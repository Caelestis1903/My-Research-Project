"""V11 full-state shadows on a fixed native SN path and local counterfactuals."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import v3_audit_20261016 as audit
import v3_train_20261016 as train
import v3_local_20261016 as local

base = audit.base


def adam_delta(gradient, m, v, step, lr=.003, beta1=.9, beta2=.999, m_only=False):
    """Input gradient already includes coupled L2; input history is read-only."""
    next_m = m.lerp(gradient, 1 - beta1)
    next_v = v * beta2 + gradient.square() * (1 - beta2)
    numerator = next_m / (1 - beta1 ** (step + 1))
    denominator = torch.ones_like(v) if m_only else next_v.sqrt() / ((1 - beta2 ** (step + 1)) ** .5) + 1e-8
    return -lr * numerator / denominator


def histories(model, opt):
    return tuple(torch.cat([opt.state[p][name].detach().flatten() for p in model.parameters()])
        for name in ('exp_avg', 'exp_avg_sq'))


def stored_batches(directory, epoch):
    data = np.load(directory / f'batches_epoch{epoch:03d}.npz')
    starts = np.r_[0, np.cumsum(data['sizes'])]
    return [data['rows'][starts[i]:starts[i+1]] for i in data['order']]


def checkpoint_check(model, opt, directory, epoch):
    path = directory / f'epoch{epoch:03d}.pt'
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert base.sha(path) == receipt['artifact_hashes'][path.name]
    state = torch.load(path, map_location='cpu', weights_only=False)
    records = []
    for name, actual, expected in [('parameters', local.vector(model).detach().cpu(),
            torch.cat([state['model'][n].flatten() for n, _ in model.named_parameters()]))]:
        error = float((actual.double()-expected.double()).norm())
        bound = 1e-7 + 1e-5 * float(expected.double().norm())
        assert error <= bound, (name, epoch, error, bound)
        records.append(dict(quantity=name, l2_error=error, bound=bound))
    saved = list(state['optimizer']['state'].values())
    for name, actual in zip(('exp_avg', 'exp_avg_sq'), histories(model, opt)):
        expected = torch.cat([s[name].flatten() for s in saved]).to(actual.device)
        error = float((actual.double()-expected.double()).norm())
        bound = 1e-9 + 1e-5 * float(expected.double().norm())
        assert error <= bound, (name, epoch, error, bound)
        records.append(dict(quantity=name, l2_error=error, bound=bound))
    return dict(epoch=epoch, checkpoint_sha=base.sha(path), comparisons=records)


def evaluate(model, b, device, split, theta, delta, metadata, groups):
    local.load_vector(model, theta + delta)
    predictions = train.predict(model, b, 'val', device)
    row = dict(**metadata, train_empirical_objective=local.objective(model, b, device))
    cells = []
    for name in ('selection', 'diagnostic'):
        result = train.metrics(b, predictions, 'val', split[name])
        for cell in result:
            cells.append(dict(**cell, **metadata, split=name))
        for metric in ('auc', 'logloss'):
            row[f'{name}_{metric}'] = float(np.mean([c[metric] for c in result]))
    for group, indices in groups.items():
        row[f'displacement_{group}'] = float(delta[indices].double().norm())
    row['displacement_ALL'] = float(delta.double().norm())
    return row, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--natural', type=Path, required=True)
    parser.add_argument('--balanced', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--oracle-epoch1-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    configs = [json.loads((d / 'config.json').read_text()) for d in (args.natural, args.balanced)]
    for key in ('dataset', 'seed', 'optimizer', 'lr', 'batch_size', 'data', 'init_sha256', 'source_hashes'):
        assert configs[0][key] == configs[1][key]
    config = configs[0]
    assert config['optimizer'] == 'adam' and config['lr'] == .003
    assert [c['arm'] for c in configs] == ['SN', 'SB']
    args.out.mkdir(parents=True, exist_ok=False)
    base.setup_seed(config['seed'])
    b = audit.bundle(config['dataset'], config['seed'])
    model = base.fixed_model(b, config['seed']).to(args.device)
    assert base.initial_state_hash(model) == config['init_sha256']
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-5)
    p = np.asarray(config['p'])
    weights = [torch.tensor(p/q, device=args.device, dtype=torch.float32)
        for q in (p, np.ones(3)/3)]
    theta = local.vector(model)
    shadow_m = torch.zeros_like(theta)
    shadow_v = torch.zeros_like(theta)
    groups, _ = audit.block_indices(model)
    split = np.load(audit.OUT / 'splits' / f'{config["dataset"]}_seed{config["seed"]}.npz')
    stages = [1] if args.oracle_epoch1_only else [1, 10, 40]
    step = 0
    checks = []
    rows, cells, draws = [], [], []
    producers = [json.loads((d / 'receipt.json').read_text()) for d in (args.natural, args.balanced)]
    for epoch in range(1, max(stages)+1):
        for d, r in zip((args.natural, args.balanced), producers):
            name = f'batches_epoch{epoch:03d}.npz'
            assert base.sha(d/name) == r['artifact_hashes'][name]
        batches = [stored_batches(d, epoch) for d in (args.natural, args.balanced)]
        assert len(batches[0]) == len(batches[1])
        for sn, sb in zip(*batches):
            model.train()
            theta = local.vector(model).detach()
            gradient, _ = audit.grad(model, b, sb, weights[1], args.device)
            gradient = gradient.add(theta, alpha=1e-5)
            shadow_m.lerp_(gradient, .1)
            shadow_v.mul_(.999).addcmul_(gradient, gradient, value=.001)
            audit.grad(model, b, sn, weights[0], args.device)
            opt.step()
            step += 1
        if epoch in stages:
            checks.append(checkpoint_check(model, opt, args.natural, epoch))
            if not args.oracle_epoch1_only:
                theta = local.vector(model).detach().clone()
                native_m, native_v = histories(model, opt)
                metadata = dict(seed=config['seed'], epoch=epoch, current_arm='BASE',
                    condition='BASE', norm_control='BASE', norm_informative=True)
                row, cc = evaluate(model, b, args.device, split, theta, torch.zeros_like(theta), metadata, groups)
                rows.append(row); cells.extend(cc)
                for arm_index, (arm, q) in enumerate((('SN', p), ('SB', np.ones(3)/3))):
                    local.load_vector(model, theta); model.train()
                    sampler_seed = 20261016 + config['seed']*1000 + epoch*10 + arm_index
                    batch = base.Sampler(b, q, sampler_seed, 'iid').sample(config['batch_size'])
                    gradient, _ = audit.grad(model, b, batch, weights[arm_index], args.device)
                    gradient = gradient.add(theta, alpha=1e-5)
                    deltas = dict(native=adam_delta(gradient, native_m, native_v, step),
                        swap_m=adam_delta(gradient, shadow_m, native_v, step),
                        swap_v=adam_delta(gradient, native_m, shadow_v, step),
                        full_shadow=adam_delta(gradient, shadow_m, shadow_v, step),
                        beta1_zero_v_only=adam_delta(gradient, native_m, native_v, step, beta1=0.),
                        beta2_zero=adam_delta(gradient, native_m, native_v, step, beta2=0.),
                        m_only=adam_delta(gradient, native_m, native_v, step, m_only=True))
                    target_norm = float(deltas['native'].double().norm())
                    draws.append(dict(seed=config['seed'], epoch=epoch, arm=arm,
                        sampler_seed=sampler_seed, row_sha=hashlib.sha256(batch.tobytes()).hexdigest(),
                        native_delta_norm=target_norm, steps=step,
                        shadow_m_difference_norm=float((shadow_m-native_m).double().norm()),
                        shadow_v_difference_norm=float((shadow_v-native_v).double().norm())))
                    for condition, raw_delta in deltas.items():
                        norm = float(raw_delta.double().norm())
                        informative = min(norm, target_norm) >= 1e-12
                        for norm_control in ('raw', 'match_native_global_norm'):
                            if norm_control != 'raw' and not informative:
                                continue
                            delta = raw_delta if norm_control == 'raw' else raw_delta*(target_norm/norm)
                            metadata = dict(seed=config['seed'], epoch=epoch, current_arm=arm,
                                condition=condition, norm_control=norm_control, norm_informative=informative)
                            row, cc = evaluate(model, b, args.device, split, theta, delta, metadata, groups)
                            rows.append(row);cells.extend(cc)
                local.load_vector(model, theta)
        base.dump(args.out / 'progress.json', dict(epoch_completed=epoch, steps=step, time=time.time()))
    pd.DataFrame(rows).to_csv(args.out/'summary.csv', index=False)
    pd.DataFrame(cells).to_csv(args.out/'cells.csv', index=False)
    base.dump(args.out/'receipt.json', dict(status='PASS_NATIVE_REPLAY_ORACLE' if args.oracle_epoch1_only else 'COMPLETE',
        source_sha=base.sha(__file__), seed=config['seed'], stages=stages, draws=draws, checkpoint_checks=checks,
        source_runs=list(map(str, (args.natural,args.balanced))),
        source_receipt_sha=[base.sha(d/'receipt.json') for d in (args.natural,args.balanced)],
        design_sha=base.sha(audit.OUT/'V11_state_decomposition_design.json'),
        freeze_sha=base.sha(audit.OUT/'analysis_rules_frozen.json'), test_access=False,
        scope='Full shadow moments along common SN theta path; local conditional counterfactuals, not long-run causal mediation'))


if __name__ == '__main__':
    main()
