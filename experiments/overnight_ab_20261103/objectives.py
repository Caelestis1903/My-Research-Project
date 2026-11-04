"""Native single components and the two frozen combined objective entries."""
import torch
from torch.nn import functional as F
from experiments.mdmt_history_information_20261103 import objectives as history_objectives
from experiments.mdmt_history_information_20261103.mixing import build_mix_batches
from experiments.mdmt_history_information_20261103.optimizers import HistoryOptimizer, norm
from experiments.mdmt_history_information_20261103.protocol import update_ema
from experiments.conditional_information_momentum_20261102_v3 import objectives as pair_objectives
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import sample, forward_logits
from experiments.conditional_information_momentum_20261102_v3.optimizers import PairOptimizer
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.objectives import losses as gaussian_losses
from .optimizers import SerialCPOptimizer

CP = 'C_MIX_P01_SERIAL_V1'
CG = 'C_MIX_GAUSSIAN_V1'


def native_config(config, arm=None):
    arm = config['arm'] if arm is None else arm
    c = {key: config[key] for key in ('dataset', 'lr', 'weight_decay', 'rho', 'epochs', 'batch_size')}
    c['arm'] = arm
    if arm in ('P01', CP):
        p = config['p01']
        if (p['pairs_per_step'], p['history_alpha'], p['history_bias_beta']) != (2048, .01, .99):
            raise ValueError('PAIR source sampling and clock constants must remain frozen')
        c.update(mode=p['mode'], dynamic=p['dynamic'], lambda_pair=p['lambda_pair'])
    if arm in ('C_MIX', CP, CG):
        c.update(config['c_mix'])
        c.update(goal='real', optimizer='current_gradient_repair', probe='conditioned_mix', c_mix=config['c_mix'])
    if arm in ('A0', 'A0f'):
        c.update(arm='A0f', goal='real', optimizer='source_adam')
    if arm in ('GAUSSIAN_A1', 'A1'):
        c.update(arm='A1', goal='source_gaussian', optimizer='source_adam', **{
            key: config['gaussian'][key] for key in ('lambda_aug', 'kappa', 'target_domain')})
    return c


def make_optimizer(model, config):
    c = native_config(config)
    if config['arm'] == CP:
        return SerialCPOptimizer(model, c)
    if config['arm'] == 'P01':
        return PairOptimizer(model, c)
    if config['arm'] in ('A0', 'A0f', 'C_MIX', 'GAUSSIAN_A1', 'A1', CG):
        return HistoryOptimizer(model, c)
    raise ValueError(f"Unregistered B arm: {config['arm']}")


def _probes(model, x, y, row_ids, c, context):
    start = (context['global_step'] - 1) % 3
    domain = next(d for d in ((start + i) % 3 for i in range(3)) if bool((x[-1] == d).any()))
    batches = build_mix_batches(x, y, row_ids, context, 'conditioned', c['tau'], domains=[domain])
    pairs = history_objectives.collect_history_probes(model, x, y, context, batches, c['probe'])
    for gradients in [g for probe in pairs for g in probe]:
        if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
            raise FloatingPointError('Nonfinite C probe gradient')
    return batches, pairs


def train_step(model, optimizer, ema, x, y, row_ids, config, context, diagnostic=False):
    arm = config['arm']
    c = native_config(config)
    context = dict(context, study='EMA02V2', split_id='original')
    if arm in ('A0', 'A0f', 'C_MIX', 'GAUSSIAN_A1', 'A1'):
        return history_objectives.train_step(model, optimizer, ema, x, y, row_ids, c, context, diagnostic)
    if arm == 'P01':
        return pair_objectives.train_step(model, optimizer, ema, x, y, row_ids, c, context, diagnostic)
    if arm not in (CP, CG):
        raise ValueError(f'Unregistered combined arm: {arm}')
    if not bool(torch.isfinite(y).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Expected complete finite binary labels')
    optimizer.zero_grad()
    work = dict(original_rows=len(y), training_forwards=0, training_forward_rows=0,
                probe_forwards=0, probe_forward_rows=0, parameter_backward_calls=1,
                shared_autograd_calls=0, optimizer_steps=1, ema_updates=1, pair_rows=0)
    phase = 'training'
    def count(module, args):
        work[phase + '_forwards'] += 1
        work[phase + '_forward_rows'] += args[0].shape[1]
    handle = model.register_forward_pre_hook(count)
    try:
        pair = None
        r = None
        info = {}
        if arm == CP:
            prediction, logits = forward_logits(model, x)
            real = F.binary_cross_entropy(prediction, y, reduction='mean')
            pair = sample(x[-1].detach().cpu().numpy(), y.detach().cpu().numpy(), optimizer.demand,
                          c['dynamic'], context, row_ids)
            loss = real
            if pair is not None:
                pos = torch.as_tensor(pair['pos'], device=x.device)
                neg = torch.as_tensor(pair['neg'], device=x.device)
                margin = logits[pos, pair['task']] - logits[neg, pair['task']]
                auxiliary = F.softplus(-margin).mean()
                ratio = float(pair['w'][pair['index']] / pair['q'][pair['index']])
                loss = real + c['lambda_pair'] * ratio * auxiliary
                r = [g.detach() for g in torch.autograd.grad(auxiliary, [p for _, p in optimizer.shared], retain_graph=True)]
                work['shared_autograd_calls'] += 1
                work['pair_rows'] = 4096
                info.update(pair_loss=float(auxiliary.detach()), w_over_q=ratio,
                            pair_count=2048, pair_index_hash=pair['pair_index_hash'],
                            pair_selected=pair['selected'], pair_active=pair['active'],
                            unique_pair_rows=pair['unique_pair_rows'])
                if not bool(torch.stack([torch.isfinite(g).all() for g in r]).all()):
                    raise FloatingPointError('Nonfinite PAIR gradient')
        else:
            gaussian = native_config(config, 'GAUSSIAN_A1')
            loss, source, info, gaussian_work = gaussian_losses(model, ema, x, y, row_ids, gaussian, context,
                                                               diagnostic=diagnostic)
            prediction = source['pred']
            real = source['real'].mean()
            for key, value in gaussian_work.items():
                if key not in ('training_forwards', 'training_forward_rows', 'probe_forwards', 'probe_forward_rows'):
                    work[key] = value
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite combined supervised objective')
        loss.backward()
        if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()):
            raise FloatingPointError('Nonfinite combined parameter gradient')
        phase = 'probe'
        batches, probes = _probes(model, x, y, row_ids, c, context)
        work['shared_autograd_calls'] += 2 * len(probes)
        if arm == CP:
            info.update(optimizer.step(pair=pair, r=r, probes=probes, diagnostic=diagnostic))
            info.update(active_clocks=dict(optimizer.active), selected_counts=dict(optimizer.selected))
        else:
            info.update(optimizer.step(probes=probes))
        update_ema(ema, model, c['rho'])
        info['mix_index_hashes'] = [mb.index_hash for mb in batches]
        if diagnostic:
            info['ema_online_distance'] = float(norm([p.detach() - e for p, e in zip(model.parameters(), ema.parameters())]))
            levels = torch.cat([mb.support_level for mb in batches])
            info['support_rates'] = {str(k): float((levels == k).float().mean()) for k in (1, 2, 3)}
        return dict(loss=float(loss.detach()), real_loss=float(real.detach()), pred=prediction.detach(),
                    diagnostics=info, counts=work)
    finally:
        handle.remove()
