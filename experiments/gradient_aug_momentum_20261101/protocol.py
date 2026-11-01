"""GAM01 configuration, state identities, and RAM-only snapshots."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from experiments.gradient_aug_20261031_v2.vendor_ga31.protocol import (
    setup, cpu_copy, equal, rng_state, restore_rng, tensor_bytes, send_memory,
    receive_memory, write_json, append_csv)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / 'resolved_config.json'


def protocol(path=DEFAULT_PROTOCOL):
    p = json.loads(Path(path).read_text())
    source = json.loads((ROOT / 'experiments/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json').read_text())
    p['model'] = source['model']
    p['datasets'] = source['datasets']
    assert p['seeds'] == list(range(2022, 2027))
    assert p['training']['epochs'] == 50 and p['training']['batch_size'] == 40960
    assert [a['id'] for a in p['arms']] == ['A0', 'A1', 'A2', 'A3']
    for arm in p['arms']:
        validate_flags(arm)
    return p


def validate_flags(c):
    if c['use_reliability_gain']:
        if c.get('num_views', 1) != 2:
            raise ValueError('Reliability gain requires K=2')
        raise ValueError('P2 is not enabled before its conditional design and review')
    if c['num_views'] != 1:
        raise ValueError('Stage1 uses K=1')
    if not 0 <= c['beta_delta'] < 1 or not 0 <= c['rho_shadow'] < 1:
        raise ValueError('EMA decay must be in [0,1)')


def arm_config(p, arm, dataset):
    c = copy.deepcopy(next(a for a in p['arms'] if a['id'] == arm))
    c['weight_decay'] = 1e-4 if dataset == 'movielens' else 1e-5
    c['loss_scale'] = 1
    validate_flags(c)
    return c


def separated(c):
    return (c['use_delta_momentum'] and c['beta_delta'] != 0 and
            c['use_augmented_view'] and c['lambda_aug'] != 0 and c['noise_scale'] != 0)


def make_optimizer(model, p, c):
    from .optimizer import SADMomentum
    options = dict(lr=p['optimizer']['lr'], betas=tuple(p['optimizer']['betas']),
                   eps=p['optimizer']['eps'], weight_decay=c['weight_decay'])
    if separated(c):
        return SADMomentum(model.parameters(), beta_delta=c['beta_delta'],
                           lambda_aug=c['lambda_aug'], **options)
    return torch.optim.Adam(model.parameters(), foreach=False, fused=False, amsgrad=False, **options)


def state_digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k, item in v.items():
                h.update(str(k).encode()); visit(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(str(v).encode())
    visit(value)
    return h.hexdigest()


def make_shadow(model, c):
    if not c['use_shadow_ema']:
        return None
    result = copy.deepcopy(model)
    result.requires_grad_(False)
    return result


@torch.no_grad()
def update_shadow(shadow, model, rho):
    if shadow is None:
        return
    for target, source in zip(shadow.parameters(), model.parameters()):
        target.mul_(rho).add_(source, alpha=1-rho)
    for target, source in zip(shadow.buffers(), model.buffers()):
        target.copy_(source)


def snapshot(model, optimizer, shadow, loaders, epoch, step, validations, device, identity):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                shadow=cpu_copy(shadow.state_dict()) if shadow is not None else None,
                rng=cpu_copy(rng_state(device)), loaders={k:g.get_state().clone() for k,g in loaders.items()},
                epoch=epoch, global_step=step, shadow_updates=step if shadow is not None else 0,
                validation=copy.deepcopy(validations), identity=copy.deepcopy(identity), resume_boundary='epoch')


def restore(model, optimizer, shadow, loaders, state, device):
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    if shadow is not None:
        shadow.load_state_dict(state['shadow'])
    for k, generator in loaders.items():
        generator.set_state(state['loaders'][k])
    restore_rng(state['rng'], device)
    return state['epoch'], state['global_step'], state['shadow_updates']


def selected_update(selected, best, kind, epoch, value):
    if epoch >= 1 and value > best[kind]:
        selected[kind] = epoch; best[kind] = value
        return True
    return False


def checkpoint_rows(result):
    rows = []
    for kind, epoch in result['selected_epochs'].items():
        for checkpoint, e in [('selected', epoch), ('E10', 10), ('E50', 50)]:
            state = result['states'][e]
            rows.append(dict(dataset=result['dataset'], arm=result['arm'], seed=result['seed'],
                attempt=result['attempt'], model_kind=kind, checkpoint=checkpoint, epoch=e,
                global_step=state['global_step'], model_hash=state_digest(state['model' if kind == 'online' else 'shadow']),
                optimizer_hash=state_digest(state['optimizer']), shadow_updates=state['shadow_updates'],
                validation=state['validation'][kind]))
    return rows
