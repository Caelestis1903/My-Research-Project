"""Frozen finite candidates, identities, independent shadows, and RAM states."""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from experiments.gradient_aug_momentum_20261101.protocol import (
    protocol as history_protocol, setup, cpu_copy, equal, rng_state, restore_rng,
    tensor_bytes, send_memory, receive_memory, write_json, append_csv, state_digest, selected_update,
    update_shadow)
from experiments.gradient_aug_momentum_20261101.optimizer import SADMomentum
from .optimizer import AIGMomentum, parameter_groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_SEEDS = [2022, 2024]
SHADOWS = ['ema995', 'H2', 'H8', 'H20']
TRIAL_ORDER = {'S1':['C','A','B','D'], 'S2':['G1','G2','G3'], 'S3':['S95','S99'], 'S4':['M1','M2']}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol():
    p = history_protocol()
    p.update(study_id='AIGM01', developer_seeds=DEV_SEEDS,
        plan_source=dict(repository='Caelestis1903/My-Research-Project', main='0973fa5f1a804aec34f88ccb2ffdd8c0231b602b',
            gam01='ee4cf0512a7723624c9ee6a1ef192f0f7f17686b'),
        aig=dict(beta_delta=.95, c=1., slope=4., epsilon=1e-12, lambda_min=0., threshold_epochs=[1]),
        nominal_budget=dict(development=44, formal_per_dataset=10, optional_aig_per_dataset=5, max_total=74))
    p.pop('arms'); p.pop('decision')
    return p


def shadow_rates(dataset, p):
    steps = p['datasets'][dataset]['steps_per_epoch']
    return dict(ema995=.995, **{f'H{h}':math.exp(math.log(.5)/(h*steps)) for h in (2,8,20)})


def candidates(stage, dataset, p, base=None, thresholds=None):
    wd = 1e-4 if dataset == 'movielens' else 1e-5
    common = dict(use_augmentation=False, use_sad=False, use_aig_m=False, use_shadow_ema=True,
        use_aig_swa=False, lr=.01, weight_decay=wd, lambda_aug=.5, noise_scale=.1, beta_delta=0.,
        lambda_min=0., lambda_max=.5, thresholds={}, shadow_rates=shadow_rates(dataset,p),
        group_rule='E_embedding_P_target_path_U_other_v1', loss_scale=1,
        betas=[.9,.999], eps=1e-8, slope=4.)
    if base is not None:
        common.update(lr=base['lr'], weight_decay=base['weight_decay'])
    if stage == 'S1':
        specs = [('A',dict(lr=.005)),('B',dict(weight_decay=.3*wd)),('C',{}),('D',dict(lr=.02,weight_decay=3*wd))]
    elif stage == 'S2':
        common['use_augmentation'] = True
        specs = [('G1',dict(noise_scale=.05)),('G2',{}),('G3',dict(lambda_aug=.75))]
    elif stage == 'S3':
        common.update(use_augmentation=True, use_sad=True)
        specs = [('S95',dict(beta_delta=.95)),('S99',dict(beta_delta=.99))]
    elif stage == 'S4':
        if thresholds is None:
            raise ValueError('AIG thresholds must be frozen from S95 E1 training statistics')
        common.update(use_augmentation=True, use_aig_m=True, beta_delta=.95)
        specs = [('M1',dict(lambda_max=.25,thresholds=thresholds['q50'])),
                 ('M2',dict(lambda_max=.5,thresholds=thresholds['q75']))]
    else:
        raise ValueError(stage)
    return {trial:dict(common, **options) for trial,options in specs}


def validate_config(c):
    if c['use_aig_swa'] or (c['use_sad'] and c['use_aig_m']):
        raise ValueError('AIG-SWA is outside this study; SAD and AIG modes are exclusive')
    if (c['use_sad'] or c['use_aig_m']) and not c['use_augmentation']:
        raise ValueError('Separated modes require augmentation')
    if not 0 <= c['beta_delta'] < 1 or not 0 <= c['lambda_aug'] <= 1:
        raise ValueError('Invalid delta decay or augmentation mixture')
    if c['use_aig_m'] and set(c['thresholds']) != {'E','P'}:
        raise ValueError('Frozen E/P thresholds required')


def active_separation(c):
    return (c['use_augmentation'] and c['noise_scale'] != 0 and
        ((c['use_aig_m'] and c['lambda_max'] != 0) or
         (c['use_sad'] and c['beta_delta'] != 0 and c['lambda_aug'] != 0)))


def make_optimizer(model, p, c, dataset):
    validate_config(c)
    options = dict(lr=c['lr'], weight_decay=c['weight_decay'])
    if active_separation(c):
        if c['use_aig_m']:
            return AIGMomentum(model.parameters(), parameter_groups(model,p['datasets'][dataset]['augmented_domain']),
                beta_delta=c['beta_delta'], lambda_min=c['lambda_min'], lambda_max=c['lambda_max'],
                thresholds=c['thresholds'], slope=c['slope'], **options)
        return SADMomentum(model.parameters(), beta_delta=c['beta_delta'], lambda_aug=c['lambda_aug'], **options)
    return torch.optim.Adam(model.parameters(), betas=tuple(c['betas']), eps=c['eps'],
        foreach=False, fused=False, amsgrad=False, **options)


def make_shadows(model, c):
    if not c['use_shadow_ema']:
        return {}
    return {key:copy.deepcopy(model).requires_grad_(False) for key in c['shadow_rates']}


def update_shadows(shadows, model, c):
    for key, shadow in shadows.items():
        update_shadow(shadow, model, c['shadow_rates'][key])


def snapshot(model, optimizer, shadows, loaders, epoch, step, validation, selected, best, c, device, counters):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
        shadows={k:cpu_copy(v.state_dict()) for k,v in shadows.items()},
        rng=cpu_copy(rng_state(device)), loaders={k:g.get_state().clone() for k,g in loaders.items()},
        epoch=epoch, global_step=step, validation=copy.deepcopy(validation), selected=copy.deepcopy(selected),
        best=copy.deepcopy(best), config=copy.deepcopy(c), counters=copy.deepcopy(counters), resume_boundary='epoch')


def restore(model, optimizer, shadows, loaders, state, c, device):
    assert digest(c) == digest(state['config'])
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    for key, shadow in shadows.items():
        shadow.load_state_dict(state['shadows'][key])
    for key,generator in loaders.items():
        generator.set_state(state['loaders'][key])
    restore_rng(state['rng'], device)
    return copy.deepcopy({k:state[k] for k in ('epoch','global_step','selected','best','counters')})


def model_state(state, kind):
    return state['model'] if kind == 'online' else state['shadows'][kind]


def checkpoint_rows(result, kind, arm=None):
    rows = []
    for checkpoint, epoch in [('selected',result['selected_epochs'][kind]),('E10',10),('E50',50)]:
        state = result['states'][epoch]
        rows.append(dict(dataset=result['dataset'], arm=arm or result['trial'], seed=result['seed'],
            stage=result['stage'], trial=result['trial'], attempt=result['attempt'], model_kind=kind,
            checkpoint=checkpoint, epoch=epoch, global_step=state['global_step'],
            model_hash=state_digest(model_state(state,kind)), optimizer_hash=state_digest(state['optimizer']),
            config_hash=result['config_hash'], run_hash=result['run_hash'],
            validation=state['validation'][kind]))
    return rows


def source_manifest():
    dirs = [HERE, ROOT/'experiments/gradient_aug_momentum_20261101', ROOT/'experiments/gradient_aug_20261031_v2']
    files = sorted(f for directory in dirs for f in directory.rglob('*') if f.suffix in ('.py','.json'))
    rows = [dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size) for f in files]
    return dict(files=rows, hash=digest(rows))
