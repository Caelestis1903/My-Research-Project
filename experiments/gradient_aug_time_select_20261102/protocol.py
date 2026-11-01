"""Frozen GATS02 protocol, configuration, identities and RAM-only state."""
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from experiments.gradient_aug_aig_m_20261101.protocol import protocol as source_protocol, digest
from experiments.gradient_aug_momentum_20261101.protocol import (
    setup, cpu_copy, equal, rng_state, restore_rng, state_digest, tensor_bytes,
    send_memory, receive_memory, write_json, append_csv, selected_update, update_shadow)
from experiments.gradient_aug_aig_m_20261101.optimizer import parameter_groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_SEEDS = [2022, 2024]
CONFIRM_SEEDS = [2023, 2025, 2026]
ARMS = ['A0', 'A1', 'B2', 'B1', 'B3']
HISTORY = ROOT/'research_reports/2026-11-02_gradient_aug_time_select/historical_aigm_baseline'


def protocol():
    p = source_protocol()
    p.update(study_id='GATS02', developer_seeds=DEV_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
        limits=dict(unique=40, additional=8, confirmed_candidates=2, retries=2, stop_hours=10, hard_hours=12),
        random_off_epochs=sorted(np.random.default_rng(1729).choice(np.arange(1, 51), 5, replace=False).tolist()))
    return p


def config(arm, dataset, **changes):
    base = arm.split('_')[0]
    c = dict(arm=arm, augmentation_enabled=base!='A0', difficulty_mask_enabled=base=='B1',
        task_conflict_projection_enabled=base=='B2', delayed_augmentation_enabled=base=='B3',
        random_mask_enabled=base=='R1', fixed_shrink_enabled=base=='R2', random_schedule_enabled=base=='R3',
        ema_enabled=True, delayed_ema_enabled=base=='A0', swa_enabled=base=='A0',
        adaptive_ema_enabled=base!='A0', replay_enabled=base!='A0',
        lambda_aug=.5, noise_scale=.05 if dataset=='movielens' else .1,
        rho_fixed=math.exp(math.log(.5)/40) if dataset=='movielens' else .995,
        rho_min=.95 if dataset=='movielens' else .98, rho_max=.995 if dataset=='movielens' else .999,
        cv_window=100, cv_c=.5, eta=1., delay_epochs=5, shrink={'E':1., 'P':1.},
        lr=.01, weight_decay=1e-4 if dataset=='movielens' else 1e-5,
        betas=[.9,.999], eps=1e-8, precision='FP32', epochs=50, batch_size=40960,
        group_rule='E_embedding_P_target_path_U_other_v1')
    c.update(changes)
    assert c['epochs']==50 and c['batch_size']==40960
    return c


def lambda_for(c, epoch, off_epochs):
    if not c['augmentation_enabled']:
        return 0.
    if c['delayed_augmentation_enabled']:
        return 0. if epoch<=c['delay_epochs'] else 25/(50-c['delay_epochs'])
    if c['random_schedule_enabled']:
        schedule=c.get('off_epochs',off_epochs)
        return 0. if epoch in schedule else 25/(50-len(schedule))
    return c['lambda_aug']


def optimizer(model, c):
    return torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'],
        betas=tuple(c['betas']), eps=c['eps'], foreach=False, fused=False, amsgrad=False)


def source_manifest():
    directories=[HERE, ROOT/'experiments/gradient_aug_aig_m_20261101',
        ROOT/'experiments/gradient_aug_momentum_20261101', ROOT/'experiments/gradient_aug_20261031_v2']
    files=sorted(f for directory in directories for f in directory.rglob('*') if f.suffix in ('.py','.json'))
    rows=[dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        bytes=f.stat().st_size) for f in files]
    return dict(files=rows, hash=digest(rows))


def identity(dataset, seed, c, p, control=None):
    return dict(dataset=dataset,seed=seed,config_hash=digest(c),data_hash=digest(p['datasets'][dataset]),
        source_hash=p['source_manifest_hash'],execution_commit=p['execution_commit'],rng=p['rng'],
        control_hash=digest(control or {}),epochs=50,batch_size=40960,precision='FP32',
        metrics='GA31_R3_six_cell_macro_v1')


def snapshot(model, opt, shadows, controller, loaders, epoch, step, validation, selected, best, c, device, counters):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(opt.state_dict()),
        shadows={k:cpu_copy(v.state_dict()) for k,v in shadows.items()},
        controller=controller.state(),rng=cpu_copy(rng_state(device)),
        loaders={k:g.get_state().clone() for k,g in loaders.items()},epoch=epoch,global_step=step,
        validation=copy.deepcopy(validation),selected=copy.deepcopy(selected),best=copy.deepcopy(best),
        config=copy.deepcopy(c),counters=copy.deepcopy(counters),resume_boundary='epoch')


def restore(model,opt,shadows,controller,loaders,state,c,device):
    assert digest(c)==digest(state['config'])
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    for key,shadow in shadows.items(): shadow.load_state_dict(state['shadows'][key])
    controller.restore(state['controller'])
    for key,g in loaders.items():g.set_state(state['loaders'][key])
    restore_rng(state['rng'],device)
    return copy.deepcopy({k:state[k] for k in ('epoch','global_step','selected','best','counters')})


def model_state(state, kind):
    return state['model'] if kind=='online' else state['shadows'][kind]


def checkpoint_rows(result,kind):
    rows=[]
    for checkpoint,epoch in [('selected',result['selected_epochs'][kind]),('E10',10),('E50',50)]:
        if kind=='swa20' and checkpoint=='E10':continue
        state=result['states'][epoch]
        rows.append(dict(dataset=result['dataset'],arm=result['trial'],seed=result['seed'],stage=result['stage'],
            trial=result['trial'],attempt=result['attempt'],model_kind=kind,checkpoint=checkpoint,epoch=epoch,
            global_step=state['global_step'],model_hash=state_digest(model_state(state,kind)),
            optimizer_hash=state_digest(state['optimizer']),config_hash=result['config_hash'],run_hash=result['run_hash'],
            source_hash=result['run_identity']['source_hash'],execution_commit=result['run_identity']['execution_commit'],
            validation=state['validation'][kind]))
    return rows
