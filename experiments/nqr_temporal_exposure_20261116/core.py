"""Schedule/config adapter; all numerical operations remain in frozen C2."""
from experiments.natural_quadratic_ranking_20261116.core import train_step as frozen_train_step
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SELECTION, digest, load_protocol,
)


ARMS = ('NQR-EARLY10', 'NQR-LATE10')
_WINDOWS = {'NQR-EARLY10': ('early10_v1', 1, 10),
            'NQR-LATE10': ('late10_v1', 41, 50)}
_FIXED = dict(cycle_id='cycle_03', objective='nqr_temporal_exposure_v1',
              source_operator='experiments.natural_quadratic_ranking_20261116.core.train_step',
              inner_arm='NQR-025', completed_risk='Q_R', probability_margin=1.,
              active_lambda=.25, off_lambda=0., epochs=50, batch_size=40960,
              precision='float32', lr=.01, betas=[.9, .999], eps=1e-8,
              loader_seed_offset=11000, gate_training_detached=True,
              probe_gate_full_derivative=False, algorithm_rng_draws=0,
              selection_hash=digest(SELECTION))
_DATASETS = {
    name: dict(weight_decay=source['weight_decay'], rho=source['rho'],
               steps_per_epoch=20 if name == 'movielens' else 49,
               tail_rows=21927 if name == 'movielens' else 40052,
               data_hash=digest(source['source']))
    for name, source in load_protocol()['datasets'].items()
}


def schedule(arm, epoch):
    if not isinstance(arm, str) or arm not in _WINDOWS:
        raise ValueError('Technical PAUSE: unregistered temporal arm')
    if type(epoch) is not int or not 1 <= epoch <= 50:
        raise ValueError('Technical PAUSE: schedule epoch must be an integer in 1..50')
    identity, first, last = _WINDOWS[arm]
    active = first <= epoch <= last
    return dict(schedule_id=identity, schedule_epoch=epoch,
                schedule_phase='on' if active else 'off', scheduled_active=active,
                effective_lambda=.25 if active else 0., external_arm=arm,
                inner_arm='NQR-025')


def _validate(config, context, row_ids):
    if not isinstance(config, dict) or not isinstance(context, dict):
        raise ValueError('Technical PAUSE: config and context must be dicts')
    current = schedule(config.get('arm'), context.get('epoch'))
    dataset = config.get('dataset')
    if not isinstance(dataset, str) or dataset not in _DATASETS:
        raise ValueError('Technical PAUSE: unregistered dataset')
    _, first, last = _WINDOWS[config['arm']]
    expected = dict(_FIXED, **_DATASETS[dataset], schedule_id=current['schedule_id'],
                    active_epoch_start=first, active_epoch_end=last)
    for name, value in expected.items():
        actual = config.get(name)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Technical PAUSE: inconsistent temporal config {name}')
    if 'mixture_lambda' in config:
        raise ValueError('Technical PAUSE: outer config must not contain mixture_lambda')
    if context.get('dataset') != dataset:
        raise ValueError('Technical PAUSE: context/config dataset mismatch')
    if type(context.get('seed')) is not int or not 2022 <= context['seed'] <= 2026:
        raise ValueError('Technical PAUSE: seed must be an integer in 2022..2026')
    for name in ('batch', 'global_step'):
        if type(context.get(name)) is not int or context[name] < 1:
            raise ValueError(f'Technical PAUSE: {name} must be a positive integer')
    pair = (context.get('purpose'), context.get('clock_kind'))
    if pair not in (('formal', 'formal'), ('engineering_check', 'engineering_virtual'),
                    ('engineering_smoke', 'engineering_virtual')):
        raise ValueError('Technical PAUSE: explicit purpose/clock_kind pairing required')
    if row_ids is None and pair[0] != 'engineering_check':
        raise ValueError('Technical PAUSE: real worker row_ids are required')
    return dict(current, clock_kind=context['clock_kind'])


def train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False):
    current = _validate(config, context, row_ids)
    inner = dict(config, arm='NQR-025', mixture_lambda=current['effective_lambda'],
                 objective='natural_quadratic_ranking_v1')
    result = frozen_train_step(model, optimizer, ema, x, y, inner, context,
                               row_ids=row_ids, diagnose=diagnose)
    active = int(current['scheduled_active'])
    B = result['ledger']['B']
    increments = dict(active_updates=active, off_updates=1-active,
                      active_rows=active*B, off_rows=(1-active)*B,
                      active_task_entries=2*active*B, off_task_entries=2*(1-active)*B,
                      lambda_update_sum=current['effective_lambda'],
                      nonzero_opportunity_updates=int(result['ledger']['opposite_mass'] > 0))
    return dict(result, schedule=current, phase_increments=increments)
