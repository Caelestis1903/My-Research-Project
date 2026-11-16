"""Finite CPU adapter checks; runtime checks are imported only after integration."""
import copy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import core
from experiments.natural_quadratic_ranking_20261116.core import train_step as frozen_step
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SELECTION, cpu_copy, digest, load_protocol, optimizer, restore_rng, rng_state, state_digest,
)


_ROOT = Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
_CYCLES = _ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles'
_PHASE_KEYS = {'active_updates', 'off_updates', 'active_rows', 'off_rows',
               'active_task_entries', 'off_task_entries', 'lambda_update_sum',
               'nonzero_opportunity_updates'}
_SOURCE_KEYS = {'loss', 'clean_loss', 'completed_loss', 'branch', 'ledger', 'row_order_hash',
                'optimizer_updates', 'ema_updates', 'ordinary_forwards', 'scalar_backwards',
                'probe_forwards', 'view_forwards', 'algorithm_rng_draws', 'seconds', 'diagnostics'}


def _config(arm='NQR-EARLY10', dataset='movielens'):
    source = load_protocol()['datasets'][dataset]
    early = arm == 'NQR-EARLY10'
    return dict(cycle_id='cycle_03', dataset=dataset, arm=arm,
                objective='nqr_temporal_exposure_v1',
                source_operator='experiments.natural_quadratic_ranking_20261116.core.train_step',
                inner_arm='NQR-025', completed_risk='Q_R', probability_margin=1.,
                schedule_id='early10_v1' if early else 'late10_v1',
                active_epoch_start=1 if early else 41, active_epoch_end=10 if early else 50,
                active_lambda=.25, off_lambda=0., epochs=50, batch_size=40960,
                precision='float32', lr=.01, betas=[.9, .999], eps=1e-8,
                weight_decay=source['weight_decay'], rho=source['rho'],
                steps_per_epoch=20 if dataset == 'movielens' else 49,
                tail_rows=21927 if dataset == 'movielens' else 40052,
                loader_seed_offset=11000, gate_training_detached=True,
                probe_gate_full_derivative=False, algorithm_rng_draws=0,
                data_hash=digest(source['source']), selection_hash=digest(SELECTION))


def _context(config, epoch, step=1):
    return dict(dataset=config['dataset'], seed=2022, epoch=epoch, batch=step,
                global_step=step, purpose='engineering_check', clock_kind='engineering_virtual')


class _Scores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.slope = torch.nn.Parameter(torch.tensor([.1, -.2]))
        self.zero = torch.nn.Parameter(torch.tensor([.7]))
        self.unused = torch.nn.Parameter(torch.tensor([.9]))
        self.register_buffer('buffer', torch.tensor([2.]))

    def forward(self, x):
        return (self.score+x[0, :, None]*self.slope+self.zero*0).sigmoid()


def _batch(B=7):
    rows = torch.arange(B)
    x = torch.stack(((rows % 11).float()/11, (rows % 3).float()))
    y = torch.stack(((rows % 2).float(), ((rows//2) % 2).float()), dim=1)
    return x, y, np.arange(100, 100+B, dtype=np.int64)[::-1].copy()


def _route(config, snapshot=None):
    model = _Scores()
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = optimizer(model, config)
    loader = torch.Generator(device='cpu').manual_seed(13022)
    if snapshot is not None:
        model.load_state_dict(snapshot['model'])
        ema.load_state_dict(snapshot['ema'])
        opt.load_state_dict(cpu_copy(snapshot['optimizer']))
        loader.set_state(snapshot['loader'])
        restore_rng(snapshot['rng'], 'cpu')
    return model, opt, ema, loader


def _snapshot(route):
    model, opt, ema, loader = route
    return cpu_copy(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         ema=ema.state_dict(), loader=loader.get_state(), rng=rng_state('cpu')))


def _same(left, right):
    assert state_digest(_snapshot(left)) == state_digest(_snapshot(right))
    for a, b in zip(left[0].parameters(), right[0].parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


def _inner(config, epoch):
    return dict(config, arm='NQR-025', mixture_lambda=core.schedule(config['arm'], epoch)['effective_lambda'],
                objective='natural_quadratic_ranking_v1')


def _adapter(route, x, y, config, context, rows, diagnose=False):
    before_config, before_context = copy.deepcopy(config), copy.deepcopy(context)
    delegated = []
    def observe(*args, **kwargs):
        assert args[5] == _inner(config, context['epoch']) and args[5] is not config
        value = frozen_step(*args, **kwargs)
        delegated.append(value)
        return value
    with patch.object(core, 'frozen_train_step', side_effect=observe) as spy:
        result = core.train_step(*route[:3], x, y, config, context, row_ids=rows, diagnose=diagnose)
    assert spy.call_count == 1 and config == before_config and context == before_context
    assert set(result) == _SOURCE_KEYS | {'schedule', 'phase_increments'}
    assert set(result['phase_increments']) == _PHASE_KEYS
    assert set(delegated[0]) == _SOURCE_KEYS
    assert all(result[k] is delegated[0][k] for k in _SOURCE_KEYS)
    assert result['optimizer_updates'] == result['ema_updates'] == result['ordinary_forwards'] == result['scalar_backwards'] == 1
    assert result['probe_forwards'] == result['view_forwards'] == result['algorithm_rng_draws'] == 0
    assert result['schedule'] == dict(core.schedule(config['arm'], context['epoch']), clock_kind=context['clock_kind'])
    a, B = int(result['schedule']['scheduled_active']), len(y)
    assert result['phase_increments'] == dict(active_updates=a, off_updates=1-a,
        active_rows=a*B, off_rows=(1-a)*B, active_task_entries=2*a*B,
        off_task_entries=2*(1-a)*B, lambda_update_sum=.25*a,
        nonzero_opportunity_updates=int(result['ledger']['opposite_mass'] > 0))
    json.dumps(result, allow_nan=False)
    return result


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def check_schedules_and_validation():
    for arm, start, end, identity in (('NQR-EARLY10', 1, 10, 'early10_v1'),
                                     ('NQR-LATE10', 41, 50, 'late10_v1')):
        values = [core.schedule(arm, e) for e in range(1, 51)]
        assert [v['schedule_epoch'] for v in values if v['scheduled_active']] == list(range(start, end+1))
        assert sum(v['effective_lambda'] for v in values) == 2.5
        for e, value in enumerate(values, 1):
            active = start <= e <= end
            assert value == dict(schedule_id=identity, schedule_epoch=e, schedule_phase='on' if active else 'off',
                                 scheduled_active=active, effective_lambda=.25 if active else 0.,
                                 external_arm=arm, inner_arm='NQR-025')
    for arm in ('CLEAN', 'NQR-025', '', None, True, []):
        _reject(lambda: core.schedule(arm, 1))
    for epoch in (0, 51, -1, True, False, 1., '1', None, []):
        _reject(lambda: core.schedule('NQR-EARLY10', epoch))
    c = _config()
    route = _route(c)
    x, y, rows = _batch()
    before = state_digest(_snapshot(route))
    invalid = []
    for key in c:
        missing = dict(c); del missing[key]
        invalid.append((missing, _context(c, 1), rows))
    for key, value in dict(arm='CLEAN', schedule_id='late10_v1', active_epoch_start=2,
                           active_epoch_end=11, active_lambda=.2, off_lambda=.25,
                           mixture_lambda=.25, rho=0., data_hash='wrong', selection_hash='wrong',
                           algorithm_rng_draws=True, probe_gate_full_derivative=True).items():
        invalid.append((dict(c, **{key: value}), _context(c, 1), rows))
    for key in _context(c, 1):
        missing = _context(c, 1); del missing[key]
        invalid.append((c, missing, rows))
    for key, value in [('epoch', 51), ('epoch', True), ('seed', True), ('seed', 2027),
                       ('batch', 0), ('batch', True), ('global_step', 0), ('global_step', 1.),
                       ('dataset', 'kuairand'), ('purpose', 'formal'), ('clock_kind', 'formal')]:
        invalid.append((c, dict(_context(c, 1), **{key: value}), rows))
    for purpose, clock in [('formal', 'formal'), ('engineering_smoke', 'engineering_virtual')]:
        invalid.append((c, dict(_context(c, 1), purpose=purpose, clock_kind=clock), None))
    with patch.object(core, 'frozen_train_step', side_effect=AssertionError('invalid input delegated')) as spy:
        for config, context, ids in invalid:
            _reject(lambda: core.train_step(*route[:3], x, y, config, context, row_ids=ids))
        assert spy.call_count == 0
    assert before == state_digest(_snapshot(route))
    return dict(status='PASS', schedule_values=100, rejected_train_inputs=len(invalid), mutation_before_rejection=False)


def check_frozen_and_clean_identity():
    cases = 0
    for dataset in ('movielens', 'kuairand'):
        c = _config(dataset=dataset)
        x, y, rows = _batch()
        for epoch in (10, 11):
            for populated in (False, True):
                base = _route(c)
                if populated:
                    _adapter(base, x, y, c, _context(c, 10), rows)
                snapshot = _snapshot(base)
                on, off, reference = [_route(c, snapshot) for _ in range(3)]
                rng_before = state_digest(rng_state('cpu'))
                for step in (1, 2):
                    context = _context(c, epoch, step)
                    value = _adapter(on, x, y, c, context, rows, diagnose=True)
                    quiet = _adapter(off, x, y, c, context, rows, diagnose=False)
                    assert quiet['diagnostics'] is None and value['diagnostics'] is not None
                    assert {k:v for k,v in value.items() if k not in ('seconds','diagnostics')} == {
                        k:v for k,v in quiet.items() if k not in ('seconds','diagnostics')}
                    _same(on, off)
                    if epoch == 10:
                        direct = frozen_step(*reference[:3], x, y, _inner(c, epoch), context, row_ids=rows)
                        assert {k:value[k] for k in _SOURCE_KEYS-{'seconds','diagnostics'}} == {
                            k:direct[k] for k in _SOURCE_KEYS-{'seconds','diagnostics'}}
                    else:
                        loss = source_clean_step(*reference[:3], x, y, _inner(c, epoch))
                        assert value['loss'] == value['clean_loss'] == value['completed_loss'] == loss
                        assert value['branch'] == 'lambda_zero'
                    _same(on, reference)
                assert rng_before == state_digest(rng_state('cpu'))
                assert state_digest(snapshot) == state_digest(_snapshot(base))
                cases += 1
    return dict(status='PASS', cases=cases, updates_per_route=2, exact_state_gradients_activity_rng_loader=True,
                initial_and_populated_Adam=True, diagnose_on_off=True)


def check_boundaries_and_activity():
    observed = []
    for arm, before_epoch, after_epoch in [('NQR-EARLY10',10,11), ('NQR-LATE10',40,41)]:
        c = _config(arm)
        route, reference = _route(c), _route(c)
        x, y, rows = _batch()
        totals = dict.fromkeys(_PHASE_KEYS, 0)
        rng_before = state_digest(rng_state('cpu'))
        for step, epoch in enumerate([before_epoch]*2+[after_epoch]*2, 1):
            previous = cpu_copy(route[0].state_dict())
            context = _context(c, epoch, step)
            value = _adapter(route, x, y, c, context, rows, diagnose=True)
            frozen_step(*reference[:3], x, y, _inner(c, epoch), context, row_ids=rows)
            _same(route, reference)
            model, opt, ema, _ = route
            assert model.unused.grad is None and model.unused not in opt.state
            assert torch.equal(model.unused, previous['unused'])
            assert model.zero.grad is not None and not bool(model.zero.grad.any())
            assert not torch.equal(model.zero, previous['zero'])
            assert all(int(opt.state[p]['step']) == step for p in model.parameters() if p.grad is not None)
            assert torch.equal(ema.buffer, model.buffer)
            assert value['diagnostics']['active_parameters'] == 3
            assert value['diagnostics']['none_parameters'] == 1
            assert value['diagnostics']['parameter_step_min'] == value['diagnostics']['parameter_step_max'] == step
            for name in totals:
                totals[name] += value['phase_increments'][name]
        assert totals == dict(active_updates=2, off_updates=2, active_rows=14, off_rows=14,
                              active_task_entries=28, off_task_entries=28, lambda_update_sum=.5,
                              nonzero_opportunity_updates=2)
        assert rng_before == state_digest(rng_state('cpu'))
        observed.append(dict(arm=arm, virtual_epochs=[before_epoch,after_epoch], actual_steps=4, phase_counters=totals))
    return dict(status='PASS', boundaries=observed, no_optimizer_ema_rng_loader_reset=True,
                None_skips_Adam=True, explicit_zero_remains_active=True)


def _mass(ledger):
    assert ledger['total_mass'] == 1. and ledger['task_mass'] == {'0':.5, '1':.5}
    assert abs(sum(c['bce_mass']+c['opposite_mass'] for c in ledger['cells'].values())-1.) <= 1e-15
    for t in range(2):
        assert abs(sum(ledger['cells'][f'd{d}t{t}']['bce_mass']+ledger['cells'][f'd{d}t{t}']['opposite_mass'] for d in range(3))-.5) <= 1e-15


def check_support_tails_and_nulls():
    cases = []
    for dataset, sizes in [('movielens',(1,3,7,21927)), ('kuairand',(40052,))]:
        c = _config(dataset=dataset)
        for B in sizes:
            x, y, rows = _batch(B)
            if B == 3:
                x[-1] = torch.tensor([0.,0.,1.])
                y = torch.tensor([[0.,1.],[1.,0.],[1.,1.]])
            for epoch in (10, 11):
                route = _route(c)
                value = _adapter(route, x, y, c, _context(c, epoch), rows)
                ledger = value['ledger']
                assert ledger['B'] == B and ledger['entries'] == 2*B
                assert value['row_order_hash'] == state_digest(rows)
                _mass(ledger)
                for cell in ledger['cells'].values():
                    if value['branch'] != 'ordinary' or not cell['n']:
                        assert all(cell[k] is None for k in ('mu0','mu1','var0','var1','bbar0','bbar1','S','Q_R','Q_B','Q_M','risk'))
                    elif cell['n0'] and cell['n1']:
                        for label in (0,1):
                            if cell[f'n{label}'] == 1:
                                assert cell[f'var{label}'] == 0.
                    for label in (0,1):
                        coefficient = cell[f'bce_coefficient{label}']
                        if cell[f'n{label}']:
                            assert coefficient == (1-ledger['mixture_lambda']+ledger['mixture_lambda']*cell[f'pi{label}'])/(2*B)
                        else:
                            assert coefficient is None
                if B == 1 and epoch == 10:
                    assert value['branch'] == 'all_single_class'
                    assert value['phase_increments']['active_updates'] == 1
                    assert value['phase_increments']['nonzero_opportunity_updates'] == 0
                cases.append(dict(dataset=dataset,B=B,phase=value['schedule']['schedule_phase'],branch=value['branch']))
    c = _config()
    x,y,_ = _batch(1)
    value = _adapter(_route(c), x, y, c, _context(c,1), None)
    assert value['row_order_hash'] == state_digest(np.arange(1,dtype=np.int64))
    return dict(status='PASS', cases=cases, synthetic_None_row_ids=True, mass_rounding_atol=1e-15)


def _frozen_evidence():
    manifest = json.loads((_ROOT/'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/EXPERIMENT_MANIFEST.json').read_text())
    rows = manifest['sourcebase']['c2_frozen_files']
    for row in rows:
        assert hashlib.sha256((_ROOT/row['path']).read_bytes()).hexdigest() == row['sha256']
    assert len(rows) == 87 and digest(rows) == 'ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae'
    c1 = json.loads((_CYCLES/'cycle_01/attempt1/RESOURCE_1_SUPERSESSION.json').read_text())['source_manifest']
    for row in c1['files']:
        assert hashlib.sha256((_ROOT/row['path']).read_bytes()).hexdigest() == row['sha256']
    assert len(c1['files']) == 79 and digest(c1['files']) == '6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1'
    proofs = []
    for name, expected in [('CORE_CPU_CHECKS.json','36cbf7f016769daa8eb5166a6c1595cc6b2ca375944a9bc40a2c09f06eb310a7'),
                           ('R1_CPU_CHECKS.json','2711118f9df32174bec330190a124c5c97b5099439af427446635ade6b1ec5de')]:
        path = _CYCLES/'cycle_02'/name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        receipt = json.loads(raw)
        if name == 'CORE_CPU_CHECKS.json':
            assert receipt['status'] == 'CORE_CPU_PASS_INTEGRATION_PENDING'
            assert all(v['status'] == 'PASS' for v in receipt['final_core_results'].values())
        else:
            assert receipt['status'] == 'PASS'
        proofs.append(dict(path=str(path),sha256=expected,reused=True))
    return dict(status='PASS', frozen_C2_files=87, frozen_C2_sha256=digest(rows),
                frozen_C1_files=79, frozen_C1_sha256=digest(c1['files']), reused_proofs=proofs,
                literal_QR_suite_rerun=False)


def run_checks():
    torch.set_num_threads(1)
    return dict(status='PASS', checks={check.__name__:check() for check in (
        check_schedules_and_validation, check_frozen_and_clean_identity,
        check_boundaries_and_activity, check_support_tails_and_nulls)},
        frozen_evidence=_frozen_evidence())


def _assert_rejected(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError('invalid runtime contract accepted')


def _epoch_record(state):
    fields = ('job', 'epoch', 'global_step', 'counts', 'selected_epochs', 'peak_allocated_bytes',
              'peak_reserved_bytes', 'peak_full_process_bytes', 'clock_kind', 'next_epoch',
              'schedule_epoch', 'schedule', 'next_schedule', 'phase_counters',
              'adam_step_counters', 'last_parameter_activity', 'engineering_steps',
              'engineering_commits', 'scientific_epochs_completed', 'sample_epoch', 'sample_batches')
    record = {name:copy.deepcopy(state[name]) for name in fields}
    record.update(continuation_hash=state_digest(state),
                  selected_hashes={k:state_digest(v) for k,v in state['selected_models'].items()},
                  **{name+'_hash':state_digest(state[name]) for name in ('model','ema','optimizer','rng','loader')})
    return record


def _runtime_boundary(protocol, runner, arm):
    c = protocol.config('movielens', arm)
    before_epoch = 10 if arm == 'NQR-EARLY10' else 40
    formal = protocol.job('movielens', arm, 2022, 'CPU_FIXTURE_NOT_SOURCE_RECEIPT')
    job = dict(formal, purpose='engineering_check', fixture_id='temporal_populated_boundary_v1')
    del job['key']; job['key'] = digest(job)
    assert job['key'] != formal['key']
    route = _route(c)
    torch.randperm(13, generator=route[3])
    state = dict(global_step=0, counts=dict(optimizer_updates=0,ema_updates=0),
                 phase_counters=protocol.empty_phase_counters())
    batches = [_batch(40960), _batch(c['tail_rows'])]
    def step(target, metadata, ordinal, epoch, batch):
        x,y,rows = batch
        context = dict(_context(c, epoch, ordinal), sample_epoch=1,
                       sample_batch=1 if ordinal % 2 else c['steps_per_epoch'])
        before = runner.adam_steps(target[0],target[1])
        value = _adapter(target, x, y, c, context, rows, diagnose=True)
        runner.observe_step(metadata,target[0],target[1],before,value)
        assert metadata['global_step'] == ordinal
    for ordinal,batch in enumerate(batches,1):
        step(route,state,ordinal,before_epoch,batch)
    state.update(job=job, epoch=1, clock_kind='engineering_virtual', next_epoch=None,
                 schedule_epoch=before_epoch,
                 schedule=dict(core.schedule(arm,before_epoch),clock_kind='engineering_virtual'),
                 next_schedule=dict(core.schedule(arm,before_epoch+1),clock_kind='engineering_virtual'),
                 selected_models={},selected_epochs={},selected_validation={},
                 peak_allocated_bytes=0,peak_reserved_bytes=0,peak_full_process_bytes=0,
                 engineering_steps=2,engineering_commits=1,scientific_epochs_completed=0,
                 sample_epoch=1,sample_batches=[1,c['steps_per_epoch']])
    saved = dict(_snapshot(route), **cpu_copy(state))
    protocol.validate_epoch_state(saved,_epoch_record(saved))
    for key,value in [('clock_kind','formal'),('global_step',before_epoch*c['steps_per_epoch']),
                      ('next_epoch',before_epoch+1),('scientific_epochs_completed',before_epoch)]:
        bad = cpu_copy(saved); bad[key] = value
        _assert_rejected(lambda:protocol.validate_epoch_state(bad,_epoch_record(bad)))
    for field in _PHASE_KEYS:
        bad = cpu_copy(saved); bad['phase_counters'][field] += 1
        _assert_rejected(lambda:protocol.validate_epoch_state(bad,_epoch_record(bad)))
    bad = cpu_copy(saved); bad['counts']['active_updates'] = 2
    _assert_rejected(lambda:protocol.validate_epoch_state(bad,_epoch_record(bad)))
    bad = cpu_copy(saved); bad['job'].pop('purpose')
    _assert_rejected(lambda:protocol.validate_epoch_state(bad,_epoch_record(bad)))

    buffer = io.BytesIO()
    torch.save(saved,buffer); buffer.seek(0)
    transported = torch.load(buffer,map_location='cpu',weights_only=False)
    saved_hash = state_digest(saved)
    assert state_digest(transported) == saved_hash
    clones = [_route(c),_route(c)]
    for clone in clones:
        runner.restore(clone[0],clone[2],clone[1],clone[3],transported,'cpu')
    for key,values in saved['optimizer']['state'].items():
        for name in ('step','exp_avg','exp_avg_sq'):
            tensors = [values[name], transported['optimizer']['state'][key][name]] + [
                clone[1].state_dict()['state'][key][name] for clone in clones]
            assert len({t.untyped_storage().data_ptr() for t in tensors}) == len(tensors)
    metadata = [cpu_copy(state) for _ in range(3)]
    for target,meta in zip([route]+clones,metadata):
        for ordinal,batch in enumerate(batches,3):
            step(target,meta,ordinal,before_epoch+1,batch)
        meta.update(epoch=2,schedule_epoch=before_epoch+1,
                    schedule=dict(core.schedule(arm,before_epoch+1),clock_kind='engineering_virtual'),
                    next_schedule=None,engineering_steps=4,engineering_commits=2)
        snapshot = dict(_snapshot(target),**cpu_copy(meta))
        protocol.validate_epoch_state(snapshot,_epoch_record(snapshot))
        assert meta['global_step'] == 4
    for clone,meta in zip(clones,metadata[1:]):
        _same(route,clone)
        assert state_digest(meta) == state_digest(metadata[0])
    orders = [torch.randperm(13,generator=r[3]) for r in [route]+clones]
    assert all(torch.equal(orders[0],order) for order in orders[1:])
    assert state_digest(saved) == state_digest(transported) == saved_hash
    return dict(status='PASS',arm=arm,actual_steps=4,commit_steps=[2,4],virtual_epochs=[before_epoch,before_epoch+1],
                populated_Adam_restore_routes=2,saved_state_unchanged=True,
                optimizer_step_and_moments_storage_disjoint=True,BytesIO_roundtrip=True,
                next_update_order_rng_ema_phase_activity_exact=True)


def check_runtime_contracts():
    package = Path(__file__).parent
    required = ('__init__.py','core.py','checks.py','protocol.py','runner.py','tables.py','README.md')
    if not all((package/name).is_file() for name in required) or not (_CYCLES/'cycle_03/RUNTIME_IMPLEMENTATION_REPORT.md').is_file():
        raise RuntimeError('Integration pending: all seven files and runtime coder report required before runtime imports')
    from . import protocol, runner

    for dataset in ('movielens','kuairand'):
        for arm in core.ARMS:
            assert protocol.config(dataset,arm) == _config(arm,dataset)
    attempt = _CYCLES/'cycle_03/attempt1'
    assert protocol.cycle_path(attempt) == attempt
    assert protocol.local_path(attempt,attempt/'runs/example.json') == attempt/'runs/example.json'
    for path in (_CYCLES/'cycle_02/attempt1',Path('attempt1'),attempt/'../other'):
        _assert_rejected(lambda:protocol.cycle_path(path))
    for path in (_CYCLES/'cycle_02/attempt1/result.json',Path('runs/example.json'),attempt/'../outside.json'):
        _assert_rejected(lambda:protocol.local_path(attempt,path))
    # These are pure arithmetic clock fixtures, not trained snapshots or claimed scientific prefixes.
    clock_cases = 0
    for dataset,N,S in [('movielens',800167,20),('kuairand',2006132,49)]:
        for arm in core.ARMS:
            j = dict(dataset=dataset,arm=arm,config=protocol.config(dataset,arm))
            for e,b in [(1,0),(1,1),(10,S),(11,0),(11,1),(40,S),(41,0),(41,1),(50,S)]:
                active_prior = sum(core.schedule(arm,t)['scheduled_active'] for t in range(1,e))
                active = core.schedule(arm,e)['scheduled_active']
                total = (e-1)*S+b
                rows = (e-1)*N+min(40960*b,N)
                active_updates = active_prior*S+active*b
                active_rows = active_prior*N+active*min(40960*b,N)
                phase = dict(active_updates=active_updates,off_updates=total-active_updates,
                    active_rows=active_rows,off_rows=rows-active_rows,active_task_entries=2*active_rows,
                    off_task_entries=2*(rows-active_rows),lambda_update_sum=.25*active_updates,
                    nonzero_opportunity_updates=0)
                counts = dict(optimizer_updates=total,ema_updates=total)
                protocol.validate_formal_clock(j,e,b,total,counts,phase)
                _assert_rejected(lambda:protocol.validate_formal_clock(j,e,b,total+1,counts,phase))
                clock_cases += 1
    admission = runner.runtime_admission_checks()
    assert admission['status'] == 'PASS' and admission['gpu_queries'] == admission['ipc_calls'] == 0
    return dict(status='PASS',config_combinations=4,new_absolute_root=True,
                formal_clock_arithmetic_cases=clock_cases,admission=admission,
                populated_boundary_restores=[_runtime_boundary(protocol,runner,arm) for arm in core.ARMS],
                gpu_queries=0,ipc_calls=0,test_accessed=False)


if __name__ == '__main__':
    print(json.dumps(dict(core=run_checks(),runtime=check_runtime_contracts()),indent=2,allow_nan=False))
