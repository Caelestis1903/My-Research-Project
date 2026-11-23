import copy
import csv
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.mdmt_shared_private_coordination_20261123 import blocks, diagnostics as diag, source, statistics as stats
from experiments.mdmt_shared_private_coordination_20261123 import scheduler, training


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor(.2))
        self.domain_expert = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.task_expert = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])
        self.tower = nn.ModuleList([nn.Linear(1, 1) for _ in range(6)])

    def forward(self, x):
        if self.training:
            random.random()
            np.random.random()
            torch.rand(())
        features = x[:1].T * self.shared_weight
        outputs = [tower(features + self.domain_expert[i % 3](features) + self.task_expert[i // 3](features))
                   .squeeze(1).sigmoid() for i, tower in enumerate(self.tower)]
        predictions = torch.stack(outputs, 1)
        return predictions.gather(1, x[-1].long().unsqueeze(1) + x.new_tensor([0, 3]).long())


def _bundle(dataset):
    ids = np.arange(1000, 1120)
    domains = np.repeat([0, 1, 2], 40)
    features = np.stack((np.sin(ids), domains), 1).astype(np.float32)
    labels = np.stack((ids % 2, (ids // 2) % 2), 1).astype(np.float32)
    labels[(domains == 2) & (diag.row_side(dataset, ids) == 0), 1] = 1
    return SimpleNamespace(val_x=features, val_y=labels, val_row_ids=ids)


def test_frozen_pairs_fallback_and_purity(tmp_path, monkeypatch):
    dataset = 'movielens'
    bundle = _bundle(dataset)
    device = torch.device('cpu')
    before_rng = source.state_digest(source.rng_state(device))
    panels = diag.prepare(bundle, dataset, tmp_path)
    assert source.state_digest(source.rng_state(device)) == before_rng
    assert diag.prepare(bundle, dataset, tmp_path).manifest == panels.manifest
    assert np.intersect1d(panels.rows['V_A'], panels.rows['V_B']).size == 0
    assert np.array_equal(np.union1d(panels.rows['V_A'], panels.rows['V_B']), panels.rows['FULL'])
    for (panel, cell), pairs in panels.pairs.items():
        task = int(cell[-1])
        assert len(np.unique(pairs, axis=0)) == len(pairs) <= 4096
        assert np.isin(pairs, panels.rows[panel]).all()
        assert (bundle.val_y[pairs[:, 0], task] == 1).all()
        assert (bundle.val_y[pairs[:, 1], task] == 0).all()
    changed = copy.deepcopy(bundle)
    changed.val_y[0, 0] = 1 - changed.val_y[0, 0]
    with pytest.raises(ValueError, match='Frozen validation panel identity'):
        diag.prepare(changed, dataset, tmp_path)

    cfg = dict(batch_size=6, weight_decay=.001, rho=.99)
    monkeypatch.setattr(source, 'config', lambda _: cfg)
    model = Toy()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=cfg['weight_decay'])
    model.tower[0].eval()
    batches = tuple((torch.from_numpy(bundle.val_x[ids]).T, torch.from_numpy(bundle.val_y[ids]))
                    for ids in (np.arange(0, 120, 20), np.arange(1, 120, 20)))
    blocks.step(model, opt, ema, (batches[0],), cfg)
    metrics, counts = diag.evaluate(model, bundle, device)
    assert counts['rows'] == 120 and counts['forwards'] == 1
    assert model.training and not model.tower[0].training
    metadata = dict(dataset=dataset, seed=2022, epoch=10, source_hash='source', data_hash='data',
                    config_hash='config', logical_id='logical', attempt_id='attempt')
    before_rng = source.state_digest(source.rng_state(device))
    physical_costs = []
    receipt = diag.forks(model, opt, ema, batches, panels, metadata, device, record_physical=physical_costs.append)
    assert source.state_digest(source.rng_state(device)) == before_rng
    assert receipt['purity']['live_before'] == receipt['purity']['live_after']
    assert len(receipt['purity']['fork_states']) == len(set(diag.FORKS)) == 19
    assert sum(cost.get('training_forwards', 0) for cost in receipt['costs']) == 38
    assert sum(cost.get('ema_updates', 0) for cost in receipt['costs']) == 38
    assert sum(cost.get('eval_rows', 0) for cost in receipt['costs']) == (2 + 2 * 19 + 6) * len(bundle.val_y)
    for key in (*diag.UPDATE_COUNTS, 'eval_forwards', 'eval_rows'):
        assert sum(row.get(key, 0) for row in physical_costs) == sum(row.get(key, 0) for row in receipt['costs'])
    assert len(receipt['endpoints']) == len(stats._expected_endpoints())
    online = next(row for row in receipt['endpoints']
                  if (row['arm'], row['readout'], row['panel']) == ('ANCHOR', 'online', 'FULL'))
    assert online['macro_auc'] == metrics['macro_auc']
    fallback = next(row for row in receipt['endpoints']
                    if (row['arm'], row['readout'], row['panel']) == ('ANCHOR', 'online', 'V_A'))
    assert fallback['d2t1_status'] == 'NOT_EVALUABLE' and fallback['d2t1_auc'] is None
    assert fallback['d2t1_fallback_auc'] == metrics['d2t1_auc']
    assert fallback['d2t1_fallback_independent'] is False
    assert all(t['pairs'] == sum(t[c] for c in ('corrected', 'damaged', 'unchanged')) for t in receipt['transitions'])
    assert {row['effect'] for row in receipt['responses']} == {
        'from_anchor', 'shared_update', 'post_u_recovery', 'endpoint_contrast'}
    assert all(row['readout'] == 'online' for row in receipt['responses']
               if row['effect'] in ('shared_update', 'post_u_recovery'))
    assert all(row['reference_phase'] == 'after_u' for row in receipt['responses']
               if row['effect'] == 'post_u_recovery')
    assert json.loads(json.dumps(receipt, allow_nan=False))['metadata'] == metadata


def test_failed_anchor_preserves_completed_physical_work(tmp_path, monkeypatch):
    dataset = 'movielens'
    device = torch.device('cpu')
    bundle = _bundle(dataset)
    bundle.train_x, bundle.train_y, bundle.train_row_ids = bundle.val_x, bundle.val_y, bundle.val_row_ids
    cfg = dict(batch_size=6, weight_decay=.001, rho=.99, epochs=50)
    panel = diag.prepare(bundle, dataset, tmp_path).manifest
    job = dict(dataset=dataset, seed=2022, arm='J_ORIG', logical_id='fixture', source_hash='source',
               data_hash='data', config_hash='config', panel_hash=panel['panel_hash'], config=cfg)
    monkeypatch.setattr(source, 'config', lambda _: cfg)
    monkeypatch.setattr(source, 'load_data', lambda _: (bundle, dict(data_hash='data')))
    monkeypatch.setattr(source, 'make_model', lambda *_: Toy())
    monkeypatch.setattr(source, 'optimizer', lambda model, _: torch.optim.Adam(model.parameters(), lr=.01))
    model = Toy()
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = source.optimizer(model, cfg)
    loader = torch.Generator().manual_seed(13022)
    order = training.next_order(len(bundle.train_y), loader)
    preview = training.v3.state_digest(training.preview_order(len(bundle.train_y), loader))
    packet = training.capture(model, opt, ema, loader, order, device, epoch=10, job=job, owner='owner',
                              selected={}, epochs=[], counts={}, next_prediction=preview)
    storage = {}

    def commit(_owner, key, _sequence, path, state, _costs):
        storage[key] = copy.deepcopy(state)
        receipt = dict(key=key)
        training.write(path, receipt)
        return receipt

    native_step = blocks.step
    calls = 0

    def fail_on_second_step_of_next_fork(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('fixture failure in second fork')
        return native_step(*args, **kwargs)

    monkeypatch.setattr(blocks, 'step', fail_on_second_step_of_next_fork)
    monkeypatch.setattr(training, 'owner_identity', lambda _: 'owner')
    monkeypatch.setattr(training, 'register', lambda *args: None)
    monkeypatch.setattr(training, 'request', lambda *args, **kwargs: None)
    monkeypatch.setattr(training, 'recover', lambda *args: (packet, dict(key='previous')))
    monkeypatch.setattr(training, 'commit', commit)
    monkeypatch.setattr(training, 'verify', lambda _owner, receipt: storage[receipt['key']])
    monkeypatch.setattr(training.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *_: None)

    with pytest.raises(RuntimeError, match='fixture failure in second fork'):
        training.worker(tmp_path, tmp_path / 'ram', job, 'retry', device, lambda: False)

    out = tmp_path / 'runs/fixture'
    attempt = training.read(out / 'attempts/retry.json')
    journal = training.read(out / 'attempts/retry/FORK_COSTS.json')
    assert not (out / 'forks_E10.json').exists()
    assert attempt['status'] == 'FAILED' and attempt['counts'] == {}
    assert attempt['diagnostic_counts'] == journal['counts']
    assert all(journal['counts'][key] == 3 for key in diag.UPDATE_COUNTS)
    assert journal['counts']['eval_rows'] == 4 * len(bundle.val_y)
    assert journal['counts']['eval_forwards'] == 4 * len(bundle.val_y) // cfg['batch_size']
    assert journal['counts']['seconds'] > 0
    assert [(row['orientation'], row['step_number']) for row in journal['costs'] if row['phase'] == 'update'] == [
        ('AB', 1), ('AB', 2), ('BA', 1)]

    accepted = dict(attempt, attempt_id='accepted', status='COMPLETE', diagnostic_counts={})
    training.write(out / 'attempts/accepted.json', accepted)
    accepted_costs = []
    result = dict(job=job, anchors={}, fork_hashes={}, selected_epochs=dict.fromkeys(training.KINDS, 1),
                  selected_hashes=dict.fromkeys(training.KINDS, 'selected'))
    for epoch in training.ANCHORS:
        key = f'accepted-E{epoch}'
        path = out / f'forks_E{epoch:02d}.json'
        row = dict(arm='J2', attempt_id='accepted', readout='online', epoch=epoch,
                   phase='end', eval_forwards=20, seconds=.1)
        training.write(path, dict(metadata=dict(fork_key=key), costs=[row], resources=dict(process_gpu_bytes=1)))
        result['anchors'][str(epoch)] = dict(bytes=1)
        result['fork_hashes'][path.name] = training.sha(path)
        accepted_costs.append(dict(row, fork_key=key))
    training.write(out / 'attempts/accepted/FORK_COSTS.json', dict(costs=accepted_costs))
    for epoch in range(1, 51):
        training.write(out / f'epoch_{epoch:02d}.json',
                       dict(attempt_id='accepted', validation=dict.fromkeys(training.KINDS, {})))
    scheduler.publish(tmp_path, [result])
    with (tmp_path / 'COSTS.csv').open() as stream:
        costs = list(csv.DictReader(stream))
    retried = [row for row in costs if row['cost_kind'] == 'fork_attempt']
    assert len(retried) == len(journal['costs'])
    assert all(row['attempt_id'] == 'retry' and row['status'] == 'FAILED' for row in retried)
    assert sum(int(row['optimizer_steps'] or 0) for row in retried) == 3
    assert sum(float(row['seconds']) for row in retried) == pytest.approx(journal['counts']['seconds'])
    assert len([row for row in costs if row['cost_kind'] == 'fork']) == len(training.ANCHORS)


def _scalar_index(gains):
    index, jobs = {}, {}
    for dataset in diag.DATASETS:
        for seed in diag.SEEDS:
            jobs[dataset, seed] = dict(source_hash='source', data_hash=dataset, config_hash=dataset,
                                      dataset=dataset, seed=seed, arm='J_ORIG', logical_id=f'{dataset}-{seed}')
            for epoch in diag.ANCHORS:
                rows = {}
                for key in sorted(stats._expected_endpoints(), key=str):
                    arm, scope, order, phase, readout, panel = key
                    offset = gains(dataset, seed, epoch, panel, readout, scope, order) if arm == 'U→P' else 0.
                    offsets = offset if isinstance(offset, (list, tuple)) else [offset] * 6
                    cells = [None if delta is None else .5 + delta for delta in offsets]
                    rows[key] = dict(zip(stats.AUC_METRICS, [stats._mean(cells), *cells]))
                index[dataset, seed, epoch] = rows
    return index, jobs


def _decision(gains):
    return stats.scope_decision(stats._comparisons(*_scalar_index(gains))[0])


def test_va_freezes_scope_without_ema_or_vb_reselection():
    def gains(dataset, seed, epoch, panel, readout, scope, order):
        if readout == 'fixed_ema':
            return .1 if scope == 'P_CELL' else 0.
        if panel == 'V_B':
            return -.0015 if scope == 'P_FULL' else .03
        return .02 if scope == 'P_FULL' else .01

    decision = _decision(gains)
    assert decision['frozen_scope'] == 'P_FULL'
    assert decision['status'] == 'COMPLETE_NO_GO_S1'
    assert decision['confirmed_datasets'] == []


def test_orders_anchors_and_cell_guard():
    def gains(dataset, seed, epoch, panel, readout, scope, order):
        if panel == 'V_A' and scope == 'P_FULL':
            return [-.002, .05, .05, .05, .05, .05]
        if panel == 'V_A' and scope == 'P_CELL':
            return .0001 if order == 'AB' else -.001
        if panel == 'V_B' and epoch != 10:
            return -.001
        return .01

    decision = _decision(gains)
    assert decision['frozen_scope'] == 'P_AXIS'
    assert decision['v_b']['movielens']['replicated_anchors'] == ['E10']
    assert decision['status'] == 'COMPLETE_NO_GO_S1'

    def clock_gains(dataset, seed, epoch, panel, readout, scope, order):
        return .02 if scope in ('P_CELL', 'P_AXIS') else .01

    decision = _decision(clock_gains)
    assert decision['frozen_scope'] == 'P_CELL'
    assert decision['s1_gate'] == 'GO' and decision['status'] == 'TECHNICAL_BLOCKED'
    assert decision['s2_clock_blocker'] == 'P_AXIS'


def test_equal_dataset_gain_and_conditional_negative_interval():
    def gains(dataset, seed, epoch, panel, readout, scope, order):
        if panel == 'V_A':
            if scope == 'P_CELL':
                return .04 if dataset == 'movielens' else -.035
            return .012 if scope == 'P_FULL' else 0.
        if dataset == 'movielens':
            return .02
        return (-.004, -.003, -.002, .002, .002)[seed - 2022]

    comparisons, rows = stats._comparisons(*_scalar_index(gains))
    result = stats.scope_decision(comparisons)
    assert result['frozen_scope'] == 'P_FULL'
    assert result['status'] == 'GO' and result['confirmed_datasets'] == ['movielens']
    row = next(row for row in rows if (row['dataset'], row['epoch'], row['panel'], row['readout'], row['scope'],
                                      row['metric'], row['seed']) == ('equal_dataset', 'anchor_mean', 'V_B', 'online',
                                                                   'P_FULL', 'macro_auc', 2022))
    assert row['delta'] == pytest.approx((.02 - .004) / 2)
    assert row['leave_seed_out_mean'] == pytest.approx(.02 / 2 + (-.003 - .002 + .002 + .002) / 8)

    def negative(*args):
        return -.002 if (args[0], args[3]) == ('kuairand', 'V_B') else gains(*args)

    assert _decision(negative)['reason'] == 'SIGNIFICANTLY_NEGATIVE_ON_OTHER_DATASET'


def test_incomplete_or_missing_class_cannot_supply_selection(tmp_path):
    def gains(dataset, seed, epoch, panel, readout, scope, order):
        return [None, .03, .03, .03, .03, .03] if panel == 'V_A' else .03

    decision = _decision(gains)
    assert decision['frozen_scope'] is None
    _, jobs = _scalar_index(gains)
    stats.write(tmp_path / 'S1_RUN.json', dict(status='COMPLETE'))
    stats.write(tmp_path / 'S1_RUN_MANIFEST.json', dict(jobs=list(jobs.values())[:-1]))
    with pytest.raises(ValueError, match='each dataset and seed'):
        stats.analyze(tmp_path)


def test_scalar_artifact_round_trip_checks_fork_arms(tmp_path):
    index, jobs = _scalar_index(lambda *args: .02 if args[-2] == 'P_FULL' else .01)
    stats.write(tmp_path / 'S1_RUN.json', dict(status='COMPLETE'))
    stats.write(tmp_path / 'S1_RUN_MANIFEST.json', dict(jobs=list(jobs.values())))
    for (dataset, seed), job in jobs.items():
        run = tmp_path / 'runs' / job['logical_id']
        hashes = {}
        for epoch in diag.ANCHORS:
            meta = {key: job[key] for key in diag.IDENTITY if key in job}
            meta.update(epoch=epoch, attempt_id='fixture')
            panel_hash = 'fixture-' + dataset
            receipt = dict(metadata=meta, panel_hash=panel_hash, endpoints=[], transitions=[], responses=[],
                           purity=dict(live_before={'model': 'fixture'}, live_after={'model': 'fixture'},
                                       fork_states=[dict(arm=arm, scope=scope, orientation=order,
                                                         live_after={'model': 'fixture'})
                                                    for arm, scope, order in diag.FORKS]))
            for key, values in index[dataset, seed, epoch].items():
                arm, scope, order, phase, readout, panel = key
                receipt['endpoints'].append(dict(**meta, arm=arm, scope=scope, orientation=order, phase=phase,
                                                 readout=readout, panel=panel, panel_hash=panel_hash, **values))
            path = run / f'forks_E{epoch}.json'
            stats.write(path, receipt)
            hashes[path.name] = stats.sha(path)
        result = dict(job=job, epochs=50, test_evaluations=0, fork_hashes=hashes)
        stats.write(run / 'result.json', result)
    decision = stats.analyze(tmp_path)
    assert decision['status'] == 'GO' and decision['confirmed_datasets'] == list(diag.DATASETS)
    assert json.loads((tmp_path / 'SCOPE_DECISION.json').read_text()) == decision
    assert (tmp_path / 'PAIRED_SEEDS.csv').stat().st_size > 0
    receipt['endpoints'].pop()
    stats.write(path, receipt)
    with pytest.raises(ValueError, match='fork identity differs'):
        stats.analyze(tmp_path)
    result['fork_hashes'][path.name] = stats.sha(path)
    stats.write(run / 'result.json', result)
    with pytest.raises(ValueError, match='Incomplete or unregistered endpoint arms'):
        stats.analyze(tmp_path)
