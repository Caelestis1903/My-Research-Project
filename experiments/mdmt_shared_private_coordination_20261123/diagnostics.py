"""Validation panels and isolated native-Adam forks for the S1 anchors."""

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from experiments.conditional_history_shared_update_20261122.b_build_panels import _pairs, row_side
from experiments.conditional_history_shared_update_20261122.b_panel_identity import digest
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.evaluate import metrics
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import array_hash

from . import blocks, source


DATASETS = ('movielens', 'kuairand')
SEEDS = tuple(range(2022, 2027))
ANCHORS = (10, 25, 40)
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
SCOPES = ('P_CELL', 'P_AXIS', 'P_FULL')
PANELS = ('FULL', 'V_A', 'V_B')
READOUTS = ('online', 'fixed_ema')
NAMESPACE = 'MDMT_SHARED_PRIVATE_COORDINATION_20261123'
MAX_PAIRS = 4096
BATCH_SIZE = 40960
IDENTITY = ('dataset', 'seed', 'epoch', 'source_hash', 'data_hash', 'config_hash',
            'logical_id', 'attempt_id')
UPDATE_COUNTS = ('training_forwards', 'parameter_backward_calls', 'optimizer_steps', 'ema_updates')
FORKS = (
    ('J2', None, 'AB'), ('J2', None, 'BA'), ('J_ACC', None, 'ACC'),
    ('U_ONLY', None, 'ACC'),
    *((scope + '_ONLY', scope, 'ACC') for scope in SCOPES),
    *((arm, scope, orientation) for scope in SCOPES
      for arm in ('U→P', 'P→U') for orientation in ('AB', 'BA')),
)


@dataclass
class Panels:
    bundle: object
    rows: dict
    pairs: dict
    manifest: dict


def _pair_generator(dataset, panel, cell):
    key = f'{NAMESPACE}|probe|{dataset}|None|{panel}|{cell}_pairs'
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))


def _freeze_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, allow_nan=False, separators=(',', ':'))
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if json.loads(path.read_text()) != manifest:
        raise ValueError(f'Frozen validation panel identity differs for {path}')


def prepare(bundle, dataset, output_dir):
    """Freeze label-blind folds and seed-independent positive/negative pairs."""
    if dataset not in DATASETS:
        raise ValueError(f'Unregistered dataset {dataset}')
    ids, labels, domains = bundle.val_row_ids, bundle.val_y, bundle.val_x[:, -1]
    if (not len(ids) or labels.shape != (len(ids), 2) or len(domains) != len(ids)
            or not np.all(ids[1:] > ids[:-1]) or not np.isin(labels, (0, 1)).all()
            or not np.isin(domains, (0, 1, 2)).all()):
        raise ValueError('Expected ordered Validation IDs, binary labels and canonical domains')
    sides = row_side(dataset, ids)
    rows = {'FULL': np.arange(len(ids)), 'V_A': np.flatnonzero(sides == 0),
            'V_B': np.flatnonzero(sides == 1)}
    pairs, panel_entries = {}, {}
    for panel, indices in rows.items():
        cells = {}
        for number, cell in enumerate(CELLS):
            domain, task = divmod(number, 2)
            in_cell = indices[domains[indices] == domain]
            positive = in_cell[labels[in_cell, task] == 1]
            negative = in_cell[labels[in_cell, task] == 0]
            selected = _pairs(positive, negative, MAX_PAIRS, _pair_generator(dataset, panel, cell))
            pair_rows = ids[selected]
            pairs[panel, cell] = selected
            cells[cell] = dict(positive=len(positive), negative=len(negative),
                               status='EVALUABLE' if len(selected) else 'NOT_EVALUABLE',
                               pair_hash=array_hash(pair_rows), row_hash=array_hash(ids[in_cell]),
                               pair_ids=list(range(len(selected))), pair_row_ids=pair_rows.tolist())
        panel_entries[panel] = dict(row_ids=ids[indices].tolist(), row_hash=array_hash(ids[indices]),
                                    cells=cells, independent=panel != 'FULL')
    manifest = dict(dataset=dataset, namespace=NAMESPACE, pair_cap=MAX_PAIRS,
                    row_rule='11-22 row_side(dataset, raw_row_ids, seed=None)',
                    labels_hash=array_hash(labels), domain_hash=array_hash(domains), panels=panel_entries)
    manifest['panel_hash'] = digest(manifest)
    _freeze_manifest(Path(output_dir) / f'PANELS_{dataset}.json', manifest)
    return Panels(bundle, rows, pairs, manifest)


def _predictions(model, bundle, device, batch_size, *, logits):
    modes = [(module, module.training) for module in model.modules()]
    probability, projection = [], []
    start = time.perf_counter()
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(bundle.val_y), batch_size):
                x = torch.as_tensor(bundle.val_x[offset:offset + batch_size], device=device).T
                if logits:
                    prob, logit = source.forward_logits(model, x)
                    projection.append(logit.cpu().numpy())
                else:
                    prob = model(x)
                probability.append(prob.cpu().numpy())
    finally:
        for module, training in modes:
            module.training = training
    return (np.concatenate(probability), np.concatenate(projection) if logits else None,
            dict(seconds=time.perf_counter() - start, forwards=len(probability), rows=len(bundle.val_y)))


def evaluate(model, bundle, device):
    """Evaluate full Validation with the authoritative source metric calculation."""
    start = time.perf_counter()
    prediction, _, counters = _predictions(model, bundle, device, BATCH_SIZE, logits=False)
    result = metrics(bundle.val_y, prediction, bundle.val_x[:, -1])
    counters['seconds'] = time.perf_counter() - start
    return result, counters


def _endpoint_metrics(panels, prediction):
    y, domains = panels.bundle.val_y, panels.bundle.val_x[:, -1]
    full = metrics(y, prediction, domains)
    values = {'FULL': full}
    for panel in ('V_A', 'V_B'):
        rows = panels.rows[panel]
        if len(rows):
            result = metrics(y[rows], prediction[rows], domains[rows])
        else:
            result = {key: 0 if key.endswith(('_count', '_positive', '_negative')) else None for key in full}
        values[panel] = result
        for cell in CELLS:
            status = panels.manifest['panels'][panel]['cells'][cell]['status']
            result[f'{cell}_status'] = status
            if status == 'NOT_EVALUABLE':
                result[f'{cell}_fallback_panel'] = 'FULL'
                result[f'{cell}_fallback_independent'] = False
                for suffix in ('auc', 'logloss', 'count', 'positive', 'negative'):
                    result[f'{cell}_fallback_{suffix}'] = full[f'{cell}_{suffix}']
    for cell in CELLS:
        full[f'{cell}_status'] = panels.manifest['panels']['FULL']['cells'][cell]['status']
    return values


def _grades(probability, pairs, task):
    positive, negative = probability[pairs, task].T
    return (positive > negative).astype(np.int8) * 2 + (positive == negative)


def _shift(before, after, label):
    if not len(before):
        return {f'{label}_{field}': None for field in ('mean', 'mean_abs', 'rms')}
    delta = after.astype(np.float64) - before.astype(np.float64)
    return {f'{label}_mean': float(delta.mean()), f'{label}_mean_abs': float(np.abs(delta).mean()),
            f'{label}_rms': float(np.sqrt(np.square(delta).mean()))}


def _state_identity(model, opt, ema):
    return dict(model=source.state_digest(model.state_dict()), ema=source.state_digest(ema.state_dict()),
                adam=source.state_digest(opt.state_dict()),
                modes=source.state_digest([[(name, module.training) for name, module in net.named_modules()]
                                           for net in (model, ema)]))


def _grad_identity(model, ema):
    return source.state_digest([{name: p.grad for name, p in net.named_parameters()} for net in (model, ema)])


def forks(model, opt, ema, batches, panels, metadata, device, *, record_physical=None):
    """Apply each registered intervention to one isolated clone and return scalar receipts."""
    identity = {key: metadata[key] for key in IDENTITY}
    dataset, seed, epoch = (identity[key] for key in ('dataset', 'seed', 'epoch'))
    cfg = source.config(dataset)
    if (dataset != panels.manifest['dataset'] or seed not in SEEDS or epoch not in ANCHORS
            or len(batches) != 2 or any(len(y) != cfg['batch_size'] for x, y in batches)):
        raise ValueError('Forks require a registered anchor and two complete next-epoch batches')
    rng = source.rng_state(device)
    live = _state_identity(model, opt, ema)
    live.update(grads=_grad_identity(model, ema), rng=source.state_digest(rng))
    ownership = blocks.groups(model)
    endpoints, transitions, responses, costs, fork_states = [], [], [], [], []
    measured = {}
    domain = panels.bundle.val_x[:, -1]

    def readout(net, arm, scope, orientation, phase, name):
        prediction, logits, cost = _predictions(net, panels.bundle, device, cfg['batch_size'], logits=True)
        if not (np.isfinite(prediction).all() and np.isfinite(logits).all()):
            raise ValueError('Nonfinite fork prediction or routed logit')
        start = time.perf_counter()
        columns = dict(**identity, arm=arm, scope=scope, orientation=orientation, phase=phase, readout=name)
        values = _endpoint_metrics(panels, prediction)
        for panel in PANELS:
            info = panels.manifest['panels'][panel]
            endpoints.append(dict(**columns, panel=panel, row_hash=info['row_hash'],
                                  panel_hash=panels.manifest['panel_hash'], independent=info['independent'],
                                  **values[panel]))
        measured[arm, scope, orientation, phase, name] = prediction, logits, values
        return dict(eval_forwards=cost['forwards'], eval_rows=cost['rows'], prediction_seconds=cost['seconds'],
                    metric_seconds=time.perf_counter() - start)

    def compare(changed, reference, effect):
        arm, scope, orientation, phase, name = changed
        pred, logits, values = measured[changed]
        prior_pred, prior_logits, prior_values = measured[reference]
        columns = dict(**identity, arm=arm, scope=scope, orientation=orientation, phase=phase, readout=name,
                       reference_arm=reference[0], reference_scope=reference[1],
                       reference_orientation=reference[2], reference_phase=reference[3], effect=effect)
        for panel in PANELS:
            info = panels.manifest['panels'][panel]
            for number, cell in enumerate(CELLS):
                d, task = divmod(number, 2)
                rows = panels.rows[panel]
                indices = rows[domain[rows] == d]
                pairs = panels.pairs[panel, cell]
                grades_before = _grades(prior_pred, pairs, task)
                grades_after = _grades(pred, pairs, task)
                table = np.bincount(3 * grades_before + grades_after, minlength=9).reshape(3, 3)
                item = info['cells'][cell]
                shared = dict(**columns, panel=panel, cell=cell, independent=info['independent'],
                              cell_status=item['status'], pair_hash=item['pair_hash'], row_hash=item['row_hash'],
                              positive=item['positive'], negative=item['negative'])
                transitions.append(dict(**shared, pairs=len(pairs), corrected=int(np.triu(table, 1).sum()),
                                        damaged=int(np.tril(table, -1).sum()), unchanged=int(table.trace()),
                                        **{f'grade_{a}_to_{b}': int(table[i, j])
                                           for i, a in enumerate(('0', '0.5', '1'))
                                           for j, b in enumerate(('0', '0.5', '1'))}))
                margin_before = prior_logits[pairs[:, 0], task].astype(np.float64) - prior_logits[pairs[:, 1], task]
                margin_after = logits[pairs[:, 0], task].astype(np.float64) - logits[pairs[:, 1], task]
                effects = {}
                for metric in ('auc', 'logloss'):
                    key = f'{cell}_{metric}'
                    a, b = prior_values[panel][key], values[panel][key]
                    effects[f'delta_{metric}'] = None if a is None or b is None else b - a
                responses.append(dict(**shared, **effects,
                                      **_shift(prior_logits[indices, task], logits[indices, task], 'logit_shift'),
                                      **_shift(margin_before, margin_after, 'pair_margin_shift')))

    def cost_row(arm, scope, orientation, phase, name, seconds, **counts):
        return dict(**identity, arm=arm, scope=scope, orientation=orientation, phase=phase, readout=name,
                    seconds=seconds, **counts)

    def record_cost(arm, scope, orientation, phase, name, start, *, seconds=None, **counts):
        row = cost_row(arm, scope, orientation, phase, name,
                       time.perf_counter() - start if seconds is None else seconds, **counts)
        costs.append(row)
        if record_physical is not None and phase != 'update':
            record_physical(row)

    def verify_live():
        after = _state_identity(model, opt, ema)
        after.update(grads=_grad_identity(model, ema), rng=source.state_digest(source.rng_state(device)))
        if live != after:
            raise RuntimeError('S1 fork changed the live online, EMA, Adam, mode, gradient or RNG state')
        return after

    def update(branch, branch_opt, branch_ema, inputs, group, arm, scope, orientation, number):
        value = blocks.step(branch, branch_opt, branch_ema, inputs, cfg,
                            names=None if group is None else ownership[group])
        if record_physical is not None:
            record_physical(cost_row(arm, scope, orientation, 'update', None, value['seconds'],
                                     step_number=number, **{key: value[key] for key in UPDATE_COUNTS}))
        return value

    try:
        for name, net in zip(READOUTS, (model, ema)):
            start = time.perf_counter()
            cost = readout(net, 'ANCHOR', None, None, 'anchor', name)
            record_cost('ANCHOR', None, None, 'anchor', name, start, **cost)
        for arm, scope, orientation in FORKS:
            source.restore_rng(rng, device)
            start = time.perf_counter()
            branch, branch_opt, branch_ema = copy.deepcopy((model, opt, ema))
            pre = _state_identity(branch, branch_opt, branch_ema)
            if pre != {key: live[key] for key in pre}:
                raise RuntimeError('Fork did not inherit exact anchor state')
            record_cost(arm, scope, orientation, 'clone', None, start)
            steps = []
            if orientation == 'ACC':
                group = 'U' if arm == 'U_ONLY' else scope
                steps.append(update(branch, branch_opt, branch_ema, batches, group, arm, scope, orientation, 1))
            else:
                order = batches if orientation == 'AB' else batches[::-1]
                groups = {'J2': (None, None), 'U→P': ('U', scope), 'P→U': (scope, 'U')}
                for number, (batch, group) in enumerate(zip(order, groups[arm]), start=1):
                    steps.append(update(branch, branch_opt, branch_ema, (batch,), group,
                                        arm, scope, orientation, number))
                    if arm == 'U→P' and number == 1:
                        read_start = time.perf_counter()
                        cost = readout(branch, arm, scope, orientation, 'after_u', 'online')
                        record_cost(arm, scope, orientation, 'after_u', 'online', read_start, **cost)
            accounting = {key: sum(step[key] for step in steps) for key in UPDATE_COUNTS}
            if (accounting['training_forwards'], accounting['parameter_backward_calls'],
                accounting['optimizer_steps'], accounting['ema_updates']) != (2, 2, len(steps), 2):
                raise RuntimeError(f'Unexpected native step accounting for {arm}/{scope}/{orientation}')
            record_cost(arm, scope, orientation, 'update', None, start, **accounting,
                        seconds=sum(step['seconds'] for step in steps), steps=steps)
            for name, net in zip(READOUTS, (branch, branch_ema)):
                read_start = time.perf_counter()
                cost = readout(net, arm, scope, orientation, 'end', name)
                record_cost(arm, scope, orientation, 'end', name, read_start, **cost)
            post = _state_identity(branch, branch_opt, branch_ema)
            del branch, branch_opt, branch_ema
            source.restore_rng(rng, device)
            fork_states.append(dict(arm=arm, scope=scope, orientation=orientation, before=pre, after=post,
                                    live_after=verify_live()))
        anchor = ('ANCHOR', None, None, 'anchor')
        for arm, scope, orientation in FORKS:
            for name in READOUTS:
                end = (arm, scope, orientation, 'end', name)
                compare(end, (*anchor, name), 'from_anchor')
                if arm != 'U→P':
                    continue
                if name == 'online':
                    after_u = (arm, scope, orientation, 'after_u', name)
                    compare(after_u, (*anchor, name), 'shared_update')
                    compare(end, after_u, 'post_u_recovery')
                for control in (('U_ONLY', None, 'ACC'), ('P→U', scope, orientation),
                                ('J_ACC', None, 'ACC'), ('J2', None, orientation)):
                    compare(end, (*control, 'end', name), 'endpoint_contrast')
    finally:
        source.restore_rng(rng, device)
    return dict(metadata=metadata, panel_hash=panels.manifest['panel_hash'], endpoints=endpoints,
                transitions=transitions, responses=responses, costs=costs,
                purity=dict(live_before=live, live_after=verify_live(), fork_states=fork_states))
