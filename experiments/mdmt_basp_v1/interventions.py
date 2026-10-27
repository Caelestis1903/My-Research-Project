"""Fixed reading interventions and one F-selected, matched reader continuation.

The parent Trainer is observed, never loaded or advanced. All branch states stay
in RAM and all Q labels are read after the intervention and H=8 plans are locked.
"""
from __future__ import annotations
from pathlib import Path
import copy
import csv
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .identity import capture_rng, restore_rng, restore, cpu_copy, write_json
from .permissions import make_optimizer
from .prefix import packet as cached_packet, synchronize
from .roles import tensor
from .losses import masked_bce
from .metrics import evaluate, weighted_auc
from .export import write_csv


def ram_equal(left, right):
    """Direct comparison includes module metadata, buffers, flags and tree maps."""
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(ram_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(ram_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, '__dict__'):
        return type(left) is type(right) and ram_equal(left.__dict__, right.__dict__)
    return left == right


def _same_training_state(first, second):
    # Wall-clock accounting cannot change a trained prediction or continuation.
    a, b = dict(first), dict(second)
    a['state'] = {k: v for k, v in first['state'].items() if k != 'cost'}
    b['state'] = {k: v for k, v in second['state'].items() if k != 'cost'}
    return ram_equal(a, b)


def paths(model):
    return sorted((int(d), int(source), str(leaf)) for d, actions in enumerate(model.action_maps)
                  for source, leaf in actions)


def path_name(path):
    return f'd{path[0]}/s{path[1]}/{path[2]}'


@torch.no_grad()
def components(model, packet, domain):
    """Cache the original state once; all actions keep its routing and policy."""
    output = model.measure_output(packet, domain)
    result = dict(reference=output.reference_logits.detach(), keep=output.logits.detach(),
                  q=output.need_mass.detach(), domain=domain.detach(), paths={})
    for d, part in output.details.items():
        rows = part['rows'].nonzero().flatten()
        coefficient = torch.einsum('bk,btka->bta', part['q'], part['alpha'])
        for a, (source, leaf) in enumerate(model.action_maps[d]):
            j = model.sources[source].leaves.index(leaf)
            values = output.source_values[rows, source, j].detach()
            phi = packet['phi'][rows, source].detach()
            result['paths'][(d, source, leaf)] = dict(rows=rows, coefficient=coefficient[..., a].detach(),
                fixed_coefficient=(part['prior'][:, a:a+1]/2).expand(-1, model.T).detach(),
                values=values, phi=phi, correction=part['c'][..., a].detach(),
                root_correction=F.linear(phi, model.readers[f'd{d}s{source}']).detach())
    return result


def fixed_logits(cached, mode='KEEP_FIXED', path=None):
    if mode == 'KEEP_FIXED':
        return cached['keep']
    if mode == 'ALL_NULL_FIXED':
        return cached['reference']
    if mode == 'FIXED_RELATION_FIXED':
        out = cached['reference'].clone()
        for item in cached['paths'].values():
            out[item['rows']] += item['fixed_coefficient']*item['correction']
        return out
    if path not in cached['paths']:
        raise ValueError('Intervention path is absent from this frozen state')
    item = cached['paths'][path]
    out = cached['keep'].clone()
    if mode == 'OFF_PATH':
        out[item['rows']] -= item['coefficient']*item['correction']
    elif mode == 'ROOT_PATH':
        out[item['rows']] += item['coefficient']*(item['root_correction']-item['correction'])
    else:
        raise ValueError('Unknown fixed intervention: '+mode)
    return out


def reader_logits(model, cached, mode='KEEP_READAPT', path=None):
    """Build the retained graph directly; removed-only w remains grad=None."""
    out = cached['reference'].clone()
    for identity, item in cached['paths'].items():
        if mode == 'OFF_PATH_READAPT' and identity == path:
            continue
        coefficient = item['coefficient']
        if not bool(coefficient.any()):
            continue
        d, source, leaf = identity
        value = item['phi'] if mode == 'ROOT_PATH_READAPT' and identity == path else item['values']
        correction = F.linear(value, model.readers[f'd{d}s{source}'])
        out[item['rows']] += coefficient*correction
    return out


def select_readapt_path(cached, y, mask, natural_domain_counts):
    valid = np.asarray(mask, bool)
    target = np.asarray(y, np.float64)
    keep = cached['keep'].cpu().numpy().astype(np.float64)
    base = np.logaddexp(0., keep)-target*keep
    domain = cached['domain'].cpu().numpy()
    counts = np.asarray(natural_domain_counts, np.float64)
    proportions = counts/counts.sum()
    rows = []
    for path in sorted(cached['paths']):
        removed = fixed_logits(cached, 'OFF_PATH', path).cpu().numpy().astype(np.float64)
        difference = np.logaddexp(0., removed)-target*removed-base
        chosen = (domain == path[0])[:, None] & valid
        score = float(proportions[path[0]]*difference[chosen].mean()) if chosen.any() else 0.
        rows.append(dict(path=list(path), path_id=path_name(path), score=score,
                         probe_valid_labels=int(chosen.sum()), natural_domain_mass=float(proportions[path[0]])))
    positive = [row for row in rows if row['score'] > 0]
    winner = min(positive, key=lambda row: (-row['score'], tuple(row['path']))) if positive else None
    return None if winner is None else tuple(winner['path']), rows


def _packet(trainer, model, role, row_ids):
    view = trainer.views[role]
    pos = np.searchsorted(view['ids'], row_ids)
    value = cached_packet(view, pos, trainer.device)
    if trainer.full:
        value['h0'], value['z0'] = value['h'], value['z']
        with torch.no_grad():
            current = model.current_native.encode(tensor(trainer.data.x[row_ids], trainer.device))
            value['h'], value['z'] = current.h, current.logits
    return value


def _branch(snapshot, trainer):
    model, optimizer, _ = restore(snapshot, trainer.device, lambda m: make_optimizer(m, trainer.cfg))
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('readers.'))
    return model, optimizer


def continue_readers(snapshot, trainer, batches, chosen_path):
    """Three original-Adam branches, identical ordered F batches, H exactly eight."""
    if len(batches) != 8:
        raise ValueError('Reader continuation requires exactly eight planned batches')
    residents, records = {}, []
    initial_optimizer = None
    for mode in ('KEEP_READAPT', 'OFF_PATH_READAPT', 'ROOT_PATH_READAPT'):
        model, optimizer = _branch(snapshot, trainer)
        initial = cpu_copy(optimizer.state_dict())
        if initial_optimizer is None:
            initial_optimizer = initial
        elif not ram_equal(initial_optimizer, initial):
            raise AssertionError('Reader branches did not inherit the same Adam state')
        original_nonreaders = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                               if not name.startswith('readers.')}
        original_nonreader_adam = {name: cpu_copy(optimizer.state[parameter]) for name, parameter in model.named_parameters()
                                  if not name.startswith('readers.') and parameter in optimizer.state}
        initial_steps = {name: float(optimizer.state[p]['step']) if p in optimizer.state and 'step' in optimizer.state[p] else 0.
                         for name, p in model.named_parameters() if name.startswith('readers.')}
        steps, losses, updated = 0, [], []
        started = time.monotonic()
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            prediction = reader_logits(model, batch['components'], mode, chosen_path)
            loss = masked_bce(prediction, batch['y'], batch['mask'])
            names = []
            if loss.requires_grad:
                loss.backward()
                names = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
                if any(not name.startswith('readers.') for name in names):
                    raise AssertionError('Reader intervention changed a forbidden gradient')
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite reader continuation gradient')
                optimizer.step()
                model.project_readers()
                steps += bool(names)
            losses.append(float(loss.detach()))
            updated.append(names)
        for name, parameter in model.named_parameters():
            if name in original_nonreaders and not torch.equal(original_nonreaders[name], parameter.detach().cpu()):
                raise AssertionError('A nonreader parameter changed in the reader intervention')
            if name in original_nonreader_adam and not ram_equal(original_nonreader_adam[name], cpu_copy(optimizer.state[parameter])):
                raise AssertionError('A frozen nonreader Adam history advanced')
        model.requires_grad_(False).eval()
        synchronize(trainer.device)
        records.append(dict(branch=mode, attempted_steps=8, actual_reader_updates=steps,
                            original_reader_adam_steps=initial_steps, losses=losses, updated_parameters=updated,
                            same_original_adam_state=True,
                            frozen_nonreader_adam_unchanged=True,
                            training_seconds=time.monotonic()-started,
                            frozen_after_eight_steps=True))
        residents[mode] = (model, optimizer)
    return residents, records


def _measure_arrays(cached, path_order):
    n, tasks = cached['reference'].shape
    arrays = dict(keep_logits=cached['keep'].cpu().numpy(), reference_logits=cached['reference'].cpu().numpy(),
                  need_mass=cached['q'].cpu().numpy(), domain=cached['domain'].cpu().numpy())
    for key, target in (('coefficient', 'read_coefficient'), ('correction', 'leaf_correction'),
                        ('root_correction', 'root_correction'), ('fixed_coefficient', 'fixed_coefficient')):
        out = np.zeros((n, tasks, len(path_order)), cached['keep'].cpu().numpy().dtype)
        for j, path in enumerate(path_order):
            item = cached['paths'][path]
            out[item['rows'].cpu().numpy(), :, j] = item[key].cpu().numpy()
        arrays[target] = out
    return arrays


def _fixed_actions(path_order):
    return [('KEEP_FIXED', None), ('ALL_NULL_FIXED', None), ('FIXED_RELATION_FIXED', None)]+[
        (mode, path) for path in path_order for mode in ('OFF_PATH', 'ROOT_PATH')]


def _domain_weighted_loss(loss, domain, mask, natural_domain_counts):
    counts = np.asarray(natural_domain_counts, np.float64)
    if counts.sum() <= 0:
        return None
    proportions = counts/counts.sum()
    total = 0.
    for d, proportion in enumerate(proportions):
        if proportion <= 0:
            continue
        selected = (domain == d)[:, None] & mask
        if not selected.any():
            return None
        total += proportion*float(loss[selected].mean())
    return total


def _score_rows(identity, role, action, path, y, mask, logits, domain, q, model, baseline,
                natural_domain_counts=None):
    metric = evaluate(y, logits, domain, mask, domains=model.D)
    ll = np.logaddexp(0., logits.astype(np.float64))-y*logits
    natural = metric['natural_logloss']
    before_natural = baseline['natural_logloss']
    if natural_domain_counts is not None:
        natural = _domain_weighted_loss(ll, domain, mask, natural_domain_counts)
        before_natural = _domain_weighted_loss(baseline['_loss'], domain, mask, natural_domain_counts)
    common = dict(identity, role=role, action=action, path_id=path_name(path) if path else '',
                  target_domain=path[0] if path else None, source_domain=path[1] if path else None,
                  source_leaf=path[2] if path else None)
    rows = [dict(common, level='GLOBAL', macro_auc=metric['macro_auc'], macro_logloss=metric['macro_logloss'],
                 natural_logloss=natural, valid_labels=metric['valid_labels'],
                 natural_weighting='FULL_F_DOMAIN_WEIGHTED_PROBE_ESTIMATE' if natural_domain_counts is not None else 'NATURAL_ROLE_REQUEST_MEAN',
                 probe_mean_logloss=metric['natural_logloss'] if natural_domain_counts is not None else None,
                 delta_macro_auc=None if baseline['macro_auc'] is None or metric['macro_auc'] is None else metric['macro_auc']-baseline['macro_auc'],
                 delta_macro_logloss=metric['macro_logloss']-baseline['macro_logloss'],
                 delta_natural_logloss=None if natural is None or before_natural is None else natural-before_natural)]
    for cell, before in zip(metric['cells'], baseline['cells']):
        rows.append(dict(common, level='CELL', **cell,
                         delta_logloss=None if cell['logloss'] is None or before['logloss'] is None else cell['logloss']-before['logloss'],
                         delta_auc=None if cell['auc'] is None or before['auc'] is None else cell['auc']-before['auc']))
    before_ll = baseline['_loss']
    for d, tree in enumerate(model.needs):
        for k, leaf in enumerate(tree.leaves):
            responsibility = q[:, k]*(domain == d)
            hard = (q.argmax(1) == k) & (domain == d)
            for task in range(model.T):
                selected = mask[:, task] & (domain == d)
                w = responsibility[selected].astype(np.float64)
                mass = float(w.sum())
                yy, zz = y[selected, task], logits[selected, task]
                value = float(w@ll[selected, task]/mass) if mass > 0 else None
                previous = float(w@before_ll[selected, task]/mass) if mass > 0 else None
                rows.append(dict(common, level='NEED_LEAF', domain=d, task=task, need_leaf=leaf,
                                 n=int(selected.sum()), hard_n=int((hard & mask[:, task]).sum()),
                                 soft_mass=mass, ess=mass*mass/max(float(w@w), 1e-30),
                                 logloss=value, delta_logloss=None if value is None else value-previous,
                                 auc=weighted_auc(yy, zz, w) if mass > 0 else None))
    return rows


def _append_rows(path, rows):
    previous = []
    if Path(path).is_file():
        with Path(path).open(newline='') as stream:
            previous = list(csv.DictReader(stream))
    write_csv(path, previous+rows)


def run_read_interventions(trainer, which):
    """Call after independent addendum review; leaves the mother's RAM unchanged."""
    if which not in ('selected', 'final'):
        raise ValueError('Expected selected or final state')
    snapshot = getattr(trainer, which)
    root = Path(trainer.root)/'interventions'/which
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(trainer.identity, state=which, epoch=int(snapshot['state']['epoch']),
                    main_step=int(snapshot['state']['step']), diagnostic='READ_INTERVENTIONS_V1')
    if not getattr(snapshot['model'], 'foreign', False):
        result = dict(status='NOT_APPLICABLE', identity=identity, cost={})
        write_json(root/'INTERVENTION_PLAN.json', result)
        return result
    if which == 'final' and _same_training_state(trainer.selected, snapshot):
        source = Path(trainer.root)/'interventions'/'selected'/'INTERVENTION_PLAN.json'
        if source.is_file() and json.loads(source.read_text()).get('status') == 'COMPLETE':
            previous = json.loads(source.read_text())
            result = dict(previous, identity=identity, status='COMPLETE',
                          reused_from='selected', source_plan=str(source),
                          Q_label_reads=0, reused_measurement_Q_label_reads=previous.get('Q_label_reads'),
                          cost=dict(fixed_forward_seconds=0., branch_training_seconds=0.,
                                    evaluation_seconds=0., additional_reader_steps=0,
                                    reused_complete_training_state=True))
            write_json(root/'INTERVENTION_PLAN.json', result)
            with (Path(trainer.root)/'interventions'/'selected'/'FUNCTION_INTERVENTIONS.csv').open(newline='') as stream:
                rows = [dict(row, state='final', reused_from='selected') for row in csv.DictReader(stream)]
            write_csv(root/'FUNCTION_INTERVENTIONS.csv', rows)
            _append_rows(Path(trainer.root)/'FUNCTION_INTERVENTIONS.csv', rows)
            return result
    mother = trainer.snapshot()
    original_rng = capture_rng()
    started = time.monotonic()
    if str(trainer.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(trainer.device)
    result = None
    try:
        base, _, _ = restore(snapshot, trainer.device, lambda m: make_optimizer(m, trainer.cfg))
        base.eval().requires_grad_(False)
        path_order = paths(base)
        ids = np.asarray(trainer.data.probe)
        yf, mf = trainer.data.labels('F', 'diagnostic', ids, method=trainer.arm['id'], state=which,
                                     diagnostic='READ_INTERVENTION_PATH_SELECTION')
        domain = tensor(trainer.data.domain[ids], trainer.device)
        synchronize(trainer.device)
        fixed_start = time.monotonic()
        fcache = components(base, _packet(trainer, base, 'F', ids), domain)
        synchronize(trainer.device)
        fixed_seconds = time.monotonic()-fixed_start
        natural_counts = np.bincount(trainer.data.domain[trainer.data.roles['F']], minlength=base.D)
        chosen, scores = select_readapt_path(fcache, yf, mf, natural_counts)
        plan = dict(status='F_PATH_LOCKED', identity=identity, paths=[list(p) for p in path_order],
                    selected_path=None if chosen is None else list(chosen), path_scores=scores,
                    selection_role='F_probe_with_full_natural_F_domain_weights',
                    selection_result='SELECTED_POSITIVE_FIXED_DEPENDENCE' if chosen else 'NO_POSITIVE_FIXED_DEPENDENCE',
                    row_ids_F_probe=ids.tolist(), natural_F_domain_counts=natural_counts.tolist(),
                    teacher_version=snapshot['state'].get('teacher_version'), relation_refreshed=False,
                    H=8, reader_training='original Adam, only original w, natural L_rec, no auxiliary loss')
        write_json(root/'INTERVENTION_PLAN.json', plan)
        f_arrays = _measure_arrays(fcache, path_order)
        f_arrays.update(row_id=ids, labels=yf, mask=mf)
        np.savez_compressed(root/'F_probe_COMPONENTS.npz', **f_arrays)
        residents, branch_records, batches = {}, [], []
        continuation_preparation_start = time.monotonic()
        if chosen is not None:
            stream = copy.deepcopy(trainer.main)
            stream.load(snapshot['state']['main'])
            row_plan = []
            for step in range(8):
                rows = stream.next()
                yy, mm = trainer.data.labels('F', 'adapt', rows, method=trainer.arm['id'], state=which,
                    diagnostic='READ_INTERVENTION_H8_SHARED_PLAN', continuation_step=step)
                cached = components(base, _packet(trainer, base, 'F', rows),
                                    tensor(trainer.data.domain[rows], trainer.device))
                batches.append(dict(row_id=rows.copy(), components=cached,
                                    y=tensor(yy, trainer.device), mask=tensor(mm, trainer.device)))
                row_plan.append(rows.tolist())
            plan['shared_next_F_row_plan'] = row_plan
            write_json(root/'INTERVENTION_PLAN.json', plan)
            continuation_preparation_seconds = time.monotonic()-continuation_preparation_start
            residents, branch_records = continue_readers(snapshot, trainer, batches, chosen)
        else:
            continuation_preparation_seconds = 0.
        plan.update(status='BRANCHES_LOCKED_Q_PENDING', branches=branch_records,
                    no_Q_label_read_before_lock=True)
        write_json(root/'INTERVENTION_PLAN.json', plan)
        # The sole diagnostic Q-label access follows both F selection and H=8 locks.
        yq, mq = trainer.data.labels('Q', 'diagnostic', method=trainer.arm['id'], state=which,
                                     diagnostic='READ_INTERVENTIONS_FROZEN_ALL_ACTIONS')
        evaluation_start = time.monotonic()
        role_cache = [('F_probe', fcache, yf, mf, ids)]
        qview = trainer.views['Q']
        qarrays, qpredictions = [], {name: [] for name in residents}
        qfixed = {f'{mode}/{path_name(path) if path else ""}': [] for mode, path in _fixed_actions(path_order)}
        for begin in range(0, len(qview['ids']), 4096):
            qids = qview['ids'][begin:begin+4096]
            component_start = time.monotonic()
            cache = components(base, _packet(trainer, base, 'Q', qids),
                               tensor(trainer.data.domain[qids], trainer.device))
            synchronize(trainer.device)
            fixed_seconds += time.monotonic()-component_start
            qarrays.append(_measure_arrays(cache, path_order))
            for mode, path in _fixed_actions(path_order):
                qfixed[f'{mode}/{path_name(path) if path else ""}'].append(fixed_logits(cache, mode, path).cpu().numpy())
            for mode, (branch, optimizer) in residents.items():
                with torch.no_grad():
                    qpredictions[mode].append(reader_logits(branch, cache, mode, chosen).cpu().numpy())
        qdata = {key: np.concatenate([part[key] for part in qarrays]) for key in qarrays[0]}
        qdata.update(row_id=qview['ids'], labels=yq, mask=mq)
        np.savez_compressed(root/'Q_COMPONENTS.npz', **qdata)
        rows = []
        for role, cache, y, mask, row_ids in role_cache:
            before = cache['keep'].cpu().numpy()
            before_metric = evaluate(y, before, cache['domain'].cpu().numpy(), mask, domains=base.D)
            before_metric['_loss'] = np.logaddexp(0., before.astype(np.float64))-y*before
            for mode, path in _fixed_actions(path_order):
                prediction = fixed_logits(cache, mode, path).cpu().numpy()
                rows.extend(_score_rows(identity, role, mode, path, y, mask, prediction,
                                       cache['domain'].cpu().numpy(), cache['q'].cpu().numpy(), base, before_metric,
                                       natural_domain_counts=natural_counts))
        before = qdata['keep_logits']
        before_metric = evaluate(yq, before, qdata['domain'], mq, domains=base.D)
        before_metric['_loss'] = np.logaddexp(0., before.astype(np.float64))-yq*before
        for mode, path in _fixed_actions(path_order):
            prediction = np.concatenate(qfixed[f'{mode}/{path_name(path) if path else ""}'])
            rows.extend(_score_rows(identity, 'Q', mode, path, yq, mq, prediction,
                                   qdata['domain'], qdata['need_mass'], base, before_metric))
        branch_arrays = dict(row_id=qview['ids'], labels=yq, mask=mq, domain=qdata['domain'])
        if residents:
            keep_reader = np.concatenate(qpredictions['KEEP_READAPT'])
            reader_metric = evaluate(yq, keep_reader, qdata['domain'], mq, domains=base.D)
            reader_metric['_loss'] = np.logaddexp(0., keep_reader.astype(np.float64))-yq*keep_reader
            for mode in residents:
                prediction = np.concatenate(qpredictions[mode])
                branch_arrays[mode] = prediction
                branch_rows = _score_rows(identity, 'Q', mode, chosen, yq, mq, prediction,
                    qdata['domain'], qdata['need_mass'], base, reader_metric)
                for row in branch_rows:
                    row['reference_action'] = 'KEEP_READAPT'
                rows.extend(branch_rows)
            np.savez_compressed(root/'Q_READER_H8_PREDICTIONS.npz', **branch_arrays)
        synchronize(trainer.device)
        cost = dict(fixed_forward_seconds=fixed_seconds,
                    branch_training_seconds=sum(record['training_seconds'] for record in branch_records),
                    continuation_preparation_seconds=continuation_preparation_seconds,
                    evaluation_seconds=time.monotonic()-evaluation_start,
                    additional_reader_steps=sum(record['actual_reader_updates'] for record in branch_records),
                    attempted_reader_steps=24 if residents else 0,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(trainer.device) if str(trainer.device).startswith('cuda') else None,
                    peak_scope='process resident allocations during diagnostic',
                    time_accounting='evaluation includes fixed Q component time; total also includes export, cloning and parity',
                    total_seconds=time.monotonic()-started)
        write_csv(root/'FUNCTION_INTERVENTIONS.csv', rows)
        _append_rows(Path(trainer.root)/'FUNCTION_INTERVENTIONS.csv', rows)
        result = dict(plan, status='COMPLETE', cost=cost,
                      unselected_path_readaptation='UNMEASURED', shared_state_storage='RAM_ONLY',
                      fixed_actions_per_role=len(_fixed_actions(path_order)), Q_label_reads=1)
        write_json(root/'COST_LEDGER.json', cost)
    finally:
        parity_start = time.monotonic()
        restore_rng(original_rng)
        after = trainer.snapshot()
        parity = ram_equal(mother, after)
        write_json(root/'STATE_PARITY.json', dict(status='PASS' if parity else 'FAIL',
            model_buffers_adam_rng_all_cursors_teacher_and_G_directly_equal=parity,
            mother_model_committed=False))
        if not parity:
            raise AssertionError('Reader diagnostic polluted its mother RAM state')
    result['mother_state_parity'] = 'PASS'
    result['cost']['state_parity_seconds'] = time.monotonic()-parity_start
    result['cost']['total_seconds'] = time.monotonic()-started
    write_json(root/'COST_LEDGER.json', result['cost'])
    write_json(root/'INTERVENTION_PLAN.json', result)
    return result
