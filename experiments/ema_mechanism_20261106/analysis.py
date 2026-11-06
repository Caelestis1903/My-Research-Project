"""Frozen, seed-level analyses of the non-feedback EMA observations."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from experiments.overnight_ab_20261103.state import digest, read_json, sha_file, write_json

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2022, 2023, 2024, 2025, 2026)
GROUPS = ('S', 'D0', 'D1', 'D2', 'T0', 'T1', 'DT00', 'DT01', 'DT10', 'DT11', 'DT20', 'DT21')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
CONDITIONS = ('clean', 'gaussian_d0')
TC = 2.776445


class Incomplete(RuntimeError):
    exit_code = 5


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_table(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=['status'])
    for col in frame:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: json.dumps(_plain(v), sort_keys=True)
                                        if isinstance(v, (dict, list, tuple)) else v)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return frame


def source_logloss(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if not len(y):
        return float('nan')
    with np.errstate(divide='ignore'):
        return float(-(y * np.maximum(np.log(p), -100) +
                       (1-y) * np.maximum(np.log1p(-p), -100)).mean())


def cell_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def supported(domain, task, group):
    return group in ('S', f'D{int(domain)}', f'T{int(task)}', f'DT{int(domain)}{int(task)}')


def seed_statistics(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    n = len(valid)
    mean = float(valid.mean()) if n else float('nan')
    sd = float(valid.std(ddof=1)) if n > 1 else float('nan')
    radius = TC*sd/math.sqrt(5) if n == 5 else float('nan')
    return dict(n_seeds=n, mean=mean, sample_sd=sd, ci95_low=mean-radius,
                ci95_high=mean+radius, positive_seeds=int((valid > 0).sum()),
                negative_seeds=int((valid < 0).sum()), zero_seeds=int((valid == 0).sum()))


def summarize_seeds(frame, keys, value='value'):
    if frame.empty:
        return pd.DataFrame(columns=keys+['n_seeds', 'mean', 'sample_sd'])
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        per_seed = group.groupby('seed', sort=True)[value].mean()
        rows.append(dict(zip(keys, key), **seed_statistics(per_seed.values),
                         seed_values={str(s): float(v) for s, v in per_seed.items()}))
    return pd.DataFrame(rows)


def _jobs(study):
    return list(study.jobs.values()) if isinstance(study.jobs, dict) else list(study.jobs)


def load_records(study, out, require_complete=True):
    out = Path(out)
    records = []
    expected = {(j['dataset'], j['objective'], int(j['seed'])) for j in _jobs(study)}
    if len(expected) != 20 or any({s for d, c, s in expected if (d, c) == (ds, obj)} != set(SEEDS)
                                  for ds in study.manifest['datasets'] for obj in CONDITIONS):
        raise Incomplete('The frozen twenty-job/five-seed matrix is incomplete')
    for path in sorted((out/'runs').glob('*/epochs/E*.json')):
        row = read_json(path)
        if (row.get('dataset'), row.get('objective'), int(row.get('seed', -1))) not in expected:
            raise ValueError('Unexpected scientific trajectory in epoch records')
        row['_path'] = str(path.relative_to(out))
        records.append(row)
    if require_complete:
        for ds, obj, seed in sorted(expected):
            selected = [r for r in records if (r['dataset'], r['objective'], int(r['seed'])) == (ds, obj, seed)]
            if sorted(r['epoch'] for r in selected) != list(range(1, 51)):
                raise Incomplete(f'Missing or repeated epochs: {ds}/{obj}/{seed}')
            last = max(selected, key=lambda r: r['epoch'])
            if not last.get('complete') or last['global_step'] != 50*study.manifest['datasets'][ds]['steps_per_epoch']:
                raise Incomplete(f'Incomplete training budget: {ds}/{obj}/{seed}')
    return records


def input_hashes(study, out, include_records=True):
    out = Path(out)
    result = dict(manifest_hash=study.manifest_hash,
                  analysis_hash=digest({p.name: sha_file(p) for p in (Path(__file__), Path(__file__).with_name('evaluation.py'))}))
    for name in ('SOURCE_RECEIPT.json', 'DATA_MANIFEST.json', 'PROBE_MANIFEST.json', 'PREPARED.json'):
        if (out/name).exists():
            result[name] = sha_file(out/name)
    if include_records:
        result['epoch_records_hash'] = digest({str(p.relative_to(out)): sha_file(p)
                                               for p in sorted((out/'runs').glob('*/epochs/E*.json'))})
    return result


def _identity(record):
    keys = ('schema_version', 'job_id', 'dataset', 'objective', 'seed', 'epoch', 'global_step',
            'commit_index', 'init_hash', 'order_hash', 'layout_hash', 'group_hash', 'identity_id')
    return {key: record.get(key) for key in keys}


def _fid(study, dataset):
    return study.manifest['datasets'][dataset].get('aliases', {}).get('F', 'F')


def _prediction(out, record, panel, readout):
    hits = [r for r in record['predictions'] if r['panel_id'] == panel and
            (r.get('physical_readout_id', r['readout_id']) == readout or r['readout_id'] == readout)]
    if len(hits) != 1:
        raise ValueError(f'Expected one prediction receipt: {record["job_id"]}/E{record["epoch"]}/{panel}/{readout}')
    receipt = hits[0]
    path = Path(out)/receipt['path']
    if sha_file(path) != receipt['sha256']:
        raise ValueError('Prediction hash differs from the committed observation')
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in arrays.files}


def panel_identity(out, dataset, seed, panel):
    out = Path(out)
    item = read_json(out/'PROBE_MANIFEST.json')['panels'][dataset][str(seed)][panel]
    path = out/item['path']
    if sha_file(path) != item['sha256']:
        raise ValueError('Frozen panel bytes changed')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def materialize_tables(study, out, records):
    validation, readouts, features, innovations, costs = [], [], [], [], []
    for r in records:
        meta = _identity(r)
        for rid, metrics in r['validation'].items():
            validation.append(dict(meta, readout_id=rid, physical_readout_id=rid,
                                   state_hash=r['model_hash'][rid], **metrics))
        readouts.extend(dict(meta, **v) for v in r.get('pulse_readouts', []))
        features.extend(dict(meta, **v) for v in r.get('pulse_metrics', []))
        innovations.extend(dict(meta, **v) for v in r.get('innovation_stats', []))
        costs.append(dict(meta, **r.get('cost', {})))
    out = Path(out)
    for name, rows in [('VALIDATION_CURVES.csv', validation), ('PULSE_READOUTS.csv', readouts),
                       ('PULSE_METRICS.csv', features), ('INNOVATION_STATS.csv', innovations), ('EPOCH_COSTS.csv', costs)]:
        write_table(out/name, rows)
    return pd.DataFrame(validation), pd.DataFrame(readouts), pd.DataFrame(features)


def audit_archives(study, out):
    """Recompute the archived factorial without training or accessing new Test arrays."""
    out = Path(out)
    sources = study.manifest['source']['archive_sources']
    inventory, rows, reuse, old_curves, old_hpo = [], [], [], [], []
    for objective in CONDITIONS:
        spec = sources[objective]
        path = ROOT/spec['path']
        if not path.exists():
            inventory.append(dict(objective=objective, path=spec['path'], status='MISSING_ARCHIVED_ARTIFACT'))
            continue
        raw = path.read_bytes()
        import hashlib
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if spec.get('git_blob') and blob != spec['git_blob']:
            raise ValueError('Archived source blob mismatch: '+spec['path'])
        table = pd.read_csv(path)
        if objective == 'clean':
            table = table[(table['arm'] == 'A0') & table['model_kind'].isin(['online', 'fixed_ema'])].copy()
            table['readout_id'] = table['model_kind'].map({'online': 'online', 'fixed_ema': 'F'})
        else:
            table = table[table['candidate_id'].isin(['online', 'F'])].copy()
            table['readout_id'] = table['candidate_id']
        table['objective'] = objective
        table['source_path'] = spec['path']
        table['source_commit'] = spec['commit']
        table['source_sha256'] = sha_file(path)
        table['source_git_blob'] = blob
        table['test_exposure'] = np.where(table['split'] == 'test', 'archived', 'validation')
        rows.extend(table.to_dict('records'))
        inventory.append(dict(objective=objective, path=spec['path'], status='VERIFIED',
                              sha256=sha_file(path), git_blob=blob, rows=len(table)))
        reuse.append(dict(objective=objective, source=spec['path'], commit=spec['commit'],
                          status='SCALARS_REUSED', parameters='RAM_RELEASED',
                          reason='New pulse, gradients and per-row observations require the twenty registered trajectories'))
    archive = pd.DataFrame(rows)
    if not archive.empty:
        keys = ['dataset', 'objective', 'seed', 'split', 'checkpoint', 'readout_id']
        if archive.duplicated(keys).any():
            raise ValueError('Duplicate archived logical endpoints')
        write_table(out/'ARCHIVED_FACTORIAL.csv', archive)
        paired = endpoint_pairs(archive)
        write_table(out/'ARCHIVED_PAIRED_SEED.csv', paired)
        write_table(out/'ARCHIVED_PAIRED_SUMMARY.csv', summarize_seeds(paired,
            ['dataset', 'objective', 'split', 'checkpoint', 'contrast', 'cell', 'metric']))
    else:
        write_table(out/'ARCHIVED_FACTORIAL.csv', [])
    for key, spec in sources.items():
        if key in CONDITIONS:
            continue
        if 'path' in spec:
            path = ROOT/spec['path']
            inventory.append(dict(source=key, path=spec['path'],
                                  status='AVAILABLE' if path.exists() else 'MISSING_ARCHIVED_ARTIFACT',
                                  sha256=sha_file(path) if path.exists() else None))
    for label, directory in [('B', 'attempt1'), ('A', 'attempt_A1'), ('C', 'attempt_C1')]:
        for name in ('ENDPOINTS.csv', 'VALIDATION_CURVES.csv', 'HPO_SUMMARY.csv'):
            path = ROOT/'artifacts/aug_history_20261105'/directory/label/name
            inventory.append(dict(source=label, path=str(path.relative_to(ROOT)),
                                  status='AVAILABLE' if path.exists() else 'MISSING_ARCHIVED_ARTIFACT',
                                  sha256=sha_file(path) if path.exists() else None,
                                  independent_seed_note='A/B same online trajectories; C is a distinct objective, KR C deferred'))
            if path.exists() and name in ('VALIDATION_CURVES.csv', 'HPO_SUMMARY.csv'):
                archived = pd.read_csv(path)
                archived['archive_route'] = label
                archived['archive_source_path'] = str(path.relative_to(ROOT))
                archived['archive_source_sha256'] = sha_file(path)
                archived['archive_source_commit'] = study.manifest['source']['implementation_ref']
                (old_curves if name == 'VALIDATION_CURVES.csv' else old_hpo).append(archived)
    write_table(out/'ARCHIVED_VALIDATION_CURVES.csv', pd.concat(old_curves, ignore_index=True) if old_curves else [])
    write_table(out/'ARCHIVED_HPO.csv', pd.concat(old_hpo, ignore_index=True) if old_hpo else [])
    write_table(out/'EVIDENCE_REUSE.csv', reuse)
    result = dict(schema_version='1.0', study_id=study.manifest['study_id'], sources=inventory,
                  manifest_hash=study.manifest_hash, no_new_training=True, fresh_test_accessed=False,
                  scientific_unit='seed', missing_observations=['parameter pulse', 'M0-M3', 'bounded row predictions'],
                  archived_version_notes=['KR old Gaussian D2 differs from current D0',
                                          'A replays B online paths; duplicate online hashes are one scientific replicate'])
    write_json(out/'AUDIT_INVENTORY.json', result)
    return result


def endpoint_pairs(endpoints):
    rows = []
    keys = ['dataset', 'objective', 'seed', 'split', 'checkpoint']
    for key, block in endpoints.groupby(keys, sort=False):
        by = {r['readout_id']: r for r in block.to_dict('records')}
        if 'online' not in by or 'F' not in by:
            continue
        meta = dict(zip(keys, key))
        for cell in (*CELLS, 'macro', 'sample_weighted_cell', 'pooled_task_mean', 'worst_cell'):
            for metric in ('auc', 'logloss'):
                name = cell+'_'+metric
                if name in by['F'] and name in by['online']:
                    rows.append(dict(meta, contrast='F_minus_online', cell=cell, metric=metric,
                                     arm_F=by['F'][name], arm_online=by['online'][name],
                                     value=by['F'][name]-by['online'][name],
                                     F_epoch=by['F']['epoch'], online_epoch=by['online']['epoch']))
    paired = pd.DataFrame(rows)
    if not paired.empty:
        interaction = []
        for key, block in paired.groupby(['dataset', 'seed', 'split', 'checkpoint', 'cell', 'metric'], sort=False):
            by = {r['objective']: r for r in block.to_dict('records')}
            if all(c in by for c in CONDITIONS):
                interaction.append(dict(zip(['dataset', 'seed', 'split', 'checkpoint', 'cell', 'metric'], key),
                    objective='interaction', contrast='Gaussian_EMA_gain_minus_clean_EMA_gain',
                    arm_F=by['gaussian_d0']['value'], arm_online=by['clean']['value'],
                    value=by['gaussian_d0']['value']-by['clean']['value']))
        paired = pd.concat([paired, pd.DataFrame(interaction)], ignore_index=True)
    return paired


def _phase(epoch):
    return 'early' if epoch <= 10 else 'middle' if epoch <= 30 else 'late'


def _stage1_selection(study, records):
    rows, gain_rows = [], []
    grouped = {}
    for r in records:
        grouped.setdefault(r['job_id'], []).append(r)
    for history in grouped.values():
        history.sort(key=lambda r: r['epoch'])
        by_epoch = {r['epoch']: r for r in history}
        first = history[0]
        meta = {k: first[k] for k in ('job_id', 'dataset', 'objective', 'seed')}
        fid = _fid(study, first['dataset'])
        selected = {rid: max(history, key=lambda r: (r['validation'][rid]['macro_auc'], -r['epoch']))['epoch']
                    for rid in first['validation']}
        for rid, best_epoch in selected.items():
            peak = by_epoch[best_epoch]['validation'][rid]['macro_auc']
            neighbors = [by_epoch[e]['validation'][rid]['macro_auc'] for e in range(max(1, best_epoch-1), min(50, best_epoch+1)+1)]
            rows.append(dict(meta, readout_id=rid, selected_epoch=best_epoch, selected_macro_auc=peak,
                             neighborhood_mean_macro_auc=float(np.mean(neighbors)),
                             peak_minus_neighborhood=peak-float(np.mean(neighbors))))
        for r in history:
            online, fixed = r['validation']['online'], r['validation'][fid]
            for cell in (*CELLS, 'macro'):
                for metric in ('auc', 'logloss'):
                    key = cell+'_'+metric
                    gain_rows.append(dict(meta, epoch=r['epoch'], phase=_phase(r['epoch']),
                                          comparison='G_same', cell=cell, metric=metric,
                                          arm_history=fixed[key], arm_online=online[key], value=fixed[key]-online[key]))
            if r['epoch'] % 5 == 0:
                config = study.manifest['datasets'][first['dataset']]
                rho, steps = config['rho_F'], config['steps_per_epoch']
                age = rho*(1-rho**r['global_step'])/((1-rho)*steps)
                delayed = r['epoch']-math.floor(age+.5)
                if delayed >= 1:
                    for cell in (*CELLS, 'macro'):
                        for metric in ('auc', 'logloss'):
                            key = cell+'_'+metric
                            gain_rows.append(dict(meta, epoch=r['epoch'], phase=_phase(r['epoch']),
                                comparison='F_minus_age_matched_online', delayed_epoch=delayed, age_epochs=age,
                                cell=cell, metric=metric, arm_history=fixed[key],
                                arm_online=by_epoch[delayed]['validation']['online'][key],
                                value=fixed[key]-by_epoch[delayed]['validation']['online'][key]))
        for comparison, eh, eo in [('G_selected', selected[fid], selected['online']),
                                    ('G_common_online_selected', selected['online'], selected['online']),
                                    ('G_common_F_selected', selected[fid], selected[fid])]:
            for cell in (*CELLS, 'macro'):
                for metric in ('auc', 'logloss'):
                    key = cell+'_'+metric
                    gain_rows.append(dict(meta, epoch=eh, online_epoch=eo, phase='selected', comparison=comparison,
                        cell=cell, metric=metric, arm_history=by_epoch[eh]['validation'][fid][key],
                        arm_online=by_epoch[eo]['validation']['online'][key],
                        value=by_epoch[eh]['validation'][fid][key]-by_epoch[eo]['validation']['online'][key]))
    return pd.DataFrame(rows), pd.DataFrame(gain_rows)


def _validation_mask(identity):
    return np.asarray(identity['indices']) >= 0


def _panel_metric_matrix(identity, p, side=None):
    result = np.full((6, 2), np.nan)
    valid = _validation_mask(identity)
    for c, (d, t) in enumerate((d, t) for d in range(3) for t in range(2)):
        mask = valid[d].copy()
        if side is not None:
            mask &= identity['validation_row_side'][d] == side
        y, values = identity['y'][d, mask, t], p[d, mask, t]
        result[c] = cell_auc(y, values), source_logloss(y, values)
    return result


def history_preferences(study, out, records):
    index = {(r['dataset'], r['objective'], int(r['seed']), r['epoch']): r for r in records}
    rows = []
    for ds in study.manifest['datasets']:
        aliases = study.manifest['datasets'][ds].get('aliases', {})
        choices = list(dict.fromkeys(aliases.get(v, v) for v in ('F', 'H1', 'H2', 'H4')))
        for objective in CONDITIONS:
            for epoch in (10, 20, 30, 40, 50):
                values = {}
                for seed in SEEDS:
                    record = index[ds, objective, seed, epoch]
                    panel = panel_identity(out, ds, seed, 'V_mech')
                    for rid in choices:
                        pred = _prediction(out, record, 'V_mech', rid)['p']
                        values[seed, rid] = [_panel_metric_matrix(panel, pred, side) for side in (0, 1)]
                        for c, cell in enumerate(CELLS):
                            rows.append(dict(dataset=ds, objective=objective, seed=seed, epoch=epoch,
                                mode='FIXED_H_RESPONSE', cell=cell, readout_id=rid,
                                auc=record['validation'][rid][cell+'_auc'],
                                logloss=record['validation'][rid][cell+'_logloss']))
                means = np.array([[np.mean([index[ds, objective, seed, epoch]['validation'][rid][cell+'_auc']
                                           for seed in SEEDS]) for cell in CELLS] for rid in choices])
                uniform = int(np.argmax(means.mean(axis=1)))
                for c, cell in enumerate(CELLS):
                    best = int(np.argmax(means[:, c]))
                    rows.append(dict(dataset=ds, objective=objective, epoch=epoch,
                        mode='IN_SAMPLE_PREFERENCE_SPACE', cell=cell,
                        conditional_readout=choices[best], uniform_readout=choices[uniform],
                        conditional_auc=means[best, c], uniform_auc=means[uniform, c],
                        value=means[best, c]-means[uniform, c]))
                for held in SEEDS:
                    for fit_side in (0, 1):
                        fitted = np.array([np.mean([values[seed, rid][fit_side][:, 0] for seed in SEEDS if seed != held], axis=0)
                                           for rid in choices])
                        uniform = int(np.argmax(fitted.mean(axis=1)))
                        conditional = np.argmax(fitted, axis=0)
                        for c, cell in enumerate(CELLS):
                            chosen = int(conditional[c])
                            selected = values[held, choices[chosen]][1-fit_side][c]
                            reference = values[held, choices[uniform]][1-fit_side][c]
                            rows.append(dict(dataset=ds, objective=objective, seed=held, epoch=epoch,
                                row_fold_direction=f'V_{"AB"[fit_side]}_to_V_{"BA"[fit_side]}',
                                mode='CELL_CONDITIONAL_DIAGNOSTIC', cell=cell,
                                conditional_readout=choices[chosen], uniform_readout=choices[uniform],
                                conditional_auc=selected[0], uniform_auc=reference[0],
                                conditional_logloss=selected[1], uniform_logloss=reference[1],
                                value=selected[0]-reference[0]))
    result = pd.DataFrame(rows)
    write_table(Path(out)/'HISTORY_PREFERENCES.csv', result)
    cross = result[result['mode'] == 'CELL_CONDITIONAL_DIAGNOSTIC']
    write_table(Path(out)/'HISTORY_PREFERENCE_SUMMARY.csv', summarize_seeds(cross, ['dataset', 'objective', 'cell']))
    return result


def _pair_scores(p, pairs):
    if not len(pairs):
        return np.array([], dtype=np.float64)
    margins = p[pairs[:, 0]]-p[pairs[:, 1]]
    return (margins > 0).astype(np.float64)+.5*(margins == 0)


def _panel_pairs(identity, panel_id, cell):
    if panel_id != 'V_mech':
        return identity['rank_pairs'][cell]
    count = int(identity.get('pair_counts', np.full(6, len(identity['pairs'][cell])))[cell])
    pairs = identity['pairs'][cell, :count]
    if 'indices' in identity and len(pairs):
        valid = _validation_mask(identity)[cell//2]
        if np.any(pairs < 0) or np.any(pairs >= len(valid)) or not valid[pairs].all():
            raise ValueError('Validation rank pair refers to an invalid container slot')
    return pairs


def temporal_rank_stability(study, out, records):
    rows = []
    by_job = {}
    for r in records:
        by_job.setdefault(r['job_id'], []).append(r)
    for history in by_job.values():
        history.sort(key=lambda r: r['epoch'])
        head = history[0]
        fid = _fid(study, head['dataset'])
        for panel_id in ('P_A', 'P_B', 'V_mech'):
            identity = panel_identity(out, head['dataset'], head['seed'], panel_id)
            for rid in ('online', fid):
                previous = None
                for r in history:
                    data = _prediction(out, r, panel_id, rid)
                    scores = []
                    for c, (d, t) in enumerate((d, t) for d in range(3) for t in range(2)):
                        p = data['p'][d, :, t] if panel_id == 'V_mech' else data['rank_p'][c]
                        pairs = _panel_pairs(identity, panel_id, c)
                        current = _pair_scores(p, pairs)
                        scores.append(current)
                        if previous is not None:
                            delta = current-previous[c]
                            rows.append(dict(_identity(r), panel_id=panel_id, readout_id='F' if rid == fid else 'online',
                                cell=CELLS[c], phase=_phase(r['epoch']), pair_count=len(current),
                                temporal_change=float(np.abs(delta).mean()) if len(delta) else np.nan,
                                positive_change=float(np.maximum(delta, 0).mean()) if len(delta) else np.nan,
                                negative_change=float(np.maximum(-delta, 0).mean()) if len(delta) else np.nan,
                                missing_reason=None if len(delta) else 'NA_SINGLE_CLASS'))
                    previous = scores
    result = pd.DataFrame(rows)
    write_table(Path(out)/'TEMPORAL_RANK_STABILITY.csv', result)
    return result


def _pulse_lookup(record):
    return {(r['panel_id'], r.get('group_id'), r['variant'], int(r['domain']), int(r['task'])): r
            for r in record.get('pulse_readouts', [])}


def _pulse_row(lookup, panel, group, variant, d, t):
    return lookup[panel, group, variant, int(d), int(t)]


def pulse_contrasts(records):
    rows, interactions = [], []
    for record in records:
        if not record.get('pulse_readouts'):
            continue
        lookup = _pulse_lookup(record)
        panels = list(dict.fromkeys(r['panel_id'] for r in record['pulse_readouts']))
        for panel in panels:
            for d in range(3):
                for t in range(2):
                    base = _pulse_row(lookup, panel, None, 'F', d, t)
                    group_deltas = {}
                    for group in (*GROUPS, 'ALL'):
                        fast = _pulse_row(lookup, panel, None if group == 'ALL' else group,
                                          'all_fast' if group == 'ALL' else 'fast', d, t)
                        slow = _pulse_row(lookup, panel, None if group == 'ALL' else group,
                                          'all_slow' if group == 'ALL' else 'slow', d, t)
                        for metric, sign in [('auc', 1), ('logloss', -1), ('rank_risk', -1)]:
                            if any(v.get(metric) is None for v in (base, fast, slow)):
                                continue
                            f, s = sign*(fast[metric]-base[metric]), sign*(slow[metric]-base[metric])
                            group_deltas[group, metric] = f, s
                            for name, value in [('fast_minus_F', f), ('slow_minus_F', s), ('fast_minus_slow', f-s)]:
                                rows.append(dict(_identity(record), panel_id=panel, cell=f'd{d}t{t}', domain=d, task=t,
                                    group_id=group, functional_support=group == 'ALL' or supported(d, t, group),
                                    metric=metric, contrast=name, value=value,
                                    fast_state_hash=fast['state_hash'], slow_state_hash=slow['state_hash'], F_state_hash=base['state_hash']))
                    for metric in ('auc', 'logloss', 'rank_risk'):
                        if ('ALL', metric) not in group_deltas:
                            continue
                        f_all, s_all = group_deltas['ALL', metric]
                        f_sum = math.fsum(group_deltas[g, metric][0] for g in GROUPS)
                        s_sum = math.fsum(group_deltas[g, metric][1] for g in GROUPS)
                        individual = [group_deltas[g, metric][0]-group_deltas[g, metric][1] for g in GROUPS]
                        denominator = abs(f_all-s_all)+math.fsum(abs(v) for v in individual)
                        residual = (f_all-s_all)-(f_sum-s_sum)
                        if not math.isclose(residual, (f_all-f_sum)-(s_all-s_sum), rel_tol=1e-9, abs_tol=1e-12):
                            raise ValueError('Pulse non-additivity contrast identity failed')
                        interactions.append(dict(_identity(record), panel_id=panel, cell=f'd{d}t{t}', metric=metric,
                            all_fastF=f_all, all_slowF=s_all, D_all=f_all-s_all,
                            sum_fastF=f_sum, sum_slowF=s_sum, sum_D=f_sum-s_sum,
                            sum_abs_D=math.fsum(abs(v) for v in individual), R_fastF=f_all-f_sum,
                            R_slowF=s_all-s_sum, R=residual, q=abs(residual)/denominator if denominator else 0.))
    return pd.DataFrame(rows), pd.DataFrame(interactions)


def rank_flips(study, out, records):
    rows = []
    for r in records:
        if not r.get('pulse_readouts'):
            continue
        fid = _fid(study, r['dataset'])
        for panel_id in ('P_A', 'P_B', 'V_mech'):
            identity = panel_identity(out, r['dataset'], r['seed'], panel_id)
            base = _prediction(out, r, panel_id, fid)
            for receipt in r['predictions']:
                if receipt['panel_id'] != panel_id or receipt.get('physical_readout_id', receipt['readout_id']) in r['validation']:
                    continue
                altered = _prediction(out, r, panel_id, receipt.get('physical_readout_id', receipt['readout_id']))
                for c, (d, t) in enumerate((d, t) for d in range(3) for t in range(2)):
                    a = altered['p'][d, :, t] if panel_id == 'V_mech' else altered['rank_p'][c]
                    b = base['p'][d, :, t] if panel_id == 'V_mech' else base['rank_p'][c]
                    pairs = _panel_pairs(identity, panel_id, c)
                    aa, bb = _pair_scores(a, pairs), _pair_scores(b, pairs)
                    delta = aa-bb
                    labels = identity['y'][d, :, t] if panel_id == 'V_mech' else identity['rank_y'][c]
                    exact_a, exact_b = a, b
                    if panel_id == 'V_mech':
                        valid = _validation_mask(identity)[d]
                        labels, exact_a, exact_b = labels[valid], a[valid], b[valid]
                    rows.append(dict(_identity(r), panel_id=panel_id, readout_id=receipt['readout_id'],
                        cell=CELLS[c], pair_count=len(pairs), correction=float(np.maximum(delta, 0).mean()) if len(delta) else np.nan,
                        damage=float(np.maximum(-delta, 0).mean()) if len(delta) else np.nan, sampled_net_flip=float(delta.mean()) if len(delta) else np.nan,
                        unchanged=float((delta == 0).mean()) if len(delta) else np.nan, new_ties=float(((aa == .5) & (bb != .5)).mean()) if len(delta) else np.nan,
                        broken_ties=float(((aa != .5) & (bb == .5)).mean()) if len(delta) else np.nan,
                        paired_sample_auc=float(aa.mean()) if len(delta) else np.nan, exact_panel_auc=cell_auc(labels, exact_a),
                        missing_reason=None if len(delta) else 'NA_SINGLE_CLASS',
                        exact_auc_delta=cell_auc(labels, exact_a)-cell_auc(labels, exact_b)))
    result = pd.DataFrame(rows)
    write_table(Path(out)/'RANK_FLIPS.csv', result)
    return result


def predictive_rows(study, records):
    index = {(r['job_id'], r['epoch']): r for r in records}
    rows = []
    for r in records:
        if not r.get('pulse_metrics') or r['epoch'] >= 50:
            continue
        future = index[r['job_id'], r['epoch']+5]
        lookup = _pulse_lookup(future)
        seen = set()
        for feature in r['pulse_metrics']:
            d, t, group = int(feature['domain']), int(feature['task']), feature['group_id']
            if feature.get('panel_id', 'P_A') != 'P_A' or not supported(d, t, group):
                continue
            if (d, t, group) in seen:
                raise ValueError('Repeated P_A predictive feature row')
            seen.add((d, t, group))
            for side in ('V_A', 'V_B'):
                base = _pulse_row(lookup, side, None, 'F', d, t)
                fast = _pulse_row(lookup, side, group, 'fast', d, t)
                slow = _pulse_row(lookup, side, group, 'slow', d, t)
                if any(v.get('auc') is None for v in (base, fast, slow)):
                    raise Incomplete('Unavailable class in a frozen future Validation row fold')
                rows.append(dict(_identity(r), **feature, cell=f'd{d}t{t}', origin_epoch=r['epoch'],
                    target_epoch=future['epoch'], label_panel=side,
                    y_fastF=fast['auc']-base['auc'], y_slowF=slow['auc']-base['auc'],
                    y_ll_fastF=base['logloss']-fast['logloss'], y_ll_slowF=base['logloss']-slow['logloss'],
                    target_F_state_hash=base['state_hash'], target_fast_state_hash=fast['state_hash'],
                    target_slow_state_hash=slow['state_hash']))
        if len(seen) != 24:
            raise Incomplete(f'Expected twenty-four supported features at {r["job_id"]}/{r["epoch"]}')
    return pd.DataFrame(rows)


def predictive_split(frame, heldout_seed, fit_panel, score_panel, spec):
    train = frame[(frame['seed'] != heldout_seed) & frame['origin_epoch'].isin(spec['train_origins']) &
                  (frame['label_panel'] == fit_panel)].copy()
    score = frame[(frame['seed'] == heldout_seed) & frame['origin_epoch'].isin(spec['score_origins']) &
                  (frame['label_panel'] == score_panel)].copy()
    if len(train) != spec['train_rows_per_fold'] or len(score) != spec['score_rows_per_fold']:
        raise Incomplete(f'Future-fold rows differ from frozen 576/240: {len(train)}/{len(score)}')
    if heldout_seed in set(train['seed']) or train['target_epoch'].max() >= score['origin_epoch'].min():
        raise ValueError('Seed or forward-time leakage in the predictive fold')
    if fit_panel == score_panel:
        raise ValueError('Validation row crossfit requires opposite label partitions')
    return train, score


def _design_dictionary():
    return [(condition, d, t, group) for condition in CONDITIONS for d in range(3) for t in range(2)
            for group in GROUPS if supported(d, t, group)]


def feature_matrices(train, score, model_id, spec):
    dictionary = _design_dictionary()
    locations = {key: i for i, key in enumerate(dictionary)}
    names = [f'identity_{i}' for i in range(len(dictionary))]
    blocks = spec['feature_sets'][model_id]
    continuous = ['epoch_over50'] + [name for block in blocks if block != 'design' for name in spec['feature_blocks'][block]]
    matrices = []
    for frame in (train, score):
        onehot = np.zeros((len(frame), len(dictionary)))
        for i, row in enumerate(frame.to_dict('records')):
            key = row['objective'], int(row['domain']), int(row['task']), row['group_id']
            onehot[i, locations[key]] = 1.
        extra = np.empty((len(frame), len(continuous)))
        for i, name in enumerate(continuous):
            extra[:, i] = frame['origin_epoch'].to_numpy()/50 if name == 'epoch_over50' else pd.to_numeric(frame[name]).to_numpy()
            if name == 'gap_pulse_cos':
                extra[:, i] = np.where(np.isfinite(extra[:, i]), extra[:, i], 0.)
        if not np.isfinite(extra).all():
            raise ValueError('Undefined non-cosine feature in the predictive model')
        matrices.append((onehot, extra))
    mean = matrices[0][1].mean(axis=0)
    sd = matrices[0][1].std(axis=0, ddof=0)
    constant = sd == 0
    scale = np.where(constant, 1., sd)
    result = []
    for onehot, extra in matrices:
        standardized = (extra-mean)/scale
        standardized[:, constant] = 0.
        result.append(np.concatenate([onehot, standardized], axis=1))
    return *result, dict(onehot_dictionary=dictionary, feature_names=names+continuous,
                         continuous_mean=mean.tolist(), continuous_population_sd=sd.tolist(),
                         constant_flags=constant.tolist())


def ridge_predict(x_train, y_train, x_score):
    """SSE + alpha=1; centering leaves the intercept unpenalized."""
    mean_x, mean_y = x_train.mean(axis=0), y_train.mean(axis=0)
    xc, yc = x_train-mean_x, y_train-mean_y
    coefficient = np.linalg.solve(xc.T@xc+np.eye(xc.shape[1]), xc.T@yc)
    return mean_y+(x_score-mean_x)@coefficient


def action_indices(prediction):
    prediction = np.asarray(prediction, dtype=float)
    return np.argmax(np.column_stack([np.zeros(len(prediction)), prediction]), axis=1)


def _row_outcomes(target, prediction, uniform, ll_target=None, numerical_prediction=True):
    chosen = action_indices(prediction)
    choices = np.column_stack([np.zeros(len(target)), target])
    chosen_value = choices[np.arange(len(target)), chosen]
    reference = choices[np.arange(len(target)), uniform]
    oracle = choices.max(axis=1)
    truth_d, pred_d = target[:, 0]-target[:, 1], prediction[:, 0]-prediction[:, 1]
    result = dict(action=chosen, uniform_action=uniform, chosen_minus_F=chosen_value,
                  chosen_minus_uniform_action=chosen_value-reference, regret=oracle-chosen_value,
                  uniform_minus_F=reference, oracle_minus_F=oracle,
                  target_direction_zero=(truth_d == 0).astype(int),
                  direction_correct=np.where(truth_d != 0, np.sign(truth_d) == np.sign(pred_d), np.nan))
    if numerical_prediction:
        result.update(abs_error_D=np.abs(truth_d-pred_d), abs_error_Y_fast=np.abs(target[:, 0]-prediction[:, 0]),
                      abs_error_Y_slow=np.abs(target[:, 1]-prediction[:, 1]))
    if ll_target is not None:
        ll_choices = np.column_stack([np.zeros(len(target)), ll_target])
        result['chosen_LL_utility'] = ll_choices[np.arange(len(target)), chosen]
        result['uniform_LL_utility'] = ll_choices[np.arange(len(target)), uniform]
    return result


def _prediction_scopes(oof):
    frames = []
    specs = [('ALL', []), ('CONDITION', ['objective']), ('CELL', ['objective', 'cell']),
             ('GROUP', ['objective', 'group_id']), ('CELL_GROUP', ['objective', 'cell', 'group_id'])]
    metrics = ['abs_error_D', 'abs_error_Y_fast', 'abs_error_Y_slow', 'chosen_minus_F',
               'chosen_minus_uniform_action', 'regret', 'direction_correct', 'target_direction_zero',
               'chosen_LL_utility', 'uniform_LL_utility']
    for scope, levels in specs:
        keys = ['dataset', 'model_id', 'heldout_seed']+levels
        frame = oof.groupby(keys, dropna=False)[metrics].mean().reset_index().rename(columns={'heldout_seed': 'seed'})
        frame['scope'] = scope
        for col in ('objective', 'cell', 'group_id'):
            if col not in frame:
                frame[col] = 'ALL'
        frames.append(frame)
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed = per_seed.rename(columns={'abs_error_D': 'MAE_D', 'abs_error_Y_fast': 'MAE_Y_fast',
                                      'abs_error_Y_slow': 'MAE_Y_slow', 'direction_correct': 'nonzero_D_sign_accuracy'})
    return per_seed


def fit_predictive_folds(study, out):
    out = Path(out)
    records = load_records(study, out)
    rows = predictive_rows(study, records)
    write_table(out/'PREDICTIVE_ROWS.csv', rows)
    spec = study.manifest['predictive']
    oof, receipts = [], []
    raw_columns = {'M2_full_raw': ['P_A_gamma_full_fastF', 'P_A_gamma_full_slowF'],
                   'M2_detach_raw': ['P_A_gamma_source_detach_fastF', 'P_A_gamma_source_detach_slowF'],
                   'M3_auc_raw': ['P_A_rank_exact_auc_fastF', 'P_A_rank_exact_auc_slowF'],
                   'M3_rank_raw': ['P_A_rank_risk_reduction_fastF', 'P_A_rank_risk_reduction_slowF']}
    for dataset in study.manifest['datasets']:
        data = rows[rows['dataset'] == dataset]
        for held in SEEDS:
            for direction in spec['row_crossfit']:
                train, score = predictive_split(data, held, direction['fit_labels'], direction['score_labels'], spec)
                target = score[['y_fastF', 'y_slowF']].to_numpy(dtype=float)
                y_train = train[['y_fastF', 'y_slowF']].to_numpy(dtype=float)
                ll_target = score[['y_ll_fastF', 'y_ll_slowF']].to_numpy(dtype=float)
                uniform_by_condition = {}
                for condition in CONDITIONS:
                    y = train.loc[train['objective'] == condition, ['y_fastF', 'y_slowF']].to_numpy(dtype=float)
                    uniform_by_condition[condition] = int(np.argmax(np.r_[0., y.mean(axis=0)]))
                uniform = score['objective'].map(uniform_by_condition).to_numpy(dtype=int)
                fold = direction['fit_labels']+'_to_'+direction['score_labels']
                predictions = {}
                for model_id in spec['feature_sets']:
                    x_train, x_score, transform = feature_matrices(train, score, model_id, spec)
                    predictions[model_id] = ridge_predict(x_train, y_train, x_score)
                    receipts.append(dict(dataset=dataset, heldout_seed=held, row_fold_direction=fold,
                        model_id=model_id, train_rows=len(train), score_rows=len(score),
                        train_seeds=sorted(int(v) for v in train['seed'].unique()),
                        train_origins=sorted(int(v) for v in train['origin_epoch'].unique()),
                        score_origins=sorted(int(v) for v in score['origin_epoch'].unique()),
                        fit_label_side=direction['fit_labels'], score_label_side=direction['score_labels'],
                        train_identity_hash=digest(train[['job_id', 'origin_epoch', 'cell', 'group_id', 'label_panel']].to_dict('records')),
                        score_identity_hash=digest(score[['job_id', 'origin_epoch', 'cell', 'group_id', 'label_panel']].to_dict('records')),
                        transform=transform, alpha=1, ridge_objective='SSE + ||B||^2; unpenalized intercept',
                        uniform_actions=uniform_by_condition))
                predictions.update({name: score[cols].to_numpy(dtype=float) for name, cols in raw_columns.items()})
                predictions['UNIFORM_ACTION'] = np.column_stack([uniform == 1, uniform == 2]).astype(float)
                for model_id, prediction in predictions.items():
                    base = score[['job_id', 'dataset', 'objective', 'seed', 'origin_epoch', 'target_epoch',
                                  'cell', 'domain', 'task', 'group_id', 'label_panel', 'y_fastF', 'y_slowF',
                                  'y_ll_fastF', 'y_ll_slowF', 'target_F_state_hash',
                                  'target_fast_state_hash', 'target_slow_state_hash']].copy()
                    base['model_id'], base['heldout_seed'], base['row_fold_direction'] = model_id, held, fold
                    base['yhat_fastF'], base['yhat_slowF'] = prediction[:, 0], prediction[:, 1]
                    base['prediction_units'] = ('AUC_gain' if model_id in spec['feature_sets'] or model_id == 'M3_auc_raw'
                                                else 'action_indicator' if model_id == 'UNIFORM_ACTION' else 'rank_risk_reduction_score')
                    for name, values in _row_outcomes(target, prediction, uniform, ll_target,
                                                     model_id in spec['feature_sets']).items():
                        base[name] = values
                    oof.append(base)
    oof = pd.concat(oof, ignore_index=True)
    for name in ('abs_error_D', 'abs_error_Y_fast', 'abs_error_Y_slow'):
        if name not in oof:
            oof[name] = np.nan
    write_table(out/'PREDICTIVE_OOF.csv', oof)
    write_json(out/'PREDICTIVE_FOLDS.json', _plain(dict(input_hashes=input_hashes(study, out), folds=receipts,
                                                     ridge_fit_count=len(receipts))))
    if len(receipts) != spec['total_ridge_fits']:
        raise ValueError('The frozen number of ridge fits changed')
    per_seed = _prediction_scopes(oof)
    write_table(out/'PREDICTION_SEED.csv', per_seed)
    metrics = ['MAE_D', 'MAE_Y_fast', 'MAE_Y_slow', 'chosen_minus_F', 'chosen_minus_uniform_action',
               'regret', 'nonzero_D_sign_accuracy', 'target_direction_zero', 'chosen_LL_utility', 'uniform_LL_utility']
    keys = ['dataset', 'scope', 'objective', 'cell', 'group_id', 'model_id']
    summary = []
    for key, group in per_seed.groupby(keys, sort=False, dropna=False):
        meta = dict(zip(keys, key))
        for metric in metrics:
            summary.append(dict(meta, metric=metric, **seed_statistics(group[metric].values),
                                seed_values={str(int(r['seed'])): r[metric] for r in group.to_dict('records')}))
    summary = pd.DataFrame(summary)
    write_table(out/'PREDICTION_SUMMARY.csv', summary)
    comparisons = []
    compare_keys = ['dataset', 'scope', 'objective', 'cell', 'group_id', 'seed']
    indexed = per_seed.set_index(compare_keys+['model_id'])
    for row in per_seed.to_dict('records'):
        if row['model_id'] not in spec['feature_sets']:
            continue
        for control in spec['simple_controls']:
            key = tuple(row[k] for k in compare_keys)+(control,)
            reference = indexed.loc[key]
            comparisons.append({**{k: row[k] for k in compare_keys}, 'model_id': row['model_id'],
                'simple_control': control, 'skill_D': 1-row['MAE_D']/reference['MAE_D'] if reference['MAE_D'] != 0 else np.nan,
                'MAE_D_delta': row['MAE_D']-reference['MAE_D'],
                'action_utility_delta': row['chosen_minus_F']-reference['chosen_minus_F'],
                'regret_delta': row['regret']-reference['regret']})
    write_table(out/'PREDICTIVE_COMPARISONS.csv', comparisons)
    return summary


def _control_triggers(study, records, interactions, contrasts):
    cal = []
    index = {(r['dataset'], r['objective'], int(r['seed']), r['epoch']): r for r in records}
    for ds in study.manifest['datasets']:
        fid = _fid(study, ds)
        for objective in CONDITIONS:
            evidence = []
            for epoch in (20, 40):
                delta_auc = np.mean([index[ds, objective, s, epoch]['validation'][fid]['macro_auc']-
                                     index[ds, objective, s, epoch]['validation']['online']['macro_auc'] for s in SEEDS])
                delta_ll = np.mean([index[ds, objective, s, epoch]['validation'][fid]['macro_logloss']-
                                    index[ds, objective, s, epoch]['validation']['online']['macro_logloss'] for s in SEEDS])
                evidence.append(dict(epoch=epoch, auc_delta=float(delta_auc), raw_LL_delta=float(delta_ll),
                                     triggered=bool(delta_auc*delta_ll > 0)))
            if any(v['triggered'] for v in evidence):
                cal.append(dict(dataset=ds, objective=objective, evidence=evidence))
    trigger_rows, selected_groups = [], []
    subset = contrasts[(contrasts['panel_id'] == 'V_mech') & (contrasts['metric'] == 'auc') &
                       (contrasts['contrast'] == 'fast_minus_slow') & contrasts['epoch'].isin([20, 40])]
    means = subset.groupby(['dataset', 'objective', 'epoch', 'cell', 'group_id'])['value'].mean()
    group_scores = {g: [] for g in GROUPS}
    keys = ['dataset', 'objective', 'epoch', 'cell']
    for key, block in subset.groupby(keys, sort=False):
        by = {group: float(means.loc[tuple(key)+(group,)]) for group in (*GROUPS, 'ALL')}
        residual = by['ALL']-math.fsum(by[g] for g in GROUPS)
        magnitude = math.fsum(abs(by[g]) for g in GROUPS)
        denominator = abs(by['ALL'])+magnitude
        q = abs(residual)/denominator if denominator else 0.
        trigger = q > .5 and max(abs(by['ALL']), magnitude) >= .0002
        if trigger:
            trigger_rows.append(dict(zip(keys, key), q=q, R=residual, D_all=by['ALL'], sum_abs_D=magnitude))
            for group in GROUPS:
                group_scores[group].append(abs(by[group]))
    if trigger_rows:
        selected_groups = sorted(GROUPS, key=lambda g: (-np.mean(group_scores[g]), GROUPS.index(g)))[:2]
    return dict(C_CAL=dict(triggered=bool(cal), strata=cal, reason='computed_from_validation'),
                C_BG=dict(triggered=bool(trigger_rows), strata=trigger_rows, groups=selected_groups,
                          group_scores={g: float(np.mean(v)) if v else None for g, v in group_scores.items()},
                          reason='five-seed mean pulse non-additivity on V_mech'))


def _decision_rows(study, out):
    seeds = pd.read_csv(Path(out)/'PREDICTION_SEED.csv')
    results = []
    for key, frame in seeds.groupby(['dataset', 'scope', 'objective', 'cell', 'group_id'], sort=False):
        summaries = {m: group for m, group in frame.groupby('model_id')}
        simple_means = {m: float(summaries[m]['chosen_minus_F'].mean()) for m in study.manifest['predictive']['simple_controls']}
        for model_id, data in summaries.items():
            if model_id == 'UNIFORM_ACTION':
                continue
            effect, increment = float(data['chosen_minus_F'].mean()), float(data['chosen_minus_uniform_action'].mean())
            useful = effect > 0 and increment > 0
            results.append(dict(dataset=key[0], scope=key[1], objective=key[2], cell=key[3], group_id=key[4],
                model_id=model_id, status='ACTION_USEFUL' if useful else 'NO_DIFFERENTIAL_ACTION_GAIN',
                effect_size=effect, above_uniform=increment, n_seeds=5,
                complexity_increment_over_all_simple=model_id in ('M2_full', 'M2_detach', 'M3_direct') and
                    all(effect > v for v in simple_means.values()),
                priority='LOW_PRIORITY_SMALL_EFFECT' if useful and min(effect, increment) < .0002 else 'STANDARD',
                seed_values={str(int(r['seed'])): r['chosen_minus_uniform_action'] for r in data.to_dict('records')},
                source_artifact='PREDICTION_SEED.csv'))
    return results


def metric_correspondence(features, contrasts, out):
    if features.empty or contrasts.empty:
        raise Incomplete('Pulse features or contrasts are missing')
    features = features[features['functional_support'].astype(bool)].copy()
    features = features[features['panel_id'].fillna('P_A') == 'P_A'] if 'panel_id' in features else features
    main_keys = ['job_id', 'epoch', 'domain', 'task', 'group_id']
    scores = {'M0_gap': 'gap_rms', 'M0_pulse': 'pulse_fastslow_rms',
              'M1': 'P_A_natural_logit_rms_fastslow',
              'M2_full': ('P_A_gamma_full_fastF', 'P_A_gamma_full_slowF'),
              'M2_detach': ('P_A_gamma_source_detach_fastF', 'P_A_gamma_source_detach_slowF'),
              'M3_auc': ('P_A_rank_exact_auc_fastF', 'P_A_rank_exact_auc_slowF'),
              'M3_rank': ('P_A_rank_risk_reduction_fastF', 'P_A_rank_risk_reduction_slowF')}
    rows, stability = [], []
    for panel in ('P_A', 'P_B', 'V_mech'):
        for target_metric in ('auc', 'rank_risk', 'logloss'):
            target = contrasts[(contrasts['panel_id'] == panel) & (contrasts['metric'] == target_metric) &
                               (contrasts['contrast'] == 'fast_minus_slow') & (contrasts['group_id'] != 'ALL')]
            merged = features.merge(target[main_keys+['value']], on=main_keys, how='inner')
            for model, columns in scores.items():
                for key, block in merged.groupby(['dataset', 'objective', 'seed'], sort=False):
                    values = block[columns[0]]-block[columns[1]] if isinstance(columns, tuple) else block[columns]
                    correlation = float(np.corrcoef(values, block['value'])[0, 1]) if len(values) > 1 and values.std() > 0 and block['value'].std() > 0 else np.nan
                    rows.append(dict(dataset=key[0], objective=key[1], seed=key[2], model_id=model,
                        target_panel=panel, target_metric=target_metric, correlation=correlation,
                        valid_rows=len(block), phase='all_anchors', simultaneous_only=True))
    for key, block in features.groupby(['dataset', 'objective', 'seed'], sort=False):
        for family, a_cols, b_cols in [
            ('M1', ['P_A_natural_logit_rms_fastF', 'P_A_natural_logit_rms_slowF'], ['P_B_natural_logit_rms_fastF', 'P_B_natural_logit_rms_slowF']),
            ('M3_auc', ['P_A_rank_exact_auc_fastF', 'P_A_rank_exact_auc_slowF'], ['P_B_rank_exact_auc_fastF', 'P_B_rank_exact_auc_slowF']),
            ('M3_rank', ['P_A_rank_risk_reduction_fastF', 'P_A_rank_risk_reduction_slowF'], ['P_B_rank_risk_reduction_fastF', 'P_B_rank_risk_reduction_slowF'])]:
            if not all(c in block for c in a_cols+b_cols):
                raise Incomplete('P_B sample stability features missing: '+family)
            a, b = block[a_cols].to_numpy(), block[b_cols].to_numpy()
            av, bv = a[:, 0]-a[:, 1], b[:, 0]-b[:, 1]
            stability.append(dict(dataset=key[0], objective=key[1], seed=key[2], model_id=family,
                direction_correlation=float(np.corrcoef(av, bv)[0, 1]) if av.std() > 0 and bv.std() > 0 else np.nan,
                mean_abs_score_difference=float(np.abs(a-b).mean()),
                action_agreement=float((action_indices(a) == action_indices(b)).mean()) if family != 'M1' else np.nan,
                n_rows=len(block)))
    write_table(Path(out)/'METRIC_CORRESPONDENCE.csv', rows)
    write_table(Path(out)/'PANEL_STABILITY.csv', stability)


def temperature_map(p, temperature):
    p = np.asarray(p)
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    if temperature == 1:
        return p.copy()
    result = p.astype(np.float64, copy=True)
    interior = (result > 0) & (result < 1)
    logits = np.log(result[interior])-np.log1p(-result[interior])
    result[interior] = expit(logits/temperature)
    return result


def _calibration_controls(study, out, records, trigger):
    if not trigger['triggered']:
        return [dict(control_id='C_CAL', status='NOT_TRIGGERED')]
    index = {(r['dataset'], r['objective'], int(r['seed']), r['epoch']): r for r in records}
    rows = []
    for stratum in trigger['strata']:
        ds, objective = stratum['dataset'], stratum['objective']
        fid = _fid(study, ds)
        for seed in SEEDS:
            panel = panel_identity(out, ds, seed, 'V_mech')
            valid = _validation_mask(panel)
            for epoch in (20, 40):
                record = index[ds, objective, seed, epoch]
                predictions = {rid: _prediction(out, record, 'V_mech', physical)['p']
                               for rid, physical in [('online', 'online'), ('F', fid)]}
                for d in range(3):
                    for t in range(2):
                        y = panel['y'][d, :, t]
                        for fit_side in (0, 1):
                            fit_mask = valid[d] & (panel['validation_row_side'][d] == fit_side)
                            score_mask = valid[d] & (panel['validation_row_side'][d] == 1-fit_side)
                            for rid in ('online', 'F'):
                                p = predictions[rid][d, :, t]
                                if not fit_mask.any() or not score_mask.any():
                                    raise Incomplete('Empty Validation calibration row partition')
                                fit = minimize_scalar(lambda temp: source_logloss(y[fit_mask], temperature_map(p[fit_mask], temp)),
                                    bounds=(.25, 4), method='bounded', options={'maxiter': 100, 'xatol': 1e-8})
                                if not fit.success:
                                    raise FloatingPointError('Frozen temperature optimization failed: '+fit.message)
                                source = p[score_mask]
                                changed = temperature_map(source, float(fit.x))
                                rows.append(dict(control_id='C_CAL', status='COMPLETED', dataset=ds,
                                    objective=objective, seed=seed, epoch=epoch, cell=f'd{d}t{t}',
                                    row_fold_direction=f'V_{"AB"[fit_side]}_to_V_{"BA"[fit_side]}', readout_id=rid,
                                    temperature=float(fit.x), iterations=int(fit.nit),
                                    original_logloss=source_logloss(y[score_mask], source),
                                    calibrated_logloss=source_logloss(y[score_mask], changed),
                                    auc_retained_from_original_p=cell_auc(y[score_mask], source),
                                    new_saturation_count=int((((changed == 0) | (changed == 1)) & (source != 0) & (source != 1)).sum()),
                                    original_unique_probability_count=int(np.unique(source).size),
                                    calibrated_unique_probability_count=int(np.unique(changed).size),
                                    extra_tie_count=int(max(0, np.unique(source).size-np.unique(changed).size)),
                                    state_hash=record['model_hash'][fid if rid == 'F' else 'online']))
    if len(rows) > 960:
        raise ValueError('Temperature control fit budget exceeded')
    return rows


def _background_controls(study, out, trigger):
    if not trigger['triggered']:
        return [dict(control_id='C_BG', status='NOT_TRIGGERED')]
    import torch
    from .state import request
    from .panels import load_panels
    from .readouts import panel_predictions, pulse_rows, group_hybrid
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.state import state_digest
    from experiments.aug_history_20261105.model_adapter import build_model
    from experiments.aug_history_20261105.registry import ParameterLayout
    from experiments.aug_history_20261105.group_registry import build_path_groups

    prepared = read_json(Path(out)/'PREPARED.json')
    cache = prepared.get('cache_root', study.paths.get('cache_root'))
    if not cache:
        raise Incomplete('Frozen cache path missing from PREPARED')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    groups = trigger['groups']
    if len(groups) > 2 or any(group not in GROUPS for group in groups):
        raise ValueError('Invalid frozen second-background groups')
    records = []
    states = 0
    for job in _jobs(study):
        bundle = cached_bundle(cache, job['dataset'], include_test=False)
        panels = load_panels(out, job['dataset'], job['seed'], bundle)
        model = build_model(bundle, job['config']).to(device).requires_grad_(False)
        layout = ParameterLayout.from_model(model)
        registry = build_path_groups(model, bundle)
        snapshot = request(out, 'get', key=job['job_id'])['state']
        for epoch in (20, 40):
            retained = snapshot['retained_pulse_states'][f'E{epoch}']
            fast, slow = retained['fast'].to(device), retained['slow'].to(device)
            buffers = {name: tensor.to(device) for name, tensor in retained['buffers'].items()}
            vector = fast.new_empty(fast.shape)
            for group in groups:
                for background, base, changed in [('all_slow_elsewhere', slow, fast), ('all_fast_elsewhere', fast, slow)]:
                    group_hybrid(base, changed, registry.indices(device), GROUPS.index(group), vector)
                    layout.copy_to(model, vector, buffers)
                    state_hash = state_digest(model.state_dict())
                    states += 1
                    for name, panel in panels.items():
                        predictions = panel_predictions(model, panel)
                        values = pulse_rows(panel, predictions, background+'_'+group, group, 'background', state_hash)
                        for row in values:
                            records.append(dict(control_id='C_BG', status='COMPLETED', job_id=job['job_id'],
                                dataset=job['dataset'], objective=job['objective'], seed=job['seed'],
                                epoch=epoch, background=background, **row))
                        del predictions
            del fast, slow, vector, buffers
        del snapshot, model, layout, registry, panels, bundle
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    if states > 160:
        raise ValueError('Second-background state budget exceeded')
    return records


def run_controls(study, out, decision):
    out = Path(out)
    decision_path = Path(decision) if isinstance(decision, (str, Path)) else out/'STAGE3_DECISION.json'
    frozen = read_json(decision_path) if isinstance(decision, (str, Path)) else decision
    if frozen != read_json(out/'STAGE3_DECISION.json'):
        raise ValueError('The supplied control decision differs from the frozen stage-three decision')
    if frozen['input_hashes'] != input_hashes(study, out):
        raise ValueError('Analysis or observations changed since the control decision')
    receipt_path = out/'CONTROL_RECEIPT.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['stage3_decision_hash'] != sha_file(out/'STAGE3_DECISION.json') or receipt['results_hash'] != sha_file(out/'CONTROL_RESULTS.csv'):
            raise ValueError('Control replay conflicts with committed results')
        return dict(receipt, status='ALREADY_COMMITTED')
    started = time.perf_counter()
    records = load_records(study, out)
    rows = _calibration_controls(study, out, records, frozen['controls']['C_CAL'])
    rows += _background_controls(study, out, frozen['controls']['C_BG'])
    write_table(out/'CONTROL_RESULTS.csv', rows)
    receipt = dict(status='COMPLETED', stage3_decision_hash=sha_file(out/'STAGE3_DECISION.json'),
        results_hash=sha_file(out/'CONTROL_RESULTS.csv'), control_seconds=time.perf_counter()-started,
        c_cal_fits=sum(r['control_id'] == 'C_CAL' and r['status'] == 'COMPLETED' for r in rows),
        c_bg_states=len({(r['job_id'], r['epoch'], r['group_id'], r['background'])
                        for r in rows if r['control_id'] == 'C_BG' and r['status'] == 'COMPLETED'}),
        new_training_steps=0, fresh_test_accessed=False)
    write_json(receipt_path, receipt)
    return receipt


def _control_summaries(out, contrasts):
    out = Path(out)
    controls = pd.read_csv(out/'CONTROL_RESULTS.csv')
    cal, background = [], []
    complete = controls[controls['status'] == 'COMPLETED']
    if 'readout_id' in complete:
        table = complete[complete['control_id'] == 'C_CAL']
        if not table.empty:
            for key, block in table.groupby(['dataset', 'objective', 'seed', 'epoch', 'cell', 'row_fold_direction'], sort=False):
                by = {r['readout_id']: r for r in block.to_dict('records')}
                raw = by['F']['original_logloss']-by['online']['original_logloss']
                changed = by['F']['calibrated_logloss']-by['online']['calibrated_logloss']
                cal.append(dict(zip(['dataset', 'objective', 'seed', 'epoch', 'cell', 'row_fold_direction'], key),
                    raw_F_minus_online_LL=raw, calibrated_F_minus_online_LL=changed,
                    absolute_LL_gap_shrinkage=abs(raw)-abs(changed), signed_LL_gap_change=raw-changed))
    table = complete[complete['control_id'] == 'C_BG']
    if not table.empty:
        index = contrasts[(contrasts['contrast'] == 'fast_minus_slow') & (contrasts['metric'] == 'auc')].set_index(
            ['job_id', 'epoch', 'panel_id', 'cell', 'group_id'])
        all_rows = pd.read_csv(out/'PULSE_READOUTS.csv')
        for row in table.to_dict('records'):
            d, t = int(row['domain']), int(row['task'])
            cell = f'd{d}t{t}'
            base_variant = 'all_slow' if row['background'] == 'all_slow_elsewhere' else 'all_fast'
            base = all_rows[(all_rows['job_id'] == row['job_id']) & (all_rows['epoch'] == row['epoch']) &
                            (all_rows['panel_id'] == row['panel_id']) & (all_rows['variant'] == base_variant) &
                            (all_rows['domain'] == d) & (all_rows['task'] == t)]
            if len(base) != 1:
                raise ValueError('Second-background baseline readout unavailable')
            effect = row['auc']-base.iloc[0]['auc'] if row['background'] == 'all_slow_elsewhere' else base.iloc[0]['auc']-row['auc']
            f_effect = float(index.loc[(row['job_id'], row['epoch'], row['panel_id'], cell, row['group_id']), 'value'])
            background.append(dict(dataset=row['dataset'], objective=row['objective'], seed=row['seed'],
                epoch=row['epoch'], cell=cell, panel_id=row['panel_id'], group_id=row['group_id'],
                background=row['background'], F_background_fast_minus_slow=f_effect,
                second_background_fast_minus_slow=effect, delta=effect-f_effect,
                sign_flip=bool(effect*f_effect < 0), state_hash=row['state_hash']))
    write_table(out/'CALIBRATION_SUMMARY.csv', cal)
    write_table(out/'BACKGROUND_SUMMARY.csv', background)
    return cal, background


def analyze_stage(study, out, stage):
    out = Path(out)
    if stage not in (1, 2, 3, 4):
        raise ValueError('Unknown frozen analysis stage')
    required = 'AUDIT_INVENTORY.json' if stage == 1 else f'STAGE{stage-1}_DECISION.json'
    if not (out/required).exists():
        raise Incomplete('Analysis dependency missing: '+required)
    records = load_records(study, out)
    started = time.perf_counter()
    curves, readouts, features = materialize_tables(study, out, records)
    result = dict(schema_version='1.0', study_id=study.manifest['study_id'], stage=stage,
                  input_hashes=input_hashes(study, out), scientific_seed_count=5,
                  fresh_test_accessed=False, status='COMPLETED', decisions=[])
    if stage == 1:
        selection, gains = _stage1_selection(study, records)
        write_table(out/'SELECTION_DIAGNOSTICS.csv', selection)
        write_table(out/'VALIDATION_GAINS.csv', gains)
        gain_summary = summarize_seeds(gains, ['dataset', 'objective', 'phase', 'comparison', 'cell', 'metric'])
        write_table(out/'VALIDATION_GAIN_SUMMARY.csv', gain_summary)
        history_preferences(study, out, records)
        temporal_rank_stability(study, out, records)
        result['decisions'] = gain_summary[gain_summary['cell'] == 'macro'].to_dict('records')
        result['next_action'] = 'ANALYZE_STAGE2_FROM_EXISTING_OBSERVATIONS'
    elif stage == 2:
        contrasts, interactions = pulse_contrasts(records)
        write_table(out/'PULSE_CONTRASTS.csv', contrasts)
        write_table(out/'GROUP_INTERACTIONS.csv', interactions)
        pulse_summary = summarize_seeds(contrasts,
            ['dataset', 'objective', 'panel_id', 'cell', 'group_id', 'functional_support', 'metric', 'contrast'])
        write_table(out/'PULSE_SUMMARY.csv', pulse_summary)
        rank_flips(study, out, records)
        metric_correspondence(features, contrasts, out)
        result['decisions'] = pulse_summary[(pulse_summary['panel_id'] == 'V_mech') &
            (pulse_summary['metric'] == 'auc') & (pulse_summary['contrast'] == 'fast_minus_slow')].to_dict('records')
        result['next_action'] = 'ANALYZE_STAGE3_FIXED_PREDICTORS'
    elif stage == 3:
        fit_predictive_folds(study, out)
        contrasts, interactions = pulse_contrasts(records)
        result.update(metric_primary_id=study.manifest['predictive']['primary_id'],
                      controls=_control_triggers(study, records, interactions, contrasts),
                      next_action='CONTROLS_OR_FREEZE', decisions=_decision_rows(study, out))
    else:
        if not (out/'CONTROL_RECEIPT.json').exists():
            raise Incomplete('Finite controls must complete or record NOT_TRIGGERED before S4')
        contrasts, _ = pulse_contrasts(records)
        cal, background = _control_summaries(out, contrasts)
        previous = read_json(out/'STAGE3_DECISION.json')
        result.update(next_action='FREEZE_ENDPOINTS', controls=previous['controls'],
                      control_receipt_hash=sha_file(out/'CONTROL_RECEIPT.json'),
                      decisions=previous['decisions'], new_training_steps=0)
        write_json(out/'MECHANISM_DECISIONS.json', _plain(_mechanism_decisions(study, out, previous['decisions'])))
    result['analysis_seconds'] = time.perf_counter()-started
    write_json(out/f'STAGE{stage}_DECISION.json', _plain(result))
    return result


def _effect_evidence(values):
    stats = seed_statistics(values)
    strength = 'WEAK' if stats['n_seeds'] < 5 else (
        'STRONG' if stats['ci95_low'] > 0 or stats['ci95_high'] < 0 else 'MEDIUM')
    return _plain(stats), strength


def _mechanism_decisions(study, out, actions):
    out = Path(out)
    gain = pd.read_csv(out/'VALIDATION_GAINS.csv')
    preferences = pd.read_csv(out/'HISTORY_PREFERENCES.csv')
    correspondence = pd.read_csv(out/'METRIC_CORRESPONDENCE.csv')
    results = []
    for dataset in study.manifest['datasets']:
        for objective in CONDITIONS:
            late = gain[(gain['dataset'] == dataset) & (gain['objective'] == objective) &
                        (gain['comparison'] == 'G_same') & (gain['phase'] == 'late') &
                        (gain['cell'] == 'macro') & (gain['metric'] == 'auc')].groupby('seed')['value'].mean()
            effect, strength = _effect_evidence(late.values)
            results.append(dict(question_id='Q1', hypothesis_id='H1', claim_level='L2',
                status='HISTORY_READOUT_GAIN' if late.mean() > 0 else 'HISTORY_READOUT_COST',
                scope=dict(dataset=dataset, objective=objective, phase='late', metric='macro_auc'),
                effect_size=effect, evidence_strength=strength, n_seeds=5,
                source_artifact='VALIDATION_GAINS.csv',
                competing_explanation_disposition='Common-epoch comparison separates the readout effect from independent checkpoint selection; age-matched online and temporal rank tables assess lag.',
                next_research_decision='Retain the measured conditional readout effect; interpret lag and pulse evidence jointly.'))
            pref = preferences[(preferences['dataset'] == dataset) & (preferences['objective'] == objective) &
                               (preferences['mode'] == 'CELL_CONDITIONAL_DIAGNOSTIC')].groupby('seed')['value'].mean()
            effect, strength = _effect_evidence(pref.values)
            results.append(dict(question_id='Q2', hypothesis_id='H2', claim_level='L2',
                status='CROSS_EVALUATED_PREFERENCE_SPACE' if pref.mean() > 0 else 'NO_AVERAGE_CROSS_EVALUATED_PREFERENCE_GAIN',
                scope=dict(dataset=dataset, objective=objective, epochs=[10, 20, 30, 40, 50]),
                effect_size=effect, evidence_strength=strength, n_seeds=5,
                source_artifact='HISTORY_PREFERENCES.csv',
                competing_explanation_disposition='Same candidate count, leave-one-seed-out and opposite Validation-row selection/scoring compare conditional against uniform history preference.',
                next_research_decision='Combine with independent future action utility; keep all cell responses.'))
            cor = correspondence[(correspondence['dataset'] == dataset) & (correspondence['objective'] == objective) &
                                 (correspondence['target_panel'] == 'V_mech') & (correspondence['target_metric'] == 'auc')]
            results.append(dict(question_id='Q3', hypothesis_id='H3', claim_level='L3', status='DESCRIPTIVE_CORRESPONDENCE',
                scope=dict(dataset=dataset, objective=objective), effect_size=_plain(cor.to_dict('records')),
                evidence_strength='MEDIUM', n_seeds=5, source_artifact='METRIC_CORRESPONDENCE.csv',
                competing_explanation_disposition='Simultaneous train-to-Validation correspondence is separate from forward prediction; P_B and source-detach diagnostics identify sample and derivative differences.',
                next_research_decision='Use the registered future folds to judge action value.'))
            scope_actions = [r for r in actions if r['dataset'] == dataset and r['objective'] == objective]
            useful = [r for r in scope_actions if r['status'] == 'ACTION_USEFUL']
            status = 'READY_FOR_TARGETED_DESIGN' if useful else (
                'DESCRIPTIVE_ONLY' if pref.mean() > 0 else 'UNIFORM_HISTORY_SUFFICIENT_IN_SCOPE')
            results.append(dict(question_id='Q4', hypothesis_id='H3', claim_level='L4', status=status,
                scope=dict(dataset=dataset, objective=objective, predictive_scope='all_preregistered_condition_cell_group_scopes'),
                effect_size=_plain(scope_actions), evidence_strength='MEDIUM', n_seeds=5,
                source_artifact='PREDICTION_SEED.csv',
                competing_explanation_disposition='Actions compare against F and a fit-fold uniform action on the same single-group opportunities; all five simple references are retained.',
                next_research_decision='Targeted algorithm design is supported in the reported useful scopes only.' if useful else 'Stop expanding metrics in this study; retain the fixed history reference.'))
        own = gain[(gain['dataset'] == dataset) & (gain['comparison'] == 'G_same') & (gain['metric'] == 'auc') &
                   (gain['cell'] == 'macro') & (gain['phase'] == 'late')].groupby(['seed', 'objective'])['value'].mean().unstack()
        values = own['gaussian_d0']-own['clean']
        effect, strength = _effect_evidence(values.values)
        results.append(dict(question_id='Q5', hypothesis_id='H4', claim_level='L2', status='CONDITIONAL_GAUSSIAN_INTERACTION',
            scope=dict(dataset=dataset, phase='late', metric='macro_auc'), effect_size=effect,
            evidence_strength=strength, n_seeds=5, source_artifact='VALIDATION_GAINS.csv',
            competing_explanation_disposition='Clean and Gaussian are distinct online objectives; the difference is between within-trajectory EMA gains, with selected/fixed and split-specific evidence kept separate.',
            next_research_decision='Report the interaction by stage and cell; preserve the frozen objective scope.'))
    return dict(schema_version='1.0', study_id=study.manifest['study_id'], input_hashes=input_hashes(study, out),
                scientific_unit='seed', decisions=results, all_registered_action_scopes=actions,
                new_algorithm_designed=False)


def _endpoint_statistics(endpoints, out):
    metric_names = [k for k in endpoints if k.endswith(('_auc', '_logloss')) and pd.api.types.is_numeric_dtype(endpoints[k])]
    keys = ['dataset', 'objective', 'split', 'checkpoint', 'readout_id']
    absolute = []
    for key, group in endpoints.groupby(keys, sort=False):
        if set(group['seed']) != set(SEEDS) or len(group) != 5:
            raise Incomplete('Endpoint absolute statistics require exactly five seeds')
        for metric in metric_names:
            absolute.append(dict(zip(keys, key), metric=metric, **seed_statistics(group[metric]),
                                 seed_values={str(int(r['seed'])): r[metric] for r in group.to_dict('records')}))
    write_table(Path(out)/'ENDPOINT_ABSOLUTE_SUMMARY.csv', absolute)
    paired = endpoint_pairs(endpoints)
    extra = []
    for key, block in endpoints.groupby(['dataset', 'objective', 'seed', 'split', 'checkpoint'], sort=False):
        meta = dict(zip(['dataset', 'objective', 'seed', 'split', 'checkpoint'], key))
        by = {r['readout_id']: r for r in block.to_dict('records')}
        for rid in by:
            if rid in ('online', 'F'):
                continue
            for reference in ('online', 'F'):
                for metric in metric_names:
                    cell, name = metric.rsplit('_', 1)
                    extra.append(dict(meta, contrast=rid+'_minus_'+reference, cell=cell, metric=name,
                        arm_history=by[rid][metric], arm_reference=by[reference][metric],
                        history_epoch=by[rid]['epoch'], reference_epoch=by[reference]['epoch'],
                        value=by[rid][metric]-by[reference][metric]))
    paired = pd.concat([paired, pd.DataFrame(extra)], ignore_index=True)
    write_table(Path(out)/'PAIRED_SEED.csv', paired)
    summary = summarize_seeds(paired, ['dataset', 'objective', 'split', 'checkpoint', 'contrast', 'cell', 'metric'])
    write_table(Path(out)/'PAIRED_SUMMARY.csv', summary)
    return paired, summary


def _cost_table(study, out, records):
    out = Path(out)
    ipc = []
    if (out/'IPC_COSTS.jsonl').exists():
        ipc = [json.loads(line) for line in (out/'IPC_COSTS.jsonl').read_text().splitlines() if line.strip()]
    replays = []
    if (out/'REPLAYS.jsonl').exists():
        replays = [json.loads(line) for line in (out/'REPLAYS.jsonl').read_text().splitlines() if line.strip()]
    else:
        (out/'REPLAYS.jsonl').touch()
    test = pd.read_csv(out/'TEST_COSTS.csv')
    rows = []
    for job in _jobs(study):
        history = [r for r in records if r['job_id'] == job['job_id']]
        cost_keys = set(k for r in history for k in r.get('cost', {}))
        cost = {k: (max if k.startswith('peak_') else sum)(r['cost'].get(k, 0) for r in history) for k in cost_keys}
        replay_costs = [r for r in replays if r.get('job_id') == job['job_id']]
        row = dict(job_id=job['job_id'], dataset=job['dataset'], objective=job['objective'], seed=job['seed'],
            logical_trajectories=1, logical_epochs=50, logical_adam_steps=job['adam_steps'],
            physical_replayed_epochs=sum(r.get('status') == 'REPLAY_VERIFIED' for r in replay_costs),
            physical_extra_full_attempts=sum(r.get('status') == 'RAM_LOSS_REPLAY' for r in replay_costs),
            ipc_seconds=sum(r['ipc_seconds'] for r in ipc if r['job_id'] == job['job_id']),
            test_seconds=float(test.loc[test['job_id'] == job['job_id'], 'test_seconds'].sum()),
            prediction_disk_bytes=sum(p.stat().st_size for p in (out/'predictions'/job['job_id']).rglob('*') if p.is_file()),
            replay_cost_records=replay_costs, **cost)
        rows.append(row)
    write_table(out/'COSTS.csv', rows)
    global_rows = []
    for stage in range(1, 5):
        decision = read_json(out/f'STAGE{stage}_DECISION.json')
        global_rows.append(dict(stage=f'S{stage}', analysis_seconds=decision['analysis_seconds']))
    global_rows.append(dict(stage='S4_controls', analysis_seconds=read_json(out/'CONTROL_RECEIPT.json')['control_seconds']))
    write_table(out/'ANALYSIS_COSTS.csv', global_rows)
    return pd.DataFrame(rows)


def _save_figure(fig, directory, name):
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/(name+'.png'), dpi=160, bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def _figures(study, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    directory = out/'figures'
    curves = pd.read_csv(out/'VALIDATION_CURVES.csv')
    combos = [(ds, obj) for ds in study.manifest['datasets'] for obj in CONDITIONS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (ds, obj) in zip(axes.ravel(), combos):
        subset = curves[(curves['dataset'] == ds) & (curves['objective'] == obj)]
        for rid, block in subset.groupby('readout_id', sort=False):
            mean = block.groupby('epoch')['macro_auc'].mean()
            axis.plot(mean.index, mean.values, label=rid)
            for r in block.loc[block.groupby('seed')['macro_auc'].idxmax()].to_dict('records'):
                axis.scatter(r['epoch'], r['macro_auc'], s=9, alpha=.35)
        for boundary in (10, 30):
            axis.axvline(boundary, color='grey', linestyle=':', linewidth=.8)
        axis.set(title=ds+' / '+obj, xlabel='epoch', ylabel='complete Validation Macro AUC')
        axis.legend(fontsize=8)
    _save_figure(fig, directory, 'TIME_CURVES')
    for ds, obj in combos:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
        subset = curves[(curves['dataset'] == ds) & (curves['objective'] == obj)]
        for axis, cell in zip(axes.ravel(), CELLS):
            for rid, block in subset.groupby('readout_id', sort=False):
                mean = block.groupby('epoch')[cell+'_auc'].mean()
                axis.plot(mean.index, mean.values, label=rid)
            axis.set(title=cell, xlabel='epoch', ylabel='Validation AUC')
            axis.legend(fontsize=7)
        _save_figure(fig, directory, 'SIX_CELL_CURVES_'+ds+'_'+obj)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    for row, (ds, obj) in enumerate(combos):
        subset = curves[(curves['dataset'] == ds) & (curves['objective'] == obj)].copy()
        subset['phase'] = subset['epoch'].map(_phase)
        for col, phase in enumerate(('early', 'middle', 'late')):
            block = subset[subset['phase'] == phase]
            means = block.groupby('readout_id')[[cell+'_auc' for cell in CELLS]].mean()
            base = means.loc['online']
            means = means.drop('online').subtract(base, axis=1)
            axis = axes[row, col]
            image = axis.imshow(means.to_numpy().T, aspect='auto', cmap='coolwarm')
            axis.set_xticks(range(len(means)), means.index, rotation=45)
            axis.set_yticks(range(6), CELLS)
            axis.set_title(ds+' / '+obj+' / '+phase, fontsize=9)
            fig.colorbar(image, ax=axis, shrink=.7)
    _save_figure(fig, directory, 'HISTORY_PREFERENCE_HEATMAP')
    pulse = pd.read_csv(out/'PULSE_CONTRASTS.csv')
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)
    for row, (ds, obj) in enumerate(combos):
        for col, contrast in enumerate(('fast_minus_F', 'slow_minus_F', 'fast_minus_slow')):
            subset = pulse[(pulse['dataset'] == ds) & (pulse['objective'] == obj) & (pulse['panel_id'] == 'V_mech') &
                           (pulse['metric'] == 'auc') & (pulse['contrast'] == contrast) & (pulse['group_id'] != 'ALL')]
            matrix = subset.groupby(['group_id', 'cell'])['value'].mean().unstack().reindex(index=GROUPS, columns=CELLS)
            axis = axes[row, col]
            image = axis.imshow(matrix.to_numpy(), aspect='auto', cmap='coolwarm')
            axis.set_xticks(range(6), CELLS, rotation=45)
            axis.set_yticks(range(12), GROUPS, fontsize=7)
            axis.set_title(ds+' / '+obj+' / '+contrast, fontsize=8)
            fig.colorbar(image, ax=axis, shrink=.6)
    _save_figure(fig, directory, 'PULSE_EFFECTS')
    temporal = pd.read_csv(out/'TEMPORAL_RANK_STABILITY.csv')
    gain = pd.read_csv(out/'VALIDATION_GAINS.csv')
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), constrained_layout=True)
    for row, (ds, obj) in enumerate(combos):
        subset = temporal[(temporal['dataset'] == ds) & (temporal['objective'] == obj) & (temporal['panel_id'] == 'V_mech')]
        for rid, block in subset.groupby('readout_id'):
            mean = block.groupby('epoch')['temporal_change'].mean()
            axes[row, 0].plot(mean.index, mean, label=rid)
        axes[row, 0].set(title=ds+' / '+obj, ylabel='fixed-pair temporal rank change', xlabel='epoch')
        axes[row, 0].legend()
        subset = gain[(gain['dataset'] == ds) & (gain['objective'] == obj) & (gain['cell'] == 'macro') &
                      (gain['comparison'] == 'G_same')]
        for metric, block in subset.groupby('metric'):
            mean = block.groupby('epoch')['value'].mean()
            axes[row, 1].plot(mean.index, mean, label='F-online '+metric)
        axes[row, 1].axhline(0, color='grey', linewidth=.7)
        axes[row, 1].set(xlabel='epoch', ylabel='raw metric difference')
        axes[row, 1].legend(fontsize=8)
    _save_figure(fig, directory, 'RANK_STABILITY_AND_LL')
    flips = pd.read_csv(out/'RANK_FLIPS.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (ds, obj) in zip(axes.ravel(), combos):
        subset = flips[(flips['dataset'] == ds) & (flips['objective'] == obj) & (flips['panel_id'] == 'V_mech')]
        for name in ('correction', 'damage', 'new_ties', 'broken_ties'):
            means = subset.groupby('epoch')[name].mean()
            axis.plot(means.index, means, label=name)
        axis.set(title=ds+' / '+obj, xlabel='pulse endpoint epoch', ylabel='sampled-pair fraction')
        axis.legend(fontsize=8)
    _save_figure(fig, directory, 'RANK_FLIPS')
    summary = pd.read_csv(out/'PREDICTION_SUMMARY.csv')
    methods = list(study.manifest['predictive']['feature_sets'])+list(study.manifest['predictive']['raw_action_diagnostics'])+['UNIFORM_ACTION']
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), constrained_layout=True)
    for row, (ds, obj) in enumerate(combos):
        for col, metric in enumerate(('MAE_D', 'chosen_minus_uniform_action')):
            subset = summary[(summary['dataset'] == ds) & (summary['objective'] == obj) &
                             (summary['scope'] == 'CONDITION') & (summary['metric'] == metric)].set_index('model_id')
            chosen = [m for m in methods if m in subset.index and pd.notna(subset.loc[m, 'mean'])]
            axes[row, col].bar(range(len(chosen)), [subset.loc[m, 'mean'] for m in chosen],
                               yerr=[subset.loc[m, 'sample_sd'] for m in chosen], capsize=2)
            axes[row, col].set_xticks(range(len(chosen)), chosen, rotation=65, fontsize=7)
            axes[row, col].set(title=ds+' / '+obj, ylabel=metric+' (mean +/- seed SD)')
            axes[row, col].axhline(0, color='grey', linewidth=.5)
    _save_figure(fig, directory, 'PREDICTIVE_COMPARISON')


def _markdown_table(frame, columns=None, digits=8):
    if columns is not None:
        frame = frame[columns]
    if frame.empty:
        return 'No applicable rows.\n'
    def value(v):
        if isinstance(v, float):
            return f'{v:.{digits}f}' if math.isfinite(v) else 'NA'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])+'\n'


def generate_report(study, out):
    out = Path(out)
    from .evaluation import validate_freeze
    validate_freeze(study, out)
    if not (out/'TEST_EVALUATION_RECEIPT.json').exists():
        raise Incomplete('Final report requires frozen Test evaluation')
    records = load_records(study, out)
    endpoints = pd.read_csv(out/'ENDPOINTS.csv')
    paired, summary = _endpoint_statistics(endpoints, out)
    costs = _cost_table(study, out, records)
    _figures(study, out)
    decisions = read_json(out/'MECHANISM_DECISIONS.json')
    prepared = read_json(out/'PREPARED.json')
    panel_counts = []
    for dataset in study.manifest['datasets']:
        for seed in SEEDS:
            panel = panel_identity(out, dataset, seed, 'V_mech')
            valid = _validation_mask(panel)
            for domain in range(3):
                panel_counts.append(dict(dataset=dataset, seed=seed, domain=domain,
                    actual_count=int(valid[domain].sum()), container_capacity=valid.shape[1],
                    V_A_count=int((valid[domain] & (panel['validation_row_side'][domain] == 0)).sum()),
                    V_B_count=int((valid[domain] & (panel['validation_row_side'][domain] == 1)).sum())))
    write_table(out/'VALIDATION_PANEL_COUNTS.csv', panel_counts)
    stages = {stage: read_json(out/f'STAGE{stage}_DECISION.json') for stage in range(1, 5)}
    # All numbers below are table-derived. The orchestrator supplies the final scientific synthesis.
    lines = ['# EMA 收益机制实验报告', '',
        '研究范围：M3oE fixed1，MovieLens/KuaiRand，clean/Gaussian D0，每条件五 seed（2022–2026），每轨迹50 epoch。',
        '完成20条逻辑轨迹、1000 epoch、34500次Adam；常驻shadow、pulse与观测均无训练反馈。', '',
        '## 核心端点与逐 seed 配对', '',
        '下表为各读出独立以完整 Validation Macro AUC 最早严格最大值选点后的 Test 差。完整 selected/E10/E50、六格、Macro、Traffic、Pooled、绝对值与seed原值见 [ENDPOINTS.csv](ENDPOINTS.csv)、[PAIRED_SEED.csv](PAIRED_SEED.csv)、[PAIRED_SUMMARY.csv](PAIRED_SUMMARY.csv) 和 [ENDPOINT_ABSOLUTE_SUMMARY.csv](ENDPOINT_ABSOLUTE_SUMMARY.csv)。', '']
    selected = summary[(summary['split'] == 'test') & (summary['checkpoint'] == 'selected') &
                       (summary['cell'] == 'macro') & (summary['contrast'].isin(['F_minus_online', 'Gaussian_EMA_gain_minus_clean_EMA_gain']))]
    lines.append(_markdown_table(selected, ['dataset', 'objective', 'metric', 'mean', 'sample_sd', 'ci95_low', 'ci95_high', 'positive_seeds']))
    selected_seed = paired[(paired['split'] == 'test') & (paired['checkpoint'] == 'selected') &
                          (paired['cell'] == 'macro') & (paired['contrast'].isin(['F_minus_online', 'Gaussian_EMA_gain_minus_clean_EMA_gain']))]
    lines.append(_markdown_table(selected_seed, ['dataset', 'objective', 'seed', 'metric', 'value']))
    lines += ['', '## Q1–Q5：分层研究决定', '',
              '所有区间以五个训练 seed 的配对差计算；同轨迹多个 epoch、cell、panel、shadow 不增加独立样本数。', '']
    titles = dict(Q1='历史读出、训练阶段与有效滞后', Q2='六格、时间尺度与路径偏好',
                  Q3='变化量、功能方向与实际收益的对应', Q4='未来预测和动作价值', Q5='Gaussian 与 EMA 交互')
    for question, title in titles.items():
        lines += ['### '+question+' '+title, '']
        for decision in [d for d in decisions['decisions'] if d['question_id'] == question]:
            scope = decision['scope']
            lines.append(f"- {scope['dataset']} / {scope.get('objective', 'two objectives')}：`{decision['status']}`，证据 **{decision['evidence_strength']}**。")
            effect = decision['effect_size']
            if isinstance(effect, dict) and 'mean' in effect:
                lines.append(f"  五seed均值 {effect['mean']:+.9f}，样本SD {effect['sample_sd']:.9f}，95% CI [{effect['ci95_low']:+.9f}, {effect['ci95_high']:+.9f}]。")
            lines.append(f"  证据：[表]({decision['source_artifact']})；{decision['competing_explanation_disposition']} {decision['next_research_decision']}")
        lines.append('')
    lines += ['## 预测比较与局部研究范围', '',
        '固定八类ridge（SSE＋1×系数平方和、截距不罚）和四项raw动作诊断，与同折UNIFORM_ACTION比较。拟合只用其他四seed早期origins5/10/15与V_A未来标签；评分用留出seed晚期origins25/30/35/40/45与V_B，再交换。共160个ridge fit。',
        '动作净益是单组读出机会的平均。各condition/cell/group/cell×group范围、MAE、相对全部简单参照skill、方向率、regret、AUC/LL动作效用在 [PREDICTIVE_OOF.csv](PREDICTIVE_OOF.csv)、[PREDICTION_SEED.csv](PREDICTION_SEED.csv)、[PREDICTION_SUMMARY.csv](PREDICTION_SUMMARY.csv) 与 [PREDICTIVE_COMPARISONS.csv](PREDICTIVE_COMPARISONS.csv)。', '']
    predictive = pd.read_csv(out/'PREDICTION_SUMMARY.csv')
    main = predictive[(predictive['scope'] == 'CONDITION') & predictive['metric'].isin(['MAE_D', 'chosen_minus_F', 'chosen_minus_uniform_action'])]
    lines.append(_markdown_table(main, ['dataset', 'objective', 'model_id', 'metric', 'mean', 'sample_sd']))
    useful = [r for r in decisions['all_registered_action_scopes'] if r['status'] == 'ACTION_USEFUL']
    write_table(out/'ACTION_DECISIONS.csv', decisions['all_registered_action_scopes'])
    lines += ['', f'预注册范围中满足正向动作净益且胜统一动作的记录数：{len(useful)}。全部有利和不利范围保留于 [ACTION_DECISIONS.csv](ACTION_DECISIONS.csv)，效应小于.0002者标 LOW_PRIORITY_SMALL_EFFECT。', '',
        '## 有限控制与竞争解释', '',
        'C_CAL/C_BG 的触发依据和冻结组见 [STAGE3_DECISION.json](STAGE3_DECISION.json)。温度仅重算源LL、AUC保留原p排序；第二背景只读取E20/E40保留状态，均不新增训练。',
        '[CONTROL_RESULTS.csv](CONTROL_RESULTS.csv)、[CALIBRATION_SUMMARY.csv](CALIBRATION_SUMMARY.csv)、[BACKGROUND_SUMMARY.csv](BACKGROUND_SUMMARY.csv) 保存实际控制数值或 NOT_TRIGGERED。',
        '固定历史组合与较合适旧online的竞争解释由 [VALIDATION_GAINS.csv](VALIDATION_GAINS.csv) 的年龄匹配对比约束；低波动与延后改进由 [TEMPORAL_RANK_STABILITY.csv](TEMPORAL_RANK_STABILITY.csv) 分开检查。',
        '单组可加性由 [GROUP_INTERACTIONS.csv](GROUP_INTERACTIONS.csv) 与第二背景读出检查。P_A/P_B样本敏感性和full/source_detach差异保留在 [PANEL_STABILITY.csv](PANEL_STABILITY.csv)、[METRIC_CORRESPONDENCE.csv](METRIC_CORRESPONDENCE.csv)。', '',
        '## 冻结、来源与修订', '',
        f"source commit：`{study.identity['source_commit']}`；implementation hash：`{study.identity['implementation_hash']}`；manifest hash：`{study.manifest_hash}`。",
        f"plan hash：`{study.identity['plan_hash']}`；analysis hash：`{input_hashes(study,out)['analysis_hash']}`。",
        '每个端点身份见 [SELECTION_FROZEN.json](SELECTION_FROZEN.json)；与源、数据、面板、分析、阶段决定的绑定见 [FREEZE_RECEIPT.json](FREEZE_RECEIPT.json)。Test由独立进程读取，归档同state复用单列，pulse/条件最优/温度不进入Test。',
        '数据身份与原CSV/cache/row/split/encoder/scaler：[DATA_MANIFEST.json](DATA_MANIFEST.json)、[PROBE_MANIFEST.json](PROBE_MANIFEST.json)。既有证据和A/B去重：[AUDIT_INVENTORY.json](AUDIT_INVENTORY.json)、[EVIDENCE_REUSE.csv](EVIDENCE_REUSE.csv)、[ARCHIVED_FACTORIAL.csv](ARCHIVED_FACTORIAL.csv)。',
        'V_mech每域有效自然行数为min(4096,该域Validation人数)，[VALIDATION_PANEL_COUNTS.csv](VALIDATION_PANEL_COUNTS.csv)记录逐seed有效人数及V_A/V_B人数。固定容量容器以indices≥0解析真实行；指标、排序pair、偏好、校准和未来标签均按真实行计算。完整Validation选点与Test仍使用源完整人口。',
        '设计、实现、审查和运行修订记录由研究交付目录中的 DESIGN_SPECIFICATION、CODER_IMPLEMENTATION_REPORT、IMPLEMENTATION_REVIEW 与原始运行日志提供。', '',
        '## 成本与产物', '',
        f"本轮实际逻辑Adam数：{int(costs['real_adam_steps'].sum())}；完整V模型读出：{int(costs['validations'].sum())}；新增pulse状态：{int(costs['pulse_state_readouts'].sum())}；M2小面板梯度：{int(costs['gradient_probes'].sum())}。",
        f"训练累计worker秒 {costs['training_seconds'].sum():.3f}；完整V/regular观测累计秒 {costs['validation_seconds'].sum():.3f}；pulse累计秒 {costs['pulse_seconds'].sum():.3f}；M2累计秒 {costs['gradient_seconds'].sum():.3f}；IPC累计秒 {costs['ipc_seconds'].sum():.3f}。",
        '分项worker时间包含共享设备等待；墙钟、GPU峰值、RAM、预测字节及故障物理成本分别保留于 [COSTS.csv](COSTS.csv)、[ANALYSIS_COSTS.csv](ANALYSIS_COSTS.csv)、[TEST_COSTS.csv](TEST_COSTS.csv)、[REPLAYS.jsonl](REPLAYS.jsonl)。',
        '标准PNG/PDF图及底表：[时间曲线](figures/TIME_CURVES.png)、[历史偏好](figures/HISTORY_PREFERENCE_HEATMAP.png)、[脉冲效应](figures/PULSE_EFFECTS.png)、[排序与LL](figures/RANK_STABILITY_AND_LL.png)、[排序翻转](figures/RANK_FLIPS.png)、[预测比较](figures/PREDICTIVE_COMPARISON.png)。', '',
        '本轮到S4、冻结Test和报告后结束；后续定向设计按上面的具体适用范围决定。']
    report = '\n'.join(lines)+'\n'
    (out/'FINAL_REPORT.md').write_text(report)
    receipt = dict(status='COMPLETED', report_hash=sha_file(out/'FINAL_REPORT.md'),
        input_hashes=input_hashes(study, out), artifact_hashes={str(p.relative_to(out)): sha_file(p)
            for p in sorted(out.iterdir()) if p.is_file() and p.name != 'REPORT_RECEIPT.json'},
        figures={str(p.relative_to(out)): sha_file(p) for p in sorted((out/'figures').glob('*'))},
        missing_required_artifacts=[name for name in study.manifest['artifacts']['required_names']
                                    if name != 'RELEASE_RECEIPT.json' and not (out/name).exists()])
    if receipt['missing_required_artifacts']:
        raise Incomplete('Required report artifacts missing: '+str(receipt['missing_required_artifacts']))
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size > study.manifest['resource']['observation_disk_cap_bytes']:
        raise MemoryError('RESOURCE_BLOCKED: report exceeds the observation disk cap')
    receipt['observation_disk_bytes'] = size
    write_json(out/'REPORT_RECEIPT.json', receipt)
    return receipt
