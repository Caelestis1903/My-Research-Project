"""Recomputable absolute and paired seed tables, without research interpretation."""
import json
import statistics
from pathlib import Path
from experiments.overnight_ab_20261103.reporting import write_csv, stats
from experiments.overnight_ab_20261103.selection import CELLS
from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file
from . import config

REFERENCES = ('G-D2', 'FIXED_EMA', 'M3OE', 'HYPERGATE', 'AUTOIFS')
CANDIDATES = ('G-D0', 'G-D02', 'G-D2')


def committed_chain(attempt, key):
    """The latest full replay starts at commit one; the shared ledger stays append-only."""
    path = Path(attempt)/'epoch_commit_ledger.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rows = [row for row in rows if row['key']==key]
    if not rows:
        return []
    start = max(i for i,row in enumerate(rows) if row['commit_index']==1)
    chain = rows[start:]
    if [row['commit_index'] for row in chain]!=list(range(1,len(chain)+1)):
        raise ValueError('Committed replay chain is nonconsecutive: '+key)
    return chain


def endpoint_rows(attempt):
    attempt = Path(attempt)
    rows = read_json(attempt/'REUSED_ENDPOINTS.json')
    for path in sorted((attempt/'runs').glob('*/DONE.json')):
        done = read_json(path)
        if any(sha_file(path.parent/name) != expected for name, expected in done['files'].items()):
            raise ValueError(f'Completed endpoint archive changed: {path}')
        rows.extend(read_json(path.parent/'endpoints.json'))
    identities = [(r['arm'], r['seed'], r['estimator'], r['checkpoint'], r['split']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate endpoint identity')
    return rows


def summarize(attempt):
    attempt = Path(attempt)
    prepared = read_json(attempt/'PREPARED.json')
    destination = config.REPORT
    rows = endpoint_rows(attempt)
    write_csv(destination/'ENDPOINTS.csv', rows)
    metrics = sorted({k for row in rows for k in row if k.endswith(('_auc', '_logloss'))})
    grouped = {}
    for row in rows:
        key = (row['arm'], row['estimator'], row['checkpoint'], row['split'])
        grouped.setdefault(key, []).append(row)
    absolute = []
    for key, values in sorted(grouped.items()):
        for metric in metrics:
            data = [r[metric] for r in values if r.get(metric) is not None]
            absolute.append(dict(zip(('arm', 'estimator', 'checkpoint', 'split'), key),
                                 metric=metric, **stats(data)))
    write_csv(destination/'ABSOLUTE_SUMMARY.csv', absolute)
    selected = [r for r in rows if r['checkpoint']=='selected' and r['split']=='test'
                and r['estimator']==('online' if r['arm'] in ('M3OE','HYPERGATE','AUTOIFS') else 'fixed_ema')]
    lookup = {(r['arm'], int(r['seed'])):r for r in selected}
    comparisons = [(c, ref) for c in CANDIDATES for ref in REFERENCES if c != ref]
    comparisons += [('G-D02', 'G-D0')]
    pairs, summaries, decisions = [], [], []
    for candidate, reference in comparisons:
        current = []
        comparison = f'{candidate}_vs_{reference}'
        for seed in config.SEEDS:
            a, b = lookup.get((candidate, seed)), lookup.get((reference, seed))
            if a is None or b is None:
                continue
            if (a['protocol_id'], a['data_protocol']) != (b['protocol_id'], b['data_protocol']):
                raise ValueError('Paired data/evaluation protocols differ')
            out = dict(comparison_id=comparison, candidate_arm=candidate, reference_arm=reference, seed=seed,
                candidate_hash=a['model_hash'], reference_hash=b['model_hash'],
                candidate_epoch=a['epoch'], reference_epoch=b['epoch'])
            for metric in metrics:
                if a.get(metric) is not None and b.get(metric) is not None:
                    out.update({f'candidate_{metric}':a[metric], f'reference_{metric}':b[metric],
                                f'delta_{metric}':a[metric]-b[metric]})
            pairs.append(out)
            current.append(out)
        for metric in metrics:
            valid = [r for r in current if f'delta_{metric}' in r]
            deltas = [r[f'delta_{metric}'] for r in valid]
            summaries.append(dict(comparison_id=comparison, candidate_arm=candidate, reference_arm=reference,
                metric=metric, candidate_mean=statistics.fmean(r[f'candidate_{metric}'] for r in valid) if valid else None,
                reference_mean=statistics.fmean(r[f'reference_{metric}'] for r in valid) if valid else None,
                positive_seeds=sum(v>0 for v in deltas), negative_seeds=sum(v<0 for v in deltas), **stats(deltas)))
        if {r['seed'] for r in current} == set(config.SEEDS):
            delta = statistics.fmean(r['delta_macro_auc'] for r in current)
            ll = statistics.fmean(r['delta_macro_logloss'] for r in current)
            cells = {c:statistics.fmean(r[f'delta_{c}_auc'] for r in current) for c in CELLS}
            decisions.append(dict(comparison_id=comparison, n=5, macro_auc_delta=delta,
                macro_logloss_delta=ll, macro_auc_gain_ge_003=delta>=.003,
                cell_protection_pass=all(v>=-.001 for v in cells.values()),
                cell_auc_deltas=cells, logloss_improved=ll<0))
        else:
            decisions.append(dict(comparison_id=comparison, status='INCOMPLETE', n=len(current)))
    write_csv(destination/'PAIRED_SEED.csv', pairs)
    write_csv(destination/'PAIRED_SUMMARY.csv', summaries)
    baseline_means = {arm:statistics.fmean(lookup[arm,s]['macro_auc'] for s in config.SEEDS)
                      for arm in ('M3OE','HYPERGATE','AUTOIFS')}
    best = max(baseline_means, key=baseline_means.get)
    classifications = []
    for arm in CANDIDATES:
        if not all((arm,s) in lookup for s in config.SEEDS):
            classifications.append(dict(arm=arm, status='INCOMPLETE'))
            continue
        auc = statistics.fmean(lookup[arm,s]['macro_auc'] for s in config.SEEDS)
        delta = auc-baseline_means[best]
        tier = 'STRONG_ALGORITHM' if delta>=.003 else 'CANDIDATE_ALGORITHM' if delta>=.002 else 'POSITIVE_WIN' if delta>0 else 'NOT_ABOVE_BEST_PUBLISHED'
        guard = next(r for r in decisions if r['comparison_id']==f'{arm}_vs_{best}')
        classifications.append(dict(arm=arm, status=tier, macro_auc=auc, reference=best,
            reference_macro_auc=baseline_means[best], delta_macro_auc=delta,
            cell_protection_pass=guard['cell_protection_pass'], logloss_improved=guard['logloss_improved']))
    write_json(destination/'DECISIONS.json', dict(best_published=best, published_macro_auc=baseline_means,
        classification=classifications, comparisons=decisions, rules=prepared['protocol']['decision']))
    curves, costs, committed_paths = [], [], []
    for job in prepared['jobs']:
        parent = attempt/'runs'/job['job_id']
        committed = {r['commit_index']:r['epoch_record_sha'] for r in committed_chain(attempt,job['job_id'])}
        files = {sha_file(p):p for p in parent.glob('launch_*/epoch_*.json')}
        for index in sorted(committed):
            path = files[committed[index]]
            r = read_json(path)
            committed_paths.append(str(path))
            for estimator, values in r['validation'].items():
                curves.append(dict(arm=job['arm'], seed=job['seed'], estimator=estimator,
                    epoch=r['epoch'], model_hash=r['model_hash'][estimator],
                    train_loss=r['train_objective']['loss'], train_real_loss=r['train_objective']['real_loss'], **values))
        if (parent/'cost.json').exists():
            costs.append(dict(arm=job['arm'], seed=job['seed'], cost_origin='new_training',
                              trajectory_id=job['trajectory_id'], **read_json(parent/'cost.json')))
        for path in sorted(parent.glob('launch_*/physical_cost.json')):
            costs.append(dict(arm=job['arm'], seed=job['seed'], cost_origin='physical_launch_with_RAM_IPC', **read_json(path)))
        for path in sorted(parent.glob('launch_*/failure.json')):
            costs.append(dict(arm=job['arm'], seed=job['seed'], cost_origin='failed_physical_attempt', **read_json(path)))
        for replay in sorted((attempt/'replay_history'/job['job_id']).glob('replay_*')):
            for path in sorted(replay.glob('launch_*/epoch_*.json')):
                row = read_json(path)
                costs.append(dict(arm=job['arm'],seed=job['seed'],cost_origin='state_lost_discarded_epoch',
                    epoch=row['epoch'],source_path=str(path),**row['cost']))
            for pattern, origin in (('launch_*/physical_cost.json','state_lost_physical_launch_with_RAM_IPC'),
                                    ('launch_*/failure.json','state_lost_failed_physical_attempt')):
                for path in sorted(replay.glob(pattern)):
                    costs.append(dict(arm=job['arm'],seed=job['seed'],cost_origin=origin,
                        source_path=str(path),**read_json(path)))
    for arm in REFERENCES:
        for seed in config.SEEDS:
            row = lookup.get((arm,seed))
            costs.append(dict(arm=arm, seed=seed, cost_origin='historical_reuse', new_training_seconds=0,
                              source_path=row['source_path'] if row else None))
    write_csv(destination/'VALIDATION_CURVES.csv', curves)
    write_csv(destination/'COSTS.csv', costs)
    statuses = read_json(attempt/'queue_state.json')['jobs']
    complete = all(r['status']=='DONE' for r in statuses.values())
    if complete:
        if len(committed_paths)!=500 or len(curves)!=1000:
            raise ValueError('Final archive lacks 500 committed epochs')
        for arm in CANDIDATES+('FIXED_EMA','M3OE','HYPERGATE','AUTOIFS'):
            if not all((arm,s) in lookup for s in config.SEEDS):
                raise ValueError('Final archive lacks a selected five-seed comparison')
    progress = dict(status='FINAL_COMPLETE' if complete else 'RUNNING',
        completed=sum(r['status']=='DONE' for r in statuses.values()), total=10,
        committed_epochs=len(committed_paths), required_epochs=500)
    write_json(attempt/'progress.json', progress)
    write_json(destination/'RUN_MANIFEST.json', dict(prepared=prepared, progress=progress,
        reuse_audit=read_json(attempt/'REUSE_AUDIT.json'),
        source=read_json(attempt/'SOURCE_LOCK.json'), committed_scalar_paths=committed_paths,
        endpoint_sha256=sha_file(destination/'ENDPOINTS.csv'),
        runtime_overrides=read_json(attempt/'RUNTIME_OVERRIDES.json')))
    (destination/'PROGRESS.md').write_text(f"状态：{progress['status']}。完成{progress['completed']}/10条训练、{len(committed_paths)}/500epoch。\n")
    return progress
