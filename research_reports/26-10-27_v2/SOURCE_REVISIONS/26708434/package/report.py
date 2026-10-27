"""Machine-readable seed summaries and the explicit final-evaluation manifest."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import t as student_t
from .config import METHODS, SEEDS, DATASETS
from .state import write_json, digest, source_identity
from .trainer import flatten_metrics


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, sample_sd=sd,
                seed_t95=[mean-half, mean+half] if half is not None else None)


def read_json(path):
    return json.loads(Path(path).read_text())


def collect(artifact_root):
    root = Path(artifact_root)
    runs, missing, records = {}, [], []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                directory = root/'runs'/dataset/method/f'seed{seed}'
                path = directory/'run_receipt.json'
                if not path.exists():
                    missing.append(str(path))
                    continue
                receipt = read_json(path)
                key = (dataset, method, seed)
                runs[key] = dict(receipt=receipt, directory=directory)
                for role in ('V', 'Q', 'E'):
                    for state in ('selected', 'final'):
                        metric_path = directory/f'{role}_{state}_metrics.json'
                        if metric_path.exists():
                            metric = read_json(metric_path)
                            records.append(dict(dataset=dataset, method=method, seed=seed,
                                                role=role, state=state, epoch=metric['identity']['epoch'],
                                                model_hash=metric['identity']['model_hash'],
                                                **flatten_metrics(metric['metrics'])))
    summary, paired, gates = [], [], {}
    for dataset in DATASETS:
        for method in METHODS:
            for role in ('V', 'Q', 'E'):
                for state in ('selected', 'final'):
                    rows = [x for x in records if (x['dataset'], x['method'], x['role'], x['state']) == (dataset, method, role, state)]
                    if len(rows) == 4:
                        fields = [k for k in rows[0] if k.endswith(('_auc', '_logloss')) or k in ('macro_auc', 'macro_logloss', 'natural_logloss')]
                        summary.append(dict(dataset=dataset, method=method, role=role, state=state,
                                            metrics={f:interval([x[f] for x in rows]) for f in fields if all(x[f] is not None for x in rows)}))
        simple = METHODS[:4]
        if all((dataset, m, s) in runs for m in METHODS for s in SEEDS):
            means = {m:np.mean([runs[dataset, m, s]['receipt']['scores']['Q_selected']['macro_logloss'] for s in SEEDS]) for m in simple}
            tied = [m for m in simple if means[m] <= min(means.values())+1e-6]
            cstar = min(tied, key=lambda m:(sum(runs[dataset, m, s]['receipt']['training_seconds'] for s in SEEDS), m))
            for method in METHODS:
                for role in ('Q', 'E'):
                    for state in ('selected', 'final'):
                        rows = [x for x in records if x['dataset'] == dataset and x['method'] == method and x['role'] == role and x['state'] == state]
                        controls = [x for x in records if x['dataset'] == dataset and x['method'] == cstar and x['role'] == role and x['state'] == state]
                        if len(rows) == len(controls) == 4:
                            control_by_seed = {x['seed']:x for x in controls}
                            fields = [k for k in rows[0] if k.endswith(('_auc', '_logloss')) or k in ('macro_auc', 'macro_logloss', 'natural_logloss')]
                            deltas = {f:[x[f]-control_by_seed[x['seed']][f] for x in rows] for f in fields if all(x[f] is not None and control_by_seed[x['seed']][f] is not None for x in rows)}
                            paired.append(dict(dataset=dataset, method=method, control=cstar, role=role, state=state,
                                               seed_order=[x['seed'] for x in rows], raw_deltas=deltas,
                                               statistics={f:interval(v) for f, v in deltas.items()}))
            main = next(x for x in paired if x['dataset'] == dataset and x['method'] == 'LOWRANK_RELATION' and x['role'] == 'Q' and x['state'] == 'selected')
            delta = main['raw_deltas']
            ll, auc, natural = (float(np.mean(delta[f])) for f in ('macro_logloss', 'macro_auc', 'natural_logloss'))
            cell_fields = [f'd{d}_t{t}_logloss' for t in range(2) for d in range(3)]
            guard = natural <= .001 and auc >= -.0005 and all(np.mean(delta[f]) <= .003 for f in cell_fields)
            practical = guard and (ll <= -.0005 or (auc >= .0005 and ll <= .0002))
            local = guard and any(all(v <= -.0005 for v in delta[f]) for f in cell_fields)
            formation = [s for s in SEEDS if (root/'runs'/dataset/'LOWRANK_RELATION'/f'seed{s}'/'diagnostics.json').exists()
                         and read_json(root/'runs'/dataset/'LOWRANK_RELATION'/f'seed{s}'/'diagnostics.json').get('formation_evidence')]
            gates[dataset] = dict(Cstar=cstar, Cstar_Q_means=means, delta_macro_LL=ll, delta_macro_AUC=auc,
                                  delta_natural_LL=natural, GUARD=bool(guard), PRACTICAL=bool(practical),
                                  LOCAL=bool(local), formation_evidence_seeds=formation,
                                  stage2_eligible=bool(practical and formation),
                                  finite_diagnostic_eligible=bool(not practical and local))
    return dict(status='DEVELOPMENT_COMPLETE' if not missing else 'PARTIAL',
                completed_adaptation_runs=len(runs), expected_adaptation_runs=48,
                missing=missing, seed_rows=records, summary=summary, paired=paired,
                development_gates=gates, training_repeat_unit='seed')


def write_report(artifact_root, report_root):
    report = collect(artifact_root)
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'RESULTS_SUMMARY.json', report)
    if report['seed_rows']:
        with (root/'SEED_METRICS.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report['seed_rows'][0]))
            writer.writeheader()
            writer.writerows(report['seed_rows'])
    return report


def freeze_manifest(artifact_root, cfg, decision_path):
    root = Path(artifact_root)
    decision = read_json(decision_path)
    if decision.get('status') != 'FINAL_METHODS_FROZEN':
        raise ValueError('Orchestrator decision must close method selection before E')
    if set(decision.get('stage2', {})) != set(DATASETS) or any(v not in ('NOT_TRIGGERED', 'COMPLETED') for v in decision['stage2'].values()):
        raise ValueError('Each dataset needs an actual closed stage2 status')
    workers = {}
    current_source_hash = source_identity()['source_hash']
    for dataset in DATASETS:
        for seed in SEEDS:
            ready = read_json(root/'workers'/dataset/f'seed{seed}'/'READY.json')
            if ready['status'] != 'READY_FOR_GLOBAL_FREEZE':
                raise ValueError('Worker is not ready')
            identity = ready['identity']
            if (identity['config_hash'] != cfg['config_hash'] or set(identity['selected']) != set(METHODS)
                    or identity['source_hash'] != current_source_hash):
                raise ValueError('Worker identity/configuration mismatch')
            workers[f'{dataset}/seed{seed}'] = identity
    value = dict(status='FROZEN', protocol=cfg['protocol'], config_hash=cfg['config_hash'],
                 source_hash=current_source_hash,
                 methods=list(METHODS), workers=workers, decision=decision,
                 thresholds=cfg['comparison'], historical_test=True)
    write_json(root/'GLOBAL_FREEZE.json', value, exclusive=True)
    return dict(path=str(root/'GLOBAL_FREEZE.json'), manifest_hash=digest(value))


def release_evaluation(artifact_root, dataset, seed):
    root = Path(artifact_root)
    manifest = read_json(root/'GLOBAL_FREEZE.json')
    identity = manifest['workers'][f'{dataset}/seed{seed}']
    worker_root = root/'workers'/dataset/f'seed{seed}'
    write_json(worker_root/'EVALUATE.json', dict(identity=identity), exclusive=True)
    return dict(released=f'{dataset}/seed{seed}')

