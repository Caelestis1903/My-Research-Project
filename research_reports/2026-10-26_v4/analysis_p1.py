"""Scientific analysis of fixed completed P1 predictions; no model fitting."""
from pathlib import Path
import argparse, csv, hashlib, json, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.mdmt_dfd_20261026.data import RoleBundle
from experiments.mdmt_dfd_20261026.metrics import evaluate_probabilities, paired_summary
from experiments.mdmt_dfd_20261026.plans import rng

ROOT = Path('/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def bootstrap(y, predictions, domain, users, seed, dataset):
    """One common user resample across every policy, both tasks, and all domains."""
    clusters = np.arange(len(y)) if users is None else users
    unique, inverse = np.unique(clusters, return_inverse=True)
    generator = rng(seed, dataset, 70, component=1)
    references = ['CONST', 'LOGIT', 'DIRECT']
    names = ['macro_LL', 'macro_AUC'] + [f'cell{c}_LL' for c in range(6)] + [f'cell{c}_AUC' for c in range(6)]
    draws = {ref: [] for ref in references}
    invalid = {ref: 0 for ref in references}
    for _ in range(1000):
        counts = np.bincount(generator.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        metrics = {k: evaluate_probabilities(y, p, domain, weights) for k, p in predictions.items()}
        a = metrics['DEMAND']
        for ref in references:
            b = metrics[ref]
            def delta(x, z): return np.nan if x is None or z is None else x-z
            values = [delta(a['macro_logloss'], b['macro_logloss']), delta(a['macro_auc'], b['macro_auc'])]
            values += [delta(x['logloss'], z['logloss']) for x, z in zip(a['cells'], b['cells'])]
            values += [delta(x['auc'], z['auc']) for x, z in zip(a['cells'], b['cells'])]
            invalid[ref] += int(not np.isfinite(values).all())
            draws[ref].append(values)
    return {'unit': 'original_user' if users is not None else 'row_fallback_independence_unverified',
            'role': 'P_form_OOF', 'repetitions': 1000, 'raw_unique_clusters': len(unique),
            'direction': 'DEMAND minus reference: LL negative favorable, AUC positive favorable',
            'common_draws_across_all_methods_tasks_domains': True,
            'rng_coordinates': [seed, 0 if dataset == 'movielens' else 1, 70, 0, 0, 1, 0],
            'metrics': names, 'comparisons': {
                ref: {'invalid_resamples': invalid[ref],
                      'percentile95': dict(zip(names, np.nanquantile(draws[ref], [.025, .975], axis=0).T.tolist())),
                      'replicate_deltas': draws[ref]} for ref in references}}

def analyze(dataset, with_bootstrap):
    out = ROOT/'analysis'/'p1'/dataset
    data = RoleBundle(dataset)
    all_metrics = {}; cells = []; macro = []; audits = []
    for seed in [5101, 5102]:
        run = ROOT/'prefix'/dataset/f'seed{seed}'
        if not (run/'P1_OOF_COMPLETE.json').exists():
            raise RuntimeError(f'P1 incomplete: {dataset}/{seed}')
        file = run/'p1_oof.npz'
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        done = json.loads((run/'P1_OOF_COMPLETE.json').read_text())
        if digest != done['npz_sha256']: raise RuntimeError('OOF checksum mismatch')
        a = np.load(file)
        y, domain, rows = a['y'], a['domain'], a['row_id']
        predictions = {k: a['p_'+k] for k in ['CONST', 'LOGIT', 'DEMAND', 'DIRECT']}
        stored = json.loads((run/'P1_METRICS.json').read_text())['metrics']
        measured = {k: evaluate_probabilities(y, p, domain) for k, p in predictions.items()}
        maximum = max(abs(measured[k][field] - stored[k][field]) for k in measured for field in ['macro_logloss', 'macro_auc'])
        if maximum > 1e-12: raise RuntimeError('Stored metric differs from prediction recomputation')
        all_metrics[seed] = measured
        for method, metric in measured.items():
            macro.append(dict(dataset=dataset, seed=seed, method=method, role='P_form_OOF',
                              macro_LL=metric['macro_logloss'], macro_AUC=metric['macro_auc'],
                              natural_LL=metric['natural_logloss'], pooled_AUC0=metric['pooled_auc'][0], pooled_AUC1=metric['pooled_auc'][1]))
            for cell in metric['cells']: cells.append(dict(dataset=dataset, seed=seed, method=method, role='P_form_OOF', **cell))
        audits.append(dict(seed=seed, npz_sha256=digest, checkpoint_hash=done['checkpoint_hash'],
                           metric_recomputation_max_abs=maximum, n_rows=len(rows)))
        boot = out/f'seed{seed}_bootstrap1000.json'
        if with_bootstrap and not boot.exists():
            started = time.perf_counter()
            users = None if data.user_ids is None else data.user_ids[rows]
            result = bootstrap(y, predictions, domain, users, seed, dataset)
            result['prediction_npz_sha256'] = digest
            result['analysis_wall_seconds'] = time.perf_counter()-started
            write(boot, result)
            print(dataset, seed, 'bootstrap complete', result['analysis_wall_seconds'], flush=True)
    for name, records in [('macro.csv', macro), ('six_cells.csv', cells)]:
        out.mkdir(parents=True, exist_ok=True)
        with (out/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    comparisons = {}
    for ref in ['CONST', 'LOGIT', 'DIRECT']:
        comparisons[ref] = {field: paired_summary([all_metrics[s]['DEMAND'][field] - all_metrics[s][ref][field] for s in [5101,5102]])
                            for field in ['macro_logloss', 'macro_auc']}
    write(out/'paired_seed_statistics.json', {'dataset': dataset, 'n_seeds': 2,
          'role': 'P_form_OOF', 'direction': 'DEMAND minus reference', 'comparisons': comparisons,
          'seed_intervals_and_bootstrap_are_distinct_uncertainties': True, 'audits': audits})
    print(dataset, 'P1 analysis complete', json.dumps(comparisons), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset', choices=['movielens','kuairand'], required=True)
    parser.add_argument('--bootstrap', action='store_true'); args = parser.parse_args()
    analyze(args.dataset, args.bootstrap)
