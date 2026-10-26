"""Four-seed, frozen SCALE reassessment; no training or model loading."""
from pathlib import Path
import csv
import argparse
import hashlib
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.mdmt_rcf_20261026.scale_probe import _hist_protocol

REPORT = Path(__file__).resolve().parent
ACTIVE = REPO / 'artifacts/mdmt_rcf_20261026/v2_four_seed'
SCALE = REPO / 'artifacts/mdmt_rcf_20261026/v1/scale'
HISTORY = REPO / 'experiments/reproduction_20261021/raw_results'
SEEDS = [2022, 2023, 2024, 2025]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(name, rows):
    path = REPORT / 'tables' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def old_run(dataset, arm, seed):
    stage = 'formal' if arm == 'B' else 'stage2'
    paths = [p for p in (HISTORY/stage/dataset/arm/f'seed{seed}').glob('attempt*/completion.json')
             if read(p).get('valid') is True and read(p).get('formal') is True]
    if len(paths) != 1:
        raise RuntimeError(f'Historical identity not unique: {dataset}/{arm}/{seed}: {paths}')
    return paths[0], read(paths[0])


def main(validation_only=False):
    # This program deliberately refuses to score exposed data before the main freeze.
    if not validation_only:
        acceptance = read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json')
        if acceptance.get('status') != 'PASS' or not acceptance.get('analysis_authorized'):
            raise PermissionError('Four-seed completion acceptance must pass first')
        freeze = read(ACTIVE/'FROZEN_MANIFEST.json')
        if freeze['four_seed_acceptance_sha256'] != sha(ACTIVE/'FOUR_SEED_ACCEPTANCE.json'):
            raise PermissionError('Four-seed inventory differs from global freeze')
    frozen = read(SCALE/'ALL_EIGHT_FROZEN_MANIFEST.json')
    if {(r['dataset'], r['seed']) for r in frozen['runs']} != {(d,s) for d in ['movielens','kuairand'] for s in SEEDS}:
        raise RuntimeError('SCALE needs four independent seeds in both datasets')
    protocol = _hist_protocol()
    metrics, identities, recomputation = {}, [], []
    per_seed, long_rows, cost_rows = [], [], []
    for row in frozen['runs']:
        dataset, seed = row['dataset'], row['seed']
        path = REPO/row['receipt']
        if sha(path) != row['sha256']:
            raise RuntimeError('Frozen SCALE receipt changed')
        current = read(path)
        if current['status'] != 'COMPLETE':
            raise RuntimeError('Incomplete SCALE run')
        config, data = read(path.parent/'config.json'), read(path.parent/'data_manifest.json')
        values = {}
        roles = [('validation','validation_predictions.npz')]
        if not validation_only:
            roles.append(('test_exposed','test_predictions.npz'))
        for role, filename in roles:
            with np.load(path.parent/filename, allow_pickle=False) as raw:
                values[role, 'selected'] = protocol.metrics(raw['y'], raw['p'], raw['domain'])
                if 'p_terminal' in raw.files:
                    values[role, 'stopped_endpoint'] = protocol.metrics(raw['y'], raw['p_terminal'], raw['domain'])
        error = max(abs(values['validation','selected'][k]-current['selected_validation'][k])
                    for k in current['selected_validation'] if isinstance(current['selected_validation'][k], (float,int)))
        if error > 1e-12:
            raise RuntimeError('Stored selected-validation metrics do not recompute')
        recomputation.append(dict(dataset=dataset,seed=seed,max_validation_metric_error=error,
                                  receipt_sha256=sha(path),prediction_sha256=sha(path.parent/'test_predictions.npz')))
        for arm in ['B','Dconst01']:
            old_path, old = old_run(dataset, arm, seed)
            old_config, old_data = read(old_path.parent/'config.json'), read(old_path.parent/'data_manifest.json')
            keys = ['raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes',
                    'base_split_fingerprints','base_split_row_id_hashes','features','train_domain_counts',
                    'physical_domain_identity']
            equal = {k:data[k] == old_data[k] for k in keys}
            equal['canonical_backbone'] = config['initialization']['canonical_backbone_sha256'] == old_config['initialization']['canonical_backbone_sha256']
            if arm == 'Dconst01':
                equal['complete_initial'] = config['initialization']['complete_initial_sha256'] == old_config['initialization']['complete_initial_sha256']
            for new_key, old_key in [('batch_size','batch_size'),('lr','lr'),('weight_decay','weight_decay'),('eps','adam_eps'),('betas','adam_betas'),('tf32','tf32'),('amp','amp')]:
                equal[new_key] = current['identity'][new_key] == old_config[old_key]
            if not all(equal.values()):
                raise RuntimeError(f'Historical SCALE mismatch {dataset}/{arm}/{seed}: {equal}')
            identities.append(dict(dataset=dataset,seed=seed,reference=arm,checks=equal,
                                   historical_receipt=str(old_path.relative_to(REPO)),historical_sha256=sha(old_path)))
            history = list(csv.DictReader((old_path.parent/'epoch_metrics.csv').open()))
            selected = next(r for r in history if int(r['epoch']) == old['selected_epoch'])
            validation = {k[4:]:float(v) for k,v in selected.items() if k.startswith('val_') and v}
            metrics[dataset,seed,arm,'validation','selected'] = validation
            if not validation_only:
                metrics[dataset,seed,arm,'test_exposed','selected'] = old['test_metrics']
            cost_rows.append(dict(dataset=dataset,seed=seed,arm=arm,source='HISTORICAL_PUBLIC_SUMMARY',
                                  selected_epoch=old['selected_epoch'],stopped_epoch=old['stopped_epoch'],
                                  steps=old['steps'],wall_seconds=old['wall_seconds']))
        for (role, endpoint), stat in values.items():
            metrics[dataset,seed,'Dconst01_NO_AUG',role,endpoint] = stat
        cost_rows.append(dict(dataset=dataset,seed=seed,arm='Dconst01_NO_AUG',source='THIS_EXECUTION',
                              selected_epoch=current['selected_epoch'],stopped_epoch=current['stopped_epoch'],
                              steps=current['steps'],wall_seconds=current['wall_seconds']))
    for (dataset,seed,arm,role,endpoint), stat in metrics.items():
        common = dict(dataset=dataset,seed=seed,arm=arm,role=role,endpoint=endpoint)
        per_seed.append(dict(common,**stat))
        long_rows.extend(dict(common,metric=k,value=v) for k,v in stat.items())
    summaries, pairs = [], []
    for dataset in ['movielens','kuairand']:
        for role in (['validation'] if validation_only else ['validation','test_exposed']):
            for candidate, reference in [('Dconst01_NO_AUG','B'),('Dconst01','Dconst01_NO_AUG'),('Dconst01','B')]:
                common_keys = set.intersection(*(set(metrics[dataset,s,candidate,role,'selected']) &
                                                 set(metrics[dataset,s,reference,role,'selected']) for s in SEEDS))
                for name in sorted(common_keys):
                    if name.endswith('_count') or name.endswith('_positive'):
                        continue
                    delta = np.array([metrics[dataset,s,candidate,role,'selected'][name]-metrics[dataset,s,reference,role,'selected'][name] for s in SEEDS],float)
                    sd = float(delta.std(ddof=1)); mean = float(delta.mean()); half = 3.182446305284263*sd/2
                    common = dict(dataset=dataset,role=role,candidate=candidate,reference=reference,metric=name)
                    summaries.append(dict(common,n_training_seeds=4,mean_delta=mean,sample_sd=sd,
                                          t95_low=mean-half,t95_high=mean+half,negative_seeds=int((delta<0).sum()),
                                          pairing='TRAINING_SEED_METRICS_ONLY_HISTORICAL_RAW_P_UNAVAILABLE'))
                    pairs.extend(dict(common,seed=s,delta=float(v)) for s,v in zip(SEEDS,delta))
    prefix = 'SCALE_validation_four_seed' if validation_only else 'SCALE_four_seed'
    write_csv(prefix+'_metrics.csv', per_seed)
    write_csv(prefix+'_metrics_long.csv', long_rows)
    write_csv(prefix+'_paired_differences.csv', pairs)
    write_csv(prefix+'_paired_summary.csv', summaries)
    write_csv(prefix+'_costs.csv', cost_rows)
    result = dict(status='PASS',four_training_seeds_per_condition=True,source='c5618e281785a4b7063ff9bf2a61cf38d4faf1be',
                  exposed_evaluation=not validation_only,historical_raw_pairing=False,identities=identities,recomputation=recomputation,
                  global_freeze_sha256=None if validation_only else sha(ACTIVE/'FROZEN_MANIFEST.json'),
                  frozen_scale_manifest_sha256=sha(SCALE/'ALL_EIGHT_FROZEN_MANIFEST.json'))
    receipt = 'SCALE_VALIDATION_FOUR_SEED_ANALYSIS_RECEIPT.json' if validation_only else 'SCALE_FOUR_SEED_ANALYSIS_RECEIPT.json'
    (REPORT/receipt).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',conditions=6,seeds_per_condition=4,metric_rows=len(per_seed),paired_summaries=len(summaries))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-only',action='store_true')
    main(parser.parse_args().validation_only)
