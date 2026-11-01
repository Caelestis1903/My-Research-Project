"""Recompute GAM01 paired evidence without altering historical outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HISTORY = ROOT / 'research_reports/2026-11-01_momentum'
PAIRS = [('EMA_A0', 'A0', 'shadow', 'A0', 'online'),
         ('A1_minus_A0', 'A1', 'online', 'A0', 'online'),
         ('A2_minus_A1', 'A2', 'online', 'A1', 'online'),
         ('A3_minus_A1', 'A3', 'online', 'A1', 'online'),
         ('A3_shadow_minus_A1_shadow', 'A3', 'shadow', 'A1', 'shadow')]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    pairs, summaries = [], []
    for split in ['validation', 'test']:
        rows = list(csv.DictReader((HISTORY / f'{split}_metrics.csv').open()))
        assert len(rows) == 240
        index = {(r['dataset'], r['arm'], r['model_kind'], r['checkpoint'], int(r['seed'])): r for r in rows}
        assert len(index) == len(rows)
        metrics = [k for k in rows[0] if k.endswith(('_auc', '_logloss'))]
        for dataset in ['movielens', 'kuairand']:
            for checkpoint in ['selected', 'E10', 'E50']:
                for effect, arm, kind, reference, refkind in PAIRS:
                    for metric in metrics:
                        deltas = []
                        for seed in range(2022, 2027):
                            candidate = index[dataset, arm, kind, checkpoint, seed]
                            baseline = index[dataset, reference, refkind, checkpoint, seed]
                            delta = float(candidate[metric]) - float(baseline[metric])
                            deltas.append(delta)
                            pairs.append(dict(split=split, dataset=dataset, checkpoint=checkpoint,
                                effect=effect, metric=metric, seed=seed, candidate=float(candidate[metric]),
                                reference=float(baseline[metric]), delta=delta,
                                candidate_epoch=int(candidate['epoch']), reference_epoch=int(baseline['epoch'])))
                        average, sd = mean(deltas), stdev(deltas)
                        radius = 2.7764451051977987 * sd / math.sqrt(5)
                        summaries.append(dict(split=split, dataset=dataset, checkpoint=checkpoint,
                            effect=effect, metric=metric, n=5, mean=average, sample_sd=sd,
                            ci95_low=average-radius, ci95_high=average+radius,
                            positive=sum(d > 0 for d in deltas), negative=sum(d < 0 for d in deltas)))
    write_csv(OUT / 'historical_paired_effects.csv', pairs)
    write_csv(OUT / 'historical_effect_summary.csv', summaries)
    manifest = json.loads((HISTORY / 'data_manifest.json').read_text())
    data = []
    for dataset, record in manifest.items():
        path = Path(record['raw_path'])
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == record['raw_sha256'], dataset
        data.append(dict(dataset=dataset, path=str(path), bytes=path.stat().st_size,
            sha256=digest, matches_GAM01=True, train_domain_counts=record['train_domain_counts'],
            split_row_id_hashes=record['split_row_id_hashes'], array_hashes=record['array_hashes']))
    (OUT / 'RAW_DATA_VERIFICATION.json').write_text(json.dumps(data, indent=2)+'\n')
    selected = [r for r in summaries if r['checkpoint']=='selected' and r['metric'] in ['macro_auc','macro_logloss']]
    (OUT / 'historical_recalculation.json').write_text(json.dumps(selected, indent=2)+'\n')
    for row in selected:
        if row['split']=='test':
            print(row['dataset'], row['effect'], row['metric'], f"{row['mean']:+.9f}",
                  'positive', row['positive'], 'negative', row['negative'])


if __name__ == '__main__':
    main()
