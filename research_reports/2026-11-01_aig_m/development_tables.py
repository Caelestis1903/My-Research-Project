"""Flatten completed development stages into checkpoint and seed-summary tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = Path(__file__).resolve().parent
    rows = []
    for stage in ['S1', 'S2', 'S3', 'S4']:
        if not (run / f'{stage}_decision.json').exists():
            continue
        for path in sorted((run / 'runs' / stage).rglob('state_summary.json')):
            for item in json.loads(path.read_text())['checkpoints']:
                row = dict(item)
                validation = row.pop('validation')
                row.update(validation)
                rows.append(row)
    assert rows
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ['dataset', 'stage', 'trial', 'model_kind', 'checkpoint'])
        groups[key].append(row)
    summary = []
    for key, values in groups.items():
        assert sorted(int(v['seed']) for v in values) == [2022, 2024], key
        for metric in [k for k in values[0] if k.endswith(('_auc', '_logloss'))]:
            x = [v[metric] for v in values]
            summary.append(dict(zip(['dataset', 'stage', 'trial', 'model_kind', 'checkpoint'], key),
                metric=metric, n=2, mean=mean(x), sample_sd=stdev(x)))
    for name, records in [('development_checkpoints.csv', rows), ('development_summary.csv', summary)]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(dict(checkpoint_rows=len(rows), summary_rows=len(summary),
        completed_stages=sorted({r['stage'] for r in rows}))))


if __name__ == '__main__':
    main()
