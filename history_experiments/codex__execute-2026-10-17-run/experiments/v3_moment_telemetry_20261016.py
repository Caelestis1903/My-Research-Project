"""Recover pure first-moment norms from immutable v3 telemetry archives."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records, inputs = [], []
    for path in sorted(args.archive_root.glob('*/budget*/optimizer_observations.csv')):
        receipt = json.loads((path.parent / 'receipt.json').read_text())
        config = json.loads((path.parent / 'config.json').read_text())
        assert sha(path) == receipt['artifact_hashes'][path.name], path
        assert receipt['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        rows = list(csv.DictReader(path.open()))
        maximum = 0.0
        for row in rows:
            mixed, second = float(row['exp_avg']), float(row['exp_avg_sq'])
            square = mixed * mixed - second * second
            assert square >= -1e-14 * max(mixed * mixed, second * second, 1e-300), path
            corrected = math.sqrt(max(square, 0.0))
            maximum = max(maximum, (mixed - corrected) / mixed if mixed else 0.0)
            records.append(dict(run=path.parent.parent.name, budget=receipt['epochs'],
                                dataset=config['dataset'], optimizer=config['optimizer'],
                                seed=config['seed'], arm=config['arm'], **row,
                                raw_exp_avg_mixed_norm=mixed,
                                corrected_m_norm=corrected, v_norm=second))
        inputs.append(dict(run=path.parent.parent.name, budget=receipt['epochs'], rows=len(rows),
                           source=str(path), sha256=sha(path),
                           maximum_relative_norm_correction=maximum))
    assert records, 'No completed archived telemetry'
    with (args.out / 'corrected_moment_observations.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.out / 'receipt.json').write_text(json.dumps(dict(
        status='PASS', source_sha=sha(Path(__file__)), inputs=inputs, rows=len(records),
        correction='m_norm=sqrt(raw_exp_avg**2-raw_exp_avg_sq**2); raw preserved',
        scope='Read-only norm correction; no optimizer or performance changes',
        output_sha=sha(args.out / 'corrected_moment_observations.csv')), indent=2) + '\n')


if __name__ == '__main__':
    main()
