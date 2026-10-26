"""Reduce verified production batch counts/weights; no predictive readout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for manifest in args.manifest:
        for task in json.loads(manifest.read_text())['tasks']:
            command = task['command']
            if '--run-dir' not in command or task['name'].startswith('V4pilot'):
                continue
            directory = Path(command[command.index('--run-dir') + 1])
            receipt = json.loads((directory / 'receipt.json').read_text())
            assert receipt['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
            config = json.loads((directory / 'config.json').read_text())
            source = directory / 'batch_trace.csv'
            assert sha(source) == receipt['artifact_hashes'][source.name]
            assert sha(directory / 'config.json') == receipt['artifact_hashes']['config.json']
            weights = [p / q for p, q in zip(config['p'], config['q'])]
            second_moment = sum(q * w * w for q, w in zip(config['q'], weights))
            fractions, variances, totals = [], [], [0, 0, 0]
            errors = [0., 0., 0.]
            examples = 0
            with source.open() as stream:
                for row in csv.DictReader(stream):
                    n = int(row['n'])
                    counts = [int(row[f'd{d}']) for d in range(3)]
                    assert sum(counts) == n
                    first = sum(c * w for c, w in zip(counts, weights))
                    second = sum(c * w * w for c, w in zip(counts, weights))
                    ess = first * first / second
                    variance = sum(c * (w - first / n)**2 for c, w in zip(counts, weights)) / n
                    maximum = max(w for c, w in zip(counts, weights) if c)
                    for j, (observed, expected) in enumerate(zip(
                            [float(row['ess']), float(row['weight_variance']), float(row['max_weight'])],
                            [ess, variance, maximum])):
                        errors[j] = max(errors[j], abs(observed - expected))
                        assert abs(observed - expected) <= 1e-9 * max(1., abs(expected))
                    fractions.append(ess / n)
                    variances.append(variance)
                    examples += n
                    totals = [a + b for a, b in zip(totals, counts)]
            assert examples == receipt['examples'] and len(fractions) == receipt['steps']
            records.append(dict(run=directory.name, dataset=config['dataset'], seed=config['seed'],
                optimizer=config['optimizer'], arm=config['arm'], lr=config['lr'],
                batch_size=config['batch_size'], epochs=receipt['epochs'], steps=receipt['steps'],
                examples=examples, p=json.dumps(config['p']), q=json.dumps(config['q']),
                asymptotic_ESS_fraction=1 / second_moment,
                mean_observed_ESS_fraction=statistics.mean(fractions),
                theoretical_weight_variance=second_moment - 1,
                mean_observed_weight_variance=statistics.mean(variances),
                maximum_importance_weight=max(weights), realized_domain_counts=json.dumps(totals),
                max_formula_errors=json.dumps(errors), batch_trace_sha=sha(source),
                producer_receipt_sha=sha(directory / 'receipt.json')))
    with (args.out / 'batch_composition.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.out / 'receipt.json').write_text(json.dumps(dict(status='PASS', source_sha=sha(Path(__file__)),
        manifests=[dict(path=str(p), sha256=sha(p)) for p in args.manifest], runs=len(records),
        scope='Verified complete batch-count/weight/ESS identities, without model or test predictions. '
              '1/E[w^2] is the large-batch limit of ESS/n, not exact finite-batch expected ESS. '
              'ESS is weight-based and does not identify gradient covariance or generalization.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
