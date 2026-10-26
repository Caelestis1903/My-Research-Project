"""Compact descriptive geometry readout with actual within-epoch step lags."""
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    moment = args.root / 'analysis/moment_telemetry_correction_initial50'
    correction = moment / 'corrected_moment_observations.csv'
    correction_receipt = json.loads((moment / 'receipt.json').read_text())
    digest = sha(correction)
    assert digest in json.dumps(correction_receipt), 'Correction receipt mismatch'
    with correction.open() as stream:
        corrected = {(r['run'], int(r['epoch']), int(r['batch']), r['block']): r
                     for r in csv.DictReader(stream)
                     if r['run'].startswith('V4final__') and int(r['budget']) == 50}
    records, identities = [], []
    for optimizer in ('adam', 'sgd'):
        for seed in range(2022, 2026):
            for arm in ('SN', 'SB'):
                run = f'V4final__kuairand__{optimizer}__{arm}__seed{seed}'
                directory = args.root / 'archive/formal' / run / 'budget050'
                receipt = json.loads((directory / 'receipt.json').read_text())
                source = directory / 'optimizer_observations.csv'
                assert sha(source) == receipt['artifact_hashes'][source.name]
                identities.append(dict(run=run, archive=str(directory),
                                       receipt_sha=sha(directory / 'receipt.json'),
                                       observations_sha=sha(source)))
                with source.open() as stream:
                    observations = list(csv.DictReader(stream))
                for epoch in (1, 5, 10, 20, 40):
                    for block in 'CPRE':
                        rows = [r for r in observations
                                if int(r['epoch']) == epoch and r['block'] == block]
                        assert [int(r['batch']) for r in rows] == list(range(6))
                        steps = [int(r['steps']) for r in rows]
                        assert steps == list(range(steps[0], steps[0] + 6))
                        record = dict(optimizer=optimizer, seed=seed, arm=arm,
                                      epoch=epoch, block=block, observed_batches=6)
                        for key in ('gradient_norm', 'update_norm'):
                            record['mean_' + key] = statistics.mean(float(r[key]) for r in rows)
                        record['mean_update_to_gradient_ratio'] = statistics.mean(
                            float(r['update_norm']) / float(r['gradient_norm']) for r in rows)
                        pure = [corrected[(run, epoch, int(r['batch']), block)] for r in rows]
                        for original, fixed in zip(rows, pure):
                            assert float(original['exp_avg']) == float(fixed['exp_avg'])
                            assert float(original['exp_avg_sq']) == float(fixed['exp_avg_sq'])
                        record['mean_corrected_m_norm'] = statistics.mean(float(r['corrected_m_norm']) for r in pure)
                        record['mean_v_norm'] = statistics.mean(float(r['v_norm']) for r in pure)
                        for lag in (1, 2, 5):
                            within = [r for r in rows if int(r['batch']) >= lag]
                            assert len(within) == 6 - lag and all(r[f'lag{lag}_cosine'] for r in within)
                            record[f'within_epoch_lag{lag}_cosine_mean'] = statistics.mean(
                                float(r[f'lag{lag}_cosine']) for r in within)
                            record[f'within_epoch_lag{lag}_count'] = len(within)
                        records.append(record)
    with (args.out / 'moving_theta_stages.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fixed = args.root / 'analysis/V1_full_coordinate_geometry.csv'
    paired = args.root / 'analysis/V1_paired_geometry.csv'
    result = dict(status='COMPLETE', source_sha=sha(Path(__file__)),
                  correction_sha=digest, fixed_geometry_sha=sha(fixed),
                  paired_geometry_sha=sha(paired), sources=identities,
                  records=len(records), independent_review=False,
                  lags='Within first six contiguous production batches only; lag5 has one pair per stage.',
                  scientific_conclusion='NONE; descriptive raw readout only')
    (args.out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
