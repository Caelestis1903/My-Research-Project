"""Descriptive readout of the completed, audited B1 reconstruction (no training)."""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016'
SOURCE = BASE / 'analysis'
OUT = BASE / 'interpretation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((SOURCE / 'analysis_receipt.json').read_text())
    assert receipt['scope'] == 'B1_RECONSTRUCTION_FORENSIC'
    assert receipt['formal_runs_completed'] == 8
    assert receipt['state_count_audited'] == 400
    assert receipt['prediction_files_audited'] == 800
    for name, digest in receipt['output_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    OUT.mkdir(exist_ok=True)
    selected = pd.read_csv(SOURCE / 'selected_timing_decomposition.csv')
    same = pd.read_csv(SOURCE / 'same_epoch_differences.csv')
    r0 = selected[selected.rule.eq('R0')]
    cells = r0[r0.quantity.str.fullmatch(r'd[012]t[01]')]
    cell_mean = cells.groupby(['split', 'metric', 'quantity'])[['F', 'DH', 'delta']].mean().reset_index()
    stages = same.copy()
    stages['stage'] = (stages.epoch // 10).map(lambda x: f'{x*10:02d}-{x*10+9:02d}')
    # One row per training seed and stage. Epochs are correlated observations,
    # and are never counted as independent replications or used for a CI.
    stage_seed = stages.groupby(['seed', 'split', 'metric', 'quantity', 'stage']).delta.mean().reset_index()
    stage_mean = stage_seed.groupby(['split', 'metric', 'quantity', 'stage']).delta.agg(['mean', 'std', 'min', 'max']).reset_index()
    stage_mean['positive_seed_count'] = stage_seed.groupby(['split', 'metric', 'quantity', 'stage']).delta.apply(lambda x: int((x > 0).sum())).values
    windows = pd.read_csv(SOURCE / 'selected_row_order_windows.csv')
    window_pair = windows.pivot(index=['seed', 'split', 'domain', 'task', 'window'], columns='arm', values=['auc', 'logloss'])
    window_rows = []
    for metric in ['auc', 'logloss']:
        w = (window_pair[metric]['DH'] - window_pair[metric]['F']).rename('delta').reset_index()
        w['metric'] = metric
        window_rows.append(w)
    window_delta = pd.concat(window_rows, ignore_index=True)
    calibration = pd.read_csv(SOURCE / 'selected_calibration_summary.csv')
    cal_pair = calibration.pivot(index=['seed', 'split', 'domain', 'task'], columns='arm', values=['ece_10bins', 'brier'])
    cal_rows = []
    for metric in ['ece_10bins', 'brier']:
        c = (cal_pair[metric]['DH'] - cal_pair[metric]['F']).rename('delta').reset_index()
        c['metric'] = metric
        cal_rows.append(c)
    cal_delta = pd.concat(cal_rows, ignore_index=True)
    training = pd.read_csv(SOURCE / 'training_observations.csv')
    training['checkpoint'] = training.apply(lambda row: 'R0_and_final' if row.selected and row.epoch == 49 else 'R0' if row.selected else 'final' if row.epoch == 49 else 'other', axis=1)
    tables = {
        'r0_selected_comparisons': r0,
        'r0_cell_means': cell_mean,
        'rule_epochs': selected[['seed', 'rule', 'F_epoch', 'DH_epoch']].drop_duplicates(),
        'stage_seed_means': stage_seed,
        'stage_summary': stage_mean,
        'final_epoch_comparisons': same[same.epoch.eq(49)],
        'r0_window_differences': window_delta,
        'r0_calibration_differences': cal_delta,
        'selected_final_training_loss': training[training.checkpoint.ne('other')],
    }
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False)
    result = {
        'scope': 'DESCRIPTIVE_READOUT_OF_AUDITED_RECONSTRUCTION',
        'training_runs_added': 0,
        'source_receipt_sha256': sha(SOURCE / 'analysis_receipt.json'),
        'source_analysis_sha256': receipt['analysis_source_sha256'],
        'readout_source_sha256': sha(Path(__file__)),
        'stage_uncertainty': 'Across four seed stage means; epochs are not independent repetitions.',
        'rule_sensitivity': 'R0 remains primary; R1-R4 are forensic sensitivity analyses.',
        'output_sha256': {f'{name}.csv': sha(OUT / f'{name}.csv') for name in tables},
    }
    (OUT / 'readout_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'tables': len(tables), 'source_hashes_verified': len(receipt['output_sha256'])}))


if __name__ == '__main__':
    main()
