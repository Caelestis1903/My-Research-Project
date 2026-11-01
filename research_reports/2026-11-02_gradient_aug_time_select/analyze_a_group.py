"""Reproduce validation-only A0/A1 time-estimator tables from completed raw runs."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gradient_aug_time_select_20261102.analyze import summarize

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True, type=Path)
args = parser.parse_args()
out = Path(__file__).resolve().parent
rows = []
# shortcut: A0/A1 completed on attempt1 here; consult the ledger when reusing this analysis with retries.
for path in args.run_dir.glob('runs/*/*/A0/seed*/attempt1/primary/state_summary.json'):
    for item in json.loads(path.read_text())['checkpoints']:
        if item.get('validation') is not None and item['seed'] in (2022, 2024):
            rows.append({**{k: v for k, v in item.items() if k != 'validation'}, **item['validation']})
frame = pd.DataFrame(rows)
assert len(frame) == 44
assert not frame.duplicated(['dataset', 'seed', 'model_kind', 'checkpoint']).any()
frame.to_csv(out / 'a_group_validation_endpoints.csv', index=False)
paired, summaries = [], []
metrics = [c for c in frame if c.endswith(('_auc', '_logloss'))]
for (dataset, checkpoint), group in frame.groupby(['dataset', 'checkpoint']):
    for effect, left, right in [('fixed-online', 'fixed_ema', 'online'),
                               ('delayed-fixed', 'delayed5', 'fixed_ema'),
                               ('swa-fixed', 'swa20', 'fixed_ema')]:
        a = group[group.model_kind == left].set_index('seed')
        b = group[group.model_kind == right].set_index('seed')
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) != 2:
            continue
        for metric in metrics:
            differences = []
            for seed in seeds:
                av, bv = float(a.loc[seed, metric]), float(b.loc[seed, metric])
                differences.append(av - bv)
                paired.append(dict(dataset=dataset, checkpoint=checkpoint, effect=effect, metric=metric,
                                   seed=seed, candidate=av, reference=bv, delta=av - bv))
            summaries.append(dict(dataset=dataset, checkpoint=checkpoint, effect=effect, metric=metric,
                                  **summarize(differences)))
pd.DataFrame(paired).to_csv(out / 'a_group_validation_paired.csv', index=False)
pd.DataFrame(summaries).to_csv(out / 'a_group_validation_summary.csv', index=False)
print('A0 validation analysis complete: 44 valid endpoints, seed is the replicate unit.')
decision = args.run_dir / 'A_decision.json'
if decision.exists():
    a1_rows = []
    for path in args.run_dir.glob('runs/development/*/A1/seed*/attempt1/state_summary.json'):
        for item in json.loads(path.read_text())['checkpoints']:
            a1_rows.append({**{k: v for k, v in item.items() if k != 'validation'}, **item['validation']})
    a1 = pd.DataFrame(a1_rows)
    assert len(a1) == 48 and not a1.duplicated(['dataset', 'seed', 'model_kind', 'checkpoint']).any()
    a1.to_csv(out / 'a1_c2_validation_endpoints.csv', index=False)
    (out / 'A1_C2_VALIDATION_DECISION.json').write_bytes(decision.read_bytes())
    print('A1/C2 validation analysis complete: 48 endpoints and original decision receipt.')
