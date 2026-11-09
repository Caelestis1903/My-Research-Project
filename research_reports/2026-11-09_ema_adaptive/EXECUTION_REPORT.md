# EXPERIMENT EXECUTION REPORT — A0

## 1. Experiment identity

- Experiment: `MDMT_ADAPTIVE_EMA_20261109`, stage A0
- Worktree: `/home/xzf/Code/phase0-3_ema_adaptive_20261109`
- Branch: `26-11-09`
- Source: `26-11-07` at `ffa3de067fcda5a1293af83a465d741e069493f1`
- Manifest: `research_plans/26-11-09/experiment_manifest.json`
- Seeds: 2022, 2023, 2024, 2025, 2026
- Completion time: 2026-11-09 03:21:50 UTC

## 2. Configuration and run status

The existing pulse metrics and prediction tables were audited, A0 numeric analysis was executed, and the numerical appendix was generated. All three commands exited 0. No model training was run.

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_adaptive_20261109.run audit --manifest research_plans/26-11-09/experiment_manifest.json --out artifacts/ema_adaptive_20261109/attempt1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_adaptive_20261109.run analyze --stage A0 --manifest research_plans/26-11-09/experiment_manifest.json --out artifacts/ema_adaptive_20261109/attempt1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_adaptive_20261109.run report --manifest research_plans/26-11-09/experiment_manifest.json --out artifacts/ema_adaptive_20261109/attempt1
```

Raw command records:

| Command | Exit code | Captured stdout | stderr |
| --- | ---: | --- | --- |
| `audit` | 0 | `logs/audit.stdout.log` | empty |
| `analyze --stage A0` | 0 | `logs/analyze.stdout.log` | empty |
| `report` | 0 | `logs/report.stdout.log` | empty |
| `release` | 0 | `logs/release.stdout.log` | empty |

## 3. Raw output inventory

All CSV row counts in `artifacts/ema_adaptive_20261109/attempt1`:

| File | Rows |
| --- | ---: |
| `A0_ACTIONS.csv` | 39,600 |
| `A0_HISTORICAL_ACTION_SCOPES.csv` | 336 |
| `A0_HISTORICAL_VALIDATION_FIT.csv` | 15,600 |
| `A0_PERSISTENCE_ROWS.csv` | 21,780 |
| `A0_PREDICTION_SEED.csv` | 7,920 |
| `A0_PREDICTION_SUMMARY.csv` | 36,432 |
| `A0_PREDICTIVE_OOF.csv` | 26,400 |
| `A0_SEED.csv` | 35,100 |
| `A0_STATIC_ACTIONS.csv` | 4,800 |
| `A0_STATIC_POLICIES.csv` | 120 |
| `A0_STATIC_SEED.csv` | 1,440 |
| `A0_STATIC_SUMMARY.csv` | 2,880 |
| `A0_S_CONTRASTS.csv` | 6,000 |
| `EVIDENCE_REUSE.csv` | 102 |
| `COSTS.csv` | 3 |

Machine decision was written to `A0_DECISION.json`; its recorded state is `NO_ACTIONABLE_SIGNAL`. Numerical tables are in `A0_NUMERICAL_APPENDIX.md`; `AUDIT_INVENTORY.json`, `SOURCE_AUDIT.md`, `RESOLVED_PROTOCOL.json`, `RESOURCE_ESTIMATE.json`, and `REPORT_RECEIPT.json` are also present.

## 4. Costs and errors

| Stage | Wall seconds | CPU seconds | Peak RSS bytes | Output bytes | New Adam steps | New GPU seconds | New Test accesses | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audit | 7.150061 | 6.985110 | 1,306,411,008 | 106,075 | 0 | 0 | 0 | 0 |
| Analyze | 56.767722 | 56.738389 | 802,738,176 | 104,307,281 | 0 | 0 | 0 | 0 |
| Report | 0.155625 | 0.155553 | 643,657,728 | 104,327,873 | 0 | 0 | 0 | 0 |
| Release | 0.001660 | 0.001657 | 605,839,360 | 104,328,499 | 0 | 0 | 0 | 0 |

Runtime errors: none. Failed runs: none. Missing A0 seed runs: none. Release was repeated after a provenance wording correction in the final scientific report; the final receipt records `RELEASED`, final report SHA-256 `4793b01d80ea8e20a35e690c752c604e07027567de4efe6bd417ce671a7927b0`, receipt SHA-256 `a2aa84ed94c335943f77fb071a1ba0ffb3c1c2ce406199ee24b05e9a9ba34e47`, and zero model/checkpoint files, training runs, Adam steps, GPU seconds, or Test accesses. The second release took 0.001231 wall seconds and exited 0; see the last row of `artifacts/ema_adaptive_20261109/attempt1/COSTS.csv`.
