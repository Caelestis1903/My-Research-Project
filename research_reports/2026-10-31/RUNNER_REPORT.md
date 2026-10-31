# GA31-v2 Experiment Execution Report

## 1. Experiment ID

GA31-v2, attempt `attempt_20261031_181603`.

## 2. Git/worktree state

The current working directory is not inside a Git worktree (`git status --short --branch` returned “not a git repository”). The attempt records the frozen source commit and source blob identities in `source_manifest.json` and `resolved_protocol.json`.

## 3. Config

- Mode: formal train, GPUs `0,1`.
- Matrix: MovieLens and KuaiRand × arms P0/P1/P2/P3 × five seeds = 40 runs.
- Seeds: 2022, 2023, 2024, 2025, 2026.
- Training: batch size 40,960; 50 epochs per run; one Adam optimizer at the frozen protocol settings. The resolved configuration is recorded in `resolved_protocol.json`; each run has `resolved_config.json`.
- State handling: in-memory coordinator snapshots; no model/checkpoint state files were found (`.pt`, `.pth`, `.pkl`, `.pickle`, `.ckpt`).

## 4. Seeds

All five configured seeds (2022–2026) were launched for each dataset-arm combination.

## 5. Run status

- Launch ledger rows: 40 (GPU 0: 12, GPU 1: 28).
- Run ledger rows: 40.
- `progress.json`: completed 40/40, active 0, pending 0.
- Epoch metrics: 40 files; every file's final recorded epoch is 50.
- `completion.json`: `COMPLETE`; 40 runs completed; 50 epochs each; test evaluation complete; 232 probe batches completed.
- Failure ledger: 0 data rows.

## 6. Raw metrics

- Validation metrics: `artifacts/gradient_aug_20261031/GA31-v2/attempt_20261031_181603/validation_metrics.csv` (120 rows: 40 runs × selected/E10/E50).
- Test metrics: `artifacts/gradient_aug_20261031/GA31-v2/attempt_20261031_181603/test_metrics.csv` (120 rows: 40 runs × selected/E10/E50).
- These files contain AUC and LogLoss at cell, domain, task, macro, pooled, and worst-cell levels as defined by the frozen metric protocol. They are raw per-run evaluation records.
- Deterministic aggregation artifacts were generated with `python -m experiments.gradient_aug_20261031.analyze --run-dir artifacts/gradient_aug_20261031/GA31-v2/attempt_20261031_181603`:
  - `paired_effects.csv`
  - `effect_summary.csv`
  - `arm_summary.csv`
  - `DECISIONS.json`
- Per-run epoch metrics, per-cell metrics, batch traces, augmentation statistics, optimizer statistics, gradient probes, and runtime records are under `artifacts/gradient_aug_20261031/GA31-v2/attempt_20261031_181603/runs/<dataset>/<arm>/seed<seed>/attempt1/`.

## 7. Log locations

Each of the 40 run directories contains `stdout.log` and `stderr.log` (40 of each). Coordinator-level ledgers and metadata are in the attempt root, including `launch_ledger.csv`, `run_ledger.csv`, `failure_ledger.csv`, `progress.json`, `completion.json`, `selection_frozen.json`, `environment.json`, and `resolved_protocol.json`.

## 8. Runtime errors

No failed runs were recorded. The coordinator exited with code 0. Worker training time summed from `run_ledger.csv` is 46,690.24 seconds; individual run training times range from 258.98 to 2,616.51 seconds. Worker `stderr.log` files contain PyTorch TF32 deprecation warnings; they did not prevent completion.

## 9. Missing runs

None. All 40 configured runs reached epoch 50; all 120 test metric rows and 232 prescribed probe batches were produced.
