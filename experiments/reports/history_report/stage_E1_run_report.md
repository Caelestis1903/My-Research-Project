# EXPERIMENT EXECUTION REPORT — Stage E1

## 1. Experiment ID

Stage E1 — formal 24-run matrix.

## 2. Git/worktree state

`D:\CODE\phase0-3` is not a Git worktree (`git rev-parse` unavailable); no Git revision is recorded.

## 3. Exact command and environment

Command issued from `D:\CODE\phase0-3`:

```powershell
$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -u experiments/stage_E1_orchestrator.py
```

The orchestrator launched its fixed queue using `D:\CODE\phase0-3\venv\Scripts\python.exe`, with `--device cuda:0`, `--batch_size 40960`, `--epoch 50`, `--learning_rate 0.01`, `--learning_rate_darts 0.001`, `--weight_decay 1e-05`, `--expert_num 4`, `--probe_interval 1`, `--instrument 1`, and `--role_sizes 150000,34500,3000`. Dataset path was `D:/CODE/phase0-3/m3oe/examples/data/ml-1m`.

Environment check: system Python with `D:\CODE\.venv\Lib\site-packages`; Torch `2.6.0+cu126`; CUDA available.

## 4. Seeds and matrix

Weighting `{equal, sample}` × setting `{d0_large, d1_large, d2_large}` × seed `{2022, 2023, 2024, 2025}` = 24 runs. Formal protocol used 50-epoch upper bound and patience 10.

## 5. Run status and raw cardinalities

All 24 runs returned code `0` and master status `OK`. Every run has `pair_status=VALID` and `pair_valid=True` after counterpart completion. `observed_epochs` are recorded in each manifest; post-stop epochs are censored.

| weighting | setting | seed | status | best epoch | stopped epoch | observed epochs | gradient_step | gradient_epoch | routing rows | nonfinite entries |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| equal | d0_large | 2022 | OK | 6 | 16 | 17 | 2550 | 510 | 153 | 918 |
| equal | d0_large | 2023 | OK | 5 | 15 | 16 | 2400 | 480 | 144 | 864 |
| equal | d0_large | 2024 | OK | 5 | 15 | 16 | 2400 | 480 | 144 | 864 |
| equal | d0_large | 2025 | OK | 7 | 17 | 18 | 2700 | 540 | 162 | 972 |
| equal | d1_large | 2022 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| equal | d1_large | 2023 | OK | 5 | 15 | 16 | 2400 | 480 | 144 | 864 |
| equal | d1_large | 2024 | OK | 9 | 19 | 20 | 3000 | 600 | 180 | 1080 |
| equal | d1_large | 2025 | OK | 11 | 21 | 22 | 3300 | 660 | 198 | 1188 |
| equal | d2_large | 2022 | OK | 10 | 20 | 21 | 3150 | 630 | 189 | 1134 |
| equal | d2_large | 2023 | OK | 10 | 20 | 21 | 3150 | 630 | 189 | 1134 |
| equal | d2_large | 2024 | OK | 6 | 16 | 17 | 2550 | 510 | 153 | 918 |
| equal | d2_large | 2025 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| sample | d0_large | 2022 | OK | 5 | 15 | 16 | 2400 | 480 | 144 | 864 |
| sample | d0_large | 2023 | OK | 4 | 14 | 15 | 2250 | 450 | 135 | 810 |
| sample | d0_large | 2024 | OK | 6 | 16 | 17 | 2550 | 510 | 153 | 918 |
| sample | d0_large | 2025 | OK | 12 | 22 | 23 | 3450 | 690 | 207 | 1242 |
| sample | d1_large | 2022 | OK | 11 | 21 | 22 | 3300 | 660 | 198 | 1188 |
| sample | d1_large | 2023 | OK | 6 | 16 | 17 | 2550 | 510 | 153 | 918 |
| sample | d1_large | 2024 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| sample | d1_large | 2025 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| sample | d2_large | 2022 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| sample | d2_large | 2023 | OK | 10 | 20 | 21 | 3150 | 630 | 189 | 1134 |
| sample | d2_large | 2024 | OK | 8 | 18 | 19 | 2850 | 570 | 171 | 1026 |
| sample | d2_large | 2025 | OK | 7 | 17 | 18 | 2700 | 540 | 162 | 972 |

## 6. Raw requested metrics

Raw per-epoch, per-domain, per-task train/validation/test AUC and LogLoss are in `metrics_epoch.csv` for every run. Raw gradient norms, weighted norms, variance/SNR and contribution fields are in `gradient_step.csv` and `gradient_epoch.csv`. Routing fields are in `gate_epoch.csv`. The collector preserved these fields in `stage_E1_summary.csv` with split, domain, task, weighting, setting, seed, parameter group, and routing columns.

Aggregate raw row counts across the 24 run directories: `metrics_epoch.csv` 7,200; `gradient_epoch.csv` 13,410; `gate_epoch.csv` 4,023; `batch_trace.csv` 2,682; `metrics_summary.csv` 144. The generated summary contains 108,000 split/parameter-group rows and retains seed-level rows.

## 7. Log locations and required artifacts

Raw root: [stage_E1_raw_results](../stage_E1_raw_results/)

Master status CSV: [stage_E1_runs.csv](../stage_E1_raw_results/stage_E1_runs.csv)

Structured summary: [stage_E1_summary.csv](../stage_E1_summary.csv)

Summary manifest: [stage_E1_summary_manifest.json](../stage_E1_summary_manifest.json)

Each of the 24 run directories contains all 11 required artifacts: `config.json`, `stdout.log`, `run_manifest.json`, `metrics_epoch.csv`, `gradient_step.csv`, `gradient_epoch.csv`, `gradient_cosine_epoch.csv`, `gate_epoch.csv`, `batch_trace.csv`, `metrics_summary.csv`, and `nonfinite_inventory.json`.

## 8. Schema, manifest, and pair validity

Required-file presence is 24/24 for every artifact. All 24 manifests report `status=OK`, `formal_valid=true`, and `pair_status=VALID`, `pair_valid=true`. Nonfinite inventory total is 24,138 entries; unexplained count is 0. The final resume-safe dry-run reported `pending: 0`.

## 9. Runtime errors and missing runs

Runtime errors: none. Missing runs: none. No retry was required. No scientific interpretation or conclusion is included in this execution report.
