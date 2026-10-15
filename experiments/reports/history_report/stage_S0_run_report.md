# Stage S0 raw run report

## Execution

- Working directory: `D:\CODE\phase0-3`
- Formal launcher: `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments\stage_S0_orchestrator.py --python C:\Users\86199\.conda\envs\EDDAenv\python.exe --rerun_failed`
- Matrix was resumed without `--max_runs`; validated `OK` artifacts were not rerun or overwritten.
- Raw collector: `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments\stage_S0_report.py`

## Environment and protocol status

- Runner environment: EDDAenv Python with PyTorch/CUDA preflight passed; formal device `cuda:0`.
- Formal matrix: 7 arms × 3 rotations × 4 seeds = 84 fresh joint runs.
- Formal budget: 50 epochs, batch size 40960, role sizes `150000,34500,3000`, instrumentation and oracle requirements enabled.

## Status and integrity

- Expected fresh joint runs: 84
- Formal-valid current artifacts: 84/84
- Valid rotation×seed blocks: 12/12
- Queue `all_blocks_valid`: `true`
- Report `complete`: `true`
- Master CSV: 85 append-only attempt rows, 84 unique formal keys; 84 current `OK` rows and 1 preserved historical `FAILED` attempt.
- The historical failed attempt was archived at `D:\CODE\phase0-3\experiments\stage_S0_raw_results\failed_attempts\adam_eq_tau__d0_large__2025\20260918T133033038980`; its formal key later completed successfully.
- No runner or training Python process remained after completion.

## Raw artifact paths

- Raw run directories: `D:\CODE\phase0-3\experiments\stage_S0_raw_results`
- Attempt log: `D:\CODE\phase0-3\experiments\stage_S0_runs.csv`
- Queue manifest: `D:\CODE\phase0-3\experiments\stage_S0_queue_manifest.json`
- Raw summary manifest: `D:\CODE\phase0-3\experiments\stage_S0_summary_manifest.json`
- Raw per-run summary: `D:\CODE\phase0-3\experiments\stage_S0_summary.csv`
- Paired raw table: `D:\CODE\phase0-3\experiments\stage_S0_paired.csv`
- Subgroup raw table: `D:\CODE\phase0-3\experiments\stage_S0_subgroup_gains.csv`
- Historical reproducibility audit: `D:\CODE\phase0-3\experiments\stage_S0_historical_reproduction_audit.csv`

The C_val/D_val selectors use validation macro-AUC only within each rotation-seed block; seed is the repeat unit and rotations are fixed crossed conditions within each seed. This is a raw aggregation artifact and contains no scientific conclusion.
