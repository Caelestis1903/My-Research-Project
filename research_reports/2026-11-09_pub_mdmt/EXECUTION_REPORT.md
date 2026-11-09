# PUB-MDMT execution report

## Run identity and state

- Experiment: PUB-MDMT, 2026-11-09.
- Worktree: `/home/xzf/Code/phase0-3_pub_mdmt_20261109`; HEAD `40c8afafedab422bdf36c02b38aff19114d95a9b`.
- Reviewed implementation SHA-256: `0eb54f049abef69e735b90af2dbca4fb88dd876687cea3ab2719472730069e4d`; reviewer verdict PASS.
- Project Python: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` (all project commands used this executable).
- Thread environment: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- Initial resource snapshot was recorded before RUN. Both GPUs and host RAM were below the prescribed limits at that snapshot; pre-existing processes were left running.

## Configuration and completed work

- Data: MovieLens and KuaiRand.
- Formal matrix: 7 arms (`A`, `C0`, `C1`, `C2`, `G0`, `G1`, `G2`) × 5 seeds (`2022`–`2026`) × 2 datasets = 70 trajectories.
- Each formal trajectory completed epochs 1–50. Unique formal epoch count: 3,500. Validation JSONL has 4,000 rows because A trajectories emit both A and F rows; epoch counts above are deduplicated within each trajectory.
- Formal results: 70 `RESULT.json` files, each status `COMPLETE`; 70 first-attempt records. No formal `ERROR.json` files were present.
- Combination decision is frozen with no additional PUB+EMA jobs selected; therefore no additional five-seed/50-epoch trajectories were due.
- Physical optimizer-step count from the approved batch/data configuration: MovieLens 35 trajectories × 50 epochs × 20 batches = 35,000; KuaiRand 35 × 50 × 49 = 85,750; total 120,750.
- Calibration covered 18 solver configurations (200/200 successful evaluations each in the generated calibration records). Pilot completed all 40 low-budget trajectories. Frozen shared choices in `SOLVER_DECISION.json` and `SEARCH_DECISION.json`: solver `niter=100`, `tol=1e-4`, `ridge=1e-8`; frequency `10`.

## Command status

Commands were run in order; each exited 0:

1. `python -m experiments.pub_mdmt_20261109.run calibrate --device cuda:0` — `command_logs/calibrate_attempt1.log`.
2. `python -m experiments.pub_mdmt_20261109.run pilot --devices 0,1` — `command_logs/pilot_attempt1.log`.
3. `python -m experiments.pub_mdmt_20261109.run train --devices 0,1` — `command_logs/train_attempt1.log`.
4. `python -m experiments.pub_mdmt_20261109.run report` — `command_logs/report_attempt1.log`; reported `COMPLETE`, 70 physical trajectories, 480 endpoint rows, and 6,732 paired summaries.
5. After report completion, approved report-only export `python research_reports/2026-11-09_pub_mdmt/analyze_diagnostics.py --out artifacts/pub_mdmt_20261109/attempt1` — `command_logs/analyze_diagnostics_attempt1.log`; exited 0 and reported `COMPLETE`, 825 paired windows, 30 seed coverage rows, and 480 seed-stage-group rows.

No traceback, CUDA OOM, runtime error, or failed formal job was recorded in the monitored worker logs. No retry was required. The run produced no model, optimizer, checkpoint, or serialized training-state files. Diagnostic pairing coverage follows the matched rows present in the logs (KuaiRand first-batch utility pairing is sparse; MovieLens pairing is recorded each epoch); this is an execution/log-coverage note only.

## Artifact locations

- All attempt artifacts: `artifacts/pub_mdmt_20261109/attempt1/`.
- Calibration: `CALIBRATION.jsonl`, `CALIBRATION_GRAMS.json`, `SOLVER_DECISION.json`.
- Pilot decision: `SEARCH_DECISION.json`.
- Formal per-trajectory results, selections, validation records, attempts, and worker logs: `formal/`.
- Frozen combination decision and completion markers: `COMBINATION_DECISION.json`, `FORMAL_COMPLETE.json`, `EXPERIMENT_COMPLETE.json`.
- Numerical report tables: `tables/` (`ENDPOINTS.csv`, `SUMMARY.csv`, `PAIRED.csv`, `SEED_PAIRS.csv`, `COSTS.csv`, `FREQUENCY100_DIAGNOSTIC.csv`, `DECISIONS.json`, `NUMERICAL_TABLES.md`).
- Diagnostics export CSVs: `tables/DIAGNOSTIC_SEEDS.csv`, `tables/MECHANISM_PAIRS.csv`, and `tables/PAIR_COVERAGE.csv`.
- Command stdout/stderr logs: `command_logs/`.

No scientific interpretation is included in this execution record.
