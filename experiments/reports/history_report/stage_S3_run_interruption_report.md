# EXPERIMENT EXECUTION REPORT — Stage S3 RUN INTERRUPTION

## 1. Experiment ID

Stage S3 Amendment A1; frozen queue `experiments/stage_S3_queue_manifest_A1.json`.

## 2. Execution command

Reviewer section 8 command was used with the one-time reviewed validator rebind:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments/stage_S3_orchestrator.py --execute --queue-manifest experiments/stage_S3_queue_manifest_A1.json --legacy-queue-manifest experiments/stage_S3_raw_results/.stage_S3_archives/s3_kuairand_natural_seed2022/parent_abort_79c649abb1994ee81d49_wave_1/queue_snapshot.json --review-report experiments/reports/stage_S3_amendment_review.md --raw-root experiments/stage_S3_raw_results --s1-raw-root experiments/stage_S1_raw_results --s2-raw-root experiments/stage_S2_raw_results --python C:\Users\86199\.conda\envs\EDDAenv\python.exe --device cuda:0 --max-workers 2 --allow-reviewed-validator-rebind
```

The process was launched through an equivalent encoded PowerShell wrapper only to prevent the active-process detector from matching its own command line. No arguments or paths were changed.

## 3. Seeds and configurations

Seeds: 2022, 2023, 2024, 2025. Frozen settings: KuaiRand/natural and MovieLens/d2_large; objectives Equal and Sample; access `TARGET_ONLY_SHARED`; Sample FULL replacement groups; device `cuda:0`; maximum training concurrency 2.

## 4. Run status

- Fresh accepted artifacts: 24/24.
- Formal launches: 24; attempt records: 26.
- Seed 2023 target-only wave 1 failed validation and was retained as a non-scientific archive. Both members were rerun together from epoch 0 in wave 2 and passed.
- All other fresh groups passed per-run validation on wave 1.
- All 24 fresh artifacts were promoted into canonical run directories.
- Final strict four-way validation stopped with a validator error before final collector completion.
- Queue status at stop: `RUNNING`; no further training was launched after the strict validation error.

## 5. Concurrency evidence

Every multi-seed or paired group was launched with `max_workers=2`. Runtime snapshots repeatedly showed one orchestrator and exactly two `stage_S3_run.py` Python children. Examples recorded in the run session include seed 2023 wave 2 and seed 2024 wave 1; both children used `cuda:0` and the paired objective settings. No group exceeded two training processes.

## 6. Failures and retries

The only child validation failure was seed 2023 wave 1, both members: exit code 0, validator reasons `final_common49_hash_mismatch` and `manifest_mismatch:dataset`. The prescribed whole-group wave 2 rerun from epoch 0 passed both members. No singleton repair and no checkpoint resume were used.

Finalization failure:

```text
strict four-way pair invalid: ['pairing_artifact_hash_mismatch:objective_coefficients.json', 'pairing_field_mismatch:coefficient_table_hash']
```

Per-run validator success did not override this strict paired validator failure. Execution stopped; no algorithm, queue, or archive files were changed to bypass it.

## 7. Raw metrics and artifact locations

Each accepted run directory contains raw `metrics_test.csv` and `metrics_summary.csv` with AUC and LogLoss for selected, final, and common epochs 9/19/29/39/49, including all six domain-task cells. Prediction artifacts are `predictions_test.npz`; run provenance is `run_manifest.json`; complete per-run logs are `stdout.log`, `launcher_stdout.log`, and `launcher_stderr.log` where applicable.

Canonical fresh artifact roots:

- `experiments/stage_S3_raw_results/target_only_shared/`
- `experiments/stage_S3_raw_results/full_replacement/sample/`

Attempt and complete archive roots:

- `experiments/stage_S3_raw_results/.stage_S3_attempts/`
- `experiments/stage_S3_raw_results/.stage_S3_archives/`

Queue and orchestrator console logs:

- `experiments/stage_S3_queue_manifest_A1.json`
- `experiments/reports/stage_S3_run_console_resume.log`
- `experiments/reports/stage_S3_run_console_resume2.log`

## 8. Missing outputs / unresolved validation

The consolidated collector outputs (`stage_S3_raw_results.csv`, paired results, interaction table, and final inventory report) were not finalized because strict four-way validation stopped on the objective-coefficient artifact hash mismatch. No scientific conclusions are reported.

