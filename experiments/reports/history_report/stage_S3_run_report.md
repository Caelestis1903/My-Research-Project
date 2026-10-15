# EXPERIMENT EXECUTION REPORT — Stage S3 A1 FINAL

## 1. Experiment ID

Stage S3 Amendment A1, frozen queue `experiments/stage_S3_queue_manifest_A1.json`.

## 2. Finalization command

Executed reviewer latest report section 8 command with `--allow-reviewed-coefficient-rebind` only. The prior validator rebind flag was not supplied. The command performed reviewed coefficient rebind, read-only reconciliation, and strict validation; it launched no `stage_S3_run.py` process.

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments/stage_S3_orchestrator.py --execute --queue-manifest experiments/stage_S3_queue_manifest_A1.json --legacy-queue-manifest experiments/stage_S3_raw_results/.stage_S3_archives/s3_kuairand_natural_seed2022/parent_abort_79c649abb1994ee81d49_wave_1/queue_snapshot.json --review-report experiments/reports/stage_S3_amendment_review.md --raw-root experiments/stage_S3_raw_results --s1-raw-root experiments/stage_S1_raw_results --s2-raw-root experiments/stage_S2_raw_results --python C:\Users\86199\.conda\envs\EDDAenv\python.exe --device cuda:0 --max-workers 2 --allow-reviewed-coefficient-rebind
```

Finalization log: `experiments/reports/stage_S3_run_console_finalization.log`.

## 3. Collector command

Executed only after queue completion:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments/stage_S3_report.py --queue-manifest experiments/stage_S3_queue_manifest_A1.json --raw-root experiments/stage_S3_raw_results --s1-raw-root experiments/stage_S1_raw_results --s2-raw-root experiments/stage_S2_raw_results --output-root experiments
```

Collector log: `experiments/reports/stage_S3_collector_console.log`.

## 4. Seeds and configurations

Seeds: 2022, 2023, 2024, 2025. Settings: KuaiRand/natural and MovieLens/d2_large. Fresh conditions: Equal FULL control reuse, Sample FULL replacement, Equal target-only shared, and Sample target-only shared. All 24 fresh artifacts are present and per-run valid. The prior seed2023 wave1 failure remains archived; its prescribed wave2 pair was completed from epoch 0.

## 5. Run and validation status

- Queue status: `COMPLETE`.
- Fresh records: 24/24.
- Formal launches: 24; attempt records: 26.
- Strict blocks: 8/8, all `pairing_valid=true`, `strict_source_identity=true`.
- Queue finalization reconciliation: 24/24 runs, 0 remaining groups, 0 remaining runs.
- Training processes before and after finalization/collector: none detected.
- No retraining, checkpoint resume, singleton repair, or additional promotion occurred during finalization.

## 6. Raw metrics and output cardinalities

Collector completed with:

- `stage_S3_raw_results.csv`: 1,680 rows;
- `stage_S3_paired_results.csv`: 2,688 rows, including 1,344 strict and 1,344 legacy-context rows;
- `stage_S3_interactions.csv`: 336 rows;
- `stage_S3_runs.csv`: 56 rows;
- `stage_S3_aggregates.csv`: generated;
- `stage_S3_summary_manifest.json`: `status=STRUCTURALLY_VALIDATED`, `strict_invalid_count=0`, `interaction_invalid_count=0`.

Per-run raw AUC and LogLoss files remain in each canonical run directory as `metrics_test.csv` and `metrics_summary.csv`, with prediction-level artifacts in `predictions_test.npz`.

## 7. Artifact locations

- Canonical fresh artifacts: `experiments/stage_S3_raw_results/target_only_shared/` and `experiments/stage_S3_raw_results/full_replacement/sample/`.
- Attempts: `experiments/stage_S3_raw_results/.stage_S3_attempts/`.
- Archives: `experiments/stage_S3_raw_results/.stage_S3_archives/`.
- Queue: `experiments/stage_S3_queue_manifest_A1.json`.
- Collector outputs: `experiments/stage_S3_raw_results.csv`, `experiments/stage_S3_paired_results.csv`, `experiments/stage_S3_interactions.csv`, `experiments/stage_S3_runs.csv`, `experiments/stage_S3_aggregates.csv`, `experiments/stage_S3_summary_manifest.json`.

## 8. Runtime errors and missing runs

No runtime errors occurred during finalization or collector. No runs are missing. The earlier strict coefficient hash mismatch was resolved only through the reviewer-authorized coefficient rebind; no code or algorithm change was made. No scientific conclusions are included.

