# Experiment Execution Report

1. Experiment ID: `M3OE-supplied-reproduction-20261021`

2. Git/worktree: run cwd `/home/xzf/Code/phase0-3`; execution used reviewed worktree sources recorded per attempt in config.json. Training artifacts are external at `/home_new/xzf_m3oe_reproduction_20261021`. The orchestrator reports reviewed implementation commit `2f1dc436` in its isolated delivery worktree.

3. Config: `batch_size=40960`, epoch cap 50, strict validation six-cell macro-AUC selection, patience 10, best-checkpoint restoration; O/B/D/G arms; native configs contain full settings and source SHA256. Six scheduler invocations followed the reviewed README order across the two datasets.

4. Seeds: 2022, 2023, 2024, 2025 for each arm and dataset.

5. Run status: 32/32 expected unique formal keys valid; missing=0, extra=0, duplicate=0, invalid completions=0, failed formal queue exits=0. All 32 keys complete.

6. Raw metrics: each formal `completion.json` preserves the complete raw test metric object; `test_metrics.csv` and per-epoch validation/per-cell metrics remain in each attempt. `VALIDATION_METRICS.json` contains the selected-epoch validation metrics for Stage 2. `RAW_METRICS.json` is a separate consolidated, uninterpreted copy of the test metric fields; it contains no comparisons or interpretation. No cross-arm interpretation is included here.

7. Logs: per-attempt `stdout.log`, `stderr.log`, and scheduler launch logs remain under OUT. Exact command arrays, attempt folders, source receipts, file paths, selected/stopped epoch, steps, wall time, peak GPU memory and causal source-map SHA256 are in `RUNS.csv` and `INTEGRITY.json`.

8. Runtime errors: none recorded; all 32 formal queue exits returned 0, all 32 formal stderr logs are empty, and no error attempt remains.

9. Missing runs: none.

Profile gate: all four MovieLens and all four KuaiRand one-epoch, full-batch profiles are valid, non-formal, and have checkpoint files; details are in INTEGRITY.json.

Raw integrity: matching raw, encoded-array, ordered split-row and encoder/scaler receipts are checked per dataset; paired initialization hashes are checked per dataset/seed; best checkpoint file SHA256 is recomputed; all 32 checkpoint payloads contain their model, optimizer, RNG, loader-generator, epoch, step and validation state (O also contains its auxiliary optimizer); selected checkpoint epoch and stopping rule are checked. The canonical causal source-map SHA256 is identical across all 32 runs and matches current files. The only source-receipt hashes that vary are for analyze.py, a non-training-path analysis helper updated during the run as reported by the orchestrator.

## Files

- [RUNS.csv](RUNS.csv) — exact run commands, status, timing, memory and artifact locations.
- [INTEGRITY.json](INTEGRITY.json) — raw receipt consistency and completeness checks.
- [VALIDATION_METRICS.json](VALIDATION_METRICS.json) — selected-epoch validation metrics only.
- [RAW_METRICS.json](RAW_METRICS.json) — uninterpreted consolidated test metrics.
