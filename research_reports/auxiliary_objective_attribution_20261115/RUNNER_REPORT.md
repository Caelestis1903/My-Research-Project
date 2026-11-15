# Runner Report — auxiliary_objective_attribution_20261115

## Run record

- Status: **COMPLETE**
- Stage: STAGE1
- Start (UTC): 2026-11-15T08:22:27.939323+00:00
- End (UTC): 2026-11-15T09:50:44.918886+00:00
- Elapsed: 5296.972 seconds
- Git commit: `e5738e52294f064c23a9b953fe74abb84de713df`
- Reviewed source SHA-256: `a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`

## Configuration and completion

- Datasets: movielens, kuairand
- Arms: B, C, Z-PAIR, Z-EXAMPLE
- Seeds: 2022, 2023, 2024, 2025, 2026
- Per trajectory: 50 epochs, batch size 40960; full resolved configuration: `artifacts/auxiliary_objective_attribution_20261115/attempt1/run_manifest.json`
- New trajectories: 40; reused CLEAN trajectories: 10; trajectories represented in tables: 50
- Verified updates from result JSON: 69000 optimizer and 69000 EMA
- Test accessed: **false**
- Failed runs: 0; retries: 0; missing runs: 0
- Final host RAM fraction reported by progress: 0.3276184373497689; scheduler admission limits were per-device memory/power below 85% and host RAM below 90%.

## Matrix counts

| Dataset / arm | Completed |
|---|---:|
| movielens / B | 5 |
| movielens / C | 5 |
| movielens / Z-PAIR | 5 |
| movielens / Z-EXAMPLE | 5 |
| kuairand / B | 5 |
| kuairand / C | 5 |
| kuairand / Z-PAIR | 5 |
| kuairand / Z-EXAMPLE | 5 |

## Raw metrics and logs

Tables exported by the runner: `ENDPOINTS.csv`, `VALIDATION_CURVES.csv`, `SUMMARY.csv`, `PAIRED_SEEDS.csv`, `PAIRED_SUMMARY.csv`, `WEIGHTS.csv`, and `COSTS.csv` under `artifacts/auxiliary_objective_attribution_20261115/attempt1/tables/`.

Worker logs are in `artifacts/auxiliary_objective_attribution_20261115/attempt1/launch_logs/`; epoch commits are recorded in `epoch_commit_ledger.jsonl`. The Test partition was not accessed. This report records execution and raw output locations only.
