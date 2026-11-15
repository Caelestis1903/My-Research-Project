# Stage 2 Runner Report — NDP-A

## Run record

- Status: **COMPLETE**
- Start (UTC): 2026-11-15T10:21:34.723766+00:00
- End (UTC): 2026-11-15T11:43:52.596843+00:00
- Elapsed: 4937.869 seconds
- Git commit: `e5738e52294f064c23a9b953fe74abb84de713df`
- Reviewed Stage 2 source SHA-256: `831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`
- Review: `STAGE2_IMPLEMENTATION_REVIEW.json` — PASS

## Configuration and completion

- Arm: NDP-A; datasets: movielens, kuairand; seeds: 2022, 2023, 2024, 2025, 2026
- Alpha: 0.03; mixture lambda: 0.25
- Each new trajectory: 50 epochs, batch size 40960
- Completed new trajectories: 10; reused CLEAN trajectories: 10; trajectories in export: 20
- Verified result totals: 500 epochs, 17250 optimizer updates, 17250 EMA updates
- Failed runs: 0; retries: 0; missing runs: 0
- Test accessed: **false**
- Selected RAM states remain with the guardian.

## Matrix counts

| Dataset | Seed | Completed |
|---|---:|---:|
| movielens | 2022 | 1 |
| movielens | 2023 | 1 |
| movielens | 2024 | 1 |
| movielens | 2025 | 1 |
| movielens | 2026 | 1 |
| kuairand | 2022 | 1 |
| kuairand | 2023 | 1 |
| kuairand | 2024 | 1 |
| kuairand | 2025 | 1 |
| kuairand | 2026 | 1 |

## Raw metrics, resources, and logs

Runner tables are in `artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/tables/`: `ENDPOINTS.csv`, `VALIDATION_CURVES.csv`, `SUMMARY.csv`, `PAIRED_SEEDS.csv`, `PAIRED_SUMMARY.csv`, `WEIGHTS.csv`, and `COSTS.csv`. They include the 10 new NDP-A and 10 reused CLEAN trajectories.

The final host RAM fraction recorded in `progress.json` was 0.25176841709286446; per-device memory/power admission limits were below 85%, and host RAM admission was below 90%. Worker logs are in `artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/launch_logs/`; epoch commits are recorded in `epoch_commit_ledger.jsonl`. The RAM guardian was left running.

This report records execution, raw accounting, and output paths; it makes no scientific assessment.
