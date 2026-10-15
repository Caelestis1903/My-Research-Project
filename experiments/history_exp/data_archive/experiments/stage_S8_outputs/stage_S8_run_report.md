# Stage S8 Run Report (Mechanical)

Per Algorithm Specification `stage_S8_design.md`.

This report is the mechanical aggregator skeleton; scientific analysis is the responsibility of the main agent.  See `stage_S8_summary_manifest.json` for the summary metadata and the four CSVs in `stage_S8_outputs/` for the raw tables.

## Coverage matrix

| dataset/seed | anchors |
|---|---|
| kuairand/seed2022 | late=OK, middle=OK, selected=OK |
| kuairand/seed2023 | late=OK, middle=OK, selected=OK |
| kuairand/seed2024 | late=OK, middle=OK, selected=OK |
| kuairand/seed2025 | late=OK, middle=OK, selected=OK |
| movielens/seed2022 | late=OK, middle=OK, selected=OK |
| movielens/seed2023 | late=OK, middle=OK, selected=OK |
| movielens/seed2024 | late=OK, middle=OK, selected=OK |
| movielens/seed2025 | late=OK, middle=OK, selected=OK |

## Phase 1 base retrain V-BASE results

| dataset | seed | selected_epoch | selected_macro_auc | initial_hash_match | vbase_ok | reasons |
|---|---|---|---|---|---|---|
| movielens | 2022 | 7 | 0.7313916750519242 | True | True | [] |
| movielens | 2023 | 8 | 0.7293003086652066 | True | True | [] |
| movielens | 2024 | 8 | 0.7319634433733193 | True | True | [] |
| movielens | 2025 | 12 | 0.7306116301896478 | True | True | [] |
| kuairand | 2022 | 30 | 0.6823676235661164 | True | True | [] |
| kuairand | 2023 | 47 | 0.6945312363811044 | True | True | [] |
| kuairand | 2024 | 41 | 0.6924487619701608 | True | True | [] |
| kuairand | 2025 | 24 | 0.6829631842501961 | True | True | [] |

All 8 base retrain runs pass V-BASE with selected_epoch/initial_hash/AUC-diff/fingerprint diff all == 0.0 versus the S6 canonical source (max per-epoch macro AUC diff 0.0 across 50 epochs; train batch fingerprints match 250/250 for ML and 2450/2450 for KR).  Manual transcription (the auto-aggregator's `_all_base_run_dirs()` only inspects `seed_dir/run_manifest.json` but the actual manifest lives at `seed_dir/joint/run_manifest.json`, so the script emitted 0 base rows).

## Phase 2 continuation V-checks (24 runs)

All 24 continuation runs pass `validate_run` (`ok=True`, no `ABORTED` audits).  Per-task V0 / V0 step-1 / V3 / V4-arch / V5 / V2 review flags (transcribed from `replay_audit.json`):

| dataset | seed | anchor | wall (s) | V0 anchor diff | V0 step-1 L_full | V0 step-1 delta rel | V3 all-equal | V4 arch (match/total) | V5 shared (obs/exp) | V5 priv_cov | V2 flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| movielens | 2022 | selected/7 | 65.0 | 0.00e+00 | 0.0 | 3.28e-09 | True | 2/2 | 40/40 | True | False |
| movielens | 2022 | middle/19 | 65.0 | 0.00e+00 | 0.0 | 2.54e-09 | True | 2/2 | 100/100 | True | False |
| movielens | 2022 | late/39 | 65.3 | 0.00e+00 | 0.0 | 2.36e-08 | True | 2/2 | 200/200 | True | False |
| movielens | 2023 | selected/8 | 66.0 | 0.00e+00 | 0.0 | 3.43e-08 | True | 2/2 | 45/45 | True | False |
| movielens | 2023 | middle/19 | 66.6 | 0.00e+00 | 0.0 | 2.75e-09 | True | 2/2 | 100/100 | True | False |
| movielens | 2023 | late/39 | 66.2 | 0.00e+00 | 0.0 | 2.88e-08 | True | 2/2 | 200/200 | True | False |
| movielens | 2024 | selected/8 | 66.1 | 0.00e+00 | 0.0 | 3.33e-08 | True | 2/2 | 45/45 | True | False |
| movielens | 2024 | middle/19 | 65.9 | 0.00e+00 | 0.0 | 2.81e-09 | True | 2/2 | 100/100 | True | False |
| movielens | 2024 | late/39 | 66.5 | 0.00e+00 | 0.0 | 2.68e-08 | True | 2/2 | 200/200 | True | False |
| movielens | 2025 | selected/12 | 67.1 | 0.00e+00 | 0.0 | 1.12e-08 | True | 2/2 | 65/65 | True | False |
| movielens | 2025 | middle/19 | 66.1 | 2.22e-16 | 0.0 | 5.21e-09 | True | 2/2 | 100/100 | True | False |
| movielens | 2025 | late/39 | 66.0 | 0.00e+00 | 0.0 | 2.32e-08 | True | 2/2 | 200/200 | True | False |
| kuairand | 2022 | selected/30 | 1023.9 | 0.00e+00 | 0.0 | 6.85e-09 | True | 2/2 | 1519/1519 | True | False |
| kuairand | 2022 | middle/19 | 1051.6 | 0.00e+00 | 0.0 | 6.04e-08 | True | 2/2 | 980/980 | True | False |
| kuairand | 2022 | late/39 | 1053.5 | 1.11e-16 | 0.0 | 7.50e-08 | True | 2/2 | 1960/1960 | True | False |
| kuairand | 2023 | selected/47 | 1050.2 | 0.00e+00 | 0.0 | 2.40e-08 | True | 2/2 | 2352/2352 | True | False |
| kuairand | 2023 | middle/19 | 1058.9 | 0.00e+00 | 0.0 | 5.52e-08 | True | 2/2 | 980/980 | True | False |
| kuairand | 2023 | late/39 | 1065.7 | 1.11e-16 | 0.0 | 7.12e-08 | True | 2/2 | 1960/1960 | True | False |
| kuairand | 2024 | selected/41 | 1067.4 | 2.22e-16 | 0.0 | 7.27e-09 | True | 2/2 | 2058/2058 | True | False |
| kuairand | 2024 | middle/19 | 1054.6 | 0.00e+00 | 0.0 | 6.48e-08 | True | 2/2 | 980/980 | True | False |
| kuairand | 2024 | late/39 | 1058.5 | 0.00e+00 | 0.0 | 7.10e-08 | True | 2/2 | 1960/1960 | True | False |
| kuairand | 2025 | selected/24 | 1056.0 | 0.00e+00 | 0.0 | 5.32e-08 | True | 2/2 | 1225/1225 | True | False |
| kuairand | 2025 | middle/19 | 1053.9 | 0.00e+00 | 0.0 | 6.79e-08 | True | 2/2 | 980/980 | True | False |
| kuairand | 2025 | late/39 | 1038.7 | 0.00e+00 | 0.0 | 7.11e-08 | True | 2/2 | 1960/1960 | True | False |

V4-main fingerprint check (consumed vs base batch_trace.csv `kind=train` over the cont run's h_max_steps window) is also all `0/24` fails (24/24 match).  The arch step `expected` is `anchor_epoch + 1` and `shared_step_expected = bpe * (anchor_epoch + 1)`.

## Phase 1 base / Phase 2 continuation aggregate (raw transcription of `stage_S8_best_auc_table.csv`)

Per `(dataset, seed, anchor, branch, horizon)` -- 288 rows = 2 datasets x 4 seeds x 3 anchors x 3 branches x 4 horizons (5steps / half / 1ep / 2ep).  Below is the full table.  `f_original_best_auc` is the original S6 canonical `selected_validation_macro_auc` (design §9.11).

(Full 288-row table is in `stage_S8_best_auc_table.csv`; columns:
`stage,dataset,seed,anchor_class,anchor_epoch,branch,horizon,horizon_steps,best_step,continuation_best_auc,final_auc,f_original_best_auc,f_original_selected_epoch,delta_best,new_best,late_rescue`.)

## Source files

- `experiments/stage_S8_protocol.py`
- `experiments/stage_S8_base_run.py`
- `experiments/stage_S8_run.py`
- `experiments/stage_S8_orchestrator.py`
- `experiments/stage_S8_report.py`