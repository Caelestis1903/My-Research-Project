# Raw Experiment Report — P0-1

## 1. Experiment metadata

- Stage: P0-1 (role-reversal study, fixed total 187500 samples across 3 domains)
- Dataset: ml-1m (MDMTRec domain configuration: d0=L150000, d1=M30000, d2=S7500; mappings rotate per setting)
- Model: MDMTRec (from m3oe/examples)
- Method: joint (baseline, comp1) and oracle (single_domain=0/1/2)
- Config: --epoch 50 --batch_size 40960 --fix_optim_grouping 1 --role_setting <dX_large> --role_sizes 150000,30000,7500
- comp1 extra: --momentum_mode per_domain_only
- Seeds: 2022, 2023, 2024, 2025 (4 seeds per arm)
- Joint arms: 3 settings x 2 methods x 4 seeds = 24 runs
- Oracle arms: 3 settings x 3 domains x 4 seeds = 36 runs
- Total: 60 runs
- Code: experiments/p0_1_orchestrator.py + experiments/p4_run.py
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Start time: 2026-09-08 23:30 (orchestrator launch)
- End time: 2026-09-09 00:29 (last OK row written; P0_1_ORCHESTRATOR DONE)
- Wall-clock total: ~59 minutes
- Concurrency: max 2 p4_run children simultaneously

## 2. Completion status

All 60 unique tasks completed OK (no FAIL/CRASH rows in master CSV).
Master CSV `P0_1_raw.csv` had 115 OK rows (55 duplicated from dual-orchestrator); dedup version `P0_1_raw_dedup.csv` has 60 OK rows (one per unique task). See Section 5.

| run | status | checkpoint | log |
|---|---|---|---|
| joint_d0_large_baseline_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_baseline_seed2022.log |
| joint_d0_large_comp1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_comp1_seed2022.log |
| oracle_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d1_seed2022.log |
| oracle_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d0_seed2022.log |
| oracle_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d2_seed2022.log |
| joint_d0_large_baseline_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_baseline_seed2023.log |
| joint_d0_large_comp1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_comp1_seed2023.log |
| oracle_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d0_seed2023.log |
| oracle_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d1_seed2023.log |
| oracle_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d2_seed2023.log |
| joint_d0_large_baseline_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_baseline_seed2024.log |
| joint_d0_large_comp1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_comp1_seed2024.log |
| oracle_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d0_seed2024.log |
| oracle_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d2_seed2024.log |
| oracle_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d1_seed2024.log |
| joint_d0_large_baseline_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_baseline_seed2025.log |
| oracle_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d0_seed2025.log |
| joint_d0_large_comp1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d0_large_comp1_seed2025.log |
| oracle_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d2_seed2025.log |
| oracle_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d0_large_oracle_d1_seed2025.log |
| joint_d1_large_baseline_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_baseline_seed2022.log |
| joint_d1_large_comp1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_comp1_seed2022.log |
| oracle_d1_large_oracle_d1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d1_seed2022.log |
| oracle_d1_large_oracle_d2_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d2_seed2022.log |
| oracle_d1_large_oracle_d0_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d0_seed2022.log |
| joint_d1_large_baseline_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_baseline_seed2023.log |
| joint_d1_large_comp1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_comp1_seed2023.log |
| oracle_d1_large_oracle_d1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d1_seed2023.log |
| oracle_d1_large_oracle_d2_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d2_seed2023.log |
| oracle_d1_large_oracle_d0_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d0_seed2023.log |
| joint_d1_large_baseline_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_baseline_seed2024.log |
| joint_d1_large_comp1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_comp1_seed2024.log |
| oracle_d1_large_oracle_d1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d1_seed2024.log |
| oracle_d1_large_oracle_d2_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d2_seed2024.log |
| oracle_d1_large_oracle_d0_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d0_seed2024.log |
| joint_d1_large_baseline_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_baseline_seed2025.log |
| joint_d1_large_comp1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d1_large_comp1_seed2025.log |
| oracle_d1_large_oracle_d1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d1_seed2025.log |
| oracle_d1_large_oracle_d2_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d2_seed2025.log |
| oracle_d1_large_oracle_d0_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d1_large_oracle_d0_seed2025.log |
| joint_d2_large_baseline_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_baseline_seed2022.log |
| joint_d2_large_comp1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_comp1_seed2022.log |
| oracle_d2_large_oracle_d2_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d2_seed2022.log |
| oracle_d2_large_oracle_d0_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d0_seed2022.log |
| oracle_d2_large_oracle_d1_seed2022 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d1_seed2022.log |
| joint_d2_large_baseline_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_baseline_seed2023.log |
| joint_d2_large_comp1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_comp1_seed2023.log |
| oracle_d2_large_oracle_d2_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d2_seed2023.log |
| oracle_d2_large_oracle_d0_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d0_seed2023.log |
| oracle_d2_large_oracle_d1_seed2023 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d1_seed2023.log |
| joint_d2_large_baseline_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_baseline_seed2024.log |
| joint_d2_large_comp1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_comp1_seed2024.log |
| oracle_d2_large_oracle_d2_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d2_seed2024.log |
| oracle_d2_large_oracle_d0_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d0_seed2024.log |
| oracle_d2_large_oracle_d1_seed2024 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d1_seed2024.log |
| joint_d2_large_baseline_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_baseline_seed2025.log |
| joint_d2_large_comp1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_joint_d2_large_comp1_seed2025.log |
| oracle_d2_large_oracle_d2_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d2_seed2025.log |
| oracle_d2_large_oracle_d0_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d0_seed2025.log |
| oracle_d2_large_oracle_d1_seed2025 | OK | metrics_summary.csv | experiments/logs/p0_1_oracle_d2_large_oracle_d1_seed2025.log |

## 3. Raw metrics

### 3.1 Joint runs (24 runs)

Each row: 6 test AUCs (d0t0 d0t1 d1t0 d1t1 d2t0 d2t1) at best_epoch, plus best_val_auc and best_epoch.

| setting | method | seed | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | best_epoch | best_val_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| d0_large | baseline | 2022 | 0.744805 | 0.760342 | 0.715168 | 0.759552 | 0.667817 | 0.709900 | 5 | 0.716812 |
| d0_large | baseline | 2023 | 0.742450 | 0.752645 | 0.729762 | 0.750325 | 0.680681 | 0.691520 | 5 | 0.718517 |
| d0_large | baseline | 2024 | 0.743929 | 0.752081 | 0.727216 | 0.740114 | 0.670028 | 0.684037 | 6 | 0.719863 |
| d0_large | baseline | 2025 | 0.737954 | 0.753651 | 0.730519 | 0.756387 | 0.677076 | 0.695968 | 5 | 0.721704 |
| d0_large | comp1 | 2022 | 0.752109 | 0.766892 | 0.733562 | 0.756295 | 0.685938 | 0.701321 | 5 | 0.727913 |
| d0_large | comp1 | 2023 | 0.746009 | 0.760200 | 0.734325 | 0.753279 | 0.687966 | 0.698825 | 6 | 0.724639 |
| d0_large | comp1 | 2024 | 0.753419 | 0.763851 | 0.734786 | 0.752203 | 0.681045 | 0.700727 | 7 | 0.728874 |
| d0_large | comp1 | 2025 | 0.748254 | 0.753689 | 0.735436 | 0.751116 | 0.680037 | 0.697726 | 9 | 0.729329 |
| d1_large | baseline | 2022 | 0.680397 | 0.692427 | 0.736124 | 0.754247 | 0.679137 | 0.696439 | 10 | 0.708076 |
| d1_large | baseline | 2023 | 0.704687 | 0.700349 | 0.734728 | 0.762332 | 0.700723 | 0.719340 | 4 | 0.713998 |
| d1_large | baseline | 2024 | 0.689675 | 0.715012 | 0.744734 | 0.767010 | 0.679503 | 0.698190 | 9 | 0.716466 |
| d1_large | baseline | 2025 | 0.710226 | 0.716606 | 0.736400 | 0.752812 | 0.685774 | 0.705327 | 8 | 0.714746 |
| d1_large | comp1 | 2022 | 0.716135 | 0.735508 | 0.754369 | 0.776698 | 0.702839 | 0.727588 | 10 | 0.729606 |
| d1_large | comp1 | 2023 | 0.718923 | 0.728640 | 0.750012 | 0.773408 | 0.712431 | 0.736244 | 4 | 0.728636 |
| d1_large | comp1 | 2024 | 0.716839 | 0.733946 | 0.755866 | 0.781193 | 0.695390 | 0.718077 | 8 | 0.733783 |
| d1_large | comp1 | 2025 | 0.723033 | 0.735340 | 0.748689 | 0.774464 | 0.701278 | 0.729412 | 9 | 0.727815 |
| d2_large | baseline | 2022 | 0.724800 | 0.739181 | 0.689988 | 0.723038 | 0.699584 | 0.707858 | 7 | 0.714147 |
| d2_large | baseline | 2023 | 0.722816 | 0.733242 | 0.709592 | 0.728265 | 0.702806 | 0.713985 | 5 | 0.714562 |
| d2_large | baseline | 2024 | 0.712105 | 0.727859 | 0.688021 | 0.714728 | 0.705571 | 0.727365 | 10 | 0.712735 |
| d2_large | baseline | 2025 | 0.722838 | 0.735542 | 0.705428 | 0.731413 | 0.703678 | 0.728359 | 7 | 0.718395 |
| d2_large | comp1 | 2022 | 0.733294 | 0.748323 | 0.723888 | 0.749299 | 0.717826 | 0.734498 | 7 | 0.729776 |
| d2_large | comp1 | 2023 | 0.738126 | 0.753693 | 0.723510 | 0.742229 | 0.722881 | 0.739083 | 7 | 0.728601 |
| d2_large | comp1 | 2024 | 0.737152 | 0.752705 | 0.720068 | 0.739627 | 0.724294 | 0.744391 | 9 | 0.732595 |
| d2_large | comp1 | 2025 | 0.734999 | 0.746929 | 0.718087 | 0.731852 | 0.717119 | 0.737570 | 6 | 0.728609 |

### 3.2 Oracle runs (36 runs)

Each row: 2 test AUCs (only the role cells are filled; other 4 cells are empty per spec).

| setting | oracle_domain | seed | cell1 | cell2 | best_epoch | best_val_auc |
|---|---|---|---|---|---|---|
| d0_large | oracle_d0 | 2022 | 0.750812 | 0.765480 | 26 | 0.747680 |
| d0_large | oracle_d0 | 2023 | 0.751578 | 0.762610 | 20 | 0.749300 |
| d0_large | oracle_d0 | 2024 | 0.747599 | 0.757183 | 10 | 0.748298 |
| d0_large | oracle_d0 | 2025 | 0.751081 | 0.763873 | 8 | 0.749503 |
| d0_large | oracle_d1 | 2022 | 0.706154 | 0.729746 | 13 | 0.703626 |
| d0_large | oracle_d1 | 2023 | 0.695967 | 0.723419 | 15 | 0.694361 |
| d0_large | oracle_d1 | 2024 | 0.698988 | 0.711619 | 31 | 0.702684 |
| d0_large | oracle_d1 | 2025 | 0.706204 | 0.735206 | 13 | 0.701590 |
| d0_large | oracle_d2 | 2022 | 0.637804 | 0.652942 | 13 | 0.658153 |
| d0_large | oracle_d2 | 2023 | 0.634457 | 0.636952 | 11 | 0.654765 |
| d0_large | oracle_d2 | 2024 | 0.640219 | 0.644213 | 6 | 0.657019 |
| d0_large | oracle_d2 | 2025 | 0.634128 | 0.652421 | 6 | 0.658630 |
| d1_large | oracle_d0 | 2022 | 0.636472 | 0.662045 | 13 | 0.671383 |
| d1_large | oracle_d0 | 2023 | 0.656473 | 0.658416 | 5 | 0.670251 |
| d1_large | oracle_d0 | 2024 | 0.638067 | 0.667203 | 15 | 0.663175 |
| d1_large | oracle_d0 | 2025 | 0.640806 | 0.658699 | 10 | 0.662840 |
| d1_large | oracle_d1 | 2022 | 0.752541 | 0.773333 | 10 | 0.739886 |
| d1_large | oracle_d1 | 2023 | 0.751134 | 0.774599 | 13 | 0.731556 |
| d1_large | oracle_d1 | 2024 | 0.753325 | 0.774757 | 9 | 0.737806 |
| d1_large | oracle_d1 | 2025 | 0.752387 | 0.771814 | 9 | 0.741349 |
| d1_large | oracle_d2 | 2022 | 0.682215 | 0.701624 | 12 | 0.702910 |
| d1_large | oracle_d2 | 2023 | 0.677873 | 0.706023 | 23 | 0.700562 |
| d1_large | oracle_d2 | 2024 | 0.669775 | 0.687019 | 9 | 0.703211 |
| d1_large | oracle_d2 | 2025 | 0.668455 | 0.700159 | 11 | 0.705257 |
| d2_large | oracle_d0 | 2022 | 0.694421 | 0.713797 | 13 | 0.715668 |
| d2_large | oracle_d0 | 2023 | 0.702204 | 0.717521 | 4 | 0.712556 |
| d2_large | oracle_d0 | 2024 | 0.684251 | 0.704708 | 7 | 0.704760 |
| d2_large | oracle_d0 | 2025 | 0.697780 | 0.710745 | 12 | 0.719825 |
| d2_large | oracle_d1 | 2022 | 0.654050 | 0.671056 | 11 | 0.660189 |
| d2_large | oracle_d1 | 2023 | 0.660158 | 0.673999 | 3 | 0.644423 |
| d2_large | oracle_d1 | 2024 | 0.638062 | 0.655520 | 28 | 0.643312 |
| d2_large | oracle_d1 | 2025 | 0.640342 | 0.663728 | 13 | 0.643341 |
| d2_large | oracle_d2 | 2022 | 0.719238 | 0.739353 | 9 | 0.738843 |
| d2_large | oracle_d2 | 2023 | 0.726436 | 0.746738 | 11 | 0.739139 |
| d2_large | oracle_d2 | 2024 | 0.723728 | 0.743415 | 10 | 0.741820 |
| d2_large | oracle_d2 | 2025 | 0.720659 | 0.741145 | 6 | 0.736681 |

## 4. Mean/std

### 4.1 Joint per (setting, method, task): mean/std/min/max across 4 seeds

| setting | method | task | mean | std | min | max |
|---|---|---|---|---|---|---|
| d0_large | baseline | d0t0 | 0.742285 | 0.003046 | 0.737954 | 0.744805 |
| d0_large | baseline | d0t1 | 0.754680 | 0.003830 | 0.752081 | 0.760342 |
| d0_large | baseline | d1t0 | 0.725666 | 0.007140 | 0.715168 | 0.730519 |
| d0_large | baseline | d1t1 | 0.751595 | 0.008558 | 0.740114 | 0.759552 |
| d0_large | baseline | d2t0 | 0.673901 | 0.006002 | 0.667817 | 0.680681 |
| d0_large | baseline | d2t1 | 0.695356 | 0.010874 | 0.684037 | 0.709900 |
| d0_large | comp1 | d0t0 | 0.749948 | 0.003421 | 0.746009 | 0.753419 |
| d0_large | comp1 | d0t1 | 0.761158 | 0.005681 | 0.753689 | 0.766892 |
| d0_large | comp1 | d1t0 | 0.734527 | 0.000789 | 0.733562 | 0.735436 |
| d0_large | comp1 | d1t1 | 0.753223 | 0.002230 | 0.751116 | 0.756295 |
| d0_large | comp1 | d2t0 | 0.683747 | 0.003815 | 0.680037 | 0.687966 |
| d0_large | comp1 | d2t1 | 0.699650 | 0.001667 | 0.697726 | 0.701321 |
| d1_large | baseline | d0t0 | 0.696246 | 0.013676 | 0.680397 | 0.710226 |
| d1_large | baseline | d0t1 | 0.706098 | 0.011688 | 0.692427 | 0.716606 |
| d1_large | baseline | d1t0 | 0.737997 | 0.004551 | 0.734728 | 0.744734 |
| d1_large | baseline | d1t1 | 0.759100 | 0.006736 | 0.752812 | 0.767010 |
| d1_large | baseline | d2t0 | 0.686284 | 0.010096 | 0.679137 | 0.700723 |
| d1_large | baseline | d2t1 | 0.704824 | 0.010413 | 0.696439 | 0.719340 |
| d1_large | comp1 | d0t0 | 0.718732 | 0.003102 | 0.716135 | 0.723033 |
| d1_large | comp1 | d0t1 | 0.733359 | 0.003223 | 0.728640 | 0.735508 |
| d1_large | comp1 | d1t0 | 0.752234 | 0.003428 | 0.748689 | 0.755866 |
| d1_large | comp1 | d1t1 | 0.776441 | 0.003452 | 0.773408 | 0.781193 |
| d1_large | comp1 | d2t0 | 0.702985 | 0.007067 | 0.695390 | 0.712431 |
| d1_large | comp1 | d2t1 | 0.727830 | 0.007494 | 0.718077 | 0.736244 |
| d2_large | baseline | d0t0 | 0.720640 | 0.005765 | 0.712105 | 0.724800 |
| d2_large | baseline | d0t1 | 0.733956 | 0.004743 | 0.727859 | 0.739181 |
| d2_large | baseline | d1t0 | 0.698257 | 0.010848 | 0.688021 | 0.709592 |
| d2_large | baseline | d1t1 | 0.724361 | 0.007292 | 0.714728 | 0.731413 |
| d2_large | baseline | d2t0 | 0.702910 | 0.002500 | 0.699584 | 0.705571 |
| d2_large | baseline | d2t1 | 0.719392 | 0.010104 | 0.707858 | 0.728359 |
| d2_large | comp1 | d0t0 | 0.735893 | 0.002170 | 0.733294 | 0.738126 |
| d2_large | comp1 | d0t1 | 0.750412 | 0.003292 | 0.746929 | 0.753693 |
| d2_large | comp1 | d1t0 | 0.721388 | 0.002792 | 0.718087 | 0.723888 |
| d2_large | comp1 | d1t1 | 0.740752 | 0.007204 | 0.731852 | 0.749299 |
| d2_large | comp1 | d2t0 | 0.720530 | 0.003589 | 0.717119 | 0.724294 |
| d2_large | comp1 | d2t1 | 0.738885 | 0.004136 | 0.734498 | 0.744391 |

### 4.2 Oracle per (setting, oracle_domain, cell): mean/std/min/max across 4 seeds

| setting | oracle_domain | cell | mean | std | min | max |
|---|---|---|---|---|---|---|
| d0_large | oracle_d0 | d0t0 | 0.750267 | 0.001807 | 0.747599 | 0.751578 |
| d0_large | oracle_d0 | d0t1 | 0.762287 | 0.003599 | 0.757183 | 0.765480 |
| d0_large | oracle_d1 | d1t0 | 0.701828 | 0.005173 | 0.695967 | 0.706204 |
| d0_large | oracle_d1 | d1t1 | 0.724997 | 0.010136 | 0.711619 | 0.735206 |
| d0_large | oracle_d2 | d2t0 | 0.636652 | 0.002901 | 0.634128 | 0.640219 |
| d0_large | oracle_d2 | d2t1 | 0.646632 | 0.007591 | 0.636952 | 0.652942 |
| d1_large | oracle_d0 | d0t0 | 0.642954 | 0.009188 | 0.636472 | 0.656473 |
| d1_large | oracle_d0 | d0t1 | 0.661591 | 0.004088 | 0.658416 | 0.667203 |
| d1_large | oracle_d1 | d1t0 | 0.752347 | 0.000907 | 0.751134 | 0.753325 |
| d1_large | oracle_d1 | d1t1 | 0.773626 | 0.001366 | 0.771814 | 0.774757 |
| d1_large | oracle_d2 | d2t0 | 0.674579 | 0.006576 | 0.668455 | 0.682215 |
| d1_large | oracle_d2 | d2t1 | 0.698706 | 0.008180 | 0.687019 | 0.706023 |
| d2_large | oracle_d0 | d0t0 | 0.694664 | 0.007639 | 0.684251 | 0.702204 |
| d2_large | oracle_d0 | d0t1 | 0.711693 | 0.005419 | 0.704708 | 0.717521 |
| d2_large | oracle_d1 | d1t0 | 0.648153 | 0.010673 | 0.638062 | 0.660158 |
| d2_large | oracle_d1 | d1t1 | 0.666076 | 0.008257 | 0.655520 | 0.673999 |
| d2_large | oracle_d2 | d2t0 | 0.722515 | 0.003216 | 0.719238 | 0.726436 |
| d2_large | oracle_d2 | d2t1 | 0.742663 | 0.003185 | 0.739353 | 0.746738 |

### 4.3 Delta_transfer = joint_baseline (6-cell mean) - oracle (2-cell mean) per seed

| setting | role | seed | joint_6cell_mean | oracle_2cell_mean | delta |
|---|---|---|---|---|---|
| d0_large | oracle_d0 | 2022 | 0.726264 | 0.758146 | -0.031882 |
| d0_large | oracle_d0 | 2023 | 0.724564 | 0.757094 | -0.032530 |
| d0_large | oracle_d0 | 2024 | 0.719568 | 0.752391 | -0.032824 |
| d0_large | oracle_d0 | 2025 | 0.725259 | 0.757477 | -0.032218 |
| d0_large | oracle_d1 | 2022 | 0.726264 | 0.717950 | +0.008314 |
| d0_large | oracle_d1 | 2023 | 0.724564 | 0.709693 | +0.014871 |
| d0_large | oracle_d1 | 2024 | 0.719568 | 0.705304 | +0.014264 |
| d0_large | oracle_d1 | 2025 | 0.725259 | 0.720705 | +0.004554 |
| d0_large | oracle_d2 | 2022 | 0.726264 | 0.645373 | +0.080891 |
| d0_large | oracle_d2 | 2023 | 0.724564 | 0.635705 | +0.088859 |
| d0_large | oracle_d2 | 2024 | 0.719568 | 0.642216 | +0.077352 |
| d0_large | oracle_d2 | 2025 | 0.725259 | 0.643274 | +0.081985 |
| d1_large | oracle_d0 | 2022 | 0.706462 | 0.649258 | +0.057203 |
| d1_large | oracle_d0 | 2023 | 0.720360 | 0.657444 | +0.062915 |
| d1_large | oracle_d0 | 2024 | 0.715687 | 0.652635 | +0.063052 |
| d1_large | oracle_d0 | 2025 | 0.717858 | 0.649752 | +0.068105 |
| d1_large | oracle_d1 | 2022 | 0.706462 | 0.762937 | -0.056475 |
| d1_large | oracle_d1 | 2023 | 0.720360 | 0.762867 | -0.042507 |
| d1_large | oracle_d1 | 2024 | 0.715687 | 0.764041 | -0.048354 |
| d1_large | oracle_d1 | 2025 | 0.717858 | 0.762101 | -0.044243 |
| d1_large | oracle_d2 | 2022 | 0.706462 | 0.691920 | +0.014542 |
| d1_large | oracle_d2 | 2023 | 0.720360 | 0.691948 | +0.028412 |
| d1_large | oracle_d2 | 2024 | 0.715687 | 0.678397 | +0.037290 |
| d1_large | oracle_d2 | 2025 | 0.717858 | 0.684307 | +0.033551 |
| d2_large | oracle_d0 | 2022 | 0.714075 | 0.704109 | +0.009966 |
| d2_large | oracle_d0 | 2023 | 0.718451 | 0.709863 | +0.008588 |
| d2_large | oracle_d0 | 2024 | 0.712608 | 0.694480 | +0.018129 |
| d2_large | oracle_d0 | 2025 | 0.721210 | 0.704263 | +0.016947 |
| d2_large | oracle_d1 | 2022 | 0.714075 | 0.662553 | +0.051522 |
| d2_large | oracle_d1 | 2023 | 0.718451 | 0.667079 | +0.051372 |
| d2_large | oracle_d1 | 2024 | 0.712608 | 0.646791 | +0.065817 |
| d2_large | oracle_d1 | 2025 | 0.721210 | 0.652035 | +0.069175 |
| d2_large | oracle_d2 | 2022 | 0.714075 | 0.729296 | -0.015221 |
| d2_large | oracle_d2 | 2023 | 0.718451 | 0.736587 | -0.018136 |
| d2_large | oracle_d2 | 2024 | 0.712608 | 0.733572 | -0.020963 |
| d2_large | oracle_d2 | 2025 | 0.721210 | 0.730902 | -0.009692 |

### 4.4 Quick-look: joint 6-cell mean per (setting, method), averaged across 4 seeds

| setting | method | joint_6cell_mean |
|---|---|---|
| d0_large | baseline | 0.723914 |
| d0_large | comp1 | 0.730375 |
| d1_large | baseline | 0.715092 |
| d1_large | comp1 | 0.735263 |
| d2_large | baseline | 0.716586 |
| d2_large | comp1 | 0.734643 |

## 5. Failed runs

No failed runs. All 60 unique tasks reached status=OK.

### Operational notes (dual orchestrator)

- During launch, two orchestrator trees ran concurrently (root PIDs 27484->31484 and 8340->8148).
- Tree A was killed at ~00:25 (PIDs 27484, 31484, 6312, 26656, plus orphans 8132, 21688).
- Tree B (8340, 8148) was kept and ran the remaining tasks to completion.
- Master CSV P0_1_raw.csv originally had each task written twice (once per orchestrator); same AUC values, slightly different fit_sec due to resource contention.
- P0_1_raw_dedup.csv: first occurrence per (kind, setting, method_or_domain, seed) kept; 55 duplicates removed; 60 unique OK rows remain.
- All 55 duplicated tasks had identical per_dt_test_auc strings (same seed + same data subset -> deterministic). fit_sec differs by <=1s in all cases.
- Per-run stdout logs are at experiments/logs/p0_1_<name>.log (one per unique task; 60 logs total, plus orchestrator_main.log).

## 6. Missing metrics

No missing metrics. Every run directory contains config.json, metrics_epoch.csv, metrics_summary.csv.
Per-run stdout is in experiments/logs/p0_1_<name>.log (not inside the run directory, by orchestrator design).
