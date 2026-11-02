# EMA02V2 Runner Report

## Experiment and execution identity

- Experiment: EMA02V2, Stage S0 only
- Result: **PASS** (`STAGE0_REPORT.json`), process exit code `0`
- Execution commit: `e5d74ccd761f1cafcc3a9c13f0d880eb2130d695`
- Parent commit: `fb74c40cad9c22ce3abdfb8348da7e98d726a61a`
- Approved source hash: `e524699526ae6e8eaf38950a0255eb359c7ca7705a64dab4f60bf9aa1ea93307` (40 files)
- Interpreter: Python 3.13.5, PyTorch 2.9.0+cu126, CUDA runtime 12.6
- Device: `cuda:0`, NVIDIA GeForce RTX 3090
- Command:
  `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.ema_view_design_20261102_v2.run stage0 --attempt /home/xzf/Code/phase0-3/artifacts/ema_view_design_20261102_v2/attempt1 --review /home/xzf/Code/phase0-3/research_reports/2026-11-02_ema_view_design_v2/INDEPENDENT_REVIEW_RECEIPT.json --device cuda:0`
- Started: `2026-11-02T08:23:28.597466+00:00`; report completed: `2026-11-02T08:26:28.730597+00:00`
- Wall duration: 180.13 seconds; summed physical worker time: 175.712 seconds

## Launch conditions

Preflight snapshot: GPU0 had 8,143/24,576 MiB in use, power 202.86/350 W (58%), utilization 99%; GPU1 had 18,189/24,576 MiB in use, power 294.29/350 W (84%), utilization 98%. CUDA Stage0 ran on GPU0. External GPU processes were observed and left untouched.

## Configuration and raw Stage0 results

Stage0 ran A0, A1, J0, J1, F0, and F1 on MovieLens and KuaiRand, two full training batches per configuration at batch size 40,960. The result contains 12 dataset/arm measurements and 13 physical run records (including invocation overhead). Every arm recorded two optimizer steps and two EMA updates. No test metrics were accessed.

| Dataset | Train rows | Validation rows | Online validation forwards / rows / seconds | EMA validation forwards / rows / seconds | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 200,042 | 100,021 | 3 / 100,021 / 0.29208 | 3 / 100,021 / 0.26371 | 0.910–0.920 | 0.988–1.029 |
| KuaiRand | 2,006,132 | 250,767 | 7 / 250,767 / 0.91351 | 7 / 250,767 / 0.82729 | 1.005–1.017 | 1.176–1.340 |

Probe probability maximum absolute error was `0.0` for all arms. FP32 directional relative errors at epsilon `1e-3` were `5.75335e-05` (MovieLens) and `0.00382691` (KuaiRand); at epsilon `1e-4` they were `0.00974589` and `0.254784`. All arms reported no teacher parameter gradients and unchanged probe RNG/state. Maximum Stage0 reserved memory was 1,438,646,272 bytes (1.340 GiB).

## Initial GPU-SVD cost forecast (superseded)

This is the original forecast for one 50-epoch trajectory, before the reviewed CPU SVD placement and repeated F1 Stage0. Combined-family values sum the two dataset trajectories. The original scalar record is retained as `cost_forecast_GPU_SVD_INITIAL.json`.

| Family | MovieLens seconds | KuaiRand seconds | Combined seconds |
|---|---:|---:|---:|
| A0 | 1,215.19 | 1,186.21 | 2,401.39 |
| A1 | 853.35 | 1,749.84 | 2,603.19 |
| J0 | 462.67 | 1,383.08 | 1,845.75 |
| J1 | 377.53 | 1,503.04 | 1,880.58 |
| F0 | 365.56 | 1,524.92 | 1,890.48 |
| F1 | 840.99 | 44,473.23 | 45,314.22 |
| **All families** | **4,114.73** | **51,820.87** | **55,935.60** |

The initial, superseded projection for S1 seeds 2022 and 2024 was 111,871.20 summed worker seconds, with a 1.20 multiplier of 134,245.44 seconds. No S1 jobs were dispatched under that forecast.

## Status, errors, and artifacts

- Runtime errors: none; exit code `0`.
- Stderr warnings: PyTorch TF32 setting deprecation warning; scalar conversion warning in the Stage0 probe check.
- Seeds: no formal seeded S1 trajectories were run in Stage0.
- Missing runs: all S1 runs remain pending; no S1 dispatch was performed.
- Guardian/model state handling followed the approved in-memory/BytesIO path; no checkpoint file was written. Guardian remains running for the parent-controlled next step.
- Full raw measurements and per-run records: [STAGE0_REPORT.json](../../artifacts/ema_view_design_20261102_v2/attempt1/STAGE0_REPORT.json)
- Stdout: [stage0.stdout.log](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/stage0.stdout.log) (SHA-256 `a62484dc9bc4adb40c7e097620dc1eb847fb287d971c916c828f2aa50a43571b`)
- Stderr: [stage0.stderr.log](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/stage0.stderr.log) (SHA-256 `0e4c7c132a862e0f86318bcf20369af7d1014ddd684cc4221b9e0190e5575d80`)
- Invocation provenance: [STAGE0_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/STAGE0_INVOCATION.json)

Formal S1 execution is waiting for explicit parent authorization after the pending reviewed execution update.


## Reviewed CPU-SVD placement: repeated F1 Stage0

- Incremental review: PASS, no blockers; approved source hash `f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`.
- Execution commit: `e3ca50380fff0535d16b73888cc6e462391021ca`.
- Command repeated Stage0 for `--arms F1` only, using batch size 40,960 on both datasets. Exit code `0`; Stage0 report status remains PASS. Test metrics were not accessed.
- Device: cuda:0, RTX 3090. Launch snapshot: 20,864/24,576 MiB used (84.90%), 184.33/350 W (52.67%), utilization 100%; 3,712 MiB free. GPU1 was at 18,189 MiB, 298.14/350 W (85.18%) and was not eligible. External processes were left untouched.
- Wall duration: 76.138 seconds. Incremental charged physical time: 75.385 seconds. Cumulative physical records now total 304.421 seconds over 17 records, including the 53.324-second SVD placement calibration.

| Dataset / arm | Two measured step times (seconds) | Peak allocated (bytes) | Peak reserved (bytes) | Updated expected worker seconds per trajectory |
|---|---:|---:|---:|---:|
| MovieLens / F1 | 2.788385, 0.307290 | 979,930,624 | 1,094,713,344 | 2,870.399 |
| KuaiRand / F1 | 0.147384, 0.160290 | 1,085,401,088 | 1,283,457,024 | 553.319 |

Both repeated entries recorded two optimizer steps and two EMA updates. MovieLens diagnostics: step1 fallback/field rows 7,957/7,957, capped rows 0; step2 0/8,011, capped rows 0. KuaiRand diagnostics: step1 fallback/field rows 1,843/1,843, capped rows 637; step2 0/1,836, capped rows 201. Full diagnostics remain in `STAGE0_REPORT.json`.

Updated forecast values, in expected worker seconds per one 50-epoch trajectory:

| Family | MovieLens | KuaiRand | Combined |
|---|---:|---:|---:|
| A0 | 1,215.19 | 1,186.21 | 2,401.39 |
| A1 | 853.35 | 1,749.84 | 2,603.19 |
| J0 | 462.67 | 1,383.08 | 1,845.75 |
| J1 | 377.53 | 1,503.04 | 1,880.58 |
| F0 | 365.56 | 1,524.92 | 1,890.48 |
| F1 | 2,870.40 | 553.32 | 3,423.72 |
| **All families** | **6,144.69** | **7,900.41** | **14,045.10** |

For S1 seeds 2022 and 2024, this revised forecast projects 28,090.20 summed worker seconds; the unchanged 1.20 cap is 33,708.24 summed worker seconds. This is the updated pre-dispatch projection; no S1 jobs have been run or dispatched.

The repeated-run provenance is [STAGE0_F1_SVD_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/STAGE0_F1_SVD_INVOCATION.json). Separate stdout/stderr logs are [stage0_f1_svd.stdout.log](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/stage0_f1_svd.stdout.log) and [stage0_f1_svd.stderr.log](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/stage0_f1_svd.stderr.log). The updated combined forecast is [cost_forecast.json](../../artifacts/ema_view_design_20261102_v2/attempt1/cost_forecast.json); the original is retained at [cost_forecast_GPU_SVD_INITIAL.json](../../artifacts/ema_view_design_20261102_v2/attempt1/cost_forecast_GPU_SVD_INITIAL.json).

S1 execution is waiting for explicit parent authorization and a frozen dispatch reservation based on this updated forecast.


## S1 execution report — 24 approved trajectories

- Dispatch: `APPROVED_S1_DISPATCH.json`, seeds 2022 and 2024; six arms (`A0`, `A1`, `J0`, `J1`, `F0`, `F1`) on MovieLens and KuaiRand.
- Training: FP32, batch size 40,960, 50 epochs per trajectory.
- Execution commit: `e3ca50380fff0535d16b73888cc6e462391021ca`; source hash `f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`.
- Result: **24/24 complete**, all at attempt 1; 0 retries, 0 failures. Scheduler exit code 0.
- S1 trajectory physical worker time: 18,530.417 seconds. Cumulative physical time including Stage0 and SVD placement calibration: 18,834.838 seconds. Dispatch reservation: 28,090.202 seconds; 1.20 cap: 33,708.242 seconds; final budget state `WITHIN_CAP`.
- Test data/metrics accessed: false. Guardian remains active with selected and E50 model states in RAM.
- Exact scheduler invocation and preflight conditions: [S1_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/S1_INVOCATION.json). Scheduler stdout/stderr files were empty; GPU/PID launch records are in `attempt1/launch_ledger.jsonl`, and per-worker logs and epoch records are under `attempt1/runs/` and `attempt1/launch_logs/`.

The following values are raw validation metrics. Each cell gives `macro_auc / macro_logloss`. “Selected” is the per-estimator validation-selected epoch recorded by the worker; “E50” is the final epoch's validation. Domain, task, and domain-task-cell AUC/loss/counts for both selected and E50 online/fixed-EMA estimators are recorded in [S1_RAW_METRICS.json](../../artifacts/ema_view_design_20261102_v2/attempt1/S1_RAW_METRICS.json).

| Dataset | Arm | Seed | Selected epochs (online/EMA) | Selected online AUC/loss | Selected EMA AUC/loss | E50 online AUC/loss | E50 EMA AUC/loss | Worker seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| movielens | A0 | 2022 | 29/26 | 0.756911/0.531181 | 0.760026/0.526644 | 0.752284/0.541741 | 0.756626/0.540346 | 291.16 |
| movielens | A1 | 2022 | 26/20 | 0.756373/0.523903 | 0.759699/0.525383 | 0.753107/0.538594 | 0.756629/0.538387 | 380.84 |
| movielens | J0 | 2022 | 28/23 | 0.755828/0.529419 | 0.759854/0.527952 | 0.753351/0.541715 | 0.756273/0.540067 | 369.91 |
| movielens | J1 | 2022 | 27/23 | 0.756742/0.523640 | 0.760121/0.525796 | 0.752564/0.540400 | 0.755986/0.539964 | 344.37 |
| movielens | F0 | 2022 | 29/20 | 0.756426/0.531447 | 0.759002/0.525914 | 0.751511/0.541029 | 0.755546/0.543346 | 324.78 |
| movielens | F1 | 2022 | 32/27 | 0.756735/0.526069 | 0.759373/0.529399 | 0.753110/0.536993 | 0.756610/0.538882 | 261.69 |
| kuairand | A0 | 2022 | 44/33 | 0.688511/0.431608 | 0.692638/0.430601 | 0.681470/0.432496 | 0.690082/0.429494 | 695.81 |
| kuairand | A1 | 2022 | 32/33 | 0.686676/0.430414 | 0.688373/0.431109 | 0.680214/0.432329 | 0.683700/0.431449 | 557.54 |
| kuairand | J0 | 2022 | 44/44 | 0.684260/0.426476 | 0.683974/0.429180 | 0.677068/0.434233 | 0.682809/0.429108 | 949.28 |
| kuairand | J1 | 2022 | 26/23 | 0.679610/0.432892 | 0.680255/0.432971 | 0.671229/0.435432 | 0.677835/0.431016 | 555.82 |
| kuairand | F0 | 2022 | 26/32 | 0.690486/0.428427 | 0.693607/0.428523 | 0.682084/0.432076 | 0.683755/0.435970 | 1307.11 |
| kuairand | F1 | 2022 | 33/50 | 0.692313/0.428953 | 0.690961/0.428581 | 0.690172/0.427363 | 0.690961/0.428581 | 1604.22 |
| movielens | A0 | 2024 | 24/20 | 0.756061/0.526846 | 0.759673/0.525143 | 0.751004/0.544616 | 0.754149/0.542598 | 131.83 |
| movielens | A1 | 2024 | 22/22 | 0.756905/0.523505 | 0.759831/0.524990 | 0.750985/0.538076 | 0.755974/0.542625 | 627.13 |
| movielens | J0 | 2024 | 22/20 | 0.756615/0.526953 | 0.759818/0.524457 | 0.749626/0.545378 | 0.754924/0.543786 | 691.47 |
| movielens | J1 | 2024 | 25/20 | 0.757015/0.524798 | 0.759784/0.524461 | 0.750970/0.543166 | 0.755510/0.540914 | 763.54 |
| movielens | F0 | 2024 | 22/20 | 0.756117/0.525418 | 0.759114/0.525029 | 0.748814/0.552478 | 0.753342/0.547077 | 690.77 |
| movielens | F1 | 2024 | 25/25 | 0.756676/0.525926 | 0.759478/0.524104 | 0.752420/0.539616 | 0.755317/0.541689 | 210.19 |
| kuairand | A0 | 2024 | 40/50 | 0.693477/0.426897 | 0.695994/0.426760 | 0.693399/0.430246 | 0.695994/0.426760 | 1199.65 |
| kuairand | A1 | 2024 | 39/39 | 0.683213/0.437431 | 0.686909/0.427553 | 0.679657/0.430024 | 0.684424/0.431303 | 1485.88 |
| kuairand | J0 | 2024 | 41/44 | 0.683689/0.430263 | 0.684988/0.429223 | 0.676457/0.430644 | 0.683162/0.427407 | 1467.24 |
| kuairand | J1 | 2024 | 40/40 | 0.691242/0.425251 | 0.690557/0.426120 | 0.683743/0.430887 | 0.689049/0.426599 | 1451.39 |
| kuairand | F0 | 2024 | 35/28 | 0.674424/0.429635 | 0.675112/0.430117 | 0.672113/0.431032 | 0.667478/0.431789 | 804.15 |
| kuairand | F1 | 2024 | 42/36 | 0.684735/0.426630 | 0.685249/0.427472 | 0.678744/0.431041 | 0.680282/0.428118 | 1364.67 |

All raw per-epoch validation, loss, counts, and diagnostic records remain in the scalar epoch JSON files. The final S1 invocation report records no missing runs and no runtime errors. No test evaluation was performed by the runner.


## S2B backup-path execution report — 12 approved trajectories

### Stage0 calibration

- S2B Stage0 result: PASS, exit code 0; arms TG/RG/AG on both datasets, two full batches of 40,960 each.
- Device: cuda:1, RTX 3090. Launch preflight had 15/24,576 MiB used and 26.50/350 W; GPU0 was above the 85% memory gate and was not selected. External processes were left untouched.
- Incremental six-arm physical worker time: 47.478 seconds; cumulative Stage0 physical record sum after this calibration: 353.442 seconds.

| Dataset | Arm | Two measured step times (s) | Peak allocated bytes | Peak reserved bytes | Expected worker seconds / trajectory |
|---|---|---:|---:|---:|---:|
| movielens | TG | 0.239240, 0.132279 | 1123684864 | 1197473792 | 274.419 |
| movielens | RG | 0.173411, 0.165220 | 1125459456 | 1245708288 | 207.245 |
| movielens | AG | 0.128850, 0.114167 | 979930624 | 1094713344 | 161.887 |
| kuairand | TG | 0.145420, 0.131346 | 1089159168 | 1388314624 | 493.568 |
| kuairand | RG | 0.184367, 0.195249 | 1087558144 | 1402994688 | 616.425 |
| kuairand | AG | 0.132093, 0.124380 | 1084165632 | 1459617792 | 461.543 |

### S2B dispatch and completion

- Approved dispatch: `APPROVED_S2B_DISPATCH.json`, path choice `backup`; 12 jobs, seeds 2022/2024, TG/RG/AG on MovieLens and KuaiRand.
- Training: FP32, batch size 40,960, 50 epochs per trajectory.
- Execution commit: `e3ca50380fff0535d16b73888cc6e462391021ca`; source hash `f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`.
- Result: **12/12 complete**, all at attempt 1; 0 retries, 0 failures. Scheduler exit code 0.
- S2B forecast reservation: 4,430.175 worker seconds. Combined S1+S2B reservation: 32,520.377 seconds; 1.20 cumulative cap: 39,024.452 seconds.
- S2B trajectory physical worker time: 14,543.898 seconds. Cumulative physical time including prior Stage0/calibration and S1: 33,427.756 seconds; budget status `WITHIN_CAP`.
- Test data/metrics accessed: false. Guardian remains active; S1/S2B model state and selected/E50 states remain in RAM.
- Exact command and preflight snapshot: [S2B_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/S2B_INVOCATION.json). Per-run GPU/PID snapshots: `attempt1/launch_ledger.jsonl`; worker epoch records and logs are under `attempt1/runs/` and `attempt1/launch_logs/`.

The following are raw validation metrics; each metric cell is `macro_auc / macro_logloss`. Selected epochs are reported as online/fixed-EMA. “E50” is the final epoch validation. Per-domain, per-task, and domain-task-cell metrics/counts for selected and E50 estimators are in [S2B_RAW_METRICS.json](../../artifacts/ema_view_design_20261102_v2/attempt1/S2B_RAW_METRICS.json).

| Dataset | Arm | Seed | Selected epochs (online/EMA) | Selected online AUC/loss | Selected EMA AUC/loss | E50 online AUC/loss | E50 EMA AUC/loss | Worker seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| movielens | TG | 2022 | 31/23 | 0.756635/0.526419 | 0.758496/0.523668 | 0.754210/0.537201 | 0.755872/0.535748 | 668.01 |
| movielens | RG | 2022 | 15/20 | 0.756108/0.524218 | 0.759458/0.525934 | 0.752218/0.538728 | 0.755334/0.540582 | 729.75 |
| movielens | AG | 2022 | 25/26 | 0.757086/0.524514 | 0.759200/0.524152 | 0.754360/0.535065 | 0.755553/0.538124 | 568.65 |
| kuairand | TG | 2022 | 30/30 | 0.665597/0.447604 | 0.663354/0.445223 | 0.621489/0.445764 | 0.648907/0.444062 | 1738.52 |
| kuairand | RG | 2022 | 49/50 | 0.693847/0.422253 | 0.694402/0.426609 | 0.690075/0.430805 | 0.694402/0.426609 | 1820.75 |
| kuairand | AG | 2022 | 26/26 | 0.673141/0.447188 | 0.669671/0.445338 | 0.658440/0.449973 | 0.655248/0.445284 | 1608.93 |
| movielens | TG | 2024 | 14/20 | 0.755719/0.523467 | 0.758169/0.523180 | 0.752762/0.533089 | 0.755788/0.535879 | 866.64 |
| movielens | RG | 2024 | 21/20 | 0.756207/0.526070 | 0.759629/0.523032 | 0.751138/0.543613 | 0.754576/0.542422 | 868.91 |
| movielens | AG | 2024 | 40/39 | 0.756970/0.529268 | 0.758234/0.528207 | 0.752175/0.539235 | 0.757330/0.533194 | 754.99 |
| kuairand | TG | 2024 | 34/29 | 0.667169/0.445458 | 0.665203/0.442615 | 0.642002/0.449616 | 0.638997/0.442757 | 1692.43 |
| kuairand | RG | 2024 | 31/35 | 0.689377/0.428195 | 0.688057/0.426653 | 0.674912/0.436737 | 0.686062/0.428489 | 1702.88 |
| kuairand | AG | 2024 | 32/28 | 0.659258/0.442178 | 0.661424/0.442650 | 0.631088/0.443971 | 0.655575/0.440830 | 1523.44 |

All S2B scalar epoch validation, loss, count, and diagnostic records are retained in the per-job epoch JSON files. No test evaluation or later-stage jobs were run by the runner.


## Frozen final test evaluation — complete

- Training freeze was created before the first test read. Frozen manifest SHA-256: `5df920da4d5af03824ed7c1a64674834110de0b89b8e954c393688ef122dce0c`; it records 36 completed runs and exactly 144 endpoints (online/fixed-EMA × selected/E50 for each run), with `test_opened=false` in the freeze receipt.
- Final evaluator: reviewed source hash `f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`, execution commit `e3ca50380fff0535d16b73888cc6e462391021ca`; command and device preflight in [FINAL_EVALUATION_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/FINAL_EVALUATION_INVOCATION.json).
- Result: **144/144 endpoint files COMPLETE**, each with metrics; 0 missing, 0 duplicate, 0 evaluation failures, no budget-stop record. Frozen endpoint keys exactly match the 144 persisted endpoint keys.
- Final evaluator exit code: 0. Endpoint physical cost: 93.330 seconds.
- Cumulative physical time: Stage0 353.442 seconds + 36 training trajectories 33,074.314 seconds + final endpoints 93.330 seconds = 33,521.086 seconds, within the cumulative cap of 39,024.452 seconds.
- Raw endpoint metrics and timing records: [final_evaluation.json](../../artifacts/ema_view_design_20261102_v2/attempt1/final_evaluation.json) and [test_metrics.csv](../../artifacts/ema_view_design_20261102_v2/attempt1/test_metrics.csv). The complete ledgers are `trial_ledger.csv`, `run_ledger.csv`, `physical_cost_ledger.csv`, `failure_ledger.csv`, `epoch_metrics.csv`, and `validation_metrics.csv` in the same attempt directory.
- Test metrics were accessed only after the fixed 144-endpoint freeze. No model/configuration was reselected from test results. Model states were handled in RAM; no checkpoint files were written. The guardian remains running after evaluation.


## Guardian shutdown

After verifying all 144 frozen final endpoints were persisted as COMPLETE, the guardian reported 36/36 run states released and zero live workers. The reviewed CLI shutdown returned `{'status': 'SHUTDOWN'}` with exit code 0; PID 4174308 was absent from `/proc` after shutdown. Invocation and output hashes are recorded in [GUARDIAN_SHUTDOWN_INVOCATION.json](../../artifacts/ema_view_design_20261102_v2/attempt1/logs/GUARDIAN_SHUTDOWN_INVOCATION.json). Scalar records, logs, data manifests, and endpoint metrics remain in the attempt directory. No model checkpoint files were written.
