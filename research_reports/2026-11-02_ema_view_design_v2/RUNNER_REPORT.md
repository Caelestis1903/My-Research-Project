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

## Frozen cost forecast

Forecast is expected worker seconds for one 50-epoch trajectory, as recorded by Stage0. Combined-family values sum the two dataset trajectories.

| Family | MovieLens seconds | KuaiRand seconds | Combined seconds |
|---|---:|---:|---:|
| A0 | 1,215.19 | 1,186.21 | 2,401.39 |
| A1 | 853.35 | 1,749.84 | 2,603.19 |
| J0 | 462.67 | 1,383.08 | 1,845.75 |
| J1 | 377.53 | 1,503.04 | 1,880.58 |
| F0 | 365.56 | 1,524.92 | 1,890.48 |
| F1 | 840.99 | 44,473.23 | 45,314.22 |
| **All families** | **4,114.73** | **51,820.87** | **55,935.60** |

For the currently described S1 development set of seeds 2022 and 2024, the frozen forecast is 111,871.20 summed worker seconds. A 1.20 multiplier gives a provisional reservation/cap of 134,245.44 summed worker seconds. These are arithmetic projections from the recorded Stage0 forecast; no S1 jobs have been dispatched or run.

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
