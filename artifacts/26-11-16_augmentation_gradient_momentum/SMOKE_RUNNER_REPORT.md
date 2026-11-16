# SMOKE RUNNER REPORT

## 1. Experiment ID

26-11-16_augmentation_gradient_momentum

## 2. Git/worktree state

Commit: dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11. Detached HEAD. Worktree was dirty before execution; existing edits were preserved.

## 3. Review identity and config

Reviewer verdict: PASS WITH FIXES; blocking issues: [].
Source SHA256: d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6
Design SHA256: 67fdebf500e80611de2f0600e0847bf0697774bcb0fa107571154b1524fff338
Smoke: CUDA:1, seed 2022, batch 40,960, FP32, MovieLens and KuaiRand; five branches per dataset; one step each. Formal manifest seeds: 2022, 2023, 2024, 2025, 2026. Formal training was not started.

## 4. Seeds

Smoke seed: 2022. No formal seed trajectories were run. The prepared manifest lists: 2022, 2023, 2024, 2025, 2026.

## 5. Run status

Prepare: exit 0, status PREPARED. Smoke: exit 0, runner status PASS. All receipt checks: True.

## 6. Raw branch verification

| Dataset | Branch | Rows | Opt/EMA updates | Seconds | Peak allocated / reserved bytes | Shuffle relative error |
|---|---|---:|---:|---:|---:|---:|
| movielens | SOURCE-CLEAN | 40960 | 1/1 | 0.319105 | 961808384 / 1044381696 | — |
| movielens | GAMMA0 | 40960 | 1/1 | 0.122844 | 970852352 / 1052770304 | — |
| movielens | TV-GRM-010 | 40960 | 1/1 | 0.463537 | 2500655104 / 2749366272 | — |
| movielens | TV-GRM-SHUFFLE-010 | 40960 | 1/1 | 0.424770 | 2500655104 / 2749366272 | 4.111813035706533e-09 |
| movielens | TV-GRM-INSTANT-010 | 40960 | 1/1 | 0.374043 | 2500655104 / 2749366272 | — |
| kuairand | SOURCE-CLEAN | 40960 | 1/1 | 0.145784 | 1065205760 / 1321205760 | — |
| kuairand | GAMMA0 | 40960 | 1/1 | 0.144618 | 1065205760 / 1321205760 | — |
| kuairand | TV-GRM-010 | 40960 | 1/1 | 0.440872 | 2905210880 / 3336568832 | — |
| kuairand | TV-GRM-SHUFFLE-010 | 40960 | 1/1 | 0.446044 | 2905210880 / 3336568832 | 3.111310442062587e-08 |
| kuairand | TV-GRM-INSTANT-010 | 40960 | 1/1 | 0.432217 | 2905210880 / 3336568832 | — |

SOURCE-CLEAN and GAMMA0 state hashes (model, optimizer, EMA, RNG, loader) are exactly equal for each dataset. All serialized numeric report values are finite. SHUFFLE relative norm errors are recorded in the JSON receipt and are within 1e-5. These are execution checks only.

## 7. Log locations

- Full smoke report: artifacts/26-11-16_augmentation_gradient_momentum/attempt1/SMOKE_REPORT.json
- Prepare stdout/stderr: artifacts/26-11-16_augmentation_gradient_momentum/runner_logs/prepare.stdout.log, prepare.stderr.log
- Smoke stdout/stderr: artifacts/26-11-16_augmentation_gradient_momentum/runner_logs/smoke.stdout.log, smoke.stderr.log

## 8. Runtime errors

None. Stderr contains a PyTorch TF32 API deprecation warning; see preserved smoke stderr.

## 9. Missing runs

None within the requested prepare/smoke scope. Formal training, epoch training, and Test evaluation were not run. No model/checkpoint files were written.
