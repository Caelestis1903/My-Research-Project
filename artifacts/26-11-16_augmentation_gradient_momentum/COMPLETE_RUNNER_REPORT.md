# Complete Runner Report

- Experiment ID: `26-11-16_augmentation_gradient_momentum`
- Attempt: `attempt1`
- Export status: `COMPLETE`
- Review: `PASS WITH FIXES`; blocking issues: `0`
- Source SHA256: `d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6`
- Design SHA256: `67fdebf500e80611de2f0600e0847bf0697774bcb0fa107571154b1524fff338`
- Git HEAD: `dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11`; branch: `(detached)`

## Configuration

- Datasets: movielens, kuairand
- Arms: TV-GRM-005, TV-GRM-010, TV-GRM-020, TV-GRM-SHUFFLE-010, TV-GRM-INSTANT-010
- Seeds: 2022, 2023, 2024, 2025, 2026
- Epochs: 50; batch size: 40960; precision: FP32
- Expected optimizer and EMA updates: 86250 each
- Full per-dataset/per-arm configuration is recorded in the JSON report under `config.per_dataset_arm`.

## Completion counts

- New runs: 50/50; reused controls: 20/20; total exported: 70/70
- Epoch records: 2500 new and 1000 reused
- Missing runs: 0; failed runs: 0; Test accessed: false

## Raw table files

| Table | Rows | SHA256 |
|---|---:|---|
| `CANDIDATE_RANKING.csv` | 3 | `99b900cdafdc756b889c563b52ef0ffd053e05da1f1bd252a6492dac6568d812` |
| `COSTS.csv` | 70 | `def9c964efcbad25ad8356a36420e9b6985259946fa8b721a06e45543744a0e4` |
| `ENDPOINTS.csv` | 560 | `2f5aec229f18f00cccd37a280b917e0d2cb6b41bf25f32eeda83daccc91f55f9` |
| `GRADIENT_DIAGNOSTICS.csv` | 250 | `3a76ed8cb717c6f3ad23a6eb8475e8dfc32578e78bafa4e719b88f482d945f79` |
| `PAIRED_SEEDS.csv` | 38280 | `e0365eb86f74f20d6fcd788f53906b8ae1ffec6bfea5343623f37f31004403db` |
| `PAIRED_SUMMARY.csv` | 7656 | `36fadfd50b3969506fd067345036a1b2ef1b2308858245f75643de864f1ad3bd` |
| `SUMMARY.csv` | 3696 | `67e82fbb98f1ebc6d3ed1505c15d7428638611a0d58238db2f13a4eac1c9b204` |
| `VALIDATION_CURVES.csv` | 7000 | `4b6be3c120a277668229bad38911aedc3bccb40a776302a0e83218ff3107535c` |
| `WEIGHTS.csv` | 36000 | `f9bc6c2c8ddba64f60a9504057fd491f398d2931cc63827a6078956289579f0c` |

## Execution logs

- Prior outer-driver event: exited at 00:26 after a reconnect warning; CLI exit code 0 and turn completed. Retired and not restarted.

- Command: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner tables --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1`
- Exit code: 0
- stdout: `artifacts/26-11-16_augmentation_gradient_momentum/runner_logs/tables.stdout.log` (SHA256 `4e561a4c5c7ef570fb727717e7e388de563eb9438ee2ce75166d98cb4f57af78`)
- stderr: `artifacts/26-11-16_augmentation_gradient_momentum/runner_logs/tables.stderr.log` (SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)
- Runtime error: none
- No training, recovery, Test evaluation, model release, or guardian termination was performed.
