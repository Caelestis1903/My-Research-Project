# Stage0 Execution Report

- Experiment: `gradient_aug_time_select_20261102` (GATS02)
- Attempt: `attempt_20261102_recovery_0440_check`
- Command mode/device: `check`, `cuda:1`
- Status: `PASS` for MovieLens and KuaiRand
- Execution commit: `be0227e510f00864b8c7ec7565def0b5a813be4f`
- Source manifest hash: `7f7a0eecb052ca0be1f5f7ffc07433e85d5e2d73bcbfd7e2f6f0ff4efac328a1`
- Receipt identity matched the requested execution commit and source hash.
- Batch size: 40960; storage: `RAM_ONLY`; test evaluation: false for both datasets and overall.

| Dataset | Status | Cases passed | Actual optimizer updates | Peak allocated bytes | Peak reserved bytes | Test evaluated |
|---|---|---:|---:|---:|---:|---|
| MovieLens | PASS | 8/8 | 13 | 2,079,853,568 | 2,298,478,592 | false |
| KuaiRand | PASS | 8/8 | 13 | 2,220,430,336 | 2,589,982,720 | false |

The eight case names and raw values for each dataset are in `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440_check/stage0_checks.json` and the per-dataset receipts in that check directory. Stage0 stdout/stderr are preserved as `attempt_20261102_recovery_0440_check.stdout.log` and `.stderr.log` in the parent artifact directory.

The original Stage0 report from the first execution identity is preserved at [INITIAL_STAGE0_REPORT.md](INITIAL_STAGE0_REPORT.md). That earlier receipt used commit `ac2ec5cc70be7dcb880a7dde1015e4634726fd96` and manifest hash `ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`; it is not the Stage0 receipt used by the completed final attempt.
