# STAGE0 REPORT

- Attempt: `attempt_20261102_0005_check`
- Command device: `cuda:1`
- Status: `PASS`
- Execution commit: `ac2ec5cc70be7dcb880a7dde1015e4634726fd96`
- Source manifest hash: `ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`
- Receipt identity matched the requested commit and manifest hash.
- Batch size: 40960.
- Storage mode: `RAM_ONLY`.
- Test evaluation: false for both datasets and overall.

## Raw dataset checks

| Dataset | Status | Cases | Optimizer updates | Peak allocated bytes | Peak reserved bytes | Test evaluated |
|---|---|---:|---:|---:|---:|---|
| MovieLens | PASS | 8/8 PASS | 13 | 2,079,853,568 | 2,298,478,592 | false |
| KuaiRand | PASS | 8/8 PASS | 13 | 2,220,430,336 | 2,589,982,720 | false |

The eight case names for each dataset are recorded in `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_check/stage0_checks.json`: `A0_source_Adam_exact`, `A1_source_Adam_exact`, `B1_R1_real_count_energy_and_RNG`, `B2_real_task_gradients_U_projection_and_RAM_restore`, `B1_disabled_exact_A1`, `B2_disabled_exact_A1`, `B3_disabled_exact_A1`, and `three_independent_shadows_nonfeedback_and_lambda_budget`.

## Stage0 artifacts

`artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_check/` contains `stage0_checks.json`, per-dataset Stage0 receipts, formula checks, data manifest, environment, resolved config, source manifest, trial plan, and failure ledger. The first Stage0 console output was not separately redirected; the Stage0 receipt and per-dataset output files are preserved in this directory.
