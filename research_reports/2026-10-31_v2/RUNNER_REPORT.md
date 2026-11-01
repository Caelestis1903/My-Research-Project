# GA31-R3 Experiment Execution Report

## 1. Experiment ID

GA31-R3 (protocol revision R3-r1-codex-readiness).

## 2. Git/worktree state

Git metadata is unavailable from `/home/xzf/Code/phase0-3` (`git` reports that the working directory is not inside a repository). The protocol records source commit `ee6e81fdfd37284e6447e8149cb93d04736dba02` on branch `26-10-31`. No source code was modified by this runner.

## 3. Config

- Entry: `python -X faulthandler -m experiments.gradient_aug_20261031_v2.run --mode all --protocol research_plans/26-10-31_v2/phase1_protocol.json --run-dir artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645 --devices 0,1`
- Batch size: 40,960; epochs: 50; optimizer: Adam, learning rate 0.01; AMP off; TF32 off.
- Formal training runs: 45 total. Stage A: 15 (B-ML 5, L2-ML 5, B-KR 5). Stage B: 30 (M1, G-IID2, G-ANTI2; each 5 seeds on each dataset).
- Probe launches: 10. Baseline and candidates frozen before the unified test stage. State storage: RAM only; release recorded after test and numeric report tables.
- Seeds: 2022, 2023, 2024, 2025, 2026.

## 4. Seeds and run status

Completion record: `COMPLETE`; 45/45 formal runs completed; every run has 50 epochs. Test evaluation completed. Validation and test metric files each contain 135 rows (3 checkpoints × 45 runs: selected, E10, E50). There are no entries in the current attempt failure ledger.

## 5. Raw metrics

Selected-checkpoint test metrics below are five-seed means ± standard deviations. These are numeric summaries copied from `arm_summary.csv`; per-seed rows remain in `test_metrics.csv`.

| Dataset | Arm | Macro AUC | Domain 0 AUC | Domain 1 AUC | Domain 2 AUC | Task 0 AUC | Task 1 AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| movielens | B-ML | 0.774992 ± 0.001604 | 0.777697 ± 0.002551 | 0.787594 ± 0.001788 | 0.759686 ± 0.000913 | 0.766796 ± 0.001531 | 0.783188 ± 0.001820 |
| movielens | L2-ML | 0.775190 ± 0.001136 | 0.777735 ± 0.002185 | 0.787789 ± 0.000969 | 0.760044 ± 0.001252 | 0.766897 ± 0.001195 | 0.783482 ± 0.001109 |
| movielens | M1 | 0.773563 ± 0.002137 | 0.773526 ± 0.003650 | 0.787955 ± 0.001182 | 0.759206 ± 0.001947 | 0.765166 ± 0.001900 | 0.781959 ± 0.002404 |
| movielens | G-IID2 | 0.774983 ± 0.001205 | 0.777791 ± 0.001311 | 0.787808 ± 0.001548 | 0.759350 ± 0.002160 | 0.766896 ± 0.001654 | 0.783070 ± 0.000962 |
| movielens | G-ANTI2 | 0.774503 ± 0.001251 | 0.777892 ± 0.001863 | 0.787135 ± 0.000955 | 0.758482 ± 0.001209 | 0.765999 ± 0.001733 | 0.783007 ± 0.000791 |
| kuairand | B-KR | 0.671907 ± 0.006145 | 0.640763 ± 0.018112 | 0.712748 ± 0.004834 | 0.662208 ± 0.003765 | 0.662570 ± 0.004859 | 0.681243 ± 0.008891 |
| kuairand | M1 | 0.673137 ± 0.002955 | 0.633606 ± 0.008100 | 0.714228 ± 0.001948 | 0.671578 ± 0.007957 | 0.664279 ± 0.004303 | 0.681995 ± 0.005515 |
| kuairand | G-IID2 | 0.674346 ± 0.001645 | 0.647056 ± 0.007433 | 0.711743 ± 0.001946 | 0.664239 ± 0.005632 | 0.662562 ± 0.005209 | 0.686130 ± 0.003002 |
| kuairand | G-ANTI2 | 0.673109 ± 0.005334 | 0.639155 ± 0.013773 | 0.713650 ± 0.001810 | 0.666521 ± 0.005371 | 0.663696 ± 0.006181 | 0.682521 ± 0.004934 |

| Dataset | Arm | Macro log loss | Domain 0 log loss | Domain 1 log loss | Domain 2 log loss | Task 0 log loss | Task 1 log loss |
|---|---|---:|---:|---:|---:|---:|---:|
| movielens | B-ML | 0.493372 ± 0.005437 | 0.494128 ± 0.007394 | 0.478490 ± 0.006308 | 0.507497 ± 0.003412 | 0.577596 ± 0.003478 | 0.409148 ± 0.007539 |
| movielens | L2-ML | 0.494891 ± 0.004690 | 0.495903 ± 0.006812 | 0.479416 ± 0.003482 | 0.509353 ± 0.004287 | 0.579617 ± 0.003341 | 0.410165 ± 0.006474 |
| movielens | M1 | 0.495424 ± 0.005588 | 0.499409 ± 0.007564 | 0.477774 ± 0.006392 | 0.509088 ± 0.005211 | 0.580561 ± 0.005196 | 0.410287 ± 0.006196 |
| movielens | G-IID2 | 0.491190 ± 0.004939 | 0.492048 ± 0.007684 | 0.475600 ± 0.005493 | 0.505922 ± 0.002034 | 0.577598 ± 0.003939 | 0.404782 ± 0.005949 |
| movielens | G-ANTI2 | 0.495393 ± 0.003002 | 0.496425 ± 0.004908 | 0.479940 ± 0.001773 | 0.509815 ± 0.003372 | 0.580407 ± 0.001872 | 0.410380 ± 0.004773 |
| kuairand | B-KR | 0.437172 ± 0.002751 | 0.235643 ± 0.002393 | 0.426730 ± 0.007995 | 0.649143 ± 0.005017 | 0.481868 ± 0.003106 | 0.392476 ± 0.003098 |
| kuairand | M1 | 0.434039 ± 0.002017 | 0.237734 ± 0.002507 | 0.420181 ± 0.004470 | 0.644203 ± 0.005559 | 0.478416 ± 0.002164 | 0.389663 ± 0.002178 |
| kuairand | G-IID2 | 0.436282 ± 0.003121 | 0.234752 ± 0.001788 | 0.424491 ± 0.004821 | 0.649604 ± 0.005997 | 0.481515 ± 0.003867 | 0.391050 ± 0.003310 |
| kuairand | G-ANTI2 | 0.435463 ± 0.004266 | 0.236417 ± 0.001251 | 0.423494 ± 0.008457 | 0.646476 ± 0.006408 | 0.480272 ± 0.004941 | 0.390653 ± 0.003805 |

Full raw validation/test rows, including per-domain and per-task AUC and log loss, are in `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/test_metrics.csv` and `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/validation_metrics.csv`. Five-seed summaries for all listed metrics are in `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/arm_summary.csv`; paired rows and descriptive effects are in `paired_effects.csv` and `effect_summary.csv`.

## 6. Runtime errors and warnings

Current attempt: no runtime errors; failure ledger is empty apart from its header. Standard error contains repeated PyTorch TF32 API deprecation warnings.

Prior attempt `attempt_20261031_230801`: recorded native `free(): invalid pointer` during KuaiRand load, process exit 134; no training launch occurred in that attempt. That failed attempt remains preserved.

## 7. Probe accounting and release

- Online temporal probe: 120 observed states, 360 extra gradients, 0 extra optimizer steps.
- Offline probe: 60 batches, 840 gradients, 300 counterfactual steps.
- Test rows: 135; run ledger rows: 45; state release: `released=true`, storage `RAM_ONLY`.
- Probe records: `temporal_probe.csv`, `increment_probe.csv`, `counterfactual_update.csv`, `view_probe.csv`, `probe_ledger.json`.

## 8. Log and artifact locations

- Attempt directory: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645`
- Coordinator stdout: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645.stdout.log`
- Coordinator stderr: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645.stderr.log`
- Run ledger: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/run_ledger.csv`
- Launch ledger: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/launch_ledger.csv`
- Failure ledger: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/failure_ledger.csv`
- Completion record: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/completion.json`
- State release record: `artifacts/gradient_aug_20261031_v2/GA31-R3/attempt_20261031_151645/state_release.json`

## 9. Missing runs

None. All 45 formal runs completed with 50 epochs; all 45 runs have selected/E10/E50 test rows.
