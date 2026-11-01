# GAM01 Experiment Execution Report

## 1. Experiment ID

GAM01 — gradient augmentation momentum; formal attempt `attempt_20261101_1340`.

## 2. Git and worktree state

- Reviewed implementation source commit: `89edb99a389c56fde86460af677a48e14fbcf116`.
- This checkout has no local `.git` metadata; source commit was supplied to the coordinator with `--code-commit`.

## 3. Configuration

- Mode: `all`; devices: `cuda:0,cuda:1`; formal matrix: MovieLens and KuaiRand × A0–A3 × five seeds.
- 50 epochs/run; batch size 40960; FP32; no early stopping. Stage0 was checked at batch 40960 on `cuda:1`.
- Formal totals recorded by coordinator: 40 runs, 2,000 epochs, 69,000 optimizer steps.
- P2 gate: `NOT_TRIGGERED` for both datasets (`test_accessed: false` in the gate record); unified test followed the frozen selection.

## 4. Seeds

2022, 2023, 2024, 2025, 2026 (each dataset/arm).

## 5. Run status

- Stage0: PASS; CPU checks recorded as 4/4 passed. CUDA Stage0 checks are in the checks artifact below.
- Formal phase 1: 40/40 complete; 0 failed; no missing runs.
- Unified evaluation: validation and test tables include selected, E10, and E50 for online and shadow model states.
- Test metrics: 240 rows (excluding header); selected online test metrics span both datasets, all four arms, and all five seeds.
- Coordinator completion record: `artifacts/gradient_aug_momentum_20261101/attempt_20261101_1340/completion.json` (complete; test evaluated; 3,203,080,124 snapshot tensor bytes held in RAM during the run).

## 6. Raw test metrics

The following are selected-checkpoint online metrics summarized across the five seeds (mean ± sample SD). Per-seed values and the E10/E50 and shadow rows are retained in `test_metrics.csv`; validation counterparts are in `validation_metrics.csv`. Values are reported as recorded, without scientific interpretation.

### movielens

| Metric | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| `macro_auc` | 0.775190 ± 0.001136 | 0.775129 ± 0.001355 | 0.774927 ± 0.001171 | 0.774793 ± 0.002160 |
| `macro_logloss` | 0.494891 ± 0.004690 | 0.493643 ± 0.003991 | 0.492380 ± 0.002469 | 0.495428 ± 0.005867 |
| `domain0_macro_auc` | 0.777735 ± 0.002185 | 0.776716 ± 0.002129 | 0.778018 ± 0.002548 | 0.777091 ± 0.002619 |
| `domain0_macro_logloss` | 0.495903 ± 0.006812 | 0.493848 ± 0.005164 | 0.492056 ± 0.004135 | 0.497963 ± 0.006107 |
| `domain1_macro_auc` | 0.787789 ± 0.000969 | 0.788175 ± 0.002072 | 0.787117 ± 0.000978 | 0.787719 ± 0.001267 |
| `domain1_macro_logloss` | 0.479416 ± 0.003482 | 0.479286 ± 0.006976 | 0.479142 ± 0.004445 | 0.477915 ± 0.003592 |
| `domain2_macro_auc` | 0.760044 ± 0.001252 | 0.760496 ± 0.001826 | 0.759647 ± 0.001380 | 0.759570 ± 0.002944 |
| `domain2_macro_logloss` | 0.509353 ± 0.004287 | 0.507796 ± 0.002142 | 0.505943 ± 0.002395 | 0.510406 ± 0.008320 |
| `task0_macro_auc` | 0.766897 ± 0.001195 | 0.767279 ± 0.001757 | 0.766962 ± 0.001407 | 0.766474 ± 0.002351 |
| `task0_macro_logloss` | 0.579617 ± 0.003341 | 0.579462 ± 0.003859 | 0.578391 ± 0.003464 | 0.580498 ± 0.005224 |
| `task1_macro_auc` | 0.783482 ± 0.001109 | 0.782979 ± 0.001012 | 0.782892 ± 0.000978 | 0.783113 ± 0.001988 |
| `task1_macro_logloss` | 0.410165 ± 0.006474 | 0.407825 ± 0.004341 | 0.406369 ± 0.002603 | 0.410358 ± 0.006889 |

### kuairand

| Metric | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| `macro_auc` | 0.671907 ± 0.006145 | 0.674491 ± 0.005178 | 0.673812 ± 0.005533 | 0.672937 ± 0.003129 |
| `macro_logloss` | 0.437172 ± 0.002751 | 0.430971 ± 0.005113 | 0.437577 ± 0.002444 | 0.434816 ± 0.001482 |
| `domain0_macro_auc` | 0.640763 ± 0.018112 | 0.643696 ± 0.016217 | 0.646696 ± 0.018437 | 0.644632 ± 0.010606 |
| `domain0_macro_logloss` | 0.235643 ± 0.002393 | 0.235874 ± 0.001591 | 0.235502 ± 0.002793 | 0.235715 ± 0.001982 |
| `domain1_macro_auc` | 0.712748 ± 0.004834 | 0.710301 ± 0.003440 | 0.712699 ± 0.002379 | 0.712212 ± 0.002821 |
| `domain1_macro_logloss` | 0.426730 ± 0.007995 | 0.416415 ± 0.013284 | 0.426852 ± 0.007137 | 0.420502 ± 0.007953 |
| `domain2_macro_auc` | 0.662208 ± 0.003765 | 0.669475 ± 0.004082 | 0.662042 ± 0.002373 | 0.661967 ± 0.002876 |
| `domain2_macro_logloss` | 0.649143 ± 0.005017 | 0.640625 ± 0.004147 | 0.650377 ± 0.003815 | 0.648231 ± 0.004152 |
| `task0_macro_auc` | 0.662570 ± 0.004859 | 0.662885 ± 0.004335 | 0.663189 ± 0.004824 | 0.662055 ± 0.004270 |
| `task0_macro_logloss` | 0.481868 ± 0.003106 | 0.476483 ± 0.005664 | 0.481579 ± 0.002769 | 0.480321 ± 0.001777 |
| `task1_macro_auc` | 0.681243 ± 0.008891 | 0.686097 ± 0.006601 | 0.684435 ± 0.006703 | 0.683819 ± 0.003466 |
| `task1_macro_logloss` | 0.392476 ± 0.003098 | 0.385459 ± 0.004657 | 0.393575 ± 0.002673 | 0.389312 ± 0.001398 |

## 7. Logs and artifacts

- Attempt root: `artifacts/gradient_aug_momentum_20261101/attempt_20261101_1340/`.
- Stage0 CUDA evidence: `artifacts/gradient_aug_momentum_20261101/attempt_20261101_1339_check/checks.json`.
- Per-run `stdout.log` and `stderr.log`: under `runs/<dataset>/<arm>/seed<seed>/attempt1/`.
- Run/config/source ledgers: `run_ledger.csv`, `launch_ledger.csv`, `resolved_config.json`, `source_manifest.json`, `environment.json`.
- Raw epoch, per-cell, validation, test, paired, and aggregate metric files: `epoch_metrics.csv`, `per_cell_metrics.csv`, `validation_metrics.csv`, `test_metrics.csv`, `paired_effects.csv`, `effect_summary.csv`, `arm_summary.csv`.
- Phase1 gate and frozen selections: `phase1_validation_decision.json`, `selection_frozen.json`.

## 8. Runtime errors

- Failed-run ledger is empty. No runtime errors recorded.
- Worker stderr contained PyTorch’s TF32 API deprecation `UserWarning`; it did not stop execution.

## 9. Missing runs

None. All 40 configured runs completed; test metrics cover the full 40-run matrix for selected, E10, and E50 states.
