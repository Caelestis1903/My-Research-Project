# Stage E3 Run Report

This is a structured artifact/status report; it contains no scientific conclusion.

- Expected joint runs: 60
- Available run directories: 60
- Missing run directories: 0
- Oracle coverage cells: 360/360

## Formal execution and integrity status

- Command: `D:\CODE\phase0-3\venv\Scripts\python.exe experiments/stage_E3_orchestrator.py --dataset_path D:/CODE/phase0-3/m3oe/examples/data/ml-1m --device cuda:0`
- Runtime: Python 3.13 venv with `D:\CODE\.venv\Lib\site-packages`; torch `2.6.0+cu126`; CUDA available.
- Seeds: `2022, 2023, 2024, 2025`; formal protocol: upper bound `50` epochs, early-stop patience `10`.
- Oracle references: `36/36` formal-valid and artifact-complete; master status `REUSED` with reason `OK` after successful generation/audit; missing/invalid `0`.
- Joint matrix: `60/60` status `OK`; `12/12` setting×seed blocks `VALID`; `60/60` arm manifests report `block_valid=true` and `common_support_valid=true`.
- Common-support raw ranges across the 12 blocks: `14–18` common epochs and `84–108` common trace rows per block.
- Pair/block validity: all five alpha arms in each of the 12 blocks passed the block comparison; no invalid pair/block was recorded.
- Non-finite inventory: `0` joint entries and `0` unexplained entries.

## Artifact/schema counts

All `60/60` joint run directories contain each required file: `config.json`, `stdout.log`, `run_manifest.json`, `metrics_epoch.csv`, `metrics_summary.csv`, `shared_gradient_step.csv`, `shared_gradient_epoch.csv`, `batch_trace.csv`, `nonfinite_inventory.json`, and `model.pth`.

All `36/36` oracle directories contain: `config.json`, `stdout.log`, `run_manifest.json`, `metrics_epoch.csv`, `metrics_summary.csv`, and `model.pth`.

Seed-level summary rows: `360` (`60` runs × `6` cells). Oracle coverage: `360/360` cells.

## Matrix status

| alpha_key | setting | seed | status | block | missing files |
|---|---|---:|---|---|---:|
| alpha_0p00 | d0_large | 2022 | OK | VALID | 0 |
| alpha_0p00 | d0_large | 2023 | OK | VALID | 0 |
| alpha_0p00 | d0_large | 2024 | OK | VALID | 0 |
| alpha_0p00 | d0_large | 2025 | OK | VALID | 0 |
| alpha_0p00 | d1_large | 2022 | OK | VALID | 0 |
| alpha_0p00 | d1_large | 2023 | OK | VALID | 0 |
| alpha_0p00 | d1_large | 2024 | OK | VALID | 0 |
| alpha_0p00 | d1_large | 2025 | OK | VALID | 0 |
| alpha_0p00 | d2_large | 2022 | OK | VALID | 0 |
| alpha_0p00 | d2_large | 2023 | OK | VALID | 0 |
| alpha_0p00 | d2_large | 2024 | OK | VALID | 0 |
| alpha_0p00 | d2_large | 2025 | OK | VALID | 0 |
| alpha_0p25 | d0_large | 2022 | OK | VALID | 0 |
| alpha_0p25 | d0_large | 2023 | OK | VALID | 0 |
| alpha_0p25 | d0_large | 2024 | OK | VALID | 0 |
| alpha_0p25 | d0_large | 2025 | OK | VALID | 0 |
| alpha_0p25 | d1_large | 2022 | OK | VALID | 0 |
| alpha_0p25 | d1_large | 2023 | OK | VALID | 0 |
| alpha_0p25 | d1_large | 2024 | OK | VALID | 0 |
| alpha_0p25 | d1_large | 2025 | OK | VALID | 0 |
| alpha_0p25 | d2_large | 2022 | OK | VALID | 0 |
| alpha_0p25 | d2_large | 2023 | OK | VALID | 0 |
| alpha_0p25 | d2_large | 2024 | OK | VALID | 0 |
| alpha_0p25 | d2_large | 2025 | OK | VALID | 0 |
| alpha_0p50 | d0_large | 2022 | OK | VALID | 0 |
| alpha_0p50 | d0_large | 2023 | OK | VALID | 0 |
| alpha_0p50 | d0_large | 2024 | OK | VALID | 0 |
| alpha_0p50 | d0_large | 2025 | OK | VALID | 0 |
| alpha_0p50 | d1_large | 2022 | OK | VALID | 0 |
| alpha_0p50 | d1_large | 2023 | OK | VALID | 0 |
| alpha_0p50 | d1_large | 2024 | OK | VALID | 0 |
| alpha_0p50 | d1_large | 2025 | OK | VALID | 0 |
| alpha_0p50 | d2_large | 2022 | OK | VALID | 0 |
| alpha_0p50 | d2_large | 2023 | OK | VALID | 0 |
| alpha_0p50 | d2_large | 2024 | OK | VALID | 0 |
| alpha_0p50 | d2_large | 2025 | OK | VALID | 0 |
| alpha_0p75 | d0_large | 2022 | OK | VALID | 0 |
| alpha_0p75 | d0_large | 2023 | OK | VALID | 0 |
| alpha_0p75 | d0_large | 2024 | OK | VALID | 0 |
| alpha_0p75 | d0_large | 2025 | OK | VALID | 0 |
| alpha_0p75 | d1_large | 2022 | OK | VALID | 0 |
| alpha_0p75 | d1_large | 2023 | OK | VALID | 0 |
| alpha_0p75 | d1_large | 2024 | OK | VALID | 0 |
| alpha_0p75 | d1_large | 2025 | OK | VALID | 0 |
| alpha_0p75 | d2_large | 2022 | OK | VALID | 0 |
| alpha_0p75 | d2_large | 2023 | OK | VALID | 0 |
| alpha_0p75 | d2_large | 2024 | OK | VALID | 0 |
| alpha_0p75 | d2_large | 2025 | OK | VALID | 0 |
| alpha_1p00 | d0_large | 2022 | OK | VALID | 0 |
| alpha_1p00 | d0_large | 2023 | OK | VALID | 0 |
| alpha_1p00 | d0_large | 2024 | OK | VALID | 0 |
| alpha_1p00 | d0_large | 2025 | OK | VALID | 0 |
| alpha_1p00 | d1_large | 2022 | OK | VALID | 0 |
| alpha_1p00 | d1_large | 2023 | OK | VALID | 0 |
| alpha_1p00 | d1_large | 2024 | OK | VALID | 0 |
| alpha_1p00 | d1_large | 2025 | OK | VALID | 0 |
| alpha_1p00 | d2_large | 2022 | OK | VALID | 0 |
| alpha_1p00 | d2_large | 2023 | OK | VALID | 0 |
| alpha_1p00 | d2_large | 2024 | OK | VALID | 0 |
| alpha_1p00 | d2_large | 2025 | OK | VALID | 0 |
