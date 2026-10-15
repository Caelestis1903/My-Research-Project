# Stage E4 Run Report

This report records artifact coverage and schema validity only.

- Expected joint runs: 12
- Formal-valid joint runs: 12
- Smoke-only joint runs: 0
- Expected matched oracle runs: 12
- Formal-valid matched oracle runs: 12
- Smoke-only matched oracle runs: 0

## Joint artifact status

| weighting | seed | status | block | common epochs/traces | path/reasons |
|---|---:|---|---|---:|---|
| equal | 2022 | FORMAL_VALID | VALID (True) | 28/1400 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\equal\seed2022  |
| equal | 2023 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\equal\seed2023  |
| equal | 2024 | FORMAL_VALID | VALID (True) | 24/1200 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\equal\seed2024  |
| equal | 2025 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\equal\seed2025  |
| sqrt | 2022 | FORMAL_VALID | VALID (True) | 28/1400 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sqrt\seed2022  |
| sqrt | 2023 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sqrt\seed2023  |
| sqrt | 2024 | FORMAL_VALID | VALID (True) | 24/1200 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sqrt\seed2024  |
| sqrt | 2025 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sqrt\seed2025  |
| sample | 2022 | FORMAL_VALID | VALID (True) | 28/1400 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sample\seed2022  |
| sample | 2023 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sample\seed2023  |
| sample | 2024 | FORMAL_VALID | VALID (True) | 24/1200 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sample\seed2024  |
| sample | 2025 | FORMAL_VALID | VALID (True) | 23/1150 | D:\CODE\phase0-3\experiments\stage_E4_raw_results\sample\seed2025  |

## Oracle artifact status

| domain | seed | status | path/reasons |
|---:|---:|---|---|
| 0 | 2022 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d0\seed2022  |
| 0 | 2023 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d0\seed2023  |
| 0 | 2024 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d0\seed2024  |
| 0 | 2025 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d0\seed2025  |
| 1 | 2022 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d1\seed2022  |
| 1 | 2023 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d1\seed2023  |
| 1 | 2024 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d1\seed2024  |
| 1 | 2025 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d1\seed2025  |
| 2 | 2022 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d2\seed2022  |
| 2 | 2023 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d2\seed2023  |
| 2 | 2024 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d2\seed2024  |
| 2 | 2025 | FORMAL_VALID | D:\CODE\phase0-3\experiments\stage_E4_raw_results\oracle_reference\d2\seed2025  |

Structural/raw summary only; no cross-run scientific analysis.

## Structured experiment execution report

1. **Experiment ID:** Stage E4 / Group 2 Prompt 4 (matched-oracle + joint weighting matrix)

2. **Git/worktree state:** unavailable; `D:\CODE\phase0-3` is not a Git worktree (`git status` returned “not a git repository”).

3. **Formal config:** dataset `D:/CODE/data/kuairand/processed/kuai_rand.csv`; device `cuda:0`; batch size `40960`; epoch upper bound `50`; learning rate `0.01`; DARTS learning rate `0.001`; weight decay `1e-05`; expert count `4`; early-stop patience `10`; joint probe interval `10`; instrumentation enabled. Oracle protocol is split-first/filter-second. Weighting arms are `equal`, `sqrt`, and `sample`.

4. **Seeds:** `2022, 2023, 2024, 2025`.

5. **Run status:** matched oracle `12/12 OK`, joint `12/12 OK`; report validation `12/12` oracle `FORMAL_VALID`, `12/12` joint `FORMAL_VALID`; all three joint arms in each seed block have `VALID` block status and `block_valid=true`. No missing runs.

6. **Raw metrics:** `experiments/stage_E4_summary.csv` contains `72` rows (`12` joint runs × `6` domain/task cells); each row preserves raw AUC and LogLoss fields from the corresponding `metrics_summary.csv`, together with per-domain/task/split fields. Per-epoch raw metric files are at each run directory listed above. Joint common-support counts are recorded in the table above and in `stage_E4_summary_manifest.json`.

7. **Log locations:** each run’s complete stdout/stderr capture is in its `stdout.log`; joint logs are under `experiments/stage_E4_raw_results/{equal,sqrt,sample}/seed{2022,2023,2024,2025}/`, and oracle logs are under `experiments/stage_E4_raw_results/oracle_reference/d{0,1,2}/seed{2022,2023,2024,2025}/`.

8. **Runtime errors:** initial direct invocation failed before any run because the parent process lacked the PyTorch path: `ModuleNotFoundError: No module named 'torch'`. This was diagnosed as an environment issue and retried with `PYTHONPATH=D:\CODE\.venv\Lib\site-packages`; the formal matrix then completed. No child-run runtime errors were recorded.

9. **Missing runs:** none. Required artifact check: `0` missing files across `12` joint and `12` oracle directories. Non-finite inventory: `24` inventory files, `0` entries.

## Execution commands

- Initial failed command: `D:\CODE\phase0-3\venv\Scripts\python.exe experiments/stage_E4_orchestrator.py --device cuda:0`
- Successful formal command: `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; D:\CODE\phase0-3\venv\Scripts\python.exe -u experiments/stage_E4_orchestrator.py --device cuda:0`
- Structural report command: `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; D:\CODE\phase0-3\venv\Scripts\python.exe -u experiments/stage_E4_report.py --formal`

No scientific interpretation, hypothesis evaluation, or result selection is included in this runner report.
