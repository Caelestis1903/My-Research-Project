# Stage S6-R2 — Retrospective Robustness Analysis

Pure-retrospective / evaluation-only analysis of the KuaiRand DH selected-solution
phenomenon. Per `experiments/stage_S6_R2_design.md` (and the research brief
`promt/S6_R2.md`).

## Layout

| Script | Role |
|---|---|
| `s6r2_common.py`           | shared constants, loaders, aggregation, gate stats |
| `s6r2_integrity.py`        | cross-checks the saved artifacts (no recomputation) |
| `s6r2_partA_selection.py`  | R0–R4 selection on validation only (+ A-grid variant) |
| `s6r2_partB_decomposition.py` | F-anchor / DH-anchor identities + grid trajectory |
| `s6r2_partC_window.py`     | ±2 selection-window analysis & classification |
| `s6r2_partD_swap.py`       | KuaiRand parameter-swap hybrid inference (only KuaiRand) |
| `s6r2_report.py`           | consolidate CSVs into `stage_S6_R2_selected_solution_analysis.md` |

Outputs land in `experiments/stage_S6_R2_raw_results/{dataset}/` and
`experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json`.

## Running

Always with the project virtualenv and `PYTHONPATH=experiments`:

```bash
# Parts A/B/C (no model load, fast):
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partA_selection --dataset kuairand
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partA_selection --dataset movielens
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partB_decomposition --dataset kuairand
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partB_decomposition --dataset movielens
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partC_window --dataset kuairand
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partC_window --dataset movielens

# Integrity (always run before the main agent reads the report):
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_integrity --dataset kuairand
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_integrity --dataset movielens

# Part D (long; only KuaiRand; needs CUDA for ~15-min total runtime):
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_partD_swap --dataset kuairand
# Append --device cpu if no GPU.

# Final report:
PYTHONPATH=experiments venv/Scripts/python.exe -m stage_S6_R2.s6r2_report
```

Two datasets are run independently; the runner can launch them concurrently
as two separate processes.

## What this stage does NOT do

- No training or optimisation step.
- No modification of `experiments/stage_S1_*.py`, `stage_S6_*`, `m3oe/**`.
- No reliance on per-epoch weights that were never saved (`predictions_test.npz`
  only covers {selected, final, common_epoch_*}).
- Verdict / H1-H6 / GO-STOP sections of the report are left as
  `[MAIN-AGENT: ...]` placeholders for the main agent.
