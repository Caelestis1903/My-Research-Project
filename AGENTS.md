# Research Experiment Orchestration

This repository is an experimental research codebase.

The primary Codex agent acts only as the ORCHESTRATOR.

It must not directly implement research algorithms when a specialized
agent exists.

## Required workflow

For every new algorithmic experiment:

DESIGN
↓
IMPLEMENT
↓
REVIEW
↓
RUN
↓
ANALYZE

The stages must not be skipped.

## Stage 1 — DESIGN

Spawn `designer`.

Designer must produce a DESIGN SPECIFICATION.

No implementation may begin before design is complete.

## Stage 2 — IMPLEMENT

Pass the approved DESIGN SPECIFICATION to `luna-coder`.

Coder must implement only the specification.

Coder may perform syntax/basic checks but must not analyze research results.

## Stage 3 — REVIEW

Spawn `reviewer`.

Reviewer must independently inspect:

- DESIGN SPECIFICATION
- actual implementation
- CODER IMPLEMENTATION REPORT

Experiments may proceed only if reviewer verdict is:

PASS

or

PASS WITH FIXES

and all blocking issues have been corrected.

If verdict is FAIL:

return to CODER.

## Stage 4 — RUN

Spawn `luna-runner`.

Unless explicitly specified otherwise, experimental seeds are:

2022
2023
2024
2025

Runner records raw results only.

Runner MUST NOT analyze research conclusions.

## Stage 5 — ANALYZE

The primary agent performs scientific analysis.

The primary agent MUST NOT assume:

- Curriculum Learning is effective.
- Momentum mechanisms are effective.
- M3OE necessarily fails on difficult domains.
- Small domains are equivalent to difficult domains.

The analysis must evaluate evidence independently.

For experiment analysis examine:

- AUC
- loss
- per-domain performance
- per-task performance
- seed consistency
- domain size
- domain difficulty
- baseline performance
- model
- training stage

Identify:

1. Stable repeated phenomena
2. Contradictory experiments
3. Evidence strength
4. Competing explanations
5. Whether domain size is a plausible main explanatory variable
6. Whether a small-domain improvement / large-domain decline trade-off exists
7. What experiment would best discriminate competing explanations

Evidence strength must be labeled:

STRONG
MEDIUM
WEAK

The orchestrator must prefer falsification-oriented experiments over
experiments that merely confirm the current hypothesis.

## Project Python environment

- The project environment is the Conda environment `phase0-3-env` at `/home_new/xzf/envs/miniconda3/envs/phase0-3-env`.
- It uses Python 3.13.5, PyTorch 2.9.0+cu126, and the exact pip package snapshot in `requirements-phase0-3.lock`.
- In new SSH terminals, activate it with `source /home_new/xzf/envs/miniconda3/etc/profile.d/conda.sh` followed by `conda activate phase0-3-env`.
- Codex must run project Python commands explicitly through `conda run --no-capture-output -n phase0-3-env python ...` or `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; do not use Conda `base`.
- Run tests with `conda run --no-capture-output -n phase0-3-env python -m pytest ...`.
- Keep this setting scoped to this repository; do not change global shell or Conda defaults.
- PyTorch includes the CUDA 12.6 runtime. This setup session had no NVIDIA device exposed (`torch.cuda.is_available()` was false), so check GPU visibility before starting GPU experiments.

## Local dataset paths

- MovieLens CSV: `/home_new/xzf/datasets/ml-1m/ml-1m.csv`.
- KuaiRand processed CSV: `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`.
- Both CSVs were verified against the existing project SHA-256 records after relocation.
- The legacy `/home/xzf/data` locations are no longer valid. Paths inside historical experiment records and pinned source snapshots record their original locations.
- Prepared data expected under `/home_new/xzf/datasets/mdmt_next_20261024_strict_v1` was not present in the migrated datasets or the local recovery directory. Restore and verify these artifacts before running workflows that require them; a corrected path alone does not make those workflows ready.
