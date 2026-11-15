# Coder implementation report — 2026-11-15

Implemented the approved DESIGN_SPECIFICATION in the isolated `experiments/heterogeneity_aware_20261115/` package. Historical experiment code, plans and reports were not edited. Source is stable for independent review.

Reviewed-source candidate SHA256: `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491` from `protocol.source_manifest()` (59 current/new Python and source JSON files). Source Git base: `79fbd441c6c87cff30838c65bf97ad6ec7f45abb`; publication is handled by the orchestrator.

## Files and behavior

| File in the new package | Responsibility |
|---|---|
| `core.py` | Temporary embedding/logit hooks; probe-only full gate derivative; A/B/C and Rademacher/Gaussian views; exact projection, pair cap, two-halving drift check; target-only auxiliary BCE; one Adam and EMA update |
| `protocol.py` | Unchanged current-main utilities, exact inherited configuration, verified data-only pickle cache, immutable source/run records, five allowed seeds |
| `run.py` | Full-batch smoke, RAM-guardian workers and selected states, per-GPU resource scheduling, separate pilot/HPO/confirmation/freeze/Test commands, raw logs |
| `selection.py` | Predeclared five-seed validation gates, bounded route/alpha selection, seed-paired statistics |
| `tables.py` | All six-cell/Macro/Traffic/Pooled endpoint exports, E10/E50 validation, five-seed summaries and paired tables, online/EMA total/component contrasts |
| `test_core.py` | Five compact scientific-correctness checks |
| `README.md`, `__init__.py` | Execution instructions/package |

Clean loss is the actual source sample-mean probability BCE. The original training gate detach remains active for clean and augmented backward. Only the sensitivity probe reconnects the gate input; its numerical function matches ordinary forward. The auxiliary loss is twice the mean target BCE, matching the specified mean positive-plus-negative pair loss. No model parameters, architecture or optimizer recurrence were added.

All stages use seeds2022–2026, following the user's override. Pilot is 60 full 50epoch runs. HPO reuses identical alpha=.03 trajectories, adds at most40 runs, and does not read Test. Frozen confirmation reuses only identical completed identities; selected states remain in the existing RAM guardian. Data caching saves audited input arrays/metadata only. No model, optimizer, gradient, embedding or checkpoint state is serialized to a disk file.

## Basic verification completed

- `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/heterogeneity_aware_20261115/test_core.py` → **5 passed in 2.01s**.
- CPU checks cover sampler/empty cells, exact projection/cap, raw/probe value equivalence, logit-sum versus individual derivatives, unrelated-row independence, full-function double-precision finite differences including gate dependence, hook cleanup, source training-detach gradients, lambda0 Adam/EMA equivalence, all view paths, RAM-only continuation, and the mandatory five-seed gate.
- `python -m compileall -q experiments/heterogeneity_aware_20261115` with the project Python → passed.
- `python -m experiments.heterogeneity_aware_20261115.run --help` with the project Python → passed.

No GPU smoke, scientific training, Test prediction or research-result analysis was performed by the coder. The independent reviewer must inspect the specification, actual code and this report before the runner starts.

## Runner handoff

Fresh attempt: `artifacts/heterogeneity_aware_20261115/attempt1`.

Review JSON path: `research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json`. Required schema:

```json
{"verdict":"PASS","blocking_issues":[],"source_hash":"a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491"}
```

From the checkout root, with `PYTHON` replaced by `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

```bash
PYTHON -m experiments.heterogeneity_aware_20261115.run manifest --attempt artifacts/heterogeneity_aware_20261115/attempt1 --review research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json
PYTHON -m experiments.heterogeneity_aware_20261115.run smoke --attempt artifacts/heterogeneity_aware_20261115/attempt1 --review research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json --device cuda:0
PYTHON -m experiments.heterogeneity_aware_20261115.run pilot --attempt artifacts/heterogeneity_aware_20261115/attempt1 --review research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json --devices 0,1
PYTHON -m experiments.heterogeneity_aware_20261115.run decision --attempt artifacts/heterogeneity_aware_20261115/attempt1 --review research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json
PYTHON -m experiments.heterogeneity_aware_20261115.run tables --attempt artifacts/heterogeneity_aware_20261115/attempt1 --review research_reports/2026-11-15_heterogeneity_aware/IMPLEMENTATION_REVIEW.json
```

The pilot returns after its60 jobs. HPO is a separate command and requires an orchestrator continuation decision. Remaining exact commands are in the package README. `B_MATCH` provides the one conditional A-versus-B norm-matched contrast; dispatch it only if its predeclared gate is met. `freeze` blocks further training scheduling. `evaluate` uses only frozen selected RAM states. `release` frees state after reporting.

GPU full-batch feasibility, numerical diagnostics on both actual datasets, runtime and memory remain to be measured by the runner. A worker failure records its attempt/failure; no batch reduction or scientific rejection is automatic. The original guardian provides at most one RAM-state retry. Current-main HyperGate per-seed endpoints remain unavailable as identified in the design, so no paired HyperGate interval is fabricated.

## Review correction R1 — numerical smoke probe

The initial source hash `fa4db1090ec06d35dc205f412afa6b3e58880b629fb71c6f3be199fccdbff50e` failed independent review because the FP32 finite-difference smoke probe used an absolute step1e-3. On the exact KuaiRand source-model row2/task0 it produced -0.306516886 versus the autograd derivative -0.298707664 (2.61% difference). Reducing the FP32 step to1e-4 caused MovieLens row3/task1 cancellation error0.00117483 (2.69%). Neither numerical probe was used for training or a scientific performance decision.

The corrected numerical check uses a temporary FP64 copy of the same FP32-initialized model coefficients and the same four real source-loader rows, fixed step1e-5, and the design's FP64 tolerances rtol1e-3/atol1e-7. It also compares the full original FP32 input gradient against that copy at the unchanged FP32 tolerances rtol1e-2/atol1e-4. The ordinary FP32 forward/probe values still must agree exactly. Only the numerical probe uses FP64; model training, augmentation, the actual full-batch smoke step, hyperparameters and performance budgets retain FP32.

The fixed1e-5 FP64 central difference passed the real MovieLens and KuaiRand source models/rows, with maximum absolute FD errors3.50265e-11 and2.95627e-11. This correction was approved by the orchestrator as an engineering-only refinement; no HPO value or training formula changed.

The final `run.check_sensitivity` helper was executed on CPU using the audited loaders, seed2022, and `randperm(generator seed13022)[:4]`; both datasets passed. All original model parameters remained FP32 after the helper.

| Dataset | Actual source row IDs | Maximum FP64 FD error | Maximum full FP32/FP64 gradient difference |
|---|---|---:|---:|
| MovieLens | 478260,664109,529990,794020 | 3.50265e-11 | 2.60029e-7 |
| KuaiRand | 279554,1406375,1764115,1732943 | 2.95627e-11 | 5.18891e-7 |

No GPU, training update or Test prediction/metric was performed during this correction. The earlier five CPU checks continue to apply to unchanged intervention and training code.
