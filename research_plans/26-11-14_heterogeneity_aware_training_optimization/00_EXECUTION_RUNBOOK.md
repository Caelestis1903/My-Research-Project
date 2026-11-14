# Execution Runbook — Heterogeneity-aware Training Optimization

**Status:** PLANNING_ONLY / NOT_IMPLEMENTED / NOT_RUN  
**Research base:** main@01979f6588935f713029b2b5e04557de9b8c153f  
**Scope:** fixed M3oE training platform; no architecture or selective-sharing changes.

## 1. Research decision

The primary candidate is **Domain–Task Pairwise Gradient-Nullspace Augmentation (DT-PGNA)**. It constructs a hard, task-specific latent perturbation from a target-task positive/negative pair, while constraining first-order changes to the other task's logits. It changes augmented training inputs, not the optimizer's parameter-gradient aggregation rule.

The fallback is **Task-Gradient Augmentation (TGA)**: the same target-task adversarial perturbation without the non-target-logit nullspace constraint. It is simpler and cheaper. The comparison decides whether the cross-task preservation constraint earns its added cost.

A label-matched pairwise auxiliary loss is not retained as a new route: the repository already implemented PAIR/P01 and associated conditions. Ordinary PCGrad, magnitude balancing, dynamic EMA, and gradient-covariance variants are not reopened.

## 2. Mandatory preflight

Codex must:
1. Verify checked-out branch, clean/dirty Git status, commit SHA, Python/PyTorch/CUDA versions, GPU availability and dataset hashes.
2. Read AGENT.md, EXP_AI_CONTEXT.md, Experimental_Global_Summary.md, Experimental_Stable_Findings.md, Experimental_Open_Questions.md, MDMT_ALGORITHM_METRICS_LEDGER.md, and this directory.
3. Confirm start.md and AGENTS.md are absent from current main; do not invent their contents. AGENT.md is the actual root engineering contract.
4. Recheck that this directory does not already exist before writing any experiment output. Do not modify historical code or reports.
5. Inspect the actual fixed1 model forward path and identify the embedding tensor and pre-sigmoid task scores. Do not assume a hook or tensor name from this plan.
6. Produce a read-only source map and implementation plan before coding.

## 3. Execution phases

| Phase | Work | Gate |
|---|---|---|
| 0 | Code/data/protocol audit; exact output and embedding access | No unresolved semantic mismatch |
| 1 | CPU unit tests + two-dataset GPU smoke; verify projection and label preservation | PASS all correctness checks |
| 2 | Five-seed low-cost pilot for TGA and DT-PGNA, with fixed pair budget | Candidate must improve the predeclared validation ranking-risk diagnostic without violating cell constraints; screening only |
| 3 | Lightweight epsilon search, paired five seeds, both datasets | Freeze one common epsilon or predeclared per-dataset rule |
| 4 | Full five-seed training confirmation for TGA and DT-PGNA plus controls | Frozen configurations; test only after freeze |
| 5 | Independent review, report, and metric-ledger update | All identities, costs, and metrics reconcile |

Do not run Phase 4 automatically if Phase 2 shows the perturbation is numerically ineffective, but do not call a route failed when the stage was not triggered. The alternative route is activated only under the decision rules in 10_EVALUATION_AND_DECISION.md.

## 4. Core controls

- Clean M3oE fixed1 online and fixed-EMA shadow.
- Gaussian D0 + fixed EMA as an existing augmentation comparator, only where source identity and protocol are reusable.
- TGA: target-task gradient perturbation without cross-task projection.
- DT-PGNA: projected perturbation preserving other-task logits to first order.
- Random-direction perturbation matched to DT-PGNA norm.
- Isotropic Gaussian perturbation matched to DT-PGNA per-sample norm.

The pairwise loss itself is not added. The base objective remains the repository's clean multi-task BCE; the augmented branch adds target-task BCE only.

## 5. Deliverables

Create immutable run_manifest.json and source_receipt.json; unit/smoke reports; HPO selection record and frozen config; per-seed validation curves and selected checkpoint records; six-cell AUC/LogLoss and macro/traffic/pooled summaries; online and fixed-EMA results; perturbation norm, target-margin change, non-target-logit drift, pair counts, wall-time and peak-memory diagnostics; paired five-seed statistics and machine-readable decision JSON; final report with NOT_TRIGGERED / BLOCKED / FAILED / COMPLETED distinctions.

No algorithm implementation or training is authorized by this planning commit.
