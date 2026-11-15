# Execution Runbook — Auxiliary Objective Attribution and Natural-Distribution-Preserving Task Augmentation

**Status:** PLANNING_ONLY / NOT_IMPLEMENTED / NOT_RUN  
**Governing seed pool:** 2022, 2023, 2024, 2025, 2026 only  
**Scope:** training optimization only; no architecture or selective-sharing changes

## 1. Mission and first decision

The 2026-11-15 validation pilot found no continuation signal for DT-PGNA (A), TGA-Pair (B), or TGA-Example (C) at alpha=.03, K=128, lambda_aug=.25. This is not a Test-based formal performance rejection: HPO, confirmation, Test and conditional ablations were NOT_TRIGGERED.

The most decision-relevant unresolved issue is that the candidate arms changed both the input view and the auxiliary training objective. The next task is therefore to isolate the auxiliary BCE effect before designing another perturbation.

**First execution is Stage 0 audit, then Stage 1 zero-perturbation attribution. Do not implement Stage 2 augmentation before Stage 1 decision.**

## 2. Governing constraints

- Every scientific performance comparison, HPO comparison and algorithm decision uses exactly seeds 2022–2026. No 2027–2031 or substitute pool.
- Five full seeds are required for each dataset/configuration used in a performance decision. Smoke tests are engineering-only.
- MovieLens and KuaiRand; M3oE fixed1; 3 domains × 2 tasks; 50 epochs; batch size 40960; inherited data split, natural sampling, Adam and fixed EMA.
- Validation checkpoint: earliest strict maximum of six-cell Macro AUC; online and fixed-EMA readouts selected independently.
- Test remains sealed until a candidate/configuration and checkpoint protocol are frozen.
- AUC primary; LogLoss independent. Report six cells, Macro, Traffic and Pooled.
- Formal published-model gate, if a candidate reaches Test: mean paired Test Macro AUC gain >= +0.003 against the strongest applicable published model and every cell mean delta >= -0.001. No requirement that every seed be positive. Report mean, sample SD, paired 95% t interval and positive-seed count.
- Fixed EMA is an internal reference. Report total candidate+EMA vs clean online separately from the incremental component effect vs clean fixed EMA.
- Small domain is not difficult domain. Do not analyze or test this fixed boundary; never use domain size as a difficulty proxy.
- Do not reopen PAIR/ECCM, PCGrad, generic gradient balancing, dynamic EMA, PUB migration or architecture/knowledge-sharing routes.
- Do not modify historical source, reports, metrics ledger, model files, data, or checkpoints. New implementation and outputs belong in unique new directories only.

## 3. Stage sequence

| Stage | Work | Runs | Advance condition |
|---|---|---:|---|
| 0 | Read-only main/branch/source/data/protocol audit | 0 | Exact source and result identities recorded; no unresolved semantic mismatch |
| 1 | Z-PAIR and Z-EXAMPLE, delta=0 | 20 new full runs maximum | Interpret auxiliary-target effects and matched perturbation increments |
| 2 | Natural-distribution-preserving task augmentation | Conditional; start with 2 arms × 2 datasets × 5 seeds = 20 | Stage 1 identifies a viable auxiliary objective and the chosen objective is frozen |
| 3 | Bounded HPO | Conditional; 5 seeds for every candidate value | At least one Stage 2 candidate shows a positive, decision-relevant validation increment |
| 4 | Frozen five-seed evaluation | Conditional | Freeze code/config/data/selection before Test |
| 5 | Conditional mechanism ablation and independent review | Only if it changes the algorithmic claim | One specific unresolved mechanism could change the decision |

Stage 1 may reuse CLEAN, B and C from the 2026-11-15 branch only after exact identity verification. Z-PAIR and Z-EXAMPLE are new. Maximum additional Stage 1 work is 20 trajectories / 1000 epochs; if reuse fails, rerun only the necessary matched arms and report the expanded count before dispatch.

## 4. Stage 1 decision in one sentence

Determine whether the current loss augmentation is itself harmful, and whether the perturbation adds or subtracts value conditional on exactly the same sampled auxiliary examples, labels, task cells, weights and loss normalization.

## 5. Execution order for Codex

1. Read this directory and all referenced main context before touching code.
2. Verify main HEAD and the 26-11-15 branch status; do not merge the branch as part of this task.
3. Verify exact 26-11-15 experiment source, raw data hashes, resolved protocol, configs, seeds, training logs and report tables.
4. Produce a read-only audit and reuse matrix. No training until identity is adjudicated.
5. Implement only Stage 1 support-index recording/replay and zero-delta arms; leave the original experiment source untouched.
6. Run compact CPU correctness tests and full-batch engineering smoke on each dataset. These are not performance evidence.
7. Execute the five-seed Stage 1 matrix; no HPO and no Test.
8. Generate seed-paired validation results and decision JSON; independently review the evidence and exact run identities.
9. Only if the decision tree triggers Stage 2, implement the selected natural-distribution-preserving objective in a new experiment module.
10. Complete bounded HPO and frozen evaluation only when their gates are satisfied.

## 6. Stop and failure semantics

- NOT_TRIGGERED is not a negative result.
- ENGINEERING_BLOCKED is not a scientific failure.
- A validation pilot may prioritize a route but is not a formal Test rejection.
- A candidate that loses to CLEAN in five-seed validation should not receive an expanded alpha/K/lambda search merely to rescue it.
- If the auxiliary objective is materially harmful at delta=0, do not continue to tune its perturbation direction. First correct the objective semantics or stop that objective family.
- If zero-delta controls are neutral/positive but perturbation increments are negative, stop the current perturbation recipe; do not infer that all task-sensitive input augmentation is impossible.
- Prefer the simplest design that achieves the same objective with fewer assumptions and less compute.
