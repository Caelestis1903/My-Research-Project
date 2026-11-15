# Autonomous Research Runbook — TV-GRM

**Status:** PLANNING_ONLY / NOT_IMPLEMENTED / NOT_RUN. Target main. Plan date 2026-11-16.

## Decision
The original task's auxiliary-objective attribution and natural task-view experiments are already complete on current main. Do not rerun them. The next experiment tests a distinct optimizer-state route: compute the task-view minus clean gradient residual, accumulate it in a separate first-moment state, and inject it through the clean Adam second-moment preconditioner under a global update-norm cap. This is a candidate hypothesis, not an accepted algorithm.

## Preflight before any implementation/training
1. Read all main context/source files in 01_RESEARCH_CONTEXT.md.
2. Record current main HEAD, clean worktree, exact source/data/config hashes and NDP-A/CLEAN identities.
3. Confirm NDP-A/CLEAN evidence is audited; do not rerun.
4. Test native Codex Subagent capability: independent child context, actual `6-Astra-exhigh`, XHigh effort, specified repository reads.
5. Parent must be `6.1-sol-exhigh` (or exact runtime equivalent); record actual metadata.
6. If independent context/model/effort cannot be verified, set `AGENT_DISPATCH_BLOCKED` and stop before formal training. Do not simulate Astra using parent or a separate ChatGPT session.

## First Codex round
Read-only audit → isolated implementation → CPU tests → full-batch GPU smoke on both datasets → independent review → frozen five-seed matrix → validation-only analysis → report/main-context update/commit. Reuse CLEAN/NDP-A only after exact identity checks.

## Autonomous cycle
After report, review and main-context commit, automatically launch a fresh Astra child. Astra reads current main, full report, raw paired results and diagnostics; explores at least three distinct mechanisms; writes a complete Codex plan under cycle_01; returns to parent. Parent checks hard completeness only. PASS launches next experiment; REVISE returns once; STOP/BLOCKED recorded. Maximum two Astra-driven cycles after initial round. No user confirmation inside the loop. Pro exits after this initial plan and never participates again.

## Invariants
MovieLens + KuaiRand; M3oE fixed1; 3 domains × 2 tasks; natural sampling; 50 epochs; batch 40960; FP32; seeds exactly 2022–2026; validation six-cell Macro AUC earliest strict max, online/EMA independently selected; AUC primary, LogLoss separate; Test sealed until freeze. **Small domain is not difficult domain**; no experiment/analysis about this boundary. No architecture/gate/expert/fusion/selective-sharing/data-split changes.

`NOT_TRIGGERED`, `ENGINEERING_BLOCKED`, `NO_CONTINUATION_SIGNAL` and `PASS` must remain distinct. Failure of TV-GRM does not terminate the research line.
