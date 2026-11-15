# Codex Implementation Specification

Create isolated package `experiments/task_view_gradient_residual_momentum_20261116/`. Do not modify historical source, vendor model, reports, data, checkpoints, metrics ledger, or merge 26-11-15.

Read Stage2 NDP-A core/protocol/runner/tables; reuse only audited pure helpers and current-main model/data/EMA/evaluation utilities.

Implement clean Adam m/v from clean gradient + inherited coupled WD; separate q from (g_v-g_c) without WD; consistent bias correction; clean v preconditioner; global cap; exactly one parameter update and one EMA update per batch; complete resumable state (model, m/v, q, step, EMA, RNG, loader).

Compute gradients at identical theta with no parameter mutation. Preserve grad=None semantics; parameters with undefined clean Adam state receive no residual-only update. Apply coupled WD once. SHUFFLE uses deterministic parameter-space Rademacher vector matched to residual norm.

Tests: clean loss/gradient equivalence; task-view gradient; residual arithmetic; moment recurrence/bias correction; cap; gamma=0 state equivalence; grad=None; no double WD; resume determinism; finite differences/hooks; full-batch GPU smoke both datasets. Hash source/config/data/selection.
