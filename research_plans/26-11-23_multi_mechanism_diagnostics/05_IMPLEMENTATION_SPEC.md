# 05 — Codex 实现规格

## 1. 新目录

源码：
experiments/multi_mechanism_diagnostics_20261123/

工件：
artifacts/multi_mechanism_diagnostics_20261123/

报告：
research_reports/multi_mechanism_diagnostics_20261123/

不得修改原始 M3oE source、D2 source 或历史报告。允许从 D2 分支复用经过审查的纯函数，但必须记录来源 blob/hash；不得直接把 D2 scheduler/arm 当作新机制实现。

## 2. 模块

- protocol.py：读取并冻结 effective config、seed、dataset、split。
- source_adapter.py：canonical fixed1 model/data/EMA/Adam adapter。
- parameter_registry.py：运行时 U/P/FROZEN 分组和 support。
- state_store.py：state save/load、hash、atomic commit、readback。
- panels.py：Train calibration、V_A/V_B row/pair panels。
- ranking.py：six-cell AUC、LogLoss、pair margins、pair correction/destruction。
- m1_consistency.py：per-cell g_U、g_J、L_cons、二阶 P update。
- m2_attribution.py：per-row/task gradient contribution、virtual Adam step、calibration margin influence proxy。
- m3_interaction.py：固定总 batch 的 A/B group interaction arms。
- m4_m5_audit.py：feature semantics and label-preserving transformation audit only.
- diagnostics.py：M6 output sensitivity and prediction delta.
- scheduler.py：state machine, admission, retry, worker ownership.
- statistics.py：paired seed reduction, CIs, multiple-comparison registry.
- reports.py：CSV/JSON/Markdown outputs.
- cli.py：audit, smoke, baseline, screen, confirm, short-run, report。

## 3. 核心接口

~~~python
def load_protocol(dataset: str, seed: int) -> Protocol:
    """Return exact frozen D2-compatible effective protocol."""

def capture_state(model, optimizer, ema, rng, loader, epoch, step) -> StateReceipt:
    """Capture complete resumable state and identity hashes."""

def restore_state(receipt: StateReceipt) -> RuntimeState:
    """Restore and verify predictions, optimizer state and next batch IDs."""

def compute_cell_shared_grads(model, batch, cell_ids, create_graph=True):
    """Return per-cell gradients on the runtime-verified U parameter set."""

def consistency_objective(cell_grads, joint_grad, weights):
    """Scalar cosine inconsistency objective."""

def private_consistency_step(state, batch, inner_lr, inner_steps):
    """Update P only; prove U and U-Adam state are unchanged."""

def sample_task_scores(state, calibration_pairs, sampled_rows):
    """Return per-row/task/cell local ranking-margin impact proxy."""

def run_matched_fork(anchor, arm, batch_a, batch_b) -> ForkReceipt:
    """Clone, execute, evaluate and atomically persist one arm."""

def compute_group_interaction(anchor, group_a, group_b, batch_size):
    """Run equal-size A-only/B-only/A-B/A-A/B-B arms."""

def analyze_and_decide(manifest, outputs) -> DecisionRecord:
    """Apply frozen stage gates; no post-hoc arm selection."""
~~~

## 4. M1 algorithm semantics

1. Compute per-cell losses from the same canonical forward and same Train batch.
2. Obtain g_U,c with create_graph=True.
3. Obtain g_U,J using the canonical joint objective and same batch population.
4. Build L_cons over runtime U registry.
5. Compute ∇_P L_cons; do not materialize Hessian.
6. Apply exactly one native Adam step to P only; preserve U and U optimizer state bitwise.
7. Recompute joint loss on batch B and perform one canonical joint step.
8. Record P displacement, U displacement, m/v/step transitions, gradient cosine matrix and pairwise ranking changes.

M1-2 P-BCE control executes one private-only BCE step then one joint step. M1-3 uses a deterministic permutation of cell-to-gradient association, frozen before outcomes. All arms use identical anchor, A/B row IDs and optimizer state.

## 5. M2 implementation

- Select fixed sample panel from Train only.
- Compute per-row/task gradient on U using exact task-specific BCE contribution.
- Compute virtual Adam update using anchor m/v/step and effective weight decay; do not mutate live optimizer.
- Compute train-calibration pairwise margin gradients per cell.
- Score = negative margin-gradient dot virtual update.
- Normalize weights within each domain-task to mean 1; clip and log effective total mass.
- M2-2 averages score across tasks before weighting; M2-3 shuffles score within domain.
- All arms retain identical row IDs and batch size.

If exact per-row gradients are too expensive, S0 may reduce fixed panel size only before S1 and record the reduced estimand. It may not switch to loss ranking or gradient norm ranking.

## 6. M3 implementation

Construct equal-sized batch groups A, B and A/B mixture. Use same anchor optimizer state, same batch size and same step count. Record interaction residual and all five arm outcomes. If group support fails, do not run.

## 7. Output contract

Each run writes:
- run_manifest.json;
- config.json;
- source_receipt.json;
- state_parent.json;
- batch_identity.json;
- action_receipt.json;
- metrics.json;
- costs.json;
- COMPLETE_VERIFIED.json or failure.json.

Global outputs:
- RUN_MANIFEST.json;
- SEED_MANIFEST.csv;
- BASELINE_MANIFEST.csv;
- SNAPSHOT_MANIFEST.csv;
- ADMISSIONS.jsonl;
- EXITS.jsonl;
- FORK_ENDPOINTS.csv;
- CELL_METRICS.csv;
- PAIR_TRANSITIONS.csv;
- M1_GRADIENT_CONSISTENCY.csv;
- M2_SAMPLE_TASK_SCORES.csv;
- M3_GROUP_INTERACTIONS.csv;
- COSTS.csv;
- DECISION_RECORD.json.

## 8. Mandatory tests

- U/P/FROZEN coverage, disjointness or documented overlap;
- model forward parity and six-cell output ordering;
- source vs adapter loss parity;
- Adam None-vs-zero semantics;
- P-only step leaves U parameters and U optimizer state unchanged;
- consistency objective finite gradients and correct sign;
- state save/restore prediction parity;
- next batch row IDs and RNG parity;
- M2 virtual Adam does not mutate optimizer;
- M2 weight mass and row exposure parity;
- M3 fixed batch-size/exposure parity;
- Validation panel disjointness;
- Test remains inaccessible;
- deterministic replay of one fork;
- OOM/retry and atomic-write recovery;
- metrics and statistics recomputation.

## 9. Review gate

Before any S1 baseline training, independent Reviewer must PASS:
- design conformance;
- code/source hash closure;
- model/data fidelity;
- M1 second-order correctness;
- M2 sample-score correctness;
- M3 budget parity;
- state restore;
- Test sealing;
- resource smoke.

A failed review blocks run. Coder self-review is not independent review.


## 10. 条件性 CGPA trainer

新增 cgpa_train.py，仅当 M1 S2/S3 PASS 后启用。

~~~python
for batch in train_loader:
    # step 1: P consistency adaptation
    freeze_update(U)                 # U remains in graph as differentiation target
    cell_grads = per_cell_grad_U(create_graph=True)
    joint_grad = joint_grad_U(create_graph=True)
    loss_cons = consistency_objective(cell_grads, joint_grad)
    optimizer.zero_grad(set_to_none=True)
    loss_cons.backward()
    assert all(p.grad is None for p in U)
    optimizer.step()                 # P state only

    # step 2: canonical joint training
    unfreeze_all()
    optimizer.zero_grad(set_to_none=True)
    joint_bce(batch).backward()
    optimizer.step()

    ema.update_once()                # one tick per batch, unchanged rho
~~~

Implementation must prove U tensor and U Adam state unchanged after step 1; P m/v/step changes as expected; EMA tick count increments exactly once per batch. P-BCE-prestep control uses the same two-step sequence but replaces loss_cons with private BCE. CGPA and control share batch IDs and initialization.

Do not update EMA after the private inner step. Do not change rho, batch size, loss reduction, or optimizer configuration. This candidate is not a direct implementation of ConsMTL's full bilevel method; report the exact local consistency objective and its computational cost without claiming theoretical equivalence.
