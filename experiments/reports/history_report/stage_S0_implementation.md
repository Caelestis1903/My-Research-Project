# CODER IMPLEMENTATION REPORT — Stage S0

## Scope

Implemented only the approved `DESIGN SPECIFICATION — Stage 0: Matched
Momentum Closure` in `experiments/reports/stage_S0_design.md`.  No Stage 1+
algorithm, Curriculum mechanism, controller, gamma schedule, bias-corrected
comp1, stacked momentum, or new optimizer was added.  No formal run and no
research-result analysis was performed.

## Modified/added files

- `experiments/stage_S0_run.py`
- `experiments/stage_S0_protocol.py`
- `experiments/stage_S0_orchestrator.py`
- `experiments/stage_S0_report.py`
- `experiments/tests/test_stage_S0_matrix.py`
- `experiments/tests/test_stage_S0_momentum.py`
- `experiments/tests/test_stage_S0_pairing.py`
- `experiments/reports/stage_S0_implementation.md`

The approved design file was read-only.  Historical raw results, E1–E7
artifacts, core M3OE definitions, and archived reports were not modified.

## Design-to-implementation mapping

### Fixed task registry and controls

- `build_tasks()` creates exactly `7 arms × 3 rotations × 4 seeds = 84`
  unique fresh joint tasks, with 12 tasks per arm.
- The arm registry is closed to `sample_adam`, `comp1_domain`,
  `uniform_090`, `uniform_p_tau`, `uniform_eq_tau`, `adam_p_tau`, and
  `adam_eq_tau`.
- Formal defaults are locked to MovieLens, L/M/S=`150000/34500/3000`, batch
  size 40960, 50 epochs, learning rate 0.01, DARTS learning rate 0.001,
  weight decay `1e-5`, seeds 2022–2025, and the three role rotations.

### Objective and optimizer arms

- Every main and architecture update uses the same task-major
  sample-proportional objective `w_d = D*p_d` applied to six cell-mean BCE
  terms.
- B/C use the same forward, per-domain `autograd.grad`, EMA-buffer update,
  full-objective private backward, and optimizer path.  Only the gamma vector
  changes.  Their shared Adam beta1 is 0; private beta1 remains 0.9.
- A/D use the same no-EMA path.  Only shared beta1 changes among 0.9,
  0.9431818, and 0.980453; private beta1 remains 0.9.
- `DomainEMA` updates each domain buffer once per optimizer step and replaces
  shared gradients with the sum of the three buffers.
- The runner computes and records tau, half-life, and effective sample length
  for every gamma.

### Budget, checkpoints, pairing, and persistence

- Training always completes all 50 epochs.  Patience 10 is simulated as
  shadow state and never terminates training.
- The selected checkpoint is the first validation macro-AUC argmax within
  the shadow-visible window.  Epoch 49 and common epochs 9/19/29/39/49 are
  always retained.
- The manifest records initialization, split/selected-row, batch-order, code,
  and final-state hashes plus examples processed and main/architecture/
  scheduler/EMA update counts.
- RNG state is saved/restored around extra test evaluation and the restoration
  digest is persisted per epoch.
- Selected/final model states, Adam state, and (for B/C) EMA buffer state are
  saved separately.
- Matching oracle reuse goes through the existing E3 artifact validator and
  requires seed, setting, role sizes, dataset/device identity, and remapped
  per-domain train/validation/test row fingerprints.  Formal runs fail closed
  on incomplete oracle coverage.
- The orchestrator validates each run and all seven-arm blocks.  Completed
  successes are not overwritten.  Failed attempts are retained and moved to
  a timestamped `failed_attempts` archive only when `--rerun_failed` is
  explicitly supplied; the append-only master retains every attempt row.

### Logging and report aggregation

- `metrics_epoch.csv` stores weighted train loss, validation/test AUC and
  LogLoss, matched-oracle transfer deltas, all four NT indicators, roles,
  physical domains, tasks, and checkpoint views for every epoch/cell.
- `momentum_step.csv` stores raw per-domain shared-gradient norm, EMA norm,
  aggregate gradient norm, actual shared update norm, Adam moment norms,
  EMA/raw ratio, EMA counts, startup coefficient, and learning rate.
- `batch_trace.csv` and manifest hashes make paired batch order auditable.
- The report emits macro and explicitly named sample-weighted metrics,
  worst-domain/worst-cell values, NT rates, selected/final/common-epoch
  trajectories, role gains, and B comparisons against A, C1, D1, C_val, and
  D_val.
- C_val/D_val are selected only from selected-checkpoint validation macro-AUC
  inside each `(rotation, seed)` block.  Test metrics are not read by the
  selector.
- Pair aggregation first averages the three rotations within each seed, then
  reports mean, median, sample SD, range, and direction count over the four
  seed units.  Rotation summaries remain separate and are not treated as 12
  independent seeds.

## Basic checks performed

1. Python byte-code compilation for all three S0 modules and all three S0
   tests: **PASS**.
2. Matrix, pairing, fail-fast, resume, and validation-selection tests:
   **11 passed** in the base environment.
   - exact 84 unique tasks and arm/rotation/seed cardinalities;
   - formal command budget and absence of forbidden mechanisms;
   - C_val/D_val validation-only selection despite adversarial test ranking;
   - deterministic pre-registered tie order.
   - explicit rejection of a seven-arm block with a mismatched batch-order
     hash.
3. Formal orchestrator dry-run: **PASS**.  It reported exactly
   `7 arms × 3 rotations × 4 seeds = 84` and printed one correctly parameterized
   task when limited to one; it launched no child run.
4. The base Python still skips the torch-dependent pytest file, but the
   reviewer-identified `EDDAenv` PyTorch 2.0.0 environment was used for direct
   tensor checks and the corrected real tiny EMA-arm smoke described below.

## Known limitations / deviations

- The corrected EMA path has one real tiny CUDA smoke, but formal-scale
  50-epoch execution and the complete formal artifact validator remain for
  REVIEW/RUN.  The smoke was explicitly nonformal and was removed afterward.
- The formal 84-run matrix is intentionally unexecuted.
- `sample_weighted_auc` is an explicitly named weighted average of cell AUCs;
  it is not mislabeled as a prediction-level pooled AUC.  Raw per-cell metrics
  are retained so a true pooled metric can be added only if a future approved
  protocol supplies prediction-level artifacts.
- No scientific effect, KEEP/STOP decision, or Stage 1 recommendation is made
  in this implementation report.

## Review #1 blocking corrections

The first reviewer verdict was `FAIL`.  The following corrections implement
only the requested Stage S0 protocol fixes.

### C1 — non-finite values and disconnected gradients

- Added fail-fast checks for every cell loss, combined objective, expected
  shared gradient, EMA buffer, aggregate shared gradient, optimizer moment,
  post-step model parameter, architecture gradient/loss/state, evaluation
  prediction, and all six per-cell evaluation metrics.
- `DomainEMA.update()` now raises on an expected shared `None` gradient instead
  of silently replacing it with zero.
- The report's numeric helpers now raise on missing/NaN/Inf values.  They no
  longer filter values and silently reduce `n`.
- Formal validation requires 300 unique finite epoch-cell metric rows and
  finite raw diagnostic rows.  A focused regression injects a NaN test AUC
  and confirms explicit rejection.

### C2 — formal protocol lock

- Added the single-source `stage_S0_protocol.py` registry and exact
  `FORMAL_PROTOCOL` covering device, role sizes, batch size, epochs, both
  learning rates, weight decay, expert count, shadow patience,
  instrumentation, and mandatory oracle coverage.
- The launcher rejects any critical CLI mismatch before launching a child.
- Each config/manifest stores the protocol snapshot, mismatch map, and
  `formal_valid` flag.  `validate_run()` independently compares every field
  and rejects instrumentation-off/nonformal artifacts.
- Formal validation now checks exact optimizer/example/architecture budgets,
  six-cell support, raw row cardinality, artifact non-emptiness, optimizer
  group betas/steps, EMA update counts, oracle completion, and RNG restoration.

### C3 — architecture-update batch pairing

- Every architecture-update batch is now recorded in `batch_trace.csv` with
  its kind, fingerprint, size, domain counts, and finite weighted loss.
- Manifest fields now include architecture batch count, ordered hash, examples
  processed, and optimizer steps.
- Architecture evidence fields are part of the seven-arm `PAIR_FIELDS` audit.
  A regression test changes only the architecture-batch hash and confirms the
  block is rejected.

### C4 — strict and complete reporting

- Report aggregation is allowed only for blocks that pass the full seven-arm
  audit; validation family selection additionally requires six unique finite
  selected cells.
- Added selected-checkpoint B-vs-A subgroup summaries for L/M/S role, task
  0/1, and physical domain 0/1/2.
- Added a 24-row historical-vs-fresh A/B audit.  Historical artifacts are
  descriptive only and are permanently marked `included_in_formal_gate=0`;
  missing/invalid historical artifacts remain explicit instead of being
  substituted.

### M1 — resume and failure handling

- Current artifacts are now the sole source of completion truth; stale `OK`
  rows in the append-only master cannot cause a missing/corrupt run to be
  skipped.
- Existing failed/invalid artifacts block preflight unless explicit
  `--rerun_failed` is supplied, after which the old directory is timestamped
  and archived before retry.
- `completed_count` now counts current formal-valid artifacts only.
- The orchestrator returns immediately after the first failed child and
  persists the failure row and fail-fast queue manifest.  A two-task mocked
  regression confirms only the first child is invoked.
- A runner Python/PyTorch/CUDA preflight executes before the first real child,
  preventing 84 repeated dependency failures.

### M2 — complete optimizer state

- `architecture_optimizer_state.pth` is now saved and required alongside the
  main optimizer state.
- Manifest optimizer audits record group names, beta values, parameter counts,
  state parameter counts, and min/max step.  Formal validation requires shared
  and private beta contracts and exact main/architecture state steps.

### M3 — added verification

- Base-environment focused tests after correction: **11 passed**.
- PyTorch 2.0.0 environment manual tensor checks: disconnected and NaN shared
  gradients fail fast; domain EMA aggregation and split-group Adam update
  equivalence pass.
- One nonformal, one-epoch real `comp1_domain` tiny smoke completed with one
  main step, one architecture step, three EMA updates, three momentum rows,
  two batch rows, finite optimizer audits, an architecture batch hash, and all
  four raw state artifacts.  It was marked `formal_valid=false` as required.
  The temporary smoke directory was removed after inspection.
- Corrected orchestrator dry-run still reports exactly 84 tasks and launches
  no child.

No formal Stage S0 run was launched and no scientific result was analyzed
during these corrections.
