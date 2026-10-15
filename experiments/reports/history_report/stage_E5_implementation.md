# Stage E5 Implementation Report

## Scope

Implemented only the approved Stage E5 specification in
`experiments/reports/stage_E5_design.md`.  The implementation is isolated from
the E3/E4 shared files and does not implement E6 or E7, adaptive control,
momentum/EMA, role reversal, a new sampler, a new router, or a schedule grid.

The formal queue is fixed at `5 arms × 4 seeds = 20` runs on MovieLens-1M,
`r50`, canonical `d0_large`, with 18 epochs.  The E3 frozen single-domain
oracle artifacts are resolved and identity-audited; no oracle retraining is
performed by the E5 orchestrator.

## Implementation mapping

- `experiments/stage_E5_schedule.py`
  - closed arm registry: `static_sample`, `static_interior`,
    `static_matched`, `forward`, `reverse`;
  - exact epoch 0–17 alpha functions and early/middle/late labels;
  - E3 domain-share normalisation for arbitrary scheduled alpha values;
  - alpha/coefficient and optimizer-step-weighted exposure tables;
  - 20-task matrix validator and per-seed five-arm batch-support comparator.
- `experiments/stage_E5_run.py`
  - reuses E3 data/model/optimizer/architecture-update and shared observer
    paths;
  - changes only the main six-cell loss coefficient vector once per epoch;
  - records schedule, metrics, contribution, batch, RNG-restore, oracle and
    non-finite inventories;
  - runs the requested fixed epoch budget and never invokes early stopping.
- `experiments/stage_E5_orchestrator.py`
  - audits the 12 E3 oracle references, enumerates exactly 20 E5 tasks, and
    provides resume-safe artifact/block validation and dry-run support.
- `experiments/stage_E5_report.py`
  - enumerates all 20 expected paths and writes structured seed/cell/view raw
    summaries plus completeness/block/oracle inventories; it does not make a
    scientific comparison or GO/STOP decision.
- `experiments/tests/test_stage_E5_schedule.py`
  - boundary/complementary schedule, E3 coefficient sum/task-copy, toy loss
    and backward-gradient, and exposure checks.
- `experiments/tests/test_stage_E5_matrix.py`
  - exact 20-task queue and ordered batch-support mismatch/acceptance checks.

## Checks performed

- `py_compile` passed for all four implementation modules.
- Focused pytest passed: `7 passed` for the two E5 test files.
- Orchestrator dry-run passed with oracle audit `12/12 reused`, fixed matrix
  `5 arms × 4 seeds = 20`, and `20` pending; no child run was launched by the
  dry-run.
- One 1-epoch CUDA `forward` smoke passed.  It produced all required raw
  files, logged alpha `0.75`, coefficient sum `6`, five main steps, and a
  non-finite inventory with zero unexplained entries.  The temporary smoke
  directory was removed after inspection.

## Limitations and handoff

- No formal E5 matrix was launched and no performance or mechanism conclusion
  is made here.  The runner/reporter deliberately leave cross-arm inference to
  REVIEW, RUN, and main analysis.
- The formal `d0_large` setting is enforced by the E5 runner.  Nonformal short
  runs are supported only for smoke/debug checks and are not accepted by the
  formal orchestrator.
- `model.pth` stores the validation-selected state, while fixed-final metrics
  remain in the epoch trajectory; both views are retained in the summary.
- The only shared code used is read-only reuse of E3/E1 APIs; no core model or
  trainer file was modified.

## Formal launch command

After reviewer approval, runner may invoke:

```text
python experiments/stage_E5_orchestrator.py
```

The command uses the fixed protocol and may be resumed through the generated
`experiments/stage_E5_runs.csv` and `experiments/stage_E5_raw_results/` paths.
