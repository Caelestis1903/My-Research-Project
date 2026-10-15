# Stage E6 implementation report

## Scope

Implemented only the Stage E6 design specification in
`experiments/reports/stage_E6_design.md`.  No prompt file was read during
implementation, and no E7 or Curriculum/E5 arm was added.  The implementation
does not launch the formal matrix and does not make a scientific conclusion.

## Design-to-code mapping

- `experiments/stage_E6_calibration.py`
  - closed six-arm registry and registry hash;
  - data-share and uniform targets;
  - `r_d = sum_t ||g^S_dt||_2` raw shared-gradient magnitude;
  - detached float64 EMA initialized from the first batch;
  - exact `beta = 2**(-1/5)`;
  - closed-form `c_d` with dtype epsilon and `sum_d c_d = D`;
  - target/magnitude-share/calibration-error diagnostics.
- `experiments/stage_E6_run.py`
  - isolated E3-compatible MovieLens/M3OE single-run entry point;
  - one forward, six cell losses, `autograd.grad` raw observer, detached
    controller coefficients, one main backward/optimizer step;
  - architecture objective/cadence and sampler inherited from the E3 path;
  - sample-EMA matched control updates the same EMA state but applies fixed
    `D*p_d` coefficients;
  - per-cell metrics, controller traces, shared-gradient traces, batch trace,
    RNG-safe test evaluation, checkpoint and nonfinite inventory.
- `experiments/stage_E6_orchestrator.py`
  - exact `6 arms × d0_large × 4 seeds = 24` task registry;
  - E3 oracle-only audit/reuse for 12 frozen references;
  - resume-safe artifact checks and per-seed common-support comparator;
  - allows only legal suffix censoring for unequal early stop and checks the
    sample/control loss identity.
- `experiments/stage_E6_report.py`
  - fixed 24-path raw collector;
  - seed/cell summary and ablation mapping with explicit missing/invalid
    status; no scientific comparison.
- `experiments/tests/test_stage_E6_calibration.py`
  - toy raw-gradient norm, closed-form target tracking, EMA, uniform-target,
    and no-`.grad` observer checks.
- `experiments/tests/test_stage_E6_matrix.py`
  - fixed matrix cardinality, common batch identity mismatch, and fixed-path
    report enumeration checks.

## Checks performed

1. `D:\Programs\anaconda3\python.exe -m py_compile` on all four E6
   implementation modules: PASS.
2. Focused tests with the torch-enabled environment:

   ```text
   pytest experiments/tests/test_stage_E6_calibration.py \
          experiments/tests/test_stage_E6_matrix.py -q
   8 passed
   ```

3. Orchestrator dry-run with the formal CUDA identity: E3 oracle audit
   `REUSED`, exactly 24 pending E6 tasks printed, no child launched.
4. One-epoch `adaptive_ema` CUDA smoke: 5 controller updates, 30 shared
   step rows, 6 metric rows, coefficient sum-domain 3, formal=false as
   required for a smoke, unexplained nonfinite count 0.
5. One-epoch `sample_static`/`sample_ema_control` CUDA matched-control smoke:
   final parameter hashes equal and maximum common train-loss difference 0;
   both inventories had unexplained nonfinite count 0.
6. Temporary smoke outputs were removed after inspection.  No formal E6 raw
   matrix was produced by the coder stage.

## Review #1 corrections

The first review found two reporting/diagnostic issues; both were corrected
locally without changing the controller or formal matrix:

- `coefficient_variation` now extracts the single D-length domain coefficient
  copy before computing `||c_k-c_{k-1}||_1/D`.  The first observation is blank
  (no transition), is excluded from all transition means, and is not replaced
  by a zero.
- `stage_E6_report.py` now calls the six-arm `compare_e6_block`, requires both
  a fresh VALID block and persisted `block_valid=true`, marks pending/invalid
  cross-arm identity as INVALID, and augments each valid run's
  `metrics_summary.csv` with the actual common-support flag and last-common
  epoch/test/transfer endpoint.  The runner's pre-block local endpoint is no
  longer a hard-coded zero; the report replaces it with cross-arm support.

Added focused regression coverage for the domain-only transition formula and
for invalid-block rejection plus valid-block last-common endpoint material.
After the corrections: `10 passed`; all E6 modules and tests compile; formal
orchestrator dry-run again reports E3 oracle `REUSED`, 24 fixed pending tasks,
and launches no child.

## Final review #1 follow-up correction

The shared-step logger now preserves the first transition-free
`coefficient_variation` as an empty value and writes a numeric value only for
`k>0` transitions.  A direct `_shared_step_rows` regression test covers both
cases; focused tests now report `11 passed`, and all E6 modules/tests compile.

## Constraints and limitations

- The formal 50-epoch matrix remains for the runner stage.
- The implementation reuses E3's already-audited data/model/oracle path;
  controller inputs are restricted to train `p_d` and detached shared raw
  magnitude.  Validation/test/oracle, stage, variance and SNR are logging or
  evaluation quantities only.
- This report records implementation evidence only.  Reviewer must inspect
  the design, code and this report before any formal run; main analysis must
  be written only after the runner stage.
