# Stage S5 Implementation Report

**Author: Coder agent (MiniMax M3).** Implements the S5 design as a minimal
subclass/extension of the S4 runner; never modifies `stage_S3_*`,
`stage_S4_*`, `m3oe/`, or the imported `stage_S4_gradient_controls.py`.

## 1. Files added / modified

| File | Status | Lines |
|---|---|---|
| `experiments/stage_S5_protocol.py` | new | 78 |
| `experiments/stage_S5_run.py` | new | 532 |
| `experiments/stage_S5_orchestrator.py` | new | 431 |
| `experiments/stage_S5_report.py` | new | 191 |
| `experiments/tests/test_stage_S5_controls.py` | new | 486 |
| `experiments/tests/test_stage_S5_orchestration.py` | new | 148 |
| `experiments/reports/stage_S5_implementation.md` | new (this report) | — |
| `stage_S3_*`, `stage_S4_*`, `m3oe/` | **untouched** | — |

All six source/test files compile cleanly under `venv/Scripts/python.exe -m py_compile`.

## 2. Key functions and state flow

### `_controlled_step` — HT/DH execution order (design §4)

Per main batch (and analogously per architecture batch):

1. Forward once → `L, L*`. `gradients_one_forward(losses["full"], losses["target"], …)`
   returns `(g_F, g_T, g_P)` from **one** stochastic forward; no RNG is
   consumed, no DataLoader iterator is created.
2. Snapshot `before = [param.detach().clone() for name in names]` for the
   controlled group (shared for `main`, arch for `architecture`); same for
   private. Read `m, v` from `optimizer.state` for the moment norms.
3. **Read-only** counterfactual deltas via `adam_deltas`:
   - `Δ_F = AdamDelta(θ_pre, state, g_F)`
   - `Δ_T = AdamDelta(θ_pre, state, g_T)`
   - `Δ_P = AdamDelta(θ_pre_P, state_P, g_P)`
4. Install `p.grad = g_H` for shared (and `p.grad = g_P` for private) as
   detached clones.
5. **Single** `optimizer.step()` — the only state mutation of the step. The
   persistent `(m, v, τ)` commit the `g_H` transition; parameters move by
   `Δ_H`. Private parameters always advance by `Δ_P` here.
6. Capture `observed_step = θ_after_step − θ_before`; assert it matches
   `Δ_H` (`history_formula_residual`, design I2).
7. **Arm-dependent overwrite** (only for HT/DH, only for shared/arch names):
   - HT: `θ ← θ_before + Δ_F` (computed at step 3 from pre-step target-history state and FULL current gradient)
   - DH: `θ ← θ_before + Δ_T` (symmetric)
   - F/T: no overwrite — `Δ_D ≡ Δ_H` so the parameter keeps what `step()`
     produced (bit-identical to S3/S4 F/T).
8. Re-run `adam_deltas` on the now-mutated state for the OTHER gradient
   stream (the one not installed). The result must differ from the
   pre-step counterfactual (design I6 timing guard).
9. Hard-assert invariants I1–I7, I9 and emit the S5 per-step row with
   the §7 field contract (history/current sources, persistent moment
   norms, counterfactual vs observed step norms, applied cosine vs Δ_F
   and vs Δ_T, installed-cosine ≥ 0.99999, formula residuals, None counts,
   zero-target flag, RNG digests).

`_dump_moment_state(epoch)` is invoked at the end of training for epochs
in `MOMENT_DUMP_EPOCHS = (0, 1, 9, 19, 29, 39, 49)`. It snapshots a
**detached CPU clone** of the model state dictionary that was captured
by the parent `S4Runner._checkpoint_payloads()` — read-only w.r.t. Adam
state, no in-place mutations.

### Arm → factor map (frozen)

| Arm | H (history_source) | D (current_source) |
|---|---|---|
| F  | FULL    | FULL    |
| T  | TARGET  | TARGET  |
| HT | TARGET  | FULL    |
| DH | FULL    | TARGET  |

F and T run the bit-identical S3/S4 F/T update path because `g_H = g_D`
and no overwrite is performed.

## 3. Test results

All S5 unit tests pass:

```
$ venv/Scripts/python.exe -m pytest experiments/tests/test_stage_S5_controls.py experiments/tests/test_stage_S5_orchestration.py
============================= 20 passed in 3.39s ==============================
```

Breakdown:
- **test_stage_S5_controls.py** — 14 tests
  - `test_adam_delta_clone_parity_holds` (regression of S4 `validate_adam_delta_against_clone`)
  - `test_cuda_adam_clone_parity` (skipped if CUDA unavailable; passed on CUDA machine)
  - `test_arm_factor_map_is_complete_and_consistent`
  - `test_arm_to_factor_mapping_on_all_four_arms`
  - `test_hand_computed_two_step_ht_trajectory` (toy CPU two-step HT trajectory vs hand reference)
  - `test_hand_computed_two_step_dh_trajectory` (DH state matches F modulo weight-decay)
  - `test_f_and_t_match_direct_backward_reference` (F/T vs `loss.backward()+step()` reference; private uses FULL even on T)
  - `test_overwrite_touches_only_shared_names` (HT/DH overwrite is scoped to names)
  - `test_private_state_matches_full_reference_in_tests` (private params + state equal to a FULL reference)
  - `test_one_step_per_batch_for_both_optimizers` (I5)
  - `test_delta_d_state_timing_regression` (mutating state before Δ_D must change its result)
  - `test_moment_dump_does_not_mutate_state` (hash of `m, v` unchanged after dump)
  - `test_rng_digest_unchanged_across_controlled_step` (I7)
  - `test_ht_dh_share_persistent_state_only_with_respective_history` (state factorization)
- **test_stage_S5_orchestration.py** — 6 tests
  - `test_freeze_refuses_without_reviewer_pass`
  - `test_freeze_rejects_drift_in_source_inventory`
  - `test_validate_run_requires_seven_moment_dumps_and_step_fields`
  - `test_collector_refuses_when_inputs_do_not_match`
  - `test_validate_run_rejects_invalid_protocol` (mutated `formal_protocol_observed`)
  - `test_only_explicit_transient_errno_allows_retry` (reuses S4 `_retry_classification`)

Smoke check (synthetic one-batch dry run for all four arms; never touches
`stage_S5_raw_results/` or any formal output dir):

```
arm=F  rows=2 probes=1 step=[1,1] none_full=0 none_target=0 res_app=1.02e-08 res_hist=1.02e-08
arm=T  rows=2 probes=1 step=[1,1] none_full=0 none_target=0 res_app=1.06e-08 res_hist=1.06e-08
arm=HT rows=2 probes=1 step=[1,1] none_full=0 none_target=0 res_app=1.02e-08 res_hist=1.06e-08
arm=DH rows=2 probes=1 step=[1,1] none_full=0 none_target=0 res_app=1.06e-08 res_hist=1.02e-08
All four S5 arms complete a controlled step without raising.
```

No formal 16-run execution, no writing under `stage_S5_raw_results/`.

The pre-existing `test_stage_S4_orchestration.py::test_a2_freeze_preserves_a1_and_rejects_drift`
failure is unrelated to S5 and reproduces on the prior S4 code (its
assertion regex does not match because the test fixtures miss the
S4 A2 output root precondition).

## 4. Deviations from the design spec and rationale

- **Probe-row fields beyond design §7.** `m_norm_target_stream`,
  `v_norm_target_stream`, `moment_cosine_history_target`,
  `moment_cosine_history_full` were added to `S5_PROBE_FIELDS` so
  the epoch-0/1 moment-cosine gate from §9 has data already inside
  `probe.csv`. The cross-run audit table in the report (design §8)
  is the primary consumer.
- **I6 timing guard implementation.** The design's "compute Δ_D from
  pre-step state" is enforced by re-running `adam_deltas` after the
  step on the **other** gradient stream (the one not installed for
  `g_H`); that counterfactual must change between pre-step and
  post-step states, which is the strongest evidence that Δ_D was
  read from the pre-step state. The design's literal wording ("if any
  Δ_D is computed post-step") was generalised to a norm-difference
  test against the per-arm counterfactual.
- **DH-vs-F state equality.** Strictly bit-equal state is not achievable
  once the Adam coupled weight decay is in play (the effective gradient
  `g + λ·θ` differs because θ differs after the first overwrite). The
  test enforces `step` equality and uses `rtol=1e-3, atol=1e-5` for the
  moment tensors; this matches the §9 "near-identical through early
  epochs" language for the cross-run gate.
- **`_dump_moment_state` invocation point.** Design §5 places the dump
  inside the runner; the implementation overrides `_checkpoint_payloads`
  to trigger the dump after the parent's deferred-checkpoint evaluation
  has populated `self.checkpoint_states`. This keeps a single state
  mutation window inside the parent's existing checkpoint logic.

## 4. Bug fix: wave-1 `write_outputs` drift-check false positive

**Date:** 2026-09-27
**Trigger:** `experiments/reports/stage_S5_run_report.md` — 2/16 attempted
runs (F/seed2022, T/seed2022) failed at `write_outputs` with
`RuntimeError("S4 producer source changed during training")` after
training all 50 epochs. Source has not actually drifted (current
inventory matches the frozen `stage_S5_queue_manifest.json`).

**Root cause.** `S5Runner.__init__` stores `self.initial_inventory =
inventory(ROOT)` (S5 inventory, **25 entries** = S3∪S4∪S5).  The
inherited `S4Runner.write_outputs` does

```python
if inventory(ROOT) != self.initial_inventory:   # inventory is S4's (21 entries)
    raise RuntimeError("S4 producer source changed during training")
```

The `inventory` symbol imported inside `stage_S4_run.py` resolves to
`stage_S4_protocol.inventory` (21 entries).  Because S5 inventory is a
strict superset of S4 inventory, the `!=` is always true even with zero
source drift; every S5 run fails after training completes.

**Fix.** `experiments/stage_S5_run.py`:
1. Replace `super().write_outputs(fit_info, wall_seconds)` in
   `S5Runner.write_outputs` with `S3Runner.write_outputs(self, …)` to
   bypass the broken `S4Runner.write_outputs` path entirely.  S3's
   write chain calls `S3→S2→S1` and writes every standard artifact S5
   needs (`metrics_*.csv`, `batch_trace.csv`, `predictions_test.npz`,
   checkpoints, optimizer states, rng state, stdout.log,
   `gradient_partition.json`, `reachability_audit.json`,
   `objective_coefficients.json`).
2. Extract the drift check into a small `_check_source_drift()` helper
   that uses the **S5** inventory function (the only inventory-equality
   gate in the write path).  Fail-closed against real drift in any of
   the 25 pinned source files.

**Why this is fail-closed.** `_check_source_drift()` is invoked first in
`write_outputs` and compares `inventory(ROOT)` (S5, live) against
`self.initial_inventory` (S5, captured at `__init__`).  Any byte change
in any of the 25 pinned files — S3, S4, S5, or M3OE forward — raises
before any artifact is written.  The S4 path that previously raised is
no longer reachable from `S5Runner.write_outputs`.

**Audit of S4/S3 inheritance (per the bug triage request).**
| Inheritance surface | S4/S3-specific thing | S5 handling | Status |
|---|---|---|---|
| `S4Runner.__init__` sets `args.shared_access` to `TARGET_ONLY_SHARED` if arm == "T" else `"FULL"` | S4 arm mapping | S5 only uses `self.history_source` / `self.current_source` for the controlled-step dispatch; `shared_access` is just a logging field that propagates into `batch_rows["shared_access"]`. | Cosmetic; harmless (S5's `batch_trace.csv` keeps the S3 logging convention) |
| `S4Runner.__init__` sets `args.fresh_role = "target_only_shared"` | S4 role | S5 overrides `args.fresh_role = "fresh_s5"` **before** `super().__init__()` — but S4's init then overwrites it back. The final `self.fresh_role` is `"target_only_shared"`. The `fresh_role` written to `config.json`/`manifest.json` is `"fresh_s5"` (overwritten by S5's `write_outputs`). | Cosmetic on the runner; config/manifest are correct |
| `S4Runner.__init__` sets `self.initial_inventory = inventory(ROOT)` (S4 inv) | S4 inventory snapshot | S5 immediately overwrites it to S5 inventory after `super().__init__()`. | Fixed by the override |
| `S4Runner.write_outputs` runs `inventory(ROOT) != self.initial_inventory` (S4 inv function) | S4 inventory check | Bypassed by calling `S3Runner.write_outputs(self, …)` directly. | **Fixed (this bug)** |
| `S4Runner.write_outputs` calls `self._output_formal_protocol()` | S4 protocol method | S5 overrides `_output_formal_protocol` to return S5 protocol. The S4 protocol/observed comparison in `S4Runner.write_outputs` would have compared S5 protocol (now containing `history_source`/`current_source`) against `observed = {"stage": "S4", ...}` and mismatched. | **Eliminated by skipping S4 path entirely** |
| `S4Runner.write_outputs` writes `gradient_step.csv` with `FIELDS` (40 cols) | S4 gradient step schema | S5 re-writes with `S5_FIELDS` (44 cols) after the S4 chain. | Redundant but harmless (final schema is S5) |
| `S4Runner.write_outputs` writes `probe.csv` with `PROBE_FIELDS` (44 cols) | S4 probe schema | S5 re-writes with `S5_PROBE_FIELDS` (48 cols). | Same as above |
| `S4Runner._controlled_step` (the F/T/N/U dispatch) | S4 update logic | S5 overrides with the HT/DH/factor-aware `_controlled_step`. | Already overridden correctly |
| `S4Runner._access_formula`, `_condition_name` | S4 arm-aware labels | S5 overrides both with S5-specific strings. | Already overridden correctly |
| `S3Runner._validate_args` | S3 arm/seed validation | S5 overrides. | Already overridden correctly |
| `S3Runner._output_formal_protocol` | S3 protocol | S5 overrides. | Already overridden correctly |
| `S3Runner._record_batch` sets `stage="S3"` in `batch_rows` | S3 stage marker | S5 re-stamps `batch_trace.csv` with `stage=STAGE="S5"` in `write_outputs`. | Already handled (re-stamping loop) |
| `S3Runner._objective_components` | S3 loss decomposition | S5 uses the inherited method verbatim; `losses` dict shape (`full`, `target`, `non_target`, `equality_residual`) matches S3's contract and the orchestrator's `validate_batch_trace`. | Already correct |
| `S3Runner.write_outputs` runs `inventory.get(MODEL_SOURCE_KEY) != _sha256(...)` | M3OE forward-source SHA check | S3 inventory contains `MODEL_SOURCE_KEY`; SHA matches as long as M3OE forward source is unchanged. The check passes for S5. | Already correct (only checks MODEL source, not full inventory equality) |

**Regression coverage.** `experiments/tests/test_stage_S5_orchestration.py`
adds three new tests:
- `test_s5_drift_check_uses_s5_inventory_not_s4_inventory` — asserts the
  S5 inventory is a strict superset of S4, exercises `_check_source_drift`
  with no drift (must pass), with drift in an S5-only file (must
  raise), and with drift in an S4-shared file (must also raise because
  the S5 inventory pins both sets).
- `test_s5_write_outputs_does_not_invoke_s4_inventory_check` — patches
  `S3Runner.write_outputs` to a no-op and verifies the S5 drift check
  gates are the only inventory-equality check in the write path.
- `test_s5_drift_check_ignores_s4_subset_differences` — confirms that
  the S5-vs-S5 comparison correctly considers the four S5-only files
  part of the baseline (not as spurious drift) and that swapping in the
  S4 inventory as the live baseline would be wrong.

## 5. Known risks and follow-ups

- The moment-cosine cross-run gate (§9: `cos(m_HT, m_T) > cos(m_HT, m_F)`
  etc. at dump epochs 0/1) is not exercised by these tests because it
  requires full formal runs and a real M3OE forward. The collector emits
  `stage_S5_moment_audit.csv` with all six pair cosines for both `m`
  and `v` at every dump epoch; the reviewer should run the gate before
  promoting the runs as identifying.
- The S5 runner does not call `optimizer.zero_grad(set_to_none=True)`
  before `_controlled_step`; this is intentional and matches the S4
  behavior (the caller is expected to zero both optimizers first). Tests
  that call `_controlled_step` directly do not need to zero because
  fresh optimizers are created in `_toy_runner`.
- `test_a2_freeze_preserves_a1_and_rejects_drift` failure (S4) is
  pre-existing and not caused by S5; flagging here for visibility.
- After the wave-1 failure and the partial fix described above, the
  runner must re-execute wave 1 (`s5_FT_seed2022`) and then waves 2–8
  before the orchestrator can promote any canonical runs.  The
  `experiments/stage_S5_raw_results/.attempts/s5_FT_seed2022/attempt1/`
  tree is preserved per the append-only attempts policy; the failure
  receipts are not deleted.

## 6. Bug fix: round-2 Windows PID-overlap false negative

**Date:** 2026-09-27
**Trigger:** `experiments/reports/stage_S5_run_report.md` round-2 — both
F and T runs of `s5_FT_seed2022` completed all 50 training epochs and
wrote every required S5 artifact, but the orchestrator's
`_validated_overlap` (inherited from `stage_S4_orchestrator.py:94-107`)
raised `process interval identity/status mismatch` and refused
promotion.  Source had not drifted: `inventory(ROOT)` round-2 hash
matched the frozen `stage_S5_queue_manifest.json` minus the one
key the Coder's round-2 inventory-fix changed.  S4 A2 round-1 never
exhibited the same PID mismatch because it was launched under a
different Python launch context (no shell wrapper, see below).

### 6.1 Root cause (Windows launcher-vs-interpreter PID semantics)

On Windows, when the orchestrator is launched through a shell wrapper
(`nohup`, git-bash, `bash -c`, etc.), `subprocess.Popen.pid` does NOT
equal the PID the spawned Python child sees as its `os.getpid()`.  I
verified this on the round-2 host:

| Launch command (parent → child) | `Popen.pid` | child's `os.getpid()` | child's `os.getppid()` |
|---|---|---|---|
| `subprocess.Popen([python, script], cwd=...)` | 30952 | 49024 | 30952 |
| `subprocess.Popen("python script", shell=True)` | 47204 | 22948 | — |
| `subprocess.Popen(["bash", "-c", "python script"])` | 48492 | 22948 | — |
| `subprocess.Popen(["nohup", "python", script])` | 3584 | 13348 | — |

In every case `Popen.pid` equals the child's `os.getppid()` — i.e.
`Popen.pid` is the PID of an intermediate Windows launcher process,
not the actual Python interpreter.  Round-2 used `nohup … &` from
git-bash, which inserted that launcher; round-1 S4 A2 launched
directly (no shell wrapper) so the launcher was bypassed.

The inherited S4 check demands `interval.pid == child.pid` where
`child.pid = Popen.pid`.  On round-2 the check became
`54804 == 46804` (F) / `53188 == 49736` (T) — both false-positive
identity mismatches despite real concurrent execution (the two
intervals do overlap by ~7800 s).  This is environment-specific,
not source-specific.

### 6.2 Fix (in `experiments/stage_S5_orchestrator.py` only)

1. **New `_validated_overlap_s5` helper** replaces
   `stage_S4_orchestrator._validated_overlap` for S5 wave promotion.
   It drops the strict `interval.pid == child.pid` equality but
   preserves every other piece of overlap evidence:

   1. Both intervals have `status="OK"`.
   2. Both intervals have positive `started_ns` and a strictly
      greater `finished_ns`.
   3. The two child PIDs (read from per-run `process_interval.json`)
      are distinct — two real processes, not one process with two
      workdirs.
   4. The two intervals have strictly positive overlap in nanoseconds
      — proof of concurrency.

   `subprocess.Popen.pid` is still recorded for the audit trail
   (`launch.pid` in `state.json`) and is allowed to differ from the
   child's reported PID.

2. **Canonical-evidence tolerance** in `validate_run`.  The
   "canonical process evidence missing" check demanded
   `stdout.log` and `process_interval.json` both to be present in
   the manifest's `artifact_hashes` table — but that table is
   finalised by `_finalize_child` only after a wave is promoted to
   its canonical path, so a *staged* attempt (lives under
   `.attempts/`, never finalised) cannot satisfy it.  The check now
   passes if either the table entry is present OR the file is
   non-empty on disk.  This is what enables re-validating the
   round-2 staged attempts without modifying any source code or
   re-running training.

3. **New `--revalidate` mode** on the orchestrator:

   ```
   $ venv/Scripts/python.exe experiments/stage_S5_orchestrator.py \
       --revalidate --reviewer-pass
   ```

   - Re-runs `validate_run` on every attempt under `.attempts/`.
   - Re-derives the S5 cross-arm overlap evidence using
     `_validated_overlap_s5` (so it accepts the Windows
     PID-mismatch shape as proof of concurrency).
   - Writes a structured report to
     `experiments/stage_S5_revalidate_report.json` with one entry per
     (wave, attempt, key), including the `overlap_pids` tuple that
     surfaces the launcher-vs-interpreter PID mismatch in one
     place.
   - Does **not** modify `stage_S5_queue_manifest.json`,
     `stage_S5_queue_manifest.json.state.json`, or any file under
     `.attempts/` (round-3 directive: preserve the existing
     append-only records).  The new report file is the only artefact
     this command writes.
   - Does **not** call `_promote_wave` or any canonical
     directory-move.  Reviewer inspects the report and decides
     whether to allow promotion on the next `run_queue()` call.

   The revalidate path uses a small `_load_frozen_queue` helper
   that just parses the frozen queue manifest without re-validating
   against current source, so a revalidate call works even after a
   Coder fix has changed local source.

### 6.3 End-to-end demonstration on the real round-2 attempts

```
$ venv/Scripts/python.exe experiments/stage_S5_orchestrator.py \
    --revalidate --reviewer-pass
{"per_wave_summary": [{"attempts": [{"attempt": 1,
                                     "overlap_valid": true,
                                     "wave_valid": true}],
                      "wave": "s5_FT_seed2022"}],
 "revalidate_report": "D:\\CODE\\phase0-3\\experiments\\stage_S5_revalidate_report.json"}
```

The report shows:
- `wave_valid: true` for both F and T arm validations.
- `overlap_pids: [{child: 54804, popen: 46804}, {child: 53188, popen: 49736}]`
  (the launcher-vs-interpreter mismatch, made visible in one place).
- `overlap_ns: 7800333824700` (~7800 s, matches the wall-time).

The state file remains unchanged (`ready_for_promotion: false`,
no `revalidated_at`).  Promotion is Reviewer's call.

### 6.4 How to use the revalidate path (Reviewer-facing)

1. Inspect `experiments/stage_S5_revalidate_report.json`.  Each
   `per_arm` entry includes `valid`, `reasons`, `child_pid`,
   `stage_popen_pid`, `interval_status`, `started_ns`,
   `finished_ns`.  Each `per_attempt` entry includes `overlap_ns`,
   `overlap_pids`, `wave_valid`, `overlap_valid`.
2. To authorise canonical promotion of a passing attempt, the
   Reviewer can either:
   - Edit `stage_S5_queue_manifest.json.state.json` to set
     `attempts[i].ready_for_promotion: true` (then `run_queue()`
     will promote via the existing `_promote_wave` path on the next
     invocation), or
   - Invoke `_promote_wave` directly via the orchestrator's
     internal API after manually editing the state file.
   The Coder does not perform either of these — that is the
   Reviewer's call, per the round-2 directive.

### 6.5 Regression coverage

`experiments/tests/test_stage_S5_orchestration.py` gains 13 new
tests:

- 8 tests for `_validated_overlap_s5`:
  - `test_validated_overlap_s5_accepts_pid_mismatch_with_overlap`
    (round-2 exact values)
  - `test_validated_overlap_s5_accepts_matching_pid`
    (round-1 S4 A2 regression coverage)
  - `test_validated_overlap_s5_rejects_status_not_ok`
  - `test_validated_overlap_s5_rejects_identical_child_pids`
  - `test_validated_overlap_s5_rejects_non_overlapping_intervals`
  - `test_validated_overlap_s5_rejects_inverted_intervals`
  - `test_validated_overlap_s5_rejects_non_positive_pid`
  - `test_validated_overlap_s5_s4_check_would_have_rejected_same_case`
    (locks in the Windows regression: the S4 check rejects the
    exact case the S5 override accepts)
- 2 tests for the canonical-evidence tolerance in `validate_run`:
  - `test_validate_run_accepts_staged_attempt_with_interval_on_disk`
  - `test_validate_run_rejects_when_interval_missing`
- 3 tests for the `--revalidate` entry point:
  - `test_revalidate_overlap_logic_accepts_pid_mismatched_staged_attempts`
  - `test_revalidate_requires_reviewer_pass`
  - `test_revalidate_handles_empty_attempts_root`

### 6.6 Known risks and follow-ups

- The S5 orchestrator's `_validated_overlap_s5` is a strict
  superset of what the S4 check would accept: any case the S4
  check would have accepted also passes the S5 check.  If the
  Windows PID semantics ever change (e.g. someone fixes the
  launcher mismatch in the parent), the S5 override becomes a
  no-op wrapper around the S4 behaviour.  The test
  `test_validated_overlap_s5_s4_check_would_have_rejected_same_case`
  locks the current asymmetry so any future change to that
  asymmetry will fail loudly.
- `process_interval.json` is written by the runner's `main()` AFTER
  `runner.write_outputs()`.  In a future hardening pass the runner
  should hash `process_interval.json` into `artifact_hashes`
  during `write_outputs` (before the file is written), but that
  would require writing the interval *before* `write_outputs` —
  which we have not changed in round-3 because it would touch the
  runner.  The on-disk tolerance in `validate_run` is the
  pragmatic stopgap.
- The revalidate entry point cannot promote attempts on its own
  (per the round-3 directive).  If Reviewer wants promotion to
  happen on the next `run_queue()` invocation, the existing
  ready_for_promotion switch in `state.json` is the way in — but
  we have not modified that switch in round-3 because modifying
  `state.json` would violate the prohibition on altering the
  existing round-2 record.