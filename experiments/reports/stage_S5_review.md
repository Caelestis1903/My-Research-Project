# Stage S5 Code Review Report

**Reviewer: Stage S5 Reviewer agent (K3 high). Independent of the Coder.**
**Date: 2026-09-27. Baseline: `stage_S5_design.md` (Designer spec), `stage_S5_implementation.md` (Coder report).**

The Coder's initial implementation had the correct core §4 state-flow algebra (HT/DH
install-then-overwrite with pre-step read-only `AdamDelta`, F/T no-overwrite), but
shipped **five release-blocking defects** in the audit/evidence layer, all confirmed by
code reading and reproduced or closed by new tests. All were fixed in place by this
review. Final verdict: **Ready for runner: YES**.

## Reviewed files

- `experiments/stage_S5_protocol.py`, `stage_S5_run.py`, `stage_S5_orchestrator.py`, `stage_S5_report.py`
- `experiments/tests/test_stage_S5_controls.py`, `test_stage_S5_orchestration.py`
- Reference (read-only, verified unchanged): `stage_S3_run.py`, `stage_S4_run.py`,
  `stage_S4_protocol.py`, `stage_S4_orchestrator.py`, `stage_S4_gradient_controls.py`,
  `stage_S1_run.py` (fit loop / checkpoint machinery)

## Algorithm-spec compliance (S5.md checklist, item by item)

1. **F/T keep S3/S4 semantics — PASS.** H=D ⇒ no overwrite; `_controlled_step` installs
   `g_H` and the single `step()` is the whole update. `_train_epoch`/`_architecture_update`
   are now inherited literally from S4 (see Fix 5), so the batch/exposure/logging loop is
   byte-identical to S4. `test_f_and_t_match_direct_backward_reference` proves F/T
   bit-agreement with a direct `backward()+step()` reference including optimizer state.
2. **HT = TARGET-history + FULL-current — PASS.** `g_T` is installed (persistent `m/v/τ`
   commit the target transition), then shared params are overwritten with
   `θ_before + Δ_F` where `Δ_F = AdamDelta(pre-step state, g_F)` computed before `step()`.
   Hand-computed two-step trajectory test passes on CPU and CUDA.
3. **DH = FULL-history + TARGET-current — PASS.** Symmetric; state-factorization test
   (`test_ht_dh_share_persistent_state_only_with_respective_history`) proves HT state ≡ T
   state and DH state ≡ F state after one step, and parameters follow the D stream.
4. **Current computation does not pollute persistent history — PASS (after Fix 4).**
   `df/dt/dp` are computed before `optimizer.step()` in program order; I2/I3 hard-assert
   them against the observed/applied deltas every step (this is also what catches the
   "Δ_D computed post-step" bug: post-step deltas would mismatch I2's observed step).
   The overwrite only calls `param.copy_` on controlled names; a bit-exact optimizer-state
   digest on probe steps proves nothing outside `step()` writes `optimizer.state`.
5. **Step counter / bias correction — PASS.** Exactly one `step()` per batch per optimizer;
   `adam_delta` uses `t = τ_pre + 1` (inherited, parity-validated); I5 asserts uniform
   counters; no shadow optimizer exists anywhere.
6. **Shadow/counterfactual state side effects — PASS.** `AdamDelta` is read-only
   (clone-parity validation re-asserts source params unchanged); no shadow objects.
7. **RNG / DataLoader trajectory — PASS (after Fix 2).** Real `rng_state_digest()` is now
   captured before and after every controlled step; probe steps hard-assert equality (I7);
   the orchestrator validates digest presence+equality on all 100 probe rows. No new
   DataLoader iterator or forward is created by S5 code (the eval loader is S3-inherited,
   identical across arms; arch batch keeps `preserve_rng_state`).
8. **Private parameters — PASS.** `g_P` always installed; overwrite loop covers only
   shared/arch names; I4 per-step assertion; `test_private_state_matches_full_reference_in_tests`
   proves private params+state equal a FULL reference on all four arms.
9. **Loss/objective/exposure — PASS.** `_objective_components`, Equal coefficient table,
   target masking, loader construction and batch order are inherited unchanged from the
   S3/S4 chain; pairing audit (`PAIR_FIELDS` incl. `batch_order_hash`, coefficient table
   hash) enforced cross-arm in `validate_block`.
10. **Audit signals prove independent factor manipulation — PASS (after Fixes 1, 3, 6, 7).**
    `gradient_step.csv` carries `history_source`/`current_source` per step (orchestrator
    checks them per row), moment dumps now contain actual `(m, v, τ)` for all 7 epochs,
    and `stage_S5_moment_audit.csv` computes the four gate cosines for `m` and `v` with an
    explicit per-seed epoch-0/1 gate verdict in the collector manifest. I1–I5, I7, I9 are
    hard per-step assertions in the runner; I6 via digest + ordering test; I8 via pairing
    audit.

## Problems found (all blocking unless noted)

1. **Moment dumps had the wrong content and could never exist for epochs 0/1 (critical).**
   `_dump_moment_state` saved `self.checkpoint_states[epoch]` — the *model parameter*
   snapshot, not the Adam moments the design (§5, §7) requires — and
   `checkpoint_states` is only populated at COMMON_EPOCHS (9…49) by S1, so the epoch-0/1
   dumps that the §9 cross-run gate depends on were silently skipped. The orchestrator's
   REQUIRED list would then have failed every run, and `stage_S5_moment_audit.csv` would
   have computed cosines of model weights (the collector even built `m_concat` and
   `v_concat` as the *same* tensor). **Fixed:** dump now writes
   `{name: {"exp_avg","exp_avg_sq","step"}}` for shared+arch params from both optimizers
   (fail-closed if state missing), fired at true epoch end via a thin `_architecture_update`
   wrapper (after the arch step, before validation — identical loop point for all arms).
2. **RNG digest evidence was vacuous (critical for I7).** The runner wrote
   `self.rng_digest() if hasattr(self, "rng_digest") else ""` — no ancestor defines
   `rng_digest`, so both fields were always `""` — and assigned the *same* captured value
   to `rng_digest_before` and `rng_digest_after`. The I7 probe check could never fail.
   **Fixed:** real `rng_state_digest()` captured around the controlled step, hard-asserted
   equal on probe steps, logged on every row, and validated by the orchestrator.
3. **Probe-row creation crashed on any real model (critical).** `_moment_cosine` built a
   1-element moment vector (first param only) against an 80-element gradient vector;
   `cosine → dot` raises `ValueError("vector length mismatch")`. Tests passed only because
   the toy model has exactly one shared parameter; the first formal probe step
   (epoch 0, step 0) would have aborted every run. **Fixed:** cosine now aligns the full
   persistent-`m` vector with the full gradient vector; regression test with two shared
   parameters added.
4. **Coder's I6 deviation (post-step norm-difference guard) carried a false-abort risk.**
   The guard re-ran `adam_deltas` after the step on the other gradient stream and
   hard-failed if the norm moved by less than `max(1e-6, 1e-4·ref)`. It does catch the
   "Δ_D used post-step state" bug, but the same condition can trigger legitimately
   (near-converged streams where `g_H ≈ m`, decay-dominated deltas), aborting a healthy
   formal run deep into training — and the design assigns the pre-step property to tests,
   not to a per-step heuristic. **Fixed (deviation superseded):** the guard is replaced by
   (a) the existing I2/I3 per-step hard assertions, which *directly* catch post-step Δ_D
   (a post-step Δ_H cannot match the observed pre-step-based move), (b) the ordering
   regression test, and (c) a new bit-exact optimizer-state digest on probe steps that
   proves overwrite/logging/dump paths never mutate `optimizer.state` — zero false-positive
   risk by construction.
5. **Orchestrator checkpoint filename bug (critical).** The S5 label style
   `"common_epoch_9"` made `label[6:]` = `"_epoch_9"`, producing
   `common_epoch__epoch_9.pth` (S4's labels were `"common9"`, where the inherited slice
   worked). Every run would have failed validation with "checkpoint missing".
   **Fixed:** `label.rsplit("_", 1)[-1]`.
6. **Orchestrator referenced a non-existent CSV field (critical).** `validate_run` read
   `r["private_delta_norm"]`, which was not in `S5_FIELDS` — a `KeyError` on the first
   step row, caught by the broad `except`, failing every run as "validation exception".
   **Fixed:** `private_delta_norm` added to `S5_FIELDS` and populated in the runner
   (mirrors S4's field; used as the I4 tolerance reference).
7. **Collector moment audit was uncomputable/meaningless (critical, with 1).** Rewritten
   to read `exp_avg`/`exp_avg_sq` separately per dump, with a schema check, per-arm
   moment norms and step ranges, and the §9 epoch-0/1 hard gate evaluated per seed
   (`moment_gate_m_pass`/`moment_gate_v_pass` columns; `moment_gate` +
   `moment_gate_overall_pass` in the summary manifest). Gate arithmetic verified on
   synthetic dumps in both passing and violating configurations.
8. **Vacuous test assertion (non-blocking but misleading).** The private-integrity check
   in `test_overwrite_touches_only_shared_names` was
   `allclose(p, p_before) or not equal(p, p_before)` — true by construction. **Fixed:**
   asserts the private Adam step actually happened (`step == 1`, parameter moved).
9. **Duplicated S4 loop bodies (drift risk, non-blocking).** S5 overrode `_train_epoch`
   and `_architecture_update` with byte-copies of S4's, plus a `_checkpoint_payloads`
   override that double-fired during the S1/S2 write chain. **Fixed:** both loops are now
   inherited from S4 (they contain no arm-dependent logic — dispatch lives in
   `_controlled_step`), and only a thin `_architecture_update` wrapper remains for the
   epoch-end dump hook.
10. **Missing test coverage (non-blocking).** HT two-step trajectory now also runs on
    CUDA when available (previously dead code); probe-row digest fields asserted.

## Fixes applied (files touched by this review)

- `experiments/stage_S5_run.py` — dump rewrite + hook; real RNG digests + I7 gate;
  `_moment_cosine` alignment fix; `private_delta_norm` field; I6 exact state digest
  replacing the norm-difference guard; removed duplicated `_train_epoch` /
  `_architecture_update` / `_checkpoint_payloads`; stale comments corrected; unused
  imports pruned.
- `experiments/stage_S5_orchestrator.py` — checkpoint filename fix; probe RNG-digest
  validation; removed dead no-op check.
- `experiments/stage_S5_report.py` — moment audit rewritten for the real dump schema;
  §9 epoch-0/1 gate columns + manifest summary.
- `experiments/tests/test_stage_S5_controls.py` — dump test rewritten (schema, content,
  non-dump epochs); vacuous private assertion fixed; CUDA trajectory; RNG probe-field
  assertions; new multi-shared-parameter probe regression test.

No changes to `stage_S3_*`, `stage_S4_*`, or `m3oe/` (verified — see below).

## Coder's self-declared deviations — adjudication

- **probe.csv extra fields** (`m_norm_target_stream`, `v_norm_target_stream`,
  `moment_cosine_history_target/full`): acceptable as informational additions; they do not
  feed the §9 gate (the cross-run dump audit does). Kept; crash bug inside them fixed
  (Problem 3). Note the "target_stream" names are misleading for F/DH arms (they read the
  persistent state, which is FULL-history there) — cosmetic only.
- **I6 timing-guard generalization:** did capture the highest-risk bug but with
  false-abort risk; superseded by the stronger exact mechanism (Problem 4). Design intent
  preserved and detection strengthened.
- **DH-vs-F state tolerance relaxation (`rtol=1e-3, atol=1e-5`, exact step equality):**
  legitimate. Coupled weight decay (`g + λθ` with diverged θ) and diverged trajectories
  make bit-equality impossible after step 0; the §9 gate's cosine comparison is the
  designed instrument for later epochs. The relaxation does not weaken the factorization
  evidence (step equality is exact; one-step factorization is covered by the stricter
  `test_ht_dh_share_persistent_state_only_with_respective_history`).
- **`_dump_moment_state` invocation point:** the declared deviation (checkpoint-payload
  hook) was part of the critical bug; replaced by the epoch-end architecture-step wrapper,
  which matches the design's "dump at epoch end" intent more precisely.

## Verification performed

- `venv/Scripts/python.exe -m pytest experiments/tests/test_stage_S5_controls.py experiments/tests/test_stage_S5_orchestration.py -x -q` → **21 passed** (20 before review; +1 new regression test; several existing tests strengthened).
- Synthetic one-batch + one-arch-step smoke for all four arms (F/T/HT/DH): formula
  residuals ≤ 8e-9, private residual 0, step counters `[1,1]`, probe RNG digests equal
  and non-empty, moment dump schema `{shared, arch} → {exp_avg, exp_avg_sq, step}`.
- Collector gate arithmetic on synthetic dumps: gate passes when HT≈T and DH≈F, fails
  when HT is moved toward F.
- `py_compile` on all six S5 source/test files: clean.
- Inheritance-chain integrity: this workspace is not a git repository, so zero-change was
  verified by hashing — the current S5 source inventory matches the frozen S4 A2 queue
  manifest (`stage_S4_queue_manifest_A2.json`, 21 entries) on **every inherited key**
  (S3/S4 runners, protocols, gradient controls, gradient access, M3OE forward
  `a7702a64…`); the only new keys are the four S5 files. No `stage_S3_*`, `stage_S4_*`,
  or `m3oe/` file was modified.

## Remaining risks

- The §9 cross-run moment gate is only exercisable after formal runs; the collector now
  computes it and records `moment_gate_overall_pass`, but the *reviewer block* on gate
  failure remains a human/main-agent step (by design). Check the manifest before any
  mechanism interpretation.
- Probe-step-only I6/I7 hard gates sample 100 steps/run; non-probe steps rely on I2–I5
  (per-step) plus the deterministic nature of the read-only helpers. Acceptable per design.
- CUDA foreach dispatch is not bitwise-reproducible across processes (design pitfall 4);
  all cross-run comparisons correctly use tolerances/cosines — no action needed.
- The pre-existing, unrelated `test_stage_S4_orchestration.py::test_a2_freeze_preserves_a1_and_rejects_drift`
  failure flagged by the Coder was not investigated (out of S5 scope; S4 sources hash-identical
  to the frozen A2 inventory).

## Ready for runner: **YES**

All ten S5.md checklist items pass; all blocking defects are fixed and covered by tests;
the two intervention factors (persistent Adam history source × current applied update
source) are independently manipulated with per-step hard evidence (I1–I5, I7, I9), exact
probe-step pollution/RNG gates (I6/I7), cross-arm pairing audit (I8), and a computable
epoch-0/1 cross-run moment gate (§9). The queue may be frozen and the Runner started.

---

# Re-review (round 2): wave-1 `write_outputs` false-positive drift check

**Date: 2026-09-27. Trigger: `stage_S5_run_report.md` — F/seed2022 and T/seed2022
completed all 50 epochs (all seven moment dumps were written during training, confirming
the round-1 dump fix works in production), then failed at `write_outputs` with
`RuntimeError("S4 producer source changed during training")` (`stage_S4_run.py:262`).**

## Root cause (confirmed independently)

`S5Runner.__init__` stores `self.initial_inventory` = the **S5** inventory (25 entries),
while the inherited `S4Runner.write_outputs` recomputes `inventory(ROOT)` with the **S4**
inventory function (21 entries) and compares with strict `!=`. Superset-vs-subset is
always unequal, so every S5 run would fail after training with zero actual drift. This
was a genuine miss in round 1: I verified the inventory contents but did not trace the
S4 write path's own inventory-equality gate against the S5 snapshot. Additionally, had
the drift check not fired first, S4's `observed` protocol block (stage `"S4"`, no
`history_source`/`current_source`/`moment_dump_epochs`) would have mismatched S5's
polymorphic `_output_formal_protocol()` and raised anyway — so bypassing the S4 write
layer entirely (rather than patching only the drift line) is the correct fix shape.

## Coder's fix, verified item by item

`S5Runner.write_outputs` now calls `_check_source_drift()` (S5 inventory vs S5 snapshot,
fail-closed, runs before any artifact write) and then `S3Runner.write_outputs(self, …)`
directly. What `S4Runner.write_outputs` used to do, and where each piece now lives:

| S4 write-path responsibility | Fate in the new S5 path | Verdict |
|---|---|---|
| S4 inventory drift check | Replaced by `_check_source_drift()` (S5 inventory, superset) | Correctly substituted, strictly stronger |
| `super()` → S3→S2→S1 chain (predictions npz, checkpoints, optimizer/rng states, metrics CSVs, gradient_partition, reachability, objective_coefficients, config/manifest skeleton) | Called directly as `S3Runner.write_outputs(self, …)`; the intermediate S3-schema `gradient_step.csv` write with S5 rows is safe because `write_rows_csv` uses `extrasaction="ignore"` + `restval=""` (same tolerance S4 relied on in production) | Preserved |
| Stage re-stamping of `batch_trace/metrics_epoch/metrics_test/metrics_summary` | S5's own re-stamping loop (unchanged) | Preserved |
| `gradient_step.csv`/`probe.csv` with stage schema | S5 writes with `S5_FIELDS`/`S5_PROBE_FIELDS` (as before, final write wins) | Preserved |
| Protocol observed-vs-expected gate | S5's own block executes: `protocol` (with `history_source`/`current_source`/`moment_dump_epochs`) vs `observed`; mismatch ⇒ `RuntimeError` before manifest stamping | **Still executed and fail-closed** |
| config/manifest stamping (stage, arm, fresh_role, formal_valid, code_hashes, digests, counts) | S5's own update block (superset of S4's fields, with `fresh_s5`, moment-dump hashes, none counts) | Preserved; nothing S4-specific is lost |
| artifact_hashes / artifacts manifest update | S5's own block (unchanged) | Preserved |

No functionality loss found; the S4 layer contributed nothing that S5 does not redo.

## `_check_source_drift()` fail-closed verification

- Code: `inventory(ROOT) != self.initial_inventory` with both sides the 25-entry S5
  inventory (snapshot taken at `__init__` **after** `super().__init__`, so it overwrites
  S4's snapshot — verified at `stage_S5_run.py:125`). The inventory pins all S3/S4 files
  and `m3oe/MDMTRec/models/multi_domain/model.py`, so drift in any of the 25 pinned files
  raises before any artifact is written.
- Test `test_s5_drift_check_uses_s5_inventory_not_s4_inventory`: no-drift passes; drift in
  an S5-only file raises; drift in an S4-shared file raises. Real, not trivially-pass.

## Inheritance-surface audit spot checks (Coder's 12-row table)

- **`__init__` order**: correct — S5's `initial_inventory` assignment post-dates
  `super().__init__`; `args.fresh_role` is overwritten back to `"target_only_shared"` by
  S4's init (as the table states), but the final config/manifest `fresh_role` is
  `"fresh_s5"` via S5's stamping, which is what the orchestrator validates. No behavioral
  use of `fresh_role`/`shared_access` exists on the S5 path (S2 never reads them — grep
  verified; `shared_access` only feeds logging columns).
- **`_record_batch` re-stamping**: correct — S3 stamps `stage="S3"` per batch row; S5's
  re-stamping loop rewrites `batch_trace.csv` to `stage="S5"`; metrics CSVs likewise.
- **`_objective_components` shape**: correct — inherited verbatim from S3; the
  `full/target/non_target/equality_residual` dict matches `_controlled_step`'s consumption
  and the batch-trace contract.
- No omissions found in the table; the one stale row is the table's own description of
  the old I6 guard (superseded in round 1) — cosmetic, in the implementation report only.

## Regression tests — validity check

- Round-1 finding: the Coder's `test_s5_write_outputs_does_not_invoke_s4_inventory_check`
  was **near trivially-pass** — it patched `S3Runner.write_outputs` but only called
  `_check_source_drift()` inside the block, never `write_outputs`, so it could not detect
  a regression back to the S4 dispatch (it also contained dead `... if False else ...`
  code). **Rewritten by this review** into a true end-to-end regression of the wave-1
  failure mode: a minimal `S5Runner` + temp run dir, `S3Runner.write_outputs` stubbed,
  `S4Runner.write_outputs` poisoned to raise the production error, then the real
  `S5Runner.write_outputs` must complete and produce S5-stamped CSVs
  (`S5_FIELDS`/`S5_PROBE_FIELDS` headers) and a manifest with `stage=S5`,
  `fresh_role=fresh_s5`, `history_source`/`current_source`, `formal_valid=True`, empty
  `formal_protocol_mismatches`, and `code_hashes` equal to the S5 inventory;
  `s4_write.assert_not_called()` guards the dispatch.
- Negative control (executed this round): with the same poison, dispatching through
  `S4Runner.write_outputs` (the old code path) does raise — so the strengthened test
  genuinely discriminates fixed vs broken code.
- `test_s5_drift_check_ignores_s4_subset_differences`: partially redundant with the first
  drift test but not vacuous (pins the superset precondition).

## Test runs and zero-change verification

- `venv/Scripts/python.exe -m pytest experiments/tests/test_stage_S5_controls.py experiments/tests/test_stage_S5_orchestration.py -x -q` → **24 passed**.
- S4 regression: `test_stage_S4_controls.py + test_stage_S4_orchestration.py` →
  **20 passed, 1 failed** — the failure is the pre-existing
  `test_a2_freeze_preserves_a1_and_rejects_drift` (fixture misses the S4 A2 output-root
  precondition; S4 sources are hash-identical to the frozen A2 inventory, so this
  predates S5 and is unrelated).
- Zero-change: current inventory vs frozen `stage_S4_queue_manifest_A2.json` — **NONE** of
  the 21 inherited keys (S3/S4/m3oe) drifted. Current S5 inventory digest
  `ac22b371fbca9e16…`; vs the frozen `stage_S5_queue_manifest.json` (`79cebd121df5daf7…`)
  exactly one key differs: `experiments/stage_S5_run.py` (the reviewed bug fix). Test-file
  edits are outside the pinned inventory.

## Operational note for the Runner restart

`freeze_queue` is fail-closed against source drift, so the existing frozen queue (pinning
the pre-fix `stage_S5_run.py`) will be rejected. The Runner/main agent must archive the
old `stage_S5_queue_manifest.json` + `.state.json` (preserving the append-only wave-1
attempt record under `stage_S5_raw_results/.attempts/`) and re-freeze with
`--reviewer-pass` against the fixed source, then re-execute all 8 waves.

## Ready for runner (round 2): **YES**

The wave-1 blocker is fixed at the correct layer, the fix is fail-closed against real
source drift, no S4 write-path functionality is lost, the observed-vs-expected protocol
gate (with `history_source`/`current_source`) still executes, regression coverage now
genuinely discriminates the failure mode, all 24 S5 tests pass, and the inherited chain
remains bit-identical to the frozen S4 A2 inventory.



# Re-review (round 3): Windows venv PID-overlap false rejection

Round-2 wave `s5_FT_seed2022` (F and T arms) completed both training runs cleanly
(exit 0, 34 artifacts/run, `formal_valid=True`, 2500 gradient-step rows = 2450 main +
50 architecture, 100 probe rows, wall ~7800 s, `overlap_ns=7800333824700`), yet the
wave was rejected by `_validated_overlap` because `interval.pid != child.pid` for both
children (child PIDs 54804/53188 vs Popen PIDs 46804/49736). The Coder diagnosed a
Windows venv-redirector shim as the root cause and patched the overlap check, the
staged-attempt validator, and added a `--revalidate` path. This round independently
verified all of it, found and fixed one additional blocking bug the Coder missed, and
ruled on the fate of the round-2 F/T runs.

## Root cause — independently reproduced

`venv/Scripts/python.exe` on this machine is a **venv redirector shim** (253 KB; base
interpreter `D:\Programs\anaconda3\python.exe`), not the interpreter itself. Reproduced
directly: `subprocess.Popen([sys.executable, "-c", ...])` returns `Popen.pid=45812`,
while the actual Python child reports `os.getpid()=45264`, `os.getppid()=45812`. The
shim forwards into the real interpreter, so **any** plain list-form `Popen` of
`sys.executable` under this venv yields a Popen-handle PID that differs from the
training process PID. This has nothing to do with `nohup`/shell wrapping; the S4 strict
check (`interval.pid == child.pid`) can **never** pass on this host. Consequences:

- The Coder's fix is **necessary**; there is no alternative route of "roll back the
  orchestrator and change the launch command" — the mismatch is inherent to the
  interpreter path the experiment is required to use.
- The strict S4 check is not wrong on its own turf; regardless, S5 must not inherit a
  check that is physically unsatisfiable on the mandated runtime.

## The six verifications

1. **Root cause**: confirmed empirically (above). Not nohup-related; shim-inherent.
2. **`_validated_overlap_s5` (experiments/stage_S5_orchestrator.py:345)**: keeps four
   pieces of evidence — both statuses OK, monotone intervals, the two child PIDs
   distinct from each other, strictly positive overlap — and drops only the
   interval-to-Popen-handle PID binding. Evidence strength remains sufficient for the
   actual threat model (bookkeeping integrity, not adversarial forgery): serial
   execution produces non-overlapping intervals → rejected; two directories driven by
   one process produce identical PIDs → rejected; stale interval files are impossible
   because each `attempt_root` is freshly created and the child creates `run_dir` with
   `exist_ok=False`.
3. **`validate_run` relaxation for staged attempts**: the new condition is effectively
   "both interval files exist on disk and are non-empty" (the hashes-table clause is
   absorbed/redundant). A truncated/corrupt `process_interval.json` is still caught
   downstream by `_read_json` parsing plus the status-OK and `finished>started` checks
   (stage_S5_orchestrator.py:184-187). Promoted canonical runs remain bound by
   `_finalize_child`-written hashes and `_artifact_hash_reasons` validation. **Safe.**
4. **`--revalidate` purity**: verified empirically — hashed the queue manifest,
   `state.json`, and the entire `.attempts/` tree (70 files), ran
   `--revalidate --reviewer-pass`, re-hashed: all three **unchanged** (SHA-256
   identical). The report content is correct: `wave_valid=true`,
   `overlap_ns=7800333824700`, `overlap_pids` documents the 54804/53188 vs
   46804/49736 mismatch. One residual documentation bug found and fixed by this review
   (below).
5. **Coder's 13 new tests**: read individually — not trivially-passing (positive and
   negative cases, round-2 real numbers, state.json immutability assertions, and the
   S4/S5 lock-in asymmetry test). Genuine discrimination.
6. **Source inventory**: `stage_S5_protocol.inventory('.')` vs the frozen S4 A2
   manifest — all 21 inherited keys **zero drift**; vs the round-2 S5 queue
   (`source_inventory_digest ac22b371fbca9e16…`) — **exactly one** differing key,
   `experiments/stage_S5_orchestrator.py` (56eedd76e025… → 8ae510cfba02…), which is
   precisely the Coder's fix plus this review's fix below.

## Blocking bug found by this review and fixed (would have detonated every wave)

`_promote_wave` / `_recover_promotion` were inherited from `stage_S4_orchestrator`
(stage_S4_orchestrator.py:200, :162), and both resolve `validate_run` in the **S4
module namespace** — i.e. the S4 validator (stage="S4", `fresh_role="fresh_s4"`, S4
protocol, no moment dumps). Any S5 wave reaching promotion would have failed with
"staged copy invalid" at best or been mis-validated at worst. Rounds 1–2 never reached
promotion, so this was never exercised.

**Fix applied by this review** (within Reviewer bug-fix mandate; logged here):
- `experiments/stage_S5_orchestrator.py:56` — import reduced to
  `_finalize_child, _journal_path, _monitor_children, _retry_classification, sha`
  (dropped `_promote_wave, _recover_promotion, _validated_overlap`).
- `experiments/stage_S5_orchestrator.py:427` and `:465` — module-local
  `_recover_promotion` and `_promote_wave`, verified **byte-identical** to the S4
  originals (`inspect.getsource` equality) except that `validate_run` now resolves to
  the S5 validator. All call sites (`run_queue`) use the same names — no call-site
  changes needed.
- New regression tests pin the namespace binding for both helpers (S4 validator
  patched to raise on any call; S5 validator asserted called) plus a positive
  two-canonical commit case.

## Documentation bug fixed

`--revalidate` help text claimed "ready_for_promotion is set on passing attempts" —
contradicted by the implementation (revalidate never touches `state.json`). Rewritten
to state that only the report is written and no promotion decisions are taken
(stage_S5_orchestrator.py:802-807).

## Test evidence

`pytest experiments/tests/test_stage_S5_controls.py experiments/tests/test_stage_S5_orchestration.py -x -q`
→ **40 passed** (Coder's 37 + 3 new from this review). Full suite green after all fixes.

## Ruling on the round-2 F/T runs: rerun all 16 runs

Promotion of the two finished runs is **NOT** approved. Reasoning chain:

1. The shim mismatch is inherent to the mandated interpreter, so the orchestrator fix
   is unavoidable → the orchestrator file hash changes → `freeze_queue` refuses the
   round-2 queue (digest `ac22b371fbca9e16…`) → a third freeze is mandatory.
2. Under a new freeze, both the completed-wave re-check and `_promote_wave` run with
   `source_hashes=<new inventory>`; the round-2 F/T manifests pin the **old**
   orchestrator hash in `code_hashes` → "source identity mismatch". `validate_block`
   additionally requires all four arms of the seed2022 block to carry equal
   `code_hashes` (stage_S5_orchestrator.py:302) → a mixed block (F/T old, HT/DH new)
   fails "pairing source mismatch".
3. Relaxing either the source gate or the pairing rule is a **design change**, outside
   Reviewer authority.
4. Promoting the two runs before re-freezing is circularly impossible: it would
   require rolling the orchestrator back, which simultaneously restores the strict
   overlap bug **and** the S4-validator promotion bug.
5. The round-2 F/T attempts are retained as superseded evidence, mirroring round-1's
   `.attempts.superseded_wave1/` handling. Cost: one wave (~2.2 h).

## Operational steps for the Runner

1. Archive `experiments/stage_S5_queue_manifest.json` → `...json.superseded_wave2`
   and `...state.json` → `...state.json.superseded_wave2`; move
   `experiments/stage_S5_raw_results/.attempts/s5_FT_seed2022` to
   `.attempts.superseded_wave2/` (retained, never re-read by the orchestrator).
2. `python experiments/stage_S5_orchestrator.py --reviewer-pass --freeze-only` to
   freeze the third queue against the current inventory (orchestrator hash
   `8ae510cfba02ea976f21aabddff4a30a411cbffe290f6df72eedd25cf516d65d`).
3. Run all 8 waves (16 runs).

## Ready for runner (round 3): **YES**

The root cause is proven and the fix is necessary and minimal; the overlap evidence
remains sufficient; the validator relaxation is safe; revalidate is empirically
side-effect-free; the promotion-path validator-binding bug that would have failed
every wave is fixed and pinned by regression tests; 40/40 tests pass; and the source
inventory shows exactly the one expected changed file. Conditional on the Runner
performing the three operational steps above (re-freeze, then all 16 runs fresh).


# Moment-gate investigation (round 3, post-run): seed 2024 / dump_epoch 1 / m_pass=False

Round-3 execution completed 16/16 runs (all `formal_valid=true`, `overlap_valid=true`,
`retry_class=none`, zero retries), `pairing_valid=true`, but the collector reports
`moment_gate_overall_pass=false` with exactly one failing cell:
**seed 2024, dump_epoch 1, m** (`moment_gate_m_pass=False`; v passes in that cell;
all other 14 gated cells pass m and v). This chapter is the formal investigation and
ruling required by design §9. Per the investigation mandate, no performance metrics
(AUC/LogLoss/contrasts) were consulted; every conclusion below rests solely on audit
artefacts (`stage_S5_moment_audit.csv`, `gradient_step.csv`, `moment_state_epoch*.pth`,
`initialization_audit.json`, batch fingerprints).

## 1. Failing cell — raw values and localization

Gate (§9): at dump epochs 0/1, cos(m_HT,m_T) > cos(m_HT,m_F) **and**
cos(m_DH,m_F) > cos(m_DH,m_T) (same for v).

Seed 2024, dump_epoch 1 (collector values, independently recomputed below):

| comparison | value | counterpart | value | required | margin |
|---|---|---|---|---|---|
| cos(m_HT, m_T) | 0.04128798 | cos(m_HT, m_F) | 0.12218710 | HT_T > HT_F | **−0.0809 FAIL** |
| cos(m_DH, m_F) | 0.24060157 | cos(m_DH, m_T) | 0.09491421 | DH_F > DH_T | +0.1457 pass |
| cos(v_HT, v_T) | 0.99989555 | cos(v_HT, v_F) | 0.61778655 | HT_T > HT_F | +0.3821 pass |
| cos(v_DH, v_F) | 0.99998172 | cos(v_DH, v_T) | 0.61805507 | DH_F > DH_T | +0.3819 pass |

The failure is on the **HT side only**, margin −0.0809.

Contrast cells (m, HT side, epoch 1): seed 2022 +0.1177, seed 2023 +0.1097,
seed 2025 +0.1167 — all pass. All four seeds pass at epoch 0 (m HT margins
+0.134/+0.405/+0.134/+0.185). The smallest *passing* m margin anywhere in the gate is
**+0.016** (seed 2025, epoch 1, DH side), i.e. the metric's passing margins themselves
span +0.016 … +0.46.

## 2. Independent recomputation (collector arithmetic verified)

Loaded `moment_state_epoch1.pth` for all four seed-2024 arms directly
(`torch.load` → CPU, float64, sorted-key concat of `exp_avg` / `exp_avg_sq`):

```
cos_m_T_F  = 0.0936844806962670    cos_v_HT_F = 0.6177865455859267
cos_m_HT_F = 0.1221870970155883    cos_v_HT_T = 0.9998955503337397
cos_m_HT_T = 0.0412879798124302    cos_v_DH_F = 0.9999817159828865
cos_m_DH_F = 0.2406015673055741    cos_v_DH_T = 0.6180550732918367
cos_m_DH_T = 0.0949142133430077    cos_m_HT_DH = 0.0080033859035862
m_norms: F 2.1239e-03, T 8.2231e-04, HT 1.5288e-03, DH 1.4394e-03
```

Every value matches the collector CSV to all 16 printed digits. The collector's
arithmetic is correct; the failure is a property of the dumped state, not of the
scoring code.

## 3. Diagnosis (mandated checks a–d)

**(a) Numerical edge case? No — and yes, depending on the noise scale.** The margin
−0.0809 is five orders of magnitude above float64 noise, so this is *not* a numerical
tolerance issue. It *is*, however, inside the metric's own seed-level statistical
spread: the same quantity's passing margins range from +0.016 to +0.46 across the
other 15 gated m comparisons, so a single −0.08 draw is unremarkable for this metric.

**(b) Tiny-norm instability? No.** The four m norms (0.82e-3 … 2.12e-3) are the same
scale as every passing cell; no denominator pathology.

**(c) Concentrated or diffuse? Diffuse, but block-structured.** Decomposing the gate
numerator Σᵢ[cos-contrib(HT,T) − cos-contrib(HT,F)] per parameter tensor:

- Total: −0.0809. **Embedding block (34 tensors): −0.0911. Non-embedding block
  (50 tensors): +0.0102 — i.e. the HT-side gate actually *passes* on the dense
  parameters and fails only through the embedding tables.**
- 49/84 tensors contribute negatively; the largest single contributor
  (`embedding.embed_dict.31.weight`) accounts for only −0.0142. No single expert,
  arch scalar, or tensor dominates; this is not a localized defect.
- Block-wise cosines, seed 2024 ep1 — embeddings: cos(HT,T)=−0.0116 vs
  cos(HT,F)=+0.1737; non-embedding: cos(HT,T)=+0.0725 vs cos(HT,F)=+0.0703.
  For the passing seed 2025 the same decomposition is positive on both blocks
  (embeddings 0.2283 vs 0.0979; non-embedding 0.2555 vs 0.1596).

**(d) Per-step invariants I1–I9 for seed 2024 HT/DH — all hold, profile identical
to passing seeds.** Over all 2450 main steps of both runs:

- `history_source`/`current_source` exactly as designed (HT: TARGET history + FULL
  current; DH: FULL history + TARGET current), constant per arm.
- `installed_cosine_history` = 1.000000 exactly on every step (I1: the persistent
  state received exactly the designed gradient source).
- `applied_cosine_full` (HT) / `applied_cosine_target` (DH) = 1.000000 exactly (I3:
  the parameter overwrite is exactly the designed counterfactual delta).
- `history_formula_residual`, `applied_formula_residual`, `private_full_residual`
  ∈ [7e-7, 1.2e-6] — the float32 noise floor, **statistically identical across all
  four seeds** (same min/max band for 2022/2023/2025).
- `decomposition_residual` ≤ 3e-8; zero `zero_target_gradient_step`; zero None
  gradients; Adam step counters uniform; batch fingerprints identical across the four
  arms at every step (verified for the first 98 steps, and the collector validated all
  steps at promotion); `initialization_audit.json` SHA identical across all four arms
  (`960842fc…`) — the §9 premises "相同初始化 + 相同 batch" hold for seed 2024.

## 4. Did the persistent-history decoupling actually happen for seed 2024? **Yes.**

Three independent lines of evidence:

1. **Per-step invariants (above)**: the g_T sequence installed into HT's Adam state
   and the g_F sequence installed into DH's Adam state were exactly the designed ones,
   every single step, with formula residuals at the float32 floor. The manipulation
   is not inferred — it is asserted per step and never failed.
2. **The v evidence is overwhelming**: cos(v_HT,v_T)=0.99990 vs cos(v_HT,v_F)=0.61779
   and cos(v_DH,v_F)=0.99998 vs cos(v_DH,v_T)=0.61806. HT's second-moment state is
   essentially identical to T's and far from F's; DH's mirrors this. The persistent
   state carries the intended source identity with margins of ~0.38.
3. **No-leak is proven by the failing numbers themselves**: if g_F had leaked into
   HT's persistent state, cos(m_HT,m_F) would be ≈1. The observed value is 0.122 —
   weak correlation, exactly what "different gradient source, correlated parameter
   trajectory" produces. HT's m is nearly orthogonal to *everything*
   (max pairwise cos 0.122; cos(m_HT,m_DH)=0.008) — the signature of noisy
   signed-EMA decorrelation, not contamination.

**Mechanism of the gate failure.** Adam m is a *signed* EMA (β₁, time constant
≈1/(1−β₁) ≈ 10 steps). By step 98 (dump_epoch 1) the shared-initialization
contribution has washed out (β₁⁹⁸ ≈ 3e-5); what remains is the correlation between
the gradient *sequences* g_T(θ_HT(t)) and g_T(θ_T(t)) — which decays as the arm
parameter trajectories diverge. This divergence is **by design**: HT applies the
full-gradient counterfactual delta to its parameters, T applies the target delta, so
their parameters — and hence the evaluation points of their g_T sequences — separate
from step 1. The §9 premise "HT/T history 早期近乎相同" is therefore only a
statistical bet that divergence is slow enough at the dump epochs; it is not an
invariant. Embedding rows amplify the effect: their gradients are sparse, so each
row's m is dominated by a few recent touches whose *signed* values rotate quickly
with the parameter point — hence the block-wise result (dense params still pass,
embeddings flip). v is immune to sign rotation (elementwise squares are
all-positive), which is why the v gate passes with huge margins everywhere, including
this cell. Supporting trend: seed 2024's cross-trajectory m cosines decay unusually
fast (cos(m_T,m_F) at ep1 = 0.094, lowest of the four seeds; HT-side margin decays
+0.134 @ep0 → −0.081 @ep1), while its per-step cos(g_F,g_T) is actually the
*highest* of the seeds (0.49/0.66 for ep0/ep1) — the failure is trajectory-divergence
speed in signed space, not gradient-source aliasing.

## 5. Ruling: **A — gate failure is a statistical edge case; identification stands**

- **Why not B**: the identification question — "were HT/DH's persistent histories
  independently manipulated as designed?" — is answered affirmatively by per-step
  hard invariants (I1–I9, 2450/2450 steps, audit profile indistinguishable from
  passing seeds), by the v-cosine decoupling margins (~0.38), and by the absence of
  any leak signature. Declaring seed 2024's HT/DH unidentified would contradict
  direct evidence; the failing m comparison does not measure manipulation fidelity.
- **Why not C**: 15/16 gated comparisons pass, the gate's v component and its epoch-0
  m component are well-behaved, and the failure mechanism is fully understood. The
  epoch-1 m component is nonetheless empirically fragile — a strict `>` over a
  quantity whose seed-level spread (passing margins +0.016…+0.46, observed failure
  −0.081) straddles zero — so this review **recommends** that the designer consider
  revising that component for future stages (e.g. tolerance, block-restricted scope,
  or reliance on v + epoch-0 m). That is a design decision; this review makes no
  change to the gate.
- **Honest characterization of the margin**: −0.0809 is *not* within float64/numerical
  noise (that reading of A is explicitly rejected), but it *is* within the metric's
  own seed-sampling noise scale, with a complete mechanistic explanation (signed-EMA
  decorrelation under by-design trajectory divergence, concentrated in sparse
  embedding rows).

**Conditions attached to this ruling** (binding on the main analysis):

1. This gate deviation — seed 2024, dump_epoch 1, m HT side, margin −0.0809, 1/16
   gated cells — must be cited as a limitation wherever S5 mechanism conclusions are
   stated, together with the reason (above) that it does not impugn identification.
2. Mechanism interpretation for seed 2024's HT/DH contrasts should note that at epoch 1
   the signed momentum of HT had already decorrelated from T's (cos 0.041) — i.e.
   early-training momentum-state claims for this seed rest on epoch 0 and on the v
   state, not on epoch-1 m proximity.
3. If any downstream analysis produces seed-2024-specific anomalies, this cell is the
   first place to re-examine, and ruling A should be revisited.

No code or data was modified during this investigation; all findings are reproducible
from `experiments/stage_S5_outputs/stage_S5_moment_audit.csv`,
`experiments/stage_S5_raw_results/{F,T,HT,DH}/seed2024/joint/moment_state_epoch1.pth`,
and the four arms' `gradient_step.csv` / `initialization_audit.json`.
