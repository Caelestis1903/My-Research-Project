# REVIEW REPORT

1. **Verdict: FAIL.** One **MAJOR** blocking deviation remains; no CRITICAL or MINOR issues were identified. The numerical core passed the focused CPU checks. The infrastructure's required continuation identity check is incomplete. No source is authorized for execution.

   Both final coder reports and actual files were inspected. The observed 78-file aggregate is `4690be03973b1f9da4173ffd436cd09a49c4e27c75fd40f1c719b535f338c881`, matching the final infrastructure report. This is an inspected identity, **not** an accepted source binding: JSON `source_hash=null`. The approved seven-file design remains `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`; CODEX_EXECUTION_PLAN and the normative addendum govern.

2. **Requirement-by-requirement verification**

| Requirement | Verification |
|---|---|
| D1: Approved seven-file design, overrides, exact candidate and constants | **PASS** — Bundle independently recomputed as f682…9754 and DESIGN_APPROVAL PASS; only AXIS-EVEN, AXIS-IID2, ISOTROPIC-EVEN; alpha=.03, mixture=.25. |
| D2: Exact 30 trajectories / 1500 epochs / 51750 Adam and EMA updates; FP32 40960 and seeds 2022–2026 | **PASS_STATIC** — Scheduler Cartesian product and worker epoch range match the manifest; actual tails retained; per-trajectory and total update assertions. Formal execution not performed. |
| D3: Frozen NDP-A probe and source embedding/routing/gate semantics | **PASS** — Unchanged make_views/forward/sensitivity hashes verified; full-support detached deltas, original x routing, trainable ordinary embeddings, training gate detach and probe-only full derivatives. |
| D4: Natural full-2B support and .75/.125/.125 coefficient mass including zeros and tails | **PASS** — Actual core and CPU ledger checks preserve every row-task and task mass .5; no selected-subset or population balancing. |
| D5: Exact target-only objective, prescribed four view forwards and one backward | **PASS** — a_tj=BCE(view[:,t],y[:,t]); R1=.5*(a01+a11), R2=.5*(a02+a12), V=.5*(R1+R2), total=.75*C+.25*V. Required order and actual-step state equality fixtures pass. |
| D6: Even/IID2 expectation and actual input-norm isotropic control | **PASS** — Exact four-sign-pair expectation fixture and reversal symmetry pass; independent row/task/replica signs; Gaussian radius inheritance and registered zero-q fallback. Measured norm error 1.6109599698665988e-7 <=1e-6. |
| D7: Keyed private view RNG and private source loader | **PASS** — Literal UTF-8 SHA256 key, unsigned little-endian seed, registered streams/draw shapes and no arm/GPU field verified. Global/loader RNG invariance passes CPU checks. Runner loader seeded once seed+11000. |
| D8: All-zero exact CLEAN identity, source Adam None/zero/counter/WD and EMA once | **PASS** — Canonical full/tail source gradients, None masks, model/moments/counters/EMA match exactly. Source activity and coupled decay fixture passes; one Adam and source EMA call per actual arm. |
| D9: Exact historical CLEAN/NDP-A controls and baseline preservation | **PASS_READ_ONLY** — 20 pinned controls, 10 paired initial hashes, 500 epoch orders and frozen source dependencies pass. d443 baseline unchanged; all 50 old result files equal frozen archive; 9 tables match completion hashes; whole journals/results/index unchanged during review. |
| D10: Cycle-root frozen serve/request, BytesIO-only RAM and protected old storage | **PASS_STATIC** — Only new-root request wrapper; unmodified frozen serve import/call; no new-path ensure_guardian/driver/detached launch. Root routing mocked checks pass. No real owner/IPC was started. |
| D11: Verified continuation and retained-state identity before reattachment | **FAIL** — Endpoint/job/epoch checks exist; required continuation tensor/counter/metadata comparisons against acknowledged epoch records do not. |
| D12: Parent-owned storage lifecycle and explicit handoff/release | **PASS_STATIC_RUNTIME_PENDING** — Distinct endpoint/session receipts, COMPLETE selected hash/compaction checks and explicit cycle-only release/shutdown gates present. Actual parent-owned tool-session survival and acceptance are RUN gates, not established here. |
| D13: Owned scheduler worker pipes/signals, both stop roots, no heartbeat TTL or process-count cap | **PASS_STATIC** — Popen uses owned stdin PIPE/PID and no detach; EOF/actual parent loss/signals pause at atomic boundary; both manual-stop controls read; short resource polling has no lease expiry or worker-count cap. |
| D14: Per-card memory AND power <85%, host <90% including pending/snapshots | **PASS_STATIC_AND_FOCUSED_CPU** — Focused independent-card rejection/admission, pending and host checks pass. Actual measured reservations/power and headroom remain smoke/RUN gates. |
| D15: Batch/epoch logs, validation-only selection, full raw metrics and registered seed contrasts | **PASS_STATIC** — Required diagnostics schedule, hashes/costs/support logs, strict-greater independent online/EMA selection and E10/E30/E50 implemented. Export requires 30+20 exact coverage; n=5, ddof1 and exact CI multiplier; no ranking or scientific conclusion. |
| D16: Current-source review/smoke gate and Test sealed | **PASS_STATIC_NOT_AUTHORIZED** — check_review requires accepted verdict, empty blockers, current source, approved bundle and cpu_checks_passed=true. Smoke requires both datasets/all arms/full and tail source identities. This FAIL authorizes neither storage bootstrap nor GPU smoke/formal execution. No Test evaluation path or reviewer Test access. |

3. **Bugs/deviations found**

   **R1 — MAJOR / BLOCKING: unverified continuation state can be accepted for recovery.**

   DESIGN_COMPLETENESS_ADDENDUM §10.4 explicitly requires “liveness/source/state-hash checks before reattachment.” Its §7.3 and CODEX_EXECUTION_PLAN §9 require verified continuation from the acknowledged RAM epoch.

   In [runner.py](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/task_axis_even_views_20261116/runner.py:586), `recover()` compares only the snapshot job and epoch with the owner record. It then calls recovery or clears `PAUSE_TRAINING`. It never reads the matching acknowledged epoch record or compares model, EMA, optimizer, global RNG, loader, selected-state hashes, step/counts or selection metadata. [The worker load path](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/task_axis_even_views_20261116/runner.py:199) checks job/initialization and directly loads the continuation. `attach()` checks endpoint/source receipts; `handoff()` compares selected hashes only for COMPLETE records. Those checks do not supply the missing unfinished-state verification.

   **Independent evidence:** a focused read-only CPU fixture called the actual `recover()` with its already-validated outer prerequisites stubbed. The snapshot matched job/epoch but contained a changed model tensor, inconsistent global step/counts and missing continuation components. The function returned `{"status":"RECOVERY_READY","key":"fixture","epoch":1,"test_accessed":false}`, invoked no state hash check and requested pause removal. IPC and `Path.unlink` were mocked; no storage process, actual IPC, state mutation or file deletion occurred. A missing field may fail later in the worker; an otherwise loadable changed tensor has no recorded-hash comparison either.

   **Consequence:** endpoint/source/job/epoch equality is not the required proof that the continuation tensors and metadata are the acknowledged state. This finding does not claim that any existing protected RAM was corrupted.

   **Minimal correction for a NEW fresh coder:** locate the key/epoch's `RAM_COMMITTED` epoch file through the cycle-local ledger; reuse the model/EMA/optimizer/RNG/loader/selected hashes and metadata already emitted at [runner.py:253](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/task_axis_even_views_20261116/runner.py:253). Compare the returned snapshot, epoch/global step/counts and selected metadata before accepting an unfinished-state reattachment, clearing a pause/recovering, or loading continuation. Do not select a later uncommitted numeric file by glob alone. Keep frozen serve and all research formulas unchanged. One focused CPU check should reject altered state/counters without pause removal/recover calls and accept an exact acknowledged snapshot, including epoch-50 finalization without further updates.

   No algorithm redesign, new baseline or extra trajectory is requested. The other listed requirements passed static/CPU inspection or remain explicitly identified RUN gates.

4. **Corrections made**

   None. Reviewer ownership permits only these two review artifacts. No source, tests, coder reports, historical artifacts, shared state or guardian state was changed. Return R1 to the original parent for a **NEW fresh coder** and then a **NEW distinct reviewer**; do not reuse closed contexts.

5. **Remaining risks**

   Actual cache/raw-data loading, real cycle-owner IPC, parent tool-session survival across native runner closure, actual resource measurements, full-batch/tail GPU source equality and the formal matrix were not exercised. They remain RUN gates after R1 is corrected and a new accepted review binds the revised source.

   Independent preservation checks: old source aggregate remains `d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6`; all 50 old result bytes equal the hash-pinned frozen archive; all nine old tables match completion receipts; whole old TV/NDP journal lengths/hashes, complete result inventories and the twenty-control index stayed equal before/after this review. Both workspaces' manual stop flags were clear, status ACTIVE, Test false.

6. **Whether experiments are safe to run**

   **No.** This FAIL does not pass the first static/CPU review gate and authorizes **no parent storage bootstrap, GPU smoke or formal training**. The four core CPU fixture groups passed; the overall implementation gate fails because the recovery negative check exposed R1. After a fresh correction/review passes, the parent-owned storage lifecycle and actual current-source smoke must still pass before formal execution. Test stays sealed.

Skipped GPU/storage execution and broad regression checks, as requested. The concrete remaining implementation risk is unverified RAM continuation.

