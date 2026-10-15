# F0 execution-control v1.2 final independent review

Date: 2026-10-12 (Asia/Shanghai). Reviewer: independent REVIEW agent. Reviewed approved design §§12–13, actual source/test snapshot, coder report v1.2.1, pending gate and unchanged incident artifacts. No Python, signal, training or result-analysis actions were performed.

## 1. Verdict and blocking list

**PASS — implementation review.** MAJOR-01–04 from the previous two reports are closed; no remaining CRITICAL, MAJOR or MINOR implementation defect identified in this targeted review. No source corrections were made. This new report is the sole reviewer write.

**RUN remains unauthorized in the current state.** Launch/formal/test/recovery authorizations are false, reviewer gate verdict is PENDING, and the production hold is active. Bind this report and the reviewed code hashes reproducibly, retain all 47 held keys, and obtain the separate principal-approved, identity-bound incident release/prelaunch decision before any new execution. This implementation PASS does not authorize recovery of held keys or assert that the historical native crash is resolved.

## 2. Independently verified identities

Actual file bytes and sorted compact JSON aggregates were checked with shell SHA-256 and Perl JSON/Digest, without importing project Python:

| Artifact | SHA-256 |
|---|---|
| Approved design including v1.2 | `85b4ff62e0128b8c22e521f0c6a377f35839e348e249be2603e47623cd0dfece` |
| Coder report v1.2.1 | `25abbf6fad4bec20ee996f77b100fb31840feb5c61d6a0355c2d0d420bf95908` |
| Pending gate | `6d4ea7526763ba1925e223db2d50ba6dd658c25dcd6a400fa1e947213cae9f1f` |
| Active global hold, unchanged | `de33c8228d0ba7f0988bf01cd3bf374cbef7290dd30654109286fbd80d4d603e` |
| Prior search gate, unchanged | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Independent incident audit v3 | `a8d71510f065523a7282776debb7c790e20704290a0b820ead676a3c889c8091` |
| Actual 14-file control inventory | `8e493cdf8e61fe2d0323e1eec5483a86ff5ea4fd035d422d3512e247966f6997` |
| Actual 28-file runtime inventory | `81bd8e390af4099b82c4a8b8130e602b25bc83a50a180b70b12676073209c63a` |
| Independently reconstructed sorted 47-key set | `ccbf64945396b542ee2746f4bab37d7d2fcdaea49b00817f096facec45f7f8bf` |

All 14 individual control hashes matched the gate. Runtime inventory was independently rebuilt from actual `stage_F0_queue.IMPLEMENTATION_FILES` (28 files). Both aggregates and coder-report gate binding matched. The frozen scientific canonical identity remains `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`; executable runtime amendment has its separate digest above.

## 3. Requirement verification and previous issue closure

**MAJOR-04 / §13.4 — PASS.** Actual `stage_F0_worker_control.py:53–110` freezes a 47-key constant independent of the caller, execution gate and release document. Reviewer independently reconstructed its source: unchanged active hold top-level `preserved_held_run_ids` supplies 45; unchanged v3 audit `worker_attempts[].canonical_claim.run_id` supplies exactly `b3a55740aadd0e3377a9` and `cfbf6cc0e541e76cfb92`. This union equals the constant and gate array exactly, with 47 unique items, no missing/extra substitutions. All 47 actual queue status files read `RECOVERY_BLOCKED`.

`validate_execution_gate:165–167` refuses any nonexact set before authorization/read of search data. `frozen_held_run_ids:223–245` requires gate equal to the fixed set, rejects duplicate gate/caller entries, preserves prior search holds, and requires caller equality. `_hold_is_released:260–276` now requires that same exact set as well as current hold ID/hash/principal/release/reviewer metadata. The prior 45-item release is retained as historical provenance, not accepted by the new active-hold release validator. Gate/caller checks occur before supervisor child creation and runner queue claim; hold consumption occurs at the existing serialized claim barrier.

Mandatory tests are present: `test_stage_F0_launcher.py:373–396` rejects an approved-fixture gate with a 47-item new-key replacement or old-45 truncation. `test_stage_F0_worker_control.py:1076–1090` tests omission of either new key from caller and a complete positive case; `:1093–1126` tests old-45 and same-count release substitution refusal plus exact-47 positive acceptance. These test the production gate/caller/release functions rather than checking only the length. Generic malformed/duplicate/current-hold-hash failures remain enforced by source. Production gate and hold were not mutated for review.

**MAJOR-01 / §13.1 pidfd — PASS, no regression.** Actual operator wrapper, supervisor and child source hashes are byte-identical to the preceding v1.2 review:

- Supervisor `2689e7cc0b9f364afb2b96569615316e90b7381e775d4e5cb42000e8333cbab8`.
- Child `4343e928c266869fb9723509373da3dfe75c30ffe4de85a241a38e9a708f86e5`.
- Wrapper `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4`.

`run_search_worker.sh:29–34` enters a one-slot-pre-reserved helper. Supervisor `:273–300` opens pidfd before live identity verification, validates descriptor target and host/worker/PID=PGID/startticks/nonce, rechecks pinned target, then sends solely through pidfd. Unavailable/expired/unauthorized descriptor fails closed; no naked PID or group TERM fallback exists. Peer and operator share this path. Tests retain real wrapper stale-identity refusal/success and unavailable/exit-at-send fault cases (`test_stage_F0_launcher.py:157–239`). STOP/CONT remain negative PGID.

**MAJOR-02 / §13.2 protected receipt — PASS, no regression.** Runner SHA remains `5ad73c898063fb6abddb14251fc98201c9e5cf7cdbe9d7845aab6e74c1602599`. `stage_F0_run.py:1543–1674` freezes both held lock owners and process/canonical/attempt/checkpoint identities, holds execution-state serialization, writes matching status/manifest and commits receipt while canonical/GPU remain held. Actual writers flush/fsync complete temporary files, atomically replace, and fsync directories. Caller `:2004–2039` completes this before either resource-lock release. The last complete checkpoint supplies rollback epoch/state; no partial-epoch composition or extra recovery allowance is introduced. Child outer handling only reads/validates the committed receipt after runner invocation, and cannot create a lock-external successful interruption receipt. Supervisor independently validates immutable receipt/manifest/checkpoint/owner identities.

**MAJOR-03 / §13.3 real barriers — PASS, no regression.** Actual fixture retains manifest cleanup repeats and deterministic pre/post receipt hooks. At A, receipt is absent and independent pytest cannot acquire either resource lock; real child TERM/INT/HUP are delivered and acknowledged. At B, receipt writer has completed fsync/rename/directory fsync, both locks remain excluded, and checkpoint hash/epoch, canonical/attempt/process identities and frozen owners agree across status/manifest/receipt. Initial TERM exit reason remains; six child repeats, one supervisor forward and unchanged deadline are asserted. After unwind, canonical/GPU/worker/process-slot locks are actually reacquired; one child only and no next claim/continuation are asserted (`test_stage_F0_worker_control.py:306–329,660–796`).

Receipt-writer injected failure fixture (`:332–342,815–885`) retains no receipt or commit marker, preserves committed checkpoint, records held failure, and never advances queue or fabricates validated success. It injects before writer entry; fsync/rename failures and production outer-child behavior are inspected by source rather than separately runtime-injected here. Partial status/manifest evidence is explicitly incomplete, not a committed interruption transaction.

**§§12.1–12.3 unchanged controls — PASS by source/test inspection.** Supervisor remains the sole child stop forwarder, shared first-stop latch and deadline are not reset by repeats/global hold. Child duplicate handler raises only once. Production timeout remains 30 seconds and failure/hold/peer request/escalation event are durable before own group KILL; persistence failure preserves supervisor and stops only child. Existing timeout/default assertions, group STOP/CONT and PDEATHSIG/race fixtures remain. Fresh child per trajectory, same worker PGID, parent-death guard before runner import, two-slot worker reservation, at most two GPU workers, operator one-slot reservation and shared total Python≤5 policy remain intact. Adopted child descriptor close does not unlock the supervisor lease.

**§13.5 gate/hold — PASS, closed.** Actual launch, formal/test and recovery authorization values are all false; verdict/amendment remain PENDING. Gate/hold validation precedes runner queue load and supervisor child start. Active hold hash is unchanged. New 47-key enforcement prevents old historical releases from silently dropping the two new keys. Scientific data/model/loss/routing/selection/protocol and numerical checkpoint content were not redesigned by this repair; this review approves only control behavior.

## 4. Test evidence, corrections and risks

Coder reports reservation-wrapped phase0-3-env syntax checks, receipt-failure fixture **1 passed**, signal/pidfd focused fixtures **3 passed**, and final affected worker-control/launcher/runner regression **55 passed**, with at most pytest plus one supervisor and one child simultaneously (3 Python processes), CPU only. Reviewer independently inspected actual tests and accounting topology, but did not rerun Python or reproduce the reported outputs. Source/test review and independent byte hashes support this verdict; it is not claimed as independent runtime/CUDA reproduction.

No corrections were made by reviewer. All four blocking findings are closed. No remaining implementation fix is requested for this snapshot. The native crash cause remains UNKNOWN; CPU fixtures do not establish CUDA/native cleanup latency or guarantee future training will not crash. Full historical experiment artifact byte invariance cannot be independently proved without a prior complete snapshot. No research metrics or actual test-set evaluation were analyzed.

Experiment execution requires the separate current-snapshot gate/report binding, incident-release adjudication and prelaunch controls described above. An implementation PASS does not itself release the hold, authorize a held-key recovery, or approve formal/test work.
