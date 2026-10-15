# REVIEW REPORT — F0 v1.5 successor control design revision 2

## 1. Verdict

**PASS — DESIGN ONLY. B1–B3 are resolved; no remaining design blocker identified.** Implementation may proceed under the reviewed specification. This is not implementation approval, staged-gate GO, principal release or experimental RUN authorization.

Independent reviewer: `/root/f0_v15_design_review`. Specification: `experiments/reports/stage_F0_v15_successor_control_design_v2.md`, SHA256 `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`. Compared the actual v1/v2 text and current control artifacts; did not rely solely on the designer's verdict.

## 2. Requirement-by-requirement verification

| Requirement | Verification |
| --- | --- |
| B1: terminal release/publication | Resolved. Final publication defines completion independently of function return. Pre-publication incomplete graphs reject. Exact visible terminal graph after rename is fenced by file/parent fsync under execution-state serialization before acceptance; a non-surviving rename permits only exact-identity predetermined publication recovery. Crash after durable publication permits read-only acknowledgement and legitimate progress, without hold reactivation or hidden completion bit. Tests now distinguish these states explicitly. |
| B2: memory race after child spawn | Resolved. Pre-spawn NO-GO creates no child. Post-spawn preclaim threshold NO-GO has dedicated schema/outcome and exit 75, random per-invocation pipe-delivered secret, full-core HMAC, exact identities, phase and independent canonical-absence proof. Exit code alone is insufficient. Live serialized check precedes any key mutation. Valid deferral leaves hold/peer untouched, exits supervisor orderly and releases slots, with no automatic respawn. Malformed/replayed/forged evidence or any actual failure retains hold/peer-stop behavior. Both races and peer continuity are required qualification cases. |
| B3: freshness and PREPARED re-entry | Resolved. Reviewed immutable target-selection/source audit remains historical evidence; fresh same-target live observations control each decision and age out after 60 monotonic seconds. Re-entry after >60 seconds refreshes live observations without modifying PREPARED, audit, targets or immutable graph. Separate observation log is excluded from immutable/COMMITTED ledger hashes, preserves malformed tails, and cannot justify evidence/nonce substitution. Post-COMMITTED publication refresh does not append release ledger; terminal acknowledgement neither queries telemetry nor appends logs. |
| Historical v1.4 ledgers and old-release replay | Preserved. New one-use successor release/ledger binds new incident hold/nonce and old historical release/ledger separately. Historical validation does not require old current-runtime hash or released hold to match successor runtime. Old ledgers/artifacts/events are immutable; later holds reject every replay. |
| Current active incident hold | Compatible. Source is exact current worker1 preclaim failure with null claim and absent receipt; staged preparation does not overwrite gate/hold/release pointer. Root may release only after actual implementation review, qualification and staged GO. No held key becomes retryable. |
| Single-card GPU threshold | Compatible. Nonempty chosen subset may contain GPU0 alone or GPU1 alone; unrelated utilization/unreadability/other-process occupancy does not veto readable target. Integer strict threshold rejects exactly 90%. Every chosen index/UUID must independently qualify; subsequent admission checks only its assigned target. |
| Transaction ordering and re-entry | Closed by design. Durable preimages/PREPARED → immutable release → pending gate/pointer → durable inactive exact hold/HOLD_RELEASED → COMMITTED → final authorized gate. Pending gate blocks hold-transition window; final graph includes immutable release/pointer/hold/COMMITTED/control/review evidence without a circular final-gate ledger hash. Exact expected intermediate bytes only; changed identity/path/hold/evidence rejects; no truncation or rollback that erases incidents. |
| Fresh keys/concurrency/scientific preservation | Preserved. Frozen C97/H51/F140, lexists/history and full canonical identity, no retry/recovery/checkpoint/formal/test, one child per GPU, shell-reserved supervisor/child slots, all Python processes within five, unchanged scientific loops/data/queue/seeds/baselines. |
| Observability/workflow | Sufficient. New explicit phase/deferral authentication and separate live-observation log supplement immutable incident/protection inventories, exact digests, resource/slot/lock/claim and transaction evidence. AGENTS stage order and raw-only runner role remain intact. No control qualification implies an AUC/loss conclusion. |

Actual read-only SHA256 checks matched v2's design for canonical v1.4 gate, active failure hold, immutable v1.4 release, committed release ledger and committed migration ledger. These remain the same bytes checked during v1 review. No evidence of an intervening production transition was observed.

## 3. Bugs/deviations found

No remaining CRITICAL, MAJOR or MINOR design defect identified. The v2 changes answer the three earlier MAJOR blockers without changing the scientific hypothesis or historical release semantics.

## 4. Corrections made

None. This report is the only new artifact. No specification edits, production writes, Python processes, qualification tests or training were performed.

## 5. Remaining implementation risks

Actual-code review must establish that the specified contracts are implemented, including:

* Execution-state locking is coherent across claim, supervisor, child, release and durability fence; an already-held lock must not be reacquired in a way that deadlocks, and all paths must read/revalidate the same locked graph.
* Peer progress attribution during deferral must accept authenticated **in-progress** peer claims and distinguish them from the deferred child's mutations. A peer must not be required to finish training merely to authenticate its legitimate current ownership. Missing/ambiguous ownership remains failure. Actual tests must exercise a continuing live peer, not only a completed peer receipt.
* Deferral secrets must remain pipe/memory-only, FDs must not leak across later invocations, and neither telemetry/phase fields nor exit 75 alone may bypass independent absence and identity checks. Child receipt publication must not be overwritten by the runner's legacy RUNNER_NONZERO handler.
* Runtime observation logging must not silently alter COMMITTED ledger/final-gate digests or immutable source inventories; terminal acknowledgement must tolerate legitimate F140 progress while rejecting every later incident hold.
* Post-lock fresh-key recheck must distinguish the current newly-created lock from earlier durable history without permitting a crashed or previously touched key; tests/readiness must not create production key artifacts or exceed task-process slots.

These are verification obligations under the existing v2 requirements, not new algorithm or protocol requirements. The selected target subset is immutable after PREPARED; a transient failure of a selected target defers that administrative transaction, and adding another target requires separate reviewed authority.

## 6. Whether experiments are safe to run

**Not yet.** Design is approved for implementation. Experiments require the subsequent actual implementation/report review, runner qualification with the specified fault and deferral matrices, separate staged-graph GO and principal release. The present active hold remains in force; do not replay v1.4 release or launch workers on this design verdict alone.
