# REVIEW REPORT — F0 v1.5 successor control design

## 1. Verdict

**PASS WITH FIXES — DESIGN ONLY, conditional on resolving B1–B3 before implementation.** No implementation, coder report, or v1.5 staged artifacts exist within this review's scope. This verdict does not authorize production release or experiment RUN.

Reviewed specification: `experiments/reports/stage_F0_v15_successor_control_design_v1.md`, SHA256 `4376f30686e65bdc35045b982308b3ddcf4a4be9abd10ff1c66e76e13d5a08fa`. Reviewer: independent `/root/f0_v15_design_review`.

## 2. Requirement-by-requirement verification

| Requirement | Assessment |
| --- | --- |
| AGENTS DESIGN → IMPLEMENT → REVIEW → RUN → ANALYZE | Preserved. Design explicitly authorizes implementation/review only and requires independent actual-code/report review plus runner qualification before release. No scientific conclusion is claimed. |
| Current incident binding | Correct. Independently read current active hold and failure; hold is active with ID `cd2dbf50-3a50-4b81-b976-38f1b03991a3`, latest event is the specified worker1 failure, and failure has `canonical_claim:null` and absent receipt hash. |
| Historical v1.4 preservation | Correct design. Separates immutable historical validation from new runtime validation, archives mutable pointer preimages, preserves all hold events, C97/H51 assets and old ledgers; forbids rewriting old review/inventory evidence. |
| Old-release replay prevention | Correct design. Separate one-use v1.5 ledger/artifact; binds exact new incident hold/nonce and rejects old v1.4 identity, different successor nonce, later active hold and unexpected projection. COMMITTED replay must be read-only. |
| Release transaction safety | Prior durable evidence, pending gate while hold changes, COMMITTED before final authorization, no final-gate/ledger hash cycle, and no optimistic HOLD_RELEASED event are specified correctly. Terminal fault acceptance is contradictory: B1. |
| Fresh-only/no retry/recovery | Correct design. Frozen F140 membership plus full canonical identity and `lexists` history checks; C97/H51 preserved; explicit UNCLAIMED/lock/tree/history excludes a key; no attempt-01 or checkpoint reuse. Implementation review must verify owned newly-created lock handling during post-lock recheck. |
| Complete bootstrap approval | Correct design. APPROVED plus accepted independent verdict, scientific/runtime/control hashes and complete inventories are required at all entry points; no legacy fallback or missing-file omission. |
| GPU target semantics | Integer predicate `10*U < 9*T` correctly rejects exactly 90%, malformed/bool values and UUID ambiguity. Each card is independently queried; non-target unreadability/utilization/other-process ownership does not veto a valid card. Worker ID remains physical index. Live resource-deferral outcome is underspecified across the spawned-child race: B2. |
| Five-Python-process limit | Correct design. Two shell-reserved slots per supervisor/child, at most one child per physical GPU, two workers use four slots, all helpers/tests count toward five. Actual FD inheritance and all test subprocesses require implementation verification. |
| Failure handling | Bootstrap/corruption/UUID mismatch/abnormal claimed work activates hold and preserves peer-stop behavior; later incidents cannot be released by replay. No training/resource retry is authorized. Administrative memory deferral requires B2 clarification. |
| Observability | Sufficient intended fields: exact raw preimages/inventories/path kinds/receipt absence, separate runtime/control digests, target rows/reasons/time, disk bytes, slots/locks/claims, transaction phases, identity/receipts/failures. Separate immutable audit freshness versus live refresh/re-entry semantics needs B3. |
| Formula/tensors/gradients/baselines | Memory formula verified. No new mathematical model, tensor operation or gradient path is proposed. Frozen scientific loops/data/queue/configuration/seeds and baselines remain unchanged by design; actual preservation must be reviewed after coding. |

Read-only SHA256 verification matched specification for the canonical execution gate (`experiments/stage_F0_outputs/execution_isolation_gate.json`), active hold, immutable v1.4 release and current release pointer, v1.4 release/migration ledgers, source snapshot, frozen search gate/queue, current worker failure and stderr. The canonical gate remains v1.4; this review made no production changes and ran no Python or training process.

## 3. Bugs/deviations found

**B1 — MAJOR, blocking: terminal publication and fault criteria conflict.** Section 7.2 makes the fully bound final authorized gate the last authorization action; §7.3 says an already-published final gate permits read-only exact-graph acknowledgement and legitimate F140 progress. Section 10 nevertheless requires *every* interrupted state to reject launch until re-entry, explicitly including “final gate before return.” After publication the listed admission predicate is already satisfied; a fault before returning cannot distinguish that valid graph from successful completion. Likewise, a rename-before-parent-fsync crash cannot be inferred solely from readable artifact bytes. Clarify the publication/linearization boundary and its fault oracle: incomplete pre-publication states reject; a complete terminal graph follows the explicitly defined committed/admission and read-only acknowledgement rule. Specify what rename/fsync fault states are considered terminal. Do not leave coder to add a hidden authorization phase or to reactivate a released hold. Add the corresponding explicit boundary tests.

**B2 — MAJOR, blocking: resource deferral after child spawn lacks an explicit contract.** Section 5.3 says a transient memory NO-GO before any claim causes “no training-child spawn,” but separately requires live checks in an already-spawned child and at claim. Memory may cross the threshold between supervisor observation, child bootstrap and the serialized claim. Current actual supervisor recognizes VALIDATION_COMPLETE, QUEUE_EMPTY and INTERRUPTED receipts; any other child outcome writes a failure hold and stops the peer. Specify an identity-bound administrative resource-deferral receipt/exit/outcome for this race, recognized independently by the supervisor, with null canonical claim and proof that no canonical lock/status/tree/history was created. Restrict “no spawn” to NO-GO observed before spawn. Define whether the deferred supervisor exits or waits, without automatic failed-child retry. Missing/malformed/forged deferral evidence and non-memory bootstrap faults must retain fail-closed behavior. Test both child-preclaim and immediately-before-lock races while an eligible peer continues.

**B3 — MAJOR, blocking: immutable reviewed audit freshness versus PREPARED re-entry is ambiguous.** Section 5.3 binds exact immutable prelaunch telemetry/time and says observations have maximum age 60 seconds “at release”; §7.3 permits exact-identity administrative re-entry with unchanged PREPARED evidence, but also says newly stale target audits require new reviewed evidence before PREPARED and targets/evidence cannot change afterward. A crash lasting more than 60 seconds could make the reviewed immutable audit stale forever under one reading, contradicting exact-identity completion. Define separately the reviewed target-selection snapshot and each fresh live admission/re-entry observation. Specify whether PREPARED re-entry can refresh only live telemetry for the same frozen target UUIDs, how that observation is logged without changing hashed PREPARED evidence/final ledger after commit, and exactly when stale reviewed evidence is fatal. Require a simulated >60-second re-entry case, changed-memory/UUID case, and immutable-evidence preservation checks.

No CRITICAL algorithmic defect is identified; these are control-specification contradictions/omissions that must be resolved by the designer, not invented by the coder.

## 4. Corrections made

None to specification, implementation, production control assets, queue or run trees. This independent report is the only created artifact. Requested clarifications preserve the intended incident-bound protocol and scientific hypothesis.

## 5. Remaining risks

Actual code and coder report still require independent review. Particular checks are coherent version dispatch across every launcher/bootstrap/claim path; exact historical graph validation without old-current hash comparisons; post-lock freshness without counting the current lock as earlier history; no Python subprocess outside slot accounting; complete fault matrix and later-hold injection; actual final review/gate digest graph without cycles; and default production paths without fixture bypasses.

The reviewed design intentionally fixes the admitted target subset for its one-use transaction. A later eligible non-target requires separately reviewed admission authority; no implicit worker addition is allowed. This is consistent with admitting any currently eligible single card and must remain visible in operational logs.

## 6. Whether experiments are safe to run

**NO-GO.** Resolve B1–B3 in a new immutable designer revision and review that revision before coding this successor. Then follow IMPLEMENT → independent actual-code/report REVIEW → isolated RUN qualification → staged-graph review → principal release. Keep the current hold active and v1.4 history untouched. This is a design clarification requirement, not a request for new user permission.
