# REVIEW REPORT — F0 v1.5 B6 incident disposition v1

## 1. Verdict

**FAIL / B6 OPEN (MAJOR evidence limitation); RELEASE and RUN NO-GO.** Independent reviewer `/root/f0_v15_incident_review` performed a bounded read-only review of design v2, coder implementation v2, independent implementation review v2/evidence, current incident log, and current COW fixture source. This disposition does not approve implementation, staged GO, principal release, worker admission, or training.

## 2. Requirement-by-requirement verification

| Requirement / question | Independently verified disposition |
|---|---|
| Design §7.1 raw protected preimages and pre-edit inventory; §9 exact source diff | Full v1.5 pre-edit queue/status/run/checkpoint inventory and original runner byte diff remain unavailable in the reviewed evidence. Existing v1.4 frozen snapshots/hashes are narrower historical evidence, not the missing v1.5 baseline. The repository is not a Git worktree. |
| Design §10 no production mutation by tests | Violated historically: current production `audited_worker_0.jsonl` has 138 lines, including exactly nine `CHILD_STARTED` rows at 130–138. All nine event/nonce/receipt-path/child-PID/supervisor-PID identities exactly match independent reviewer v2 evidence. They reference `.v15-qualification-*` or `.v15-child-smoke-*` clone paths. |
| Can the append be restored to the original state? | **No authenticated restoration is supported by the available evidence.** Removing a known suffix can construct a candidate prefix, but cannot establish that prefix as the exact pre-incident bytes without a trustworthy contemporaneous preimage/hash. Do not delete, truncate, rewrite, or relabel the nine events; no rollback was attempted. |
| Does COW close B6? | **No.** Actual test source at lines 460–484 replaces shared regular clone-log inodes before child dispatch; smoke/E2E call it at lines 674/832 and compare source hashes after dispatch. Reviewer v2 independently reports the selected COW test passing within `28 passed, 8 deselected`; coder reports 259 detached files, zero shared inodes/hash changes and subsequent E2E/regression passes. This supports prospective containment for those tested fixtures, not historical preservation or recovery of lost evidence. No tests were rerun for this disposition. |
| Current containment | Hold is `active=true`, ID `cd2dbf50-3a50-4b81-b976-38f1b03991a3`, with all 19 existing hold events retained. The nine worker-log incident rows are retained. Canonical gate/hold/v1.4 snapshot hashes match the bound values below. Production v1.5 release and observation ledgers are absent. |

## 3. Bugs / deviations and required disposition

B6 remains an actual production append incident plus an unrecovered preservation-evidence gap. **STRONG** evidence establishes the current log, its nine incident identities, and current active hold. Existing COW source and independent focused test provide **MEDIUM** bounded evidence of repeat prevention in those fixtures. Evidence for complete historical pre/post equality is **WEAK / unavailable**; passing later tests cannot strengthen a missing contemporaneous baseline.

Keep the incident disclosed and B6 unresolved in subsequent reports. Preserve current raw evidence and bind any newly collected inventory explicitly as a post-incident baseline. Return remaining implementation blockers to IMPLEMENT, then require fresh independent REVIEW and separately reviewed staged graph GO under the approved workflow. This reviewer cannot waive design §7.1/§10 or retroactively manufacture compliance. Any proposed forward disposition of the irrecoverable gap needs an explicit designed and independently reviewed evidence policy; this artifact grants no such exception.

## 4. Corrections made

None to source, tests, production artifacts, or existing reports. Only this new disposition report was written. No Python process, spawned-child test, production lock, release, claim, training, or rollback was executed.

## 5. Remaining risks / statements that must not be made

Do not declare “all historical tests were production-mutation-free”, “original log restored”, “full pre-edit inventory recovered”, “scientific runner byte identity established from its original diff”, “B6 closed by COW”, or “full historical protected equality proved”. Current frozen hashes prove equality only for their specifically bound artifacts and moments. The append rows establish test-child starts; by themselves they prove neither production training nor comprehensive absence of every possible other mutation. No AUC/loss, curriculum/momentum, domain-size/difficulty, or M3OE conclusion follows from this control incident or qualification tests.

## 6. Safety and evidence bindings

**Experiments are not safe to authorize on this review.** Preserve the active hold, nine appended rows, and previous implementation-review FAIL. Other B1–B5 findings are outside this narrow disposition and are not cleared.

SHA-256 values independently read from disk:

| Evidence | SHA-256 |
|---|---|
| Design v2 | `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316` |
| Coder implementation v2 | `3ce9125bb56ca7c42c17c930e125113930b14b387829714c3122f5540024b5b2` |
| Independent implementation review v2 | `78d5cebbf0bb74e6b71b8bfe2237d4aa0b4dc6fa9ad98bf8885e174aca8af23a` |
| Independent review v2 evidence JSON | `3b46d1a20f70f62fb13089c7416d4ae7c03d5bec103da64857b7787dab892353` |
| Current production worker log | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` |
| Raw current log lines 130–138, including their newlines | `ada4692312d6929b8b3f67cd261f7dc37ed6a2f3867e5787317b9ffacdb163c1` |
| Inspected COW/test source | `83d17018984f7605df03e148e2a6d5c3ed757d4799b316848ddc57b39faee07f` |
| Active hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Historical v1.4 source snapshot | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |

The current log hash is a post-incident evidence binding, **not** an inferred original hash. The slice digest binds preserved incident bytes only and is not a proposed rollback digest.

## 7. Addendum — shared production execution-state flock incident

**B6 remains OPEN / FAIL (MAJOR); RELEASE and RUN remain NO-GO.** The initial review and its hash table above record the first review snapshot. This addendum records a subsequently disclosed fixture-isolation incident and changed COW/test source without replacing that historical evidence.

The orchestrator reports that a real-supervisor E2E acquired flock through a hardlinked clone of production `experiments/stage_F0_outputs/runs/execution_control/execution_state.lock`, inode **54657656**. This is **production lock access and a runtime synchronization side effect**, even if no artifact content was changed. An empty lock can still serialize or block other production actors. It must not be classified as production-free testing merely because the file's content digest is unchanged.

The supplied timeout-diagnostic directory is `/tmp/pytest-of-xzf/pytest-267/test_real_pending_graph_releas0/v15-timeout-evidence-SERIALIZED_PRECLAIM_MEMORY_CHECK/`. It **does not exist at this review's read attempt**; `live_process_snapshot.json`, `resource_snapshot.json`, and phase-log bytes therefore could not be independently read or SHA-bound. Historical flock ownership and any previously reported diagnostic digests are attributed to the coder/orchestrator record, not promoted to independently verified raw evidence here. No hash for an unavailable file is inferred.

Independent read-only checks for this addendum confirm the production lock currently has inode 54657656, size 0, mode 664, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, and mtime `2026-10-12 20:37:08.643643767 +0800`. `/proc/locks` contains no row for that inode at the observation time. The process snapshot contains no matching pytest/F0 supervisor/trajectory-child process (apart from the read-only search command itself). The orchestrator's prior verification also reports all incident processes exited and the lock's mtime was unchanged. This reviewer has no separate pre-access stat snapshot bound here: the independently read mtime is a current value, while its before/after equality is attributed to the supplied incident record. These observations support **no current owner and no observed content-write evidence**, not proof of no historical synchronization effects or complete absence of other historical side effects.

Current fixture source now calls `_detach_clone_mutable_locks` from `_detach_clone_worker_logs` before dispatch. The helper recursively selects all existing regular `*.lock` files under production runs, detaches matching regular clone files by COW replacement, rejects remaining shared inodes, and verifies source bytes. The focused COW test explicitly includes `execution_control/execution_state.lock` and checks source/clone inode separation for every returned lock. The orchestrator reports **`1 passed in 19.43s`**; that result was not independently rerun by this reviewer, and no subprocess-producing test was run for this addendum. This is prospective isolation evidence for the revised fixture, not recovery of a pre-incident inventory or evidence that previous flock acquisition was harmless.

The reported focused invocation used `run_f0_python_task.sh` task label `v15-cow-lock-isolation-focused` and test node `experiments/tests/test_stage_F0_successor_v15.py::test_clone_worker_logs_are_detached_and_production_bytes_stay_fixed`. No readable raw command/output artifact with an independently computed SHA was supplied for the `19.43s` result; its provenance remains the orchestrator/coder record. The actual revised test source is SHA-bound below.

The current inspected test file SHA-256 is `0ad347d44219e09a1a79ddc789b35ed84a56f0ea0391ef6bd5316cddb76044f4`. Production worker-log SHA remains `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2`; its preserved nine incident rows remain at 130–138. Active hold SHA remains `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51`. No existing event, lock content, hold, source, or test was modified by this reviewer. Only this report was updated.

Required disposition is unchanged: retain both disclosed fixture incidents, the missing historical evidence limitation, the active hold and all events; strengthen isolation qualification under the approved workflow; do not certify historical no-side-effects, close B6, grant staged GO, or authorize RUN from this addendum.
