# REVIEW REPORT: F0 v1.5 implementation v5 — pre-qualification code review

1. **Verdict: FAIL for the reviewed v5 bytes.**

Independent reviewer `/root/f0_v15_implementation_review` reviewed approved DESIGN v2 SHA `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`, implementation v5 SHA `7f37bb2bf7c662d86902839c9d5a04484c758a535221cbd73ba61259469e8097`, frozen test source SHA `461f30d4dac24bc80d7eb73e0cac4b3abfde5f5044a45be6c8ac83948158edda`, and prior C3/D1–D3 findings. Runtime successor remains `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.

**Precheck for a separate long isolated qualification RUN: NOT READY until V5-1 is corrected and reviewed. Full qualification criteria: NOT CLOSED. Production release/scientific RUN: NO-GO; B6 OPEN/FAIL.** Unexecuted qualification criteria are not claimed to be implementation failures merely because this is the pre-RUN review; the concrete harness race below is an implementation blocker before spending a long run. No long E2E, training or production action was performed.

2. **Requirement-by-requirement verification**

| Contract / finding | Actual v5 code/evidence | Result |
|---|---|---|
| Changed-peer snapshot ordering, §5.3 B2 | supervisor wrapper calls real capture, publishes exact bytes/hash, waits before returning state; real `_spawn_one` calls capture outside execution-state/canonical locks (worker source 919–930); child no longer waits before runner or inside deferral callback | Direction correct, parent rendezvous race remains V5-1 |
| Genuine live supervisor schema, C1 | focused fixture now starts session-leader supervisor, separate child with canonical/GPU flocks, real supervisor-owned worker/control nonce/PID and CHILD_STARTED row; runtime joins retained | Correct fixture identity model by inspection; final coder group passes; count/accounting report wrong (V5-2) |
| Fixture integrity/compiler check | embedded scripts compile and string-order checks pass | Confirms script syntax/order, not parent waiting for asynchronous marker production |
| C2 observation correction | runtime unchanged from reviewed v4, stage/reason/fraction/current graph checks retained | No new runtime regression observed |
| Diagnostic v4 D2 correction | corrected report SHA `3c89908be8dcb70dbb8abe52f81c01cfc45ff105100fccb22b71fb4d0be02d37` now matches actual sleep wchan, instantaneous owner/no deadlock proof and missing raw pytest limitation | D2 reporting corrected; v4 root cause still undetermined |
| Raw outputs/manifest | five archived stdout hashes match, final `5 passed in 0.15s`; manifest SHA `a8097d1305ad57c66913ef9d1e14b71585bea5e03050559c020a20c8a3d9625f`, no self-entry | Verified; earlier failures retained honestly |
| Full §§8/10 qualification, C3/D1 | v5 explicitly did not run actual changed-peer two-boundary E2E, controlled long reentry/full fault matrix or current broad regressions | Pending required RUN evidence, not established by focused five cases |
| B6/D3 preservation | incident disposition OPEN/FAIL, hold active; no claimed reconstructed preimages | Unresolved gate; no scientific/production authorization |

No runtime/model/objective/data/metric/runner edit is asserted or observed for v5; tensor/gradient/formula changes are not applicable. Original runner diff/full pre-edit inventory remain unavailable under B6. The v5 source/report were copied to an external review archive before subsequent coder work, so this verdict attaches to the exact hashes above, not later workspace edits.

3. **Bugs/deviations found**

**V5-1 — MAJOR, blocking long qualification RUN: parent consumes markers before producer completion.** In test `test_real_pending_graph_release_reentry_terminal_and_f140_claim`, lines 2043–2049 wait only for `peer_ready`. At lines **2063–2067**, parent immediately reads `defer_ready` and `defer_resumed`. The corrected producer order explicitly puts these later: peer publishes ready; worker0's supervisor wakes from `capture_then_signal`, writes resume and returns captured state; only then does it spawn/bootstrap the child, which writes the deferral-decision marker. Reading either later marker immediately after peer_ready is a valid scheduling race and can raise `FileNotFoundError` or observe incomplete JSON. There is no producer-liveness/deadline/complete-marker rendezvous on those reads.

Independent CPU schedule probe extracted the **actual AST marker-read group** from the frozen v5 test (2063–2067). With peer_ready/snapshot/peer-before-claim published and later resume/child-decision markers not yet published, executing that group raised `FileNotFoundError` for `defer_ready.json`. This is an admissible schedule, not a long E2E result. Raw reproduction is in `/tmp/f0-v15-v5-review-20261014-a/audit_v5.py` and archived `independent-v5-audit.json`.

Add a bounded marker-completion/liveness rendezvous before parsing later markers, preserve non-75/timeout/partial-marker diagnostics before cleanup, and verify the parent path with delayed producers. The existing compile/string-order test deliberately passes without executing this parent path, so its PASS does not resolve the race. Findings were returned to coder; later fixes require fresh review of their own bytes.

**V5-2 — MAJOR documentation/accounting correction before qualification planning: reported Python peak is false.** V5 says “never exceeded two: one pytest process and ... supervisor/child process chain.” Actual live positive source starts a supervisor using `Popen` at line 391–395; that script starts a separate child at line 373. At the live predicate assertion all three are alive: **pytest + supervisor + child = 3**, within user's maximum five but not two. Only pytest's wrapper reservation is evidenced; these fixture Popen calls do not independently reserve the two extra task slots. Correct the count, distinguish actual processes from a chain, and use truthful slot/process accounting for any subsequently run fixture alongside other tests/audits. This review did not independently rerun the three-process fixture.

Pending gate **B6 — OPEN/FAIL** is preserved separately from this code precheck. Missing incident preimages/full protected before inventory and original runner diff cannot be filled by v5 focused tests or the proposed future qualification. Prior production incident rows must remain raw and the hold active.

4. **Corrections made**

None to implementation, repository test source or production artifacts. Reviewer created an ordinary copied source clone, unique basetemp and external archive, and this review/evidence. No PASS is issued while V5-1 is unresolved. The parent has already returned these issues to coder for a subsequent revision.

5. **Independent verification and remaining qualification obligations**

Archive `/tmp/f0-v15-v5-review-20261014-a/archive` contains exact frozen v5 test/report, corrected v4 report, all five coder stdout files/manifest, the independent schedule audit and raw independent pytest outputs.

The independent clone ran only the non-spawning compile/order and missing-marker diagnostic helper checks through the mandatory project slot wrapper with unique basetemp. Result **2 passed in 0.09s**. An initial underspecified copy clone lacked stage_F0_queue at conftest import (exit 4 before collection); its raw output is retained. Copying required top-level dependencies into that same isolated clone fixed the harness setup, and the final two cases passed. No runtime validator was changed. Independent schedule probe separately demonstrates V5-1 even while the compile/order case passes.

Coder raw final five-case output independently verified as `5 passed in 0.15s`; earlier initial/subset/import/EPERM outputs all match their manifest. They are not evidence of successful actual `TrajectorySupervisor.run()` changed-peer lifecycle, controlled >60-second PREPARED reentry with unchanged audit/identity/COMMITTED graph, both race boundaries, all specified syscall fault/restart/later-hold opportunities, or current relevant regressions. Those remain required by DESIGN §§8/10 for the separately supervised qualification RUN after code precheck passes. No scientific metric rerun is warranted by this test-only revision.

Corrected v4 report now accurately states child wchan=hrtimer_nanosleep/supervisor=0, one-instant FLOCK ownership does not prove deadlock, archive payload 171 matches plus one self-entry exception, pre-kill 38 matches, and absent outer pytest raw stdout. It is reporting correction, not closure of failed qualification or reconstruction of previous evidence.

6. **Whether experiments are safe to run**

**Current v5 long qualification test: NO-GO because of V5-1. Full scientific RUN/principal release: NO-GO.** After concrete harness/accounting fixes and fresh code review, a bounded non-training isolated qualification RUN can be evaluated separately while B6 stays OPEN. Its results must not be conflated with production admission. Preserve current hold and incident evidence.
