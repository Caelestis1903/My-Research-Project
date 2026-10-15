# REVIEW REPORT: F0 v1.5 implementation v6 — pre-qualification review

1. **Verdict: FAIL. `qualified_RUN_precheck=false`.**

Independent reviewer `/root/f0_v15_implementation_review` inspected approved DESIGN v2, exact v6 implementation report SHA `b4a0dfe4331f8abcc12da6d5e21c44495f1f979f0198d231fe88b4f665c33b6e` and frozen test SHA `46005445c78a471cc998ced47264e7172a81701bdd2314f351c488fe6ef38b06`. Runtime successor stays `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`. The original v5 missing-file race is corrected, but a different, independently reachable partial-publication race remains. **Separate long qualification RUN remains NO-GO until corrected/reviewed. Full qualification is not closed. Production/scientific RUN and release remain NO-GO; B6 OPEN/FAIL.**

This review is limited to static/focused isolated CPU checks; no long E2E or training was launched, and no implementation or production source/artifact was edited.

2. **Requirement-by-requirement verification**

| Contract | Actual v6 behavior | Result |
|---|---|---|
| Wait for later resume/child markers | parent calls `_wait_for_peer_rendezvous` before parsing five marker records | V5 missing-file race corrected |
| Strict snapshot → peer-wait → peer-claim → supervisor-resume → child-defer ordering | helper checks strictly increasing numeric timestamps | Implemented; positive helper test passes |
| Early supervisor exit | missing-marker loop polls producer; failure captures returncode/output and marker state | Implemented; independent early-exit test passes |
| Timeout/order diagnostics | bounded deadline and common failure record paths exist | Implemented for absence/order errors; partial JSON bypasses them (V6-1) |
| Complete marker publication | helper equates `is_file()` with readable complete JSON at test lines 56–71 | FAIL, V6-1 |
| Process report | v6 correctly reports pytest + isolated supervisor + child = at most three, under five | V5 false peak corrected in v6 report; no two-process claim relied on |
| C1/C2 runtime corrections | runtime unchanged from reviewed v4; identity joins and observation constraints retained | No new runtime change or regression observed |
| Actual full qualification, DESIGN §§8/10 | long changed-peer E2E and complete acceptance obligations explicitly not run | Pending separate qualification, not established by seven focused passes |
| B6 | report/disposition retains OPEN/FAIL, current active hold/evidence | Unresolved; no production authority |

3. **Bugs/deviations found**

**V6-1 — MAJOR, blocking pre-qualification: existing-but-incomplete marker raises before failure evidence is archived.** `_wait_for_peer_rendezvous` in `experiments/tests/test_stage_F0_successor_v15.py:56–71` waits only for every path to be an ordinary existing file, then immediately calls `json.loads(path.read_text(...))`. Embedded producers still use ordinary `Path.write_text`, which opens/truncates the path before writing and closing it. A reader can see that existing empty/partial path under a normal scheduling interruption. JSON/read errors are outside the common failure block at lines 81–106; they raise directly and produce no `rendezvous_failure.json`.

This is independently reproduced using the actual project Python 3.13 file-write implementation, not an invented permanently malformed payload. A real subprocess called `Path.write_text` for the fifth marker. A trace callback paused it at the actual `PathBase.write_text` scheduling point **after its `open(mode='w')` and before `f.write(data)`**. Four earlier markers were complete; the fifth existed with size 0 and its producer was alive. Calling the copied, unchanged v6 helper raised **JSONDecodeError** (`Expecting value: line 1 column 1 (char 0)`); diagnostic directory and failure JSON were both absent. Releasing the producer completed its same ordinary write, process exited 0, and the helper then accepted the same five markers. The pause represents a scheduler preemption at a reachable boundary; no runtime/helper/file-publishing code was monkeypatched.

Reproducer: `/tmp/f0-v15-v6-review-20261014-a/partial_publication_probe.py`. Raw result: external `archive/partial-publication-probe.json` and `.stdout.txt`. Exact hashes are in evidence JSON. First probe setup traced the outer Python 3.13 Path.write_text delegator, which contains no open/write local, so the intended scheduling point was not reached; that initial failed setup output is retained. The corrected probe traces the actual PathBase implementation and reaches the intended boundary.

Require complete publication before parsing: use an atomic producer publication contract or a bounded retry of incomplete/readable JSON with early-exit/deadline evidence handling. Read/decode/invalid-structure failures must follow the same archival path before cleanup. Add this concrete delayed-publication case to helper tests. Correcting the v5 absent-file case alone does not guarantee complete JSON. Findings have been returned to coder; this verdict attaches only to frozen v6 bytes.

**B6 gate — OPEN/FAIL.** The production append/flock incident, missing trusted pre-incident protection inventory and original runner diff remain unresolved. Successful helper corrections cannot close B6. Preserve current raw evidence and hold.

4. **Corrections made**

None to repository implementation/tests/production. Reviewer used ordinary copied source clone, unique basetemp, externally archived raw results, and this report/evidence. No raw historical evidence was deleted or rewritten.

5. **Independent checks and evidence limits**

Independent copied-source CPU helper checks through mandatory project task wrapper and unique absolute basetemp returned **2 passed in 0.21s** (raw exact timing is also retained in evidence). They execute the successful delayed-marker and early-exit paths. The raw stdout is `/tmp/f0-v15-v6-review-20261014-a/archive/independent-rendezvous.stdout.txt`. At most two project Python processes were involved in focused/probe checks; no CUDA/training was requested.

The separate actual-write scheduling probe exited 0 after confirming the expected uncaught JSONDecodeError/no-archive case and post-publication helper success. Positive focused passes therefore coexist with a demonstrated failure schedule. This is an isolated CPU reproducibility result, not long E2E evidence.

Coder raw seven-case stdout `fa4a9a82967353cefdeab95317b1b7614e623061babc255833d883911886d20a` and two-entry manifest `e50c4ceaa958b136c337a7c676187ce51c916f77c2dc9413bffe5592cfc2981e` match supplied hashes. The manifest binds that stdout and retained v5 output manifest; it has no self-entry. Full qualification remains separately required: actual both-boundary changed-peer supervisor lifecycle/receipts/continuity, controlled long PREPARED reentry/immutable graph preservation, specified failure/restart/later-hold opportunities and current regressions. A code-review PASS would permit considering that isolated non-training qualification separately, not production admission; v6 has not earned that precheck because V6-1 remains.

Production frozen gate/hold/queue/remediation/audit hashes remain current, F140 has no materialized status/lock/run path, hold is active and successor ledgers absent. No scientific formula/tensor/gradient/model/objective/data/metric change is introduced by this test-only checkpoint. B6's missing original baseline is not reconstructed by present equality.

6. **Whether experiments are safe to run**

**Long isolated qualification precheck: NO-GO pending V6-1 correction/re-review. Production/scientific RUN and principal release: NO-GO.** Keep B6 OPEN/FAIL and active hold. No algorithmic efficacy conclusion follows this review.
