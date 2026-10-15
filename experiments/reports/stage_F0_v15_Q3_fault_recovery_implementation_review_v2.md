# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v2

## 1. Verdict

**FAIL. qualification_RUN_precheck=False.** Three MAJOR blockers remain. The 30+8 authored syscall cases are counted and mapped independently, but the first new case has a statically reachable oracle error, recovery-only fsync paths are missing, and two interrupted PREPARED states lack nonce-isolation checks. No matrix was executed; the predicted exception is a static control-flow counterexample, not an observed pytest result. Return v2 to coder and separately review v3.

B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. This FAIL does not allow isolated Q3 qualification RUN, production action, staged graph GO, release or scientific RUN.

## 2. Requirement-by-requirement verification

| Requirement | Result |
|---|---|
| Frozen source/report identity | Actual test SHA de825ab0e3305775984652d00a9005c11755a7e0722eb5e0189bc37fa9a54a21; coder report SHA be4e30de42079d511be5aa402d5e42203e2b41e5e79272f7d90773f42dd3821b; copied ordinary snapshots before successor changes. |
| Only harness changed | PASS within approved baseline03 exp/m3oe .py/.sh inventory: only this test differs. Runtime SHA 0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2 unchanged. No algorithm/formula/tensor/mask/routing/gradient behavior changes. No universal filesystem preservation claim. |
| Count concrete forward fsync cases | AST independently finds 30 unique boundary_specs at :2312–2343: 6 immutable archive fsyncs, 10 later release-ledger event fsyncs, 4 observation fsyncs, 8 replacement-temp/rename-parent fsyncs, 2 terminal-gate fence fsyncs. Preimage loops add 8: three preimages ×2 plus PREPARED append ×2. Each map is stored in evidence JSON. |
| File versus parent fsync | Selected hook runs real_fsync first then raises, correctly representing interruption after successful fsync. FD fstat distinguishes regular vs directory; parent roots distinguish canonical gate parent vs control parent. Regular classification uses basename/event/stage/payload rather than full path for many classes; present transaction has unique names, but recovery classes are absent (R2). |
| Ledger/observation distinct events | Release ledger last-row event distinguishes five event callsites; observation last stage distinguishes LOCKED_RELEASE_DECISION and PRE_HOLD_RELEASE_DECISION. Correct event-prefix expectations are explicitly stored. Data fields target.index/uuid agree with actual runtime observation schema. |
| Full concrete callsite coverage | FAIL R2: forward 38 is not every recovery syscall. Existing immutable archive fsyncs and mutable _durability_fence_path calls are never faulted. |
| Preimage/PREPARED state oracle | Improved: original gate/hold/pointer/source/prelaunch bytes, ordered archive presence, no optimistic release artifacts, admission rejection, later active hold preservation. Exact PREPARED first line is preserved and PREPARED/COMMITTED counts one. The two interrupted PREPARED states still defer alternate nonce until terminal recovery (R3). |
| 30-case exact visible phases/admission | FAIL R1: source v1.4 gate is mislabeled terminal by launch_authorized alone. First nine cases attempt absent final archive. Expected final publication set exists but does not drive/validate is_terminal. |
| Different nonce/later hold | 30-case matrix tests valid nonce-derived foreign identity before telemetry and preserves eight transaction/log paths. Later hold preserves hold and transaction graph; observation log is omitted from the nonterminal later-hold graph snapshot, so no promise of zero nonterminal diagnostic observations is inferred. Added terminal later-hold check includes both ledgers and no telemetry. Preimage cases have correct pre-PREPARED hold rejection; R3 applies once PREPARED exists. |
| Exact same-identity resume | 30 cases assert ordered six release events exactly once, unchanged PREPARED first-row bytes/source/audit and matching final canonical/archive. Eight preimage cases assert exact captured PREPARED line when already present, one PREPARED/COMMITTED, unchanged source/audit. These assertions have not executed. |
| Rename visible/nonvisible | Existing selected post-rename-before-dir-fsync hooks remain for pending gate, pointer, hold and final gate. Final surviving branch :1706–1725 and alternate pending-byte restore :1915–1928 remain. No real restart/power-loss simulation is claimed. |
| Terminal all bytes/no telemetry | Written terminal snapshots compare six graph files plus release/observation ledgers; query counter stays fixed. No event append is inferred from exact preserved log bytes and runtime terminal-return flow. These do not repair earlier oracle/callsite gaps. |
| COW/cleanup | PREPARED case release/observation ledgers detached before append; worker logs/locks detach, mutable pointers use atomic replacement; finally removes per-case/base roots and restores modules/path. RUN must still start in an independently built ordinary-copy isolated clone, with unique basetemp and external scheduler. |
| Basic-check evidence | AST parse independently passes. Coder says 3 collected and compile exit0; no bound raw stdout/manifest independently verified here. Report honestly says all fault matrices NOT EXECUTED. Collection is not execution evidence. |

### Concrete boundary map

| Logical family | Cases | Actual runtime callsites/helper |
|---|---:|---|
| Three source preimages | 6 | :2295–2302 → durable_bytes_create :604–621 |
| PREPARED event | 2 | :2321 → _append_jsonl_durable :559–575 |
| Release/pending/final immutable creation | 6 | :2404/:2412/:2485 → durable_bytes_create |
| Five later release events | 10 | :2407/:2426/:2439/:2463/:2475 → _append_once → append helper |
| Two live observation stages | 4 | :2273/:2452 → fresh_decision → observation append helper |
| Four atomic replacements | 8 | :2417/:2434/:2457/:2490 → temp file fsync :633 and parent fsync :639 |
| Final terminal fence | 2 | :2496 → validate_terminal_graph → _durability_fence :687–696 |
| Four post-rename/pre-dir interruption hooks | 4 selected points | Same atomic replacement calls after os.replace :636 and before parent :639; retained named matrix |
| Existing immutable archive recovery | **6 missing fsync points** | durable_immutable existing branch :2398/:2400 × release/pending/final |
| Expected mutable path recovery | **6 missing fsync points** | _durability_fence_path :704/:707 at callers :2423/:2437/:2460 |

The two fence helper implementations are distinct concrete syscall callsites. Normal final-gate fence cases do not cover _durability_fence_path or existing-archive branch merely because all call os.fsync.

## 3. Bugs/deviations found

### Q3-V2-R1 — MAJOR: old authorized gate mistaken for current terminal graph

Test :2490 sets is_terminal = terminal_gate.get("launch_authorized") is True. The frozen source gate, independently read from accepted baseline03, SHA f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64, has launch_authorized:true at :42 and schema F0-EXECUTION-ISOLATION-v1.4 at :132. It is not a v1.5 committed terminal graph and the current incident hold is active.

At first case observation:LOCKED_RELEASE_DECISION:file, interruption occurs before release/pending/final creation; canonical source gate is unchanged. Consequently :2504 takes final_archive.read_bytes(), although that file does not exist. This statically predicts FileNotFoundError. The same problem affects the first nine specs through pending_temp:file. If byte assertion were bypassed, :2537–2547 would also choose the wrong admission expectation. DESIGN §7:127/129 and §10:162 explicitly require incomplete graphs/old authority to reject.

Required correction: derive expected phase from boundary, assert source/pending/final bytes and matching schema/transaction-state/ledger identity, and assert actual terminal classification equals the expected phase. Do not rely only on actual launch_authorized or existence to choose an oracle.

### Q3-V2-R2 — MAJOR: recovery fsync branch coverage still absent

Test base :2295–2304 contains PREPARED only, then each initial fault call starts from this source state. The fsync monkeypatch context :2468–2475 ends before later recovery :2611–2614. Thus first-pass creation/replacement injections never cover runtime durable_immutable existing branch :2398/:2400 for release/pending/final artifacts, or mutable-path fence helper :699–707 called at :2423/:2437/:2460.

These are twelve logical file/parent recovery points. identify_regular_file :2388–2414 also returns no class for a pending canonical gate, current release pointer or current hold regular FD, so adding an interrupted state alone will not inject these fences. The terminal_fence class only recognizes canonical launch_authorized:true, and executes a different helper. DESIGN §10:162 requires every relevant file/fsync boundary, including recovery; Q3 v1-R1 is only partially closed.

Required correction: add exact partial-state roots that reach each recovery helper branch, classify the concrete file/path/callsite and phase, interrupt both successful file and parent fsync, and apply independent state/nonce/later-hold/admission/restart/terminal oracles. No runtime/scientific redesign is requested.

### Q3-V2-R3 — MAJOR: foreign nonce not tested at interrupted PREPARED fsyncs

In preimage matrix, label=prepared_ledger already has durable PREPARED at :2107–2118. It then resumes the original identity at :2184, and only after completion tests foreign nonce at :2205–2232. Therefore the actual PREPARED file-fsync and parent-fsync interruption states have no alternate-identity rejection before recovery. This remains the durable-state part of Q3 v1-R2 and DESIGN §7:133/137, §10:162.

Required correction: for those two cases, perform the valid alternate identity negative immediately after fault while PREPARED and source projections remain interrupted, preserve graph/log bytes and query count, then proceed to later-hold/original recovery. Pre-PREPARED preimage cases have no durable nonce to compare; not requiring a foreign-nonce rejection there is justified.

## 4. Corrections made

No implementation, test or production edits. Froze reviewed source/report to a new external directory, inspected AST and data, captured diff, and wrote this report/evidence. One initial reviewer audit heredoc had a bracket typo and failed at parse time before any execution; corrected archived audit then succeeded. This was not a harness/test failure and is recorded separately in static audit metadata.

## 5. Remaining risks

No matrix, E2E, pytest or model/checkpoint action was run. The static counterexample is reachable from exact runtime order and archived source bytes, but not presented as observed execution. Authored remaining assertions cannot establish passing qualification until fixed, reviewed and separately executed with raw archival evidence. Coder basic checks remain report-only. Source comparison covers approved source scope, not every transient filesystem/lock state. Scientific formulas/tensor/gradient checks are not applicable because scientific source is unchanged.

## 6. Whether experiments are safe to run

**NO-GO for Q3 isolated qualification RUN at this checkpoint.** Return all three blockers to coder. A future independent implementation PASS may authorize only an explicitly bounded isolated qualification RUN; actual qualification acceptance is a separate review. B6 OPEN/FAIL, NOT_GRANTED, NO_GO and the bans on production release, staged graph GO and scientific RUN remain regardless of harness progress.

### Evidence bindings

- `/tmp/f0-v15-Q3-review-v2-20261014-a/test_stage_F0_successor_v15.py` — SHA `de825ab0e3305775984652d00a9005c11755a7e0722eb5e0189bc37fa9a54a21`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v2.md` — SHA `be4e30de42079d511be5aa402d5e42203e2b41e5e79272f7d90773f42dd3821b`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/audit.py` — SHA `5c4adcc83de94f7b2a4bd9fabdfd83bb392d214f87b52ff0ae3340b0b31d2659`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/audit.stdout.txt` — SHA `ba2745e1c3d04fc2448270d579da25f055a0cee3b64723359bcd500d958b9d21`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/source-vs-v1.diff` — SHA `0133c444edd265fd749b50ea5de106a8f613815d8415afa68d7312460d1a5ba6`.
- `/tmp/f0-v15-Q3-review-v2-20261014-a/static_audit.json` — SHA `6d6821f2b61aee31dbdfa24b970ec9e86e1717f834386ece35bd49742b7166b6`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v2_evidence.json` — SHA `beac7020c9e0ff585fccc7381b8b0aae04fcbadbda424cac8643818ff51d386f`.
