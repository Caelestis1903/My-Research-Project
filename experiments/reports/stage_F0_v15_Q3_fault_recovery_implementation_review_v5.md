# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v5

## 1. Verdict

**FAIL. qualification_RUN_precheck=False.** V4's temp-file versus destination defect is corrected, but a new MAJOR eager payload-dictionary read makes three pre-final temp-file cases fail before their recovery oracles. This is a static counterexample; no matrix or pytest was executed. Return the exact finding to coder and review v6 separately.

Historical B6 OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. No isolated Q3 qualification RUN, production action, staged graph GO, release or scientific RUN is authorized.

## 2. Requirement-by-requirement verification

| Requirement | Independent result |
|---|---|
| Frozen test/report/runtime | Test 1e7fc1382de18575510c059849015b7cec804b6fa02fc447a895cdd0cdc06df3; coder report 35fee081a0f6f5d05f8afb23cdae8e2d2cb698b8376fce4e5c1a8be246ff2e01; runtime unchanged 0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2. Frozen ordinary copies exclude successors. |
| Only harness/scientific preservation | Approved baseline03 exp/m3oe .py/.sh comparison finds only harness changed. Runtime/scientific algorithms, formulas, tensors, masks, routing, weighting and gradients unchanged; no model execution applicable. |
| V4 temp-file defect | CLOSED STATIC: four *_temp classes use same destination parent, exact basename prefix and lowercase 32-hex suffix, recorded regular FD and actual temp bytes. Destination bytes captured before exception match old source gate/pointer/hold or pending gate and remain exact afterward. Temp path is no longer required to equal destination. |
| Four intended payloads | Pending graph/release payload/released-hold PREPARED SHA/final archive and COMMITTED checks are authored, but three cannot reach validation due eager final archive read (R1). Hold's direct byte comparison is supplemented by exact prepared released_hold_sha256 and active:false, avoiding a self-comparison-only oracle. |
| Prior collision/normalization/seed fixes | Unchanged and statically intact: 21 unique phase-aware directory rows, six normalized recovery file IDs; seed phase suppresses all 12 recovery selectors, verifies exact named interruption/state before enabling recovery mode. |
| 42+8 mapping | 42 unique forward/recovery specs and 21 directory rows equal frozen v4 declarations; eight preimage cases unchanged. FD kind/path/phase/match count and exact preimage descriptor sequence retained. Declarations are not observed execution coverage. |
| Recovery/identity/immutable state | Exact PREPARED prefix/source/prelaunch/preimage bytes, same identity, different nonce before resume, later-hold preservation, incomplete admission rejection, append-once and final terminal graph checks remain unchanged outside temp payload block. Unexecuted. |
| Terminal/rename/no telemetry | Four-part v1.5+COMMITTED+launch flag+ledger digest oracle, expected publication phase, final rename surviving/lost branches and all terminal graph/ledger bytes/no query checks retained. |
| Isolation/COW/cleanup | Ledger/log/lock detachment, clone-only atomic replacement and finally cleanup retained. Future RUN must use independently built isolated ordinary-copy clone, unique basetemp/external scheduler; no production mutation. |
| Evidence limits | Coder reports compile PASS and 2/62 collect-only, 60 deselected, 0.12s; no bound raw logs independently verified here. Independent AST/static audit PASS. No harness module imported, no pytest/matrix/worker/claim/release/model/checkpoint/sample/GPU action. |

## 3. Bugs/deviations found

### Q3-V5-R1 — MAJOR: eagerly reads artifact absent at selected phase

Test :2675–2680 constructs a payloads dictionary including "final_gate_temp": final_archive.read_bytes() at :2679. Python evaluates every dictionary value immediately, before selecting payloads[expected_file_key] at :2681. Therefore final archive is read even for pending_temp, pointer_temp and hold_temp.

| Selected boundary | Visible ledger phase | Final archive |
|---|---|---|
| pending_temp:file | RELEASE_ARTIFACT_DURABLE | Absent |
| pointer_temp:file | PENDING_GATE_DURABLE | Absent |
| hold_temp:file | RELEASE_POINTER_DURABLE | Absent |

Runtime appends COMMITTED at :2475 and creates final archive only at :2485. These three injections stop earlier at temporary-file fsync :633, so no final archive exists. Thus each has a statically inevitable FileNotFoundError at test :2679 rather than a successful phase-specific payload assertion. This defeats DESIGN §10:162's exact visible-state/recovery test. The final_gate_temp:file case is later and its final archive legitimately exists; it is not the failing case.

Required correction: construct/validate only the selected file class's payload, reading final archive exclusively when selected class is final_gate_temp. Preserve regular/UUID/parent/destination-preimage checks, pending/release exact bytes, hold PREPARED digest/active flag and final COMMITTED/archive checks. Do not create an artificial final archive early, widen exception acceptance or delete payload assertions.

## 4. Corrections made

No implementation/test/coder-report/production edits. Created new frozen ordinary copies, diff, independent AST/static-expression audit and this review/evidence only. The static audit confirms literal eager call placement and actual runtime phase ordering without running reviewed functions.

## 5. Remaining risks

Only static implementation evidence is available. Temp path construction is now correct, but the first three payload cases cannot execute as intended. After correction and independent review, actual isolated matrix execution must verify all 50 fsync specs and retained rename/recovery/terminal cases, preserving raw outputs and exact identities. Source-only preservation does not reconstruct B6 missing pre-incident inventory or prove absence of all transient filesystem/lock effects.

## 6. Whether experiments are safe to run

**NO-GO for isolated Q3 qualification RUN at v5.** Correct R1 and review the successor. A later implementation PASS permits only bounded isolated qualification RUN; production/scientific authorization remains separate. B6 OPEN/FAIL, NOT_GRANTED and NO_GO remain.

### Evidence bindings

- `/tmp/f0-v15-Q3-review-v5-20261014-a/test_stage_F0_successor_v15.py` — SHA `1e7fc1382de18575510c059849015b7cec804b6fa02fc447a895cdd0cdc06df3`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v5.md` — SHA `35fee081a0f6f5d05f8afb23cdae8e2d2cb698b8376fce4e5c1a8be246ff2e01`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/audit.py` — SHA `63af50ae97956c0a8ac685619c1b52a3d5bd20cef8b7e2969a82ef761cb314da`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/audit.stdout.txt` — SHA `ea2f7838b0424d37682f8fc1c9c622ae92d729000eea072d10764c2f4fd40f7f`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/static_audit.json` — SHA `f45ba020416e539c5412be8620198954672843c4f4cb5329272c73d82865dc47`.
- `/tmp/f0-v15-Q3-review-v5-20261014-a/source-vs-v4.diff` — SHA `834319b3e96269c6817038baea8a8be542060145ec7f7987f0bb3f80823c07dc`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v5_evidence.json` — SHA `0cd49e03f6c4c2671bb0b88289a0c9bafffc4fff8155e41c8406a74260604867`.
