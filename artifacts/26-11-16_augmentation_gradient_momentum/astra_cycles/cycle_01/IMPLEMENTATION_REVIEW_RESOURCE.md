# REVIEW REPORT — RESOURCE-1 correction

1. **Verdict: PASS.** No blocking issues; one MINOR test-only finding. Independently approved evidence reuse applies only to this exact correction and source.

   Current source: `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. Original source: `2ee2bdddb67861e65449cb3577c1d835f13395000a90b4ac101508c855f7b1a1`.
   Unchanged seven-file design: `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`.
   Original smoke SHA-256: `4d50a42e5a56e0c2df2392b18ae40cec8768714f4315b63c6148efe7113521f1`.

2. **Requirement-by-requirement verification**

| Requirement | Verification |
|---|---|
| Approved design and scientific invariants | PASS: Seven design files independently hashed and byte-compared to the pre-fix archive; bundle unchanged. Exactly the three specified files differ among 79 source files. All 76 other files, including core, test_core, tables, package lock and frozen scientific/storage dependencies, are unchanged. |
| Power reservation arithmetic and semantics | PASS: schedule uses max(original dataset increment, all original arm board peaks): MovieLens195.61W and KuaiRand206.31W, labeled CONSERVATIVE_OBSERVED_BOARD_PEAK_WATTS. Original telemetry is unchanged. Focused actual choose_device fixture rejects loaded283/350W and accepts idle20/350W for KR with admissible synthetic memory/host. |
| Existing admission behavior preserved | PASS: choose_device, host admission, pending allocations and actual resampling are unchanged. Strict memory AND power below85%, independent cards, and no arbitrary process cap or scientific budget reduction remain. |
| Explicit independent source/evidence supersession | PASS: check_review remains byte-identical and exact-current-source-specific. resource_1_evidence alone cannot authorize adoption: reviewed_resource_1_receipt requires a fresh exact-source PASS containing the entire matching evidence object. This reviewer explicitly approves that object after independent archive/diff/smoke inspection. Old R1 PASS alone remains insufficient. |
| One effective manifest through all affected callers | PASS: validate_attempt, validate_continuation and require_storage use effective_run_manifest. schedule creates all formal job keys from its current source; worker validation and epoch/source receipts consume the same manifest. The original source remains only the explicit storage-bootstrap and smoke identity. Receipt/review/source/budget tampering or missing supersession is refused. |
| Same owner and original receipts preserved | PASS: Original source/run/smoke/owner/guardian/handoff/parent-acceptance bytes independently compared with the archive and pinned hashes. Owner root/PID1570727/socket/endpoint/session59365 remain bound. New RESOURCE_1_ handoffs expose revised verified source and manifest/receipt hashes while preserving original filenames. No replacement/storage IPC was performed. |
| Pre-formal-only adoption and no implicit run approval | PASS: adopt_resource_1 refuses duplicate receipt, formal artifacts/nonempty output directories, registered keys/retained state/live workers, changed endpoint/session, or manual stop. It checks existing owner only and creates no server. Real supersession/dispatch/progress/launch/commit/completion artifacts are absent; owner liveness and empty RAM inventory are for parent adoption to check. |
| Mathematics, shapes, masks, routing, weighting, gradients and baselines | PASS_BY_VALID_EVIDENCE_REUSE: No change to core, worker, source identity checks or numerical helpers;40 unaffected mixed-module functions independently byte-compared. Original R1 numerical evidence is reused only for those unchanged paths. Independently compared original smoke full/tail signatures for all null arms vs SOURCE-CLEAN and delegated one-sided vs SOURCE-NDP-A on both datasets, including model/optimizer/EMA/gradient/None-mask/counter/RNG/loader signatures; six40960 FP32 arm steps each record one Adam/EMA update. Historical baseline isolation old_before==old_after. No new research conclusion. |
| Focused CPU verification | PASS_WITH_NONBLOCKING_TEST_FINDING: 20 focused pytest cases passed in3.39s; independent in-memory optimizer assertion positive/negative condition passed; three changed files compile without bytecode writes. Minor displaced assertion documented separately; no runtime defect found. |

3. **Bugs/deviations found**

RESOURCE-1 / original MAJOR blocker is corrected. No CRITICAL or MAJOR issue remains in this correction.

RESOURCE-TEST-1 / **MINOR, nonblocking**: `test_recovery.py:345` contains the original optimizer-hash assertion after the expected exception in `test_effective_manifest_cannot_downgrade`; it is unreachable and no longer executes in `test_acknowledged_continuation`. Runtime continuation is unchanged. A temporary in-memory restoration passed on the real CPU fixture and rejected a deliberately corrupted returned optimizer hash. Source was preserved. Minimal eventual cleanup is to move that assertion back after the recovery fixture's online-hash assertion; this does not block RESOURCE-1 adoption.

4. **Corrections made by reviewer: none.** Only this Markdown report and its JSON companion were written. The JSON contains the exact `protocol.resource_1_evidence(ATTEMPT)` object, explicitly approved after independent inspection, rather than trusting the coder's proposed approval. This approval reuses unchanged numerical evidence; it does not validate the original zero-watt reservation or act as an adoption receipt.

   Focused checks: **20 passed in 3.39s**. Independent optimizer positive/negative condition passed. All99 archived files,79 source identities,76 unchanged source files,40 unchanged runtime functions,seven design files and original receipt bytes were verified. Actual smoke full/tail signatures were compared without rerunning smoke. Three changed files passed `compile()`.

5. **Remaining risks:** measured board peaks are conservative budgets, not guarantees of all future peaks. Existing actual monitoring remains required. Owner liveness and zero RAM inventory were not queried; parent adoption must verify them. Both observed controls are PAUSED, manual stop false, Test false, with no stop sentinel; technical PAUSED status is not a user-stop signal. Test remains sealed.

6. **Whether experiments are safe to run:** the correction is implementation-safe for the parent's one-time adoption. **No execution approval is issued; FORMAL30 remains unstarted pending parent adoption and live run gates.** Same owner PID1570727/session59365 must be retained. No GPU, storage IPC, adoption, smoke/formal training, source/design/shared-state change, commit or publication occurred.

Unchanged core/raw-data/broad suites were not rerun; valid prior evidence was reused. Return to the original parent for permanent closure of this reviewer; never resume or reuse.
