# Independent execution scheduler review

## Current review — 7 GiB execution-budget amendment

Date: 2026-11-10. Verdict: **PASS**.

Current reviewed supervisor SHA-256: `ec28e47176bae0cb534f9555e75e357b14837a0bf5a588dd78a6fc731156a07c`.

Reviewed the updated execution amendment, coder report, two source-line changes and existing focused scalar check. Independently reversed only `CAP=7<<30` and its rejection-message text in memory; the resulting full source hash exactly matches the previously reviewed `f22601d6af5014095db2e001bcc35af8589580744d564e986bb917dfaaad9dc0`. Thus all forecast formulas, adoption/locking paths, guardian 4 GiB predicate, host 90% rule and per-GPU 85% checks remain unchanged.

Independently reran `check_scheduler.py` through the required project Python with `PYTHONPATH=.`: PASS. It checks the exact 7 GiB boundary, boundary+1 denial, the supplied three-KR-worker snapshot, and the existing independent guardian/host denials. The supplied physical forecast 6,965,659,752 bytes fits 7,516,192,768 bytes; its guardian predicate 4,238,324,112 bytes fits 4,294,967,296 bytes. This is conditional admission for that snapshot, not a guarantee of three concurrent KR workers. Also independently checked that the later supplied physical 7,448,875,768 / guardian 4,721,540,128-byte snapshot is rejected specifically by the unchanged guardian predicate.

PREPARED SHA-256 remains `741df1a95597f516add9ea505006dd1de72947bb24b87d859b8223d1f692a82e`. The source change is confined to the external scheduler; scientific identity and attempt remain unchanged. Historical dry receipts below retain their original identities.

No blocking findings. This PASS releases the exact new supervisor hash for the specified supervisor-only replacement after the matching review receipt is updated. Runner must confirm no attribution child is active, bind the old supervisor identity, terminate only that supervisor, wait for identity/ownership-lock release, and adopt the existing workers and guardian. The next real decision must use refreshed resource measurements and record whichever constraint actually controls admission.

No model checks, scientific analysis, source edits or process signals were performed during this narrow review. Live replacement and its first scheduling decision remain runner work.

## Historical review — 5 GiB supervisor

Date: 2026-11-10. Verdict: **PASS**.

Reviewed supervisor SHA-256: `f22601d6af5014095db2e001bcc35af8589580744d564e986bb917dfaaad9dc0`.

Scientific implementation remains `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`. Attempt2 PREPARED SHA-256 remains `741df1a95597f516add9ea505006dd1de72947bb24b87d859b8223d1f692a82e`. Both identities were independently recomputed. Reviewed amendment SHA-256: `150ea1ac24c1fbd1c7b756d2d4d204bc318665e6bc45bb2b1efa686e37fe226d`.

Inspected the amendment, coder report, external supervisor, scalar assertion check and dry receipt; followed the existing guardian commit/finalize, worker export/recovery, observation schedule, resource helper and attribution call paths relevant to this execution change. No blocking findings remain within this scope.

## Handover and scheduling

- Dry mode requests scalar status, scans processes and evaluates resource candidates without dispatch, signals or guardian mutations. The supplied dry receipt correctly reports `boundary_ready: false`; its hypothetical finalize candidate is not authorization to run.
- Actual handover requires the independent hash-bound PASS receipt and exclusive attempt-level flock. It waits for the ML-C complete receipt, guardian block registration and all five FINALIZED epoch50 rows. The old controller's PID/start/boot identity and exact attempt command are checked before signaling only that PID, then its identity must disappear before dispatch. Guardian and workers are not signaled or restarted.
- Exact-attempt process scans include workers awaiting guardian registration. Registered live workers must reconcile with process identity; both registered and pending workers exclude their jobs from candidates. Guardian launch counters plus the scheduler launch ledger bound total attempts, including its own pre-registration launch failures. Existing workers are adopted without re-registration.
- Frozen block order and one-block lookahead are retained. There is no fixed two-worker ceiling. At most one attribution child is selected, and a separate exclusive attribution lock prevents simultaneous writers. The original finalizer performs all scientific work. A complete receipt permits only the pending guardian finalization after child loss; incomplete attribution stops with anchors retained and a log path.
- Completion requires all four finalized block receipts, exactly 20 finalized epoch50 jobs, and expected block epoch/Adam-step totals before writing TRAINING_COMPLETE. The supervisor neither invokes Test evaluation nor bypasses the existing downstream analysis/freeze/evaluate/report/release flow.

## Resource admission

The external physical-tensor forecast includes current guardian payload, future retained state, active worker/history state, export/deserialization and old commit coexistence, pending anchors, and two possible fetched finalizer rows plus scratch. Only a guaranteed lower bound on replaced guardian parameter tensors is subtracted; arrays/buffers are retained conservatively. The epoch/stage enumeration preserves the existing peak-active floor and models remaining growth rather than only current idle GPU state.

The independently forecast guardian predicate matches the existing commit structure: resident rows plus incoming state/new anchors plus RUNNING old-state payload. It is checked separately against the unchanged 4 GiB guardian limit. Serialized IPC buffers are separately reserved against host RAM rather than counted as additional parameter tensors. The launch reservation includes observed process RSS, payload growth and a Python/data allowance. Measured per-GPU memory and power must each be below 85%; projected host usage must remain below 90%.

The specified fifth KR worker case evaluates to 4,654,122,048 physical bytes under the external 5 GiB cap and 4,039,824,232 bytes under the guardian's 4 GiB predicate. Passing the physical cap alone does not admit a request that fails the guardian predicate. Prefetch and finalization must each pass their full concurrent forecast; the fifth-worker example is not a blanket concurrency approval.

## Independent verification and release

Executed the existing scalar-only check with the project interpreter:

```text
PYTHONPATH=. /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_plans/26-11-10_ema_causal_attribution/check_scheduler.py
PASS: independent GPU eligibility, no worker ceiling, payload/RSS denial, adoption, serial attribution, all forecast terms
```

This includes the fifth-worker dual-cap fit and rejection when physical 5 GiB passes but guardian 4 GiB fails. The assertions support the inspected logic; they are not a live handover demonstration.

This PASS releases the exact supervisor above for the specified boundary takeover after its matching SCHEDULER_REVIEW.json is installed by the orchestrator. Record the first actual scheduling decision and full reservation after takeover, including a concrete denial reason if no launch fits. Existing controller PID1014311 and guardian PID1014402 were not modified by this review.

No code changes, model runs, broad new tests, live handover or scientific analysis were performed. The live scheduling decision remains the required execution check.
