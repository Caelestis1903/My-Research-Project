# CODER SCHEDULER REPORT

Date: 2026-11-10. Status: 7 GiB external-cap revision ready for independent review. This coder has not deployed this revision or mutated any running process.

## Files and identities

- External implementation: `experiments/ema_causal_scheduler_20261110.py`.
- Single scalar check: `research_plans/26-11-10_ema_causal_attribution/check_scheduler.py`.
- Read-only execution receipt: `research_plans/26-11-10_ema_causal_attribution/SCHEDULER_DRY_RECEIPT.json`.
- Current supervisor SHA-256: `ec28e47176bae0cb534f9555e75e357b14837a0bf5a588dd78a6fc731156a07c`.
- Prior reviewed 5 GiB supervisor: `f22601d6af5014095db2e001bcc35af8589580744d564e986bb917dfaaad9dc0`.
- Current amendment SHA-256: `1c50a3227276580bcff2b8b7a88ce3349038b44a15843526f18bbaaee4b9978a`.
- Unchanged scientific implementation: `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`.
- Unchanged PREPARED SHA-256: `741df1a95597f516add9ea505006dd1de72947bb24b87d859b8223d1f692a82e`.
- Amendment at dry run: `150ea1ac24c1fbd1c7b756d2d4d204bc318665e6bc45bb2b1efa686e37fe226d`.

No scientific-package file, model, data, preparation, smoke receipt, or training result was modified. The implementation remains external to `implementation_identity()`.

## Implementation

The supervisor verifies preparation/current science identity and both existing smoke receipts, requires an independently written `SCHEDULER_REVIEW.json` with `verdict: PASS` and the exact supervisor SHA-256, and holds an exclusive attempt-level flock before handover or dispatch. Dry mode only reads state and prints JSON.

Takeover waits for matching ML-C block receipt and all five guardian rows FINALIZED, then verifies the original controller PID/start/boot/argv before sending SIGTERM to that process only. It waits for identity disappearance and scans exact attempt processes, including workers not yet registered. Existing workers are adopted; their guardian launch counts remain authoritative. An extra launch receipt counts pre-registration failures without permitting more than two total launches. The guardian is never restarted. A separate attribution lock prevents simultaneous finalizers; process scans permit adoption after supervisor restart. Failed attribution preserves anchors and stops with its log path. A completed attribution receipt can finish only the pending guardian finalize operation.

Candidates follow frozen block order with one-block lookahead. There is no fixed worker-count ceiling. Only one attribution child is launched; it invokes original `finalize_block` and then guardian finalize. Training completion requires all 20 finalized epoch50 jobs and matching four block receipts/step totals. Downstream analysis/freeze/evaluate/report/release remain original commands.

## Resource accounting and limitation

The schedule is enumerated by epoch and stage, distinguishing mid-epoch history growth from commit-time history after pruning. Each active/proposed job reserves its maximum remaining stage, rather than adding the maximum historical window count to its final anchor count. Existing guaranteed parameter tensors in a running guardian row are subtracted before replacing them with the future stage bound; arrays and buffers are not subtracted. Completed guardian rows remain counted through actual scalar `payload_bytes`.

Commit stage counts worker tensors, exported CPU clone, old guardian commit, and newly deserialized tensors. Pending anchor copies and retained earlier anchors are included. All three saved panels contain 294,912 bytes per FP32 prediction set; a packet has up to seven equivalent prediction sets and is conservatively allotted 4 MiB beyond its parameter vectors. Selected/source/matched packets and transient worker vectors are included. The original `peak_active_bytes` is retained as a lower floor for the training-stage reservation, and its value appears in the receipt.

Attribution reserves two complete-row bounds because the `_rows` generator may briefly retain the old row while fetching the next, plus 100P+32 MiB scratch. It is reserved only when active or being proposed; otherwise finalization must separately pass admission before it can overlap workers. Measured per-card memory/power thresholds use the existing helper. Serialized send/receive byte buffers belong to host RSS under 06§5.2, not the 4 GiB tensor budget. Their explicit reservation is added to observed-process RSS, model-growth and Python/data allowances under host90%.

The initial designer-approved budget correction raised only the external physical tensor allowance from 4 GiB to 5 GiB (5,368,709,120 bytes). The original guardian is unchanged and must independently pass its 4 GiB predicate. For each projected commit the formula is guardian resident payload plus incoming state/new anchor tensors plus every RUNNING old state. The supervisor enumerates each committing job and its possible epoch against the other workers' retained-state bounds. It reports the external total, guardian total and host reservation separately; all must pass.

The concrete fifth KR-clean seed case now passes: four complete row bounds plus the remaining worker require **4,654,122,048 physical bytes** (714,587,072 bytes below 5 GiB). The independently evaluated **guardian predicate is 4,039,824,232 bytes** (255,143,064 bytes below unchanged 4 GiB). The prior external cap was short by 359,154,752 bytes. This resolves the previously documented fifth-seed bound gap without excluding a tensor copy, editing the scientific package, or counting serialized buffers as parameter tensors. This fifth-seed fit does not imply automatic admission of another prefetched worker.

## Initial 5 GiB verification

Executed only with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- Syntax/import and CLI help succeeded.
- `PYTHONPATH=. .../python research_plans/26-11-10_ema_causal_attribution/check_scheduler.py` passed. Checks cover GPU0 eligibility with GPU1 absent/ineligible, admission after two live workers, payload and host denial, adopted-job exclusion, second-finalizer rejection, all forecast terms, the concrete fifth-seed fit under both tensor predicates, and rejection of a case passing physical 5 GiB but failing guardian 4 GiB.
- Read-only dry command against attempt2 succeeded with controller PID1014311/start10948746 and guardian PID1014402/start10949401. ML-C boundary was still false; no workers were dispatched or signaled. Eligible device was GPU0. Hypothetical attribution physical forecast was 2,107,337,520 bytes, guardian predicate 1,407,769,320 bytes, and extra host RSS reservation 3,946,760,152 bytes. Both scientific hash and PREPARED checksum match the first dry receipt.

The dry output's hypothetical `action: finalize` describes a resource candidate only; `boundary_ready: false` keeps execution non-dispatching. No model tests, formal experiment, guardian mutation, or real scheduling decision were executed. Independent review and the first post-approval real scheduling decision remain outstanding.

## 7 GiB correction: implementation and focused verification

The updated design authorizes an external physical-tensor cap of **7,516,192,768 bytes (7 GiB)**, replacing **5,368,709,120 bytes (5 GiB)**. Exactly two source lines changed: `CAP=7<<30` and the associated rejection reason. Reversing those two replacements reproduces the entire prior source SHA-256 `f22601d6af5014095db2e001bcc35af8589580744d564e986bb917dfaaad9dc0`. The original guardian predicate remains **4,294,967,296 bytes**, projected host limit remains90%, per-device memory/power limits remain85%, and no execution code path changed.

The supplied first live three-KR-worker candidate required6,965,659,752 physical bytes, with guardian predicate4,238,324,112 bytes. Its host usage37.7GB plus reservation12.36GB fit below90% of67,181,834,240 bytes and GPU1 was eligible. The new external limit leaves550,533,016 bytes of headroom for that candidate; the old external limit alone had denied it.

The existing scalar assertion check was extended only to test the exact7GiB boundary, reject boundary+1, and admit that supplied three-worker snapshot when its other checks pass. All existing assertions, including independent guardian rejection and host rejection, still pass. Command: `PYTHONPATH=. /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_plans/26-11-10_ema_causal_attribution/check_scheduler.py`; result: PASS.

A later read-only live decision had physical forecast7,448,875,768 bytes but guardian predicate4,721,540,128 bytes. The amended supervisor will correctly deny that later snapshot on the unchanged guardian constraint. The cap correction does not guarantee immediate admission when another constraint fails.

PREPARED SHA-256 remains741df1a95597f516add9ea505006dd1de72947bb24b87d859b8223d1f692a82e. No scientific-package files, scientific identity, guardian, workers, attribution child, or running supervisor were changed by this revision. The earlier dry receipt retains its original5GiB source identity. No new model tests, process signals, restart, or dispatch were performed. Independent review must precede the runner's restart and worker adoption.
