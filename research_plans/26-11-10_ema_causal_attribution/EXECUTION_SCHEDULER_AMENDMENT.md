# EXECUTION SCHEDULER AMENDMENT — resource-admitted overlap

Date: 2026-11-10. Status: DESIGN COMPLETE; independent implementation review required before takeover.

## Responsibility and unchanged scientific identity

The current controller executes `finalize_block` synchronously. During attribution it does not admit any new training, even when an individual GPU meets the user's memory/power rule. Replace only this execution ordering with a small external supervisor so pending training can use admissible resources while one condition block is attributed.

Own one new file, `experiments/ema_causal_scheduler_20261110.py`, plus its execution receipt and this amendment. Do not modify `experiments/ema_causal_20261110/`, PREPARED, source functions, or scientific hashes. Record the full unchanged PREPARED implementation hash, supervisor SHA-256, amendment hash and takeover receipt separately. Reuse existing `_worker`, `finalize_block`, guardian requests, process identity/alive helpers, resource admission and protocol constants. No new training or model tests.

The same 20 jobs, seeds2022–2026, 50 epochs, source training order within each job, independent RNGs, output definitions, block finalization order ML-C→KR-C→ML-G→KR-G, 2GiB observations, restart/replay limits and Test freeze remain in force. The external physical-tensor forecast cap is corrected to 7GiB as justified below; the existing guardian's internal 4GiB commit predicate remains unchanged and separately enforced. Scheduling changes execution time and contention, not scientific comparisons. Completion still requires all four attributed blocks, followed by the original analysis/freeze/evaluate/report/release flow.

## Boundary handover without losing RAM

Current controller PID1014311 and guardian PID1014402 are contextual observations, not permanent identities. Bind PID plus process creation identity and exact attempt command before any signal.

1. Prepare/review the new supervisor in dry mode while current ML-C attribution continues. It must not dispatch anything in this phase.
2. Wait for both the ML-C `blocks/<block>/BLOCK_COMPLETE.json` receipt and guardian `status.blocks` containing ML-C; all five ML-C rows must be FINALIZED. A table file alone is insufficient. Do not signal the old controller while it is doing attribution or while guardian finalization is pending.
3. At that boundary, stop dispatch by terminating only the verified old controller process, never its process group, guardian, training worker or other user processes. Confirm that controller identity is no longer alive before enabling the new scheduling loop. If it has already exited, do not signal a reused PID. The old controller may have launched next-block workers in the interval; preserve them.
4. Reconcile existing guardian worker identities and exact attempt `_worker` processes before dispatch. Include a launched-but-not-yet-registered child in the pending set until it registers or exits. Adopt registered live workers without re-registering or counting them as new launches. Do not rely on `Popen` child ownership or `waitpid` for adopted processes; guardian epoch/status and process identity remain authoritative.
5. Use one exclusive supervisor ownership lock for this attempt and keep it for its lifetime. The takeover receipt records old/new identities, guardian identity, boundary receipt/hash, adopted jobs and their committed epochs, resource state, and unchanged scientific hash. No second scheduler can dispatch concurrently.

If boundary state changes or cannot be reconciled, remain in non-dispatch mode and report the concrete dependency; never kill a worker to simplify takeover. Do not restart or replace the live guardian for this handover.

## Small event loop and attribution child

Use the frozen job order and original worker argv/env/log location. The event loop tracks live registered workers and pending launches, checks completion/failures, and admits another pending job whenever resources permit. Remove the old fixed two-worker concurrency ceiling; actual GPU, host and payload admission determine concurrency. Refresh measurements after each launch and allow the existing short startup interval before the next admission.

Prefer unfinished jobs of the earliest unfinalized block. Once they are already active/complete, allow the immediately following condition to prefetch; do not create a second future condition buffer. This one-block lookahead is enough to overlap training with attribution and bounds retained anchors. While a block is being attributed, its five rows remain COMPLETE until the original finalization operation releases their anchors.

Launch at most one attribution child globally, in frozen condition order, after the block has five COMPLETE epoch50 jobs and resource admission succeeds. The child calls the unchanged `finalize_block(out, jobs, device)` and, only after its successful complete receipt, the unchanged guardian `finalize` request. Its own argv/process identity and block are recorded so a restarted supervisor can adopt it. No two processes write a block's attribution tables simultaneously.

If attribution tables completed but the child died before guardian finalization, verify its complete receipt and the five COMPLETE rows, then finish only the guardian finalization. If attribution itself failed, preserve all anchors, record the actual error and use the existing recovery policy; do not silently release state or pretend completion. Training may continue only within reserved resource capacity. Worker restart remains at most one after the first launch; inherited guardian launch counters are authoritative.

## Admission must cover future retained state, not only current idle GPU

Each GPU is independent: measured memory.used/memory.total < .85 and power.draw/power.limit < .85; host usage from MemAvailable < .90. Reuse the existing measurement helper. A busy other card does not veto an eligible card. Both attribution and training consume capacity and are included in accounting.

Before a new training launch, forecast the maximum concurrent parameter payload while that worker progresses to completion without the current attribution releasing anchors. Include:

- guardian's current actual payload;
- still-to-arrive retained anchors/selected/source/matched packets for every active or proposed job, conservatively bounded from the existing object layout and observation schedule;
- active worker state, parameter history and recovery copies using existing `peak_active_bytes` and the actual pending-packet layout; IPC copies here mean actual deserialized parameter tensors, not serialized byte buffers;
- old/new epoch-commit coexistence, including commits from already admitted workers;
- attribution child's one copied job row plus its scratch/construction temporary parameter vectors. The current `_anchors` fetches one complete row from guardian; do not omit that copy merely because guardian owns the originals.

Do not fetch every full guardian row just to poll resource use; use scalar status and a conservative precomputed per-job bound. State all terms and bytes in the admission receipt. A conservative bound may double-count a current component, but must not omit future retained growth or concurrent copies. In particular, `guardian.payload_bytes + next active snapshot` alone is not sufficient for cross-condition prefetch.

Require the external physical parameter-tensor forecast ≤7GiB (7,516,192,768 bytes). As specified by 06§5.2, this cap counts actual parameter-class tensor storage via `tensor_bytes`: live parameters/history/optimizer tensors, real tensor clones and deserialized tensors, and old/new commit tensors simultaneously resident. Count an actual shared tensor storage once; do not treat a serialized representation as another parameter tensor. IPC serialization buffers (`BytesIO`, `send_bytes`/`recv_bytes` bytes), Python objects, data mappings, predictions, pair caches and CPU sorting arrays instead enter the separate host-RAM forecast governed by 90%. Serialized buffers still require host reserve while coexisting with their source/destination tensors, but do not consume the physical tensor allowance a second time.

### Measured budget correction and separate guardian predicate

Coder's concrete KR Clean fifth-worker E39 forecast includes four actual tensor copies (worker, export, guardian old state, newly deserialized state), 19 history events and four anchors: 2,306,417,184 bytes additional. Four completed KR rows contribute 2,347,704,864 bytes, giving 4,654,122,048 bytes total. This exceeds the old external 4GiB estimate by 359,154,752 bytes; serialized buffers are excluded from these numbers. The first correction to 5GiB accommodated this measured fifth-seed case with 714,587,072 bytes of physical-tensor headroom; it is superseded by the measured concurrency correction below. Record these terms, old/new caps and the measured host reserve in the execution receipt.

06§5.1 identifies 4GiB as a proactively inherited conservative task budget, not a user hardware limit. Correcting this underestimated execution budget preserves the user requirements of host RAM below90% and per-device memory/power below85%. It changes only the external supervisor's execution policy; no scientific package edit, PREPARED/hash change, new attempt or additional experiment is required.

Independently forecast and require the unchanged guardian commit predicate at every projected commit: `size + tensor_bytes(old_row_state) + worker_payload <= 4GiB`, using the source `Guardian.handle('commit')` definitions of `size` and `worker_payload`. This internal predicate counts a different set of resident tensors than the external four-copy physical forecast; passing7GiB does not imply passing the guardian. Do not modify or bypass the guardian cap. Admission requires all three checks: physical tensors≤7GiB, guardian predicate≤4GiB, and projected host usage<90%, as well as the GPU rule. Log each total and denial reason separately.

Current/next-condition prefetch uses the remaining physical headroom only when its full retained-growth/worker/commit/finalizer forecast also fits both tensor predicates and host reserve. The current 7GiB cap does not automatically admit an arbitrary additional worker; every candidate still receives the complete forecast. If prefetch does not fit, admit it after attribution releases anchors; do not exceed either predicate to keep a GPU occupied.

Reserve expected extra host RSS from current smoke/calibration and observed worker/finalizer RSS before dispatch so host90% is not crossed by a launch that only passed a pre-launch check. Report physical-tensor, guardian-commit and host-memory forecasts as three separately named totals with their own terms. The currently reported roughly27GiB available host RAM is an observation to refresh at admission, not a permanent reservation. No checkpoint or temporary model persistence is allowed.

If current GPU readings pass but payload/RSS reservation prevents launch, log the exact limiting term and bytes. Keep polling and admit after resources are released; do not present an unconditional two-worker cap or synchronous attribution wait as a resource restriction. Do not terminate unrelated processes, shrink batch, alter evaluation or skip seeds to create room.

### First live decision: external cap corrected from 5GiB to 7GiB

After ML-C was finalized and the supervisor adopted KR seeds2022/2023, its first proposed third worker (KR2024) had physical-tensor forecast6,965,659,752 bytes. The independent original guardian predicate was4,238,324,112 bytes, below4GiB=4,294,967,296 bytes. Recorded host usage37.7GB plus reserved12.36GB was below90% of67.18GB total; GPU1 met the per-device85% rule. Only the self-imposed external5GiB cap rejected this launch.

Approve the external cap7GiB=7,516,192,768 bytes, giving550,533,016 bytes of headroom for that measured candidate. This also removes the known5GiB obstruction to the intended approximate KR-completed-block3GB + finalizer1.4GB + next-ML-worker1.2GB overlap; actual admission must still use exact forecasts and current host/GPU readings. Record the first-decision evidence and5GiB→7GiB correction in the supervisor receipt. Original guardian4GiB, projected host90%, GPU85% and all other policies remain unchanged.

Implementation scope is one external budget constant and corresponding execution reporting. Reuse the same focused admission assertion with the observed third-worker boundary numbers; no additional model test or scientific matrix change. Following independent review, restart only the verified supervisor, wait for its ownership lock to release, then adopt the two live workers and guardian. At this correction no attribution child is active; confirm that fact before the restart and preserve all worker/guardian processes and RAM. PREPARED, scientific package/hash and attempt remain unchanged.

## Minimal validation and review

1. Syntax/import check the external script; verify its help/dry mode makes no dispatch, signal or guardian mutation.
2. One compact assertion-based admission/state check using supplied scalar snapshots: GPU0 eligible while GPU1 ineligible admits on GPU0; two already live workers do not impose an artificial ceiling; an exceeded payload forecast denies; a live/adopted job is not launched again; a second finalizer is denied. No real model run is needed.
3. Dry mode against the current attempt records boundary-wait status, adopted/pending identities, candidate job, eligible device and complete memory forecast without mutation. Verify the package implementation hash and PREPARED remain unchanged.
4. Independent reviewer inspects handover exclusivity, adoption of unregistered launches, existing restart counters, serial attribution, lookahead bound, both tensor-cap predicates, host reserve and Test/finish flow. Include the measured fifth-seed fit and a case that passes the physical7GiB check but fails the guardian4GiB predicate in the same compact admission assertion. PASS or corrected PASS WITH FIXES is required before takeover.

After takeover, confirm one real scheduling decision from logs: either a worker dispatch with its resource reservation or a specific blocking resource/dependency. This is an execution check, not a scientific experiment. Write final TRAINING_COMPLETE only when guardian has all four finalized blocks and the original 20 job/epoch/step totals are satisfied; preserve the original downstream scientific commands.
