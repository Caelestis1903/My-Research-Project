# REVIEW REPORT

1. **Verdict: PASS.** No CRITICAL, MAJOR or MINOR findings; `blocking_issues: []`.

2. **Requirement-by-requirement verification**

- **Strict current-resource gates — PASS.** choose_device retains current per-card NVML memory/total <0.85 AND current measured power/limit <0.85 and current host RAM <0.90. Incoming peak is used only in the separate physical-capacity predicate; predicted power and host-byte budgets do not gate admission.
- **Measured physical fit at usable capacity — PASS.** Actual code converts NVML MiB to bytes and requires current bytes + incoming peak_bytes +352321536 <= measured usable25307578368 bytes for the selected card. Equality admits. No projected85% threshold, worker-count cap or invented padding.
- **Measured correction and actual OOM predecessor — PASS.** Independently recomputed full-minus-smoke ML336MiB/KR332MiB; universal maximum336MiB. At19118MiB/280W, usable free5017.1875MiB; smoke-based KR5288MiB rejects and ML4306MiB admits. Focused fixture exercises these actual inputs.
- **Updated real peaks feed physical fit — PASS.** Frozen schedule raises each dataset reservation bytes from active RAM records peak_memory_bytes before admission, then passes reserve[bytes] into the patched choose_device. The overlay uses that incoming peak directly. Fixture verifies4968MiB rejects where4952MiB fits.
- **Independent cards, pending ACK, resampling — PASS.** Actual frozen scheduler derives pending_cards from active records lacking allocated ACK, adds each started card immediately, resamples rows after every launch and again each loop. No process-count cap or requirement that both cards fit. Scheduler code is in unchanged79-file aggregate.
- **Exact independent review and runtime bindings — PASS.** policy_binding independently executed against actual files and loaded profile; exact equality with coder JSON confirmed. Binds overlay, historical policy, correction, measurements, loaded correction/capacities, frozen worker and design. Review checker requires new reviewer role, independent identity, CPU checks and exact binding; runtime rejects changed loaded measurements.
- **Append-only history and accurate output policy — PASS.** Existing policy JSONL remains one byte-identical record with first-line/file SHA50bf6a6efdaa4b40673d9d5ef8116acffe69f2fc0f0c1df1d3bf569105cb0200. Old b5 overlay archive matches historical record. Actual append path binds both reviews; CURRENT_UTILIZATION_AND_PHYSICAL_CAPACITY and worker_resource_estimates describe the new semantics. Mock fixture verifies append and patch restoration.
- **Scientific formulas, tensor shapes, masks, routing, weighting and gradient paths — PASS_BY_VALID_EVIDENCE_REUSE.** Independently recomputed exact frozen79-file aggregate6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1 and seven-file design bundlef682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754. Overlay only patches scheduler choose_device and progress/report writing; scientific workers launch unchanged frozen.MODULE. Existing separate scientific review and numerical evidence remain applicable; no new numerical suite.
- **Baselines, owner, recovery and trajectory preservation — PASS.** Frozen source covers original worker/recovery/baseline paths; exact owner receipt hash verifies PID1570727/session59365 unchanged. Worker-review SHAa6b5e138827fdf55cc71c1a59873f2ae31e7aa3d8e665a918d9e0f96f9f6b3e5 unchanged. No scientific source, owner, live state, Test, dispatch, recovery or policy history changed by reviewer.
- **Focused CPU verification — PASS.** Reviewer independently ran existing focused fixture once using required Python -B;29 passed in1.26s, exit0. Hardware/IPC/process launch effects are mocked. No new checks or scientific tests added.

3. **Bugs/deviations found:** None in this correction.

4. **Corrections made:** None. Reviewer writes only these two review reports.

5. **Remaining risks:** Measured residency supports the specified fit correction; it does not guarantee that future workloads cannot exceed observed peaks. Existing runtime monitoring/ACK/pause behavior remains required. No live owner IPC, recovery or GPU validation was performed. Reported16 COMPLETE +7 ACK-PAUSED +1 FAILED(epoch0/noACK/stateNone) +6 pending is parent handoff context, not an independently queried inventory.

6. **Whether experiments are safe to run:** No physical-capacity implementation blocker. Safe for the original parent to proceed through existing same-owner recovery and stop-control gates, then a fresh current-conversation runner for14 unfinished trajectories. This reviewer performs no recovery/dispatch. Preserve16 COMPLETE trajectories and all attempt costs; Test remains sealed.

Reviewer `01a1e335-ba9d-70b2-a3c6-4e7ca4724a01` is distinct from coder `01a1e32f-cd7d-77f1-b74c-993b15012384`; native identity and closure are parent-verified. The companion JSON contains the exact independently computed `policy_binding()`; equality with the coder report was verified against actual files, rather than accepted as assurance.

CPU evidence: **29 passed in1.26s**, one reviewer invocation. No GPU/IPC/recovery/run or scientific numerical suite was performed. Parent retains the existing recovery and live-run gates.
