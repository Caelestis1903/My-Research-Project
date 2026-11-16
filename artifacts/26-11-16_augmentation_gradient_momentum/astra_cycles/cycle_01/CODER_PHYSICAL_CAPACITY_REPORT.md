# Physical-capacity coder report

Implemented only the actual-CUDA-OOM correction in the existing overlay and focused fixture. Awaiting a fresh distinct review; no experiment was run.

Current per-card memory/power <85% and current host RAM <90% remain unchanged. The additional physical check is:

```text
current NVML occupancy bytes + incoming torch peak bytes + 352321536
    <= measured CUDA-usable capacity for that card (25307578368 bytes)
```

The 336MiB correction is the measured maximum of ML(full4306-smoke3970)=336MiB and KR(full5284-smoke4952)=332MiB. It accounts for the smaller original smoke peaks and full-process residency without inventing a cushion. With smoke inputs the full-process needs are ML4306MiB/KR5288MiB. With completed torch peaks they are ML4326MiB/KR5304MiB: the universal correction stays conservative when frozen active-worker measurements raise the incoming peak. This uses 100% of measured CUDA-usable capacity24135.1875MiB, rather than NVML24576MiB or a projected85% limit. Predicted power195.61/206.31W and predicted host bytes remain unused.

At the predecessor sample19118MiB,280W/350W,RAM33%, usable free space is5017.1875MiB: KR fails physical fit and ML fits. Independent-card selection, pending allocation ACKs, fresh sampling and no process-count cap are retained by the unchanged frozen scheduler. No batch, precision, model, optimizer, training or recovery change.

The new policy name is CURRENT_UTILIZATION_AND_PHYSICAL_CAPACITY. Current output metadata uses worker_resource_estimates because its memory bytes now feed physical fit. Historical outputs are preserved.

- New overlay SHA256: `b7eeba76d21727b596897eec84e0e5766c01e8a41fba832c1036d38bdf7c6f3f`.
- New focused fixture SHA256: `baa0d97e7332c70c0038ac5abab88b55199756981556845420105ad33fd9b590`.
- Correction SHA256: `3a051f809e2301f906c477e97261ce546f34c92d5f1b47afe3d56712866121ca`.
- Measurements SHA256: `aa78e32ed82fccc38632793b9b5680c303fa8f6468daa95cd7d62109e2572ce5`.
- Frozen79-file aggregate before/after: `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`.
- Design bundle remains `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`.

Before editing, archived the old overlay and fixture byte-for-byte under BEFORE_PHYSICAL_CAPACITY_ARCHIVE/orchestration/. Verified old hashes remain `b5c71d81bf48f3bc5075ca40624a336a7be554789e7cbe1efe2cc30ace2a0b9e` and `49dc43c4b23c9a16db73b4408b332f529a1fde83d3826115d71d85391c1bcec0`. The existing execution-policy JSONL remains one unchanged record; its file and first-line SHA256 both remain `50bf6a6efdaa4b40673d9d5ef8116acffe69f2fc0f0c1df1d3bf569105cb0200`. The old overlay hash remains valid history. Owner1570727/session59365's receipt and the scientific review remain unchanged; no live-owner IPC was attempted.

Focused CPU verification: **29 passed in1.09s**, exit0.

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m pytest -q -p no:cacheprovider orchestration/test_task_axis_even_views_current_utilization_20261116.py
```

Checks cover the original current-utilization sample with huge predicted power ignored; actual-OOM predecessor KR rejection/ML admission; exact physical boundary and one-byte excess; updated real peaks; current85%/90% boundaries; independent cards/pending ACKs; runtime measurement and review hash binding; append-only mocked receipts, accurate output labels and restored patches. All hardware/IPC/worker effects were mocked; no broad suite was run.

The fresh review must write IMPLEMENTATION_REVIEW_PHYSICAL_CAPACITY.json with verdict PASS or PASS WITH FIXES and no blocking issues; role NEW_DISTINCT_PHYSICAL_CAPACITY_REVIEWER; independent_of_coder=true; an actual fresh reviewer_agent_id; cpu_checks_passed=true; and execution_policy exactly equal to execution_policy_binding in the companion JSON report. That binding includes overlay, unchanged old policy, correction and measurements paths/hashes, loaded runtime capacity values, frozen worker source, design and both admission predicates. The runtime rejects a measurements file inconsistent with its loaded profile. Original parent verifies the distinct native fork_context=false lifecycle and permanently closes children.

The existing scientific IMPLEMENTATION_REVIEW_RESOURCE.json remains a separate required --worker-review, with SHA256 `a6b5e138827fdf55cc71c1a59873f2ae31e7aa3d8e665a918d9e0f96f9f6b3e5`. No generic adoption, source migration or new smoke is needed. At a future authorized RUN, the existing append path records a new version with correction/measurements/overlay hashes and both review identities; it preserves the first historical JSONL line.

Parent handoff only: reported inventory16 COMPLETE +7 ACK-PAUSED +1 FAILED(epoch0/noACK/stateNone) +6 pending. After distinct PASS, recover failed key3ab9dab5e7347854f7c4956e31a5e1073e908ee35d930559e4f11acc7112cb9f first, then validate/recover seven ACK-paused keys through the existing frozen protocol. Its epoch0/noACK/stateNone case remains supported and unchanged. Preserve costs and owner; fresh native runner continues14 unfinished trajectories without replaying16 COMPLETE trajectories. Check both workspaces' stop controls. This coder did not recover, clear pause, dispatch or release storage.

From /home/xzf/Code/phase0-3_tvg_rm_20261116, after those prerequisites, the fresh runner's command is:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m orchestration.task_axis_even_views_current_utilization_20261116 run --devices 0,1 --review artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/IMPLEMENTATION_REVIEW_PHYSICAL_CAPACITY.json --worker-review artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/IMPLEMENTATION_REVIEW_RESOURCE.json
```

No GPU/IPC/real run, Test, disk models, shared-state changes, source-control commit or publish. Return this completed coder to the original parent for permanent closure; never resume/reuse.
