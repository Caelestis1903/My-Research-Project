# DESIGN SPECIFICATION — Cycle04 operational restart, attempt2

Status: COMPLETE; ready for parent acceptance and implementation. Original orchestrator: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. Date: 2026-11-17. This addendum changes operational execution only. The seven-file accepted scientific design remains authoritative and immutable.

Paths: `W=/home/xzf/Code/phase0-3_tvg_rm_20261116`; `O=/home/xzf/Code/phase0-3`; `A4=W/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04`; `oldN=A4/attempt1`; `N=A4/attempt2`; `E=W/experiments/completion_weight_attribution_20261117`; `R=W/research_reports/completion_weight_attribution_20261117`.

## 1. Research question

Complete the already registered Cycle04 completion-weight attribution comparison after irreversible RAM loss, with unchanged scientific interpretation. Can the confirmed pending-allocation race be repaired while retaining the existing independent-card admission policy and full experimental budget?

Inspection: W/O AGENTS, W start.md decoded GB18030, buchong.md, current runbook correction, accepted DESIGN_SPECIFICATION, actual protocol/runner/server interfaces, pause reports, source receipt and parent RAM_STATE_LOSS.json. Both workspaces were ACTIVE/manual_stop_requested=false with no stop sentinel at inspection; parent rechecks before dispatch.

## 2. Hypothesis being tested

Operational hypothesis: binding pending startup to the newly launched PID and expected owner registration attempt prevents an older `allocated=true` from authorizing another allocation. It leaves the frozen W/S objectives and training protocol unchanged. Scientific hypotheses, competing explanations and decision thresholds remain those in A4/DESIGN_SPECIFICATION.md; no efficacy assumption is added.

Confirmed incident: old recovery released 19/20 pending latches before the new worker STARTED, admitted 17 workers on GPU0 and 3 on GPU1, and added zero ACK epochs; 15 workers FAILED and 5 PAUSED. The documented failure was physical-capacity guard rejection, without logged CUDA OOM evidence. NVML used 25,525,813,248 bytes exceeded CUDA-usable 25,307,578,368 bytes. Preserve these historical facts and receipts.

## 3. Variables changed

Canonical writable attempt becomes attempt2; new source48 identity, twenty new source-bound job keys, fresh reviews, engineering receipts and foreground storage owner. Scheduler pending records gain launched PID and expected registration attempt. Update only their necessary audit records, narrow CPU regression cases and package run instructions.

RAM_STATE_LOSS.json records four missing owners: C4 2429109, C3 2097722, C2 1944719, C1 1570727. TV owner 1025015 remains alive and protected: no IPC, signals or changes. The twenty old C4 full states are lost; their 275 ACK epochs/7,733 useful updates are historical numeric evidence, never restart credit. Old designer Boyle is closed/not_found and must not be resumed.

## 4. Variables held constant

- Accepted seven-file design bundle `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`; six-file research bundle `a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a`. Do not alter bundle membership, algorithm definitions or scientific acceptance gates.
- Source closure is the same exact 48 paths enumerated by oldN/source_receipt.json and protocol.source_paths(). Old source identity `00ce7925c3589deb239f57e8374673e5afe8e3fe155e159dd34c140ae08cd553`. Parent preserved exact old bytes under oldN/source_snapshot with original-path mapping in RAM_STATE_LOSS.json. All48 live files matched that receipt during DESIGN inspection.
- E/core.py SHA256 `790ae97d01a3ab3d39e42712580e333aa017b96d5869f36aed682181b03de683`; E/checks.py `7bbf10c3e4249d9a3af7e1ad80b7c93267a66f99ddc52e3050c709ba99605ae1`. Both remain byte-identical. All shared helpers, storage server, source Adam, model, EMA, loader, RNG, selection and data protocol remain byte-identical.
- Same two datasets × W/S × seeds2022–2026 ×50 epochs =20 trajectories,1,000 useful epochs,34,500 live Adam and EMA updates. Batch40960; ML20 steps/tail21927; KR49 steps/tail40052; FP32, existing hyperparameters and diagnostics at first batch of epochs1/10/30/50. No HPO, reduced epochs/seeds, changed batches or additional research arms.
- Existing CLEAN/NQR numeric controls remain read-only: cycle_03/attempt1/reused_reference_index.json SHA256 `e8d67c6b18ad9ddbfa19b752e35c6ab051e3170a3d28df635b86b27922441fdb`,20 results+1,000 epochs=1,020 files. Reuse numeric controls without old-owner IPC or historical tensor availability claims.
- Raw ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`, SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`; KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`, SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`. Use existing trusted train/validation cache `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache/{dataset}.pkl` and existing manifests/fingerprints; both cache paths exist. Existing prepare performs the required data/control validation. Test stays sealed.

## 5. Mathematical/algorithmic definition

Incorporate the entire original scientific §5 unchanged: C is source BCE, W=(1−lambda)C+lambda A, S=mC, m=1−r, lambda=.25, N=W+R for diagnostics/reference; same reductions, identity branches, gradients, coupled Adam and fixed EMA. No implementation change to this math is authorized.

For a launch on device d, capture prior owner attempt count b (zero if key absent) before Popen. After Popen, store pending[d]={key, pid=Popen.pid, expected_attempt=b+1}. It must exist before the next scheduling/admission decision. Global experiment attempt2 and per-key owner registration attempt are distinct: the first registration in attempt2 is number1.

Release pending[d] only when the corresponding active child is still running and a fresh status record satisfies ALL:

`record.job == expected_job`, `record.worker_pid == pending.pid`, `record.attempts == pending.expected_attempt`, `record.status == RUNNING`, `record.allocated is True`.

Absent registration, old PID, old/future attempt, or current registration with allocated=false keeps that card pending. The frozen server resets allocated=false on every register; only its subsequent allocated command sets true. Consequently matching PID+attempt+allocated proves the new generation's allocation ACK, rather than relying on an old flag. Preserve the worker's ACK after actual first FULL Adam/diagnostic allocation and its post-ACK measurement. No server protocol extension or invented server ACK field is needed. Add worker PID to the existing numeric allocation ACK and release/resample audit records so chronology is directly reviewable. Clear on observed child exit only through existing exit handling, matching the pending launch; a failed child triggers PAUSE and must never look like a successful allocation release.

After qualified release take a new current NVML sample before another start. Independently for each card require current memory/NVML total<.85, current power/power limit<.85, host RAM<.90, no pending startup, and current used+incoming measured full-process envelope<=current CUDA-usable capacity. The physical fit ceiling is100% usable capacity; incoming peak is not added to the85% predicate. Keep existing first-fitting queue and measured envelope compatibility/updates, independent devices, resample-after-start behavior and cold-engineering policy. No worker-count cap, both-card fit requirement, forecast power, projected85% ceiling or foreign-process interference.

## 6. Exact affected modules/files

Coder write scope: E/protocol.py (ATTEMPT and attempt-specific diagnostics only), E/runner.py (pending generation logic, audit fields, finite regression), E/README.md (current attempt2 commands/stop-after-report correction). Keep source_paths membership and hash recipe unchanged. E/__init__.py, tables.py, core.py, checks.py and the other41 closure files remain unchanged. Do not use external monkeypatching, source-hash exceptions or changes to historical source consumers.

Coder reports: new A4/RESTART_CODER_IMPLEMENTATION_REPORT.md and RESTART_CPU_CHECKS.json. Reviewer: new A4/RESTART_IMPLEMENTATION_REVIEW.md/json. Runtime writes only N; engineering signoff/release/handoff use existing names within N. Parent alone owns RAM-loss/snapshot preservation, report/context path migration, stop-state handling and publication. No old receipt, jobs, ACK, result, binding, design or ledger is overwritten. Parent already migrated report tools; coder must leave R alone.

## 7. Implementation requirements

1. Before source edits verify parent's RAM-loss receipt and48-file snapshot mapping against old source receipt. Preserve oldN and old seven-design acceptance. New restart design acceptance is separate; parent can hash these two addenda without altering protocol.DESIGN_HASH.
2. Point canonical protocol.ATTEMPT/new_root/new_path diagnostics at attempt2. Existing job digest already includes source_hash; new actual48 digest must differ from old and yield20 distinct new keys. Do not add attempt to scientific config or change the existing job recipe. No copying old registered jobs/ACKs/selected models, reconstructing states from hashes or run --recover against oldN.
3. Implement §5 in run(), including launch, pending release and exit cleanup call sites. A small local predicate/helper is acceptable for the exact regression; do not build a scheduler framework.
4. Freeze final actual48 source using existing digest recipe and external installed Adam provenance. Every new review, manifest, job, owner binding, engineering receipt and formal record must consume that actual new source hash plus unchanged scientific design/control identities. Future source edits invalidate dependent gates.
5. Keep existing check_review separation contract. Its coder_ids currently requires at least two distinct IDs: list actual inherited core/runtime contributors plus the fresh restart coder, with their report provenance; do not invent another coder, rerun unchanged core work, or weaken this check. Fresh code/resource reviewers exclude all contributing coder IDs; resource reviewer also differs from code reviewer.
6. After code PASS parent runs prepare for N, then starts a fresh foreground storage session by calling frozen serve(N) directly. Never call its detached ensure_guardian/start helper. Record N/guardian.json and private STORAGE_OWNER.json per existing README contract, mode0600, live PID/tool session, endpoint hash, new source, unchanged design and actual server hash. New owner must differ from all historical owner identities including lost C4 PID2429109; do not broaden the frozen server. Parent verifies empty initial owner and keeps the foreground tool session live. Only N storage IPC is authorized; no disk model/optimizer/RNG states.
7. Perform exactly the existing four actual GPU engineering cases on the new source/owner: GPU0 ML/W, GPU0 KR/S, GPU1 ML/S, GPU1 KR/W, seed2022. Each fixed plan remains8 returned calls,2 diagnostics,6 extra sweeps,8 virtual Adam steps, storage ordinals1/2 at logical steps2/4 and zero scientific epochs. Retain exact source/lambda0, immutable prefix, diagnostic neutrality, real pause/register/get/BytesIO/deep restore/replay, selected validation and all measured phase/full-process profiles. Full/tail shapes and lazy/populated states remain mandatory. Old engineering PASS establishes history only and cannot release this source gate.
8. Fresh independent resource reviewer validates all four new cases/profiles and measured admission/physical capacity; parent performs actual get/hash acceptance, records existing ENGINEERING_RELEASE_ACCEPTANCE and releases exactly those four engineering keys. Formal RUN requires actual state_released flags, matching new source and zero blockers. Do not infer fit from old profiles or alter foreign occupancy (user reports roughly15/17GiB; runner measures afresh).
9. Formal RUN initializes all20 trajectories from source seed initialization, validates control initialization/order hashes, and completes50 whole ACK epochs per key. Existing commit/journal/status/get/hash, lost-reply reconciliation, exact deep restore with RNG last, immutable numeric prefixes, logical/physical accounting, selection/export and retained selected-RAM handoff remain unchanged. A later within-attempt2 recovery requires live matching RAM and existing fresh recovery review; missing acknowledged state means technical PAUSE, never fabricated continuation.

### Exact finite CPU checks

Coder executes compile(source_text,...,'exec') only for edited Python files; no py_compile/compileall. Reuse unchanged core/checks evidence. Add one finite in-memory regression using the actual production pending predicate/run path (mock launch/status/telemetry, no GPU/IPC). Assert this sequence: absent record retains latch; stale allocated=true with old PID/attempt retains; expected PID/wrong attempt retains; wrong PID/expected attempt retains; expected PID+attempt with allocated=false retains; exact running PID+attempt+allocated=true releases once and resamples before next same-card launch; pending on GPU0 permits fitting GPU1 admission; nonzero child exit takes PAUSE/exit cleanup without ACK release. Each is one deterministic case, no timing stress loop or new GPU case. Preserve existing admission assertions for strict85%/90% and physical100% boundaries.

Run the existing bounded runtime checks once after the edit:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -c 'from experiments.completion_weight_attribution_20261117 import protocol,runner,tables; print({m.__name__:m.finite_checks() for m in (protocol,runner,tables)})'
```

These checks already use finite memory transcripts/CPU fixtures; no broad suite, unchanged-core rerun, GPU query or live owner IPC by coder/designer/reviewer.

### Parent dispatch and execution handoff

Parent alone dispatches fresh native children with explicit fork_context=false: restart coder -> distinct code reviewer -> DEFAULT gpt-6-luna/high engineering runner -> distinct resource reviewer -> fresh DEFAULT gpt-6-luna/high formal runner. Non-RUN children use the parent's current model. Actual resolved model/reasoning/thread metadata must be verified from native receipts before GPU work; requested settings alone do not pass. Use runbook's five-seed DEFAULT role correction, no global role edits. Close each completed child permanently; no descendants, independent/background AI or resumed child. Technical failure PAUSES for parent-directed repair.

The following are commands for the parent/authorized runner AFTER applicable gates; designer does not execute them. Set W/A4/N explicitly in that supervised shell, with N ending attempt2:

```bash
W=/home/xzf/Code/phase0-3_tvg_rm_20261116
A4="$W/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04"
N="$A4/attempt2"
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.runner prepare --root "$W" --attempt "$N" --review "$A4/RESTART_IMPLEMENTATION_REVIEW.json"
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -c 'from experiments.task_view_gradient_residual_momentum_20261116.runner import serve; serve("/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2")'
```

Keep serve in its own current-conversation foreground tool session; parent binds/verifies it before the runner starts. Separate later supervised calls:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner smoke --root "$W" --attempt "$N" --review "$A4/RESTART_IMPLEMENTATION_REVIEW.json" --devices 0,1
# After distinct resource PASS, parent acceptance and actual four-key release:
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner run --root "$W" --attempt "$N" --review "$N/ENGINEERING_SIGNOFF.json" --devices 0,1
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.tables --root "$W" --attempt "$N"
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.runner handoff --root "$W" --attempt "$N"
```

Parent verifies selected live RAM before closing formal runner, then performs ANALYZE, complete report, distinct evidence review, context update and actual publication/delivery through the existing publication route. Finally USER_STOPPED. No Cycle05/Astra. Keep immediate stop latch false during this authorized cycle; check both workspaces' actual stop controls before dispatch and honor any intervening immediate stop. Test remains sealed under this restart specification.

## 8. Ablations required

No new scientific ablation. Preserve W/S with exact numeric C/N comparisons from original design. The finite stale-generation regression is the operational discriminating comparison; running a deliberately broken GPU scheduler is unnecessary.

## 9. Expected logging requirements

Preserve all original scientific/numeric schemas, AUC/loss per cell/task/domain, selected and fixed epochs, seed pairing, losses/masses and diagnostic counters. Pending startup/release logs identify key, new PID, expected and observed attempt/PID, time and fresh sample; worker allocation ACK carries its actual PID and attempt. Existing STARTED, allocation ACK and release chronology must establish matching generation before release.

Keep attempt1 costs/physical logs and48 snapshot immutable. Report lost attempt1 work separately (275 historical ACK epochs/7,733 ACK updates plus actual failed/replayed/engineering work); attempt2 useful budget starts at zero and ends at1,000/34,500. Aggregate physical costs across both roots using source/experiment-attempt/key/registration-attempt/PID and original receipt paths; engineering and unknown partial work stay separate, overlapping times are not summed, and no durations are inferred from hashes. Parent's migrated ANALYSIS.py/FIGURES.py/PACK_RAW.py own final report/archive aggregation. Public archives exclude guardian credentials/endpoints and all tensor states.

## 10. Acceptance criteria for reviewer

Code PASS or PASS WITH FIXES with zero remaining blockers requires exact authorized diff, unchanged45 source files outside the three-file write scope including cores, preserved48 snapshot and seven-design/control identities, actual new48 digest and20 new keys, attempt2-only writes, production-linked finite stale-generation regression and bounded CPU checks. Every contributing coder/reviewer identity must be genuine and separated. No old source/review hash may approve attempt2.

Resource PASS requires the four new actual source-bound GPU cases, exact continuation proofs, new foreground owner and measured full-process envelopes, PID/attempt-qualified pending behavior, independent-card current85%/host90% and usable100% fit, and parent four-key release. Formal completion requires20×50 ACKs,34,500 useful updates, existing exact exports and parent actual selected-RAM acceptance (40 selected online/EMA models;191,017,920 tensor bytes) before evidence review/publication/USER_STOPPED. Candidate scientific GO is not an engineering acceptance criterion.

## 11. Conditions that would falsify the hypothesis

An old PID/registration attempt clearing pending, same-card launch before new-generation allocation ACK/resample, cross-card blocking introduced by the fix, or a source/config change to training falsifies the operational claim and blocks RUN. A current-generation match still permitting the documented startup race warrants a targeted repair, not altered admission thresholds. Scientific hypotheses remain falsifiable under original §11; negative/contradictory W/S results must be reported with original metrics, seed evidence and competing explanations. RAM loss or resource failure is a technical PAUSE, not scientific rejection or successful completion.

DESIGN work performed: read-only inspection and these two restart specification files. No source implementation, CPU/GPU experiment, owner IPC or subagent dispatch was performed.
