# DESIGN SPECIFICATION ADDENDUM — third native crash, bounded thread diagnostic

Date: 2026-10-29. Protocol remains MDMT-SOTA-2526-v1. **No new package source revision is proposed.** Existing R1/R2 source and freezing checks remain in force. This design supersedes the R2 addendum's stop condition only to permit the finite diagnosis and one conditional replacement attempt below, after implementation and independent review. It authorizes no training from the designer.

## 1. Question and evidence

Can an explicitly single-threaded CPU-library execution profile avoid the recurrent native failure while preserving the exact model trajectory? The hypothesis is a runtime/thread interaction, not a demonstrated NumPy/PyTorch bug or an algorithmic change.

Inspected: prior runtime IO design; R2 data/evaluation/report implementation; runtime implementation/review/replay reports; `artifacts/mdmt_baselines_2526/run_logs/formal_hypergate_kuairand_seed2025.stderr.log`; incident ledger and run sources. Third failure occurred under R2 at `evaluation.py:55` in the compound expression `tensor(data.x[ids[...]],device)`, after19 complete epochs of HyperGate/KuaiRand/2025. Parent confirms23 complete formal entries, queue stopped, E never read; runner has archived the failed attempt and its recorded failed-epoch cost is286.686639784seconds. Parent also confirms actual launch OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4; OPENBLAS_NUM_THREADS was not explicitly controlled, torch intra/inter-op counts were not explicitly set. Do not infer their exact prior effective counts from the run's incomplete thread metadata.

R2 removed the feature memmap, yet this failure recurred: **resident X is not sufficient to eliminate the problem**. The new Python stack has no frame inside data.tensor and cannot locate the failing native instruction within ID slicing, advanced indexing, NumPy copying or tensor construction. The native fault may also surface after earlier memory corruption; the stack does not prove the fault originated on that line.

## 2. Static audit of the suspected paths

- `Data.x` is ordinary np.load ndarray and read-only. Roles are loaded without mmap, concatenated and sorted into ordinary arrays. `domain` is an int64 copy; no role/index memmap path was found.
- `tensor(value,device)` is `torch.as_tensor(np.array(value),dtype=torch.float32,device=device)`. NumPy array defaults to copy=True. The helper therefore does not hand the original read-only X directly to a CPU zero-copy tensor. On CPU as_tensor may share the **new owned copy**; on CUDA conversion copies to device memory. There is no application non_blocking=True, pinned buffer, custom CUDA stream or user-thread array mutation in the inspected path.
- No demonstrated buffer-lifetime error currently justifies replacing tensor(), adding clone()/extra copies/synchronizations to production, or loading y eagerly. A temporary array expression alone is not proof of dangling memory: PyTorch supports retaining NumPy-backed tensor storage. Diagnose actual values/lifetime, not an assumed bug.
- Installed local torch `_torch_docs.py` and version-matched [torch.as_tensor documentation](https://docs.pytorch.org/docs/2.9/generated/torch.as_tensor.html) describe CPU sharing/device copying. [NumPy2.1 array](https://numpy.org/doc/2.1/reference/generated/numpy.array.html) documents the default copy; [Tensor.to](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.to.html) documents non_blocking=False by default. These API contracts support the audit but cannot rule out an implementation/library fault.

## 3. Changed and constant variables

Only candidate mitigation: process-local **CPU library thread limits**, keeping exact current R2 package bytes. Set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1 before importing their libraries; then set torch intra-op and inter-op thread counts to1 in a fresh process before computation. Record actual effective counts and NumPy runtime/BLAS information. CUDA/runtime service threads may still exist: this is a CPU compute-pool profile, not a claim that the OS process has exactly one thread.

Unchanged: source revisions R1/R2, Python/library versions, X/y/roles/preprocessing, labels/masks and permissions, model math, seed, RNG initialization/stream coordinates, row-plan, batch/epoch/optimizer schedule, checkpoint selection/patience, device GPU1, formal30 matrix. Thread count can change floating-point reduction order in principle; exact replay is a required check, not an assumed consequence. No dependency upgrade, hash, new algorithm, hyperparameter adjustment or library backend substitution.

## 4. Finite diagnostic implementation and ownership

Pipeline coder owns **one small external diagnostic/launch script** under `research_reports/mdmt_baselines_2526/` (or another owned non-package report-tool directory), its evidence JSON/logs and a short reproduction note. No edits to the18 frozen package files, config, existing successful artifacts or R1/R2 amendment. No R3 source map/framework. The script is ordinary tooling and its exact contents are retained for reviewer inspection; no checksum required.

First run two **fresh diagnostic processes**, sequentially:

- D4 reconstructs the known original launch limits OMP=4/MKL=4 and records inherited OPENBLAS/NUMEXPR limits and effective torch counts. It must call this the reconstructed original launch profile, not pretend missing historical thread counts were measured.
- D1 sets all four named CPU limits to1 before imports and sets torch intra/inter-op1. Fail if requested limits cannot be established before workload. Do not change these process-global settings midway through an imported process or mutate shell/global environment defaults.

Both use project Python, PYTHONFAULTHANDLER=1 and the same cuda:1 device. They read only current KuaiRand X and the existing **V row IDs**; no y file or labels need be opened. No model training or E access. Record type/base, shape, dtype, strides, owning/writeable flags for X/index/batch/copy; ID bounds and equality of the two profiles' fixed row traversal. Avoid dumping raw arrays to chat.

Exact bounded workload per process: **two complete sequential sweeps of existing V rows using the registered HyperGate eval batch size**, keeping the real final short batch. No random draws for sampling. In each batch separate the compound line into named diagnostic steps:

1. Slice V IDs and advanced-index X.
2. Construct the same `np.array(batch)` copy.
3. Construct float32 CPU as_tensor of that copy, and the original-style float32 CUDA as_tensor of that copy; explicitly synchronize CUDA **inside diagnostics only** before validating the returned device values.
4. Check exact CPU and CUDA roundtrip value equality with the indexed batch, original shape/dtype/order; log a short flushed stage marker before/after each native operation with sweep/batch and row bounds. Do not change feature writeability, dtype or values to conceal a failure.

Include one tiny synthetic lifetime check per profile: CPU tensor from an owned temporary float32 array still has the expected values after dropping the Python array variable/collecting garbage; same for completed CUDA conversion. Use synthetic constants, no stochastic sampling. Verify Python, NumPy and Torch CPU RNG states unchanged by the diagnostic data/conversion work. For CUDA compare RNG state immediately before/after work **after initial CUDA context initialization**, so initialization is not confused with a random draw. Retain before/after direct equality results, not hashes. The diagnostic is not proof of training-state equivalence.

Output one compact evidence record per profile and stage log to an owned artifact path. Each process runs at most the two declared V sweeps; no retry-until-pass, increasing loops or long stress matrix. A clean run is only failure-to-reproduce in this workload. If it crashes reproducibly, one additional invocation of that exact diagnostic/profile under installed `/usr/bin/gdb` may collect `thread apply all bt`; do not attach to others' processes, install debugging packages, run another training trajectory, or gather a core dump automatically. Retain exit signal/native frames and stop the workload when this one diagnostic replay ends.

## 5. What outcomes mean and permitted next action

- D4 fails but D1 passes: supports a thread-sensitive runtime association for this reproducer, not proof of a particular race or library cause. Continue only if value/RNG checks pass and independent reviewer approves the one replacement attempt.
- D4 and D1 both pass: no reproducer obtained. The known4→1 CPU limit change remains a concrete, low-impact **weakly motivated operational probe**, unlike repeating the exact crashing setting. It may proceed once after reviewer PASS with uncertainty disclosed and the exact19-epoch replay gate below.
- D1 fails or either profile produces wrong values/RNG drift: do not resume training. Preserve findings and native trace; no automatic source/dependency changes or repeated attempts.
- If diagnostics already show a concrete separate lifetime/index/shape violation, report it and return to design with that evidence rather than applying unreviewed patches. No speculative broad code repair is authorized.

No extra model/ablation is needed to compare diagnostic profiles; exact feature/CPU/CUDA equality and the real fixed-seed replay are the relevant tests. Native service thread counts alone do not demonstrate concurrent mutation of these arrays.

## 6. Independent review and one conditional formal replacement

Reviewer independently inspects this design, external script, its actual finite results, source-byte comparisons showing the R2 package unchanged, and coder report. PASS/PASS WITH FIXES (all blocking issues corrected) is required before real training resumes. Reviewer may approve the bounded attempt after both diagnostic profiles pass, explicitly treating the causal evidence as weak.

Then launch **only the original formal HYPERGATE/kuairand/seed2025**, from its original initialization, with D1 process-local CPU limits. The external launcher may set thread counts then use runpy/the existing train entry unchanged; it must preserve original argument/config/device/seed handling and consume no RNG before the original seed setup. No new training code or config field. Record actual process environment, torch counts and source identity before entering the existing CLI, linked to this run and process ID in a separate runtime-profile sidecar. Do not edit existing successful run/checkpoint JSON to retrofit thread fields. The new run's existing environment capture remains truthful; the sidecar supplies the missing thread detail.

Archive the failed HG2025 attempt once under a unique owned attempt path without changing its original run/log/source/checkpoints. The successful replacement occupies the canonical matrix path as one result for seed2025; failures remain extra cost, not statistical repeats.

The replacement uses the full original training budget/patience, not a19-epoch shortened config. Compare **all first19 complete scientific epoch records** to the preserved failed attempt, with exact direct equality for every field except `seconds` and separately identified process/timing diagnostic fields. Specifically retain train loss/components/update/exposure counts, six validation cells, macro AUC/LL, temperature/stage/epoch. Maximum absolute V macro AUC and LL difference must be0. Do not relax tolerances after observing results. Log any absent fields as unavailable, not compared. Inputs/source/config/role manifests must match the original R2 identities.

Only after that exact replay **and** clean completion of full normal checkpoint/save/posttrain checks may the remaining6 formal entries proceed, using the same D1 profile and unchanged R2 source. If any first19 scientific value differs or any further native crash occurs, stop further launches, retain evidence and report the blocker. No second HG2025 retry, seed replacement, switching back to4 threads, new backend, CUDA_LAUNCH_BLOCKING=1 production change or growing experiment budget is automatically allowed.

The retry is a runtime diagnostic replacement within the approved30 matrix. It does not establish thread causality merely by succeeding. It also does not establish absence of latent faults in already successful runs.

## 7. Preserve23 successful results and existing final freeze

Keep all23 complete trajectories, both checkpoint views, original run identities and source snapshots unchanged, whether trained under R1 or R2. The remaining/replacement runs still produce **exact R2 snapshots**, so the current explicit R1/R2 source-equivalence freezer needs no amendment logic or allowed-source expansion. Default strict/source-pair checks are not bypassed. Exact source/config/data/checkpoint comparisons and complete30 requirement remain as currently reviewed.

Thread-profile sidecars and this third-incident/replay evidence are separate actual runtime provenance. Link them in final reporting and the final-freeze delivery record, without editing frozen source inventories, retroactively changing23 run environments, or asserting all runs used one thread profile. If desired, retain an adjacent final-freeze evidence index referencing the generated freeze plus these sidecars; no code change to build_freeze is needed.

The R2 review and seed2022 replay remain true evidence of that specific source-equivalence event. Add the third crash as evidence that it did not eliminate failures; do not rewrite the prior receipt to claim a stronger fix. E stays locked until all30 valid results and the existing full mixed-R1/R2 freeze actually pass. No E label is used in diagnosis or replay decisions.

## 8. Logging and cost

Required small records: actual thread-profile diagnostic outputs, coder/reviewer reports, third incident/failure cost entry, external launcher content, actual replacement runtime-profile sidecar and19-epoch replay comparison. Record successful/failed process statuses, elapsed diagnostic/failed-run costs and selected/final result paths. Different thread profiles affect runtime comparisons; disclose that limitation and preserve actual per-run timing. No new timing study or rerun of23 successes is required for performance-cost standardization.

## 9. Acceptance and falsification

Accepted implementation evidence: exactly bounded fresh D4/D1 diagnostics; no package/config/dependency drift; actual X/index/copy storage properties documented; exact feature/CPU/CUDA values and unchanged RNG; label files not read; reviewer PASS; original HG2025 seed/config full replay with19 exact scientific records; complete checkpoints and unchanged final-freeze checks. Existing tests need not be rerun wholesale when no package source changes; run the small lifetime/value/RNG assertions in the diagnostic and necessary launch/source checks.

H_THREAD is weakened by no differential reproduction and falsified as a sufficient mitigation if the explicitly limited D1 workload/retry still crashes. Exact numerical trajectory equivalence is falsified by any nonzero scientific replay difference. Either failure stops the queue. Neither justifies algorithm improvement, tuning or a broad source-compatibility layer.

## 10. Evidence strength and scientific interpretation

**STRONG**: a crash recurred after resident-X R2; X/roles storage and tensor-copy expression are as inspected; third failure occurred on the compound conversion/index line. **WEAK**: the particular failing native operation, threading as root cause, dangling-buffer or readonly-write explanations. A successful19-epoch replay would provide direct bounded trajectory evidence for this environment change; a full clean run adds operational completion evidence, not a root-cause proof. Existing efficacy questions, seed/domain/task analysis and sample-size cautions remain unchanged.

## 11. Current boundary

Designer has inspected files/docs only, changed only this design document, and run no CPU workload or real training. Coder implements the small external diagnostic, runner records its raw outcomes, reviewer inspects before any formal replacement; orchestrator interprets evidence and enforces the stop gates. No parallel research experiments or new source revision are required.
