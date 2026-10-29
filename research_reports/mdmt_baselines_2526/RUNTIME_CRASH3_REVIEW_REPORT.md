# REVIEW REPORT — third native crash / external thread diagnostic

Date: 2026-10-29. **Verdict: PASS — bounded original HG/KuaiRand/2025 D1 replacement only.** No remaining blocking implementation issue was found in this review. The remaining6 runs are conditional on exact first19 scientific replay and clean full completion; final freeze/E remain subject to the existing complete30 guards. This PASS is not evidence of a diagnosed or eliminated native fault.

## 1. Design inspection

Read `DESIGN_RUNTIME_CRASH3_ADDENDUM.md` and the current R2 `data.py`, `evaluation.py`, `training.py` and CLI caller. The proposed scope is sufficiently narrow: one external tool, unchanged frozen package/config/dependencies, two fresh diagnostic profiles with exactly two full V-feature sweeps each, and one conditional original HyperGate/KuaiRand/2025 replacement attempt after independent implementation review.

The current `tensor` helper calls `np.array(value)` before `torch.as_tensor`; current X is ordinary read-only ndarray, and the role/index construction uses ordinary arrays. These facts do not establish the failing native instruction or root cause. A new tensor-copy patch or R3 source inventory is not justified by the inspected evidence.

D4 must describe reconstructed OMP=4/MKL=4 launch limits and record inherited/effective counts without claiming historical measurements. D1 must establish all four CPU-library limits before imports and torch intra/inter-op limits before computation. Diagnostic assertions must cover actual CPU/CUDA values, synthetic temporary-array lifetime, RNG conservation after CUDA initialization, storage properties, fixed complete V traversal and no label access. No training equivalence claim follows from a clean diagnostic alone.

The external launch caller must preserve existing CLI arguments/config/device/seed handling and remain reusable for later authorized formal runs and post-freeze evaluation. Stage and queue authorization belongs to orchestration; no separate permission framework is required in the launcher. Sidecars must record actual per-process thread provenance without rewriting existing successful artifacts.

## 2. Final review gate and reviewed deliverables

The initial design-only review withheld a verdict until implementation and actual evidence were available. Final review has now independently inspected the sealed `runtime_crash3_tool.py`, `CODER_RUNTIME_CRASH3_REPORT.md`, `CRASH3_DIAGNOSTIC_RAW_RESULTS.json`, both profiles' `diagnostic.json`, `runner_execution.json`, stage JSONL and stderr logs. The reviewer independently ran the tiny CPU selfcheck and a stubbed launcher check; no real CLI command, GPU diagnostic sweep, training or E evaluation was launched by the reviewer.

After a possible PASS, only the single original full-budget HG/KuaiRand/2025 replacement is eligible. Its first19 complete scientific epoch records must compare exactly, excluding timing/process diagnostics, and the full run/checkpoint/posttrain checks must finish cleanly before the remaining6. Any scientific difference or further native crash stops launches. E retains the existing complete30/mixed-R1-R2 freeze gate.

## 3. Requirement-by-requirement verification

| Requirement | Independent verification | Result |
|---|---|---|
| No new package revision or config drift | Decoded all18 approved R2 `content_base64` records and compared actual current file bytes and sizes directly; all18 equal. This includes configuration, training, model/loss adapters, data/evaluation/report and reference JSONs. No hashes computed. | PASS |
| Fresh process limits before library workload | Top-level tool imports only standard library. `_load_stack` sets environment before importing NumPy/PyTorch; D1 sets/checks torch intra/inter-op1. Distinct actual D4/D1 PIDs and sequential process times retained. | PASS |
| Reconstructed D4 distinguished from measured history | Actual D4 OMP/MKL4, inherited OpenBLAS/NumExpr unset; actual torch4/24, NumPy OpenBLAS4/OpenMP4. These are new diagnostic measurements, not reconstructed historical effective counts. | PASS |
| Effective D1 limits | All four named environment limits1; actual torch1/1 and NumPy runtime OpenBLAS1/OpenMP1. | PASS |
| Exactly two complete V sweeps each | Each profile recorded250767 V rows,490 batches per sweep,512 registered batch size and399-row tail; independently parsed13728 stage records/profile and verified all980 ordered batch operation sequences and row offsets/counts. | PASS |
| Same fixed V traversal | Independently loaded both saved ID arrays and the existing prepared V array; direct full equality after the same sorting. No feature/label workload rerun. | PASS |
| Actual storage and conversion semantics | Evidence shows owned read-only float32 X[2507666,34], owned int64 V IDs, ndarray index slice, owned float32 indexed batch/copy and CPU/CUDA float32 tensors. Script separately marks slice/index/copy/conversion/synchronize/validation; no production helper changes. | PASS |
| Exact values and temporary lifetime | Both real profiles report980/980 CPU and CUDA exact batches. Every final per-batch validation marker independently checked true. CPU and completed CUDA temporary-array lifetime checks pass. | PASS |
| RNG and labels | Both actual profiles report Python/NumPy/Torch CPU/CUDA direct state equality after initial CUDA context setup; inspected code makes no random sampling call. Script opens X and V IDs only, not Data/y/labels/E. | PASS |
| Actual finite process outcome and cost | D4 PID1506624 exit0,1.231062860s; D1 PID1506791 exit0,1.172181181s. Separate stderr logs empty; summary objects exactly equal per-process original records. No gdb replay was needed or run. | PASS |
| Existing CLI/seed and sidecar caller | Independently intercepted runpy without entering CLI: original HG2025 argv exact, config and KuaiRand prepared manifest exact, threads1/1, sidecar process/argv recorded and Python/NumPy/Torch CPU preseed RNG unchanged. Sidecar uses a new path and refuses overwrite. Existing CLI remains responsible for validation and E authorization. | PASS |
| Model formulas, shapes, masks, gradients and baselines | No executable package/config byte changed from approved R2, so prior accepted semantics and tests remain applicable. Diagnostic uses the same float32 NumPy-copy/as_tensor conversion; no model training or gradient modification. | PASS |
| Preserve successful artifacts and source/freeze guards | Tool has no path that rewrites existing runs/checkpoints or the source-pair record. It creates new diagnostic output directories/sidecars and delegates requested commands unchanged. Existing full-matrix/source/config/data/checkpoint/E code remains byte-identical R2. | PASS |

CPU selfcheck independently returned `PASS_CPU_SYNTHETIC_ONLY`: exact values/lifetime, all three CPU RNG states unchanged, torch threads1/1. Launcher verification replaced only the runpy boundary with an assertion stub and raised `SystemExit(0)`; no actual training entry was called. Its temporary sidecar was outside existing research artifacts.

## 4. Bugs/deviations and corrections

No CRITICAL, MAJOR or MINOR implementation defect requiring correction was found within this finite diagnostic/launcher scope. No implementation correction was made by the reviewer. Only this report was written. Diagnostic elapsed times are supplied by the runner's separate process receipts, satisfying the design without modifying the sealed script.

## 5. Remaining risk

Native root cause remains unresolved. If both finite diagnostics pass, threading remains a weak operational hypothesis rather than an established cause. Existing successful runs and timing provenance must remain truthful and unchanged.

Both profiles did pass, so there is no differential reproducer. The diagnostic contains no model forward/backward workload and cannot establish training-state equivalence. A successful replacement would supply bounded trajectory/completion evidence only. Different thread profiles limit direct runtime-cost comparison; retain actual per-run times and separate failure costs.

## 6. Whether experiments may run

**Only one original formal HYPERGATE/kuairand/seed2025 replacement, from its original initialization and full unchanged budget/config on cuda:1, may now use the D1 launcher.** Preserve the failed attempt, its19 complete epochs and original R2 snapshot. Record the new per-process sidecar without retrofitting23 successful run environments.

Before any remaining6 launch, compare all first19 scientific epoch objects exactly against the archived attempt, excluding only `seconds` and explicitly identified process/timing diagnostics. Preserve loss/components, stage/epoch/temperature, updates/exposures, all six validation cells and macro AUC/LL; both maximum absolute macro differences must be0. Record unavailable fields as unavailable. Require matching original source/config/data/roles and clean full completion of checkpoint/save/posttrain checks. Any scientific difference or another native crash stops the queue; this review authorizes no second replacement or fallback profile.

The remaining6 may use the same D1 launcher only after that gate is actually satisfied. E remains locked until all30 valid trajectories and the existing explicit mixed-R1/R2 freeze pass; later E launcher reuse does not bypass those checks. No real replacement,19-epoch comparison or E result is claimed by this pre-retry report.
