# C1 runtime release review

## 1. Verdict

**PASS.** Both historical-control binding defects from the prior review are corrected. The C1 implementation gate permits the foreground runner to freeze the initial C2/C3 roster and enter supervised RUN after the required stop and admission checks. This verdict covers C1 source `f06680455940737f70e91f8054f57e3366ba1ebc610b80b1ff17a3936220d61f`. [Execution design](C1_EXECUTION_DESIGN.md) · [Prior implementation review](C1_FINAL_IMPLEMENTATION_REVIEW.md)

## 2. Requirement verification

| Requirement | Independent evidence |
| --- | --- |
| Preserve the exact original source and controls | Read-only checks verified all 16 blobs in archive commit `70455ed0ebc898ae80275d17f14e7be80f8cfd57`. Both A1 and B1 rejected the current shared bytes as launch identities. The live A1 and C0 metadata resolved 20 exact original B0/P0 controls, 120 Validation endpoints and 50 compact receipt identities. The original stage, arm, config, source, scope, journal and completion bindings remain on the linked records. [Archive manifest](../../artifacts/three_directions_20261128/A_B_SOURCE_SNAPSHOT.json) · [Control validation](../../experiments/three_directions_20261128/c1_protocol.py#L180) |
| Bind the complete P0 warmup and E10 teacher | Authentic live metadata produced ten prefixes covering 3,450 committed batches, and all ten per-seed validation callbacks passed. Reopening the virtual C1 roster succeeded with Train/cache and compact RAM decoding disabled. Rehashed substitutions for control keys, online E10 identity and the entire missing prefix failed independently. The CPU regressions also rejected modified selected and E50 rows and a modified batch inside a rehashed prefix. [Prefix reconstruction](../../experiments/three_directions_20261128/c1_protocol.py#L277) · [Seed callback](../../experiments/three_directions_20261128/c1_protocol.py#L378) · [Roster reopen](../../experiments/three_directions_20261128/c1_protocol.py#L473) |
| Link committed E10 and E50 Validation endpoints | Each control requires exactly six endpoints across both readouts. Selected rows bind to the earliest strict Macro AUC maximum and the same checkpoint's metrics and compact model receipt. E10/E50 metrics bind to committed epoch records and the original A1 publication. In a disposable copy of the real MovieLens B0 seed-2022 metadata, replacing the E50 fixed-EMA Macro AUC with `0.125` in completion and journal failed against the committed `0.756626326787877`. It also failed when the supplied publication repeated the substitution. [Endpoint validation](../../experiments/three_directions_20261128/c1_protocol.py#L202) |
| Maintain the approved matrix and numerical path | The current metadata resolves ten C2 and ten C3 jobs with the exact P0 base config over seeds 2022–2026. C4 remains `NOT_TRIGGERED`. The targeted CPU suites passed 49 cases with two CUDA cases deselected and the existing PyTorch TF32 API warning. These tests cover bank selection, routing and weighting, detached targets and gradients, matched fallback, the clean BCE no-event path, and verified E10 rollback and replay. The previous review's archived P0 parity, CUDA acceptance, publication and scheduler evidence applies to the unchanged numerical and runtime paths. [CPU regressions](../../experiments/three_directions_20261128/tests/test_c1_memory.py#L554) · [Integration cases](../../experiments/three_directions_20261128/tests/test_c1_integration.py) · [Prior review](C1_FINAL_IMPLEMENTATION_REVIEW.md) |
| Preserve stage and data gates | B1 is complete at 30/30, the C0 20/20 accounting and independent scientific gate verified, and all ten registered stop paths were clear. This review used existing Validation and provenance metadata, with zero formal C1 jobs and zero Test accesses. [B1 status](../../artifacts/three_directions_20261128/B1/STAGE_STATUS.json) · [C0 decision](../../artifacts/three_directions_20261128/C0/decisions.json) · [Stop paths](../../artifacts/three_directions_20261128/B1/STOP_CONTROLS.json) |

## 3. Bugs and deviations

- **MAJOR, resolved.** Recomputed roster hashes previously admitted changed A1 controls and P0 E10/prefix evidence. The reopen path now requires exact equality to records reconstructed from the original evidence.
- **MAJOR, resolved.** Fresh links previously admitted E50 metrics changed in both completion and journal. Both readouts now require their committed E10 and E50 metrics.

## 4. Corrections

The coder corrected both issues in `c1_protocol.py`. This review added the release report. [Binding fix](C1_BINDING_FIX_REPORT.md)

## 5. Remaining verification

Formal RUN must check exact end-to-end P0 parity at E10 for each dataset and seed before memory exposure. Live compact decode remains part of the runner's fresh roster freeze. The focused re-review verified historical compact receipt identities using metadata and relied on the existing live readback and CUDA acceptance evidence. [Transition guard](../../experiments/three_directions_20261128/c_step.py#L58) · [Roster freeze](../../experiments/three_directions_20261128/c1_protocol.py#L443)

## 6. RUN gate

**OPEN for supervised C1 roster freeze and the initial 20 C2/C3 trajectories.** Recheck the scoped stop controls before dispatch and apply the registered current-telemetry and post-OOM physical-fit requirements. Test stays sealed. [Approved runtime commands](C1_EXECUTION_DESIGN.md#7-implementation-requirements-and-commands)
