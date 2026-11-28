# C0 implementation review

## 1. Verdict

**PASS.** The foreground CPU entry is approved for supervised C0 inspection of completed A1.B0 and A1.P0 trajectories. Review covered C0 source SHA-256 `a9b36fab67dd27687892442a648e3d4c685f8cea4c0d833752655aa421abaeda`. [Design](C0_EXECUTION_DESIGN.md) · [Coder handoff](C0_CODER_IMPLEMENTATION_REPORT.md)

## 2. Requirement verification

| Requirement | Evidence |
|---|---|
| Approved inputs and source binding | The entry verifies the existing frozen A1 roster and enumerates exactly the two control arms across both datasets and five seeds. The cache loader opens the feature schema and the three immutable Train arrays. It compares array and ordered row hashes, content fingerprints, feature encoding and scaling metadata, and six-cell support with the audited and pinned identities. The full data hash comes from the existing Train/Validation receipt and A1 binding. Validation input consists of recorded metrics. [Roster and cache](../../experiments/three_directions_20261128/c0_audit.py#L48) |
| Completion and named model readback | The job record and full binding must agree with the epoch-50 completion and journal. E05, E10 and selected readback each supply the named key, kind, readout, epoch, logical step and model hash. Journal readback checks generation, writer, packet bytes, decoded digest and model digest. The fixture exercised all three keys for both arms through actual synthetic RAM packets. Same-hash keys each receive a fresh readback and share inference. [Completion and snapshot binding](../../experiments/three_directions_20261128/c0_audit.py#L149) · [Journal](../../experiments/three_directions_20261128/ram_state.py#L210) |
| Validation and timing | The code checks the earliest exact maximum over 50 committed six-cell Macro AUC records and joins same-checkpoint LogLoss. The selected result agrees with the final selection and Validation endpoint. E10’s published endpoint provides an additional hash check. Selection before or at a snapshot yields null pre-selection rates, and a same-epoch snapshot requires identical hashes and margins. E5→E10 formation requires selection at or after E10. [Selection and windows](../../experiments/three_directions_20261128/c0_audit.py#L149) |
| Candidates, inference and gradients | The registered pools and context RNG produce sorted unique ranks and fixed candidate identities shared by each arm pair. The pinned M3oE factory is used on CPU with a checked parameter scope, strict state loading, FP32 logits and inference mode. The audit uses FP64 subtraction on the routed FP32 logits. A separate synthetic check with both pinned dataset configurations reproduced direct tower indexing and logit margins for both tasks and all three domains. Sigmoid parity held at the specified tolerances. [Candidate and scoring paths](../../experiments/three_directions_20261128/c0_audit.py#L109) |
| Macro6 metrics and replay | Strict signs govern reversal, tie loss and retention. Weighted conditional rates divide separately weighted numerator and denominator. Confidence clipping and the 0.001 sensitivity band are diagnostic. Replay uses the A1 sampler and requires condition and pair-hash agreement at the first and last logical steps of each eligible window, using the committed epoch writer and verifying earlier attempts at matching boundary steps. An independent stable-ID counter reproduced repeated draws, observed members, historical-positive hits and uniform expected hits with three batches per epoch. [Replay and aggregation](../../experiments/three_directions_20261128/c0_audit.py#L317) |
| Coverage, baseline preservation and outputs | The two-arm integration fixture reports all 20 expected dispositions, preserves the original A1 fixture file set and hashes, and produces paired summaries with distinct selected horizons. Pending inputs and technical input failures yield explicit dispositions and preserve observations from eligible jobs. The audit checks stop controls before inspection and between job units. C0 publishes candidate IDs, aligned margins, numeric summaries and costs under its own directory. Conditional memory exposure remains a formula with an unassigned bank size. [Inventory and publication](../../experiments/three_directions_20261128/c0_audit.py#L659) · [Focused tests](../../experiments/three_directions_20261128/tests/test_c0_audit.py#L315) |

## 3. Bugs or deviations

No critical, major or minor implementation defects were found in the reviewed path.

## 4. Corrections

The implementation and fixtures passed as delivered. The project Python reported 16 passing focused tests, and the CLI help worked from the specified working directory. The inherited PyTorch TF32 API emitted one deprecation warning.

## 5. Remaining risks

The review exercised synthetic Train rows and synthetic RAM packets. Production cache verification, availability of the 20 A1 inputs and fresh readback of the retained models are RUN checks. RAM read, integrity verification and decode share one measured duration because the reader exposes a synchronous call. The stop paths registered for this study and the current history study were clear in both original workspaces and the study worktree at review time.

## 6. Execution gate

Safe for the approved C0 offline RUN. Complete verified accounting is required for scientific interpretation. C1 remains subject to the dataset-specific scientific gate and a governing capacity ruling. Test stays sealed.
