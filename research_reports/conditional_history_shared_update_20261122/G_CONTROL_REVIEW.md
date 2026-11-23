# REVIEW REPORT — G_CONTROL

## 1. Verdict: PASS

The Gaussian control identity audit is correct for its frozen sources. This verdict approves the audit and the documented technical pause; it does not authorize G_FROZEN training or establish a negative scientific result. The trigger remains TRIGGERED, while the supplement remains TECHNICAL_BLOCKED.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Original scope and provenance | Read original 00, 06 §11, 08 and 12 and the DESIGN SPECIFICATION. All four original plan files equal immutable plan commit 4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983. Separately hashed all 29 recorded Git blobs; every digest matches the audit. |
| Trigger and fixed configuration | Trigger is Validation-only A21, S_DENSE, pair lambda .05; D0, mix .5, kappa ML .05/KR .1, seeds 2022–2026. The audit checks its A_DOSE gate binding and frozen Gaussian settings. |
| Historical population and endpoint identity | Inspected comparisons of nine population/preprocessing fields against S0, raw CSV hashes, original batch order and training row population. Both official five-seed tables retain source/config/model/trajectory/selection/endpoint identity; MovieLens selections match the pinned source audit, and KuaiRand selections are the earliest maximum over all 50 Validation epochs. |
| Actual RNG mismatch | Inspected pinned EMA02 normal/objective and GA31 draw_view/apply_view/objective implementations. EMA02 uses row/coordinate keyed SplitMix64 and Box–Muller; GA31 uses a step-keyed device-local torch.Generator and one torch.randn matrix per sparse field. Recomputed all ten actual first batches and registered FP32 noise fingerprints against archived native GPU draw receipts. All ten differ. This is a concrete sampled-stream mismatch, not an inference from generator names or a CPU emulation of historical CUDA draws. |
| Exact available controls | The three row-keyed A1 endpoint lineages agree in selected epoch/model hash and supply only MovieLens 2022/2024. The original EMA02 configuration independently confirms KuaiRand target_domain=2; those records are ineligible for D0. Completion aliases point back to the AIG-M source, rather than adding row-keyed controls. The 11/06 source and 11/10–11/11 adapters retain the implicit GA31 path. |
| Formula, dimensions, masks and weighting | Both objective paths implement [sum(C)+.5*(sum_D0(A)-sum_D0(C))]/(2B). Noise shape is [D0 rows, sparse fields,16], with six ML and 33 KR fields; domain embedding is excluded. ML D0 uses age 1/18; KR D0 uses tab0 after the original split/filter. The full-batch and targeted-row identities are consistent with the recorded draws. |
| Gradient semantics | The source applies detached per-field embedding RMS scaling and retains differentiable clean/view predictions. The audit calls no model forward/backward, Adam or EMA and creates no research update path. A21/G composition gradients are outside this identity-only implementation and have not been approved here. |
| Baseline preservation and scope | The reviewed code only reads pinned sources/data and generates a scalar identity audit. No baseline source or endpoint is modified. Original 00 S4 requires pausing this supplement when enhanced-control identity mismatches; original 08 conditions its ten candidate jobs on reusable matched controls. No additional control trajectory is warranted by that original matrix. |
| Correct disposition | Missing controls are ML 2023/2025/2026 and KR 2022–2026. The report records TECHNICAL_BLOCKED rather than NO_GO or a measured performance failure, preserves the existing scientific evidence, and does not claim Gaussian or A21 is ineffective. No G_FROZEN run artifact is present; no scientific training or new Test scoring was performed by this review. |

The audit was rerun in memory through the required project Python environment. After JSON serialization, its complete output equals the saved audit exactly, including every pinned source, endpoint and first-batch fixture. The focused existing final-evaluation control-gap test passed: 1 passed, 5 deselected. That test rejects an augmented final selection, a self-review, an unbound receipt and a claimed complete-control matrix for this blocked disposition.

## 3. Bugs/deviations found

None. CRITICAL: 0; MAJOR: 0; MINOR: 0. The missing matched controls are an evidenced technical prerequisite gap, not an implementation bug or a scientific negative finding.

## 4. Corrections made

None. Only this independent report and its receipt were written; research code, the final audit and trigger were not edited.

## 5. Remaining risks

Coverage is a finding about the registered pinned sources and inspected source lineages, not a proof that no unregistered archive exists anywhere. The first mismatching batch is sufficient to reject exact trajectory identity; full 50-epoch replay was unnecessary and was not performed. The two usable MovieLens controls do not form the required two-dataset, five-seed matrix. This review does not grant current RAM-state or Test-evaluation eligibility.

## 6. Whether experiments are safe to run

The read-only CPU identity audit is safe to reproduce. G_FROZEN scientific training remains blocked under original 00/08 until its matched-control prerequisite is legitimately resolved; this review neither adds a control matrix nor authorizes extra trajectories. S5 may document this technical gap while retaining the independently resolved unaugmented candidate. No new Test evaluation is authorized.

## Exact reviewed identities

- `artifacts/conditional_history_shared_update_20261122/SOURCE_G_CONTROL_AUDIT.json`: `10a8b2ed10b944ce21292526cc9a0920c3dfcc850acc2199d17ccaa68ed002c3`
- `artifacts/conditional_history_shared_update_20261122/G_FROZEN_TRIGGER.json`: `dd2c2d1589113ddf2f1bec9f757431659586d4962a501e1dbaff27bc6dc84179`
- `experiments/conditional_history_shared_update_20261122/gaussian_control_audit.py`: `bfcfac3d739987a2751b41ba96090343235ef0ad3d13780c714dee5aff6494ef`
- `research_reports/conditional_history_shared_update_20261122/CODER_IMPLEMENTATION_REPORT_G_FROZEN.md`: `aa1549e7a0f109690c2f329b54505028bb00c83dfeee331bb42b7ec2ee33b43c`
- `research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md`: `607fb9b7ada44aebae7cd8d80c1fd163e83206607e751de455a46efbcd7a3272`

Not checked: new model performance or full-trajectory replay. Remaining limitation: the registered matched D0 control matrix is incomplete.
