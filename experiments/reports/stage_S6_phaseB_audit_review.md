# REVIEW REPORT

Independent post-run review of S6 Phase B, 2026-10-08. Scope: implementation fidelity and audit validity, without examining performance conclusions or running training. No algorithm, training, collector, queue, or raw artifact was modified.

## 1. Verdict

**FAIL for an unqualified claim that the complete S6 matrix satisfies every design identification requirement.** KuaiRand fails the explicitly specified M1 first-moment proximity criterion at four of eight seed/epoch checks. This remains a failed requirement; this review does not replace it with another criterion, waive it using S5 precedent, or redefine its threshold.

**Execution/implementation fidelity: PASS for both datasets. MovieLens M1: PASS.** The existing raw matrices are suitable for transparent diagnostic analysis of the four implemented optimizer policies. The KuaiRand results must not be described as having passed M1 or as a fully M1-validated separation of matched realized Adam states. No detected implementation defect requires discarding the raw runs or rerunning the same correct implementation. A coder cannot repair an observed cross-trajectory inequality without changing the experimental conditions or validity rule; such changes would require a new design, outside this review.

## 2. Requirement-by-Requirement Verification

Authoritative inputs read: `experiments/stage_S6_design.md`, the actual S6/S6ML protocol/run/orchestrator/report sources, inherited S3/S4 objective and Adam controls, `experiments/reports/stage_S5_implementation.md`, `experiments/reports/stage_S5_review.md`, `stage_S6_phaseB_extension_scope.md`, and `stage_S6_phaseB_run_report.md`. A standalone S6 coder implementation report and pre-run S6 reviewer report were unavailable in `experiments/reports`; historical approval cannot be independently established from those missing documents. The user explicitly requested direct runner completion using completed code. This review is post-run and does not retroactively supply a pre-run approval.

| Requirement | Independent finding |
|---|---|
| Original Phase A STOP discipline | The extension scope expressly records the user's override of the no-Phase-B execution branch. Phase B is diagnostic completion after STOP; original STOP thresholds and decisions remain historical facts. |
| Four-arm definitions | PASS. F=(FULL history,FULL current), T=(TARGET,TARGET), HT=(TARGET,FULL), DH=(FULL,TARGET), in both protocol maps, actual dispatch, and every gradient row. |
| O1 Sample coefficients | PASS across all 32 canonical runs. All loss weights are 1, effective domain shares equal count/total, cell shares equal half the domain share, population weight=1, double weighting=false. KuaiRand coefficient pin is `30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7`; MovieLens is `e0616863f6970789ae92b1d2f975d497da83d89bd224af7416496f5ccbf63463`. |
| O2/O3 objective calls and paths | PASS. Every recorded objective call uses only its Sample coefficient pin; every batch trace says sample. The inherited `loss_components` Sample branch uses ones and ordinary population mean; the target term is masked BCE divided by the whole batch size, with no target renormalization. |
| I1 history installation | PASS. Actual code installs the selected `g_H`; installed/history cosines across all rows are within rounding of 1. |
| I2/I3 actual history and applied moves | PASS. Both counterfactual deltas are computed before the single optimizer step. The observed intermediate move matches the history delta; the final move matches the current delta within frozen tolerances. |
| I4 private update | PASS. Private gradients come from full loss. The overwrite loop covers shared or architecture names only. Private update residuals pass; all final main optimizer states, including private tensors, have the expected counter. |
| I5 counters | PASS. Independently checked every row for exact sequential counters separately for main and architecture steps. Final main state counters: 2450 KuaiRand, 250 MovieLens; architecture: 50. All private and shared state tensors agree within their optimizer. |
| I6 state contamination | PASS within the available evidence. `adam_delta` performs read-only state arithmetic; only `optimizer.step()` commits moments. HT/DH overwrite parameters only. Probe digest equality is hard-asserted after overwrite/logging; all runs completed. Raw artifacts do not store both digest values, so this is code plus successful assertion evidence, rather than independently replayable per-step moment transitions. |
| I7 RNG / I8 pairing | PASS. Every gradient row, including non-probes, has equal before/after RNG digests. All four arms have identical ordered `(kind,epoch,step,batch_fingerprint)` sequences within every seed. Pairing fields were independently compared across phases. |
| I9 missing/zero gradients | PASS. Zero shared/architecture None counts and zero target-zero steps across both matrices. |
| L1/L2 locks and source identity | PASS. Data, splits, initialization, partitions, parameter manifests, domain counts, batch cardinality, and frozen hyperparameters match the corresponding S5/S5ML anchors. KuaiRand config-lock table has 192/192 OK rows. MovieLens locks were independently checked despite no analogous config-lock table being emitted. Producer-critical live hashes match every manifest. |
| P2 matrix cardinality | PASS. Each dataset has 16 canonical runs, 672 test cell rows, 1600 probes, 28 moment rows. KuaiRand has 40,000 gradient rows (39,200 main+800 architecture); MovieLens has 4,800 (4,000 main+800 architecture). |
| C1 checkpoint discipline | Manifests and actual validation code preserve validation-only unweighted macro-AUC selection, earliest tie, frozen selection before test, and no test evaluation during training. The collectors validate final/common49 parameter equality. All corresponding artifact hashes independently recomputed correctly. Final/common49 is one observation, not an independent replication. |
| M1 early moments | **KuaiRand FAIL:** m conjunction passes 4/8 seed-epoch checks, v passes 8/8. **MovieLens PASS:** m and v both pass 8/8. Independent recomputation from all 64 early moment dumps reproduces all reported pair cosines exactly. |
| Two-process concurrency | Runner report and frozen queue/state receipts record two HT/DH workers per seed wave, positive overlap, no retries, 16/16 newly completed runs. This review did not launch processes. |

The actual S6 initialization intentionally calls `S3Runner.__init__` directly to avoid S4's hardcoded Equal objective, and recreates the small S4 initialization fields. This departs from the design's literal clone instruction but is necessary to execute the specified Sample objective. Actual objective artifacts prove Sample was used. The `_controlled_step` bodies match their S5/S5ML counterparts after normalizing stage strings; moment snapshot/digest/cosine helper logic matches as well (documentation text differs).

Phase A and B differ in the hashes of protocol/orchestrator/report plumbing. Cross-phase comparison excludes exactly those three files and retains all inherited producer sources plus the stage run file. Every producer-critical live hash was independently checked. Excluding protocol bytes is not unconditional permission to change training constants: actual arm maps, dataset/objective/initialization pins, counters, optimizer settings, and target identities must still agree. They do for these artifacts. The missing standalone S6 implementation/review documents limit verification of the historical amendment process.

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

The inherited Adam helper implements coupled weight decay: `effective=g+weight_decay*theta`, `m_next=beta1*m+(1-beta1)*effective`, `v_next=beta2*v+(1-beta2)*effective^2`, bias correction at `step_pre+1`, and the configured epsilon/lr. No second optimizer step occurs. All deltas use the local arm's pre-step parameters and persistent state. These are source-factorized optimizer policies along different parameter trajectories, not transplantation of F/T state tensors from other runs.

`gradients_one_forward` differentiates full and target losses from the same forward graph; private gradients are full-loss gradients. For main updates the manipulated partition is C; for architecture updates it is A_arch. Moment dumps contain C plus A_arch, with the separate optimizer clocks preserved. At epochs 0/1 KuaiRand dump step sets are {1,49}/{2,98}; MovieLens {1,5}/{2,10}. Those ranges reflect separate architecture/main clocks, not counter desynchronization.

| All-row independent audit | KuaiRand | MovieLens |
|---|---:|---:|
| Maximum history residual | 1.63853e-6 | 1.59794e-6 |
| Maximum applied residual | 1.63512e-6 | 1.59794e-6 |
| Maximum private residual | 1.13656e-6 | 8.33395e-7 |
| Maximum history residual / allowed tolerance | 0.09357 | 0.09797 |
| Maximum applied residual / allowed tolerance | 0.10304 | 0.09700 |
| Maximum private residual / allowed tolerance | 0.06451 | 0.04844 |
| Decomposition residual | exactly 0 | exactly 0 |
| None / target-zero / RNG mismatches | 0 / 0 / 0 | 0 / 0 / 0 |

Absolute residuals above 1e-6 are allowed when the frozen relative tolerance is larger; no tolerance was adjusted during this review.

### Gradient flow & Loss participation

Autograd computes both required loss gradients before detaching their copies for stock Adam installation. The detaches occur after differentiation and do not disconnect a trainable component from its intended gradient. Full private loss participates in each main optimization. Architecture parameters participate through their dedicated full/target dispatch. Finite loss/gradient/delta/parameter checks and empty nonfinite inventories provide successful-run evidence against NaN/Inf propagation.

### Controlled variables & Baseline preservation

The review read the existing F/T and Equal anchors without modifying them. Same-seed pairing was checked independently for every manifest field used by the cross-phase collector, including data/splits/initialization/partition/exposure/main and architecture batch hashes and clocks/metric implementation/checkpoint protocol. All recorded artifact hashes were recomputed for all 32 runs and matched. Target identities remain KuaiRand d1 and MovieLens d2. Sample and Equal differ in objective coefficients; between-objective batch ordering is informational rather than a required lock under design §6.3.

### Fail-fast enforcement

No silent objective/configuration fallback was found. Unsupported Adam settings, invalid losses, nonfinite tensors, update discrepancies, missing shared/architecture gradients, source drift, and required-artifact/identity failures raise or return an explicit failed validation. Validation's exception wrapper records failure reasons; it does not mark failed validation successful. The inherited StopIteration handler resets the architecture loader by design and does not suppress training errors.

M1 is **not** a fail-fast collector condition: `stage_S6_report.py` computes its Boolean and writes `moment_gate_overall_pass=false` while retaining `status=STRUCTURALLY_VALIDATED`. This is transparent in the machine-readable manifest but must not be read as scientific identification approval. The runner report's structural success counts do not establish M1 success.

## 4. Over-Engineering & Code Bloat Audit

No new factory, registry, compatibility hierarchy, configuration framework, or training abstraction was introduced by this review. The existing cloned stage modules are the design's prescribed arrangement. Source/identity checks, formula residuals, pairing, and process receipts protect experimental comparability and should remain. No speculative defensive additions or removals were made.

## 5. Bugs / Deviations Found

**CRITICAL:** No detected algorithm, gradient, update-routing, contamination, data-corruption, or baseline-modification bug in the inspected implementation/artifacts. This statement is limited to the recorded evidence, not a proof from unavailable full per-step tensors.

**MAJOR — unmet M1 identification requirement in KuaiRand.** Define HT margin as cos(HT,T)-cos(HT,F), and DH margin as cos(DH,F)-cos(DH,T), separately for m/v. Both margins must be strictly positive.

| Seed | Epoch | m HT margin | m DH margin | m conjunction | v conjunction |
|---|---:|---:|---:|---|---|
| 2022 | 0 | -0.034231 | -0.015792 | FAIL | PASS |
| 2022 | 1 | +0.161342 | -0.125357 | FAIL | PASS |
| 2023 | 0 | +0.005663 | +0.008741 | PASS | PASS |
| 2023 | 1 | -0.157011 | +0.239659 | FAIL | PASS |
| 2024 | 0 | -0.013301 | +0.000869 | FAIL | PASS |
| 2024 | 1 | +0.009808 | +0.152827 | PASS | PASS |
| 2025 | 0 | +0.009947 | +0.010285 | PASS | PASS |
| 2025 | 1 | +0.073282 | +0.621579 | PASS | PASS |

This is five failed individual inequalities out of sixteen, and four failed conjunctions out of eight. It is not floating point noise. Early KuaiRand m norms range approximately 3.247e-4 to 3.631e-3; the states are nonzero. KuaiRand v margins remain positive (approximately 0.01407–0.04393). MovieLens has every m/v margin positive (m approximately 0.01037–0.06172; v 0.01452–0.07195).

The failure does **not itself establish an implementation or state-contamination defect**. HT's installed sequence is `g_target(theta_HT)` and T's is `g_target(theta_T)`; likewise DH and F use full gradients at different parameters. Coupled decay also evaluates each arm's own theta. Because current updates differ, the realized signed first-moment tensors need not be closer to a different run with the same nominal gradient source. The main clocks differ substantially by dataset: 49 versus 5 steps per epoch; with beta1=0.9, initial signed-history retention after the first epoch is about 0.00573 versus 0.59049. Trajectory divergence is therefore a plausible diagnostic limitation, not an independently proven complete explanation of these particular failures. Passing v and local update invariants support correctly dispatched policies but do not logically imply that failed m inequalities should pass.

S5's post-run review examined one failed m comparison and explicitly discussed signed-EMA trajectory divergence. That prior ruling does not authorize changing S6 M1. The S6 failures occur across three seeds, include epoch 0, and include both HT and DH sides. This review does not label them a negligible isolated S5-style exception.

**MAJOR — validity disclosure gap.** The Phase B runner report records structural success without reporting the KuaiRand M1 failure. This review supplies the missing disclosure and binds downstream interpretation. Raw success must not be summarized as every scientific gate passing.

**MINOR — incomplete historical documentation.** Standalone S6 coder/pre-run reviewer reports are unavailable. Existing raw source hashes and this independent review provide current fidelity evidence but cannot establish the missing historical approval record.

## 6. Corrections Made

Only this review document was added. It records exact M1 failure localization, independently reproduced gate arithmetic, execution-fidelity findings, and restrictions on interpretation. No source, criterion, tolerance, seed, training result, or manifest status was changed. A diagnostic explanation is not treated as a correction to the failed criterion.

## 7. Remaining Risks

- KuaiRand cannot support an unqualified statement that the designed early realized-history proximity validation succeeded. Realized Adam m/v and weight-decay terms follow arm-specific trajectories.
- Digest no-contamination enforcement is sampled at probes; successful code assertions are available, but pre/post optimizer tensors are not recorded at every training step. Code inspection found no mutation path outside the single optimizer step.
- Phase B follows user-authorized completion after STOP; it must remain labeled diagnostic and cannot retroactively become the original GO-triggered confirmation.
- Selected checkpoints are independently chosen by arm; same-epoch contrasts must be distinguished. Final and common49 are duplicate observations.
- The absent S6 implementation/pre-run review reports prevent a complete historical workflow audit.

## 8. Whether Experiments Are Safe to Run

All requested missing runs are already completed; this review launched no experiment. Existing code shows no detected correctness defect requiring repair before another execution, but **no additional run or gate modification is authorized by this review**. Repeating these seeds to force M1 acceptance or dropping failing checks would be improper.

Main may analyze the complete existing matrix as a diagnostic comparison of faithfully implemented optimizer policies, disclose KuaiRand's failed M1 explicitly, and keep mechanistic evidence conditional. Main may not state that all design identification checks passed or that matching realized cross-arm m states has been established for KuaiRand. MovieLens satisfies the specified M1 check, within the same limits of policy/trajectory interpretation. Any future attempt to change the validity criterion must return to a designer; it is not a coder repair.
