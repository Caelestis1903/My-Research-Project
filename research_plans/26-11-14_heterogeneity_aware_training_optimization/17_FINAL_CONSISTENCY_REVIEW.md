# 17 — Final Plan Consistency Review

**Status:** REVIEWED / PLANNING_ONLY / NOT_IMPLEMENTED / NOT_RUN  
**Review base:** main@bcb7e2e61d1524a7efab2cb059f47fe042ec16b7  
**Final governing artifacts:** 16_CURRENT_MAIN_AUDIT_AND_EXECUTION_REFINEMENT.md and experiment_manifest_v3.json.

## 1. Scope and source integrity

- Current-task repository audit is main-only.
- Required main context documents were read, including AGENT.md, EXP_AI_CONTEXT.md, Global Summary, Stable Findings, Open Questions, Experiment Index, Evidence Ledger, metrics ledger and PROJECT_ARCHITECTURE.md.
- start.md and AGENTS.md are absent from main; AGENT.md is authoritative.
- The existing target directory was inspected before updates. No existing file was overwritten blindly; prior plan and critical-review history were retained.
- No historical report, training code, data, checkpoint or project-wide metrics ledger was changed.

## 2. Historical de-duplication

- PAIR already enumerates target-positive/negative joint-label conditions and includes the proposed same-other-task-label subset. A new label-only PAIR variant is excluded.
- Generic PCGrad, parameter-gradient magnitude balancing, tested ECCM variants and current adaptive EMA/PUB versions are not reopened.
- Fixed EMA remains a strong internal control; its existing benefit is not credited to the new method.
- Domain size is not used as a difficulty proxy. No experiment or analysis is proposed to prove or disprove the fixed boundary “small domain is not difficult domain.”

## 3. Algorithm consistency

- DT-PGNA changes input embedding views using per-example target-task logit sensitivity projected into the exact two-task nullspace of the other task's logit sensitivity.
- TGA-Pair removes the projection but keeps pair construction/cap.
- TGA-Example removes pair construction/cap and uses per-example target BCE sensitivity.
- Matched random-direction and isotropic Gaussian controls isolate whether the target direction and projection add value.
- No new trainable model parameters, expert/gate/fusion changes, optimizer recurrence changes or inference-time operations are introduced.
- The exact two-task projection is authoritative; no lambda_J search remains.
- First-order output preservation is not claimed as exact nonlinear preservation; actual post-forward drift is measured.

## 4. Experiment consistency

- Development/HPO seeds 2027–2031 and formal confirmation seeds 2022–2026 are disjoint.
- HPO is validation-only. Test remains sealed until the formal configuration and checkpoint-selection rule are frozen.
- Formal training retains MovieLens and KuaiRand, M3oE fixed1, natural sampling, 50 epochs, inherited optimizer/batch/data protocol and five formal seeds.
- Validation selection uses the earliest strict maximum of six-cell Macro AUC.
- Published models alone are formal baselines. Internal EMA and candidate routes are internal controls.
- Formal AUC gain and six-cell protection gates match the current main rules. LogLoss is separate.

## 5. Implementation and operational checks required of Codex

- Verify exact active entry point, data hashes, model/optimizer/EMA lifecycle and runtime versions before editing.
- Confirm per-example gradient extraction is valid under the actual device/distribution wrapper.
- Verify embedding hook cleanup, no in-place clean-tensor mutation, detached first-order perturbation and exactly one optimizer/EMA update per batch.
- Run finite-difference, zero-gradient, rank-one projection, nonlinear drift, clean-path equivalence and resume-determinism tests.
- Profile actual wall-time and peak memory; do not lower the training budget to conceal added cost.
- Preserve NOT_TRIGGERED, ENGINEERING_BLOCKED and scientific NO_GO as distinct statuses.

## 6. Remaining decision-relevant risks

1. DT-PGNA may be too expensive at the full batch size.
2. Nonlinear forward drift may be larger than first-order projection predicts.
3. The projection may not add AUC beyond pair-directed TGA or pairless TGA-Example.
4. Adversarial embedding views may improve AUC while worsening LogLoss.

These risks are directly tested by the existing staged plan. No additional speculative validation matrix is needed before implementation.

## 7. Final decision

The plan is sufficiently specified for Codex to begin with a read-only source audit and correctness implementation. It is not evidence that the method has been implemented, trained, or shown to improve performance. The primary research route remains DT-PGNA, with TGA-Pair and TGA-Example as decisive alternatives/controls. Prefer the simplest route that demonstrates an independent five-seed AUC increment under the inherited evaluation rules.
