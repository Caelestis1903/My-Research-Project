# 16 — Current-Main Audit, Algorithm Refinement and Codex Execution Addendum

**Authority:** This addendum governs this current task's scope and execution. Where it conflicts with older plan text, follow this addendum and experiment_manifest_v3.json.  
**Audit snapshot:** main@bcb7e2e61d1524a7efab2cb059f47fe042ec16b7 (2026-11-15).

## A. Main-only scope

The current task read only main-branch files. No remote branch enumeration or branch checkout was performed. The canonical current audit is 01_REPOSITORY_AUDIT.md. The old branch inventory in 15_CRITICAL_REVIEW_AND_REFINEMENT.md is a historical document already present in main; it was not revalidated or extended during this task. Do not retrieve branch-only sources during Codex execution unless the user separately authorizes it.

Main's latest context is the 2026-11-15 handoff in EXP_AI_CONTEXT.md. The main context records the 11/09 PUB transfer, 11/10 EMA causal attribution, and 11/11 EMA next-phase outcomes. These are accepted context, not new work to repeat:
- fixed EMA remains a strong internal reference; its tested AUC gain is mainly carried by shared parameters, especially EMBED;
- current PUB P0/P1/P2 migration did not beat normal Adam/F under its tested settings;
- A_MATCH feedback harmed F in both datasets; no general claim about every feedback controller follows;
- two-state B is a prediction-time historical readout, not a training optimizer; its incremental AUC over F is near zero/negative in primary settings, with a small KuaiRand diagnostic retained separately.
Do not relaunch these EMA/PUB directions as the main contribution.

## B. Confirmed implementation semantics relevant to DT-PGNA

The current fixed1 model code is vendored under experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/. The wrapper captures the EmbeddingLayer output with a temporary forward hook. The reference model computes all domain-task towers, then selects the relevant tower output by the sample's domain. Tower outputs before sigmoid are therefore available through the existing forward-hook pattern used by boundary_sampler.py.

The reference graph uses per-row LayerNorm, Linear, ReLU and expert/tower operations; the inspected model contains no BatchNorm or Dropout. This supports row-wise gradient extraction from logits.sum(), but Codex must still test:
1. gradient from sum of row logits equals stacked per-row gradients within tolerance;
2. a row's logit gradient is unchanged when unrelated rows are added to the batch;
3. the exact current training wrapper and device/DataParallel setup preserve this property;
4. finite-difference output changes match the Jacobian prediction for small perturbations.

The current model wrapper already uses an embedding forward hook for Gaussian perturbation. Add only a narrow, optional embedding-delta override or an isolated context-manager hook for the new experiment. Do not modify the reference model architecture, experts, gates, fusion or towers. Hook cleanup must occur in finally blocks; no in-place modification of the clean embedding; the perturbation must be detached for the first-order method.

## C. Final route hierarchy

Keep the refined hierarchy from 15_CRITICAL_REVIEW_AND_REFINEMENT.md and manifest_v2, not the earlier two-route-only manifest:
- **A — DT-PGNA:** pair-directed target-logit perturbation projected into the exact two-task nullspace of the other task's logit sensitivity. Primary candidate.
- **B — TGA-Pair:** same target positive/negative pair selection, perturbation radius and target-only BCE, but no cross-task projection. Projection-ablation control.
- **C — TGA-Example:** per-example target-BCE adversarial embedding perturbation, without positive/negative pair construction or pair-margin cap. Simpler standard alternative and decisive comparator.
- Matched random-direction and isotropic Gaussian controls are required to determine whether target sensitivity or projection adds value.
- Label-matched PAIR-only ranking, generic PCGrad/ECCM, gradient magnitude balancing and adaptive EMA are excluded as new routes.

For the two-task case, use the exact rank-one projection:
\[
q_t=g_t-\frac{g_o^\top g_t}{g_o^\top g_o}g_o
\]
when the other-task gradient norm exceeds the frozen numerical threshold; otherwise q_t=g_t. This removes the unnecessary damping hyperparameter. Skip near-zero q rather than normalizing numerical noise. Record the residual g_o^T q_t and finite-difference drift.

For pair-directed routes, use the fully specified first-order pair cap in 15_CRITICAL_REVIEW_AND_REFINEMENT.md. Do not invent a kappa or margin floor. Skip near-zero clean margin or predicted margin change; report skip rates. The cap does not guarantee nonlinear output preservation, so measure actual non-target-logit drift after the augmented forward.

## D. Separate HPO from formal confirmation

The earlier plan reused seeds 2022–2026 for pilot/HPO and then described a later confirmation stage. To satisfy the requirement that HPO and formal confirmation be distinct, use:

- **Development/HPO seeds:** 2027, 2028, 2029, 2030, 2031.
- **Formal confirmation seeds:** 2022, 2023, 2024, 2025, 2026.

Stage 1 pilot and Stage 2 HPO use only development seeds and validation metrics. They must not read Test. Stage 3 uses the untouched formal seed pool, freezes all route/config/code/data identities, and evaluates Test only after the validation checkpoint-selection protocol is complete. If a clean baseline on development seeds is required for paired validation selection, train it under the same fixed protocol. For formal confirmation, reuse existing 2022–2026 published-model endpoints only if model source, data split, preprocessing, model configuration, seed and selection protocol match exactly; otherwise run a matched baseline.

The development seed pool is not used to claim formal five-seed Test performance. The formal 2022–2026 pool is not used to select HPO values. HPO must use five seeds; no single-seed or two-seed performance rejection.

## E. Binding staged execution

1. **Stage 0 — read-only source and semantic audit:** inspect active training entry point, current main SHA, data/config hashes, batch ordering, domain/task mapping, fixed1 output and EMA path, checkpoint selection, GPU/RAM lifecycle. No training code edits until source map is reviewed.
2. **Stage 1 — correctness and GPU smoke:** CPU unit tests plus one real-batch smoke per dataset. Smoke checks only engineering/numerical correctness; no scientific inference.
3. **Stage 2 — five-seed development pilot:** both datasets, seeds 2027–2031; arms clean, TGA-Pair, DT-PGNA, TGA-Example, matched random and matched Gaussian. Test sealed. Record validation six-cell Macro AUC, six cells, actual non-target drift, target margin change, skip rate, throughput and peak memory. Do not gate on gradient cosine or a learned proxy.
4. **Stage 3 — bounded HPO:** validation only, development seeds 2027–2031. Search alpha in {0.01, 0.03, 0.05} for at most the two most promising routes; K=128 and lambda_aug=0.25 fixed initially. Only change K or lambda_aug if pilot evidence identifies a concrete cost/scale issue and the small search is pre-registered. No lambda_J because the exact two-task projection removes it.
5. **Stage 4 — formal five-seed confirmation:** freeze route, alpha, pair policy, cap, objective normalization, code, data and checkpoint selection. Run selected route and decisive controls on seeds 2022–2026. Keep clean online and fixed-EMA readouts. No changes after freeze.
6. **Stage 5 — Test and report:** after freeze only, evaluate Test. Report each of six AUC/LogLoss cells, Macro AUC/LogLoss, Traffic and Pooled metrics, mean/SD/paired 95% CI/positive seed count, runtime and memory.
7. **Stage 6 — independent review:** verify all identities and decision labels. Update the project metric ledger only after the experiment is actually executed; this planning task must not change it.

## F. Decision semantics

- A stage not run is NOT_TRIGGERED, not a failed algorithm.
- Smoke failure is ENGINEERING_BLOCKED, not scientific negative evidence.
- Five-seed development validation can select a route/configuration for formal confirmation, but it is not a Test rejection and cannot be reported as final performance.
- A route is not rejected because another route fails. A route not run remains NOT_TRIGGERED.
- AUC is primary. The formal gate is mean paired Test Macro AUC gain >= +0.003 against the strongest applicable published model and all six cell mean AUC differences >= -0.001. Do not require all seeds positive.
- Report paired mean, sample SD, paired 95% CI and positive seed count. The five seeds are the independent units.
- LogLoss is independent: report its paired mean and six cells. It cannot substitute for AUC.
- For every candidate with EMA, report total gain vs ordinary online and the incremental component gain vs clean fixed EMA separately.
- Do not use domain size as a difficulty proxy. Do not design or analyze any experiment to prove or disprove “small domain is not difficult domain.”

## G. Final scientific decision criterion

DT-PGNA is worth promoting only if it provides an independent ranking increment over both TGA-Pair and TGA-Example at matched perturbation scale, while keeping actual non-target drift acceptably small and satisfying the six-cell AUC protection rule. If TGA-Example is equally effective, prefer the simpler method and describe its weaker method-level novelty honestly. If none of the routes adds AUC over matched Gaussian/random controls, stop this specific family rather than expanding into generic gradient surgery or EMA controllers.

The plan remains PLANNING_ONLY. This addendum authorizes no implementation and no training.
