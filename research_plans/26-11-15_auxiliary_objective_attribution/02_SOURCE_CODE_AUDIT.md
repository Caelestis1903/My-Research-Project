# Source Code and Repository Audit

## 1. Repository state at planning time

- Repository: Caelestis1903/My-Research-Project.
- Main HEAD at audit: 79fbd441c6c87cff30838c65bf97ad6ec7f45abb.
- Remote branch 26-11-15 exists and is exactly two commits ahead of main, zero commits behind; it is not merged.
- Main did not contain research_plans/26-11-15_auxiliary_objective_attribution/ at audit time.
- Do not merge branch 26-11-15 in this task. Read it as the source of the completed pilot, but write only the new plan to main.
- Main governing context: AGENT.md, EXP_AI_CONTEXT.md, experiments/Experimental_Global_Summary.md, Experimental_Stable_Findings.md, Experimental_Open_Questions.md, Experimental_Experiment_Index.md, Experimental_Evidence_Ledger.md, MDMT_ALGORITHM_METRICS_LEDGER.md, PROJECT_ARCHITECTURE.md.
- Existing plan: research_plans/26-11-14_heterogeneity_aware_training_optimization/, especially 15_CRITICAL_REVIEW_AND_REFINEMENT.md, 16_CURRENT_MAIN_AUDIT_AND_EXECUTION_REFINEMENT.md, 17_FINAL_CONSISTENCY_REVIEW.md and experiment_manifest_v3.json.

## 2. Pilot source map

Branch 26-11-15 files directly inspected:
- research_reports/2026-11-15_heterogeneity_aware/REPORT.md
- RESULT_TABLES.md
- NUMERICAL_APPENDIX.md
- DESIGN_SPECIFICATION.md
- SOURCE_MAP.md
- IMPLEMENTATION_REVIEW.md
- FINAL_REVIEW.md
- COMPLETION.json
- artifacts/heterogeneity_aware_20261115/attempt1/PILOT_DECISION.json
- experiments/heterogeneity_aware_20261115/core.py
- protocol.py
- run.py
- selection.py
- tables.py

The report's accepted training source hash is a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491; implementation commit bf36e454adc2d0c569218a7993834f6d56cfc97f. The branch completion file records 60 trajectories, 3000 epochs, 103500 updates, no Test access and all later phases NOT_TRIGGERED.

## 3. Source semantics confirmed from code

core.py:
- sample() deterministically samples pair rows or example rows from domain/task cells.
- Pair route selects k=min(128,n_positive,n_negative) positive and negative rows without replacement.
- Example route samples 2k rows uniformly from the domain, where k is based on available positive/negative counts; this approximately retains the natural label mixture in expectation, not exact finite-batch label proportions.
- sensitivity() differentiates each selected task logit with respect to the inserted embedding.
- A projects target logit sensitivity against the other task's logit sensitivity; B omits projection.
- C uses (sigmoid(z)-y) * grad(logit) as the BCE gradient direction.
- Pair route has margin cap and skip rules; A also checks non-target-logit drift and backtracks.
- The generated delta is detached before the augmented forward.
- L_aug=2*mean(target-task probability BCE) on accepted augmented rows.
- L=L_clean + lambda_aug*L_aug; one backward, one Adam step and one after-Adam EMA update per batch.

protocol.py:
- Seeds exactly 2022–2026.
- MovieLens and KuaiRand.
- Batch 40960, 50 epochs, lr=.01, Adam betas(.9,.999), eps=1e-8, coupled weight decay ML=1e-4 / KR=1e-5.
- Fixed EMA rho ML=.9828205985452511 / KR=.995.
- The 26-11-15 branch intentionally overrode the older 26-11-14 plan's 2027–2031 development pool and reused 2022–2026 for pilot; this task's seed constraint continues to govern every stage.

selection.py:
- Validation selected fixed-EMA Macro AUC used the five seeds.
- The original pilot continuation gate compared each route with CLEAN, RANDOM and GAUSSIAN.
- The branch's gate did not trigger HPO, confirmation or Test.

## 4. Important loss-weight derivation Codex must reproduce

Let batch size be B rows and let M be the number of accepted auxiliary row-task entries in one step. Then:

L_clean = (1/(2B)) * sum over all i,t of ell_clean(i,t)

L_aug = 2 * (1/M) * sum over accepted auxiliary entries j of ell_aug(j)

L_total = L_clean + 0.25 * L_aug

Therefore the total scalar coefficient on the auxiliary mean is 0.5, and each selected auxiliary entry has coefficient 0.5/M. Its coefficient relative to one clean row-task entry is B/M. This is not equivalent to assigning 25% extra weight to each selected sample. The realized M, domain-task counts, label counts and repeated row IDs across target tasks must be logged per batch and aggregated per epoch/seed.

If M is zero, the auxiliary term is zero and the objective must exactly equal clean loss.

## 5. Identity and reuse matrix requirements

Before reusing any 26-11-15 result, compare:
- source commit and complete source hash;
- data raw SHA256, processed arrays and row-order hashes;
- preprocessing and split identities;
- model factory/config and fixed1 coefficients;
- optimizer hyperparameters, coupled weight decay, batch size, epoch count;
- fixed EMA initialization, decay and update timing;
- RNG setup, loader order, sampler implementation and task/domain mapping;
- objective definition and probability-BCE implementation;
- validation selection (earliest strict max of six-cell Macro AUC);
- metric calculation and software environment.

Expected exact reuse candidates:
- CLEAN trajectories for shared baseline references;
- B trajectories for the perturbed pair route;
- C trajectories for the perturbed example route.

A/B/C can only be reused as source results, not silently mixed with new implementations. If the zero-control sampler/replay semantics cannot reconstruct exact support and loss weights from stored batch logs, rerun the necessary paired arms. Do not assume that report-level metrics are sufficient to establish trajectory identity.

## 6. Main integration path to inspect at implementation time

The pilot source imports model/data/protocol utilities from experiments/conditional_information_momentum_20261102_v3/.
- fixed1 model factory: vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/model.py::make_model
- forward reference: corresponding reference.py; embedding output is [batch, embedding_dim]
- tower output ordering: task*3+domain
- optimizer and EMA: vendor/ema02/protocol.py
- data protocol and hashes: conditional_information_momentum_20261102_v3/protocol.py and imported audited_data.py
- source clean objective: sample-mean probability BCE over all rows and both tasks
- model gate input is detached during ordinary training. Sensitivity probe reconnects the gate path only to calculate the numerical forward Jacobian; do not change training detach semantics.

Codex must re-read exact current main paths and compare them with the branch's imported utilities before coding. Do not copy old source into a new model or modify vendor implementation.
