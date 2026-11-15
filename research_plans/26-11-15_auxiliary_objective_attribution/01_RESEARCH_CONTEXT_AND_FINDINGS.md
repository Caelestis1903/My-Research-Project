# Research Context and Accepted Findings

## 1. Research question

The current line is heterogeneity-aware training optimization in multi-domain, multi-task recommendation. The immediate question is not whether adversarial perturbations can be generated; the 2026-11-15 code shows that they can. The question is whether the training objective used to learn from those views is correctly aligned with the natural recommendation objective, and whether perturbations provide an independent ranking benefit after objective effects are controlled.

## 2. 2026-11-15 pilot: accepted facts

The remote branch 26-11-15 contains a completed validation pilot:
- MovieLens and KuaiRand; CLEAN, A, B, C, RANDOM and GAUSSIAN; five seeds 2022–2026; 50 epochs each.
- 60 complete trajectories, 3000 epochs, 103500 Adam updates and 103500 EMA updates; zero failed or missing runs.
- Test was not accessed. HPO, formal confirmation and conditional ablations were NOT_TRIGGERED.
- A/B/C are DT-PGNA, TGA-Pair and TGA-Example respectively.
- Relative to CLEAN fixed EMA selected validation Macro AUC: MovieLens A -0.001436, B -0.001353, C -0.001643; KuaiRand A -0.022282, B -0.021557, C -0.009188. Each of the six dataset-route comparisons was 0/5 positive.
- On KuaiRand C exceeded RANDOM by +0.009541 and GAUSSIAN by +0.011498, each 5/5 positive, but remained below CLEAN fixed EMA. This is a conditional control comparison, not proof of an independent task-gradient benefit because sample support and radius policy were not identical.
- MovieLens C improved Macro LogLoss by 0.004180 while reducing AUC by 0.001643. This is evidence that probability-loss and ranking outcomes can diverge.
- A's projection numerically controlled non-target logit drift (accepted-row drift <=.001), but did not show AUC improvement over B. Projection residual was tiny; the mechanism operated, but its numerical property did not translate into ranking gain.
- A's projected direction norm was about 36% of original on MovieLens and 16% on KuaiRand. This is a plausible design cost, not by itself proof that projection caused the AUC loss.
- Early/late AUC effects reversed in KuaiRand; A/B were positive early and negative in later stages. Selected checkpoint results remained negative.
- All A/B/C, RANDOM and GAUSSIAN include an auxiliary objective. The current study has not yet observed those perturbation arms without their auxiliary loss.

## 3. Why the zero-perturbation experiment is decisive

The current objective is:

L_total = L_clean + 0.25 * L_aug

The actual source semantics are:
- L_clean = mean(BCE_probability(model(x), y)) over every row and both tasks in the natural batch.
- A/B sample up to 128 positive-negative pairs in each of six domain-task cells; auxiliary loss is twice the mean target-task BCE over accepted pair members.
- C samples up to 256 rows per domain-task cell from the domain population, without explicit positive-negative pairing; its target-task BCE is also multiplied by two.
- A/B/C optimize target-task BCE on the perturbed views only; no augmented loss is applied to the other task.
- The accepted row support can differ because A/B have pair-validity, margin and (for A) non-target-drift rejection, while C has its own near-zero-gradient/RMS skip rule.
- The auxiliary branch changes the loss composition and weighting, not just the input representation.

With 6 cells × up to 256 rows, the selected auxiliary set can be a small fraction of the batch while its aggregate coefficient is 0.5 times a selected-row mean. Thus lambda=.25 does not mean “the auxiliary objective contributes only 25% of the natural per-example training weight.” Exact effective weights depend on per-batch selected counts and cell composition and must be computed from the actual accepted row indices.

## 4. Historical findings that constrain novelty and repetition

- Fixed EMA has repeatable five-seed AUC gains in the repository and is retained as a strong internal reference; this plan must not attribute those gains to the new method.
- Prior PAIR/ECCM already explored domain/task-conditioned positive-negative sampling and label conditions. Do not relabel it.
- Prior GATS and gradient experiments already explored parameter-gradient transformations, projection and magnitude adjustment. This plan acts on input-embedding sensitivity and auxiliary objective semantics, not parameter-gradient surgery.
- Gaussian augmentation has dataset-conditional outcomes and must be a matched control, not a universal baseline.
- Dynamic EMA, PUB migration and generic gradient balancing have not produced a stable independent AUC route under the tested settings; they are outside this plan.
- 11/10 EMA attribution found fixed EMA gains primarily associated with shared parameters, especially embedding; this is background context only.
- Main branch 26-11-14 plan proposed A/B/C task-sensitive augmentation. The 26-11-15 branch implemented and tested it; the present plan is a new, narrower follow-up prompted by the report's own unresolved confounding.

## 5. Evidence levels and status discipline

| Statement | Status |
|---|---|
| Current A/B/C configurations underperform CLEAN fixed EMA on both datasets in selected validation Macro AUC | STRONG, five-seed validation evidence |
| Projection achieves its local non-target-logit drift constraint | STRONG numerical mechanism evidence |
| Auxiliary label/domain weighting caused most of the AUC degradation | COMPETING EXPLANATION, not yet isolated |
| Task-sensitive perturbation has no possible ranking value | NOT ESTABLISHED |
| Test performance of A/B/C | NOT MEASURED |
| HPO values alpha=.01/.05 | NOT RUN |
| Zero-delta auxiliary objective | NOT RUN |
| Natural-distribution-preserving task augmentation | NOT RUN |

## 6. Research boundary

The plan retains the fixed boundary that small domain is not difficult domain. No experiment, hypothesis or analysis is designed to prove or disprove it. Domain counts and label rates may be reported descriptively as properties of the natural population, not as difficulty labels.


## 7. Historical research evidence table

| Historical line | Intervention and evidence | Accepted conclusion | Constraint for this plan |
|---|---|---|---|
| PAIR/ECCM (11/02) | Within-domain target-task positive/negative boundary sampling; limited early two-seed comparisons and later supplements | Label-conditioned pair construction is already explored; small-sample outcomes remain conditional | Do not rename label matching or ECCM as a new contribution |
| Conditional information / label donor (11/03) | Cross-task label donor and mixed conditional sampling; early signals followed by five-seed completion | Complete-stack gains included fixed EMA; independent component gains over fixed EMA were not uniform | Any new component must show increment over clean fixed EMA |
| Five-seed completion (11/04) | ML/KR full five-seed comparisons for C/H/P01 and Gaussian combinations | Some full systems beat ordinary online; that does not identify the incremental contribution of each component | Separate total system gain from component gain |
| Gaussian domain coverage (11/04) | Different domain coverage under Gaussian augmentation | MovieLens Gaussian+EMA can beat published M3oE; KuaiRand coverage variants did not beat strongest published model | Gaussian is a dataset-conditional matched control |
| GATS / parameter-gradient work | Parameter-space gradient transformations, projection and magnitude adjustment | Related gradient operations are not unexplored; generic gradient balancing did not establish a stable new route | Current plan intervenes on input embedding views and objective weighting, not parameter-gradient surgery |
| Adaptive EMA / PUB (11/09) | Multiple adaptive EMA and PUB-transfer variants | Tested variants did not establish a stable independent ranking improvement under their tested protocols | Do not reopen these lines |
| Fixed EMA attribution (11/10) | Parameter-group and historical-window attribution | Shared parameters, especially embedding, carry most of the measured fixed-EMA AUC gain; this is a background control result | Keep fixed EMA fixed; do not attribute its benefit to this proposal |
| EMA feedback/readout (11/11) | Multi-step feedback and two-state historical readout | Single-step local signal did not reliably transfer to repeated feedback; prediction-time readout gains were small/conditional | Avoid another proxy-to-action mechanism without direct AUC value |
| Task-sensitive embedding views (11/15) | DT-PGNA, TGA-Pair, TGA-Example with target-only auxiliary BCE | Current alpha=.03/K=128/lambda=.25 configurations had no validation continuation signal; auxiliary-objective effect remains confounded with perturbation | First run Z-PAIR and Z-EXAMPLE |

## 8. Directly relevant literature and novelty boundary

The existing 26-11-14 critical review identifies the following neighbors; they should be retained in Codex's literature note rather than rediscovered or described as absent:
- CROSSGRAD (ICLR 2018): domain-guided cross-gradient input perturbation. It establishes gradient-guided input augmentation as a known mechanism.
- Rep-MTL (ICCV 2025): representation-level task saliency and sample-wise cross-task saliency alignment. It is a close multi-task representation optimization neighbor, but not the same per-row target-logit Jacobian perturbation with a non-target nullspace constraint.
- DTME-MTL (ICCV 2025): token-space gradient conflict analysis and token manipulation. It establishes active representation-space conflict handling in MTL.
- SALARec (Expert Systems with Applications, 2026): preference-aware adversarial item-embedding perturbation for sequential recommendation. Adversarial embedding augmentation in recommendation is established, though its task/objective differs.
- SAF, M3oE, MMFI and PMTRec remain relevant MDMT/multi-task recommendation context in the main literature records.

The potential distinction is not "first gradient-guided augmentation" or "first adversarial embedding perturbation." The candidate distinction is a carefully normalized task-specific view objective under natural row/task weights, with direct measurement of cross-task effects. Similar prior work narrows the claim but does not automatically invalidate a useful MDMTRec application. If the resulting method is simply ordinary FGSM plus a BCE weight, describe it accurately and do not overclaim method novelty.

## 9. Repository context precedence

The 26-11-14 manifest contains a 2027–2031 development/HPO pool. That is superseded for this task by the explicit higher-priority requirement: all performance comparisons and HPO use exactly 2022–2026. Reusing those seeds does not constitute independent unseen-seed confirmation.

The 26-11-15 branch's report is a validation pilot. Its "NO_CONTINUATION_SIGNAL" means the predeclared validation continuation gate did not trigger; it is not a Test-based formal rejection.
