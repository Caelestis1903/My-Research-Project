# Cycle02 literature and novelty audit

Primary-source web check: 2026-11-16. The recommendation is a falsification experiment using established ranking-loss ideas, not a claim to have invented pairwise AUC optimization. A repository-local difference cannot establish global novelty.

## 1. Primary literature and what is actually supported

| Primary source | Relevant result | Boundary for this design |
|---|---|---|
| Yiming Ying, Longyin Wen, Siwei Lyu, *Stochastic Online AUC Maximization*, NeurIPS2016, [paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/c52f1bd66cc19d05628bd8bf27af3ad6-Paper.pdf), especially Eq.(3), Theorem1/Eq.(6–7) | A squared opposite-label ranking risk can be expressed using score moments and a saddle formulation, avoiding explicit storage of all pairs. | NQR uses exact minibatch moments and ordinary Adam, with probabilities and a natural pair-completion/BCE mixture. It is not SOLAM; no convex convergence guarantee transfers to M3oE. The moment reduction is established algebra, not our novelty. |
| Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang, *Large-scale Robust Deep AUC Maximization*, ICCV2021, [author preprint](https://arxiv.org/abs/2012.03173), [full paper](https://arxiv.org/pdf/2012.03173), Sections3.2–3.3 | The paper analyzes problems of AUC square loss and proposes a margin-based alternative. Its decomposition separates within-class score variance from mean separation. It identifies adverse gradients on sufficiently separated unbounded scores and sensitivity to noisy labels. | NQR's bounded probabilities exclude a gap greater than1, but noise, saturation and variance compression remain concerns. The paper supplies objections and lineage, not evidence NQR will work on recommendation. No PESG/min-max optimizer, class resampling, margin tuning or medical-dataset performance claim is imported. |
| Wei Gao, Zhi-Hua Zhou, *On the Consistency of AUC Pairwise Optimization*, IJCAI2015, [author-hosted paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/ijcai15auc.pdf), [proceedings](https://ijcai.org/proceedings/2015/137) | Calibration is necessary but insufficient in general; the paper proves consistency results for specific surrogate families and conditions. | Being a pairwise objective is not a consistency proof. This natural-weighted, BCE-mixed, bounded-score, finite-batch M3oE objective is not covered by a guarantee established here. In particular, do not substitute a theorem about square hinge or realizable data for a theorem about this mixture. |
| Takeru Miyato et al., *Virtual Adversarial Training*, [primary paper](https://arxiv.org/abs/1704.03976) | Prediction-distribution divergence defines local smoothness and a label-independent adversarial direction. | Candidate B changes the divergence being regularized; it is not simply a smaller NDP radius. It remains an established VAT-family approach and has no demonstrated increment in this protocol. |
| Pierre Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR2021, [primary paper](https://arxiv.org/abs/2010.01412), [full paper](https://arxiv.org/pdf/2010.01412), Section2 | SAM optimizes loss in a parameter neighborhood through a tractable gradient approximation. | Candidate C moves parameters temporarily, not inputs, scores, or history. Its published empirical successes are not M3oE evidence and do not justify selecting a radius from current validation outcomes. |

Access note: the direct CVF PDF returned403 and OpenReview requested a browser challenge. Author preprints were used; the Gao/Zhou author PDF and NeurIPS paper were accessible. The initial IJCAI capitalized PDF URL failed, so the author PDF/proceedings URL above replaces it. No review, blog, search snippet, or secondary summary is used as authority for a technical claim. Source-result dates are not treated as paper dates.

## 2. Exact local comparison with historical PAIR/ECCM

Read sources in the current frozen dependency tree:

- `experiments/conditional_information_momentum_20261102_v3/objectives.py`, SHA256 `9794b3aa5f10370c4b09d9bb040f925334ea47a591516ef0fe3e4caee8ac51d0`.
- `experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py`, SHA256 `278d22da232199922f3b70e4735966c61c09d5d4ed800dcf78ed8ffe2e5164eb`.

The historical objective takes one natural model forward, obtains routed logits, selects one valid (domain,task,positive joint-label group,negative joint-label group) condition, and samples2048 positive/negative pairs with replacement. Its auxiliary is mean softplus(−(z_positive−z_negative)); the scalar is `C + lambda_pair*(w_selected/q_selected)*auxiliary`. w includes natural domain mass and conditional joint-label proportions within the two classes. The proposal distribution mixes uniform, w, and demand-weighted terms. Some arms add demand/history machinery; even static ordinary-Adam arms differ from NQR. Historical lambda values include .05 and .2. Zero lambda delegates to source ordinary training.

NQR instead uses all natural rows and all conceptual within-domain ordered pairs, with same-label BCE completion and opportunity mass2*pi_positive*pi_negative. Its score is a probability, surrogate is squared difference to margin1, moments are computed exactly, and the new risk **replaces** .25 of BCE in a convex mixture. There is no selected joint-label condition, pair RNG, importance factor w/q, persistent demand, separate history, or pair-specific optimizer. These are exact mathematical and procedural differences, not a renamed PAIR arm.

This audit does not show which difference causes any historical/new performance difference: too many factors differ. Historical PAIR is therefore context, not a matched C2 mechanism control. The new BRIER and MEAN-GAP controls hold the currently selected opportunity measure and mixture fixed. If the formal designer finds an additional historical source with the same complete objective, it must report that redundancy and return to fresh research; it must not change a coefficient merely to manufacture a distinction.

## 3. Exact local comparison with DT-PGNA/TGA, NDP, TV and C1

The C1 DESIGN_COMPLETENESS_ADDENDUM identifies the original pilot implementation commit `bf36e454adc2d0c569218a7993834f6d56cfc97f`, aggregate `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`, and exact source receipts. It established local non-equivalence between selected one-sided target-entry extra BCE and full-support even views. This proof must be preserved; it is not a global novelty certificate for either cycle.

| Historical method | Actual changed quantity | Why NQR is different |
|---|---|---|
| DT-PGNA/TGA-Pair | Selected opposite-label examples, logit-Jacobian displacement, projection/caps/acceptance, extra target BCE mass1.5 | NQR changes a score-space loss on original inputs; no Jacobian, accepted subset, displaced embedding or extra-BCE coefficient mass |
| TGA-Example / Stage1 zero-view controls | Budgeted selected entries; BCE ascent or zero displacement, still added target BCE | NQR has natural pair opportunities, no selected budget, fixed replacement mixture |
| NDP-A | .75C+.25 natural full-support target BCE at detached one-sided input displacement | NQR uses original p only; no displacement or new view population. Natural support alone does not make the objectives equal |
| TV-GRM | Adam clean moments plus separately accumulated/preconditioned residual view-gradient channel | NQR feeds one scalar data objective to ordinary Adam with no independent q/history or injected update |
| AXIS-EVEN / AXIS-IID2 / ISOTROPIC-EVEN | Natural BCE at paired or sampled signs, controlled input axes | NQR has no axis/sign/noise variable and no same-input perturbed replica. Its quadratic score gap is not a second-order Taylor approximation to the input-view objective |

The current worker's79-file manifest reproduces `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. The original `source_receipt.json` still records the earlier RESOURCE_1 identity; `RESOURCE_1_SUPERSESSION.json` and current file hashes establish the accepted identity. Do not overwrite old receipts to make them look uniform.

## 4. Novelty verdict and permissible wording

**Global novelty: NOT ESTABLISHED; no claim authorized.** Pairwise square losses, efficient score-moment reductions, mixing supervised objectives, prediction smoothing and parameter-neighborhood minimization all have clear prior art. A finite primary-source search is not an exhaustive patent/literature clearance.

**Local non-equivalence: supported for the audited neighboring implementations.** The exact natural pair completion and its two controls offer a new experiment in this frozen research sequence. The possible contribution is a carefully identified empirical result about this protocol and decomposition, including a negative result. The strongest admissible pre-run statement is: “We will test a fixed probability-ranking mixture and matched scalar-objective controls against CLEAN selected EMA.”

No formal published-model baseline is added. CLEAN, NDP and the new ablations are internal source-matched controls, not published model names. Published M3oE/HyperGate/AutoIFS comparisons from history remain under their original data/selection/Test protocols and are not transplanted into this validation screen.
