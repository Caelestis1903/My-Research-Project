# Critical Review and Binding Refinements

**Status:** ADDITIVE REVIEW / PLANNING_ONLY / NOT_IMPLEMENTED / NOT_RUN  
**Applies to:** this directory's original runbook, route specifications, experiment matrix, and manifest  
**Rule:** This file is an append-only refinement. It does not alter historical reports or replace the original documents. If any original planning statement conflicts with this file, the more specific rules below govern the Codex execution plan.

## 1. Repository snapshot and audit reconciliation

The original plan directory was already present on `main` before this review. Its initial planning commit is `bcb8758b5c3ee88311294a1597662e3ed596fa44` (2026-11-14 08:52:13 UTC). Subsequent commits corrected EMA attribution, verified adjacent literature, added audit filename pointers, and sharpened the DTME distinction. The current reviewed `main` HEAD is `c27d7f91d210046ccab42d4b31ceaeadcceb96c9` (2026-11-14 09:00:31 UTC). The original audit's base snapshot `01979f6588935f713029b2b5e04557de9b8c153f` is therefore historical, not the current HEAD.

The remote branch inventory contains 46 branches including `main`. The `26-11-11` research branch is 19 commits ahead of and 100 commits behind the current `main`; its full compare response is capped, so the audit uses branch tip metadata plus direct reads of the relevant report and plan sources. The 11-11 report and plan are not fully merged into main.

Main contains `AGENT.md`, not `AGENTS.md`; neither `start.md` nor `AGENTS.md` exists at current main. `AGENTS.md` exists on the 26-11-11 branch, but it is not the current main engineering contract. Codex must read current main `AGENT.md` and must not silently substitute branch-only instructions. The source report may refer to historical `start.md`; that is historical evidence, not a current-main file.

The audit directly examined the main research context and metrics ledger, the relevant PAIR/ECCM, conditional-history, five-seed completion, Gaussian coverage, AUG_HISTORY and EMA reports, the 11-10 EMA attribution report, the 11-11 EMA report/plan, and key implementation files including:
- `experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py`
- `experiments/conditional_information_momentum_20261102_v3/objectives.py`
- `experiments/conditional_information_momentum_20261102_v3/optimizers.py`
- `experiments/gradient_aug_time_select_20261102/objectives.py`
- `experiments/gradient_aug_momentum_20261101/objectives.py`
- `experiments/gradient_aug_aig_m_20261101/objectives.py`
- `experiments/mdmt_history_information_20261103/objectives.py`

This is a targeted source audit of the training-optimization line, not a claim that every legacy binary artifact, every log line, or every source file in all 46 branches was re-read. The historical reports and implementation paths above are the evidence base for this plan. Codex must still inspect the exact current-main fixed1 model implementation before integration.

## 2. Historical conclusions that materially constrain this plan

| Historical line | Direct evidence and interpretation | Consequence for this plan |
|---|---|---|
| PAIR / ECCM, 2026-11-02 v3 | The sampler explicitly forms within-domain, target-task positive/negative boundaries conditioned on the joint two-task label; 2048 pairs are sampled with replacement. P00/P11 completed 24 two-seed runs. The original S2/S3 and full P01/P10 training were NOT_TRIGGERED at that point. | A design based only on requiring the other task's label to match is already covered. Do not revive it under a new name. |
| PAIR P01/P10 supplement | The static/dynamic selection and ECCM factors were separated in a two-seed 2×2. KuaiRand P01−P00 Test Macro AUC was +0.012099 while validation was −0.000101; the P11 interaction was −0.012347 on Test. This is useful interaction evidence, not five-seed component proof. | Do not claim ECCM is independently useful or that the PAIR family was universally disproven. The current proposal must change the mechanism, not only the label condition. |
| 2026-11-03 conditional information / real history | Cross-task label-donor M and C_MIX/H_MIX were actually implemented. Early one-seed KR results were positive, with validation/cell/traffic costs. The later five-seed completion found full-stack gains against ordinary M3oE, but not a clean independent gain over fixed EMA across all candidates. | Cross-task label information and conditional sample construction are not unexplored. A new method must use a different signal and show its own increment over fixed EMA. |
| Five-seed completion | C/H/P01/M complete-stack gains against ordinary M3oE included EMA; ML component gains against fixed EMA were negative, while KR P01 had a cell-protection violation. | Separate complete-stack gains from new-component gains. Fixed EMA remains an internal control. |
| GATS B2 / GATS02 | Five-seed KuaiRand total Macro AUC gain was +0.006097847 vs ordinary training, +0.001481447 vs fixed EMA, and +0.000005156 vs Gaussian. The implementation includes task-gradient projection over parameter-gradient groups and magnitude balancing; its independent projection increment is weak. | Do not rebrand parameter-gradient projection or magnitude balancing. The proposed operation is on input-embedding sensitivities and non-target task logits, which is a different intervention object; compare it directly with existing augmentation controls. |
| Gaussian coverage | MovieLens Gaussian D0 + fixed EMA reached +0.004167 vs M3oE with six-cell protection, with a +0.000931 LogLoss cost. KuaiRand Gaussian coverage variants did not beat HyperGate. | Gaussian is a matched control and a dataset-conditional internal reference, not the proposed method or a universal baseline. |
| AUG_HISTORY and EMA mechanism | Direction-response, grouped timescales and gradient-drift variants did not yield stable independent AUC gains over fixed EMA. Fixed EMA itself has strong five-seed ranking evidence and late-stage effects. | No dynamic EMA or new momentum controller is part of this study. |
| 11-09 PUB transfer | Five-seed migration variants were materially below ordinary Adam; the tested transfer failed under the actual M3oE update scale/weight-decay setting. | Do not reopen a simple PUB coefficient repair as the main route. |
| 11-10/11 EMA research | Shared representation/embedding historical corrections dominate fixed EMA gains; complex prediction signals did not reliably produce action gains. The 11-11 B method is a prediction-time historical function combination, not a training optimizer. Its incremental AUC over F was near zero or negative in primary configurations; the KuaiRand two-state diagnostic was a small separate signal. | EMA is a fixed internal control only. Do not attribute its gain to this proposal or confuse prediction readouts with training optimization. |

Evidence strength must retain the original sample sizes and statuses. In particular, two-seed and one-seed results are preliminary; NOT_TRIGGERED/NOT_RUN/COST_BLOCKED is not a negative performance result.

## 3. Critical algorithmic finding: TGA is close to ordinary task-loss adversarial augmentation

For binary BCE with logit (z_t(x)), probability (p=sigma(z_t(x))), and label (y),
[

abla_e ell_{mathrm{BCE}}(z_t,y)=(p-y)
abla_e z_t.
]
For (y=1), the adversarial ascent direction is opposite to (
abla_e z_t); for (y=0), it is along (
abla_e z_t). After per-example norm normalization, this is exactly the sign convention used by the current TGA positive/negative perturbations, apart from the pair-specific margin cap and sampling policy.

Therefore TGA must be described as a task-conditioned, first-order adversarial embedding-augmentation control—not as an independently novel pairwise-gradient algorithm. The pair itself does not create a new gradient direction. Its possible added value lies only in pair selection and the margin-based radius cap.

**Required additional comparator (Route C / TGA-Example):** use the target task's per-example BCE gradient with respect to the embedding, generate a norm-matched first-order adversarial perturbation for the same number of examples per domain-task cell, and do not construct positive/negative pairs or apply a pair-margin cap. Keep the same perturbation-radius rule, target-only augmented BCE, clean objective, optimizer and EMA. This is the simpler standard alternative that tests whether pair construction/capping is actually useful.

Interpretation:
- A > B at matched norm: evidence for the cross-task output-preservation projection.
- B > C: evidence that pair selection/capping contributes beyond ordinary task-loss adversarial augmentation.
- C ≈ B: pair construction is unnecessary; simplify to C unless A's projection adds a clear independent benefit.
- A ≈ B ≈ C: no evidence that the extra structure earns its complexity; do not promote DT-PGNA.

## 4. Exact two-task projection and removal of an unnecessary hyperparameter

The current manifest uses a damped projection
[
P=I-J^	op(JJ^	op+lambda_J I)^{-1}J.
]
For (lambda_J>0), this is not an exact orthogonal projector, so (Jq) is generally nonzero. The plan already measures this residual, but (lambda_J) is an additional hyperparameter with no defined search.

The current platform has exactly two tasks. Let (g_t=
abla_e z_t) and (g_o=
abla_e z_o), where (o) is the other task. Use the exact rank-one orthogonal projection:
[
q_t=
egin{cases}
g_t-dfrac{g_o^	op g_t}{g_o^	op g_o}g_o,&|g_o|_2^2>	au_{mathrm{zero}},\
g_t,&	ext{otherwise}.
end{cases}
]
This is the projection of target-task sensitivity onto the orthogonal complement of the other-task logit sensitivity. If the other-task sensitivity is numerically zero, there is no local first-order direction to preserve and the target direction is retained. If (q_t) is near zero, skip that sample. For future (T>2), use a numerically stable SVD/pseudoinverse projector, but do not implement that generalization in this two-task study.

This removes (lambda_J) from the search and gives a directly interpretable first-order property (g_o^	op q_tapprox0). Use float64 for the tiny scalar dot-product/denominator calculation if required, then cast the perturbation back to the embedding dtype. Record the residual and compare it with finite differences.

## 5. Make the pair cap executable and auditable

The original specification says to apply a frozen pair-level margin cap but does not provide its exact formula. Codex must not invent one.

For a pair with clean margin (m=z_t(x^+)-z_t(x^-)), raw perturbations
[
delta^+=-alpha,mathrm{RMS}(e^+)rac{q^+}{|q^+|_2+epsilon_0},qquad
delta^-=alpha,mathrm{RMS}(e^-)rac{q^-}{|q^-|_2+epsilon_0},
]
calculate the predicted first-order margin change using the *unprojected target gradients*:
[
Delta m_{mathrm{lin}}=g_t(x^+)^	opdelta^+-g_t(x^-)^	opdelta^-.
]
Set
[
c=minleft(1,rac{|m|}{|Delta m_{mathrm{lin}}|+epsilon_0}ight),quad
delta^+leftarrow cdelta^+,quaddelta^-leftarrow cdelta^-.
]
If (|m|le	au_{mathrm{margin}}) or (|Delta m_{mathrm{lin}}|le	au_{mathrm{zero}}), skip the pair and record it. This cap has no tunable (kappa) or (m_{mathrm{floor}}), and bounds the predicted first-order margin displacement by the current absolute margin, so it does not intentionally cross the pair's current margin sign in the linear approximation. Actual nonlinear margin and non-target-logit drift must still be measured after the forward pass. The tolerances are numerical correctness thresholds, not performance hyperparameters; freeze them from dtype/finite-difference tests before HPO.

## 6. Revised experiment matrix and stage gates

Keep the fixed platform, natural data proportions, 50-epoch budget, seeds 2022–2026, six-cell validation selection, fixed EMA implementation, and all inherited protocol identities.

### Stage 1 — five-seed diagnostic pilot

Run both datasets and all five seeds for:
- B0: clean online + fixed EMA (reuse only if exact source/data/protocol identity matches);
- B1: pair-directed TGA without projection;
- B2: pair-directed DT-PGNA with exact two-task projection;
- B3: TGA-Example, per-example target-BCE adversarial perturbation without pair construction;
- B4: matched random direction;
- B5: matched isotropic Gaussian perturbation.

Reuse Gaussian D0 only when the source identity, data, training budget, and checkpoint protocol match exactly. Otherwise retrain the matched control. Keep the test set sealed.

The Stage 1 gate is based on actual five-seed validation Macro AUC and six-cell validation differences, together with correctness, non-target drift and measured cost. **Do not gate on a proxy ranking-risk/difficulty predictor or on gradient statistics alone.** Prior EMA research demonstrated that better proxy prediction need not produce action value. Record all diagnostics, but they do not replace validation AUC.

If no route shows a useful validation signal, stop HPO and record the next phase as NOT_TRIGGERED; do not describe this as a five-seed Test rejection. If a route has a useful validation signal but misses a six-cell protection cell, retain the exact cell-level trade-off for the decision rather than discarding the whole research direction.

### Stage 2 — lightweight HPO

Only if Stage 1 provides a validation-based reason to continue, search (alphain{0.01,0.03,0.05}) for the best two of B1/B2/B3. Use all five seeds and both datasets. Do not search (lambda_J), because the exact two-task projection removed it. Keep the pair budget (K=128) and (lambda_{mathrm{aug}}=0.25) fixed initially; change either only if a measured scale/cost issue makes it decision-relevant, and then predeclare a small search.

Select configurations only from validation. Do not use Test to choose route, alpha, cap, or checkpoint.

### Stage 3 — frozen five-seed confirmation

After HPO, freeze the selected route, alpha, pair policy, objective normalization, tolerances, code and data identities. Run five-seed confirmation for the selected candidate and its matched decisive comparator (at minimum B0 and the strongest of B1/B2/B3; include the other route if it is needed to isolate projection or pair effects). Test is evaluated only after all configurations and checkpoint-selection rules are frozen.

### Stage 4 — targeted ablation

Only if B2 (DT-PGNA) shows a positive independent increment over B1 and B3:
- exact projection vs no projection at matched perturbation norm;
- pair-directed vs per-example TGA at matched number of augmented rows;
- actual non-target logit drift vs first-order Jacobian prediction;
- target-only augmented BCE vs no augmented loss, with all else held fixed.

Do not launch a broad factorial search.

## 7. Revised research claim and novelty boundary

The defensible claim is not “gradient-guided augmentation is new,” “pairwise ranking is new,” or “task gradients have not been used in representation space.” The defensible candidate is:

> In a fixed MDMT recommendation model, construct domain-task-conditioned adversarial embedding views using the target task's local sensitivity, and constrain the perturbation to the local nullspace of the other task's logit sensitivity; evaluate whether this output-preserving constraint yields a reproducible AUC increment over matched task-adversarial, pairless, random and Gaussian controls.

This is a candidate contribution, not an established result. It is meaningfully different from the repository's PAIR/ECCM label-conditioned sampler and parameter-gradient transformations, but the conceptual neighborhood is established.

Add the following directly relevant 2025–2026 literature to the novelty review:
- **Rep-MTL**, ICCV 2025, DOI 10.1109/ICCV51701.2025.00326. It uses representation-level task saliency, entropy regularization and sample-wise cross-task saliency alignment. It is a close representation-level MTL optimization neighbor, though not the same per-sample logit-Jacobian nullspace perturbation. Primary source: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Rep-MTL_Unleashing_the_Power_of_Representation-level_Task_Saliency_for_Multi-Task_ICCV_2025_paper.html
- **DTME-MTL**, ICCV 2025, DOI 10.1109/ICCV51701.2025.00277. It performs token-space conflict analysis and token modulation/expansion. It establishes that task-conflict-aware representation-space manipulation is already an active MTL direction; the proposed method's exact mathematical operation and recommendation setting must be compared, not described as the first such method. Primary source: https://openaccess.thecvf.com/content/ICCV2025/html/Jeong_Resolving_Token-Space_Gradient_Conflicts_Token_Space_Manipulation_for_Transformer-Based_Multi-Task_ICCV_2025_paper.html
- **SALARec**, Expert Systems with Applications, published 2026-03-01, DOI 10.1016/j.eswa.2025.130070. It uses preference-aware adversarial perturbation of item embeddings for sequential recommendation. This makes adversarial embedding augmentation in recommendation an established precedent, although its sequential/contrastive objective differs from MDMT output-preserving training. Primary source: https://www.sciencedirect.com/science/article/pii/S0957417425036863
- **CROSSGRAD**, ICLR 2018, uses domain-guided cross-gradient input perturbations. It is an older but directly relevant precedent for gradient-guided input augmentation. Primary source: https://arxiv.org/abs/1804.10745

The existing SAF, M3oE, MMFI, PMTRec and literature sources remain relevant. The literature section must describe the overlap and the distinction precisely. Similarity is not an automatic rejection reason; an equivalent output-preserving Jacobian projection would, however, materially narrow the method-level novelty claim.

## 8. Updated decision hierarchy

1. **Primary research candidate:** DT-PGNA with the exact two-task projection, but only if it adds AUC beyond both pair-TGA and pairless task-BCE adversarial augmentation.
2. **Simpler alternative / decisive comparator:** TGA-Example. If it performs as well as DT-PGNA, prefer it for engineering simplicity, but assess its lower standalone novelty honestly.
3. **Mechanism control:** pair-directed TGA without projection. It isolates the projection increment and is not a separate novelty claim.
4. **Do not revive:** label-match-only PAIR, generic PCGrad/ECCM, generic gradient-magnitude balancing, dynamic EMA, or unqualified Gaussian search.

The current plan is suitable for a bounded, falsification-oriented training-optimization pilot. It is not yet a confirmed paper method. The critical scientific question is whether the cross-task output-preservation constraint contributes independent ranking value after comparison with the simpler task-BCE adversarial augmentation.
