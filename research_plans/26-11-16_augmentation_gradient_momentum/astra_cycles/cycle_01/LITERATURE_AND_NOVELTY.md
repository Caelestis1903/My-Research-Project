# Literature, provenance, and limits of the novelty claim
Primary papers and official documentation were browsed for this design on 2026-11-16. Mathematical identities below are our derivations; benchmark outcomes come from this repository's raw evidence. No new method is called novel or effective merely because it has a new label.

## Candidate A: smoothing, task geometry, and antithetic sampling
Bishop's noise-regularization analysis connects training with noise and derivative regularization under its assumptions: [Training with Noise is Equivalent to Tikhonov Regularization, Neural Computation 1995](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf). It supports investigating loss smoothing, not claiming that a finite detached task-axis perturbation implements its exact penalty in M3oE.

Antithetic estimators are established: [Ren, Zhao and Ermon, Adaptive Antithetic Sampling for Variance Reduction, ICML 2019](https://proceedings.mlr.press/v97/ren19b.html). Our paired signs are fixed, not their learned sampler. Conditional equality of the AXIS-IID2 expected objective and AXIS-EVEN is a direct two-sign average, not a literature novelty claim.

[Virtual Adversarial Training, Miyato et al.](https://arxiv.org/abs/1704.03976) uses a local distributional-smoothness construction. This proposal instead evaluates supervised probability BCE at two opposite, frozen task-logit sensitivity directions. It uses no KL objective, power iteration or second-order backpropagation. Binary local sensitivity can overlap with adversarial directions; “different target loss” alone would not establish global novelty.

The actual distinct object relative to the supplied NDP-A implementation is:
L_N= Lc+.25(E+O), L_A=Lc+.25E,
where E=(Lplus+Lminus)/2-Lc and O=(Lplus-Lminus)/2.
This is an exact finite-difference decomposition at a fixed model and detached direction. The mechanism is deleting the complete odd-in-delta component, not reducing gamma or rebranding an ascent view.

In AXIS-EVEN the sign of the normalized label-conditioned direction cancels. Except for threshold/epsilon effects, the retained axis is the task logit Jacobian, not a supervised ascent orientation. A claim that the pair retains the useful label sign would be incorrect.

Historical Gaussian ANTI2 already tested +/- Gaussian views and did not beat IID2 despite conditional variance reduction. The current matrix changes the conditional direction distribution to a task-sensitive axis and explicitly includes both axis-IID2 and isotropic-even controls. Antithetic sampling itself is not new. The historical metrics were produced under a different protocol (the ledger's ANTI2 rows include historical Test results); they are context only, never a current-cycle Test query or a source-matched endpoint control.

## Candidate B: output-error decomposition
[Gradient Centralization, Yong et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613.pdf) centers gradient vectors in parameter space. Candidate B centers samplewise scalar loss derivatives within domain-task output cells, then applies their VJP and restores the existing final bias gradients. The axes of decomposition and the retained bias term differ. This distinction is mathematical, not proof that the entire construction is unpublished.

[Independent Component Alignment for Multi-Task Learning, Senushkin et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf) is a relevant alternative involving task-gradient conditioning. Candidate B computes no task-gradient Gram eigensystem, whitening map or task balancing coefficient.

[CAGrad, Liu et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html), with [official implementation](https://github.com/Cranial-XIX/CAGrad), is a conflict-aware multi-task optimization comparator. B does not solve its gradient-combination problem or project conflicting task gradients. It is not justified by the old clean/residual data-gradient cosine.

The within-cell AUC invariance to adding a constant logit is a ranking identity. It does not imply invariance to the proposed parameter update; shared Jacobians can alter individual rankings. That distinction is part of B's falsification test.

## Candidate C: known gradient transport, new experimental question
[Cutkosky and Orabona, Momentum-Based Variance Reduction in Non-Convex SGD, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/b8002139cdde66b87638f7f91d169d96-Abstract.html), [paper text](https://arxiv.org/html/1905.10018v2), supplies the STORM family of same-sample, two-parameter gradient recurrences. C uses the fixed-coefficient form
d_k=g_k+.9(d_{k-1}-h_k)
and then passes d through the already frozen coupled Adam and fixed EMA. The transport recurrence is known; its composition here is an experimental adaptation. Original SGD results are not transferred as a convergence or efficacy claim for this composition.

[AdaBelief, Zhuang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf), [official implementation](https://github.com/juntang-zhuang/Adabelief-Optimizer), changes the adaptive second-moment construction. C changes the supplied data-gradient estimator but retains Adam's ordinary second moment. It does not rebrand that optimizer.

[PyTorch 2.9 Adam documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html) specifies the installed-version optimizer family, coupled weight decay and bias correction. The design delegates actual arithmetic to the frozen local helper, including its active-parameter treatment, rather than reimplementing equations from a newer documentation version.

## Explicit non-relabeling audit

| Existing line | Actual distinction of selected A |
|---|---|
| NDP-A | Same radius/support/probe, different objective: remove .25 O by using both signs. Source one-sided control retained. |
| TV-GRM | No augmented-minus-clean residual recurrence, cap, post-Adam residual injection or gamma. Full even objective enters Adam moments. |
| PAIR / pair selection | No selected row pairs, pair budget, target subset, dropped support or stratified weighting. Every original row-task contributes. |
| PCGrad / GATS projection | No pairwise gradient projection, orthogonalization, conflict test or projected update. |
| ECCM / conditional history | No auxiliary memory of selected examples, errors, views or residuals. |
| Generic magnitude balancing / PUB | No cross-task/domain coefficients, bargaining solve, update normalization or magnitude target. Isotropic control matches input norm only. |
| A_MATCH | No historical-update matching target, feedback controller or learned update coefficient. |
| Adaptive EMA | Fixed source EMA only; no history-dependent decay or layer-specific averaging. |
| Gaussian ANTI2 | Antithetic mechanism acknowledged as old. New task-axis conditional distribution is tested against its own IID2 and an input-norm-matched isotropic pair. |
| DT-PGNA / TGA | Supplied current handoff names these as stopped, but local searched research-plan/report records do not supply their exact mathematical definitions. Exact historical equivalence is unresolved; no claim of novelty over those unspecified formulas is made. |

The last row is a provenance limitation, not permission to invent a historical formula. If the parent has a frozen DT-PGNA/TGA specification that already deletes this same odd component with these controls, it should mark this proposal redundant before implementation and request a fresh research cycle. No broad novelty or publication claim is authorized by this dossier.

## Evidence separation
Measured: raw five-seed deltas, complete trajectory/update counts, source identities, support ledgers, recorded costs and memory.
Derived: odd/even identity, IID2 conditional expectation, input-norm matching, B's coefficient decomposition, C's recurrence expansion.
Assumed and tested: useful task geometry, harmful odd objective component, benefit of temporal correction, and any endpoint advantage.
Not measured: performance of any new candidate, its full-run throughput, its peak memory, or its efficacy on Test.

