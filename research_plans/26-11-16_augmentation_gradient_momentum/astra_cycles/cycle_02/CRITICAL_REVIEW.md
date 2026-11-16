# Cycle02 critical review

Research-child verdict: **GO TO FORMAL DESIGN, with efficacy confidence WEAK.** This is a self-critique, not the required distinct implementation review. Parent must dispatch a fresh designer, then coder, then distinct reviewer before any runner. A candidate STOP rejects the candidate and returns to the parent's research loop; it cannot stop that loop.

## 1. Reasons this candidate may fail

1. **BCE plus EMA may already capture the useful ranking information.** Older PAIR, feedback and perturbation histories make a generic “ranking should help AUC” story inadequate. Prediction: NQR does not exceed CLEAN; possibly all three new arms fail. That is a valid fixed-candidate negative outcome after all five seeds, not a reason to adjust lambda.
2. **The apparent effect may be pointwise squared-error regularization.** BRIER has the same support/opportunities and a matched value/first derivative at uninformative scores. If it matches or exceeds NQR, no ranking-specific superiority is demonstrated. BRIER is not globally gradient-norm matched; this limit must survive the report.
3. **The square loss compresses meaningful within-class variation.** NQR−MEAN-GAP is exactly the variance penalty. Mean-gap winning would implicate this term, while smaller training variance alone proves neither improved ranking nor mechanism. Different trajectories still prevent a claim that a diagnostic mediator uniquely caused an endpoint difference.
4. **Natural opportunity weights can weaken minority-class signal.** No inverse-prior correction is used; 2*pi_1*pi_0 can be small. Rare positives are present, but their effective influence is not identical to BCE. This is a declared design property, not missing data or an invitation to rebalance after seeing results.
5. **Loss-scale/Adam/decay interaction can explain an effect.** Mixture coefficients sum to1; loss derivatives do not have equal norms. Fixed coupled decay and adaptive moments can turn scale changes into different optimization. Log data-gradient and actual update norms separately. Do not claim a pure ranking-direction causal effect or add a norm-rescaled arm mid-run.
6. **Probability saturation and label noise remain.** The bounded range prevents a score gap>1, but it can suppress gradients through sigmoid or amplify wrong supervision relative to useful fine ordering. There is no proven consistency or robustness theorem for this mixture.
7. **Selection and repeated research on the same validation set can overstate promise.** Earliest-max selection is matched, but the broader unlimited research loop reuses validation evidence. Five paired seeds and descriptive intervals do not remove adaptive selection bias. This cycle makes validation-screening claims only; Test cannot be opened to resolve an ambiguous candidate while ACTIVE/PAUSED.

## 2. What would disprove the proposed story

| Observation after complete seeds | Allowed conclusion | Forbidden rescue |
|---|---|---|
| NQR mean AUC−CLEAN<=0 on ML and KR | Registered NO_CONTINUATION_SIGNAL for NQR-025; no champion | Choose a favorable online endpoint, E50, seed subset, dataset average or new lambda to reverse the primary decision |
| Positive on only one dataset | MIXED_DATASET_SIGNAL, with exact cell/LL costs | Present cross-dataset benefit or drop the other dataset |
| NQR−BRIER mean<=0 on a dataset | Ranking-specific advantage unsupported there | Rebrand BRIER as a successful NQR mechanism or promote it posthoc |
| NQR−MEAN-GAP mean<=0 | Positive contribution of variance compression unsupported there | Treat small variance as proof of useful regularization |
| Macro gain>=.003 but any cell mean<−.001 | Report macro target met, cell protection not met | Erase the gain, conceal the cell, or veto an entire future research direction |
| LL mean<0 but AUC fails | LL improvement, separate from AUC | Claim a ranking win from loss reduction |
| Nonfinite loss, identity mismatch, bad graph, OOM, owner loss | TECHNICAL_PAUSE with raw evidence | Count incomplete seeds as scientific failures or silently rerun under a changed protocol |

No scientific gate acts on1–4 completed seeds, a single cell-seed, sign unanimity or a CI crossingzero. There are no hidden total/revision caps. The30-run matrix is one fixed experiment budget, not a limit on later research cycles.

## 3. Control sufficiency and limits

The two controls are the smallest useful decomposition here: a pointwise competitor and an exact removal of within-class variance. They match masks, class statistics, data population, lambda, forward count, selection, optimizer and EMA. They do not match every derivative away from p=.5. In particular, Q_B−Q_R includes both a common-mean term and half the variance sum; do not call that comparison a pure offset intervention.

No control establishes a general benefit of AUC loss families. No control establishes that size or difficulty causes an effect. None can make an internal objective ablation a published-model baseline. A larger factorial or matched-gradient framework would change the task and cost; it is not needed to answer this narrower question.

The most serious evidence gap is positive efficacy evidence under this exact natural weighting and fixed M3oE protocol. It is intentionally left for the experiment. No amount of symbolic equivalence checking can fill it. The most important engineering gap is that the candidate implementation and its full-process resource profile do not yet exist; they require coder/reviewer/runner stages.

## 4. Gates the formal designer/reviewer must resolve

- Verify the O(B) loss and gradient against the literal conceptual pair sum on a tiny fixed fixture; include both labels, one-class cells, absent domains, singleton classes, unequal prevalence and tails. No model training is needed for this CPU correctness check.
- Verify exact source-clean lambda0 identity for parameters, Adam state/counters, EMA and RNG; compare `grad=None` masks. Verify all-single-class delegation and mixed-cell formulas. Confirm no full-gate probe or accidental detached means.
- Bind new source and transitive imports, the frozen source receipt,10 CLEAN identities, initialization hashes and500 epoch-order hashes. Source changes after review invalidate that review; scientific fixes require the same distinct-review gate, no silent patch during RUN.
- Review RAM owner/root separation and the current-utilization/physical-capacity predicate. The old C1 bootstrap projected-reservation rules are superseded. Keep C1 sources, receipts, raw evidence, owner and live states untouched.
- Run engineering full-batch smoke under a current native runner only after code review. Profile real process residency including CUDA context and validation/diagnostic/serialization phases. Do not admit formal workers from torch-only peaks or predicted power.
- Parent checks stop controls in both workspaces before dispatch. If stopped, preserve the atomic transaction and summarize; do not start a new research child. No Test while ACTIVE or PAUSED.

These are finite correctness/identity gates required by this experiment, not a new framework or an open-ended hardening program. Implement the narrow loss and cycle adapter, reuse audited model/Adam/EMA/evaluation/BytesIO helpers, and stop adding checks once the registered checks pass unless a concrete failure appears.

## 5. Scope and handoff

This research child wrote exactly the six requested Cycle02 planning artifacts. It did not alter source, root state, history or C1 files. It checked on-disk stop flags and known source/raw receipts only; it did not contact RAM owners or verify live capacity. Publication evidence was read from the supplied satisfied gate and local HEAD. No new remote write or publication is part of this task.

Unselected VAT/SAM ideas remain alternatives, not fallbacks to implement when NQR looks negative. The fresh formal designer may reject NQR for mathematical redundancy or an unresolved contract, but should route that outcome through the parent's report/context/new-research sequence. The research child's context is then permanently closed, without reuse or descendants.
