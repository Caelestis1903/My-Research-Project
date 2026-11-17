# Research self-critique and downstream acceptance criteria

Research verdict: **GO_TO_FORMAL_DESIGN**, zero unresolved research blockers. This is Astra's critical self-review, **not** the required distinct implementation reviewer or engineering signoff. Parent must dispatch those later. Existing C3 Halley PASS applies to C3, not to unimplemented C4.

## Why this experiment is worth its finite budget

The selected rank deficit survived variance removal and late-exposure removal, but completion weights were not independently removed or isolated. W deletes the ranking term entirely, and S prevents the remaining global BCE coefficient deficit from being misattributed to heterogeneous weighting. Twenty new full trajectories plus twenty reused numeric controls answer that conditional question; further variants would dilute it. Candidate success is not required for informative attribution.

The strongest prior evidence is ML-specific: C2 NQR and C3 EARLY selected EMA AUC intervals are negative, and task0 rank costs repeat. KR directional evidence is weak. The exact C3/C2 E10 identity prevents counting those prefixes as independent corroboration. C3's lower mean task0 LL does not repeat C2's joint rank-and-loss harm. Favorable TV ML gamma .20 and C1 KR IID2 means remain weak counterexamples. No statement that every intervention, seed, task or large domain failed is supported.

## Principal inferential risks and bounded responses

| Risk | Required response | Residual limitation |
|---|---|---|
| W removes both ranking gradient and its coefficient mass | Include S=mC with exactly W's batch BCE mass; report N-W as a composite ranking-addition effect. | N-W does not isolate direction at fixed gradient norm. |
| W-S changes class, cell and task allocation together | Log all coefficients, per-task/cell masses and natural counts. State the aggregate weighting question explicitly. | Cannot attribute solely to class prevalence or domain weighting without another intervention. |
| Adam may partly cancel a scale change | Keep source coupled decay/epsilon/private moments; measure common-state real-Adam counterfactual steps. | Local transformations cannot establish full-path optimizer mediation. |
| Selected epochs differ | Report chosen epoch/hash and fixed E1/E10/E30/E50 for both streams. Never select LL separately. | Selection is part of the registered treatment outcome, not a fixed-epoch causal comparison. |
| EMA explains total-versus-online gain | Keep W/EMA-C/EMA primary; report W/EMA-C/online and within-arm EMA-online. | EMA is a path-dependent readout; no frozen-history causal decomposition. |
| n=5 and many cells/stages | Raw paired five seeds, ddof1 SD, t4 descriptive intervals, sign counts; no multiplicity discovery claim. | Crossing intervals do not prove absence/equivalence; six cells are not extra seeds. |
| Repeated adaptation to the same validation populations | Scope the evidence as exploratory; preserve exact shared prefixes and prior experiments as correlated history. | Neither five seeds nor repeated cycles provide an untouched generalization estimate. Test remains sealed under the existing gate. |
| Reused controls may have incomplete old schema | Verify exact numeric files, source/order/selection provenance; retain null reasons for absent diagnostics/RNG/count/Test fields. | No retroactive per-epoch RNG/loader/Test evidence can be fabricated. |
| Diagnostics might perturb training or restore alias moments | Finite on/off equivalence and RAM round-trip checks at lazy/populated Adam states; deep-copy restore. | Engineering review must validate actual code, not this statement. |
| New diagnostics raise residency above C3 | Measure new full-process profiles on both cards, including clone/CPU-copy/serialization peaks, then physical fit. | C3 concurrency/profile numbers are context, not new admission limits or promises. |

## Specific falsifiers and evidence labels

* A repeated W-versus-C ML rank deficit with R identically absent falsifies the necessity of ranking gradients for that deficit. It does not establish that ranking never hurts.
* W-S adverse with coherent seeds and a signed interval, while S-C is precise and near zero, is **MEDIUM** evidence for heterogeneous completion weighting under this protocol. Replication across the two datasets strengthens scope; it is not required to state the narrower result. No new equivalence margin is invented after seeing results.
* S-C and W-C moving similarly, without precise W-S separation, leaves global scaling plausible (**WEAK** or **MEDIUM**, according to actual precision/consistency). It does not prove scale is the only cause.
* N-W adverse while W-C is recovered supports a ranking-addition cost conditional on W. The local gradient/Adam diagnostics may suggest a route, but full-path attribution remains **WEAK** without an intervention on that route.
* W-C failing practical thresholds means no qualified candidate regardless of favorable mechanism contrasts. Both means <=0 is NO_CONTINUATION_SIGNAL; one positive is MIXED. Neither stops the outer research loop or opens Test.

Use **STRONG** only for tightly scoped, repeated, well-audited phenomena with clear signed evidence or exact identities; **MEDIUM** for consistent but limited/conditional support; **WEAK** for broad intervals, conflicting seeds, exploratory local association or extrapolation. Labels do not convert descriptive intervals into confirmatory inference. Report competing explanations even when one label is stronger.

## What an independent code reviewer must accept

1. The forthcoming formal specification, actual new sibling implementation and coder report agree; no old frozen source was changed. W/S exact support/mass, all-single-class/lambda0 source delegation, absent-class nulls and FP algebra fixtures pass.
2. Source ordinary gated forward, detached gate input, architecture, Adam coupled decay/private steps/None-versus-zero, one actual optimizer/EMA update and loader RNG remain unchanged. No custom optimizer approximation or hidden loss normalization.
3. W/S production never applies NQR gradients; NQR is a read-only pure diagnostic operator. Diagnostics leave live gradients, states and RNG equivalent to diagnostics-off execution within the exact replay contract. Virtual optimizer state never aliases live or saved state.
4. All five seeds and all 50 epochs are enumerated before training. Primary, comparator, six-cell gate and same-state LL are fixed. Historical control schemas are accepted as recorded, not embellished.
5. New roots, explicit source dependencies, hash-bound numeric control import, RAM owner scope and storage exclusions are concrete. No old prepare/run/export/owner mutation, no disk tensors, no hidden driver.
6. Runtime implements current per-card memory/power and host-RAM gates, first-full-step/lazy-Adam ACK, physical usable-capacity fit, no count caps/both-card gate/projected85 rule, and whole-epoch ACK recovery accounting.

Verdict must be PASS or PASS WITH FIXES with every blocking fix corrected. FAIL returns to a fresh coder under parent control; no child reuse. A distinct fresh reviewer must also sign off the real engineering results before formal training.

## Remaining work, not missing scientific choices

Formal design must translate the specified interfaces/CSV fields, FP tolerances, source dependency closure, engineering fixture sequence and ACK state machine into testable contracts. Coder then implements only that specification; reviewer and runner provide actual verification. New source hash, measured C4 residency, endpoint/session ID and physical runtime are necessarily unknown until those stages. They are recorded as null/pending, never invented. No mechanism, coefficient, seed, epoch, primary metric or control choice remains unresolved.

The work deliberately excludes HPO, coefficient/timing rescue, a full factorial, architecture/gate/routing changes, synthetic difficulty, additional models, detached AI, guardian replacement, unrelated missing-data restoration and Test. Those omissions preserve the identified question; none blocks the authorized research handoff.
