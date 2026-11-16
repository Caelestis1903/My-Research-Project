# Cycle03 research self-critique

Verdict: **GO_TO_FORMAL_DESIGN**, not an independent implementation-review PASS. Select only EARLY10 with LATE10 and exact numeric CLEAN/constant-NQR controls. There is no observed Cycle03 efficacy and no eligible champion. The parent must obtain a fresh COMPLETE specification before coding, then a distinct source-bound reviewer verdict and runtime checks before formal training.

## Strongest objections and their consequences

1. **The proposed candidate was prompted by a favorable E10 slice.** That makes this hypothesis adaptive to C2, not externally confirmed. Freeze E1–10 and E41–50 now; do not later move windows, alter lambda, drop seeds or select a new endpoint. The first ten EARLY epochs are numerically inherited from constant NQR and cannot count as new evidence. Generalization remains untested.

2. **Selected checkpoints give the arms different opportunities.** LATE reproduces CLEAN through E40; all five ML CLEAN EMA optima are already available. Selected LATE≥CLEAN is therefore a structural bound, not evidence of benefit. Report whether it selects a pre41 or post40 epoch. Use fixed E50 online/EMA for the prespecified timing contrast. Keep selected EMA for actual candidate efficacy; do not replace it with E10 or E50 when unfavorable.

3. **Equal exposure is not equal effective dose.** Ten complete natural epochs and identical lambda sum match row exposure and active-update count. Actual `2*pi0*pi1`, gradient norms, optimizer scales and EMA ages differ. Log support/opportunity, active counters, sampled data-gradient norms and actual update norms. These observations do not magically identify a mediator. A claim that phase alone caused a gain would exceed this experiment.

4. **Turning NQR off also changes total dose relative to constant NQR.** EARLY−constant cannot separate duration from placement. EARLY−LATE addresses placement at equal scheduled exposure, but still includes path/history effects. A low-lambda constant arm would ask another question and is not added. If both limited-exposure arms behave alike, retain reduced dose as a competing explanation and return to fresh research without retrospectively adding a control.

5. **The strongest negative evidence is not just late overfitting.** C2 selected ML Macro AUC lost in all five seeds and ML task0 AUC/LL deficits repeat C1. Old E5/S8 also limit optimistic schedule narratives. The primary claim fails if EARLY does not beat CLEAN at selected EMA, even if it improves versus failed NQR or rescues E50. A late control cannot be silently promoted to candidate.

6. **KR evidence is heterogeneous.** Smaller domains have opposite AUC signs while dominant d1 LL improves. A mean Macro AUC deficit with better pooled LL is coherent: metrics weight different populations and properties. Neither calibration improvement nor traffic AUC is a substitute primary endpoint. Domain difficulty must be described using source baseline metrics, task, labels and counts, not inferred from size.

7. **Five reused seeds are not ten independent observations.** Analyze within-seed differences, n=5 per dataset. Curves, tasks, domains, online/EMA and comparisons are correlated. Report all registered outputs and t intervals as descriptive uncertainty, not multiplicity-adjusted discoveries. No CI, p-value or unanimity rule is an early science gate.

8. **Reuse could conceal an implementation change.** Source bytes currently match; that alone does not prove a future wrapper preserves behavior. Require dataset/split/encoder/environment/selection identity, all source-order hashes, and matching numerical prefixes. Compare tensor-state hashes independently of new config/job hashes. PAUSE on mismatch; do not automatically rerun controls, change tolerance or accept a “close enough” baseline.

9. **Diagnostics and recovery can perturb a supposedly identical path.** No extra RNG or gradient probes, no mutable global config and no hidden reset at transitions. The C2 deep-copy optimizer restore fix is mandatory. Test diagnostic/no-diagnostic identity, populated Adam restore isolation and boundary transitions. All model state stays in new RAM storage; old owners and frozen journals remain untouched.

10. **Historical throughput is not a capacity reservation.** C2's 26 concurrent workers and 45m27s are evidence of achieved execution, not a cap or a runtime promise. Own measured full-process peaks must cover diagnostics, evaluation and serialization. Current per-card 85% checks plus physical CUDA-usable capacity are mandatory; forecast power and projected85% reservations are forbidden.

## Prespecified scientific decision rules after completion

For each named comparison, dataset, endpoint and model stream, report five paired differences, mean, sample SD, 95% t CI, and favorable/unfavorable/zero counts. For AUC, positive is favorable; for LL, negative is favorable. Cell protection is the minimum of the six **mean paired cell AUC deltas**, threshold −.001. Do not use a worst raw cell score or mix seed-specific endpoint selections across arms.

The primary efficacy screen is EARLY10−CLEAN at selected fixed EMA. Practical target is +.003 mean Macro AUC for each dataset, with all six cell means ≥−.001, assessed per dataset. Passing both datasets is a cross-dataset validation signal, not a Test result. Passing only one is dataset-specific. If both primary means are ≤0, label NO_CONTINUATION_SIGNAL. Other outcomes are reported as mixed, below-target or cell-protection failure as applicable; none opens Test. LL results remain separately labeled and cannot substitute for AUC. Apply the same numerical thresholds to any explicitly stated practical claim in another comparison, while calling it recovery/control evidence rather than primary efficacy.

For the timing question, examine fixed E50 EARLY−LATE online and EMA, then EARLY−constant and LATE−constant. Consistent positive online/EMA differences with descriptive CIs excluding zero support a scoped timing effect; disagreement or crossing intervals weakens it. An effect below .003 is not a practical-target success even if its CI excludes zero. An EMA-only effect supports no optimizer-independent explanation. Track selected epochs and all E1–50 curves to distinguish retained old peaks from later improvement.

## Execution blockers and handoff limits

No unresolved hyperparameter, arm choice or endpoint choice blocks the research handoff. The following are mandatory **future run gates**, not completed by this child: fresh formal COMPLETE specification; implementation of that specification in a new sibling package; distinct code review PASS or PASS WITH FIXES with all blockers fixed; exact control compatibility and source/data/environment receipts; a parent-started isolated foreground RAM store; four disposable engineering smoke cases and separate source-bound smoke/resource signoff. Technical failures PAUSE for reconciliation. No early scientific kill based on losses or AUC is allowed.

Only explicit USER_STOPPED normally ends the outer loop. Candidate STOP/NO-GO leads through reports/context to fresh parent-directed research. Test stays sealed through ACTIVE, PAUSED and candidate outcomes; future access requires explicit user stop, final freeze and the existing eligible full-five-seed selected-live-RAM verification gate. These planning files neither grant eligibility nor authorize that access.

Self-review checked the corrected C2 LL counts: NQR−BRIER favorable 4/5 on **both** datasets, not 1/5; final evidence R1 PASS with zero blockers. The preserved initial FAIL is not overwritten or treated as current. No source/data rerun follows that prose correction.
