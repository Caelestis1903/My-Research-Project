# Independent implementation review — package B

Final verdict after narrow correction review: **PASS**. B1 below was identified, corrected by the coder, and independently rechecked before formal S2/S3 use. No blocking package-B issue remains. The original PASS WITH FIXES findings are retained below as revision history.

Scope: `demand.py`, `formation.py`, `mechanisms.py`, `verify_formation.py`; approved DESIGN_SPECIFICATION, authoritative plan05, decision plan04, acceptance plan08, CODER_IMPLEMENTATION_REPORT_FORMATION and the four coder verification receipts. Shared protocol/readers and campaign calls were inspected for these interfaces. This is independent implementation review, not a new performance experiment or a scientific conclusion.

## Blocking finding B1 — saturated probabilities are inverted for functional interventions

`mechanisms.py::_frozen_delete_predictions` reconstructs a frozen logit with `logit(clip(final_p, 1e-15, 1-1e-15))` before deleting/replacing a delta contribution. `diagnose_functions` similarly reconstructs the GROUP_CAL logit from clipped probabilities before its fixed plus/minus interventions. A probability rounded to 1 or 0 no longer determines the actual frozen logit, so these interventions can change the background model as well as the selected function.

The independent bounded numerical counterexample uses one foreign-domain delta of 40 on one of 4,000 rows, delta zero on the source and the other foreign domain, source S=0, foreign R=0.2 (within the 0.25 budget), and legal reader weight +1. With B=0 the full prediction rounds to 1. Exact frozen deletion must yield sigmoid(0)=0.5; the implementation yields 0.0042337517220676114. Absolute probability error is 0.49576624827793236. This is a mathematical fixture, not artificial research data or a trained performance result.

The corresponding fixed-action case has calibrated background logit 40 and delta -40 on that row; exact action yields 0.5, whereas clipping/inversion introduces the same erroneous background. GROUP_CAL permits this background, and the formation RMS bound does not bound every individual row's delta.

Path to a changed conclusion: clipping changes delete/replace or fixed-action risks → function use/cross-group advantage is measured incorrectly → mechanism-only or attribution decisions may be altered. The correction is therefore required before consuming these diagnostics, although core formation training itself does not use the erroneous inversion.

Minimum required correction: derive each frozen logit directly from its actual selected alpha/beta, B_refit and full feature matrix times its frozen coefficients; implement the intervention in that logit. GROUP_CAL uses its actual selected alpha*B_refit+beta, or exact B_refit for OFF. Do not recover these backgrounds from probabilities. Add the two bounded saturated-logit checks above to the existing necessary numerical verification, and rerun the ordinary reader/diagnostic integration check. No new training matrix is needed.

## Inspected semantics that match the design

- Demand moments are conditional FP64 means; nuisance profiling has the specified ridge, pi enters the shared metric once, excluded support is renormalized, and finite weighted bipartitions use the prescribed deterministic tie rule. K0/K1/K2 and LOW_SUPPORT have distinct bookkeeping.
- The actual query constructor uses assigned rows, Z and omega as specified; ASSIGN_SHUFFLE stays inside each target's eligible rows. Auxiliary source-fit, source-P_form and both foreign-target streams are independent cyclic permutations. Per-slot fresh Adam, full 12 passes, real short batches, pass-end dual updates and final conservative feasible scaling implement T/K/S/R correctly.
- Only delta parameters enter the formation optimizer. Native, pilot, B and A are numpy/frozen inputs. K0 creates no optimizer. Copied hidden parameters plus paired perturbation and zero output have the intended first-step learning behavior.
- POOLED support is independent of RCF requests/K, retains nominal two slots and whole eligible target domains. LOCAL_ACTIVE has 24 heads, group-conditional T without extra division, no source KL/S constraint, and only its local RMS budget. ANCHOR_OFF likewise removes source terms without reapplying S during projection.
- K1_WIDE replaces only actual K2, preserves K1/K0, uses copied 64-wide hidden initialization and the registered wider coefficient bound. SOURCE_INIT_RANDOM only changes delta initialization. SOURCE_LABEL_SHUFFLE uses an isolated pilot/request lineage and actual source labels for S without modifying the real pilot.
- Spectral directions solve the stated generalized problem. Both genuine RRR starts optimize six-cell-weighted BCE plus actual combined-correction RMS and V ridge for 20 passes, discard temporary read coefficients, and are passed together to the common reader for four lambda/start candidates plus exact OFF. Ungrouped RRR changes only formation grouping.
- Candidate deltas and frozen input-only request maps are available on all roles. Outer diagnostic labels require global freeze; no outer sign or function pair is selected inside package B. State objects remain in RAM and no model/checkpoint files are written by these modules.

## Independent checks actually executed

`FORMATION_INDEPENDENT_REVIEW_CHECKS.json` records an independent rerun of the required numerical/gradient checks, plus direct tests of the actual fractional query constructor, independent pooled whole-domain support, KL gradient and auxiliary wrap/resume. Profile error 5.55e-17, metric inverse-root identity error 1.29e-14, query constructor error 0, target-weight gradient error 0, and RAM next-step replay error 0. The valid-RMS diagnostic counterexample is recorded separately in `FORMATION_DIAGNOSTIC_BLOCKER.json`.

A saturated FP64 softplus gradient differs from the ideal expression by 1.26e-11 at logit25 due to PyTorch's default threshold20 linear approximation; specified FP32 training agrees to 5.96e-8. This is not a material training discrepancy and does not justify an extra experiment or an algorithm change.

No formal native/pilot/formation trajectory, full trained-prefix raw-X reconstruction or trained outer retention evaluation was executed by this reviewer. Those remain runner/export acceptance work. No new AUC, LL or method-effectiveness claim follows from this review.

Original release condition: coder corrects B1 in both diagnostic paths, provides its implementation delta and actual regression receipt, and independent rereview confirms the correction. This condition is now satisfied.

## Narrow correction rereview — PASS

The corrected `_frozen_reader_logits` evaluates the actual frozen expression from B_refit, selected alpha/beta and the complete feature matrix times its frozen coefficients. Delete/replace constructs the edited feature bank and reevaluates that expression; the fixed GROUP_CAL plus/minus path calls the same direct-logit helper. Neither path inverts a clipped probability. OFF remains exact B_refit through its selected alpha1/beta0/zero feature coefficients.

The coder correction report and `FORMATION_LOGIT_INTERVENTION_CORRECTION_CHECKS.json` were read. Independent actual checks in `FORMATION_INDEPENDENT_REREVIEW.json` exercise legal saturated and ordinary nonsaturated cases with two feature columns, deletion, replacement and both calibrated action signs. All maximum probability errors against the explicit direct formulas are 0. The valid-R=.2 saturated deletion now yields exactly .5; calibrated40 plus delta−40 also yields exactly .5. These checks are bounded numerical fixtures, not formal training or performance evidence. No additional training matrix was introduced.

Final reviewed source SHA256 values are recorded in `FORMATION_INDEPENDENT_REREVIEW.json`; those replace the earlier pre-correction mechanisms.py hash for authorization. Package B is approved at that identity, subject to the separately required shared-reader/campaign acceptance and trained-data export/raw-X checks during RUN.
