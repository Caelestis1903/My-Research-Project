# N18 review gate

Reviewer: primary agent self-review under latest user-invoked start.md principle 11.
This is not independent review. Inputs: DESIGN_SPECIFICATION, AMENDMENT_01_SCHEMA,
actual five source files and CODER_IMPLEMENTATION_REPORT.

Verdict: **PASS WITH FIXES — all blocking fixes applied before substantive run.**

1. **Relevant / fixed:** raw ML has rating timestamps, unlike cached features.
   Added a raw-row-ID join, rating-label identity check and descriptive time
   quartiles. Do not infer exposure policy or a causal nuisance from them.
2. **Critical / fixed in design and implementation:** coefficient degrees of
   freedom alone would overstate deployment savings. Persist/report basis,
   normalizer, rank factors and common trunk; preserve existing DT-dependent
   gates/experts. The quadratic additive certificate never applies to full M3oE.
3. **Critical / fixed by canonical span:** centered repair and ordinary extra
   cell head have the same span. Use identical orthonormal coordinates and ridge
   metric. Reporting two independently regularized parameterizations as different
   methods would be invalid. Only one fit and one equivalent representation occur.
4. **Checked:** all six P4 terms use actual domain masks and task ordering (d,t),
   retained graph for first five calls; source actions are the same for predictors.
   No unsupported learned-fusion or private-parameter assumption is introduced.
5. **Checked:** P6 q has full support, p/q is exact, per-row gradient averages both
   tasks, the mean is subtracted from second moments, and design/heldout rows do
   not overlap. Proxy and mixture are explicitly distinct from full EMAIS/OMIS.
6. **Checked:** all P3 fitting and residual selection use train data; no test
   outputs are read. P2 oracle is evaluated separately from feasible bounds.
   Random-audit estimates use finite-population SE, not exact population equality.
7. **Scope limitation:** no faithful SAF/D3/AutoIFS/DSMTL/ExtraDebias/WCP-DR neural
   reproduction is implemented. Only assumption and information comparisons,
   conventional strong local baselines and exact restricted-class statements
   are authorized. No superiority or paper failure claim may be made against them.

Syntax/basic checks complete. No unresolved blocking issue. Proceed to raw
theory/data and checkpoint measurements; then delegate only mechanical P3 fitting.

## Final scientific review after authorized amendments

Amendment 02 corrects the Critical training-field/functional-risk Jacobian issue:
identical forward gate inputs are asserted, and a discrepancy-sensitive finite
difference agrees with the complete derivative (relative error about .28%).
Earlier raw measurements and source remain archived; formal P4/P6 use corrected
measurements only. P3 features/fitting are unaffected.

Amendment 03 completes direct four-measurement P4 costs on the same rows, and
decision accuracy is also reported on the common non-abstention set. Single
paired timings are explicitly weak evidence under GPU concurrency.
P5's declared same-function output-bias construction and ordinary population
baseline finish the decision-consequence check without inventing trained gates.

Actual stored FP64 head/basis and FP32 normalizer/trunk byte counts are reported
beside coefficient dimension. Output/history-root environment overrides change
paths only and allow clean reproduction without overwriting archived experiments.
All six reports distinguish restricted math, controlled data, frozen-point
measurements, new analytic fitting and unrun end-to-end training.

`SCIENTIFIC_ACCEPTANCE.json` passes the conclusion-relevant numerical identities.
No unresolved Critical issue. Final verdict: **PASS (primary self-review only)**.
