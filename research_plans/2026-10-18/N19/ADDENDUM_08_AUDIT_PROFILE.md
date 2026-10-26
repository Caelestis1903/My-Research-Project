# P2 development-triggered audit profiling rather than latent co-adaptation

All40 P2 primary/control runs completed100epochs. Selected joint-axis AUC
.716590 versus nested-axis composite .717265; difference-.000675,2/4 positive,
95% t interval[-.003783,.002432]. Loss difference-.003245 favors joint in4/4,
but interval[-.007588,.001098]. Axis-minus-independent AUC-.000670 in0/4.
Selected channel RMSE: audit-fixed .035074, axis .098250, joint .094240. Final
joint improves rate RMSE to.077806, whereas axis degrades to.127579. Prediction
selection and channel recovery therefore disagree. Joint optimization failure
or weak anchoring competes with genuine ineffectiveness of cross-axis structure.

Alternative mechanism: fit measurement rates ONLY from audited MAIN clean/noisy
pairs before network training, then freeze them. No network posterior can absorb
this measurement fit. For each clean class separately, collect cell flip count k
and eligibility n. Fit eta_dt=.35 sigmoid(mu+r_d+c_t), with r_0=c_0=0, by minimizing
-sum_dt[(k+.5)log eta+(n-k+.5)log(1-eta)]. This uses the same Jeffreys cell
pseudocounts as independent audit_fixed, but estimates only4 effective parameters
per channel rather than6. L-BFGS-B, bounds[-12,12], analytic gradient, max1000iter,
ftol1e-12/gtol1e-8, start all rates .1. Save convergence, counts, coefficients,
fitted rates and objective. No true simulator rates enter calibration.

Four new arms x4seeds, same full100epoch network budgets:

- profile_axis: frozen axis-calibrated channel, independent clean marginals,
  composite likelihood. Compare axis_channel to isolate anchored calibration;
  audit_fixed to isolate shared calibration under identical posterior family.
- profile_nested: same frozen axis rates, nested posterior, composite likelihood.
- profile_joint: same frozen axis rates, nested posterior, joint likelihood.
  Compare profile_nested to isolate co-observation likelihood.
- audit_joint: independent audit-fixed rates, nested posterior, joint likelihood.
  Compare profile_joint to isolate channel-axis sharing at the same posterior/loss.

This is a bounded mechanistic factorial, not a penalty/search grid. It addresses
an observed rate-recovery failure with a structurally different fit. Audit GLM
profiling and frozen forward correction are conventional statistics, not a claimed
new identifiability theorem. Positive outcomes may still fail independent novelty.
All deep network parameters train from scratch; no frozen checkpoint/backbone.

Primary self-review acceptance: fitting function receives ONLY audited rows;
class-specific denominators; same smoothing as audit_fixed; independent fixed
rates unchanged; additive rates match canonical mu/row/column reconstruction;
analytic gradient/convergence checked on actual audit counts. Existing nested,
joint and frozen-channel training paths are already exercised. Verdict: pending
the targeted fit check, then PASS for all16 complete runs. Test remains deferred.

## Fit check and full-run authorization

PASS: analytic-gradient L-BFGS rates agree with an independent numerical-gradient
solve within4.40e-7 (channel a) and2.78e-7 (b). Canonical channel reconstruction
matches fitted rates within float32 precision. Independent audit_fixed rates
are exactly unchanged. All audit fits converge on actual main/audit counts;
no true simulator rates or validation labels enter fitting. New frozen-channel
branches reuse exercised full-network/nested/joint paths. All16 full jobs added;
no engineering proxy replaces their100epoch outcomes.
