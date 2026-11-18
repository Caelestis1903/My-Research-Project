# P6 development-triggered residual-field joint design

All24 primary P6 runs completed100epochs. Selected copula-minus-independent
AUC+.000107(3/4),95% t interval[-.001983,.002197]; copula-minus-antithetic
+ .000413 but only1/4 positive. Copula+CV minus independent+CV +.000450(3/4),
interval[-.002382,.003282]. Adding CV to copula gives-.000180 AUC and+.001271
loss with mixed signs. No stable independent endpoint advantage is established.
Observed copula joint inclusion .10069/.10950/.10834/.10274 is close to the
antithetic lower endpoint .1, though not identically that fixed policy.

Competing explanation: the combined estimator uses prediction augmentation,
but the declared initial policy still scores UNaugmented gradient alignment.
For B_t=grad BCE(p,pEMA), g_t=grad BCE(p,y), its stochastic field is
h_t=g_t-B_t. The conditional covariance cross term is proportional to
(r/pi^2-1)<h0,h1>, not <g0,g1>. This is a specific estimator-design mismatch,
not an implementation failure of the already declared heuristic. Preserve all
completed raw-alignment runs as a distinct candidate, with their full budgets.

New candidate mask_residual_cv estimates <h0,h1> at the SAME shared STAR
representation, using only jointly revealed labels and inclusion correction.
Teacher predictions are detached and predictable from previous steps. Choose
r=.1 for positive EMA residual alignment, .55 for negative, pi=.55 unchanged.
Loss/EMA remain the exact previous unbiased prediction control variate. This
does not confer full-parameter variance optimality or independent novelty merely
from applying conventional covariance design to residuals.

Necessary strong control mask_antithetic_cv: same teacher, augmented loss and
low joint inclusion .1, but no learned policy. It separates residual adaptivity
from simply combining a classical fixed design with a prediction baseline.
Compare residual to raw copula_cv, independent_cv and antithetic_cv. Two new
arms x4seeds=8 complete100epoch ML trainings, same rows/initialization/order and
mean1.1 labels per visit. No batch/penalty/probability grid or early stop.

Record exact high-endpoint row counts for these two NEW policies in checkpoints,
receipts and last-step diagnostics; do not fabricate such counters for legacy
runs. Extra gradient/teacher cost is measured. Test remains deferred.

Primary self-review acceptance: no unqueried label enters residuals; teacher
detached; jointly observed residual gradients corrected by current predictable r;
same loss and row/task marginal field; static CV control has no copula update;
prior variants' branches unchanged. PASS after minimal loss/gradient identity
and policy marginal checks, then all8 full jobs complete regardless of signs.

Targeted check PASS: exact four-mask enumeration at r=.1/.3025/.55 preserves
loss within1.12e-16 and logit gradients within1.39e-17. Detached-teacher residual
gradient equals teacher-minus-observed-label on queried entries. New adaptive
policy selects declared endpoints; all source parses. No network optimizer step
was used as a substitute for the eight full trainings. Review: PASS.
