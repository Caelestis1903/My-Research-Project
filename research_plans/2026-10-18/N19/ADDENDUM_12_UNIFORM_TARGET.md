# P5 fixed-target credit falsification

Design before implementation/run. All4 KuaiRand individual-regret runs completed.
Selected regret-minus-full macro AUC+.001167 in4/4,95% descriptive interval
[-.001607,.003940], weighted AUC-.001284 in2/4; loss-.001031 is driven by one
seed(3/4 worsen). Versus utility AUC+.008264 in4/4, weighted+.003457 in4/4, but
loss+.002017 worse3/4. A consistent small macro difference remains despite weak
overall evidence. At selected states, mean source-weight distance from uniform
is.018610 and entropy1.093691 (maximum log3=1.098612). Full routing has
uniform distance.291492. This triggers the declared conventional-mixture
explanation check; it does NOT establish an effective new regret algorithm.

One control entropy_router, BOTH datasets x4seeds=8full from-scratch trainings.
Same full source-router architecture, initial own-domain prior, Adam, learning
rate, batch,50KR/100ML epochs, mean-row risk and selection rule as regret_router.
Replace data-dependent target q=softmax(-finite_loss/.1) with FIXED q=(1/3,1/3,1/3).
Auxiliary remains .1/6 sum_cells KL(q || router_weight). All network/router
parameters train, including the same18raw coefficients. No finite alternative
source evaluations are needed for this control. Preserve ordinary full/utility,
individual-regret and leave-one-out arms.

This fixed-target distillation is a closer falsifier than forcing router weights
uniform from initialization: it holds architecture, trajectory initialization
and auxiliary objective scale fixed, isolating data-dependent credit. It is
ordinary regularization, explicitly not a new algorithm. No temperature/lambda
or multiple mixture-strength grid. If it explains the gain, finite credit loses
its independent interpretation even if a macro AUC advantage remains.

Implementation: add entropy_router to mix families; request router weights,
compute fixed-uniform KL, and omit finite-function forwards only for methods
that do not use them. Keep regret/coalition predictions, source alternatives and
loss paths unchanged. Save exact source snapshots, full receipts and comparisons
against regret/coalition/full/fixed. Test remains unassessed; no freeze yet.

Acceptance: initialized entropy and ordinary full predictions identical; same
parameter count; analytical KL matches functional computation; nonzero router
gradient; no optimizer steps in primary check; same complete seeds/budgets.
Primary self-review is not independent review.

Implementation and primary self-review: PASS. Initial prediction difference from
ordinary full router is exactly0; parameter counts match773488 on ML. Fixed
KL=.34341416, functional-reference error2.98e-8, nonzero router-gradient norm
.198863. Previous committed regret/coalition code gives exactly identical base
and finite-alternative predictions on a balanced real-schema sample. No primary
optimizer steps. Raw evidence: uniform_target_basic_check.json.
