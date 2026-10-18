# P5 first complete MovieLens primary reading

All4seeds x100epochs completed for finite individual-source regret, ordinary
full routing and specified train-meta utility routing. KuaiRand continues; test
remains deferred. No claim to reproduce all validation-allocation methods.

|Method|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Fixed|.730898|.550470|.739844|
|Ordinary full routing|.730463|.549649|.738824|
|Train-meta utility|.730603|.550316|.739158|
|Finite regret|.730719|.551931|.739312|

Regret-minus-full selected AUC+.000256 in3/4, interval[-.000805,.001316],
loss+.002282 worse in3/4, interval[-.002796,.007360]. Versus utility AUC+.000116
in2/4, interval[-.000685,.000916], loss+.001615 with mixed signs. Versus fixed
AUC-.000179 in2/4, weighted AUC-.000532 in3/4 and loss+.001461 worse in3/4.
Evidence of an independent selected benefit is WEAK.

Final-epoch regret-minus-fixed AUC+.004801 in4/4, interval[-.002559,.012161],
but loss+.010232 worse in4/4, interval[-.005264,.025728]. Versus utility, final
AUC+.006165 in4/4 and loss-.040209 in4/4, intervals still include0. This stage
difference is consistent with optimization/regularization effects; it is not a
uniform generalization advantage or proof that finite credit is more faithful.

Preplanned alternative now launches8 full runs: leave-one-source-out conditional
mixture credit, both native datasets x4seeds. It compares finite removal costs
through the same tower rather than competing individual expert outputs. Ordinary
full routing/utility and the individual-source arm isolate this change. Primary
self-review in ALTERNATIVE_REVIEW passed its sign and actual prediction path;
all network weights train and all budgets complete regardless of early scores.

Source permutation covariance alone cannot establish an MDMT contribution;
finite counterfactual routing/regret have direct primary-source coverage.
Competing explanations include near-uniform regularization targets, router
capacity, auxiliary KL and training-stage effects. If a repeatable advantage
remains on KuaiRand/coalition with near-uniform weights, a fixed uniform-mixture
control is the bounded discriminator; it is not launched from this null selected
MovieLens result alone. Endpoint decision comparisons cannot replace training.
Per-cell size/difficulty, both datasets, routing behavior/cost and frozen test
will determine scope; the broad function-credit question is not rejected here.

For completed MovieLens checkpoints, regret routing source entropy is.805296
selected/.951829 final, versus ordinary full .762772/.745510 (maximum log3).
Mean source-weight distance from uniform decreases from.250203 to.172964.
This is consistent with auxiliary distillation encouraging broader mixtures,
not by itself proof of correct expert attribution. Selected regret operational
training time averages216.0s versus177.4s ordinary full and186.2s utility, with
1142.5MiB peak reserved versus916MiB. These concurrent costs are descriptive;
no isolated speed claim. KuaiRand/coalition receipts and final tests remain pending.

## Complete KuaiRand primary reading and matched fixed-target discriminator

All4seeds x50epochs completed. Selected regret AUC .676326/loss .443091/weighted
AUC .701129, versus full .675159/.444122/.702412 and utility
.668062/.441074/.697671. Regret-minus-full macro AUC+.001167 in4/4, interval
[-.001607,.003940], but weighted AUC-.001284 with2/4 positive signs; mean loss
-.001031 is driven by one seed (3/4 worsen). Versus utility AUC+.008264 in4/4,
weighted+.003457 in4/4, interval[.000918,.005997], but loss+.002017 worse3/4.
Versus fixed AUC+.003426 in3/4, weighted-.001167 in3/4.

Task1 accounts for the regret-minus-full macro advantage (+.003895,4/4),
while task0 decreases-.001562 in3/4. Final regret-minus-full AUC instead
-.005462 in0/4 positive seeds, interval[-.011192,.000269]. A selected macro
advantage is not a stable all-task/all-stage improvement.

Selected source entropy1.093691 is near log3=1.098612, and mean weight distance
from uniform is.018610; full routing distance.291492. The repeated macro sign
and near-uniform behavior trigger ADDENDUM_12:8 full trainings of entropy_router,
same architecture/init/auxiliary coefficient but fixed-uniform distillation
target. This is better matched than static forced-uniform weights and explicitly
conventional regularization. It tests whether data-dependent finite credit adds
value beyond uniformization, not whether another lambda can produce a win.
Leave-one-out and fixed-target arms both complete before final conclusions/test.
