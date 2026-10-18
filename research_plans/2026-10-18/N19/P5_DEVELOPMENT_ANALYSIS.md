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
