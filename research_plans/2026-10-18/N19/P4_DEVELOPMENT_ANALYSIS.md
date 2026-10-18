# P4 first complete MovieLens primary reading

Four seeds x100epochs each completed for fixed/domain/full/interaction. KuaiRand
primary and axis-additive controls continue. Test remains deferred.

|Method|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Fixed|.730898|.550470|.739844|
|Domain-only|.730529|.549941|.738742|
|Full cell mixing|.730463|.549649|.738824|
|Centered interaction|.730879|.549460|.739332|

Interaction-minus-domain selected AUC+.000351 in4/4, interval[-.000006,.000707],
loss-.000481 in4/4, interval[-.000847,-.000115]. Yet interaction-minus-fixed
AUC-.000019 in2/4, loss-.001009 in3/4, both intervals include0. Full mixing
comparison has AUC+.000416 in3/4, interval[-.000345,.001178]. Evidence of an
independent interaction benefit is WEAK; outperforming one learned router does
not establish improvement beyond the strong fixed model. Against MMOE AUC
+.001796 in4/4, but loss+.000211 with wide uncertainty. PLE also has lower
selected AUC, yet its architecture alone cannot identify the interaction effect.

At the final epoch, interaction-minus-domain AUC=-.006473 in0/4 positive seeds,
interval[-.011788,-.001158]. Interaction-minus-fixed loss=+.040323 in4/4. The
selected/final reversal argues against a stage-invariant sharing mechanism.

Preplanned replicated-confidence alternative now enters8 full runs, both
datasets x4seeds. Its engineering update receipt already passed; all ordinary
weights train. The corresponding confidence-off arm is interaction, with full
cell and additive-axis controls. Sampling noise in two train-meta halves, generic
shrinkage and subspace capacity remain competing explanations. Effective-logit
centering is a constraint, not evidence of utility or causal domain structure.
KuaiRand, actual replicated signs/cost, per-cell size/difficulty and final frozen
test results remain pending; no broad direction closure is declared here.

Checkpoint routing diagnostics use the implemented router projection and actual
saved selected/final coefficients, reporting all18 source weights and effective
logit margins without any data/test forward. Primary self-review (not independent):
PASS; the same fixed prior is used as training, and logit constraints are kept
distinct from probability margins. Source entropy alone is not useful credit.
