# P4 direct-cost completion — before P4 result analysis

The additive P4 estimator was initially calculated from full-cell gradients.
This alone cannot measure its potential cost advantage. Complete the specified
cost comparison with a direct four-measurement estimator on identical A/B rows:
three domain task-average gradients plus one global task-contrast gradient.
This is ordinary additive ANOVA, not a new algorithm. It spans all four additive
cell components, but contains no information on the two interaction components.

One paired timing per seed, order alternated by seed, compares 8 versus 12
training-gradient calls. Verify equality to projected full gradients to 1e-4
relative error. Do not repeat a benchmark to find favorable timings. Report
measurement noise and validation-gradient overhead separately. C labels never
fit the predictor. This completes the original full-versus-marginal cost intent;
no new training, tuning or additional sample population is introduced.

Primary self-review: **PASS**. Four terms are explicit linear combinations of
the same six cell BCE means because A/B are exactly balanced by domain. Negative
task-contrast loss is intentional, and permits a signed linear measurement.
