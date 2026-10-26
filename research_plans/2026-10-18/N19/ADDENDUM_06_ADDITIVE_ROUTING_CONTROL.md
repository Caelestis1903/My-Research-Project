# P4 necessary axis-additive control before natural routing results

Domain_mix lacks task main effects; full_mix adds both task effects and their
interaction. To ask whether destination interactions provide value beyond both
ordinary axes, add axis_mix with effective logits
v_add=mean_d(v)+mean_t(v)-mean_dt(v).
This spans additive destination domain/task effects plus a common source offset,
but removes destination interactions. For D=3,T=2,K=3, softmax-effective degrees
are8, versus12 full and4 pure interaction. All three store18 raw parameters;
do not confuse raw storage with function dimension.

Eight full natural runs, both datasets x four seeds, same optimizer/data/budgets.
No new supervision or inference branch beyond the reviewed learned-mixture path.
Compare interaction/replicated confidence to axis_mix as well as domain/full.
This control changes scientific interpretation, so it is necessary rather than
a generic robustness grid. No existing run is cancelled or shortened.

Primary self-review acceptance: additive projection is idempotent and its
double-centered residual vanishes; baseline zero logits match fixed mixing;
main network fully trainable. No extra engineering training required for this
single projection assembled from already exercised means/mixture operations.
Verdict after algebra/numerical projection check: PASS.
