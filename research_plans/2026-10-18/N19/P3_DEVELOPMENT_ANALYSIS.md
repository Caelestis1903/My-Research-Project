# P3 first primary development reading; alternatives launched

MovieLens uniform/protected each completed4seeds x100epochs. KuaiRand protected
is still running; partial means are not used for final conclusions. Test deferred.

MovieLens selected macro AUC: fixed .730898, uniform .729633, protected .729633.
Protected-minus-uniform AUC=-.0000001535, descriptive95% paired interval
[-.000022,.000022]; LogLoss=+.00002878, interval[-.000092,.000149]. Selected
checkpoints do not support an allocation-specific gain (WEAK evidence of benefit).
Final AUC difference +.002209 in3/4 seeds, interval[-.002344,.006763], and loss
-.014284 in3/4 are directionally different but uncertain. Training stage matters.

Protected ranks by seed are [6,16,26], [9,16,23], [6,18,24], [7,14,27]. Main
domain sizes are [32775,2850,142500]. Largest domain receives more capacity,
but the medium domain receives less than the smallest; this is not a monotone
size allocator. Scores, baseline risk and optimization are competing explanations.

Both preplanned alternatives now train on both datasets with all4seeds:
shared domain dictionary and task-conditioned shared dictionary. Slots are
generated BEFORE nonlinear STAR, distinct from a repair head; task conditioning
doubles STAR/expert paths. Storage matching is not runtime matching. This tests
shared factors and task codes after the first allocation mechanism showed no
stable selected benefit. Full KuaiRand, cell metrics, cost and test remain pending.

## Bounded checkpoint sensitivity measurement

Motivation: MovieLens selected epochs are11/13/11/10 for both rank arms, with
near-zero selected differences despite different allocations. To distinguish
weak current slot influence from successful allocation, n19_rank_diagnostics.py
sets all rank masks to zero at each completed selected/final checkpoint on a
fixed train-meta sample (up to512/domain). It reports per-cell loss and absolute
probability change, with no optimizer steps or test access. This is conditional
sensitivity, not retrained ablation or independent generalization evidence.
Primary self-review, not independent: the intervention changes only rank_masks,
retains skip/biases and uses identical rows for both arms; PASS for this bounded
measurement. No new training algorithm or cancelled formal comparison results.
