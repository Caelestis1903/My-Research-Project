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

## Complete KuaiRand primary comparison and active-capacity limitation

Both rank arms have now completed all4seeds x50epochs. Their selected AND final
cell metrics match exactly in every paired seed. Selected macro AUC .666786
versus full fixed .672900; selected loss .445601 versus .444001. This is strong
descriptive evidence of endpoint equivalence in this regime, not proof that
allocation is intrinsically irrelevant. Protected allocations are [16,16,16],
[14,19,15],[9,21,18],[13,19,16]; the effective capacity decision actually differs.

Finite-removal diagnostics found zero prediction changes at epochs5/10,selected
and final for all4seeds in both KuaiRand rank arms and the dense fixed baseline.
Double-precision norms avoid treating float32 norm underflow as exact parameter
zero. Across4seeds, fixed shared norm falls from1.599 at epoch1 to8.21e-5 at5,
2.74e-8 at10,4.28e-15 selected and1.03e-33 final. Uniform shared norm at10 is
3.65e-10 and generated slot norm4.96e-12. The slot path is already numerically
inactive before the epoch10 allocation. No claim of a neural capacity limit or
size-driven failure can be supported by this comparison.

MovieLens selected mean absolute probability sensitivity is .002591 for dense
fixed,.001093 uniform,.000974 protected; final uniform/protected .000429/.000333.
Selected slot removal slightly lowers train-meta loss on average, but this is a
conditional sample intervention, not a retrained ablation or held-out benefit.
Optimization regularization, skip-path dominance and training-update count
compete with the proposed allocation explanation.

ADDENDUM_11 therefore launches20 matched KuaiRand AdamW full trainings across
dense/uniform/protected/domain-dictionary/cell-dictionary. This is one bounded
optimizer discrimination with same decay coefficient, rates and budgets; it
changes all parameter updates and does not isolate slot weight decay causally.
The original Adam results remain fully archived. Final capacity conclusions
wait for this active-capacity check, dictionary results and frozen test.

A diagnostic-script container mismatch when adding the dense ParameterList was
corrected before any new diagnostic table was written; old completed training
and initial rank-only measurements were unaffected. The final script stacks
slot lists without mutating parameter dtypes.
