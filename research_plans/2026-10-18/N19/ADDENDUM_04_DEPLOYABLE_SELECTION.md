# Relevant scientific correction before formal P3 rank jobs

Risk: an unrestricted selected-checkpoint search can pick a warmup epoch with
96 active ranks, although the claimed deployed comparison requires total48.
That changes model capacity and invalidates a fixed-budget performance claim.
Classification: CRITICAL for P3 interpretation. No formal rank run has started
at this amendment. Existing completed/running non-rank jobs are unaffected.

Both rank arms now allow checkpoint selection only at epochs10..budget, after
the common spectral pruning transition. All warmup trajectories remain recorded.
Full training budgets and all other arms' selection rules are unchanged. Final
test evaluation exports only active factor tensors and verifies prediction
equivalence on1024rows, recording physical counts/errors and artifact paths.
This establishes parameter storage, not faster inference: Hadamard combination
with STAR's shared matrix still reconstructs a dense slot, and runtime is measured.

Primary implementation/self-review: PASS, prerequisite is epoch10 pruning before
validation/selection and identical eligibility for uniform/protected rank arms.
