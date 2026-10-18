# Alternative implementation report and primary self-review

Authority: revised start.md11; primary performs non-training stages. This is
NOT an independent review. Inputs: main DESIGN SPECIFICATION, addenda02/03,
actual n19_models.py/n19_train.py and their exact source snapshots.

P4 replication samples disjoint train-meta halves with deterministic seed/epoch
streams, evaluates +/-0.1 perturbations, restores coefficients from clones,
updates the checkpointed confidence buffer only, and projects again after gating.
The effective logits' largest numerical mean across either destination axis was
1.49e-8 with heterogeneous random confidence. No selection/development/test label
access occurs in this update. This is heuristic reproducibility evidence, not
a confidence interval. Cost is saved per update.

P2 audit_fixed counts false positives among audited clean-zero MAIN rows and
false negatives among audited clean-one MAIN rows, separately by domain/task.
Jeffreys smoothing, declared clipping and frozen parameter exclusion are explicit.
Training likelihood and audit terms match independent-cell correction. Audit
calibration counts/rates are retained. No unobserved clean training labels enter
the fit. These additions leave all existing formal arms' mathematical paths
unchanged; actual per-run source snapshots preserve earlier variants.

P3 codebook/domain_codebook were already in the reviewed implementation. A single
dictionary generates slots before nonlinear STAR. Task codebook evaluates two
STAR/expert paths, so parameter matching does not establish equal runtime.
P5 coalition_router computes actual leave-one-source-out mixture predictions
through the same tower and uses negated loss for contribution targets; supervised
weights still train. It is a heuristic credit rule and can overreward redundancy.

Verdict: PASS for two targeted engineering runs (reliability update at epoch5;
audit-fixed likelihood at epoch1). Full alternatives may run after those receipts
and first respective primary development analyses. No extra boundary-test grid.

## Targeted engineering receipts

Both new cases exit0/COMPLETE_ENGINEERING. Reliability epoch5 records two disjoint
587-row halves, nontrivial finite slopes, confidence updates and6.67s extra cost.
Audit-fixed records class-specific denominators (smallest cell13 audited positives)
and frozen transition likelihood training. Verdict: PASS for full alternatives
and audit control; respective primary development readouts remain launch triggers.
