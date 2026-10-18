# P4 alternative: replicated finite-risk confidence

Designed before implementation/results for this alternative. Primary-only design
and review follow the revised start.md. Existing formal jobs are unchanged.

At epochs 5,10,... draw two DISJOINT train-meta sets, up to256 rows/domain/half.
For each raw interaction coordinate (d,t,k), evaluate BCE for target cell(d,t)
at coefficient +/-0.1 on each half. Freeze network weights during this measurement
only. Let s_A,s_B be the two symmetric finite slopes. Evidence=1 when
s_A*s_B>0, otherwise .25. Update confidence c=.9c+.1 evidence. All ordinary
training remains end-to-end. Confidence does not use validation or test labels.

Effective logits are P(c * P(v)), where P double-centers over destination domain
and task. The SECOND projection is necessary: heterogeneous confidence otherwise
reintroduces main effects. Confidence-off arm is existing interaction, with exact
same model family. Full_mix controls unrestricted capacity. This is a train-meta
reproducibility heuristic, not a calibrated posterior or unbiased risk certificate.
Near-zero but same-sign slopes may be unstable; actual results must adjudicate.

Two disjoint halves require main/meta exclusion already shared by every arm.
Use deterministic epoch/seed sampling; restore each raw coefficient exactly after
each pair. Save slopes/confidences each update, actual additional wall time and
full checkpoints. At most72 small forward passes/update (18coordinates x2 signs
x2 halves). Four seeds, both full populations, all ordinary epoch budgets.
Only launch after first interaction development readout; no proxy cancellation.

Review acceptance: zero destination margins after gating; +/- restores weights;
disjoint meta rows; absent validation/test access; confidence buffer checkpointed;
baseline branch unchanged. Review is primary self-review, not independent.
