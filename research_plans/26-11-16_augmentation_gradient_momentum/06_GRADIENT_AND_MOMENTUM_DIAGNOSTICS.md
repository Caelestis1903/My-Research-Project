# Gradient and Momentum Diagnostics

At fixed early/mid/late steps (epochs 1,10,25,40,50), record clean/view/residual norms, residual-clean cosine, clean Adam m/v norms, q norm, preconditioned update norms before/after cap, cap activation, parameter delta, EMA delta, and parameter-group summaries where source mapping is exact.

Use deterministic fixed batches for expensive decomposition; do not retain full gradients every batch. Verify same theta/support/normalization, task-specific target loss, detached deltas, no probe parameter gradients, no double WD, no residual-only grad=None update, one parameter and EMA update per batch, gamma=0 equivalence and SHUFFLE norm match.

Cosines/norms/state magnitudes are diagnostics, never AUC substitutes. Mechanism claims require paired five-seed endpoint evidence.
