# Gradient and Momentum Diagnostics

Each batch: actual B/row-order hash, clean/view loss and denominators, natural support, active/None counts, complete parameter/EMA update counts, cap coefficient/activation, ||u_c||/||u_r||/||b||/||lr*b||. SHUFFLE also logs its OWN post-preconditioner/post-cap target norm, applied norm, relative error and independent random key; zero target requires exact zero.

At fixed batches in epochs1/10/25/40/50 record g_c/g_v/r/q/m/v norms, cosine, per-parameter step range and exact source-group summaries. INSTANT logs current residual with no q history, and cap/amplitude changes. No full gradient/tensor disk snapshots.

Verify same-theta data gradients, all-row natural normalization, target-only loss, delta detach with trainable embedding, no probe parameter .grad, WD once, current clean None skip/zero-tensor update/re-entry per-parameter n_p, one complete parameter transaction then one EMA/RAM commit, gamma0 source equality and correct SHUFFLE norm domain. Norms/cosines are diagnostics, never AUC substitutes.
