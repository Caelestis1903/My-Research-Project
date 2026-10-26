# Production batch weights and composition

Read-only verification covers all48 completed formal runs: fixed-reference V2 (16), V4 calibrated/B2 (24), and MovieLens V9 (8). Every stored batch's ESS, weight variance and maximum weight were recomputed from its exact physical-domain counts and configured p/q; total examples/steps and producer hashes also match. The16 calibration-only pilots are excluded. Source: experiments/v3_batch_composition_20261016.py; numeric table: batch_composition.csv; producer/source hashes: receipt.json.

| Environment/arm | Large-batch ESS/n limit | Observed per-run mean ESS/n range | Var(w) | max(w) |
|---|---:|---:|---:|---:|
| KuaiRand SN | 1 | 1–1 | 0 | 1 |
| KuaiRand SB | .435695 | .435644–.435743 | 1.295181 | 2.607851 |
| MovieLens SN | 1 | 1–1 | 0 | 1 |
| MovieLens SB | .494478 | .494280–.494555 | 1.022336 | 2.400000 |

These telemetry weights are the float64 p/q values; actual production loss uses the existing FP32 tensor and audited numerical tolerances. Training computes the mean of weighted losses over batch size, not a ratio normalized by the sum of weights. ESS is a diagnostic and does not change the objective.

For WR IID draws at a fixed theta, E[w]=1 and E[w²]=Σ p_d²/q_d. The weight-based ESS fraction converges to 1/E[w²]; this is not exact finite-batch expected ESS. Gradient covariance instead equals

Cov(g_hat_q) = (Σ_d p_d²/q_d E[g_i g_iᵀ|d] − G*G*ᵀ) / B.

The domain gradient second moments matter. Weight ESS does not determine covariance trace, orientation, update quality or AUC. The V1 observed trace ratios2.316–2.752 are separate full-gradient measurements; they are not obtained by substituting inverse ESS. Doubling a full IID batch at the same theta halves that covariance, while production B2 trajectories also change theta, update count and the final partial batch. The IID expression is not a claim of conditional unbiasedness for PERM.

The domain-size fractions are p_KR=(.086127,.869284,.044589) and p_ML=(.184,.016,.8). Their max/min ratios are about19.5 and50, while Σp² is about.7651 and.6741 respectively. Thus the environment called “more imbalanced” depends on the chosen scalar. A single imbalance label cannot replace the full counts, label support, difficulty proxies and matched performance readouts. No downstream mechanism or cross-environment outcome is inferred from this table alone.

**STRONG** evidence for the recorded batch-weight/composition identities, **WEAK** evidence for a unique explanation of downstream performance without the remaining interventions. WR/PERM with identical domain-count sequences also have identical per-batch weight ESS, despite different within-domain rows; ORDER preserves the exact batch/ESS multiset while permuting its temporal sequence. Those numerical identities must still be verified on the completed variant artifacts.
