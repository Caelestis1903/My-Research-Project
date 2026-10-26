# Critical amendment 02 — training field versus predictive risk derivative

After raw measurements, before substantive result analysis, source review found
that P4/P6 risk-response language requires the **complete prediction-function**
Jacobian. Routing reads `emb.detach()`: its training derivative omits a genuine
forward sensitivity. This has a direct path to changing decision signs and
directional covariance, and is therefore Critical, not generic defensive testing.

Preserve original `measurements/seed*/` and archive the original probe/readout
source. P3 uses those valid frozen features and continues unchanged. Re-run only
gradient measurements in `measurements_functional/seed*/`, skipping repeated
feature inference. The same random draws are retained by advancing the same RNG.

P4 saves a 24×24 Gram: A/B/C training fields (18 vectors) plus six complete
prediction-function C gradients. Training-field interaction stability and errors
still use A/B/C. Risk-response decision oracle uses functional C and identical
fixed source actions mean(A,B). P6 uses normalized functional C risk directions.
All source sample gradients remain actual training fields, appropriate to the
updates that would be applied. Fixed objective mean preservation is unchanged.

Implementation: read-only hooks capture `skip_conn` and `star_mlp` outputs and
reconnect their sum as gate input only in the functional diagnostic forward.
Assert bitwise-identical gate input values; remove hooks after six gradients.
No checkpoint or historical implementation is mutated. No additional tuning,
seeds or optimizer experiment is introduced.

Review: primary self-review **PASS WITH FIXES**, blocker corrected. Check hook
reconnection against one finite difference of a shared embedding perturbation
with autograd-only versus complete Jacobian; this is the sole new basic check.
Full measurements are authoritative for risk-direction conclusions; old
stop-gradient-direction measurements are superseded, not independent evidence.
