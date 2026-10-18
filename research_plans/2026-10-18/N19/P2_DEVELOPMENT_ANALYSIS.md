# P2 complete primary/control development reading; profiling extension pending

All10methods x4seeds completed100epochs. Same noisy population,2% main-training
audits, clean validation selection for all methods; test remains deferred. Native
labels are latent truth only in this declared simulator. No natural MNAR claim.

|Method|Selected macro AUC|Selected macro LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Naive, audit replacement|.706140|.583441|.717404|
|Learned fusion, observed|.705704|.586661|.717460|
|Audit only|.592813|.700875|.612362|
|Independent channel|.712854|.575757|.718548|
|Axis channel|.712184|.574174|.717620|
|Nested independent channel|.717177|.569737|.721680|
|Nested axis composite|.717265|.569180|.721803|
|Nested axis joint|.716590|.565935|.721810|
|Independent audit-fixed|.709405|.574123|.720813|
|Clean-label privileged reference|.730898|.550470|.739844|

**Stable phenomena — MEDIUM for correction under this fixed simulator.** Joint
versus naive improves AUC in4/4 seeds(+.010450) and loss in4/4(-.017507), but
AUC interval[-.005351,.026250] is broad and train-weighted gain only+.004406.
Sparse audit-only training is substantially worse. This supports combining noisy
and audited supervision, not a new identification result.

**Stronger controls contradict a broad method claim.** Axis-minus-independent
AUC-.000670 in0/4 positive seeds, interval[-.002069,.000728]. Joint versus
axis-only AUC gain+.004406 in4/4 looks attractive, but nested support explains
that comparison: joint-minus-nested-axis AUC-.000675 in2/4, versus nested-ind
-.000587 in1/4. Axis sharing under the SAME nested composite architecture gives
only+.000088 in2/4. Joint loss is lower than nested-axis in4/4(-.003245), but its
interval[-.007588,.001098] crosses zero. Evidence of independent AUC benefit:
**WEAK**; no generic failure of the broader measurement problem follows.

**Rate recovery contradicts prediction selection.** Audit-fixed rates have RMSE
.035074; selected axis .098250, independent .099833, joint .094240. Joint's final
RMSE improves to.077806; axis worsens to.127579. Thus clean prediction selection
is not selecting accurate measurement estimates. Rates and predictors can absorb
each other's misspecification/optimization error. The additive parameter family
has8 functional channel degrees (12 raw parameters with gauge redundancy),
versus12 independent rates; parameter sharing alone does not guarantee recovery.

**Competing explanations and next experiment actually launched.** Finite-audit
variance, latent posterior/channel co-adaptation, nested-label constraints and
optimization geometry compete with genuine structural information gain. The
stronger rate fit from ordinary audited counts makes an anchored, audit-only
axis fit more discriminating than a blind channel-LR grid. ADDENDUM_08 trains
16 new full networks after frozen audit calibration, with independent/axis and
nested composite/joint controls. Calibration is conventional statistics, so a
positive result still needs a separate originality judgment.

Domain size, audit eligibility and true channel values differ together. The
smallest domain has only13 audited task1 positives; neither macro gains nor
loss scales identify size as the primary cause. Detailed per-cell/stage/seed
analysis and final disposition wait for profiling and frozen test evaluation.
Raw evidence: development_readout/*.csv, channel_recovery.csv/channel_rmse.csv,
per-run audits/receipts/trajectories; PROTOCOL_LIMITS applies.
