# P2 complete development reading: calibration and assumption falsification

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

## Completed audit-profile alternative and non-additive falsification

All72 P2 formal networks finished100epochs:40 primary/control,16 frozen-profile,
16 non-additive stress. Four optimization seeds each; no test performance read.
These are end-to-end trained networks, including frozen calibrated channels with
all predictor parameters trainable. The frozen component is the measurement
channel, not a backbone/head-only substitute.

|Profile method, additive truth|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Frozen axis/composite|.709643|.567187|.720204|
|Frozen axis/nested composite|.715551|.559000|.721428|
|Frozen axis/nested joint|.714467|.559659|.720078|
|Frozen independent/nested joint|.715211|.561842|.721020|

The audit-only axis rate estimate has RMSE .027248 versus independent .035074.
These estimates repeat identically across network seeds because the audit draw
is shared; they are ONE calibration realization, not4 independent replications.
Profile-axis minus independent-audit-fixed loss=-.006936 in4/4 seeds, but AUC
+.000239 in1/4 and weighted AUC-.000608 in1/4. Profile-nested minus trainable
nested-axis loss=-.010179 in4/4, interval[-.018859,-.001500], while AUC-.001714.
Anchoring the nuisance channel improves calibration here without establishing
better ranking. This separates co-adaptation from the axis prior itself.

Joint likelihood has no residual advantage: frozen profile-joint minus
profile-nested AUC=-.001084 and loss=+.000659. Versus frozen independent-joint,
profile-joint AUC=-.000744 and loss=-.002183, with intervals crossing zero.
A new joint-identification algorithm is not supported. Conventional audited
likelihood calibration is a sufficient competing explanation for the loss gains.

ADDENDUM_10 then falsifies the structural assumption with ONE declared
non-additive channel (gamma1), same clean rows, audits, split and flip uniforms:

|Method, non-additive truth|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Independent audit/composite|.715606|.562677|.722345|
|Axis audit/composite|.710750|.587145|.718594|
|Independent audit/nested joint|.717587|.559438|.723237|
|Axis audit/nested joint|.715098|.577269|.721929|

Axis RMSE worsens to .074270 versus independent .034253. Composite axis-minus-
independent AUC=-.004856 in0/4 positive seeds, interval[-.010206,.000493]; loss
=+.024468 in4/4, interval[.003293,.045643]. Joint-axis minus independent-joint
AUC=-.002488 in0/4, loss=+.017832 in4/4, interval[.005843,.029820]. The direction
of the earlier loss benefit reverses under misspecification. Evidence for this
conditional failure is **MEDIUM** across4optimization seeds, not a population-
wide theorem. The deterministic increase in rate error on this draw is directly
measured (**STRONG** descriptive evidence).

The medium-size domain0 (N32775) experiences the largest composite loss rises:
+.077361 task0 and+.033319 task1, each4/4. Task1 AUC drops-.012714,4/4 with
interval[-.016784,-.008643]. The smallest domain1(N2850) has mixed AUC changes
(-.012169/+.004367), and the largest domain2(N142500) loses AUC
(-.001880/-.003724) with weaker intervals. The channel interaction is confounded
with domain identity, and the smallest-domain crossing term was deliberately
zero. Thus size is not a plausible uniquely identified main cause. There is no
stable small-gain/large-decline effect in these paired controls.

All final-epoch losses are much worse than selected losses (e.g. non-additive
independent composite1.470514, axis1.478652; joint1.552820/1.520953). Final joint
axis has AUC+.001610 in3/4 but weighted AUC-.002608 in3/4 and loss mean driven
by one seed. This contradicts a stage-invariant ranking conclusion and does
not rescue selected generalization. Operational times for stress arms are
155.8–175.0s under queue contention; rate freezing does not establish a speedup.

Disposition at development stage: sufficient evidence to stop tuning the tested
axis/joint families; broad unknown MNAR identification remains open. Do not claim
an independently competitive contribution from calibrated loss improvement. The
most discriminating future study would vary independently randomized audit
realizations and crossed channel misspecification while holding the native
population and network fixed; naturally collected randomized reveal/audit data
would be necessary for real observation-policy claims. These are future studies,
not unrun parts of the present declared matrix. Frozen test readout will assess
consistency without further method selection.
