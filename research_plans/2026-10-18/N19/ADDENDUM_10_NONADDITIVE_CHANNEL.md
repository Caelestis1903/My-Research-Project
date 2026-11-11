# P2 single falsification of the axis-channel assumption

All16 profiling runs completed100epochs. Audit-only axis fit RMSE .027248 versus
independent audit .035074 (one shared audit draw, NOT four independent calibration
replications). Profile-axis minus audit-fixed loss-.006936 in4/4, AUC+.000239
in1/4. Profile-joint minus audit-joint loss-.002183 in4/4, AUC-.000744 in2/4.
Profile-nested has AUC .715551/loss .559000 versus profile-joint .714467/.559659.
Calibration improvement is real for this draw, but no stable joint-likelihood AUC
benefit appears. Prespecified assumption stress is triggered by the remaining
axis-calibration/loss effect, not by claiming an already competitive new algorithm.

One fixed non-additive measurement condition, gamma=1:
a_dt=.35 sigmoid(A_d+B_t+gamma*u_d*v_t),
b_dt=.35 sigmoid(C_d+F_t-gamma*u_d*v_t),
u=(-1,0,1),v=(-1,1). Other simulator constants remain as in main design. Preserve
the EXACT native clean population, row IDs, main/meta, audit mask and underlying
flip uniforms. Only flip thresholds differ. Validation/test stay native clean.
No use of true rates in fitting; all methods retain the same declared .35 bound.

Four arms x4seeds=16 full100epoch from-scratch networks:
audit_fixed/profile_axis (same independent-posterior/composite likelihood), and
audit_joint/profile_joint (same nested-posterior/joint likelihood). Within each
pair only audit-calibrated independent versus axis-constrained rates differ.
This directly tests whether the axis prior's finite-audit loss advantage survives
model misspecification. Do not expand all14methods or multiple gamma values.
The clean-oracle reference can be reused by exact training-objective identity;
do not count that reuse as new training or independent evidence.

Implementation: optional noise_interaction argument default0 in population and
trainer, tag noise_interaction1. Preserve old default generator exactly. Carry
variant into config, analysis cache keys, channel tables and evaluator. Comparisons
must remain within tag; no pooling primary/perturbed rows or seeds. Save true rates
as analysis metadata, never optimization inputs.

Primary self-review acceptance: gamma0 bitwise matches archived main data; gamma1
changes observed labels but preserves clean labels/IDs/audit; rates remain in(0,.35),
nonzero logit interaction; all4 arms share identical perturbed observation process;
same full budgets, no test tuning. PASS after this minimal data identity check.
