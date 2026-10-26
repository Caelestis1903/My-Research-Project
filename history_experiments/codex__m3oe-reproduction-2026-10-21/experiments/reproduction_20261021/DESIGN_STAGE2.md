# DESIGN SPECIFICATION — Stage 2, validation-only amendment

Design completed: 2026-10-21T04:01:47.408473+00:00. Frozen Stage 1 `DESIGN.md` is unchanged. This is the only Stage 2 design amendment; no algorithm code or experiment was implemented/run by the designer. The evidence receipts and machine-readable decision are in `DECISION_STAGE2.json`.

## 1. Research question

Two questions are worth the remaining controlled budget: (i) does amplitude scaling itself cause G's apparent validation deterioration, after matching G's per-cell gradient accumulation and zero-filled Adam update path? (ii) are learned sample-specific weights necessary for D's gains when the actual trained weights collapse almost to the constant lower bound? The design deliberately challenges the proposed mechanisms rather than searching for a better score.

## 2. Hypotheses being tested and triggering evidence

Only validation and training-mechanism files were inspected. No `test_metrics.csv`, test columns, test comparison tables, test plots, `paired_test`, or `all_seed_results` were used. Stage 1 tests may exist on disk; their existence is not evidence used to select controls.

Validation D−B macro AUC: MovieLens +0.003463763690257554, KuaiRand +0.009545570148739746, both 4/4 seeds practical positive at the already frozen 0.001 threshold. D−B macro LogLoss improves in 4/4 seeds on each dataset, but confidence intervals include zero. KuaiRand D−B pooled task-mean AUC is −0.0016029336152035834 (3/4 negative), despite positive macro AUC. G−D macro AUC is −0.0010437799705124151 on MovieLens (3/4 negative, only 2/4 at most −0.001) and −0.004017177122367377 on KuaiRand (3/4 practical negative); neither paired 95% interval excludes zero. MovieLens G−D macro LogLoss increases +0.005769011759389264 in 4/4 seeds with the paired interval above zero. These are validation screening observations, not final test conclusions.

Training probes at epoch 5: all 16 D/G dataset×seed runs have mean weight 0.10000001639127731. Maximum sample weight SD across these probes is 6.376330019541854e-11; maximum mean specificity is 2.205500539143035e-11. Later probes show small departures from the lower bound. This supports a falsifiable competing explanation: sustained informative sample discrimination may not be needed; transient early gradients, near-constant loss scale, augmentation or their interaction may account for the gain. It does not prove that any of those explanations is correct. Training-only smoke previously showed small prediction-update differences between G(tau=0) and ordinary backward, so full-trajectory control is necessary before pure scaling attribution.

H-A: G's unfavorable contrast is attributable predominantly to amplitude scaling, not the G implementation path.

H-W: The dynamic learned specificity weighting trajectory is necessary for D's gains. This includes its early transient representation gradient; it is broader than the claim that late weights are informative.

## 3. Variables changed and exact matrix

| Stage 2 ID | Existing implementation parent | Sole controlled change | Primary comparisons |
|---|---|---|---|
| Gtau0 | G-supplied-ported | tau=0, while keeping G gradient computation, scale calculation, summation and zero-filled parameter gradients | G−Gtau0 for amplitude; Gtau0−D for implementation path |
| Dconst01 | D-supplied-ported | Replace all real/copy loss weights by the fixed float32 scalar 0.1, with no gradient through weights | Dconst01−D for necessity of the dynamic weighting trajectory; Dconst01−B as contextual bundle comparison |

Both controls on both datasets, seeds 2022,2023,2024,2025: **16 additional complete runs**, exactly two controls per dataset. Stage 1 O/B/D/G results are reused as frozen matched references; do not rerun them under changed source or reinterpret an old pipeline run as a formal control.

## 4. Variables held constant

Everything in Stage 1 remains frozen except the explicitly changed variable above: audited data byte identities and train-only/OOV encoding; full natural MovieLens and processed KuaiRand; row-order splits; same original M3oE backbone and fixed coefficients; identical canonical initial backbone and auxiliary states; loader/noise seed schedule; scarce-domain selection; copy count; noise std=0.1 and embedding target; all-parameter norm definition; Adam lr/decay/betas/eps; batch40960; six-cell group-mean sum; no scheduler/AMP; 50-epoch maximum, patience10, strict validation macro AUC best and earliest tie, always restored best. Every Stage 2 initial state and split/feature hash must match its Stage 1 parent. Early stop is valid completion; no seed may be dropped for its direction.

## 5. Mathematical / algorithmic definition

Retain D/G's generated batch and differentiable real specificity weights from the frozen design. Let L_k(theta) be the six weighted cell losses, g_k=grad L_k, G_k=||g_k||, and k in task-major order.

**Gtau0** executes the actual G branch with tau=0. Thus nonzero groups have s_k=1, the final assigned gradient is the per-cell sum g=sum_k g_k, and every trainable parameter receives a Tensor gradient, including zeros for parameters disconnected from all losses. Keep the original G parameter order, precision, separate autograd passes, norm/scale calculation and optimizer. Do not implement it by calling the ordinary D backward branch; that would eliminate the very path under test. Gtau0 uses dynamic specificity, copy and noise exactly as G. Probe scales must all equal one (including zero norm groups). Its model module/initial-state receipt matches G.

**Dconst01** calls the same D model and augmented forward, including specificity and complexity for observation, but uses w_i=0.1 for every real and copied sample, with requires_grad=False. L_C=sum_k mean_{i in augmented cell k}[0.1*BCE_i]. Use ordinary summed backward and D's `grad=None` semantics. No extra loss, detached-specificity auxiliary loss, weight normalizer, moving mean, data-dependent constant, coefficient schedule, or warmup is added. Specificity/complexity modules remain present with the same initialization; they are disconnected from this loss and naturally receive grad=None. The predictor still trains through its original embedding/backbone path. This intentionally removes both sample discrimination and the direct specificity-weight gradient into embeddings. Retaining the unused module is an architectural control, not evidence that it has trained.

The exact 0.1 constant is fixed by the supplied method's mathematical lower bound and the observed training collapse, not by any validation/test hyperparameter sweep. Do not select 0.35, the global historical mean, a per-seed mean, or an epoch-dependent historical schedule. Dconst01 does **not** match D's first-epoch average scale, and cannot separately identify transient scale versus transient weight-network gradients. Its interpretation is whether the full dynamic weighting mechanism is necessary relative to a simple prespecified constant control.

For any same scalar metric and paired seed the decomposition is exact: (G−D)=(G−Gtau0)+(Gtau0−D), even though selected checkpoints may differ. The decomposition concerns performance contrasts, not additive mechanistic effects on the nonlinear optimization trajectory.

## 6. Exact affected modules/files

Coder may minimally extend `experiments/reproduction_20261021/{protocol.py,run.py,schedule_runs.py}` and requested package/example entrypoints as necessary for two explicit control identities and effective configuration receipts. No new model/backbone is needed. The callable path must preserve Stage 1 O/B/D/G behavior. Gtau0 maps to G model/gradient path with effective tau=0; Dconst01 maps to D model/ordinary gradient path with effective weight_mode=constant, constant_weight=0.1. Never alias these to D/G run IDs in result directories.

Add Stage2 coder report, review and smoke artifacts; never modify frozen DESIGN.md, historical source snapshots, Stage1 output files or earlier reports. Save new source hashes; Stage1 results remain associated with their original hashes. A separate Stage2 runner or explicit experiment-stage flag may avoid breaking the Stage1 `formal` classifier, which currently requires tau=.5. A valid intended tau0 control must be classified as Stage2 formal, not silently sent into a pipeline tree.

## 7. Implementation requirements and minimal checks

One matched real-data batch for each control is sufficient before full execution. Assert Gtau0's forward predictions/weights/cells match G at the common initial state and identical noise seed, and its computed scales are exactly one. Compare Gtau0 with manually summed per-cell gradients (including assigned zeros); audit the known difference from D's ordinary summed backward and report gradient/update and prediction differences by backbone/auxiliary group, without claiming numerical equality where Adam amplifies tiny accumulation differences.

For Dconst01 verify identical parent D forward predictions and generated batch, every weight exactly the dtype representation of 0.1, all six losses equal 0.1 times their unweighted augmented cell means, no specificity parameter grad, and a nonzero backbone update. Verify real/copy counts and augmentation noise identity against D. Full data/epoch pipeline, earlystop/val-best rules and output schema inherit Stage1; do not add generic tests.

Run controls in new attempts under an explicit runtime stage2 tree. Resume requires full state and identical hash; otherwise use a new attempt rather than overwrite. Record exactly one accepted completed attempt per dataset/control/seed. Code review must PASS or PASS WITH FIXES with all blocking fixes applied before luna-runner starts.

## 8. Ablations required, exclusions, and budget rationale

Only the two controls above are authorized in this amendment. Gtau0 resolves a mandatory attribution ambiguity documented in the original design. Dconst01 is an explicit narrow amendment to the previously listed removal controls, justified by universal observed weight collapse; it is more discriminating against the informative-weight explanation than w=1, which simultaneously removes the weighting network and increases the late loss scale about tenfold. Do not substitute a w=1 control while still describing it as loss-scale matched.

Noise0 / no-copy would help isolate augmentation, but currently leave the stronger observed weight-collapse alternative unanswered. The real/generated cosine probes do not establish augmentation necessity; that limitation will remain explicit. Shared-only norm, tau sweep, noise sweep, complete historical VUG/mask, momentum/curriculum, additional objectives, hyperparameter optimization, and any new combination arm are excluded. Original G already contains D's augmentation/weighting, so D+G is not an independent untested factorial combination.

## 9. Expected logging and decision rules

Inherit all per-seed train/validation/test metrics, checkpoint/source/optimizer/data receipts, mechanism logs, time and memory from Stage1. Log effective `stage`, `control_id`, `parent_arm`, `gradient_path`, `tau`, `weight_mode`, `constant_weight`, whether each auxiliary parameter has grad/state, and its selected checkpoint. Gtau0 must log scales=1; Dconst01 must log w mean/min/max=0.1 and std=0 within representation tolerance, and auxiliary gradient norm=0. Keep unweighted real BCE distinct from optimized loss. No need to save an optimizer at every epoch.

Before accessing **any** Stage2 test table, freeze the Stage2 collection and comparisons to D/G/B, using exactly these IDs, without choosing a winner to suppress others. Report all Stage1 and Stage2 test results transparently once the experiment matrix is frozen/completed. Tests must never trigger a third control or new hyperparameter. The orchestrator may inspect validation while monitoring completion; this does not authorize changing the frozen matrix or stopping unfavorable seeds.

Analysis unit is seed, n=4; report raw paired differences, mean, SD, signs, and descriptive paired t intervals. The pre-existing practical macro AUC band remains ±0.001. For attributing a repeatable practical negative/positive contrast, require at least 3/4 seeds beyond the corresponding 0.001 band; report interval uncertainty independently rather than relabeling a crossing-zero interval as proof. A contrast with mean inside the band does not demonstrate equivalence.

For a conservative Dconst01 noninferiority description relative to D, require the lower paired 95% interval for macro AUC >−0.001 **and** the upper paired 95% interval for macro LogLoss <+0.002; these are prespecified descriptive tolerances, not a powered equivalence trial. Always inspect worst cell and each domain/task regardless of this screen. If these checks fail, report uncertain or adverse rather than declaring learned weighting necessary. A seed-consistent practical D advantage over Dconst01 falsifies constant sufficiency but still cannot distinguish transient loss scale from weight gradients. The unchanged protocol and comparisons allow corresponding test contrasts to be reported after the freeze; validation itself cannot serve as independent confirmatory evidence.

## 10. Acceptance criteria and Stage 3 stop/continue rule

Reviewer independently checks this spec, code, coder report, new arm mappings, formulas and minimal checks. Blocking failures include Gtau0 using D backward; Dconst01 retaining a gradient through specificity weights or using w=1; initialization/data/noise mismatch; tau0 misclassified as nonformal; test-informed selection; mutable or overwritten Stage1 outputs; wrong best-checkpoint/metric logic; or omitted seeds.

**Normal completion and stop:** finish these 16 runs and the required original-plus-fixed comprehensive analysis, report remaining attribution limits, and stop additional training for the current reproduction. Null, adverse, inconsistent or ambiguous results are scientifically valid endpoints. No result automatically warrants an open-ended Stage3.

**Stage3 need assessment (future explicitly reviewed design only):** only consider objective/combination research if a surviving component has practical positive macro AUC at least 3/4 seeds on each dataset and its validation curves/per-cell trade-off leave a specific objective-dependent scientific question that cannot be answered by the current controlled contrasts. Evidence that a copied/noise bundle is still confounded with constant loss scaling prioritizes a simple matched 0.1×B control over a large Equal/Sample matrix. No independent D+G experiment is meaningful for these supplied definitions. Merely observing opposite macro/pooled movement, a positive test maximum or a wide CI is not a continuation trigger. The assessment can conclude “objective dependence unmeasured” and recommend one discriminating future experiment without running it. Any genuinely required Stage3 must receive a new finite design/review and independent authorization scope before implementation; it is not silently included in this amendment.

## 11. Conditions falsifying hypotheses and interpretive limits

If G−Gtau0 is null while Gtau0−D reproduces G−D deterioration, the claim that amplitude scaling is responsible is falsified or strongly weakened. If Gtau0−D is unstable or material, G−D cannot be treated as a pure amplitude contrast. If G−Gtau0 remains adverse while Gtau0≈D within measured precision/practical band, evidence favors adverse scaling under this protocol; it does not establish a universal failure.

If Dconst01 is noninferior or repeatedly close to D with comparable per-cell/calibration behavior, ongoing learned specificity is unnecessary within this measured protocol, contradicting an informative-sample-weight necessity claim. It does not prove that noise is the effective factor. If Dconst01 loses, early transient scale or weight-gradient learning are competing explanations, not automatically meaningful specificity estimation. If both approaches retain macro/pooled conflict, report the explicit aggregate trade-off rather than selecting the favorable metric.

The final analysis retains every required domain/task/seed/size/difficulty/baseline/model/training-stage dimension and labels evidence STRONG/MEDIUM/WEAK. Natural-domain contrasts alone remain WEAK for causal size explanations. This amendment adds no manufactured historical data and no assumptions that small means difficult or that Adam update balancing follows from raw-gradient amplitude scaling.
