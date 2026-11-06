# Frozen performance, contradictions and novelty disposition

The full856-model frozen readout supports a bounded mechanism audit, **not a generally beneficial new rescue algorithm**. Several consistent architecture-control effects are real on these fixed populations, but activity restoration, selected ranking, probability risk, stronger baselines and population generalization do not agree sufficiently for a broad method claim. Publication disposition is **NO-GO for universal collapse/a novel general remedy; CONDITIONAL for the bounded empirical mechanism study**. This agrees with the development decision frozen before test.

Every reported test model uses its original earliest strict maximum selection-AUC checkpoint. Final checkpoints are diagnostics, never substituted because their test result looks preferable. Test evaluation changes no optimizer parameters. FINAL_AUDIT verifies856 valid receipts,25,920 unique cell rows and1,712 readable prediction archives with CRC; ARTIFACT_INDEX binds their local files. The repeatedly used historical N19 test is exploratory, not an untouched confirmation population.

## Selected risk and the important contradiction

All differences below are candidate minus its native selected model, using the same physical tasks/domains and four paired optimization seeds. Native optimizer comparisons use Adam as reference. Higher AUC and lower LogLoss are favorable.

|Complete comparison|Development ΔAUC (positive seeds)|Test ΔAUC (positive seeds)|Test ΔLogLoss|
|---|---:|---:|---:|
|KR fixed, global AdamW|+.000836 (2/4)|+.003817 (3/4)|+.000682|
|KR fixed, local factor AdamW|+.007599 (4/4)|+.000591 (1/4)|+.000861|
|KR STAR task1, local factor AdamW|+.009015 (4/4)|+.015709 (4/4)|+.009671|
|ML STAR task0, local factor AdamW|-.002829 (0/4)|-.003045 (0/4)|+.058007|
|ML STAR task1, local factor AdamW|-.002001 (0/4)|+.003951 (4/4)|+.013908|
|ML STAR task0, no-modulation retraining, Adam|+.004117 (4/4)|+.013746 (4/4)|-.055984|
|ML STAR task1, no-modulation retraining, Adam|+.003617 (4/4)|+.017369 (4/4)|-.041740|
|ML HAMUR task0, no-adapter retraining, Adam|+.044933 (4/4)|+.022595 (4/4)|-.546250|

The most direct fixed-model rescue candidate loses its stable seed pattern on test: KR local AdamW's development interval [.003292,.011906] becomes [-.010073,.011255], with only one positive seed. Against global WD0 its test AUC difference is-.002241 (1/4 positive), and versus global AdamW-.003225 (2/4). Its intervals against MMoE/PLE also overlap zero. This materially weakens performance usefulness; it must not be hidden behind the clear functional restoration.

There are also positive effects that deserve preservation. KR STAR task1 local AdamW improves test AUC in all seeds, but worsens native LogLoss and has a wide AUC interval versus global AdamW including zero. ML STAR task1 changes AUC sign across development/test while worsening loss on both. These are contradictory task/split outcomes, not a general AdamW benefit. ML nominal AdamW shrink rounds to identity in this backend, so its global WD0 equivalence further constrains optimizer interpretation.

No-modulation retraining on ML STAR improves selected AUC and LogLoss on development and test, with four positive AUC signs. **MEDIUM** evidence for this fixed-population control effect. Nevertheless test task0 still trails same-task fixed/Adam by.003478 (all four signs negative; interval includes0), while task1 differs by-.000661 with mixed signs and worse LogLoss+.010299. Both can be useful simplification evidence relative to their native STAR adaptation without establishing a new competitive MDMT algorithm. No-modulation retains legitimate biases and the auxiliary route; it is not removal of the entire network.

HAMUR's no-adapter gain is relative to a weak public-code-style native baseline. On test the resulting task0/task1 models remain below same-task fixed by.036314/.038747 AUC (all four seeds), with loss higher by.062037/.155681. Public mixed-batch BN/DN, shared factors and inference-batch dependence limit paper-wide interpretation. Gains over this baseline do not establish leading performance or novelty.

## Natural baselines and training stage

Whole two-task native models under Adam, test four-seed means:

|Population/model|Selected AUC|Selected LogLoss|Final AUC|Final LogLoss|
|---|---:|---:|---:|---:|
|KR fixed|.666951|.442080|.648995|.487054|
|KR PEPNet|.671633|.436792|.668169|.455625|
|KR MMoE|.671033|.440934|.631769|.536264|
|KR PLE|.668671|.441406|.635705|.540982|
|ML fixed|.733443|.533803|.675340|1.165258|
|ML PEPNet|.727846|.598939|.700310|1.230099|
|ML MMoE|.729497|.534751|.658341|1.257753|
|ML PLE|.726174|.550527|.660080|1.234850|

These point means are not an independently confirmed leaderboard or a pure architecture causal estimate. Full paired signs/intervals, both optimizers and all task/domain cells remain in the raw tables. Native single-task STAR/HAMUR comparisons use the fixed model's same head/cell rather than its two-task macro. The strong-reference export includes fixed/MMoE/PLE at both optimizer recipes.

KR fixed AdamW's selected test AUC gain is+.003817 with interval[-.007395,.015028]. At final it becomes-.018722 (4/4 worse), while LogLoss rises.140856 (4/4). ML fixed AdamW selected AUC is-.001824, interval[-.007334,.003686], and final LogLoss rises.174350 in all seeds. Increasing path activity alongside worse final risk contradicts a simple restoration-is-benefit story. Larger LogLoss is probability-risk deterioration, not evidence of a specific ECE/calibration decomposition.

Original source-style protocol test AUC is.653279(KR)/.707611(ML), below fixed. Original_sample largely closes that gap (.668193/.733661). Equal weighting, overlapping fusion updates and protocol adaptations are combined differences; this is not proof the original paper architecture fails. Private MMoE/PLE controls often lose selected risk, especially KR and ML MMoE. For example MMoE private/Adam test AUC falls.013010(KR) and.008952(ML), all seeds negative, with worse LogLoss. Shared computation is useful in these cases; universal harmful sharing is falsified.

## Fixed-state neutralization is not retraining

Selected native Adam test, neutral minus natural:

|Intervention|ΔAUC|ΔLogLoss|Meaning|
|---|---:|---:|---|
|KR fixed modulation-off|0|0|All four selected test predictors are exactly unchanged|
|KR fixed bypass-off|-.166654|+.279231|Bypass remains influential and useful|
|KR STAR task0 modulation-off|-.053414|+.293265|Independent STAR has useful native ranking contribution|
|ML STAR task0 modulation-off|-.008833|-.041521|Ranking and probability risk disagree|
|ML PEPNet EP gate-neutral|+.005567|-.060845|An active gate can worsen selected risk|
|ML PEPNet PP gate-neutral|-.002583|-.051146|Same-family interventions can disagree by metric|
|ML MMoE shared-output-off|-.229665|+.119662|All shared experts removed; differs from PLE's retained private paths|

ML STAR no-modulation **retraining** gains AUC while zeroing the already trained native module reduces AUC. Training adaptation and architecture/regularization differences matter. PEPNet gate1, HAMUR residual0, MMoE expert0 and PLE shared-expert0 are different estimands; probability-change magnitude is not a common cross-model quality scale. Native numerical inactivity, functional usefulness at fixed weights and useful from-scratch replacement are three separate questions.

## Cells, weighting, size and long training

KR local fixed AdamW illustrates why macro AUC alone is insufficient. Its test macro ΔAUC is+.000591 (1/4 positive), original-size-weighted cell ΔAUC+.002550 (3/4), but weighted LogLoss worsens+.004047 (3/4). Per-cell AUC changes for domain0 task0/1 are+.001616/-.008643, domain1+.003131/+.003216, domain2+.002087/+.002140. The strong development gain on domain0 task1 reverses on test. This is not a stable small-domain-gain/large-domain-loss pattern. Weighted-cell AUC here averages within-domain AUCs using original row weights; it is not pooled ROC AUC.

G4 matched exposure/update control, fixed/Adam selected test:

|Population/unique target fraction|Target-domain ΔAUC|Other-domain ΔAUC|Target ΔLogLoss|
|---|---:|---:|---:|
|KR25%|-.043272|-.010225|+.073660|
|KR50%|-.022109|-.005904|+.052860|
|ML25%|-.027805|-.021290|+.047282|
|ML50%|-.013834|-.009809|+.020753|

Every target AUC sign is negative. **MEDIUM** evidence that unique information/replay affects risk under this fixed-subset intervention, also seen in active-path STAR. It does not explain why KR fixed remains inactive while ML stays active. Target domains remain the original largest even at25%, only one nested subset per fraction is used, and repeat sampling changes replay noise. Natural size is not intrinsic difficulty; the test task/domain baseline ordering also differs from development. No general small/large trade-off or universal size irrelevance is established.

All16 ML long runs select the same epochs as the500-update baselines, with exactly zero selected risk differences. Final2,350-minus500 test AUC differences are-.031095/-.021080 for fixed Adam/AdamW and-.049954/-.026322 for STAR task0; all seed signs are negative. Corresponding final LogLoss increases are.885752/1.050329/2.500233/2.492379. Training stage is a strong repeated phenomenon. Fixed modulation remains nonzero at2,350 steps, so update count alone is insufficient to explain the KR/ML difference.

## Novelty and evidence disposition

|Axis|Supported conclusion|Unconfirmed claim|
|---|---|---|
|Structural scope|STRONG bounded KR M3oE suppression; independent STAR and other families are counterexamples|Universal conditional/shared-path collapse: NOT_ESTABLISHED|
|Mechanism|STRONG local coupled-L2 group intervention evidence; moment history, bypass and scaling matter|A general finite-time Adam theorem or norm-ratio criterion: NOT_ESTABLISHED|
|Selected predictive value|MEDIUM for particular repeated controls; WEAK for a general useful rescue|Consistent AUC+loss benefit against strong baselines across populations/tasks: NOT_ESTABLISHED|
|Prior art/novelty|AdamW/WD0, Hadamard sparse-penalty geometry and STAR+ alternatives cover the tested remedies|A new algorithm/theorem with competitive confirmed value: NOT_ESTABLISHED|

[AdamW](https://arxiv.org/abs/1711.05101), [Hoff's Hadamard sparsity work](https://doi.org/10.1016/j.csda.2017.06.007), [Smoothing the Edges](https://doi.org/10.1007/s10994-026-06997-0) and [Star+](https://arxiv.org/abs/2406.16568) constrain novelty independently of our measured outcomes. The scale-minimization algebra is not presented as a new result. Gradient Starvation and normalization/rotation literature are competing explanations, not identical theorems for this nonlinear MDMT graph. See LITERATURE_LEDGER for primary sources and fidelity boundaries.

No extra seed/data grid was triggered after test. Development had already selected no novel broadly useful algorithm worthy of conditional new-population confirmation. Test did not change models, optimizer coefficients, budgets or selection. Independent temporal/user population confirmation remains necessary for a population-general method claim. The most discriminating future experiment is the preregistered function-preserving factor-gauge/regularizer study in OPEN_PROBLEMS, not another outcome-driven grid.

All856 training tasks are actual separate from-scratch executions, not856 independent statistical observations. The72 aliases were preregistered equivalences. Additional executed ML recipes can be numerically identical (12 global AdamW/WD0 final model-SHA pairs); they do not multiply the evidence count. The statistical repetition unit remains four optimization seeds on one frozen population; t intervals are descriptive, unadjusted for multiple comparisons, and cells/tasks are correlated.
