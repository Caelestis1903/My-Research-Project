# Optimizer mechanism and falsification evidence

The best-supported mechanism is a **configuration-specific suppression of the KR M3oE input-weight product by coupled L2 in Adam**. It is not a universal failure of STAR, conditional modulation, task sharing or shared experts. Preventing suppression does not establish a general improvement in selected AUC and LogLoss. Evidence strength is STRONG for the local functional observation and controlled attribution, MEDIUM for individual repeated performance effects, and WEAK for a universal useful remedy. All numbers in the mechanism tables below use development probes/metrics; frozen full-test risk is reported separately in PERFORMANCE_AND_NOVELTY and the test_readout tables.

## What disappears

The intervention zeros only the applied multiplicative input-weight matrix of M3oE. Additive biases, the remaining STAR MLP, bypass, experts and heads remain legitimate contributors. On KR fixed/Adam, selected predictions are bitwise unchanged by this intervention in24/24 seed/domain/task probe cells, across512 rows per physical domain. The effect is also zero at updates200/500/1000. At final,18/24 cells remain exactly zero and the overall mean absolute probability change is5.85e-9, maximum4.77e-7: there is a tiny late rebound in one seed. This supports numerical inactivity at the measured checkpoints, not an irreversible absorbing-state theorem. The selected full-main intervention still changes probabilities by.0281 on average, and bypass-off by.2701; an entire-network-death interpretation is false.

Original source-style KR training also attenuates modulation: selected mean1.60e-7 (maximum5.48e-6), final exactly zero in24/24 cells. The original_sample bridge is approximately2.31e-9 selected and7.87e-9 final. These are three protocols on a related graph, not three independent architecture replications. On ML fixed/Adam the selected/final means are .002303/.000985, with no exactly zero cells.

## Controlled long-training interventions

Each row below is four seeds on the same KR fixed model/population. Nonmodulation groups keep native Adam/L2 except in global interventions. ΔAUC/ΔLogLoss are candidate minus native, using each run's frozen selected checkpoint. Path sensitivity is measured on a fixed development probe.

|Intervention|Selected mean absolute Δp|Final mean absolute Δp|Selected ΔAUC|Selected ΔLogLoss|
|---|---:|---:|---:|---:|
|Native Adam, L2 on both factors|0|5.85e-9|0|0|
|Remove shared-factor L2|2.86e-5|1.20e-6|+.007870|-.002918|
|Remove domain-factor L2|8.52e-6|8.75e-7|-.000528|-.000850|
|Remove both factor groups' L2|.031587|.071141|+.003000|-.001388|
|Local AdamW on both factor groups|.035344|.071639|+.007599|+.001303|
|Global Adam WD0|.028265|.092102|+.001570|+.000676|
|Global AdamW|.029856|.086572|+.000836|+.001621|

Removing only one factor's decay does not reproduce both-factor restoration. Removing both provides a causal intervention on the relevant groups while leaving the remaining optimizer protocol fixed. This converges with common-state next-step evidence. However selected AUC effects are heterogeneous: only local AdamW has all four positive signs with an interval excluding zero versus native fixed Adam ([.003292,.011906]); it worsens LogLoss. Its paired AUC intervals versus global WD0, global AdamW, MMoE and PLE all include zero. A restored path is not automatically an algorithmic rescue.

## Local counterfactuals and state history

The48 planned native fixed/STAR diagnostic runs reuse model weights and Adam moment buffers at actual saved checkpoints. They compute a train-meta eval-mode gradient, hypothetically update only factor parameters with Adam-L2, zero-L2 Adam or AdamW, record loss/probability changes and restore the parameters. These are one-step common-state counterfactuals, not alternative optimizer histories or full training results.

At KR fixed initialization, mean shared norm changes are -1.3193 under Adam-L2 and +.01966 under zero-L2/AdamW. Initial BCE changes are about -6.02e-6 versus -4.69e-5 on this meta probe. By update50, even the zero-L2 hypothetical step shrinks the norm (about-.0714) because it retains moments accumulated under the native trajectory. At the collapsed selected state all three next-step rules have exactly zero measured probability effect. A single late switch does not instantly repair the path; this does not prove that prolonged switching/reinitialization could never recover it.

The first training update's candidate L2-vector/data-gradient norm ratios are about7.42 for KR fixed shared factors and12.86 for slots. ML fixed values are .00729/.01264. This is consistent with a scale-sensitive mechanism. Yet independent KR STAR ratios can be larger (task0 about23.4/39.6), while its path later grows to substantial functional contribution. An initial norm ratio alone is not a sufficient collapse criterion. Per-parameter preconditioner statistics are scales of η/(sqrt(v-hat)+epsilon), not actual functional learning rates. Loss surface, layer depth/PN, trainable embeddings/biases, bypass training and evolving gradient scale remain relevant alternatives.

Raw gradient rows include an explicit gradient_observation_step. A row labelled selected can otherwise refer to the last observed gradient from another step; selected common-state analysis instead loads the actual selected checkpoint and its moments. No selected-gradient conclusion relies on the label alone.

## Numerical behavior of nominal AdamW

DECAY_ROUNDING_CHECK.json records a direct CPU/CUDA float32 check:

|LR|Nominal WD|Mathematical multiplier|Applied float32 multiplier|Changed values among10,001|
|---:|---:|---:|---:|---:|
|.001|1e-5|.99999999|1.0|0|
|.003|1e-5|.99999997|.9999999403953552|10,001|

The frozen optimizer uses foreach=False with a separate scalar multiplication. All12 ML fixed/STAR global AdamW versus Adam-WD0 pairs have identical final model-state SHA and selected epochs. ML local AdamW likewise matches both-factor WD removal in the collected outcomes. Thus ML's nominal positive AdamW coefficient does not test a representable positive decoupled shrink in this backend. It tests removing the coupled adaptive-L2 term. KR's decoupled shrink is representable but very small, and quantized above the real-valued3e-8 shrink. These facts prevent attributing ML outcomes to a beneficial positive AdamW penalty or comparing effective decay amounts by their nominal coefficients alone. They do not invalidate the measured coupled-L2 group interventions. No result-dependent change to the frozen decay/LR/precision recipe was made.

## Bypass, initialization and parameterization

KR fixed no-bypass retraining has selected mean modulation sensitivity .04178, yet selected AUC falls .089489 and LogLoss rises .018891; all four AUC signs are negative. Half of final probe cells are again exactly zero. Competition with the bypass can matter, but removing the bypass is not a performance remedy. Initial RMS balancing alone leaves modulation zero in all24 selected/final cells under Adam. It is insufficient in this recipe.

Additive factors preserve KR fixed activity (.03598 selected) but provide mixed seed gains (+.005233 AUC, +.001785 LogLoss under Adam). They substantially hurt ML STAR selected AUC. Equal parameter count does not equal initial effective function or regularization geometry. The no-modulation model freezes only the relevant weights; remaining biases/branches can retrain. Its sometimes positive performance further separates activity from utility.

For one effective coordinate acrossD domains, w_d=s*v_d. Minimizing (lambda/2)(s²+sum v_d²) over the scale gauge at fixedw yields lambda*||w||2. With additive w_d=s+v_d, the corresponding optimum is (lambda/2)[sum w_d²-(sum w_d)²/(D+1)]. This is established Hadamard/structured-sparsity algebra, not a novel theorem or a proof of finite-time nonlinear Adam dynamics. [Hoff2017](https://doi.org/10.1016/j.csda.2017.06.007), [Smoothing the Edges2026](https://doi.org/10.1007/s10994-026-06997-0), [AdamW](https://arxiv.org/abs/1711.05101), [Gradient Starvation](https://arxiv.org/abs/2011.09468), [Rotational Equilibrium](https://arxiv.org/abs/2305.17212) and [Star+](https://arxiv.org/abs/2406.16568) delimit the novelty claim. See LITERATURE_LEDGER for distinctions from their precise settings.

## Size and accumulated steps as falsification controls

Reducing the original largest KR domain to25%/50% with either native epochs or matched original exposure/update counts leaves selected fixed modulation exactly zero. Independent STAR remains active. ML fixed remains nonzero under all size controls. Extending ML fixed to2,350 steps still gives final sensitivity .000312, with no zero cells; ML STAR remains strongly active. Therefore neither unique-domain size over this tested range nor total update count alone explains the population difference. This does not establish that size never matters: fewer unique rows clearly harm prediction, the same nested subset is reused across seeds, and matched sampling changes replay/noise. LR, feature dimensionality, task definition and initialization sensitivity also differ across KR/ML.

The supported mechanism is conditional and useful as an audit finding. General structural collapse, irreversible starvation, sufficient norm-ratio thresholds, and broad selected-risk benefits remain NOT_ESTABLISHED. A future independent-population test with matched effective initialization and explicitly representable factor regularizers would discriminate the remaining geometry/gradient-scale explanations better than another arbitrary hyperparameter grid.

Reproducible evidence: development_readout/evidence/probe_condition_summary.csv, paired_comparisons.csv, gradient_updates.csv, offline_readout/common_state_counterfactuals.csv, common_state_moments.csv.gz, probe_conditional_intervals.csv and DECAY_ROUNDING_CHECK.json. Interventions and checkpoints are indexed in ARTIFACT_INDEX.csv.

## Complete frozen-test endpoint

The full-test functional summary corroborates the bounded observation: KR fixed/Adam selected modulation changes zero predictions across all four models (24/24 cell aggregates exactly zero). Final mean sensitivity is6.12e-9, maximum7.75e-7,18/24 cells zero; original KR final is exactly zero in24/24 cells, and independent STAR selected KR/ML paths remain active. The performance implication weakens: KR local AdamW selected test AUC+.000591,1/4 positive seeds, interval[-.010073,.011255], versus development+.007599,4/4 positive. Native test LogLoss increases.000861; against global WD0/AdamW the AUC means are negative. Functional restoration therefore does not confirm the selected-risk rescue. Full risk, metric conflicts and positive controls are in PERFORMANCE_AND_NOVELTY.md and test_readout/evidence.
