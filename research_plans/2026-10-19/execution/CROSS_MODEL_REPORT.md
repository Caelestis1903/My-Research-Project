# Cross-model scope, modules and competing explanations

Evidence rejects a universal conditional/shared-path-death claim. M3oE's KR input-weight product is numerically suppressed in several training protocols; independent STAR, PEPNet, public-code-style HAMUR and audited MMoE/PLE keep substantial native module activity. Their interventions have different meanings and scales, so numerical magnitudes cannot rank model quality across families. This report's mechanism and control numbers are development results. Full frozen-test natural/neutral performance is retained in test_readout, with the final endpoint interpretation in PERFORMANCE_AND_NOVELTY and FINAL_REPORT.

## Fidelity and scope

MODEL_FIDELITY.md and SOURCE_LINEAGE.json distinguish pinned N19/public-source graph semantics from our controlled adaptations. All models use the same frozen public populations, features and paired optimization seeds; these are not industrial paper-performance reproductions. STAR and native HAMUR train task0/task1 separately. Joint two-head adaptations are G0M, not original-paper MDMT models. G0MC trains two independent backbones with the same mean-task objective and twice the core capacity. MMoE/PLE use repository-audited equations, with one CGC extraction stage for PLE. PEPNet uses the available priors; KR has no user/item IDs.

HAMUR preserves public-source shared/all-ones factors and mixed-batch normalization including eval, rather than strict paper domain-specific normalization/factors. Its output depends on inference batch composition. Associative contraction was checked against the preserved reference; forward/gradient differences were within the frozen tolerances. The conclusions below apply to this declared implementation. Conditional row-bootstrap intervals resample already computed predictions and do not recompute batch normalization, so they do not model uncertainty from changing inference batches.

## Native activity and fixed-state intervention

Selected development probe mean absolute probability changes under each model's specified neutral intervention:

|Population/Adam|M3oE fixed modulation|STAR task0/task1 modulation|PEPNet EP/PP|HAMUR task0/task1 residual|MMoE/PLE shared outputs|
|---|---:|---:|---:|---:|---:|
|KR|0|.279/.257|.0815/.0541|.0584/.0718|.199/.151|
|ML|.00230|.174/.178|.0534/.0774|.1484/.0823|.226/.146|

Exact family-specific values, seed ranges and zero counts are in development_readout/evidence/probe_condition_summary.csv and probe_seed_summary.csv. PEPNet gate1, HAMUR residual0 and expert-output0 without gate renormalization are different counterfactual functions. Zeroing M3oE input weights also differs from zeroing the whole main output or bypass.

**STRONG within the measured population/implementation:** native independent STAR activity is a counterexample to all STAR-style multiplicative paths necessarily dying under Adam. Its KR initial candidate L2/data-gradient norm ratio can exceed fixed M3oE's, yet it develops substantial contribution. PEP/HAMUR/expert activity similarly contradicts a universal numerical degeneration claim. An active module can still hurt a particular risk measure, and these observations do not establish universal helpfulness.

The original M3oE protocol also attenuates KR modulation (selected mean1.60e-7, final exactly0); original_sample is near float32 resolution at selection. Thus the observation extends beyond the fixed-fusion optimizer recipe on a related graph. Original's Equal cell-sum objective and overlapping fusion optimizer are a combined protocol contrast. Original is about.019 selected AUC below fixed on both populations, while original_sample largely closes that difference; this cannot isolate a single architectural cause or prove the original paper fails.

## Full retraining is a separate question

|Complete control vs its native selected model|Development ΔAUC|ΔLogLoss|Seed pattern / implication|
|---|---:|---:|---|
|KR fixed, remove bypass, Adam|-.089489|+.018891|4/4 AUC worse despite restored selected modulation|
|ML STAR task0, no modulation, Adam|+.004117|-.022967|4/4 AUC better while native modulation is active|
|ML STAR task1, no modulation, Adam|+.003617|-.025748|4/4 AUC better; active does not imply selected-risk value|
|ML STAR task0/task1, additive, Adam|-.033956/-.023415|+.021239/-.010544|All AUC signs negative; AUC/loss can conflict|
|KR PLE matched private experts, Adam|-.020840|+.004037|4/4 AUC worse; blanket harmful-sharing claim contradicted|
|KR MMoE matched private experts, Adam|-.016387|+.008418|4/4 AUC worse|
|ML HAMUR task0 no adapter, Adam|+.044933|-.452883|4/4 AUC better relative to a weak native baseline|
|ML HAMUR task1 no adapter, Adam|+.019371|-.112398|4/4 AUC better; descriptive interval includes0|

The no-adapter ML HAMUR task0/task1 models still trail same-task fixed/Adam by.017587/.030073 AUC (all four seeds), with LogLoss higher by.077463/.139614. Gains over this native baseline do not establish a competitive new algorithm, and cannot generalize to strict paper DN. Static adapters improve some ML native comparisons; private hypernetworks can reduce task1 AUC and markedly worsen loss. These mixed controls do not support a single shared-hypernetwork degeneration story.

Private expert alternatives differ in parameter placement and width; matched widths170(MMoE)/426(PLE) are approximate capacity controls. G2 additive/capacity-replacement aliases have exactly the same graph and trainable factor count, but a different effective initialization and penalty geometry from products. Frozen unused parameters can remain serialized and some computation remains; no physical-pruning speed claim follows from the ablation names. Every positive and negative G2/G3 comparison remains in paired_comparisons.csv and paired_cell_deltas.csv.

## Task interaction and meaningful baselines

Joint versus normalized independent-backbone selected AUC changes:

|Model/population|Adam|AdamW|Interpretation|
|---|---:|---:|---|
|STAR KR|-.006299|+.000162|Adam4/4 negative, AdamW mixed; not one stable sharing sign|
|STAR ML|+.005136|+.005414|4/4 positive under both recipes; Adam loss worsens .004514|
|HAMUR KR|+.005618|+.011188|AdamW4/4 positive; doubled private capacity is not a benefit guarantee|
|HAMUR ML|+.012984|+.014959|Mixed seeds and wide intervals; loss changes disagree across recipes|

The G0MC mean-task objective resolves the missing .5 normalization in a naive comparison with two separately trained single-task runs; it does not hold backbone capacity fixed. No pure equal-capacity causal estimate of task conflict is claimed. Single-task native metrics compare with the fixed model's same task/domain, never a two-task macro average. Strong-reference tables additionally match candidate cells against native fixed/MMoE/PLE at both optimizer recipes. KR local fixed AdamW exceeds fixed Adam in AUC but its paired intervals against MMoE and PLE cross0; ML no-adapter HAMUR remains below fixed. Baseline strength, not gain over the weakest comparator, limits utility claims.

## Size, difficulty and stage

KR main domain counts are164143/1656704/84980. Fixed/Adam selected task0 AUCs are.635691/.693332/.659878; the smallest domain is not the lowest-AUC one. ML counts32775/2850/142500, with task0 AUCs.727854/.713992/.743631; size and lower AUC align in this one case, without causal identification. Task1 and probability-risk diagnostics are preserved separately. Constant LogLoss uses the evaluation cell's own prevalence (oracle constant entropy); skill_logloss is entropy minus model LogLoss, an absolute gap rather than a ratio or a train-fitted deployment predictor.

The G4 intervention reduces the original largest domain's unique rows, keeping other domains/evaluation fixed. Matched sampling preserves domain exposure, mixture and total updates, with target replay. Fewer unique rows reduce selected AUC in both populations and active-path STAR. At25%, fixed target ΔAUC is KR-.043395/ML-.025246, while other domains also decline -.016192/-.014979. This supports an information/replay effect on risk, not a claim size is irrelevant. But KR fixed remains numerically inactive and ML fixed remains active throughout the tested range; unique-N changes do not explain that boundary. Target domains remain largest even at25%; all seeds share one fixed nested subset, so population/subset generalization remains unconfirmed.

ML2,350-update fixed/STAR controls retain activity, keep the same selected epochs as500-update controls, and worsen final AUC and LogLoss. Cumulative steps alone are insufficient. Natural KR/ML differences also include LR, feature dimension, labels, priors and initialization-scale differences. A final-only gain, a path norm increase or a correlation with domain size cannot replace a selected per-cell risk result.

A small-domain-gain/large-domain-loss rule is not stable. KR local fixed AdamW improves mid-sized domain0 task0/1 AUC by.021717/.030408, changes largest domain1 by+.002029/+.002771 and changes smallest domain2 by-.000298/-.011032. Individual seed signs remain mixed in several cells. Other optimizer interventions have different patterns. **WEAK** evidence for a universal size trade-off; **MEDIUM** evidence for heterogeneous model/task/stage-specific effects on these fixed populations.

The four required axes remain separate: bounded structural scope; locally supported decay attribution; heterogeneous selected-risk value; substantial existing-method/theory coverage. Broad structural collapse and a general novel rescue are NOT_ESTABLISHED.
