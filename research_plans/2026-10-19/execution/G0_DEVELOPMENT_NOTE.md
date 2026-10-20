# G0 development evidence — interim, not final disposition

All 160 preregistered G0 training runs completed, four optimization seeds per condition, with full budgets. This note uses the development split and validation probes only. G0M/G0MC and G1–G4 remain required; no direction is cancelled and no test evaluation is released. Results below cannot establish out-of-population generalization.

Reproduction: `python -B experiments/cm19_collect.py --split development --stage G0`, then `python -B experiments/cm19_g0_readout.py`. Tables are in `development_g0_readout/`. Intervals below are descriptive paired-seed t intervals with df3, without multiplicity correction. Tasks/domains inside a seed are not independent replication units.

## Direct pathway observations

**STRONG, narrowly scoped numerical observation:** fixed M3oE with Adam on KuaiRand has exactly zero float32 prediction sensitivity to modulation neutralization in all 24 seed/domain/task probe cells at the selected checkpoints. The entire 24-cell probe is also zero at updates200,500 and1000. At final,18/24 cells are exactly zero; the remaining seed has tiny sensitivity (overall mean5.85e-9, maximum individual change4.77e-7). Thus permanent mathematical absorption is not established. Selected effective product norms are at underflow scale. The bypass remains influential and the selected main branch still changes outputs when removed: modulation inactivity does not establish complete main-branch death.

**STRONG counterexample within these implementations:** independent STAR retains functional multiplicative FCN pathways under Adam. Mean absolute selected prediction changes when neutralized are0.279/0.257 for KuaiRand task0/task1 and0.174/0.178 for MovieLens; all corresponding cells have nonzero sensitivity. Per-layer product norms remain nonzero. The public-source-style HAMUR adapters, PEPNet EP/PP gates and MMoE/PLE shared pathways also remain functionally active. Their interventions have different meanings and these magnitudes are not a cross-model ranking. This rejects a universal inactivity claim for the tested condition, while source fidelity limits remain in MODEL_FIDELITY.md.

Fixed MovieLens Adam modulation sensitivity is nonzero at every measured step0–500 and at selected/final checkpoints. It is weaker than several other branches but cannot be called numerically dead. Dataset, architecture, initialization, optimization count and task context are competing explanations; domain size alone has not been isolated.

## Functional activity does not establish selected performance benefit

AdamW restores substantial KuaiRand fixed-model modulation sensitivity (selected mean absolute probability change0.0299 versus0 for Adam), but selected development performance is mixed:

| Fixed M3oE, AdamW minus Adam | Selected macro AUC | Paired seed signs | Selected macro LogLoss | Final macro AUC | Final macro LogLoss |
|---|---:|---|---:|---:|---:|
| KuaiRand | +0.000836 | 2 positive / 2 negative | +0.001621 | -0.012721 | +0.130647 |
| MovieLens | -0.000142 | 2 positive / 2 negative | -0.000778 | -0.000628 | +0.137456 |

Lower LogLoss is better. The selected AUC paired95% intervals are[-0.012245,+0.013918] and[-0.001640,+0.001356]. **WEAK evidence for selected benefit; no stable improvement demonstrated.** Final LogLoss is worse for AdamW in all four seeds on both datasets; this cannot be hidden by reporting path activity alone.

Across the20 model/dataset optimizer contrasts, seven selected AUC mean differences favor AdamW and thirteen favor Adam. These correlated conditions are not20 independent trials. MovieLens STAR task0/task1 selected AUC decreases by0.003989/0.004132 with AdamW, each in all four seeds, alongside worse LogLoss. KuaiRand PLE selected AUC decreases by0.009977 in all four seeds. These are **MEDIUM fixed-population evidence against a generally beneficial AdamW replacement under the frozen protocol**, not proof that tuned AdamW is intrinsically inferior. G1 group-specific retraining and common-state moment counterfactuals are still needed for decay attribution.

## Baselines, tasks and training stage

Selected fixed Adam macro AUC/LogLoss is0.668287/0.443914 on KuaiRand and0.730805/0.550816 on MovieLens. KuaiRand PLE/MMoE Adam AUC means are0.671927/0.670721, but their paired differences against fixed have mixed seeds and descriptive intervals including zero. MovieLens fixed slightly exceeds those implementations in AUC; MMoE has slightly lower mean LogLoss. No universal model ranking follows.

Original Equal+overlapping-optimizer protocol has selected AUC about0.019 below fixed on both datasets and both optimizers, negative in every paired seed. Original_sample largely closes that difference. **MEDIUM protocol-comparison evidence**: loss weighting plus the optimizer schedule matters, but these jointly changed features do not isolate one cause. This is not an architecture-only comparison.

The task-matched tables compare single-task STAR/HAMUR only against the same fixed-model task. For MovieLens Adam, STAR task0/task1 trails fixed by0.006895/0.006205 and HAMUR by0.062520/0.049444, all four paired seeds negative. Backbone/task sharing, capacity and source-style HAMUR batch normalization remain confounders. G0M/G0MC supplies the planned normalized task-sharing controls; averaging separate single-task models is not one MDMT model.

**MEDIUM evidence for training-stage dependence:** final development AUC is materially below selected AUC for most conditions, and final LogLoss worsens broadly. For fixed Adam, final-minus-selected AUC is-0.021489/-0.053741 on KuaiRand/MovieLens, negative in all seeds; LogLoss rises0.047374/0.630248 in all seeds. Final online-training AUC/LogLoss is0.838996/0.379322 and0.976472/0.143644. Online values use changing within-epoch weights and are not static training-set checkpoint estimates. Overfitting, calibration and scheduling can explain poor finals even when a pathway remains active. KuaiRand Adam STAR/PEPNet are exceptions to a universal final-AUC decline, and must remain visible.

## Physical domain size, difficulty and trade-off

Main-training domain counts are KuaiRand[164143,1656704,84980] and MovieLens[32775,2850,142500]. Fixed Adam selected AUC by domain/task is:

| Dataset | Domain | Task0 | Task1 |
|---|---:|---:|---:|
| KuaiRand | 0 | 0.635691 | 0.636624 |
| KuaiRand | 1 | 0.693332 | 0.718990 |
| KuaiRand | 2 | 0.659878 | 0.665204 |
| MovieLens | 0 | 0.727854 | 0.735673 |
| MovieLens | 1 | 0.713992 | 0.723073 |
| MovieLens | 2 | 0.743631 | 0.740606 |

The smallest KuaiRand domain is not the lowest-AUC domain, whereas MovieLens's smallest domain has lower baseline AUC. Difficulty depends on task/metric/model; positive prevalence and skill relative to a constant predictor are provided in seed_cell_metrics.csv. Natural correlations do not establish size as a main cause.

Fixed KuaiRand AdamW has positive mean selected AUC changes in both domain0 tasks and negative changes in both largest-domain1 tasks, but cell signs vary across seeds; smallest domain2 has opposing task means(+0.003575,-0.008346). MovieLens smallest-domain1 task means also disagree(-0.001907,+0.001851). **WEAK evidence for a general small-domain gain/large-domain decline trade-off.** The registered G4 matched-update/mixture nested-subset intervention is the strongest next discriminator of unique sample count; ordinary-epoch subsampling also changes exposure and optimization count. G4LONG separately addresses the500-versus2350-update cross-dataset confound.

## Interim disposition

Do not promote a universal path-failure mechanism or a performance repair from G0. Continue every frozen condition. The falsification priorities remain: whether isolated factor decay reproduces the functional/performance changes, whether no-modulation retraining can match active models, whether STAR's architectural counterexample persists with normalized joint controls, and whether unique-N interventions change the phenomenon with update count and mixture held fixed. No new hyperparameter search is justified by this interim readout. Publication GO/NO-GO and any independent confirmation decision are deferred until the complete development matrix is available.
