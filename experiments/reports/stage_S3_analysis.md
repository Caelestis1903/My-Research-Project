# Stage S3 Main Research Analysis

## 0. Decision

**Stage S3 supports a narrow causal conclusion: under M3OE on the preregistered KuaiRand Large domain, cross-domain access to shared-parameter updates is a major cause of the Equal-objective deficit. Evidence strength: STRONG.**

This is not proof of “gradient conflict,” and it is not a universal statement about M3OE or all domains. The intervention identifies harmful access by non-target-domain losses to parameters that can affect multiple domains; it does not identify the geometry of those gradients. MovieLens supports the selected-checkpoint direction but shows a clear late-training convergence exception, so cross-dataset generalization is **MEDIUM**.

The Equal-versus-Sample gap on KuaiRand is strongly attenuated but not fully eliminated: at the selected checkpoint it shrinks from `-0.04338/-0.04288` AUC to `-0.00480/-0.00344` on the two tasks. Thus objective weighting retains a small practical residual under the access control. The data do not support replacing Sample with target-only access as a method: the target-specific control damages many non-target cells.

The next experiment should therefore be a falsification-oriented decomposition of **gradient composition versus gradient magnitude/optimizer conditioning**, not a new recommendation method or an assertion that gradient conflict has been proven.

## 1. Evidence base and validity

The analysis uses only the accepted S3 A1 outputs:

- `stage_S3_raw_results.csv`: 1,680 rows;
- `stage_S3_paired_results.csv`: 2,688 rows, including 1,344 strict source-identical contrasts and 1,344 legacy SD-context contrasts;
- `stage_S3_interactions.csv`: 336 strict four-way interaction rows;
- `stage_S3_runs.csv`: 56 inventory rows;
- `stage_S3_aggregates.csv` and `stage_S3_summary_manifest.json`.

The queue is `COMPLETE`; all 24/24 fresh artifacts and 8/8 strict four-condition blocks passed validation. Seeds 2022–2025 are the repeat unit. Prediction-level AUC and LogLoss were structurally validated with zero invalid strict pairs/interactions. Equal FULL is the accepted S2 source-proven control; Sample FULL was freshly reproduced because the S1 model-source hashes were incomplete. The D=1 reference remains legacy context and is not source-identical causal evidence.

The strict S3 estimands are:

- `R_E = Equal target-only shared − Equal FULL`;
- `R_S = Sample target-only shared − Sample FULL`;
- `I = R_E − R_S`;
- `D_W_FULL = Equal FULL − Sample FULL`;
- `D_W_BLOCK = Equal target-only shared − Sample target-only shared`.

Positive rescue and interaction AUC are favorable. The reported `neg_logloss` interactions reverse LogLoss so that positive is favorable.

## 2. Complete selected-checkpoint results

### 2.1 Setting-level macro results

Each value first averages all six domain-task cells within a seed and then averages four seeds.

| Dataset | Condition | Macro-AUC ± seed SD | Macro-LogLoss | Mean worst-cell AUC |
| --- | --- | ---: | ---: | ---: |
| KuaiRand natural | Equal FULL | 0.655341 ± 0.006857 | 0.451970 | 0.637305 |
| KuaiRand natural | Equal target-only | 0.657056 ± 0.002316 | 0.438955 | 0.614822 |
| KuaiRand natural | Sample FULL | 0.673389 ± 0.004478 | 0.434691 | 0.635742 |
| KuaiRand natural | Sample target-only | 0.654221 ± 0.002999 | 0.436797 | 0.602364 |
| KuaiRand natural | legacy SD reference | 0.637355 ± 0.015565 | 0.443367 | 0.561014 |
| MovieLens d2_large | Equal FULL | 0.718879 ± 0.006501 | 0.572496 | 0.687123 |
| MovieLens d2_large | Equal target-only | 0.729572 ± 0.001995 | 0.523831 | 0.714366 |
| MovieLens d2_large | Sample FULL | 0.733106 ± 0.004457 | 0.555678 | 0.716381 |
| MovieLens d2_large | Sample target-only | 0.728441 ± 0.001884 | 0.526444 | 0.712323 |
| MovieLens d2_large | legacy SD reference | 0.688175 ± 0.001501 | 0.589647 | 0.604986 |

Macro averages are not decisive. KuaiRand Equal target-only slightly raises macro-AUC while lowering mean worst-cell AUC because the target rescue is accompanied by non-target damage. Sample target-only is worse than Sample FULL at macro level in both datasets.

### 2.2 All domain-task cells

The table reports four-seed means at each condition's validation-selected checkpoint. `E-F`, `E-T`, `S-F`, `S-T`, and `SD` denote Equal FULL, Equal target-only, Sample FULL, Sample target-only, and legacy D=1 context.

| Dataset / domain / role / task | E-F AUC | E-T AUC | S-F AUC | S-T AUC | SD AUC | E-F LL | E-T LL | S-F LL | S-T LL | SD LL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRand tab0 M click | 0.647280 | 0.615894 | 0.637185 | 0.602689 | 0.568365 | 0.287267 | 0.290447 | 0.290462 | 0.292123 | 0.295006 |
| KuaiRand tab0 M long-view | 0.652980 | 0.620838 | 0.648495 | 0.615548 | 0.567735 | 0.181845 | 0.182632 | 0.183277 | 0.183293 | 0.185875 |
| KuaiRand tab1 L click | 0.653956 | 0.692832 | 0.697339 | 0.697636 | 0.698961 | 0.532359 | 0.509186 | 0.502015 | 0.500482 | 0.501793 |
| KuaiRand tab1 L long-view | 0.682714 | 0.722290 | 0.725597 | 0.725733 | 0.726175 | 0.367360 | 0.347634 | 0.341101 | 0.341366 | 0.342443 |
| KuaiRand tab4 S click | 0.638481 | 0.631557 | 0.655541 | 0.628929 | 0.623631 | 0.672756 | 0.653533 | 0.646143 | 0.653305 | 0.666243 |
| KuaiRand tab4 S long-view | 0.656634 | 0.658926 | 0.676174 | 0.654789 | 0.639265 | 0.670235 | 0.650298 | 0.645150 | 0.650215 | 0.668842 |
| MovieLens age_25 S rating>3 | 0.687947 | 0.725927 | 0.718012 | 0.723964 | 0.604986 | 0.802289 | 0.614449 | 0.726364 | 0.620363 | 0.761616 |
| MovieLens age_25 S rating>4 | 0.720982 | 0.751890 | 0.733189 | 0.750356 | 0.624852 | 0.468871 | 0.405481 | 0.482496 | 0.408390 | 0.562083 |
| MovieLens ages_1_18 M rating>3 | 0.726439 | 0.714366 | 0.736337 | 0.712323 | 0.709117 | 0.644015 | 0.624339 | 0.622583 | 0.626444 | 0.682247 |
| MovieLens ages_1_18 M rating>4 | 0.745371 | 0.719420 | 0.749791 | 0.718194 | 0.727798 | 0.437534 | 0.431692 | 0.429630 | 0.433458 | 0.458220 |
| MovieLens ages_35_45_50_56 L rating>3 | 0.704419 | 0.723597 | 0.721801 | 0.723744 | 0.722512 | 0.623710 | 0.613263 | 0.615012 | 0.614402 | 0.616101 |
| MovieLens ages_35_45_50_56 L rating>4 | 0.728114 | 0.742232 | 0.739506 | 0.742062 | 0.739782 | 0.458559 | 0.453758 | 0.457980 | 0.455608 | 0.457613 |

## 3. Primary KuaiRand diagnosis

### 3.1 Equal access rescue

| Target task | `D_W_FULL` AUC | `R_E` AUC ± SD | Positive seeds | `R_S` AUC ± SD | `I` AUC ± SD | Positive interaction seeds | `I` neg-LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| click | -0.043383 | +0.038876 ± 0.003950 | 4/4 | +0.000297 ± 0.004833 | +0.038579 ± 0.005653 | 4/4 | +0.021641 |
| long-view | -0.042883 | +0.039576 ± 0.004107 | 4/4 | +0.000135 ± 0.004718 | +0.039440 ± 0.005285 | 4/4 | +0.019990 |

Both preregistered tasks pass the access-rescue and interaction magnitude, seed-direction, and LogLoss gates. Sample receives essentially no target benefit, so the Equal rescue is not a generic consequence of the target-only implementation.

The interaction persists at every common epoch:

| Task | Epoch 9 | Epoch 19 | Epoch 29 | Epoch 39 | Epoch 49 |
| --- | ---: | ---: | ---: | ---: | ---: |
| click `I` AUC | +0.025579 | +0.035257 | +0.040050 | +0.044829 | +0.045037 |
| long-view `I` AUC | +0.025231 | +0.035884 | +0.040763 | +0.045459 | +0.045459 |

All ten task-stage means are positive and above `+0.003`; all are 4/4 positive by seed. Final-epoch negative-LogLoss interactions are also positive (`+0.031060`, `+0.028006`). The effect is therefore not selected-checkpoint-only.

### 3.2 Attenuation, not complete elimination

For click, the Equal-minus-Sample gap changes from `-0.043383` under FULL access to `-0.004805` under target-only access, an 88.9% attenuation in absolute magnitude. Long-view changes from `-0.042883` to `-0.003443`, a 92.0% attenuation. Both residual blocked gaps remain at or beyond the prespecified `-0.003` practical line.

Relative to the legacy SD context, Equal target-only remains `-0.006130` on click and `-0.003885` on long-view. Those comparisons imply 86–91% attenuation of the historical Equal deficit but not elimination. Because the D=1 source is not strict source-identical evidence, the causal claim rests on `R_E` and `I`, not on these SD comparisons.

**Conclusion:** harmful non-target shared-update access explains most, but not all, of the KuaiRand Equal deficit. A smaller weighting/optimization residual remains.

## 4. MovieLens replication and convergence exception

At the selected checkpoint, MovieLens's preregistered Large domain is compatible with KuaiRand:

| Target task | `D_W_FULL` AUC | `R_E` AUC ± SD | Positive seeds | `R_S` AUC ± SD | `I` AUC ± SD | Positive interaction seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rating>3 | -0.017382 | +0.019177 ± 0.003851 | 4/4 | +0.001943 ± 0.004726 | +0.017235 ± 0.002046 | 4/4 |
| rating>4 | -0.011391 | +0.014118 ± 0.010428 | 4/4 | +0.002557 ± 0.005275 | +0.011561 ± 0.005160 | 4/4 |

The blocked Equal-minus-Sample gaps are approximately zero (`-0.000147`, `+0.000170`), and Equal target-only is slightly above the legacy SD context (`+0.001085`, `+0.002450`).

However, the common-epoch evidence is not stable:

- rating>3 interaction remains positive at 5/5 common epochs but drops below practical magnitude at epoch 49 (`+0.002186`); its negative-LogLoss interaction reverses late.
- rating>4 interaction is positive only at epochs 9 and 19, then becomes negative at epochs 29, 39, and 49; final AUC interaction is `-0.002490` and final negative-LogLoss interaction is `-0.052019`.

MovieLens therefore provides **MEDIUM selected-stage replication** and a strong warning that checkpoint/convergence can alter the apparent mechanism. It cannot upgrade the result to universal cross-dataset evidence.

## 5. Non-target effects and the gain/decline trade-off

The intervention is target-specific and redistributes performance.

- KuaiRand Equal: target cells gain `+0.039226` AUC on average across task-seed observations, while non-target cells lose `-0.017040` (14/16 negative).
- KuaiRand Sample: targets are unchanged (`+0.000216`), while non-target cells lose `-0.028860` (16/16 negative).
- MovieLens Equal: target cells gain `+0.016647`; non-target mean is `+0.007716`, but signs split 8 positive/8 negative.
- MovieLens Sample: targets gain only `+0.002250`; non-target mean is `-0.008123`, with mixed signs.

Cell-level counterexamples are substantial. On KuaiRand tab0, target-only access lowers both Equal and Sample AUC by about `0.031–0.034`. On MovieLens ages_1_18, it lowers Equal by `0.012/0.026` and Sample by `0.024/0.032`. Conversely, MovieLens's Small age_25 cells improve strongly under target-only access.

This means the control is diagnostic, not a safe global mitigation. It also supplies direct evidence of a **Large-target improvement / non-target decline trade-off** on KuaiRand. The older S2 result showed the converse Equal-versus-SD pattern—Large decline with Medium/Small gains. Together, these results show that access routing can move benefit across domains, but they do not show that the trade-off is unavoidable.

## 6. Domain size, difficulty, baseline, model, and training stage

### Domain size

S2's fixed-physical-domain MovieLens rotations showed all six physical-domain/task combinations changing from negative Equal-minus-SD when Large to positive when Medium/Small. That is **STRONG association** between assigned size role and where the deficit appears.

S3 intervenes only on the Large target in KuaiRand natural and MovieLens d2_large. Its large target rescue is consistent with size being an important condition, but S3 does not independently manipulate target size. Size changes empirical share, coefficient scale, diversity, and optimization trajectory together. It remains a plausible main explanatory variable, not an isolated causal mechanism. Causal evidence strength for “size itself” remains **MEDIUM**.

### Difficulty and baseline

The two target datasets have high SD-reference AUCs, whereas several lower-baseline Medium/Small cells receive positive transfer. The S2 correlation between Equal-minus-Sample and SD baseline AUC was near zero, and the same physical domains changed sign when their size roles rotated. A simple “hard domains suffer” account is therefore **WEAK/unsupported**. Baseline performance and physical identity cannot explain the intervention interaction by themselves.

### Model scope

All evidence is conditional on M3OE, the audited source code, fixed optimizer, 50-epoch budget, two datasets/settings, and the selected target domains. S1 showed that small residual effects under Sample were backbone-dependent across M3OE/MMOE/PLE; S3 does not test whether the access interaction transfers to those other backbones.

### Training stage

KuaiRand Equal FULL selects much earlier (mean epoch 15.5) than Equal target-only (33.75), Sample FULL (35.0), and Sample target-only (41.5). MovieLens selections are earlier overall: 5.5, 10.0, 8.75, and 9.0. The matched common-epoch analysis is therefore essential. It strengthens the KuaiRand conclusion and weakens MovieLens generalization.

## 7. Stable phenomena, contradictions, and evidence strength

| Phenomenon | Supporting evidence | Contradiction / limitation | Strength |
| --- | --- | --- | --- |
| Cross-domain shared-update access is harmful for KuaiRand Equal target cells | `R_E≈+0.039` on both tasks, 4/4 seeds, favorable LogLoss, 5/5 common epochs | Conditional to M3OE/natural/tab1 | **STRONG** |
| The effect is Equal-specific rather than a generic target-only effect | `R_S≈0`; `I≈+0.039`, 4/4 seeds and 5/5 epochs on both tasks | Equal still trails Sample slightly after blocking | **STRONG** |
| Access explains the entire Equal-versus-Sample gap | 89–92% selected attenuation | `D_W_BLOCK=-0.00480/-0.00344`, so practical residual remains | **MEDIUM / incomplete** |
| MovieLens replicates the interaction | Selected `I=+0.0172/+0.0116`, both 4/4 positive | rating>4 reverses at 3/5 common epochs; late LogLoss reverses | **MEDIUM** |
| Target-only access is a generally beneficial method | Strong target rescue | Large non-target losses, especially KuaiRand and MovieLens Medium cells | **WEAK / falsified as universal claim** |
| Domain size locates the original Equal deficit | S2 physical-domain rotations; both S3 targets are Large | S3 does not independently manipulate size | **STRONG association; MEDIUM causality** |
| Baseline difficulty is the main explanation | Some target baselines are high | Near-zero S2 correlation and rotation sign changes | **WEAK** |
| The result proves gradient conflict | Gradient removal changes outcomes | No geometric conflict/alignment intervention or directional test | **Unsupported** |

## 8. Competing explanations

1. **Structured non-target shared-update interference — strongest current explanation.** The strict factorial interaction is large, seed-stable, metric-concordant, and stage-persistent on both KuaiRand tasks.
2. **Shared-gradient magnitude / Adam conditioning — not separated.** Removing `L_non` changes both composition and norm of the applied shared gradient and therefore optimizer moments. This could explain some or all of the rescue without requiring harmful directional conflict.
3. **Persistent objective-weighting/private-path effect — bounded residual support.** The Equal-minus-Sample gap remains slightly practical after blocking on KuaiRand. The experiment does not separate coefficient scale, private-parameter optimization, or calibration.
4. **Convergence/checkpoint behavior — strongly relevant for MovieLens, weak for KuaiRand.** KuaiRand survives matched stages; MovieLens partly reverses.
5. **Domain size and update economics — plausible condition.** Size controls where the original deficit appears, but share, coefficient, diversity, and stage co-vary.
6. **Physical-domain identity or inherent difficulty — weak.** Rotation and baseline evidence argue against either as a sufficient explanation.
7. **Random seed fluctuation — unlikely for the KuaiRand primary effect.** Both tasks are 4/4 consistent with effects far above their seed SDs; it remains relevant for small residuals and MovieLens late-stage behavior.

## 9. Best discriminating next experiment

Proceed only to a narrow mechanism-discrimination stage, not method development.

The highest-value falsification test is an **Equal-objective, KuaiRand-tab1 shared-gradient magnitude/composition control** with the same initialization, batches, exposure, optimizer, and checkpoints:

1. FULL shared gradient (`g_target + g_non`);
2. target-only shared gradient (`g_target`), reproducing S3;
3. a preregistered **norm/optimizer-matched FULL control** that preserves the FULL gradient composition but matches the target-only shared-gradient norm and optimizer-step scale;
4. preferably a complementary target-only-plus-null control that restores the removed norm without restoring structured non-target direction.

The experiment should be designed to falsify the current interpretation:

- If a magnitude/optimizer-matched FULL arm reproduces the rescue, structured cross-domain direction is not needed; magnitude/conditioning is sufficient.
- If target-only remains superior after matching norm and optimizer scale, composition/access gains stronger support.
- If the effect fails on either KuaiRand task or at matched common epochs, the current access explanation is weakened.

Only after that distinction is resolved would donor-specific leave-one-domain-out access be justified to identify whether tab0, tab4, or their combination drives the effect.

## 10. Stage disposition

**GO, narrowly, to a falsification-oriented mechanism-discrimination stage.**

The GO is justified by a large, stable, source-identical causal interaction on both primary KuaiRand tasks. It is not permission to claim universal gradient conflict, to deploy target-only routing as a method, or to skip a new stage-specific Designer specification. Any next stage must begin with Designer and must preserve the two-process multi-seed execution rule.
