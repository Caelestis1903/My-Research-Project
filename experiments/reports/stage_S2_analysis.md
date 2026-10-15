# Stage S2 Final Research Analysis

## 0. Scope, evidence, and terminology

This report is the Main Agent analysis for Stage S2. It uses only the formally accepted artifacts:

- `experiments/stage_S2_raw_results.csv` (2,016 unique rows);
- `experiments/stage_S2_paired_results.csv` (2,016 unique, valid paired rows);
- `experiments/stage_S2_runs.csv` (80 runs: 16 fresh Equal and 64 reused S1 runs);
- the four-seed, role, rotation, difficulty, common-epoch, and training-stage summary tables;
- the final independent data-acceptance record in `experiments/reports/stage_S2_review.md`.

All 16 fresh runs and all 64 reused runs passed validation. AUC and LogLoss were independently recomputed from all 80 prediction artifacts with zero mismatch. There were no missing or invalid pairs. The label normalization used during post-run collection changed no raw artifact and bypassed no non-dataset provenance check.

The protocol calls the reference `STL`, but the actual reference is a **single-domain, two-task MTL model** (`D=1,T=2`), not a separate model per task. This report therefore uses **SD reference** where precision matters.

Unless explicitly marked otherwise, the primary results are validation-selected test results, averaged over seeds 2022–2025. A negative AUC delta means the left method is worse. A positive LogLoss delta means the left method is worse.

---

## 1. Complete experimental results

### 1.1 Setting-level macro results

Each setting-level value first averages its six domain-task cells within a seed, then averages the four seeds.

| Dataset | Setting | Method | Macro-AUC | Seed SD | Macro-LogLoss | Mean worst-cell AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| KuaiRand | natural | Equal | 0.655341 | 0.006857 | 0.451970 | 0.637305 |
| KuaiRand | natural | Sample | 0.675240 | 0.004074 | 0.436404 | 0.638355 |
| KuaiRand | natural | SD reference | 0.637355 | 0.015565 | 0.443367 | 0.561014 |
| MovieLens | d0_large | Equal | 0.712391 | 0.003066 | 0.614724 | 0.656369 |
| MovieLens | d0_large | Sample | 0.729286 | 0.001964 | 0.542514 | 0.680670 |
| MovieLens | d0_large | SD reference | 0.692316 | 0.006413 | 0.566936 | 0.598233 |
| MovieLens | d1_large | Equal | 0.711575 | 0.004377 | 0.591968 | 0.678313 |
| MovieLens | d1_large | Sample | 0.735732 | 0.002401 | 0.537035 | 0.705010 |
| MovieLens | d1_large | SD reference | 0.686353 | 0.003932 | 0.567226 | 0.599782 |
| MovieLens | d2_large | Equal | 0.718879 | 0.006501 | 0.572496 | 0.687123 |
| MovieLens | d2_large | Sample | 0.733140 | 0.004377 | 0.555162 | 0.716463 |
| MovieLens | d2_large | SD reference | 0.688175 | 0.001501 | 0.589647 | 0.604986 |

Across the 16 setting-seed blocks, mean Macro-AUC was 0.699546 for Equal, 0.718349 for Sample, and 0.676050 for the SD reference. Thus aggregate Macro-AUC alone would hide the harmed Large cells documented below.

Setting-level selected-checkpoint paired Macro-AUC deltas were:

| Dataset / setting | Equal − SD | Sample − SD | Equal − Sample |
| --- | ---: | ---: | ---: |
| KuaiRand natural | +0.017985 | +0.037884 | -0.019899 |
| MovieLens d0_large | +0.020075 | +0.036970 | -0.016895 |
| MovieLens d1_large | +0.025222 | +0.049379 | -0.024157 |
| MovieLens d2_large | +0.030704 | +0.044965 | -0.014261 |

All four Equal-minus-Sample macro deltas were negative in all four seeds. All Equal-minus-SD macro deltas were positive in all four seeds. The coexistence of these facts is the central aggregation warning of S2: overall benefit and cell-specific negative transfer occur simultaneously.

### 1.2 Complete selected-checkpoint domain-task table

`neg` is the number of negative AUC deltas among four paired seeds. LogLoss values are included for all methods; the complete seed-level records and all seven checkpoint views remain in the accepted CSVs.

| Dataset | Setting | Domain | Role | Task | Equal AUC | Sample AUC | SD AUC | Equal LL | Sample LL | SD LL | E−SD ΔAUC | S−SD ΔAUC | E−S ΔAUC | E−SD neg | S−SD neg | E−S neg |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRand | natural | tab0 | M | click | 0.647280 | 0.638559 | 0.568365 | 0.287267 | 0.288629 | 0.295006 | +0.078914 | +0.070193 | +0.008721 | 0/4 | 0/4 | 1/4 |
| KuaiRand | natural | tab0 | M | long_view | 0.652980 | 0.656398 | 0.567735 | 0.181845 | 0.180685 | 0.185875 | +0.085245 | +0.088663 | -0.003418 | 0/4 | 0/4 | 2/4 |
| KuaiRand | natural | tab1 | L | click | 0.653956 | 0.698229 | 0.698961 | 0.532359 | 0.504228 | 0.501793 | -0.045006 | -0.000732 | -0.044273 | 4/4 | 3/4 | 4/4 |
| KuaiRand | natural | tab1 | L | long_view | 0.682714 | 0.726658 | 0.726175 | 0.367360 | 0.345202 | 0.342443 | -0.043461 | +0.000483 | -0.043944 | 4/4 | 1/4 | 4/4 |
| KuaiRand | natural | tab4 | S | click | 0.638481 | 0.653964 | 0.623631 | 0.672756 | 0.652237 | 0.666243 | +0.014850 | +0.030332 | -0.015483 | 0/4 | 0/4 | 4/4 |
| KuaiRand | natural | tab4 | S | long_view | 0.656634 | 0.677632 | 0.639265 | 0.670235 | 0.647443 | 0.668842 | +0.017369 | +0.038367 | -0.020998 | 1/4 | 0/4 | 4/4 |
| MovieLens | d0_large | ages_1_18 | L | rating>3 | 0.735118 | 0.748742 | 0.750088 | 0.619769 | 0.609636 | 0.610645 | -0.014970 | -0.001346 | -0.013624 | 4/4 | 3/4 | 4/4 |
| MovieLens | d0_large | ages_1_18 | L | rating>4 | 0.749590 | 0.760450 | 0.763373 | 0.423916 | 0.424415 | 0.422622 | -0.013783 | -0.002923 | -0.010859 | 4/4 | 4/4 | 4/4 |
| MovieLens | d0_large | age_25 | M | rating>3 | 0.723703 | 0.735518 | 0.703086 | 0.639810 | 0.625597 | 0.693757 | +0.020617 | +0.032431 | -0.011814 | 0/4 | 0/4 | 4/4 |
| MovieLens | d0_large | age_25 | M | rating>4 | 0.738889 | 0.752387 | 0.727783 | 0.425871 | 0.416602 | 0.460147 | +0.011106 | +0.024604 | -0.013498 | 0/4 | 0/4 | 4/4 |
| MovieLens | d0_large | ages_35_45_50_56 | S | rating>3 | 0.656369 | 0.680670 | 0.598233 | 0.844714 | 0.679269 | 0.672202 | +0.058136 | +0.082437 | -0.024301 | 0/4 | 0/4 | 4/4 |
| MovieLens | d0_large | ages_35_45_50_56 | S | rating>4 | 0.670675 | 0.697949 | 0.611330 | 0.734262 | 0.499568 | 0.542242 | +0.059345 | +0.086619 | -0.027274 | 0/4 | 0/4 | 4/4 |
| MovieLens | d1_large | ages_1_18 | S | rating>3 | 0.682824 | 0.724236 | 0.599782 | 0.786266 | 0.669383 | 0.689420 | +0.083042 | +0.124455 | -0.041412 | 0/4 | 0/4 | 4/4 |
| MovieLens | d1_large | ages_1_18 | S | rating>4 | 0.696771 | 0.728310 | 0.617261 | 0.576286 | 0.454857 | 0.503682 | +0.079510 | +0.111050 | -0.031539 | 0/4 | 0/4 | 4/4 |
| MovieLens | d1_large | age_25 | L | rating>3 | 0.732020 | 0.754472 | 0.750837 | 0.636917 | 0.593993 | 0.603922 | -0.018817 | +0.003635 | -0.022452 | 4/4 | 0/4 | 4/4 |
| MovieLens | d1_large | age_25 | L | rating>4 | 0.756227 | 0.776738 | 0.773357 | 0.409893 | 0.397602 | 0.408837 | -0.017130 | +0.003381 | -0.020511 | 4/4 | 0/4 | 4/4 |
| MovieLens | d1_large | ages_35_45_50_56 | M | rating>3 | 0.689681 | 0.705010 | 0.677462 | 0.653504 | 0.633207 | 0.681732 | +0.012219 | +0.027548 | -0.015329 | 0/4 | 0/4 | 4/4 |
| MovieLens | d1_large | ages_35_45_50_56 | M | rating>4 | 0.711925 | 0.725624 | 0.699417 | 0.488943 | 0.473166 | 0.515764 | +0.012508 | +0.026207 | -0.013699 | 0/4 | 0/4 | 4/4 |
| MovieLens | d2_large | ages_1_18 | M | rating>3 | 0.726439 | 0.736256 | 0.709117 | 0.644015 | 0.622838 | 0.682247 | +0.017322 | +0.027139 | -0.009817 | 0/4 | 0/4 | 4/4 |
| MovieLens | d2_large | ages_1_18 | M | rating>4 | 0.745371 | 0.749781 | 0.727798 | 0.437534 | 0.429471 | 0.458220 | +0.017573 | +0.021982 | -0.004409 | 0/4 | 0/4 | 4/4 |
| MovieLens | d2_large | age_25 | S | rating>3 | 0.687947 | 0.718126 | 0.604986 | 0.802289 | 0.726124 | 0.761616 | +0.082961 | +0.113140 | -0.030179 | 0/4 | 0/4 | 4/4 |
| MovieLens | d2_large | age_25 | S | rating>4 | 0.720982 | 0.733384 | 0.624852 | 0.468871 | 0.479402 | 0.562083 | +0.096130 | +0.108533 | -0.012402 | 0/4 | 0/4 | 3/4 |
| MovieLens | d2_large | ages_35_45_50_56 | L | rating>3 | 0.704419 | 0.721782 | 0.722512 | 0.623710 | 0.615050 | 0.616101 | -0.018093 | -0.000730 | -0.017362 | 4/4 | 2/4 | 4/4 |
| MovieLens | d2_large | ages_35_45_50_56 | L | rating>4 | 0.728114 | 0.739508 | 0.739782 | 0.458559 | 0.458087 | 0.457613 | -0.011667 | -0.000274 | -0.011393 | 3/4 | 2/4 | 4/4 |

---

## 2. RQ1 — Did Equal reproduce negative transfer?

### Observed facts

1. At the selected checkpoint, **8/24** domain-task groups had Equal-minus-SD mean AUC at or below -0.003; all eight were also at or below -0.005.
2. All eight were the **Large** role. Seven were negative in 4/4 seeds and one in 3/4 seeds.
3. The two KuaiRand tab1 groups were the strongest and most reproducible: click `-0.045006 ± 0.000602`, long-view `-0.043461 ± 0.000968`, both 4/4 negative, LogLoss worse in 4/4 seeds, and negative at all five common epochs.
4. MovieLens Large groups ranged from -0.011667 to -0.018817. All six reached practical and strong AUC magnitude at the selected checkpoint, but only the d2_large `rating>3` group passed the full preregistered combination of AUC magnitude, at least 3/4 negative seeds, LogLoss concordance, and negative mean AUC at at least 3/5 common epochs.
5. Therefore only **3/24 groups** passed the full strict candidate gate: the two KuaiRand tab1 tasks and MovieLens d2_large `rating>3`.
6. Every setting-level Equal-minus-SD Macro-AUC remained positive. Negative transfer is localized rather than global.

### Training-stage qualification

The MovieLens result is checkpoint-sensitive. For example, the six Large selected deltas were all negative, but at final/common epoch 49:

- d0_large changed from `-0.014970/-0.013783` to `+0.006619/+0.006041`;
- d1_large changed from `-0.018817/-0.017130` to `+0.008396/+0.002686`;
- d2_large changed from `-0.018093/-0.011667` to `-0.001715/-0.000374`.

By contrast, KuaiRand tab1 worsened through training and ended at `-0.047369/-0.046085`. Thus the KuaiRand negative transfer is stage-stable, whereas much of the MovieLens selected-checkpoint effect reflects different convergence and checkpoint-selection behavior.

### Interpretation

- **Fact:** Equal produces reproducible, practically large negative transfer in specific Large domain-task cells.
- **Not established:** Equal causes universal negative transfer, or that a shared-interference mechanism is the root cause.
- **Evidence strength:** **STRONG** for KuaiRand tab1; **MEDIUM** for the MovieLens Large-role selected-checkpoint pattern because most MovieLens deficits reverse or shrink below the practical threshold at common late epochs.

---

## 3. RQ2 — Did Sample improve the Equal outcome?

### Direct Equal-versus-Sample result

- Equal was lower than Sample in 23/24 selected domain-task groups.
- The mean difference reached -0.003 in 23/24 groups and -0.005 in 21/24 groups.
- At least 3/4 seeds favored Sample in 22/24 groups.
- All four setting-level Equal-minus-Sample Macro-AUC differences were negative in 4/4 seeds.
- The only selected mean favoring Equal was KuaiRand tab0 click (`+0.008721`, only 1/4 seeds negative). KuaiRand tab0 long-view was nearly tied and seed-unstable (`-0.003418 ± 0.032701`, 2/4 negative).

This establishes a broad objective-ranking result under the fixed S2 protocol: **Sample generally outperformed Equal**.

### Did Sample remove actual negative transfer relative to SD?

Yes at the primary selected checkpoint:

- Sample had **0/24** groups at or below the practical `-0.003` mean threshold.
- Its worst group was MovieLens d0_large rating>4 at `-0.002923`; this was 4/4 negative but below the prespecified practical magnitude and lacked LogLoss/stage concordance.
- KuaiRand tab1 changed from Equal deficits of `-0.045006/-0.043461` to Sample deltas of `-0.000732/+0.000483`.
- The six MovieLens Large groups changed from Equal deficits of -0.011667 to -0.018817 into Sample deltas between -0.002923 and +0.003635.

Thus Sample did not merely raise overall Macro-AUC; it specifically removed the prespecified practical negative-transfer classification in all eight Equal-harmed groups.

### Metric and stage counterexamples

The result is not “Sample improves everything at every stage”:

- Some Sample-minus-SD selected means remained slightly negative.
- At final epoch 49, MovieLens d2_large Sample-minus-SD was `-0.005795/-0.008084` for the two Large tasks, despite being near zero at the selected checkpoint.
- Equal's AUC gains on several Small groups coexisted with worse LogLoss. Seven Equal-minus-SD groups had positive AUC but positive (worse) mean LogLoss, mostly Small roles. Ranking and calibration/probability quality therefore cannot be treated as interchangeable.

### Conclusion for RQ2

Sample's selected-checkpoint mitigation of Equal negative transfer is **STRONG**. Generalization of that statement to every fixed training stage is only **MEDIUM**, because late-epoch counterexamples exist.

---

## 4. RQ3 — Is the weight effect related to domain size?

### Fixed-physical-domain MovieLens rotation

For every MovieLens physical-domain/task combination, Equal-minus-SD changed sign with size role:

| Physical domain | Task | Large | Medium | Small |
| --- | --- | ---: | ---: | ---: |
| age_25 | rating>3 | -0.018817 | +0.020617 | +0.082961 |
| age_25 | rating>4 | -0.017130 | +0.011106 | +0.096130 |
| ages_1_18 | rating>3 | -0.014970 | +0.017322 | +0.083042 |
| ages_1_18 | rating>4 | -0.013783 | +0.017573 | +0.079510 |
| ages_35_45_50_56 | rating>3 | -0.018093 | +0.012219 | +0.058136 |
| ages_35_45_50_56 | rating>4 | -0.011667 | +0.012508 | +0.059345 |

This controls physical identity and task more effectively than a cross-domain comparison. All six combinations are negative when Large and positive when Medium or Small. KuaiRand is compatible with the same sign pattern: its Large domain is negative on both tasks, while Medium and Small are positive versus SD.

### The trade-off is real, but not a simple “Equal helps small domains” story

Role-level selected means for Equal-minus-SD were:

| Dataset | Large | Medium | Small |
| --- | ---: | ---: | ---: |
| MovieLens | -0.015743 | +0.015224 | +0.076521 |
| KuaiRand | -0.044233 | +0.082080 | +0.016109 |

This is strong evidence of a **small/medium-domain gain and large-domain decline trade-off relative to the SD reference**.

However, Sample produced even larger gains than Equal for nearly all Medium and Small cells. In MovieLens, Equal-minus-Sample means were negative for all six Large, all six Medium, and all six Small groups; the Small-role mean was `-0.027851`. Therefore the data do **not** support the stronger claim that sacrificing Large domains under Equal was necessary to rescue Small domains. Sample usually achieved better Small-domain AUC without practical selected-checkpoint negative transfer on Large domains.

### Size versus difficulty

Across the 24 selected cell groups, Equal-minus-Sample had essentially no linear relationship with the SD test AUC (`r≈0.009`) or SD validation macro-AUC (`r≈0.012`); rank correlations were also small. This weakens a simple “hard domains suffer” explanation. The fixed-domain rotations further show that the same physical domain changes sign as its role changes.

Size is therefore a plausible main explanatory variable for the **location** of the Equal-minus-SD sign change. But causal attribution remains **MEDIUM**, not STRONG, because changing size also changes empirical share, inverse loss coefficient, per-example gradient scale, data diversity, and convergence. S2 identifies their combined protocol effect; it does not isolate which component is causal.

---

## 5. Stable phenomena, contradictions, and evidence strength

| Phenomenon | Supporting evidence | Counterevidence / limitation | Strength |
| --- | --- | --- | --- |
| Equal creates stable practical negative transfer on KuaiRand tab1 | Both tasks about -0.044 to -0.045, 4/4 seeds, worse LogLoss, 5/5 common epochs | One dataset/natural size configuration | **STRONG** |
| Equal selected checkpoints harm MovieLens Large roles | 6/6 Large domain-task means below -0.011, at least 3/4 negative; pattern repeats under all three physical-domain rotations | Most reverse or shrink below threshold at late common epochs; only one passes the complete stage/concordance gate | **MEDIUM** |
| Sample removes selected practical negative transfer | 0/24 Sample-minus-SD groups at or below -0.003; all eight Equal-harmed groups move above threshold | Some late-epoch Sample deficits, especially MovieLens d2_large | **STRONG** for selected; **MEDIUM** across stages |
| Sample generally outperforms Equal | 23/24 selected group means, 22/24 seed-stable; all four macro blocks 4/4 seeds | KuaiRand tab0 click favors Equal; tab0 long-view is unstable | **STRONG** |
| Size role is associated with Equal-minus-SD sign | Same six MovieLens physical-domain/task combinations: negative at L, positive at M/S; compatible KuaiRand pattern | Size co-varies with target coefficients, per-example scale, diversity, and convergence | **STRONG** association; **MEDIUM** causal explanation |
| Equal exhibits a Small-gain/Large-decline trade-off versus SD | Role averages and all MovieLens rotations show it | Sample improves Small domains even more, so the trade-off is not necessary or uniquely beneficial | **STRONG** descriptive trade-off |
| Baseline difficulty explains the effect | No meaningful correlation with SD AUC/validation macro-AUC | Difficulty measures are limited and observational | **WEAK / unsupported** |
| AUC and LogLoss tell the same story | Strong concordance in KuaiRand Large negative cells | Several Small groups gain AUC but worsen LogLoss; two MovieLens Large tasks lack selected LogLoss seed concordance | **WEAK** as a universal claim |

---

## 6. Competing explanations and unidentified factors

### 6.1 Effective target weighting

This is the directly manipulated factor. Sampling, batch order, example exposure, model, optimizers, and checkpoints were paired; Equal changed the fixed loss coefficient from `w_d=1` to `w_d=1/(3p_d)`. The broad Equal-versus-Sample difference is therefore causally attributable to the **objective-weighting intervention as a package** under this protocol. It does not identify whether the decisive mechanism is gradient magnitude, optimization conditioning, calibration, or interaction with shared parameters.

### 6.2 Domain update opportunities

Equal and Sample used the same ordinary mixed loader and batch order. Domain appearance/update opportunities did not change between them. This makes a pure “different numbers of domain updates” explanation unlikely for their difference. Realized gradient contribution did change through the loss coefficient.

### 6.3 Shared-parameter interference

The harmed Large cells are compatible with shared-update interference after their effective coefficient is reduced, but S2 did not intervene on sharing or measure gradient conflict. “Gradient conflict,” “capacity competition,” and “representation incompatibility” remain hypotheses, not conclusions.

### 6.4 Domain size, diversity, and per-example scale

MovieLens rotations make physical identity an unlikely sole explanation. Yet the size intervention simultaneously changes sample count, empirical frequency, inverse coefficient, data diversity, and coefficient variance. “Domain size” is a useful condition label, but not yet a singular mechanism.

### 6.5 Checkpoint and convergence

Equal selected substantially earlier than the SD reference and, on KuaiRand, earlier than Sample. MovieLens Equal Large deficits often disappear late, while KuaiRand deficits deepen. This is strong evidence that training stage moderates the effect and that a single selected-checkpoint comparison is insufficient for mechanism claims.

### 6.6 Random-seed variability

KuaiRand tab1 has very small seed SD and is the cleanest target. Several MovieLens groups have much larger SD, including d1_large, and should not be treated as equally strong evidence despite 4/4 selected signs.

---

## 7. RQ4 and stage decision

### Answers to RQ1–RQ4

1. **RQ1:** Yes, Equal reproduces stable practical negative transfer, but only under identifiable conditions. The strongest condition is KuaiRand's Large tab1 on both tasks. MovieLens Large-role selected deficits repeat across rotations but are often stage-dependent.
2. **RQ2:** Yes, Sample clearly improves Equal and removes every selected-checkpoint practical negative-transfer classification. It does not dominate at every cell/seed/stage.
3. **RQ3:** Size role is strongly associated with the sign of Equal-minus-SD, independent of physical MovieLens domain identity. Size is plausible but not yet an isolated causal mechanism.
4. **RQ4:** S2 provides a valid diagnosis object, but not a universal one. Follow-up work must be restricted to stable Large-domain conditions and must model training-stage dependence.

### Formal decision

The evidence best matches **Situation D: condition-dependent negative transfer**, with the additional feature from Situation A that Sample mitigates the selected Equal deficit.

The negative-transfer line may continue only as:

> **Diagnosis of training-objective and shared-update effects in stable Large-domain negative-transfer conditions.**

It must not be framed as a general M3OE or general multi-domain failure. Primary targets should be KuaiRand tab1 on both tasks; MovieLens d2_large rating>3 is a secondary replication target. The other MovieLens selected deficits are useful convergence counterexamples, not equally strong mechanism targets.

### Best next falsification-oriented experiment

The most discriminating next experiment is a preregistered factorial intervention on the stable target cells that crosses:

1. **effective target weighting** (Sample versus Equal), and
2. **cross-domain shared-gradient access** (normal sharing versus a matched control that blocks cross-domain updates to shared parameters while preserving batches, exposure, optimizer steps, parameter initialization, and evaluation).

It must evaluate selected, matched common, and final checkpoints. If the Large-domain deficit persists when cross-domain shared updates are blocked, a shared-interference explanation is weakened and objective-scale/optimization/calibration explanations gain weight. If it disappears only in the blocked-sharing control while exposure and weighting remain matched, shared-update interference gains support. MovieLens stage reversals should be preregistered as a falsification check rather than treated as noise.

No such follow-up experiment is authorized or executed by Stage S2.

---

## 8. Final conclusion

S2 closes the original evidence gap. Equal and Sample are not merely differently named configurations: their verified effective objectives yield materially different outcomes. Equal creates a reproducible Large-domain penalty and a Medium/Small gain relative to the SD reference, while Sample generally achieves higher performance across all roles and eliminates practical selected-checkpoint negative transfer. The cleanest stable phenomenon is KuaiRand tab1; MovieLens demonstrates that convergence and checkpoint selection can reverse the apparent effect.

Accordingly, the evidence supports a **narrow, condition-specific mechanism study**, not a universal negative-transfer claim and not an assumption that domain size alone is the root cause.
