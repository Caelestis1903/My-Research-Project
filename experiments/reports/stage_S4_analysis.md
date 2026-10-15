# Stage S4 ANALYSIS REPORT — Main Agent

**Scope.** S4 mechanism diagnosis of the S3 KuaiRand Equal `tab1` target-only rescue: four fresh arms `F` (FULL gradient), `T` (target-only shared gradient), `N` (FULL gradient norm-matched to target-only before Adam), `U` (FULL Adam moments with applied shared delta magnitude-matched to the same-state counterfactual target-only step), seeds 2022–2025, M3OE, KuaiRand `natural`, 50 epochs. Executed under Amendment A2 (queue `c7353a2e…`, producer digest `8dc07f3d…`, M3OE forward `a7702a64…`): 16/16 canonical runs, 4/4 four-arm blocks, `pairing_valid: true`, zero retries, zero failures, all per-step norm-matching gates closed (max matching residual 5.0e-07 vs tolerance ≥1e-6; `zero_case = regular` for all 1,600 probes). Data sources: `experiments/stage_S4_A2_outputs/stage_S4_{runs,raw_results,paired_results,aggregates,probe,gradient_steps}.csv`. All statements below trace to those files; AUC differences are `left − right`; LogLoss differences are favorable when negative.

Conventions: `d1` = target domain `tab1`; `d0`, `d2` = non-target. Tasks: `t0` = click, `t1` = long_view. Predeclared practical threshold 0.003 (strong 0.005); 3/4-seed and 3/5-common-epoch stability rules; no p-values.

---

## 1. Complete results

### 1.1 Absolute test AUC at the selected checkpoint (4-seed means)

| Cell | F | T | N | U |
|---|---|---|---|---|
| d0t0 | 0.6473 | 0.6159 | 0.6344 | 0.6267 |
| d0t1 | 0.6530 | 0.6208 | 0.6461 | 0.6308 |
| **d1t0** | 0.6540 | **0.6928** | 0.6499 | 0.6534 |
| **d1t1** | 0.6827 | **0.7223** | 0.6778 | 0.6818 |
| d2t0 | 0.6385 | 0.6316 | 0.6484 | 0.6340 |
| d2t1 | 0.6566 | 0.6589 | 0.6652 | 0.6558 |

### 1.2 Target-domain contrasts (mean ΔAUC, sign count over 4 seeds)

| Contrast | Checkpoint | d1t0 | d1t1 |
|---|---|---|---|
| T−F | selected | **+0.038876** (4/4) | **+0.039576** (4/4) |
| T−F | final | **+0.043197** (4/4) | **+0.043518** (4/4) |
| N−F | selected | −0.004048 (0/4) | −0.004873 (0/4) |
| N−F | final | −0.002522 (0/4) | −0.002008 (1/4) |
| U−F | selected | −0.000587 (2/4) | −0.000942 (2/4) |
| U−F | final | −0.000009 (2/4) | −0.000133 (2/4) |
| T−N | selected | +0.042924 (4/4) | +0.044448 (4/4) |
| T−N | final | +0.045720 (4/4) | +0.045525 (4/4) |
| T−U | selected | +0.039463 (4/4) | +0.040517 (4/4) |
| T−U | final | +0.043206 (4/4) | +0.043651 (4/4) |

Seed-level T−F on d1, selected: t0 = +0.0392 / +0.0382 / +0.0342 / +0.0438; t1 = +0.0405 / +0.0389 / +0.0345 / +0.0444 (seeds 2022/2023/2024/2025). Worst seed +0.0342 — every seed clears the 0.003 threshold by an order of magnitude. Target LogLoss concurs: T−F = −0.0232 / −0.0197 (selected), −0.0262 / −0.0235 (final); every seed favorable.

### 1.3 Common-epoch trajectory of T−F on d1 (mean ΔAUC, sign count)

| Epoch | d1t0 | d1t1 |
|---|---|---|
| 9 | +0.024295 (4/4) | +0.023694 (4/4) |
| 19 | +0.032148 (4/4) | +0.032485 (4/4) |
| 29 | +0.038454 (4/4) | +0.038841 (4/4) |
| 39 | +0.042629 (4/4) | +0.042974 (4/4) |
| 49 | +0.043197 (4/4) | +0.043518 (4/4) |

The same monotone, 4/4-positive trajectory holds for T−N and T−U (e.g. T−N d1t0: +0.0283 → +0.0457; T−U d1t0: +0.0244 → +0.0432), and N−F / U−F stay at or below zero at every common epoch on d1.

### 1.4 Control-execution audit facts

- N matching: `r_N` mean 0.306 (range 0.030–4.278); target-only gradient norm is on average 34.8% of the FULL norm — the magnitude difference the N arm removes is large and real. Max matching residual 1.4e-08.
- U matching: `r_U` mean 0.914 (range 0.218–2.191). Max matching residual 5.0e-07. U behaves as designed: U−F is ≈ 0 on d1 (≤ 0.001 at every checkpoint) and small elsewhere, confirming the applied-step rescale was executed and that step-size alone changes almost nothing.
- Probes: 800 main + 800 architecture rows, all `zero_case = regular`; no direction-undefined step occurred in any arm/seed.

---

## 2. Replication gate (predeclared §6 prerequisite): PASSED

All four required conditions hold: mean R_T ≥ +0.003 on both d1 tasks (+0.0389 / +0.0396 selected); positive in 4/4 seeds (≥ 3/4 required); mean LogLoss favorable; R_T positive at 5/5 common epochs including epoch 49 (≥ 3/5 required).

Replication fact, stated as observation only: the fresh S4 T−F selected-checkpoint means (+0.03887596 / +0.03957562) coincide with the accepted S3 values (+0.038876 / +0.039576) to the reported precision. Under frozen seeds, data, and source inventory the S4 F/T arms deterministically reproduce the S3 F/T training trajectories. The S3 access-effect is therefore not a run-to-run artifact of that stage's execution.

## 3. Magnitude hypotheses (H-M, H-U): WEAKENED — both controls fail

- **H-M (raw gradient magnitude).** R_N is *negative* on d1 at both selected (−0.0040 / −0.0049, 0/4 seeds positive) and final (−0.0025 / −0.0020). N does not merely underperform T; it slightly underperforms F on the target domain while its matching audit passes exactly. Scaling the FULL gradient to the target-only norm reproduces **0%** of the rescue (80% of R_T would require ≈ +0.031). H-M is weakened decisively within this matrix.
- **H-U (applied update magnitude / Adam conditioning of the step).** R_U ≈ 0 on d1 (≤ 0.001 in absolute value at every checkpoint, 2/4 seeds). Matching the same-state target-only Adam *step magnitude* while preserving FULL moments reproduces **0%** of the rescue. H-U is weakened decisively within this matrix.

The failure is dissociated from matching quality: both controls hit their norm targets with residuals three orders of magnitude inside tolerance. The null is a property of the intervention, not of its execution.

## 4. Composition/history (H-C): SUPPORTED under the predeclared rule

Predeclared conditions, all met on both d1 tasks:

1. mean D_N ≥ +0.003 at selected and final: +0.0429/+0.0444 and +0.0457/+0.0455 — 4/4 seeds;
2. mean D_U ≥ +0.003 at selected and final: +0.0395/+0.0405 and +0.0432/+0.0437 — 4/4 seeds;
3. D_N and D_U positive at ≥ 3/5 common epochs including 49: 5/5 for both contrasts, both tasks, all 4/4 seeds;
4. same-state probes show a consistent target-favorable direction of Δ_T^cf over the equal-norm FULL update in ≥ 3/4 seeds aggregated over the 50 fixed main probe steps: satisfied **4/4 seeds in every arm** (mean `target_gain_target − target_gain_equal_norm_full` = +0.0039 to +0.0083). Architecture-step probes are neutral (mean diffs ~1e-8, sign split 19–35/50) — the effect localizes to the main shared parameters `C`, not the four `deep_weights` scalars.

Probe texture, reported honestly: the raw gradients are only moderately aligned — cos(g_F, g_T) ≈ +0.30–0.37 in F/U state (positive, not adversarial) and ≈ +0.60–0.76 in T state — while Adam's preconditioning aligns the *updates* far more (cos(Δ_F, Δ_T^cf) ≈ 0.90 in F/U state, ≈ 0.59–0.66 in T/N state). In F/U states the target-favorable advantage of Δ_T^cf holds on only ~17–25 of 50 individual steps yet the aggregated mean is positive in all seeds; in T/N states it holds on 42–48/50 steps. The probe signal is consistent in aggregate, not uniform step-by-step.

**Design boundary (binding):** this matrix cannot hold Adam's moment history fixed while changing gradient composition. What the four arms jointly establish is sharper than "composition matters" but narrower than "geometric conflict proven":

- The only arm whose Adam moments are built from the target-direction gradient (T) is the only arm that rescues d1.
- Arm N commits FULL-direction moments at target magnitude and fails; arm U commits FULL-direction moments with a target-magnitude applied step and fails.
- Within this matrix, the discriminator of the d1 rescue is therefore the **directional composition of the gradient stream committed to Adam's moments**, not the raw gradient norm (N) and not the instantaneous applied-step norm (U).

Whether the operative carrier is the moment history itself or the per-step direction (with moments as its passive integral) is **not separated** by S4; the design's pre-registered interpretation ("direction/composition plus its induced optimizer history") stands, and no claim of pure geometric gradient conflict is made.

## 5. Checkpoint/convergence (H-K): WEAKENED

The d1 rescue is present at every common epoch from 9 (+0.024) and grows monotonically to epoch 49 (+0.043); it is not a selection artifact. Selection behavior does differ across arms — T selects late (epochs 41/24/23/47) versus F (19/11/15/17) — i.e. removing non-target shared gradients shifts the validation optimum later, but the effect survives fixed-epoch comparison at all five nodes, so H-K does not explain it. Late-stage reversals occur only in non-target cells (§6), not on the target.

## 6. Non-target trade-offs (mandatory report; no deployment-safety claim)

| Contrast | Cell | selected | final | Note |
|---|---|---|---|---|
| T−F | d0t0 | **−0.031386** (0/4) | −0.017851 (1/4) | real cost, largest at selected |
| T−F | d0t1 | **−0.032142** (0/4) | −0.004876 (1/4) | real cost, decays late |
| T−F | d2t0 | −0.006924 (0/4) | **+0.019810** (4/4), ΔLL **−0.1478** | sign reversal over training |
| T−F | d2t1 | +0.002292 (2/4) | **+0.036160** (4/4), ΔLL **−0.1443** | sign reversal over training |
| N−F | d2t0 | +0.009914 (3/4) | **+0.025385** (4/4), ΔLL −0.1172 | N captures the d2 effect |
| N−F | d2t1 | +0.008535 (3/4) | **+0.028093** (4/4), ΔLL −0.1116 | N captures the d2 effect |
| U−F | d0t0 | −0.020549 (0/4) | −0.010862 (1/4) | U inherits F-like d0 behavior |

Three observations:

1. **d0 cost is real and seed-stable at selected** (−0.031/−0.032, 0/4 positive, magnitude > 0.005 = "strong" band), shrinking but not vanishing at final. The d1 rescue is not a general improvement; it is a reallocation.
2. **d2 shows a late-emerging benefit under T that N also reproduces** (final AUC +0.020–0.036, 4/4, with large LogLoss improvements ≈ −0.11 to −0.15). Absolute LogLoss shows F and U overfit d2 late (selected 0.67 → final 0.80) while T and N do not (≈ 0.65–0.69). The d2 effect is **magnitude-compatible** — reducing the non-target gradient norm (N) suffices there — in sharp dissociation with d1, where only target-direction moments work. This within-run dissociation is itself evidence that the d1 rescue is not a generic small-gradient effect.
3. **AUC/LogLoss disagreement** is confined to non-target cells (notably d2 at final: F's AUC is competitive while its LogLoss is much worse); on d1 the two metrics always agree in sign. The rescue claim is metric-consistent; the non-target story is metric-dependent and checkpoint-dependent.

## 7. Stable phenomena and counterexamples

| Phenomenon | Supporting evidence | Counterexamples / limits | Strength |
|---|---|---|---|
| S3 d1 target-only rescue reproduces fresh | T−F +0.0389/+0.0396 selected, +0.0432/+0.0435 final; 4/4 seeds both tasks; 5/5 epochs; deterministic match to S3 values | none in-matrix | **STRONG** |
| Raw gradient-norm account insufficient (H-M ✗) | R_N ≤ −0.002 on d1 at all 7 checkpoints, ≤1/4 seeds; matching residual ≤1.4e-08 | N *does* reproduce the d2 late benefit — magnitude matters somewhere, just not on d1 | **STRONG** (for d1) |
| Applied-step magnitude account insufficient (H-U ✗) | R_U ≈ 0 (≤0.001) on d1 at all checkpoints; U ≈ F everywhere; matching residual ≤5e-07 | U cannot address historical moments by construction | **STRONG** (within design boundary) |
| Composition/history support (H-C ✓) | D_N, D_U ≥ +0.039 at selected/final, 4/4 seeds, 5/5 epochs; probes target-favorable 4/4 seeds aggregated | step-level probe sign is mixed in F/U states (17–25/50); composition vs induced history not separable | **MEDIUM-STRONG** |
| Effect is not checkpoint artifact (H-K ✗) | monotone +0.024→+0.043 over epochs 9–49, 4/4 at every node | selection epoch itself shifts later under T (41/24/23/47 vs 19/11/15/17) | **STRONG** |
| d0 non-target cost at selected | −0.031/−0.032, 0/4 positive, both tasks | cost decays by final (−0.018/−0.005, 1/4) — magnitude is checkpoint-dependent | **STRONG** (at selected) |
| d2 late benefit is norm-driven | T and N both +0.02–0.036 AUC / −0.11–0.15 LL at final, 4/4; U absent | absent at selected; metric-dependent | **MEDIUM** |

## 8. Competing explanations and unidentified factors

- **Gradient direction/composition committed to Adam moments** — the only factor that perfectly separates T from {F, N, U} in this matrix and is supported by the same-state probes. Preferred account, bounded by the design's identification limit.
- **Optimizer-history (moment) pathway** — inseparable from composition here; U demonstrates that holding FULL moments while rescaling the step preserves F behavior, which is *consistent with* moments being the carrier but does not prove it.
- **Update-opportunity / exposure** — held constant (identical batches, step counts, exposures); excluded as a confound by construction.
- **Checkpoint selection** — excluded as the driver (§5), though it modulates the measured size of non-target effects.
- **Domain identity/difficulty** — d0 vs d2 respond oppositely at final despite both being non-target; physical-domain identity, not size role alone, determines the sign of the spillover. S4 was not designed to rank these factors.
- **Seed noise** — negligible for d1 conclusions (all key contrasts 4/4 with worst-seed margins ≫ threshold); material for d0/d2 at individual checkpoints, where sign counts are mixed.
- **Scope limits** — one dataset (KuaiRand), one backbone (M3OE), one objective (Equal), one target domain, four seeds, fixed 50-epoch budget. No universality claim is made.

## 9. Stage classification and decision

Applying the predeclared §6 rules mechanically: replication gate passed → magnitude-compatible test failed for both controls (R_N, R_U ≈ 0% of R_T; D_N, D_U both ≥ +0.003, 4/4 seeds) → probe criterion met (4/4 seeds, main steps) →

**Classification: Composition/history-supported (H-C).** H-M and H-U are weakened decisively; H-K is weakened; H-0 is rejected (the rescue reproduced and controls were executed exactly).

**Answer to the S4 research question.** Keeping the FULL shared-gradient direction while matching either the target-only gradient norm (N) or the same-state target-only Adam step magnitude (U) does **not** reproduce the d1 rescue — it recovers none of it. The rescue is associated with committing the target-direction gradient to Adam's shared-parameter moments; the cost is a seed-stable d0 degradation at the selected checkpoint (−0.031/−0.032) plus a metric-dependent d2 pattern that turns positive late and is separately reproducible by norm reduction (N).

**Stage decision.** S4 is complete and its question is answered within its pre-registered identification boundary. Recommended next-stage question (to be designed independently, not executed under S4): *interventions that swap or freeze Adam moment composition versus per-step gradient direction* (e.g., target-direction moments with FULL applied steps, or FULL moments with target-direction applied steps and matched norms) to separate the moment-history carrier from the instantaneous direction — the one dissociation this matrix could not perform. Any such stage must also carry forward the d0 trade-off as a first-class outcome; the d1 rescue must not be presented as a general improvement.

---

*Prepared by the main Agent from `stage_S4_A2_outputs` collector products only (queue `c7353a2e…`, 16/16 canonical runs, 4/4 blocks, pairing valid). Raw execution facts: `experiments/reports/stage_S4_A2_execution_report.md`; design and amendments: `experiments/reports/stage_S4_design.md`; reviews: `stage_S4_review.md` (A1), `stage_S4_A2_review.md` (A2).*
