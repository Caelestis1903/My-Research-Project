# Stage S5 ANALYSIS REPORT — Main Agent

**Scope.** S5 mechanism factorization of the S3/S4 KuaiRand Equal `tab1` target-only rescue: four fresh arms `F` (FULL history + FULL current), `T` (TARGET history + TARGET current), `HT` (TARGET history + FULL current), `DH` (FULL history + TARGET current); seeds 2022–2025; M3OE; KuaiRand `natural`; Equal objective; 50 epochs. Executed in round 3 (queue digest `f6eceb00…`, 25-entry inventory, M3OE forward `a7702a64…`): 16/16 canonical runs, 4/4 four-arm blocks, `pairing_valid: true`, zero retries, zero failures. Per-step invariants I1–I9 held on every one of the 2,450 main steps in every run (`installed_cosine_history` = 1.0 exact, formula residuals ~1e-6, batch fingerprints identical across arms within seed). Data sources: `experiments/stage_S5_outputs/stage_S5_{runs,raw_results,paired_results,aggregates,probe,gradient_steps,moment_audit}.csv`. All statements below trace to those files; AUC differences are `left − right`; LogLoss differences are favorable when negative.

Conventions: `d1` = target domain `tab1`; `d0`, `d2` = non-target. Tasks: `t0` = click, `t1` = long_view. Predeclared practical threshold 0.003 (strong 0.005); 3/4-seed and 3/5-common-epoch stability rules; no p-values. `R_T = T−F`, `R_H = HT−F`, `R_D = DH−F`.

**Execution-history note (provenance, not results).** Rounds 1–2 each failed after completed training on inherited-code environment bugs (S4-inventory drift check; Windows venv-redirector PID binding in the overlap check), were root-caused, fixed under Coder→Reviewer cycle, and archived append-only (`.attempts.superseded_wave1/2`). Round-2 F/T runs were ruled non-promotable by the Reviewer (source-identity pinning) and are superseded evidence only. All numbers below are round-3 canonical runs.

**Moment-gate note (binding constraint from Reviewer ruling A).** The predeclared cross-run moment gate is 15/16: the single failing cell is seed 2024 / dump epoch 1 / first-moment m / HT side (cos(m_HT,m_T)=0.0413 vs cos(m_HT,m_F)=0.1222, margin −0.0809). The Reviewer ruled identification **valid**: per-step invariants prove the manipulation was executed exactly on every step; v-state decoupling passes everywhere with large margins; the m-cosine at epoch 1 measures trajectory-divergence speed of a signed EMA (concentrated in the sparse embedding block), not manipulation fidelity; leakage is positively excluded (cos(m_HT,m_F)=0.122, not ≈1). Constraints carried through this report: (i) the deviation is cited wherever mechanism conclusions are stated; (ii) no epoch-1 m-proximity claims are made for seed-2024 HT/DH; (iii) seed-2024-specific phenomena are flagged where they occur (see §7).

---

## 1. Complete results

### 1.1 Absolute test AUC at the selected checkpoint (4-seed means)

| Cell | F | T | HT | DH |
|---|---|---|---|---|
| d0t0 | 0.6351 | 0.6078 | 0.6284 | 0.6428 |
| d0t1 | 0.6521 | 0.6147 | 0.6457 | 0.6634 |
| **d1t0** | 0.6531 | **0.6946** | **0.6942** | 0.6540 |
| **d1t1** | 0.6815 | **0.7238** | **0.7221** | 0.6824 |
| d2t0 | 0.6378 | 0.6336 | 0.6528 | 0.6407 |
| d2t1 | 0.6580 | 0.6578 | 0.6772 | 0.6597 |

Selected epochs: F = 16/20/17/23; T = 49/43/43/40; HT = 37/32/41/49; DH = 15/15/15/25 (seeds 2022/2023/2024/2025). Note the selection-epoch shift itself follows the **history** factor: DH selects early like F; HT selects late like T.

### 1.2 Target-domain contrasts (mean ΔAUC, sign count over 4 seeds)

| Contrast | Checkpoint | d1t0 | d1t1 |
|---|---|---|---|
| T−F (R_T) | selected | **+0.041426** (4/4) | **+0.042254** (4/4) |
| T−F | final | +0.031068 (3/4) | +0.032166 (3/4) |
| HT−F (R_H) | selected | **+0.041085** (4/4) | **+0.040528** (4/4) |
| HT−F | final | **+0.044950** (4/4) | **+0.044909** (4/4) |
| DH−F (R_D) | selected | +0.000875 (3/4) | +0.000821 (2/4) |
| DH−F | final | +0.001634 (2/4) | +0.002708 (2/4) |
| HT−T | selected | −0.000341 (2/4) | −0.001726 (1/4) |
| HT−T | final | +0.013882 (1/4) | +0.012743 (1/4) |
| DH−T | selected | −0.040551 (0/4) | −0.041433 (0/4) |

Seed-level R_T on d1, selected: t0 = +0.0430/+0.0396/+0.0379/+0.0452; t1 = +0.0444/+0.0399/+0.0385/+0.0462 (2022/2023/2024/2025). Seed-level R_H: t0 = +0.0412/+0.0378/+0.0398/+0.0455; t1 = +0.0414/+0.0359/+0.0400/+0.0449. Seed-level R_D: t0 = +0.0020/−0.0042/+0.0003/+0.0054; t1 = +0.0020/−0.0030/−0.0001/+0.0044 — every seed within ±0.006, i.e. indistinguishable from F under the 0.003 practical threshold (2/4 and 3/4 sign counts).

**Fraction of the rescue recovered at selected:** HT = **99.2% / 95.9%** of R_T (t0/t1); DH = **2.1% / 1.9%**.

Target LogLoss concurs: T−F = −0.0305/−0.0275, HT−F = −0.0239/−0.0202, DH−F = +0.00002/+0.00025 (selected). One metric texture: HT−T d1 LogLoss = +0.0066/+0.0073 at selected — T's *current-direction* component adds a small LogLoss benefit on d1 beyond what history alone gives HT, despite AUC parity.

### 1.3 Common-epoch trajectory on d1 (mean ΔAUC, sign count)

| Epoch | T−F t0 | HT−F t0 | DH−F t0 | T−F t1 | HT−F t1 | DH−F t1 |
|---|---|---|---|---|---|---|
| 9 | +0.0234 (4/4) | +0.0237 (4/4) | −0.0002 (2/4) | +0.0230 (4/4) | +0.0231 (4/4) | −0.0007 (2/4) |
| 19 | +0.0313 (4/4) | +0.0311 (4/4) | −0.0013 (2/4) | +0.0318 (4/4) | +0.0311 (4/4) | −0.0005 (2/4) |
| 29 | +0.0271 (3/4) | +0.0368 (4/4) | −0.0006 (2/4) | +0.0273 (3/4) | +0.0363 (4/4) | −0.0003 (1/4) |
| 39 | +0.0427 (4/4) | +0.0368 (4/4) | −0.0020 (1/4) | +0.0428 (4/4) | +0.0367 (4/4) | −0.0014 (1/4) |
| 49 | +0.0311 (3/4) | +0.0450 (4/4) | +0.0016 (2/4) | +0.0322 (3/4) | +0.0449 (4/4) | +0.0027 (2/4) |

HT−F is positive 4/4 at **every** checkpoint and grows monotonically to epoch 49. DH−F stays within |0.0027| at every checkpoint — below the practical threshold at all times. T−F is positive in mean at all epochs but dips to 3/4 at epochs 29 and 49 (§7 counterexample: T seed-2024 late collapse).

### 1.4 Audit facts (execution fidelity)

- I1–I9 per-step hard invariants: zero violations in 16 runs × 2,450 main steps + 50 arch steps. `installed_cosine_history` = 1.0 exact; applied-delta formula residuals ≤ ~1.2e-6; batch-fingerprint sequences identical across the four arms within each seed (`pairing_valid: true`).
- Cross-run moment audit: 15/16 gate cells pass; the one failure (seed 2024, epoch 1, m, HT side) is ruled a statistical property of the signed first-moment EMA under legitimate trajectory divergence, concentrated in the sparse embedding block, with leakage positively excluded (see head-note and `stage_S5_review.md` Moment-gate investigation).
- HT's applied delta is the FULL counterfactual under target-history state and DH's the TARGET counterfactual under FULL-history state — verified per-step by I3 (applied cosine vs the D-source = 1.0).

## 2. Replication gate: PASSED

Mean R_T = +0.0414/+0.0423 at selected (≥ +0.003, both d1 tasks); positive 4/4 seeds (worst +0.0379, an order of magnitude above threshold); mean LogLoss favorable (−0.0305/−0.0275, every seed favorable); positive mean at 5/5 common epochs including 49. The S3/S4 rescue therefore replicates a third time, in fresh runs, at slightly larger magnitude than S4 (+0.0389/+0.0396). Unlike S4, this replication is **not bit-exact** to earlier stages (selected epochs differ from S4: F 16/20/17/23 vs 19/11/15/17) — expected, since the S5 code path inserts read-only diagnostics into the step and CUDA atomic ordering is not bitwise-stable across code paths; within-S5 pairing is exact and is the basis for all contrasts.

## 3. The S5 research question: answer

> When the persistent Adam history source and the current applied-update source are manipulated independently, which factor does the d1 rescue follow?

**The rescue follows the persistent history. Almost perfectly.**

- **HT (TARGET history + FULL current) recovers the entire rescue:** 99.2%/95.9% of R_T at selected, 4/4 seeds on both tasks, at every checkpoint from epoch 9 to 49. HT−T ≈ 0 at selected (−0.0003/−0.0017). The current-step gradient seen by the shared parameters in HT is the FULL gradient — the target direction is *absent* from the current update composition in the naive sense, yet the rescue is fully present.
- **DH (FULL history + TARGET current) recovers nothing:** 2.1%/1.9% of R_T at selected, |R_D| ≤ 0.0027 at every one of the 7 checkpoint views, sign counts mixed (1–3/4). DH−T = −0.0406/−0.0414 ≈ −R_T. The current-step target direction, when applied through FULL-built moments, is worth zero on d1.

**Classification: Pattern A — history-dominant** (HT ≈ T, DH ≈ F), and in the strong form: the instantaneous target direction is *neither necessary* (HT rescues without it) *nor sufficient* (DH fails with it). Pattern E is excluded by the DH arm itself: if HT≈T were produced by implementation leakage, the same leakage would lift DH, which stays glued to F; the Reviewer's leakage audit and the per-step I3 assertions independently exclude this.

This refines the S4 interpretation. S4 established that the discriminator was "the directional composition of the gradient stream **committed to Adam's moments**" but could not separate the moment history from the per-step direction. S5 separates them: the carrier is the **persistent moment state** (m, v, step counter) shaped by the target gradient stream. What the current update is made of hardly matters for d1; what the preconditioner has become over training is essentially everything.

Mechanistically, the Adam applied step is the current gradient preconditioned by history-built m̂/√v̂. Under HT the FULL current gradient — which contains the target's own gradient components every step — is preconditioned by a target-shaped state; under DH the target current gradient is preconditioned by a FULL-shaped state. The results say the d1 effect lives in the shape of that preconditioner, not in the composition of the vector being preconditioned. *Which* component of the state (first moment, second moment, bias-correction effective step size) carries the effect is not separated by this matrix (§8).

## 4. Non-target trade-offs (first-class outcome)

| Contrast | Cell | selected | final | Note |
|---|---|---|---|---|
| T−F | d0t0 | **−0.027338** (0/4) | −0.007278 (1/4) | real cost, largest at selected |
| T−F | d0t1 | **−0.037392** (0/4) | −0.025811 (0/4) | real cost |
| HT−F | d0t0 | −0.006705 (2/4) | **+0.017840** (4/4) | cost mostly absent; turns positive late |
| HT−F | d0t1 | −0.006369 (2/4) | **+0.023840** (4/4) | cost mostly absent; turns positive late |
| DH−F | d0t0 | +0.007671 (3/4) | +0.007972 (2/4) | no cost |
| DH−F | d0t1 | +0.011365 (3/4) | −0.000821 (3/4) | no cost |
| T−F | d2t0 | −0.004570 (2/4) | +0.013070 (4/4), ΔLL −0.1477 | late benefit, as in S4 |
| T−F | d2t1 | −0.000168 (2/4) | +0.022614 (4/4), ΔLL −0.1396 | late benefit |
| HT−F | d2t0 | **+0.014991** (4/4) | **+0.045920** (4/4), ΔLL −0.1561 | benefit early AND late, largest of all arms |
| HT−F | d2t1 | **+0.019234** (4/4) | **+0.053451** (4/4), ΔLL −0.1501 | benefit early AND late |
| DH−F | d2t0 | +0.002883 (2/4) | +0.023719 (4/4), ΔLL −0.1090 | late-only benefit |
| DH−F | d2t1 | +0.001695 (3/4) | +0.021204 (4/4), ΔLL −0.0938 | late-only benefit |

**d0 damage follows neither single factor — it requires both.** The seed-stable d0 cost (−0.027/−0.037, 0/4 positive) exists only in T, where the target-shaped preconditioner acts on the target-only direction. Remove the target current direction while keeping the history (HT) and the cost shrinks ~80% at selected (−0.0067/−0.0064, 2/4) and reverses to a uniform 4/4 *benefit* at final (+0.018/+0.024). Remove the target history while keeping the current direction (DH) and there is no cost at all (+0.008/+0.011). The d0 damage is therefore an **interaction phenomenon** — it is the target-history-preconditioned *target-direction step* (which contains no d0 gradient signal yet moves along target-momentum-shaped coordinates) that displaces shared parameters away from d0's optimum. When the same preconditioner instead acts on the FULL gradient (HT), the d0 components of g_F are still present in every step and d0 is preserved.

**d2 benefit follows primarily the history factor.** HT shows it earliest and largest (selected +0.015/+0.019 4/4; final +0.046/+0.054 4/4), T shows it late (+0.013/+0.023 final), DH shows a smaller late benefit (+0.024/+0.021 final). S4 found the d2 effect norm-reproducible (arm N); consistently, all three arms here that reduce non-target gradient *influence* on the applied step show some d2 relief late, but the history-carried version (HT) dominates at every checkpoint. F's late d2 LogLoss overfitting (selected ~0.66–0.70 → final ~0.76–0.82) is avoided by all three intervention arms.

**Observation, stated without method claim:** at the final checkpoint HT is above F on all six cells with 4/4 sign counts, and at selected on 4/6 cells. This is recorded as mechanism evidence (rescue without damage when history and current are decoupled this way), **not** as a deployable configuration: HT's selected-checkpoint d0 cost reaches −0.038/−0.030 in seed 2024 (§7), its d1 LogLoss is slightly worse than T's, and S5's mandate is diagnosis, not method innovation.

## 5. AUC / LogLoss consistency

On d1 the two metrics agree in sign for every contrast at every checkpoint (rescue favorable in both; DH neutral in both). Disagreements are confined to non-target cells and to the HT−T texture noted in §1.2 (T's current-direction component adds d1 LogLoss benefit without AUC benefit over HT). The d2 LogLoss improvements (all arms, final, −0.09 to −0.16) are larger in relative terms than the AUC changes — as in S4, the d2 story is metric-dependent and the d1 story is metric-consistent.

## 6. Checkpoint / training-stage consistency

The d1 conclusion (Pattern A) holds identically at selected, final, and all five common epochs — it is not view-dependent. Within that: T's rescue is stable early but shows late instability in one seed (§7); HT's rescue is monotone and 4/4 everywhere; the d0 damage is largest at selected and decays late (as in S4); the d2 benefit grows late. Selection-epoch behavior itself tracks the history factor (DH selects F-like early; HT selects T-like late) — further evidence that the persistent state, not the current direction, shapes the training trajectory.

## 7. Counterexamples and seed-level deviations

1. **T seed 2024 late collapse (the main counterexample).** T−F d1 at final = −0.0169/−0.0132 for seed 2024 (other seeds +0.046 to +0.048); T seed-2024 d1t0 falls from 0.6926 at its selected epoch (43) to 0.6344 at epoch 49 with LogLoss 0.5109→0.5694. This makes T−F final 3/4 and produces the epoch-29/49 sign-count dips. No such collapse appears in HT (same target history, FULL current) — HT seed-2024 d1 final = +0.0433/+0.0436 vs F. The target-only *current* update thus carries a late-training instability on d1 that the target *history* does not. S4 did not observe this (T final was 4/4 at +0.043/+0.044); with n=1 seed it is recorded as a counterexample, not a trend.
2. **HT seed-2024 d0 damage at selected** (−0.0382/−0.0296, near T's level for that seed): the ~80% average d0-cost reduction under HT is not uniform across seeds (2/4 sign count). Per Reviewer constraint (iii): this is the same seed whose epoch-1 m-gate cell failed; the per-step invariants still prove correct manipulation, and seed-2024 HT's d1 rescue is unremarkable (+0.0398/+0.0400), so identification is unaffected — but the coincidence is flagged honestly as a limitation on any seed-2024-specific HT claim.
3. **DH d0 selected** mean is positive (+0.0077/+0.0114) but seed-sign-mixed (3/4; range −0.0048 to +0.0277): treated as "no d0 cost", not as a real d0 benefit.
4. **Moment-gate cell** (seed 2024, epoch 1, m): see head-note; ruled non-identifying-relevant by the Reviewer with binding constraints.

## 8. Competing explanations, confirmed and unconfirmed

**Confirmed within this matrix (conditions: KuaiRand-Pure, M3OE, natural, Equal, tab1 target, 4 seeds, 50 epochs):**
- The d1 rescue is carried by the persistent Adam history built from the target gradient stream (Pattern A, strong form: target-current neither necessary nor sufficient). Strength: **STRONG** (4/4 seeds, both tasks, all 7 checkpoint views, ~96–99% recovery vs ~2%).
- The d0 damage requires the joint presence of target history and target current (interaction), and is largely avoidable by the HT configuration. Strength: **STRONG** at selected (T 0/4 vs HT/DH non-negative means), **MEDIUM** on uniformity (HT seed-2024 exception).
- The d2 late benefit follows primarily the history factor, with a secondary late current-direction contribution. Strength: **MEDIUM** (checkpoint- and metric-dependent).
- The S4 magnitude exclusions (N, U) stand and are now complemented: not the norm, not the step size, not the current direction — the state.

**Not confirmed / still open:**
- **Which state component carries the effect** — m, v, or the bias-corrected effective step size were manipulated as a bundle. A targeted follow-up (e.g., transplanting only v, or only m, between streams) is the natural next factorization if this question is taken up.
- **The geometric mechanism** of why target-shaped preconditioning of the FULL gradient helps d1 (anisotropic step-size reweighting vs implicit regularization vs curvature alignment) — S5 localizes the carrier but does not open it.
- **Generality** — one dataset, one backbone, one objective, one target domain, four seeds; no cross-setting claim.
- **Whether the seed-2024 T collapse and the seed-2024 HT d0 damage share a cause** — single-seed phenomena, flagged, unexplained.
- **HT as a configuration** — characterized diagnostically only; its selected-checkpoint d0 cost in one seed and d1-LogLoss gap vs T prohibit any "safe method" reading (CLOSED-12 stands).

## 9. Stage classification and decision

Applying the predeclared S5 logic: replication gate passed (§2); HT recovers ~96–99% of the rescue at selected and 100%+ at final (4/4, all epochs); DH recovers ~2% (below threshold at every view) → **Classification: Pattern A — history-dominant, strong form.**

**Answers to the S5 mandate (§十一):**
1. F/T rescue replicated: **yes** (+0.0414/+0.0423 selected, 4/4; LogLoss favorable).
2. HT recovers d1 rescue: **yes, essentially completely** (HT ≈ T at selected; HT > T at final due to T seed-2024 collapse).
3. DH recovers d1 rescue: **no** (≤0.0027 at every checkpoint; ≈2% of R_T).
4. Which factor explains the rescue: **the persistent Adam history**; the instantaneous target direction is neither necessary nor sufficient.
5. Joint contribution / interaction: **none for d1 rescue** (history alone accounts for ~all of it); **yes for d0 damage** (requires both factors jointly).
6. d0 damage follows: **neither factor alone — the interaction**; it is ~80% reduced under HT at selected and fully reversed at final (4/4).
7. d2 late effect follows: **primarily the history** (HT earliest and largest), with secondary late current-direction contribution (DH final).
8. AUC/LogLoss: **consistent on d1**; non-target LogLoss texture noted (d2 strongly favorable in all intervention arms late; T's d1-LogLoss edge over HT).
9. selected/final/common-stage: **the d1 conclusion is view-invariant**; non-target effects are checkpoint-dependent as documented.
10. Counterexamples: **T seed-2024 late d1 collapse; HT seed-2024 d0 damage at selected; DH d0 seed-sign-mixed; one moment-gate cell** (ruled non-identifying).
11. Confirmable now: history is the carrier of the rescue; damage and rescue are **different causal objects** (different factor dependence).
12. Not confirmable: which moment component; the internal geometry; generality; the seed-2024 anomalies; any safe-method claim.
13. Next stage warranted: **yes** — S5 shows the rescue (history-carried) and the d0 cost (interaction-carried) are separable in principle, making "target rescue without non-target damage" a well-posed design question for S6; a secondary candidate question is the m/v/effective-step-size sub-factorization of the history carrier. S5 itself designs no fix and claims no method.

**Boundary restated.** All conclusions are conditional on KuaiRand-Pure / M3OE / natural / Equal / tab1 / seeds 2022–2025 / 50 epochs. No universal negative-transfer mechanism, no universal gradient-conflict mechanism, no Safe-MTL method, and no deployable algorithm is claimed. The moment-gate deviation (seed 2024, epoch 1, m, HT side; Reviewer ruling A) is a standing qualification on every mechanism statement above.

---

*Prepared by the main Agent from `stage_S5_outputs` collector products only (round-3 queue `f6eceb00…`, 16/16 canonical runs, 4/4 blocks, pairing valid, moment gate 15/16 with documented ruling). Raw execution facts: `experiments/reports/stage_S5_run_report.md`; design: `stage_S5_design.md`; implementation: `stage_S5_implementation.md`; reviews incl. moment-gate investigation: `stage_S5_review.md`.*
