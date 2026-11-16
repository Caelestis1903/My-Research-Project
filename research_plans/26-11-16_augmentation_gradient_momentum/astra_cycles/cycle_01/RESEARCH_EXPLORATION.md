# Fresh Astra cycle 01 — research exploration
Date: 2026-11-16. Status: DESIGN COMPLETE; implementation and independent review pending. This is a file-based handoff from a fresh native child to foreground parent thread `01a1dfb8-605b-7430-8462-5fd3da0b048f`. No child delegation, implementation, training, Test evaluation, commit, or push occurred.

## Decision
Select **task-axis even views (AXIS-EVEN)**, with **AXIS-IID2** and **ISOTROPIC-EVEN** controls. This changes the objective's odd/even decomposition at fixed input radius and natural support. It does not tune TV-GRM gamma, add another residual channel, or assume momentum works. The three explored mechanisms span input-view construction, output-gradient decomposition, and a same-batch optimizer recurrence. Only the first is selected.

Question: does removing the one-sided, loss-ascent component of NDP-A recover useful task-axis smoothing, or merely recover CLEAN by making the perturbation almost ineffectual? Random-axis and independent-sign controls make those explanations separable. A recovery relative to NDP-A alone will not count as improvement over CLEAN.

## Source and policy
Workspace: `/home/xzf/Code/phase0-3_tvg_rm_20261116`.
Published, remote-verified main/HEAD: `c4dc36bd38bd4f7a2a3319377a8289ba8185532d`.
Published tree: `bae323bf8c33e8144eaf225633d2fd481f553df8`.
Publication receipt: `artifacts/26-11-16_augmentation_gradient_momentum/PUBLICATION_RECEIPT.json`; SHA-256 `7f77c799f034118c3885221f88561aa5dea19c59c8939f48b6038cc0766550f3`.
Old TV-GRM training-source aggregate remains `d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6`; published commit identity and historical training-source identity are different fields.

Read policy first: AGENTS.md, buchong.md, CURRENT_SESSION_POLICY_CODER_REPORT.md, CURRENT_SESSION_POLICY_REVIEW.md. Then current AGENT.md, EXP_AI_CONTEXT.md, MDMT_ALGORITHM_METRICS_LEDGER.md, PROJECT_ARCHITECTURE.md and all seven requested research_context/experiments records. Read the complete current TV-GRM report, decision, numerical appendix and final evidence review; loaded all six current raw tables and COMPLETE_RUNNER_REPORT.json. Read NDP-A REPORT.md and Stage2 design, decision and final review; loaded Stage2 ENDPOINTS/PAIRED_SEEDS and costs. The NDP handoff dossier's 34 files, including exact sibling raw results, were hash-verified. Machine-readable identities and exact 20 reusable-control records accompany the manifest.

Both workspaces' stop-state files were ACTIVE / DESIGN_RESEARCH with manual_stop_requested=false and test_accessed=false when inspected; no STOP_AUTONOMOUS sentinel was found. The old driver is DISABLED_BY_USER_POLICY / RETIRED. This observation does not authorize skipping the next stop check.

## Evidence that constrains the next experiment

All deltas below are validation, earliest strict Macro-AUC-selected fixed EMA versus its source-matched CLEAN fixed EMA, unless stated otherwise. Five paired seeds are the unit.

| TV-GRM gamma | MovieLens Macro AUC delta | KuaiRand Macro AUC delta |
|---|---:|---:|
| .05 | -0.000328815 | -0.000226846 |
| .10 | -0.000062272 | -0.000166450 |
| .20 | +0.000153992 | -0.001310579 |

There were 50/50 new full trajectories, 2,500 epochs, 86,250 Adam and EMA updates, plus 20 exact historical controls. Test remained false. Scientific review PASS had zero blockers. NO_CONTINUATION_SIGNAL means no champion emerged; the ranking winner .10 remained negative on both datasets. The all-gamma/all-dataset-nonpositive NO-GO condition is **false**, because ML .20 is positive. Its AUC is WEAK evidence: 3/5 positive, 95% paired-t interval [-0.000388187,+0.000696171]. Its logloss change is -0.001597351 with 5/5 reductions. Neither fact erases the other.

KR .10 minus NDP-A is +0.005608869, 5/5 positive, CI [+0.000165186,+0.011052552]: MEDIUM evidence of recovery from NDP-A, not superiority to CLEAN. NDP-A's selected-EMA changes were ML -0.000101256 and KR -0.005775319. NDP-A already fixed objective mass and covered all 2B row-task entries; 17,250 Stage2 batches passed the natural-mass audit. Thus its failure cannot be assigned to the older extra-BCE mass defect without new evidence.

TV .10 minus actual post-cap norm-matched SHUFFLE has positive mean AUC on both datasets (ML +0.000204767, KR +0.004101641), but both intervals cross zero. KR logloss improves in all five seeds. TV .10 minus INSTANT is ML -0.000084288 and KR +0.001715651, again intervals crossing zero. Direction and residual history therefore have WEAK efficacy evidence. The near-one q/current-residual cosines do not establish a useful long-memory mechanism.

Training stage changes the story: TV .10 E50 deltas are ML +0.000713 and KR +0.001645, whereas selected endpoints are negative. Earlier ML E10/E30 are approximately -0.000450/+0.000138; KR -0.001551/-0.000663. Keep selected endpoints primary and fixed epochs descriptive. EMA also explains much of the apparent improvement versus online CLEAN: ML .20 is +0.003150 versus online but only +0.000154 versus fixed EMA; KR .10 is +0.001477 versus online but -0.000166 versus fixed EMA.

Aggregation changes the story too: KR .10 domain deltas are approximately [-0.005369,+0.001452,+0.003418], task deltas [-0.001305,+0.000972], with traffic-weighted +0.001393 and pooled +0.001403 despite negative Macro AUC. ML .20 domain changes [+0.000469,-0.000255,+0.000248], task changes [+0.000247,+0.000061], and worst-cell -0.000016 are not a uniform six-cell gain. KR .20 improves worst-cell AUC about +0.000601 while degrading Macro AUC.

Engineering evidence is STRONG: SHUFFLE's actual post-cap norm matching passed (maximum relative error approximately 3.30e-8 ML / 5.87e-8 KR), as did gamma-zero/CLEAN equivalence. The logged clean/residual cosine is a **data-gradient cosine**, not an Adam-update cosine. Five first-batch diagnostic epochs per trajectory are dependent observations, not extra statistical samples.

## Size, difficulty, baseline, and competing explanations

CLEAN selected EMA mean AUC/logloss: ML .759768224/.524690036; KR .691098041/.428639572.

| Dataset, domain | Natural training rows | CLEAN task0 AUC / LL | CLEAN task1 AUC / LL |
|---|---:|---:|---:|
| ML d0 | 155,949 | .764868 / .583580 | .771048 / .443893 |
| ML d1 | 318,599 | .758080 / .575002 | .755677 / .481985 |
| ML d2 | 325,619 | .755200 / .582951 | .753736 / .480730 |
| KR d0 | 172,782 | .678220 / .288469 | .694874 / .151442 |
| KR d1 | 1,743,898 | .702408 / .497339 | .729578 / .341356 |
| KR d2 | 89,452 | .668583 / .645034 | .672924 / .648199 |

These are descriptive populations and baseline outcomes, not a difficulty ordering inferred from size. In KR the two smaller domains move in opposite directions under TV .10, while the largest improves. ML .20 improves the smallest and largest and declines in the middle. A stable small-domain gain / large-domain loss trade-off is not supported. Domain size is therefore WEAK as a main explanation here; domain/task gradients, label prevalence, representation, calibration, training stage, and selection interact. No resizing, resampling, synthetic domains, or size-proxy experiment is proposed.

Stable observations: fixed EMA is a strong internal control; fixing mass alone did not repair NDP-A; TV recovered KR NDP-A losses without clearing CLEAN; diagnostics can pass while endpoint gains fail. Contradictions to preserve: ML .20 positive, LL/AUC disagreement, online/EMA disagreement, endpoint/fixed-epoch disagreement, macro/traffic disagreement.

Competing explanations:
1. One-sided ascent adds a harmful first-order term; even task-axis smoothing may retain a useful second-order effect.
2. Any recovery is simply a smaller effective regularizer. Isotropic views or CLEAN may perform equally well.
3. Antithetic sampling reduces conditional estimator variance but adds no endpoint benefit, as the earlier ANTI2/IID2 experiments already suggest.
4. Calibration-oriented gradients can help LL while hurting ranking. Candidate B tests a distinct output-gradient decomposition.
5. Current-batch gradient change, not augmented-view residual history, may matter. Candidate C tests a STORM-derived transport recurrence.

## Alternatives and selection
A: even task-axis views removes the odd objective component exactly at a frozen probe and admits two compute-matched controls. B: cell-centered output-error routing separates common output error from within-cell contrast without new input views. C: same-batch two-parameter gradient transport changes temporal estimation without an augmentation residual. B and C are specified but not authorized for implementation by this cycle.

Select A because its algebra addresses a documented failed objective and exposes a clear “recovery only” outcome; B requires a less evidenced calibration-interference assumption, and C spends compute on temporal variance reduction despite weak existing history evidence. None is claimed novel or effective from its name.

## Fixed matrix and budget
New arms AXIS-EVEN, AXIS-IID2, ISOTROPIC-EVEN; each MovieLens and KuaiRand; every arm uses seeds 2022,2023,2024,2025,2026. Exactly 30 new trajectories, 1,500 epochs, 51,750 Adam and EMA updates. Conditional exact reuse: 10 CLEAN + 10 NDP-A historical trajectories. Failed identity proof pauses the matrix for revised design/review; it does not silently add confirmations or substitute a universal baseline.

Recorded NDP-A means are 1328.847 s/ML and 3362.975 s/KR per full trajectory; reserved peaks 2.354/2.813 GiB. Three such ten-trajectory blocks cost 19.549 worker-hours. Five training forwards rather than three gives an operation-count planning point of 32.582 worker-hours; 19.55–32.58 is a planning range, not a measured bound. Probe/evaluation overhead, RNG, and shared-device contention alter the ratio. The last 50 new TV runs consumed 37.774 worker-hours over about 5.47 elapsed hours; do not equate those units. Prospective A memory is roughly 3.92/4.69 GiB by a crude 5/3 graph scaling, with 4.5/5.5 GiB initial admission reservations until a measured full-batch smoke replaces them. Full batch and FP32 never shrink to fit.

Both per-card memory and power must remain below 85%, host RAM below 90%; admit additional workers on each card when feasible. Only current-conversation runner supervision is allowed. No outer cycle, Astra, experiment, or revision cap is introduced. Candidate failure produces evidence and fresh research. Only explicit USER_STOPPED normally ends the line.

