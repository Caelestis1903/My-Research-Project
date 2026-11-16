# Cycle03 research exploration — 2026-11-16

Decision: **GO_TO_FORMAL_DESIGN**, for one narrow falsification experiment, **NQR temporal exposure**. The candidate is `NQR-EARLY10`; `NQR-LATE10` is its equal scheduled exposure control. Exact numeric CLEAN and constant `NQR-025` trajectories are historical controls, conditional on compatibility verification. This is a temporal intervention on an established objective, not a claim of a new ranking loss or proven curriculum benefit. Efficacy is unproved.

This six-file handoff is research preparation, not the formal COMPLETE design or permission to train. The original current-conversation parent alone dispatches a fresh designer, coder, distinct reviewer and runner, each with `fork_context=false` and explicit file handoffs. No child descendants. All cycle/experiment limits remain null; candidate rejection returns to the parent's research loop.

## Evidence basis and scope

Read the current W/AGENTS.md, GB18030-decoded O/start.md, W/buchong.md and execution runbook; the current foreground policy overrides retired background-driver instructions. Both workspaces' stop controls were ACTIVE, manual stop false, no USER_STOPPED termination or STOP_AUTONOMOUS sentinel. Only this directory's six requested files are written. No source edits, training, GPU queries, IPC, owner contact, Git operations, remote writes or Test access occurred.

Full-history inputs include EXP_AI_CONTEXT.md, MDMT_ALGORITHM_METRICS_LEDGER.md and all five Experimental context files, including historical blocks; NDP Stage1/Stage2 reports and tables; full TV and AXIS-C1 reports, decisions, numerical appendices and raw tables; C1's seven design files including the DT-PGNA/TGA completeness addendum; and corrected C2 reports, seven-file design, source package and numeric epoch records. The large CSVs were streamed across all rows rather than pasted into context. This is an evidence synthesis, not a fresh audit of every historical experiment's implementation.

Read-only checks verified all 87 frozen C2 source file hashes and aggregate `ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae`; all ten reused CLEAN result hashes and 500 epoch hashes matched. All 30 C2 result files and 1,500 numeric epoch records were read: 51,750 updates, Test=false throughout. No model tensor was loaded. The manifest records file receipts and numeric inventory.

Publication is already satisfied by the supplied/locally read receipt: commit `ed34eb243099dc2693f06ba566fc689c7657f641`, tree `ad6b794d83de8a75ba161c1e3466016c31f6e069`, 150 published files/161101288 bytes, 18 archive parts/146878639 bytes, 5305 members/391354662 uncompressed bytes, whole archive SHA `86b7cf1faa0cccd096fbc4e0a72b532322f4f60f4a0a36afa969abeb0606cef0`. No new remote verification or publication is attempted by this child.

## What the history does and does not establish

| Evidence | Observation | Strength and implication |
|---|---|---|
| S0–S8 and older curriculum studies | Matched ordinary EMA weakened the uniqueness of domain-conditioned momentum. Sample joint training did not exhibit universal practical negative transfer. Equal-weight/history interventions were conditional; matching raw gradient norms did not isolate the Adam mechanism. E5 did not beat static Sample; S8 late recovery did not beat the original selected best. | MEDIUM within those settings. Neither generic curriculum efficacy nor a universal M3oE failure on hard domains is established. Old artificial populations, other backbones and old Test results are not current CLEAN controls. |
| NDP attribution | Stage1's mass-1.5 controls were confounded. Repaired natural `.75C+.25V` still failed to beat CLEAN EMA: ML −.000101256, KR −.005775319 mean selected Macro AUC. | MEDIUM that the original mass mistake is not a complete explanation of later negative outcomes; not proof that every augmentation fails. |
| TV | ML gamma .20: +.000153992, 3/5 positive, CI crosses zero. The all-gamma/all-dataset nonpositive predicate is FALSE. Rank winner .10 remained negative on both datasets. | WEAK local positive counterexample; no champion. Recovery relative to failed NDP does not establish improvement over CLEAN. |
| AXIS-C1 | Primary AXIS-EVEN: ML −.000237520, KR −.003540710. KR AXIS-IID2 +.001269552, 2/5 positive, CI crosses zero. ML task0 AUC −.000570378 and LL +.002315558, all five unfavorable. | MEDIUM negative primary evidence; WEAK IID2 counterexample. No control promotion. |
| C2 primary | NQR selected EMA minus CLEAN: ML −.000343516900, 0/5 positive, CI [−.000547122,−.000139912]; KR −.000328734160, 2/5 positive, CI [−.006717219,+.006059751]. All three new arms have negative mean AUC on both datasets. | STRONG for the observed five-seed ML selected deficit, WEAK for a KR directional deficit. C2 remains NO_CONTINUATION_SIGNAL and ineligible. |
| C2 ingredients | Four NQR-minus-BRIER/MEANGAP AUC CIs cross zero. NQR-minus-BRIER LL improves in **4/5 seeds on each dataset**, both CIs cross zero. ML NQR-minus-MEANGAP LL −.002033597, 4/5 favorable, CI below zero, but both are worse than CLEAN. | WEAK evidence for ranking-specific AUC benefit; MEDIUM for the narrow ML relative LL difference. Corrected final R1 is PASS, zero blockers. Preserve initial FAIL as history, not current verdict. |
| C2 time | ML E10 EMA AUC +.001151311, 5/5 positive, CI [+.000005901,+.002296722], then selected −.000343517, 5/5 negative; E30 −.000289384 and E50 −.000565383. | MEDIUM stage dependence within these trajectories. Epochs are not independent replications. This motivates a bounded timing falsifier; it does not prove an early-only intervention will work. |
| C2 tasks and calibration | ML task0 AUC −.000416581 and LL +.003874856, all five unfavorable, repeating C1. KR d1 LL −.006196973 and pooled LL −.006051198, all five favorable, despite negative Macro AUC mean. KR traffic/pooled AUC positive in 4/5, both CIs cross zero. | MEDIUM recurring task-specific vulnerability and metric separation. Improved calibration-like loss is not a ranking gain. |

These evidence labels describe the scoped observations, not universal truth or multiplicity-adjusted proof. Five seeds and repeated reuse of the same validation populations constrain generalization.

## Domain size, difficulty and competing explanations

Natural train counts ML d0/d1/d2 = 155949/318599/325619; KR = 172782/1743898/89452. KR d0 positive prevalence is about .090889/.041214 across tasks, d1 .377832/.256083, d2 .616867/.484807. Counts and label opportunities therefore covary with identity and task; small is not synonymous with hard. Baseline cell AUC/LL must be shown alongside every delta before describing difficulty.

C2 KR selected domain AUC deltas are −.006386723/+.000165424/+.005235096. The two smaller domains have opposite signs; the dominant domain has near-zero AUC change and clear LL improvement. C1 also had opposite signs in its smaller domains. **Size as the principal causal explanation is WEAK. There is no unified small-domain gain/large-domain harm trade-off.** Neither natural observations nor the older fixed-N role experiments alone identify size causality.

At least six explanations remain live: (1) continued exposure erodes an initially useful trajectory; (2) an insufficiently aligned objective harms the selected solution regardless of timing; (3) prevalence-dependent pair opportunity changes effective influence; (4) feature/task sharing produces the repeated ML task0 deficit; (5) calibration and ranking respond differently; (6) Adam scale/history, EMA lag and selection create stage-dependent observations. Raw gradient norms cannot alone choose among them. This experiment addresses (1) against (2), while explicitly retaining (3)–(6).

## Three mechanisms considered

Exact formulas and local non-equivalence are in ALGORITHM_DESIGNS.md; primary sources are in LITERATURE_AND_NOVELTY.md.

| Mechanism | Intervention | Why selected or deferred |
|---|---|---|
| A: temporal exposure | Apply the exact frozen NQR scalar loss only in epochs 1–10, versus 41–50 at identical lambda .25. | SELECTED. Uses the new stage reversal, keeps objective/optimizer/model intact, and needs only two new arms. Can directly falsify preservation of the early benefit. |
| B: opportunity flattening | Replace natural `2*pi0*pi1` completion mass by a fixed .5 in two-class cells, retaining natural domain weights. | DEFERRED. Mathematically changes class influence, not just timing. It breaks the natural pair marginal and neutral-score derivative identity. Rare-positive noise and changed gradient scale demand additional attribution controls. |
| C: parameter-neighborhood minimization | Known SAM perturbation of model parameters before the training gradient. | DEFERRED. A different mechanism from score moments or feature augmentation, but it needs a radius and another backward. Already considered, unrun, in C2; no new curvature evidence makes it the best next discriminator. |

No evidence supports adding all three to the matrix. No lambda, radius, schedule-window or seed search is authorized.

## Selected falsifier and interpretation

Twenty new full trajectories: two datasets × five seeds (2022–2026) × EARLY10/LATE10. Historical CLEAN and constant NQR contribute twenty exact numeric control trajectories. All run 50 epochs, 40960 FP32, source natural rows and loader order, source Adam and fixed EMA. Both new arms expose every row to the active objective ten times; each has equal scheduled lambda sum 2.5 over epochs. This is not equal realized gradient dose: parameters, Adam moments, EMA recency and batch class composition differ by phase.

Primary efficacy remains EARLY10 minus CLEAN, selected EMA Macro AUC, separately for ML and KR. The prespecified mechanism contrast is EARLY10 minus LATE10 at **fixed E50**, online and EMA separately, followed by EARLY10 minus constant NQR at E50. Selected comparisons are also reported but cannot resolve the timing mechanism alone.

Selection trap: LATE10 reproduces CLEAN through E40. All five ML CLEAN EMA best epochs (23,19,20,28,21 in seed order) precede E41, so its selected EMA AUC cannot fall below CLEAN if identity is preserved. A zero difference can simply retain an old best. Some KR CLEAN best epochs are after E40 (44,39,32,50,37), so that bound does not hold there. EARLY10 reproduces constant NQR through E10: its initial positive finding is an identity check, not new replication.

If EARLY10 fails to beat CLEAN at its selected endpoint, reject the proposed preservation claim even if it recovers some constant-NQR damage. If EARLY10 and LATE10 both improve relative to constant NQR, reduced total exposure is plausible; it is not isolated from timing. If only EARLY10 improves fixed E50 online and EMA relative to LATE10, timing under this exact training system is supported, but no pure phase mechanism independent of optimizer/EMA history is identified. An EMA-only advantage leaves EMA lag as a strong competitor. A positive selected result with a negative E50 result is selection-dependent, not durable recovery. Wide CIs or inconsistent seed directions remain inconclusive; they do not authorize a rescue sweep.

The practical validation screen is mean Macro AUC +.003 with all six mean cell deltas at least −.001, comparison-specific; LL is assessed separately. No conclusion is drawn before all registered trajectories finish. This cycle cannot establish external generalization or open Test. The parent alone performs the eventual scientific analysis and chooses fresh research after any candidate outcome.
