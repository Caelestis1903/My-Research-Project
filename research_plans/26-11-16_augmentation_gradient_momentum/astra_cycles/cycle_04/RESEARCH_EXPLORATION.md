# Cycle04 research exploration — completion-weight attribution

Decision: **GO_TO_FORMAL_DESIGN**. Research blockers: **0**. This is a research handoff, not a DESIGN_SPECIFICATION, implementation approval, or permission to bypass review. Candidate efficacy is unproved.

Research identity: fresh native child `01a1e51b-7e46-77a0-8096-f7871fce0a03` (Wegener), actual `gpt-6-astra/xhigh`, explicit `fork_context=false`, parent `01a1dfb8-605b-7430-8462-5fd3da0b048f`. Identity is supported by native session metadata/first turn context and `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/PARENT_RESEARCH_PREFLIGHT.json`. This child dispatches no descendants and closes permanently after this handoff.

All relative paths are under `/home/xzf/Code/phase0-3_tvg_rm_20261116` (W). Control workspace O is `/home/xzf/Code/phase0-3`. The six-file manifest binds evidence paths and hashes. Published Cycle03 is commit `4c95dc877b85cc676bf7366958f8c7de0291871e`, tree `0dbbe5c0c53dc843a1effd929df197cc7399f18d`; its receipt records actual remote-main verification of 105 files before this dispatch. Local detached HEAD/tree agree. No new remote query or publication was performed by this child.

## Question and selected experiment

Does NQR's selected MovieLens ranking cost require its quadratic ranking gradient, or can its retained BCE completion weights and their overall scale already cause it?

Run two new arms for both datasets and all five seeds: **COMPLETION-BCE-025** (candidate W) and **MASS-SCALE-025** (control S). Reuse the exact ten CLEAN (C) and ten constant NQR-025 (N) numeric trajectories. The objective chain is `C -> S=mC -> W -> N=W+R`, with the same batch BCE coefficient mass m in S and W. It separates a particular global scaling intervention, nonuniform completion weighting conditional on that mass, and adding the ranking term conditional on W. The source Adam, model, batch orders, EMA and selection rule stay fixed.

This deletes a whole component from the objective; it is not a renamed NQR schedule, a smaller coefficient, or an epoch rescue. Lambda remains .25 throughout all 50 epochs. Two new arms are the minimum for this conditional decomposition. A full weighting-by-ranking factorial would add `S+R`, ten more trajectories, to estimate an interaction that is not needed to answer the selected question. That interaction is explicitly unidentified here.

## Evidence read and numerical audit

Read complete REPORTs and availability statements for C3 `nqr_temporal_exposure_20261116`, C2 `natural_quadratic_ranking_20261116`, C1 `task_axis_even_views_20261116`, TV `task_view_gradient_residual_momentum_20261116`, and NDP `auxiliary_objective_attribution_20261115`. Read their raw paired, endpoint, diagnostic and statistic records, not only report introductions. C3's DECISION, full numerical appendix, selection structure, resource audit, phase counters and schedules were examined. C2's full appendix and mass summaries were examined. NDP's EVIDENCE and STAGE2_EVIDENCE were parsed in full.

All 18,020 C3 and 12,580 C2 paired rows were recomputed against their respective 3,604 and 2,516 summaries: paired delta, seed set, mean, sample SD, t4 interval and sign counts agreed (maximum numerical discrepancy 0.0). These are many correlated measurements, **not** 30,600 independent observations. C3's 34,500 schedule records and 1,000 epoch phase counters reconcile to 6,900 active plus 27,600 off updates. All 1,020 result/epoch files in C3's reused-reference index were read and their SHA-256 verified. Their ten CLEAN plus ten NQR trajectories are available without old-owner IPC.

Read EXP_AI_CONTEXT, MDMT_ALGORITHM_METRICS_LEDGER, all five main `research_context/experiments/Experimental_*` files (Global_Summary, Stable_Findings, Open_Questions, Hypothesis_History, Dead_Ends), and relevant ledger/contradiction/index history. Historical scale, private-Adam, selected-versus-late, EMA and gradient-interference findings inform the alternatives below. Frozen source was traced through loss algebra, gated model forward, loader RNG, private Adam, EMA, evaluator and RAM restore/admission. C1's authoritative source is its RESOURCE_1_SUPERSESSION, not its superseded initial receipt.

## What the completed experiments establish

All deltas below are left minus CLEAN, validation-selected fixed EMA unless explicitly qualified; negative log loss is favorable. Intervals are descriptive paired t4 intervals, not multiplicity-adjusted discovery tests.

| Evidence | Observation | Strength and implication |
|---|---|---|
| C3 EARLY10, MovieLens | AUC -0.000340045; 0/5 positive; CI [-0.000459987,-0.000220103]. LL -0.000928551, interval crosses zero. | **STRONG** observed arithmetic/sign fact; **MEDIUM** repeated ML rank evidence under this reused validation protocol. Its loss result is uncertain. |
| C3 EARLY10, KuaiRand | AUC -0.000068613; 3 positive/2 negative; CI [-0.008122287,+0.007985061]. LL -0.000928905, interval crosses zero. | **WEAK** evidence of any directional effect; mean-negative does not mean every seed failed. |
| C3 EARLY10 minus constant NQR, MovieLens | Selected LL -0.003069812, 5/5 lower, CI negative; AUC +0.000003472, CI crosses. | **STRONG** observed loss arithmetic; **MEDIUM** evidence that stopping the last 40 intervention epochs improves relative loss without recovering selected rank. Not proof of equivalence or general timing causality. |
| C3 E50 EARLY minus LATE | All four AUC intervals cross: ML online/EMA negative, KR online positive/EMA negative. | **WEAK** timing-direction evidence; neither dataset establishes a simple timing law. |
| C3 LATE selected ML | Five selected EMA states exactly equal CLEAN, at epochs 23,19,20,28,21 before intervention begins at 41. | **STRONG** structural identity; it bounds selected harm and is not a late-intervention benefit. KR late +0.000738140, two positive/three exact zeros, CI crosses, is an internal-control result only. |
| C2 NQR, ML/KR | AUC -0.000343517 / -0.000328734; ML 0/5 positive, CI negative; KR mixed and wide. | **MEDIUM** ML-specific repeated deficit, **WEAK** KR direction. C3 E10 is the exact C2 prefix, not replication. |
| C2 alternatives | BRIER AUC ML -0.000200870/KR -0.000977209; MEAN-GAP -0.000348939/-0.001818484; all four intervals cross. NQR minus MEAN-GAP ML AUC +0.000005422, LL -0.002033597 (4/5 lower, CI negative). | **MEDIUM** evidence against a simple variance-removal rescue of ML rank; not evidence all quadratic interventions are universally harmful. |
| C2 vs C3 ML task0 | C2 AUC -0.000416581 and LL +0.003874856, both 5/5 adverse with signed intervals. C3 AUC -0.000408706, 4/5 negative with negative interval, but LL -0.000986610, interval crosses. | **MEDIUM** repeated task0 rank cost, **not** repeated joint rank-and-loss cost. The experiments share populations, seeds and early history. |
| C2 KR large d1 | LL -0.006196973, 5/5 lower, CI negative; pooled LL -0.006051198, 5/5 lower. | **STRONG** counterexample to universal large-domain loss harm. Macro rank remains a separate outcome. |
| C1 AXIS-EVEN | ML AUC -0.000237520, 0/5 positive; KR -0.003540710, 1/5 positive; intervals cross. ML LL +0.001264852, 5/5 worse. | **MEDIUM** cumulative reason not to assume deterministic view regularization helps. |
| C1 AXIS-IID2 KR; TV gamma .20 ML | +0.001269552 (2/5 positive) and +0.000153992 (3/5 positive), respectively; both AUC intervals cross. TV ML LL -0.001597351, 5/5 lower. | **WEAK** positive AUC counterexamples retained. TV's original all-gamma-nonpositive condition was not triggered; lack of practical qualification is different. |
| NDP natural-support correction | After correcting the earlier mass-1.5 setup to natural 2B mass1, ML -0.000101256 (1/5 positive), KR -0.005775319 (0/5 positive). | Correct support is necessary for interpretation, not sufficient for efficacy. |

C3 completed 20 new and 20 reused numeric trajectories, all 50 epochs/five seeds, with 1,000 new epochs, 34,500 updates, zero failed formal attempts. It observed 20 concurrent CUDA workers, ten per card. Its final evidence review is Halley PASS with zero blockers. These execution facts validate the receipts, not the algorithm's efficacy. The candidate outcome was NO_CONTINUATION_SIGNAL; no champion or Test evaluation resulted.

## Size, difficulty and task structure

| Dataset | Train/validation rows | Train domain sizes d0,d1,d2 | CLEAN selected EMA domain AUC d0,d1,d2 |
|---|---|---|---|
| MovieLens | 800167 / 100021 | 155949,318599,325619 | .767958341,.756878065,.754468266 |
| KuaiRand | 2006132 / 250767 | 172782,1743898,89452 | .686547442,.715993098,.670753584 |

CLEAN selected EMA macro AUC/LL is .759768224/.524690036 on ML and .691098041/.428639572 on KR. Baseline performance and label prevalence are descriptive difficulty indicators, not an identified latent difficulty variable. KR C3 domain AUC deltas are -0.003499683/-0.001043873/+0.004337717, all intervals crossing zero. The two smaller domains move in opposite directions. C2's corresponding AUC directions also differed by small domain. **WEAK** evidence that size is the main explanation; no established small-domain-gain/large-domain-decline trade-off. C3 KR macro LL has a lower mean while pooled LL has a higher mean (+0.000520874); both are uncertain. Report both, without using pooled performance to rescue the macro primary.

All current evidence is for this fixed1 M3oE adaptation, these natural populations, source optimizer and training/selection stages. It does not establish M3oE failure on difficult domains, or that small domains are difficult. Curriculum and momentum receive no presumption of efficacy. The earlier S4 norm-matching, S7/S8 selected-continuation, 11/09 coupled-WD scale and 11/11 open-loop/feedback results specifically discourage equating gradient magnitude, local gain, late gain and selected trajectory gain.

## The unresolved common component

C2 NQR, BRIER and MEAN-GAP all retained completion-weighted BCE. C3 changed when that same composite objective was active, not its active-step completion weights. Thus these experiments do not isolate that component.

From C2's complete NQR MASS_SUMMARIES, the epoch row-weighted BCE coefficient mass averages .894768747600 on ML (range .894766142410–.894772479792) and .899469997620 on KR (range .899467793321–.899472466681), over five seeds by 50 epochs. These are epoch summaries, not batch extrema or independent efficacy replicates. Roughly ten percent of nominal mass is allocated to the ranking component. Dropping ranking while retaining its BCE component must therefore compare against a correspondingly scaled CLEAN objective. A normalized W/m would ask another question and is excluded.

## Three distinct mechanisms considered independently

1. **Deterministic completion-only pointwise BCE (selected).** Delete the entire opposite-class ranking term, retain exact class/opportunity completion coefficients. The S control matches total BCE coefficient mass each actual batch. No auxiliary gradients are applied, no new RNG/history/state exists. The decisive falsifier is a persistent ML rank deficit without any ranking gradient. This is an attribution experiment even if the candidate fails.
2. **Conditionally zero-mean view-gradient noise.** At a common state use `g_C + (.25/2) zeta (g_plus-g_minus)`, with an independent symmetric Rademacher zeta and frozen paired task views. Conditional expectation is g_C, whereas AXIS-EVEN changes the mean objective. This tests whether stochastic perturbation, rather than mean view regularization, explains the weak KR IID2 signal. A direction claim additionally requires variance/norm-matched isotropic noise; CLEAN, AXIS-EVEN and IID2 are existing contextual controls, not substitutes for that control. At least two new arms/20 trajectories and extra view derivatives would be needed. C1's 2/5-positive KR signal is weak; current common weighting confounding is better evidenced. Noise is neither a scalar weight change nor a rank objective. Not selected.
3. **Instantaneous ranking-gradient projection.** At a common state set `g_W + (I-P_gW)g_R`, where `P_g v=<v,g>g/||g||^2` and P=0 if g=0. This removes the component parallel to the pointwise gradient without history; a control `g_W + kappa g_R`, with kappa matching projected residual norm, is needed to distinguish direction from magnitude. Projection plus matched control would need W as an additional baseline if run now (at least 30 new trajectories), alongside reused N/C. This is a generally nonconservative vector update, unlike the selected scalar objective, and unlike TV's accumulated residual momentum. Historical norm/interference and feedback results weaken an untested assumption that local conflict causes selected harm. No direct evidence yet warrants its more complex private-Adam/activity contracts. Not selected.

These are mathematically different interventions on deterministic risk, conditional gradient variance, and gradient direction. None is offered as a globally novel algorithm. The selected test has the clearest controlled contrast and smallest sufficient new-run budget.

## Precommitted interpretation and next research

Primary remains W selected EMA minus CLEAN selected EMA separately per dataset. W versus S and N versus W are mechanism contrasts; a favorable S, epoch10, pooled score or LL cannot replace the candidate primary. Both primary mean AUC deltas <=0 means NO_CONTINUATION_SIGNAL for W alone; one positive means MIXED. Practical qualification requires mean AUC >=+.003 on each dataset and all six per-cell mean AUC deltas >=-.001 on each dataset. LL is judged separately at those same selected states.

If W keeps the ML deficit, the ranking term is not necessary for that deficit. If W-S is repeatedly adverse while S-C is small with sufficiently precise intervals, heterogeneous weighting is implicated. If S-C tracks W-C while W-S remains uncertain, global scaling remains plausible but is not proved the sole cause. If deleting R recovers N's deficit yet W still fails CLEAN, removing ranking is useful attribution, not an effective candidate. Zero-crossing intervals never establish equivalence.

Common-state diagnostics will separate raw gradients from immediate source-Adam updates, and online/EMA comparisons will describe the EMA readout interaction. They cannot identify full-trajectory causal mediation by Adam, parameters or EMA. N-W is the effect of adding the ranking objective conditional on W, including its magnitude and downstream path, not a pure direction-only effect. The four objective means telescope algebraically; they are not percentages of a biological-style mediation decomposition.

After parent analysis and real main publication, fresh Astra must choose the next falsifier from the observed uncertainty: e.g. the missing S+R interaction if ranking-by-weighting interaction becomes central, a specifically justified scale/regularization discriminator if scale dominates, or the zero-mean noise mechanism if deterministic-risk explanations fail. Do not preauthorize those runs, tune lambda, shorten training, select a favorable control as champion, or end the uncapped loop on this candidate's outcome.
