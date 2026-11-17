# C05 research choice: remove class weighting while preserving cell mass

Research child: **Sagan / 01a1e9a9-73f5-7242-87d8-f78f0b61d86a**, original parent **01a1dfb8-605b-7430-8462-5fd3da0b048f**, fresh native context, `fork_context=false`; authorized model/effort gpt-6-astra/xhigh. Research and planning only. Parent performs the subsequent formal DESIGN, implementation, independent review, supervised RUN and scientific analysis. This handoff authorizes no descendants from this child.

Choose **CELL-MASS-BCE-025 (U)**: replace W's label-dependent coefficient inside each domain/task cell by its empirical mean in that cell. Preserve W's exact cell, task and global BCE coefficient masses on every natural batch. Add **one new arm, both datasets, all five seeds2022–2026, fifty epochs**: 10 new trajectories, 500 epochs, 17,250 Adam/EMA updates. Reuse C, W and S as thirty exact numeric controls. No HPO or follow-up arm is conditional on results.

The experiment asks whether W's class-dependent allocation is needed for its observed task1 loss benefit, and whether removing it changes KR d0t0 harm while retaining the cell allocation. This is a narrower, more identifiable question than explaining all of W−S by class imbalance. The proposed objective is a new intervention relative to this evidence chain, not a claim of a new general-purpose loss family.

## Scope and verified starting point

W=`/home/xzf/Code/phase0-3_tvg_rm_20261116`; O=`/home/xzf/Code/phase0-3`. Both `artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json` files read ACTIVE, `manual_stop_requested=false`, `termination_reason=null`, `stop_after_current_cycle=5`, `cycle06_authorized=false`. Neither workspace-root nor artifact-root `STOP_AUTONOMOUS` sentinel existed when inspected. Recheck controls before each later dispatch/admission.

Latest user scope is finish C04, complete exactly one C05, then deliver and USER_STOPPED. No C06 and no next Astra after C05. This overrides preserved infinite-loop and stop-after-C04 paragraphs. Stop That Shit CHANGE/hash=allow and Ponytail FULL apply. PREFLIGHT_RECEIPT PASS is reused; there is no new capability pilot.

C04 was actually published, not merely reviewed: `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/PUBLICATION_RECEIPT.json` records main/local/remote commit `cbccf5ae234e46bc3f976467de2b3327139f021b`, tree `e874ff09381557117d9e357048e3b59a475bd366`. Include this receipt in C05 publication. Frozen C04 source48 `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f` is untouched.

C04's 20 trajectories, 1,000 ACKs and 34,500 useful updates are complete. Forty selected models were directly verified at 11:21:33 UTC; owner2476891 was absent at 12:02:24, and its foreground tool later returned143. Exact exit time, sender and cause remain unknown. Current C04 models are unavailable. Original evidence FAIL is preserved; R1 is PASS WITH FIXES, scientific numeric evidence PASS. Numeric controls and archive remain valid. No C04 rerun, owner replacement, reconstruction, warm start or Test is part of C05. New U starts from approved source initialization.

## Evidence used to choose the next intervention

All seven maintained contexts are inputs: `EXP_AI_CONTEXT.md`, `MDMT_ALGORITHM_METRICS_LEDGER.md`, and `research_context/experiments/Experimental_{Global_Summary,Evidence_Ledger,Experiment_Index,Open_Questions,Stable_Findings}.md`. Architecture/history is read with the later natural-population evidence taking precedence. `AGENTS.md`, GB18030-decoded `start.md`, `buchong.md`, the runbook and Astra prompt supply execution constraints. The manifest binds these inputs and the full reports below.

| Evidence | Observation and counterevidence | Strength and design implication |
|---|---|---|
| NDP-A, `research_reports/auxiliary_objective_attribution_20261115/REPORT.md` | Natural 2B support and unit mass were implemented. Selected EMA−CLEAN AUC ML−.000101256, KR−.005775319; ML online+.000359628 and KR d2t0+.005083339 coexist with d0 harm. | MEDIUM against this fixed view configuration; natural weighting alone did not solve it. Do not relabel a supervised adversarial view or claim all views fail. |
| TV, `research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md` | γ=.10 improves KR relative to NDP-A by+.005608869, all five seeds, but still loses to CLEAN. ML γ=.20+.000153992, 3/5, CI crosses0. Residual history versus INSTANT has opposite dataset directions. | MEDIUM for recovery from NDP-A; WEAK for extra momentum necessity or useful matched-EMA increment. |
| C01, `research_reports/task_axis_even_views_20261116/REPORT.md` | AXIS-EVEN fails the primary on both means. All six directional/antithetic mechanism AUC intervals cross0. KR IID2+.001269552, 2/5, remains a weak exception. | MEDIUM fixed-configuration negative result; WEAK advantage for even geometry. Avoid another sign/radius variant. |
| C02, `research_reports/natural_quadratic_ranking_20261116/REPORT.md` | NQR ML selected EMA AUC−.000343517, all five negative; E10+.001151311, all five positive. KR largest-domain LL improves, but two smaller domains have opposing AUC directions. | MEDIUM stage and local task sensitivity; neither variance removal nor Brier control establishes ranking superiority. |
| C03, `research_reports/nqr_temporal_exposure_20261116/REPORT.md` | EARLY ML selected AUC−.000340045, all five negative. Turning N off improves LL versus constant N without restoring rank. Four E50 early/late AUC intervals cross0. KR LATE+.000738140, two positive/three exact-zero, is preserved. | MEDIUM ML evidence against late N exposure being the sole explanation; WEAK timing advantage. N's removal also removed W, leaving a confound. Identical prefixes are not new replications. |
| C04, full `research_reports/completion_weight_attribution_20261117/REPORT.md`, METHODS, DECISION, DATA_AVAILABILITY and EXECUTION_HISTORY | W EMA−C EMA: ML−.00015462144763278562, KR+.0025550857333183517, both intervals cross0. Both miss+.003; KRd0t0−.0031413726095260277 misses−.001. W−S task1 LL is−.002988932/−.005124581, all five lower on both datasets. | STRONG arithmetic/identity evidence, MEDIUM conditional task-loss signal, WEAK stable selected-AUC efficacy. The class/cell/task decomposition is the next untested, cheap intervention. |

C04's positive counterevidence stays central: KR W−C Macro LL−.003942334, five improvements; KR E50 AUC+.003242111, five positive; d2 selected domain AUC+.009201240, five positive; dominant d1 LL−.006935433, five lower. ML W−C task1 LL improves while task0's previous joint rank/loss cost is not consistently reproduced. W−S is not class-only. N−W on ML changes from E1 AUC+.019277577 to E50−.001496855 with E50 LL+.007416875; this warrants a timing competitor, not an assumption that curriculum works.

Natural train domain counts ML=[155949,318599,325619], KR=[172782,1743898,89452]. CLEAN selected EMA domain AUC ML=[.767958341,.756878065,.754468266], KR=[.686547442,.715993098,.670753584]. ML's smallest domain has the highest observed baseline AUC. KR's two smaller domains do not share effect signs. Label priors, validation counts, feature/domain identity, task, optimization and stage remain competing explanations. Size as the main cause and a uniform small-domain gain/large-domain decline trade-off remain **WEAK/unestablished**.

## Three mathematically distinct candidates

Full recurrences, assumptions, alternatives, resource costs, falsifiers and five-seed matrices are in ALGORITHM_DESIGNS.md; primary literature is in LITERATURE_AND_NOVELTY.md.

| Candidate | Changed mathematical object | Minimum complete matrix | Choice |
|---|---|---|---|
| U, cell-mass BCE | Conditional averaging of the coefficient field before differentiation; within-cell class residual removed, all six cell masses held exact | 1 new arm×2 datasets×5 seeds=10; C/W/S30 numeric controls | **CHOSEN.** Directly splits the confounded C04 allocation contrast with no new tuned coefficient, model pass, optimizer state or empirical premise. |
| V, predictive-KL view | Label-free local predictive-distribution curvature determines detached embedding perturbation; KL consistency instead of augmented supervised BCE | VAT and matched-radius random-KL, each2×5=20; C10 numeric controls | Lower priority. Mathematically distinct from NDP-A and axis even views, but more computation and overlap with established VAT; prior view gains are weak. |
| G, signed two-step gradient filter | Pre-Adam data-gradient recurrence with current/previous gradients; both Adam moments see the transformed input | Lead and lag, each2×5=20; W/C20 numeric controls | Lower priority. Tests temporal gradient processing but observed late/early trajectories do not establish one-step predictability; noise amplification is a concrete rival. |

Two more C04 questions were considered explicitly. Holding W fixed while switching only R early/late would require two new ten-epoch-exposure arms trained for fifty epochs each (20 new trajectories); unlike C03 it isolates R timing conditional on W, but retains within-class weighting and cannot explain W−S. Adding S+R requires one new arm (10 trajectories) and identifies the W/S×R interaction in the existing four-cell factorial; it still combines cell and class allocation in that interaction. Neither is a mandate. U is selected because one arm isolates the missing class component and also leaves an identifiable U−S cell/task contrast.

## What one complete C05 can decide

At common batch/state, W−U is exactly a within-cell class-coefficient residual; U−S is exactly cell/task mass redistribution at fixed global mass. After full training their paired outcomes identify the respective complete interventions, including parameter/Adam/EMA history; they do not estimate additive mediation percentages.

Directional mechanism predictions are precommitted separately from qualification: U−W should improve KRd0t0 loss/rank if W's within-cell class skew explains that harm; U−S should retain negative task1 LL on both datasets if cell/task mass allocation suffices for the repeated task1 benefit. Opposing means falsify these directional predictions; five-seed intervals/signs state their evidence strength. A crossing interval never establishes equivalence or necessity. If U loses the LL benefit while W retains it, the class component is implicated conditionally; if U retains it, the strong claim that class skew is necessary is weakened. Residual confounding with shared-parameter optimization remains explicit.

Primary efficacy stays independently selected U fixed EMA−C fixed EMA, separately ML/KR, five paired seeds, mean AUC≥+.003 and every cell mean≥−.001. Mean unscaled same-state LL<0 is judged separately. C05 ends with a complete report/review/publication even if every scientific prediction fails. No alternative candidate or C06 is launched.

## Grounded execution costs affecting this choice

The user's 13:19:49UTC measurement gives W/home3.0GB available, /tmp176GB, hostRAM27/62GiB (~44%, available30GiB), swap4.6GiB full; GPU0 used19571/24576MiB and251.66/350W, GPU1 used20071/24576MiB and287.73/350W. These are historical planning inputs, not launch permission. Preserve co-tenants including TV PID1025015.

C04 archive member accounting shows admission.jsonl1,372,015,267B and duplicated tables/RESOURCE_EXECUTION_AUDIT.json1,842,164,364B; runs196,703,067B, WEIGHTS246,011,471B, common-state diagnostic CSV224,978,014B. Thus repeated history/object metadata, rather than the ten new scientific trajectories, is the concrete capacity problem. CODEX_EXECUTION_PLAN defines a task-specific /tmp numeric root, normalized complete records, full-process witness references, and a quantified storage/publication budget. It preserves every decision and measurement.

No implementation, training, GPU manipulation, owner IPC/restart, new agent, background session or Test operation is performed by this research child. The only authored files are this directory's six requested artifacts.
