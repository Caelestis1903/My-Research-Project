# CONDITIONAL DESIGN ADDENDUM — v3 P2 and locked reevaluation

Status: DESIGN COMPLETE, before conditional implementation or execution. This document activates only previously specified branches; it does not revise P1 algorithms, thresholds, metrics, reported outcomes or source. Authority: v3 file02 §§2.8,3.12,3.13; file03 §§7–10; approved DESIGN_SPECIFICATION §§8–12; and pre-result `artifacts/mdmt_next_v3/v3_0/ORCHESTRATOR_DECISION_CLARIFICATIONS.json`. The designer has inspected current interfaces but has not implemented or run an extension. The numerical support booleans below were supplied by the orchestrator and verified as values in `P2_DECISION_ARITHMETIC.json`; no scientific outcome analysis is performed here.

## 1. Research questions and activated scope

1. Does current paired acceptance add value beyond historical decisions when history shares exactly the same first initialized PB main block?
2. Does the L4 read mechanism outperform a probability-residual explanation in its own reachable continuous read space?
3. Does cross-target meta supervision add something that same-owner meta supervision cannot explain?
4. Does learned formation improve beyond supported fixed geometric formation under the same modern bank/read graph?

Activated full24 P2 queue:

| Branch | Data | Seed | Number | Frozen activation basis |
|---|---|---:|---:|---|
| R7_PB_HISTORY | KuaiRand only |3101|1| R6 has restricted KR support against all R0–R5; complete the current-feedback attribution before a principal RHD formal lock |
| LPS_CAL_PROB_CLIP | ML and KR |3101|2| L4 versus L2 and versus L3 each meets the original global AUC compute-allocation heuristic |
| LPS_OWNER_META | ML and KR |3101|2| Same supported AUC attribution path |
| LPS_GEOM | ML and KR |3101|2| Same supported AUC attribution path |

Exactly seven complete24 trajectories are activated. Six LPS trajectories satisfy the maximum six-run LPS P2 budget. There is no PAIR branch because the AUC attribution branch is supported. STATE/C14 are not activated by these booleans. TARGET_DIRECT_WRITE, new reader/backbone, and joint RHD/LPS remain outside this task.

The FEATURE comparison is a separate principal-claim boundary. Its failed global support does not erase the already specified L2/L3-driven bounded P2 attribution work, and those P2 runs do not automatically authorize formal LPS evaluation. Complete all seven even when intermediate results are negative; do not stop/reconfigure one method based on partial quality.

## 2. Constants and variables

All data/row/encoder identities, logical40960/drop_last=False, main natural BCE, FP32/noAMP/noTF32, coupled Adam, RNG namespaces, main/support/panel/query/coverage plans, fixed24 endpoint, eligible≥12 selection, six-cell metrics, and complete policy selection rules remain as approved. All reading-family forks start from the verified L1–L5 shared **epoch10 pre-refresh** full state, not an already L4-shaped state. LPS new variants share exactly the epoch1–10 main graph/coverage trajectory; each performs its own required epoch10 initialization afterward.

Only the following change:

- PB_HISTORY: timing of the deployed action choice after the shared first block; the current event can affect future history but not the already fixed current action.
- CAL_PROB_CLIP: deployment read policy and the required coefficients interface; formation remains L4.
- OWNER_META: formation outer query target; allocator, fit graph, support, virtual Adam and read policy remain L4.
- GEOM: owner responsibility generator; fixed geometric weights replace learned φ and its meta steps, not the modern detached bank/full-owner/continuous-read graph.

All new code belongs in a separate conditional package, recommended `experiments/mdmt_next_v3_conditional_20261025/`, with imports/subclasses of the reviewed base where appropriate. The existing71-file base receipt and source hash `3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2` remain immutable. This addendum does not authorize rewriting the P1 files or disguising different conditional semantics under an old arm ID.

## 3. Implementation interfaces and provenance

Actual base interfaces are in `experiments/mdmt_next_v3_20261025/lps.py`: `LpsModel.features`, `from_features`, `predict`; `TargetReadPolicy.coefficients`; `LpsTrainer._main_weights`, `main_step`, `_meta_features`, `_virtual_bank`, `_outer_risk`, `fit_policy`, `_policy_risk`, `evaluate`, `refresh`; functional Adam already exists. `LpsTrainer` currently rejects conditional names. Actual R7 timing/first-block fork is present in `rhd.py` but dormant and is **not** covered merely because P1 review passed.

Use explicit extension classes and coefficient context arguments. A common extension prediction helper may take `(features, current_packet, policy, context)` with contexts MAIN_DEPLOY, META_SUPPORT, META_QUERY, READ_CACHE, EVAL, DELETION. A CAL policy needs `coefficients(u,z0,residuals,domain)`; do not force it into a two-argument API through global mutable cached features or hidden state. Inspect and override **every** path that invokes the base two-argument API, including main fit, support/query construction, cache P_cal risk, refresh gate statistics, final evaluations, deletion/cost benchmarks. Ordinary read gates can use a stateless adapter ignoring z0/residuals. A complete packet is produced before CAL coefficient selection.

Add an extension CLI (or an equivalent explicit entrypoint) without editing the frozen base CLI. Suggested commands: `validate-semantics`, `build-p2-plan`, `run-stage --stage p2 --resume`, `run-one`, `build-formal-plan`, `run-stage --stage formal --resume`, `run-reversal-diagnostic`, `summarize`. It must execute only exact frozen manifest/config rows after independent conditional review and concrete complete-package budget admission. A receipt's scope must explicitly include P2/formal/diagnostic as applicable; base P1 approval is insufficient.

Conditional source identity is a composite manifest: pinned base files/hashes plus all extension files/dependencies. Record extension hash, base hash, data/protocol/design hashes, actual method ID and parent checkpoint identity in each config/checkpoint. A fork across source versions is allowed **only** by an explicit receipt that verifies the exact frozen parent checkpoint and its source/data/seed/state boundary. Do not relabel an old source hash as a new one to pass `restore`, nor silently disable global source checking. Resume of a conditional trajectory requires its own unchanged composite source.

## 4. R7_PB_HISTORY definition

### 4.1 Shared prefix

Fork KR3101 from R6's recorded `first_block.pt`: epoch10 full-C8 Hblock initialization already measured/revealed/committed, and main block10.0→10.5 already executed with the accepted **or STATIC-fallback** action. Verify model, optimizer/moments/individual steps, all gradients, RNG, plan cursor, PB initialized flag, action/history, seed/data and parent checkpoint SHA. The checkpoint is an intra-epoch state: `epoch=10`, next main position=20 for actual KR40-step epochs. Do not interpret it as a completed epoch11 or replay that first half-block.

Import warmup/prefix validation records and shared event/main-action records with their old identities/provenance; do not invent missing timings. Independent deployment cost includes epoch1–10 and the first main block even though they are not physically rerun. Reset only method-specific future selected tracking when this is logically an epoch10.5 fork; keep controller history from the one initialization. Preserve the first-block physical cost relationship in all three ledgers.

### 4.2 Events after10.5

For each current block, before revealing current labels: generate the same-rule shortlist (STATIC, prior Hblock EMA-best nonstatic, least-recently measured remaining legal action), then choose `argmax m_j(a)` among that shortlist, with STATIC mean fixed0; ties STATIC then action ID. Record this `precommitted_history_action` before probing.

Run the same Hblock support/branch count/panel plan and preserve same-budget current P_update measurement and selected P_cal pair bookkeeping. Restore the true main state, observe the event once into history for future events, then deploy the **precommitted** history action. A newly favorable candidate or cal acceptance cannot change that block. This deliberate timing differs from CUM/EMA, which correctly observe their current H1 feedback before choosing the immediately next block.

After divergence, R7 must use only its own states/candidates/labels. It may not replay R6's later utility table. Candidate identities may diverge while generation rules and upper budgets remain matched. First-block equality and pre-reveal action timing need focused review even though dormant R7 source exists.

### 4.3 Costs and records

Standalone24 PB_HISTORY has the same89H upper-bound shadow updates as R6 (KR1780) including shared8H initialization. The conditional suffix physically adds at most81H (KR1620), plus27 event scoring/state/cost components, while the shared first block is charged once in research actual and to both deployments. Record actual legal shortlist sizes and operations rather than assume every ceiling is reached. All normal24 validation and endpoint/selected evaluation requirements remain.

## 5. CAL_PROB_CLIP exact policy

### 5.1 Probability head and training objective

Six independent heads `u34→19 ReLU→1`, standard hidden Xavier/bias0, last weight/bias0. Same current u normalization as L4, initialized on epoch10 P_read and frozen. For each task/head raw correction is `c_eta(u)` and estimated probability `sigmoid(z0+c_eta(u))`. Its P_read training loss is **ordinary un-clipped** BCE on `z0+c_eta`; it fits the conditional probability residual. It uses P_read only, not all P_label. No residual r is an input to the head.

Adam .001,betas(.9,.999),eps1e-8,wd0; batch512; first10 passes, refresh2; six head/domain-specific shuffles as base. Candidates are old complete parameter+optimizer state and each full pass. **P_cal selection scores the actual clipped deployment logits**, never the un-clipped probability fit score. Select the entire six-head/optimizer group, with exact ties retaining the earlier/old group. Fit/cal objective difference is deliberate and must be logged.

### 5.2 Reachable read projection

For each row/task current eight real scalar foreign residuals r, `lo=sum_j min(r_j,0)`, `hi=sum_j max(r_j,0)`, `delta=clip(c_eta,lo,hi)`. The deployed logit is exactly `z0+delta`, implemented as `z0+sum_j alpha_j*r_j` with the unique minimum-L2 coefficients:

`argmin .5*sum(alpha**2) subject to alpha∈[0,1]^8, r·alpha=delta`.

Compute in FP64. If delta=0, alpha=0. Otherwise set q=max(sign(delta)*r,0), T=abs(delta); assign only q>0 entries. For current free set, mu=T/sum(q_free**2), alpha_free=mu*q_free. Cap entries>1 to1, subtract their q contribution from T, remove them and repeat, at most8 rounds. At boundary T=sum(q), all aligned entries are1; zero-r entries remain0. Enforce residual tolerance `abs(r·alpha-delta) <= 1e-7+1e-6*sum(abs(r))`; record maximum error. Handle the trivial all-zero-residual interval without divide-by-zero. Convert coefficients for FP32 main prediction only after FP64 solve, and record both solve and deployed arithmetic error.

This minimizes coefficient norm, not the number of blocks. The method still computes all needed residuals and is not sparse inference. Preserve current source/task ordering and signs.

### 5.3 Gradient and context rules

Main fit computes current real features/owner z0/all foreign residuals, obtains alpha, then detaches **all alpha** before native/owner/receiver backward. Neither probability head nor foreign bank gets fit gradients. Target reader remains trainable through `sum alpha*r`; native and owner learn through z0. Preserve None state for an unused reader when alpha is exactly0.

Meta support computes alpha from its current **real** state and detaches it, then uses that fixed support background to calculate routed owner-bank gradients with the original L4 differentiable Adam step. Meta query likewise computes alpha from real-current z0/residuals and freezes it. The75% deployed outer term replaces only foreign content by virtual bank outputs but **does not recompute alpha from virtual residuals**. Own z0 is current real detached. The25% .5 background is unchanged. Thus φ→g→virtualAdam→virtualforeign remains intact while the deployed read decision is fixed at the current snapshot. Neither head nor real bank/moments updates in meta.

Epoch10: L4 allocator16 meta steps first, using the initialized probability head/current read policy; probability-head fit afterward. Epoch12/14/.../24:8 meta steps then2 head passes.72 total meta steps and24 read passes. Restore full probability-head optimizer/normalizer state. Record interval coverage, lower/upper saturation, zero-alpha fraction, projection error, source usage, cal-clipped versus raw-head risks and prediction equivalence when c lies within the reachable interval. Do not use an un-clipped FEATURE score as CAL's deployment score.

## 6. OWNER_META outer-loss change

Use the same block allocator, support/query row schedules (fit512/source,P_form512/target), natural source masses, differentiable bank Adam,72 meta steps, two backgrounds, read gate, coverage and direct-write permissions as L4. Only outer query content changes.

For target/source t, use **its own virtual bank** for the owner contribution:

`zown_virtual_tq = detach(native_z_tq) + v_tq,t(detached parameters)^T sum_k e_tk(h;theta_virtual)`.

Foreign contributions come from the **current real bank**, with real reader values held constant; no virtual foreign replacement. The75% background uses current real deployment coefficients; the25% background uses all foreign coefficients .5. Compute two BCE risks then mix .75/.25 and average target/task cells equally. Consequently the two backgrounds may alter the own-virtual gradient via BCE, but only the query's own virtual block carries the φ path. The same-source query is additional owner meta supervision; the method is not the cross-target L4 objective relabelled.

Real native/bank/reader/moments and read gate do not update during this meta step. Only φ commits. The uniform formation reference uses identical real state, support, own-query definition and both real-foreign backgrounds. Record which outer bank paths are virtual/real and per-background query losses; unchanged fit gradients remain owner-only. Do not train owner virtual bank parameters directly on query or submit query gradients to real bank.

## 7. GEOM formation replacement

Start from epoch10 pre-refresh reading-family warmup, where w1 held throughout. Compute current a32 on per-source fit IDs, at most20000/source, with a dedicated predeclared label-independent subset plan; retain IDs/hash. K-means++ K4, seed=run_seed+91000+1009*s,n_init5,max_iter300,tol1e-4. Record package versions, convergence/inertia, centers/order and counts. Do not fit centers on P_form/P_read/P_cal or choose clustering by evaluation quality.

For each source, rho_k=.9*softmax(-squared_distance(a,center_k)/tau)+.025, w=4rho. Lock tau on this same fit feature sample to mean entropy `.5*log(4)` of the complete rho distribution specified in file02. Let v be median of positive squared distances, or1 if none; base=max(v,1e-6), bracket tau=base×[1e-4,1e4]. Bisect log tau60 times. If target entropy is outside attainable endpoint range, select endpoint with smaller absolute entropy error; ties select smaller tau. Record target, endpoints, attained entropy and any unattainability. No label-based choice.

Freeze centers and temperature from epoch10 onward; a continues to follow the current native representation. Use **current a** for weights on each main batch. Same responsibility floor, full owner sum, task application (same geometric weights for both tasks), detached bank input and real w detachment as L4. Same 8-dimensional continuous gate/read fit/coverage/selection; centers do not become foreign demand keys.

GEOM has no learned φ optimizer and performs **zero** formation meta steps, zero formation support/query label accesses and zero uniform virtual-reference computations. Do not charge it fictitious72 meta steps or perform dummy meta work to match an irrelevant budget. Its own fit subset feature extraction, K-means/temperature computation, repeated geometry/weight computation, cache/read fitting and checkpointing costs are real and must be logged. Read fitting still totals24 passes; epoch12/14/... refreshes the read head only. Save centers/tau/feature-subset identity and geometry allocation state in every required full checkpoint; resume must never refit centers.

## 8. Focused acceptance criteria

Independent reviewer reads base design+this addendum, actual extension code and conditional coder report. Verdict PASS/PASS WITH FIXES with all blocking issues corrected before admission. Required checks target changed semantics only:

1. **Provenance/forks:** base71-file digest unchanged; conditional composite hash complete; warmup and R7 first-block source crossing explicitly verified. Restoring a conditional last reproduces mode, normalizer, probability/φ optimizers or fixed geometry, main cursor and next row/action. No reinitialization at epoch10 resume; no repeated first block or event.
2. **PB_HISTORY:** shared initialization/main first-block digest equality to R6, then a constructed event with different prior-history winner/current measured winner proves deployment uses the recorded pre-reveal winner. After observe, current feedback enters only future means. P_cal pair bookkeeping cannot rewrite the committed history action. Own later labels, identical plans/candidate budget, candidate-list restriction and no extra C8 initialization verified.
3. **CAL projection:** signed residuals, zero/boundary/inside/outside cases satisfy[0,1], equality tolerance and unique min-L2 KKT structure (active/free alpha aligned with q). One independently solved tiny case may use a generic convex solver to verify minimal norm. Demonstrate actual clipped P_cal policy scoring and complete-state optimizer selection while raw un-clipped BCE trains the head. On a real batch, head/foreign bank receive no fit gradient, native/owner/used reader do; alpha is detached. A controlled virtual perturbation changes meta foreign content but not current-real alpha; φ hypergradient remains nonzero when fixture permits.
4. **OWNER_META:** a toy query perturbation of virtual owner changes outer loss/φ gradient; perturbing virtual foreign does not. Real foreign background remains present as required. Uniform reference shares the same own-query object. No real model/bank/moments/reader mutation except φ update; same query/support dimensions/natural masses.
5. **GEOM:** predeclared fit-only subset and deterministic center/tau receipts; responsibility sum4 and range[.1,3.7]; freeze/resume preserves centers/tau; current changed a changes weights without refitting. Meta step/access count0; two read-refresh passes and main owner routing still occur. Do not infer an error merely because entropy target is unreachable; record endpoint choice.

Existing functional-Adam/hypergradient receipts remain valid for unchanged helper source; add one focused changed-path meta check for CAL and OWNER. Maintain original tolerances. No new exhaustive test matrix or broad refactor.

## 9. Budget, logs and seven-run CLI plan

Freeze `P2_MANIFEST.json`, exact configs, dependencies and selected parent checkpoint hashes. Use the same deterministic coverage/meta/read plans and new method IDs; never modify P1 manifests. Register physical shared prefixes and independent deployment counts separately.

Budget each complete24 method/data from actual retained planned work; first necessary component run may stop at a registered retained boundary solely for measurement and then resume unchanged. Use matched load, observed component maxima where repeats are insufficient, counts and1.35 package margin. No fixed hour quota exists and no24-native-epoch deadline is allowed. P2 body:

- R7: suffix main13.5 epochs,27 Hblock events up to81H shadow updates (KR1620), scoring/snapshot/prefix import/eval. Deployment includes original complete warmup+first initialized block89H total shadow cost.
- CAL and OWNER each:14 main post-warmup epochs,72 meta steps including uniform references,24 complete read passes,8 refreshes (epoch10,12,...,24), all validation/cache/cal/projection/save costs. There are four such runs total.
- GEOM each:14 main post-warmup epochs, epoch10≤60000-row feature/cluster/temp work,24 read passes, no formation meta steps, normal repeated geometry costs. Two runs total.

Record research actual, independent deployment and development/search ledgers, including inherited prefix attribution and every diagnostic. Whole-package budget uses actual component sums without overlapping inclusive refresh double-counting. GPU0≤2 experiment processes; occupied GPU1 unavailable. Count inference/P50/P95/projection compute under a common declared setting; no unmeasured GPU-hour/energy claims.

Extend event/refresh/registry/access schemas with versioned explicit fields: policy_kind, raw_head_correction, projected_delta/error, feasible interval/inside fraction, coefficient_context/current-real-state hash, virtual-own/foreign flags, geometry fit IDs/centers/tau hashes, no-meta declaration, shared-prefix state digest, precommitted history action and selected complete-policy optimizer identity. Locally retain required selected/epoch24 predictions, validation each epoch, normal checkpoint set and exact deployed inference. Aggregate only COMPLETE24 rows; negative and failed run costs remain visible.

## 10. RHD bounded reversal diagnostic

The pre-result orchestrator clarification activates this diagnostic for a seed3101 dataset only if (i) R6's common epoch24 Q_explore macro LL is worse than STATIC, and (ii) at least2 PB events accepted a non-STATIC action with positive current paired utility. Verify both from existing full endpoint/event tables and write an eligibility receipt; do not substitute selected LL for the common24 trigger.

Eligible dataset(s) use only stored `diagnostic_before_epoch12` and `diagnostic_before_epoch20`, defined before P1 as the **first block in epochs12/20**, coordinates11.0/19.0 before probe/main. At most four snapshots overall; if only ML qualifies, exactly two. Do not choose a different favorable snapshot or manufacture an unavailable one.

For each snapshot compare its recorded PB candidate and STATIC from exactly the same full S using the **already occurred** main row block, not independent support: first H1, then H_hold under the fixed respective action. Starting from the two H_hold branch states, execute one identical next H_hold continuation row/action plan taken from the archived factual trajectory, independent of branch risk. Preserve current candidate even if original P_cal rejected it; candidate versus STATIC is the frozen diagnostic object. Report the original deployed action separately. The continuation policy must be identical in the two branches, using archived factual main actions, not branch-specific controllers/revealed future risk. No action or checkpoint search.

The orchestrator has supplied the completed trigger audit: ML3101 meets the common24/repeated-positive-event trigger, KR3101 does not. At **both** registered ML snapshots, the recorded P_update winner is already STATIC (action5). The candidate sets were[5,0,1] at coordinate11 and[5,4,3] at coordinate19; do not select one of their nonstatic losers, change snapshot, or redefine candidate as historical winner to create a nonzero contrast.

Register exactly two ML diagnostic snapshots with status `DEGENERATE_IDENTICAL_ACTIONS`. Execute one physical replay path per snapshot:8 STATIC main steps for the first hold block, then8 archived common-continuation steps/actions; score the sequential states after1,8,16 updates. Reuse each state/risk as the two logical candidate/STATIC paths **only** after proof that initial full-state/RNG, compared action IDs, row plans and continuation action sequence are identical. There are32 physical logical-batch updates in total, not64; H1 is an intermediate state, not an additional independent one-step run. Save raw risk arrays/row IDs and shared state/optimizer/RNG digests at all three checkpoints plus a logical-reuse receipt. A final source-snapshot restore is required. The contrast is identically0 by design; report that these preregistered anchors cannot discriminate the reversal mechanism. No additional attribution claim follows from running the duplicate. P_update panels from the original events are available; if no original cal pair was opened for a STATIC winner, label cal unavailable/not-accessed instead of inventing a nonzero cal contrast. Optional already-authorized Q_explore scores remain explicitly retrospective and do not choose actions.

Score matched six-cell risks on the archived event risk panels and explicitly named Q_explore reference if already authorized; every label access is diagnostic, not method selection. Save per-horizon state/optimizer digests, row/action plan, exact comparisons and cost. This is a retrospective mechanism check, not an unseen confirmation or replacement for formal training. All branches restore the source snapshot at completion; no changes to P1 states. If required factual rows/actions are unavailable, record precise ARTIFACT_BLOCKED for the diagnostic rather than substitute random support.

## 11. Formal eligibility without relaxing the original rules

Let `G_cmp` be global support for a specified comparison:≥3/4 positive beneficial gains, both per-data means>0, at least one per-data mean≥.0005. Let `D_cmp(d)` be restricted-data support: both seed gains>0 and mean≥.0005 in that **same dataset**. Principal comparison sets are R6 versus R0–R5; L4 versus L2,L3,L0_FEATURE. All gains use predeclared validation-selected Q_explore metrics and sign conventions; no use of test for locking.

Global principal eligibility is conjunction of G_cmp over every required primary comparator. If that fails, restricted eligibility for dataset d is conjunction of D_cmp(d) over **all** required comparators. This is not `(G_L2 OR D_L2(KR)) AND (G_L3 OR D_L3(KR)) AND D_FEATURE(KR)`: mixing a globally supported comparator with a restricted-data comparator would let another dataset's favorable seed erase a failed restricted-data replication. Such mixing is not permitted by the approved global-versus-restricted rule.

Applying supplied booleans only: LPS global principal eligibility fails FEATURE. ML restricted eligibility fails FEATURE. KR restricted eligibility fails L2 because one KR development seed is negative. Thus **no principal LPS50 formal package is currently authorized**. Complete the six triggered P2 runs and report them; a positive new P2 variant cannot replace original L4 or manufacture its missing second development seed. A separate future candidate plan would be required for that. This no-start is a compute-allocation decision, not a claim that all LPS scientific effects are zero.

RHD KR meets restricted support against all R0–R5; ML does not. After the necessary KR PB_HISTORY24 result and external comparisons are complete, a KR-only principal RHD50 package is eligible to be locked: R6_PB_HB, BEST_SIMPLE, R5_PB_H1, R7_PB_HISTORY, STGU_DOMAIN_ADAM ×seeds4101–4104 =20 logical runs. PB_HISTORY result must be visible before lock and may limit the current-feedback claim; do not invent a new numeric history threshold after seeing its result. If the orchestrator concludes current-feedback attribution does not warrant principal reevaluation, state the fixed evidence/claim reason explicitly rather than silently drop that comparator. An independently triggered simple-history discovery package follows the prior clarification, not automatic replacement of a negative PB method.

Lock BEST_SIMPLE **per KR** among R0–R4 by two-seed mean validation-selected macro LL (not Q_explore or test), tie lower standalone deployment cost, then fixed method ID if still exactly tied. Resolve exact duplicate BASE/STATIC by identical-source/state/action/RNG/plan proof and record both logical roles if needed; do not score-match deduplicate. Freeze that same chosen method for all four formal seeds. This dataset-specific formal method lock is distinct from selecting a comparator for developmental mechanism analysis.

STGU formal must be a **fresh** seed4101/4102/4103/4104 model/optimizer/controller/main-plan at epoch0, same audited sequential-group algorithm and constants, full50. The historical3101 resume bridge hardcodes an old24 lifecycle and must not be reused unchanged or mislabelled a fresh formal run. A new scoped wrapper can import its audited STGU step object while owning fresh native initialization, full-state plan/optimizer snapshots,50 schedule, selection and cost. It must not import seed3101 optimizer/affinity/rewind/epochs from old checkpoints.

Formal RHD uses fresh common native10 warmup for each new seed. PB_HB/H1 fork their appropriate initialization; PB_HISTORY forks that seed's PB_HB first initialized main block; BEST_SIMPLE forks appropriate warmup and uses its exact known schedule. All50 outputs retain selected+epoch50, AUC/LL-best secondary aliases and independent predictor identities. No patience stop. Per50 ceiling PB89→245H, H1=245; FREEZE96; CUM/EMA92 C8 events (736 shadow steps including final research-only event). PB_HISTORY shares one8H initialization/first main block physically, then237H suffix, but independent deployment still245H.

Before any formal process: save immutable `FROZEN_CONFIRMATION_SPEC.json` with support/no-start reasons for both lines, comparator identities, new seed availability audit, method/config/source/data/checkpoint rules, endpoint/selection protocol, exact20-run manifest or explicit reduced/no-start rationale, and concrete1.35-reserved component budget. Independent reviewer explicitly approves freshSTGU and conditionalR7 formal lifecycle. New seeds on old Q_confirm/test remain exposed-benchmark locked reevaluation, not unseen-data confirmation.

## 12. STATE/C14 nonactivation and falsification criteria

STATE requires existing directly comparable Hblock evidence that history shortlist omissions have useful residual risk structure predictable strictly from a past prefix or another seed, with forecasts committed before reveal. An unmeasured action in a3-candidate event has no observed utility and cannot be assigned an oracle winner. Warmup H1 cannot be renamed Hblock labels. Existing full-C8 initialization alone is not enough independent temporal evidence. Do not pay for a new full-C8 oracle grid just to manufacture STATE eligibility. If compatible prior raw tables do not establish this condition, record NO_TRIGGER and perform no STATE implementation/run.

C14 requires already recorded six-cell responses showing opposing task effects within the same domain, repeatedly and reproducibly across stages, in a way the common domain-mass C8 action cannot express. A single opposing noisy event, different domains, or macro failure alone does not satisfy it. Document exact existing events/stages/seed consistency if a trigger is claimed. STATE has priority if both qualify; at most one extension group. With no supplied or independently documented trigger, both remain off.

The conditional hypotheses fail to gain their specific evidence if history-only matches/current PB, probability clipping matches block gating, same-owner meta matches cross-target meta, or fixed geometry matches learned formation. A nonzero α/φ gradient, moved center-independent gate, entropy or differing source outputs demonstrates execution only. Scientific conclusions, evidence strength, seed consistency, six-cell tradeoffs, domain size versus difficulty, common endpoints, costs and competing explanations remain the orchestrator's ANALYZE stage after reviewed runs complete.
