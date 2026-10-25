# DESIGN SPECIFICATION — MDMT-NEXT-v3.0

Status: DESIGN COMPLETE; no v3 algorithm implementation or training performed by the designer. Frozen before inspecting any v3 outcome. Authority: all four files in `research_plans/26-10-25_v2/` and `start.md` (GB18030). This specification makes their implementation requirements explicit; their full mathematical definitions remain normative. Date: 2026-10-25. Designer role: `designer`, configured model `gpt-6.1-sol`; the available tools do not expose the parent runtime model, so model equality has not been independently verified.

## 1. Research questions

1. RHD: does paying to measure current shared-update utility over the actual action holding horizon improve complete natural-proportion MDMT training beyond BASE, fixed preference, simple history and one-step measurement?
2. LPS: does cross-target readable risk shaping owner-only block updates improve complete training beyond uniform block formation, scalar meta reweighting, coarse/constant reads and a no-foreign conditional prediction correction?

These are independent lines. Do not combine RHD with LPS in this package. Mechanism execution, fixed-state utility, complete training quality, and quality versus cost are four different claims.

## 2. Testable hypotheses and competing explanations

| Hypothesis | Discriminating comparison | Counterexample / competing explanation |
|---|---|---|
| Current paid utility has incremental value | PB_HB against simple policies, then PB_HISTORY | Persistent fixed action or historical average is sufficient |
| Matching measurement and hold horizons matters | PB_HB against otherwise identical PB_H1 | H1 is equally good; gains come from candidate/acceptance changes |
| Target-driven within-source formation matters | L4 against L2 and L3 | Ordinary continuous reading or scalar sample reweighting explains gains |
| Local block reads matter | L2 against L1; L4 against L5, including same-bank cache comparison | Source-level or per-cell constant combinations suffice |
| Real source content adds useful information | L4 against L0_FEATURE; conditional CAL_PROB_CLIP | Extra local features/meta labels explain the effect |

No assumption is made about curriculum, momentum, difficult domains, or a small-domain advantage. The old negative results remain negative results; v3 changes do not retroactively rescue them.

## 3. Changed and fixed variables

Changed RHD variables: the decision rule, measured horizon (H1 versus holding block), history update rule, and a conditional history-only/current-feedback attribution. The native predictive graph and controlled parameter set remain identical across RHD arms.

Changed LPS variables: full owner block sum; detached native features for every bank path; current projected feature descriptor; owner-only gradient routing; bank-only differentiable Adam formation; read granularity/conditioning; scalar versus block meta routing. All new LPS arms share the new graph conventions. Old K1/K4 numbers are not pure formation ablations of this graph.

Fixed across paired arms:

- Real ML/KR inputs, original natural row proportions, original role/encoder identities. No synthetic data, resampling ratio or label-conditioned redraw.
- Three domains, two independent Bernoulli tasks. `cell=task*3+domain`. Fit BCE is `sum(row,task loss)/(2*B)`; macro risk is the unweighted average of six cell risks.
- Logical batch 40960; `drop_last=False`; one appropriate Adam step per logical batch. Microbatching only performs exact accumulation. Loss denominators refer to the logical batch, including its short last batch.
- Adam: lr .001, betas (.9,.999), eps 1e-8, **coupled** weight decay 1e-5. FP32, no AMP or TF32. None gradient skips a parameter; a zero Tensor is not None. No scheduler or quality early stopping.
- Development seeds 3101/3102, fixed 24 epochs. Eligible checkpoints start at epoch12. RHD selects minimum validation macro LL; LPS selects maximum validation macro AUC. Exact ties use earliest epoch. Retain common epoch24, AUC-best and LL-best; formal adds epoch50.
- RNG namespaces: native=seed; main plan=seed+11000; bank/receiver=seed+50000; support/panels=seed+61000; controller/meta=seed+71000; fixed projection=seed+81000. Substreams have deterministic recorded offsets; additional work must not consume the native/main stream.
- Same dataset/seed arms share main row plans, support/panel schedules and coverage plans when these semantics are common; different closed-loop states produce their own labels and are never allowed to borrow another trajectory's revealed feedback.

## 4. Existing modules and exact implementation boundary

Inspected actual files:

| Existing file | Safe reuse boundary |
|---|---|
| `experiments/mdmt_next_v2_20261025/models/fixed1_adapter.py` | `Fixed1Adapter.encode` defaults to the historical graph and returns `native_logits`, `expert_outputs [B,4,64]`, domain. Training must use `risk=False`; `risk=True` removes the historical gate detach. Verify against actual `fixed.MDMTRec.forward` before admission. |
| `.../models/ownership.py` | Module-reference shared/private partition includes task experts and supplies the two state-feature moment groups. Verify the manifest. |
| `.../utility/state.py` | `FullState` captures model/buffers/modes, optimizer, None/Tensor gradients, requires-grad flags, RNG and registered objects. Register all new mutable plans/controllers/optimizers; budget/access journals remain outside rollback. |
| `.../data/roles.py` and `experiments/mdmt_next_20261024/data_roles.py` | Verify original prepared arrays, fit-only vocabulary/scaler and row hashes. Existing sorted `P_structure`/`P_value` use the exact v3 subdivision seed and can be aliased to P_form/P_read **after equality verification**. |
| `.../actions/c8.py` | Coefficients and module-gradient ownership can be reused, but candidate filtering, STATIC-relative labels and PB pair access are new. |
| `.../baselines/{runner,stgu}.py` and prior baseline adapters | Retain audited method mechanisms; replace obsolete timeout and invocation lifecycle, retain full workflow state. |
| `.../evaluate.py` | Six-cell risk machinery is reusable; v3 deployment predictions and predictor-specific checkpoint identities are new. Compute AUC from logits or stable probabilities without accidental clipping ties. |

Do **not** copy old `train.py`, `SourceBank`, `measure` or discrete selectors and call them v3: old bank sends owner gradients into h and uses geometry-weighted owner means; old coverage selects row/task A51 actions; old measurement enumerates all C8 relative to BASE and exposes full cal arrays. The old trainer's `epoch>=5` midpoint condition does correctly represent coordinate4.5 and can retain that coordinate convention. The changed graph, policy and cal access semantics remain mandatory.

New owned package: `experiments/mdmt_next_v3_20261025/`. Suggested modules: `protocol.py`, `data.py`, `plans.py`, `native.py` (or verified imports), `rhd.py`, `lps.py`, `functional_adam.py`, `read_policies.py`, `train.py`, `baselines.py`, `semantics.py`, `cli.py`, `aggregate.py`. Names may differ; semantics may not.

Outputs: `artifacts/mdmt_next_v3/v3_0/`; reports: `research_reports/2026-10-25_v3/`. New code/outputs must not mutate old source archives, checkpoints, reports or prepared arrays. Large new state files may live on a separately declared local filesystem and be referenced by recorded paths/hashes. Do not push data, row predictions or checkpoints; the user explicitly authorized pushing the completed report to a new time-named branch.

### 4.1 Common/core/LPS implementation interface

Use a common read-only data wrapper with `Data(dataset, campaign, log=None)`, attributes `features`, `x`, `domain`, `roles`, `pools[(role,domain)]`, `hash`, and `batch(ids,role,device,purpose,authorization=None)` returning x[B,F], y[B,2], domain[B], row_id[B], role. Alias `P_form`/`P_read` without changing old identities. All writes in the wrapper are new manifests/access logs only.

`Plans(data,seed)` exposes `main(epoch)->list[np.ndarray]`, independent `support(event_id,H)->list[np.ndarray]` (or equivalent `.at()` returning support plus panels), `panels(event_id)->dict[P_update/P_cal,ids]`, `coverage(epoch,position)->bool`, and `meta_rows(refresh_epoch,meta_step)->dict[fit/P_form,source,ids]`. Derive support/panel global event ID as `int(2*(coordinate-4.5))`: PB initial10.0 is11; next10.5 is12. This matches row plans across H1/HB where the first support is a common prefix without mixing their utility estimands. Controller-local indices still begin0 for EMA aging. Declare all row substream offsets in protocol manifest before use.

Common helpers: atomic JSON writer, append-only JSONL writer, source manifest/hash, and `CostLedger` with named component/seconds/event/epoch/method fields. `ExperimentIO(cfg,campaign)` owns only that run's output/state directory, config/data/source identities, registry updates, epoch records, access/cost journal, atomic checkpoint writes and checkpoint aliases. Native/RHD and LPS may each implement an explicit Trainer lifecycle consuming these common objects; neither imports another family's mutable training objects. LPS provides `LPSModel`/equivalent prediction/formation state, deployment predictor selection and complete `state_dict` including φ/read optimizer and normalization. Agree exact argument spelling directly between coders before edits; no semantic requirement depends on names.

The prepared checkpoint symlink is `artifacts/mdmt_next_v3/v3_0/checkpoints -> /home_new/xzf_mdmt_next_v3_2026-10-25_15-56-04`; preserve it. Python runtime is `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`, torch2.9.0+cu126, subject to runtime receipt. Publication branch is orchestrator-owned `2026-10-25_15-56-04`.

## 5. Data, role and evaluation contract

Prepared data expected at `/home/xzf/data/mdmt_next_20261024_strict_v1/{movielens,kuairand}`. Verify rather than assume:

| Data | fit d0 | fit d1 | fit d2 | fit N | main steps/epoch |
|---|---:|---:|---:|---:|---:|
| ML | 124759 | 254879 | 260495 | 640133 | 16 |
| KR | 138225 | 1395118 | 71561 | 1604904 | 40 |

Reuse saved disjoint roles first. Existing role arrays are source-row sorted. P_form/P_read are the first/second halves after per-domain `PCG64(20261025+60000+1009*d)` permutation of sorted P_label. Independently verify their old alias arrays, then keep aliases in the manifest without changing their row identities. P_update is RHD feedback; P_form is LPS meta query; P_read fits reading policies; P_cal selects a pair or a complete policy state. All are meta training/selection data. Q_explore is development analysis. Old Q_confirm/test are exposed benchmarks, never described as unseen confirmation.

If role files cannot be recovered but raw/preprocessed input identity is verified, reconstruct exactly file03 §2.1 and declare new artifacts. Never reconstruct from exposed test to manufacture an apparently unseen set. Verify raw files, x/y hashes, PREPROCESSOR fit hash, encoder/scaler hashes, all row partitions and task labels.

Development comparisons use selected checkpoint predictions on Q_explore, while all checkpoint/method selection uses validation only. Save validation each epoch; selected, epoch24 and optional AUC/LL-best score tables on Q_explore remain distinct. If exposed test/legacy-confirm are evaluated, freeze the relevant method/selection first and label the role explicitly; do not use them for P2/parameter selection. Formal new-seed evaluation is also an exposed-benchmark reevaluation.

Record six AUC/LL cells, n/positive rate, macro AUC/LL, per-task pooled AUC and natural pooled LL separately. Preserve an undefined single-class AUC consistently rather than dropping inconvenient cells. Report per-task and per-domain differences, worst-cell identity and delta, seed values/differences, mean/sample SD/median/direction count; seed is the replicate, not cell/epoch/query pair. With two development seeds, intervals are descriptive and unstable. Formal paired t intervals have n=4 and fixed-data limitations.

## 6. RHD algorithms, schedules and transactions

### 6.1 Native ownership and C8

Shared controlled parameters: embedding; STAR shared_weight/shared_bias; star_mlp; skip_conn; shared experts; cross-domain task experts. Private: all remaining native parameters (including domain slot parameters, domain experts, gate/tower parameters). Enumerate by live module objects, with unique complete coverage.

For each logical batch, `p_d=n_d/B`, `L_d=sum(loss rows d,tasks)/(2*n_d)`. Shared receives `grad(sum(alpha_d*L_d))`; private receives `grad(sum(p_d*L_d))`. C8 order is `BASE`, `SHARED_EQUAL`, then `TRANSFER_i_TO_j` in lexicographic i,j with i!=j. BASE alpha=p; EQUAL normalizes over present domains; transfer subtracts/adds `.25*min(p_i,p_j)`. Detach alpha. Deduplicate/mark illegal actions for missing domains; a fixed illegal action maps to BASE. Names and hashes define identity, not unchecked integer IDs.

STATIC is frozen ML TRANSFER_1_TO_2, KR BASE; apply it only after epoch10. Do not re-search it for a seed. One real coupled-Adam update follows each mixed shared/private gradient; do not add separate domain Adam displacements.

### 6.2 Common native warmup and simple rules

All RHD arms share native BASE epochs1–10, if state/RNG identity is proven. Record full C8 H1 feedback at fractional epoch4.5,5.0,...,10.0 (12 events). These probes restore the exact main state. BASE/STATIC do not require probes for independent deployment, although a shared research warmup can generate diagnostic events. FREEZE uses the warmup's P_update mean utility per action to select once and holds it to the end.

For CUM/EMA, full C8 H1 events continue at10.5,11.0,...,24.0 (40 total); formal ends50.0 (92 total). Each event: (1) commit prediction using only prior revealed events; (2) measure branches; (3) restore; (4) observe exactly once; (5) choose the **immediately following** main half-epoch. No extra block delay. At10.0, reveal the twelfth warmup event before choosing the first controlled block. Last24.0/50.0 event is research-only and may be excluded from independent deployment cost. CUM averages all observed H1 events; EMA uses last32 global events with half-life8; absent actions are missing, never zero. Choose highest STATIC-relative macro utility, STATIC priority on exact ties then action ID. Relative-to-BASE old utilities can be converted by subtracting the same event's STATIC utility before averaging.

### 6.3 PB_HB and PB_H1

Divide each epoch11–24 into its two predeclared logical-batch blocks; for current data H_hold=8/20. Derive actual lengths from the main plan, including short last logical batch; do not hardcode inferred lengths. PB_HB uses H_probe=H_hold independent fit support batches; PB_H1 uses1; both hold the action for H_hold main batches.

At epoch10.0 PB has its own **single** full C8 probe in its target estimand. It directly decides block10.0→10.5; do not duplicate this initial event and do not reuse H1 warmup as Hblock history. Afterward shortlist at most3 legal actions from prior events: STATIC; EMA-best non-STATIC; least-recently measured remaining action. Missing histories are not filled with0. Fixed action ID breaks history/coverage ties. Never force main-model exploration; exploration is in shadow branches. Keep estimand/horizon identity in each history; do not pool H1/H8/H20 unlabelled.

For each shortlisted action a and cell c, `U_update[a,c]=R_STATIC[c]-R_a[c]` on this event's P_update. Choose max six-cell mean, STATIC first on ties. Only after selecting this candidate may policy access P_cal **for that candidate/STATIC pair**. It cannot search all candidates on P_cal.

P_update and P_cal panels independently take1024 rows/domain without replacement per event, or all if insufficient; predeclare IDs, no label-based redraw. With per-row pair `D_i=(loss_STATIC_task0-loss_a_task0 + loss_STATIC_task1-loss_a_task1)/2`, accept only if

`Delta=(mean_d0 D + mean_d1 D + mean_d2 D)/3 > SE`,

`SE^2=sum_d[(1-n_d/N_d)*sample_var(D_d)/n_d]/9`.

N_d is that role's complete domain pool. Full-pool contribution is0. If n=1<N, deterministically take the next planned row to reach2, irrespective of labels. STATIC candidate implies fallback without unnecessary cal opening. No six separate veto gates; log all cell tradeoffs. One-SE is an empirical rule, not a coverage or long-run safety guarantee.

### 6.4 Full-state branch invariant

S contains model parameters and buffers, modes, every Adam group/m/v/per-parameter step, None/Tensor gradients, requires-grad flags, CPU/CUDA/Python/NumPy RNG, main/support plan cursors, mutable native attributes, controller/history and caches. Every action starts from identical S and identical independent support/RNG. For H>1 recompute all forward/backward steps using updated parameters/moments; private **rules** match, but private numerical gradients need not match as trajectories diverge.

Restore S after every branch and finally before committing the chosen controller history/action once. Main loader must be untouched by support. Never submit a probe's trained model as the real main state. Persist measured labels/cost/access outside transaction rollback. Budget/accounting is not restored to pretend probes were free. A crash-resumable checkpoint identifies completed main position and whether an event was measured, revealed and committed, preventing repeat observation.

At24 epochs PB has28 blocks, branch updates `8H+27*3H=89H` (ML712/KR1780), H1=89. At50 epochs:80 blocks,245H (ML1960/KR4900), H1=245. CUM/EMA total C8 updates320/736; independent deployment312/728. FREEZE96. Actual missing-action masks can reduce these ceilings and must be logged.

## 7. LPS graph and exact gradients

### 7.1 Features, capacities and initialization

Native `encode(x[B,F])` supplies logits `[B,2]`, experts `[B,4,64]`. `h=concat(LN_0(expert_e),e=1..4)` is256, expert-LN eps1e-5. All bank paths use detached h. `a=LN_0(R*h_detached)` is32, R `[32,256]` independent Gaussian row-normalized, projection-LN eps1e-6. `u=[a,detach(native_logits0),detach(native_logits1)]` is34, **not** owner-corrected z0 and not foreign geometry keys.

Three sources × four no-bias blocks `D:256→4`, GELU, `U:4→64`;15360 parameters. D Xavier, U Xavier times .01. Reader `v[target,task,source]`64, same reader for all four blocks of a source;1152 parameters, independent Xavier Linear(64,1,no bias). Owner forward is **sum** all four blocks (no rho average), so exactly the concatenated rank16 bottleneck. Never initialize D/U both zero.

Six heads `u34→16 ReLU→8 sigmoid` (4176 parameters), outputs two foreign sources ×4, no across-block softmax. Hidden Xavier/bias0; final weights N(0,std=.001), bias logit(.1). First refresh normalizes u using epoch10 P_read per-target mean/std, same domain normalization for both tasks; std floor1e-3 then frozen, clip±10. L0 heads use their own same-role normalizer on their independent trajectory. Record clipping/saturation.

Allocator per source `a32→16 tanh→8` (664 each/1992 total), final layer0; reshape2 tasks×4. `rho=.9*softmax_k(b)+.025`, `w=4*rho`; sum_k w=4 each row/task, range[.1,3.7], initial1. Remaining MLP hidden initialization Xavier/bias0 with dedicated streams.

### 7.2 Main fit permission matrix

`z0_tq=native_z_tq + v_tq,t^T sum_k e_tk`.

Owner values use task-specific backward routing:

`e_real=e.detach()+w.detach()*(e-e.detach())`.

Forward is e exactly; only bank gradient is multiplied by w. Reader/native gradients are unweighted. Foreign `e` is detached before trainable target reader; read coefficients are detached. Main fit does not update φ or read heads. All bank inputs h detach; native learns through its native logit. No per-block aux loss (aux=0). Do not multiply an already routed bank gradient by w again.

Main native/bank/receiver optimizer is Adam(.001, coupled wd1e-5), one logical step. Keep φ/read parameters outside main optimizer, with their own optimizers. Distinct source/block parameter objects preserve per-parameter Adam state and None semantics. An entirely inactive foreign reader must have grad=None (not zero gradients that decay or advance momentum). A source absent from a logical batch must not receive an update just because it was evaluated on foreign rows. Preserve the verified historical native graph rather than silently changing its zero/None behavior.

### 7.3 Functional Adam formation

Each meta step uses preplanned fit512 rows/source and P_form512 rows/target (all if less), labels never control row selection. Compute current native features in eval mode/no-grad; restore main mode afterward. Freeze native, real bank starting state, real reader, and deployed gate coefficients. Each virtual source support contributes natural source mass `p_s=N_fit,s/N_fit` times its mean two-task BCE.

Use differentiable routing `e_meta=e.detach()+w*(e-e.detach())`; do not detach w. With `create_graph=True`, get the bank gradient defined by this routed graph once. It is `p_s/(2*n_s)` times task/row derivative, **not** a second application of w. The virtual support predicts under the current deployed read policy, without inserting a coverage mixture.

Functional bank Adam uses current real theta/m/v/per-parameter step as detached constants, and g with its φ graph intact:

`gbar=g+wd*theta`, `m1=.9*m+.1*gbar`, `v1=.999*v+.001*gbar**2`, `t1=t+1`,

`theta1=theta-.001*(m1/(1-.9**t1))/(sqrt(v1/(1-.999**t1))+1e-8)`.

Uninitialized parameter state means m=v=0,t=0; None gradients skip that parameter. Do not reset existing moments, use AdamW, detach g/m1/v1 or replace with SGD. No real optimizer step is taken on bank/native/reader during meta.

For each target query keep its own real-current z0 fixed; replace only foreign content by virtual theta1. Reader values are frozen but the operation must preserve derivative through virtual foreign e. Compute macro BCE in two backgrounds **separately**, then `J=.75*J_deployed+.25*J_all_foreign_coeff_.5`. Do not average logits before BCE. Only φ receives/commits hypergradients, optimizer Adam .001/wd0. Discard virtual theta/moments. Log uniform w1 virtual reference from identical state/support; this is FORMATION_ADAM1, not full-model UPDATE_H1. Meta target feedback indirectly shapes owner formation; do not claim complete absence of target influence.

L3 allocator per source `32→19 tanh→2` (667/source), final0, `v=softplus(b)+.001`, `zeta=v/mean_same_source_same_task(v)`. Same ζ for all four blocks. Real denominator spans the **entire logical batch** before any microbatch backward; virtual denominator spans full same-source512 support and retains its φ derivative. Exact accumulation may precompute detached real ζ on the whole logical feature batch. Do not normalize each microbatch. L3 uses identical query/support plans, functional-Adam formula, meta-step budget and outer backgrounds as L4.

### 7.4 Read fitting, schedule and six arms

Read cache contains current detached u,z0,eight scalar residuals/labels/domain/row IDs plus native/bank/reader/projection/normalizer hashes. Recompute each refresh; stale snapshots are not interchangeable. Each head independently shuffles its target P_read and minimizes actual `BCE(z0+sum_j g_j*r_j)` with batch512, Adam .001/wd0. A pass means **all six heads each traverse their domain pool once**. No discrete51 expected-action loss.

First fitting10 passes; later refresh2 passes. Candidates include the untouched pre-refresh complete six-head parameters **and optimizer state**, plus each full pass. P_cal macro LL chooses one complete state, exact ties earliest/old. Never choose each cell's best pass and splice. Restore the selected optimizer with parameters.

Epoch schedule:

1. Epoch1–5: full owner, w1, foreign OFF, aux0.
2. Epoch6–10: L1–L5 use fixed .5 for all foreign blocks on predeclared10% **whole logical batches**, OFF on others. L0 stays OFF throughout. The selected whole-batch plan is common to L1–L5 and fixed without labels.
3. Epoch10 end: fork reading-family full state before any arm-specific formation/read fit. L3/L4/L5 execute16 meta steps; L1/L2 do not invent dummy updates. Fit each arm's read policy, then validate and checkpoint epoch10 after required refresh.
4. Epoch11 onward: same10% whole-batch coverage; other90% deploy that arm's read coefficients. Every even epoch≥12 first8 meta steps if applicable, then read refresh2 passes, then validation/save. Odd epochs validate/save too. Endpoints24/50 are after refresh.
5. L0 follows its independent all-OFF trajectory, with MONO/FEATURE heads fitted epoch10 and every even epoch afterward; these heads never feed back into the main trajectory.

| Arm | Formation | Deployed read |
|---|---|---|
| L0_OWN | w1 | OFF all epochs |
| L1_FULL_COARSE | w1 | six34→16→2 heads; same scalar on each source's4 blocks (594 params/head) |
| L2_BLOCK_UNIFORM | w1 | standard six8-output gates |
| L3_SCALAR_META | scalar ζ | same standard gates |
| L4_BLOCK_META | block w | same standard gates; principal LPS-BF |
| L5_BLOCK_CONSTANT | block w, same meta schedule |48 constants,8/cell in[0,1] |

L5 each cell solves its convex P_read BCE in FP64 L-BFGS-B, bounds[0,1], maxiter100, gtol1e-8. Candidate **whole policies**: unchanged old coefficients; optimized from old; optimized from all .1. First old is .1. P_cal chooses one whole policy, ties old. Report solver status; do not weaken this control to a few sigmoid steps.

L0_MONO: `z'=softplus(a)*z0+b`, initial slope1,b0, each cell. L0_FEATURE: six34→19 ReLU→1 residuals (685 params/head), final0; `z'=z0+f(u)`. Both fit entire P_label, same Adam/read pass/whole-policy P_cal rules; choose **independent** validation AUC checkpoints from their real predictions. MONO preserves each cell's ordering. Head fit/selection/inference/access costs are charged. FEATURE contains no foreign output or geometry key.

At24 epochs meta72 steps, read24 full passes; at50 meta176/read50. These apply to each meta arm independently after forks. L4 added training parameters22680 (bank15360+reader1152+gate4176+allocator1992); allocator not needed at inference. Projection has8192 fixed values. Inference computes all eight foreign blocks, reused across tasks; no claimed sparse speedup.

## 8. Required matrix and bounded conditional branches

P0 precedes P1; DESIGN→IMPLEMENT→independent REVIEW→RUN→ANALYZE gates cannot be bypassed. Coder does not analyze outcomes; runner records raw results only. An independent reviewer must inspect this specification, actual code and coder report, returning PASS or PASS WITH FIXES with blockers corrected before formal training.

P1: exactly7 RHD×2 data×2 seeds=28 and6 LPS×2×2=24 logical full24 trajectories. Share exact warmups, not post-divergence state. Register every logical arm even if execution is deduplicated. Old BASE/STATIC can only be reused with proof of code/data/main-plan/evaluation/optimizer/initial state identity; score equality is insufficient. KR BASE/STATIC may deduplicate if all future state/action/RNG/plan identity is proven; count both logical entries and one actual execution.

After cores, finish4 old aux0 paths (ML/KR seed3101 K1/K4) to24 from verified epoch5 pre-aux state or verified existing resume. Keep original discrete graph/gate/owner/selection identities. Reuse aux.1 old complete results only as the corresponding old factorial. If strict resumes cannot be verified mark old factorial ARTIFACT_BLOCKED; do not manufacture matched history. A newly rebuilt factorial occupies the same four slots and is explicitly a new version. A true ML K4 aux0 resume from epoch10 already contains its refreshed actor: skip the old trainer's unconditional begin11 initialization refresh, while retaining legitimate initial fork refresh and normal epoch12+ refreshes. Import prefix metrics and existing primary/AUC/LL checkpoint aliases as well as last; otherwise an unchanged old best could disappear from the resumed comparison. Copy across filesystems, use hardlinks only within a filesystem, and never alter old aliases. Resume AutoIFS's actual nested workflow optimizers/rewind/selector state, not its unused empty outer optimizer or a newly constructed workflow. Preserve remaining method/source/protocol identities.

Finish all6 external development adapters (three methods×two data×3101) to24; no obsolete native-equivalent deadline. Retain pinned author/adaptation receipts: STEM `769e2af0d0d1a0be9f58b475b95e05f502a20df5`, AutoIFS `65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e`, STGU `e5bc4565a4a1d33a8e9d6d8a2f13a63a72e7e3d3`. Retain STEM borrowed-expert detach. AutoIFS search10, rewind to **start of search epoch10** (epoch9 complete), frozen selector/selector embedding reuse to total24, temperature1→100, sparse .001/task, selector Adam1e-5/wd0; all search/rewind work counted. STGU performs actual sequential groups with real Adam, loss ratios including active private, natural domain masses, beta.001, shuffled group order, mutually positive-affinity partition among five partitions; inactive private grad=None. Do not replace STGU by a C8 EMA.

### 8.1 Frozen development support function

For a **specified comparator**, define beneficial gain=`reference_LL-candidate_LL` for RHD and `candidate_AUC-reference_AUC` for LPS using validation-selected Q_explore values. `global_support` requires≥3 of4 gains strictly positive, each dataset's two-seed mean>0, and at least one dataset mean≥.0005. Single-dataset support requires both seed gains>0 and mean≥.0005; report the other dataset's negative/limited outcome. This is a compute-allocation heuristic, not significance.

Always compute comparisonwise support for every fixed primary comparator. A principal package-level claim requires support against **all** prescribed primary comparators, not merely the weakest. Record detailed booleans and reasons rather than a single flattering promotion flag. Preserve a complete comparator table and common endpoints even when promotion is false.

### 8.2 Conditional RHD queue (maximum4 full24)

- PB_HISTORY when PB_HB has complete-training support against the locked best simple R0–R4 (best simple locked per dataset by mean development validation selected LL, cost tie). Execute supported dataset(s)×3101, at most2.
- PB_HISTORY shares PB_HB full C8 epoch10 initialization and **the first main block10.0→10.5 under its accepted or fallback action**, even if that action is STATIC; never wait for the first non-STATIC acceptance. From10.5, choose from its shortlist by historical mean *before* seeing current feedback; STATIC historical mean fixed0; tie STATIC then ID. Then measure/restore/observe for future only; current pair cannot revise the fixed action. Same horizon/candidate budget/panel plans, but after divergence its own states and labels. Identical cost/access budget; it is a deliberate information ablation, unlike CUM/EMA timing.
- One extra extension at most: STATE if strict past-only/other-seed candidate-omission structure exists; otherwise C14 only if same-domain tasks repeatedly oppose across stages. At most supported data×3101=2. No support→no extension. STATE details file02 §2.8 are normative (24 fixed features, per-action25→1 ridge, recent32/half-life8, non-intercept L2=1, prefix residual labels and fixed normalization). C14 adds six within-domain task transfers of p_d/8 to baseline task masses p_d/2; private unchanged. Neither replaces R6 in formal main table.
- If probe and final LL direction conflict, only registered snapshots per data3101 epoch12/20: interpret these as the **first block starting in epoch12/20** (coordinates11.0/19.0, before that block's probe/main update), save full state/row plan/decision then. Compare then-selected PB candidate/STATIC on already-observed common main sequence atH1, H_hold, and H_hold+one common H_hold continuation. Four snapshots, no full action×epoch grid. Store this interpretation before running; it resolves the prose's unspecified first/second half without selecting a favorable snapshot afterward.

### 8.3 Conditional LPS queue (maximum6 full24)

If L4 AUC support against L2/L3 (and report FEATURE primary comparison independently), execute supported dataset3101 in order: CAL_PROB_CLIP; OWNER_META; GEOM. At most2 runs/group, six total. A failure against FEATURE limits the unique-source-content claim, even if same-architecture formation ablations justify these already-bounded attribution controls.

If AUC unsupported but LL gains against both L2/L3 satisfy the same support function, execute only PAIR version on LL-supported dataset3101, max2; not the three AUC groups. Keep BCE main rows. Positive PAIR requires a separately planned second development seed/control package before any future lock, and is not substituted into this version's formal principal method. If neither AUC nor LL trigger, no formation extensions. TARGET_DIRECT_WRITE/strong-reader/backbone/joint controller are outside this package.

CAL_PROB_CLIP preserves L4 formation/bank/reader/coverage. `u34→19 ReLU→1` outputs c with final0, fits P_read only, P_cal chooses by actual clipped deployed LL. For residuals r8, feasible correction `[sum min(r,0),sum max(r,0)]`; clip c to delta. Obtain unique minimum-L2 coefficients in[0,1] satisfying r·alpha=delta: use positive aligned q=max(sign(delta)*r,0), repeatedly solve alpha=mu*q with capped entries1, at most8 rounds. FP64; residual tolerance `1e-7+1e-6*sum(abs(r))`. delta0→all0. Main fit detaches alpha; formation deployed background fixes alpha from current real residuals, not recomputed virtual residuals. Record correction coverage/saturation/FEATURE-equivalent fraction.

OWNER_META changes only outer query to same-source owner risk using virtual owner, real foreign state. GEOM fits four K-means++ centers/source from epoch10 current a at most20000 fit rows, seed=seed+91000+1009s,n_init5,max_iter300,tol1e-4; freeze centers. Lock temperature to mean entropy .5log4 by60 bisections in log tau over median-positive-distance×[1e-4,1e4] (median fallback1; base floor1e-6), choose nearest endpoint if unreachable (smaller on ties); same responsibility floor and gate.

PAIR replaces only outer BCE with per-cell softplus of negative positive-negative logit gaps; fixed independent pair RNG chooses≤256 distinct ordered +/- pairs from the original512 query, all if fewer; missing class masks explicitly. Same separate .75/.25 backgrounds, no added raw labels or pseudo-replicates.

### 8.4 Formal lock and counts

Only after P1 and triggered necessary P2 complete, generate immutable `FROZEN_CONFIRMATION_SPEC.json` with exact hashes, methods, constants, comparisons, selected rule, cost model and support decisions. Four seeds4101–4104 fixed50. Verify not previously executed. Do not start formal on negative principal lines merely to fill an upper bound.

RHD principal package: PB_HB, BEST_SIMPLE (validation-selected per data), PB_H1, completed PB_HISTORY, STGU:≤40. Principal claim requires development support of R6 against the fixed simple references and H1; PB_HISTORY can falsify current-feedback necessity and must be reported without silently adding a new threshold. If only a simple historical rule is supported, explicitly label a simple-history discovery reevaluation comparing locked history, BASE, STATIC, STGU:≤32; no PB superiority claim. If only one dataset supports, run only that dataset's corresponding package.

LPS supported principal package: L0 (with independently scored MONO/FEATURE), L1,L2,L3,L4,L5, completed CAL_PROB_CLIP, and best complete AutoIFS/STEM adapter locked per dataset by mean applicable development validation selected AUC (pilot seed3101; cost tie):≤64. Do not choose an external opponent on Q_explore/test. Main L4 claim must survive L2/L3/FEATURE; partial support is reported without calling formal confirmation. OWNER_META/GEOM one-seed evidence is developmental and does not become four-seed mechanism evidence.

Formal Q_confirm/test are exposed data despite new model seeds. Label findings STRONG/MEDIUM/WEAK based on repeated effects and counterevidence, not the promotion heuristic or p-value alone. No new parameter search after lock.

## 9. P0 and minimum reviewer acceptance

P0 audit hashes source archives cbe015/f07322 as archive identities (not assumed Git commits); roles/encoder;48 old completed cores;4 aux;6 external pilots; full states; predictions; event/tree/cost caches. Separate independently recomputed, report-only, missing. Recompute selected/endpoint six-cell tables from actual predictions when possible. Do not infer missing raw rows from report numbers.

P0-R1 reconstruct raw→shrink→history gate→gamma→state acceptance→actual action where logs permit; distinguish forced-gamma proposal and actual deployment. P0-R2 strictly prefix-only STATIC/FREEZE/CUM/EMA on available H1/H5 feedback, separate horizons/pools. P0-L1 inspect tree candidate/support/value/shrink/error/cal stages only when tables exist; missing candidate tables mean unresolvable depth0 cause. At most three single-factor cached replacements (shrink, margin, shallow preference). P0-L2 same snapshot owner whole bank/single/pair/full response and target risk; P0-L3 MONO/CAL42/no-foreign-key correction/probability selection on matched caches. A single small cached receiver comparison only if owner-useful/target-unreadable evidence supports it. No diagnostic training grid.

Reviewer acceptance requires four compact semantic receipts, using small controlled examples and an actual batch/schema where necessary:

1. Native/role: adapter sigmoid matches fixed historical native forward for both actual schemas with new paths OFF; pre-sigmoid logits used directly; ownership complete/unique; raw/array/encoder/role hashes and P_form/P_read alias equality. No training `risk=True`.
2. RHD transaction: one real logical batch demonstrates None versus zero/Adam-state preservation, rollback digest equality; one H>1 branch shows per-step recomputation and untouched main plan. Check STATIC-relative sign; first PB event exactly once; CUM/EMA immediate-observe timing; pair SE averages tasks before variance and uses correct finite-pool N.
3. LPS capacity/permissions: block-sum forward and gradient at w1 equals concatenated rank16; foreign target fit has no source-bank direct gradient; owner bank gradient changes once by w, reader/native not multiplied; φ/read gate absent from main updates; L3 full-logical/source/task normalization. Parameters with no use do not spuriously advance their Adam state. Show whole-batch coverage differs correctly from per-row random actions.
4. Meta: small-bank functional Adam agrees with true cloned-bank Adam at same gradients/current m/v/step including nonzero decay and initialized state; one finite-difference check verifies φ hypergradient direction/magnitude through virtual bank. Confirm native/real bank/reader/moments unchanged after φ-only update; both outer backgrounds evaluated separately, virtual foreign graph not detached. CAL constraint check occurs only if its P2 activates.

Numerical tolerances must be recorded before checking: native probability max absolute error≤1e-6; FP32 gradient/update comparisons allclose atol1e-6,rtol1e-4 with actual max errors; finite difference uses FP64 small bank and a step sweep around1e-4 with relative error≤1e-2 for a non-negligible derivative, or absolute≤1e-7 for near-zero derivatives. Rank16 equivalence uses FP64 or FP32 tolerance reflecting reduction order. Failing tolerances require diagnosing the same semantics, not loosening after observing training performance.

Blocking issues are only those with a concrete path to changed training/prediction/comparison/conclusion. Coder report lists source changes, deviations, syntax/basic checks, receipt locations and unresolved blockers; no outcome analysis. Reviewer independently inspects code/report/spec. PASS WITH FIXES requires every blocking fix and affected receipt rerun before RUN. No unrelated test suite/refactor.

## 10. Budget and execution requirements

Before launching an entire new package measure first **scheduled, retained** work under declared hardware/load: first main stage, full probe, meta refresh, read refresh, evaluation/checkpoint/cache/IO. Preserve state/RNG and continue it to24; this is not a disposable quality pilot. Estimate each method/data from actual component counts; use maximum observed component when too few samples for P90, not fabricated quantiles.

`C_upper=startup+E*c_main+branch_steps*c_branch+N_risk*c_risk+N_meta*c_meta+N_read*c_read+eval+checkpoint/cache/IO`.

Freeze concrete seconds/hours for all remaining complete comparisons plus `1.35` scheduling margin and reporting/cache diagnostics; no invented resource limit and no old24-native-epoch timeout. The initial component measurement is allowed before freezing the full matrix budget only as a retained portion of its first planned trajectory. The inability to predict a component before running it is not permission to discard that trajectory after seeing quality.

GPU0 at most2 experiment processes. GPU1 at most1 process total; a known external vLLM process occupies it, so it is unavailable until independently verified free. Do not kill unrelated work. Track load/co-resident processes. Compare inference/components in a common declared short hardware/concurrency setting. Physical experiment wall time, sum of overlapping process wall, and GPU hours are distinct; do not equate them. No energy estimate without measurements.

Three ledgers: research actual (all executed work, shared warmup once); standalone deployment training (each method's required warmup/probe/meta/read/eval, research-only terminal probe separately excluded); development/search (STATIC selection, P2, external search/rewind). Log failed/restarted startup/cache work too. Cost-quality curves only use legal preregistered completed checkpoints.

Atomic epoch last includes all optimizers/plans/controllers/RNG/selected-state trackers and immutable config/data/source identity. Resume from original state; no silent seed/reset. Status RUNNING/COMPLETE/RESOURCE_BLOCKED/ARTIFACT_BLOCKED/NUMERICAL_FAILED and actual completed epoch. A true hard quota unable to accommodate an indivisible package allows RESOURCE_BLOCKED; no artificial timeout to manufacture it. Incomplete runs never enter complete-quality tables. No numerical hyperparameter retries until success.

## 11. Logging and deliverables

Minimum machine-readable outputs: protocol_manifest.json, run_registry.csv, per-run config.json/data_manifest.json/epoch_metrics.csv/test_metrics.csv/model_best.pth/optimizer_state.pth/stdout.log; rhd_events.jsonl and pair per-row differences; lps_refresh.jsonl; cost_components.csv; information_access.csv; method_adapter_receipt.json; selected_predictions locally. Save full epoch10, selected, AUC-best, LL-best, epoch24, formal50, plus atomic last. Aliases/hardlinks allowed with identities recorded. Retain RHD diagnostic epoch12/20 block snapshots on registered3101 runs only if needed for bounded diagnosis.

RHD event fields: state digest, event/estimand/Hprobe/Hhold, legal masks/action names, support and risk row hashes/IDs, pre-reveal predictions/candidate shortlist, all P_update candidate six-cell risks, unique cal-pair loss rows/mean/SE/FPC, reject reason, committed/held action, actual hold count, labels accessed and component cost. Seal unselected P_cal values from policy even if raw arrays are saved internally for reproducibility.

LPS refresh fields: bank/reader/allocator/gate/feature/normalization hashes, ordered support/query IDs, .75/.25 risk separately, uniform formation reference, real-bank no-submit digest, φ step/gradient, w/ζ support distributions by source/task/block including positive/negative owner counts; routed versus uniform bank gradient cosine/norm and Adam-displacement change on a registered sample; gate distributions/clipping/saturation, per-pass full-policy cal risk and selected optimizer identity, solver statuses, label access and costs. At selected/common endpoint record all-source/individual-block deletion responses and prespecified input/gate-quantile groups, not label-mined winning subgroups. Nonzero gate/entropy/output difference alone is not useful knowledge evidence.

Information ledger records role, purpose, parameter or selector affected, row-plan hash, total visits and distinct rows, domain/task. Count all support/query repeats; two tasks on one row are not independent samples. Predicted logits contain exact deployed policy identity and row IDs.

Reports required: ARTIFACT_AUDIT.md; SPEC_RESOLUTION.md (explicitly no material revision if none); IMPLEMENTATION_REVIEW.md; P0_MECHANISM_REVIEW.md; RHD_CLOSED_LOOP_REPORT.md; LPS_FORMATION_REPORT.md; ABLATION_AND_BASELINE_REPORT.md; COST_REPORT.md; FORMAL_LOCKED_REEVALUATION.md; FINAL_REPORT.md; OPEN_QUESTIONS.md. Generate tables from artifacts. Give exact complete/incomplete scope, seed-paired deltas, cell/task/domain/stage comparisons, limitations, protocol revisions, actual budget errors and negative outcomes. Label scientific evidence STRONG/MEDIUM/WEAK. Evaluate domain-size versus baseline difficulty separately and explicitly test whether a small-domain gain/large-domain loss tradeoff exists; three domains do not identify size causality.

## 12. Pre-result clarifications for SPEC_RESOLUTION

These are implementation choices/clarifications, not post-result algorithm changes. Coder/orchestrator must mirror them into protocol/spec-resolution before training:

1. The user explicitly requests a report push; this overrides file04's conditional no-push text. Source data/predictions/checkpoints remain local. Report publication alone does not imply external dataset upload.
2. File02§3.5 explicitly sets P_form=P_structure/P_read=P_value, and actual v2 files implement the exact specified per-domain sorted-row PCG subdivision. Verify byte-equality rather than assume a role name proves it. Reusing verified aliases preserves all label permissions.
3. A10% logical-batch coverage rate cannot always be an exact integer within16 batches. Freeze a label-independent whole-batch schedule over successive blocks of10 main logical batches from epoch6, exactly one selected position in each block via a dedicated seeded permutation; carry position across epochs, use common plan L1–L5. Truncated final ten-batch blocks have their actual realized rate logged. This satisfies predeclared whole-batch10% without replacing it by row-level coverage. Do not fit the schedule to outcomes.
4. For odd main-step counts, split the precomputed epoch batch list at floor(nsteps/2) into two nonempty blocks and use each actual length; current verified16/40 requires no change. Different horizons remain separately identified in history. No short-batch padding in main training.
5. PB utility uses STATIC as event reference; simple CUM/EMA averages may be computed from BASE-relative labels by same-event subtraction, which does not change actions. PB's first Hblock history is not warmup H1 relabelling.
6. Package promotion is comparisonwise, with all fixed primary comparisons reported; single-dataset promotion uses two positive seeds and mean≥.0005 against the required comparisons. This resolves the text's global heuristic and limited-scope exception without cherry-picking a weak opponent. Formal method lists remain bounded and negative lines stop; exact no-start reasons are mandatory.
7. Same-source ownership should be represented with distinct parameter objects to preserve per-parameter None/step semantics. Vectorized storage is allowed only if it demonstrably implements equivalent independent per-source/block update and decay/step masks.
8. The authoritative dates/branch names are provenance, not proof of actual upstream contents. Audit source archive/file hashes and remote baseline separately. Existing archives are not assumed to be Git commits. Any missing upstream fetch cannot justify replacing v3 by old v2.

No outcome-dependent threshold changes, broad extra ablations, new backbone, stronger reader, direct-target-write, RHD+LPS combination or infinite literature search is authorized by this design.
