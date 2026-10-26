# DESIGN SPECIFICATION — DFD-v4.0

Date: 2026-10-26. Stage: DESIGN COMPLETE; no new experiment result is claimed.

**Design verdict: PASS for implementation under the locked 04/05 contract.** Training admission additionally requires the independent implementation review and applicable P0 receipts. This document translates the already approved design into implementation requirements; it does not reopen hypotheses, constants, comparisons, seeds, or promotion thresholds.

## 1. Research question

Can predictable target-side differences in the risk of fixed source reads organize source-supervised functions that improve the complete selected predictor beyond strong direct probability, local capacity, uniform, geometry, shuffled responsibility, constant-read, label-pass, and pooled-source explanations?

Readability of a fixed source, conditional readability, usefulness of formation, and usefulness of the whole pipeline are separate claims. Positive weights, different parameters, nonzero gradients, or foreign deletion effects alone do not answer the question.

## 2. Hypotheses and falsification targets

- **H1:** Target demand is usable from prediction-time features. P_form OOF DEMAND must pass the exact comparison against CONST and LOGIT below. DIRECT matching DEMAND does not reject demand; it rejects estimator necessity.
- **H2:** Target demand can guide useful source function formation across support distributions. DFD must outperform matched UNIFORM/GEOM/SHUFFLE, with source extrapolation and fixed-decision content diagnostics reported.
- **H3:** The complete pipeline has practical incremental value. It must surpass every prespecified simple control while respecting AUC and cell protections. A successful local intervention cannot substitute for this comparison.

Competing explanations include full-h local prediction, extra nonlinear capacity, more target-label passes, source-label or confidence strata, generic geometry, pooled training, calibration, checkpoint selection, population support changes, and finite optimization. No assumption is made about curriculum, momentum, small domains, or M3oE on difficult domains.

## 3. Authority, source identity, and scope

Scientific authority: `research_plans/2026-10-26/04_核心算法与Codex实现规范.md` for mathematics/gradient/modules; `05_完整实验矩阵与研究决策.md` for matrix/statistics/budget/selection/promotion. All eight plan files, AGENT.md, and the GB18030 start.md were read. Latest start.md clause 11 makes delegation advisory and requires non-RUN agents to match the parent model; clause 12 requires concurrency when the same available GPU has both memory and power below 85% of its limits. These direct user instructions override the default single-heavy-process suggestion in 05, without authorizing use of devices owned by other workloads.

The parent reports current remote main `8b87d3fc2ac393faa34cc870bc3befee0e8693ca`; its new isolated checkout must independently record this before implementation. The original workspace contains source/data but has invalid Git metadata (missing HEAD/config); do not repair, reset, or reinterpret it as a valid checkout. Work from current main in an isolated implementation branch, import only necessary audited files, and do not merge an old experimental branch wholesale.

Historical evidence is pinned to `8a4b3b296ca5dc7349956fae9eca5e202aefe2c0`. Locally read REPORT_INDEX, FINAL_REPORT, IMPLEMENTATION_REVIEW, SPEC_RESOLUTION and relevant v3 source interfaces. Historical PASS does not admit new DFD code. Historical reports separate selected failure from late endpoint stabilization; those observations do not prove any new DFD claim.

Readable local interfaces inspected:

| Source | Finding / reuse boundary |
|---|---|
| `experiments/mdmt_next_20261024/data_roles.py` | Audited fit-only prepared schema/arrays, label-free integer roles, raw identities. Read/check or vendor necessary dependencies; do not inherit old purpose permissions without DFD changes. |
| `experiments/mdmt_next_v3_20261025/data.py` | P_form/P_read verified aliases of deterministic P_label halves. Old main row RNG is not the new SeedSequence contract. |
| `experiments/mdmt_next_v2_20261025/models/fixed1_adapter.py` | Mechanical pre-sigmoid logits and shared expert tensors; `risk=False` preserves gate detach. Direct fixed construction here does not implement the required original-then-copy canonical initialization. |
| `experiments/reproduction_20261021/protocol.py:make_model` | Original MDMTRec is constructed first, `_weight_*` removed from its state, then canonical backbone copied to fixed. Reuse this identity rule; never its lr=.01/Equal trainer. |
| `experiments/mdmt_next_v3_20261025/common.py` | Complete-state/cost ideas are useful; its logit LL/AUC metric semantics do not satisfy new probability-clipped evaluation LL and original-probability AUC exactly. |

Inspected file SHA256s: adapter `628e3966938cbc586612acb5fff534369f15142bc665e33741b67ee5c4d32596`; v3 data `d4b07a16a115082ce3521adb8d66687079623a4eb761e62a6afd77a5be5df2bb`; historical data_roles `f8cb866a711330af2caed6ed696a134c2e78dd964213a00e0c9152262d4ae363`. These are content hashes, not Git commits.

If local source identity is unresolved, extract `c5618e281785a4b7063ff9bf2a61cf38d4faf1be:experiments/data_generate/MDMTRec` and dependencies to `vendor/pinned_fixed/`, including original.py for canonical initialization. `controlled.py` from `f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e` is a reference, not a trainer to inherit. Local or pinned route is valid after parity; neither is automatically old-byte reproduction.

## 4. Variables changed and held constant

Changed: source-function training responsibility (DFD/UNIFORM/GEOM/SHUFFLE), local versus source versus pooled supervision, direct probability versus mixture readout, constant versus conditional reading, and the fixed U20/U60 label-pass comparison. No other degrees of freedom are searched.

Held constant where the comparison allows: true natural rows/proportions; fit-only encoder; roles; source/native/pilot prefix by own seed; architecture; canonical native initialization; paired formation perturbations and fit plans; logical batch/optimizer/precision; fixed passes; validation/P_cal selection; metrics; exposure status. LOCAL_CAP and POOL are explicitly stronger but not globally parameter/label-pass matched controls; their differences must remain visible.

### Data and roles

Use existing `/home/xzf/data/mdmt_next_20261024_strict_v1/{movielens,kuairand}` only after raw, array, encoder/scaler and role hash checks. Raw expected SHA256s are ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`, KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`.

Fit counts ML `[124759,254879,260495]` (640133), KR `[138225,1395118,71561]` (1604904). P_form/P_read/P_cal expected totals ML 24004/24006/16003, KR 60183/60184/40123. P_label is their historical parent, not another fitting pool to add.

Roles are disjoint at the atomic partition level: fit; P_form; P_read; P_cal; P_update; Q_explore; Q_legacy_confirm_exposed; validation; test_exposed. Alias/parent relations must not be mistaken for overlap. Keep original row IDs/order, outer 80/90 integer boundaries and per-domain PCG protocol exactly as 04§2.1. If regenerated identities differ, record PROTOCOL_MISMATCH rather than silently substituting.

fit trains native/pilot/source and fits input normalizers/geometry. P_form trains demand and common reader prefit. P_read trains final readouts. P_cal selects complete six-head policies. validation selects native only. Q_explore scores development final predictors. P_update is unused in DFD. Exposed test/legacy labels remain locked until final method/run freeze. Loading labels for integrity audit must never create a training dependency.

ML labels are rating>3 and rating>4; KR labels are actual is_click/long_view columns; do not infer a logical relation from task names. ML raw has original user_id/movie_id. The inspected KR processed raw header has no original user or item ID: if no trusted row mapping can recover them, use the explicitly allowed row-bootstrap fallback, mark user/item support MISSING, and state the insufficient independence assumption. Never invent identity from profile features or merge all UNK users into one real user. This allowed fallback does not block core training.

### Training and RNG

Native Sample loss is `sum_iq BCEWithLogits(z_iq,y_iq)/(2B)`. Logical B=40960, drop_last=False, expected steps ML16/KR40 per epoch. Microbatch4096 is allowed with each sum divided by the full current logical-batch denominator and exactly one Adam step. Short last B uses its actual size.

Native/pilot/formation use torch Adam lr=.001, betas(.9,.999), eps1e-8, coupled wd1e-5; eta/readers wd0, batch512. FP32 gradients, no AMP/TF32; risk conversion/mixing diagnostics, solver and summaries FP64. No resampling, scarce copying, noise, learned fusion, AutoML/DARTS, AdamW, target-writing source gradients, meta Adam or second order paths.

Development seeds 5101/5102; formal 6101/6102/6103/6104. Audit prior use for this method before any run; collision is recorded and not resolved by replacing the seed. Distinct datasets have independent streams. Use `SeedSequence([run_seed,dataset_id,namespace,source_id,task_id,component_id,pass_id])`, ML0/KR1, unused coordinates0. Namespaces: native-init10, main-plan20, pilot-init30, pilot-plan31, eta-init40, eta-plan41, formation-perturb50, formation-plan51, shuffle52, geometry-subset53, geometry-fit54, read-init60, read-plan61, bootstrap70. Freeze the component-ID mapping and per-head target/source ownership in PLAN before training; disjoint stage/fold IDs must prevent correlated accidental RNG reuse. Paired arms deliberately share only prescribed initialization/row streams, not trained optimizer state.

P_form folds are independent of model seed: per-domain sorted rows permuted by PCG64(20261026+70000+1009*d), position modulo2. Both tasks use the same row fold. Normalizers never see fold labels or heldout labels.

## 5. Mathematical and algorithmic specification

### A. Native packet, training, cache and pilots

Fixed native: three domains, two tasks, four shared experts, fcn_dims[512,256,256,64], exp_d=exp_t=bal_d=bal_t=1. Build original canonical backbone then copy; no `_weight_*` becomes trainable. Adapter input `[B,F]`, transpose once to native `[F,B]`; return actual logits `[B,2]`, actual shared expert tensor `[B,4,64]`, and `h=concat_e layer_norm(E_e,64,affine=False,eps=1e-5)` `[B,256]`. The STAR intermediate has the same width and is prohibited as an h substitute. Preserve gate `emb.detach()` on real training.

Train each own seed native from scratch 24epochs. Select validation macro LL minimum among completed epoch12–24, exact tie earliest, and save full selected plus epoch24. Freeze selected model and all mutable state/eval mode permanently. Cache h/z with native weight/extractor/encoder/row hashes; never relabel another checkpoint's cache.

Six independent pilot MLPs f_s,q:256→32ReLU→1, 8257 parameters each, total49542; hidden Xavier/bias0, output weight/bias0. Only rows d=s update that pilot. Loss `sum_iq BCE(z_iq+f_diq(h_i),y_iq)/(2B)`, 12 full fit passes. Freeze pilot afterward. LOCAL=own pilot residual probability, NATIVE=sigmoid(z). No foreign pilot gradient in source fit.

### B. Demand eta, cross-fitting, and centered responsibilities

For target t/task q and source s≠t, on actual-domain features compute p0=sigmoid(z_q+f_tq(h)) and pa=(p0+sigmoid(z_q+f_sq(h)))/2. Diagnostic a0/a1 are logits after clipping these probabilities to [1e-7,1−1e-7] in FP64. Actual observed difference is d=softplus(a0)−softplus(a1)−y*(a0−a1).

eta input v=[h256,z0,z1,f_0q,f_1q,f_2q], 261 dims. Per target/task head 261→32ReLU→1, total50502. Fit target-specific normalizer from target fit frozen v, std floor1e-3, clip normalized coordinates ±10, freeze for all folds/refit/extrapolation. Hidden Xavier/bias0, output weight0, output bias logit of fitting fold's positive rate clipped [1e-4,1−1e-4]. Train20 full passes, batch512, Adam .001/wd0 ordinary BCE.

`mu=softplus(a0)−softplus(a1)−eta(v)*(a0−a1)` has no y input. Train eta on fold0 and predict only fold1, and reverse. Save actual train-ID hash and OOF row IDs; never overwrite OOF with full refit predictions. Only after two-seed dataset P1 passes, initialize fresh and train all-P_form eta20; formal locked seeds always perform required folds/refit and are never filtered by their own P1 scores.

For each ordered (s,t,q), m is full P_form,t mean observed d. For fixed (s,q), concatenate the two target OOF arrays b=OOFmu−m with each target total weight1/2, compute population weighted SD around the pooled weighted mean, tau=max(.001,SD). Do not use a raw second moment as SD. m/tau are frozen and used only in formation, never P1 OOF actions.

On actual source fit row d=s, retain the input's actual domain s and its h_s/z_s. Evaluate the target normalizer/eta on v built from these features; a0 uses target pilot f_t(h_s), source term uses f_s(h_s), both add actual z_s. For sorted two targets k, set b_k=mu_s→tk,q−m_s→tk,q and `w_k=.1+1.8*softmax_k(b_k/tau_s,q)`. Cache once. Each w in[.1,1.9], sum_k w=2. Do not relabel row domain, use target labels for source rows, recalculate w as F moves, or normalize each function by its batch/global mean.

### C. Formation

Twelve independent F_s→t,q:256→32ReLU→1, 99084 total. Copy matching pilot parameters, then add independent std1e-3 Gaussian noise to output weight only, not bias. Each slot's perturbation is identical across DFD/UNIFORM/GEOM/SHUFFLE. Fresh optimizer states start empty, never inherit pilot moments.

`Lform=sum_i,q,k w_iqk*BCE(z_iq+F_di→tk,q(h_i),y_iq)/(4B)`, 12 full fit passes, fixed final library. Only actual source/task functions receive supervised gradients. native/pilot/eta/w remain frozen and detached. Zero with set_to_none=True; build losses only for present sources. Missing-source parameter grad=None preserves Adam step/m/v/decay. Source conditionally positive weighted BCE does not theoretically guarantee distinct Bayes functions; any finite-capacity functionality claim needs outcomes/controls.

### D. Ordinary reader and selection

For target/task, ordered candidates are local pilot plus BOTH intended-target functions from each foreign source, sorted(source,intended target), five probabilities. Never filter to only intended_target=t, and never read local source's formed functions in DFD. Input `[h256,z0,z1,p0..p4]`263→32ReLU→5softmax, 8613/head,51678 total. hidden Xavier/bias0, output weight0, bias log[.9,.025,.025,.025,.025]. Target fit input normalizer is computed for this fixed library and frozen (std floor.001, clip±10).

Train six heads independently on their own target rows, P_form20 full passes then P_read10, batch512/Adam.001wd0 with continuous optimizer state. All candidate inputs detach. Stable mixed BCE uses logsumexp(logalpha+logsigmoid(candidate logits)) and the analogous negative logits, without clipping mixture-training probabilities. Candidate states are complete six-head pass0 after prefit, complete read passes1–10 and exact LOCAL_ONLY OFF; never select a prefit epoch. P_cal macro LL chooses one complete group, exact ties OFF then pass0 then earliest. No per-cell pass stitching. OFF exactly equals LOCAL and still bears actual incurred formation/training cost.

## 6. Required comparisons / ablations

No additional ablation is authorized merely to obtain positive evidence. All required controls are the finite matrix below.

### P1 four OOF strategies, each train-fold only

1. CONST: best constant simplex of THREE probabilities (local and two fixed half-read candidates), FP64 solver below. No all-P_form fit. Prediction is convex blend, not sampled hard action.
2. LOGIT: 5→32ReLU→1 eta from `[z0,z1,f0q,f1q,f2q]`, corresponding target fit normalizer columns, fitting-fold positive-rate bias,20passes. Compute two mu. Action argmax{0,mu_s0,mu_s1}, ties OFF then lowest source ID. Prediction is selected fixed candidate.
3. DEMAND: full261 eta and same fixed hard action. No m, tau, observed heldout d, heldout calibration, or action fallback.
4. DIRECT: normalized261→32ReLU→3softmax, hidden Xavier, output weight0,bias log[.9,.05,.05],20passes, stable true mixture BCE on train fold. Predict ordinary soft mix of same three probabilities. No heldout/pass selection.

All four use identical train-fold unique labels, same heldout rows and fixed pilot/native. Report six-cell and macro OOF AUC/LL; P1 gate is on P_form OOF, not Q_explore. Save fold-specific η/coefficients, full row predictions/actions and loss.

### P2 ten outputs

NATIVE, LOCAL, H_DIRECT, ETA_DIRECT, UNIFORM, GEOM, SHUFFLE, CONST_UNIFORM, LOCAL_CAP, DFD. All launched outputs finish even if an early opponent is strong.

- H_DIRECT: six258→32→1 heads, total49926, target fit normalizer and eta initialization rules. P_form20→P_read10, complete groups pass0–10 P_cal choice, earliest tie; no OFF is added to these direct-head candidates unless specified for EXT_INFO.
- ETA_DIRECT: clone all-P_form refit eta AND its Adam, preserve original eta frozen; continue own target P_read10, choose complete pass0–10. Contains foreign pilot features, not a foreign-free method.
- UNIFORM: formation w=1, same slots/perturbations/plans/passes/reader.
- GEOM: per source label-free subset min(n,20000), Kmeans++K2,n_init5,max_iter300,tol1e-4, shared task centers. rho=.05+.9softmax(−dist²/taugeo), w=2rho. Base=median all nonzero subset-center dist², else1,floor1e-6; tau in base*[1e-4,1e4], log60 bisections toward entropy .5log2. Unreachable choose closest endpoint, tie smaller tau; no retries. Center order maps to sorted intended targets.
- SHUFFLE: source/task local pilot fit probability linear deciles, strata=(actual y_q,confidence bin); sort row IDs then one dedicated permutation assigns entire two-coordinate w vectors. Fixed all12passes, both coordinates same permutation. Preserve every stratum's exact joint weights and each row sum2. Record singleton/unchanged rows, no manufactured noise.
- CONST_UNIFORM: same uniform library; solve five-component constant simplex on unique P_form∪P_read per head; complete constant group vs exact OFF on P_cal, tie OFF.
- LOCAL_CAP: four extra local256→32→1 per t,q,24functions198168params, copy local pilot+output-weight perturbation, own fit labels only,12passes, each BCE coefficient1/(4B). Its total scalar loss equals twice two-task sample mean due four heads; native remains frozen. Same five-component reader. Total added deployed299388 vs DFD200304; do not claim global parameter match.

Convex solver for P1_CONST and CONST_UNIFORM: SLSQP FP64, [0,1] bounds,sum1,analytic gradient,maxiter500,ftol1e-12. Fixed component-prob clip[1e-15,1−1e-15] BEFORE mixing. Starts own-only,uniform,and own-heavy ([.9,.05,.05] or [.9,.025×4]). Select smallest TRAIN loss; exact ties lower foreign mass then lexicographic coefficients. Require sum error≤1e-8,bounds error≤1e-10 and FWgap `g·alpha−min_jg_j`≤1e-7. Bad near-optimality is an implementation failure, not a scientific negative control result. No mixture clipping flat zones, weak few-step optimizer, or heldout restart selection.

### P3 two additional controls and fixed-decision interventions

U_BUDGET uses UNIFORM library but P_form60→P_read10. Branch from saved UNIFORM prefit20 state, never from its P_read-fitted state; keep both U20/U60 outputs. It matches training label-pass only, not FLOPs/walltime.

POOL: task-pooled g_q12passes, loss sum_iqBCE(z+g,y)/(2B). Initialize twelve slots from g_q+same perturbation, then every slot sees all natural fit rows12passes, no source mask/no demand, loss sum_iq,6slots BCE/(12B). Retain original source own pilot component0, same readout. Source slots now only positions, no source ownership claim; extra coverage/cost is explicit.

Fixed final DFD and matched UNIFORM content diagnostics: replace each foreign source's two probabilities by p0 (2), replace individual DFD functions by uniform correspondence (4), swap same-source two intent probabilities (2). At most8/cell. Freeze original alpha even though the reader normally consumes probabilities; do NOT reroute or retrain. Group by P1 frozen OOF mu quartiles and predefined support. These identify current content contribution, not retrained necessity.

### P4 fixed author baselines

Only P3-passing datasets run four scratch seeds,12internal outputs plus STEM_DOMAIN_TASK/AutoIFS_MDMT raw+EXT_INFO. No seed-level quality filter. Lock exact source adaptation, plans,12outputs, four external predictor outputs and evaluation roles before any formal/exposed access. Full external contracts remain 05§6.2; do not substitute defaults:

- STEM author pin769e2af0d0d1a0be9f58b475b95e05f502a20df5; sixhead3*q+d,embedding16/block,7blocks112/field,shared8,specific1/head,expert[512,512,512],gate/tower[128,64],one layer,noBN/dropout,foreign stop-gradient. Dense field Linear1→112 no bias then split.
- AutoIFS pin65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e;embedding16,MLP[1024,512,256],rank2,3domains/2tasks,four flow types,noBN/dropout/cross additions. Dense Linear1→16 no bias. Search10, temperature100**((e−1)/9), naturalBCE+.001mean(sigmoid(temp gate0))+.001mean(sigmoid(temp gate1)); selectorAdam1e-5wd0,predictorAdam.001coupledwd1e-5. Save predictor after9 before10, finish search10, rewind predictor9, freeze search10 selector/embedding, fresh predictorAdam, reuse logical10–24(15passes), total physical25/effectivepredictor24. Select only reuse12–24 by validationLL. No author training-time test or AUC early-stop.
- EXT_INFO: frozen selected external, target fit normalized `[r_tq,z0,z1]`→32→1, P_form20/read10, P_cal complete pass0–10 plus exact raw, ties raw then earlier. STEM r64=tower hidden before finalLinear;AutoIFS r776=concat(d_dnn256,d_share256,d_lora256,gate0hard4,gate1hard4). Tap parity≤1e-6; no activation/forward alteration. Additional explicit net/embed regularizers0; unified coupledwd only.

## 7. Exact stage decisions and budget

### Promotion gates

P1 per dataset only after both5101/5102: each G=LL_CONST−LL_DEMAND>0,meanG≥.0001;each seed≥4/6cell LL gains≥0;mean(LL_LOGIT−LL_DEMAND)≥0;no missing main metrics/implementation blocker. P1 fail→P2/P3/P4NO_START for this dataset. Another dataset decides independently. DIRECT has no gate veto.

P2: for each of all other9 outputs c, eachseed G_c=LL_c−LL_DFD>0,meanG≥.0005;mean(AUC_DFD−AUC_c)≥−.0005;max_cell mean_seed(LL_DFD−LL_c)≤.003. Nondegenerate means at least one source/task fit w SD>1e-6 AND at least one cell final predictor−LOCAL max absolute difference>1e-6; full receipts required. Do not average per-seed worst cells instead of taking worst mean cell.

P3: both U_BUDGET and POOL must separately pass the same seedLL/meanLL/AUC/cell conditions. P3 fail→P4NO_START. Interventions remain descriptive and do not override failed gates or cause added searches. P4 complete all locked seeds/output even if individual P1/P2-like checks fail.

Formal support rule: versus every other11 internal outputs,4/4 LL gain>0,mean≥.0005,t95 lower bound>0,mean AUC/cell protections; complete mechanism/cost. External superiority is separate and requires corresponding support versus both raw and INFO of both methods. Label support LOCKED_BENCHMARK_SUPPORT / LIMITED_MECHANISM_SUPPORT / SMALL_OR_UNCERTAIN / VERSION_NOT_SUPPORTED as 05; unresolved artifacts/implementation are not scientific negatives.

### Finite matrix and executable cost admission

P1 four native24 prefixes. P2 twenty final outputs per P1-passing dataset, zero new native; P3 four extra outputs per P2-passing dataset. P4 per P3-passing dataset four native24, eight author baseline trajectories and48internal+16external predictor outputs. Maximum28large24epoch trajectories,128formal logical predictors, not128independent seeds. AutoIFS physical search overhead counted.

Use first REQUIRED run's nativeepoch1,pilotpass1,eta firstfoldpass1,formationpass1,readerpass1 as retained complete-state measurement boundaries; no extra full timing trial. Measured remaining component max rates + cache/eval/I/O times, times1.35 reserve, admit full stage once disk/device support it. No invented hard time budget or new approval request. Estimate storage for caches/full states before stage; FP32 hfit≈.66GB ML/1.64GB KR plus all other roles/components. Preserve required shared prefix once in research actual cost, charge independently necessary prefix in standalone training cost, keep development/search cost distinct. Concurrent wall sums are not GPU hours.

## 8. Required modules and implementation boundaries

New root `experiments/mdmt_dfd_20261026/`, original workspace artifact root `artifacts/mdmt_dfd_v4/v4_0/`, report root `research_reports/2026-10-26_v4/`.

| File | Required responsibility |
|---|---|
| data.py | RoleBundle, immutable array/raw/encoder/row identity, atomic role permissions, fit supports, native natural batches |
| native_adapter.py | canonical initialization, logits/shared/h packet, train graph parity |
| functions.py | independent pilot/function owners, all source/local/pooled formation losses |
| demand.py | normalized eta folds/refit, FP64d/mu/m/tau, frozen responsibility cache |
| readers.py | stable soft mixture, exact OFF, direct heads, simplex optimizer, whole-policy choice |
| plans.py | SeedSequence namespace/component map, fold and row plans, immutable full dependencies/run IDs |
| state.py | complete stage state/cursors/RNG/optimizer/selection/source identity; strict resume |
| diagnostics.py | four OOF policies, support/functional interventions, isolated bounded RHD |
| metrics.py | true probability AUC,epsilon1e-7LL,sixcell/domain/task/macro/natural,raw outputs,bootstrap |
| costs.py | nonoverlapping stage leaves, label access, actual/standalone/development accounts |
| cli.py | inspect/build-plan/verify-semantics/run-stage/decide/freeze-evaluation/evaluate-exposed/summarize genuinely execute |

Coder may split implementation owners by module while sharing frozen interfaces. A coder report must state actual files/changes, tests, omitted conditional components and the exact scientific semantics preserved. No algorithmic implementation begins before this completed specification; independent reviewer reads this, 04/05, actual code and coder reports before admitting training. Conditional P3/P4 implementation can receive its own pre-run review when triggered, without altering locked design. Untriggered later stages should return honest NO_START, not pretend execution.

## 9. State, logs, metrics and output

Save model parameters/buffers/modes; all actual optimizersm/v/step;stage/pass/logicalbatchcursor;full RNG; row/fold/shuffle plans;code/config/data/source hashes; native selected/endpoint/history; normalizers;m/tau/w/geometry; final library;whole-policy candidates;cost/access state. Resume only at full logicalstep boundaries; a half-microbatch accumulation is not a completed step. Never repeat eta folds/readerprefit/perturbation or reselect backbone on resume. New stage optimizer fresh except explicitly copied ETA_DIRECT and continuous readerP_form→P_read. Missing-source grads remainNone.

Per-row files contain actual row_id/domain/y/p/role/checkpoint_hash,fold where applicable; model prediction has no y input. Save native epoch24, source/pilotpass12,reader pass10 and selected separately. All metrics sixcellcounts/positives/AUC/LL,domain/taskmacros,sixmacro,naturalLL,taskpooledAUC,worstcell and harm. Evaluation p→FP64→clip[1e-7,1−1e-7] for LL across all roles/methods; AUC uses original p. NA singleclass/emptycell never becomes silent five-cell macro.

Demand logs: dmean/variance,OOFmuerror vs train-fold constant mean,OOFeta/actions,candidate probabilities,quartile cutpoints,fit support/clip rates,source extrapolationmu/bw distributions,label/confidence strata,weight mass/ESS/correlation,actual shuffle changes. No source-demand accuracy is observable from this extrapolation and none may be fabricated.

Formation logs: active owner map,w applied once,per-functionsteps/momentnorms,loss/gradient ownership. Read logs: every fullcandidate group/P_calchoice/OFF,normalizeridentity,finalpredictor reconstruction. DFD added deployedparams200304;train-onlyeta50502;measure actualnative,total/activeparams,caches/latency instead of using counts as speed claims. Latency batch4096,5warmup,30sync repeats,median/P95/precision/hardware/load.

Must emit SOURCE_MANIFEST,DATA_ROLE_MANIFEST,PLAN,SEMANTIC_RECEIPTS,DECISION_P1,DECISION_P2,fullfrozen evaluation spec,source role/access/cost ledgers and artifact index. Final required reports: REPORT_INDEX,FINAL_REPORT,IMPLEMENTATION_REVIEW,DATA_AND_ROLE_AUDIT,P1_DEMAND_REPORT,FULL_PIPELINE_REPORT,ATTRIBUTION_REPORT,COST_REPORT,OPEN_QUESTIONS. Publish compact source/reports to date-named user branch after completion; no rawdata/checkpoints automatic upload.

Statistical unitseed;never6cells,epochs,OOFfolds,latencyrepeats or interventions. Formal4seed pairedrawdifferences,mean,sampleSD,t95=mean±3.182446*s/2. Original-user pairedclusterbootstrap1000 with same sampleduserweights across methods, both tasks/all domains;report invalidmissingclassresamples. If rawuser unavailable(rowfallbackKR),explicit independence limitation. Do not mix seed and bootstrap uncertainty into fictitiously larger n. Developmentn2 stays separate;no significance-driven additionalseed.

## 10. Reviewer acceptance: minimal necessary checks

1. **Native/data:** both real schemas one batch parity sigmoid(adapter) vs pinned actual fixed≤1e-6; actual4shared LN256;gate detach;canonicalcopyidentity;fitonlyencoder;atomic rolesdisjoint;microbatchSample loss/grad against full logicalbatch atol1e-6rtol1e-4 with measuredmax.
2. **Demand/weights:** FP64BCEdifferenceidentity≤1e-10;mu no labels;OOFtrainIDdisjoint;actualsource domain unchanged;centeredmean/weightedSDcorrect;wsum2≤1e-6/range;UNIFORMexact1;SHUFFLEentirejointstratifiedvectors preserved;GEOMfitonly.
3. **Gradient/Adam/resume:** real smallbatch native/pilot/eta no formationgradient;foreign functions no targetwrite;absentsourcegradNone and states unchanged;parametercopy but newoptimizerempty;samecompletecheckpoint nextlogicalstep allclose1e-6/1e-4 includingmoment/cursor/RNG. Do not accept state_dictweights-onlyresume.
4. **Prediction/selection:** exactOFF=LOCAL;stablemixtureBCE vs reference FP64abs≤1e-10/FP32p≤1e-6;P_calcompletegroupselection;simplexfeasibility+FWgap;noexposedlabeldependencies;independent finalpredictor reconstruction.

No unbounded engineering tests or new hyperparameter ablations. Relevant uncertainty gets smallest direct check; irrelevant style/generalengineering risks do not stop real training. Nonfinitevalues/roleleakage/wronggradient/wrongselection/failedreconstruction block affectedruns, with deterministicbugfix rerunning every affected pairedmethod/seed;no selective rerun of losing scientificresults. A failed single seed is retained,not replaced.

## 11. Falsification, stopping, bounded RHD and issue resolution

P1 failure rejects usable demand at this frozen estimator/representation/support scale and closes formation;it does not prove all conditional heterogeneity absent. P2 failure against UNIFORM rejects incremental formation;failure againstGEOM/SHUFFLE leaves general specialization/weightmarginal explanations;failure againstdirect/localcontrols rejects practical necessity. P3failure againstU_BUDGET/POOL leaves label-pass/pooling sufficient. AUC/cellharm keeps risk-ranking/celltradeoffs explicit. Nonvariableweights/OFF are real degeneracy outcomes,not permission to alterdesign. Crossseed/rolecontradictionsanddomain/task/stagedecompositions remainvisible;domainsize is not identified as maincause without manipulation.

RHD remains bounded and independent: only KR3101/3102 R0_BASE COMPLETEDepoch12/20 fouranchors with model/Adam/RNG/originalnextrowplan. Missingany→MISSING_ARTIFACT,nonewoldbaseline,noalternativeseed/epoch/PBanchor. FourbranchesALL/PRIVATE/SHARED/STOP;H1/20/40/80 onsame80stepbranch;STOPparentonlyeval0trainforwards/RNGmovement;freezegradNoneandmutablebuffers. Shared=embedding,STARsharedweightbias,star_mlp,skip_conn,sharedexperts,taskexperts;remainingdomain slots/domainexperts/gates/towersprivate. Max960updates,640newifALLverifiedreusable. Q_explore/P_update scoreonly,noactionselection. PRIVATEeligible criterion from05 only yields futureRESEARCH_REOPEN_ELIGIBLE,neverautomaticsecondresearchline.

Resolved apparent conflicts before implementation:

- start.md legacy seed2022–2027 and `examples/` instructions are historical general examples;this explicit newpackage/seed5101/6101CLI contract controls the DFD run.
- start concurrency overrides05defaultsingleheavyprocess;onlyactuallyavailabledevice,same-device85%memoryANDpowerdecision,recordload,dontkillotherprocesses.
- AGENT.md historicalD3/V1/V2 tests are not automatically applicable toDFD;only04P0checks thatcanchangeitsconclusionapply.
- Main/sourcebranch and finalpublication branch naming serve distinct purposes;implementfrommainindependently,publishcompletedsource/reportstouserdatebranchwithoutoldbranchmerge.
- P1 scoresOOFP_form;Q_exploreonlyP2/P3;neverm/tauusedinP1strategy. This is explicit05table,notaresult-dependentchoice.
- ηcrossfitOOFandallP_formrefitaredifferentartifacts;finalreaderprefitreusesP_formlegallyandremovesOOFstatusforthefinalpredictor.
- FormerRHDfirstblockepoch12/20anchorsinoldSPEC_RESOLUTIONdonotmatchthisroundcompletedepoch12/20;donotsubstitute.
- KRoriginaluserabsencehas05authorizedrowbootstrapfallback;recordMISSINGsupportandlimits rather thaninventidentitiesorunnecessarilyblockcore.

No unresolved mathematical contradiction in locked04/05 requires redesign. Runtime source/data/seed/diskavailability and pinnedadapterparity remain P0 factual checks, not assumedPASS. If a real external blocker prevents completion, preserve complete resumable state and state the exact dependency; do not rename implementation failure as algorithm failure or deliver only a proposed plan as completed experiments.

## 12. Shared implementation API and staged admission

The following explicit interchange contract prevents the prefix/P1 owner and the formation/readout owner from implementing different tensor axes or role rules. Internal helper names can be changed together before review; the manifest must record the final contract. This section changes no algorithm.

**RoleBundle** in data.py contains `dataset`, `dataset_id`, `x` memory map indexed by original global row_id, `y` memory map with task columns0/1, `domain` int64 array indexed by row_id, ordered `schema`, atomic `roles:dict[str,int64[n]]`, original user/item identity arrays if recoverable (else None with reason), `manifest` and `data_hash`. Expose `ids(role,domain=None)`, `batch(ids,role,device,purpose,authorization=None)` and `verify(output_dir)`; membership is checked before tensors are returned. batch output has x[B,F],y[B,2],domain[B],row_id[B],role. Atomic exposed names are test_exposed and Q_legacy_confirm_exposed; historical test/benchmark_test_exposed/Q_confirm aliases map once and never become extra data pools. Parent P_label is not an atomic permission for downstream fitting. The label-free role splitter and method freeze authorization are separate from model code.

**NativePacket** in native_adapter.py has `.logits[B,2]`, `.shared[B,4,64]`, `.h[B,256]`; actual domain is provided by batch. `NativeAdapter.forward(x_batch_major)` returns actual logits and `encode(x_batch_major)` returns packet, always with historical gate detach. If accepting old risk keyword for bridging, reject risk=True in all production DFD paths. A feature cache may store packet outputs for every original row in memmaps, or per-role arrays plus row-to-cache indexes; its access method must accept global row_id and return the same order. On-disk cache record must include its layout, native selected checkpoint hash, x/encoder/source extraction hashes, dtype and completed-row ranges. Cache tensors carry no gradients.

**FrozenCache** shared by modules provides `get(ids,device)`→`{h:[B,256],z:[B,2],domain:[B],row_id:[B]}`. y is obtained from authorized RoleBundle.batch, not secretly from feature extraction. A dense cache's first dimension is original global row count; a fit-only weight file must instead explicitly store `fit_row_id` and its inverse map. Never index fit-position arrays by global row_id without that mapping.

**PilotBank** in functions.py: owner order is head index `3*q+s` (task-major; same cell order throughout reporting). `.residuals(h)` returns `[B,3,2]` with axes row/source/task; `.local_logits(h,z,domain)` returns `[B,2]`. Training loss builds only active-source heads; an all-source inference tensor is not reused for loss in a way that creates zero gradients for absent sources.

**FunctionBank** stores independent heads with explicit tuple keys `(source,intended_target,task)`; intended_target differs from source and is ordered ascending for each source. A batch full forward can return `[B,3,2,2]` with axes row/source/intended-slot/task. LOCAL_CAP uses `(target,local_slot0..3,task)` instead. Export `.candidates(h,z,domain,pilot)`→`logits[B,2,5]` in exact local-first,source-ascending,intended-target-ascending order; probabilities are sigmoid of these logits. The bank owner map is serialized, so the readout owner need not infer it from list positions.

**Demand outputs** include OOF records indexed by original P_form row_id with fold,eta[B,2],mu[B,2,2],candidate probabilities[B,2,3] for that row's actual target; the foreign-source axis is ascending source excluding target. A pair table keyed `(source,target,task)` stores m and OOF quartile cuts; a table keyed `(source,task)` stores tau. The responsibility cache stores fit-row IDs plus w `[nfit,2,2]` with axes row/task/intended-slot and uses the row's actual source to identify sorted intended targets. This axis contract differs from bank inference's task-last layout intentionally; convert explicitly once and record dimensions.

**Reader input contract:** candidates `[B,2,5]`, h[B,256],z[B,2],domain[B]. Per-head `[h,z,p0..p4]` stays263; do not flatten both tasks' probabilities into the input. Reader output p[B,2], alpha[B,2,5], and selected policy metadata with one global candidate ID. Direct eta/H heads return probabilities[B,2] and have their own input contracts. Stable loss accepts candidate logits and log-alpha, never a clipped probability pseudo-logit.

**State helpers** in state.py should expose atomic JSON, file/array/state hash, RNG capture/restore, and atomic `save_checkpoint(path,payload)` / `load_checkpoint(path,expected_identity)`. Payload keys include `identity={method,run_id,dataset,seed,source_hash,config_hash,data_hash,plan_hash}`, `stage`, `pass`, `next_logical_step`, `models`, `optimizers`, `rng`, `plans_or_hashes`, `normalizers`, `selection`, `dependencies`, `cost_state`. Store model owner maps and optimizer param-group ordering. Loading validates immutable identity before mutation. CPU load then intended device restore must handle optimizer tensors correctly. Save only a completed logical step, not a partially accumulated gradient.

**Metric and result helpers** should share a single `evaluate_probabilities(y,p,domain)` implementation and return explicit `cells` in q-major/d-minor order plus macro/domain/task/natural/pooled fields, counts and clip rate. `save_predictions` preserves original row order and complete predictor checkpoint identity. Every output declares whether selected or fixed stage endpoint and which role was scored.

**Prefix/P1 owner:** data/state/plans/native/prefix/cache/pilot stage orchestration/P1 diagnostics/metrics/cost/CLI. **Formation/readout owner:** functions/readers plus formation/controls/conditional P2/P3 orchestration; coordinate PilotBank interface with prefix owner immediately. If both need functions.py, designate one owner to create the base classes and the other calls them; do not concurrently overwrite the file. Share helpers through explicit imports, not copied divergent versions. Report/review ownership is separate from algorithm implementation.

P0/P1 can receive an independent source-matched review and begin its prescribed retained runs once its own prerequisites pass, while P2/P3 implementation proceeds separately. P2/P3 execution requires both dataset promotion and its actual independent implementation review. P4 author adaptations are implemented and reviewed only if triggered; their locked formulas/configuration remain known now. Dormant unimplemented commands must honestly return NO_START until triggered rather than fabricate work. All source modifications used by a newly launched stage must be in that stage's reviewed manifest; a later module addition cannot silently change an already running reviewed prefix.
