# DESIGN SPECIFICATION — Cycle04 completion-weight attribution

Status: **COMPLETE**. Unresolved design blockers: **0**. Experiment: `completion_weight_attribution_20261117`. Date: 2026-11-17. This is the formal DESIGN handoff to original parent `01a1dfb8-605b-7430-8462-5fd3da0b048f`. Parent acceptance, implementation, independent review and engineering signoff are subsequent stages. Candidate efficacy remains to be measured.

The designer's sole write is this file. The six accepted research files remain immutable. This design supplies engineering decisions within their scientific scope; it does not add interventions or change the queue, candidate, comparisons or budget. Implementation and experiment execution were not performed in this DESIGN stage. Parent alone dispatches subsequent fresh native children, with `fork_context=false`, and closes each permanently after its file/report handoff. This designer has no descendants and is not to be resumed.

## 1. Research question

Does the MovieLens selected-AUC deficit of natural quadratic ranking require its opposite-class quadratic ranking gradient, or can the retained completion BCE weights and their reduced global BCE mass already produce it? Under the identical natural-source training protocol, compare the objective chain `C -> S=mC -> W -> N=W+R`. Evaluate the same question separately on KuaiRand rather than presuming transfer.

The accepted candidate is **COMPLETION-BCE-025 (W)**. The only new control is **MASS-SCALE-025 (S)**. Reuse ten exact numeric CLEAN (C) and ten constant NQR-025 (N) trajectories. W and S have identical total BCE coefficient mass in a matched batch; their class, cell and task allocations differ. The W-S contrast therefore concerns combined heterogeneous completion weighting. Class-only or domain-only causality is not identified.

Design inputs were read: both workspaces' governing policies (start.md decoded GB18030), current W runbook top correction, all six P4 research files, A4 research acceptance/cleanup receipts, frozen loss/model/Adam/EMA/evaluator/RNG/copy/IPC APIs, actual historical numeric schemas, complete latest C3 REPORT and DECISION, and protected publication/source receipts. Both workspaces' stop controls were ACTIVE, manual stop false, with neither stop sentinel present at the design read. Parent must read them again before each dispatch.

## 2. Hypotheses being tested

1. **Ranking necessity hypothesis:** the repeated ML rank deficit requires R. W retaining that deficit with R absent rejects necessity under this protocol.
2. **Heterogeneous weighting hypothesis:** W-S can explain an adverse part of W-C conditional on the exact shared batch mass. Coherent adverse W-S with a signed descriptive interval provides conditional evidence; uncertain W-S does not establish equality.
3. **Global scaling competitor:** S-C may account for a similar deficit under unchanged coupled Adam/EMA. Scaling affects the data term relative to coupled weight decay and epsilon; source Adam need not be invariant to it.
4. **Candidate hypothesis:** W improves independently AUC-selected fixed EMA over CLEAN fixed EMA on both datasets with the registered practical/cell protection requirements. A useful attribution result can accompany candidate failure.

Retain competing explanations involving shared representation, label prevalence/opportunity, private optimizer history, EMA readout and checkpoint selection. The four means telescope algebraically; they are not percentages of trajectory mediation. N-W measures ranking addition conditional on W, including magnitude and downstream dynamics. The missing S+R arm leaves the ranking-by-weighting interaction unestimated. Local common-state diagnostics do not identify a full-path optimizer or EMA mediation effect.

## 3. Variables changed

Only the scalar data objective changes in new formal training:

| Arm | Objective | Status |
|---|---|---|
| COMPLETION-BCE-025 / W | `.75*C + .25*sum_c w_c H_c` | new; sole candidate |
| MASS-SCALE-025 / S | `m*C`, exact actual-batch m | new; scale control |
| CLEAN / C | exact source probability BCE | reused numeric control |
| NQR-025 / N | frozen C2 `.75*C+.25*sum_c w_c(H_c+o_c Q_c)` | reused numeric control; pure diagnostic reference |

Lambda is .25 at every formal update. Lambda0 is an engineering source-identity branch, never an additional formal arm. No alternate coefficient, mass normalization, learning-rate change, extra algorithm, control retraining, augmentation, curriculum, projection, residual momentum or RNG is authorized.

## 4. Variables held constant

| Contract | Fixed value |
|---|---|
| Datasets | natural MovieLens-1M and KuaiRand populations/labels, existing original 80/10/10 split and train-only preprocessing |
| Seeds | exactly 2022, 2023, 2024, 2025, 2026 on each dataset/arm |
| Model | source M3oE fixed1; 3 domains, 2 tasks; embedding16; experts4; FCN `[512,256,256,64]`; source routing, sharing, towers and fusion |
| Gate | exact ordinary source `gate(emb.detach())`; no probe full-derivative route |
| Training | 50 whole epochs; batch40960; real final tail; FP32; no autocast or early efficacy stop |
| Adam | installed source `torch.optim.Adam`; lr .01, betas (.9,.999), eps1e-8; coupled WD ML1e-4/KR1e-5; foreach=False, fused=False, amsgrad=False |
| EMA | initial deep model copy; rho ML .9828205985452511/KR .995; one source update per actual Adam step; copy source buffers |
| Loader | private CPU torch.Generator seeded `seed+11000`; exactly one randperm per epoch; source row order and row IDs |
| Environment | Python3.13.5, PyTorch2.9.0+cu126; exact requirements-phase0-3.lock; one CPU thread; source seeds; TF32 matmul/cuDNN off; cuDNN benchmark=False |
| Selection | independently online/fixed_ema earliest strict maximum validation six-cell Macro AUC over epochs1..50 |
| Evaluation | unmodified source evaluator, unperturbed validation, source probability/loss precision and undefined-cell semantics; LL at the same AUC-chosen state |

No new deterministic-algorithms setting, altered `None` handling, parameter group or model buffer is allowed. Tensor-zero gradients remain active for source coupled decay, moments and private step; `grad is None` skips that parameter. Never infer activity solely from an absent domain. Coupled WD is unscaled in both arms.

ML train/validation rows 800167/100021, domain train counts `[155949,318599,325619]`, 20 updates/epoch, tail21927. KR train/validation rows 2006132/250767, domain counts `[172782,1743898,89452]`, 49 updates/epoch, tail40052. Actual class counts and source difficulty/baseline performance remain covariates, not manipulated difficulty or inferred functions of domain size.

## 5. Mathematical and algorithmic definition

### 5.1 Exact loss construction and support

For a nonempty batch B, probabilities/targets have shape `[B,2]`. Model input has source layout `[F,B]` with domain in `x[-1]`. Use `torch.nn.functional.binary_cross_entropy` in probability space, unchanged clipping/reduction behavior. Let entry loss be ell_it and `C=BCE(p,y,reduction='mean')`. For cell `c=(d,t)`, n_c is its row count, n_ca class count, `pi_ca=n_ca/n_c`, `w_c=n_c/(2B)`, and bbar_ca is the present class mean of `BCE(...,reduction='none')`.

`H_c=sum_{a present} pi_ca**2*bbar_ca`

`o_c=2*pi_c0*pi_c1`

`A=sum_c w_c*H_c`

`W=(1-lambda)*C+lambda*A`

`r=sum_c (w_c*lambda*2*pi_c0*pi_c1)` and `m=1-r`.

`S=m*C`.

Counts, proportions, coefficients, m and r are detached host numeric quantities. In ordinary support, form A by the frozen C2 completion reduction order: domain0..2, task0..1, present class0 then1; initialize a scalar `clean.new_zeros(())`; add cell weight times H; finally form W directly. Do not obtain production W by subtracting R from N. S multiplies source C by that unrounded Python-float m in its probability dtype. Do not pre-round m or substitute a dataset-average mass.

An independent row expression for fixtures is `W=mean_{2B}[q_c,y*ell_it]`, `q_ca=1-lambda+lambda*pi_ca`. The scalar C, W and S are differentiable in probabilities and source parameters; the ledger contains detached numbers only. No derivative flows through counts or weights.

For both classes present, define `mu_ca=mean(p_ca)`, `var_ca=mean((p_ca-mu_ca)**2)` and `Q_c=.5*((1-mu_c1+mu_c0)**2+var_c1+var_c0)`. Then `R=lambda*sum_c w_c*o_c*Q_c` and N=W+R in real arithmetic. The read-only reference for N is `experiments.natural_quadratic_ranking_20261116.core.loss_components(...,arm='NQR-025',mixture_lambda=.25)['loss']`. Use it only on scheduled diagnostics and finite checks; never call its training step in W/S production. Explicit opposite-class pair enumeration is a tiny CPU check, not a training operation.

### 5.2 Actual coefficient ledger

In ordinary support:

| Quantity | W | S | counterfactual N |
|---|---|---|---|
| present-entry BCE coefficient | `q_ca/(2B)` | `m/(2B)` | W coefficient |
| cell BCE mass | `w_c*(1-lambda*o_c)` | `m*w_c` | W mass |
| task t BCE mass | `sum_d w_dt*(1-lambda*o_dt)` | `m/2` | W mass |
| total BCE mass | m | m | m |
| applied opposite-class mass | 0 | 0 | r |
| total applied objective mass | m | m | 1 |

For lambda .25, `.875 <= m <= 1`. Actual W/S total mass is m; no normalization by m or redistribution of r. Retain both actual masses and counterfactual removed opportunity mass. W task masses can differ; S task masses are m/2. A minority-class row generally receives a smaller W coefficient than a majority-class row. This is not inverse-frequency balancing.

If n_c=0, n0=n1=0, weight/masses/opportunity=0, pi0/pi1=null; all class statistics and absent-entry coefficients=null. If n_c>0 but n_ca=0, pi_ca=0 is defined, its mean/variance/BCE/coefficient=null; the present class has pi=1. Opposite-class Q is null unless both classes exist. An ordinary single-class W cell has entry coefficient `1/(2B)` while S still uses global `m/(2B)` whenever other cells have mixed classes. Never renormalize supported domains, fabricate missing means, drop an entry, or confuse absence with measured zero.

If every nonempty cell is single-class, set r=0,m=1 and delegate the actual update to frozen `source_clean_step`. Lambda0 likewise delegates exactly to source CLEAN, bypassing `.75*C+.25*C` arithmetic. This branch decision occurs before any model forward. Branch strings are `ordinary`, `all_single_class`, `lambda_zero`, with lambda_zero taking precedence. Batch0, bad shape/device/dtype, nonbinary labels, domain outside integral0/1/2, or nonfinite/out-of-range probabilities are technical errors. Labels/probabilities use matching FP32 or FP64 in the pure helper; actual training is FP32.

### 5.3 Core public interfaces: fixed contract for both coders

`core.ARMS=('COMPLETION-BCE-025','MASS-SCALE-025')`; `OBJECTIVES=('C','S','W','N')`; `CELLS=tuple('d%dt%d'%(d,t) for d in range(3) for t in range(2))`.

`loss_components(probabilities, targets, domains, *, arm, mixture_lambda=.25, diagnose=False)` returns a dict with exactly these semantic fields:

| Field | Type/meaning |
|---|---|
| loss | differentiable scalar for selected W/S arm |
| clean_loss | differentiable source C |
| same_class_loss | differentiable A in ordinary support; source C on identity branch |
| completion_loss | differentiable W; source C on identity branch |
| scaled_loss | differentiable S; source C on identity branch |
| nqr_loss | differentiable frozen C2 N when diagnose=True; otherwise None |
| branch | one of the three branch strings |
| ledger | detached schema below |

On identity branches all four objectives equal C; the reference helper is still used for N when requested. On ordinary S steps the W tensor may be constructed for consistent measurement, but only `loss` receives the live backward. Off diagnostics no N/Q/score-variance graph is constructed. `same_class_loss` is distinct from completion_loss: the former A is unmixed, the latter W includes .75C. Do not reuse C2's ambiguous `completed_loss` name for W.

Ledger `schema_version=1` fields: `arm,branch,B,entries,mixture_lambda,bce_mass_scale,removed_opposite_mass,total_mass,bce_mass,opposite_mass,task_mass,completion_task_mass,cells,diagnose,statistics_status`.

Here bce_mass_scale=m; removed_opposite_mass=r; total_mass=bce_mass=m; opposite_mass=0 for both new arms; task_mass is the actual arm's task BCE allocation with string keys `'0','1'`; completion_task_mass always records W's allocation. `cells` has exactly six fixed keys. Each cell contains:

`n,n0,n1,pi0,pi1,weight,opportunity,removed_opposite_mass,completion_bce_mass,scale_bce_mass,bce_mass,opposite_mass,bce_coefficient0,bce_coefficient1,completion_coefficient0,completion_coefficient1,scale_coefficient0,scale_coefficient1,mu0,mu1,var0,var1,bbar0,bbar1,H,Q_R,statistics_status`.

`bce_*` fields without a completion/scale prefix describe the actual arm. All applied opposite_mass fields are numeric zero; removed_opposite_mass is the hypothetical N rank coefficient. No unit-mass `total_mass=1` survives from C2 into a new-arm ledger. Present q can be recovered as `2B*completion_coefficienta`; exporters must include `q0,q1`, null for absent entries. Empty-cell opportunity=0 is a coefficient convention; pi/statistics remain null. Status reasons are per statistic as necessary: `not_scheduled`, `empty_cell`, `absent_class`, `requires_both_classes`, or `observed`. Structural absence takes precedence over not_scheduled. A measured zero statistic is numeric zero with observed status.

Off diagnostic batches all `mu*,var*,bbar*,H,Q_R` ledger values are null/not_scheduled for present support. W's differentiable computation still uses its class BCE reductions internally; no claim of having recorded those reductions is made. On diagnostics compute detached class means and population variances with the source dtype/count denominators from that same ordinary probability output. No additional graph derivatives or model forwards are needed for these statistics.

`train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` preserves source calling order. It returns JSON-safe finite detached values:

`loss,clean_loss,same_class_loss,completion_loss,scaled_loss,nqr_loss,branch,ledger,row_order_hash,optimizer_updates,ema_updates,ordinary_forwards,scalar_backwards,probe_forwards,view_forwards,algorithm_rng_draws,extra_autograd_sweeps,virtual_adam_steps,seconds,diagnostic_seconds,diagnostics,parameter_activity,adam_step_counters`.

Each successful call has optimizer_updates=ema_updates=ordinary_forwards=scalar_backwards=1; probe_forwards=view_forwards=algorithm_rng_draws=0. extra_autograd_sweeps/virtual_adam_steps are 3/4 iff diagnose, otherwise0/0. No persistent channel or state outside source model/optimizer/EMA exists. `parameter_activity` maps every source name to `none`, `tensor_zero`, or `nonzero`; adam_step_counters maps every name to its actual integer step or null for uninitialized. Tensor_zero means every gradient element equals zero, not a norm threshold. Runtime stores an additional boolean last_parameter_activity if needed, derived from that mapping.

row_ids are actual source train row IDs in complete-batch order, converted to np.int64 before source state_digest. Require them for real engineering/formal calls. Small CPU fixtures may omit them and use local arange. Inputs/config/context are not mutated.

Config fields are fixed: `cycle_id='cycle_04',dataset,arm,objective='completion_weight_attribution_v1',mixture_lambda=.25,epochs=50,batch_size=40960,precision='float32',lr=.01,betas=[.9,.999],eps=1e-8,weight_decay,rho,steps_per_epoch,tail_rows,loader_seed_offset=11000,gate_training_detached=True,probe_gate_full_derivative=False,algorithm_rng_draws=0,data_hash,selection_hash`. Bind model/source/environment at job/manifest level. Engineering config may use mixture_lambda0 with a distinct declared check purpose; formal validators reject it. No C3 schedule fields or phase switch are accepted.

Context formal fields: `purpose='formal',clock_kind='formal',dataset,seed,epoch,batch,global_step`. Epoch1..50, batch1..K, global_step is the target post-step ordinal `(epoch-1)*K+batch`; saved runner state immediately before it has global_step one less. Actual B is40960 except batchK's exact tail. diagnose iff batch1 and epoch in1/10/30/50. Validate int fields excluding bool, seed membership and matching dataset/config before mutation. At every diagnosed real call, runtime also supplies `pre_loader_hash=state_digest(loader.get_state())` in context, captured immediately before train_step; core records this identifier without owning or advancing the loader. Core captures the actual pre-RNG hash itself. CPU diagnostic fixtures supply their actual fixture-loader hash by the same rule. This closes the loader-audit interface without introducing algorithm state.

Context engineering fields: `purpose='engineering_smoke'` or `'engineering_check'`, `clock_kind='engineering'`, dataset,seed,case_id,engineering_call_ordinal,engineering_state_step,sample_epoch=1,sample_batch,branch_role. Formal epoch/global_step fields are absent from core engineering contexts. sample_batch is1 or dataset K for real cases; state_step counts updates in that branch only, not all physical fork work. CPU fixtures may use their own finite sample cardinality explicitly. Context is measurement/clock data, never algorithm history.

### 5.4 Common-state gradient and real-Adam diagnostics

At each formal trajectory's first actual batch of epochs1/10/30/50, compute C,S,W,N from its single ordinary forward at current parameters and current arm's pre-update Adam state. Three `torch.autograd.grad(...,allow_unused=True,retain_graph=True)` calls obtain the objectives other than the live arm in C,S,W,N order with that arm omitted. Leave live `.grad` unset until the one scalar `.backward()` of actual loss. Do not materialize unused gradients as zero. Capture all four gradients detached in source named-parameter order after the live backward. `g_S=m*g_C` and `g_N=g_W+g_R` are algebra checks, not replacements for each objective's real floating backward.

Before the live optimizer update, apply each gradient in C,S,W,N order to one disposable `copy.deepcopy(model)` at a time, with a newly constructed source Adam and a deep CPU copy of the exact pre-update live optimizer state loaded into it. Copy/deepcopy does not call model constructors or consume RNG. Set clone .grad to detached cloned tensors or None according to that objective's actual autograd activity. Use `optimizer(...)` from the frozen helper and its actual installed Adam.step, including lazy initialization and its parameter-private steps. Never implement Adam numerically, substitute historical CLEAN moments, share mutable moments between clones/live/saved state, or copy hooks into clone optimizers. Record the clone update before discarding that clone; four clone models must never coexist on device.

Then call the single live Adam.step and source update_ema. The actual arm's virtual post-model and post-Adam hashes must equal the live post-step hashes exactly; record actual-vs-virtual update difference norm zero. EMA is updated only on the live route. At identical pre-EMA state, theoretical counterfactual update differences satisfy `deltaEMA_left-right=(1-rho)*(deltaTheta_left-right)`. Log the scaled vector norms/relationship algebraically; no four additional EMA loops or EMA-history substitutions. FP roundoff in separately expressed EMA arithmetic uses the algebra tolerance, not a relaxed replay hash.

Identity branches must still call the frozen source_clean_step, including when diagnose=True. For that uncommon diagnostic case, use scoped observational PyTorch hooks: a temporary outer-model forward hook receives the one output and obtains the three retained-graph gradients of the other objectives; an optimizer step pre-hook, reached after source scalar backward and before its Adam update, captures the live gradient and performs the four disposable steps. Hooks must return None, never replace output/arguments/gradients, be removed in finally, and be absent on the clones. Remove the temporary model observation hook as soon as it has captured the one ordinary output/probes, before any clone is deepcopied; the newly built clone optimizer has no live step hook. Use detached prestate copies prepared before the helper call. For lambda0 diagnostics the pure frozen C2 helper receives mixture_lambda=0; all-single-class diagnostics retain the configured .25. This preserves the source delegate while fulfilling the same 3/4 accounting without an extra forward. This is local diagnostic observation, not monkeypatching any old module or forward. Verify it in the finite identity/neutrality fixture. Ordinary production uses the direct sequence above; there is no general hook/adapter framework.

For global `ALL` and every existing top-level named-parameter prefix, flatten conceptually in the same full coordinate space. Missing components count as mathematical zero for norm/dot computations only; do not create a tensor grad for them. Compute dot products and squared norms in FP64. Never use the old TV cosine helper's intersection-only normalization when activity differs. Record:

* data gradient L2 norm for each objective, gradient residual norms W-S and N-W, gradient cosines W/S and N/W;
* actual-source-Adam parameter update norm for each objective, update difference norms W-S and N-W, update cosines W/S and N/W;
* active/None/tensor-zero parameter counts and coordinate counts per objective, activity-mask differences per pair, pre/post private-step min/max and uninitialized count;
* pre-model/Adam/RNG/loader hashes, actual host arm, row order, objective loss values, m/r, same-state provenance, actual-vs-virtual exact hashes, and scaled EMA difference norms.

Undefined cosine is null with reason `zero_norm`; an observed zero norm is numeric0. Every difference/cosine uses actual objective gradients/updates at the current W/S state. It is never labelled as a historical CLEAN trajectory. Historical unavailable measurements are null/`not_recorded_in_source`; off-schedule values are null/`not_scheduled`. No per-cell gradient sweeps, view forwards, RNG, live additional optimizer step or extra data batch is allowed. Clone preparation, diagnostic computation and disposal must leave mode, buffers, model/Adam/EMA, live gradients, all RNG and loader state exactly equivalent to diagnose=False after a call/suffix.

### 5.5 Preregistered numeric tolerances versus exact identities

Fix before implementation: FP64 algebra/gradient comparisons `atol=1e-12,rtol=1e-10`; FP32 algebra/gradient/EMA-reexpression comparisons `atol=2e-6,rtol=2e-5`, elementwise `abs(a-b)<=atol+rtol*abs(reference)`. These apply to independently reduced row/cell/pair formulas, N=W+R, g_S=m*g_C and equivalent gradient formulas. Count/mass host arithmetic uses atol1e-12/rtol1e-10. Require finiteness separately; tolerances never excuse NaN/Inf. Tiny derivative fixtures use interior probabilities so boundary BCE behavior does not dominate the equality check; source validation still covers endpoints without changing BCE.

Exact contracts have **no tolerance**: source delegate versus direct source path from identical states; initialized source model/EMA/Adam identities; diagnose on/off poststate, gradients, RNG/loader; sequential clone actual-arm poststate; source same-path validation; selected hashes/epochs; snapshot restore/replay and saved-state immutability; source/design/control file hashes; counters; owner ACK identity. Use tensor equality plus source state_digest, activity equality and private-state structure equality. Do not compare an entire timing/resource receipt hash for replay: the exact scientific identity projection is defined below. An algebra-tolerance failure or exact mismatch is a technical blocker to inspect, never a reason to broaden tolerances after results.

## 6. Exact affected modules/files and provenance

### 6.1 Authorized future write roots and two disjoint coder scopes

Path abbreviations, all absolute after expansion:

| Name | Root |
|---|---|
| W | `/home/xzf/Code/phase0-3_tvg_rm_20261116` |
| O | `/home/xzf/Code/phase0-3` |
| P4 | `W/research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04` |
| A4 | `W/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04` |
| N | `A4/attempt1` |
| E | `W/experiments/completion_weight_attribution_20261117` |
| R | `W/research_reports/completion_weight_attribution_20261117` |

Parent can dispatch two fresh coders with these exclusive file scopes and this shared interface contract:

| Coder | Files owned | Responsibilities/report |
|---|---|---|
| core | E/core.py, E/checks.py | exact loss/ledger, train_step, common-state diagnostics and finite core fixtures; A4/CORE_IMPLEMENTATION_REPORT.md and CORE_CPU_CHECKS.json |
| runtime | E/__init__.py, E/protocol.py, E/runner.py, E/tables.py, E/README.md | new roots/freeze/reference import, clocks/ACK/storage/recovery/admission/engineering, numeric extraction; A4/RUNTIME_IMPLEMENTATION_REPORT.md and RUNTIME_CPU_CHECKS.json |

These seven new package files are sufficient. No new dependency/framework/test tree is required. checks.py exposes `check_core()` and a CLI `--scope core|all`; `--scope all` calls runtime-owned `protocol.finite_checks()`, `runner.finite_checks()`, `tables.finite_checks()` after integration. Runtime finite checks use fixture data/in-memory fake request transcripts only; no GPU/IPC/training is run by coders. Core and runtime coders do not edit each other's files; explicit import interfaces here resolve integration. Parent combines the two reports as A4/CODER_IMPLEMENTATION_REPORT.md, then a distinct reviewer inspects actual files. No old package, P4 file, context, rootstate, Git/global config or historical receipt is a coder write target. Numeric run outputs are only N; final analysis/publication is parent's scope in R/authorized context paths.

### 6.2 Read-only helper imports and actual dependency closure

Use frozen pure functions with their existing signatures:

* auxiliary_objective_attribution_20261115.protocol: `SELECTION,digest,optimizer,load_protocol,make_model,setup,cpu_copy,state_digest,rng_state,restore_rng,evaluate,sha_file`; `immutable/write_json` only with new absolute paths.
* conditional_information_momentum_20261102_v3.protocol: `update_ema`; its guardian `request(attempt,command,**payload)` only with N supplied by the new protocol wrapper.
* task_view_gradient_residual_momentum_20261116.core: `source_clean_step`; auxiliary stage2.core: `synchronize`.
* natural_quadratic_ranking_20261116.core: pure `loss_components` only for N diagnostics/checks. Its train_step is not a production delegate.
* frozen TV runner `serve(str(N))` only as the parent-owned storage process. The server's own recovery command internally reads the existing pure stop predicate; no retired driver lifecycle function is invoked.
* audited_data: `array_hash,split_row_fingerprint,row_ids_hash` for explicit train/validation objects, never whole-bundle Test-inspecting convenience methods.

New protocol implements its own config, job, data-cache verification, source_manifest, reference-index verification, root checks, stop checks, prepare/review gates, epoch readers and extraction. Do not call old root-bound prepare/run/admission/export/ensure_guardian/source_manifest/reference_index functions, copy a frozen source file into a mutable edited variant, monkeypatch old globals, or scan historical artifact trees to define identity. Importing a module containing old routines does not authorize calling them.

For an explicit import closure, let `V=experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31` and `G=V/..`. Freeze these exact read-only repository files (with W-relative POSIX paths), plus all seven E files, requirements-phase0-3.lock and V/resolved_protocol.json:

* `experiments/auxiliary_objective_attribution_20261115/{__init__.py,core.py,protocol.py,runner.py,stage2/__init__.py,stage2/core.py}`;
* `experiments/conditional_information_momentum_20261102_v3/{__init__.py,guardian.py,protocol.py,vendor/__init__.py,vendor/ema02/__init__.py,vendor/ema02/data.py,vendor/ema02/evaluate.py,vendor/ema02/protocol.py,vendor/ema02/vendor/__init__.py}`;
* `G/{__init__.py,protocol.py,views.py}`;
* `V/{__init__.py,audited_data.py,metrics.py,model.py,protocol.py,reference.py,source_protocol.py,basic/__init__.py,basic/activation.py,basic/features.py,basic/initializers.py,basic/layers.py,utils/__init__.py,utils/data.py}`;
* `experiments/natural_quadratic_ranking_20261116/{__init__.py,core.py}`;
* `experiments/task_view_gradient_residual_momentum_20261116/{__init__.py,core.py,protocol.py,runner.py,autonomous.py}`.

The last autonomous.py is a frozen dependency of the storage server's pure stop read on recovery, not an execution entry point. The closure above follows actual helper/module imports, including imports needed to load the frozen serve module. It is not the union of all historical 94/87/79 manifests. Review the new imports against this explicit list; any genuinely required additional local dependency is enumerated with import reason and frozen before review. No rglob or silent import expansion. Hash external installed `torch/optim/adam.py` separately as runtime implementation provenance (`b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c` at the inspected environment), alongside package-lock verification. No editable vendor optimizer is introduced.

### 6.3 Immutable research/design/source identities

Accepted six-file P4 research bundle SHA256: `a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a`. All six hashes were checked against A4/ASTRA_HANDOFF.json during this design:

| Logical name | SHA256 |
|---|---|
| ALGORITHM_DESIGNS.md | 35a2fa2ad974511e2401bb6e8a5291f9715d12453127344ad035407195ae2988 |
| CODEX_EXECUTION_PLAN.md | 56e8b8a90add63d694655a1b908e84e63a6ebdc11bb0012e950a10c88255b558 |
| CRITICAL_REVIEW.md | 696fe772efb113aa4304d014a9c4a573cb240ccff02bebad2bba4781fd64044f |
| EXPERIMENT_MANIFEST.json | 1e93bc945533d15ed869ee6e7716cbcf701eaebfd6ccfdb330e0f3707c118b10 |
| LITERATURE_AND_NOVELTY.md | 5f0ddf966283c241638a52e038834efc79b4dc530a35e437b7b04a6b59a649c3 |
| RESEARCH_EXPLORATION.md | 47e94eb2bf62b312b7a692bfed3b3af6e40106c83ac76247350b87655c0d5503 |

After accepting this formal design, parent freezes **seven** logical files: six at P4 and `DESIGN_SPECIFICATION.md` at A4. Define sorted `[logical_name,sha256]` pairs by basename and hash their UTF-8 compact JSON with separators `(',',':')`, ensure_ascii=False. This retains the historical bundle recipe while permitting two directories. A4/DESIGN_ACCEPTANCE.json must contain verdict COMPLETE_APPROVED_FOR_IMPLEMENTATION, the seven pairs, seven-bundle hash, six-bundle hash, and a separate explicit logical-name -> absolute-path mapping. Require all seven names unique and correct path provenance. Do not attempt P4/DESIGN_SPECIFICATION.md. This file cannot contain its own final hash; parent computes it after completion. Preserve A4/ASTRA_HANDOFF.json context_closed=true and bind its SHA in acceptance/source/run receipts.

New source receipt is `files=[{path,sha256},...]` sorted by exact W-relative path; source hash is frozen auxiliary `digest(files)` (sort_keys=True, compact JSON). Keep separate external-runtime files and separate design/control/data identities. New source hash is pending actual implementation, never a historical hash. N/source_receipt.json and run_manifest.json bind source, seven-design, six-research, controls index, environment/lock, selection and data provenance; reviews and owner bindings must agree. If reviewed source changes, invalidate dependent reviews/freeze and re-review only affected gates; no silent overwrite of frozen receipts.

Published sourcebase commit `4c95dc877b85cc676bf7366958f8c7de0291871e`, tree `0dbbe5c0c53dc843a1effd929df197cc7399f18d`, is established by A3/PUBLICATION_RECEIPT.json, SHA256 `9bf02b0bc2189e556aafbfc98f7dc0cc118494820546cd77f56fa69a4c6b4bc9`, status REMOTE_MAIN_VERIFIED,105 files. Latest C3 evidence review is A3/FINAL_EVIDENCE_REVIEW.json, SHA256 `5ab48bb76c276c2d42b7d233d181dec4caf6a8bc190cdeee0a22b6f84319005b`, PASS/zero blockers; it certifies C3 only.

Forever protect these existing receipts, their enumerated files and old owners:

| Historical source | Receipt under W/artifacts/26-11-16_augmentation_gradient_momentum | Effective source / file count |
|---|---|---|
| C1 | astra_cycles/cycle_01/attempt1/RESOURCE_1_SUPERSESSION.json | 6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1 /79 |
| C2 | astra_cycles/cycle_02/attempt1/source_receipt.json | ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae /87 |
| C3 | astra_cycles/cycle_03/attempt1/source_receipt.json | a7666566be5790a0e97613d11c37792122a4028f72716669f52be1ceffe99002 /94 |

Their receipt SHAs respectively are `b214b1a45564acd29b47824d7fb6b96d7c59c8f0c4ce311b19800bd8f91bb0d1`, `15c62913dc9fb9188ebb5f0f22bab34d57bd40bcccb8ee41c81305a6d2b7c944`, `85b10a334cb109553f954d77639afad690aa6bca8207a65d4347f32ab2034938`. Do not use superseded initial C1 identity. Bytecode cleanup was already completed and source94 reverified in A4/PARENT_BYTECODE_CLEANUP_RECEIPT.json; do not repeat that audit absent a new concern. Every project command uses `PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python ...`. Syntax verification can use compile(source_text,...); never py_compile/compileall against old packages.

## 7. Implementation requirements

### 7.1 Data and historical numeric reference contracts

Use existing trusted prepared cache `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache/{dataset}.pkl`. Raw CSVs are `/home_new/xzf/datasets/ml-1m/ml-1m.csv` and `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`. Expected raw SHAs ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`, KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`. Bind complete recorded data/config identities from frozen V/resolved_protocol.json and A3/attempt1/{dataset}_data_manifest.json, with their numeric receipts. Verify raw CSV bytes and train/validation arrays/row IDs/fingerprints, schema/encoder/scaler, natural train cell counts and shape/cardinality. Source-model construction uses the same features and source dataset configuration.

Implement a small new dataset_cache loader for this exact cache tuple `(bundle,manifest)`; no reconstruction or calls to old source_cache. The old source_cache hashes every array and calls whole-bundle split_fingerprints/row_id_fingerprints, which traverse Test. New code must only dereference train_x/train_y/train_row_ids, val_x/val_y/val_row_ids and features; verify their exact recorded hashes using pure audited functions. Cached Test members may be carried opaquely by the existing pickle container; do not inspect, hash, count, iterate, transfer to a model, predict or compute metrics on them. Existing recorded Test metadata remains recorded metadata, not a new verification claim. Do not call bundle.cell_stats/split_fingerprints/row_id_fingerprints or raw-data preparation. No unrelated mdmt_next restoration. Missing required cache/path/hash -> technical PAUSE with exact reason.

Numeric controls index is A3/attempt1/reused_reference_index.json, SHA256 `e8d67c6b18ad9ddbfa19b752e35c6ab051e3170a3d28df635b86b27922441fdb` (882651 bytes). It contains entries20, initial_hashes10, epoch_order_hashes500, selection_hash, environment, shared_operator_provenance, source_index_path/source_index_sha256, test_accessed=false. Each entry supplies dataset/arm/seed/key, result receipt, exact epochs50 with path/hash, epoch_receipts_sha256, available_fields, reused=true, and historical prefix metadata. Read only those explicit numeric files; verify all1020 result/epoch bytes against receipts before formal launch, without old IPC. Bind the index unchanged; a new N-local mapping may copy numeric bytes into `N/reused/` and record original and copied paths/hashes. Do not edit historical index or its source files.

Actual inspected schemas:

* CLEAN result: job,epochs,global_step,initial_hash,selected_epochs,selected_validation,selected_hashes,common_validation,final_hashes,optimizer_hash,counts,worker_seconds,peak_allocated_bytes,peak_reserved_bytes,test_accessed. Epoch: epoch,global_step,seconds,loss,clean_loss,auxiliary_loss,validation,selected_epochs,model_hash,ema_hash,optimizer_hash,epoch_order_hash.
* Constant NQR result additionally has costs/full-process/physical-attempt provenance. Epoch contains job,counts,source_hash,design_bundle_sha256,continuation_hash,selected_hashes,rng_hash,loader_hash,test_accessed,validation/validation_cost,objective_ledger,completed_loss,batch_log/hash/count and the common source model/order/optimizer fields.
* Actual estimator keys are `online` and `fixed_ema`; initial_hashes key is `dataset/seed`, epoch order key `dataset/seed/epoch`. Validate integer epoch identity and complete1..50 coverage; require final global_step1000 ML/2450 KR and counts optimizer_updates=ema_updates=that value.

Verify every control's recorded config/optimizer/source/shared-operator/environment and selection provenance using the index's explicit receipts, not new C4 config-byte equality. CLEAN's unused auxiliary fields/probe flag do not change its source-CLEAN path; validate active shared fields and the traced operator. Selection string is exactly `earliest strict maximum of validation six-cell Macro AUC; independent online and EMA`, selection hash `c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89`. Recompute selected epochs/validation/state binding from all50 numeric epochs, keeping earlier exact ties. Require recorded initialization identity and every epoch order hash to match new W/S. Do not demand W/S updated-state equality with C/N or inherit C3 prefix_identity_required_through_epoch. CLEAN epoch RNG/loader/counts/continuation/Test flags remain unavailable with reason `not_recorded_in_source`; result-level Test=false remains available. Do not fabricate or rerun controls to fill those fields. Missing required numeric bytes or mismatched provenance PAUSEs; it does not authorize extra control training.

### 7.2 Minimal runtime API and clocks

protocol exposes `config(dataset,arm)`, `job(dataset,arm,seed,source_hash,purpose='formal',case_id=None)`, `design_bundle()`, `source_manifest()`, `dataset_cache(dataset,cache_dir=DEFAULT_CACHE)`, `reference_index()`, `stop_requested()`, `request(attempt,command,**payload)`, `prepare(attempt,review_path)`, `validate_attempt(attempt,require_engineering=False)`, `validate_formal_clock(...)`, `validate_epoch_state(snapshot,record)`, `epoch_files(attempt,result)`, `verify_selection(result,epochs)`, `finite_checks()` and the imported pure numerical aliases above. Job key is digest of the fixed identity fields before key is added: cycle,dataset,arm,seed,config/config_hash,source_hash,design_bundle_sha256,controls_index_hash,purpose and case_id when engineering. Engineering keys must be disjoint from all20 formal keys. `request` rejects any attempt except canonical absolute N before reading any guardian file; all new write path helpers similarly enforce new-root ownership. No secret is returned in public reports.

runner exposes `admission(...)`, `restore(model,ema,opt,loader,state,device)`, `commit_snapshot(attempt,key,snapshot,record,path,...)`, `smoke_case(...)`, `smoke(...)`, `run(...,recovery=False)`, `recover(...)`, `handoff(...)`, `finite_checks()`. CLI module E.runner has `prepare`, `smoke`, `run` (with `--recover`), `handoff`, and internal supervised worker/case commands. No ensure_guardian or background AI command. tables exposes `export(attempt=N)` and `finite_checks()`; it extracts arithmetic only, does not select a winner or infer scientific conclusions.

Keep legacy-compatible `counts` **exactly** `{optimizer_updates,ema_updates}` in saved state, records and source-server commits. Each value is an int equal to current branch/global update count. Expanded counters live in a separate `operations` dict:

`training_rows,task_entries,ordinary_forwards,scalar_backwards,live_adam_steps,live_ema_steps,diagnostic_points,extra_autograd_sweeps,virtual_adam_steps,probe_forwards,view_forwards,algorithm_rng_draws`.

All are nonnegative integer cumulatives. Per ordinary successful step increments B,2B,1,1,1,1,diagnose,3*diagnose,4*diagnose,0,0,0. Source None private steps are recorded independently, not assumed equal to global_step. Validate them every returned step against actual grad activity and previous private step map; uninitialized + active becomes1, initialized + active increments1, inactive leaves existing state/step/decay unchanged.

Formal saved clock fields are `purpose='formal',clock_kind='formal',epoch,global_step,next_epoch,scientific_epochs_completed`. Initial epoch0/step0/next_epoch1. At batch b in epoch e, global_step=(e-1)K+b. Whole-epoch snapshot e has global_step=eK, scientific_epochs_completed=e,next_epoch=e+1 (51 after50), exact rows=e*n_train/task_entries=2*rows. Diagnostics cumulative `sum(1 for k in [1,10,30,50] if k<=e)`, with 3/4 extra/virtual multiples. All formal commits bind complete data/order including the tail. `PHASE_COUNTERS` means constant-objective operation accounting, never fictitious early/late/on/off fields; branch counts can be recorded as ordinary/all_single_class/lambda_zero, with lambda_zero=0 formal. No off_updates/scheduled_active/schedule_epoch or obsolete temporal clock.

Engineering saved fields are `purpose='engineering_smoke',clock_kind='engineering',epoch=engineering_commit_ordinal,engineering_commit_ordinal,engineering_state_step,global_step=engineering_state_step,next_epoch=None,scientific_epochs_completed=0,sample_epoch=1,sample_batches=[1,K]`. Server epoch is only its monotonic storage ordinal: prefix commit1 at logical step2; restored suffix commit2 at logical step4. Physical calls total8 per case includes source/delegate forks and replay; those costs belong to physical operations, not branch logical operations. Engineering result status is ENGINEERING_PASS after review, never formal COMPLETE. The source server's complete command (requires epoch50) is forbidden on engineering keys.

### 7.3 Snapshot schema, exact replay identity and selection

RAM snapshot includes `job,source_hash,design_bundle_sha256,controls_index_hash,owner_endpoint_hash,model,ema,optimizer,rng,loader`, source modes, selected_models/selected_epochs/selected_validation,best,common_validation,initial_hash,initial_ema_hash,initial_optimizer_hash,clock fields,counts,operations,adam_step_counters,last_parameter_activity, and numeric costs/resources. At formal epoch boundaries no live permutation is needed; engineering prefix additionally holds sample row-index identities/permutation in RAM for exact repeated full/tail suffix. All tensors are immutable CPU copies; no live/selected/saved tensor aliases a mutable checkpoint. `cpu_copy` clones recursively. Restore opt with `opt.load_state_dict(cpu_copy(state['optimizer']))`; model/EMA use load_state_dict copies; loader gets copied saved state, and original Python/NumPy/CPU/CUDA RNG is restored **last**, after all constructors/device moves/state reconstruction. No source reset of populated moments.

Define `scientific_identity_hash=state_digest(identity)` for the stable ordered identity dict containing job/provenance, model,ema,optimizer,rng,loader, source modes, selected_models/epochs/validation,best,common_validation,initial identities, scientific clock, counts, logical operations excluding diagnostics-only work, private steps and last activity. Diagnose-on/off comparisons also compare actual gradients and None masks explicitly. For engineering suffix comparisons use the same branch logical state/commit ordinal and selection observation identity; physical attempt/call ordinals and diagnostic counters are excluded from that equality projection and compared to their separately expected values. `continuation_hash=state_digest(full_CPU_snapshot)` binds the exact actual transaction including its captured counters/costs; it is checked against that very returned owner payload, not a different replay's elapsed times. Record component hashes model_hash,ema_hash,optimizer_hash,rng_hash,loader_hash and selected_hashes for independent inspection. Stable insertion order is preserved; source state_digest is the same as historical hashes. Never broaden a scientific identity tolerance because receipt timing differs.

For each completed formal epoch evaluate full validation online then fixed_ema with source evaluate(batch_size40960), preserve modes and RNG, reject undefined required macro/cell AUC or nonfinite metrics. Select with strict `value>best[kind]`, no epsilon/tolerance/tiebreak by LL. Initially best=-1.0 and empty selections (finite JSON); selected model states are cloned CPU bytes of that state. common_validation records fixed epochs1/10/30/50. Both selected LL and every cell metric are tied to the same selected AUC state/hash/epoch. Retain no extra fixed-epoch model tensors after they have supplied numeric records. Source evaluator fields, counts and aggregation semantics are preserved.

### 7.4 Actual frozen storage API, ownership and commit chronology

After code PASS, parent alone starts foreground `serve(str(N))` imported from frozen TV runner using the explicit project Python/PYTHONDONTWRITEBYTECODE. Create N first; no ensure_guardian, detached process, replacement driver, old-owner call or global-role edit. Parent records its new tool session/PID/server-file SHA/source/seven-design and endpoint hash in N/STORAGE_OWNER.json. Actual `guardian.json` fields are `address,authkey,pid`, mode0600. Exclude guardian.json, authkey/address/socket and connection material from public reports/archives; a local owner binding may contain the endpoint, public handoff uses only its hash/PID/session. Wrapper verifies canonical N, owner/source/design/server hash and live owner identity before requests. No credentials are printed.

Frozen `serve` actual command/reply contract:

| Command | Request fields beyond command | Required reply/effect |
|---|---|---|
| status | none | pid,retained_states,retained_bytes,jobs (no tensor state) |
| register | key,job,pid | `{attempt:int,state:RAM snapshot or None}`; identical job; REGISTERED/PAUSED and no registered live worker; attempts increments without cap |
| allocated | key,peak_memory_bytes | `{status:'ALLOCATED'}`; only after actual first FULL Adam allocation |
| commit | key,state,epoch_file | `{status:'RAM_COMMITTED',epoch:e}`; state.epoch exactly previous+1; server writes its N journal |
| get | key | `{state,result}`; result may be None before completion |
| paused | key,reason | `{status:'PAUSED'}`; clears worker_pid |
| failed | key,failure | `{status:'FAILED'}`; clears worker_pid |
| recover | key | RECOVERY_READY plus key,previous_status,epoch,attempts,time,unresolved,test_accessed; only FAILED/dead RUNNING, no live workers; preserves ACK state |
| complete | key,result | COMPLETE; only epoch50; retains **exactly** selected_models,selected_epochs,selected_validation |
| release | key | RELEASED; permitted only temporary new engineering keys after acceptance |

`roundtrip(value=...)` exists but does not establish commitment; engineering must use real commit/get. Frozen server transports via CPU torch.save(BytesIO)/send_bytes and torch.load(BytesIO,map_location=cpu,weights_only=False). No tensor payload is written to disk. get has no epoch/hash/ACK fields: inspect state and separately own status/journal. Source server does not itself validate all C4 state semantics; new wrapper must validate them before commit and after get. It does not gain unimplemented commands, job fields, hash returns or fake failure injection.

Formal transaction order: finish all K batches -> both validation streams -> strict selection -> CPU immutable snapshot -> validate record/snapshot -> write numeric pending epoch record and bind batch-log SHA -> BytesIO serialization -> request commit -> receive real RAM_COMMITTED ACK -> status/get -> check returned epoch/counts/job/full continuation and component/selected hashes -> reconcile exact own server journal -> write numeric ACK/committed epoch acceptance. Preserve monotonic commit1..50 per formal key; pending JSON alone is never a committed epoch. Numeric ACK fields: key,attempt,epoch,epoch_file,epoch_file_sha256,batch_log_sha256,source_hash,design_bundle_sha256,controls_index_hash,owner_endpoint_hash,continuation_hash,scientific_identity_hash,ram_commit_status,ram_get_verified,test_accessed=false, plus serialization/get/copy bytes/time and chronological event timestamps. The actual owner journal row fields are key,epoch,attempt,epoch_file,time,status='RAM_COMMITTED'; bind that exact row rather than expecting C4 hashes inside it.

On lost register/commit/complete reply, reconcile only N status/get/journal. An owner epoch equal to the candidate epoch with exact hash/job/counts and journal permits reconstructing the local ACK; previous epoch means the candidate was uncommitted, so PAUSE/recover/replay as appropriate. Conflicting epoch/payload is technical PAUSE; do not blindly resend or fabricate ACK chronology. Preserve transport_warning_reconciled as evidence, not automatic fatal failure after successful reconciliation. Use source complete only after all50 ACKs/selected provenance check; verify its compact retained state and result by get afterward.

Old RAM owners are TV PID1025015, C1 PID1570727/session59365, C2 PID1944719/session52263, C3 PID2097722/session61982. No old IPC/status/key queries/release/restart or journal writes. Record their protected identities from existing receipts without contacting them.

### 7.5 Four real engineering cases and measured calibration

Only a fresh DEFAULT native gpt-6-luna/high RUN-only child, dispatched by parent after code PASS and new owner startup, executes these actual cases. Seed2022, FP32 source model/data/Adam/EMA:

| GPU | Dataset | Arm |
|---:|---|---|
|0|movielens|COMPLETION-BCE-025|
|0|kuairand|MASS-SCALE-025|
|1|movielens|MASS-SCALE-025|
|1|kuairand|COMPLETION-BCE-025|

These assignments cover engineering only. Within each case draw the source seed+11000 epoch1 permutation once; take its true first40960 rows and true final tail, with exact original IDs. Save that permutation/loader identity in RAM. Physical call ordinals and actions:

1. Ordinary arm first full batch, diagnostics **on**, lazy Adam; allocated ACK only after its actual full backward/live Adam/EMA allocation and measured diagnostic clone phases. Logical state_step1.
2. Ordinary arm real tail, diagnostics off. Save immutable populated **prefix** at state_step2. Commit it to the new engineering key with storage ordinal1; real ACK -> get/hash -> numeric ACK. Save a separate CPU prefix copy and its identity hash.
3. Temporary source-CLEAN full step from the identical populated prefix (including restored RNG/loader), no diagnostic. Record model/Adam/EMA/RNG/loader/gradient/activity hashes, then discard that temporary route.
4. Temporary lambda0 source delegate full step from that same populated prefix, no diagnostic. Require exact equality to call3, including tensor-zero/None/private-state semantics. Discard temporary route; prefix hash unchanged.
5. Original ordinary-arm suffix full batch from the unmodified prefix, diagnostics off. Logical state_step3.
6. Original ordinary-arm suffix real tail, diagnostics **on** at populated Adam. Logical state_step4. Full online/EMA validation and engineering selection bookkeeping; retain expected suffix scientific identity in CPU RAM, not the server. This full/tail diagnostic placement implements the user's explicit diagnosed-full-and-tail requirement; it changes no research intervention, diagnostic count or physical-call budget from P4. The first full diagnoses lazy state; this tail diagnoses populated state.
7. Pause new engineering key, re-register that same key under actual worker PID (attempt2), get prefix ordinal1, verify it matches the saved prefix. Serialize/deserialize via BytesIO and restore with independent optimizer tensors/RNG-last. Replay ordinary-arm suffix full with diagnostics off, logical state_step3.
8. Replay its real tail with diagnostics off, logical state_step4; full online/EMA validation/identical engineering selection bookkeeping. Require exact scientific identity, model/Adam/EMA/RNG/loader/selected hashes and validation equality to call6. Assert saved prefix and its optimizer bytes were not mutated. Commit restored suffix to engineering storage ordinal2 -> real ACK -> get/hash -> numeric ACK. Mark paused ENGINEERING_VERIFIED; retain temporary key until reviewer/parent accepts it.

Engineering selection uses observation ordinal2 after the suffix, explicitly `selection_clock='engineering_commit_ordinal'`; selected_epochs stores ordinal2, never a scientific epoch claim. Prefix selections are empty; expected/restored suffix compare the same observation. No pretend epoch50, formal result, Test evaluation or efficacy gate. Two full validation passes of both streams per case imply80 total engineering validation chunks (ML24,KR56); any additional acceptance evaluation is counted separately. Exactly8 physical train calls/case,32 total; two diagnostic points/case, eight total,24 extra autograd sweeps and32 virtual Adam steps. Actual branch logical steps finish at4; engineering scientific_epochs_completed=0. Physical instrumentation must not miscount virtual steps or source/delegate forks as live formal updates.

Resource profiling must begin at process/context initialization and cover all these real phases: staged allocation, context, lazy Adam, full/tail forward/backward/live step, diagnostic sequential clone peak, populated diagnostics, online/EMA validation, CPU-copy/serialization, real owner get, restore and replay. Measure own-PID total CUDA residency via NVML including context/non-torch usage, allocator current/peak allocated/reserved separately, actual CUDA-usable capacity, host RAM fraction/RSS, timing and phase boundaries. Sample during transient diagnostic allocations and at their barriers, not merely after clone deletion; compare full-process peak and reserved+measured non-torch overhead. Profiles are dataset/arm/device/source/design/hardware bound and grow to maxima actually observed.

The four-case matrix observes both arms for each dataset and each dataset on each card. For formal placement on either card, use a **measured dataset envelope**, not an invented arm/device measurement: require a real case for that dataset on the target card, both arm cases present, and compatible measured hardware (GPU model/compute capability/CUDA-usable capacity, environment/source/design). Take the maximum of `full_process_peak_bytes` and `torch_reserved_peak_bytes+non_torch_bytes` over those compatible actual C4 profiles, including the target-card context and both arms' diagnostic peaks. Record every contributing profile's original arm/card/hash and that the admission basis is this measured envelope; do not relabel one as an observation of the other combination. Both arms' diagnostics evaluate the same four gradients and virtual optimizers, and this envelope uses all measured branches without extrapolating a new numeric peak. Grow it with compatible live observations; never shrink it to admit a worker. If hardware compatibility or target-card dataset coverage is absent, PAUSE for the actual resource issue before formal RUN. Do not add unregistered engineering cases or pin formal arms to avoid the missing evidence.

Cold calibration is an engineering allocation, not a formal worker admission. Start only on a card satisfying fresh current memory/power/host gates. Reserve pending startup on that card; measure actual context and staged actual full/tail/Adam/diagnostic allocations against available CUDA-usable capacity before continuing. No assumed C3 peak or guessed multiplier is a C4 profile. No other admission on that card before its full calibration ACK/profile. If staged calibration cannot proceed safely/finish, PAUSE with actual failed stage/resource evidence; do not infer capacity or change batch size. After profiles exist, all starts use measured physical fit below. A fresh independent resource/engineering reviewer must sign actual four-case receipts, chronology, diagnostics/replay, measured profiles and accounting before formal run.

### 7.6 Admission, supervised RUN and recovery

Before each parent dispatch/worker admission read BOTH W and O `artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json`, each root STOP_AUTONOMOUS and artifact STOP_AUTONOMOUS. A manual flag, USER_STOPPED status/reason or sentinel latches user stop. A technical PAUSE blocks dependent starts until corrected/reviewed. Do not edit rootstate in new package code or confuse an old retired driver's historical status with a new instruction.

Per GPU independently require CURRENT measured `used/NVML_total < .85` AND `power/enforced_limit < .85`, plus CURRENT host `(MemTotal-MemAvailable)/MemTotal < .90`. Additionally, because actual prior OOM established a capacity mismatch, require `current_device_used_bytes + incoming_measured_full_process_bytes <= current_CUDA_usable_total_bytes`. This second test is usable capacity100%, not projected85%. Re-measure current CUDA capacity; old25307578368 is context only. Incoming residency uses the matching measured C4 full-process profile, including diagnostic clone peaks and measured context overhead. If the incoming process already allocated a phase, subtract its already-counted residency when checking its remaining allocation; do not count it twice.

No forecast worker power, projected85%-memory, projected90%-RAM, fixed worker cap, both-card gate or artificial formal card pinning. Protect a pending first allocation only until its real first FULL lazy-Adam ACK; update profile and freshly resample current memory/power/RAM before another admission on that card. A calibration-in-progress may hold that card until all its new phases are measured. Allow the other card independently. Record a numeric admission decision with measurement times, profile path/hash, current fractions, usable bytes, pending ACK, physical-fit arithmetic, rejection reason and selected queue index.

Formal queue: KuaiRand first, then MovieLens; seed ascending2022..2026; W then S. Choose first physically fitting eligible job for each independently eligible GPU. If an earlier KR job cannot physically fit but later ML can, launch first fitting ML and retain the skipped reason/order. Do not idle an eligible card because another card fails a gate. Worker subprocesses are mechanical, foreground-supervised by the active current-conversation runner with retained process handles; no detached/headless AI/session, tmux/nohup/systemd or autonomous.py lifecycle call. A mechanical resource sampling thread is not an AI session.

On actual technical failure or CUDA OOM: stop affected admissions, finish/safely preserve atomic parameter/EMA work, retain last acknowledged whole-epoch RAM state and every physical attempt/cost; PAUSE for a fresh source/design-bound recovery review under parent control. Correct only the evidenced issue; re-review/check affected contracts. Parent then automatically dispatches a fresh DEFAULT Luna/high runner using this new package's `run --recover`, without user prompt or hidden retry cap. Recovery review binds source/design/owner/failure/last ACK and verifies the new physical profile when applicable. No batch/precision/seed/epoch/model/optimizer change is a routine recovery fix.

Recover N keys only. Verify no registered workers remain live before frozen server recover; reconcile each RUNNING/dead or FAILED state using actual journal/get/hash. PAUSED keys re-register directly. Epoch0 with no ACK/state may restart its original exact initialization; a nonzero acknowledged state must exist and match its exact ACK. Restore whole-epoch state/RNG/loader, replay every uncommitted step from original order, preserve wasted/replayed work and distinct physical attempt numbers. Epoch50 can reconcile completion from acknowledged state without training again. Completed compacted keys must not be re-registered or rerun. Missing live acknowledged state is reported as technical loss, never replaced by fabricated tensors/checkpoints. Do not invoke old autonomous.py recover.

Explicit USER_STOPPED halts admissions and automatic recovery/outer continuation; finish/checkpoint current atomic step safely. A fully completed epoch can be validated/ACKed; partial epochs stay explicitly uncommitted, with available RAM/cost record and no fake full-epoch ACK. Parent writes final summary/freeze. Test remains sealed through ACTIVE, PAUSED and every candidate outcome. Opening requires explicit USER_STOPPED + final freeze + existing eligible full-five-seed selected-live-RAM verification gate. No champion/eligibility exists merely from this design or storage acceptance.

### 7.7 Required budget and retained selected state

20 new trajectories =2 arms*2 datasets*5 seeds;20 reused numeric controls. New scientific epochs1000. ML new10000 and KR24500 updates =34500 each ordinary forwards, scalar backwards, live Adam and live EMA updates. New rows1403149500 and task entries2806299000. Diagnostic points80, additional autograd sweeps240, disposable real-Adam steps320. Validation epoch chunks10000: ML3000,KR7000 for both streams. Initialization, engineering, probes, copies/get/restore, failures/uncommitted/replays and selected acceptance are additional measured physical work. These totals are not an observed runtime forecast.

Retain all20 new selected online/EMA pairs in the new parent-owned server,40 model states, including S and failed candidates, through parent direct get/hash/chosen-epoch/validation acceptance and subsequent retention policy. Expected source unchanged tensor bytes191017920 is a consistency check, verified from actual tensors. Parent writes N/PARENT_STORAGE_ACCEPTANCE.json before closing formal runner; per pair bind job/source/design/owner, epochs, selected hashes/validation, get verification and any evaluator recheck. Perform real source re-evaluation when required by the existing selected acceptance gate and count it. Source complete compaction removes final/optimizer/RNG duplicates only after ACK50; selected pairs remain. Candidate failure never triggers release. No disk model/Adam/EMA/RNG/loader/selected/checkpoint tensors or tensor-valued JSON/base64 payloads; only CPU RAM/BytesIO. New engineering keys may be released after reviewer/parent acceptance only. Old owners never receive release.

## 8. Ablations and experimental comparisons required

No extra formal arm. For each dataset and each endpoint selected/E1/E10/E30/E50, compute these contrasts for **both** source streams online and fixed_ema: W-C, S-C, W-S, N-W, N-C. Add total-package W/fixed_ema-C/online and within-arm fixed_ema-online for each W,S,N,C. These are15 comparisons per dataset/endpoint. Selected endpoints compare independently selected states on each side; fixed endpoints compare the same epoch. Report each side's epoch/hash, including within-arm selected comparisons. The primary is only **selected W/fixed_ema minus selected C/fixed_ema**, separately for ML and KR.

All four arms retain their numeric50-epoch online/EMA curves. No C3 E40 endpoint, prefix-equality or timing exposure requirement carries forward. Lambda0, all-single-class, diagnostics on/off, source versus delegate, restored suffix and tiny enumerated-pair checks are engineering identities, not efficacy arms or seed replicates. S+R, W/m, inverse weights, lambda/schedule sweeps, noise/projection/momentum and alternative models remain outside this budget.

## 9. Expected logging requirements and numeric schemas

### 9.1 Raw records and aggregate definitions

Use source-safe finite JSON (allow_nan=False); undefined numeric CSV values are empty plus explicit null_reason, never string `nan` or imputed zero. Record version and source/design/job/physical-attempt/purpose/clock provenance in each raw family. Keep N/jobs, runs/<key>/attempt<number>, smoke/<case>/attempt<number>, tables, own ACK/journal/recovery/admission/allocation/cost logs. Preserve every failed/uncommitted log; extraction accepts scientific epochs only through verified N server ACKs. Attempt's pending record/old duplicate does not supersede a different committed epoch by lexicographic order.

Batch records include context, B, real row_order_hash, branch/lambda, actual/core loss values, support ledger, operations and counts increments/cumulatives, private steps/activity, diagnostic result or null reason, seconds/resource phase and test_accessed=false. Bind each complete batch log's hash/count in its epoch record. Epoch records add validation/validation_cost, strict selected epochs/hashes, common_validation, source/model/Adam/EMA/RNG/loader/continuation/scientific identity hashes, actual epoch_order_hash, objective_ledger summaries, clock/counters, source/design/control/environment/owner binding, and ACK linkage. Result includes all50 completion/selection/order references, final hashes and numeric costs/physical attempts; no tensors.

Aggregate mass fields with `update_sum=sum_b M_b`, `update_mean=update_sum/K`, `row_weighted_sum=sum_b B_b*M_b`, `row_weighted_mean=row_weighted_sum/sum_b B_b`; preserve denominator units and actual number of batches/rows. The natural tail gets its real B weight. Cell epoch n/n0/n1 sums are exact natural counts. Do not calculate an epoch mass from class proportions aggregated across batches and present it as actual batch mass.

Diagnostic class statistics have `observed_batches,null_batches,sum,count_weighted_sum,observed_count,update_mean,count_weighted_mean,denominator_units`. Use n0 for mu0/var0/bbar0, n1 for class1, n for H/Q_R; skip undefined values with explicit reasons and zero observed_count -> null means. Only scheduled observations contribute, never every batch's n when no statistic was measured. Count-weighted variances are averages of within-batch population variances; they are not a pooled class variance. No running pooled second-moment computation is required. Where Q_R needs both classes, its denominator is n for observed mixed cells only. Average within seed before aggregating five seeds; do not count batches/groups as independent replicates.

### 9.2 CSV contracts

Export under N/tables; parent may copy required views to R preserving hashes. Keep these exact names and semantic columns:

| File | Required columns/coverage |
|---|---|
| ENDPOINTS.csv | key,dataset,arm,seed,source_hash,design_bundle_sha256,reused,test_accessed,estimator,endpoint,epoch,selected_epoch,state_hash,epoch_order_hash; source metric fields below; availability/null_reason |
| ENDPOINT_CELLS.csv | endpoint provenance above plus cell,auc,logloss,count,positive,negative |
| CURVES.csv / CELL_CURVES.csv | same provenance with all epochs1..50, actual state_hash/validation; both streams/four arms |
| ALL_PAIRED_SEEDS.csv | dataset,comparison,left_arm,left_estimator,right_arm,right_estimator,endpoint,metric,primary_metric,seed,left_key,right_key,left_epoch,right_epoch,left_state_hash,right_state_hash,left,right,delta,available,null_reason,favorable_direction,favorable |
| ALL_PAIRED_SUMMARY.csv | dataset,comparison,left_estimator,right_estimator,endpoint,metric,primary_metric,available,n,mean,sd,ci95_low,ci95_high,positive,negative,zero,favorable,unfavorable,null_reason |
| WEIGHTS.csv | one row per new committed batch/cell; key/attempt/context/row hash/B/entries/arm/branch/lambda/m/r, actual and completion task/total/cell masses, counts/pi/w/opportunity/q/coefficients, statistic values/status reasons |
| MASS_SUMMARIES.csv | key,dataset,arm,seed,epoch,scope(total/task/cell),scope_id,mass_kind,update_sum,update_mean,row_weighted_sum,row_weighted_mean,batches,rows,denominator_units; actual and removed/completion/scale quantities distinguished |
| STATISTICS.csv / STATISTIC_MEANS.csv | raw per-seed/epoch/cell/statistic counts/sums/weight denominators above; means view records statistic/aggregation/valid_seed_count/n_seed5 and mean/SD across seed-level values; no null imputation |
| COMMON_STATE_DIAGNOSTICS.csv | key,dataset,host_arm,seed,epoch,batch,global_step,row_order_hash,pre_model_hash,pre_optimizer_hash,pre_rng_hash,pre_loader_hash,parameter_group,objective_or_pair,metric,value,defined,null_reason,coordinates,parameter_names,activity_counts,source_hash,design_bundle_sha256,extra_autograd_sweeps,virtual_adam_steps; counterfactual_at_host_state=true |
| DIAGNOSTIC_MEANS.csv | dataset,host_arm,epoch,parameter_group,objective_or_pair,metric,seed_aggregation,valid_seed_count,mean,sd,min,max,available,null_reason; retain five individual seed values in raw table |
| DIAGNOSTIC_AVAILABILITY.csv | dataset,arm,seed,epoch,metric_family,available,null_reason,source_path/source_sha256; explicitly distinguish common-state new diagnostics from historical on-trajectory norms |
| SELECTION_STRUCTURE.csv | key,dataset,arm,seed,estimator,selected_epoch,selected_hash,selected_macro_auc,selected_macro_logloss,selection_hash,first_strict_max_verified,reused |
| PHASE_COUNTERS.csv | new formal committed epoch key/dataset/arm/seed/clock/epoch/global_step, counts, operations, rows/task_entries, branch counts, private-step/activity maps, actual lambda, next_epoch,scientific_epochs_completed; constant objectives only |
| PHYSICAL_ATTEMPTS.csv / COSTS.csv / SMOKE_COSTS.csv | every physical attempt/status/pid/device/source/design/resume ACK/time/seconds semantics, returned/unknown operation counts, measured phase/validation/diagnostic/copy/serialization/get/replay costs/bytes and resource peaks; logical work labelled separately |
| RESOURCE_EXECUTION_AUDIT.json | actual devices/profile/admission/allocation ACK sources and hashes; queue decisions/current gates/physical fit, four engineering signoffs, all20*50 formal ACK completeness, physical attempts/replays/uncommitted work, owner/selected acceptance, logical/physical budget reconciliation, Test=false |

Source metrics are exactly six `{d}t{t}_auc/logloss/count/positive/negative`, macro_auc/macro_logloss, sample_weighted_cell_auc/logloss, domain0..2_macro_auc/logloss, task0..1_macro_auc/logloss, task0..1_pooled_auc/logloss, pooled_task_mean_auc,pooled_logloss,worst_cell_auc. Preserve original source names `d0t0_...` etc. No newly invented flattened-label pooled AUC. Endpoint400 rows =40 trajectories*2streams*5endpoints; endpoint cells2400; curves4000/cell curves24000. New batch weights207000 rows=34500*6; PHASE_COUNTERS1000 rows. Diagnostic group/metric rows are not diagnostic-point counts or samples. Missing historical counts/diagnostics do not erase available metrics. Historical NQR statistics can be separately linked from exact numeric receipts; do not claim they are common-state counterfactual diagnostics.

### 9.3 Statistics, costs and reporting

Paired seed is the unit within dataset, n=5. For each metric/contrast print all five deltas, mean, sample SD(ddof1), descriptive95% interval `mean +/- 2.7764451051977987*sd/sqrt(5)`, positive/negative/exact-zero and favorable/unfavorable counts. Positive AUC is favorable; negative LL is favorable. Null rows do not enter arithmetic, and incomplete primary n cannot qualify. Repeated validation adaptation/shared populations/seeds/history remain exploratory and correlated; no multiplicity-adjusted discovery, equivalence or independent-replication claim from zero-crossing intervals. No post-hoc equivalence margin.

Log actual summed worker seconds and foreground wall time separately; GPU/worker time by actual attempt, device residency including context, allocator allocation/reservation, host RSS/fraction, real concurrency/current admission records. Separate live work, diagnostic autograd/virtual Adam, evaluation, CPU-copy/BytesIO/get/restore, engineering, parent acceptance, failed/uncommitted/replayed work. Snapshot-capture costs and that snapshot's subsequent transfer cost are separate, linked by ACK; do not put the transfer's own unknown duration into the already-hashed snapshot. Lost/killed partial operations are null with their observation/upper-bound semantics, not zero cost. C3's worker-hours/foreground duration are historical observations, not C4 predictions. Costs outside last-ACK logical state remain in physical attempts and cannot vanish on restore.

Parent's complete report must cover all per-seed/per-domain/per-task AUC/LL, baseline performance, label prevalence, domain size versus difficulty, model, selection and fixed training stages; stable repeated facts, contradictions, competing explanations, evidence STRONG/MEDIUM/WEAK, size plausibility, small-domain/large-domain trade-off and the next discriminating experiment. Context must retain C3: ML primary -0.000340045 with5/5 adverse AUC; KR -0.000068613 with3positive/2negative and wide intervals; C3/C2 E10 exact prefix is correlated history, LATE ML selected identity is a selection bound, C2 KR large-domain LL improvement counters universal large-domain harm. These motivate the comparison, not its result.

## 10. Acceptance criteria for reviewer

### 10.1 Finite CPU/basic implementation checks

Coder reports and independent reviewer must provide actual evidence for this finite set; no experiment or GPU/IPC use by coder is needed:

1. Tiny FP64/FP32 mixed cells including different task/class prevalence: row versus ordered-cell W, S=mC, m/r bounds, actual task/cell/total coefficient sums, N=W+explicit pair-risk R and probability/parameter gradients against pure frozen C2 using the preregistered tolerances. Check one mixed cell with an ordinary single-class other cell distinguishes W coefficient1/(2B) from global-scaled S. Include real-form cardinality/full/tail fixture checks.
2. Missing domain/absent class/all-single-class/lambda0 and invalid input branches: exact null/count/mass behavior, no mutation before invalid config/support rejection, direct source-clean state/gradient/activity identity for both arms. All-single and lambda0 are actual delegates. A zero-data-gradient parameter and genuinely unused parameter distinguish tensor_zero from None in both lazy and populated source Adam with coupled WD; test steps/decay/moments, not merely final loss.
3. Diagnostic on/off exact identity at lazy and populated states for both arms, ordinary and identity branch; three extra grads/four real optimizer steps, one live forward/backward/Adam/EMA, no RNG/loader change, virtual actual-arm exact poststate, frozen C2 pure read-only N, global/prefix vector semantics including disjoint activity and zero-norm null. Saved prestate/live/clones deeply isolated, one clone resident at a time. Source deepcopy constructors do not reseed/draw RNG.
4. CPU BytesIO roundtrip and populated-prefix restore/suffix identity, saved prefix immutability and `data_ptr`/storage isolation of optimizer step/exp_avg/exp_avg_sq where mutable. Loader next-permutation continuation identity; source EMA parameter/buffer identity. These are finite source fixtures, not historical fields invented for CLEAN.
5. Strict selection synthetic sequence with ties: earliest equal maximum retained, independent streams, LL/hash bound to selected AUC state. Parse actual one CLEAN and one NQR result/epoch from the explicit index, preserve their real schema differences/null availability; verify required1020 files in the subsequent preparation gate, once per final frozen input.
6. New roots/design seven-file mapping/source closure/config/job checks. Storage fake transcripts model the actual serve replies above: commit ordinal, ACK->get/hash->numeric chronology, lost-reply reconciliation, no epoch50 engineering complete, no skipped uncommitted epoch, two-key counts compatibility and source/design/owner mismatch rejection. These test consumer protocol only; actual real-owner evidence remains the engineering gate. Admission fixtures show current85/85/host90 plus usable100 physical capacity, per-card independence, first-fitting queue, pending ACK release/resample and no projected thresholds/caps.

Run only syntax/basic checks and these finite behavior fixtures required by this change. No broad hardening, HPO, efficacy pilot, GPU query, training launch, Test read, old IPC or frozen-source bytecode writes. Reports list actual commands/counts/assertions and source/design hash, not invented PASS results. A missing integration file can await the other coder's report, but all final required checks must pass before reviewer acceptance.

### 10.2 Independent code and real engineering gates

Code reviewer is a fresh identity distinct from both coders. It independently reads this design, all actual new code, imported frozen paths and combined coder report. Verdict PASS, or PASS WITH FIXES with every blocking issue corrected and verified, is required. FAIL returns to fresh coder/review. Reviewer verifies the exact scientific protocol, production never applying N gradients, diagnostic observational isolation, owner API, clocks/selection/recovery, explicit freeze and disk/old-owner boundaries. Review source/design hashes must equal final manifest; historical C3 PASS does not certify C4.

After code PASS, parent opens new foreground owner and verifies actual fresh DEFAULT gpt-6-luna/high metadata/fork_context=false for the engineering runner. The old named luna-runner role hardcodes four seeds and is forbidden. Engineering receipts must demonstrate all four assigned dataset/arm/card cases, actual full/tails, lazy+populated diagnostics,32 calls/24 extra sweeps/32 virtual steps, source-vs-lambda0 from populated prefix, two real engineering commits/case, real pause/re-register/get/deep-isolated restore, exact neutral suffix/full validation, immutable saved prefix, scientific_epochs0 and measured complete residency phase profiles. A separate fresh resource reviewer accepts actual receipts before formal admission; a written intended profile is insufficient.

Fresh DEFAULT Luna/high formal runner then executes the frozen20 jobs, all five seeds/50epochs, in the supervised current conversation. Runner records raw evidence only. Completion acceptance requires1000 whole-epoch verified ACKs,34500 useful Adam/EMA steps and all logical/diagnostic/validation totals reconciled, exact50 recorded order identities per trajectory, complete20 reused controls/1020 files, full endpoint/selection provenance, all physical costs and parent selected-RAM get/hash/validation acceptance. An actual technical failure produces reviewed recovery, not an efficacy conclusion or a reduced budget.

### 10.3 Scientific decision gates and next handoff

Evaluate W selected fixed EMA minus CLEAN selected fixed EMA separately for ML/KR. Both primary mean AUC deltas<=0 -> NO_CONTINUATION_SIGNAL for W alone. Exactly one positive -> MIXED. Both positive but failing practical/cell requirements -> POSITIVE_SIGNAL_BELOW_QUALIFICATION. Practical AUC qualification requires each dataset mean>=+.003 and all six per-cell mean deltas>=-.001. Preserve primary signs and separate booleans for macro target and cell protection even when category names overlap; no control, online/pooled/LL or favorable epoch rescues the primary. LL decrease is a separate per-dataset mean/cell/seed axis at those exact AUC-selected states. Any existing further eligibility gate remains in force; qualification here does not open Test or automatically select a champion.

Evidence labels: STRONG for audited exact identities/numerical repeated facts narrowly stated; MEDIUM for coherent but conditional protocol-local support; WEAK for wide/crossing intervals, contradictory seeds, local associations or extrapolation. Do not presume curriculum/momentum efficacy, M3oE failure on difficult domains, or small=difficult. Do not label size a main cause from six correlated cells alone.

After parent ANALYZE: R/REPORT.md, DECISION.json, NUMERICAL_APPENDIX.md, DATA_AVAILABILITY.md, paired/diagnostic/statistic/resource evidence and numeric-only reproducible archive; substantive EXP_AI_CONTEXT, MDMT_ALGORITHM_METRICS_LEDGER and five main Experimental_* context updates. Parent obtains required independent evidence review and actually publishes reviewed code/design/report/context to current remote main with expected-head protection, records verified commit/tree, then dispatches fresh Astra C05. Preserve unrelated concurrent work. Secrets/tensors stay out of publication. No candidate NO-GO/MIXED/NO_CONTINUATION_SIGNAL or Astra STOP terminates the outer loop. max_cycles/max_astra_cycles/max_experiments/max_revisions=null; technical failures PAUSE with recoverable action; only explicit USER_STOPPED normally terminates. This designer completes only the DESIGN stage and returns control permanently to parent.

## 11. Conditions that would falsify the hypotheses

* W retaining a repeated adverse ML W-C selected EMA AUC with ranking coefficient/gradient identically absent falsifies the claim that ranking gradients are necessary for that deficit. Mixed/uncertain W-C yields unresolved necessity evidence, not a forced rejection or confirmation.
* W-S improving while S-C is adverse, or W-S remaining uncertain with broad intervals, weakens/fails to support heterogeneous weighting as the principal adverse explanation. A precise adverse W-S with a comparatively small S-C supports only the conditional combined-weighting claim; no class-only explanation follows.
* W-C and S-C separating coherently in opposite directions weakens global scaling as a sufficient explanation. Similar means with uncertain W-S leave scaling plausible without establishing it as sole cause. No unregistered equivalence threshold is introduced.
* Removing R recovering N's deficit but leaving W below CLEAN supports a conditional ranking-addition cost while rejecting candidate efficacy. N-W signs or local gradient cosines cannot alone establish optimizer-history causality.
* Failure of either dataset's+.003 primary practical target or any registered cell floor rejects practical qualification; any missing full seed/epoch/source/ACK/selected-state evidence blocks evaluation rather than counting as a scientific negative.
* Contrary dataset, task, domain, online/EMA or fixed-versus-selected results must be retained. A small-domain gain/large-domain decline pattern absent or contradicted across cells/seeds fails the corresponding explanatory claim. Natural size, prevalence and baseline performance remain distinct measured quantities.

The most discriminating next experiment is chosen by fresh Astra from actual C4 evidence after publication, for example the missing S+R interaction if central, a justified scale/regularization intervention if scale remains competitive, or a distinct zero-mean-noise question if the deterministic-risk explanations fail. None of those future runs is authorized by this specification.

**Formal DESIGN COMPLETE — 0 blockers. Parent acceptance and seven-file freeze are the next actions.**
