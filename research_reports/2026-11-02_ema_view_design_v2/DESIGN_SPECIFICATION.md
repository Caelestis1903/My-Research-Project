# DESIGN SPECIFICATION — EMA02V2

Design completed 2026-11-02. Implementation may begin after orchestrator approval. Authority: `start.md` and `research_plans/26-11-2_v2/` (00–07, YAML). This document resolves implementation details within that contract; formulas and gates retain their planned values.

## 1. Responsibility and boundaries

Implement J0/J1/J_SUM, F0/F1/FR/F_PR and the conditional TG/RG/AG family against fixed1 A0/A1, complete the legal staged queue, and preserve raw numerical evidence for scientific analysis and publication. All new executable files belong to `experiments/ema_view_design_20261102_v2/`; reports to `research_reports/2026-11-02_ema_view_design_v2/`; scalar runtime artifacts to an isolated attempt directory. Original CSVs, historical source, plans and reports remain source evidence.

The specific view formulas in plan03 authorize the representation augmentations in this study. The general natural-data rule in start.md applies to the observed dataset population, labels, row ordering and proportions: retain natural original rows, without synthetic replacement data, resampling or label changes.

Only seeds 2022–2026 are legal. Development uses 2022/2024, confirmation 2023/2025/2026, W1 2023/2025. No disk model, optimizer, EMA, RNG, loader state, tensor checkpoint, model pickle or disguised state dump. Source code, scalar numerical records, configuration, hashes, logs and reports are allowed.

Use the explicit project Python interpreter. Do not terminate existing external GPU processes. Batch40960, FP32, 50 epochs, no AMP/TF32, no early stopping; change concurrency rather than the scientific configuration.

## 2. Source closure and affected components

Source release identity is GATS `e59efa2fd954c4d2231b679a742406cfbe2a1077`; execution identities and release identities remain separate in SOURCE_RECEIPT. Main observed by parent is `fb74c40cad9c22ce3abdfb8348da7e98d726a61a`. Parent has verified 59 local source/report blobs against pinned releases. Main lacks the required recent execution packages.

Vendor the exact local/pinned bytes under `vendor/gradient_aug_20261031_v2/`:

- top level `__init__.py`, `objectives.py`, `views.py`, `protocol.py`;
- `vendor_ga31/`: `__init__.py`, `model.py`, `reference.py`, `data.py`, `audited_data.py`, `source_data.py`, `source_protocol.py`, `protocol.py`, `metrics.py`, `resolved_protocol.json`;
- `basic/{__init__,activation,features,initializers,layers}.py` and `utils/{__init__,data}.py` inside vendor_ga31.

Preserve these blobs; add the new vendor package initializer outside them. Their relative imports work under the new namespace. Do not call the old top-level `protocol.protocol()`, which resolves a historical plan path. Read dataset/model identities directly from the vendored resolved_protocol JSON, and construct this study's protocol from the current YAML. Do not invoke source_data.load_data(), which persists a prepared pickle.

Reuse exact `objectives.losses` for A0/A1, source `views.make_model`, the natural loaders and preprocessing helpers, source probability BCE semantics and source metric arithmetic. Reuse vendor_ga31.protocol setup/copy/RNG/BytesIO IPC/JSON/CSV helpers. Implement the short fixed EMA update and strict selected-update predicates in new protocol code following GAM's source helpers; record this helper adaptation and its source path/blob. GATS run/shadow/recovery and GAM/AIGM optimizer modules are evidence, not required runtime imports.

New responsibilities can be split between independent implementation owners without changing the contract:

| Area | Responsibility |
|---|---|
| input probe, view construction, objective, semantic checks | Correct mathematical intervention and exact source closed paths |
| protocol/data, guardian, worker, scheduler, stage0, evaluation | Source/data identity, RAM lifecycle, legal queue, cost and numerical records |
| analysis utilities | Scalar-only paired statistics and tables; orchestrator interprets conclusions |

Callers, fixtures and configs affected are confined to this new package, its tests, resolved study protocol and report/artifact consumers. Existing public training callers need no changes.

## 3. Data and fixed protocol

Original split is global raw-row 80/10/10 before KR tab filtering. Fit vocabularies/scaler only on train; OOV=0. Pass absolute relocated CSV paths explicitly. Verify raw SHA256, six arrays, row IDs/fingerprints, encoder/scaler against pinned identities. Original train totals: ML800167, domain counts155949/318599/325619; KR2006132, counts172782/1743898/89452. ML val/test100021 each; KR250767 each. Target MLd0/KRd2. Sparse perturbable fields6/33, embedding16. Route and dense fields remain unchanged.

Use fixed1, 3 domains, 2 tasks, four shared experts, fcn_dims=[512,256,256,64], fusion1, no learned fusion/AutoML/DARTS/DBO, no BN/Dropout. Parameter counts773470/1614254. Adam lr.01, betas(.9,.999), eps1e-8, coupled wd ML1e-4/KR1e-5, foreach/fused/amsgrad=False. One real optimizer step per original batch. Loader seed offsets follow the source11000/33000/44000; shuffled without replacement, tail retained, num_workers0.

Source identity hashing of all six arrays is allowed before final evaluation. Stage0 and queue decisions do not inspect test predictions, test AUC/LL or test-dependent diagnostics. Avoid source manifest_fields()'s automatic test cell-statistics call before the final evaluation; form the needed identity manifest and train/validation statistics explicitly.

W1 is implemented only as the fixed second window, using source preprocessing helpers in a new adapter: global raw rows [0,floor(.60N)), [.60N,.70N), [.70N,.80N), fit preprocessing anew on W1 train, then filter KR tabs. Do not derive W1 by slicing the already-encoded original bundle. Retain raw IDs, recompute window identities/counts, and retain fixed rho despite different step count. No window search or boundary adjustment.

## 4. Exact losses and gradients

For actual original batch size B, real=[B,2] unreduced source probability BCE, S=target rows:

`Lreal=real.sum()/(2B)`.

For a single J/F/A1 view, `L=Lreal+.5*(aug.sum()-real[S].sum())/(2B)`. Do not divide by target count, donor count or valid-view count. For TG/RG each view t contributes only its task t loss; subtract the corresponding clean loss once. AG shares one augmented forward for both tasks. Auxiliary loss0.

Pre-known disabled augmentation, lambda0, kappa0 or empty S goes directly through the exact source A0 callable without RNG/probe draws. An entirely zero perturbation discovered during construction retains the already-spent probe/RNG cost, skips the extra supervised augmented forward/backward, and uses exact source A0. A1 goes through the exact source A1 callable. To use the new split/row RNG contract, pass the frozen generated Gaussian via source `explicit_views`; source equivalence compares the same explicit view. This keeps the source formula and training gradient unchanged without claiming historical trajectory reuse. A1 noise remains per-field iid standard Gaussian scaled by detached source field RMS. No double augmentation hook.

All constructed perturbations are fully detached. Supervised training uses current `h+delta`, with the original gate detach preserved. Donor, teacher, radius, selection and gain construction receive no supervised gradient. Probe uses autograd.grad(create_graph=False), never parameter .backward(). Preserve existing parameter gradients as None; optimizer.zero_grad(set_to_none=True) before the step.

Observed labels must be finite binary, complete as expected. If source data actually contains missing labels, first resolve a loader mismatch. If authentic missingness is established, report the protocol difference and apply plan03's observed-task masks with original2B denominator; never manufacture negatives.

## 5. Read-only complete input probe

Create a dedicated new-package forward adapter from the pinned reference numerical forward, sharing/replacing the same parameter values via a detached mapping. It accepts a full embedding leaf or dimensionless field-gain leaf, computes STAR/skip/experts/gates/towers identically, removes only `gate(emb.detach())`'s input detach, and returns routed true last-Linear raw logits [n,2] before sigmoid. Do not reconstruct logits from clipped probabilities. Training continues to call the original forward.

One viable minimal implementation is an nn.Module wrapper around the source model plus a copied forward-from-embedding routine, called with torch.func.functional_call and detached named parameters/buffers. Parameter replacement alone cannot remove source detach. The adapted forward must contain the complete gate input graph. Use input-derived dtype/device allocations so FP64 finite differences exercise FP64 arithmetic. The adapter changes neither source classes nor source files.

For embedding probe, use a detached leaf containing the target embedding; take each task's unreduced probability BCE sum with respect to that leaf, then select permitted slices. Because the network has no cross-row state, a summed loss yields each row's derivative in one traversal. For gain probe, each model uses its own detached embedding h_omega, gains [n,F] at zero, and h_omega,f*(1+a_f). Teacher parameters are constants but the gain leaf requires grad, so teacher probe must not be inside no_grad.

Accept when source probabilities and sigmoid(probe logits) differ by <=2e-6; FP64 smooth-point finite-difference relative error<=1e-4; real FP32 directional check error<=1e-2 over a stable step range1e-3/1e-4. Record absolute error as well when the true directional derivative is near zero. Probe leaves online/EMA/optimizer/loader/RNG hashes unchanged, parameter grads None, source training gradient unchanged.

## 6. J family

Within target batch, pool each row by (domain,complete two-label vector), exclude itself, order candidates by original row ID, draw at most two distinct donors without replacement. First draw up to two, then discard selected donor directions with norm<=1e-12; do not resample replacement donors. ML(0,1) count must be zero; measure all KR combinations without assuming nesting.

For surviving donor c, d_ic=detach(h_j,s-h_i,s), r_i=min(kappa*||h_i,s||,.2*min_c||d_ic||), delta_ic=r_i*d_ic/||d_ic||. One donor uses the same formula, no donors/zero radius yields zero. Both candidate energies match exactly and each interpolation coefficient<=.2.

J1 task embedding gradients are normalized independently; select the candidate maximizing minimum observed nonzero-task cosine. Ignore task gradients norm<=1e-12; all zero uses uniform random selection. Exact score ties use independent tie RNG. There is no positive-score coverage gate. J0 uses the same donor draws and radius, random selection, no probe. J_SUM replaces min by task arithmetic mean. Entire chosen delta is detached. EMA is nonfeedback for every J arm.

Record target coverage, joint-label support, valid/double donors, candidate and selected scores, chosen identity hashes, radius/energy and missing/zero cases. Extra detail is limited to first two train steps per epoch; accumulate construction counters throughout.

## 7. F family

Student gain gradient g=grad_a(mean of the two per-row BCEs); teacher J consists of two per-row raw-logit input gain gradients at old phi. Normalize each nonzero J row to norm1; rows<=1e-12 become zero.

Compute batched FP64 SVD on [n,2,F], keep singular directions s>1e-6*smax; smax<=1e-12 means rank0 and identity projector. Apply Pi*v=v-V_r^T(V_r*v) without allocating [n,F,F]. F1 starts v=Pi*g. If ||g||<=1e-12 or ||Pi*g||/||g||<1e-3, use one keyed Gaussian xi and v=Pi*xi; if still norm<=1e-12 set zero, no repeated draws. At equal teacher/student the projected supervised gradient is algebraically zero: the check must recognize fallback, rather than label it active student gradient.

Set w_if=||h_theta,if||², W=sum_f w. If W or sum_f(w*v²)<=1e-24 set a0. Otherwise a0=kappa*sqrt(W)*v/sqrt(sum_f(w*v²)), and a=a0*min(1,.25/max_abs(a0)). This is one row scalar multiplier, never coordinate clipping. Cast the final gain/delta to training FP32 after FP64 projection/scaling diagnostics. Final delta=detach(a*h_theta) as a whole; training is h_theta+delta.

F0 uses v=g without teacher; g zero gets one independent Gaussian direction. FR constructs F1 counterfactual delta* from its own current theta/phi, E=||delta*||², then a=sqrt(E/W)*independent Rademacher signs (W0=>0). Match counterfactual row energy at this current state, not the F1 trajectory's energy. F_PR uses Pi*xi always and omits student probe.

Log rank, row-zero constraints, retained ratio, fallback, Ja residual, cap rate and actual energy. First two steps per epoch additionally evaluate finite teacher logit response on the constructed gains and charge that extra forward.

One EMA is initialized exactly at theta0, read before current Adam step for views, then updated after each real Adam step: phi=rho*phi+(1-rho)*theta_new, buffers copied. Fixed rho ML.9828205985452511/KR.995; no bias correction/delay. F1/FR/F_PR use this same state for teacher and average evaluation; disabling only EMA evaluation does not disable teacher, and missing teacher state is an invalid F configuration.

## 8. Conditional TG/RG/AG

Only legal on global backup path. Per-task target embedding gradients give delta_it=kappa*||h_i,s||*g_it/||g_it||. Zero gradient gets a single normalized Gaussian direction; zero embedding yields zero. RG uses two independent random directions at that same row radius; AG normalizes g_i0+g_i1, with random fallback, one physical view shared across tasks. Full deltas detached, allowed slices only, one task per TG/RG view. No extra parameters/state/higher-order gradients; fixed nonfeedback EMA.

## 9. Randomness

Use keys containing study,dataset,split_id,seed,epoch,global_step,original_row_id and role. J0/J1/J_SUM share donor keys; all arms share initialization and loader stream. Donor, selection/tie, Gaussian, null fallback, FR signs and TG/RG roles are separate. Diagnostics/validation consume none of these streams; disabled components draw nothing.

Implement keyed random values simply with per-row NumPy default_rng(derive_seed(full_key)) if sufficient in Stage0. A vectorized stateless uint64 SplitMix64 mapping of (base-key digest,row_id,coordinate,draw role) with Box-Muller normals is permitted to remove per-row Python generator overhead; keep constants/algorithm frozen in source and use independent counters for paired uniforms. Either implementation must preserve row identity under row reordering and batching, two distinct donor draws, and identical J-family donor pools; choose one implementation before Stage0 and freeze it for all formal jobs. Do not substitute only an ordered-batch hash for original_row_id. Any change after scientific dispatch requires new scientific identity and regeneration of affected pairs.

## 10. RAM guardian and worker lifecycle

Create one independently launched long-lived guardian process per attempt, reachable through existing multiprocessing.connection local IPC and the reused BytesIO send_memory/receive_memory functions. torch.save into BytesIO is allowed; no file-backed state. Use a socket/address scalar receipt for reconnection. Guardian/worker processes are independent of the orchestration process lifecycle and do not exit merely when the controller's connection closes.

Guardian stores by scientific job key: latest complete epoch recovery state, online/EMA selected models and metadata, final E50 models, completion status and worker liveness identity. Latest recovery includes model, Adam, EMA, Python/NumPy/CPU/CUDA RNG, loader generators, epoch/global step, best values/selected epochs/model identities, diagnostics/counters and scalar-log commit boundary. Transfer each complete epoch; only the guardian's successful receive/ownership ACK permits RAM_COMMITTED. Snapshot selected models at strict improvements; E50 at final. Do not retain all50 epoch tensors. Completed jobs may discard optimizer recovery state after completion ACK while keeping evaluation models.

Worker failure with live guardian restores the last ACK epoch in a new worker at the identical configuration, preserving selected states and next batch/RNG. Scalar records beyond the committed epoch are uncommitted cost records, not duplicate scientific epochs. Reconnecting controller first reads frozen dispatch ledger, guardian statuses and live worker identities; it attaches to live/completed work instead of dispatching duplicate queues. A controller loss allows only already-frozen jobs to finish. If all RAM state is lost, report it; one essential recomputation is legal only under retry and cost rules, never replay the entire queue.

A failed necessary job gets at most one retry, original configuration, after cause assessment or isolated resource retest. Disk logs record physical failed/partial work and its cost. Completed scalar results retain their value if evaluation states are lost; essential missing test states are EVAL_STATE_LOST unless one budgeted recomputation is available.

Keep retained S1/conditional evaluation states in guardian until the global freeze and final evaluation, including a legal stop path. Release state after model hash/metrics receipt; final shutdown requires queue completion and evaluation/disposition for retained jobs.

## 11. Legal staged decisions

All decisions use fixed_ema, selected complete-validation six-cell macro AUC. A stage decision JSON is written before dispatch, recording inputs, chosen methods/configs, guards, required paired controls, scientific keys and cost reservation. Orchestrator owns scientific interpretation; runner records raw observations and executes approved dispatch.

S1 is24 complete50-epoch trajectories: datasets x dev seeds x A0/A1/J0/J1/F0/F1. Do not early-prune this initial necessary matrix. S1R adds FR per dataset only if F1-A0 mean>=.0005,2/2 positive, no protection trigger, and F1-F0 mean>0. Without completed FR, F1 cannot enter confirmation or claim independent direction advantage.

Development complex gate: versus A0 EMA mean>=.001,2/2 positive; versus A1 and strongest completed required simple reference mean>=.0005,2/2 positive; no harm guard. Simple candidates require only A0/A1 practical gates plus guards. FR is a control, not a deployable winner.

Protection checks every cell and macro LL against A0 and strongest necessary reference: LL mean deterioration>=.002 with both dev seeds adverse, or any cell AUC mean deterioration>=.002 with both adverse, blocks promotion and records TRADEOFF.

Choose one global path. Main if any dataset has complete or eligible boundary main family, or a complete simple alternative; backup only when all main/simple alternatives have no advance/calibration signal and no S2M/S3M started; stop on no executable condition/budget. The path is immutable after first exclusive job. This resolves the wording gap for an already-eligible simple alternative: it proceeds on main without inventing a calibration.

Boundary family only: A0 gain in[.0005,.001),2/2 positive, no guards, positive mean increment over required simple reference; one family per dataset and one half-kappa point (ML.025/KR.05). Family J gets J0/J1; F gets F0/F1/FR. Choose family by A0 gain; differences<.0005 prefer measured lower cost, then J. Choose default versus half-kappa by eligible mean; differences<.0005 keep default. No lambda/rho/cap/lr/threshold search.

S3M runs J_SUM or F_PR only for otherwise-promotable respective family, two dev seeds. A promising F1 with >90% fallback still gets F_PR; a failed F1 does not get explanatory extra training. Reapply full gate and simple preference after controls.

S2B is TG/RG/AG x both datasets x two dev seeds, max12, no tuning; globally excludes S2M/S3M. Family effects and guards use A0/A1 and both matching controls.

Per dataset choose at most one eligible winner. Difference<.0005 prefers simpler/lower measured cost; final tie order J0,F0,RG,AG,J_SUM,F_PR,J1,F1,TG. Strongest references are chosen only among completed dev controls, frozen before reading confirmation outcomes.

S4 uses held seeds2023/2025/2026. J1 pairing=A0,A1,J1,R_J where R_J is frozen strongest J0/J_SUM; J_SUM=A0,A1,J0,J_SUM. F1=A0,A1,F0,F1,R_F where R_F strongest FR/F_PR; F_PR=A0,A1,F0,FR,F_PR. TG=A0,A1,RG,AG,TG. Simple winner=A0,A1,winner. R_J tie prefers cheaper then J0, R_F cheaper then F_PR. References differing<.0005 use measured cost.

Held confirmation gate: A0 mean>=.001 and>=2/3 positive; A1/frozen reference mean>=.0005 and>=2/3 positive; combined5 seeds>=4/5 positive. Harm means retain.002 thresholds, require>=2/3 held adverse; also check combined>=4/5 adverse. Failure stops that candidate without retuning.

S5 only after S4 validation gate passes: W1 x seeds2023/2025 x A0,winner,strongest frozen completed simple alternative (A1 if winner itself simple), max6/dataset. Frozen config/rho, no retuning.

Max unique counts: main86, backup82, global86. S1 minimum24, conditional counts are ceilings. Reuse an identical already-completed trajectory within this attempt once, with scientific identity; shadow/endpoint/epoch/retry is not extra n. Explicit stop states follow plan05.

## 12. Stage0 and cost

Run the plan validator first. Minimum CPU semantic checks grouped into a few assert-based tests: source A0/A1 and disabled exact paths; original2B/tail/empty budget; donor joint-label/self/exact radius; numerical probe/full gate derivatives; fully detached delta with teacher/donor isolation; SVD rank0/1/2, equal-model degeneracy and uniform cap; FR energy; TG task-specific budget; Adam None versus explicit zero and coupled decay; EMA order; keyed RNG isolation/row invariance; guardian ACK, epoch restore and next-step equality.

Independent reviewer inspects design, implementation and coder report before research dispatch; only PASS or corrected blocking issues allows RUN. CPU semantic checks and short real CUDA Stage0 belong to acceptance, not scientific performance evidence. If reviewer requires a fix, return to coder and obtain corrected review before the24-run queue.

Stage0 real-data CUDA uses training batches and validation only: measure joint-label/domain counts, ML nesting, donor coverage, F rank/fallback/residual/actual energy, probe finite differences, no teacher parameter grads, backward and peak allocated/reserved GPU memory. Use full original B for allocation/cost measurements. Measure at least first two training steps per core method/dataset and complete validation forwards for online/EMA; record measured versus extrapolated epoch50 costs explicitly. Include setup, data/probe/teacher/diagnostics/validation/IPC costs; S0 work and retry work enter physical ledger.

Freeze cost_forecast per scientific job from these measurements: expected train steps (ML20/KR49 per epoch),50 epochs,100 full validation passes, guardian transfer overhead. Every legal new conditional dispatch adds only its S0-derived reservation; never inflate a job reservation after a failure. Physical cap=1.20*sum frozen expected cost of approved dispatched queue. Before dispatch, reserve enough for necessary matched pairs; prune optional calibration/ablation before unmatched mandatory controls. A true resource/cost stop is BUDGET_STOP_UNMATCHED, not gate success.

Scheduler checks each GPU independently. Add a worker when both current device memory and power are<85% and the measured next worker's peak memory fits physically remaining device memory; wait for its allocation report, then reassess current utilization before another placement. Concurrency can be>1 on one device; another busy device does not block a free one. Respect available CPU RAM for guardian/data. OOM changes concurrency only, with charged partial cost. No arbitrary wallclock expiry replaces the specified50 epochs or legal budget/stop rules.

## 13. Freeze, final metrics, analysis and delivery

After the last legal stage decision and all required training, freeze the complete configuration set, path/stop decisions, selected epoch for each estimator/split, E50 and model hashes in one scalar final-evaluation manifest. Open original test and W1 confirmation only then, in a single scheduled phase. Evaluate every retained completed S1 endpoint/control as well as conditional jobs, including when no candidate advances. Evaluate online and fixed_ema separately, selected and E50, natural unperturbed inputs. Deduplicate physical evaluation only when states/split identities match; retain logical endpoint labels. Never adjust method selection using final test.

Metrics per six cells: AUC, source unscaled probability BCE LL, count, positives, negatives; macro, sample-weighted traffic, per-domain and per-task, pooled per-task/mean, worst-cell AUC and minimum paired cell gain. The last two differ. For an undefined cell AUC, write null and null six-cell macro, retain LL/population; do not drop a cell. Source metric arithmetic is reused where all cells are defined. W1 missing support becomes INCONCLUSIVE_WINDOW without another window search.

Store run/trial, decision, physical-cost and failure ledgers; configurations, epoch validation and train losses; endpoint/model hashes; Stage0 diagnostics; per-seed final cells and scalar runtime. Aggregate with seed as n, sample SD, paired effects, positive counts and unadjusted t95% CI(df=n-1). Develop n2, held n3 and combined n5 separately. No task/epoch/prediction pseudo-replication.

Final test practical gates mirror held validation; stable scope further requires W1 A0 gain>=.001,2/2 positive and simple-reference gain>=.0005,2/2 positive, no guards. Report WINDOW_LIMITED if original split passes but W1 fails. Interpret original test as historically exposed; W1 is a different raw-row window, not new external or verified chronological data.

Orchestrator analysis must address AUC/LL, domain/task, seed consistency, domain size versus measured difficulty, baseline, model and stage; stable phenomena, contradictory observations, STRONG/MEDIUM/WEAK evidence, competing explanations, size as explanatory variable, small/large tradeoff, and the most discriminating next experiment. Separate ordinary EMA gain from view increment; F feedback online/EMA difference does not isolate teacher causality. Do not assume any candidate works. If all fail, retain A0+EMA and close this registered queue.

Deliver SOURCE_RECEIPT, this DESIGN, coder report, independent review, Stage0, protocol/queue, raw ledgers/tables, paired effects and REPORT. Parent owns publication to the newly available date branch26-11-02_v2, retaining main history, no reset/force push, only this study's source/report/scalar bundle. Original plan status remains planning history.

No unresolved mathematical choices remain. Future implementation blockers affecting these semantics must return to the orchestrator for an explicit design addendum; style or unrelated engineering issues do not delay experiments.

## 14. Core/execution implementation interface

These signatures are the coordination boundary for two implementation owners; file layout inside each responsibility may be simpler if equivalent.

`protocol.py` (core owner) exports:

```python
load_protocol() -> dict
config(arm: str, dataset: str, *, kappa: float | None = None) -> dict
make_optimizer(model, config) -> torch.optim.Adam
update_ema(ema, model, rho: float) -> None
state_digest(value) -> str
# Re-export reused vendor helpers:
setup, cpu_copy, equal, rng_state, restore_rng, tensor_bytes,
send_memory, receive_memory, write_json, append_csv
```

load_protocol returns the complete original YAML as p and enriches each `p["datasets"][dataset]` with `source`, the exact dataset record from vendored resolved_protocol.json. The original source record contains perturbable field names, raw/array/row/encoder/scaler identities; the YAML field count is not a replacement for this list. Add `vendor_source_hash` and `study_id="EMA02V2"`. No protocol reader writes files by itself. Execution owns the scalar resolved-protocol receipt.

config returns at least `arm,dataset,lambda_aug,kappa,target_domain,rho,lr,weight_decay,epochs,batch_size`; family/selector derive from the registered arm table. It accepts only registered arms and default/allowed half-kappa. make_optimizer follows section3 exactly. setup is called before model construction. Execution creates the source views model with `make_model(bundle,p["datasets"][dataset]["source"])`, then `copy.deepcopy(model).requires_grad_(False)` as EMA. Core does not load real data.

`objectives.py` (core owner) exports:

```python
train_step(model, optimizer, ema, x, y, row_ids, config, context,
           *, diagnostic=False) -> dict
```

x=[fields,B] and y=[B,2] on the training device; row_ids is ordered NumPy int64[B] original raw row IDs. context has `study,dataset,split_id,seed,epoch,global_step`, global_step starting at1. train_step owns zero_grad, view construction, scalar loss, one backward, one Adam.step and one EMA update, in that order. Execution must not repeat these operations. It returns:

```python
{
  "loss": float,              # exact mixed supervised scalar before update
  "real_loss": float,         # clean probability BCE mean before update
  "pred": detached_tensor,    # [B,2], no retained graph
  "diagnostics": dict,        # JSON-compatible aggregates; detailed only when requested
  "counts": dict,             # actual work, integers unless documented
}
```

counts includes `original_rows,target_rows,extra_view_rows,training_forwards,training_forward_rows,probe_forwards,probe_forward_rows,teacher_forwards,teacher_forward_rows,input_gradient_calls,parameter_backward_calls,optimizer_steps,ema_updates,diagnostic_teacher_forwards`. Distinguish teacher probes from training forwards; parameter backward always1, optimizer_steps1 and ema_updates1 for a successful step. Real timing and peak-memory measurement are execution's responsibility. Count disabled branch probes/draws as0.

Core can additionally expose pure `losses(...)->(scalar,info)`, `build_delta(...)`, and `input_probe.probe_logits(model,x_target,*,embedding_leaf=None,gains=None)` for its tests. build_delta returns target sparse deltas [n,F,16] (two tensors only for TG/RG); every tensor is fully detached. These internals are not required worker imports.

Core owns `protocol.py`, `rng.py`, `input_probe.py`, J/F/TG view modules, `objectives.py`, `vendor/` and core semantic tests. Execution owns `data.py`, `guardian.py`, `run.py`, evaluation/gate/stage0 modules and execution semantic tests. Execution reads core protocol; it must not edit core files without coordinating. Each owner writes a separate coder report. The independent reviewer reads both reports and the composed source.

Execution's public commands are a single new-package CLI with stage0, run/attach and final-evaluate modes; their exact spelling is documented by the execution coder. CLI defaults never dispatch unreviewed research training. New run requires the approved review receipt; attach reconnects to the attempt's guardian/frozen jobs. Source freeze identity is computed after both owners finish and before scientific dispatch. A blocking semantic fix changes the source identity and is reviewed before resuming affected jobs.


