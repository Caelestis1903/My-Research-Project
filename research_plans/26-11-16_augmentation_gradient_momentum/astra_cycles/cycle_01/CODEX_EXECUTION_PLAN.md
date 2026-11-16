# CODEX EXECUTION PLAN — implementation-ready DESIGN SPECIFICATION
Status: frozen research specification for candidate A; not implementation approval. Source base is published main/HEAD c4dc36bd38bd4f7a2a3319377a8289ba8185532d, tree bae323bf8c33e8144eaf225633d2fd481f553df8. No algorithm code was written by this Astra child.

DESIGN_COMPLETENESS_ADDENDUM.md is mandatory: it resolves the exact DT-PGNA/TGA history comparison and replaces the shared-guardian write proposal with isolated cycle-root mechanical RAM storage. Its two overrides govern older statements in this six-file Astra dossier; formulas, matrix, baselines and literature remain unchanged. Designer completeness verdict PASS remains subject to the original parent's hard-completeness gate and subsequent distinct implementation review.

## 1. Ownership and exact next stages
Only foreground parent thread 01a1dfb8-605b-7430-8462-5fd3da0b048f dispatches:
1. DESIGN: consume the six original Astra files plus DESIGN_COMPLETENESS_ADDENDUM.md; parent owns the hard-completeness gate. Close each design child permanently.
2. IMPLEMENT: fresh native luna-coder, explicit fork_context=false, file-only handoff. Implement only A and its two controls. Coder performs syntax/basic direct correctness checks, not research-result analysis.
3. REVIEW: a distinct fresh native reviewer, fork_context=false, reads specification, actual source and CODER IMPLEMENTATION REPORT independently. Verdict PASS or PASS WITH FIXES with all blockers corrected is required. FAIL returns to a fresh coder; revised source gets a fresh review. Close each child permanently.
4. RUN: fresh native luna-runner, fork_context=false, supervised in this current conversation, records raw results only. Reviewer-approved full-batch smoke precedes formal trajectories. Runner may supervise concurrent mechanical training processes, but no AI/session/driver delegation.
5. ANALYZE: original parent analyzes the complete raw evidence, records scientific decision and contradictions, releases designated discarded RAM states, then sends files to a fresh native Astra. No child inherits the parent transcript.

Never run codex exec/resume/fork, create_thread, tmux, nohup, systemd, or autonomous.py start/run/recover. Never start/repair the retired driver or replace/restart the old guardian. The current user's isolated cycle-root mechanical RAM owner is authorized only as specified in section8/addendum; it is not an AI/session/driver. Existing historical source and receipts remain frozen. No commit/push is requested by this plan.

## 2. Narrow implementation boundary
Suggested new package: experiments/task_axis_even_views_20261116/ with only the direct core, protocol/client runner and required package entry files. Use installed torch/numpy/pandas/standard-library helpers; no framework, dependency, generalized optimizer registry, automatic search or new configuration surface.

Use the frozen imports:
- auxiliary_objective_attribution_20261115/core.py: forward and sensitivity (SHA-256 0999b534aba4f632f9fbd35a3313d5eeb4fd04405d323f881d2e39a7f4a2bfe3).
- auxiliary_objective_attribution_20261115/stage2/core.py: make_views and source NDP-A train_step for identity checks (fe662da1b8c00229a6a234bb7e00d659fab1d278470092531cdeefa6bb2f3d58).
- task_view_gradient_residual_momentum_20261116/core.py: source_clean_step only for canonical control identity (5915d97983c79ccb33302146fde287419ef611caee5bda2f13898c973845c9d9).
- Existing source setup, make_model, optimizer, update_ema, evaluate, cpu_copy, state_digest, memory IPC and verified data helpers. Record every transitive imported source hash.
Do not invoke an executable driver entry point or call old prepare/schedule/ensure_guardian. Frozen task_view_gradient_residual_momentum_20261116/runner.py::serve is the sole storage entry-module import exception; use it unmodified at the new cycle root, not its main/scheduler/worker. Its recover branch may reach only the existing read-only stop predicate, never driver actions. Reviewer checks transitive import side effects. The old core/protocol remain pure helper imports.

All new numerical artifacts go under artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/. New reports go under the corresponding cycle report directory chosen by parent. Do not overwrite existing attempts or revise this frozen design after launch. A source repair creates a new reviewed attempt identity; no silent mixing.

## 3. Data/model/training identity
Datasets: movielens and kuairand. Natural train counts 800167 and2006132; validation100021 and250767; original split, rows, labels, feature encoding and support. Raw paths and every manifest identity are in EXPERIMENT_MANIFEST.json. Verified cache directory exists at /home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache. Use the existing data protocol with relocated raw paths; verify raw/train/validation/encoder/scaler and row-order identities. No re-split, resizing, balancing, synthetic rows or fitted-on-validation preprocessing. Missing unrelated mdmt_next prepared artifacts are not silently substituted.

M3oE fixed1: 3 domains,2 tasks; fcn_dims=[512,256,256,64], expert_num=4, exp_d=exp_t=bal_d=bal_t=1, embedding_dim16. Ordinary model/gates/experts/fusion/sharing unchanged. No BatchNorm/dropout in the frozen model; retain ordinary training gate detach. Probe-only full gate derivatives remain the source behavior. Embedding perturbation covers exactly the NDP-A embedding output, including its existing field coverage; domain routing uses original x.

Environment: /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python, Python3.13.5, PyTorch2.9.0+cu126, requirements-phase0-3.lock. Invoke project commands via that executable or conda run --no-capture-output -n phase0-3-env python. FP32; no autocast/GradScaler/TF32. Source setup uses one CPU torch thread, CUDA matmul/cudnn TF32 false, cudnn benchmark false; seed Python, numpy, torch CPU and assigned CUDA with run seed. Do not invent new deterministic-kernel flags that change source comparability.

Each dataset/arm uses seeds [2022,2023,2024,2025,2026],50epochs,batch40960,tail included. Per epoch use private CPU torch.Generator seeded once seed+11000 and torch.randperm(N,generator=loader). Same order across arms. Record original row IDs/order digest. Exactly20 ML /49 KR updates per epoch;1000/2450 per trajectory. No early stopping; selected model is a RAM copy, not a reason to stop training.

## 4. Deterministic A core
Fixed constants alpha=.03, mixture lambda=.25. No gamma/cap/residual memory. Define each step at current theta in this order:

1. optimizer.zero_grad(set_to_none=True).
2. Call unchanged source make_views(model,x,y,.03) to obtain delta_0,delta_1 and probe diagnostics. Its sensitivity returns detached e,z,J, using autograd.grad with create_graph=false; it must not populate parameter .grad. r=sqrt(mean(e^2)); direction=(sigmoid(z_t)-y_t)*J_t; valid iff direction norm>1e-10 and r>1e-12; delta=.03*r*direction/(norm+1e-12), else zero. All deltas detach. Release the probe graph before training forwards.
3. Construct two perturbation replicas per target task:
   - AXIS-EVEN: delta_t1=delta_t; delta_t2=-delta_t.
   - AXIS-IID2: independent scalar Rademacher signs for each row/task/replica; delta_tj=s_itj*delta_t. Signs do not alter labels or route.
   - ISOTROPIC-EVEN: q_it drawn Gaussian, u_it=norm(delta_it)*q_it/norm(q_it); use +u,-u. Zero delta -> zero u. Exactly zero q norm with nonzero delta -> first-coordinate unit direction with that radius. No resampling.
4. Run clean probabilities=model(x), C=F.binary_cross_entropy(probabilities,y) with source mean reduction.
5. If all original delta coordinates for both tasks are zero, take the canonical clean branch: total=C, no view forwards, same one backward/Adam/EMA as source. Preserve ordinary .grad activity. This is an exact null identity, not a scientific arm.
6. Otherwise do four ordinary source forward calls, **task0 replica1, task0 replica2, task1 replica1, task1 replica2**, each full batch with its additive detached embedding perturbation. Let a_tj=F.binary_cross_entropy(view_probabilities_tj[:,t],y[:,t]) with mean over B. Both outputs can exist; only target task t contributes from that view.
7. R1=.5*(a_01+a_11); R2=.5*(a_02+a_12); V=.5*(R1+R2); total=.75*C+.25*V, using this parenthesization in all three arms. For even arms R1=Lplus,R2=Lminus. IID2 R1/R2 are sampled replicas, not positive/negative losses. Avoid swapping to summed losses, logits BCE or sequential backward.
8. Assert finite total, total.backward() exactly once, inspect gradients for nonfinite values without changing them, optimizer.step() exactly once, source update_ema exactly once. Drop references to all view graphs after the step.

Mass ledger for every batch and every domain-task-label cell: count n, clean coefficient .75/(2B), each view replica coefficient .125/(2B), total natural mass n/(2B). All rows contribute to both tasks; zero perturbations remain included. Per-task mass .5; total mass1. Canonical zero branch uses one clean coefficient1/(2B) and records the equivalent conceptual view weights. Actual tail B determines every denominator.

### Private RNG
No view draw may consume Python/numpy/global torch/loader RNG. For each needed stream create torch.Generator(device=x.device). Seed from the first eight SHA-256 bytes interpreted unsigned little-endian of the UTF-8 string formed by joining these exact fields with "|" and no trailing newline:
axis_even_v1 | dataset | decimal_run_seed | decimal_epoch | decimal_batch | decimal_task | stream
(the displayed spaces around separators are explanatory; actual string has none).
Epoch/batch are one-based; task0/1. Stream names: sign1, sign2, isotropic. For signs call torch.randint(0,2,(B,1),generator=g,device=x.device,dtype=torch.int64), convert 2*s-1 to FP32. For isotropic call torch.randn(delta.shape,generator=g,device=x.device,dtype=torch.float32), then rowwise L2 normalization. No arm or GPU index in the key. Frozen same hardware/software and assigned-device implementation apply; global RNG identity must remain unchanged by draws. After RAM recovery, epoch/batch keys plus restored loader state reproduce draws. Never use Python hash().

### Adam, activity and EMA
Use source optimizer helper: lr=.01, betas(.9,.999),eps1e-8, coupled WD ML1e-4/KR1e-5, foreach=false,fused=false,amsgrad=false. g is the derivative of total; h=g+wd*theta once; n_p increments only if grad_p is present; m=.9m+.1h, v=.999v+.001h^2; update uses m/(1-.9^n_p) and v/(1-.999^n_p). grad=None skips WD/state/counter/update; explicit zero does not. Do not attach a second optimizer, precondition separately, clip, project or rescale the combined gradient.
EMA copies initial model, then rho*EMA+(1-rho)*updated_model once per optimizer step, copying buffers. rho ML=.9828205985452511,KR=.995. No EMA bias correction.

## 5. Frozen matrix and comparisons
New:
- AXIS-EVEN, both datasets, seeds2022–2026,50each.
- AXIS-IID2, both datasets, seeds2022–2026,50each.
- ISOTROPIC-EVEN, both datasets, seeds2022–2026,50each.
30new,1500epochs,51750Adam updates,51750EMA updates. Do not execute B or C, HPO, added seeds, new confirmation runs or adaptive stopping.

Historical controls are exactly the manifest's twenty source-indexed CLEAN/NDP-A result records. All selected, fixed-epoch and online/EMA comparisons stay within the same dataset and seed and the same evaluation protocol.
Primary: AXIS-EVEN minus CLEAN selected fixed-EMA six-cell Macro AUC, separately ML and KR.
Mechanism contrasts: AXIS-EVEN minus NDP-A; minus ISOTROPIC-EVEN; minus AXIS-IID2. Report both datasets for each, no ranking of the three new arms to nominate an unregistered champion. A control's unexpectedly good outcome informs fresh research rather than promotion within this experiment.

### Historical reuse gate
Before formal runs, verify twenty exact result-file hashes; original per-seed configuration, model/source dependency identities, initial model hashes, environment settings, data arrays/row order, Adam/WD/EMA/selection and support. Confirm historical CLEAN is the exact source null branch; confirm NDP-A is the exact source one-sided objective and unchanged probe. Their input/objective differences are intentional factors, not an excuse to reuse an unrelated baseline.
Direct equality checks use the original source steps, not a newly rewritten algebraic equivalent. Record gradient None masks, every parameter's Adam counter and all model/moment/EMA tensors, including buffers. New view RNG must not perturb initialization/loader/global state. Require exact equality for the delegated null/source identity, not only a close scalar loss.
If model, preprocessing, objective, optimizer or relevant software/source identity differs without exact proof, PAUSE this frozen matrix. Parent must specify a source-matched new control and its exact budget in a revised design, then fresh coder/reviewer. Never silently treat old CLEAN as universal.

## 6. Minimal correctness and smoke gates
No new testing framework or broad test suite. Use installed/pure helpers and a short direct assertion entry point:
1. On a small deterministic in-memory fixture, check mass/target support including zero directions, all-zero canonical branch, actual tail denominator, sign reversal symmetry of the even objective, and equality of the average of the four replica-sign combinations to the even loss for one row-task. No random expectation tolerance masquerading as exact identity.
2. Verify keyed RNG reproducibility and no global/loader RNG consumption; verify isotropic input-norm relative error <=1e-6 for nonzero deltas, zeros stay zero; record actual error. This is not update-domain norm matching.
3. Verify Adam parameter activity on a tiny unused-parameter fixture: None skips WD/counter, explicit zero stays active, coupled WD occurs once, EMA once. Inspect existing torch behavior; do not implement a toy competing optimizer.
4. After independent source review passes, runner performs full40960 FP32 training-only smoke on the first source-ordered2022 batch of each dataset for each new arm. Record timing/peaks and support; discard smoke states and reinitialize formal runs. Include direct null/source and one-sided/source identity checks on that batch and a tail-sized slice using unchanged source helpers; do not train smoke epochs or inspect validation for tuning.

Source review must pass before GPU experimental execution. A smoke failure pauses and returns through a fresh coder/reviewer if implementation changes. Smoke only verifies correctness/admission; it cannot pick alpha/lambda/arm or alter the matrix.

## 7. Raw records, evaluation and statistical contract
Every batch: dataset/arm/seed/epoch/batch/global_step, row-order hash, actual B, cell support/mass, clean/replica/total losses, zero-direction counts, RNG key version, optimizer/EMA count.
At first batch of epochs1,10,25,40,50 only: actual input radius/norm-match error, sign duplication fraction for IID2; E and O for even arms (loss finite differences, not gradient directions); total data-gradient norm, None count, per-parameter step range, actual update L2 norm. For actual update measure detached theta_after-theta_before in stable named-parameter order with FP64 scalar reductions; no Adam shadow simulation. No cosine is required. Reuse existing names/parameter group metadata where convenient without retaining GPU histories.
Per epoch: training loss/probe/construction/training/evaluation seconds, peak GPU allocated/reserved, host RSS and RAM occupancy, source/model/EMA/optimizer hashes, validation six-cell metrics for online and EMA, selected epoch/model hashes and all counts.
Evaluate only natural validation after every epoch. Independent strict-greater selection for online and fixed EMA; selected LL belongs to AUC-selected model. Keep E10/E30/E50 endpoints as secondary diagnostics. Store all six cell AUC/LL/count/positives, per-domain/task macro, six-cell Macro, traffic-weighted cell and pooled metrics, worst cell. No Test forward, metric, tuning or selection path; existing Test hashes are provenance metadata only.

Runner emits endpoint, paired-seed inputs, cost and diagnostic tables plus COMPLETE_RUNNER_REPORT containing exact coverage, physical cost, state-retention keys and test_accessed=false. Runner records facts, not conclusions.
Parent computes paired differences over the five seeds: mean, SD(ddof1), count positive/negative/zero, and descriptive95% CI mean +/-2.7764451051977987*SD/sqrt(5). AUC positive is better; LL negative is better. No pooled-row significance, diagnostic-batch pseudo-replication, multi-seed averaging of models, or joint AUC/LL success score. Report all specified contrasts, not post-hoc winners; no familywise significance claim from these descriptive intervals.

## 8. Admission, RAM and current-conversation supervision
Before each child dispatch and mechanical worker launch, read both workspaces' stop controls, including their existing STOP_AUTONOMOUS/manual-stop controls. Honor pause/failure records. Both memory **and** power admission must be <.85 per individual3090; host RAM<.90 including current states, selected states, pending allocations and transient CPU/IPC snapshot copies. Use measured smoke peaks to reserve GPU bytes and measured process power increments; initial planning reservations4.5GiB ML/5.5GiB KR are replaced by measurements, not hard limits.
Scan both cards independently and launch additional training per card whenever admissible. Track pending allocations to avoid admission races; sample actual power/memory after launch before admitting further work on that card. If no card fits, wait in short supervised polls and communicate. Never lower gates, reduce batch/precision or set an arbitrary process-count cap. Actual new peak larger than reservation updates reservations and pauses further admission until safe.

Preserve the existing guardian and its old states without any new-key registration, journal append, result mirror, release or shutdown there. During reviewed RUN, reuse the audited frozen task_view_gradient_residual_momentum_20261116/runner.py::serve(attempt) in a **new cycle-root mechanical memory owner** at the absolute new cycle attempt directory above. Its root argument already isolates all journal/result writes; no new server/framework is needed. The parent owns its persistent foreground tool session/PID, and the current-conversation runner supervises it and the workers. No detached launch, ensure_guardian, old scheduler/driver or background AI is allowed. Preflight verifies a distinct resolved root/owner endpoint, unchanged helper hashes and old inventories; missing/incompatible storage pauses. Exact launch/API/source and lifecycle requirements are in addendum section7.
New job key = SHA-256 of canonical JSON (sort_keys=true,separators=(",",":")) containing cycle_id, dataset, arm, seed, complete config, new source hash and design-bundle hash. Design-bundle hash covers the six original files **plus DESIGN_COMPLETENESS_ADDENDUM.md** after finalization. Hash the lexicographically sorted cycle-relative `[path,SHA256(file_bytes)]` pairs as a UTF-8 compact JSON array with no appended newline, per addendum section10; store it only in a separate receipt, with no self-hash embedded in these seven files.
Snapshot current model, EMA, optimizer, loader/global RNG, epoch/global_step, selected online/EMA models/epochs/validation, best scores/counts and accumulated cost. No residual state exists. At epoch end write numerical records, send snapshot via RAM IPC, wait for ACK, release temporary snapshot references. Keep only the last committed continuation state plus current working state and current selected models; replace superseded selected tensors, do not accumulate epochs.

All new requests use request(new_cycle_attempt,...) and the new owner's generated guardian.json. Commit appends only its cycle-local epoch_commit_ledger.jsonl; complete writes only new_cycle_attempt/runs/<key>/result.json. Send absolute cycle-local epoch_file paths. Before/after smoke and formal writes, require whole old journal length/hash and sorted old result path/hash inventory equality, not prefix equality. Old50 local results plus20 reused controls remain the same70 export inputs. New export uses exactly30 cycle-local results plus the immutable20-control index. No new-key mirror, old entry mutation, old recovery/driver automation or archive change is permitted.
At epoch50 use the frozen TV serve complete operation, which compacts state to selected_models/selected_epochs/selected_validation. Parent accepts a numeric storage-ownership receipt with the live parent-owned tool-session handle, root/PID/key inventory and retained hashes/bytes before the runner closes permanently. Keep that mechanical owner alive under parent supervision across fresh-agent handoffs; fresh runners attach to the recorded endpoint, never restart over lost RAM. After parent analysis, release only explicitly discarded new-cycle keys; retain designated eligible/incumbent selected sets per existing policy and protect old live states. Shut down only a new owner with no retained state and no live workers. At cycle handoff purge superseded discarded tensors so unbounded cycles do not accumulate unbounded model history. Unsupported lifecycle/isolation means PAUSE, not a detached fallback. No model, optimizer, tensor checkpoint, pickle snapshot or RAM spill is written to disk.

Mechanical workers are launched/monitored under the current runner's live foreground tool session with owned PIDs and pipes, not detached process sessions. They contain no Codex/agent invocation. The parent must keep the runner context supervised until completion or safe pause; if supervision is lost, stop new admission and pause workers at the next epoch/RAM-ACK boundary.

## 9. Stop, recovery, scientific gates and next research
An atomic operation is the current epoch's batches + validation + acknowledged RAM commit. If user stop is seen before an epoch, do not start it. If seen during an epoch, safely complete that boundary when technically possible, cease dispatch, retain selected/current RAM and report USER_STOPPED. OOM/nonfinite state can prevent finishing: preserve the previous committed snapshot and report the failure honestly rather than claim a checkpoint.
Technical failure -> PAUSED, Test sealed. Resume only from a verified existing RAM epoch boundary under a fresh current-conversation runner and reviewed source. Lost RAM cannot be reconstructed from disk; parent specifies a frozen rerun/new attempt after resolving the fault. Preserve failed receipts and physical cost. No hidden retry or revision cap; no source mutation inside a running trajectory.

After complete matrix:
- Both dataset AUC means AXIS-EVEN-CLEAN <=0: candidate NO_CONTINUATION_SIGNAL; no HPO/confirmation.
- One positive, one nonpositive: MIXED_SIGNAL; preserve positive evidence and contradictions, do not call all-dataset nonpositive.
- Both positive: PROVISIONAL_CROSS_DATASET_SIGNAL, not a champion or Test authorization. Evaluate seed consistency/CIs, LL, every cell/task/domain, fixed epochs, compute and the three mechanism contrasts before choosing further research.
- Odd-component, axis-information and antithetic endpoint hypotheses are separate: compare A-NDP, A-isotropic and A-IID2 respectively. If a contrast fails its directional prediction, state that; a CI crossing zero is unresolved evidence, not proof of equality.
- Label evidence STRONG/MEDIUM/WEAK with reasons, alternatives and observed contradictions. Domain size remains descriptive; no small-domain/difficulty equivalence.

Every candidate outcome including NO-GO or Astra STOP returns through parent report/context to fresh native research. max_cycles=null,max_astra_cycles=null,max_experiments=null; no hidden total/revision cap. Normal line termination only USER_STOPPED.
Test remains sealed through ACTIVE, PAUSED and all candidate outcomes. Only explicit manual stop + final freeze + the existing eligible full-five-seed selected-RAM verification gate may enable later Test work, by the original parent. This runner has no automatic Test step. Preserve older validation-to-Test eligibility requirements; this plan neither relaxes them nor declares an eligible final model.

