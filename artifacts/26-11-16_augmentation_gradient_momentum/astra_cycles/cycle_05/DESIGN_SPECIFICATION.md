# C05 DESIGN SPECIFICATION — CELL-MASS-BCE-025

Status: COMPLETE DESIGN — ready for parent acceptance and fresh coder dispatch. Implementation starts only after the designer's complete handoff and parent acceptance. This file is the seventh design artifact; the six accepted research files remain immutable.

Designer: Lorentz, native thread `01a1e9c0-2e59-71d0-903b-d132bc65d1cd`, gpt-6.1-sol/high, `fork_context=false`. Original foreground parent: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. No descendants. The designer returns permanently after this handoff.

## 1. Research question

Does removing COMPLETION-BCE-025's within-cell class coefficient variation, while preserving its six domain/task coefficient masses on every natural batch, retain its task1 validation LogLoss benefit and change its KuaiRand d0t0 harm? Independently, does the resulting fixed candidate qualify against matched CLEAN fixed EMA on both datasets?

Only U=`CELL-MASS-BCE-025` is new. C=`CLEAN`, W=`COMPLETION-BCE-025`, and S=`MASS-SCALE-025` are exact numeric controls. The parent completes this one C05 through analysis, report, distinct evidence review, publication and delivery, then USER_STOPPED. No C06, next Astra, efficacy early stop, fallback candidate, HPO, architecture change, sampling change or Test evaluation.

The controlling inputs are W/AGENTS.md, GB18030-decoded W/start.md, W/buchong.md, the latest runbook policy, the six P5 files, and A5/ASTRA_HANDOFF.json. Both workspace state files were read: ACTIVE, manual_stop_requested=false, termination_reason=null, stop_after_current_cycle=5, cycle06_authorized=false. All four workspace/artifact STOP_AUTONOMOUS sentinels were absent. Recheck before subsequent dispatch/admission. The latest stop-after-C05 instruction supersedes preserved stop-after-C04 and uncapped-continuation paragraphs.

## 2. Hypotheses being tested

H1, efficacy: U fixed EMA minus C fixed EMA, independently selected by validation AUC, has mean MacroAUC gain at least .003 and each of six cell mean AUC deltas at least -.001 on each dataset. Unscaled same-state mean LogLoss improvement is a separate axis.

H2, cell/task allocation sufficiency: U minus S selected fixed-EMA task1 macro LogLoss has negative mean on both datasets. This tests whether retaining W's cell allocation can retain the repeated task1 benefit after removing its within-cell class policy.

H3, class-skew harm: KR U minus W selected fixed-EMA d0t0 has positive mean AUC delta and negative mean LogLoss delta. Evaluate both directions separately and retain every other cell, including d2 and dominant d1, in the comparison.

C04 provides MEDIUM evidence for the conditional task1 loss observation and WEAK evidence for stable selected-AUC efficacy. U efficacy is untested. Full-trajectory differences include their induced parameter, Adam and EMA histories. Local diagnostic decompositions do not estimate mediation fractions. Domain size and observed baseline difficulty remain separate variables.

## 3. Variables changed

The sole scientific intervention is the BCE coefficient policy. W's class-dependent coefficient q_ca is replaced by its empirical conditional mean a_c inside each domain/task cell. Lambda is fixed at .25, inherited without tuning. U−W compares the complete class-policy intervention at fixed batchwise cell/task/global masses; U−S compares cell/task redistribution at fixed global mass; U−C tests efficacy.

Engineering changes are confined to the new C05 package: physical /tmp roots and explicit publication mapping; named bounded NVML sampling; terminal facts independent of cleanup; and append-only resource witnesses instead of copied contributor histories. These changes do not alter the scientific recurrence.

## 4. Variables held constant

| Item | Fixed requirement |
|---|---|
| New matrix | ML and KR × seeds 2022,2023,2024,2025,2026 × U × 50 epochs = 10 trajectories |
| Reuse | Exactly 30 C/W/S trajectories, each result plus 50 epoch files = 1,530 files; no old-owner IPC |
| Train/validation rows | ML 800167/100021; KR 2006132/250767 |
| Domain train counts | ML [155949,318599,325619]; KR [172782,1743898,89452] |
| Batch/precision | 40960, natural tails ML21927/KR40052, FP32, no autocast |
| Updates/epoch | ML20; KR49 |
| Model | Source M3oE fixed1, 3 domains × 2 tasks, embedding16, four experts, FCN [512,256,256,64] |
| Ordinary gate | Source `gate[i](emb.detach())`; no probe reconnection |
| Adam | lr .01, betas (.9,.999), eps 1e-8, coupled WD ML1e-4/KR1e-5, foreach/fused/amsgrad false |
| Fixed EMA | ML .9828205985452511; KR .995; initialized from online; one update per live Adam |
| Initialization | Source setup/make_model; exact per-dataset/seed CLEAN initialization identity |
| Order | Natural train order from private CPU generator seed+11000; all 50 permutation hashes equal matched references |
| Selection | Earliest strict maximum of validation six-cell MacroAUC over epochs1..50; independent online/fixed_ema |
| Validation | Source chunking 40960, unmodified natural validation; same selected state for AUC, LL and cells |
| Data | Source80/10/10 split, train-only preprocessing, unchanged features, encoders and scaler |

Source `setup` remains authoritative, including one CPU thread, TF32 disabled and source deterministic settings. Model RNG and the private loader generator are separate. No algorithm RNG draw is introduced.

C04 actual publication is `cbccf5ae234e46bc3f976467de2b3327139f021b`. Its source48 hash is `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`. Its forty selected models were historically accepted, then owner2476891 was lost; exact exit cause is unknown. Numeric controls remain valid. U starts at source initialization, with independent new parent-owned storage. No C04 reconstruction, warm start or retention rerun.

## 5. Mathematical and algorithmic definition

For a natural batch B, probabilities and labels are [B,2]. Domain IDs are [B]. In the actual source model call, x is [F,B], y is [B,2], and domains=x[-1]. CPU cache inputs are [N,F]; preserve the source transpose. Use probability-space `torch.nn.functional.binary_cross_entropy`, including its source boundary behavior; do not use logits BCE, manual clipping, sample synthesis or score-dependent coefficients.

For c=(d,t), let n_c be its row count, n_ca its label count, pi_ca=n_ca/n_c, w_c=n_c/(2B), ell_it=BCE(p_it,y_it), and C_c=mean(ell|c). Counts are detached integer observations. With lambda=.25:

    q_ca = 1-lambda+lambda*pi_ca
    kappa_c = pi_c0^2+pi_c1^2
    opportunity_c = 2*pi_c0*pi_c1
    a_c = 1-lambda*opportunity_c
    C = mean over all 2B ordinary BCE entries
    H = sum_c w_c * sum_present_a pi_ca^2 * mean(ell|c,a)
    W = (1-lambda)*C + lambda*H
    U = sum_observed_c w_c*a_c*mean(ell|c)
    removed_mass = sum_c w_c*lambda*opportunity_c
    m = 1-removed_mass
    S = m*C

The production U reduction order is fixed: obtain ordinary unreduced BCE entries; initialize a scalar zero of their dtype/device; visit d=0,1,2, then t=0,1; for each nonempty cell add `(w_c*a_c)*entries[domains==d,t].mean()` to that scalar. w, pi and a use detached Python count ratios in the same ordering as frozen C04 `_support`; multiply w*a before the tensor mean product. Never reconstruct U by W−residual, divide by m, normalize cell/task coefficients, balance classes, or flatten all weighted entries into an alternative production reduction. Those formulas are mathematical fixture oracles only.

Reuse frozen C04 `loss_components(probabilities,targets,domains,arm='COMPLETION-BCE-025',mixture_lambda=...,diagnose=False)` for exact C/H/W/S reduction and support. Its invocation has no forward or sweep. Derive U from the same probabilities and BCE entries; add U fields to this fresh returned ledger. For scheduled detached class statistics reuse frozen `_statistics(ledger,probabilities,targets,domains,True)`. Do not call its diagnose=True loss path, which would also construct the unneeded NQR objective. Do not call C04 train_step/config/write-path wrappers with altered globals or pretend U is a C04 live arm.

For every observed cell, U and W share exactly the computed field `completion_bce_mass=w_c*a_c`; U's coefficient for either present label is a_c/(2B). U class mass is n_ca*a_c/(2B), W class mass is n_ca*q_ca/(2B), and S class mass is n_ca*m/(2B). U task masses are the d-ordered sums of w_c*a_c. Mathematically .875≤a_c,m≤1. S retains the frozen `1-removed_mass` floating-point order, rather than recomputing m by another sum. Equality of separately summed floating-point masses uses the fixture tolerances below; the shared W/U cell-mass field itself must be identical.

For empty cells n,n0,n1,w,mass=0; pi, a and absent per-entry coefficients/statistics are null with reason `empty_cell`. For absent classes in observed cells, class mass=0 and class mean/variance/coefficient=null with reason `absent_class`; present class coefficients remain defined. Singleton variances are population variances and equal zero. No null class mean is multiplied into a tensor loss. Balanced cells satisfy mathematical W=U locally; one-class cells a=1. The zero-lambda engineering branch and any batch whose observed cells are all single-class delegate the entire live call to exact frozen `source_clean_step`. They are not implemented as an arithmetically equivalent U loop.

For present data gradients d_p the explanatory source recurrence is:

    z_p=d_p+wd*theta_p; k_p←k_p+1
    m_p←.9*m_p+.1*z_p; v_p←.999*v_p+.001*z_p^2
    theta_p←theta_p-.01*(m_p/(1-.9^k_p))/(sqrt(v_p/(1-.999^k_p))+1e-8)
    thetaEMA←rho*thetaEMA+(1-rho)*theta

Production calls the installed source Adam, never a manual recurrence. Source update_ema copies source buffers. `grad=None` skips Adam state initialization, decay and private-step increment; present tensor-zero remains active. One ordinary forward, one scalar backward, one live Adam and one live EMA occur per natural batch. No extra momentum tensors exist.

Useful decompositions, for interpretation and fixtures:

    W-U = lambda * sum_c w_c * Cov_(a~pi_c)(pi_ca,mean(ell|c,a))
    U-S = sum_c w_c*(a_c-m)*C_c
    grad(W)-grad(U) = lambda*sum_c w_c*sum_a
                     pi_ca*(pi_ca-kappa_c)*grad(mean(ell|c,a))

The within-cell residual has zero coefficient mass, not necessarily zero loss or gradient.

### 5.1 Pinned shared core interface

New package name: `experiments.cell_mass_bce_20261117`. No generic registry/plugin layer.

`loss_components(probabilities, targets, domains, *, arm='CELL-MASS-BCE-025', mixture_lambda=.25, diagnose=False)` returns `loss` (U), `clean_loss`, `same_class_loss` (H), `completion_loss` (W), `scaled_loss` (S), `cell_mass_loss` (U), `branch`, `ledger`. Only engineering accepts lambda0; formal lambda=.25. No N objective.

For a mixed-label lambda0 delegated engineering call, C/U/W/S equal the returned source CLEAN value, but H is not generally C. If the delegate observation did not compute H, same_class_loss is null with reason `not_computed_delegate`; do not relabel the frozen helper's lambda0 placeholder as an observed H. This null cannot occur in an ordinary formal mixed-cell call. All-single-class H=C. Loss logging/reduction must handle the explicit engineering null.

`train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` keeps C04's caller signature. It returns the six scalar losses above, branch/ledger/row_order_hash, optimizer_updates=ema_updates=ordinary_forwards=scalar_backwards=1, probe_forwards=view_forwards=algorithm_rng_draws=0, extra_autograd_sweeps=3 iff diagnosed else0, virtual_adam_steps=4 iff diagnosed else0, seconds, diagnostic_seconds, diagnostics or null, parameter_activity and adam_step_counters. Core returns names in the latter two dictionaries; the disk writer normalizes names through a single parameter dictionary.

Config is C04's fixed scientific keys with cycle_id=`cycle_05`, objective=`cell_mass_bce_v1`, arm=`CELL-MASS-BCE-025`; same dataset, mixture_lambda=.25, epochs50, batch_size40960, precision=`float32`, lr/betas/eps/WD/rho, steps_per_epoch/tail_rows, loader_seed_offset11000, gate_training_detached=true, probe_gate_full_derivative=false, algorithm_rng_draws0, data_hash=digest(source dataset definition), selection_hash=`c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89`. Reject schedule/switch/off-update keys. Do not change the source optimizer factory signature.

Formal context keys: purpose=`formal`, clock_kind=`formal`, dataset, seed, epoch=e, batch=b, global_step=(e−1)K+b, pre_loader_hash. Require 1≤e≤50, 1≤b≤K, actual full/tail cardinality, row_ids [B], and diagnose exactly b=1 at e∈{1,10,30,50}. Engineering context has purpose=`engineering_smoke` or CPU-only `engineering_check`, clock_kind=`engineering`, case_id, branch_role, engineering_call_ordinal, engineering_state_step, sample_epoch=1, sample_batch∈{1,K}, pre_loader_hash; no formal epoch/global_step. CPU fixtures state fixture_rows explicitly. Real calls always supply actual natural row IDs.

### 5.2 Same-state diagnostics

At the first ordinary batch of E1/10/30/50 of each new U trajectory, retain its single forward graph. Objectives are ordered C,S,U,W. Run autograd.grad only for C,S,W (`allow_unused=True`, retain_graph=True); live U backward supplies U gradients. Four disposable source-Adam operations start from the same populated or lazy pre-live Adam state and same pre-live parameter values, using the post-forward model buffers. Clones are created, stepped, measured and deleted sequentially. No clone forward, clone EMA, live extra Adam, RNG draw, loader advance, validation or new sampled batch is permitted. The live U Adam runs once, followed by its ordinary EMA.

For exact source-delegate branches, use C04's one-forward observation hook and optimizer pre-step hook pattern; remove the forward hook before cloning and always remove hooks on exit. Do not substitute another forward for a missed observation. Validate every virtual gradient/state is finite. Confirm frozen pre-Adam copies stay unchanged and the live optimizer and post-forward RNG remain unchanged by diagnostic work. U virtual post-model and post-optimizer hashes equal actual post-model and post-optimizer hashes exactly; update difference norm is exactly zero. Diagnosed versus undiagnosed restored calls must have exactly identical scientific state, gradients/activity/private steps, buffers/modes, RNG and loader state.

Diagnostic metadata, once per point: point_id, job/physical-attempt/context IDs, row_order_hash, pre_model/optimizer/RNG/loader hashes, C/S/U/W loss values, m and removed_mass, actual_objective=`U`, actual_post hashes, four virtual_post pairs, actual_virtual_exact and zero update difference, sweep/step counts, timing. Group dictionary is ALL plus frozen parameter-name prefixes before the first dot, with exact ordered names/numels/dtypes/shapes stored once per model schema. Do not repeat name lists or point hashes per metric.

For each group and objective record data-gradient norm and parameter-update norm; numbers of none/tensor-zero/nonzero parameters and coordinates; pre/post private-step min/max/uninitialized. Norms/dot products use detached FP64 accumulation in named-parameter order; missing coordinates contribute zero for comparisons without installing a zero .grad. Pairs are U−W, U−S and S−C: left/right norm, difference norm and cosine for data gradients and parameter updates; activity-mask difference counts; algebraic EMA difference norm=(1−rho)*update difference norm. Cosine is null with reason `zero_norm` if either norm is zero. Required check g_S≈m*g_C uses the FP32 tolerances below and equal None masks. The diagnostic records remain local response evidence at U-host states.

## 6. Exact affected modules and frozen source closure

Only these seven new source files are authorized for implementation under W:

| File under experiments/cell_mass_bce_20261117/ | Responsibility |
|---|---|
| __init__.py | Package marker; no dispatch or import-time mutation |
| core.py | U loss, call validation, source-delegate branches, C/S/U/W diagnostics |
| checks.py | Finite CPU mathematical/state/selection fixtures |
| protocol.py | C05 roots, seven-file design identity, source/reuse/data/review/owner and snapshot contracts |
| runner.py | Supervised workers/engineering, named telemetry, compact witnesses/admission, ACK/recovery/handoff |
| tables.py | Raw numeric exports, finite arithmetic/mapping checks, explicit numeric archive/parts |
| README.md | Commands, schemas, logical/physical mapping and handoff instructions |

Do not change any existing source file, frozen snapshot, source48 member, root-bound C04 wrapper, historical receipt or accepted P5 file. Parent/reviewer/runner author their stage receipts in A5/N5/R5 as assigned below. Coder implementation reports belong to A5, with scopes and actual native IDs; these are reports, not additional executable files. Later scientific reports/context/publication changes are parent-owned and outside coder source scope.

The exact immutable dependency set is the 48-entry `files` list in C04 attempt2/source_receipt.json, file-byte SHA256 `10f3000ae43e2a394b71cdb2febc291888f06bf216dd99c05ddd7b842186fec3`. Its sorted path/sha256 list has source identity `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`. Referencing this pinned explicit list avoids a second manually copied closure. All 48 W files matched during DESIGN. The new source closure is exactly their union with the seven files above: 55 unique repository files. Sort by repository-relative path; each entry is {path,sha256}; `source_hash=digest(files)`, where digest is frozen auxiliary protocol's SHA256 of sorted-key compact JSON. Exclude generated receipts, reports and snapshots from their own source hash.

Pin external `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/lib/python3.13/site-packages/torch/optim/adam.py` byte hash `b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c` separately in external_runtime, with Python/Torch/CUDA/package-lock/environment identities. requirements-phase0-3.lock hash is `5ca1c78d5a1f58636eb3a23b51fd2dbd550c9b2619b9fd918882d622070ebb13`. Do not claim these are hashes of unimplemented C05 files.

Existing numerical imports are frozen auxiliary protocol's digest/optimizer/load_protocol/make_model/setup/cpu_copy/state_digest/rng_state/restore_rng/evaluate/sha_file; frozen C04 loss_components/_statistics and numeric support; frozen TV core source_clean_step; conditional protocol update_ema; audited_data array_hash/split_row_fingerprint/row_ids_hash; and stage2.core synchronize. Frozen TV runner `serve(attempt)` is the actual generic storage function. Frozen guardian `request(attempt,command,**payload)` is the transport. None requires patching its module globals. Never call ensure_guardian, old schedule/run/prepare, autonomous.py, or root-bound C04 protocol writers. Keep source48's retired driver bytes as provenance only.

### 6.1 Seven-file design identity

P5 is W/research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05; A5 is W/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05. The six accepted file byte hashes are exactly A5/ASTRA_HANDOFF.json `files`. They matched during DESIGN. Their bundle hash is `bfd6230c9cae8971a3a8d793027fc54ba6209d193429340107bf64bb22dc4caf`.

For both six-file and seven-file bundles use SHA256(UTF8(json.dumps({logical_filename:SHA256(file_bytes)},sort_keys=True,separators=(',',':')))), no newline. The seventh mapping entry is DESIGN_SPECIFICATION.md→A5/DESIGN_SPECIFICATION.md; the other six point to P5. C04's list-of-pairs bundle serialization must not be reused. The completed file's own byte hash and seven-file bundle hash are returned externally, avoiding self-reference. Parent writes the existing-style A5/DESIGN_ACCEPTANCE.json with verdict=`COMPLETE_APPROVED_FOR_IMPLEMENTATION`, no blockers, actual designer ID, seven-name path mapping, filename/hash mapping, both bundle hashes, ASTRA_HANDOFF byte hash and parent acceptance. No second design-manifest framework.

### 6.2 Minimal disjoint coder scopes

Parallel coding is useful only across this pinned boundary: numerical coder owns __init__.py/core.py/checks.py; runtime coder owns protocol.py/runner.py/tables.py/README.md. The parent may assign one coder all seven if coordination would cost more. No minimum coder count. If split, both read this shared API before coding and neither edits the other's files. Runtime fixtures live in the responsible module's finite_checks; checks.py does not import a partially written runner. Each coder reports changed files, shared API compliance and basic CPU check results only. Every contributing coder's actual native ID is listed in review; reviewer must differ from all of them.

## 7. Implementation requirements

### 7.1 Environment, data and numeric reuse

Every project Python command explicitly uses `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B`, with PYTHONDONTWRITEBYTECODE=1; no base Python. Compile source strings in memory; no py_compile, __pycache__, disk state or new dependency. Coder may perform the finite CPU checks below, not engineering training or research analysis. Parent's A5/PARENT_ENVIRONMENT_CHECK.json records at13:47:55UTC Python3.13.5, Torch2.9.0+cu126, CUDA available with two devices, W free3175661568 bytes and /tmp free188973625344 bytes with ample inodes. It is environment evidence, not a resource-smoke/admission sample.

Prepared caches exist at `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache/{movielens,kuairand}.pkl`. Do not build/relocate/reprepare datasets. Adapt C04's Test-opaque projection: deserialize the existing trusted container in RAM, dereference only train_x/train_y/train_row_ids/val_x/val_y/val_row_ids/features, and immediately return a SimpleNamespace with only those attributes. Do not iterate all bundle fields or call whole-bundle fingerprints/asdict, and never dereference Test. Verify those train/validation arrays, row IDs, row fingerprints, feature encoder/scaler and natural class counts against the exact protocol data manifests in the research manifest. Verify the relocated raw CSV hashes at preparation as C04 did; DESIGN need not reread those raw datasets.

Use P5/EXPERIMENT_MANIFEST.json `reused_controls` as the sole list of 30 control result/epoch pointers. Exactly ten each C/W/S and one per dataset/seed/arm; reject NQR entries. All 1,530 pointed files were present and byte-hash-valid during DESIGN. At preparation repeat the receipt verification and verify their dataset/arm/seed/key, epochs1..50, selected hashes/validation, source identities and order/init matches. Keep original CLEAN source identity; W/S retain source48. Earliest strict maxima must reconstruct from all50 epochs. Historical missing RNG fields remain null with reason `not_recorded_in_source`.

The C03 reused_reference_index.json byte hash is `e8d67c6b18ad9ddbfa19b752e35c6ab051e3170a3d28df635b86b27922441fdb`; its initial_hashes and epoch_order_hashes provide the ten initial model hashes and500 expected order hashes. Bind all50 W/S epoch hashes and control selection to the supplied pointers. Write R5/reused_reference_index.json once with deterministic sorted control entries, original pointers/byte hashes, numeric archive mapping, source identities, initial/order dictionaries and selection provenance; its actual byte hash becomes C05 controls_index_hash. The C03 index hash is a separate input identity, not mislabeled as the new 30-control hash. No old owner IPC. Missing required numeric evidence causes technical PAUSE; no substitute trajectory or automatic control rerun.

### 7.2 Physical roots, writes and publication mapping

| Symbol | Exact root and role |
|---|---|
| W | /home/xzf/Code/phase0-3_tvg_rm_20261116, source workspace |
| O | /home/xzf/Code/phase0-3, control workspace |
| N5 | A5/attempt1, compact mapping/review/publication metadata |
| R5 | /tmp/phase0-3-c05-01a1e9a9/attempt1, canonical physical execution and owner root |
| S5 | /tmp/phase0-3-c05-01a1e9a9/publication, archive/index/part staging |

New protocol `new_root(attempt=R5)` requires absolute equality to R5 and resolved equality to R5; `new_path(path)` permits resolved descendants of R5 only. N5 and S5 have separate exact-root metadata/publication writer checks, not a generic arbitrary-root escape. No symlink links into old guards or across these roots. `json_safe` rejects tensors, ndarrays, bytes/bytearray and nonfinite floats; null uses JSON null, never NaN. No model/EMA/Adam/RNG/loader/BytesIO serialization reaches disk. Frozen source text/JSON is ordinary code/provenance, not training state.

Parent/prepare writes N5/ROOT_MAPPING.json and identical public mapping in R5 with physical_root=R5, logical_root=N5, publication_staging_root=S5, cycle/attempt, source/design/research/reuse identities and numeric_only=true. Paths in batches/epochs/transactions/ACKs are R5-relative; logical path=N5-relative suffix. No `.relative_to(W)` on /tmp paths. Source snapshot mapping is W/<relative source path>→R5/source_snapshot/<relative source path>. Verify all55 hashes and installed Adam before engineering and on launch/restore; execute pinned W modules with cwd=W and report actual module files. Never change pinned source after snapshot/run_manifest freeze. Needed source correction returns through review and a parent-defined attempt decision; no mutation under an existing source identity.

Exact historical read allowlist is the manifest's explicit control file pointers, C03 reference index/data manifests, C04 source receipt and its48 listed inputs, C04 publication/storage-availability/table/archive receipts and the four bootstrap profiles below; new source reads add only the seven C05 files. The packer enumerates those explicit inputs; it never traverses old artifact roots indiscriminately. If an original control file is unavailable later, recover only its exact numeric member from the already-bound C04 archive using expected bytes/hash, into R5/reused_controls; no model unpickling and no foreign writes/deletes.

R5 layout keeps jobs/, runs/<key>/attempt<number>/, smoke/<case>/attempt<number>/, transactions/<key>/attempt<number>/, acks/<key>/, telemetry/, dictionaries/, profiles/, tables/, source_snapshot/, run_manifest.json, source_receipt.json, reused_reference_index.json and compact journals. Every new physical attempt remains present. N5 stores root mapping and parent/review/publication receipts; raw run trees stay R5. Credential guardian.json and STORAGE_OWNER.json are private0600 and excluded from package; public binding contains endpoint hash, PID/session and source/root identities without address/authkey.

### 7.3 Launch APIs and gates

Preserve C04's callable shapes in the new modules: protocol config(dataset,arm), job(dataset,arm,seed,source_hash,purpose='formal',case_id=None), prepare(attempt,review_path), validate_attempt(attempt=R5,require_engineering=False), request(attempt,command,**payload), verify_selection(result,epochs), component_hashes(snapshot), scientific_identity(snapshot), validate_epoch_state(snapshot,record). Runtime restore(model,ema,opt,loader,state,device), commit_snapshot(attempt,key,snapshot,record,path,*,attempt_number), reconcile_ack(attempt,key,snapshot), admission(current,incoming_measured_full_process_bytes=0,*,pending=False,already_resident_bytes=0), smoke/run(attempt=R5,review_path=None,devices=(0,1),recovery=False,recovery_review=None), handoff(attempt=R5). Retain smoke's optional case_id for reviewed recovery of an unfinished case. No `cold=True` physical-fit bypass.

Job identity has cycle=`cycle_05`, dataset, arm, seed, fixed config/config_hash, source_hash, design_bundle_sha256, controls_index_hash, purpose; engineering adds case_id. key=digest(the identity before key). Physical registration attempt is not a new scientific job. Deterministic formal queue: KR seeds2022..2026 then ML seeds2022..2026. Check both state files and all four sentinels before dispatch/admission and safe boundaries; unreadable controls mean PAUSE. Scheduled stop-after-C05 does not trigger immediate latch. Only parent writes stop controls.

Runner CLI commands are prepare, smoke, run, handoff, _worker and _case, retaining C04 --root=W, --attempt=R5, --review, --recovery-review, --recover, --devices=0,1, --device, --key, --case-id, --dataset, --arm. `tables` has export and pack commands; `checks` has check. Each module's finite_checks is callable by coder CPU checks. README provides exact project-Python commands. The parent uses native fresh DEFAULT mechanical runner gpt-6-luna/high, fork_context=false, preserving the five-seed runner-role resolution. Workers are supervised mechanical processes; no detached/session-start flags or AI dispatch by runtime.

Concrete launch sequence, all cwd=W, with the full absolute project Python and environment prefix on EVERY invocation (short labels below denote paths, not shell environment overrides):

    env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.checks check
    env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.runner prepare --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1 --review A5/IMPLEMENTATION_REVIEW.json

After parent owner acceptance, same Python/environment prefix, runner `smoke --attempt R5 --review A5/IMPLEMENTATION_REVIEW.json --devices 0,1`; after consumed resource signoff/release, runner `run --attempt R5 --review R5/ENGINEERING_SIGNOFF.json --devices 0,1`. Recovery adds `--recover --recovery-review <fresh bound receipt>` and engineering optionally `--case-id <unfinished case>`. `runner handoff --attempt R5`, `tables export --attempt R5`, then `tables pack --attempt R5` supply raw delivery. README expands A5/R5 in these commands. No command in DESIGN is executed by the designer.

Before preparation/engineering require accepted design, distinct independent implementation review PASS or PASS WITH FIXES with all blockers closed, actual coder/reviewer IDs and exact reviewed source. Before formal run additionally require ENGINEERING_SIGNOFF and ENGINEERING_RELEASE_ACCEPTANCE with the consumed fields below. Code review fields: verdict, blocking_issues, actual parent_id, designer_id, coder_ids (one or more unique IDs), reviewer_id not in coder_ids and not designer_id, source_hash/source_files/external_runtime, design_bundle_sha256, research_bundle_sha256, controls inputs, coder_report path/hash list, finite-check receipt path/hash, test_accessed=false. No inherited C04 len(coders)>=2 rule. Resource reviewer is fresh and distinct from coders and code reviewer. `PASS WITH FIXES` requires explicit fixes_closed references and zero outstanding blocking issues; a textual verdict alone never releases RUN.

### 7.4 Parent-owned foreground RAM storage and transactions

After code review/prepare the original parent starts the existing generic `experiments.task_view_gradient_residual_momentum_20261116.runner.serve(R5)` in its own foreground tool session, using explicit project Python/PYTHONDONTWRITEBYTECODE. Call serve directly; do not call its ensure_guardian or CLI dispatcher/old scheduler. The reviewed function accepts an arbitrary actual root, writes generic owner metadata there, and keeps all tensors in RAM. It is a new C05 storage server, not an old-owner restart or training driver. Record actual PID, tool_session, endpoint_hash=digest({address,pid}), server_file_sha256=`5ddcb68c206c6d78ad6c3cecf59441081aa9a9ddde3233ade4af55d158b32cfa`, source/design/root binding. No assumed PID/session values. Preserve TV PID1025015 and all co-tenants.

Actual server methods and responses: register(key,job,pid)→{attempt,state}; status→{pid,retained_states,retained_bytes,jobs}; allocated(key,peak_memory_bytes)→{status:'ALLOCATED'}; commit(key,state,epoch_file)→{status:'RAM_COMMITTED',epoch}; get(key)→{state,result}; paused(key,reason), failed(key,failure), recover(key), complete(key,result), and engineering-only release(key). No server hash/attempt token ACK fields exist. Client derives and verifies hashes from returned state and the actual epoch_commit_ledger.jsonl. Never fabricate owner methods or silently infer missing hashes from metadata.

Snapshot schema retains C04: job/source_hash/design_bundle_sha256/controls_index_hash/owner_endpoint_hash; CPU model/ema/optimizer/rng/loader; source_modes; selected_models/epochs/validation; best; common_validation; initial_hash/initial_ema_hash/initial_optimizer_hash; purpose/clock_kind/epoch/global_step/next_epoch/scientific_epochs_completed; counts; operations; adam_step_counters/last_parameter_activity; branch_counts and measured costs. Engineering adds engineering_commit_ordinal/engineering_state_step/sample_epoch/sample_batches/sample_order and selection_clock. Counts has exactly integer optimizer_updates and ema_updates. CPU snapshots are independent deep copies; NumPy order/RNG, tensors and BytesIO stay RAM. Restore model/EMA/Adam, private loader and per-module train flags, then RNG LAST; no RNG-consuming operation between that restoration and the next intended source work.

At each complete formal epoch e, global_step=e*K, next_epoch=e+1, scientific_epochs_completed=e, counts={optimizer_updates:e*K,ema_updates:e*K}. Operation counters are exact integer totals: training_rows=e*N, task_entries=2eN, ordinary_forwards=scalar_backwards=live_adam_steps=live_ema_steps=eK; diagnostic_points=# registered epochs≤e, extra_autograd_sweeps=3*points, virtual_adam_steps=4*points; probe/view/algorithm_rng_draws=0. Epoch0 has zeros. Snapshot epoch fields are storage clocks, not inferred completion from worker exit.

An epoch transaction is: finish all K ordinary calls; evaluate both source streams preserving RNG/module flags; update independent strict-AUC selections; deep CPU capture; write immutable numeric epoch record binding batch log/hash, continuation_hash and component hashes; measure RAM BytesIO roundtrip and verify digest; write transaction record; send commit; receive ACK; obtain status and get; validate returned CPU snapshot and hashes/counts/job/owner plus one actual server journal row; then write numeric ACK. Preserve C04 commit chronology and lost-reply reconciliation. Journal has key,epoch,attempt,epoch_file,time,status, not generated scientific hashes. A failed transport is reconciled by status/get/journal before any replay. Never resend a conflicting commit or replay a live optimizer call because telemetry failed.

`continuation_hash=state_digest(full snapshot)` includes actual costs; `scientific_identity_hash` uses C04 IDENTITY_FIELDS plus branch_counts and operations excluding only the three diagnostic operation counts, and excludes costs/timestamps/diagnostic records. Component hashes are model,EMA,Adam,RNG,loader and selected models. Exact equality fixtures also compare dtype,shape,ordered keys and values: the frozen digest alone does not encode all such metadata. Validation/selection values, gradients and mode identities are checked as well. No research metric is used for stopping.

On complete, require all50 ACKs/epoch hashes and selected identities first, then call complete. Generic storage compacts to exactly selected_models/selected_epochs/selected_validation (two selected model states per job), with numeric result retained. No release of formal states. Engineering-only release is gated by parent acceptance of all four cases and independent resource review. Parent directly verifies 500 formal ACKs and20 selected model identities before closing runner, and rechecks current availability before evidence review/publication/final delivery.

Parent actively polls the actual foreground owner tool session at intervals≤60 seconds through RUN, ANALYZE, report, review, publication and delivery; use short tool waits and record transition liveness. Polling responsibility remains with parent, including while children work. Retain the same owner on technical recovery. Finish an atomic call safely; commit a valid completed epoch or preserve the previous full ACK, counting returned uncommitted work separately. Recovery verifies same owner/source/design/review, exact continuation and RNG/loader identities, and actual dead/stale startup facts. No completed-job replay, owner replacement, automatic rerun after owner loss or disk checkpoint. A lost owner with retained state is technical PAUSE with precise unavailable-state disclosure.

### 7.5 Named telemetry, terminal records and admission

Replace C04's anonymous `_ok(code)` at the actual query boundary. Query names are nvmlInit_v2, nvmlDeviceGetHandleByIndex_v2, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetEnforcedPowerLimit and nvmlDeviceGetComputeRunningProcesses_v2. Record name, return code, device, calling/observed PID, phase, source/job/registration-attempt identity, incident/sample-attempt IDs, wall/monotonic timestamps, arguments and requested/returned process-list capacity. Use the existing v2 ABI and capacity256; on insufficient-size/error record the returned count and reject that whole sample rather than claiming a truncated process list is complete. Query failure is recorded before propagation. Owner/job dictionaries and query-name definitions are stored once, referenced by compact IDs.

Each incident has at most three TOTAL fresh full-sample attempts, with a newly acquired handle and≤.1s separation. Success requires all required memory/power/limit/process fields and current host memory from the same full attempt; no mixed-time success, cached substitution or zero sentinel. Failed attempts and later successful attempts remain recorded. At admission evaluation sample age from monotonic end time must be≤1s; otherwise take a fresh sample. Persistent failure causes technical PAUSE. Retry samples only, never optimizer/model/EMA work. Include CUDA-usable capacity from the actual target hardware identity and positive, finite totals/limits. Sampling does not create a CUDA context on another worker's behalf.

Preserve C04's .005s worker profile sampling and explicit synchronized phase boundaries. Store raw measurements once in per-process compressed telemetry logs, including failed samples and each required phase observation. Profile summaries/witnesses refer to those logs rather than embedding them. Serialized writes to each log must retain query/measurement ordering. Controller observations use their own log; unknown/failure records remain parseable on terminal paths. Do not add fallback telemetry providers or generalized retry infrastructure.

Independent per-card admission is exactly:

    fresh_known_used/NVML_total < .85
    AND fresh_known_board_power/enforced_limit < .85
    AND current_host_used_fraction < .90
    AND no unresolved pending allocation on this card
    AND used_bytes + max(0, measured_incoming_bytes-already_resident_bytes)
                         <= measured_CUDA_usable_total_bytes

Already_resident_bytes is zero for a new formal process. Only an explicitly observed partially resident engineering process may subtract its own current residency; do not double count it. No projected85% fit, predicted power, fixed worker cap, both-card gate or admission from unknown/stale measurements. GPU utilization is recorded, not gated. Re-sample after each launch and FULL allocation ACK. Pending identity is exact new PID + job key + expected next registration-attempt + current owner RUNNING generation, with that PID alive. Release pending only when status matches all identities and allocated=true was sent after its first real FULL batch and lazy Adam allocation. Status's stale previous allocated flag never suffices; timeout alone never releases pending. Dead startup is resolved using actual child exit/process/owner facts. Smoke also uses pending allocation protection, not C04's one-active-case-per-card cap.

For the first source-specific smoke before a C05 measurement exists, use the four existing C04 engineering profiles ONLY as explicitly labeled `historical_bootstrap` measured-residency witnesses, never as current telemetry or formal C05 signoff. They cover the same architecture/batch/FP32/full-tail/validation/clone/serialization path and both GPUs. C04 ENGINEERING_SIGNOFF.json byte hash=`ee7dc227bdcfc73798e5920da6eee441574378f25fe894cbf6d0c4c1b22868ac`; its four profile pointers/hashes are authoritative:

| C04 smoke case/attempt1/case_profile.json | SHA256 | Full process peak bytes |
|---|---|---:|
| case1_gpu0_movielens | 1f196d597c2dbadf70eb6c748d46d8f80e5c1d1f4275fca337922627dc074450 |1675624448|
| case2_gpu0_kuairand | 9d841b078b6b2ded6a9b5e022630a0e35040d3533722118c0347262e16929081 |2027945984|
| case3_gpu1_movielens | 793856cbdb3c6d3063165bf83b91bc7d28f0d2c987a305867cc687964a296f18 |1675624448|
| case4_gpu1_kuairand | a92798765283423c60099955420ceedfc79ac2db250b477b0df48dfab34ca895 |2027945984|

These profiles have RTX3090/compute8.6/CUDA-usable25307578368-byte identity. Bootstrap incoming measured envelope is the maximum of full_process_peak and reserved_peak+non_torch_peak over all four (2027945984 bytes in those receipts). Record this provenance/scope explicitly. Current hardware must match; bootstrap still applies the full predicate and contains no cold/fit exemption. As C05 phase witnesses arrive, engineering uses the maximum of bootstrap and observed compatible C05 residency; formal admissions use only reviewed C05 witnesses. This historical number is an initial engineering estimate, not a guarantee of C05 peak residency. Measure C05's actual full path before releasing formal work.

One append-only witness registry replaces duplicated profile-contributor histories. Each immutable witness contains witness_id=digest(payload excluding witness_id), producer PID/job/attempt/purpose, dataset/arm/source/design/environment/hardware/device, full_process_peak_bytes, reserved_peak_bytes, non_torch_peak_bytes, allocated_peak_bytes, host RSS, phase coverage and raw-log path/range/hash. Scope keys include dataset, U arm, source/design/environment/hardware; device coverage requires an actual reviewed target-card case. The conservative measured incoming value is max(full_process_peak,reserved_peak+non_torch_peak) over the compatible bound prefix. Do not sum incompatible workers or infer a new power estimate.

Registry producer journals are single-writer and the controller appends each discovered immutable witness once to profiles/witness_registry.jsonl, validating IDs and raw references. Envelope versions contain scope keys, registry byte/record prefix boundary and hash, maxima and IDs attaining the full-process, reserved/non-Torch and conservative-envelope maxima. Reconstruct compatible membership by filtering that exact prefix; max IDs do not replace the registry. Each admission/denial row stores decision ID/time, queue candidate/device, exact current sample ID, incoming/remaining bytes, pending identity, envelope version/max witness references, every predicate Boolean and reasons. Recording a denial is required even when no worker launches. RESOURCE_EXECUTION_AUDIT contains references/hashes and recomputed summaries, not a second copy of admission or contributor rows. New actual C05 observations may raise envelopes; never overwrite earlier versions or decisions.

STARTED and TERMINAL_CORE records use only local process facts. Enter the outer try/finally before Profile construction/registration-dependent work can fail; initialize optional profile/owner/result variables to null. Local fields include key/purpose/PID/device/known registration attempt, wall/monotonic start/end and elapsed, source/design, returned train calls/operations, last completed operation, current pending call and ordinal, last observed ACK, original exception type/message/traceback reference, and unknown/null reasons. Set call_pending before calling train_step and clear only after its returned receipt is recorded. An unreturned call contributes unknown work, not a fabricated update count. Original exception remains primary.

Write the terminal core in finally independently of profile.close, final telemetry, owner queries and other cleanup; isolate each cleanup and append its error separately. Do not call telemetry or owner status to decide whether the core can be written. A cleanup failure following successful training makes the physical attempt status/report explicit without deleting completed ACKs. Supervisor separately records launch/observed exit time, PID, return code/signal, wall duration, last observed owner/ACK facts and whether worker terminal data exists. If killed before finally, worker fields stay null and supervisor facts remain available. Do not add overlapping worker/controller durations into a single elapsed total.

### 7.6 Actual engineering plan and consumed resource signoff

Four seed2022 engineering cases, all U: case1_gpu0_movielens, case2_gpu0_kuairand, case3_gpu1_movielens, case4_gpu1_kuairand. This keeps C04 case names and its useful eight-call proof skeleton. Each uses the source E1 private permutation and exact FULL first batch plus actual natural tail; engineering logical clock is separate from formal epochs.

| Physical call | State step | Sample | Diagnostics | Role |
|---:|---:|---|---|---|
|1|1|FULL|on|U prefix; lazy Adam; actual FULL allocation ACK|
|2|2|tail|off|U prefix; CPU capture/BytesIO/owner commit1/get|
|3|3|FULL|off|restore prefix, exact source CLEAN fork|
|4|3|FULL|off|restore prefix, U lambda0 fork; exact identity to call3|
|5|3|FULL|off|restore prefix, expected U suffix|
|6|4|tail|on|expected suffix; populated diagnostics; full online+EMA validation|
|7|3|FULL|off|re-register/get/BytesIO/restore actual ACK prefix; replay suffix|
|8|4|tail|off|replay suffix; same full validation; commit2/get|

Calls3/4 and the expected suffix are disposable branches, never formal updates. Prefix snapshots remain immutable. Call6 and call8 observations compare exactly, including gradients/activity/private steps; expected and restored suffix scientific identities and validation/selection match exactly. Actual/virtual U hashes match at calls1 and6. Final engineering snapshot has epoch=engineering_commit_ordinal=2, engineering_state_step=global_step=4, counts4/4, scientific_epochs_completed=0, next_epoch=null. Restored logical state contains only calls1,2,7,8; its diagnostic counters contain prefix point1, whereas physical evidence contains both points1 and6. Record that distinction, not a fabricated second logical diagnostic.

Per case verified physical counts are8 returned train calls,2 diagnostic points,6 extra sweeps,8 virtual Adam. All four cases:32 calls/8 points/24 sweeps/32 virtual steps, zero scientific epochs, 1191074 training rows and2382148 task entries. Expected+replayed full validation supplies80 actual source validation chunks and2806304 validation rows over both streams across all cases. Per-case case.json and proof/log references bind actual physical counts; earlier failed/replayed work is additional, never discarded to force these totals. No extra GPU fixture or capability pilot.

Measure context/model/EMA/lazy and populated Adam, ordinary and diagnosed FULL/tail, both full validation streams, CPU snapshot copy, BytesIO serialization, commit/status/get, restore and compact retained states. Retain actual full-process NVML residency, Torch allocated/reserved, non-Torch estimate, CUDA-usable total, per-phase host RSS/peak and wall/transfer bytes/durations. Parent directly accepts eight engineering ACKs and all four exact proofs before release; no engineering state remains to distort formal host RAM accounting.

R5/ENGINEERING_SIGNOFF.json is written by a fresh independent resource reviewer and CONSUMED by validate_attempt(require_engineering=True). Required fields: verdict/blocking_issues/fixes_closed; actual reviewer_id distinct from all coder_ids and code_reviewer_id; exact source/design/research/environment/owner/root identities; four case/proof/log receipt hashes and four actual C05 profile/witness references; required matrix and phase coverage; per-case verified_plan_counts; physical totals including additional attempts and null partials; scientific_epochs_completed=0;80 useful engineering validation chunks; source_lambda_zero_exact,diagnostics_neutral_exact,prefix_immutable,actual_virtual_exact; bound C05 envelopes/registry prefix/raw sample references; terminal-core/supervisor evidence; code-review receipt; parent_engineering_acceptance receipt/hash; test_accessed=false. Validate every field and count against actual receipts rather than merely checking names. Do not accept a C04 profile as one of the four C05 cases.

Parent ENGINEERING_RELEASE_ACCEPTANCE lists exactly these four engineering keys, signoff path/hash, parent_accepted=true, accepted ACKs/proofs and release status. Only after that may release remove their RAM state. Parent-owned owner remains alive; no formal selected models are released. Before run, validate source snapshot, distinct code/resource identities, all signoff receipts, actual release and same foreground owner.

### 7.7 Capacity and packaging

Measure free bytes/inodes on actual R5 and W filesystems at setup, after engineering, each epoch commit and before pack. Log category bytes (runs/epochs/telemetry/witnesses/admissions/tables/controls/archive) and measured remaining-growth estimate from completed versus outstanding work, with unknown physical work separate. The8GiB /tmp planning allocation (2 raw/tables/controls +2 archive/parts +4 transient) and128–512MiB compressed forecast are estimates, not output caps. If remaining physical capacity cannot finish the next transaction/package, PAUSE with exact need; do not truncate evidence, shorten training, delete foreign files or spill state to disk.

Pack in S5 using tables.pack after raw export and parent storage acceptance. Explicit map: R5/<suffix>→archive `attempt1/<suffix>`; each of the30 verified controls→`reused_controls/<symbol>/<dataset>/seed<seed>/<key>/result.json` and epoch_01..50.json; accepted seven design inputs→`design/<filename>`; public stage/source/environment/review/storage-availability/publication receipts→`receipts/<unique logical name>`; W source55 snapshot→`attempt1/source_snapshot/<repo-relative path>`. Include C04 PUBLICATION_RECEIPT.json and selected-availability update; bind its existing archive rather than repack C04 admission/audit/retired attempts. Only regular files with checked root/expected hash enter the list; reject symlinks, traversal, credentials, opaque tensor/training-state extensions and nonnumeric logs/records. Code/Markdown/source JSON are allowed provenance. Public credential redaction must retain endpoint_hash/source/PID/session/root facts.

Store member logical path, absolute source path, category, bytes and SHA256 in the existing-style RAW_RUNS_MANIFEST.json, plus complete archive bytes/hash and ordered8MiB part bytes/hashes. Read back archive headers/members to check mapping, lengths and hashes; do not deserialize state. Manifest and part-index bytes do not include themselves recursively. Keep the full archive/index/parts in S5; only published parts and compact manifests/receipts go into N5, without a second unsplit archive in W. Derived tables use normalized dictionaries; a reviewer can reconstruct a denormalized view in /tmp.

Before copying/staging publication parts, use measured compressed size A and account for `3*A +256MiB +1GiB free headroom` on W, or an explicitly measured parent publication/object-staging strategy. Actual free3175661568 bytes is an observation, not launch/capacity permission. Parent publishes reviewed code/report/context/numeric artifacts with expected-head protection to the existing main target, includes C04 publication receipt, and verifies actual remote commit and local alignment. Keep R5/S5 evidence and foreground owner through verified publication/delivery; no automatic foreign cleanup or storage-owner shutdown.

## 8. Required ablations and minimal meaningful fixtures

The complete trained comparison is U/C/W/S only. C, S and W are the ablations/references and are reused numerically; NQR remains historical context. No extra trained arm is required or authorized. Engineering lambda0 and all-single-class branches are identity fixtures, not scientific ablation trajectories.

Finite CPU acceptance consists of the following checks, implemented once in the owning modules. Use actual installed source Adam and source numerical helpers; CPU fixture checks do not access prepared data or Test. Tolerances for algebraic identities only: FP64 atol1e-12/rtol1e-10; FP32 atol2e-6/rtol2e-5. Exact state comparisons use tensor/array equality with dtype/shape, ordered structures and frozen state_digest; hashes/selection/None masks/operation counts/zero actual-virtual difference have no approximate tolerance.

1. A small explicit FP64 probability/label/domain fixture with unequal domain sizes, unequal class proportions and distinct task proportions verifies U against direct hand-enumerated coefficient sums, W−U covariance and U−S identities, six W/U cell masses and task/global masses, class-mass difference, all2B natural entries and finite gradients. Include balanced and one-class observed cells plus an empty domain in the same finite fixture set. Check singleton population variance, all declared null reasons, lambda0, all-single-class and boundary probabilities0/1 with source BCE behavior. Reuse frozen W/S scalar output as the exact reference, not a rewritten W objective. FP32 version checks the actual production reduction and prescribed algebraic tolerances. No gradcheck sweep or exhaustive random fuzzing.
2. Reuse C04's minimal stochastic/detached-gate fixture: shared and gate parameters, present tensor-zero parameter, unused parameter, a forward buffer and one ordinary RNG draw. On a populated Adam state compare U lambda0 and all-single-class calls exactly to source CLEAN, with diagnostics on/off. Verify None versus zero private counters/decay, one forward/live Adam/EMA, source buffers and RNG. For ordinary mixed labels compare diagnosed/undiagnosed restored U exactly and all four virtual source steps, with immutable pre-Adam copies and g_S≈m*g_C. A tiny instance of the real source architecture separately verifies gate embedding detach, layout [F,B]→[B,2], finite source backward and expected parameter prefix dictionary; reuse C04's useful source-model fixture rather than adding a second suite.
3. RAM-only BytesIO/deep-copy restore of the populated fixture reproduces a short suffix exactly, including optimizer moments/private steps, buffers/flags, gradients/activity, RNG and private loader. Snapshot/scientific identity distinctions and clocks are checked. Use a small in-memory fake transport with the actual generic server's response/journal schema to cover successful epoch ACK and lost-reply reconciliation, plus stale registration-attempt allocated=true rejection. Do not test against an old owner or launch a storage server from coder checks.
4. A tiny explicit six-cell validation fixture verifies source ordinary/unscaled metrics, strict ties selecting the earliest epoch independently for online/EMA, and LogLoss bound to the AUC-selected state. Undefined AUC is null in fixture metrics; a required undefined formal validation cell causes PAUSE. Check 17-family enumeration, metric34 epoch having no favorable direction, five-seed paired mean/SD/t4 interval arithmetic and declared logical table counts. No scientific winner conclusion in checks/tables.
5. Runtime finite fixtures inject one named NVML failure followed by success, and three failures; assert complete-sample retry bound, code/query/phase/attempt capture, no stale/unknown admission and no optimizer replay. Inject sampler-close and owner-status failures into the worker exit path; assert terminal local facts and original exception survive, with cleanup errors separate and unknown call work null. Fixed numeric witness/prefix fixtures reproduce full-list envelope and per-card predicate, including .85/.90 boundaries, physical100% fit, pending current/stale identity and independently eligible cards. A small numeric file/map fixture checks R5/N5/S5 containment and explicit archive member paths/credential exclusion. These fixtures are the requested actual-failure checks, not a general infrastructure test framework.

Run each relevant check once on the final source. Repeat only after a failure, source change or unresolved concern affecting this comparison. Coder reports syntax/basic CPU results; independent reviewer inspects actual code and these evidence paths. Actual full/tail GPU/RAM engineering belongs only to RUN after review.

## 9. Expected logging, raw export and analysis contract

### 9.1 Exact useful scientific counts

| Quantity | ML | KR | New useful total |
|---|---:|---:|---:|
| Trajectories |5|5|10|
| Committed epochs |250|250|500|
| Ordinary forward/backward/live Adam/live EMA, each |5000|12250|17250|
| Training rows |200041750|501533000|701574750|
| Task entries |400083500|1003066000|1403149500|
| Scheduled diagnostic points |20|20|40|
| Extra autograd sweeps |60|60|120|
| Virtual source Adam steps |80|80|160|
| Validation chunks, both streams |1500|3500|5000|
| Validation rows, both streams |50010500|125383500|175394000|
| Selected model pairs/models |5/10|5/10|10/20|

Per trajectory ML1000/KR2450 updates; each has50 epochs,4 diagnostic points,12 extra sweeps and16 virtual steps. No ordinary probe/view forwards or algorithm RNG draws. Forecast selected tensor bytes95508960 is the source-size forecast; acceptance uses actual tensor bytes, not a fabricated constant measurement. Reused30 controls contribute no new formal/physical training completion. Scientific count reconciliation uses acknowledged unique epoch transactions, not the sum of all worker calls.

Keep physical counts separately: successful32-call engineering plan; every earlier engineering/formal failure; uncommitted returned calls; replay; restored suffix branches; failed/unreturned calls with null unknown work; diagnostic/validation/capture/serialization/transfer cost. Physical returned totals must be at least their useful operations; do not subtract unknowns into a precise total or turn engineering calls into scientific epochs. Keep foreground wall versus summed worker times separate and preserve contention context. No claim of an isolated efficiency gain from historical timings.

### 9.2 Raw record schema and normalization

Record job metadata once per key (dataset/arm/seed/config/source/design/reuse/data/selection/environment identities). Record ordered parameter/group schema once per model dictionary. Per-call compact records contain job/physical-attempt/context references; e,b,global_step,B,entries,row_order_hash, pre_loader_hash; C/H/W/S/U scalar losses; branch; six-cell ledger; elapsed/diagnostic seconds; integer operation deltas/cumulative logical counters; parameter activity/private steps in dictionary order; diagnostic point reference or null; test_accessed=false. Full lists of parameter names, entire jobs and all virtual hashes must not be repeated per metric. No individual predictions, labels, embeddings or state tensors are written. Aggregate counts/masses/statistics are allowed numeric observations.

For each of the six cell ledgers on every batch store n/n0/n1, pi0/pi1, w, opportunity, a, kappa, natural/W/U/S cell mass, present-class q0/q1, natural/W/U/S per-entry coefficients and class masses, removed mass, and null statuses. Batch stores m,total_mass, task0/task1 masses for W/U/S and exact W/U shared-cell-mass identity. Use null coefficients for absent classes and zero actual class masses. Statistics mu0/mu1,var0/var1,bbar0/bbar1,H,Q_R are observed only on scheduled points; otherwise null `not_scheduled`, with empty/absent/requires_both_classes taking precedence. Detached BCE statistics are not extra model forwards or autograd sweeps.

Store500 immutable epoch records/ACKs/transactions with source/order/continuation/scientific/component/selected identities, validation and actual cost receipts;17250 useful batch records;103500 derived cell-weight rows. Batch/phase/control costs remain reconstructable through receipt joins. Parameter_names and point-level hash maps exist once in dictionaries/diagnostic-point records; long scalar diagnostics reference point_id/group_id/objective_or_pair/metric/value/null_reason. Tables/schema.json documents keys, join paths, units, formulas, null reasons and source raw references. Raw facts live once; derived CSV receipts identify their inputs and row counts. RESOURCE_EXECUTION_AUDIT only summarizes/references complete raw measurement/decision/witness/physical journals.

### 9.3 Numeric tables and paired families

Analyze40 unique trajectories (10 new+30 reused). tables.export produces receipt-bound ENDPOINTS400 rows, ENDPOINT_CELLS2400, CURVES4000 (all50×2 streams), CELL_CURVES24000; SELECTION_STRUCTURE80 stream selections; WEIGHTS103500; MASS_SUMMARIES including row-weighted and update-weighted summaries; STATISTICS and STATISTIC_MEANS; COMMON_STATE_DIAGNOSTICS, DIAGNOSTIC_MEANS and DIAGNOSTIC_AVAILABILITY; PHASE_COUNTERS500 useful epoch rows; PHYSICAL_ATTEMPTS/COSTS/SMOKE_COSTS; compact resource audit; dictionaries/schema and TABLE_RECEIPT.json. Availability rows distinguish observed C05 local diagnostics from unavailable historical common-state data. If a logical view is stored normalized, its receipt gives logical row count and documented joins. Never fabricate missing historical diagnostics.

Endpoint order is selected,E1,E10,E30,E50. Streams are online,fixed_ema. Six matched-arm families U−C,U−W,U−S,W−S,S−C,W−C in EACH stream yield12; U/C/W/S fixed_ema−online yield4; U fixed_ema−C online yields1. Exactly17 families. Use the manifest's ordered34 metric_names (33 source AUC/LL metrics plus epoch) exactly. This yields28900 ALL_PAIRED_SEEDS rows (17×2 datasets×5 endpoints×34 metrics×5 seeds) and5780 ALL_PAIRED_SUMMARY rows. No N family, cross-task pseudo-replicate or per-epoch replication is added.

Each paired-seed row binds left/right trajectory+endpoint+stream references, actual selected/fixed epoch/hash, values and delta, available/null_reason, metric and favorable_direction. Summaries have n=5, arithmetic mean, sample SD ddof1, mean±2.7764451051977987*SD/sqrt(5), positive/negative/exact-zero and favorable/unfavorable counts. AUC favors positive, LL favors negative; epoch direction/favorable/unfavorable is null `no_favorable_direction` (its raw sign counts remain available). Incomplete data is not silently reduced to n<5 for qualification. Intervals are descriptive paired t4 intervals; task/cell/epoch/control reuse does not enlarge n.

Source metrics are the six cell AUC/LL pairs; equal-cell macro, traffic-weighted cell, domain/task macro, task-pooled AUC/LL, pooled-task-mean AUC, pooled LL and worst-cell AUC. Keep pooled and macro names distinct. Validation loss is ordinary unscaled source BCE on the AUC-selected state, never its own minimum. Selected online versus EMA may use different epochs; common endpoints and full curves allow stage comparison without reselection.

Runner exports raw arithmetic and costs only. Parent performs scientific analysis after raw-run acceptance. It reports every per-seed value/delta, seed consistency, AUC and LL, per-domain/per-task effects, online/EMA, selected/fixed stages, natural domain/class counts and CLEAN baseline AUC/LL/seedSD. Required discussion: stable repeated phenomena, contradictions including positive prior counterevidence, evidence strength STRONG/MEDIUM/WEAK per claim, competing explanations, size as a plausible main explanation, presence/absence of a small-domain gain/large-domain decline trade-off, and the best discriminator of remaining explanations. Do not assume curriculum/momentum works, difficult-domain M3oE failure or small=hard.

Primary decision, using full five-seed selected matched EMA data: both dataset mean AUC deltas≤0→NO_CONTINUATION_SIGNAL; exactly one positive→MIXED; both positive but any mean .003/cell−.001 gate fails→POSITIVE_UNQUALIFIED; both datasets pass both inclusive AUC gates→INTERNAL_SCREEN_QUALIFIED. LL mean<0 is reported separately. No CI-exclusion/all-positive-seed gate, champion automation or Test gate is added. Every decision still ends this one C05 after report/review/publication/delivery.

## 10. Acceptance criteria for reviewer and parent

Implementation review is independent and examines this complete design, actual55-file closure, actual implementation, coder report(s) and meaningful CPU fixture evidence. PASS/PASS WITH FIXES with no remaining blockers is mandatory before engineering. It must establish the registered objective/reduction, exact delegate branches, real probability BCE/source Adam/gate detach/fixed EMA, natural support/order/init, None/zero behavior, diagnosed-state neutrality and actual U virtual/post identity; new physical roots and numeric-only pack map; the concrete telemetry/terminal/witness fixes; and absence of extra trained arms/architecture/data/sampling/Test/state-to-disk paths. Record actual coder/reviewer identities, not configured role labels.

Resource review must consume the four actual C05 engineering proofs and phase witnesses,32 useful physical calls/8 points/24 sweeps/32 virtual steps plus all additional costs, source freeze/owner binding and exact ACK/restore identities, independent per-card current predicate/physical-fit/pending generation, terminal facts and compact reconstructable registry. It must check the fields in7.6 are actually used by launch. A report PASS without launch-bound code/source/evidence does not qualify.

Parent directly checks the same foreground owner, eight engineering ACKs, exact case proofs and release before formal admission. At formal completion parent independently reconstructs500 unique ACKs and epoch identities, totals500/17250/17250 and diagnostic40/120/160, all ten results and exact30 numeric controls, and obtains each completed job's compact selected pair via get. For all20 selected models, recompute actual state hash, bind dataset/seed/stream/epoch/validation to numeric result and earliest strict selection; verify retained tensor bytes and owner/root/source/design identity. PARENT_STORAGE_ACCEPTANCE.json records actual PID/tool_session/endpoint hash, selected state rows, ACK/reuse/table/cost receipt hashes, useful and physical counts, current liveness/availability and test_accessed=false. No old-model IPC is needed.

Before closing runner, accept RUNNER_HANDOFF.json with code/resource/recovery review identities, source55/external Adam/design/research/reuse/data/config/selection hashes, job/results/epoch/ACK/control pointers, current storage binding/selected identities, exact useful and all physical work, null partials, resource/witness/disk observations, table receipts and raw-only runner statement. Parent keeps foreground owner polled during analysis/review/publication. Independent evidence review then checks report claims against the exact numeric/archive/source receipts. It records original and revised verdicts; correct scientific blockers before publication. Published C05 includes REPORT, METHODS, full numeric appendix/tables, DATA_AVAILABILITY, execution history, machine decision, source/design/review/run/reuse/storage/cost/archive/publication receipts and reproducible arithmetic/figures, plus updates to all seven maintained contexts and per-cell absolute-metrics ledger. Parent owns analysis/report/context work; runtime tables do not choose scientific conclusions.

Actual verified remote publication and local alignment, final current selected-state availability/liveness observation, final delivery and scheduled USER_STOPPED complete C05. No C06. Test stays sealed for this entire specification, including final delivery; any later eligible frozen Test request would be a separate explicit action under existing gates.

## 11. Conditions that falsify hypotheses or invalidate identification

F1: Failure of either dataset's U−C selected matched-EMA mean .003 gate or any cell−.001 gate rejects cross-dataset candidate qualification. Unimproved same-state LL rejects the separate LL improvement claim. Complete the report even on a negative result.

F2: U−S selected EMA task1 mean LL≥0 on either dataset falsifies that dataset's registered sufficiency direction. If W−U LL<0 while U−S≥0, the pattern is consistent with W's class component contributing conditionally; it does not prove universal necessity.

F3: KR U−W d0t0 mean AUC≤0 or mean LL≥0 falsifies the respective class-skew rescue prediction. Opposing d1/d2/task effects must be reported as trade-offs where present.

F4: U−S task1 LL<0 on both datasets with retained W-relative rank/probability behavior weakens a strong claim that class skew is necessary for that repeated benefit. A crossing interval establishes neither equality nor necessity; no posthoc equivalence margin or mediation percentage.

F5: Nonmatching W/U six cell masses, natural support/order/source initialization, uncontrolled diagnostic mutation, wrong selection/state binding, missing required controls or changed scientific recurrence invalidates the intended contrast and requires correction before interpretation. It is a technical identification failure, not scientific evidence against H1–H3. Telemetry/storage failure similarly causes technical PAUSE with preserved facts, not a fabricated efficacy result.

Design blockers: none identified. Required downstream code/resource/storage/publication evidence is specified above and has not been substituted by DESIGN-stage inspection. Designer performed no implementation, training, preparation, GPU query, owner IPC or Test access. The only authored file is this DESIGN_SPECIFICATION.md.
