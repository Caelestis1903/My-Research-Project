# DESIGN SPECIFICATION — EMA_NEXT_20261111

Status: DESIGN COMPLETE. Date: 2026-11-11. Implementation, independent review, smoke and scientific runs follow this specification.

The frozen research plan is `research_plans/26-11-11_ema_next_phase` from plan commit `01979f6588935f713029b2b5e04557de9b8c153f`; implementation source is `5ef45b18b973ed11be6fbb5f654562349998de60`. This specification maps that plan to code without changing its experiments. Algorithm authority: 04 and 12–14; decisions: 09 and its matching manifest fields; matrix: 05. Preserve the original 19 plan files and record their actual byte hashes.

## 1. Responsibility and source boundary

Implement and execute the complete finite study: five Clean parents per dataset with simultaneous B/C readouts; required A short feedback; all triggered RAW, continuation, AB, KD and Gaussian branches; global selection freeze; Test; tables, scientific report, release and repository delivery. Seeds are exactly 2022–2026. Dataset decisions are independent. Negative five-seed results are valid completion; resource or implementation failures are not scientific STOP.

New scientific code lives only in `experiments/ema_next_20261111/`, its tests and execution/report outputs. Existing source model, loss, data, registry, metrics and prior results are read-only. Reuse source implementations directly where semantics match. No new dependency, new model framework, larger grid, extra baseline training or unrelated hardening.

Inspected source interfaces:

| Source | Use and necessary adaptation |
|---|---|
| `aug_history_20261105.model_adapter`, `registry`, `group_registry` | Build original fixed1 model, canonical parameter coordinates and S/D/T/DT support. Domain embedding is outside S. |
| `ema_mechanism_20261106.worker.create_session/train_batch` | Preserve initialization, natural loader and source loss; a local training adapter inserts A after native Adam and before F. Do not call its observation loop or pulse bank. |
| `ema_mechanism_20261106.shadows.FixedShadowBank` | Exact FP32 F update and clock; instantiate only F. |
| `ema_mechanism_20261106.readouts.read_predictions/panel_predictions` | Same forward produces raw logits and FP32 source probabilities. Preserve `3*task+domain` tower indexing and restore train/eval flag. |
| `ema_causal_20261110.interventions.group_masks/source_direction_states` | Exact EMBED mask and alpha=1 one-step oracle, not a feedback trainer. |
| `ema_causal_20261110.history.SparseHistory` | Reference-count design only: it stores R, while this study must store actual FP32 F. |
| `overnight_ab_20261103.data_bridge` and mechanism panels | Reuse audited natural cache and source P_A/P_B/V_mech population. Training/selection loads `include_test=False`. |
| `overnight_ab_20261103.selection.score` | Reuse six-cell score. Its `consider` rejects e0, so the new selector implements strict `>` for normal E1–50 and KD e0–5. |
| causal state and scheduler | Reuse RAM IPC, owner identity, admission measurements and transaction pattern; replace their hardcoded 50-epoch completion, four blocks and old packet schema. |

All Python commands use `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`. Keep source train batch 40960, 50 epochs, Adam .01/(.9,.999)/1e−8, dataset WD/rho, source natural BCE and Gaussian D0 unchanged. No AMP, TF32, clipping or probe training graph. Runtime data/feature/panel/group/environment identities belong in PREPARED.

## 2. One required specification reconciliation

`14_ROUTE_C.md` §3.2 orders risk ties by distance to current a, then distance to .5, then smaller alpha. Manifest `route_C.risk_tie_order` omits the middle rule. Resolve the runtime copy to `[closest to current old mass a, closest to 0.5, smaller alpha]`, following algorithm authority 14. Write a correction receipt identifying both original fields and the resolved rule; hash both raw plan and resolved configuration. Do not rewrite the downloaded plan or silently choose a rule. This changes no grid, budget or scientific threshold.

Other implementation distinctions already settled by the plan: source data cache may contain Test arrays but no Test loader/readout is available before global freeze; input cache is data, not a model checkpoint. Prefix packet parent-F is optional when it has no KD consumer; current teacher-candidate best must retain the real same-epoch FP32 parent F. Scalar selection coefficients are permitted outputs; model vectors, optimizer state and RNG-containing recovery packets remain RAM-only.

## 3. Minimum implementation split and common interface

Use two coders with disjoint file ownership, followed by a different reviewer. Both coders use the same inherited model as the orchestrator. No coder analyzes research conclusions.

**Coder 1 — scientific execution:** `trajectory.py`, `optimizer_direction.py`, `history_readout.py`, `conditional_readout.py`, `distillation.py`, `evaluation.py`, and `tests/test_science.py`. Own the source adapter, readouts, per-epoch records, compact predictor packets, state export/restore and numerical tests. No scheduler or gate decisions.

**Coder 2 — orchestration and accounting:** `__init__.py`, `protocol.py`, `decision.py`, `state.py`, `scheduler.py`, `run.py`, `report.py`, and `tests/test_execution.py`. Own frozen config/DAG, metadata and provenance, immutable RAM store, resources/recovery, gates, CLI, tables and report assembly. No independent model/loss implementation.

Use ordinary dictionaries plus existing tensors/arrays, not a new class hierarchy. Agree these functions before parallel edits:

- `protocol.resolve_plan(plan_dir)` returns the resolved Study dict: configs, manifest, identity, readout menu, explicit job templates and resource caps. `prepare` supplies cache root and per-dataset/seed panel identities.
- `trajectory.create_parent(job, prepared, device)` creates the source-compatible LiveSession. `fork_at_epoch(job, prepared, device, snapshot, objects)` restores a complete immutable anchor; KD uses `distillation.create_student` with its explicit teacher/F initializer. Restore must not alias mutable tensors.
- `trajectory.advance_epoch(session)` returns a scalar epoch `record` and newly immutable predictor/fork/history packets. `trajectory.export_session(session)` exports only mutable live state plus immutable object IDs; existing packet tensors are not recursively cloned into each commit. `restore_session` resolves those IDs from the RAM store. Implement `train_one_step` and `continue_to_epoch` as the same underlying scientific loop, not separate algorithms.
- Packets carry `kind`, parameter parts/buffers, source epoch/step/config/hash, H/h/a/scope/mode, applied C weights and real `parent_F_at_same_epoch` reference when needed. Scope S stores shared parts and shared N once; CELL PARAM can retain the same two-state packet and coefficients, reconstructing during evaluation.
- `evaluation.evaluate_full_population(packet, bundle, split, eval_batch_size)` returns source metrics and bounded panel diagnostics. Test evaluation requires the frozen endpoint registry supplied by the controller. `update_earliest_best` takes the complete predictor packet and preserves ties. `freeze_endpoints/test_frozen` expose final metadata/evaluation.
- `state.commit_epoch(job_id, live_state, new_objects, object_refs, record)` commits live state and incremental objects atomically; `fork_snapshot`, `put_immutable`, `release_consumed`, `recover` preserve their named semantics. Finalized scalar records and endpoint metadata are the only gate inputs.
- `decision` pure functions consume complete seed/config records and return DecisionRecords containing source record IDs, statistics, status, reasons and successor IDs. No inference from presence of files alone.

Coders may consolidate functions within their assigned files, retaining these responsibilities and agreeing exact argument shapes directly. Each writes a CODER IMPLEMENTATION REPORT listing source reuse, files, interface and checks. The orchestrator combines them for review.

## 4. Scientific loop

### A: real update feedback

One step is source zero_grad → forward/loss → backward → capture pre-G theta/data gradient → native Adam once → optional A patch → F once. Keep current branch m/v/per-tensor step. Compute q=g+WD*pretheta without writing q into `.grad`. Match the actual native G displacement radius using FP64 current-preconditioned direction and mixture from 12; cast patched theta to FP32. Non-G coordinates remain native. Alpha0 is exact native no-op. Zero radius no-op, zero direction/near cancellation use documented native fallback with counts; NaN/Inf is an implementation failure. Actual FP32 radius relative error must be ≤1e−4. Subsequent gradients use patched parameters.

A_RAW splits parameter groups without changing parameter objects or state: EMBED beta1=0, all others .9, preserving v/step/F. E20/E35 anchors contain optimizer, F, RNG, loader, history, best-reference/prefix state and complete identity. Two alphas × two anchors × five seeds each run six real epochs; match parent E25/26 and E40/41. Full runs continue the chosen E26 state through E50, retaining parent E1–20 selected prefix. E35 never becomes a hidden full-run start choice.

### B: actual F history and bounded readouts

Capture actual F after updates only at origins `max(0,e*steps-H)`, including F0. Capture precedes consume at shared events; full history cache peaks are ML5/KR11. Compute h=min(H,t), a=rho**h, old=F[t-h], recent=(F[t]−a*old)/`-expm1(h*log(rho))` in FP64 then FP32 forward states. S scope gives both states the current F_N; all scope uses complete two states. Record double and cast barycenter errors. Logit/probability mixtures reuse two component forwards. B_SHARED_ONLY is F_S+online_N, a different predictor.

All 12 B configurations and B_SHARED have independent full-V E1–50 bests. Save current teacher best with real same-epoch F, never a cast-centroid reconstruction. At E20 independently pin all B prefix-bests and O/F prefix-bests; later parent best updates cannot release them. After five-parent B selection retain only the frozen B prefix and needed controls. A captures that one B history only when G_B passes. AB curve is the exact parent prefix plus real A short/continuation curve.

### C: one-epoch-lagged training signal

Use main H/S/logit. Build fixed unique P_A/P_B rank pairs once using the exact ordered compact UTF-8 JSON and BLAKE2b-128 contract of 14, taking min(4096,positive×negative) smallest `(hash,pos_id,neg_id)`. Keep source row/panel identity; new C pair identity is distinct from source random pairs. Grid is unique within 1e−12: 0,.25,.5,.75,1,a. FP64 stable softplus pair risk; global equal valid-cell mean, local argmin and shrinkage s=.5/1; no projection. Apply the resolved three-level tie order.

At e1 apply current a. At e≥2 apply absolute weights fitted on P_A at e−1 to current two states. Evaluate all six complete-V curves and P_B first, fit next weights using current P_A afterward, then commit used and pending separately. Rank population supplies panel AUC/risk; natural population supplies LL. CELL PARAM evaluates each domain/task's actual interpolated parameter forward and selects task column k, restoring original rows. PARAM risk must not borrow PRED margins. Missing pair cells produce NA/fallback and BLOCKED_PANEL where gate input is incomplete.

Store scalar future-window (E21–50) differences, alpha differences and fixed-pair sign-change rates, keeping seed as statistical unit. Small registered E20/E35/E50 and selected pair diagnostics suffice; do not persist all full-V predictions.

### KD and inherited Gaussian

Freeze one qualifying B/C teacher per dataset with each seed's selected packet. All seven possible student arms start from its exact same-epoch parent F, fresh Adam, same paired loader order, F_student initialized at theta0. Cache teacher and Fteacher soft probabilities on original train row IDs only. Loss is `(1-lambda)*source_BCE(y)+lambda*source_BCE(q)` with source natural means and T=1. Four λ/LR candidates and two LABEL LRs complete five seeds × five epochs before KD selection; then run the selected λ/LR Fteacher control. Both O/F use legal e0–5 bests, with student F primary.

Gaussian runs only for a qualifying Clean main family and inherited hyperparameters. A/AB uses E20 switch and 30 real continuation epochs, B/C uses frozen readouts. KD/LABEL/Fteacher Gaussian boundary is Gaussian-parent teacher/init followed by three Clean-student five-epoch arms: selected main LR, Clean KD winning nonzero λ for both soft arms and zero for LABEL. No Gaussian HPO or changed loss.

## 5. Gates and finite DAG

Use exactly 09 and matching manifest formulas, including these completeness and numerical boundaries:

| Gate | Required decision |
|---|---|
| G_A | Current dataset all 20 MATCH forks complete; per alpha all 10; mean paired F gain over E25/26/40/41 ≥1e−4, each cell ≥−.001, E25/26 mean >0. Tie within1e−6 chooses .5. |
| G_A_RAW | Its own 10 forks complete, same numeric gates. At most one MATCH plus passing RAW continuation. Full MATCH/RAW tie within1e−6 prefers RAW. |
| G_B/G_B_SHARED | Own-selected U gain versus F ≥2e−4 and cell guard. B internal tie: S, logit, base/short/long H. Both passing and within1e−4 prefer shared-only deployment. |
| G_C_MECH | Complete E21–50: mean |cell−global alpha| >1e−6, pair sign changes >0, future PB risk delta <0 and AUC delta >0; V delta ≥1e−4 with cell guard. Compare same mode. |
| G_C_SELECTED/GLOBAL | CELL additionally beats same-mode GLOBAL and best unified readout by ≥2e−4 with both guards; GLOBAL alone versus F uses ≥2e−4. PARAM preferred within1e−4 when eligible. |
| G_AB | Beat both A_F and parent B by ≥2e−4 with both guards; preserve selected and E50 difference-in-differences separately. |
| G_KD_ENTER | Qualified teacher beats best available one-forward unified alternative by ≥2e−4. Consider F, B_SHARED_ONLY, C_GLOBAL_PARAM and qualifying completed A_F; keep O as ordinary control. Choose teacher U with1e−6 tie order B, GLOBAL, CELL. |
| G_KD_HISTORY | Student F beats parent F, same-LR LABEL_F and FTEACHER_F by ≥2e−4 with all guards; retains ≥.5 of teacher gain. Also report better LABEL LR. Simple LABEL/Fteacher may qualify on their own versus F. |
| G_G | One Clean main per dataset: gain versus F ≥2e−4 and guard. Main-family ties within1e−4 use measured forwards, parameter bytes, extra updates, fixed ID. |
| FORMAL_AUC | Frozen Test five-seed gain versus each declared published baseline ≥.003 and each cell ≥−.001; primary ML M3oE / KR HyperGate. LL separately reports gain/cost. |

Cell guards always use mean paired seed deltas. No CI-positive or positive-seed-count gate. U means mean over five seeds of that configuration's own selected full-V metric. Checkpoint tie is strict `>` with no epsilon. Report sample SD and t4 CI with 2.7764451052, never treating epochs/cells as independent repeats. Full A component and Gaussian eligibility still require the general ≥2e−4 selected increment after its shorter mechanism gate.

Expand all templates and consumers before training. Core=740 epochs/25,530 updates; all-condition maximum=2,490/85,905. Count no-SGD readouts separately; parent full-V maximum is20 forward equivalents per epoch (2 O/F +1 shared +12 B +5 C PARAM). Gaussian alternatives are mutually exclusive. Smoke/retry costs have separate physical counts.

Logical order for each dataset: parents → B freeze/C records → A MATCH → conditional RAW/continuation/AB → teacher selection/KD → main selection/Gaussian → retain final endpoints. All datasets/triggered controls then global freeze → Test/report → release. Every untriggered row references actual gate inputs; incomplete seeds produce WAITING, not a failure verdict.

## 6. RAM design and feasibility

One scheduler owner and RAM guardian per attempt. Objects are immutable and keyed by state hash/identity; references are small IDs. Mutable active state is exported independently. B modes can share a packet only when actual epoch/state match; prefix references survive later best replacement. Scope-S packets store shared states plus N once, and C PARAM stores two states/weights rather than six full models. Completed parents discard online optimizer/live cache after their immutable anchors and bests have been committed. Physical storage accounting must deduplicate shared tensor storage; inherited recursive `tensor_bytes` alone counts references repeatedly and cannot describe the object store.

A conservative planning calculation, before measured buffers/arrays and retained earlier-dataset endpoints: current best ≤42P+11S scalar FP32 elements/seed; E20 prefixes ≤20P+6S; E20/E35 anchor cores plus histories ≤(8+2h)P with h=5/11; active current state ≤(4+h)P. Five completed-parent stores plus one active state and two extra active copies for the inherited guardian predicate give ML1.383GiB/KR3.246GiB. Physical active/export/observer copies raise the corresponding illustrative baseline to ML1.458GiB/KR3.475GiB. These estimates show a compact implementation is plausible under 4/7GiB, but omit buffers, RNG, panels, accumulated prior-dataset endpoints and actual commit overlap; they are not acceptance measurements.

Prepare must simulate actual capture/consume, prefix pin, best replacement, anchor creation, seed completion/selection and commit old/new coexistence for the whole five-seed block, including worst-case distinct selected epochs. Its object layout must match code. Enforce original guardian predicate ≤4GiB and total parameter-tensor peak ≤7GiB, host <90%, each independently selected GPU memory/power <85% with measured growth reservations. Include prior dataset final endpoints and at most one prefetched block; if lowest concurrency cannot fit, report named excess objects before training. No fixed two-worker cap; fill eligible cards subject to these constraints.

No parameter/Adam/F/history/selected/recovery files, including temporary NPZ/mmap/tmpfs. Input arrays, bounded predictions and teacher targets are data and may use source data cache. Persistent observations ≤2GiB with precomputed shapes, bounded panel records, compressed scalar curves and logs. Do not save full-V/Test epoch/grid logits.

Epoch transaction: reserve → train/observe → submit new objects/live state while old commit survives → atomic scalar/allowed-prediction outputs → COMMITTED receipt → release consumed old references. Worker loss recovers from live guardian once/job; guardian loss deterministically replays same parent and branch once/logical job with prior scientific hashes, no reselection. Code/config/data mismatch blocks affected jobs. OOM reduces concurrency/eval microbatch, never train batch. Preserve guardian across controller interruption. Final release requires all frozen Test/report consumers completed.

## 7. Required verification and review exit

One focused test module per ownership set; tests cover only these scientific failure paths:

1. Native/alpha0 and observations preserve θ/m/v/step/F/RNG/loader exactly on a short source sequence. Alpha1 agrees with source direction oracle; WD once, per-tensor step, None/zero fallback, non-G unchanged, actual norm tolerance, second-step feedback.
2. B initial mass and h clipping, sparse capture-before-consume (ML5/KR11, KR277), FP64 and FP32 barycenter residuals; S common N and shared-only distinction. Constructed-vector residual tolerance is explicit machine-scale error, not a scientific gate.
3. C unique deterministic pairs and identities, documented tie rule, e1 fixed-a/e−1→e timing, no validation/PB fit input, exact task-index PARAM output, PARAM risk from actual forward, empty-pair NA/fallback.
4. KD stopgrad/eval, teacher=y source-loss equivalence, row IDs and same paired batch order, fresh Adam/F initialization, e0 selection, same λ/LR Fteacher control.
5. Full-population metric versus concatenated batch scores; earliest strict maximum; A prefix retained; Test unavailable until global freeze. Five-seed gate boundary/equality and missing-seed WAITING tests; exact matrix counts.
6. RAM fork storage independence, single epoch loss/recovery next-step equivalence, incremental packet retention/release including E20 prefix; whole-block resource forecast and independent per-card admission; duplicate execute cannot duplicate work.

Reviewer independently checks this specification, all actual code and both coder reports; PASS or corrected PASS WITH FIXES is required before experiments. Then runner executes two dataset GPU smokes with engineering seed2022: source/A step, B/C readouts and KD student step on real batches, plus one complete train epoch/full-V registered readout sweep to measure train/eval/panel/IPC and actual resource peaks. Smoke metrics never prune candidates. Bind PASS and smoke receipts to implementation hash. Run prescribed tests once after fixes; no unrelated regression expansion.

## 8. Delivery and scientific analysis

Deliver required source/prepare/data/panel/group/job/readout/resource receipts; complete curves; HPO/Decision records; global freeze; gzip ENDPOINTS/PAIRED/COSTS; FINAL_REPORT, QUANTITATIVE_APPENDIX, PROVENANCE and ARTIFACT_INDEX; release receipt. Reports include all seeds/cells, AUC/LL, worst cell, same-time versus independently selected identities, controls and published baselines, phase, costs and revisions. Archive negative and untriggered outcomes with actual reasons.

Orchestrator analyzes A real feedback versus F, B bounded expression/deployment and KD retention, C action/future/unified-policy increment. Label evidence STRONG/MEDIUM/WEAK, give repeated phenomena, contradictions and competing explanations. Record source domain counts and observed baseline difficulty without equating size with difficulty; do not add domain-size experiments. Separate total performance, incremental contribution, LL cost and applicable dataset/condition. Reuse fixed published-baseline endpoints with exact filters and same-seed pairing; do not retrain them. Scientific report is written only after actual records exist.

Skipped by design: old q/attribution reruns, extra windows/rho/models/controllers/temperature/HPO, model weight exports and unrelated engineering tests. Remaining execution acceptance is implementation review plus measured data/state/resource correctness and complete prescribed runs.
