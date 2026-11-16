# DESIGN SPECIFICATION — Cycle03 NQR temporal exposure

Date: 2026-11-16. Role: fresh DESIGNER; original current-conversation parent; `fork_context=false`; file handoffs; no descendants.

**Verdict: COMPLETE. Blocking design/API ambiguities: none.** This fixes the implementation contracts for the already selected EARLY10/LATE10 experiment. Candidate efficacy is UNPROVED. COMPLETE permits the parent to dispatch coders; independent implementation REVIEW and later smoke/resource signoff remain required before formal RUN.

Paths used below:

- W = `/home/xzf/Code/phase0-3_tvg_rm_20261116`; O = `/home/xzf/Code/phase0-3`.
- P = W/`research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03`.
- A = W/`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03`.
- N = A/`attempt1`, the new absolute storage/attempt root.
- K = W/`experiments/nqr_temporal_exposure_20261116`.
- R = W/`research_reports/nqr_temporal_exposure_20261116`.
- C2N = W/`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1`.

Read/applied: current W/O AGENTS, O/start.md decoded GB18030, W/buchong.md, execution runbook, all six completed Cycle03 artifacts and A/ASTRA_HANDOFF.json, C2 DESIGN_SPECIFICATION, core/protocol/runner/tables, populated-restore correction, latest full C2 REPORT, FINAL_EVIDENCE_REVIEW_R1/correction, C1 physical-capacity correction, and C2 resource audit. Ponytail FULL and stop-that-shit CHANGE apply. Historical lifecycle instructions yield to the current foreground policy. Astra `01a1e494-023e-7ac2-93b0-59c57b3c8785` is permanently CLOSED and is never resumed.

The six-file handoff aggregate is `58aa43e9093429bf232da9c07ff8820e952216eb151e8aa603077aa611b9c03e`; all six file hashes matched its receipt. Frozen C2 source is 87 files, aggregate `ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae`; all actual bytes matched. Frozen C1 source is 79 files, aggregate `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. Previous publication commit/tree are supplied and receipt-bound as `ed34eb243099dc2693f06ba566fc689c7657f641` / `ad6b794d83de8a75ba161c1e3466016c31f6e069`; this designer performs no Git/publication action.

## 1. Research question

Does applying the exact frozen C2 NQR-025 operator only in epochs 1–10 preserve a benefit at the full 50-epoch, independently selected fixed-EMA validation endpoint, relative to CLEAN on each dataset? At equal ten-epoch scheduled exposure, does EARLY differ from exposure in epochs 41–50 at fixed E50, separately for online and EMA?

The motivating observation is C2 MovieLens E10 EMA AUC gain followed by a negative selected-EMA gain. EARLY E1–10 reproduces that observation by construction; its later trajectory is the new test. The C2 corrected result remains NO_CONTINUATION_SIGNAL. NQR-minus-BRIER LL was lower in **4/5 seeds on both datasets**, with crossing intervals; that relative LL finding supplies no primary AUC success.

## 2. Hypothesis being tested

Primary directional prediction: EARLY10 minus CLEAN selected fixed-EMA six-cell Macro AUC has a positive mean on each dataset. The practical screen is +.003 mean Macro AUC with all six mean paired cell deltas at least −.001, per dataset.

Timing prediction: EARLY10 minus LATE10 fixed-E50 Macro AUC is positive in both online and EMA streams. EARLY-minus-constant NQR distinguishes preservation/recovery relative to continued exposure, but changes dose as well as placement. Similar EARLY/LATE responses favor reduced exposure or other shared effects over a specific early-placement explanation. All predictions can fail.

Ten active natural epochs match row exposures, scheduled update counts and lambda sum. They do not match realized pair opportunities, gradients, Adam moments, EMA recency or parameter paths. The interpretation includes those histories. No effective curriculum, useful momentum, intrinsic domain difficulty inferred from size, or generic difficult-domain M3oE failure is assumed. This is a schedule intervention on an established loss; no renamed-objective or global novelty claim is permitted.

## 3. Variables changed

Only the epoch-dependent use of the frozen NQR scalar objective changes:

| External arm | Epochs with lambda .25 | Epochs with lambda 0 | Status |
| --- | --- | --- | --- |
| `NQR-EARLY10` | 1–10 | 11–50 | Only candidate; 10 new trajectories |
| `NQR-LATE10` | 41–50 | 1–40 | Timing control; 10 new trajectories |
| `CLEAN` | None | 1–50 | 10 numeric reused trajectories |
| `NQR-025` | 1–50 | None | 10 numeric reused trajectories |

No other arms, endpoints, coefficients, sweeps, stopping thresholds or baseline reruns are added. Opportunity flattening and SAM remain deferred. C2 BRIER/MEAN-GAP and earlier experiments are context, not extra Cycle03 comparison arms.

## 4. Variables held constant

| Item | Contract |
| --- | --- |
| Model | Frozen M3oE fixed1; domains3/tasks2, embedding16, FCN [512,256,256,64], experts4, exp_d/exp_t/bal_d/bal_t=1; source parameter order, gates, sharing and fusion |
| Graph | Ordinary source forward; gate inputs detached, gate parameters trainable; no full-gate probe or feature view |
| Data | Original natural train/validation split, rows, labels, feature schema, encoders/scalers fitted on train only; no balancing, synthetic population or resampling |
| Seeds | Exactly 2022,2023,2024,2025,2026, each dataset/arm from fresh source initialization |
| Training | 50 epochs, batch40960, retained actual tails, FP32; one ordinary forward/scalar backward/Adam/EMA per batch |
| Adam | Frozen factory, lr .01, betas(.9,.999), eps1e-8, coupled WD ML1e-4/KR1e-5; foreach/fused/amsgrad false; no clipping/scheduler/accumulation |
| EMA | Starts equal online; rho ML .9828205985452511/KR .995; source buffer copying; one update after Adam; no reset or bias correction |
| Loader | Source setup once; private CPU generator seed+11000; one randperm per epoch; continuous generator; no boundary reset or algorithm RNG |
| Environment | `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`, Python3.13.5/torch2.9.0+cu126, existing lock, `PYTHONDONTWRITEBYTECODE=1`, one CPU thread, TF32/autocast/cuDNN benchmark off |
| Validation | Full natural validation online and fixed_ema after every epoch; frozen evaluator and reduction; each stream independently selects earliest strict Macro AUC maximum |

| Dataset | Train / validation rows | Natural train domain counts | Steps/epoch | Tail | Total / active / off updates per new run |
| --- | --- | --- | --- | --- | --- |
| movielens | 800167 / 100021 | 155949,318599,325619 | 20 | 21927 | 1000 / 200 / 800 |
| kuairand | 2006132 / 250767 | 172782,1743898,89452 | 49 | 40052 | 2450 / 490 / 1960 |

Twenty new trajectories total **1000 epochs, 34500 Adam and 34500 EMA updates, 6900 active and 27600 off updates**. Each row has ten active and forty off visits. Per trajectory lambda sum over updates is ML50/KR122.5; sum over epochs is2.5. Historical constant NQR has update sums250/612.5. No early science gate changes this budget.

Use the immutable cache `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache` through the existing audited source cache helper after checking the cache exists. Do not invoke its missing-cache reconstruction fallback. Raw CSVs are `/home_new/xzf/datasets/ml-1m/ml-1m.csv` and `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`. Verify the Cycle03 manifest's raw/train-validation arrays, row IDs/fingerprints, encoder/scaler, features and source data config identities. Preserve the frozen cache verification behavior; Test fingerprints in existing receipts are provenance only. No Test inference, metric, exploratory inspection or export is authorized. The absent mdmt_next preparation is not a dependency.

## 5. Mathematical/algorithmic definition

### 5.1 Operator and branch semantics

For actual batch B, x:[F,B], binary y/p:[B,2], domains d=x[-1] in{0,1,2}, let C be source mean probability BCE over2B entries. For cell c=(domain,task), n=n0+n1, pi_a=n_a/n, w=n/(2B). For each present class, mu_a is the probability mean, v_a its centered population variance and bbar_a the mean source BCE. For two-class cells:

`S_c = pi1^2*bbar1 + pi0^2*bbar0`

`Q_R,c = .5*((1-mu1+mu0)^2 + v1 + v0)`

`A_c = S_c + 2*pi1*pi0*Q_R,c; A = sum_c w_c*A_c`

`L_e = (1-lambda_e)*C + lambda_e*A`.

EARLY: lambda_e=.25 for1≤e≤10, else0. LATE: lambda_e=.25 for41≤e≤50, else0. Epochs are integers1…50. No margin/variance/BCE/reduction-order change is permitted. Means and variances retain autograd; only logs detach. No pair matrix or pair sampling is introduced.

The explicit BCE coefficient per observed row/task is `(1-lambda+lambda*pi_y)/(2B)`; opposite mass per cell is `lambda*w*2*pi0*pi1`; total coefficient mass1, task mass.5 each. This budget does not imply equal gradient dose. Empty cells contribute0 with null absent statistics. Singleton present-class variance is0. One-class cells use ordinary mean BCE. At lambda0, or if every nonempty cell is single-class, the frozen C2 core delegates to its exact source CLEAN helper. The branch labels remain `lambda_zero`, `all_single_class`, `ordinary`; direct branches report C=A=L and null unobserved moment fields. Off A is a reporting convention, not a measured counterfactual risk. No extra forward fills those nulls. An active all-single-class batch counts as scheduled-active but has zero nonzero-opportunity increment.

**Mandatory reuse:** import and invoke `experiments.natural_quadratic_ranking_20261116.core.train_step` exactly once per adapter call. Do not reimplement QR, source CLEAN, backward, Adam, EMA, ledger or diagnostics. The inner arm is always `NQR-025`; only its local effective `mixture_lambda` is0/.25. External job/config/log identities always retain the temporal arm.

### 5.2 Shared core API — fixed for both coders

Core imports frozen helpers only; it never imports the new protocol/runtime. Plain dicts suffice.

`core.schedule(arm, epoch) -> dict`

- Accept exactly the two new arm strings and an actual integer (exclude bool),1≤epoch≤50; invalid input fails before any model/optimizer/RNG mutation.
- Return exactly `schedule_id`, `schedule_epoch`, `schedule_phase`, `scheduled_active`, `effective_lambda`, `external_arm`, `inner_arm`.
- schedule_id is `early10_v1` / `late10_v1`; schedule_epoch=epoch; schedule_phase=`on`/`off`; scheduled_active is bool; effective_lambda=.25/0.; external_arm=arm; inner_arm=`NQR-025`.

`core.train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False) -> dict`

- Keep the frozen C2 positional signature. Require config fields from7.1 and reject an inconsistent schedule/window/coefficient. Determine schedule only from external arm and explicit context epoch. Copy config locally; override only inner `arm='NQR-025'`, `mixture_lambda=effective_lambda`, and `objective='natural_quadratic_ranking_v1'` before delegation. Never mutate caller config/context, a shared dict or an old module global. Extra temporal metadata is inert to the frozen callee.
- Context requires `dataset`, `seed`, `epoch`, `batch`, `global_step`, `purpose`, `clock_kind`. Core requires context.dataset=config.dataset and seed to be an integer in2022–2026; the caller additionally requires context.seed=job.seed. Epoch is the schedule epoch; batch/global_step are positive integers (exclude bool). `purpose='formal', clock_kind='formal'` uses the real source clock. Engineering uses purpose `engineering_check` or `engineering_smoke` and clock_kind `engineering_virtual`; only those explicit combinations permit the virtual clock in7.5. No automatic clock-mode inference.
- Core validates arm/schedule/epoch/context types and clock-mode pairing, but does not infer elapsed scientific epochs from an engineering fixture. The formal worker independently enforces every clock/counter equation in5.3 before/after delegation. Core need not own mutable counters. Real worker row_ids are true ordered source IDs; None is allowed only for synthetic engineering checks, using the frozen callee convention.
- Return every frozen C2 key unchanged: `loss`, `clean_loss`, `completed_loss`, `branch`, `ledger`, `row_order_hash`, `optimizer_updates`, `ema_updates`, `ordinary_forwards`, `scalar_backwards`, `probe_forwards`, `view_forwards`, `algorithm_rng_draws`, `seconds`, `diagnostics`. Add exactly `schedule` and `phase_increments`. schedule is the seven-field result above plus `clock_kind`. No tensors/graphs/state are returned.
- phase_increments has exactly `active_updates`, `off_updates`, `active_rows`, `off_rows`, `active_task_entries`, `off_task_entries`, `lambda_update_sum`, `nonzero_opportunity_updates`. For active a∈{0,1}, B=ledger.B: increments are a,1−a,aB,(1−a)B,2aB,2(1−a)B,lambda_e, and int(ledger.opposite_mass>0), respectively. The last can be1 only on an active update.
- Source counts remain optimizer_updates=ema_updates=ordinary_forwards=scalar_backwards=1 and probes/views/algorithm RNG=0. Existing seconds cover the delegated step/diagnostic work, while outer worker wall time captures adapter overhead. Add no backward/probe to measure it.

**Ledger and diagnostic schemas are unchanged frozen C2 schema1** (C2 DESIGN_SPECIFICATION5.3 and actual core.py). Do not rename/rebuild those nested fields. In particular retain all six cell entries and Q_R/Q_B/Q_M detached diagnostics, BCE coefficients, mass/null conventions, and diagnostic active/None counts/group names/Adam counter ranges. Schedule metadata belongs beside the ledger, not inside it. tables flattens the added metadata without altering source numeric columns.

### 5.3 Formal clock, phase counters and continuity

Let S=steps_per_epoch and Ntrain=train_rows. At entry to real batch b of epoch e, `global_step_before=(e−1)S+b−1`; context.global_step is that plus1, with1≤b≤S. Successful return increments global_step and exactly the two source `counts={optimizer_updates,ema_updates}`. Preserve this two-key source dict; keep new totals separately in `phase_counters` with the eight keys of phase_increments.

Define a(e)=scheduled_active. Through batch b, active_updates is `S*sum_{j<e}a(j)+b*a(e)`; off_updates=global_step−active_updates. Let R_b=min(40960*b,Ntrain). Active_rows is `Ntrain*sum_{j<e}a(j)+R_b*a(e)`; off_rows is `(e−1)*Ntrain+R_b−active_rows`; task entries are twice the corresponding rows. lambda_update_sum=.25*active_updates. nonzero_opportunity_updates is the exact accumulated ledger-positive count, bounded by active_updates and reconciled against logs, not inferred from schedule.

At a committed real epoch, global_step=eS, source counts both equal eS, phase totals obey those equations with b=S. State `epoch=e`, `next_epoch=e+1`, `clock_kind='formal'`, `schedule_epoch=e`; `next_epoch=51` at completion is a terminal marker and is never passed to schedule(). Record current schedule and next schedule (null after50). Every formal transition10→11/40→41 checks counters and changes the effective lambda only. No optimizer/EMA/model/loader/RNG/reselection reset occurs. Adam parameter steps are taken from populated optimizer state per named parameter; do not set all steps to global_step because source None-gradient parameters legitimately differ. None skips Adam decay/moments/step, explicit-zero gradients follow source Adam activity.

Setup/model initialization happen once for a fresh worker; on recovery their construction effects are superseded by restoring acknowledged tensors, private loader and complete source RNG, in that order with RNG last. A formal epoch draws exactly one permutation using the continuously advancing private generator. Evaluation, diagnostics, hashing, snapshotting and transport preserve training RNG/loader and model mode. No boundary reseed, extra probe or algorithm random draw is permitted.

## 6. Exact affected modules/files and disjoint ownership

| Fresh coder | Exclusive new files | Responsibilities |
| --- | --- | --- |
| Core/checks | K/`__init__.py`, K/`core.py`, K/`checks.py` | Package marker, schedule adapter and finite identity/clock/check fixtures |
| Protocol/runtime | K/`protocol.py`, K/`runner.py`, K/`tables.py`, K/`README.md` | New identities/reuse/roots, supervised mechanical worker/smoke/admission/ACK/recovery, numeric exports and commands |

Neither coder edits the other's files or this design/six-file handoff. Reports/check receipts go to new A paths assigned by the parent (`CORE_IMPLEMENTATION_REPORT.md`, `CORE_CPU_CHECKS.json`, `RUNTIME_IMPLEMENTATION_REPORT.md`, `RUNTIME_CPU_CHECKS.json`). No extra source module/config framework or dependency is needed. Coders can work in parallel using5.2/7.1; after both finish, the parent obtains the integrated source receipt and distinct reviewer.

No change/addition under `natural_quadratic_ranking_20261116`, `task_axis_even_views_20261116`, `auxiliary_objective_attribution_20261115` (including stage2), `conditional_information_momentum_20261102_v3` (including vendor), or `task_view_gradient_residual_momentum_20261116`. K is outside every protected scan root. Preserve pycache contents/membership; use no bytecode-writing compile/import. Preserve all other workers' changes. No root state, history, Git, publication, dependency or old orchestration edits.

Allowed frozen numerical helpers include source `digest`, `immutable`, `sha_file`, verified cache reader, `optimizer`, `load_protocol`, `make_model`, `setup`, `cpu_copy`, `state_digest`, `rng_state`, `restore_rng`, `evaluate`, `write_json`, and C2 core.train_step. Read-only root-independent C2 `admission`, `incoming_profile`, `host_ram`, `NVML`, `device_identity` may be imported after tracing their globals. Construct/query measurement helpers only in reviewed RUN. C2 `Profile` calls its old-root numeric writer and cannot be reused unchanged; the new runner owns its local profiling/writing. Likewise own all local preparation, jobs, commit/recovery and export plumbing.

Never call old root-bound `prepare/run/worker/export/recover/handoff`, `ensure_guardian`, old source-manifest generators, or patch their globals/default roots. The frozen storage `serve(attempt)` and transport `guardian.request(attempt,...)` are explicit-root exceptions: only the parent launches serve with N after review; the new protocol wraps request with absolute N validation. Source inspection/reuse is not authority to contact an old owner.

## 7. Implementation requirements

### 7.1 New protocol, immutable identities and callable surface

Required public interfaces (same dict/path conventions as C2; no new framework):

- `protocol.config(dataset, arm) -> dict`, `job(dataset, arm, seed, source_hash) -> dict`.
- `protocol.source_manifest() -> dict`, `design_bundle() -> dict`, `reference_index() -> dict`.
- `protocol.cycle_path(attempt) -> Path`, `local_path(attempt, path) -> Path`, `request(attempt, command, **payload)`, `stop_requested() -> bool`.
- `protocol.prepare(attempt, cache_dir, review_path) -> dict`, `validate_attempt(attempt, review_path=None, require_smoke=False) -> dict`, `validate_epoch_state(state, record)`, `epoch_files(attempt, result) -> dict`, `results(attempt) -> list`.
- `runner.restore(model, ema, opt, loader, state, device)`; `runner.runtime_admission_checks() -> dict` for pure fixtures; `runner.worker(attempt, key, gpu)`, `smoke(attempt, devices, review_path)`, `run(attempt, devices, review_path, recovery=False)`, `handoff(attempt)`.
- `tables.export(attempt=N) -> dict` with numeric output paths/counts, Test=false and no scientific verdict.
- `checks.run_checks() -> dict` for core-only finite CPU checks; `checks.check_runtime_contracts() -> dict` after both coders' files exist (lazy import, no GPU/IPC). CLI checks invokes both only after integration. No broad old test suite.

Config includes exactly registered scientific choices: cycle_id=`cycle_03`, dataset/external arm; objective=`nqr_temporal_exposure_v1`; source_operator=`experiments.natural_quadratic_ranking_20261116.core.train_step`; inner_arm=`NQR-025`; completed_risk=`Q_R`; probability_margin=1.; schedule_id; active_epoch_start/end=1/10 or41/50; active_lambda=.25; off_lambda=0.; epochs50/batch_size40960/precision float32; source lr/betas/eps/weight_decay/rho/steps_per_epoch/tail_rows; loader_seed_offset11000; gate_training_detached=true, probe_gate_full_derivative=false, algorithm_rng_draws=0; source data and selection hashes. External config has no ambiguous fixed `mixture_lambda`; the inner local copy adds it. Seed is a job/context field, not a reason for different scientific config. Include unchanged model/environment/data/loader/selection identities in the run manifest. Do not use a C2 config/job/source hash as a Cycle03 worker identity.

New source receipt starts from the **explicit frozen87 list** in Cycle03 EXPERIMENT_MANIFEST.sourcebase.c2_frozen_files/C2N/source_receipt.json. Check each path/hash and the87 aggregate, plus nested frozen79 identity, without invoking an old scanner. Add the seven K files and actual additional transitive local imports discovered by byte-only AST traversal, including package initializers. Requirements lock is already in the frozen list; do not duplicate it. Sort unique `{path,sha256}` rows by W-relative path; new source hash is frozen `digest(rows)` (compact sorted-key JSON). At least94 unique files; actual extras are enumerated and hashed. New receipt separately binds operator qualified name/core file SHA, frozen87 and79 hashes, source closure and design hash. Source receipt itself is not an input to its own hash.

For design identity, retain C2's sorted `[filename,sha256]` list convention over the six original files plus this specification; hash compact JSON with that order. Bind A/ASTRA_HANDOFF.json hash and the original six aggregate separately. Parent DESIGN_ACCEPTANCE must bind the exact seven files with verdict `COMPLETE_APPROVED_FOR_IMPLEMENTATION`. The six-file manifest's old `formal_design_complete=false` describes its research-handoff stage; preserve its bytes and use this specification plus the new acceptance receipt for formal design authority. Do not recursively store the new bundle hash inside its inputs.

Job contains cycle_id,dataset,external arm,seed,config,config_hash,source_hash,design_bundle_sha256; `key=digest(job_without_key)`. Formal jobs have no engineering purpose. Engineering adds purpose/fixture identity before key hashing and can never equal a formal key. Full receipts bind environment, data, reused-control index and source-adapter identities; internal per-step config copies cannot change the external job.

All paths must equal N or be verified descendants of N, absolute, resolved without symlink aliases into old roots. New manifest/review/storage/smoke/table receipts must agree on source/design/config/job identities. Original-C2 dynamic root gates are not substitutes for these new gates. Use source helper checks directly or new local equivalents where the C2 wrapper binds old paths.

### 7.2 Exact numeric control reuse and prefix comparisons

Use exactly20 entries in Cycle03 manifest.controls.entries: ten CLEAN and ten NQR-025, both datasets/all five seeds. No broad result glob. CLEAN comes through the C2N/reused_reference_index.json SHA `ba6fb786d675c0a85fede482f6ae203cb3fa292583c4f5ac4855ef692a04b89b`; use its explicit entries[].epochs. Constant NQR comes from the exact manifest result paths and epoch_directory/epoch_01…50 receipts, checked against the registered epoch-receipts aggregate. Numeric result/epoch bytes only; no old state/IPC/access key is read or cloned.

New `reference_index()` returns `entries` (the20 augmented explicit result paths/hashes and50 epoch path/hash receipts each), `initial_hashes` (10 dataset/seed keys), `epoch_order_hashes` (500 dataset/seed/epoch keys), `selection_hash`, `environment`, `shared_operator_provenance`, `source_index_path`, `source_index_sha256`, `test_accessed=false`. Preserve original control keys/config/source/cost identities and explicit reused=true; do not present controls as new trajectories or extra independent seeds.

Compatibility requires same environment/lock, natural population/labels/arrays/row IDs/features/encoder/scaler, model/parameter order, source operator semantics, initialization, optimizer/EMA/loader/evaluator and selection. Trace the historical CLEAN source branch as C2 did; irrelevant inactive legacy config flags need not equal new config fields. Verify50epochs/batch40960/FP32, source step counts, ten initialization hashes, all500 reference permutations, and selected epochs/hashes/metrics reconstructed with strict `>` from curves. All1000 new arm-epoch permutations match the corresponding500 reference values. Do not regenerate extra randperms to check them.

**Actual availability checked by designer:** all20 result hashes and1000 epoch receipt hashes/aggregates matched. All500 CLEAN epochs contain `model_hash`, `ema_hash`, `optimizer_hash`, `epoch_order_hash`, `global_step`, `validation`, `selected_epochs`. They lack C2 rng_hash/loader_hash/continuation_hash, counts and per-epoch test_accessed; do not require or fabricate these. All CLEAN results explicitly have Test=false. C2 NQR epochs contain those four prefix hashes plus rng_hash/loader_hash/counts/continuation and Test=false. E40 validation and state hashes exist for all controls despite old common_validation omitting E40.

As formal records arrive, enforce EARLY E1–10 against same-seed constant NQR, and LATE E1–40 against same-seed CLEAN. Compare exact model/EMA/optimizer/order hashes, global_step and same-stream validation numbers; where both sides have rng_hash/loader_hash, compare those too. Do not compare whole jobs/configs/continuation hashes or cost/diagnostic fields. They legitimately differ. EARLY prefix equality is100 epoch comparisons; LATE prefix equality400. Local populated-state source/diagnostic/restore fixtures provide RNG/activity proofs absent from historical CLEAN numeric records. No claim of historical RNG measurement is made.

Any missing required record, semantic compatibility gap or exact prefix mismatch means technical PAUSE with the specific field/path/epoch/operator condition. No automatic baseline rerun expansion, approximate hash acceptance, tolerance relaxation, state reconstruction or source repair. A design-level unresolved compatibility condition returns REVISE to the parent.

### 7.3 RAM owner, snapshots, ACKs and recovery

Parent alone launches frozen TV `serve(str(N))` in a current-conversation foreground exec **after distinct implementation review**. Record new STORAGE_OWNER root/PID/session/socket identity, endpoint hash, frozen server file hash and new source/design hashes; guardian.json stays private0600, auth never in receipts/archive. No background AI/driver, codex exec/resume/fork, tmux/nohup/systemd, detached session, autonomous.py, replacement guardian or old ensure_guardian.

Protect TV PID1025015, C1 PID1570727/session59365, C2 PID1944719/session52263, and their states including C2's30 selected pairs/286526880 bytes. No old endpoint IPC, keys, mirrors, restarts, releases or journal appends. Parent inventories live ownership; designer/coders do not claim fresh live verification. New clients always route to N's own endpoint.

Use the C2 snapshot/commit/continuation schema and transaction order, with minimal added fields. RAM snapshot includes job, model, ema, optimizer, rng, loader, epoch/global_step/source counts, selected_models/epochs/validation, best, common_validation, initial_hash, costs/physical_worker_seconds and peak allocated/reserved/full-process bytes. Add `clock_kind`, `next_epoch`, `schedule_epoch`, `schedule`, `next_schedule`, `phase_counters`, and `adam_step_counters` (source name→integer step or null if uninitialized). Record source parameter activity at the last update as `last_parameter_activity` (source named-parameter order, bool values), for checking without synthesizing gradients. This numeric activity observation never changes grad=None. Schedule counters are observations, not optimizer state.

At real epoch boundaries require5.3, derive Adam counters from the populated optimizer and verify snapshot values; counter differences across schedule boundaries obey actual activity. Checkpoints preserve all source RNG streams and private loader, selected state history and next whole epoch. No partial-epoch scientific resume. Every tensor/subdict snapshot is deeply isolated with frozen cpu_copy; no references back to live optimizer or a saved baseline.

`restore` must load model/EMA from saved state and **`opt.load_state_dict(cpu_copy(state['optimizer']))`**, then loader and source RNG last. Each restore gets fresh step/exp_avg/exp_avg_sq storage. Restore mutable counters/costs/selection state through independent copies too; no CPU alias into retained RAM or fixtures. Check saved baseline byte/state hashes remain unchanged after updating each restored clone. Do not clear/recreate moments or reset schedule counters at a boundary.

After completed training+validation: create immutable numeric epoch record, then send matching CPU snapshot using frozen BytesIO IPC to N, get RAM_COMMITTED ACK, get retained state and verify exact continuation hash, then persist numeric ACK binding key/physical attempt/epoch/absolute epoch-file path+hash/continuation hash/source/design/owner endpoint. The frozen server ACK alone is epoch-only. Preserve source record keys and add the clock/schedule/phase/Adam/activity fields above; hash the snapshot before adding receipt-only hashes. Numeric epoch includes model/EMA/optimizer/RNG/loader/selected hashes and batch-log hash. Update acknowledged frontier only after this complete verification. Release temporary snapshot references after ACK, never acknowledged selected states.

The server commits epoch=r.epoch+1 and compacts at50. New validate_epoch_state enforces formal equations and exact hashes; engineering validation follows7.5 explicitly. Source `counts` remains the two-key dict. Final result must carry all final counters/schedules/provenance before `complete` compacts RAM to selected_models/epochs/validation. Keep selected online/EMA pairs for all20 new trajectories through parent storage acceptance and scientific analysis; do not release to fit more workers. Preserve all prior pairs.

For uncertain commit/reconnect, reconcile owner status, journal, get/state hash and numeric record before retrying; a recovered warning is not a fatal scientific result. Recovery attaches to the same N owner, checks identities and last ACK, and resumes next whole epoch only. All registered workers must be dead before server global recover. Never replay an acknowledged epoch or a completed trajectory. Epoch0 reinitialization needs proof of epoch0/no-state/no-ACK and retained failed physical-attempt cost. Loss of acknowledged RAM/owner is technical PAUSE. Numeric hashes are not restorable tensors. An acknowledged epoch50 finalizes without more training. Retain orphan numeric/uncommitted costs separately and exclude them from useful curves.

Before/after implementation and smoke/formal/export, check the explicit frozen source closure and protected-tree file membership/bytes, and read-only old TV/C1/C2 journal lengths/hashes plus result path/hash inventories. Store inventories only under N/A. No old source-manifest generators or owner IPC are necessary. Mechanical costs and transport timing belong to new numeric receipts. No Tensor/model/EMA/optimizer/checkpoint file, temp checkpoint, mmap, archive tensor or path-based torch.save is permitted; RAM/BytesIO only.

### 7.4 Resource admission and queue

Reuse the audited pure predicate or an identical new local caller: independently per GPU, CURRENT NVML_used/NVML_total<.85, CURRENT measured power/power_limit<.85, CURRENT host RAM<.90, no pending first-allocation ACK on that card, and physical fit `current_used_bytes+incoming_measured_full_process_bytes <= measured_CUDA_usable_total_bytes`. Threshold equality fails the85/90 checks; equality passes physical capacity. Re-sample after every startup and real first full forward/backward/lazy-Adam/EMA allocation ACK before another admission on that card. The other card remains independently eligible. Keep admitting queued work as both cards permit; no worker-count limit or both-card coupling.

Do not add forecast195/206W, projected85 memory, projected90 host RAM, peak-memory85 reserve or a fixed worker cap. C1 actual OOM explains the additional physical100%-usable condition: historical usable25307578368B/card versus NVML25769803776B. Its19118MiB/280W predecessor fits incoming ML4306MiB and fails KR5284MiB despite current85 passing. Preserve that focused fixture and strict thresholds/pending-ACK/independent-card/profile-growth fixtures.

Only initial engineering startup may bootstrap from historical C2 formal full-process peaks ML1384120320/KR1658847232B (smoke1377828864/1648361472B, measured context329252864B). These numbers are not Cycle03 profiles. Never add context again to a full-process peak. All four own smoke cases must measure both effective lambdas, true full/tail, lazy Adam, diagnostic temporaries, online/EMA evaluation and CPU-copy/BytesIO serialization/transfer, with per-phase PID NVML full residency, torch allocated/reserved/current/peak, measured non-Torch overhead, host fraction/RSS, CUDA-usable capacity, allocation ACK/resample and timing/bytes.

Incoming residency is the maximum of compatible own full-process observations, allocator-reserved peak plus measured compatible non-Torch overhead, and subsequent larger live observations. Carry higher observations forward. Sampling must run during phases as well as at endpoints; do not substitute torch-only memory. Local profiler writes only N. Four cases should cover each dataset on each card (assign opposite arms across cards); all cases include both lambdas. Source/resource reviewer must explicitly bind cross-arm/device compatibility using the actual hardware identity, own on/off profiles and measured device overhead; use the compatible maxima, not an unmeasured missing profile. Formal admission requires those own four-case profiles and fresh signoff. Hardware/profile incompatibility is PAUSE, not an automatic extra smoke matrix.

Longer observed KR work may be queued first with deterministic arm/seed tie order to reduce tail idle time. Historical C2 cost/concurrency observations (including26 workers) are scheduling context, not admission thresholds, caps, runtime promises or evidence of speedup. Actual OOM preserves raw evidence and prior ACK, PAUSEs for measured profile correction/review, and leaves current85 unchanged. No batch/precision/population modification.

### 7.5 Exactly four real engineering smoke cases; explicit virtual clock

After distinct code PASS, fresh current-conversation runner executes dataset×new-arm, seed2022, **four cases only**, using N's new owner. Each case has purpose=engineering_smoke and a disjoint key; all costs/metrics remain outside research tables. Engineering_check CPU fixtures use the same clock distinction but no owner.

Use the source seed2022 epoch1 natural permutation once, retain its real first full40960 IDs and last true-tail IDs (ML21927/KR40052), and record `sample_epoch=1`, `sample_batch=1/S`, row hash and B. Reusing these two index lists for the disposable boundary fixture is explicitly engineering; it is not a formal data trajectory or ten/forty scientific epochs. The loader state advances by that one actual randperm and is thereafter preserved/restored without extra draws.

Each case's four-step main fixture is fixed:

| Arm | Actual engineering step1 | Step2 | Step3 | Step4 |
| --- | --- | --- | --- | --- |
| EARLY | virtual e10, full, on | virtual e10, tail, on | virtual e11, full, off | virtual e11, tail, off |
| LATE | virtual e40, full, off | virtual e40, tail, off | virtual e41, full, on | virtual e41, tail, on |

All four main calls diagnose=true to profile full/tail diagnostic residency. Source initialization starts real step0; after two steps Adam/EMA are populated. The transition changes only schedule input; actual global_step goes2→3, never200→201/1960→1961. context.epoch is the listed virtual label; context.batch/global_step are actual fixture step1…4; clock_kind=engineering_virtual. Add sample_epoch/sample_batch as numeric engineering context. Core validates the schedule label while the engineering runner validates actual fixture counts. Formal worker rejects this clock_kind/purpose and cannot invoke a bypass.

At the populated boundary, perform natural online/EMA validation, BytesIO roundtrip and RAM commit/get/hash verification as owner **engineering epoch1**, actual global_step2. State/record epoch denotes engineering commit ordinal, schedule_epoch=virtual10/40, next_epoch=null, next_schedule=virtual11/41. Capture `engineering_steps=2`, `engineering_commits=1`, `scientific_epochs_completed=0`. Obtain an uninterrupted two-step suffix reference from an isolated restoration of this populated snapshot, then pause/re-register the actual smoke key, get and independently restore its acknowledged boundary state, and execute the same two-step suffix. Compare exact model/EMA/optimizer, parameter activity/Adam counters, loader/RNG, schedule/phase counters and validation to the uninterrupted route, excluding timings/job transport identity. Commit engineering epoch2, actual global_step4, engineering_steps4/engineering_commits2/scientific_epochs_completed0. Do not call complete (frozen server requires50).

At each of the four main steps, one isolated direct frozen-C2 call with the same inner config and diagnose=false must match the adapter's diagnosed numerical state/activity/RNG/loader/ledger/loss. This checks real full/tail on/off adapter and diagnostic neutrality together; CPU checks isolate them separately. Execute reference routes sequentially using CPU deep snapshots/restores so extra GPU model copies do not define the profile. Total is ten physical ordinary train_step calls/case: four main, four direct-C2 reference, two uninterrupted suffix; the logical acknowledged engineering frontier is four updates. Count every physical call/time, including reference replay, separately from frontier counts. No50-epoch smoke, full historical QR algebra rerun, efficacy threshold, extra seed or randomized probe.

Engineering `validate_epoch_state` checks key purpose/clock, commit ordinal1/2, actual step2/4, source counts2/4, phase totals and the registered virtual/sample labels. It must not apply e*S to virtual labels or accept engineering records into formal results. Saved baselines and optimizer step/moment tensors remain hash-unchanged and storage-disjoint after all restorations. Smoke release is allowed only for its four keys after paused/recovered state verification; no release of formal/old keys. Four real case receipts plus separate source/design-bound resource reviewer PASS (zero blockers) precede formal20. A code/source/design change invalidates the relevant review/smoke receipts.

## 8. Ablations required

Exactly LATE10, numeric CLEAN and numeric constant NQR. Required ordered comparisons are EARLY−CLEAN, EARLY−LATE, EARLY−constant, LATE−CLEAN, LATE−constant, separately by dataset, stream and each registered endpoint. No BRIER/MEAN-GAP reruns, low-lambda constant arm, changed window, gradient-dose matching arm or rescue sweep. This experiment can leave dose, opportunity, task sharing and optimizer/EMA path effects unresolved; record those explanations and return them to fresh research.

## 9. Expected logging requirements

Every new formal batch records frozen C2 return/context/ledger plus schedule and phase_increments, cumulative source counts and phase_counters, actual B/true row hash/external job key. Current and cumulative fields are distinct. Epoch records retain C2 numeric schema with the minimal fields7.3 and row-weighted L/C/A (`sum B_batch*loss/Ntrain`). Keep active/off weighted loss/mass summaries separate with row/update denominators. Off score/class BCE/Q/risk fields remain null per source convention; no fabricated counterfactuals. Aggregate moments keep observed/null batch counts, class/cell observed-count denominators and count-weighted sums; they are not pooled population variances. Distinguish scheduled active_updates from nonzero_opportunity_updates and from actual gradient norms.

At first actual batch of real epochs1,10,11,40,41,50, retain frozen data-gradient L2, actual Adam/decay parameter-update L2, active/None parameter counts, Adam counter ranges, overall and by source prefix. This is120 new diagnostic observation points, not120 independent trials or cumulative dose. No extra forward/backward/RNG. Old diagnostics missing at11/41 or absent for CLEAN remain unavailable/null with provenance; never fill zeros or rerun controls.

Each epoch online/EMA: six-cell AUC/LL/count/positive/negative, domain/task means, equal-cell Macro AUC/LL, source traffic-weighted/pooled metrics and worst-cell values, selected epochs/hashes, state/order/RNG/loader hashes, clocks/counters/ACK. Full validation is once per stream per scientific epoch; E40 is read from this existing evaluation, not an extra pass. Select each stream earliest strict maximum validation Macro AUC; no LL tiebreak or tolerance. Undefined required real validation metrics cause technical PAUSE rather than dropping cells.

Required views: **selected, E10, E30, E40, E50**, both streams for all40 trajectories. Old E40 comes from numeric epochs and matching state hashes. No extra old state snapshot is needed. New common_validation records10/30/40/50; old common_validation omissions are resolved by epoch data. Exports: ENDPOINTS400 rows, ENDPOINT_CELLS2400, CURVES4000, CELL_CURVES24000; committed new epochs1000, useful new batch records34500, new six-cell WEIGHTS207000. Keep new-only schedule rows distinguishable from historical unavailable fields. STATISTICS, MASS_SUMMARIES, DIAGNOSTICS, COSTS, SMOKE_COSTS, PHYSICAL_ATTEMPTS, PAIRED, PAIRED_SUMMARY, DESCRIPTIVE and reuse/source/storage/resource audits retain explicit inputs/hashes. Historical raw ledgers may be referenced where present, with reused=true; they never inflate the new34500/207000 counts.

Export exactly20 new manifest results plus20 explicit controls. New epochs resolve through verified RAM journal/ACK paths, excluding orphan/uncommitted files. Control epochs resolve through hash-bound reference_index. All five comparisons use matching stream/seed; retain absolute left/right metrics, selected epochs/state hashes, deltas and metric availability. Include every registered six-cell/domain/task/macro/pooled/traffic metric, with the primary Macro AUC clearly named. table arithmetic may compute paired summaries; runner/coders do not issue scientific decisions.

Record useful versus attempted/replayed updates, all physical-attempt seconds/statuses/failures, source forward/backward/probe/view counts, evaluation forwards/time, diagnostic time inclusion, serialization/IPC bytes/time, preparation/export/storage cost, peaks/RSS/current resource samples/admission reasons/allocation ACK/resample. New versus reused and smoke versus formal costs remain separate. Historical unmeasured fields are null with provenance, never zero; prior worker times under different concurrency do not establish isolated algorithm overhead or speedup. Preserve failed and paused costs without adding replay to useful scientific counts.

## 10. Acceptance criteria for reviewer

1. Independently inspect this seven-file design, both coder reports, actual seven-file implementation and source closure. Reviewer is fresh and distinct from every contributing coder; verdict PASS or PASS WITH FIXES is usable only with all blocking fixes complete. COMPLETE here is a design verdict, not reviewer PASS.
2. Core is exactly a schedule/config adapter over the frozen C2 train_step, with unchanged nested ledger/diagnostics and true external arm. Check both windows/all50 schedule values, invalid arms/epochs/types/configs before mutation, all new return keys and increments. No local QR/Adam/EMA reimplementation, algorithm tensors/RNG or shared-global mutation.
3. Finite CPU checks: active adapter versus exact frozen C2; off adapter versus direct source CLEAN from initial and populated Adam/EMA; both boundary sequences with no reset; diagnostic on/off state/activity/RNG identity; None versus explicit-zero behavior; mixed/empty/one-class/singleton support and actual tail B/mass/null delegation. Reuse frozen C2 mathematical proof/check receipts by byte hash; do not re-run its entire literal QR suite. Deterministic state/gradient identities are exact; existing scalar mass-rounding conventions are retained. No tolerance relaxation to hide numerical drift.
4. Populated optimizer restore checks must show saved state hash unchanged, step/exp_avg/exp_avg_sq no storage alias to saved state or another restored route, and exact next update/order/RNG/EMA/counters versus uninterrupted route. A fixture starting with empty Adam does not satisfy this gate. New phase totals are observed/validated from real calls and populated states; artificial setting of Adam/global counters to virtual epochs fails review.
5. New formal clock is strict at batch/epoch/restore/commit/finalization; engineering_virtual is permitted only for registered engineering keys and has actual step/commit counts. Smoke cannot claim scientific10/40/50 prefixes, bypass formal validation, create selected evidence or call complete. Both coders' APIs/configs/counter names match5.2/7.1; checks.check_runtime_contracts verifies integration without GPU/IPC.
6. Verify exact20-control index/provenance/selection/initial/order/available-prefix schemas; do not reject CLEAN for absent historical RNG/loader/test-per-epoch keys. No old endpoint access or control rerun expansion. Prefix checks use state/procedure hashes and available metrics, excluding new job/config/continuation/cost identities. Mismatch PAUSEs with a specific condition.
7. Explicit87 frozen files plus seven new files and real extras have new hash; frozen79/87 and all protected tree membership/bytes remain intact. No old scanner, root-bound engine call, pycache/write or monkeypatch. New paths, source adapter/core hash, environment/controls/design/jobs/storage/snapshots/export hashes agree. Preserve historical failed review and correction receipts.
8. Root-local BytesIO-only storage, two-stage ACK/get hash validation, source optimizer deep-copy fix, same-owner recovery, complete compaction/retention and user-stop boundaries are implemented. No owner launch/contact before code PASS. No old owner/state operation. After code PASS, parent launches new foreground N owner, runner does four actual GPU cases, distinct reviewer signs actual source/design/resource receipt, then formal RUN. Historical C2 smoke/peaks cannot replace these four cases.
9. Admission fixtures distinguish independent current85/host90 from physical usable100, include C1 predecessor, strict thresholds/equality fit/pending allocation/resample/profile growth and independent other-card opportunity. No power forecasts/projected85/projected90/fixed caps/card coupling. Profile coverage includes both lambdas/transitions/lazy Adam/full/tail/diagnostics/evaluation/BytesIO, with own measured full-process peaks and compatible device signoff.
10. Export counts/endpoints/seeds/sign conventions/selection and useful-versus-physical costs reconcile. Test stays sealed. Once the finite gates pass, proceed to the fixed matrix; add checks only for a concrete new failure that can affect this experiment.

Registered scientific decisions, made by the parent only **after all20 new trajectories/all five seeds per cell of the matrix finish and identity gates pass**:

- Primary is EARLY selected fixed_ema minus CLEAN selected fixed_ema, separately for each dataset. Report all five deltas, mean, sample SD(ddof1), descriptive95% t4 interval `mean ± 2.7764451051977987*SD/sqrt(5)`, positive/negative/zero and favorable/unfavorable counts. Seed is n=5; datasets/cells/epochs/streams are not extra replicates. No multiplicity-adjusted discovery claim.
- Both primary means≤0: `NO_CONTINUATION_SIGNAL`, no candidate champion. Exactly one positive: `MIXED_DATASET_SIGNAL`. Both positive below practical target: `POSITIVE_VALIDATION_SIGNAL_BELOW_TARGET`. No early efficacy gate, CI-sign gate or unanimous-sign gate alters the run budget.
- Per named comparison/dataset, mean Macro AUC≥.003 and each of six mean paired cell AUC deltas≥−.001: `VALIDATION_SCREEN_AUC_TARGET_AND_CELL_PROTECTION_MET`. Macro target met but any cell below−.001: `MACRO_TARGET_MET_CELL_PROTECTION_NOT_MET`. Cross-dataset EARLY efficacy target requires both datasets passing EARLY−CLEAN. A different comparison's pass cannot substitute.
- LL improvement means **negative** delta. Report mean LL decrease separately (`COMPREHENSIVE_LOSS_DECREASE` when mean Macro LL delta<0). LL is neither an AUC substitute nor an undeclared veto. A joint rank/loss claim requires both. All LL sign counts use negative as favorable, preserving the corrected C2 interpretation.
- Mechanism contrast is fixed E50 EARLY−LATE, online and EMA separately, then comparisons to constant NQR. Report selected/E10/E30/E40/E50 for all five comparisons. No posthoc LATE/constant-control champion or endpoint/seed rescue.
- **Structural selection bound:** ML CLEAN selected EMA epochs2022…26 are23,19,20,28,21, all in LATE's exact prefix. Therefore LATE selected ML EMA Macro AUC≥CLEAN per seed; equality can merely retain a pre41 maximum and is not efficacy. Report pre41/post40 selection. KR CLEAN epochs44,39,32,50,37 do not supply that uniform bound. EARLY E10 identity is not new replication.

Parent analysis examines AUC/loss, each task/domain/seed, natural sizes/prevalence/CLEAN performance, fixed model, stream and training stage. Identify stable repeated phenomena, contradictions, evidence strength STRONG/MEDIUM/WEAK, competing explanations, whether size plausibly dominates, whether a small-domain gain/large-domain decline actually exists, and the next most discriminating experiment. Baseline AUC/LL describe observed difficulty under this model, not intrinsic domain difficulty. Repeated validation adaptation constrains evidence strength.

## 11. Conditions that would falsify the hypothesis

Nonpositive primary means on both datasets reject the registered directional preservation prediction and trigger NO_CONTINUATION_SIGNAL. One positive dataset rejects a cross-dataset improvement claim. Failure of +.003 or cell protection rejects the corresponding practical claim even when a small mean benefit exists. E10-only, E50-only, LL-only, pooled-only, online-only, or improvement over failed constant NQR cannot rescue a failed selected-EMA EARLY−CLEAN primary claim.

Nonpositive E50 EARLY−LATE means in either stream undermine the stated timing prediction there; inconsistent seed signs/crossing intervals weaken attribution, without proving equivalence. Similar EARLY/LATE gains versus constant keep reduced dose as a competitor. An EMA-only advantage leaves EMA recency/history as a competitor; a selected gain with E50 loss is selection-dependent. A positive timing contrast with nonpositive EARLY−CLEAN efficacy supports only a relative timing response within a failed candidate. Equal lambda exposure is insufficient to claim phase independent of Adam/EMA history, opportunity or gradient dose.

Compatibility/source/loss/loader/clock/restore/RAM/resource failures are technical PAUSE/REVISE with explicit missing conditions, not efficacy falsification. No new baseline reruns or scientific target changes resolve them silently.

Only the original parent controls DESIGN→IMPLEMENT→distinct REVIEW→current-conversation RUN→ANALYZE→fresh native Astra→next plan. Children are fresh, fork_context=false, explicit file handoffs, permanently closed on completion. No child descendants/independent/headless/background AI or retired driver restart. `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden revision/experiment cap. Candidate NO-GO/Astra STOP routes through report/context to fresh research; technical failure PAUSEs; only explicit USER_STOPPED normally ends the outer loop.

Read both W/O artifact AUTONOMOUS_STATE manual_stop_requested/termination_reason and root/artifact STOP_AUTONOMOUS before each dispatch/admission and safe boundaries. Unreadable controls PAUSE. User stop admits no new work, completes/checkpoints the supported atomic epoch transaction safely and produces final raw/storage handoff. Retain the20 new selected pairs, source/review/reuse/resource receipts,1000 committed epoch identities, owner/session/PID/endpoint hash and selected hashes/bytes until parent accepts supervision before runner closure. No auth secrets in handoff. Test remains sealed throughout ACTIVE/PAUSED/candidate outcomes; future access requires explicit USER_STOPPED + final freeze + existing eligible full-five-seed selected-live-RAM verification gate. Reused numeric control records confer no live-Test eligibility.

Designer completion observations: both W/O file stop controls were ACTIVE/manual_stop_requested=false/Test=false, termination_reason null and all four sentinel paths absent. All six handoff bytes,87 frozen source bytes and20 result/1000 numeric epoch receipts matched. CLEAN field availability and selected epochs were inspected directly. Only P/DESIGN_SPECIFICATION.md was written. No implementation, experiment, GPU query, IPC/owner contact, source/rootstate/history/Git/publication/dependency edit or Test access occurred. Future implementation compatibility checks, actual capacity and live storage verification belong to the subsequent gated stages. Parent closes this designer permanently after this handoff.
