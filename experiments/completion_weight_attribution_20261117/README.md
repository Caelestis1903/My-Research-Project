# Cycle04 runtime handoff

Implementation root is `/home/xzf/Code/phase0-3_tvg_rm_20261116` (W). The only run-output root is `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2` (N). The formal specification is at Cycle04 **artifacts**, with six immutable research files at Cycle04 **research_plans**. `protocol.design_bundle()` verifies the accepted seven-file mapping and both bundle hashes. No command below starts an AI session or the retired driver.

All commands execute with W as the working directory. Every Python command uses `PYTHONDONTWRITEBYTECODE=1` and the explicit project interpreter. Tensors, optimizer/RNG/loader states and selected models stay in CPU RAM/BytesIO. Output JSON/CSV contains numeric receipts only. The cache loader dereferences only train/validation arrays, row IDs and features, returning a projection without Test attributes. The existing pickle container carries Test opaquely. Full raw/train/validation verification and all1020 numeric-control verifications happen in preparation, after code review.

The public interfaces are `protocol.config/job/design_bundle/source_manifest/dataset_cache/reference_index/stop_requested/request/prepare/validate_attempt/validate_formal_clock/validate_epoch_state/epoch_files/verify_selection/finite_checks`, `runner.admission/restore/commit_snapshot/smoke_case/smoke/run/recover/handoff/finite_checks` and `tables.export/finite_checks`. `runner.commit_snapshot` takes `attempt,key,snapshot,record,path,attempt_number=...`. `admission` takes a current measurement dict plus incoming measured residency, pending status and already counted incoming residency. `runner.run(...,recovery=True,recovery_review=...)` implements formal `run --recover`. `runner.smoke(...,recovery=True,recovery_review=...,case_id=...)` implements case-scoped `smoke --recover`; it operates before resource signoff.

Attempt2 restarts all twenty trajectories from source initialization after irreversible Cycle04 RAM loss. The approved restart addendum is A4/RESTART_DESIGN_SPECIFICATION.md/json with RESTART_DESIGN_ACCEPTANCE.json. Attempt1 numeric records, costs and exact48-file source_snapshot remain immutable provenance; its275 ACK epochs/7,733 updates give no completion credit to attempt2. No old-owner IPC or state reconstruction is permitted. Attempt2 has a new actual48-file source identity and20 new source-bound keys; its first per-key registration is number1, independently of the experiment attempt name. The unchanged useful budget is20 trajectories ×50 epochs, seeds2022–2026,1,000 epochs/34,500 updates.

Finish this cycle, complete analysis/report, evidence review, context and publication/delivery, then USER_STOPPED. Do not start Cycle05 or Astra. The immediate stop latch remains false during authorized completion unless a new immediate stop arrives.

## Code review and preparation

Parent freezes the final `protocol.source_manifest()` output after both coder reports and a distinct review. This function only computes identities; the coder does not create N or freeze a receipt. `files` is the explicit48-file closure from specification6.2, including all seven new package files, the lock and resolved protocol. No tree scan or historical source-manifest helper is used. The external installed Adam file is separately hashed. The expected external Adam hash is checked.

Code review JSON required fields:

```json
{
  "verdict": "PASS",
  "blocking_issues": [],
  "source_hash": "actual protocol.source_manifest()['sha256']",
  "design_bundle_sha256": "2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7",
  "coder_ids": ["all contributing core/runtime/repair coder identities"],
  "reviewer_id": "fresh independent reviewer identity",
  "test_accessed": false
}
```

Use all four actual contributing coder IDs: core Volta `01a1e54b-0376-79c0-b1a7-20a6893345c7` (A4/CORE_IMPLEMENTATION_REPORT.md), runtime Chandrasekhar `01a1e54b-03a0-7613-a537-dba4cc0f0e80` (A4/RUNTIME_IMPLEMENTATION_REPORT.md), R1 Leibniz `01a1e6cd-54c0-7a11-80ca-fb7194be1f15` (A4/R1_IMPLEMENTATION_REPORT.md), and restart coder `01a1e747-c8be-7a73-af93-5a9f2966d3b1` (A4/RESTART_CODER_IMPLEMENTATION_REPORT.md). The existing at-least-two-coders check remains unchanged. The reviewer must be excluded from the entire coder set. `PASS WITH FIXES` requires an empty blocking list after correction. Actual source and design hashes must match.

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.runner prepare --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/RESTART_IMPLEMENTATION_REVIEW.json
```

The canonical absolute N is required. Any missing cache, data hash, numeric control, review or identity produces a technical pause. Preparation uses no storage IPC and never reconstructs data. It writes the immutable new source/run/data/control receipts and20 formal jobs. The shared numerical aliases are frozen helpers, never old root-bound preparation, source-cache, runner, admission or export functions.

## Parent-owned foreground RAM storage

Only the original parent starts the new foreground owner after code PASS. N must exist first. Keep its foreground tool session live:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -c 'from experiments.task_view_gradient_residual_momentum_20261116.runner import serve; serve("/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2")'
```

Parent records N/STORAGE_OWNER.json with exactly these consumed fields:

- `pid`: integer from the new N/guardian.json.
- `tool_session`: new foreground tool session identifier (nonempty).
- `source_hash`: new source receipt `sha256`.
- `design_bundle_sha256`: accepted seven-design hash.
- `server_file_sha256`: SHA256 of W/experiments/task_view_gradient_residual_momentum_20261116/runner.py.
- `endpoint_hash`: `protocol.digest({'address': guardian['address'], 'pid': guardian['pid']})`.

An optional `address` is accepted in that **local private** receipt if it equals guardian.address and the receipt mode is0600. `guardian.json` must be0600. Do not publish either local file or address/authkey/socket/connection material. Public owner/handoff/audit records contain only PID, session, source/design/server and endpoint hash. The wrapper verifies these identities and accepts only N. Old owners1025015/1570727/1944719/2097722 are rejected and never contacted. Parent must also verify that the new owner differs from lost Cycle04 owner2429109; no historical owner is contacted or replaced in place.

The frozen server accepts `status`, `register(key,job,pid)`, `allocated(key,peak_memory_bytes)`, `commit(key,state,epoch_file)`, `get(key)`, `paused(key,reason)`, `failed(key,failure)`, `recover(key)`, `complete(key,result)` and `release(key)`. Its `get` returns only `{state,result}`. Commit returns `{status:'RAM_COMMITTED',epoch}`; its journal is `epoch_commit_ledger.jsonl`, rows `{key,epoch,attempt,epoch_file,time,status}`. The consumer computes hashes itself. There is no added server command, fake failure injection or invented owner hash/ACK field.

A transaction writes numeric pending record and request timestamp, serializes CPU state through BytesIO, receives ACK, gets/status-checks actual RAM, validates full returned continuation/component/scientific hashes, binds the actual journal row, then writes numeric ACK. Lost reply reconciliation uses only this owner's get/status/journal and original pending request time; it never blindly resends a commit. Local ACK lives at `N/acks/<key>/epoch_XX.json`; physical transactions live at `N/transactions/<key>/attempt<number>/epoch_XX.json`. Each ACK includes `transaction_file` and `transaction_sha256`, binding the one journal-selected committed transaction. Failed physical transactions remain immutable. An uncommitted next attempt writes a new transaction; a committed lost ACK is reconciled from that exact journal attempt/path without resending. Full continuation hashes bind each actual payload separately. Extraction accepts only ACK-bound numeric records, never the last lexicographic attempt file.

## Real engineering and resource signoff

Parent dispatches a fresh DEFAULT native `gpt-6-luna/high`, `fork_context=false`, RUN-only child, verifies its actual resolved model/reasoning/thread metadata before GPU work, and supervises this command. Use the documented five-seed DEFAULT role correction without global role edits:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner smoke --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/RESTART_IMPLEMENTATION_REVIEW.json --devices 0,1
```

The four registered cases are GPU0/ML/W, GPU0/KR/S, GPU1/ML/S, GPU1/KR/W. Each case has a fixed verified plan of eight calls: diagnosed lazy full, tail and prefix commit1; source and lambda0 full forks from that prefix; expected full and diagnosed populated tail with both validation streams; actual pause/re-register/get/BytesIO/deep restore; replay full/tail with diagnostics off and both validation streams; suffix commit2. Exact gradients/activity/scientific identity and immutable prefix are required. Server ordinals1/2 correspond to logical steps2/4. Scientific epochs stay0. Engineering keys are never sent `complete`; they remain paused until acceptance.

NVML sampling starts before CUDA/model allocation. The main worker explicitly initializes its own device context, enables allocator counters immediately, and records context overhead before constructing the model/EMA/Adam. The sampler never initializes CUDA. A5ms mechanical sampling thread records own-PID total residency, allocator memory, non-torch overhead and phase peaks. Synchronized phase boundaries cover lazy/full/tail/diagnostics, online/EMA validation, copy/serialization/get/restore/replay. Both arm and target-card real profiles feed each dataset envelope. Hardware name/compute capability/CUDA-usable capacity, source/design and environment must match. Formal placement has no arm/card pinning. Compatible live observed peaks grow this envelope.

Parent obtains a fresh independent resource review and writes N/ENGINEERING_SIGNOFF.json. Required fields:

```json
{
  "verdict": "PASS",
  "blocking_issues": [],
  "reviewer_id": "fresh reviewer distinct from all coders and code reviewer",
  "source_hash": "actual new source hash",
  "design_bundle_sha256": "accepted seven-design hash",
  "cases": [{"path": "absolute smoke/.../case.json", "sha256": "actual bytes hash"}],
  "profiles": [{"path": "absolute smoke/.../case_profile.json", "sha256": "actual bytes hash"}],
  "verified_plan_counts": {"physical_train_calls": 32, "diagnostic_points": 8, "extra_autograd_sweeps": 24, "virtual_adam_steps": 32},
  "physical_train_calls": 32,
  "diagnostic_points": 8,
  "extra_autograd_sweeps": 24,
  "virtual_adam_steps": 32,
  "scientific_epochs_completed": 0,
  "test_accessed": false
}
```

The numeric example gives the no-failure totals. The four actual physical totals must instead equal the sums of the four case receipts, including failed/replayed returned calls. `verified_plan_counts` stays exactly as shown. Per-case `verified_plan_counts` is `{physical_train_calls:8,diagnostic_points:2,extra_autograd_sweeps:6,virtual_adam_steps:8}`; its actual returned-work fields can be larger. Unknown unreturned work remains separate in physical attempt records.

There must be four actual case receipts and four profile receipts. Per-case receipts must prove eight calls/two diagnoses/six extra sweeps/eight virtual steps, exact source/lambda0 and replay/neutrality, immutable prefix and matching measured profile. Parent writes N/ENGINEERING_RELEASE_ACCEPTANCE.json with `keys` (exactly these four engineering keys), `parent_accepted:true`, and absolute `review_path` to the independent engineering signoff. Parent then uses **this package's** `protocol.request(N,'release',key=...)` for those four keys only. Formal admission requires their actual owner `state_released` flags. Selected formal pairs are retained even for failed candidates.

### Engineering recovery before resource signoff

For a technical failure, a fresh independent reviewer supplies N/ENGINEERING_RECOVERY_REVIEW.json using the review/source/design/Test fields above, plus `scope:"engineering"`, `case_keys:["exact existing case key"]`, `last_acks:{"that same key":0 or 1 or 2}`, `owner_endpoint_hash`, and `pause_sha256` of the retained N/TECHNICAL_PAUSE.json. The scope contains exactly one of the four fixed keys. `resource_profiles` can bind available measured receipts; a completed engineering signoff is not a prerequisite. Every registered worker must have exited before recovery. The code review path below is the fresh parent-accepted restart review; recovery applies only within attempt2 with its live matching RAM owner.

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner smoke --recover --case-id case1_gpu0_movielens --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/RESTART_IMPLEMENTATION_REVIEW.json --recovery-review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/ENGINEERING_RECOVERY_REVIEW.json --devices 0,1
```

The same registered key is reconciled via its actual status/get/journal and passed through frozen recover only when FAILED/dead RUNNING. Ordinal0 restarts its exact initialization. Ordinal1 restores its populated prefix, saved permutation, model/EMA/Adam, loader and RNG, then genuinely reruns the source/lambda0 forks and expected/restored suffix proofs. Its original prefix calls remain part of the fixed eight-call proof. Every repeated physical call retains its actual owner attempt in `smoke/<case>/attempt<number>/batches.jsonl`; physical attempt segments and restore costs retain those numbers too.

The verified suffix's epoch record binds `verified_plan.json` before commit2. That proof binds the original prefix ACK, hashed call logs, actual fork/replay observations, lazy/populated diagnostic evidence, exact expected/replayed scientific identity and both validation passes. With an already acknowledged suffix, recovery verifies this proof against the retained suffix and finalizes the case without training calls. Missing or changed proof blocks finalization. Original physical `resource_profile.json` files remain; `case_profile.json` combines their measured maxima with original profile receipts and phase provenance. Case completion remains pending the independent four-case resource review and parent release acceptance.

A successful one-case recovery returns ENGINEERING_CASE_VERIFIED_PENDING_OTHER_CASES until all four case receipts exist; the full matrix then returns ENGINEERING_PENDING_INDEPENDENT_REVIEW. Other unfinished retained cases each require their own reviewed recovery. A later ordinary smoke invocation skips already completed case receipts and can start untouched cases.

## Formal supervision, stop and recovery

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner run --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/ENGINEERING_SIGNOFF.json --devices 0,1
```

Queue: KR before ML, ascending seeds2022..2026, W then S; choose the first fitting queued job independently per card. Admission requires current measured memory<85%, power<85%, hostRAM<90%, then current used+incoming **measured** full-process envelope<=fresh CUDA-usable capacity. Before Popen, capture that key’s prior owner registration count (zero if absent); pending records bind key, launched PID and expected registration count+1. Pending first allocation blocks only that card until a fresh owner status matches the active live child, exact job, worker PID, expected registration attempt, RUNNING status and allocated=true. Registration resets allocated=false; only the new generation’s actual first FULL Adam allocation ACK qualifies. Allocation ACKs record worker_pid; STARTUP_RESAMPLE and PENDING_ACK_RELEASE_RESAMPLE bind PID/expected attempt, with observed PID/attempt on release. Resample before another admission. Matching child exit clears its pending record through exit handling; a nonzero exit PAUSES without an allocation-release event. Calibration may retain its pending card until its new phases are measured. No worker cap, projected85% memory/RAM or forecast power is used.

Both W/O manual flags/status and both root/artifact STOP_AUTONOMOUS sentinels are read at starts and atomic step boundaries. USER_STOPPED halts admissions and recovery. Complete parameter/EMA calls safely; a complete epoch may validate/commit, while partial epochs stay explicitly uncommitted. Technical failures preserve last ACK, numeric partial logs and physical attempt costs, set N/PAUSE_TRAINING and N/TECHNICAL_PAUSE.json, and return to parent for fresh review. They never become a scientific result or an outer-loop stop.

N/RECOVERY_REVIEW.json must have passing review/source/design/Test fields above, a fresh `reviewer_id`, `owner_endpoint_hash`, `pause_sha256` of the actual TECHNICAL_PAUSE.json, `last_acks:{key:integer_epoch}` for every unfinished formal key in the owner, and `resource_profiles:[{path,sha256},...]` binding measured resource evidence. No registered workers may remain live. Parent automatically dispatches a fresh RUN-only child after that review:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner run --recover --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/ENGINEERING_SIGNOFF.json --recovery-review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2/RECOVERY_REVIEW.json --devices 0,1
```

Recovery reconciles missing local ACKs against existing actual RAM/journal, uses frozen server recover only for failed/dead RUNNING keys, registers PAUSED keys directly, restores deep-copied populated optimizer state and RNG last, and replays only uncommitted work. COMPLETE compacted keys are excluded. Epoch50 acknowledgement can complete without another training call. Retry/experiment/cycle/revision totals have no hidden cap. Physical logs remain outside restored logical counters.

## Snapshots, extraction and selected RAM handoff

Snapshot `counts` is **exactly** `optimizer_updates,ema_updates`; expanded integer counters are in `operations`. Stable `scientific_identity_hash` excludes diagnostic work and physical timing/attempt ordinals; `continuation_hash` binds the full actual CPU payload including its captured costs. Diagnose-on/off comparisons also compare actual gradients/activity. Formal clock is whole epochs0..50 with next_epoch51 after50; all1000 ACKs must cover actual natural tails. Engineering has its separate clock/selection observation ordinal2.

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.tables --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.completion_weight_attribution_20261117.runner handoff --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt2
```

Tables export the specification9.2 families with exact source metrics, selected/E1/E10/E30/E50, all50 curves,15 comparisons, five individual paired seeds, sample SD and registered descriptive interval, masses weighted by actual full/tail rows, scheduled class statistics and flattened actual core common-state group/objective/pair schema. They preserve missing historical fields with `not_recorded_in_source`. Extraction accepts exactly400 endpoints/2400 cells/4000 curves/24000 cell curves/207000 weight rows/1000 phase counters and the registered logical totals. It makes no candidate decision.

`handoff` directly gets all20 selected online/EMA pairs and verifies chosen epochs/hashes/metrics,50 ACKs each and actual191017920 tensor bytes. N/RUNNER_HANDOFF.json is numeric/public. Parent must complete direct selected-RAM get/hash/selected validation acceptance and N/PARENT_STORAGE_ACCEPTANCE.json before closing the formal runner; any required actual evaluator recheck is additional measured work. Test stays sealed through ACTIVE/PAUSED/all candidate outcomes. Only explicit USER_STOPPED+final freeze+the existing eligible selected-RAM gate permits later opening, under parent control.

COSTS.csv includes `cost_kind='diagnostic'` for every returned core batch across committed, failed and replayed physical logs: `key,attempt,purpose,context,call,seconds,seconds_semantics,extra_autograd_sweeps,virtual_adam_steps,source_path,source_hash,design_bundle_sha256,null_reason`. `seconds` is the returned `diagnostic_seconds`, including measured zero on ordinary calls. Incomplete physical batch logs add a null-duration record with `incomplete_physical_batch_log`. `cost_kind='restore'` comes from N/costs.jsonl, measured around the real deep state restore with CUDA completion barriers and RNG restored last; it includes actual attempt and role/resume ACK. Engineering validation streams, snapshot copies, owner get and BytesIO replay costs are written as they return, so a subsequent failure preserves them. Every physical transaction also retains its own measured BytesIO preflight cost. Logical epoch summaries and committed ACK transfer summaries remain labelled separately; their overlapping components are not additive independent durations. SMOKE_COSTS.csv includes the engineering physical diagnostic/restore/serialization rows and case summaries. Unknown partial work stays null; no duration is inferred by subtraction.

CPU syntax uses `compile(source_text,filename,'exec')`, never py_compile. For the restart repair, compile only edited Python files, reuse the exact unchanged core/checks evidence and run the following bounded checks once:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -c 'from experiments.completion_weight_attribution_20261117 import protocol,runner,tables; print({m.__name__:m.finite_checks() for m in (protocol,runner,tables)})'
```

The CPU checks use finite in-memory files/owner transcripts, mocked NVML/CUDA telemetry, CPU actual-core recovery proofs, and actual numeric CLEAN/NQR schema receipts; they perform no GPU query, IPC, actual preparation or training launch. The runtime finite check includes eight production-run scheduler cases: absent status, stale PID+attempt, right PID/wrong attempt, wrong PID/right attempt, allocated=false, qualified release/resample before next launch, independent GPU1 admission while GPU0 is pending, and nonzero exit/PAUSE without ACK release. Parent alone performs analysis/report, evidence review, context and publication/delivery, then USER_STOPPED; no C05/Astra.
