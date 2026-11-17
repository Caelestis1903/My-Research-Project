# C05 CELL-MASS-BCE-025 runtime

Accepted design: `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/DESIGN_SPECIFICATION.md` (SHA256 `a1f61ae81535bb33873da72185d9400187c5b988e81c0d3a5019c5505e821af9`). U is the only new arm: two datasets, seeds2022–2026, 50 epochs, batch40960, FP32. C/W/S are 30 numeric controls (1530 files). If an original numeric control is unavailable at preparation, its exact bound C04 archive member may be recovered into R5/reused_controls with byte/hash verification; no historical tree is modified. Test stays sealed. The parent performs scientific analysis after raw acceptance and stops after C05 delivery.

Run every command from `/home/xzf/Code/phase0-3_tvg_rm_20261116`. W is that source workspace. O is `/home/xzf/Code/phase0-3`. R5 is `/tmp/phase0-3-c05-01a1e9a9/attempt1`; N5 is W's `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/attempt1`; S5 is `/tmp/phase0-3-c05-01a1e9a9/publication`. No cross-root symlink is used. R5 contains all physical raw work, source snapshots, jobs, ACKs and owner credentials. N5 contains compact mapping/review/publication metadata and archive parts. S5 retains the complete archive, index and parts.

All model/EMA/Adam/RNG/loader/BytesIO state stays in RAM. Commands below are downstream stage instructions; coder execution is limited to finite CPU checks. The source closure is the exact immutable C04 source48 receipt plus the seven C05 package files. Preparation freezes the 55-file snapshot only after independent implementation review.

## Commands

Numerical CPU fixtures:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.checks check
```

Runtime finite checks (no data loading, owner IPC or GPU queries):

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -c 'import json; from experiments.cell_mass_bce_20261117 import protocol,runner,tables; print(json.dumps({m.__name__:m.finite_checks() for m in (protocol,runner,tables)},allow_nan=False))'
```

After fresh distinct code review:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.runner prepare --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/IMPLEMENTATION_REVIEW.json
```

The original foreground parent then starts the existing generic RAM server in its own foreground tool session and polls it at intervals of at most60 seconds through delivery:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -c 'from pathlib import Path; from experiments.task_view_gradient_residual_momentum_20261116.runner import serve; serve(Path("/tmp/phase0-3-c05-01a1e9a9/attempt1"))'
```

Parent writes private0600 R5/STORAGE_OWNER.json using actual pid/tool_session, endpoint_hash=digest({address,pid}), source_hash, design_bundle_sha256, research_bundle_sha256, physical_root, logical_root and server_file_sha256. The generic server writes private0600 guardian.json. Public receipts retain PID/session/endpoint hash/root/source identities and omit address/authkey. Existing storage owners and co-tenants are preserved.

After parent owner acceptance, supervised mechanical engineering:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.runner smoke --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/IMPLEMENTATION_REVIEW.json --devices 0,1
```

Exactly four U seed2022 cases: case1_gpu0_movielens, case2_gpu0_kuairand, case3_gpu1_movielens, case4_gpu1_kuairand. Each has eight returned calls, two diagnostic points, six sweeps, eight virtual source-Adam steps, two owner commits and zero scientific epochs. Expected/replayed suffix observations and validation match exactly. Failed/unreturned attempts remain separate physical work. Engineering's initial measured residency envelope is explicitly bound to C04's four profiles; formal admission consumes reviewed C05 witnesses only.

After independent resource signoff, direct parent acceptance of eight ACKs/proofs, and parent-authorized release of exactly four engineering keys:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.runner run --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1 --review /tmp/phase0-3-c05-01a1e9a9/attempt1/ENGINEERING_SIGNOFF.json --devices 0,1
```

Queue is KR2022..2026 then ML2022..2026. Each card independently admits when fresh current memory<85%, power<85%, host RAM<90%, no unresolved current allocation, and current used+measured incoming<=100% actual CUDA-usable capacity. There is no worker cap, projected85% gate, predicted-power term, cold-fit bypass or both-card requirement. Pending release binds live PID/job/current RUNNING registration generation and an actual first FULL call allocation receipt. A second engineering registration obtains its own FULL ACK. Re-sampling follows startup and allocation acknowledgment.

Recovery uses the same owner and source, a fresh distinct recovery review, and `--recover --recovery-review /absolute/path/to/review.json`. Engineering additionally selects `--case-id caseN_gpuD_DATASET`. Recovery consumes pause_sha256, owner_endpoint_hash, exact last_acks, measured resource_profiles; engineering requires scope=engineering and its single case_keys list. Never replace a lost owner, restart old owners, replay completed jobs, or recreate unavailable C04 models. Technical failure pauses; user stop reaches an atomic safe boundary. Both workspace controls and all four stop sentinels are read before admission.

Raw handoff/export, followed by pack only after parent storage acceptance:

```sh
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.tables export --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.runner handoff --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1
env PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m experiments.cell_mass_bce_20261117.tables pack --attempt /tmp/phase0-3-c05-01a1e9a9/attempt1
```

Internal `_worker` consumes --attempt/--key/--device; `_case` consumes --attempt/--case-id/--device/--dataset/--arm and optional --recovery-review. They are launched as ordinary supervised project-Python subprocesses. No AI/session/driver is started by runtime.

## Consumed review schemas

Every review is JSON, verdict PASS or PASS WITH FIXES, blocking_issues=[], test_accessed=false, exact source_hash/design_bundle_sha256/research_bundle_sha256. PASS WITH FIXES additionally supplies nonempty fixes_closed receipt references. Code review records actual parent_id/designer_id/coder_ids/reviewer_id; reviewer differs from all coders and designer. `source_files` is source_receipt.files and `external_runtime` pins installed Adam. `controls_inputs` contains source_reference_sha256 and manifest_sha256. `coder_reports` and `finite_check_receipts` are lists of {path,sha256}; runtime checks every referenced byte hash.

ENGINEERING_SIGNOFF.json additionally consumes actual distinct reviewer_id, code_reviewer_id, coder_ids, owner (the public owner_binding dictionary), physical_root/logical_root, environment, cases/profiles lists of {path,sha256}, verified_plan_counts={physical_train_calls:32,diagnostic_points:8,extra_autograd_sweeps:24,virtual_adam_steps:32}, actual physical totals under those same four top-level keys, scientific_epochs_completed=0, validation_chunks=80, four true flags source_lambda_zero_exact/diagnostics_neutral_exact/prefix_immutable/actual_virtual_exact. `parent_engineering_acceptance`, `code_review`, `terminal_core`, `supervisor_evidence` are byte-bound receipts; the parent receipt has parent_accepted=true, owner and exactly eight ACK receipts. `registry_prefix` has path/bytes/records/sha256; `witnesses` has witness_id references to actual C05 engineering rows within that prefix. `envelopes` is a nonempty list of version receipts; `required_phase_coverage` equals actual profile phase union; `physical_totals` records all attempts/additional/null partial work; partial_work_policy=unknown_is_null. Cases, proofs, profiles, call logs, native identities, matrix, physical arithmetic, raw prefix hashes and owner binding are checked before launch.

ENGINEERING_RELEASE_ACCEPTANCE.json has parent_accepted=true, exactly four keys, review_path, signoff_sha256 and accepted proof/ACK references. Parent invokes protocol.request(R5,'release',key=...) for those keys only. Formal launch checks actual storage state_released for every key. Formal states are never released.

PARENT_STORAGE_ACCEPTANCE.json used by pack has parent_accepted=true, source_hash, design_bundle_sha256, owner, test_accessed=false; parent also records actual500 ACK/17250-update totals,20 selected model identities/tensor bytes, reuse/table/cost receipts and liveness, as required by the design. N5/PACK_INPUTS.json explicitly lists public extra stage receipts as {path,sha256,logical_name}, and binds source_hash/design_bundle_sha256. Allowed receipt roots are A5/N5/R5; C04 publication/availability/archive-manifest receipts are explicit additional inputs. Parent redacts credentials before supplying public receipts.

## Numeric evidence and joins

Native query dictionary names all six NVML v2 functions. At most three fresh complete attempts per incident retain return code/device/phase/PID/sample-attempt/capacity facts, never mixed or stale substitutes. Profile cadence is .005s with synchronized phase marks. Raw compressed process logs contain ordered queries/samples/Torch-phase rows once. Profile summaries and append-only producer/witness journals reference raw byte/record prefixes. Each immutable envelope binds a registry prefix, exact scope, maxima and witness IDs; admission rows reference current sample and version. Resource audit contains journal references and summaries rather than copied history.

Local STARTED and TERMINAL_CORE precede registration/cleanup dependencies. Terminal core preserves returned operations, pending ordinal, last ACK, original exception and unknown unreturned work. Sampler/owner cleanup failures append separate errors. Supervisor records observed exit/signal/wall time and absent-terminal nulls independently. Worker and controller wall durations are not added.

Model parameter dictionaries store ordered names/dtypes/shapes/numels and prefix membership once. Batch activity/private-step lists join parameter_dictionary_id and ordered index. Diagnostics are stored once in diagnostics/<point_id>.json; flattened numeric metrics join point_id/group/objective-or-pair/metric. Epoch job_key joins jobs/<key>.json. Epochs bind batch log hash, RAM continuation/component/scientific identity and transaction/ACK/server-journal facts. Strict validation AUC chooses earliest maxima separately for online/EMA; LL belongs to the same selected state. Engineering lambda0 H is explicit null/not_computed_delegate.

Export counts: ENDPOINTS400, ENDPOINT_CELLS2400, CURVES4000, CELL_CURVES24000, SELECTION_STRUCTURE80, WEIGHTS103500, PHASE_COUNTERS500. Exactly17 paired families ×2 datasets ×5 endpoints ×34 metrics ×5 seeds=28900 rows; summary5780. Paired SD uses ddof1 and mean±2.7764451051977987*SD/sqrt(5). Epoch has no favorable direction. Missing history retains null reasons. schema.json documents joins/formulas/units; TABLE_RECEIPT.json binds outputs. Runtime computes arithmetic and physical costs; parent makes research decisions.

Pack maps R5 suffixes to attempt1/<suffix>,30 explicit controls to reused_controls/<symbol>/<dataset>/seed<seed>/<key>/{result,epoch_NN}.json, seven inputs to design/<filename>, source55 to attempt1/source_snapshot/<repo path>, public receipts to receipts/<logical name>. Only checked regular files are included. Credentials, opaque state formats, symlinks and unstructured worker stdout logs are excluded; terminal failures remain recoverable in numeric JSON. Archive readback verifies every header/path/length/hash. S5 retains unsplit archive/index/ordered8MiB parts; N5 receives parts and compact manifest/index only. Workspace staging requires measured3A+256MiB+1GiB headroom. No model state enters an archive.

Hardware identity uses context-free CUDA driver device queries; NVML samples create no worker context. Handoff follows export so its table receipt binds actual outputs. It reconciles all500 ACKs, useful17250 operations, physical call/unknown work, live selected tensors and current owner/review/resource identities.
