# MAINTENANCE PLAN — preserve resident prefixes after reader numerical failure

Status: DESIGN_ONLY. No runtime file changed, no maintenance code implemented, and no IPC command sent by this task. The immediate objective is to export the exact failed `FULL_SOURCE`, target0/task0/group0, lambda0.001, P_read numerical fixture from the existing MovieLens seed5201 context. Its recorded size is2359×64; implementation must verify the actual size rather than assume it. The subsequent solver choice belongs to the separate numerical designer and independent reviewer.

## Why the existing ordinary API is insufficient

The existing authenticated local `Client/Listener` accepts only the fixed dispatch commands. `status` returns resident identities; `prefix` returns an already-resident identity; neither exposes the P_read matrices or reloads a module. A new external Python process cannot access another process's `prefix._LIVE`. Restarting the service or a worker would discard model/Adam state and force unnecessary prefix reconstruction. No existing normal command therefore performs the requested fixture extraction or later reviewed reader reload.

The service and worker pipes already use Python pickle for trusted task-owned commands. One narrowly scoped, source-reviewed `maintenance.py` can use a typed command object's `__reduce__` to call its whitelisted maintenance function during deserialization. This is an explicit extension of our authenticated local execution path, not an arbitrary-code or general RPC interface. The helper returns an ordinary existing command dictionary so the unmodified loop continues. All arguments are task-owned action enums, fixed identity fields and an authorization-receipt path; no expression, shell command, arbitrary module/function name, model, optimizer or RNG payload is accepted.

## Exact interfaces and actions

The new module exposes only these two action values:

1. `READ_FAILED_READER_FIXTURE` — allowed initially and read-only with respect to trained objects.
2. `RELOAD_APPROVED_READERS` — unavailable until the numerical designer's concrete change, implementation and independent review have produced a separate authorization.

Use immutable typed data:

- `MaintenanceRequest(action, artifact_root, config_hash, authorization_path, target_dataset, target_seed)`.
- `ControllerEnvelope(request).__reduce__` resolves only to `controller_maintenance(request)`.
- `WorkerEnvelope(request).__reduce__` resolves only to `worker_maintenance(request)`.

Authorization is a dedicated JSON receipt under the fixed new artifact root's `maintenance/` directory. It contains schema/version, exact action, expected controller PID, expected worker PID/dataset/seed, config hash, permitted artifact root, expected helper file SHA256, reviewed verdict and zero blocking issues. Reload authorization additionally contains the approved new `readers.py` file SHA256 and the preceding fixture receipt identity. A generic earlier training PASS does not authorize maintenance. The client verifies the helper hash before connecting; controller and worker independently verify the token and helper hash before locating objects or performing each action. A successful shared-file import is not itself treated as authorization.

## Controller sequence

1. Client uses the existing socket path/authentication derived from the unchanged config and transmits the reviewed controller envelope. No listener/socket is replaced and no unrelated process is accessed.
2. The controller helper catches all errors internally. It uses `gc.get_objects()` plus `isinstance(obj, Campaign)` to find exactly one Campaign whose resolved root/config/PID match the request. It never examines caller frames.
3. Require target worker exists, its PID matches authorization, `process.is_alive()` is true, its tracked pending command is None, and no unconsumed reply is present. Any mismatch records refusal without sending a worker action.
4. Send one worker envelope through that worker's existing pipe, record the ordinary returned reply, and save a small maintenance receipt. All other workers remain untouched for fixture extraction. Reload may list all four workers explicitly, with independent PID/scope verification for every worker.
5. Return the ordinary dictionary `{'command':'status','args':{},'config_hash':...}`. The existing server then runs its unmodified status dispatch and replies normally. The detailed action result is in the owned receipt, not in a serialized model payload.

The server processes one accepted client at a time; the action runs only after the failed S1 client has returned and the target pipe is idle. No overlapping training operation is interrupted.

## Worker sequence and error containment

The worker helper also catches `BaseException` internally because deserialization occurs before the old worker loop's try/except. It must never propagate a numerical/token/file error through `conn.recv()` and kill the resident worker.

1. Validate the dedicated authorization, helper hash, action/config/root and current PID.
2. Import the already-resident prefix module and find exactly one matching completed bundle in `prefix._LIVE` by dataset, seed, config and resolved output root. Do not rebuild a prefix or call any trainer if the lookup fails.
3. Compute before hashes of the live native, source/local/initial pilot and B model states; only hashes are written. No weights, optimizer dictionaries, RNG states or module objects are serialized or transmitted.
4. Execute the authorized action.
5. Recompute the same model hashes and require exact equality. Save a small result/refusal receipt.
6. On success, return `{'op':'prefix'}` only after unique resident-bundle proof. The old worker's local bundle is already present, so that existing branch returns identity without training. On any failure, return `{'op':'MAINTENANCE_DENIED'}`: the old worker raises its ordinary ValueError *inside* its existing try/except and sends an error result while retaining RAM. It must not fall back to prefix construction when bundle lookup fails.

The helper module must have only standard-library imports at module import time and no import-time action. Syntax/import checks precede any trusted envelope transmission. A stale/malformed authorization cannot select a fallback mutation.

## First action: exact numerical fixture only

Read `bundle.views['P_read']`, `bundle.base_logits['P_read']`, and the real frozen pilot's standardized hidden arrays. Select exactly `domain==0` and `group[:,task0]==0` in the frozen role order. Build columns in the same order as the approved FULL_SOURCE path: source1's32 features followed by source2's32 features, with no resampling or new normalization. Export:

- b[N] and x[N,64] in the FP64 values seen by the convex solver;
- y[N], row_id[N], domain[N], group IDs and source/column order;
- lambda0.001, coefficient ridge, alpha/beta bounds and unbounded FULL feature mask;
- dataset/seed/role/source/config/native/base identities, array hashes, exact shape, finite/support summary and empirical matrix condition/rank data;
- before/after model hashes and actual loaded-reader code hashes (marshal of the existing relevant function code objects) separately from on-disk source hashes.

Write only the small data NPZ and JSON receipt under `artifacts/mdmt_rcf_20261026/v1/maintenance/`. Do not invoke the solver, select an alternative policy, read Q/test performance, change training labels or parameters, or reload any module. Access the already-authorized P_read rows under a diagnostic/audit ledger entry. Raw fixture values are allowed experimental data, not a checkpoint. The designer receives this real fixture before choosing a numerical remedy.

## Later action: narrowly reviewed reader reload

Only after the separate numerical design/implementation/review, validate the new readers source hash and approved fixture identity. Reload only `experiments.mdmt_rcf_20261026.readers`; never reload protocol, prefix, state, baselines or native/vendor modules. Existing method branches re-import `fit_reader` at their next command, so no worker-loop edit is needed. Preserve the current module object dictionary before reload and restore it on an import/reload error, without touching trained tensors or optimizer state. Record actual before/after loaded function-code hashes and source hashes and unchanged model hashes.

The primary orchestrator updates the dedicated normal training authorization only after successful reader installation in the listed workers. If a controller identity refresh is required, it is a separately validated metadata mutation using the approved current source manifest: retain old and new identities and explicitly state which core modules remained loaded from the original version. Do not relabel prior native/base training as though it used the later collector/solver source. This plan does not implement a delivery hook or a general maintenance framework.

## Acceptance and rollback

Independent review must confirm: only two whitelisted actions; all token checks precede operations; authentic local endpoint only; unique idle controller/worker/bundle proof; catch-all worker deserialization boundary; ordinary status/prefix or denied-operation continuation; no arbitrary imports/commands; immutable fixture row ordering; exact FULL64 column mask; before/after live-model hashes unchanged; no saved model/state; source/config/loaded-code identities reported honestly. A small disposable process test verifies envelope success/refusal and survival, without contacting the live service.

Any refusal leaves the service/workers alive and old reader module usable. Fixture action has no rollback mutation. A failed reader reload restores the old module dictionary and records the failure; no scientific run is marked completed or negative. Other users' processes, including PID235316, are outside scope and remain untouched.
