# F0 execution isolation v1.3 implementation report

This implementation follows `experiments/reports/stage_F0_execution_isolation_design.md`
v1.3 §14 (SHA-256 `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7`).
The independently frozen H49 is v1.2 H47 plus `7fd2006bdc8f1817a6f7` and
`1f04c7fb623d18f940ac`; its SHA-256 is
`60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e`.

The implementation adds the administrative migration and provenance
validation in `stage_F0_migration.py`, requires exact H49 at search-claim,
execution-gate, caller, hold-release, migration-audit, status, and prelaunch
boundaries, and refreshes the pending execution gate to the current
control/runtime hashes. The migration code records a durable PREPARED event
with original raw hold/status bytes and complete protected-asset inventories
before any status write; serializes with execution, worker, GPU, and target
canonical locks; updates only the two target queue statuses and active hold;
then validates the complete post-state and appends COMMITTED. Runtime
validation pins the original 52 completed keys and their 573-file hash, keeps
all 49 H49 entries RECOVERY_BLOCKED, and permits fresh queue states to advance
through the protocol after a separately reviewed release. A release binds the
complete provenance graph, fresh-only scope, and the same current reviewer
path/hash frozen by both the execution gate and its v1.3 protection record.
The incident inventory also binds the peer's durable failure audit through the
original hold event's canonical key, worker, nonce, path, and byte hash. The
migration does not load or rewrite a checkpoint, resume or retry a trajectory,
create a new attempt, or authorize formal/test work. Failure injection covers
PREPARED, both status writes, and hold write/fsync boundaries; retry of the same
PREPARED transaction is idempotent and fails closed on source drift.

`apply_migration()` checks a caller-frozen provenance-correction SHA before
starting its administrative transaction. The matching-hash fixture commits
the temporary migration and records that digest in COMMITTED. The mismatched-
hash fixture fails before creating a migration ledger; both target statuses
and the active-hold bytes remain unchanged.

The implementation report and gate are ready for independent review. Per the
orchestrator instruction, production migration is intentionally not run before
that review passes and a later Runner authorization. No production queue
status, active hold, receipt, manifest, metric, checkpoint, or completed-run
asset was written by this implementation turn.

## Verification

All project Python commands used the reserved-slot wrapper and the
`phase0-3-env` interpreter. The final combined focused suite ran in one pytest
Python process; process fixtures reserved their own child/supervisor slots and
stayed within the five-process ceiling. Command:

```text
experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v13-migration-final-focused /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_migration.py experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_worker_control.py experiments/tests/test_stage_F0_run.py experiments/tests/test_stage_F0_launcher.py
```

Result: **103 passed**, with one existing PyTorch TF32 deprecation warning.
The suite includes temporary-directory migration/failure-injection fixtures,
queue and gate H49 rejection boundaries, reviewed-release and dynamic fresh
claim transitions, complete status projection and provenance-edge checks,
source/incident inventory drift refusal (including the hold-event-bound peer
failure audit and old-review substitution), caller-frozen correction SHA
match/mismatch behavior, PID/signal/receipt transaction fixtures, and runner
lock/receipt tests. The migration fixtures independently
copy mutable incident/control inputs and assert different inodes; the 52
completed-run assets are read-only hash-checked as 573 source files instead
of copied (4.72 GB). Fixtures therefore do not exercise writes to those
completed assets; the production migration/runtime validator performs the
full content hash scan. CPU runner fixtures include a synthetic
optimizer/checkpoint case. No research or CUDA training was run, and no
experimental results were analyzed.

A separate reserved-wrapper, read-only production preflight verified:

| Check | Observed state |
|---|---|
| Search statuses before migration | 52 `VALIDATION_COMPLETE`, 47 `RECOVERY_BLOCKED`, 187 `UNCLAIMED`, 1 `RUNNING`, 1 `INTERRUPTED` |
| H49 additions | `7fd2006bdc8f1817a6f7` remains `RUNNING`; `1f04c7fb623d18f940ac` remains `INTERRUPTED` |
| Completed-run inventory | 573 files; aggregate SHA-256 `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |
| Active global hold | `active: true`; SHA-256 `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |
| Target peer receipt | SHA-256 `3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7` |
| Target SIGSEGV status | Original `RUNNING` record SHA-256 `74cbabc600e19e4e977c3b38d3068419c72c3be0b678cea5857a1867ec66dbbe` |
| Target peer status | Original `INTERRUPTED` record SHA-256 `61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478` |
| Incident attempt/log/hold-failure inventory | 27 files; SHA-256 `10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e` |
| Execution locks | Free at preflight |
| Administrative artifacts | Provenance correction and v1.3 migration audit do not exist yet |

The prelaunch validator rejects the production gate with
`v1.3 H49 protection evidence failed validation` because the correction and
committed migration evidence are intentionally absent. This confirms the gate
cannot authorize a worker before the administrative migration and review.

## Frozen identities and pending gate

| Inventory | Count | SHA-256 |
|---|---:|---|
| Execution-control source/test inventory | 16 files | `e98c6ef9d23974420bebcb053f3a791b5efddc09b055fb3bdc6867b2fd572fa6` |
| Runtime implementation inventory | 28 files | `edf85d7bee7e3b7396b191a9cc896007a35c454734c30f7b35920ab6c54dc32f` |
| Frozen H49 | 49 run IDs | `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| Completed-run asset inventory baseline | 573 files | `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |
| Prior pending gate, superseded | — | `a416d0947e09b5f566251348081ef9550e0edfa1426d4ef9fb022d9d79761e95` |
| Current pending execution gate | — | Its SHA is stored on disk; it is not duplicated here because the gate binds this report's SHA |
| Active global hold, unchanged | — | `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |

The pending gate freezes all 49 IDs, includes the current 16-file control
inventory and current 28-file runtime hash, and records the pre-migration
completed and incident asset baselines. It has `reviewer_verdict: PENDING`,
`launch_authorized: false`, `formal_or_test_authorized: false`,
`recovery_authorized: false`, and `automatic_retry_authorized: false`.
Its correction and migration hashes are empty until a reviewed Runner performs
the administrative migration. The active hold remains on and unchanged.

The 16 execution-control inventory paths are:

```text
experiments/stage_F0_outputs/runs/run_search_worker.sh
experiments/stage_F0_outputs/runs/run_f0_python_task.sh
experiments/stage_F0_outputs/runs/run_search_claim_worker.py
experiments/stage_F0_outputs/runs/run_search_trajectory_child.py
experiments/stage_F0_worker_control.py
experiments/stage_F0_queue.py
experiments/stage_F0_run.py
experiments/stage_F0_migration.py
experiments/stage_F0_outputs/runs/WORKER_CONTROL.md
experiments/reports/stage_F0_execution_isolation_design.md
experiments/tests/test_stage_F0_launcher.py
experiments/tests/conftest.py
experiments/tests/test_stage_F0_worker_control.py
experiments/tests/test_stage_F0_queue.py
experiments/tests/test_stage_F0_run.py
experiments/tests/test_stage_F0_migration.py
```

## Changed-file hashes

| File | SHA-256 |
|---|---|
| `experiments/stage_F0_worker_control.py` | `06ca5287c3624633e0cac94482e9ddcff9c0882731eab7c990e48ae3e9c57af9` |
| `experiments/stage_F0_migration.py` | `02ee3b12203ddfa9184e45a2d8109a13ac48141dd750c91c029a305ad99f1d3d` |
| `experiments/stage_F0_queue.py` | `c97e3c2cf87c906f9d5ac8841e45751702ae0b0ebcfa0bc25f80a5ca4d5a640b` |
| `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` | `6cee17008a7e57e61f681439282616ffe7482cda7eb7c12e66b016bcc37cb4f4` |
| `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md` | `3f520c89b3e78c52b9e52ebea931388ad849934d113cecb3db945956506c0fca` |
| `experiments/tests/test_stage_F0_migration.py` | `54e65a8dbbe80a8193e054b00bcf65f1a37ba661252df489139dc6ee615ca51d` |
| `experiments/tests/test_stage_F0_queue.py` | `15ca840f059cfdf4cddbd4cb29cd0b41814b07c4b2c0722b65fd2b22ee1dabb2` |
| `experiments/tests/test_stage_F0_worker_control.py` | `6f5cf977b447cf88c1c1364cb9aca96d6cc4f578c3a2e221c419394c9ee0ff53` |
| `experiments/tests/test_stage_F0_run.py` | `73b74c389a947028e04516cb6bfb7933768d43dd705f3f7654fecfb780f2927c` |
| `experiments/tests/test_stage_F0_launcher.py` | `e029317bae888e313acf25a267618d3d7074ba3563a9ea09ae8f869440855f3e` |

The design and review sources are not modified. Historical artifact invariance
is not claimed beyond the individually hashed sources, statuses, and complete
asset inventory available for this preflight; no prior full snapshot exists.
