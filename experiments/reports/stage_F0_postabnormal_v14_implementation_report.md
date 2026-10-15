# F0 execution control v1.4 coder implementation report

Date: 2026-10-13. Scope: execution-control safeguards only. No research
algorithm, model, training configuration, scientific result, or training
process was changed or run. The approved specification is
`experiments/reports/stage_F0_postabnormal_design_v14.md`, SHA-256
`c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9`. The
frozen source candidates matched before implementation; the design and
independent v2 review files still match their stated hashes.

## Implemented controls

The current path keeps historical v1.3 H49 verification separate from live
v1.4 H51 authorization. H51 validation accepts iterable caller inputs while
normalizing only after duplicate and exact-membership checks; the runner's
actual `set(H51)` argument is covered. Production-root detection resolves
canonical paths, so symlink aliases continue through frozen-queue, recovery,
F140, and live-GPU checks. Missing hold, out-of-F140 claim, unverified GPU,
and any missing/extra incident source evidence fail closed.

The source snapshot now binds the frozen design SHA, exact source file set,
incident audit/receipt and both nonces' stdout, stderr, and phase logs, plus
the historical v1.3 graph projection. Current independent implementation
review is structured and must bind the design, exact snapshot, coder report,
complete control-file hash inventory, historical projection, and current
implementation bytes. Gate and migration identity must point to that same
review; the prior v1.3 review cannot substitute.

Pending migration/release validation requires the untouched frozen baseline
and rejects a claimed F140 member. Post-release validation allows only lawful
progression within frozen F140. Migration baseline checks, idempotent replay,
ledger handling, writes, and final verification stay under the existing
execution/worker/GPU/canonical locks. Partial JSONL tails are repaired only
under the transaction lock; immutable matching evidence and partial hold
reentry are re-fsynced before durable stage records. Immutable artifacts are
published atomically, and mutable clone pointers are atomically replaced so a
same-filesystem hardlink cannot mutate the source inode. Identity-bound
principal release, hold invalidation, and fresh live GPU checks remain
required at release and claim. A committed release replay with an active hold
now validates the full pending graph immediately before the hold transition,
including when the persisted gate is already launch-authorized. It verifies
the release artifact and pointer, ledger hash, canonical gate, and frozen
source pointers before changing the hold. The pending validation view lowers
only its in-memory authorization bit; persisted release bindings remain
unchanged. A valid already-inactive duplicate still permits only the
post-release F140 progression allowed by the release graph.

## Full-validator CPU fixtures

The expanded clone fixture uses the real source snapshot, source validator,
administrative migration, current gate/release validators, and public claim
path without monkeypatching acceptance validators. It exercises postmigration
GO, current review identity, identity-bound release, idempotent reentry, and a
successful in-F140 live claim. After injecting failure at gate authorization,
it confirms reentry rejects F140 status, run-directory, canonical-lock,
reviewer-byte, incident-stderr, and active-hold-event drift while leaving the
hold active. The unchanged graph then reconciles, and a duplicate after
legitimate live F140 progress remains idempotent. Other negative/failure cases cover missing hold and
incident stderr, forged design/history metadata, old-review substitution,
pending F140 already CLAIMED, altered hold history, out-of-F140 rows, actual
runner set semantics, symlink aliases, unverified GPU telemetry, new hold
between release/claim, interruption after release artifact and gate
authorization, partial JSONL recovery, immutable file re-fsync, and hold
re-entry fsync ordering. The same-filesystem clone test confirms mutable gate
replacement breaks hardlinks. These are CPU fixtures only.

## Verification

Commands ran sequentially through
`experiments/stage_F0_outputs/runs/run_f0_python_task.sh` with the
`phase0-3-env` interpreter. No more than one Python test process ran at a
time.

| Check | Result |
|---|---:|
| Full v1.4 post-abnormal controls and actual-validator fixtures | 22 passed in 306.76s |
| Worker-control, queue, runner, launcher regressions | 78 passed in 25.38s |
| Historical v1.3 migration graph | 30 passed in 87.25s |
| `py_compile` for changed runtime modules and focused tests | passed |

Total: 130 tests passed. The regression suite emitted one existing PyTorch
TF32 API deprecation warning. The complete v1.4 suite includes the
real-validator end-to-end migration/release/claim fixture; it passed without
source-validator monkeypatching. The standalone actual-validator clone also
passed after the code change (1 passed in 192.48s). No scientific result was
inspected or interpreted.

## Production state

Production administrative migration: **NOT EXECUTED; implementation and
temporary CPU fixtures only.** Production release: **NOT EXECUTED.** Training:
**NOT LAUNCHED.** The active production hold remains byte-identical at SHA-256
`7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`.
Production v1.4 migration/release ledgers, release artifact, and execution
snapshot remain absent. This report does not authorize migration, release, or
training; independent review and the specified postmigration GO/release gates
remain required.

## Touched-file SHA-256 inventory

| File | SHA-256 |
|---|---|
| `experiments/stage_F0_postabnormal_v14.py` | `10d81444afa55c13c8aab09605da0ea9840354e0be1eff9d2b4cc7738bd3ae16` |
| `experiments/stage_F0_worker_control.py` | `70860b1fe626d744d9adb8112c640f0db35a64f8ececf382394a07f171c49a2a` |
| `experiments/stage_F0_queue.py` | `58ceca9d193f62993c74e53e6964ca20441dcb9117f346988fba202cecbf4d31` |
| `experiments/stage_F0_run.py` | `1d72a7772d9092b2857956f8d833c50dd99a113104b344a86107a10b76e2ec31` |
| `experiments/stage_F0_migration.py` | `9e4c903277505a0e513f3e5f673ce3e263371722bf807a4a0cd7c4b2e4d103a0` |
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `cbcbff227e12b72175c3d7b586a6774a2e467a00b947fb6eb3b0299587e94d9b` |
| `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` | `a914327267cb650f8011e56f0164268fcee131aa1c361b5a145e2c53431d63e3` |
| `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md` | `2bdced6ef4af2b3c9d711cb2c2a38df53d692d663d9141e76f6b2b1c07345045` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `67fbeb1198bbb57455d9025a7fbb185399e911bbbd57d832c08a07f7e675e038` |
| `experiments/tests/test_stage_F0_worker_control.py` | `5268ba5de2588c6bf264d282ea17efad4645c7538006e96bdbf736499dd267c6` |
| `experiments/tests/test_stage_F0_queue.py` | `c297862e1e8a35d07668b675b8eaf9ddde1baecc0b0b8a1b46088d292e288b36` |
| `experiments/tests/test_stage_F0_run.py` | `4da8ba9cbb4803b48a90526728d9e9b4dec9601da133a95412d7975c39282b8e` |
| `experiments/tests/test_stage_F0_migration.py` | `4aadbc98ee01680e228a05d1c550c4b28c3f4c5748ce717fcdddd342114149d4` |
| `experiments/tests/test_stage_F0_launcher.py` | `42ce45868eb21e026e1da998da58af99c2b71ef1dd4e72ce1239d1e3fd4fad1f` |

The SHA-256 of this report is recorded separately in the coder handoff because
the report cannot contain its own hash.
