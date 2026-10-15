# F0 v1.5 Successor Control Implementation Report

Status: **IMPLEMENT complete; awaiting independent REVIEW.** No v1.5 production migration, release, gate switch, claim, worker launch, or training run was performed. The production hold remains active and the canonical gate remains the frozen v1.4 artifact.

## Bound specification and immutable inputs

- Design: `stage_F0_v15_successor_control_design_v2.md`, SHA-256 `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`.
- Independent design review: `stage_F0_v15_successor_control_design_review_v2.md`, SHA-256 `9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb`.
- Bootstrap remediation implementation: `stage_F0_worker_bootstrap_hash_remediation_implementation_v2.md`, SHA-256 `491bf69c0bdbc8708e2e9bbd050d3213270e643e460e66369d9a165c464140ea`.
- Independent remediation review: `stage_F0_worker_bootstrap_hash_remediation_review_v2.md`, SHA-256 `1a2196eda5a182901bfa646ab2964c8461af4aa0fe587874d70132a1aa3c0065`.
- Immutable amendment record: SHA-256 `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138`.
- Immutable no-go candidate: SHA-256 `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`.

## Implementation

The successor protocol is implemented in the v1.5 control module and connected to queue admission, runner dispatch, supervisor, and worker entry paths. It binds source and prelaunch graphs to the immutable v1.4 evidence, the current incident hold, and the reviewed target identities. It enforces fresh per-target GPU telemetry and the strict per-device memory threshold, the five-task Python process ceiling, and no-respawn behavior. Typed resource deferrals remain distinct from runner failures and preserve valid peer work.

The durable release path uses ordered identity-bound PREPARED/COMMITTED evidence, failure-injection boundaries, exact partial-reentry validation, and terminal acknowledgement without telemetry or append-only log changes. Queue admission validates canonical JSON, full SHA-256 identity, the specified run-ID prefix, exact frozen F140 membership, the active hold, and symlink-resolved roots. The runner and v1.5 validator are covered by the same approved amendment predicate.

The old v1.4 positive migration/release/claim integration fixture was changed to match design §58: it validates immutable historical raw artifacts and transaction identity, then verifies that the v1.4 current-release validator rejects the historical graph because its frozen control inventory predates v1.5 code. The rejection is exercised on an isolated clone and the clone's canonical gate, queue, hold, release pointer, and ledgers are byte-identical before and after. No historical review, ledger, or fixture was rewritten to force acceptance.

## Verification

All final suites used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` via `run_f0_python_task.sh`. Final groups ran sequentially, each with its own scheduler root and one Python process:

1. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-final-group1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_successor_v15.py experiments/tests/test_stage_F0_run.py experiments/tests/test_stage_F0_postabnormal_v14.py`

   Raw result: `78 passed, 1 warning in 227.46s (0:03:47)`. The warning is PyTorch's TF32 deprecated-API notice.

2. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-final-group2 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_worker_control.py`

   Raw result: `40 passed in 8.82s`.

3. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-final-group3 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_launcher.py experiments/tests/test_stage_F0_migration.py`

   Raw result: `44 passed in 95.11s (0:01:35)`.

The full real-validator release → terminal replay → in-F140 claim fixture passed separately with `1 passed in 198.23s`. It also covers altered identities, all durable fault boundaries, fresh telemetry at re-entry, unchanged COMMITTED digest and observation-log bytes on terminal acknowledgement, out-of-F140 rejection, and missing-hold rejection.

The historical v1.4 fixture passed separately after its §58 update: `1 passed in 2.77s`.

For transparency, an earlier parallel broad regression run had hardlink-clone source-inventory contention. Its group results were 76 passed/2 failed in group 1, 37 passed/3 failed in group 2, and 43 passed/1 failed in group 3. The group-1 failures were a clone inventory race caused by concurrent source/test edits and the obsolete v1.4 positive-authority expectation; the group-2 failures were three isolated GPU-supervisor fixtures that set `RUNS` but omitted their temporary execution-gate path; the group-3 failure expected the old `v1.4` wording rather than the fail-closed unknown-schema error. The latter fixture issues were corrected only in tests. The old v1.4 test was replaced with the §58 historical-rejection test above. The final sequential suites passed. An earlier E2E attempt was interrupted after a lock-path deadlock; lock ownership is now explicitly propagated and the complete standalone E2E passed. None of these earlier failures were discarded from this report.

The final suites used at most one Python process at a time. No active pytest, runner, or F0 worker process was present during the final read-only process check. The execution-state lock is an existing zero-byte ordinary file; no F140 lock files, status files, or run roots exist. All 51 H51 statuses remain `RECOVERY_BLOCKED`.

## Protected state and inventory boundary

Post-implementation SHA-256 checks matched the frozen values:

| Artifact | SHA-256 |
|---|---|
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Historical v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| Historical v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Historical v1.4 migration ledger | `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30` |
| Historical v1.4 source snapshot | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |
| Frozen search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Amendment record | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` |
| No-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` |

The full protected queue/status/run/checkpoint inventory was not retained as a pre-edit snapshot, so no historical pre/post inventory equality is claimed. The post-implementation read-only inventory was: queue 288 entries; frozen F140 140 keys; H51 51 keys; 155 queue-status entries (inventory SHA-256 `15b5f462286b47d804ee5ff4905885f8d93ce7d16a7d650e22a6ab7c068c3715`); 115 canonical-lock entries (inventory SHA-256 `b8ca3d67096219b2ab16e6ab855330e7c81d8f3c040a98b9e00fcb9563fcadcb`); 109 existing queue-key run roots (inventory SHA-256 `7a38f2134898e74124861a6e9e97c9cb7add644771227631022411304d959b05`); and 535 checkpoint files (inventory SHA-256 `d07d1edaf6f32435aec259c6bd41b585e712da0245591f9b72137c66271af1d9`). No F140 key had a status file, canonical lock, or run root. All 51 H51 statuses were blocked. These current inventories are reported separately from the absent pre-edit inventory and must not be interpreted as an inferred baseline.

## Modified implementation and test file hashes

| File | SHA-256 |
|---|---|
| `experiments/stage_F0_successor_v15.py` | `6617d8fe0f7986b99b0b2c6eee615b8618d8b7413256ceaabab1ec45ebbe5712` |
| `experiments/stage_F0_worker_control.py` | `69e332244fbac4fb7d957cc6a5b398733ee88e679015ec42263b656e88961717` |
| `experiments/stage_F0_queue.py` | `da4a1d0b279bc3881753de4d4e5f9f2ab8f15ec1e3222e0faad4f295d577b234` |
| `experiments/stage_F0_run.py` | `3fbb2ef5d5aa05342d034a8cb2774c47fc62f38bc5571eea060469499a24a66c` |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `d4fe34f65942eddd0f9084d34a8d8da648127c1ef8908c8dbfa7f627f2bdad26` |
| `experiments/stage_F0_outputs/runs/run_search_trajectory_child.py` | `0808eee556cd73f056caaea719ad6d61d5c735dce43b1fc12933f82c060e1a6e` |
| `experiments/tests/test_stage_F0_successor_v15.py` | `efcbe43aa8eaee828c90ade939588faee032abf6aceb0bbccf7e8fcd7898f8a9` |
| `experiments/tests/test_stage_F0_worker_control.py` | `2cee640070dffe9deb209d46939ad1db7e3035dc7f9f5e17b3cc86b0c9505e50` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `experiments/tests/test_stage_F0_launcher.py` | `b2761475f4b6c98476964cab8fea6918a188c2bd255234a7cd2a8e86cf54a824` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `079cbae1940a1c37671e323f85fec3c25babc5b80ccabcb470fb1a2e97c3b5e6` |

Six `cpython-314` cache files created by an earlier wrong-interpreter command remain untouched as previously directed. Final test and verification commands used the project Python 3.13.5 environment.
