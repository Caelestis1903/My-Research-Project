# F0 v1.5 Successor Control — Implementation Report v2

Status: **IMPLEMENT evidence updated; awaiting fresh independent REVIEW.** The earlier independent implementation verdict is FAIL and remains in force until re-review. This report does not authorize a production migration, release, worker claim, or training run.

## Bound specification and immutable inputs

- Design v2 SHA-256: `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`.
- Independent design review PASS SHA-256: `9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb`.
- Bootstrap-remediation implementation v2 SHA-256: `491bf69c0bdbc8708e2e9bbd050d3213270e643e460e66369d9a165c464140ea`.
- Independent bootstrap-remediation review PASS WITH FIXES SHA-256: `1a2196eda5a182901bfa646ab2964c8461af4aa0fe587874d70132a1aa3c0065`.
- Immutable amendment record SHA-256: `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138`.
- Immutable no-go candidate SHA-256: `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`.

## Implementation and qualification updates

The v1.5 control graph and principal transaction are exercised on isolated clone roots. The current transaction implementation has one effective `commit_principal_release` definition. The runner/worker path distinguishes typed resource deferrals from ordinary runner failures, validates the ordered child phase and receipt identity for both `CHILD_PRECLAIM` and `SERIALIZED_PRECLAIM_MEMORY_CHECK`, and validates frozen F140 key identity from canonical JSON and full SHA-256.

The qualification fixture now detaches every regular worker log in a hardlink clone before child dispatch. The COW check covered 259 source/clone log files: 259 clone inodes were detached, zero remained shared, and zero source hashes changed. The real two-branch supervisor → child → runner E2E then passed. It verified both deferral boundaries and durable authenticated receipts, no canonical claim/attempt/checkpoint mutation, stable F140 state, inactive hold only inside the isolated clone, and unchanged production worker-log bytes. The E2E also contains negative graph checks for altered source snapshot, prelaunch audit, approved candidate core, and staged-GO digests.

The real child invokes the runner through the worker wrapper with telemetry supplied only in the isolated test wrapper; no production API accepts a telemetry provider. Both branches stop at resource deferral before claim mutation. No training function was invoked.

The v1.4 historical fixture follows design §58: it validates frozen historical raw bytes and identity, then verifies that the v1.4 current-release validator rejects the graph against the newer v1.5 source inventory. The test checks the isolated clone's protected pointers and evidence remain unchanged; historical review/ledger bytes are not rewritten to manufacture acceptance.

## Verification

All Python commands used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` through `experiments/stage_F0_outputs/runs/run_f0_python_task.sh`. Test groups were run sequentially, one pytest parent at a time. Real E2E children were sequential and used the shared scheduler root `/tmp/phase0-3-F0-scheduler-a095f9a9ed5f9da0`; no competing test group ran concurrently.

The standalone real two-branch E2E command was:

```text
bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-full-e2e /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -s experiments/tests/test_stage_F0_successor_v15.py::test_real_pending_graph_release_reentry_terminal_and_f140_claim
```

Raw result: `1 passed in 590.28s (0:09:50)`. The production worker log SHA-256 was `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` immediately before and after this run.

The isolated COW-only test had previously passed with `1 passed in 16.22s`; its direct check reported `detached 259 shared_inodes 0 production_hash_changes 0`.

Focused control tests, excluding the two real child E2E tests already run separately, passed with `34 passed, 2 deselected in 23.49s`. The observation-ledger fixture now supplies the telemetry target identity required by the real validator. The supervisor receipt fixture uses an isolated repository layout with the frozen source snapshot and queue; production validation was not patched. The three targeted fixture tests passed with `6 passed in 0.13s`.

The final sequential regression groups were:

1. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-regression-group1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_successor_v15.py experiments/tests/test_stage_F0_run.py experiments/tests/test_stage_F0_postabnormal_v14.py`

   Raw result: `84 passed, 1 warning in 833.22s (0:13:53)`. The warning is PyTorch's TF32 deprecated-API notice in `test_run_preserves_gpu_uuid_mapping_in_both_lock_owners`.

2. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-regression-group2 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_worker_control.py`

   Raw result: `40 passed in 8.02s`.

3. `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-regression-group3 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_launcher.py experiments/tests/test_stage_F0_migration.py`

   Raw result: `44 passed in 94.91s (0:01:34)`.

The final read-only process check found no pytest, F0 runner, supervisor, or child process. No production lock was acquired or altered by the post-run inventory check.

For transparency, the first post-COW-focused attempt returned `2 failed, 32 passed, 2 deselected in 25.16s`: the observation fixture omitted `telemetry.index`/`uuid`, and the synthetic supervisor fixture derived `/tmp` as its repository root. A first repair rerun also failed at fixture directory creation (`5 failed, 1 passed in 0.15s`). Those fixture-only problems were corrected, then the targeted cases passed 6/6 and the final 34-case focused run passed. No production validator was weakened.

## Protected state and incident disclosure

The pre-edit full queue/status/run/checkpoint inventory and the original runner byte diff were not retained. They cannot be reconstructed from the current workspace, so no historical pre/post equality is claimed. The following values are the current read-only observations only.

`_assert_frozen_artifacts(ROOT)` passed. Exact protected hashes include:

| Artifact | Current SHA-256 |
|---|---|
| Canonical v1.4 execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| v1.4 migration ledger | `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30` |
| v1.4 source snapshot | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |
| Frozen search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Immutable amendment record | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` |
| Immutable no-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` |
| Production `audited_worker_0.jsonl` (current) | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` |

The current hold JSON reports `active=true`. The v1.5 production release ledger and v1.5 runtime-observation ledger remain absent. Current queue/status/run/checkpoint observations are: 288 queue entries; 140 frozen F140 identities; 51 H51 identities, all 51 `RECOVERY_BLOCKED`; 155 queue-status paths; 115 canonical-lock paths; 109 existing queue-key run roots; and 535 checkpoint files. The current path/type/content inventory digests, computed from sorted repository-relative rows, are: queue statuses `5a3cea6c969ebce10cd076d59051e48482bd41210aa327af9ff0bc539d6c8cd7`; canonical locks `ea7d2378048a6c67a368f61cd86ae91b27196e8b87af1369bfa794f53f544f98`; existing run-root paths `314190c995eb9b887d8a5a0b7810c11ee2250ed19e88eb47f5fb5b8829ec73fb`; checkpoint files `65233b07cfc93ccee690e26cfcb680c8aa2f741228086f2c19c0e4aba38a9079`. No F140 ID currently has a status, canonical lock, or run root.

**Append-only incident:** an earlier hardlink-clone qualification run appended nine `CHILD_STARTED` records to production `experiments/stage_F0_outputs/runs/worker_logs/audited_worker_0.jsonl`, at current lines 130–138. Those records are preserved; they were not deleted or rewritten. The pre-incident file hash and exact pre-edit inventory were not retained, so the original bytes cannot be restored or inferred. COW was added and verified before the successful standalone E2E and final regression group; the production log hash remained exactly `b27ac612…e3eb7c2` across those runs.

No production migration, release, gate update, queue/status/run/checkpoint write, worker launch, or training run was performed. The production hold remains active, the v1.4 canonical gate and release graph remain current, and the successor release remains inactive/unpublished. The earlier independent implementation review is still FAIL pending a new review of the updated source, tests, and this report.

## Current implementation/control hashes

| File | SHA-256 |
|---|---|
| `experiments/stage_F0_successor_v15.py` | `64151db7950e198c69e06da81f10a33ce6944a9464cfb0b22cd9a4b77edc119c` |
| `experiments/stage_F0_worker_control.py` | `69e332244fbac4fb7d957cc6a5b398733ee88e679015ec42263b656e88961717` |
| `experiments/stage_F0_queue.py` | `da4a1d0b279bc3881753de4d4e5f9f2ab8f15ec1e3222e0faad4f295d577b234` |
| `experiments/stage_F0_run.py` | `4c94a52d54212bd872efa5499b1a7d55eacaecb4e5d2957c88f4d95ca2837377` |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/stage_F0_migration.py` | `9e4c903277505a0e513f3e5f673ce3e263371722bf807a4a0cd7c4b2e4d103a0` |
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `e44114d1d9f6a12635e4c0d0d42ed55d32b272bf775c6e42d966af839c40ac90` |
| `experiments/stage_F0_outputs/runs/run_search_trajectory_child.py` | `0808eee556cd73f056caaea719ad6d61d5c735dce43b1fc12933f82c060e1a6e` |
| `experiments/stage_F0_outputs/runs/run_search_worker.sh` | `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4` |
| `experiments/stage_F0_outputs/runs/run_f0_python_task.sh` | `d8a317ee314c4eae0b2b10d5b8e2e9cfce345ae5164b6104a0452c02dfe34e17` |
| `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` | `a914327267cb650f8011e56f0164268fcee131aa1c361b5a145e2c53431d63e3` |
| `experiments/tests/test_stage_F0_successor_v15.py` | `83d17018984f7605df03e148e2a6d5c3ed757d4799b316848ddc57b39faee07f` |
| `experiments/tests/test_stage_F0_worker_control.py` | `2cee640070dffe9deb209d46939ad1db7e3035dc7f9f5e17b3cc86b0c9505e50` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `experiments/tests/test_stage_F0_launcher.py` | `b2761475f4b6c98476964cab8fea6918a188c2bd255234a7cd2a8e86cf54a824` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `079cbae1940a1c37671e323f85fec3c25babc5b80ccabcb470fb1a2e97c3b5e6` |
| `experiments/tests/test_stage_F0_migration.py` | `4aadbc98ee01680e228a05d1c550c4b28c3f4c5748ce717fcdddd342114149d4` |
| `experiments/tests/test_stage_F0_queue.py` | `c297862e1e8a35d07668b675b8eaf9ddde1baecc0b0b8a1b46088d292e288b36` |
| `experiments/tests/conftest.py` | `bc1ab7562c07a4f68711db525138deb313fa9e6da93caca6d7cd616a86631bb7` |
