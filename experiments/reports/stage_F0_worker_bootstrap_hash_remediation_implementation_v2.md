# IMPLEMENTATION REPORT: F0 worker bootstrap amendment remediation v2

Implementation status: approved-scope follow-up complete. Production migration/release: **not executed**. Active hold: **preserved**. Training or worker launch: **not performed**.

## Follow-up changes

The isolated positive v1.4 fixture now rebuilds a valid historical graph only inside a byte-isolated temporary clone using the archived frozen source snapshot. It executes the real administrative migration, constructs the reviewed postmigration/prelaunch graph, and proves `validate_current_gate` and the runner's `validate_runtime_control_amendment` both accept the approved amendment for both accepted verdicts. Dataset-load and claim sentinels remain untouched. The current reheld-graph negative fixture clones production state and verifies both replay of the exact old release identity and a different request nonce are rejected without changing hold, gate, release artifact, release pointer, ledger bytes, or ledger stages.

The fixture uses `shutil.copy2` for all source material and mutates only its temporary clone. No production runtime/control source changed in this follow-up.

## Test verification

Command:

```text
experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v14-hash-remediation-suite-v2 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py experiments/tests/test_stage_F0_run.py
```

Exact result:

```text
48 passed, 1 warning in 250.88s (0:04:10)
```

The warning is PyTorch's deprecation notice for legacy TF32 settings in `test_run_preserves_gpu_uuid_mapping_in_both_lock_owners`. The positive isolated graph, current-rehold release replay/different-nonce rejection, amendment matrix, v1.4 tests, and runner tests all passed. No training or scientific-result analysis was performed.

## Modified file identities

| File | SHA-256 |
| --- | --- |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `1827aab62cfe71632bcef9ad0e299ddb74482970919367d9958d2da7f8e8bfbf` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `experiments/reports/stage_F0_worker_bootstrap_hash_remediation_implementation_v1.md` | `0dd9e8ac65eb86ab11a4a310a6e82a507e1076f87a414bf4a609a6280317fada` |

The implementation report v1 remains unchanged. This v2 report is outside the frozen control inventory. Runtime digest remains `7d6bbd9174d3d26c4dce958173444e9ba9778313166aefbc3a32c30f6e9aceb3`; the separately computed 21-file control inventory digest is now `340266d56dc55fd44cdf01a3fb83be6f22e202313e3f8b645b6715eb99d4570a`.

## Complete runtime inventory

Runtime digest is `stage_F0_queue.code_hash(require_tests=True)`, computed over the 28 files below. The resulting digest is `7d6bbd9174d3d26c4dce958173444e9ba9778313166aefbc3a32c30f6e9aceb3`. It is a distinct digest domain from the 21-file control inventory.

| Runtime path | SHA-256 |
| --- | --- |
| `stage_F0_protocol.py` | `a72fb69d750464d396f3125d425a21573b67d8ce80bfe0d26b76956587d99647` |
| `stage_F0_data.py` | `e39ab4ea8ab1282f2ec939a987d02674a52d91a3423314c3786fca34e2d1df42` |
| `stage_F0_models.py` | `1ac5b538fd906b24805c4eb7cbfa7a7158dd7ff4e1fcebf64add9f30ea62cf81` |
| `stage_F0_objectives.py` | `27329abec8d943912ac71016332be9ebf44d625e8f643b0072317625f133c1b8` |
| `stage_F0_queue.py` | `58ceca9d193f62993c74e53e6964ca20441dcb9117f346988fba202cecbf4d31` |
| `stage_F0_run.py` | `1d72a7772d9092b2857956f8d833c50dd99a113104b344a86107a10b76e2ec31` |
| `tests/test_stage_F0_protocol.py` | `6446757460cbd8f7fda82ee92877510134c21aa4e4d3e16dd229fb6730403bab` |
| `tests/test_stage_F0_data.py` | `bc4d4e5cac80b0a771ea10563d6fd41b8af5ad4015214ba938ca8e34120c42bc` |
| `tests/test_stage_F0_models.py` | `613df1ffff18b3dcc0361e272c58ed7945a98ff47ac43c56e4fc475fc9af49d7` |
| `tests/test_stage_F0_objectives.py` | `438de067f46dc5b591daf709a633981b640bce0c3f170011007e38487a752ed1` |
| `tests/test_stage_F0_queue.py` | `c297862e1e8a35d07668b675b8eaf9ddde1baecc0b0b8a1b46088d292e288b36` |
| `tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `tests/conftest.py` | `bc1ab7562c07a4f68711db525138deb313fa9e6da93caca6d7cd616a86631bb7` |
| `stage_S1_data.py` | `245a408d5e8cb84bfad43a0a0236fc213a212243092731c968aa1cfd1f0abe25` |
| `stage_S1_models.py` | `1f6fa05358b047b048b115681cd62ac2f833ffd1d0892ad68b64b91216caf691` |
| `stage_S1_protocol.py` | `76ed4a851e6cdf15493be5628a08db4647a28eb3a2a11e4e1d7c1974a704218f` |
| `stage_S1_metrics.py` | `15187827f1e57e57e287ffd49f820d2879bb320f1ba650afd686aede9e37fc89` |
| `history_exp/first/p7_models.py` | `7b727c4c933e93860eb9ce1dd57d58f4c40adbf3975d2d7ca6dd76afb8b6e839` |
| `../m3oe/MDMTRec/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `../m3oe/MDMTRec/basic/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `../m3oe/MDMTRec/basic/activation.py` | `152c5909e966055f83b7ec49de7634c853fbae8b029fbd37759371c14c921046` |
| `../m3oe/MDMTRec/basic/features.py` | `116025a45517021bfe86447b8229e59cf04feb665549544ce30173d04421a2b0` |
| `../m3oe/MDMTRec/basic/initializers.py` | `a9a6a2923eebeca7f541ffaddbccf2671a42e7f9b6a1ec8d38ba2872c53f6196` |
| `../m3oe/MDMTRec/basic/layers.py` | `e3eb8db91ff0aaf79c8e389107df5f64841992fab02a18b29c37bb446b22763a` |
| `../m3oe/MDMTRec/utils/__init__.py` | `13c62ae9d561769c4597c2f279bf03ab25363ad64ada1e77af58c4f7979cc7a3` |
| `../m3oe/MDMTRec/utils/data.py` | `71d08bd9487afc702473d8813449d21ffd85c223ed05457791486b6df708501a` |
| `../m3oe/MDMTRec/models/multi_domain/__init__.py` | `e11c58c743dd3125f0a85eeb89c0266706b4716b10700eb99ed1f18d19cb23e8` |
| `../m3oe/MDMTRec/models/multi_domain/model.py` | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |

## Complete control inventory

The independent 21-file control inventory digest is `340266d56dc55fd44cdf01a3fb83be6f22e202313e3f8b645b6715eb99d4570a`.

| Control path | SHA-256 |
| --- | --- |
| `experiments/reports/stage_F0_execution_isolation_design.md` | `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7` |
| `experiments/reports/stage_F0_postabnormal_design_v14.md` | `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9` |
| `experiments/reports/stage_F0_postabnormal_v14_implementation_report.md` | `aa8c709dc9b349340f9613181e28d4e8c72b5301b9ae40cb8262ce0146c3b080` |
| `experiments/stage_F0_migration.py` | `9e4c903277505a0e513f3e5f673ce3e263371722bf807a4a0cd7c4b2e4d103a0` |
| `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` | `a914327267cb650f8011e56f0164268fcee131aa1c361b5a145e2c53431d63e3` |
| `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md` | `2bdced6ef4af2b3c9d711cb2c2a38df53d692d663d9141e76f6b2b1c07345045` |
| `experiments/stage_F0_outputs/runs/run_f0_python_task.sh` | `d8a317ee314c4eae0b2b10d5b8e2e9cfce345ae5164b6104a0452c02dfe34e17` |
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `cbcbff227e12b72175c3d7b586a6774a2e467a00b947fb6eb3b0299587e94d9b` |
| `experiments/stage_F0_outputs/runs/run_search_trajectory_child.py` | `4343e928c266869fb9723509373da3dfe75c30ffe4de85a241a38e9a708f86e5` |
| `experiments/stage_F0_outputs/runs/run_search_worker.sh` | `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4` |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/stage_F0_queue.py` | `58ceca9d193f62993c74e53e6964ca20441dcb9117f346988fba202cecbf4d31` |
| `experiments/stage_F0_run.py` | `1d72a7772d9092b2857956f8d833c50dd99a113104b344a86107a10b76e2ec31` |
| `experiments/stage_F0_worker_control.py` | `70860b1fe626d744d9adb8112c640f0db35a64f8ececf382394a07f171c49a2a` |
| `experiments/tests/conftest.py` | `bc1ab7562c07a4f68711db525138deb313fa9e6da93caca6d7cd616a86631bb7` |
| `experiments/tests/test_stage_F0_launcher.py` | `42ce45868eb21e026e1da998da58af99c2b71ef1dd4e72ce1239d1e3fd4fad1f` |
| `experiments/tests/test_stage_F0_migration.py` | `4aadbc98ee01680e228a05d1c550c4b28c3f4c5748ce717fcdddd342114149d4` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `1827aab62cfe71632bcef9ad0e299ddb74482970919367d9958d2da7f8e8bfbf` |
| `experiments/tests/test_stage_F0_queue.py` | `c297862e1e8a35d07668b675b8eaf9ddde1baecc0b0b8a1b46088d292e288b36` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `experiments/tests/test_stage_F0_worker_control.py` | `5268ba5de2588c6bf264d282ea17efad4645c7538006e96bdbf736499dd267c6` |

## Protected evidence and execution state

The seven frozen candidate identities matched before editing and remained unchanged afterward:

| Protected artifact | SHA-256 |
| --- | --- |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Active `global_hold.json` | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Worker1 failure record | `1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4` |
| Worker1 stderr | `2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a` |
| Frozen search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Immutable v1.4 release | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |

The active hold ID remains `cd2dbf50-3a50-4b81-b976-38f1b03991a3`. Read-only process inspection found no matching live F0 worker/supervisor processes; `fuser` found no holders on the execution-state or worker lock files; the incident record's `canonical_claim` remains null. Test processes ran one at a time through the five-slot task wrapper. Tests used isolated temporary roots or read-only validation, made no canonical claim, and loaded no dataset.

RELEASE NO-GO remains in force. No production migration, release, gate update, hold transition, worker launch, or training occurred.

## Protected production inventories and process state

The exact pre-implementation per-file queue/status/run/checkpoint manifests were not recorded before the initial implementation. They are unavailable and are not reconstructed or inferred here. The table below instead records exact manifests captured immediately before and after the final v2 regression suite. Each manifest is the SHA-256 of the sorted `(relative path, per-file SHA-256)` list; identical before/after digests show the suite left those production inventories unchanged.

| Production inventory | Pre-suite count | Pre-suite manifest SHA-256 | Post-suite count | Post-suite manifest SHA-256 |
| --- | ---: | --- | ---: | --- |
| Queue files under `stage_F0_outputs/queues` | 2 | `1431f78ad0618f91d88e04f8a1c80b19f795d824132b5371b648131eb0bfcd03` | 2 | `1431f78ad0618f91d88e04f8a1c80b19f795d824132b5371b648131eb0bfcd03` |
| Queue-status files under `runs/queue_status` | 155 | `82c06fc57e0eee9740236e9c63a33d6af2ec585ee238761ee92137ba290216d1` | 155 | `82c06fc57e0eee9740236e9c63a33d6af2ec585ee238761ee92137ba290216d1` |
| Run-asset files in 115 canonical run directories | 1,193 | `a0db7e3fb969fbd517451e29374dfeca0ae99723c865a8828df5dddfe7c2042e` | 1,193 | `a0db7e3fb969fbd517451e29374dfeca0ae99723c865a8828df5dddfe7c2042e` |
| Checkpoint files (`*checkpoint*`, `.pt`, `.pth`, `.ckpt`) | 535 | `219623122ae9c9525dd77aeac34f454006754cac8d16fd8bd2af2e4e69679d63` | 535 | `219623122ae9c9525dd77aeac34f454006754cac8d16fd8bd2af2e4e69679d63` |

At pre- and post-suite checks, the canonical execution gate, active hold, frozen search gate, v1.4 release ledger, and immutable release retained the exact hashes in the frozen evidence table above. The hold remains active with ID `cd2dbf50-3a50-4b81-b976-38f1b03991a3`. Process inspection found no live F0 worker/supervisor/child; `fuser` found no holders on execution-state or worker lock files. The incident failure record still has `canonical_claim: null`. Regression tests used one task Python process at a time through the five-slot wrapper. Their migration/release writes occurred only in isolated temporary clones.

RELEASE NO-GO remains in force. No production migration, release, gate update, hold transition, worker launch, queue/status/run/checkpoint mutation, or training occurred.
