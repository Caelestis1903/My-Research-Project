# Stage F0 v1.5 successor control — implementation v4 diagnostic report

**Status: IMPLEMENT remains incomplete; independent review is not requested.** This report records the approved host-binding adjustment and one clone-only diagnostic attempt. It does not close review findings C1–C3, authorize RUN, or change the active production hold. B6 remains OPEN/FAIL.

## Scope and changes

The runtime adjustment in `experiments/stage_F0_successor_v15.py` binds worker lock/control host fields to the non-empty host in the authenticated `CHILD_STARTED` invocation. GPU owner host is bound to that same invocation host. A status record may omit `host`; if it supplies one, it must match. Optional richer GPU-owner identity checks remain in place. The archived `owner_and_locks.json` from the prior `accepted:false` attempt actually contains `status_owner.host == gpu_owner.host == "ubun"`, so the missing-status-host case was **not** established as that rejection's cause. This is a robustness/schema correction only.

The test fixture in `experiments/tests/test_stage_F0_successor_v15.py` covers missing status host acceptance and status/GPU host mismatch rejection. Its generated supervisor wrapper now traces only the `_authenticated_peer_progress` code object, recording executed source line numbers and boolean return values without status bodies, PID, nonce, or resource secret. The non-75 failure path archives a source-line map, phase/log/receipt artifacts, process state, lock probes, and `/proc/locks` to the external evidence directory.

## Verification

Project interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` (Python 3.13 environment).

- `py_compile` for `experiments/stage_F0_successor_v15.py` and `experiments/tests/test_stage_F0_successor_v15.py`: passed.
- Focused CPU test, after the trace and diagnostic additions: `1 passed in 0.09s`.
- One authorized clone-only Serialized peer diagnostic ran with `CUDA_VISIBLE_DEVICES=1`, a unique basetemp, external evidence root, and a 600-second cap. Recorded result: `1 failed in 332.42s (0:05:32)`, at the fixture assertion that the real peer failed to reach `canonical_claim_after`. The original outer pytest stdout was not retained as a standalone raw file; the bounded result and failure evidence were copied to the evidence archive. This run is not a pass and was not retried.

The deferred worker-0 child reached `child_preclaim_entry`; it did not reach `serialized_preclaim_memory_check`, `canonical_claim_before`, or `training_before`. The auth trace contains only `hook_installed`, with no validator invocation or return value. Thus this run does not explain the prior `accepted:false` result. The archived process snapshot records worker-0 child PID `2192979` at `wchan=hrtimer_nanosleep`; worker-0 supervisor PID `2192463` had `wchan=0`. A `/proc/locks` row for device/inode `08:01:54138857` names the worker-0 supervisor as lock owner at snapshot time. That ownership snapshot does not prove a deadlock or establish the cause of the wrapper wait. The wrapper waited for a peer marker while the peer did not reach canonical claim; whether lock contention contributed remains undetermined. This is an incomplete test orchestration attempt, not an authenticated-peer acceptance result.

No training phase was reached in this attempt. The child used an empty sentinel dataset and fake resource telemetry. The peer worker log copied to evidence contains historical rows; the current attempt's phase record has only supervisor start, child spawn, and cleanup signal. No research conclusions were analyzed.

## Production state integrity

The production pre-run inventory is `/tmp/f0-v15-v4-authtrace-20261014-a/production-before.json`, SHA-256 `cde34a0fb684066ff74105b447d1153f059284631771b85fb7c5123d7a6c4025`. The post-run inventory is `/tmp/f0-v15-v4-authtrace-20261014-a/production-after.json`, with the same SHA-256. Their protected-file maps and captured F140 state are identical: 140 frozen queue members and zero materialized status/run/checkpoint state items.

Protected pre/post hashes:

| Asset | SHA-256 |
|---|---|
| Active hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| `audited_worker_0.jsonl` | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` |
| `audited_worker_1.jsonl` | `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960` |
| v1.5 approval record | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` |
| v1.5 no-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` |
| Production execution-state lock bytes | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The regular `worker_1.log` was not included in the saved before/after file map, so this report does not claim a pre/post hash comparison for it. Its current hash and the clone evidence copy hash are both `b349368dd3330234032eba7456f8daeb83e73757b798a385ff59431f2c1651c4`; the test's COW helper detaches every regular worker-log file before running. No production file was intentionally written.

## Evidence and hashes

External evidence root: `/tmp/f0-v15-v4-authtrace-20261014-a/evidence/`.

- Pre-kill process/lock/phase snapshot: `evidence/lock-blocked-prekill/`.
- Peer preclaim failure artifacts: `evidence/peer-preclaim-failure/`.
- Redacted peer-auth trace: `evidence/peer-auth-decisions.jsonl` (only `hook_installed`; no predicate call).
- Aggregate manifest: `evidence/SHA256SUMS.txt`, SHA-256 `75849b0b59e51ba72b97e76040e159a827d558f0136f8d49c79f409579dae144`.
- Pre-kill snapshot manifest: `evidence/lock-blocked-prekill/SHA256SUMS.json`, SHA-256 `5ab2c0ce0468f17d3dcfb25c32dedf7b2cf047ce66933895fe7e6427564b5ffa`.

The aggregate manifest has 172 rows including its own entry: 171 payload entries match, while the self-entry does not. The pre-kill manifest has 38 rows and all 38 entries match. These are archive-integrity facts, not evidence of the wrapper wait's cause.

The test-owned supervisor process groups were stopped after the process snapshot was saved; no external scheduler process was targeted. The pytest basetemp and evidence archive preserve phase and process artifacts, but the original outer pytest stdout was not separately retained.

## Current file hashes and disposition

- `experiments/stage_F0_successor_v15.py`: `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `experiments/tests/test_stage_F0_successor_v15.py`: `ea670cb5b3c21c38a8fa80fdd41877d61a4a61e30e1cce1102f5e5d919b27e0b` was the v4-era hash; the v5 report records the current hash.
- This report: hash to be recorded after it is written.

The production hold remains active; canonical gate, v1.4 release pointer, release ledger, queue, status/run/checkpoint state, and immutable remediation artifacts were not changed. No release or training was run. B6 incident disposition remains unresolved because the trusted pre-incident hash and complete pre-edit inventory do not exist; this report does not infer or claim recovery.
