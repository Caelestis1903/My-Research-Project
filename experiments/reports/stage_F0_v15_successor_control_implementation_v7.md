# Stage F0 v1.5 successor control — implementation v7 report

**Disposition: IMPLEMENT checkpoint for independent review.** This revision fixes the marker publication race found in the v6 review. It does not authorize RUN, release, or training. B6 remains OPEN/FAIL; the production hold remains active.

## Complete-marker rendezvous

`_wait_for_peer_rendezvous` now reads and parses each marker on every poll. It accepts a marker only when its content is valid JSON, the top-level value is an object, and `monotonic` is a finite numeric value (booleans are rejected). Missing files, an empty file between open and write, truncated JSON, encoding/read errors, and malformed timestamp fields remain pending until the bounded timeout or supervisor exit. The helper then requires the strict event order `snapshot < peer preclaim < peer claim < supervisor resume < child deferral`.

On supervisor exit, timeout, or invalid ordering, the helper writes a unified failure record before raising. The record includes the reason, missing/incomplete marker names, raw marker contents and parse/read errors, supervisor PID/PGID/alive status/return code, elapsed time, and stdout/stderr when the process has exited. This keeps the existing `Path.write_text` producers and waits for their completed bytes rather than treating file existence as publication.

The parent peer fixture calls this same helper after observing the peer claim. A new CPU writer test invokes actual `Path.write_text`; it intercepts `Path.open` only for the final marker, pauses 300 ms after the `open` returns and before `write_text` calls `write`, and confirms the marker exists at size zero. The parent then calls `_wait_for_peer_rendezvous` and verifies it waits until complete JSON is written. The early-exit test now leaves an empty marker file before exiting and confirms the unified diagnostic reports it as invalid with a `JSONDecodeError` alongside the missing final marker.

## Verification

All Python commands used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` (project Python 3.13). `py_compile` passed for the focused test, v15 runtime, and worker runner. No CUDA test or training was launched.

The focused regression group ran through `run_f0_python_task.sh` with unique scheduler root `/tmp/f0-v15-v7-focused-20261014-b/scheduler` and unique pytest basetemp `/tmp/f0-v15-v7-focused-20261014-b/basetemp`. It executed the actual parent rendezvous helper for ordered success, early exit with a partial marker, and the real `Path.write_text` open-before-write pause, plus wrapper ordering, non-75 diagnostics, live-peer validation, forged-peer rejection, and completed-peer receipt validation.

Raw result: `........ [100%]`, `8 passed in 0.75s`, exit 0. The earlier v7 focused run before adding the partial-marker early-exit assertion also passed (`8 passed in 0.83s`) and remains archived. Final stdout: `/tmp/f0-v15-v7-implementation-evidence-20261014/focused-partial-marker-final.stdout.txt`, SHA-256 `1558b4d8f56605cb75539366aacdb2b0fb3a49bc2fc2d93cef48055fc5408786`. The evidence manifest is `/tmp/f0-v15-v7-implementation-evidence-20261014/SHA256SUMS.txt`, SHA-256 `5e4a64af5a3d7b497f9cf29c05ec49184440be2a02433f00270634e333be3999`.

Peak Python process count in the focused tests was three: pytest, one fixture supervisor subprocess, and its child subprocess. The process count stayed below the five-process limit. This was CPU-only qualification. The long two-boundary `TrajectorySupervisor.run()` peer-continuity E2E, >60-second PREPARED reentry, and fault/restart matrix were not run in this checkpoint.

## Production protection

Post-test read-only hashes remain equal to the frozen values:

| Protected asset | SHA-256 |
|---|---|
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| Search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| `audited_worker_0.jsonl` | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` |
| `audited_worker_1.jsonl` | `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960` |
| v1.5 approval record | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` |
| v1.5 no-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` |

No production hold, gate, release pointer/ledger, queue, status/run/checkpoint, worker audit log, or immutable remediation artifact was written. No migration, release, or training was executed. B6 remains OPEN/FAIL because trusted incident-preimage and full pre-edit inventory evidence are absent.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: `10c7e1e673de0b8e3b36526701bc1a03bb301fcb6e44ee14bf4b9d37333c2350`.
- `experiments/stage_F0_successor_v15.py`: `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2` (unchanged).
- `experiments/stage_F0_outputs/runs/run_search_claim_worker.py`: `e44114d1d9f6a12635e4c0d0d42ed55d32b272bf775c6e42d966af839c40ac90` (unchanged).
- v6 implementation report: `experiments/reports/stage_F0_v15_successor_control_implementation_v6.md`, SHA-256 `b4a0dfe4331f8abcc12da6d5e21c44495f1f979f0198d231fe88b4f665c33b6e`.
- This v7 report: SHA to be recorded after report write.
