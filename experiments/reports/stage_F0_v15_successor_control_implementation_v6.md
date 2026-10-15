# Stage F0 v1.5 successor control — implementation v6 report

**Disposition: IMPLEMENT checkpoint for independent review.** This report addresses the v5 parent-harness marker race and process-count correction. It does not authorize RUN, release, or training. B6 remains OPEN/FAIL, and the production hold remains active.

## Marker rendezvous correction

The parent test previously observed `peer_ready` and immediately read `defer_resumed` and `defer_ready`. Those markers are produced later: the worker-0 supervisor resumes snapshot capture, spawns the deferred child, and only then the child reaches its deferral decision. A valid schedule could therefore make the parent read a file before it existed.

The parent fixture now uses `_wait_for_peer_rendezvous` in `experiments/tests/test_stage_F0_successor_v15.py`. After peer claim, this helper waits with the configured bounded supervisor timeout for all five markers: snapshot, peer preclaim, peer claim, supervisor resume, and child deferral. It requires strictly increasing monotonic timestamps in this order. Every wait iteration checks the worker-0 supervisor process. If the supervisor exits before all markers, times out, or publishes out-of-order markers, the helper writes `rendezvous_failure.json` with the reason, missing/present markers and their contents, supervisor PID/PGID/return code/alive state, elapsed time, and available stdout/stderr, then fails. When the supervisor has exited, it collects its stdout/stderr before writing the diagnostic. A still-running timeout records its process state and marker snapshot without blocking to drain its pipes.

The real peer qualification fixture calls this helper after observing the peer claim, rather than opening later marker files immediately. No long real peer E2E was run in this checkpoint.

## Focused tests and raw evidence

Project interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` (Python 3.13). `py_compile` passed for the v15 test, runtime, and worker runner modules. The complete focused CPU group ran through the required `run_f0_python_task.sh` launcher with unique scheduler root `/tmp/f0-v15-v6-focused-20261014-a/scheduler` and unique basetemp `/tmp/f0-v15-v6-focused-20261014-a/basetemp`.

The group covered embedded-wrapper compilation/order, successful parent-rendezvous ordering, early-supervisor-exit diagnostic archiving, non-75 diagnostic preservation, live-peer authentication, forged-peer rejection, and real completed-peer receipt validation. Raw result: `....... [100%]`, `7 passed in 0.35s`, exit 0. The parent-rendezvous helper itself was executed by both new CPU tests; the long `test_real_pending_graph_release_reentry_terminal_and_f140_claim` E2E was not run.

Raw stdout: `/tmp/f0-v15-v6-implementation-evidence-20261014/focused-rendezvous.stdout.txt`, SHA-256 `fa4a9a82967353cefdeab95317b1b7614e623061babc255833d883911886d20a`.

Evidence manifest: `/tmp/f0-v15-v6-implementation-evidence-20261014/SHA256SUMS.txt`, SHA-256 `e50c4ceaa958b136c337a7c676187ce51c916f77c2dc9413bffe5592cfc2981e`. It covers the v6 raw stdout and the previous v5 five-output manifest. That earlier manifest is retained as `v5-raw-outputs-SHA256SUMS.txt`; it preserves the v5 run's initial synthetic-positive failure, two fixture startup failures, intermediate subset, and final five-test pass.

The v5 final focused group passed `5 passed in 0.15s`. Its stdout SHA-256 is `bba54d0d911e4f4b3ff7099be56f23a579126dd0cbb36d88eb246b10ef320d5b`. Earlier v5 failures remain documented and archived; the new v6 result supersedes only the rendezvous helper coverage. The v5 report incorrectly stated that Python process count never exceeded two. Correct count for the v5 synthetic live-peer test is at most three simultaneous Python processes: pytest, its isolated supervisor, and that supervisor's child. The v6 focused group also stayed at or below three, within the five-process cap.

No CUDA test or training was started. The v6 run used temporary test roots and a unique scheduler root.

## Production protection and disposition

Post-test read-only hashes match the frozen values:

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

No production hold, canonical gate, v1.4 release pointer/ledger, queue, status/run/checkpoint, audit log, or immutable remediation artifact was written. No release or production migration was executed. No training was launched. B6 remains OPEN/FAIL because trusted incident-preimage and full pre-edit inventory evidence are absent.

## File hashes

- `experiments/stage_F0_successor_v15.py`: `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2` (unchanged).
- `experiments/stage_F0_outputs/runs/run_search_claim_worker.py`: `e44114d1d9f6a12635e4c0d0d42ed55d32b272bf775c6e42d966af839c40ac90` (unchanged).
- `experiments/tests/test_stage_F0_successor_v15.py`: `46005445c78a471cc998ced47264e7172a81701bdd2314f351c488fe6ef38b06`.
- v5 implementation report: `experiments/reports/stage_F0_v15_successor_control_implementation_v5.md`, SHA-256 `7f37bb2bf7c662d86902839c9d5a04484c758a535221cbd73ba61259469e8097`.
- This v6 report: SHA to be recorded after report write.
