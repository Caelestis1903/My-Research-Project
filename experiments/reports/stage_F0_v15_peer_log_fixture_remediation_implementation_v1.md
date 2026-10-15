# Stage F0 v1.5 peer-log fixture remediation — implementation report

Only `experiments/tests/test_stage_F0_successor_v15.py` changed. In the peer-continuity branch, after `_detach_clone_worker_logs(repo)` and creation of the external peer evidence root, the fixture now archives the inner clone's historical `audited_worker_1.jsonl` bytes and records SHA-256, size, mode, device, and inode metadata. It asserts the inner clone does not share an inode with the outer source and that the detached clone bytes match the source snapshot.

Before any supervisor starts, the fixture writes a new empty regular file to a same-directory temporary inode, fsyncs it, atomically replaces only the inner clone's peer audit-log path, and fsyncs the parent directory. It verifies the replacement inode differs from both the original clone inode and outer source inode, the outer source bytes remain unchanged, and the strict `v15._read_jsonl_exact` reader accepts the empty file. Subsequent simulated live-worker events are appended as structured JSONL. No parser or runtime-validator behavior was modified; the historical malformed production log is preserved and remains unresolved, so B6 remains OPEN/FAIL/NO-GO.

## Verification

Project Python: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

- `py_compile experiments/tests/test_stage_F0_successor_v15.py`: passed.
- Focused command used the required slot wrapper with unique scheduler root `/tmp/f0-v15-peer-log-cow-fix-20261014-a/scheduler` and basetemp `/tmp/f0-v15-peer-log-cow-fix-20261014-a/basetemp`:

```text
/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-peer-log-cow-focused /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q --basetemp=/tmp/f0-v15-peer-log-cow-fix-20261014-a/basetemp experiments/tests/test_stage_F0_successor_v15.py::test_peer_log_clone_freshening_archives_history_and_changes_only_clone experiments/tests/test_stage_F0_successor_v15.py::test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral
```

Raw result: `2 passed in 0.14s`, exit 0. This focused test exercises hardlink-alias rejection, preservation of malformed historical bytes, clone-only atomic replacement, inode separation, source-byte stability, and strict empty-log parsing. The fixture-integrity check also confirms the peer continuity integration calls the intended generated wrapper path. The long `TrajectorySupervisor.run()` E2E and training were not run.

Raw stdout: `/tmp/f0-v15-peer-log-cow-evidence-20261014/focused.stdout.txt`, SHA-256 `0a97b591ab7ce9644866a9aea8e19c37ff6647585418c546d0481e0cb92bfbc7`. Manifest: `/tmp/f0-v15-peer-log-cow-evidence-20261014/SHA256SUMS.txt`, SHA-256 `e1969d4fe7adb76cedce6d9d045d1bc20442ec049791d719da9c97dc160042e5`.

## Production side effects and hashes

The focused tests use pytest temporary roots. Current outer worker audit logs retain their frozen hashes; no production log bytes were written:

- `experiments/stage_F0_outputs/runs/worker_logs/audited_worker_0.jsonl`: `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2`.
- `experiments/stage_F0_outputs/runs/worker_logs/audited_worker_1.jsonl`: `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960`.
- `experiments/tests/test_stage_F0_successor_v15.py`: `c2365231af4deebc8bdbeff48f359c8b7a7a61646ba882ccc36f38f4b8f1cd76`.

No production gate, hold, queue, status/run/checkpoint, release pointer/ledger, or immutable evidence artifact was changed. No release, migration, or training was executed. This fixture reset does not repair or authorize the malformed historical production log; B6 remains OPEN/FAIL.
