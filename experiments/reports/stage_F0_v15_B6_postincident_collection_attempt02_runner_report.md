# Stage F0 v15 B6 post-incident collection — attempt02 runner report

Raw execution record for baseline ID `f0-v15-postincident-20261014-02`. This report records invocation and collected fields; it does not assign baseline acceptance or forward disposition.

## Invocation and process record

- Started: `2026-10-13T21:10:19.423981Z`
- Ended: `2026-10-13T21:18:25.406132Z`
- Wrapper: `/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh`
- Project Python: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`
- `CUDA_VISIBLE_DEVICES` was empty; scheduler root was external at `/tmp/f0-b6-postincident-20261014-02-run/scheduler`.
- Exact command:

```text
timeout --signal=TERM --kill-after=30s 3600s env CUDA_VISIBLE_DEVICES= F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-b6-postincident-20261014-02-run/scheduler /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh b6-postincident-collection /home/xzf/miniconda3/envs/phase0-3-env/bin/python /tmp/f0-b6-postincident-20261014-02-preflight/collect.py
```

- Outer timeout: 3600 seconds; did not fire.
- Exit status: `0` (`/tmp/f0-b6-postincident-20261014-02-run/exit_status.txt`).
- Peak project-Python process count: `1` during collection; offline manifest validation also used one project-Python process.
- Collector stdout: `/tmp/f0-b6-postincident-20261014-02-run/stdout.raw`, SHA-256 `fbd2dff9cc8a5042923f3f99ed009a6d7b099bbb674851cee8d5fcc3b428e815`.
- Collector stderr: `/tmp/f0-b6-postincident-20261014-02-run/stderr.raw`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Raw collector status: `COLLECTED_UNREVIEWED`; `errors=[]`; `execution_authorization=NO_GO`.

## Archive and inventory fields

- Archive: `/home/xzf/Code/phase0-3/research_F/F0/evidence/B6/f0-v15-postincident-20261014-02`
- Manifest: `/home/xzf/Code/phase0-3/research_F/F0/evidence/B6/f0-v15-postincident-20261014-02/manifest.json`
- Manifest SHA-256: `570c22e80520aff3fa602005e1ef7128c09372f6097071bb365b15654a5f74a4`.
- `inventory_complete=true`; `stable_double_read=true`; first and second inventories each contain 2462 rows.
- `required_paths=2082`; `protected_paths=569`; `absence_paths=560`.
- All 560 F140 status/lock/run/attempt paths were represented under scope root ID `runs` and recorded absent.

Coverage counts recorded in the manifest:

| Scope category | Path count |
|---|---:|
| C97/H51 state, runs, checkpoints | 1232 |
| F140 status/lock/run/attempt absence set | 560 |
| Audit logs and incident slice | 2 |
| Current gate/hold/release | 4 |
| Frozen data/split manifests | 1 |
| Incident failure stream evidence | 14 |
| Queue and search gate | 2 |
| Source/runtime/control/tests | 183 |
| v14 historical controls | 37 |

The manifest lists four quiescence observation IDs:

| Observation | Scope path | SHA-256 |
|---|---|---|
| claims | `q:claims.txt` | `822ec99f81b048698a57c8e0b041653e5736a72f36f3f0002e158d28adc33009` |
| process locks | `q:proc_locks.txt` | `1801bde13adb9a39a9d885409a0c2e6d906d1c85e2e45ad961b4ebb28f4b9754` |
| processes | `q:processes.txt` | `2f76a008965ced8b791845ba09153d8934bd763df72d41f9e698575baa8eb705` |
| slots | `q:slots.txt` | `79a1cf9a46ba433f2b8b5585dacf03548d62a0674c4e5a0c354bb22d65abf475` |

The four source snapshot files are under `/tmp/f0-b6-postincident-20261014-02-preflight/quiescence/`.

## Protected hashes and offline validation

- Protected after-hash list: `/tmp/f0-b6-postincident-20261014-02-run/protected_sha_after.txt`, SHA-256 `3c0b3d13b173bbd99f0f4a6f6d67f31e314c16fb75661e1c239b7f030fc25d69`.
- Protected before/after comparison: after-list exactly matched `/tmp/stageF0v15qualification-20261014-attempt4/archive/protected_sha_before.txt` (`cmp` exit `0`); `protected_sha_diff.txt` was empty.
- Offline `validate_postincident_manifest` exit: `0`.
- Offline validator raw output: `/tmp/f0-b6-postincident-20261014-02-run/offline_validation.raw`, SHA-256 `dd66d0ac55a26530424d81ed047de9b06482479d1545fb018cd4ec99feeeea44`.
- Offline validator stderr: `/tmp/f0-b6-postincident-20261014-02-run/offline_validation.stderr`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Final external artifact manifest: `/tmp/f0-b6-postincident-20261014-02-run/final_artifact_manifest.sha256`, SHA-256 `988d32d55f9993c5cad887077e5881df840deb8c8895538075a224a4417e1ba0`.

## Attempt01 record

Attempt01 remains preserved at `/tmp/f0-b6-postincident-20261014-01-run/`; its external artifact manifest SHA-256 was `d0ee323dc9291fd567829a83f39ebe9009dd727281281a5681dc1bf68ca54e33`. Its invocation failed during policy input validation with `EvidenceError: F140 attempt path does not match canonical member identity`, before baseline leaf creation, source scan, hashing, or copying. Attempt01 was not counted as a collection pass and was not modified by attempt02.
