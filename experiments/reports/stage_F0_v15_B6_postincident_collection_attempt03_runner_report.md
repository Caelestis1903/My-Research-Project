# Stage F0 v15 B6 post-incident collection — attempt03 runner report

Raw execution record for `f0-v15-postincident-20261014-03`. It records the requested scope additions, invocation, collection fields and validation output. It does not assess scientific conclusions or B6 closure.

## Invocation and execution record

- Collector start: `2026-10-13T21:36:03.811891051Z`
- Collector end: `2026-10-13T21:44:02.644879399Z`
- Project Python: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`
- Required wrapper: `/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh`
- `CUDA_VISIBLE_DEVICES` was empty. External scheduler root: `/tmp/f0-b6-postincident-20261014-03-run/scheduler`.
- Outer timeout: 3600 seconds; it did not fire.
- Exact command:

```text
timeout --signal=TERM --kill-after=30s 3600s env CUDA_VISIBLE_DEVICES= F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-b6-postincident-20261014-03-run/scheduler /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh b6-postincident-collection /home/xzf/miniconda3/envs/phase0-3-env/bin/python /tmp/f0-b6-postincident-20261014-03-preflight/collect.py
```

- Exit status: `0`; raw value: `/tmp/f0-b6-postincident-20261014-03-run/exit_status.txt`.
- Peak observed project-Python count: `1` for collection and `1` for offline validation.
- No training, sample loading, model construction, CUDA initialization, release, claim or production lock acquisition was invoked by this run.
- External working files: `/tmp/f0-b6-postincident-20261014-03-preflight/` and `/tmp/f0-b6-postincident-20261014-03-run/`.

## Added scope bindings

In addition to the reviewed attempt02 scope, this attempt added the following exact source path identities under the `m3oe` root `/home/xzf/Code/phase0-3/m3oe`:

| Relative path | SHA-256 |
|---|---|
| `MDMTRec/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `MDMTRec/basic/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `MDMTRec/basic/activation.py` | `152c5909e966055f83b7ec49de7634c853fbae8b029fbd37759371c14c921046` |
| `MDMTRec/basic/features.py` | `116025a45517021bfe86447b8229e59cf04feb665549544ce30173d04421a2b0` |
| `MDMTRec/basic/initializers.py` | `a9a6a2923eebeca7f541ffaddbccf2671a42e7f9b6a1ec8d38ba2872c53f6196` |
| `MDMTRec/basic/layers.py` | `e3eb8db91ff0aaf79c8e389107df5f64841992fab02a18b29c37bb446b22763a` |
| `MDMTRec/models/multi_domain/__init__.py` | `e11c58c743dd3125f0a85eeb89c0266706b4716b10700eb99ed1f18d19cb23e8` |
| `MDMTRec/models/multi_domain/model.py` | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |
| `MDMTRec/utils/__init__.py` | `13c62ae9d561769c4597c2f279bf03ab25363ad64ada1e77af58c4f7979cc7a3` |
| `MDMTRec/utils/data.py` | `71d08bd9487afc702473d8813449d21ffd85c223ed05457791486b6df708501a` |

It also added `experiments/history_exp/first/p7_models.py` under root `exp` with SHA-256 `7b727c4c933e93860eb9ce1dd57d58f4c40adbf3975d2d7ca6dd76afb8b6e839`, and `/tmp/f0-B6-policy-v3-review-20261014-a/probe.py` under root `probe` with SHA-256 `5b265e759aa35aa2328a0b6b0a45ffc3877d697fe975b9325699ed6da87d9a5d`.

The `probe.py` row is assigned to category `source_runtime_control_and_tests`; its selected path is distinct from paths under the sibling `review` archive root.

## Collection output and inventory fields

- Raw stdout: `/tmp/f0-b6-postincident-20261014-03-run/stdout.raw`, SHA-256 `9f0a0fa3678ea2f8b0f55b398fc1bf8169b7b94d5d9153d7e8a7b35d43c0ce00`.
- Raw stderr: `/tmp/f0-b6-postincident-20261014-03-run/stderr.raw`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Collector status: `COLLECTED_UNREVIEWED`; `errors=[]`; `execution_authorization=NO_GO`.
- Archive: `/home/xzf/Code/phase0-3/research_F/F0/evidence/B6/f0-v15-postincident-20261014-03`.
- Manifest SHA-256: `9acc4f5ca2e7a93d21fc532207a13b566b006b0a9a20d74c8dda7c89a44aa1ce`.
- `inventory_complete=true`; `stable_double_read=true`; 2476 rows in each inventory; 560 explicitly absent F140 paths; `required_paths=2094`; `protected_paths=569`.

Coverage path counts recorded by the preflight and manifest:

| Scope category | Path count |
|---|---:|
| C97/H51 state, runs, checkpoints | 1232 |
| F140 status/lock/run/attempt absence set | 560 |
| Audit logs and incident slice | 2 |
| Current gate/hold/release | 4 |
| Frozen data/split manifests | 1 |
| Incident failure stream evidence | 14 |
| Queue and search gate | 2 |
| Source/runtime/control/tests | 195 |
| v14 historical controls | 37 |

Quiescence observations are archived from these four fresh snapshot paths:

| Observation | Snapshot | SHA-256 |
|---|---|---|
| claims | `/tmp/f0-b6-postincident-20261014-03-preflight/quiescence/claims.txt` | `822ec99f81b048698a57c8e0b041653e5736a72f36f3f0002e158d28adc33009` |
| process locks | `/tmp/f0-b6-postincident-20261014-03-preflight/quiescence/proc_locks.txt` | `1801bde13adb9a39a9d885409a0c2e6d906d1c85e2e45ad961b4ebb28f4b9754` |
| processes | `/tmp/f0-b6-postincident-20261014-03-preflight/quiescence/processes.txt` | `2f76a008965ced8b791845ba09153d8934bd763df72d41f9e698575baa8eb705` |
| slots | `/tmp/f0-b6-postincident-20261014-03-preflight/quiescence/slots.txt` | `79a1cf9a46ba433f2b8b5585dacf03548d62a0674c4e5a0c354bb22d65abf475` |

## Protected after-check and offline validation

- Protected before list: `/tmp/f0-b6-postincident-20261014-03-run/protected_sha_before.txt`, SHA-256 `3c0b3d13b173bbd99f0f4a6f6d67f31e314c16fb75661e1c239b7f030fc25d69`.
- Protected after list: `/tmp/f0-b6-postincident-20261014-03-run/protected_sha_after.txt`, SHA-256 `3c0b3d13b173bbd99f0f4a6f6d67f31e314c16fb75661e1c239b7f030fc25d69`.
- Before/after `cmp` exit: `0`; diff file was empty (empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`). All nine protected-file checks returned `OK`.
- Offline `validate_postincident_manifest` exit: `0`; raw result reported `COLLECTED_UNREVIEWED`, `NO_GO`, complete and stable double inventory, 2476 rows per read, 560 absences, and `errors=[]`.
- Offline validator stdout: `/tmp/f0-b6-postincident-20261014-03-run/offline_validation.raw`, SHA-256 `e692b6a268007d66162d8890bf9cc93a332a4fccc643ad56e6f634c3e25803e7`.
- Offline validator stderr: `/tmp/f0-b6-postincident-20261014-03-run/offline_validation.stderr`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- The validator output lists all twelve added identity rows and matching source/archive SHA-256 values.
- Final external artifact manifest: `/tmp/f0-b6-postincident-20261014-03-run/final_artifact_manifest.sha256`, SHA-256 `c815bb0da73ad7dc95a1f3d40701e51a69bb196d2fcf66ce29bbfe8c4ed32a5e`.

Attempts01 and 02, their external logs/reports, and the attempt02 baseline archive were preserved unchanged. This report does not change the collector manifest or protected production paths.
