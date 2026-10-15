# Stage F0 v15 Q3 fault-recovery attempt 01: raw runner report

This report records raw execution facts for the three review-v6-authorized pytest nodes. It contains no research or closure assessment.

## Review and source bindings

- Review-v6 verdict: PASS; evidence states `qualification_RUN_precheck=true` and authorizes only these three nodes. Evidence SHA-256: `bc863621a092418ec0884deff52059b68cfede3212a1eccaf6083f0dce5b1322`. Review report SHA-256: `cdd613c3124c151965e1d08d3a9f8986ab03b03cb64ec9d1a08db1386333bdd7`.
- Test source SHA-256: `80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9`. Runtime source SHA-256: `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- Coder report v6 SHA-256: `1667b761c43d68f8d33452b6557409a2a8caf15735a701ffb335f84e82f44723`. Design v2 SHA-256: `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`; design review v2 SHA-256: `9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb`.
- Clone: `/tmp/f0-v15-Q3-run-20261014-a/clone`; ordinary-copy manifest: `/tmp/f0-v15-Q3-run-20261014-a/clone_manifest.json`, SHA-256 `b3b91752996a4cbb023c51ca53933304aa7b3471976000902e2fe37f58e984b4`. It records 1,473 exp regular files and 10 m3oe regular files. Data archives and dataset files were excluded per clone selection.
- External archive: `/tmp/f0-v15-Q3-run-20261014-a/archive`. Unique basetemps: `/tmp/f0-v15-Q3-run-20261014-a/basetemp-boundaries`, `/tmp/f0-v15-Q3-run-20261014-a/basetemp-preimage`, `/tmp/f0-v15-Q3-run-20261014-a/basetemp-syscall`. Shared scheduler root: `/tmp/f0-v15-Q3-run-20261014-a/scheduler`.

## Invocation and raw node results

Wrapper: `/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh`. Project Python: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`. Working directory: clone root. `CUDA_VISIBLE_DEVICES` was empty. Outer timeout was 3,600 seconds; it did not fire. Nodes ran concurrently under one scheduler root.

### `test_each_release_fault_boundary_uses_independent_prepared_clone`

- Command: `timeout --signal=TERM --kill-after=30s 3600s env CUDA_VISIBLE_DEVICES= F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-v15-Q3-run-20261014-a/scheduler PYTHONPATH=/tmp/f0-v15-Q3-run-20261014-a/clone/experiments:/tmp/f0-v15-Q3-run-20261014-a/clone/experiments/stage_F0_outputs/runs:/tmp/f0-v15-Q3-run-20261014-a/clone/m3oe /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-fault-boundaries /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider experiments/tests/test_stage_F0_successor_v15.py::test_each_release_fault_boundary_uses_independent_prepared_clone --basetemp=/tmp/f0-v15-Q3-run-20261014-a/basetemp-boundaries`
- Start/end UTC: `2026-10-13T22:42:55.195848665Z` / `2026-10-13T22:42:56.063831020Z`.
- Pytest exit status: `1`; wrapper recorder status: `0`; outer-timeout status: not fired.
- Raw stdout: `/tmp/f0-v15-Q3-run-20261014-a/archive/fault_boundaries/stdout.raw`; SHA-256 `1d67e5cd043fc42bdea96d229436715ba4fb8a995ad7ef9b777726db6f9ef1c8`.
- Raw stderr: `/tmp/f0-v15-Q3-run-20261014-a/archive/fault_boundaries/stderr.raw`; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Pytest output: `1 failed in 0.4s`; failure occurred in `_build_valid_pending_graph` → `build_source_snapshot` → `_assert_frozen_artifacts`. Fault-boundary/phase iteration counts reached: 0.
- Exception text: `PermissionError: frozen v1.4/incident evidence drift: experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`.

### `test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity`

- Command: `timeout --signal=TERM --kill-after=30s 3600s env CUDA_VISIBLE_DEVICES= F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-v15-Q3-run-20261014-a/scheduler PYTHONPATH=/tmp/f0-v15-Q3-run-20261014-a/clone/experiments:/tmp/f0-v15-Q3-run-20261014-a/clone/experiments/stage_F0_outputs/runs:/tmp/f0-v15-Q3-run-20261014-a/clone/m3oe /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-preimage-fsync /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider experiments/tests/test_stage_F0_successor_v15.py::test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity --basetemp=/tmp/f0-v15-Q3-run-20261014-a/basetemp-preimage`
- Start/end UTC: `2026-10-13T22:42:55.196128590Z` / `2026-10-13T22:42:56.063574672Z`.
- Pytest exit status: `1`; wrapper recorder status: `0`; outer-timeout status: not fired.
- Raw stdout: `/tmp/f0-v15-Q3-run-20261014-a/archive/preimage_fsync/stdout.raw`; SHA-256 `54fa28e1d16516b052ce26a61d8262a9f38d56576c65390d9b55e779c42edca3`.
- Raw stderr: `/tmp/f0-v15-Q3-run-20261014-a/archive/preimage_fsync/stderr.raw`; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Pytest output: `1 failed in 0.4s`; failure occurred in `_build_valid_pending_graph` → `build_source_snapshot` → `_assert_frozen_artifacts`. Fault-boundary/phase iteration counts reached: 0.
- Exception text: `PermissionError: frozen v1.4/incident evidence drift: experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`.

### `test_each_release_syscall_boundary_has_independent_restart_oracle`

- Command: `timeout --signal=TERM --kill-after=30s 3600s env CUDA_VISIBLE_DEVICES= F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-v15-Q3-run-20261014-a/scheduler PYTHONPATH=/tmp/f0-v15-Q3-run-20261014-a/clone/experiments:/tmp/f0-v15-Q3-run-20261014-a/clone/experiments/stage_F0_outputs/runs:/tmp/f0-v15-Q3-run-20261014-a/clone/m3oe /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-syscall-restart /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider experiments/tests/test_stage_F0_successor_v15.py::test_each_release_syscall_boundary_has_independent_restart_oracle --basetemp=/tmp/f0-v15-Q3-run-20261014-a/basetemp-syscall`
- Start/end UTC: `2026-10-13T22:42:55.196183178Z` / `2026-10-13T22:42:56.052867552Z`.
- Pytest exit status: `1`; wrapper recorder status: `0`; outer-timeout status: not fired.
- Raw stdout: `/tmp/f0-v15-Q3-run-20261014-a/archive/syscall_restart/stdout.raw`; SHA-256 `930af110556b3bf49e17e582db444ce680e4b7c50b68ba280a669e040e5677d7`.
- Raw stderr: `/tmp/f0-v15-Q3-run-20261014-a/archive/syscall_restart/stderr.raw`; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Pytest output: `1 failed in 0.4s`; failure occurred in `_build_valid_pending_graph` → `build_source_snapshot` → `_assert_frozen_artifacts`. Fault-boundary/phase iteration counts reached: 0.
- Exception text: `PermissionError: frozen v1.4/incident evidence drift: experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`.

## Frozen file identity record

- Expected SHA-256 recorded by reviewed frozen-artifact evidence: `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`; evidence path: `experiments/reports/stage_F0_v15_successor_control_implementation_review_v6_evidence.json` (the same digest is present in prior v1/v2/v3/v5 review evidence files).
- Production source path: `experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`; observed SHA-256: `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`.
- Clone path: `/tmp/f0-v15-Q3-run-20261014-a/clone/experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`; file absent (no clone-file SHA).
- B6 attempt03 raw archive: `research_F/F0/evidence/B6/f0-v15-postincident-20261014-03/manifest.json`, manifest SHA-256 `9acc4f5ca2e7a93d21fc532207a13b566b006b0a9a20d74c8dda7c89a44aa1ce`. The named remediation file has no raw-file entry/path in the B6 manifest observations; it therefore has no B6 archived-file SHA. The review-evidence expected SHA above is recorded separately from B6 archive contents.

## Process and protected-path records

- Process samples: `/tmp/f0-v15-Q3-run-20261014-a/archive/process_samples.tsv`. Sampled project-Python counts: 0, 3, 3, 3; sampled peak 3 (PIDs 1614674, 1614675, 1614676). No project Python processes remained after the run.
- Production protected before check: `/tmp/f0-v15-Q3-run-20261014-a/archive/production_protected_before_check.txt`; all nine entries `OK`, exit 0. After check: `/tmp/f0-v15-Q3-run-20261014-a/archive/production_protected_after_check.txt`; all nine entries `OK`, exit 0.
- Clone protected comparison: `/tmp/f0-v15-Q3-run-20261014-a/archive/clone_protected_before_check.txt` and `clone_protected_after.txt`; cmp exit 0 and diff empty (`clone_protected_cmp.txt`, `clone_protected_diff.txt`).
- Run group start/end UTC: `2026-10-13T22:42:55.188297455Z` / `2026-10-13T22:42:56.064663059Z`; group exit 0. Individual pytest exits were 1.
- No CUDA initialization, model execution, training, sample/data loading, or production claim/release/lock API action was recorded. Each node stopped during frozen-source snapshot validation before the fault/restart cases.
- No pytest node was retried. The initial clone-builder preflight exit 1 and corrected builder exit 0 are preserved in `clone_build.*` and `clone_build_v2.*`; those setup commands are not pytest retries.

## Raw artifact references

- Per-node command, timestamps, statuses, and stdout/stderr files are under `/tmp/f0-v15-Q3-run-20261014-a/archive/{fault_boundaries,preimage_fsync,syscall_restart}/`.
- Clone builder raw logs and exit statuses are under `/tmp/f0-v15-Q3-run-20261014-a/archive/clone_build*` and `clone_build_v2*`.
- Reviewed source hashes: `/tmp/f0-v15-Q3-run-20261014-a/archive/reviewed_source_hashes.txt`. Clone manifest SHA-256: `b3b91752996a4cbb023c51ca53933304aa7b3471976000902e2fe37f58e984b4`.
- External final artifact manifest is `/tmp/f0-v15-Q3-run-20261014-a/archive/final_artifact_manifest.sha256`; its SHA-256 is reported by the runner after it is written.
