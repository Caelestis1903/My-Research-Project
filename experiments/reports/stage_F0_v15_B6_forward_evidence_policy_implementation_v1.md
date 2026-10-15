# Stage F0 v1.5 B6 forward evidence policy — implementation report v1

**Disposition: IMPLEMENT checkpoint for independent review only.** This work adds a read-only evidence collector/validator and synthetic tests. It does not collect a production baseline or authorize collection RUN, qualification RUN, release, claim, training, or GPU work. Historical B6 remains `OPEN_FAIL_IRRECOVERABLE_GAP`; execution authorization remains `NO_GO`.

## Implemented behavior

`experiments/stage_F0_B6_evidence.py` exposes `collect_postincident_baseline` and `validate_postincident_manifest`. The collector requires an explicit baseline ID, scoped roots/paths, required paths, explicit expected absences, protected paths, four quiescence snapshots (processes, claims, slots, `/proc/locks`), a nine-category policy coverage map, archive destination, policy identity, collector metadata, and history-missing/residual-risk records. The coverage map binds explicit required inventory entries for the current gate/hold/release, v1.4 historical controls, incident/failure streams, audit logs and incident slice, C97/H51 state and checkpoints, F140 status/lock/run/attempt paths, queue/search gate, source/runtime/control/tests, and frozen data/split manifests. Each F140 status/lock/run/attempt path must be explicitly expected absent. The collector does not discover production paths on its own.

The collector inspects each path component using `lstat`, refuses symlinked roots/intermediate components, never recurses through a symlink, records symlink targets as UTF-8 bytes in ordinary archive files, and distinguishes broken symlinks from absent paths. It refuses protected-path symlinks, regular source files with `nlink != 1`, duplicate source inodes, special filesystem objects, unsafe relative paths, and archive roots that overlap scoped or forbidden production trees. Unexpected absence and an expected-absent path that exists are recorded as rejected status rather than silently omitted.

It reads each regular file through `O_NOFOLLOW`, compares pre/open/post/path stats, hashes raw bytes, and takes two complete sorted inventories. An unstable double read is retained in both observation archives and marked `REJECTED_UNSTABLE`; a missing required path is marked `REJECTED`. Stable data is `COLLECTED_UNREVIEWED`, never a review acceptance. The inventory has deterministic sorted-key UTF-8 JSON serialization and a SHA-256. Each observation's raw files are created exclusively as new ordinary files, fsynced, read back without following links, checked for single-link status and source inode inequality, then referenced by path and SHA in the manifest. The manifest and containing directories are fsynced and read back. Existing baseline IDs are never overwritten.

All results encode `historical_B6_status=OPEN_FAIL_IRRECOVERABLE_GAP`, `forward_disposition_status=NOT_GRANTED`, `execution_authorization=NO_GO`, and the three required historical proof fields as `false`. The required policy declaration is embedded verbatim. The validator rejects attempted GO/history-PASS fields, malformed or digest-mismatched manifests, unsorted/duplicate inventory paths, inventory drift, archive aliases, and raw archive/hash mismatches. It refuses to validate `REJECTED` or `REJECTED_UNSTABLE` manifests as usable baselines. It only reads persisted evidence and has no mutation path.

The tool has no imports or calls for model loading, training, CUDA, claim/release, scheduler mutation, or lock acquisition. It writes only beneath a caller-supplied new archive directory; it never writes beneath scope roots. Inventory includes scope-root identity and deterministic child names, and records per-path device/inode/link-count/mode/size/times and before/after stats. Synthetic tests use disposable roots outside the repository's production state.

## Verification

All Python commands used the required project interpreter `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Syntax check (exit 0):

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m py_compile experiments/stage_F0_B6_evidence.py experiments/tests/test_stage_F0_B6_evidence.py
```

Final focused test command (one pytest Python process; repository slot wrapper; unique scheduler slot and absolute basetemp):

```text
experiments/stage_F0_outputs/runs/run_f0_python_task.sh b6-evidence-focused-v12-20261014 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q --basetemp=/tmp/f0-b6-evidence-focused-v12-20261014/basetemp experiments/tests/test_stage_F0_B6_evidence.py
```

Raw output:

```text
.............                                                            [100%]
13 passed in 2.50s
```

Exit status: 0. The exact raw stdout is `/tmp/f0-b6-evidence-focused-v12-20261014/archive/focused.stdout.txt`, SHA-256 `b5b558ca9b355ec9af7fe8ec6b3fc70b1864b0051a70ff091a18340fa05cfdfb`.

The 13 synthetic cases cover stable raw-byte archive and NO-GO fields; required policy-category coverage and explicit F140 absences; explicit expected absence versus unexpected absence; expected-absent path unexpectedly present; symlink target archival without target traversal; protected symlink and symlinked-root rejection; broken symlink classification; source hardlink refusal; archive/scope overlap refusal; drift between complete reads with both observations preserved; archive corruption and inventory digest tampering; GO rejection; deterministic inventory and existing-ID overwrite refusal; and missing root/malformed manifest fail-closed behavior.

During implementation, focused runs exposed and corrected fixture/tool issues. The initial command had 13 pytest setup errors because the unique basetemp parent had not been created. Subsequent iterations exposed archive/scope check ordering, archive inode leakage into the deterministic inventory digest, required-versus-expected-absent semantics, and synthetic fixture ordering/stability assumptions. Intermediate failed stdout was not separately retained; only the final v12 output is archived and reported as passing.

No production collector invocation, baseline capture, model/data load, GPU process, claim/release, production lock acquisition, migration, or training was started. No existing F0 v15 source/harness, production gate/hold/log/queue/status/run/checkpoint, or immutable incident/remediation artifact was edited by this task. The actual incident log and original runner preimage remain unreconstructed; this tool does not change B6's historical status.

## Policy binding and file hashes

- PASS-reviewed design `experiments/reports/stage_F0_v15_B6_forward_evidence_policy_design_v1.md`: SHA-256 `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a`.
- Design review `experiments/reports/stage_F0_v15_B6_forward_evidence_policy_design_review_v1.md`: SHA-256 `5d37146a57737f510469f77283255ab260e6e50adefba193bf404cece01544d6`.
- `experiments/stage_F0_B6_evidence.py`: SHA-256 `7cde4df2f06c24ecacce852953f061d1091e92466c93e7ad7ad58723c50881b7`.
- `experiments/tests/test_stage_F0_B6_evidence.py`: SHA-256 `357657470e1bd3a1ee5de47dd52f109112e321ec8ac422731b5819accd4a26e5`.
- This report: SHA-256 to be recorded in the final evidence manifest after writing.

This implementation report documents tool behavior and synthetic verification only. It does not collect, accept, or review a post-incident baseline, close B6, approve a forward disposition, or change any execution authority.
