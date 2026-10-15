# Stage F0 v1.5 B6 forward evidence policy — implementation report v2

**Disposition: IMPLEMENT correction for independent review.** This update addresses the blocking manifest-validation probes. It changes only the B6 evidence tool, its synthetic tests, and this report. It does not collect a production baseline or authorize qualification RUN, release, claim, training, or GPU work. Historical B6 remains `OPEN_FAIL_IRRECOVERABLE_GAP`; execution authorization remains `NO_GO`.

## Corrections

The validator now requires all nine policy coverage categories to carry non-empty, uniquely referenced, exact required evidence paths. The eight evidence categories must point to regular files with hashes and before/after inventory metadata. The F140 category must identify distinct status, lock, run, and attempt paths, all explicitly absent. Coverage paths cannot be reused across categories or reused as quiescence snapshots.

Quiescence has an enforced schema containing exactly `processes`, `claims`, `slots`, and `proc_locks`. Those snapshots must reference four distinct required inventory paths, each an archived regular file with a content hash and matching before/after stat metadata. The manifest must contain the complete quiescence section, its observation map must contain exactly those four categories, and each observation must equal its inventory row. Missing sections, paths, absent paths, cross-category aliases, symlinks, and malformed rows fail closed.

Manifest validation now compares every archived raw inode against the recorded source inode and the currently observed source path identity; source metadata must still match the recorded inventory. The archive remains an ordinary, single-link file. Symlink leaf evidence remains supported without following its target; intermediate symlink components remain rejected.

Four adversarial cases were added for the reported empty-category, quiescence-path alias, and missing-section probes, plus coverage cross-category reuse and archive/source inode aliasing. No production baseline was collected and no protected path was changed.

## Verification

All Python checks used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Syntax check:

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -c "from pathlib import Path; files=['experiments/stage_F0_B6_evidence.py','experiments/tests/test_stage_F0_B6_evidence.py','experiments/tests/test_stage_F0_successor_v15.py']; [compile(Path(p).read_bytes(), p, 'exec') for p in files]; print('syntax compile: OK (3 files)')"
syntax compile: OK (3 files)
```

Final B6 focused test command (one Python process, synthetic temporary roots, unique scheduler root and basetemp):

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-b6-policy-v2-final-scheduler-20261014 experiments/stage_F0_outputs/runs/run_f0_python_task.sh b6-policy-v2-final /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/f0-b6-policy-v2-final-20261014/basetemp experiments/tests/test_stage_F0_B6_evidence.py
```

Raw output:

```text
..................                                                       [100%]
18 passed in 3.37s
```

An earlier complete run exposed two failures because the first inode-check implementation rejected symlink leaf paths. That check was narrowed to allow a symlink leaf while still refusing intermediate symlinks. The final 18-case run above passed after that correction. No long v15 qualification E2E was run; its PRE_HOLD assertions were corrected in the separate v15 test fixture but remain unexecuted here.

No model/data load, CUDA test, claim, release, production lock acquisition, migration, training, or production baseline collection was performed. No gate, hold, release ledger, queue, status, run, checkpoint, worker log, incident log, or immutable artifact was written by these tests; test-generated evidence lived under pytest temporary roots.

## File hashes

- `experiments/stage_F0_B6_evidence.py`: SHA-256 `76451b1f6f4933464c7a28b07bea6215531b2b67c10bccdf8d42d693c6b68ea4`.
- `experiments/tests/test_stage_F0_B6_evidence.py`: SHA-256 `bf495141763c6b2e65062f18c8d971aeed61e81492f805eab536ec44a385d674`.
- This report's own SHA-256 is supplied alongside its path.
