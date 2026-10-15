# Stage F0 v1.5 B6 forward evidence policy — implementation report v3

**Disposition: IMPLEMENT correction for independent review.** This update fixes the F140 coverage schema and preserves the v2 archive/quiescence protections. It changes only the B6 evidence collector, its synthetic tests, and this report. It does not collect a production baseline, change protected state, or authorize release, qualification RUN, claim, training, or GPU work. Historical B6 remains `OPEN_FAIL_IRRECOVERABLE_GAP`; execution authorization remains `NO_GO`.

## F140 schema correction

F140 absence coverage is now represented as `F0-B6-F140-ABSENCE-v1`, with exactly 140 sorted unique canonical 20-hex run IDs. Every member has an explicit `paths` mapping with exactly `status`, `lock`, `run`, and `attempt` roles. Each role is bound to its canonical relative path under one declared runs scope root:

- status: `queue_status/<run_id>.json`
- lock: `canonical_locks/<run_id>.lock`
- run: `<run_id>`
- attempt: `<run_id>/attempts/attempt-00`

All 560 paths must be required inventory entries and explicitly expected absent. The schema rejects missing or misassigned roles, duplicate member IDs, path reuse within or across members, and path reuse with another evidence category. Role semantics are explicit in the manifest; the validator does not infer a role from a filename suffix. This permits realistic names such as `queue_status/<run_id>.json` and permits the same canonical `.lock` role for each of the 140 different members without collapsing their identities.

The v2 safeguards remain: all nine policy categories must contain nonempty evidence; non-F140 categories bind required regular files with hash and before/after inventory metadata; the F140 member/role set is exact; quiescence must contain four distinct regular archived snapshots (`processes`, `claims`, `slots`, `proc_locks`) with hash and stable before/after metadata; no coverage/quiescence path reuse is accepted; archive/source inode equality is rejected; execution authorization stays `NO_GO`.

Tests include a complete synthetic 140-member coverage graph with realistic queue-status and canonical-lock paths, asserting all four role records per member and 140 distinct lock paths. Negative fixtures reject incomplete roles, misassigned role paths, cross-category reuse, empty categories, one absent path reused for all quiescence observations, a missing quiescence section, and archive/source hardlink aliasing.

## Verification

All Python checks used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Syntax check:

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -c "from pathlib import Path; files=['experiments/stage_F0_B6_evidence.py','experiments/tests/test_stage_F0_B6_evidence.py']; [compile(Path(p).read_bytes(), p, 'exec') for p in files]; print('syntax compile: OK (2 files)')"
syntax compile: OK (2 files)
```

Final focused command (one Python process, entirely synthetic temporary roots, unique scheduler root and basetemp):

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-b6-policy-v3-scheduler-20261014 experiments/stage_F0_outputs/runs/run_f0_python_task.sh b6-policy-v3-focused /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/f0-b6-policy-v3-focused-20261014/basetemp experiments/tests/test_stage_F0_B6_evidence.py
```

Raw output:

```text
....................                                                     [100%]
20 passed in 5.88s
```

No production baseline collection, production lock acquisition, GPU work, release, claim, migration, model/data access, or training was performed. No gate, hold, release ledger, queue, status, run, checkpoint, worker log, incident log, or immutable artifact was written. The tests create only synthetic evidence in pytest temporary directories.

## File hashes

- `experiments/stage_F0_B6_evidence.py`: SHA-256 `f76433a63a9fafa9332752f837a058c470473405b3596bfbba6e9e599da5b404`.
- `experiments/tests/test_stage_F0_B6_evidence.py`: SHA-256 `d0bf4b46c08e800620d237788be8ef93c46b98ce2b38c522ca8fea34a148674d`.
- This report's SHA-256 is supplied alongside its path.
