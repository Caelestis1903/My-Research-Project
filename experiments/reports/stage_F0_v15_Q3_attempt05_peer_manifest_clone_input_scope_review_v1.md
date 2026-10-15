# REVIEW REPORT — attempt05 historical peer manifest clone-input scope

## 1. Verdict

**PASS — scope only for exactly one proposed ordinary-copy input in a future fresh Q3 clone.** This does not authorize copying, setup, retry, tests, qualification RUN, production release/claims, staged GO or scientific RUN. Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain.

Source: `experiments/stage_F0_outputs/runs/1f04c7fb623d18f940ac/attempts/attempt-00/run_manifest.json`.

Exact expected SHA-256: `2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6`.

## 2. Requirement verification

Attempt05 raw preflight stderr directly reports `FileNotFoundError` for the exact clone path at `stage_F0_migration.py:468`, called by `validate_provenance` at preflight :80. Actual repository validator :466–469 constructs the manifest path from historical TARGET_PEER `1f04c7fb623d18f940ac` and original peer attempt ID, and unconditionally requires the exact digest above. This existing historical validation dependency is narrowly necessary; adding its exact bytes changes no algorithm or validation rule.

The attempted clone manifest contains no such path, and the path is absent on disk. Independently inspected baseline03 manifest has no matching file inventory record. This is therefore a live supplement, not a claimed B6 archived payload. Attempt05 report records a single failed read-only preflight, exit 1, with no pytest launch or retry; raw stdout is empty and traceback is in stderr.

Two read-only `O_RDONLY|O_NOFOLLOW` observations used mandated project Python via the task-slot wrapper and external scheduler, at `2026-10-14T02:52:08.591321+00:00` to `2026-10-14T02:52:08.691697+00:00`. Both hashes matched and all before/after lstat/open-FD fstat observations were identical:

| Property | Observed value |
|---|---|
| Kind | Ordinary regular file |
| Device / inode | 2049 / 54657795 |
| Size / mode / nlink | 13,742 bytes / 0664 / 5 |
| mtime_ns | 1791819200992205992 |
| ctime_ns | 1791933146964063596 |
| SHA-256 | `2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6` |

This historical H51 peer manifest lies outside the 560 F140 expected-absent paths. All 560 remain absent in production and attempt05 clone. The protected current incident worker1 receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json` also remains absent. A supplement copy must preserve those controls and must not modify the production H51 artifact.

## 3. Issues and limits

No blocking scope defect found for this exact file/digest. **MINOR / provenance risk:** source mode 0664 is writable and nlink 5 indicates shared inode aliases. Stability is bounded to the observation window; source must never be rewritten, chmodded or linked as a destination. Require new copy-time validation and an independent ordinary destination.

The next actual validator step :471–474 hashes `resume_epoch_017.pt`; its inclusion is outside this request and is not approved by this report. This PASS does not establish full transitive fixture completeness or that future preflight succeeds. No checkpoint loading, training or data access occurred in this review.

## 4. Required later provenance

- Preserve failed attempt05 and earlier roots. In a separately authorized fresh clone, classify this exact path/digest as `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` under `F0-Q3-CLONE-INPUT-PROVENANCE-v1`; bind this review's actual path/hash. Retain previous six supplements, two `EXISTING_REVIEWED_INPUT` design bindings and all reviewed source identities.
- Record exact source/destination paths, SHA, UTC, source lstat and opened-FD fstat before/after two reads and after copying. Require stable dev/inode/kind/size/mode/nlink/mtime/ctime and exact hashes; reject symlinks or any mismatch rather than changing the expected digest.
- Use `O_EXCL` ordinary byte copy; no hardlink/reflink. Destination must be regular, non-symlink, nlink 1 and distinct source/destination inode identity. Record source mode, destination mode/stat, exact size/hash readback, file fsync and parent-directory fsync. Never modify the source or its shared aliases.
- Update and verify the complete clone inventory, exact reviewed seven source/design/review bindings, all supplements and 19 frozen dependencies. Verify protected before/after hashes, incident worker1 receipt absence and all 560 F140 absences. Archive commands/exits/raw logs and a final artifact manifest outside the clone. Use mandated project Python and reserved task slots.
- Separately review preflight/RUN disposition. Preserve and stop on any future failure; this report grants no retry or RUN authority.

## 5. Corrections made

None. No copy, test, implementation change or protected path alteration. Only this independent review report was added.

## 6. Evidence identities

Evidence root: `/tmp/f0-v15-Q3-run-20261014-e`.

| Evidence | SHA-256 |
|---|---|
| archive/preflight.stderr.raw | `515eccd5c1123e706afc95843c2644f56bf31af94da7079bd5d2994248835634` |
| archive/preflight.stdout.raw | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| archive/attempt05_runner_report.md | `fca3be200f8e7f94e6930e073abf005d6bbb4050bc36be77e6b3f3c47505c06c` |
| clone_input_manifest.json | `169ebeae501d89cb8e145551fbd13e9a8d43129a6662187dfefe0be5e10c7a31` |
| B6 baseline03 manifest.json | `9acc4f5ca2e7a93d21fc532207a13b566b006b0a9a20d74c8dda7c89a44aa1ce` |

**Run safety: no copying/setup/test/RUN is authorized by this scope-only PASS.**
