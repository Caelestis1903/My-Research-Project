# REVIEW REPORT — historical worker0 receipt clone-input scope

## 1. Verdict

**PASS — scope only for exactly one proposed ordinary-copy input in a future fresh Q3 clone.** This PASS does not authorize copying/preparation, tests, retry, qualification RUN, production release/claims, staged GO or scientific RUN. It does not change attempt04 FAIL, Q3 NOT CLOSED, B6 historical OPEN/FAIL, forward NOT_GRANTED or production execution NO_GO.

Source relative path: `experiments/stage_F0_outputs/runs/worker_logs/trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.receipt.json`.

Exact expected SHA-256: `3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7`.

## 2. Necessity and independent verification

Attempt04 raw stdout (`a330c26076b51e75c178f753326e3fb7154f3eb8707c4d77e2a00524d9f8abc3`) fails before PREPARED at `_build_valid_pending_graph` → `build_source_snapshot` → historical validation → `stage_F0_migration.validate_provenance` :425–428 → `sha256_file`. Actual validator unconditionally hashes this historical peer receipt and requires the exact expected digest above. Clone inventory lacks it; baseline03 manifest has zero mentions of its filename. Inclusion is narrowly necessary to satisfy this observed existing validation dependency; it adds no algorithm or changes validation semantics.

Two read-only `O_RDONLY|O_NOFOLLOW` observations used project Python through the task-slot wrapper and external scheduler. Window: `2026-10-14T02:41:13.353668+00:00` to `2026-10-14T02:41:13.454213+00:00`. Both hashes matched; every lstat-before/open-fstat-before/open-fstat-after/lstat-after tuple across both observations was identical:

| Property | Observed value |
|---|---|
| Kind | Ordinary regular file |
| Device / inode | 2049 / 54136064 |
| Size / mode / nlink | 8,131 bytes / 0664 / 1 |
| mtime_ns | 1791819201072209601 |
| ctime_ns | 1791932952946285908 |
| SHA-256 | `3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7` |

This historical worker0 peer receipt differs from the protected current incident worker1 receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json`, which remains absent. The proposed source is outside all 560 expected-absent F140 paths; all 560 remain absent in production and attempt04 clone. It must not be misclassified as an archived B6 payload or fabricated current-incident evidence.

## 3. Conditions for any later attempt

- Use a fresh clone and external evidence root; preserve attempt04 and all baseline/prior evidence. Include only this exact additional relative path/digest under `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` in `F0-Q3-CLONE-INPUT-PROVENANCE-v1`, retaining existing five supplements and two `EXISTING_REVIEWED_INPUT` design bindings.
- Record source/destination paths, expected SHA, UTC window, source lstat and opened-FD fstat before/after reading and after copy. Require regular non-symlink source and unchanged dev/inode/kind/size/mode/mtime/ctime; reject hash/identity/stat mismatch. Record at least two matching reads. Revalidate at actual copy time rather than treating this scope-review window as permanent stability proof.
- Create destination with `O_EXCL` by ordinary byte copy; no hardlink/reflink. Require distinct source/destination inode identity, regular non-symlink destination, nlink 1, exact size and SHA readback, file fsync and parent-directory fsync. Preserve/record appropriate source mode.
- Record complete updated clone inventory, exact seven reviewed source/design/review hashes, all supplement/binding identities, all 19 frozen dependencies, protected before/after comparisons, incident worker1 receipt absence and all 560 F140 absence checks. Hash/archive commands, exits, logs and final artifact inventory outside clone. Use mandated project Python and reserved slots.
- Scope PASS establishes this dependency only. It does not establish all downstream fixtures complete; stop and preserve any subsequent failure. Separate preflight/RUN disposition is required.

## 4. Corrections made

None. No receipt prepared/copied, no test run, no implementation/protected paths modified. This report was subsequently persisted at the orchestrator's explicit request.

## 5. Remaining risks

Source mode 0664 is writable; stability is established only for the recorded observation window. No continuous quiescence or complete transitive dependency claim is made. New copy-time mismatch is FAIL/NO_GO, not permission to refresh the expected digest or weaken validators.

## 6. Whether experiments are safe to run

**No RUN authorized by this report.** PASS approves only the proposed one-file clone-input scope; original receipt/F140 absence controls remain mandatory and scientific execution remains NO_GO.
