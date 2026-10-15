# REVIEW REPORT — historical v1.4 design clone-input scope

## 1. Verdict

**PASS — exact-file scope/identity only.** This input is narrowly necessary only for historical v1.4 source-snapshot validation. PASS grants no copy, setup, retry, test, qualification RUN, production release/claim, staged GO or scientific RUN authority, and does not renew historical release authority or approve a new design. Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain.

Source: `experiments/reports/stage_F0_postabnormal_design_v14.md`.

Exact expected SHA-256: `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9`.

## 2. Necessity and actual validation

Attempt07 raw traceback follows `build_source_snapshot(root=clone)` at v1.5 runtime :915 into v1.4 `validate_source_snapshot`, failing at :489 with `source snapshot does not bind the accepted v1.4 design bytes`. Actual v1.4 source :484–489 requires this exact repository-relative design path, frozen `DESIGN_SHA256`, matching snapshot path/hash fields, file existence and exact raw-byte hash.

Independently inspected clone historical snapshot has the exact candidate path and expected hash. The candidate file and clone manifest entry are absent; B6 baseline03 manifest contains zero filename mentions. Thus the observed defect is a missing historical fixture, rather than an observed mismatching source-design identity. No validator executed as part of this scope review; evidence is raw attempt07 traceback, actual code, archived snapshot fields and source hashing.

## 3. Independent identity observations

Two `O_RDONLY|O_NOFOLLOW` reads used mandated project Python through the task-slot wrapper and an external scheduler, during `2026-10-14T03:33:36.167212+00:00` to `2026-10-14T03:33:36.267702+00:00`. Both hashes matched the exact expected digest. All lstat-before/open-FD fstat-before/open-FD fstat-after/lstat-after observations across both reads were identical:

| Property | Observed value |
|---|---|
| Kind | Ordinary regular file |
| Size / mode / nlink | 16,016 bytes / 0664 / 5 |
| Device / inode | 2049 / 54269858 |
| mtime_ns | 1791852557582130550 |
| ctime_ns | 1791933151636296939 |
| SHA-256 | `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9` |

The candidate is outside the runs tree and the F140 expected-absence set. All 560 F140 paths remain absent in production and attempt07 clone. Both historical SIGSEGV receipt `trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.receipt.json` and current incident worker1 receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json` remain absent in production and clone. Neither receipt is permitted by this scope PASS.

## 4. Limits and conditional provenance

No blocking exact-file scope defect found. **MINOR / stability risk:** writable mode 0664 and nlink 5 indicate shared source aliases. Stability is bounded to the recorded window; any later handling must independently revalidate source identity/hash. Do not mutate/chmod source or its aliases.

Only if separately authorized, a future fresh clone must:

- Preserve failed/prior roots; record this report's path/hash and the candidate's exact source/destination paths and expected SHA. Classify it as `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` under complete `F0-Q3-CLONE-INPUT-PROVENANCE-v1`, not as a B6 archived payload or a replacement v1.5 design.
- Record copy-time UTC and source lstat/open-FD fstat before/after matching reads and after copy. Reject changed kind/dev/inode/size/mode/nlink/mtime/ctime or digest mismatch; do not refresh expected hashes or weaken historical validation.
- Use independent `O_EXCL` ordinary byte copy, no hardlink/reflink; require regular non-symlink destination, distinct inode, nlink 1, exact size/hash readback, recorded mode/stat, file fsync and parent-directory fsync.
- Verify complete updated clone inventory, all earlier reviewed bindings/supplements, 19 frozen dependencies, protected before/after hashes, both receipt absences and all 560 F140 absences. Archive commands/exits/manifests externally under mandated project Python and reserved slots.

This PASS establishes one historical snapshot input dependency only; full downstream fixture coverage, Q3 acceptance and execution readiness remain unestablished.

## 5. Corrections and run safety

No implementation/protected path altered, no input copied, no test run. Only this report was persisted at the orchestrator's explicit request. **No copy/setup/test/RUN authority follows from this scope/identity PASS.**

## 6. Evidence identities

Attempt07 evidence root: `/tmp/f0-v15-Q3-run-20261014-g`.

| Evidence | SHA-256 |
|---|---|
| archive/preflight.stderr.raw | `a2d1956097f38c2c39a4a1076927e7aadaff562e4f612e0969be9f8293d19081` |
| archive/preflight.stdout.raw | `6a3169c2d130d82524d5bdb1ffe38acab42820c10c637df93ecbba6275b054b9` |
| clone_input_manifest.json | `3fe49cdb9fca25910dcd9235e997e217b387d57602198fa2057ffc06cbfac3f9` |
