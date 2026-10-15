# REVIEW REPORT — historical peer checkpoint scope and identity

## 1. Verdict

**PASS — exact-file scope/identity only.** This does not authorize copying, setup, retry, tests, qualification RUN, production release/claims, staged GO or scientific RUN. Checkpoint handling remains subject to explicit user authorization. Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain.

Source: `experiments/stage_F0_outputs/runs/1f04c7fb623d18f940ac/attempts/attempt-00/resume_epoch_017.pt`.

Exact expected SHA-256: `8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c`.

## 2. Necessity and hashing semantics

Actual `experiments/stage_F0_migration.py:470–472` constructs this historical TARGET_PEER checkpoint path using the original peer attempt ID and hashes it against the exact frozen digest. Line 544 hashes the same file again for returned provenance. `stage_F0_worker_control.sha256_file` opens raw bytes and streams SHA-256. These calls do not deserialize checkpoint tensors, load a model, select a resume state, run inference or train. The scope review itself used only stdlib raw-byte hashing; torch was not imported and no model was loaded.

This checkpoint is a downstream dependency visible in actual validator source; attempt05 preflight stopped earlier on missing peer run_manifest. No claim is made that attempt05 reached this checkpoint check or that this is the final missing dependency.

B6 baseline03 inventory contains no matching checkpoint record and zero mentions of this filename; attempt05 clone and its manifest lack the exact path. It would therefore be a live supplement, not a B6 archived payload, if later explicitly authorized.

## 3. Independent identity observations

Two `O_RDONLY|O_NOFOLLOW` reads used mandated project Python via the task-slot wrapper and external scheduler, during `2026-10-14T02:55:17.495742+00:00` to `2026-10-14T02:55:17.729817+00:00`. Both SHA-256 results matched the exact expected value. All lstat-before, open-FD fstat-before/after and lstat-after observations across both reads were identical:

| Property | Observed value |
|---|---|
| Kind | Ordinary regular file |
| Size | 17,207,805 bytes |
| Device / inode | 2049 / 54657763 |
| Mode / nlink | 0664 / 5 |
| mtime_ns | 1791819200228171515 |
| ctime_ns | 1791933146964063596 |
| SHA-256 | `8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c` |

This historical H51 peer checkpoint path and its run directory are outside the frozen F140 expected-absent paths. It differs from the current incident worker1 receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json`, which remains absent. Scope PASS permits no change to either the production H51 artifact or absence controls.

## 4. Risks and conditional later provenance

No blocking exact-file scope/identity defect found. **MINOR / provenance risk:** writable mode 0664 and nlink 5 indicate shared source aliases. Stability is established only within the observation window; later handling requires fresh source identity/hash verification. Never mutate, chmod or use this source inode as a clone destination.

Only after separate explicit user authorization, any proposed fresh-clone supplement must preserve prior failed attempts and:

- Bind this report's actual path/hash; classify the exact file/digest as `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` in complete `F0-Q3-CLONE-INPUT-PROVENANCE-v1` evidence, never as archived B6 data.
- Record UTC, source path and lstat/open-FD fstat before/after matching reads and after copying. Reject source kind/dev/inode/size/mode/nlink/mtime/ctime or digest mismatch rather than changing the expected digest.
- Use `O_EXCL` ordinary byte copy without hardlink/reflink; require regular non-symlink destination, distinct inode, nlink 1, exact size/hash readback, recorded mode/stat, file fsync and parent-directory fsync. Raw-byte handling grants no model-loading authority.
- Verify complete clone inventory, reviewed source/design/review bindings and all prior supplements, 19 frozen dependencies, protected before/after hashes, all 560 F140 absence paths and protected incident worker1 receipt absence. Archive commands, exits and manifests externally under mandated project Python/task slots.

These are conditional provenance requirements, not permission to perform the copy or prepare/run an attempt. Full transitive validator-fixture completeness remains unestablished.

## 5. Corrections made

None. No source, implementation or protected path changed; no checkpoint copied/deserialized; no test executed. This report was persisted only after the orchestrator explicitly requested archival of the completed read-only review.

## 6. Run safety and authorization limit

**PASS is scope/identity only and gives no copy/setup/test/RUN authorization. Checkpoint handling remains subject to user authorization.** Q3 closure, production authority and scientific conclusions do not follow from raw-byte hashing.
