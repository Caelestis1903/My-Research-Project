# REVIEW REPORT — run_search_worker.sh clone-input scope

## 1. Verdict and exact scope

**PASS — scope/identity only for exactly one passive clone input.** This report grants no copy, setup, retry, test, qualification RUN, production release/claim, staged GO or scientific RUN authorization. Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain.

Exact source: `experiments/stage_F0_outputs/runs/run_search_worker.sh`.

Exact expected SHA-256: `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4`.

Allowed prospective scope, only if separately authorized: an independent ordinary byte copy at the same relative path in a fresh isolated clone, solely for file hashing/control-inventory validation by the reviewed fixture. No execution authority follows from copied bytes or any frozen historical gate pin.

## 2. Necessity, absence and frozen pins

Actual v1.5 `V15_CONTROL_FILES` includes the candidate at runtime :85; `control_hash_inventory_v15` :154 rejects a missing/non-regular input. Attempt11 authorized node1 failed there from `_build_valid_pending_graph` :1434 and test :1750, before PREPARED/fault testing. Independently verified candidate path is absent from attempt11 clone disk and clone input manifest. B6 baseline03 manifest has zero mentions of the specific `stage_F0_outputs/runs/run_search_worker.sh` path. It is therefore a live supplement, not a B6 archived payload.

The exact candidate digest is independently pinned in all three inspected records:

| Record | File SHA-256 | Digest location |
|---|---|---|
| experiments/stage_F0_outputs/execution_isolation_gate.json | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` | control_hashes at the exact candidate path |
| experiments/stage_F0_outputs/runs/execution_control/bootstrap_amendment_remediation_v1.json | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` | digest_domains.current_control_inventory_digest_domain.inventory at the exact candidate path |
| experiments/reports/stage_F0_v14_implementation_review_structured.json | `57a3024aea4a14dfab029ad3b670ca175e31a54379277082d042402c3c5fac44` | control_hashes at the exact candidate path |

These bindings support exact-input identity; they do not release the current hold, authorize launching the script or establish completeness of remaining control-inventory fixtures.

## 3. Independent source identity and absence controls

Two `O_RDONLY|O_NOFOLLOW` raw-byte reads used mandated project Python through the task-slot wrapper and external scheduler, during `2026-10-14T04:13:01.882694+00:00` to `2026-10-14T04:13:01.983150+00:00`. Both SHA-256 results equaled the exact expected digest. All lstat-before/open-FD fstat-before/open-FD fstat-after/lstat-after observations across both reads matched:

| Property | Observed value |
|---|---|
| Kind | Ordinary regular file |
| Size / mode / nlink | 4,941 bytes / 0664 / 5 |
| Device / inode | 2049 / 54135521 |
| mtime_ns | 1791815980778919587 |
| ctime_ns | 1791933147136072189 |
| SHA-256 | `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4` |

The candidate path is outside F140 expected-absence controls. All 560 F140 paths remain absent in production and attempt11 clone. Both historical SIGSEGV receipt `trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.receipt.json` and current incident receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json` remain absent in both. Neither receipt inclusion nor canonical F140 creation is permitted here.

## 4. Operational risks and explicit exclusions

Static inspection confirms this is an active launcher/operator script with hardcoded production ROOT `/home/xzf/Code/phase0-3`. Its normal launch path reserves/writes scheduler slots and starts production supervisor Python; signal branches can signal recorded worker process groups or invoke a production operator helper. Its self-test also writes slot reservations and starts Python. Running the copy does not become isolated merely because the copy resides in a clone.

**Do not execute, source, chmod into an executable fixture, signal through, or self-test this script.** Passive hashing/copy-only scope grants none of these actions. No shell branch was executed in this review.

**MINOR / stability risk:** mode 0664 is writable and nlink 5 indicates shared source aliases. Source stability is bounded to the recorded observation window. Future handling requires fresh verification; source bytes, modes and aliases must remain untouched.

## 5. Conditional later copy-only provenance

Only if a later copy is separately authorized:

- Preserve failed/prior roots and use a fresh isolated clone. Include only this exact path/digest as `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` under complete `F0-Q3-CLONE-INPUT-PROVENANCE-v1`; bind this report's actual path/hash and retain all earlier provenance.
- Record exact source/destination paths, expected SHA and UTC, source lstat/open-FD fstat before/after matching reads and after copy. Reject changed kind/dev/inode/size/mode/nlink/mtime/ctime, symlink or digest mismatch without substituting evidence.
- Use `O_EXCL` ordinary byte copy with no hardlink/reflink. Require regular non-symlink destination, distinct source/destination inode, nlink 1, recorded mode/stat, exact size/hash readback, file fsync and parent-directory fsync. Preserve the source mode rather than adding executable permission.
- Verify complete updated clone inventory, all previous reviewed bindings/supplements, frozen/control dependencies, protected before/after hashes, both receipt absences and all 560 F140 absences. Archive commands/exits/manifests externally under mandated project Python/task slots. These checks do not authorize invoking this shell script or launching tests.

No claim is made that this one additional input resolves every remaining fixture dependency.

## 6. Corrections and provenance anchor

No implementation/protected file changed, candidate copied/executed or test run. Only this report was materialized at the orchestrator's explicit request for provenance binding.

Attempt11 evidence root: `/tmp/f0-v15-Q3-run-20261014-k`; independently audited clone input manifest SHA `177be472ca077da6d4486d6646bbcc3a69259498a76c49c2a6191ec779379666`; node1 stdout SHA `de137240deecde0a35044d68e6bb81ea57bdc103a53d50e058a85e676df732fe`.

**Scope/identity PASS grants no copy/setup/test/RUN authority and no execution/signal/self-test authority for run_search_worker.sh.**
