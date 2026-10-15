# REVIEW REPORT — seven historical incident clone-input supplements

## 1. Verdict

**PASS — exact seven-file scope/identity only.** This report incorporates the prior read-only scope PASS for the historical SIGSEGV failure audit and separately reviews the six exact historical logs below. It grants no copy, setup, retry, test, qualification RUN, production release/claim, staged GO or scientific RUN authorization. Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain.

## 2. Exact set and necessity

Every filename below has source prefix `experiments/stage_F0_outputs/runs/worker_logs/`; each is a separate path identity.

| ID | Filename | Exact SHA-256 |
|---|---|---|
| A | worker_1_failure_1791819200_6e9216332b2e4a488e8c5b7c2ed86922.json | `9772e99957fe462bb4b13b15e8482642a226595debded98794326c700291b19b` |
| B | trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.stderr.log | `a6722df04beeb98a06ae22d0edc3816e1baae5456d98b601fe18a2c0425e0f4c` |
| C | trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.phases.jsonl | `1a805fc237518b505908a48e9bfc72d48fe345f53c27951459a5d430bae246ea` |
| D | trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.stdout.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| E | trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.stderr.log | `61114ee849cec7ef4b922ffb12257f3d8c89cb260dd768e48faf9b4cc19cef0a` |
| F | trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.phases.jsonl | `03668c6d341fb6b95fcada120ea76c9ff782da6a3fbeada34ce433269021a57c` |
| G | trajectory_worker0_2_c87c8e65bc7e4b73bed880465ff2dffb.stdout.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Attempt06 raw preflight fails at migration :518 on absent/mismatched durable SIGSEGV audit. Actual v4 source binds A's exact path/hash to historical run `7fd2006bdc8f1817a6f7`, worker1 invocation2 nonce `6e9216332b2e4a488e8c5b7c2ed86922`; `validate_provenance` :514–518 requires it. The earlier A-only scope PASS observed stable matching source identities during 2026-10-14T03:18:26.244694–03:18:26.345058Z. This report revalidates A alongside B–G; it provides the requested single durable scope artifact for all seven.

Actual `_incident_asset_inventory` :266 onward includes v4-bound stderr/phase paths for both historical workers, the SIGSEGV audit, and nonce-specific stdout paths when present. Independently inspected frozen v1.3 migration PREPARED `incident_asset_inventory` contains **all seven exact paths, sizes and hashes** in the table. Historical validator :1024–1030 compares computed inventory and digest with PREPARED/COMMITTED and frozen expected digest `10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e`. Thus both empty stdout files are required for exact frozen inventory equality; omitting them changes the inventory despite their equal content hashes. D and G are distinct ordinary files with distinct source inodes and must remain distinct manifest/copy entries.

All seven are absent from attempt06 clone and clone input manifest; each filename has zero mentions in independently inspected B6 baseline03 manifest. They are narrowly necessary historical fixtures, classified as live supplements rather than B6 archived payloads. This assessment used static code/archived JSON inspection, not execution of historical inventory construction.

## 3. Independent source observations

Two `O_RDONLY|O_NOFOLLOW` reads per file used mandated project Python through the task-slot wrapper and external scheduler. Combined observation window: `2026-10-14T03:20:27.440336+00:00` to `2026-10-14T03:20:27.934279+00:00`. For each file both hashes equaled the exact expected digest, and lstat-before/open-FD fstat-before/open-FD fstat-after/lstat-after were identical across both reads. All seven were ordinary regular files, dev 2049, mode 0664, nlink 1.

| ID | Inode | Bytes | mtime_ns | ctime_ns |
|---|---|---|---|---|
| A | 54136062 | 3323 | 1791819200164168627 | 1791932994080340272 |
| B | 54136054 | 654 | 1791819131265074593 | 1791932985171884380 |
| C | 54136052 | 6106 | 1791819131557087642 | 1791932984931872087 |
| D | 54136053 | 0 | 1791819126824876261 | 1791932985363894213 |
| E | 54136060 | 2242 | 1791819201252217725 | 1791932953038290121 |
| F | 54136058 | 8954 | 1791819201204215559 | 1791932952798279130 |
| G | 54136059 | 0 | 1791819141929551494 | 1791932953154295433 |

All seven paths lie outside the frozen F140 expected-absence set; all 560 F140 paths remain absent in production and attempt06 clone. Both historical SIGSEGV receipt `trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.receipt.json` and current incident receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json` remain absent in production and clone. Neither receipt is included or permitted by this seven-file scope PASS.

## 4. Later provenance requirements and risks

No blocking exact-set scope defect found. **MINOR / stability limit:** source mode 0664 is writable; stability holds only for the recorded observation window. Later handling must revalidate rather than assume immutable sources.

Only if separately authorized:

- Use a fresh clone, preserve failed/prior roots, and bind this report's actual path/hash. Give each of the seven distinct paths its own `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` entry under `F0-Q3-CLONE-INPUT-PROVENANCE-v1`; retain earlier supplement/design/review provenance. No path deduplication by content hash, including D/G.
- Record exact expected hash, source/destination path and UTC, source lstat/open-FD fstat before/after matching reads and after copy. Reject changed dev/inode/kind/size/mode/nlink/mtime/ctime, symlinks or hash mismatch; never refresh expected digests or weaken validators.
- Create each destination using `O_EXCL` ordinary byte copy, without hardlink/reflink. Require regular non-symlink destination, distinct source/destination inode, nlink 1, exact size/hash readback and recorded mode/stat. Fsync each file and parent directory; preserve empty ordinary files at both stdout paths. Never change source bytes/modes.
- Verify complete updated clone inventory, all prior exact source/design/review/supplement identities, 19 frozen dependencies, protected before/after hashes, both receipt absences and all 560 F140 absences. Archive exact commands/exits/logs and final artifact manifest externally, using mandated project Python and reserved slots.

PASS does not establish complete transitive fixture coverage. `_incident_asset_inventory` also considers attempt trees, historical peer receipt and hold-bound peer failure audit; no additional missing input is approved by this report. Any subsequent preflight failure must be preserved with no same-attempt retry or inferred success.

## 5. Corrections made and run safety

No implementation/protected file changed, no input copied, no test run and no model loaded. Only this explicitly requested report was added. **Scope/identity PASS grants no copy/setup/test/RUN authority.**

## 6. Evidence identities

Attempt06 evidence root: `/tmp/f0-v15-Q3-run-20261014-f`.

| Evidence | SHA-256 |
|---|---|
| archive/preflight.stderr.raw | `37a85d37f81699a3be820871da49bb23138c4be6cbf5510ea2dd294f91d9d5ac` |
| clone_input_manifest.json | `bae51911927131935f327a61aaafacc6d1dea80a2c0be322b0ee8f41535b496a` |
| clone historical postabnormal_sigsegv_audit_v5_migration.jsonl | `2608edd595fcfdc46791eeb73b3aa42b1c82d75b265c7d3abb16146884284f08` |
| clone v4 source postabnormal_sigsegv_audit_v4_fresh_search.json (prior A-only scope check) | `a50162f3016841257e8c098cc0040c9ff2be4edb30073c4c2fb42f8ed416f839` |
