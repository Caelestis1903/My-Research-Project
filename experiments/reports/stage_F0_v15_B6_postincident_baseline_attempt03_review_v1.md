# REVIEW REPORT — B6 post-incident baseline attempt03

## 1. Verdict

**PASS — ACCEPTED_BOUNDED for the existing attempt03 baseline only.** Independent full-payload verification found no remaining blocking defect within the declared source/artifact scope. Attempt02 BASE02-R1 and BASE02-R2 are closed in this successor attempt; attempt02 and its FAIL report remain unchanged.

**B6 historical status remains OPEN/FAIL; forward disposition remains NOT_GRANTED; execution remains NO_GO.** This acceptance does not permit release, staged graph GO, training, claim, checkpoint loading or scientific RUN. The existing manifest remains COLLECTED_UNREVIEWED; ACCEPTED_BOUNDED is this independent review decision, not a write to the manifest.

## 2. Requirement-by-requirement verification

| Requirement | Independent result |
|---|---|
| Approved policy identity | Bound policy SHA `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a` matches current source binding and archived source; tool SHA `f76433a63a9fafa9332752f837a058c470473405b3596bfbba6e9e599da5b404` matches. |
| Actual manifest/artifact integrity | Baseline manifest SHA `9acc4f5ca2e7a93d21fc532207a13b566b006b0a9a20d74c8dda7c89a44aa1ce`; external final manifest SHA `6c98915d4855e48cc9f2ad041bd5004b76a4b4691ad14187d5a3255db25410e2`; all 25 external listed artifacts hash-match. |
| Scope completeness and category nonoverlap (§5.3) | All nine categories populated; category keys unique and present. All 2094 declared required paths have inventory rows. Exact v7 runtime dependency inventory has no missing source identity; eleven previous omissions now have explicit source-path rows. Selected source scope is bounded; this is not a full filesystem inventory. |
| Deterministic double read (§5.3) | 2476 sorted unique rows per observation; both full inventory SHA bindings recomputed; key path/kind/bytes/stat projections equal. No per-row error or within-read regular/symlink stat drift. Kinds per observation: 1755 regular, 22 symlink, 139 directory, 560 absent. |
| Raw bytes/readback/no alias (§5.3) | All 3554 payloads rehashed, totaling 15942591412 bytes; zero unverified, digest failures or inode/kind/nlink errors. Archives are ordinary single-link files, mutually distinct and unequal to every recorded source file inode. 912 archive ancestor directories are ordinary directories, with no symlink traversal. |
| F140 absence and exact IDs (§5.3) | Exact 140 frozen fresh_ids matched actual member identities; 560 status/lock/run/attempt paths match absence_paths. Each absent in both inventories and current lexists check; broken links would fail absence. |
| Eleven missing source paths (BASE02-R1) | Closed: exact absolute source identities, SHA and both raw ordinary archives verified, including empty __init__ files by path rather than digest coincidence. |
| Missing review script (BASE02-R2/§5.4) | Closed: probe:probe.py exists, source/archive/current SHA equals 5b265e759aa35aa2328a0b6b0a45ffc3877d697fe975b9325699ed6da87d9a5d. All eight nested tool-review manifest payload hashes have retained bytes; all 217 attempt4 payload references have retained bytes. |
| Incident/raw and synthetic evidence (§5.4) | Full worker0 log b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2 retained; lines130–138 are nine CHILD_STARTED events, raw slice SHA ada4692312d6929b8b3f67cd261f7dc37ed6a2f3867e5787317b9ffacdb163c1. Retained qualification archive is labeled SYNTHETIC_FRESH_PEER_LOG in baseline residual-risk metadata; it does not repair production malformed peer audit. |
| Quiescence (§5.3) | Four raw process/claim/slot/proc-lock snapshots hash-match the preflight files. Observed 14.652–44.472 ms before collector start. These are fresh point observations, not continuous quiescence proof; rearchived snapshots do not constitute a second live probe. |
| Protected before/after and production preservation (§6) | Nine listed protected before/after hashes equal byte-for-byte; protected-list SHA 3c0b3d13b173bbd99f0f4a6f6d67f31e314c16fb75661e1c239b7f030fc25d69, diff empty, current all match. All nineteen additional v7 protected current checks match. Frozen state inventories stable. Static driver collects into a new baseline and external snapshots, with no production claim/release/flock/model behavior. |
| Process/resource evidence (§9) | Actual launcher uses project Python, empty CUDA_VISIBLE_DEVICES and external scheduler. Raw ps line530 observes only one project-Python collector PID1579841; system/other-environment daemons are distinct. Independent audit PID1585239 reserved external slot0/FD10; metadata and finalizer ran sequentially, max one reviewer project Python. No full-window global process monitor is asserted. |
| Authorization/status separation (§5/§7) | Three historical proof fields false; historical OPEN_FAIL_IRRECOVERABLE_GAP, prospective RECORDED_UNREVIEWED, forward NOT_GRANTED, execution NO_GO. No automatic closure/GO. |

### Exact restored source identities

The independent evidence JSON records absolute paths, recorded source inodes, source SHA and both archive locations. Corresponding paths/digests are:

| Inventory identity | Source/archive SHA-256 |
|---|---|
| `m3oe:MDMTRec/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `m3oe:MDMTRec/basic/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `m3oe:MDMTRec/basic/activation.py` | `152c5909e966055f83b7ec49de7634c853fbae8b029fbd37759371c14c921046` |
| `m3oe:MDMTRec/basic/features.py` | `116025a45517021bfe86447b8229e59cf04feb665549544ce30173d04421a2b0` |
| `m3oe:MDMTRec/basic/initializers.py` | `a9a6a2923eebeca7f541ffaddbccf2671a42e7f9b6a1ec8d38ba2872c53f6196` |
| `m3oe:MDMTRec/basic/layers.py` | `e3eb8db91ff0aaf79c8e389107df5f64841992fab02a18b29c37bb446b22763a` |
| `m3oe:MDMTRec/models/multi_domain/__init__.py` | `e11c58c743dd3125f0a85eeb89c0266706b4716b10700eb99ed1f18d19cb23e8` |
| `m3oe:MDMTRec/models/multi_domain/model.py` | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |
| `m3oe:MDMTRec/utils/__init__.py` | `13c62ae9d561769c4597c2f279bf03ab25363ad64ada1e77af58c4f7979cc7a3` |
| `m3oe:MDMTRec/utils/data.py` | `71d08bd9487afc702473d8813449d21ffd85c223ed05457791486b6df708501a` |
| `exp:history_exp/first/p7_models.py` | `7b727c4c933e93860eb9ce1dd57d58f4c40adbf3975d2d7ca6dd76afb8b6e839` |
| `probe:probe.py` | `5b265e759aa35aa2328a0b6b0a45ffc3877d697fe975b9325699ed6da87d9a5d` |

## 3. Bugs/deviations found

No remaining CRITICAL, MAJOR or MINOR implementation blocker identified for accepting this exact bounded baseline. The driver now declares m3oe/probe roots, includes the history model-source exception and restores the missing script. Its broad history-tree exclusion is interpreted with these exact explicit source exceptions; data/sample trees remain excluded. No full-repository preservation claim is made.

## 4. Corrections made

None to collector, production, protected files, attempt03, prior attempts or prior reports. Only new independent external audit outputs and this report/evidence were written. No collection, E2E or test was rerun. No torch/model imports, checkpoint deserialization, model construction/forward, sample reads or CUDA math occurred.

## 5. Remaining risks and limits

Collection window `2026-10-13T21:36:04.045932Z` to `2026-10-13T21:44:02.491891Z` (approximately 478.446 seconds). Equal inventories and fresh pre-call snapshots support the bounded observations. They cannot prove continuous absence of transient owners/locks or mutations outside selected paths, and cannot reconstruct missing pre-incident complete inventory, original runner preimage/diff or original shared-lock diagnostics. A lock snapshot is not a deadlock/root-cause determination. Static code has no production flock acquisition; byte equality alone would not establish that claim.

Raw audit checks all archived payload hashes without deserializing checkpoints. Symlink observations are retained as target-byte evidence; targets were not used as replacement ordinary sources. Ancestor/inode checks reject archive traversal/alias risks observed at audit time, not malicious future filesystem changes. The external launch manifest/raw outputs are currently available under /tmp; this decision binds their present hashes and the retained baseline bytes, and does not promise future /tmp retention.

The incident remains disclosed and the malformed production peer log remains unchanged. Clean synthetic fixture logs and focused/E2E qualification cases support only their actual tested conditions; they do not prove all DESIGN §§8/10 criteria, scientific semantics or original production entrypoint continuity. No scientific performance conclusion is drawn.

## 6. Whether experiments are safe to run

**This baseline PASS permits no experiment or production action.** It establishes only ACCEPTED_BOUNDED for attempt03 evidence. B6 historical OPEN/FAIL, NOT_GRANTED and NO_GO remain mandatory. Forward disposition, real log continuity, complete qualification closure, staged graph GO and exact scientific run authorization each require separate approved review.

### Evidence bindings

- Evidence JSON: `experiments/reports/stage_F0_v15_B6_postincident_baseline_attempt03_review_v1_evidence.json`, SHA `c531ea2dc615a87f885703e12116c0f1890b88c1a8a492f2ef72624d2f51e2d1`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/audit.py`, SHA `f171bc4b94076e4047826642c0c341c972ac965eb3f0c2c853165b8ef225ec73`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/audit.stdout.txt`, SHA `6ad850d4e6a0ff57a4e562b304e1eb22b47b203502fa44128549da56ee83c245`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/audit.stderr.txt`, SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/audit.json`, SHA `a7e9c7c012f6cd46a36f0919fce7f11ab7597ec3fc1ba7a51f03187e5a0cf76d`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/process_record.json`, SHA `a90fe7f02b1c0f660618647199cb0cab8381c2b0e06791a70d2d5952f1d5c835`.
- Independent raw: `/tmp/f0-B6-baseline03-review-20261014-a/metadata_audit.json`, SHA `34b3e8ab20d797781e905b0f16fc22b25c72b3973903b5e22bf5a79dfa3c14b1`.
