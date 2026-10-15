# REVIEW REPORT — seventeen historical incident asset inputs

## 1. Verdict and authorization limit

**PASS — exact seventeen-file scope/identity only.** This report grants no copy, setup, retry, test, qualification RUN, production release/claim, staged GO or scientific RUN authority. **Seven `.pt` files are checkpoint/model artifacts; explicit user authorization remains required for every proposed `.pt` copy. Scope PASS cannot supply that authorization or extend prior single-checkpoint authorization.**

Q3 remains NOT CLOSED; B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain. Hashing historical training artifacts supplies no research performance conclusion.

## 2. Necessity and frozen inventory

Attempt08 raw traceback passes through v1.5 `build_source_snapshot` :915, v1.4 historical validation :533, v1.3 historical graph validation and `_incident_asset_inventory` :273 into `_tree_files` :235, which rejects the absent historical run `7fd2006bdc8f1817a6f7/attempts/attempt-00` tree. This is a missing historical validation fixture, before Q3 fault testing.

Actual `_incident_asset_inventory` hashes both historical TARGET_IDS attempt-00 trees, v4-bound incident logs and the peer failure audit uniquely bound by the hold event. `_inventory_for_paths` requires ordinary non-symlink files and records path, size and SHA. Historical validator :1024–1030 requires equality with frozen PREPARED items and PREPARED/COMMITTED inventory digests.

Independently verified PREPARED contains all seventeen requested paths with exact sizes/hashes. The full frozen incident inventory contains 27 entries; this exact proposed set equals all 17 entries absent in attempt08 clone. Reconstructed using the runtime's sorted paths and insertion-ordered `path,size,sha256` compact UTF-8 JSON, its digest is exactly `10fc3b31d3f462a89f9204db14f7e97a7bce2ad733279bf2724d3311e3e6e15e`, matching PREPARED, COMMITTED and validator expectation. Using decoded ledger object key order would not reproduce this digest; the runtime representation matters.

The seventeenth file is independently bound by one hold event: WORKER_FAILURE_HOLD, canonical run `1f04c7fb623d18f940ac`, worker0, nonce `c87c8e65bc7e4b73bed880465ff2dffb`, exact production failure path and digest in the table below. Inclusion is necessary for frozen inventory equality; no additional path is approved here.

All seventeen source paths are absent from B6 baseline03 inventory, attempt08 clone disk and clone input manifest. They must be identified as live input supplements, not B6 archived payloads. No historical inventory validator or test was executed in this review; conclusions derive from static source, archived JSON and read-only byte hashing.

## 3. Exact proposed set

Resolve these prefixes literally:

- S = `experiments/stage_F0_outputs/runs/7fd2006bdc8f1817a6f7/attempts/attempt-00/`
- P = `experiments/stage_F0_outputs/runs/1f04c7fb623d18f940ac/attempts/attempt-00/`
- L = `experiments/stage_F0_outputs/runs/worker_logs/`

| ID | Prefix + filename | Exact SHA-256 | Artifact class |
|---|---|---|---|
| 1 | S initialization_audit.json | `f0e9383f4af3c436f14c35ed50ca890d30afd955d5db69093e48ccda9bad3550` | JSON |
| 2 | S metrics_validation.jsonl | `94076017cd9c533a9d4891496c95ace75cc57a6c15e92591ec2957604772792c` | JSONL |
| 3 | S parameter_manifest.json | `fe2f9145d09acf7d217e86f457ccd36c40b47498c1810417da17b836cc8074d2` | JSON |
| 4 | S resource_epoch_metrics.jsonl | `fe06d69a1b0f876b66bc1797c9dda2b98939be670dcb27362a4476ae1f6d5470` | JSONL |
| 5 | S resume_epoch_011.pt | `336c1a13b68de659e377dda0535d8d40e7a7e995b90c8af4df9825a8f612b189` | **Checkpoint/model: user authorization required** |
| 6 | S resume_epoch_012.pt | `9e806f1169cf71b6c7900b3707a63bfd1af2ae2c59fef661d1763a7df77c506f` | **Checkpoint/model: user authorization required** |
| 7 | S resume_epoch_013.pt | `2e669175eb2752ff27cc73202687167e98799d36e2495f25e76d4543f6b0c79f` | **Checkpoint/model: user authorization required** |
| 8 | S run_manifest.json | `e304b73005ff339b63b0b2ad70c4830b7df9550bce2acc2710ff250690ee0197` | JSON |
| 9 | S selected_checkpoint.pt | `b5e19de88d46b5c9aebbb235997862111457e239509b2fde9b82b2a6ceca4f4b` | **Checkpoint/model: user authorization required** |
| 10 | P initialization_audit.json | `437f414f7c0967a0948d73b3ff2e230b4eb4cc30a918f87ba4353304724b52b6` | JSON |
| 11 | P metrics_validation.jsonl | `55033d347603eff5595e6a3a01a4f5f7c57430b3e2ad93d1721b2a84a94e392c` | JSONL |
| 12 | P parameter_manifest.json | `fe2f9145d09acf7d217e86f457ccd36c40b47498c1810417da17b836cc8074d2` | JSON |
| 13 | P resource_epoch_metrics.jsonl | `4fc23c93d49872f301ff68e8a72e1bf6a61d67c6af9398118b1f6b15ada8c06a` | JSONL |
| 14 | P resume_epoch_015.pt | `257eb8449402266286c912753971b90ad8744fd69553b7d5cab948152ebd4104` | **Checkpoint/model: user authorization required** |
| 15 | P resume_epoch_016.pt | `dafe35946ec485d081d93b490e4a27e639ec804ac20a8eb51fa28965114eb6ae` | **Checkpoint/model: user authorization required** |
| 16 | P selected_checkpoint.pt | `19f959dcdd7dde8e0a97478f0a5d419012c58692c8ac5c9980c4b61f8b97a55e` | **Checkpoint/model: user authorization required** |
| 17 | L worker_0_failure_1791819201_c87c8e65bc7e4b73bed880465ff2dffb.json | `ce1f5d4e8d68f2a304139c790982b34cef58ac971c590a852ae7bcf95bcefb9d` | JSON failure audit |

Equal hashes for parameter manifests 3/12 do not merge their path identities; each needs a separate manifest/copy record.

## 4. Independent identity and absence observations

Two `O_RDONLY|O_NOFOLLOW` raw-byte reads per file used mandated project Python through the task-slot wrapper and an external scheduler, during `2026-10-14T03:45:05.590212+00:00` to `2026-10-14T03:45:06.013784+00:00`. Every first and second digest matched its exact expected hash above. For each file all lstat-before, opened-FD fstat-before/after and lstat-after observations across both reads were identical. All were ordinary regular files, dev 2049, mode 0664; IDs 1–16 nlink 5, ID17 nlink 1.

| ID | Inode | Bytes | mtime_ns |
|---|---|---|---|
| 1 | 54657778 | 374 | 1791819131713094613 |
| 2 | 54657799 | 198570 | 1791819197276038336 |
| 3 | 54657779 | 4030 | 1791819131781097652 |
| 4 | 54657797 | 5004 | 1791819197276038336 |
| 5 | 54657794 | 17342653 | 1791819188031621635 |
| 6 | 54657802 | 17358269 | 1791819192639829290 |
| 7 | 54657796 | 17373885 | 1791819197272038157 |
| 8 | 54657783 | 11215 | 1791819197276038336 |
| 9 | 54657770 | 17250316 | 1791819159822353331 |
| 10 | 54657791 | 374 | 1791819146781768731 |
| 11 | 54657800 | 59212 | 1791819200232171696 |
| 12 | 54657792 | 4030 | 1791819146861772315 |
| 13 | 54657788 | 6585 | 1791819200236171877 |
| 14 | 54657793 | 17202621 | 1791819195087939659 |
| 15 | 54657784 | 17205245 | 1791819197652055297 |
| 16 | 54657776 | 17196236 | 1791819186843568121 |
| 17 | 54136065 | 5948 | 1791819201832243901 |

Stable ctime_ns values: IDs 1–9 `1791933147192074987`; IDs 10–16 `1791933146964063596`; ID17 `1791932992976283810`.

All seventeen paths are outside F140 expected-absence controls; all 560 F140 paths remain absent in production and attempt08 clone. Historical SIGSEGV receipt `trajectory_worker1_2_6e9216332b2e4a488e8c5b7c2ed86922.receipt.json` and current incident receipt `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.receipt.json` remain absent in both. No receipt inclusion or canonical F140 creation is approved here.

The seven `.pt` files were read **only as raw bytes for SHA-256**; torch was not imported, no checkpoint was deserialized, no model was loaded and no inference/training took place. Raw-byte verification does not authorize copying/model loading or select any resume/selected checkpoint for future training.

## 5. Conditional controls, risks and corrections

No blocking exact-set scope/identity defect found. **MINOR / source-stability risk:** writable sources and sixteen shared inode aliases are stable only within the recorded window. Copy-time revalidation is mandatory; sources and aliases must not be modified/chmodded or used as clone destinations.

Only after separate applicable authorization, including explicit user authorization for all seven `.pt` copies:

- Preserve failed/prior roots and build a fresh clone. Bind this report's exact path/hash; record each exact path/digest separately as `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` under complete `F0-Q3-CLONE-INPUT-PROVENANCE-v1`. Preserve all earlier reviewed bindings/supplements. No content-hash path deduplication.
- Record source/destination paths, UTC, expected hash and source lstat/open-FD fstat before/after matching reads and after copy. Require stable kind/dev/inode/size/mode/nlink/mtime/ctime; reject symlink/hash/stat mismatch without changing expected evidence.
- Use `O_EXCL` ordinary byte copy, no hardlink/reflink. Require regular non-symlink distinct-inode destination, nlink 1, exact size/hash readback and mode/stat records; fsync file and parent directory. Checkpoint handling, if authorized, remains byte-only unless additional model-loading authority exists.
- Verify complete updated clone inventory, frozen incident path/size/hash equality with proper digest representation, all prior source/design/review/supplement identities, 19 frozen dependencies, protected before/after hashes, both receipt absences and all 560 F140 absences. Archive raw commands/exits/manifests externally using mandated project Python/reserved slots. Preserve any later failure; no same-attempt retry is granted.

The seventeen entries cover all currently absent paths in this specific 27-item frozen incident inventory; they do not establish complete downstream source-snapshot fixtures or Q3 execution readiness. No implementation/protected path changed, input copied, test run or model loaded. Only this explicitly requested report was added.

## 6. Evidence identities and run safety

Attempt08 evidence root: `/tmp/f0-v15-Q3-run-20261014-h`.

| Evidence | SHA-256 |
|---|---|
| archive/preflight.stderr.raw | `090569c41bb0c3c2a1bcc2b421c749ad91d02a2013197a81f13cfbf6086c2adf` |
| archive/preflight.stdout.raw | `a14743d227b2b5232714edae9e39ea128a785b0d803594df73d154463dcf73d4` |
| clone_input_manifest.json | `6825f975eceb16faf2ad357840ef1f168902448e3aaeeb21cb87a5d6cc9a9b59` |
| clone historical postabnormal_sigsegv_audit_v5_migration.jsonl | `2608edd595fcfdc46791eeb73b3aa42b1c82d75b265c7d3abb16146884284f08` |

**Scope/identity PASS grants no copy/setup/test/RUN authority. Explicit user authorization remains required for each of the seven checkpoint/model artifact copies.**
