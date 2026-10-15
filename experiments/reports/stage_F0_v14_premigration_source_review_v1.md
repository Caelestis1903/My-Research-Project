# F0 v1.4 independent pre-migration source review — v1

Decision: **GO FOR THE DESIGNED ADMINISTRATIVE MIGRATION ONLY**.

Reviewer: independent reviewer agent. Live source verification completed at `2026-10-13T04:32:08Z`; GPU observations refreshed at `2026-10-13T04:32:30Z`.

This is a distinct source-snapshot review following implementation PASS. It does not release hold, authorize claim/training, permit retry/recovery/new attempt/checkpoint load, or authorize formal/test evaluation. Production migration and release have not been performed.

## Frozen identities independently checked

| Input | Actual SHA-256 |
|---|---|
| v1.4 design | `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9` |
| Final coder report | `aa8c709dc9b349340f9613181e28d4e8c72b5301b9ae40cb8262ce0146c3b080` |
| Independent implementation PASS report v4 | `5540595752ad61ef519f15979fdb4b1cb0ddf5397595de9f7082122596652410` |
| Source snapshot file | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |
| Source snapshot embedded digest | `3df82eb0ab75922503fe0b82a971e06c8a815959287a27046adcd879936e5b5f` |
| Current v1.4 module | `10d81444afa55c13c8aab09605da0ea9840354e0be1eff9d2b4cc7738bd3ae16` |
| Expanded v1.4 fixture | `67fbeb1198bbb57455d9025a7fbb185399e911bbbd57d832c08a07f7e675e038` |
| Complete current control inventory | `fd4733a4eaa967398e50b21f728f1fcaa97cc00c199ee72826349b49f0f255c8` |
| Current original execution gate | `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7` |
| Frozen scientific search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Fixed 288-entry search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Current original active hold | `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3` |
| Historical v1.3 immutable runtime projection | `43d22e38332d301667bd1b6598044dd360597faec8557b47a9bbf3fb97490251` |

Snapshot path: `experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json`. Its schema is `F0-POSTABNORMAL-SOURCE-SNAPSHOT-v1.4`, creation time is `2026-10-13T04:29:19Z`, and both supplied file/embedded digests match independently computed values.

The actual reviewed validator `validate_source_snapshot(snapshot, runs_root=RUNS)` passed with its default strict pre-release fresh policy. The actual `_verify_frozen_source_pointers(snapshot, root=RUNS)` passed without mutable-pointer exceptions. Every frozen raw source record matches its current disk bytes, including original hold/gate/release, old audit/correction/migration/prelaunch, and historical design/coder/reviewer reports. Current runtime inventory was recomputed from all production control inputs and compared with the approved revision's module/test/coder identities.

## Sets, queue baseline and inventories

| Item | Verified identity / count |
|---|---|
| H49 exact sorted list | 49; `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| H51 exact sorted list | 51; `e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86` |
| C97 exact completed list | 97; `06cba20a42bfc3f8dfc20b4df68ac7dc774c66bb34972ab4158a7227db4d7fce` |
| C97 full file inventory | 1,068 files; `462859f35bb88a5f7c668aec4e11359ea04799e7fe88da3a8c71a3a71288242d` |
| F140 exact untouched fresh list | 140; `78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b` |
| Current new-incident full inventory | 27 items; `66aaeb57a65dc512fc45e7db320416609f014cbcf62bf6942bd77c3a3f2859d1` |

Actual fixed search counts are exactly 97 VALIDATION_COMPLETE, 49 RECOVERY_BLOCKED, 140 UNCLAIMED, 1 RUNNING and 1 INTERRUPTED. The six nonsearch RESOURCE_PREFLIGHT_COMPLETE statuses were separately counted and excluded. All historical H49 statuses/projections match the frozen source. The two new incident statuses retain their original RUNNING/INTERRUPTED projections and are the sole permitted administrative targets.

All F140 members have no status, run/attempt directory or canonical lock. Full completed-tree and incident inventories were recomputed from current files, detecting changed/missing/extra files; they equal the snapshot. Historical v1.3 source/receipt/status/asset graph passes its separate validator without being used to authorize current launch.

Both incident failure audits, manifests, statuses, stdout, stderr and phase logs match the fixed run/nonce identities and frozen hashes. The peer INTERRUPTED receipt remains at SHA `cced71d99bf329a963916a38b13dd48d0f3818c62faaad1e157a00d6f9b45f1d`; the SIGSEGV receipt remains absent. The reviewed snapshot binds the original peer epoch32/checkpoint evidence. No new attempt/recovery/retry or synthesized SIGSEGV receipt was observed.

## Live execution and GPU checks

No live F0 supervisor, trajectory child, runner or manual first-epoch Python process was found in `/proc`. No current RUNNING/CLAIMED status has a live owner PID. The single RUNNING row is the already-frozen deceased SIGSEGV trajectory; it is an administrative migration target rather than an active claim.

All **122 existing production locks** were probed with nonblocking flock on read-only file descriptors and released immediately: execution_state, worker, GPU and canonical locks were free. The reviewer did not create, truncate, delete or rewrite any production lock. The verification used one task Python process through the project five-slot wrapper; no worker or training Python was launched. No v1.4 migration/release ledger or principal release artifact exists yet. Hold remains active with the exact original source bytes.

| GPU | UUID | Used / total MiB | Used fraction | Decision |
|---|---|---:|---:|---|
| 0 | `GPU-4ac4b341-3045-1093-fec3-492516863e5a` | 4,222 / 24,576 | 0.17179361979166666 | <0.90 |
| 1 | `GPU-2a59fb49-6bef-5320-74ab-ffb24d056908` | 15 / 24,576 | 0.0006103515625 | <0.90 |

Telemetry was readable. GPU utilization was 0% in this observation, but utilization and the existing external Qwen process do not affect the memory eligibility decision and were not stopped. These observations do not replace the required later live prelaunch/spawn/claim checks.

## Review outcome and scope

No source drift or pre-migration blocking discrepancy was found. All independent source checks passed. This GO is bound to the exact snapshot and code/report/inventory identities above. Structured independent review metadata must bind those same identities before the reviewed administrative migration accepts it; use a separate pending-migration gate while preserving the original frozen source gate bytes until the specified later pointer transition.

The runner may perform only the designed two-target administrative transition under the reviewed locks and durable PREPARED/transition/hold/COMMITTED procedure, with hold active and all scientific/attempt/checkpoint/receipt assets preserved. It must recheck the source live inside that transaction. Any drift is NO-GO.

After migration, an independent postmigration GO is still required. Fresh search needs the separate current prelaunch and new identity-bound principal release. Native SIGSEGV cause remains UNKNOWN, and no scientific results were analyzed.
