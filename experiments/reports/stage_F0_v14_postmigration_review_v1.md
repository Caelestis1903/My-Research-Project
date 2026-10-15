# F0 v1.4 independent postmigration review — v1

Decision: **GO FOR ADMINISTRATIVE MIGRATION CONSISTENCY ONLY**.

Independent live disk verification completed at `2026-10-13T04:45:38Z`; GPU telemetry refreshed at `2026-10-13T04:46:03Z`. This review did not invoke migration/reentry, release, claim or training and did not alter production state. It validates the completed administrative transaction, not search launch.

## Transaction and frozen identity

| Item | Verified value |
|---|---|
| Transaction ID | `f0-v14-h51-admin-50427c2469d942638fc8e13925e2dc6b` |
| Migration ledger SHA | `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30` |
| Source snapshot file SHA | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |
| Pending migration gate SHA | `14df05478e1f75bc2f2377fe5305b776e0fb659b33453c4b9d247fb652757a12` |
| Current active H51 hold SHA | `29597e3ca48a14137c1214874c3281913436a8683174c4d776504c23a663fb86` |
| Structured independent review SHA | `57a3024aea4a14dfab029ad3b670ca175e31a54379277082d042402c3c5fac44` |
| Actual control inventory SHA | `fd4733a4eaa967398e50b21f728f1fcaa97cc00c199ee72826349b49f0f255c8` |
| Original production execution-gate pointer SHA | `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7` |
| Fixed search queue SHA | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |

The ledger was read directly from disk. It contains exactly five complete records in this order: PREPARED; worker0 target STATUS_TRANSITION_DURABLE; peer target STATUS_TRANSITION_DURABLE; H51_HOLD_DURABLE; COMMITTED. All records have the expected transaction ID/schema. PREPARED and COMMITTED bind the exact source snapshot and exact H51 list/digest. PREPARED binds the approved structured reviewer path/hash. The durable-hold and COMMITTED hashes match the actual current active hold. There is one PREPARED and one COMMITTED, with no duplicate transaction.

PREPARED records time `2026-10-13T04:41:39Z`; COMMITTED records `2026-10-13T04:41:45Z`. The independent review does not claim to reconstruct past fsync syscalls: it verifies current persisted bytes, durable ledger markers and projections against the independently reviewed transaction implementation.

## Exact administrative projections and preserved assets

| Target | Original status SHA | Actual migrated status SHA |
|---|---|---|
| `0d2e6555b049c012c0e5` | `809b4dd83609f38df791afb1581f873c801aa3d408b4f51ff1d60c4dc00d4ca0` | `8522078c1db2f76ba0ffc90fbd0f803d24e9c17d8be0d669f10faa94861a5496` |
| `a80a232a3c0068a48a08` | `5dc0c4c0d656c7cc3f71772852d7b5508d9d9daa005c4d88c596dc70c5aaf1d5` | `ac361e964095f1e65bdba8f6164ee5caa94c95576190bb1f158132c8cb9ce470` |

For both targets, current status bytes equal the reviewed `held_status_projection` reconstructed from the raw original source projection using the production atomic JSON serialization. Before/after hashes, target IDs and file/directory-fsynced flags in both transition records match disk; COMMITTED's status hashes match them. Canonical JSON/full hash and all original fields outside the approved administrative changes are preserved.

The actual active hold equals the complete `_expected_h51_hold` projection reconstructed from frozen original bytes plus this transaction's single H51 event. Prior events and historical fields remain unchanged. Exact H51 is 51 IDs with SHA `e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86`; all 51 current statuses are RECOVERY_BLOCKED. Original H49 statuses and their metadata remain byte-identical to the source snapshot.

Recomputed inventories remain unchanged:

- C97: exact 97 completed IDs, 1,068 files, inventory SHA `462859f35bb88a5f7c668aec4e11359ea04799e7fe88da3a8c71a3a71288242d`.
- New incident evidence: 27 items, inventory SHA `66aaeb57a65dc512fc45e7db320416609f014cbcf62bf6942bd77c3a3f2859d1`.
- F140: exact 140 untouched IDs, list SHA `78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b`; no status/run directory/canonical lock for any member.

Both incident attempt directories still contain only attempt-00; recovery_count remains 0 and resume_checkpoint remains null. Attempt manifests, checkpoints, metrics, nonce logs and the original peer receipt remain unchanged. The SIGSEGV receipt remains absent. No retry/recovery/new attempt, receipt synthesis or training was observed. Historical H49 source/asset/release/review graph remains unchanged and validates separately.

## Transparent count-assertion issue and independent corrected count

The runner reported that its **initial postcommit reporting code exited 1** because it expected UNCLAIMED entries inside `_queue_and_statuses`. It subsequently reported corrected explicit verification passing. No separate original terminal log/report was supplied; this chronology is attributed to the runner rather than asserted as independently observed execution history.

The count semantics and completed transaction were independently checked directly from disk. `_queue_and_statuses` returns existing status files only. For the fixed search queue it returns **148** files: 97 VALIDATION_COMPLETE and 51 RECOVERY_BLOCKED. Untouched fresh keys intentionally have no status file.

Therefore:

```text
fixed queue IDs                     = 288
existing search status IDs          = 148 = 97 + 51
queue IDs without any status file   = 288 - 148 = 140
```

That exact complement equals the frozen F140 list; every member satisfies fresh eligibility. Recomputing state for each of the 288 queue rows, using implicit UNCLAIMED for missing status, gives **97 VALIDATION_COMPLETE / 51 RECOVERY_BLOCKED / 140 UNCLAIMED / 0 RUNNING / 0 CLAIMED / 0 INTERRUPTED**. Six nonsearch RESOURCE_PREFLIGHT_COMPLETE status files are separately excluded. The verified ledger is complete and the count issue does not represent a failed or partial administrative transaction. The reviewer did not rerun migration or rewrite ledger to repair a reporting assertion.

## Actual validators, locks, processes and GPU

The current actual strict source validator passed. The actual frozen-source-pointer validator passed with this transaction's exact expected H51 hold projection; it accepted no unrelated pointer changes. The actual structured implementation-review validator passed against the original snapshot file hash. Pending migration gate/control/scientific amendment identities match the reviewed current inventory and queue. The original production execution-gate pointer remains unchanged.

The current launch/hold entry point still rejects production launch, as expected: active hold is retained and no new launch graph/release exists. Full release/prelaunch validation is not represented as completed by this administrative-only review.

All **122 existing production locks** were independently probed using read-only file descriptors and nonblocking flock, then immediately released; none was busy. There are no live F0 Python supervisor/child/runner processes and no active status claim. The audit used one counted project Python process through the five-slot wrapper. No new v1.4 prelaunch audit, release ledger or principal release artifact exists.

| GPU | UUID | Used / total MiB | Used fraction |
|---|---|---:|---:|
| 0 | `GPU-4ac4b341-3045-1093-fec3-492516863e5a` | 4,222 / 24,576 | 0.17179361979166666 |
| 1 | `GPU-2a59fb49-6bef-5320-74ab-ffb24d056908` | 15 / 24,576 | 0.0006103515625 |

Both readable memory fractions are strictly below 0.90. External GPU use/utilization was observed without stopping other processes; utilization is not an eligibility condition. Later live prelaunch/spawn/claim checks remain necessary.

## GO boundary

No administrative consistency defect was found. This GO binds the exact source snapshot, transaction ledger, active hold, H51 and preserved inventories above. It is **not permission to release hold or start training**. The next gate must separately bind current postmigration review, complete runtime inventory, source graph, fresh-only scope and real-time prelaunch evidence before a new principal identity-bound release can proceed. Resume/retry/new attempt/checkpoint load/formal/test remain unauthorized.

Native SIGSEGV cause remains UNKNOWN. No AUC/loss or scientific effectiveness analysis was performed.
