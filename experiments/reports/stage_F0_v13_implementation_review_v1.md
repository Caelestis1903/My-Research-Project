# F0 v1.3 independent implementation review — submitted snapshot

Date: 2026-10-13 (Asia/Shanghai). Scope: approved DESIGN §14, actual implementation/test files, coder report and production read-only artifacts. No Python, migration, training, signals, production status/hold/gate writes, checkpoint loading, or research analysis were performed. This new report is the sole reviewer write.

## 1. Verdict

**FAIL.** Seven MAJOR blocking deviations remain. Return to Coder, retain pending authorization and active hold, and obtain a new independent review before administrative migration or search execution. Existing pidfd, fresh-child, duplicate-signal, lock-held receipt and process-limit controls do not regress in the inspected snapshot. Their preservation does not approve the new migration/runtime validation.

## 2. Independently verified hashes and production state

Actual source bytes and compact JSON aggregates were checked with shell/Perl; project Python was not imported.

| Identity | SHA-256 |
|---|---|
| v1.3 design | `163f2fbc181712e0186cdb651f8738bb72a27f7a65edf92e4e554e677aac62e7` |
| Coder implementation report | `982def77b99dabe5ecb8c53ed314374258c65b02af52fb4e44557167125ff1ad` |
| Pending execution gate | `5b2848a608d3dc4a196a099f5251618d6fe96b888bc6bcb91e5ce9369580d571` |
| Actual 16-file control aggregate | `6ecd93dcb74bff2af33e701a73b42e4d92a7eba3ea28797756e64f6b6c370bf2` |
| Actual 28-file runtime aggregate | `6e602526cb5b4ea3a31e28870bbadf1de9b9610f90285e4de31ea54a2d05879d` |
| Active hold, unchanged | `e9461efcc99e49bef4b813a3f3cb1c67720bdcd172470df2a73a9e7caca400b9` |
| Original v4 audit, unchanged | `a50162f3016841257e8c098cc0040c9ff2be4edb30073c4c2fb42f8ed416f839` |
| Independent H47 | `ccbf64945396b542ee2746f4bab37d7d2fcdaea49b00817f096facec45f7f8bf` |
| Independent H49 | `60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e` |
| Actual completed asset inventory (573 files / 52 keys) | `94039db9a66bfe20e7a4bb2d96b9249c145aa2fcf23fb53ca71bc225ffc718ae` |

All 16 individual hashes matched the gate; both aggregates matched the report. Supervisor, child and F0Runner SHA remain respectively `2689e7cc0b9f364afb2b96569615316e90b7381e775d4e5cb42000e8333cbab8`, `4343e928c266869fb9723509373da3dfe75c30ffe4de85a241a38e9a708f86e5`, and `5ad73c898063fb6abddb14251fc98201c9e5cf7cdbe9d7845aab6e74c1602599`, identical to the prior approved signal/receipt snapshot.

Independent set reconstruction used the historical 45-key source plus the prior v3 audit's two canonical claims for H47, then original v4's two new run IDs for H49. It did not treat caller/new gate as a source. The fixed H47 constant matched exactly and hashes matched DESIGN. Current gate H49 contains the complete independent union, with no missing/extra item.

Production migration has **not** occurred. The actual 288-key ledger is `52 VALIDATION_COMPLETE + 47 RECOVERY_BLOCKED + 187 UNCLAIMED + 1 RUNNING + 1 INTERRUPTED`. All old H47 remain blocked. SIGSEGV target `7fd2006bdc8f1817a6f7` is RUNNING; peer `1f04c7fb623d18f940ac` is INTERRUPTED. Hold remains active; launch/formal/test/recovery/automatic-retry authorization flags are false and reviewer verdict is PENDING.

Independent byte hashes matched DESIGN for target queue statuses (`74cbabc600e19e4e977c3b38d3068419c72c3be0b678cea5857a1867ec66dbbe`, `61921430b4975ba67720efd19fcb24a6ae4debb5f232f56d536e0e568ad41478`), peer receipt (`3a3b86cd54b0685ec09506a32706355a26c3465ed41d872c06e641d94bcbacf7`), SIGSEGV/peer manifests (`e304b73005ff339b63b0b2ad70c4830b7df9550bce2acc2710ff250690ee0197`, `2e85d70d0d97b144fd26cba06be61f9ce3d86bb3879839adde9c866c6377f1d6`) and epoch13/17 checkpoints (`2e669175eb2752ff27cc73202687167e98799d36e2495f25e76d4543f6b0c79f`, `8843afe2a382b47b7a920986ffbfc0cf996a73b0ab8822bfd138760b2517597c`). Hashing is raw read-only, not model/checkpoint deserialization.

## 3. Requirement verification

- **§14.5 independent exact H49: PASS for fixed set and gate/caller rejection.** Old set is retained and only the two specified IDs added. Claim enforces H49 at its execution-state serialization boundary (`stage_F0_queue.py:416–427`). Release also requires exact H49 and hash. Source/caller/audit negative fixtures exist, but runtime/release references remain incomplete (MAJOR-07).
- **§14.6 actual correction facts: PARTIAL.** `validate_provenance:252–370` checks frozen original v4, actual immutable peer receipt and its historical canonical/GPU owner UUID, original peer status UUID, peer manifest/epoch17 checkpoint, failure audit and process-start identity. It never replaces historical GPU identity with the current free lock owner. Generated correction carries original/verified values, methods and sources. Consumption does not fully bind those facts (MAJOR-06), and other immutable incident assets are omitted (MAJOR-04).
- **§14.7 administrative migration: PARTIAL.** Execution-state, worker, GPU and both canonical locks are acquired; original target-status raw bytes are embedded in durable PREPARED before status mutation. Writes are status-only plus active-hold extension, without torch.load/resume/new-attempt paths. COMMITTED follows post-state/asset checks. Old hold byte preservation and retry/status durability need correction (MAJOR-03/05).
- **§14.4/14.7 original 52 assets: PASS current baseline, incomplete broader incident coverage.** Reviewer rebuilt the actual 573-file sorted path/size/hash inventory and matched the frozen digest. Migration snapshots and compares that inventory. Future runtime incorrectly redefines the baseline using every current COMPLETE key (MAJOR-01).
- **§14.8 negative/fault tests: PARTIAL.** Temporary migration fixtures cover ten named failures and authorized idempotent continuation; source receipt drift, audit H49 substitution, caller/gate/release malformed sets and existing signal/receipt tests exist. They omit legal post-release progress and several source/status/reference drift cases below. Successful synthetic snapshots do not verify storage crash durability.
- **Prior pidfd/receipt/STOP/CONT/PDEATHSIG/slots: PASS by preserved code hashes and test inspection.** No new algorithm, loss, selection, model, seed or numerical checkpoint change was introduced by this control code.
- **§14.10 current NO-GO: PASS (closed).** Missing correction/COMMITTED evidence and pending gate/active hold block production. The future released/ongoing runtime path does not yet satisfy DESIGN (MAJOR-01/02/07).

## 4. Blocking deviations and required corrections

All reproductions below are **source-level logical walkthroughs, not executed**. No production artifact was changed.

### MAJOR-01 — migration's fixed counts incorrectly enforced throughout later search

`stage_F0_migration.py:549–551` requires every runtime check forever to see exactly `52 COMPLETE +49 BLOCKED+187 UNCLAIMED`. `_asset_inventory:161–187` derives its protected set from all currently complete rows and requires exactly 52. Gate, released hold and serialized claim call this validator.

After one lawful fresh claim, UNCLAIMED becomes CLAIMED/RUNNING; another worker/new child immediately fails validation. After a lawful completion, the 53rd COMPLETE also fails. This blocks the specified two-worker, multi-trajectory fresh search despite correct H49 protection.

**Fix:** keep strict pre/post administrative counts at migration commit; independently freeze the original 52-key asset set and maximum original 187 fresh candidates. Subsequent runtime must preserve original H49/52 identities/assets while validating authorized fresh progress and active claims, rather than requiring the original ledger forever. Add positive two-worker/claimed/running/completed-progress fixtures and negative unauthorized-key/source drift cases.

### MAJOR-02 — approved release has no state accepted by both gate and claim

`stage_F0_worker_control.py:217–222` calls `validate_h49_runtime` with default `require_active_hold=True`. `assert_no_global_hold:352–361` requires `hold.active=False` plus a valid release. Thus active hold passes the former but fails claim; inactive, correctly released hold fails the former. No valid released launch state exists.

**Fix:** distinguish the migration's active-hold requirement from runtime consuming an explicit approved release, retaining identity/hash/authorization bindings for the committed active-hold source and current release. Add an end-to-end production-validation fixture covering active refusal, correct inactive release acceptance, stale/incorrect release refusal, and pre-child/claim checks.

### MAJOR-03 — original global-hold bytes are not preserved before overwrite

`_make_prepared:614–643` records only `original_active_hold_sha256`; it saves raw bytes for two statuses but not the hold. `apply_migration:846–858` then overwrites the hold. Its precise original bytes cannot be recovered from a hash or reconstructed JSON. This violates §14.4/14.7.

**Fix:** durably preserve exact original hold bytes with bound SHA (raw base64 in PREPARED or an immutable independent backup) before any allowed hold/status mutation. Verify backup/hash/source on retry and consumption. Test byte identity, including formatting, through success and each failure/retry.

### MAJOR-04 — immutable incident asset protection is incomplete

`validate_provenance:292–299` hashes only peer manifest/epoch17 checkpoint plus peer receipt and SIGSEGV failure. `_asset_inventory` covers only the 52 completed trees. No validator freezes/checks SIGSEGV manifest/epoch13 checkpoint, both incident metrics, native stderr/phase logs, or the peer failure evidence as protected bytes. Migration fixture `test_stage_F0_migration.py:84–100` copies only the peer attempt manifest/checkpoint, not even the SIGSEGV attempt tree. Therefore alteration of the missing assets can pass migration/runtime while DESIGN expressly preserves them.

**Fix:** freeze and verify an explicit immutable incident source/asset inventory, including both attempt trees and original logs/receipt absence/evidence, before PREPARED, before COMMITTED and at required subsequent protection checks. Preserve original 52-key inventory separately. Add actual SIGSEGV checkpoint/manifest, both metric/log and failure-source drift refusal tests; never write production assets as fixtures.

### MAJOR-05 — migrated status drift and retry durability can be accepted

(a) Retry branch `apply_migration:776–797` accepts BLOCKED status using only migration ID, original-hash metadata, attempt/recovery/resume fields; it does not compare the entire deterministic allowed projection against saved raw source. Changing e.g. canonical fullhash/key JSON after a partial migration can be accepted. Runtime `:558–570` also ignores the emitted `COMMITTED.statuses_after` hashes (`:884–885`), so a postcommit status with forged unchanged provenance metadata may pass.

(b) A `status1_before_dir_fsync` fault happens after the second status rename. Retry sees both statuses already BLOCKED and takes reconciliation without fsyncing the queue_status file/directory anew. Ledger append fsyncs execution_control, not queue_status. COMMITTED can therefore be written without reestablishing durable status rename after the prior fsync failure. Tests check file readability, not this order.

**Fix:** strictly validate exact original raw snapshots and every allowed status projection; compare committed target status hashes/identities on consumption. In reconciliation reestablish file/directory durability before durable-transition/COMMITTED markers. Add canonical/nonce/PID/fullhash drift fixtures before retry and after commit, and observable fsync-order assertions for the second-status failure/retry.

### MAJOR-06 — correction consumer validates two values but not required provenance graph

`validate_h49_runtime:517–526` checks schema, row count and two `verified_value` fields. It does not validate original audit path/hash, correction field names, original incorrect values, source paths/source hashes, method/verifier metadata or original-asset relationships. Existing `ensure_provenance_correction:416–419` adds only original audit SHA, still not the complete required fields. A correction with correct two values but false original/source references can pass when its newly computed file hash is supplied.

**Fix:** verify the whole required correction structure against independently frozen v4 and actual source identities/hashes, including exact two explained conflicts; reject missing/changed references, originals or methods. Add correction-original-audit/field/source/UUID conflict tests. Preserve append-only old evidence rather than overwriting to hide conflict.

### MAJOR-07 — necessary gate/release/reference bindings can be empty or omitted

`validate_h49_runtime:515,531` tests supplied correction/migration digests only if truthy. Gate `stage_F0_worker_control.py:205–222` does not require them nonempty before runtime acceptance. Once evidence exists, an empty binding skips verification. Released hold path `:325–346` checks hold SHA/H49/basic approval metadata then calls runtime without frozen correction/migration bindings; it does not validate §14.10's current design/implementation/review/source/queue hashes or explicit fresh-search-only authorization. A release with those required references omitted can otherwise satisfy its current checks.

**Fix:** require nonempty valid immutable source/evidence bindings for any GO gate and consuming release; validate the acyclic graph against actual current design/code/review/original v4/correction/migration/queue and the committed/current hold relationship. Reject missing/stale/drifted hashes and forbidden scope while retaining pending/administrative stages. Add production-style gate/release fixtures rather than skipping the ROOT-dependent evidence branch or replacing it with an unconditional mock.

## 5. Evidence limits, remaining risks and experiment safety

Coder reports reservation-wrapped project-env focused command over migration/queue/control/run/launcher files **86 passed**, CPU/Linux only, plus a read-only production preflight. Reviewer inspected test sources and independently reproduced source hashes, all queue classifications, 52-key/573-file raw inventory and listed source asset hashes using read-only Perl/shell. Reviewer did not rerun Python, probe live locks/processes, or reproduce syscall/signal fixtures. Lock-free/process-free preflight claims remain Runner evidence, not a current reviewer runtime guarantee. Production correction/COMMITTED evidence remains intentionally absent.

Several tests use synthetic completed assets and production-derived copied inputs, one hardlinked checkpoint read only. The runtime implementation itself hashes checkpoint bytes; its docstring's claim that it never opens a checkpoint for reading is inaccurate (it never deserializes or resumes one). The changed legacy first-epoch helper has gate/hold checks but is not part of the 16/28 frozen control/runtime inventories; its independent report SHA is not equivalent to a gate-bound executable dependency. Freeze or retire such control entry points when finalizing GO scope.

No CRITICAL issue identified; all seven findings are MAJOR. No algorithm redesign is requested. Native SIGSEGV cause remains UNKNOWN, and no source review guarantees future native stability. Current production has not been migrated or released. Correct all blockers, refresh report/gate/inventories, independently re-review, then separately authorize administrative migration and verify its production evidence before any explicit fresh-search release. Held-key retry/recovery/formal/test remain forbidden.
