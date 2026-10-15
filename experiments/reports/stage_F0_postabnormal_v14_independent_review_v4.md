# REVIEW REPORT — F0 execution control v1.4, revision 4

Date: 2026-10-13. Independent implementation review. Scope: execution-control implementation and isolated CPU fixtures only.

## 1. Verdict: PASS

The verified v3 release-reentry blocker is corrected. No remaining CRITICAL, MAJOR or MINOR implementation finding was identified in the reviewed scope. This verdict approves the implementation; it does not itself release the production hold or authorize training. The frozen-source review, administrative migration, independent postmigration GO, current prelaunch and new principal fresh-only release remain separate requirements.

Reviewed frozen inputs:

| Input | SHA-256 |
|---|---|
| Design `experiments/reports/stage_F0_postabnormal_design_v14.md` | `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9` |
| Coder report `experiments/reports/stage_F0_postabnormal_v14_implementation_report.md` | `aa8c709dc9b349340f9613181e28d4e8c72b5301b9ae40cb8262ce0146c3b080` |
| Current v1.4 module | `10d81444afa55c13c8aab09605da0ea9840354e0be1eff9d2b4cc7738bd3ae16` |
| Expanded v1.4 test file | `67fbeb1198bbb57455d9025a7fbb185399e911bbbd57d832c08a07f7e675e038` |

Actual source, entry points and fixtures were inspected; coder assertions were not treated as verification. This workspace has no Git metadata, so there is no claimed Git baseline diff.

## 2. Requirement-by-requirement verification

| Requirement / prior finding | Independent verification |
|---|---|
| Preserve historical H49; introduce exact H51 | Fixed H49 constants and historical API retain their meaning. H51 is exactly H49 plus the two fixed current incident keys; digest/membership/duplicate checks use independently fixed constants. Historical evidence remains separate from current launch authority. |
| Current supervisor/runner/claim use current graph and H51 | Production gate dispatch reaches v1.4; hold absence denies; actual runner retains normalized H51 list. Claim normalizes iterables after exact checks and binds caller rows to the frozen 288-entry queue, excludes protected/history keys, and rejects recovery authorization. |
| Canonical-root alias safety | Queue resolves its root before locking and compares resolved roots, so symlink aliases cannot bypass frozen-row/recovery/GPU validation. Actual-validator fixture rejects both forged alias rows and unverified alias GPU identity. |
| Fixed F140/C97, source/design/history and assets | Ordered unique list digests recomputed; exact design bytes and raw source record set verified; immutable historical projection recomputed. C97 full tree inventory and H49 status projections are checked. New incident trees, both nonces' stdout/stderr/phases/failure evidence, peer receipt and SIGSEGV receipt absence are protected. |
| Current independent review binding; no old review/release replay | Structured review binds v1.4 schema/design/source snapshot/coder report/control hashes/history/current module and independent reviewer identity. Current gate verdict equals review content, and migration reviewer SHA equals current gate reviewer SHA. Old v1.3 review and stale source/release graphs do not satisfy these current identities. |
| Narrow locked administrative transaction | Snapshot/review/pointer checks, PREPARED, two exact status transitions, H51 hold event, final asset/projection verification and COMMITTED are serialized under execution/worker/GPU/target-canonical locks. Actual attempt manifests/receipts/checkpoints are not rewritten; no recovery/new attempt is created. |
| Partial migration and immutable-artifact durability | Source/partial/committed hold projections are reconstructed exactly, preserving historical fields/events. Matching immutable evidence and existing transitioned files/hold are re-fsynced; immutable publication is atomic; partial JSONL tails are reconciled under transaction locks. Ordinary retry is idempotent without duplicate COMMITTED. |
| v3 M1: release reentry after authorized gate but active hold | **Corrected.** At `stage_F0_postabnormal_v14.py:1003–1033`, active-hold replay verifies immutable release/current pointer, ledger SHA and canonical gate, then validates a pending in-memory gate view and frozen source pointers before writing the inactive hold. Source/F140/review/asset/hold drift denies before hold mutation. |
| Pending requirements vs lawful post-release progress | Pending validation requires every F140 member untouched and exact C97 baseline. Authorized-but-active reentry uses that pending validation. Already-inactive duplicate requests validate the released graph and allow the specified legal live F140 progression without changing frozen baseline or release ledger. |
| Principal release scope, hash graph and idempotence | Principal identity binds explicit fresh-only F0-B1-ML intent; resume/retry/new_attempt/checkpoint_load/formal/test remain false. Versioned artifact/pointer snapshots/append-only stages and original active/released projections are linked without rewriting old migration hashes. Unchanged interrupted release reconciles; released duplicate does not add ledger events. |
| Strict GPU memory/UUID and process/worker controls | Normal production claim reads fresh target memory/UUID under execution_state; supervisor checks reviewed target UUID and strict memory fraction before child spawn. At/above 90%, unreadable or invalid telemetry denies those paths. Existing two-worker/five-slot reservations and reviewed signal/pidfd/receipt/PDEATHSIG/STOP-CONT machinery remain intact. |
| Preserve scientific baseline, formulas, tensors, masks, routing and gradients | Amendment is administrative. Runner's held-list change affects eligibility only; scientific canonical keys, queue/configuration, model/objective/optimizer/training budget/seed/test gates are unchanged. No new learning formula/tensor/routing/gradient algorithm was introduced. |
| Preserve production state during review | Hold, old gate, fixed queue and inspected incident status/manifest/failure/peer receipt bytes retain their frozen hashes. No production v1.4 administrative artifact, training child or migration was created by this review. |

The private source-pointer helper's new `expected_mutable` argument is used in the inspected release-reentry path only after verifying the persisted release and canonical gate identities. It permits those two exact administrative pointer projections; it does not skip the current graph, hold, review or asset checks and does not grant launch authority.

## 3. Bugs/deviations found

None remaining in the inspected implementation scope.

The earlier v3 defect was independently reproduced before this revision and is now covered by the actual-validator integration fixture. The new fixture injects `after_gate_authorization` and separately introduces:

- An F140 CLAIMED status.
- An F140 run directory.
- An F140 canonical-lock file.
- Changed independent reviewer bytes.
- Changed new-incident stderr bytes.
- A new active-hold event.

Each reentry rejects while the hold remains active. Restoring the unchanged graph then completes release. A subsequent exact F140 claim succeeds with attempt-00, recovery_count=0 and no resume; duplicate principal release after that lawful progress remains idempotent and leaves ledger bytes unchanged.

The same expanded fixture also exercises earlier source/history/old-review/held-status/alias/GPU/new-hold boundaries using actual acceptance validators. Tests mock hardware telemetry and redirect the production-root identity to a temporary clone for CPU isolation; they do not replace the full integration acceptance validators with no-ops. Separate narrow migration-order fixtures still use synthetic source stubs, which are supplemented by the actual-validator integration test.

## 4. Corrections made

None by this reviewer. The implementation correction was made by the coder and independently inspected. Reviewer writes are limited to this report and pytest temporary artifacts; no production migration/release, training, scientific-result analysis or implementation edit was performed.

## 5. Independent verification and remaining risks

For this finalized revision, independently ran through the existing five-slot wrapper with `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

```text
python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py
22 passed in 220.24 seconds (0:03:40)
```

This includes the expanded actual-validator negative/positive reentry fixture and structured current-review binding against the finalized coder report/control inventory. One counted task Python process was used in this review; no GPU experiment was launched.

In the immediately preceding v3 review, independently ran 78 worker-control/queue/runner/launcher regressions and 30 historical migration tests, all passing, with one existing TF32 API deprecation warning. Their production modules and focused tests have byte-identical hashes in v4; only the v1.4 module, expanded fixture and coder report changed. Those 108 results are prior independent regression evidence, not a claim that all 130 tests were rerun in this v4 review.

Confirmed production hashes after inspection:

- Active hold: `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`.
- Old current execution gate: `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7`.
- Fixed search queue: `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23`.
- Both new incident status/manifests/failure audits and peer receipt match the frozen design values. Production v1.4 migration/release/snapshot artifacts remain absent.

CPU fixtures establish the reviewed control behavior; they do not establish native training stability or GPU timing behavior under a new production run. Native SIGSEGV root cause remains UNKNOWN. No AUC/loss/domain/seed or algorithm-effectiveness conclusion was drawn.

## 6. Whether experiments are safe to run

**Implementation PASS permits the specified staged administrative workflow.** Before production administrative migration, independently freeze/review the actual source snapshot and bind the structured review metadata to it and this exact approved control inventory. After migration, obtain independent postmigration GO and current prelaunch verification. Training remains blocked until a new principal identity-bound fresh-only release and the live GPU/process/claim checks pass.

This report does not deactivate hold, approve retry/recovery, authorize a third GPU worker, complete missing search units, or permit formal/test evaluation.
