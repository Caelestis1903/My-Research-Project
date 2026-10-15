# REVIEW REPORT — F0 execution control v1.4, revision 3

Date: 2026-10-13. Independent implementation review; execution-control scope only.

## 1. Verdict: FAIL

Most previously reported defects are corrected. One verified **MAJOR blocking issue** remains in interrupted principal-release reentry: an active hold can be released without revalidating the pending source/F140 requirements when the gate already says launch_authorized=True. Keep production hold active and return this narrowly scoped defect to the coder. No administrative migration/release or training is authorized by this report.

Reviewed inputs:

- Design: `experiments/reports/stage_F0_postabnormal_design_v14.md`, SHA-256 `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9`.
- Finalized coder report: `experiments/reports/stage_F0_postabnormal_v14_implementation_report.md`, SHA-256 `e79dad3ded99ecf4570337edbf56fe7c703a811a53d45c77443c1bce81709a05`.
- Actual source and current production entry points, the structured review/source graph, migration/release helpers, and focused fixtures. No Git metadata is available for a baseline diff; inspection used current files and frozen file hashes.

## 2. Requirement-by-requirement verification and prior findings

| Prior finding / relevant requirement | Independent verification |
|---|---|
| v2 C1: canonical-root alias claim bypass | Corrected: claim resolves output_root before locking and uses resolved production-root equality; aliases reach frozen-row/recovery/live-GPU checks. Expanded actual-validator clone asserts outside-F140 and unverified-GPU alias rejection. |
| v2 M1: ordered H51 runner/claim mismatch | Corrected: actual runner retains the normalized list, and v1.4 exact-H51 helper accepts iterables only after duplicate/string/exact-membership validation. Real clone claim with set(H51) succeeds and creates attempt-00, recovery_count=0, resume=null. |
| v2 M2: old review paired replacement | Corrected at the inspected boundary: structured current review must bind v1.4 schema/design/snapshot/coder report/control inventory/history/module bytes and reviewer identity. Current gate compares verdict to the artifact, and migration PREPARED reviewer hash must equal the current gate review hash. Old v1.3 review bytes fail structured parsing/identity verification. |
| v2 M3: incident logs/design/history omitted | Corrected: fixed design hash verified; exact required raw-source record set/order validated; immutable historical runtime projection and digest recomputed; both new nonces' stdout/stderr/phases/failure/peer receipt inventoried and compared. Missing nonce log and forged design/history fixtures reject through actual source validation. |
| v2 M4: pending touched F140 accepted | Corrected on initial migration and ordinary pending-gate validation: all F140 must be untouched, and C97 remains exact before release. **Still incomplete during release reentry**: blocker M1 below. |
| v2 M5: immutable/hold/JSONL durability | Matching immutable evidence re-fsyncs file/parent; complete immutable files publish by atomic link from a fsynced temporary file; partial JSONL tails are repaired under transaction locks; partial migration hold re-fsyncs before durable hold/COMMITTED records. Dedicated helper/order fixtures pass. Reentry must still perform the full acceptance checks before releasing hold: blocker M1. |
| v2 M6: production/adversarial fixture coverage | Expanded actual-validator fixture now covers missing hold/log, old review, forged design/history, held status drift, pending F140 claim, alias root, GPU mismatch, new hold around release/claim, two interrupted release stages, duplicate release, and real set-form claim. The remaining adversarial release-reentry-with-source-drift case is absent and independently fails. |
| v2 B1: writable hardlinked clone pointer | Corrected: full clone atomically replaces mutable execution gate and checks it is a different inode. Same-filesystem synthetic hardlink fixture verifies source bytes are preserved. |
| Historical H49 vs current H51 | Fixed H49 constants/API preserve historical meaning; H51 is exactly H49 plus the two incident keys. Historical graph verifies frozen old release/review bytes separately from current launch authority. Independent historical suite passes. |
| Original assets/scientific identity/baseline | Fixed 288-row queue remains hash-pinned; original canonical row comparison is enforced on production claim; H51 held projections and C97/full incident inventories are protected. No scientific algorithm/configuration update is introduced by this control amendment. |
| Formula/tensor/mask/routing/weight/gradient requirements | No such research implementation changed in the v1.4 administrative module. Actual runner change retains the held list; it does not change learning mathematics, routing or gradients. |
| Strict GPU memory <90%, UUID, process/worker caps | Normal production claim rereads live target memory/UUID while execution_state is locked; supervisor checks reviewed target UUID and strict memory before child spawn. Malformed/unreadable telemetry denies the checked paths. Existing two-worker/five-slot reservations and prior signal/pidfd/receipt/PDEATHSIG/STOP-CONT mechanisms remain. |
| Independent implementation → administrative RUN → postmigration GO → principal fresh-only release | The separate gates remain required. Implementation verdict is FAIL until blocker M1 is corrected and independently checked. No completed test implies native SIGSEGV root cause has been resolved. |

## 3. Bugs/deviations found

### M1 — MAJOR: authorized-gate release reentry skips pending validation before making the hold inactive

Location: `experiments/stage_F0_postabnormal_v14.py:930–1013`, especially `963–1002` and `1012`.

The release operation writes its COMMITTED ledger and launch-authorized gate before flipping the active hold. Its `after_gate_authorization` hook therefore intentionally leaves a recoverable state consisting of COMMITTED release evidence, gate.launch_authorized=True, and hold.active=True.

When reentering that state, the COMMITTED branch validates ledger/release identity and the active hold hash. It calls `validate_current_gate(..., require_release=False)` **only if gate.launch_authorized=False**. If the gate already says True, it checks only release/projection hash fields, then writes the inactive hold at line 1002. The full graph is checked afterwards with `require_release=True`, which sets `allow_live_fresh_progress=True` and permits a touched F140 member.

Consequently, the new pending F140 guard can be bypassed during this specific interrupted transaction. Full source/review/asset validation also occurs only after this hold mutation in the already-authorized branch; a changed protected source may therefore leave the hold inactive even when final validation rejects.

**Independent actual-validator reproducer, temporary clone only:**

1. Used the released graph created by the independently passing full-validator test, preserving its snapshot, migration/release ledgers, structured review, control identities, and principal identity.
2. Reconstructed the exact committed active H51 hold from the frozen snapshot and migration ID. This recreates the hold/gate/ledger state at `after_gate_authorization`.
3. Kept one authentic in-F140 CLAIMED status/canonical lock from the clone's successful claim, representing a candidate touched before the interrupted release completed.
4. Checked an ordinary pending gate: rejected with `pending H51 gate requires every F140 member untouched`.
5. Called actual `commit_principal_release` with the unchanged principal identity: returned `COMMITTED`, `idempotent=True`, and changed hold.active to False.

Probe output:

```text
pending_validator: pending H51 gate requires every F140 member untouched
committed_release_reentry: COMMITTED idempotent= True
hold_active_after_reentry: False
```

No acceptance validator was monkeypatched. All probe state changes were confined to the isolated `/tmp` clone, and its original hold bytes were restored after the probe. Production was not touched.

**Required correction within the existing design:** before an active-to-inactive hold write on every reentry path, revalidate the complete pending source/review/asset/fresh/prelaunch graph under the transaction locks even if the persisted gate already contains authorized release fields. Verify those persisted fields rather than discarding their identity. If that graph no longer meets the pre-release requirements, reject before mutation and keep the hold active. Preserve normal duplicate release once the hold is already validly inactive and lawful post-release F140 progress has occurred.

Add actual-validator failure fixtures for interruption at `after_gate_authorization` followed by F140 status/directory/canonical-lock drift, review/source/incident-asset drift, and a new hold event. They must reject while preserving active hold; an unchanged interrupted graph must reconcile durably; valid already-released duplicate requests must remain idempotent. This is a correction to the specified transaction, not a redesign of the experiment.

## 4. Corrections made

None. This reviewer inspected code and ran isolated checks only. No implementation edit, production migration/release, training, test-set evaluation, or scientific-result analysis was performed.

## 5. Independent tests, evidence and remaining risks

- Through the five-slot wrapper and project environment, independently ran v1.4 source/migration/release/claim suite: **22 passed in 163.76 seconds**.
- Independently ran historical v1.3 migration suite: **30 passed in 87.97 seconds**. These two CPU suites were allowed to overlap; task Python concurrency remained below five.
- Independently executed the actual-validator adversarial reentry probe above. Ordinary pending validation rejected the state, but release reentry accepted it and deactivated hold.
- Independently ran worker-control/queue/runner/launcher regressions: **78 passed in 29.78 seconds**, with one existing TF32 API deprecation warning. Independent total: **130 passed**. Passing regressions cannot waive the acceptance failure.
- The finalized implementation/module hashes inspected agree with the coder report: v1.4 module `474dee678b702f7c3d465c85fcc93d991b879a44b7e914b735e49edfec639734`; queue `58ceca9d193f62993c74e53e6964ca20441dcb9117f346988fba202cecbf4d31`; runner `1d72a7772d9092b2857956f8d833c50dd99a113104b344a86107a10b76e2ec31`.
- Production hashes after verification remained unchanged: hold `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`; gate `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7`; queue `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23`. No production v1.4 administrative artifacts were created.
- Native SIGSEGV cause remains UNKNOWN. No inference about algorithm efficacy, domain size/difficulty, AUC, loss, or seeds follows from control tests.

## 6. Whether experiments are safe to run

**NO-GO for production administrative migration, principal release, or training with this revision.** Correct and independently verify M1; then obtain the separate source/migration/postmigration/release approvals required by the design. The current production hold must remain active during that work.
