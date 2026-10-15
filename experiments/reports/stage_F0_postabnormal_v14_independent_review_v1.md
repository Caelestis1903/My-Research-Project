# REVIEW REPORT — F0 execution control v1.4

Date: 2026-10-13. Reviewer: independent reviewer agent. Scope: execution-control-only implementation review; no training, administrative migration, or production state mutation was performed.

## 1. Verdict: FAIL

The implementation does not yet satisfy the frozen design. One CRITICAL and eight MAJOR blocking findings remain. Do not perform the production administrative migration or release/search launch using this revision. Keep the existing global hold active and return to the coder.

Reviewed inputs:

- DESIGN SPECIFICATION: `experiments/reports/stage_F0_postabnormal_design_v14.md`, SHA-256 `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9`.
- CODER IMPLEMENTATION REPORT: `experiments/reports/stage_F0_postabnormal_v14_implementation_report.md`, SHA-256 `08c90bdc30d207a807921d92d5f9f4123e2fd948e628cb3271a96fd5c2ff40ab`.
- Actual modules, launcher/child/runner/claim paths, historical migration validation, wrappers, and focused tests. This workspace has no Git repository metadata: `git status` and `git diff` cannot provide a baseline diff. Review used the actual files and frozen provenance rather than claiming a Git diff existed.

## 2. Requirement-by-requirement verification

| Design requirement | Finding |
|---|---|
| Preserve independently fixed H49 semantics; introduce exact H51 | Constants and exact-set comparison retain H49 and define correct H51; PASS for constants only. Historical runtime separation fails: M2. |
| All current launch/claim/release paths enforce H51/F140 and original canonical identity | Normal supervisor gate dispatch reaches v1.4, and caller-held comparison requires H51. Public claim has a missing-hold bypass and does not constrain its supplied rows to F140/frozen queue: C1. |
| Frozen C97/F140 source sets, full incident/source provenance, independent snapshot review | Builder inventories C97 and both new incident trees and captures raw source files. Validation and review binding are incomplete: M3. |
| Locked administrative PREPARED → transitions → hold → COMMITTED transaction | Status updates are narrow and transition events are durable in the exercised success path. Complete baseline is checked outside locks, early idempotent return is outside locks, and no final whole-baseline proof occurs: M4. |
| Original hold/events preserved; exact partial projection on reentry | Initial normal hold update preserves fields. Partial-state acceptance permits historical event and release-field corruption: M5. |
| Durable idempotent migration and release, no duplicated transactions | Ordinary injected migration reentry passes. Hold/evidence re-fsync and ledger failure reconciliation remain incomplete; release transaction/ledger is missing: M5/M6. |
| Postmigration independent GO, current review, prelaunch, principal identity-bound release | Pending gate verifies GO artifact hash. Released path skips GO verification and cannot pass normal immutable active/released-hold transition: M1/M7. |
| Strict target GPU memory fraction <0.90 before spawn/claim; unreadable telemetry NO-GO | Ordinary threshold helper checks are present; locked claim never rechecks telemetry, target UUID is not authenticated by supervisor, and malformed values can pass: M8. |
| ≤2 GPU workers, ≤5 task Python processes; slot wrapper and prior signal semantics | Existing worker IDs/locks and five-slot wrappers remain. Focused process fixtures reported by coder exist; no new experiment or third worker was started by review. No scientific algorithm changes were introduced in this module. |
| Complete production-validator/claim/release failure-injection fixtures | v1.4 tests pass but do not exercise a valid full release/claim graph; most migration fixtures replace source validation with a no-op. Required negative edges remain absent: M9. |
| Preserve scientific baseline and production assets | Fixed queue and inspected incident status/manifest/failure/receipt hashes remain equal to design. No v1.4 administrative artifacts exist in production. No algorithm/formula/tensor/mask/routing/gradient changes are part of this amendment; these checks are not applicable to the new administrative module. |

## 3. Bugs/deviations found

### C1 — CRITICAL: claim can bypass current graph and accept keys outside F140

Locations: `stage_F0_worker_control.py:485–488`; `stage_F0_queue.py:405–427`, `253–370`.

`assert_no_global_hold` returns successfully if the hold file is absent, including the production runs root. `claim_next_entry` subsequently only requires the caller's H51 list; it does not itself require the current v1.4 gate, original frozen queue row identity, F140 membership, or rejection of recovery authorizations. The unlocked helper consumes caller-supplied entries and retains legacy recovery behavior. The current gate validates the real frozen queue, but never compares that queue to the rows supplied to this public claim function.

Independent temporary-root probe, using the actual claim function and setting the control module's production root to the temporary fixture, returned a successful claim for `outside_f140_key`, with fabricated canonical JSON/full SHA and no hold/release/gate. Only temporary files were written. This directly violates the specified current claim boundary. Make production claim fail closed on missing hold/graph and bind each claimed entry to the frozen F140 and exact original queue row; prohibit all v1.4 recovery/retry inputs.

### M1 — MAJOR: legitimate release necessarily breaks the immutable migration-hold hash check

Location: `stage_F0_postabnormal_v14.py:479–491`, `557–577`.

Migration records COMMITTED and H51_HOLD_DURABLE hashes while the hold is active. `validate_current_gate` unconditionally compares both recorded hashes to the current hold hash. Its released branch requires current hold.active=False and also binds release to that changed current hash. A legitimate release changes bytes, so these conditions cannot all hold without rewriting the immutable migration ledger. Pending active and released projections must be distinct, with a verified link from frozen active bytes to the exact released projection; historical ledger hashes must remain fixed.

### M2 — MAJOR: historical H49 validator still depends on mutable current hold and gate

Locations: `stage_F0_migration.py:1027–1038`, `1132–1211`, `758–780`, `840–843`.

Before its historical-only branch, the common implementation reads live `global_hold.json` and requires exact H49. The H51 administrative migration therefore causes the historical check used by snapshot/current validation to fail. Separately, historical release verification invokes `_release_graph`, whose `_frozen_review_identity` reads the live execution gate and its current `v13_protection` review fields instead of a frozen old gate snapshot. Thus replacing the current gate with v1.4 does not isolate old review identity.

Independent read-only probe replaced only the live-hold read with an in-memory H51 projection; actual `validate_h49_historical_graph` rejected it with `global hold does not preserve exact H49`. No production bytes were changed. Historical verification must exclusively validate frozen old hold/gate/release/review evidence and must not grant launch authority.

### M3 — MAJOR: source freeze and independent source-review identity are insufficiently bound

Locations: `stage_F0_postabnormal_v14.py:198–241`, `289–383`, `663–677`.

`validate_source_snapshot` checks declared C97/F140 digests and lengths, but does not recompute either key-list digest against its fixed digest or enforce uniqueness. It never verifies its raw `source_files` records against their raw hashes/source snapshots, nor binds the frozen historical runtime subgraph to its recorded value. `source_snapshot` hashes both failure files into an inventory, but never compares those bytes to `EXPECTED_FAILURE_SHA256`; that constant is merely copied into separate metadata. Critical new failure/nonce/fullhash/UUID correspondence is not independently verified here.

The migration accepts a reviewer JSON containing only `{"verdict":"PASS"}` with a caller-supplied matching file hash; no field binds that review to the exact source snapshot, incident/asset inventories, design, or reviewed implementation. Existing fixtures use exactly this unbound generic review. A reviewer hash is not sufficient evidence that this snapshot was independently approved. Recompute exact set digests and verify frozen source bytes/identities; require the independent source-review artifact to explicitly bind the complete frozen snapshot graph.

### M4 — MAJOR: transaction validation and idempotent replay are outside required locks

Location: `stage_F0_postabnormal_v14.py:663–705`, `706–753`, `847–871`.

Complete source validation and ledger loading happen before acquiring execution/worker/GPU/canonical locks. The committed early-return path never acquires those locks. Inside locks, the code rechecks source hold and the two statuses but not C97/H49/F140/full incident assets/reviewer/source snapshot bytes. After mutation it verifies only two status projections and a small hold subset before COMMITTED, without a final full unchanged-baseline proof. A state change between prevalidation and lock acquisition, or during commit, can therefore be accepted. Ledger/snapshot/reviewer must be loaded and fully revalidated inside the serialized transaction, with the required final inventory/status/fresh-set verification before COMMITTED.

### M5 — MAJOR: partial hold projection is not exact and durable reentry is incomplete

Locations: `stage_F0_postabnormal_v14.py:734–743`, `813–844`, `684–705`, `897–940`.

Partial migration reentry accepts any hold with active=True, H51 top-level fields, and one named migration event; it does not reconstruct and compare the full expected hold from frozen source bytes or verify the event's complete identity. It also does not re-fsync an existing hold/evidence/ledger before its idempotent return. `_create_immutable_json` likewise returns for matching existing bytes without re-fsync, and a partial failed exclusive write is left as a permanent different-byte file.

Independent temporary migration probe injected `after_hold_write`, changed an existing historical event and `released_at_utc`, and reentered. Result was `COMMITTED`; the corrupted historical event remained. The probe isolated transaction behavior using the existing migration fixture, which disables source validation; actual `validate_source_snapshot` also contains no full hold-projection comparison. This violates the explicit full-projection/history-preservation rule. Reconstruct exact expected source/partial/committed bytes and durably reconcile all existing files/directories and interrupted append boundaries without accepting unrelated edits.

### M6 — MAJOR: complete release transaction and immutable administrative pointer history are absent

Location: `stage_F0_postabnormal_v14.py` public APIs/`__all__`; runtime inventory and launcher paths.

The module provides validation of a presumed `global_hold_release_v14.json` but no serialized durable release operation, append-only release ledger, duplicate-release handling, active-to-released projection commit/reconciliation, or failure injection for release/pointer submission. The source builder captures old raw pointer bytes, but no implemented pointer-update transaction consumes that frozen snapshot and commits the new administrative pointers. A runner would have to invent these administrative mutations outside the reviewed implementation. Implement the specified operation within the approved design and cover interrupted and duplicate release requests; creating a JSON file manually is not an implemented fail-closed release transaction.

### M7 — MAJOR: released gate omits independent postmigration GO and complete new release-review binding

Location: `stage_F0_postabnormal_v14.py:432–439`, `557–593`.

Only `require_release=False` verifies postmigration_review_verdict/path/hash. The production released path neither checks that GO artifact nor requires the release to bind implementation-review/postmigration-review/prelaunch identities, incident nonce graph, frozen F140, or its ledger identity. The implementation review verdict is caller-declared in the gate and the file is hash-checked without verifying a structured identity binding to the design/code/snapshot. Older review bytes can therefore be substituted into this surface if gate fields are replaced consistently; the required rejection fixture is absent. Preserve and revalidate the independent GO and complete reviewed release identity through launch and every claim.

### M8 — MAJOR: current claim lacks live memory recheck; telemetry validation/target identity incomplete

Locations: `stage_F0_queue.py:416–427`; `stage_F0_run.py:2391–2446`; `run_search_claim_worker.py:833–868`; `stage_F0_postabnormal_v14.py:543–556`.

Runner memory telemetry is captured before lock acquisition and claim; `claim_next_entry` never rereads the actual target device while holding execution_state. The supervisor records a UUID but compares it to no frozen device identity and accepts negative used memory. The gate's recorded memory fraction accepts NaN (comparison with 0.90 is false), negative fractions, and does not recompute used/total, validate UUID, or constrain the target list to exactly the two reviewed worker devices. The runner's fallback parser expects integer MiB while its source query lacks `nounits` and normal output has units; fallback validity/UUID/ranges are not checked. Add the specified live per-target pre-spawn/pre-claim check and fail closed on malformed or mismatched telemetry, preserving the strict <90% policy. External process existence/utilization must remain observation-only.

### M9 — MAJOR: acceptance fixtures do not exercise the full critical production graph

Location: `experiments/tests/test_stage_F0_postabnormal_v14.py`, focused queue/launcher tests.

Independent execution of the v1.4 suite produced 14 passed. This is not evidence that its complete launch graph works: no test constructs a valid migrated/reviewed/released v1.4 graph and claims a genuine F140 member using unmocked current validators. Migration fixtures explicitly monkeypatch `validate_source_snapshot` to a no-op. Existing claim fixtures mainly use nonproduction roots, whose hold/release validation remains the historical fallback. Missing required edges include old release/review replay, jointly forged lists, either held member's nonblocked status, missing/mismatched failure/receipt/UUID identity, historical graph drift after H51 migration, asset addition/removal, prelaunch-to-release and release-to-claim new hold, live count progression, incomplete release/ledger/fsync boundaries, and duplicate release. Tests must exercise actual production validators/claim without disabling the acceptance decision and include the discovered defects.

## 4. Corrections made

None. Parent explicitly requested an independent inspection without implementation edits. Review only wrote this report and a temporary probe script; no production administrative artifact or research algorithm was changed.

## 5. Independent verification and remaining risks

- Ran through the existing five-slot task wrapper and `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`: `python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py` → **14 passed in 10.52 seconds**. One counted Python process was used; no CUDA training was launched.
- Temporary/read-only probes returned: historical H51 head rejected; missing-hold/outside-F140 claim accepted; corrupted historical hold event on partial migration reentry accepted and COMMITTED.
- Inspected fixed queue SHA `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` and active hold SHA `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`. Both original statuses, manifests, failure audits, and peer receipt exactly match design hashes. Production v1.4 source/migration/release files remain absent; no F0 process was present in the process snapshot.
- The report's larger 122-test claim was not independently rerun in full: broader repetition would not resolve the identified missing acceptance cases. Actual existing signal/pidfd/slot mechanisms were inspected; they require their relevant regression fixtures to pass after any correction touches them.
- Native SIGSEGV root cause remains UNKNOWN. This review does not diagnose stability or interpret AUC/loss, and no scientific effectiveness conclusion follows.

## 6. Whether experiments are safe to run

**NO-GO for administrative migration, release, and training with this revision.** Correct the blocking findings within the frozen design, produce a new coder report and meaningful production-graph fixtures, then obtain a new independent implementation verdict. Postmigration GO and principal fresh-only release remain separate required gates after implementation approval.
