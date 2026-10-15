# REVIEW REPORT — F0 execution control v1.4, revision 2

Date: 2026-10-13. Independent implementation review. No training, production migration, production release, or implementation edits were performed.

## 1. Verdict: FAIL

This revision resolves several previous defects, but one CRITICAL and six MAJOR blocking findings remain. The production runner's H51 set/list mismatch alone prevents RUN. The alias-root claim bypass and accepted old-review substitution independently violate the approved control design. Keep the existing production hold active; return to the coder.

Inputs inspected:

- Design: `experiments/reports/stage_F0_postabnormal_design_v14.md`, SHA-256 `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9`.
- New coder report: `experiments/reports/stage_F0_postabnormal_v14_implementation_report.md`, SHA-256 `a05aa1dfd8f975ccfe17682cee34faee3e893a05440a8d5a51d77b02ced3ad3e`.
- Actual current modules, shell wrappers, supervisor/child/runner/claim entry points, historical migration verifier, and focused tests. No Git baseline diff is available in this workspace; conclusions are based on inspected code and independent fixtures.

## 2. Requirement-by-requirement verification and previous blocker disposition

| Previous finding / design requirement | Current verification |
|---|---|
| C1: missing hold, caller queue identity, F140/recovery gate | Missing production hold now rejects; normal absolute-root claim compares all frozen rows and rejects recovery authorization. **Not fully fixed:** a symlink alias skips those production checks; new C1. |
| M1: impossible active/released ledger hash equality | Fixed in normal released path: migration active hash is compared to the gate's source-active hash, while current released hash is checked separately. Real clone release passes. |
| M2: historical H49 depends on current H51/review head | Exact live-H49 requirement is disabled only for the explicit historical API; old release bytes are read from the frozen v1.4 snapshot, with hash-pinned old review identity rather than current gate identity. Normal H51 migration/release clone passes this historical check. |
| M3: source sets/digests and independently reviewed source | H49/H51/C97/F140 list digests and uniqueness now recomputed; fixed failure bytes checked. Administrative source review now binds snapshot/design. **Partly fixed:** snapshot provenance omissions and current implementation-review identity remain; new M2/M3. |
| M4: administrative validation and idempotent return outside locks | Moved into execution/worker/GPU/target-canonical locks; final source/pointer validation added. **Partly fixed:** pre-release F140 may already be touched and still validates; new M4. |
| M5: hold-history tampering and durability | Exact source/partial H51 hold projection now reconstructed; dedicated tampering test rejects historical event/timestamp edits. COMMITTED replay re-fsyncs statuses/hold/ledger. **Partly fixed:** matching immutable files and partial hold reentry still lack required durability reconciliation; new M5. |
| M6: missing release implementation | Principal release API, immutable artifact, pointer snapshots and append-only stages now exist. Actual clone release, two interrupted stages, and duplicate release work. Additional durability/negative edges remain untested; M5/M6. |
| M7: released path omits postmigration GO | GO is now checked in pending and released paths and binds snapshot/migration/active-hold/H51. **Partly fixed:** runtime implementation review still only hash-checked and old review paired replacement is accepted; new M2. |
| M8: live memory checks/telemetry | Normal production claim invokes a fresh nvidia-smi memory/UUID check under execution_state. Supervisor compares target UUID and validates usage ranges; prelaunch rejects nonfinite/range-inconsistent fractions; runner query now includes nounits. **Partly fixed:** alias-root C1 bypasses the locked claim telemetry check. |
| M9: full production graph fixtures | Added one actual-validator clone test and independently reran it with the full v1.4 suite. **Partly fixed:** actual runner argument semantics and most required adversarial/failure cases remain absent; M1/M6. |
| Preserve scientific baseline, algorithms, tensors, gradients | Execution amendment remains administrative. No new scientific formula/model/mask/routing/weighting/gradient change is present in the v1.4 module. Fixed queue bytes remain unchanged. |
| ≤2 GPU workers and ≤5 task Python | Existing five-slot shell reservations, worker-ID/worker-lock mechanism and child slot adoption retained. Reviewer used one counted task Python at a time; no worker or GPU training was started. |

## 3. Bugs/deviations found

### C1 — CRITICAL: a symlink alias of the production root bypasses frozen-row/F140/recovery/GPU checks

Locations: `experiments/stage_F0_queue.py:421–439`; `stage_F0_worker_control.py:487–496`.

The hold validator identifies production using resolved paths. The queue identifies production using `os.path.abspath(output_root) == os.path.abspath(RUNS)`, which does not resolve symlinks. Passing a symlink alias of the same production runs directory validates the real release graph, then skips the queue's exact frozen-row comparison, prohibition of recovery authorization, and live GPU check. The legacy helper can then write a canonical claim to the same real directory using arbitrary caller rows.

**Independent actual-validator probe:** a symlink alias of the isolated released clone successfully claimed `alias_outside_f140_key` with fabricated canonical JSON/full SHA and `gpu_uuid='unverified-gpu'`. No acceptance validator was monkeypatched; only the control module's production root was redirected to the temporary clone. It used the actual public claim path. Fix root recognition consistently with resolved paths or reject aliases, and enforce the current graph/F140/frozen-row/telemetry rules on every path reaching this production tree.

### M1 — MAJOR: the actual runner's H51 set is rejected by production claim

Locations: `experiments/stage_F0_run.py:2366–2367`, `2455–2457`; `stage_F0_queue.py:435–436`.

Runner converts the verified held list to a set and passes that set to claim. Production claim requires `list(held_run_ids) == list(H51_RUN_IDS)`, including sorted iteration order. A set has no matching ordered-list contract. The clone fixture supplies the sorted tuple, so its successful claim does not cover the actual runner call.

**Independent actual-validator probe:** calling claim against the valid released clone with `set(H51_RUN_IDS)` failed with `production claim requires the exact frozen H51 argument`. Preserve the normalized ordered list through the actual runner or normalize/validate the iterable at claim without losing duplicate/membership checks, and test the actual caller representation.

### M2 — MAJOR: paired substitution of the old v1.3 implementation review is accepted

Locations: `experiments/stage_F0_postabnormal_v14.py:532–539`, `697–717`, `725–742`.

Runtime verifies only the current implementation-review file hash and a gate-declared PASS verdict. It does not verify that the review contents bind this v1.4 design, coder report/control inventory, source snapshot and corrected implementation. Administrative source-review JSON is checked during migration, but current gate review is a separate unchecked identity and need not equal the reviewed migration identity. Release only equates its implementation-review hash to the gate's hash.

**Independent full-graph negative probe:** replaced gate/release implementation-review references with the real old `stage_F0_v13_implementation_review_v4.md`, recomputed the dependent temporary release/ledger/gate hashes, and retained original snapshot, migration, GO, H51, F140 and hold. `validate_current_gate(..., require_release=True)` accepted. This is the explicitly prohibited old-review paired replacement. Enforce a current independent review identity tied to the complete approved v1.4 graph, and reject this exact adversarial fixture.

### M3 — MAJOR: current incident logs and frozen design/historical metadata are not protected

Locations: `experiments/stage_F0_postabnormal_v14.py:267–288`, `353–483`.

New incident inventory includes run trees, failure audits and the peer receipt, but omits both nonces' stdout, stderr and phase logs. The design explicitly requires the complete nonce logs/receipt/failure inventory. `validate_source_snapshot` also does not verify `design_sha256` against the actual frozen design or verify the stored `historical_v13_runtime` projection against a frozen historical graph. The report's claim that all these source identities are validated is stronger than the implementation.

**Independent actual-validator probes:** hiding worker0's new-incident stderr log in the clone was accepted by `validate_source_snapshot`. Changing the snapshot's design SHA to zeros and its historical runtime metadata to `{"forged":true}`, then recomputing only its self-digest, was also accepted. Complete the explicitly required source inventory/identity validation, including exact required raw-source records with no missing/duplicate replacement, and cover asset addition/removal and historical/design drift.

### M4 — MAJOR: pending release validation accepts an already touched F140 member

Locations: `experiments/stage_F0_postabnormal_v14.py:411–434`, `743–747`, `1135–1136`, `1328–1330`.

Snapshot validation only requires live fresh keys to be a subset of frozen F140 and frozen C97 to be a subset of completed keys. That permissive rule is appropriate after legitimate claims, but is also used before migration and for pending release, without requiring the frozen 97/51/140 baseline and all 140 candidates to remain untouched. It does not reject a newly existing F140 status/attempt/lock before the authorized search starts.

**Independent actual-validator probe:** reconstructed the exact committed active H51 hold in the isolated clone and checked a pending gate while the clone's genuine F140 member still had the CLAIMED status left by the previous full-validator test. `validate_current_gate(..., require_release=False)` accepted. Before release, require exact untouched F140/baseline/no-new-attempt evidence; after release, allow only the specified lawful live progression without rewriting the frozen baseline.

### M5 — MAJOR: interrupted fsync/immutable-file/ledger boundaries are not fully reconciled

Locations: `experiments/stage_F0_postabnormal_v14.py:1301–1317`, `1350–1370`, `1385–1410`, `960–974`.

`_create_immutable_json` still returns for matching preexisting bytes without re-fsyncing the file or parent directory. If its exclusive creation fails after a partial write, reentry sees different bytes and permanently rejects rather than recovering a durable published artifact. `_append_durable_jsonl` can leave a partial tail which reentry parses as malformed JSON without a defined reconciliation path. For migration interruption after a hold rename but before directory fsync, partial reentry accepts the matching H51 hold and writes H51_HOLD_DURABLE/COMMITTED without re-fsyncing that existing hold; the re-fsync happens only on a later already-COMMITTED replay.

These are actual code paths, not covered by the named hooks, which inject after the helpers have already finished their fsyncs. Implement and test durable reentry at write/rename/file-fsync/directory-fsync/append boundaries within the approved transaction; do not publish COMMITTED proof before the required existing hold/artifact durability is reestablished. Preserve immutable historical bytes and fail closed throughout.

### M6 — MAJOR: required adversarial/production-entry/failure fixtures remain missing

Location: `experiments/tests/test_stage_F0_postabnormal_v14.py` and current focused claim/launcher fixtures.

The new complete graph fixture is useful, but it exercises only a valid sorted-tuple claim, outside-row denial and two completed-helper release interruption hooks. Most migration fault fixtures still replace source validation with a no-op. Required full-validator negatives remain missing: either new held member not blocked, old release/review paired replay, jointly forged source/gate/release lists, new failure/receipt/nonce/UUID contradictions, current incident-log/asset extra/missing files, active-hold F140 drift, actual runner's set argument and alias root, prelaunch-to-release/release-to-claim new hold, legitimate completed/live-count progression, and true rename/fsync/ledger submission failures. Add meaningful fixtures that exercise production acceptance rather than disabling its validator; include all independently demonstrated defects.

### B1 — MINOR: the clone fixture can write through hardlinked production pointers on the same filesystem

Location: `experiments/tests/test_stage_F0_postabnormal_v14.py:258–274`, `gate_file.write_text(...)` in the same test.

The fixture hardlinks the original execution gate, then later truncates/writes the clone's gate with `write_text`. If the chosen pytest temporary root shares the source filesystem, that writes the production inode. The independent run was safe because `/tmp` device 66309 differs from production device 2049, so os.link fails and the fallback copies. Explicitly copy/break links for every mutable input or exclusively use atomic replacement; do not rely on incidental filesystem separation. The review did not run this test with a same-filesystem temporary root.

## 4. Corrections made

None. Independent review only; implementation edits were not authorized for this reviewer. New report and temporary probe scripts are the only reviewer outputs. All probe mutations were confined to the isolated `/tmp` clone; no training, production administrative migration/release, or source algorithm mutation occurred.

## 5. Independent verification and remaining risks

- Through the project five-slot task wrapper and `phase0-3-env` interpreter, independently ran `python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py`: **18 passed in 87.47 seconds**. This includes actual source/migration/release/claim validation in the isolated clone; acceptance validators were not patched in that test.
- Additional actual-validator probes: valid released graph accepted; actual runner H51 set rejected; symlink-root outside-F140/unverified-GPU claim accepted; old v1.3 review paired replacement accepted; missing new-incident stderr accepted; altered design/historical snapshot metadata accepted; pending gate with a CLAIMED F140 member accepted.
- Real production hashes after verification remained: active hold `7b2899b8ba587d986d4cdc8c5a7889888d213151f5a70c9393ce14f4a235c4d3`; old execution gate `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7`; fixed queue `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23`; new SIGSEGV status `809b4dd83609f38df791afb1581f873c801aa3d408b4f51ff1d60c4dc00d4ca0`; peer status `5dc0c4c0d656c7cc3f71772852d7b5508d9d9daa005c4d88c596dc70c5aaf1d5`. Production v1.4 administrative artifacts remain absent.
- Coder's remaining 108 regression-test results were not independently rerun in this review. Existing signal/pidfd/slot code remains subject to the required regression fixtures if subsequent fixes touch it. Passing the current tests cannot waive the proven gate failures.
- Native SIGSEGV cause remains UNKNOWN. No AUC/loss or scientific effectiveness analysis was performed.

## 6. Whether experiments are safe to run

**NO-GO for production administrative migration, release, and training with this revision.** The design permits continuation after corrected implementation and independent approval; it does not permit bypassing these failures. Obtain a corrected coder report and new independent verdict. Postmigration GO and a new identity-bound principal fresh-only release remain separate gates after implementation approval.
