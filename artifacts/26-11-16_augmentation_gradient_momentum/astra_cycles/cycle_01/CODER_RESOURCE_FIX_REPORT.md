# CODER RESOURCE-1 fix report

Status: **IMPLEMENTED — PENDING DISTINCT FRESH REVIEW**. This is a coder report; it grants no review verdict, adoption or formal execution approval.

Current 79-file source: `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`.
Original source: `2ee2bdddb67861e65449cb3577c1d835f13395000a90b4ac101508c855f7b1a1`.
Unchanged seven-file design: `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`.
Unchanged actual smoke SHA-256: `4d50a42e5a56e0c2df2392b18ae40cec8768714f4315b63c6148efe7113521f1`.

## Change and preserved behavior

Only `runner.py`, `protocol.py`, and the focused CPU fixture file `test_recovery.py` changed.

- Formal schedule reserves **MovieLens 195.61 W / KuaiRand 206.31 W**, computed as `max(original dataset increment, all original arm board peaks)`. It labels reservations `CONSERVATIVE_OBSERVED_BOARD_PEAK_WATTS`. Original telemetry is untouched.
- The strict independent memory AND power <85% predicate, per-device admission, host accounting, pending allocations and actual resampling are unchanged. No process cap or budget reduction.
- A single `RESOURCE_1_SUPERSESSION.json` can supply the reviewed effective formal manifest. `validate_attempt`, `validate_continuation`, `require_storage`, and existing scheduler/worker job construction use that same manifest. Jobs and numerical receipts record the revised actual source.
- Original source/run/smoke/owner/guardian/handoff/parent-acceptance bytes are pinned. The new receipt binds those exact hashes, current source manifest, design, unchanged scientific dependencies/functions, conservative budgets and a fresh exact-source PASS explicitly approving reuse.
- Original owner bootstrap source remains `2ee2...`; root, PID **1570727**, socket, endpoint and parent foreground session **59365** remain bound. No server replacement or code change to frozen serve. After adoption, handoff exposes original `source_hash` and current `source_hash_verified`, plus effective-manifest and supersession hashes. It writes `RESOURCE_1_STORAGE_HANDOFF.json` / `RESOURCE_1_PARENT_STORAGE_ACCEPTANCE.json`, preserving original handoff and acceptance.

## Scientific unchanged proof

`BEFORE_SOURCE_ARCHIVE/` preserves all **79 original source files**, the **seven approved design files**, and original provenance artifacts: **99 files**, all reverified. Archive manifest SHA-256: `de463c3db17f2b3e36d21da83ba8e2e0db9138eeac7bc334a443c854e41af697`.

**76 source files** remain byte-identical, including `core.py`, `test_core.py`, `tables.py`, the package lock and all historical/scientific dependencies. The two changed runtime modules retain **40 unaffected functions/classes** byte-for-byte, including worker, recover, frozen serve wrapper, PowerSamples, smoke, source identity checks and numerical helpers. `validate_continuation` changes only its manifest lookup; `require_storage` only its effective-manifest/bootstrap-source checks. Schedule differs only in reservation construction and labeling.

The JSON report contains exact before/after file hashes, unchanged function names, the complete current source manifest, and binding digests. No source/report/receipt self-hash cycle exists: reports and reviews are outside the source manifest; the new receipt hashes its prior fresh review, and that review does not hash the future receipt.

## Focused verification

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m pytest -q -p no:cacheprovider experiments/task_axis_even_views_20261116/test_recovery.py
```

**20 passed in 3.39s**, including the original acknowledged-continuation fixture. All IPC/owner actions in tests use temporary fixtures and mocks. Actual `choose_device` rejects **283/350 W + 206.31 W** and admits **20/350 W + 206.31 W** when synthetic memory/host conditions permit. Other fixtures cover strict thresholds, pending cards, source/evidence reuse, unchanged original receipts, stale/modified evidence and reviews, changed scientific/worker/owner identity, activity before adoption, duplicate adoption, receipt deletion and budget downgrade.

Basic `compile()` passed for all three changed files. Real read-only provenance verification passed. No broad suite or `test_core` rerun; prior numerical evidence is unchanged and available for the distinct reviewer.

## Fresh reviewer handoff

The proposed review location is `IMPLEMENTATION_REVIEW_RESOURCE.json` in this cycle. The distinct reviewer must independently inspect the accepted design, actual implementation diff, original archive and this report, then author the review. Required top-level fields:

```json
{
  "verdict": "PASS",
  "blocking_issues": [],
  "source_hash": "6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1",
  "design_bundle_sha256": "f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754",
  "cpu_checks_passed": true,
  "resource_1_evidence_reuse": "<exact object from protocol.resource_1_evidence(ATTEMPT), only after independent approval>"
}
```

The JSON report contains the complete required binding object/schema. `resource_1_evidence` only reads evidence; it neither generates an adoption receipt nor issues a review. Its required `approved:true` value expresses what the **fresh external review** must explicitly approve. `reviewed_resource_1_receipt` is a pure prospective builder used by the parent command; I did not invoke it on the real attempt.

## Parent-only action after distinct PASS

From `/home/xzf/Code/phase0-3_tvg_rm_20261116`, after the distinct review explicitly approves reuse:

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -u -m experiments.task_axis_even_views_20261116.runner adopt-resource-1 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/IMPLEMENTATION_REVIEW_RESOURCE.json --owner-session 59365
```

**This command was not executed.** It appends exactly one `attempt1/RESOURCE_1_SUPERSESSION.json` with `status: REVIEWED`, exact current source manifest, effective run manifest, review path/hash, evidence/owner/admission bindings and zero-formal-activity receipt. It refuses an existing supersession; any dispatch/progress/launch/commit/exit/completion artifact; nonempty jobs/runs/launch_logs; registered jobs, retained states/bytes or live workers; a changed endpoint/session; changed originals; stale source/review; or a manual stop. It checks the existing live owner and never starts storage. All later readers recompute the exact reviewed binding; missing/modified receipts or downgrade fail closed.

Adoption does not itself grant formal execution approval. Original parent owns the remaining review/adoption/run decisions. `attach` remains the original bootstrap operation; the existing attached owner is retained.

Both workspace controls remain PAUSED, manual stop false, Test false, sentinels absent. No supersession, dispatch, launch, progress or epoch-commit file exists in the real new attempt.

No real receipt generation/adoption, storage IPC/liveness query, GPU operation, smoke/formal run, research analysis, agent/session dispatch, shared-state/design/history/guardian edit, commit or publish occurred. Test stayed sealed. Owner liveness remains for the parent command to verify; observed board peaks remain measured conservative budgets with existing actual monitoring.

