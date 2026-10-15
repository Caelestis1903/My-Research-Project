# Stage S3 Coder Implementation Report — Amendment A1

Status: IMPLEMENTED for independent review. This is a coder hand-off only;
no formal S3 training, queue creation, recovery promotion, or scientific result
analysis was performed.

## Scope

The only binding design was `experiments/reports/stage_S3_design.md`, Amendment
A1. The earlier S3 implementation sources were not available in the checkout,
so the S3 layer below was reconstructed against the approved specification and
the existing S1/S2 interfaces. No experimental factor, threshold, seed,
setting, target, or recovery rule was changed.

## Design-to-implementation mapping

| A1 requirement | Implementation |
|---|---|
| Exact evidence matrix and provenance | `stage_S3_protocol.py` builds 24 fresh keys (16 target-only and 8 Sample FULL replacements), the 8 S2 Equal FULL + 24 S1 D=1 reuse set, and the 40-key historical audit. S1 Sample FULL remains rejected when its forward-source hash is absent. The current launch plan has exactly 12 groups. |
| Target-only shared-gradient access | `stage_S3_gradient_access.py` partitions cross-domain shared (`C`), private (`P`), and architecture (`A_arch`) parameters, audits reachability, decomposes `L_full=L_target+L_non`, routes target gradients only to C/A_arch and full-objective gradients to P, and records removed/applied norms and substitution checks. |
| Frozen runner and evidence | `stage_S3_run.py` keeps the S1/S2 model/data/optimizer/checkpoint path, exposes only the two approved objectives and accesses, and records batch-level objective/gradient traces, counters, source inventory, target identity, and deferred prediction artifacts. FULL remains the ordinary unmodified full-gradient route. |
| Two-process execution and recovery | `stage_S3_orchestrator.py` fixes concurrency at two and the A1 12-group schedule; records pair/peer/worker/process provenance; uses isolated attempt paths and locks; preserves append-only archives; permits only the specified whole-pair epoch-zero retry; and has no user-approval pause. It requires reviewer authorization before formal queue execution. |
| Wave-2 archive recovery | `stage_S3_amendment_recovery.py` provides read-only, queue-bound revalidation and a hash-bound receipt, followed by atomic two-member promotion while preserving the original archive. It was not invoked because the original queue file is unavailable. |
| Structural report | `stage_S3_report.py` validates source/reuse identity and predictions, emits the 1,680-row raw, 2,688-row paired, 336-row interaction, and 56-row run-inventory tables, and writes mechanical summaries only. The raw schema includes the A1 target/objective/access/partition fields; strict paired rows expose both forward-source hashes; interaction rows expose all four checkpoint hashes. Prediction-level AUC/LogLoss recomputation is enforced. |

## Files added or changed

- `experiments/stage_S3_protocol.py`
- `experiments/stage_S3_gradient_access.py`
- `experiments/stage_S3_run.py`
- `experiments/stage_S3_orchestrator.py`
- `experiments/stage_S3_report.py`
- `experiments/stage_S3_amendment_recovery.py`
- `experiments/tests/test_stage_S3_pairing.py` — checks strict paired source-hash columns.
- `experiments/tests/test_stage_S3_real_collector.py` — checks required raw identity/access fields.
- `experiments/reports/stage_S3_implementation.md` — this report.

No S1/S2 source or raw-result file was written by this implementation. S1/S2
artifacts were read-only inputs to audit/collector tests; fixture outputs were
created in temporary directories. The M3OE forward source remains SHA-256
`a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`, matching
the A1 reviewed hash. The current relevant-source inventory digest is
`1db449fb71baf356be2c7d9cc408cf63121687c467451dbb0c4bc6083ca1dd70`.

This workspace has no Git metadata, so a tracked pre/post diff is unavailable;
the no-write statement above reflects the scoped edits and test operations in
this coder turn, not a claim derived from Git status.

## Provenance and queue boundary

- Previously recorded original S3 queue SHA-256: `79c649abb1994ee81d4937920009aa7460cc32543dad074af75dc88ddc12623d`.
- Previously recorded original co-run plan digest: `7a3ea2972f7ec7c3bc8418ab3f134f2e466885b50d76e76339ab1e09b049bede`.
- The original queue file `experiments/stage_S3_queue_manifest.json` is absent, so its recorded hash cannot be reverified in this checkout.
- Rebuilt A1 plan digest: `cdcad4a2a5b171f66628a6e28cd4f1ba8871acd7c5768db70b1254d509a41bc7` (24 fresh keys; 12 groups).

The prior abort review records existing wave-1/wave-2 attempts and archives.
They were preserved. The orchestrator refuses to infer their task identity or
create a replacement queue when prior state exists without the exact legacy
queue manifest. Consequently archive revalidation/promotion and formal RUN
remain blocked until those original queue bytes are restored; no substitute
queue was fabricated. This does not change A1's design.

Equal FULL was not retrained: the accepted S2 Equal manifests pass the read-only
source audit with the contemporaneous model hash above and retain the required
S2 data, initialization, exposure, optimizer, batch-order, metric, and
checkpoint evidence. The eight source-unproven S1 Sample FULL artifacts are
not reused; A1 replaces them with eight fresh runs.

## Verification

Using `C:\Users\86199\.conda\envs\EDDAenv\python.exe` (Python 3.8.20,
PyTorch 2.0.0; CUDA available):

- `py_compile` passed for all six S3 implementation modules and both changed
  test modules.
- 20 tests passed across pairing, gradient-access, retry/recovery, and
  promotion-recovery suites; 5 amendment inventory/pairing tests passed; and
  4 real-manifest mapping/identity tests passed.
- Three real-artifact collector regressions passed: complete temporary
  collector dry-run, missing-prediction rejection, and tampered-prediction
  metric rejection. After the final schema extension, the full collector
  dry-run was rerun and passed with expected cardinalities; the updated raw
  schema assertions passed.
- `experiments/tests/run_stage_S3_smoke.py` passed. CPU Equal/Sample one-step
  checks and two concurrent CUDA target-only children each completed one main
  and one architecture step; both reported 1,614,258 parameters, matching
  partition hashes, passing reachability/substitution checks, and
  `max_concurrency=2`. These were synthetic smoke updates, not formal runs.
- The collector dry-run exercised the actual 40-key read-only historical
  audit, accepted the 8 S2 Equal FULL + 24 legacy D=1 contexts, and rejected
  the 8 S1 Sample FULL artifacts for the missing source hash.

The live wave-2 archive integration test was not run: it requires the missing
original S3 queue and must not be replaced with a fabricated queue. The
outdated pre-A1 parent-abort approval tests were not used because their
16-key/user-approval workflow is superseded by the binding A1 matrix and the
user's instruction to continue automatically after reviewer PASS. No formal
S3 result or scientific conclusion exists at this hand-off.

## Reviewer hand-off

Please independently inspect this report, the A1 specification, all six S3
modules, and the smoke/test outputs. A reviewer PASS (or PASS WITH FIXES after
blocking corrections) is still required before luna_runner. The runner must
retain exactly two concurrent Python processes per launch group and run the
approved preflight. Before formal execution it must also recover the exact
original queue or stop on the existing archive/queue-integrity boundary; do
not silently create a replacement queue.

## IMPLEMENT repair round after RUN fail-fast (2026-09-23)

The first A1 launch group after the retained wave-2 pair was stopped as
required when both seed-2023 children exited successfully but validation
reported `final_common49_hash_mismatch` and `manifest_mismatch:dataset`.
The failed pair remains in the append-only `wave_1` archive and the frozen
`stage_S3_queue_manifest_A1.json` remains unchanged by this repair.

Root causes were validator-contract defects, not algorithm or protocol
differences:

1. `final_checkpoint.pth` and `common_epoch_49.pth` are written by two
   separate `torch.save` calls. PyTorch serialization metadata can make their
   byte SHA-256 values differ even when every tensor in the two state dicts is
   numerically identical. The previous S3 validator incorrectly required the
   two artifact hashes to be equal. It now loads both CPU state dicts and
   compares mapping keys, tensor shapes/dtypes, and tensor values exactly;
   unreadable checkpoints still fail closed with a distinct comparison error.
2. The inherited S1/S2 bundle manifest contract stores `dataset` as the
   display label (`KuaiRand-Pure`) and `dataset_key` as the canonical protocol
   key (`kuairand`). The S3 validator incorrectly compared the display label
   directly to the canonical task key. It now requires an exact canonical
   `dataset_key` and canonicalizes the display label using the existing
   S1/S2 mapping, matching `stage_S2_orchestrator.validate_equal_run`.

Changed files in this repair:

- `experiments/stage_S3_orchestrator.py`: canonical dataset identity helper
  and semantic final/common-49 checkpoint validator.
- `experiments/tests/test_stage_S3_validator_manifest.py`: targeted regression
  for label/key canonicalization, distinct serialized hashes with equal tensor
  state, and true state mismatch rejection.
- this report.

Read-only evidence checks against both archived seed-2023 worker runs now
return `valid=True, reasons=[]` with `require_formal=False`; no archive,
queue, canonical result, or formal run was altered. The targeted regression,
S3 syntax compilation, and the existing focused tests pass. No formal S3 run
was repeated in this repair round; reviewer must independently re-review the
changed validator before RUN resumes.

Repair-round immutability observations: the frozen queue currently hashes to
`0edcc92249de8c07e556097ec3601d5e3223e8fcb00693748dda7c87a79c8766`, and the
seed-2023 failed-pair archive tree hashes to
`770cea174992e6e63fbe6d51a3a86ad6f3a818736ea22554ea9af08aa3f7d3b`. Both were
read only during this round.

## IMPLEMENT repair round after strict objective-artifact fail-fast (2026-09-23)

Status: IMPLEMENTED for independent review. This repair changes only the
objective-artifact hash validation semantics. It does not change the A1
algorithm, factor matrix, seeds, checkpoints, queue, promotion records, or
training outputs. No formal training was launched.

### Diagnosis

The strict error
`pairing_artifact_hash_mismatch:objective_coefficients.json` together with
`pairing_field_mismatch:coefficient_table_hash` was caused by the retained
KuaiRand/natural/seed-2022 Sample target-only artifact using the earlier S3
table schema. Its immutable `objective_coefficients.json` omits the fixed
`domain_num=3` and `task_num=2` metadata fields. Its declared/raw table-content
hash is `13e59c64a88b1408eff4db877cd19f2c6e14370e10b4937b8820337349529936`
and its file-byte SHA is
`58c3f307648b7a3a5b68e39d5d56a9fc177365094bf1614849c53df5aedbe087`.
The fresh Sample FULL artifact contains the same coefficient semantics with the
complete fixed schema: normalized/raw table hash
`30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7` and file
SHA `7cb4a05961986c7dbabc6d3bb8b8fabfb1d8f36fdca2719b64539023b6357830`.
Thus the previous pair validator incorrectly treated two different identity
layers as one requirement.

### Implementation

- `experiments/stage_S3_orchestrator.py` now keeps three identities distinct:
  (1) manifest `artifact_hashes[objective_coefficients.json]` versus the
  actual file-byte SHA; (2) the raw canonical table-content hash from
  `stage_S2_protocol.coefficient_table_hash` versus the table's declared and
  manifest hash; and (3) an in-memory normalized table-content hash used only
  for strict pairing.
- Per-run validation remains fail-closed for missing or mismatched file bytes,
  malformed tables, a wrong declared/raw content hash, invalid coefficients,
  or an inconsistent manifest hash.
- Strict pair validation no longer compares objective JSON byte SHAs across
  runs. It compares normalized table content. The only compatibility
  normalization adds the two fixed D/T metadata fields to a legacy Sample
  table; it never rewrites, rehashes, or overwrites the artifact. Equal tables
  still require the complete fixed schema.
- The current `stage_S3_run.py` generator was audited by regression against the
  shared `coefficient_table_hash` helper and already emits the complete D/T
  fields. It was intentionally left byte-identical in this repair so the
  frozen queue's reviewed source inventory is not silently invalidated.
- Collector-side orchestration already passes immutable run directories through
  `stage_S3_orchestrator.validate_fresh_reuse_pair`, so it uses the same
  semantic contract without changing the report module's reviewed source hash.
- Added `experiments/tests/test_stage_S3_coefficient_hash.py`, covering shared
  generator/helper hashing, legacy-schema normalization, distinct file and
  content hashes, formatting-only byte changes, and tampered-table rejection.

### Verification and immutability

- All eight strict four-way blocks pass read-only validation with the repaired
  comparator (`2 settings × 4 seeds`, `pairing_valid=true`).
- All 24 promoted objective artifacts pass the independent byte-SHA, declared
  raw table-hash, schema, and coefficient-invariant checks (`invalid=0`).
- The new coefficient regression passed 3/3 tests. Focused S3 pairing,
  amendment, dataset/checkpoint validator tests passed (4 pairing + 6
  amendment + 2 validator checks); S3 syntax compilation passed for the
  orchestrator, report, and new test.
- The 24 promoted run-tree digests were captured before and after this repair:
  count `24`, combined digest
  `a07cf3bb5d0b5504e24544dbd8020b5cda0fda809688e7eff6e424fa9b56ef5d` both
  times. The frozen queue SHA-256 was
  `1f614dee34d3b5c29752bdc539d3a9bb44e6036b1a6dc4b6266eea335cf35efa` both
  times. Queue state remains `RUNNING/PLANNED`; no queue or promoted artifact
  was written.
- The current reviewed source-inventory digest after this code-only repair is
  `c35a6e007f32f4fbb1579b65fba602f270670e7b1bf8f9679bde58f754d55484`.

### Caveat for reviewer

The historical `test_frozen_validator_repair_rebinds_only_a_temporary_exact_queue_copy`
fixture still expects the earlier two-record validator-repair queue state and
its old queue hash; the live frozen queue has since advanced to the 24-run
post-RUN state. That stale fixture was not changed because the queue is
immutable. The repair-specific tests and the focused S3 tests listed above
pass; the broader seven-test real-collector suite also has one pre-existing
fixture assumption failure because it asserts that the canonical
`target_only_shared` directory is absent, while this RUN has intentionally
promoted all 24 artifacts into that directory. No test created or modified a
canonical artifact. The frozen queue still records its pre-repair source
inventory digest (`4ec39077ed69b0483a051360dfee43e38ca7e18572680a1f7aa53a6924314fbe`),
so any formal queue resume that requires exact source rebind must stop for an
independent reviewer decision; this coder did not rebind or rewrite the queue.
No formal rerun was performed.
