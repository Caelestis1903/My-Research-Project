# REVIEW REPORT — fourth native crash / one gdb checkpoint reproduction

2026-10-29. **Latest verdict: PASS — C4-1 corrected and independently verified.** Only the single specified gdb checkpoint/V diagnostic may proceed. No training, bookkeeping recovery, remaining formal launches, final freeze or E evaluation is authorized by this review. The initial FAIL and its evidence remain below.

**Initial verdict: FAIL — one MAJOR blocking schema error.** At initial review, the single gdb invocation was blocked pending correction and independent re-review. No training or E evaluation was authorized.

## 1. Design and existing path inspection

Read `DESIGN_RUNTIME_CRASH4_ADDENDUM.md` and current R2 training/evaluation paths. The partial AutoIFS/KuaiRand/2022 attempt saved selected, fixed-final and rewind checkpoints before the native fault during fixed-final V evaluation. Final bookkeeping follows that validation, so the unchanged TRAINING status is not a completed-run receipt. This review does not claim checkpoints are absent.

The design is narrow and sufficient for the present evidence: one external tool, one fresh D1 process under gdb, selected model plus one full V sweep, selected model retained while a distinct fixed-final model is built/restored, then one full V sweep. Restore saved AutoIFS ticket/temp as well as strictly loaded state; preserve4096 evaluation batches and the final short batch, complete row/logit order and exact available logged validation metrics. No optimizer steps, backward, E access, production patch, new source revision or whole-training replay.

The existing training order is selected checkpoint restoration/validation followed by a distinct final model/validation. The external diagnostic must preserve that two-model residency/order and the original forward/conversion arithmetic while adding only flushed stage instrumentation. The original `.cpu()` transfer supplies its wait boundary; extra production or diagnostic between-operation synchronizations are outside this design.

## 2. Final review gate

Inspected sealed `runtime_crash4_tool.py`, `CODER_RUNTIME_CRASH4_REPORT.md` and existing caller/helper semantics. Independently ran the CPU synthetic selfcheck: PASS,36 rows, six batches with a short tail, exact complete logits/metrics against existing evaluator,76 marker events and expected per-batch stage order. This selfcheck does not exercise the actual manifest schema and does not close the blocking finding below.

Independently decoded the approved R2 full source records and compared all18 current package/config/reference files directly: exact bytes unchanged. No hashes, checkpoint loading, GPU computation, gdb launch, real-data evaluation or E access was performed by the reviewer. The real manifest/log reads below were JSON metadata only.

### C4-1 — MAJOR: actual prepared-role schema prevents the diagnostic

`runtime_crash4_tool.py:271` checks `len(ids) != data.manifest["roles"]["V"]["n"]`. Actual `artifacts/mdmt_baselines_2526/data/kuairand/DATA_MANIFEST.json` has roles `fit`, `P_label`, `P_cal`, `P_update`, `Q_explore`, `Q_confirm`, `validation`, `test`; it has no `V` key. `Data.roles` uses the public V alias, but its manifest retains underlying role names.

Independent metadata-only reproduction of the exact expression produced `KeyError('V')`; actual `roles.validation.n` is250767. The real diagnostic would therefore stop before either checkpoint/V sweep and spend the authorized debugger invocation on a Python schema error rather than the requested native reproduction.

Minimal correction: obtain the count from the existing V mapping (`ROLE_MAP['V']`, currently just `validation`) or this fixed diagnostic's `manifest['roles']['validation']['n']`. Add one small check/fixture using the actual manifest's role schema; metadata-only actual manifest inspection needs no labels, arrays, checkpoint or GPU. Re-review this correction before running gdb. No package change is needed.

### Other requirements inspected

| Requirement | Verification before correction | Status |
|---|---|---|
| Exact R2 package/config | Direct all18 byte comparison | PASS |
| Process-local D1 before imports | Reuses sealed Crash3 helper before importing package; checks all limits and torch1/1 | PASS |
| Partial/checkpoint identity | Actual code compares run/config/source/prepared identities, strict existing loader and embedded source, expected stage/epoch/steps/ticket/temp/state size; real checkpoint checks intentionally deferred to authorized diagnostic | IMPLEMENTED, NOT EXECUTED |
| One selected/fullV then resident selected plus distinct final/fullV | Fixed two-entry loop, selected reference retained,4096 required, no optimizer/backward/extra sync | PASS inspection |
| Conversion/forward/row/logit/metric arithmetic and markers | CPU exact evaluator parity and marker check | PASS |
| Available logged reference | Code uses reuse epoch2/3 summary equality; actual JSONL has both epochs/ticket true/temp100 | PASS inspection |
| V-only labels and separate output | Only `labels('V','select')`; new diagnostic directory, no canonical write | PASS inspection; coverage schema FAIL |
| One debugger invocation/native outcome | Coder supplies one batch gdb command, SIGSEGV stop/nopass, native stacks/libs, no core; runner must inspect inferior signal regardless of gdb exit code | IMPLEMENTED, NOT EXECUTED |

No direct implementation correction was made by the reviewer; only this report changed. Return C4-1 to coder. No other blocking issue identified in the inspected scope; synthetic checks do not claim real checkpoint parity or diagnostic success.

## 3. Interpretation and stop boundary

If both sweeps finish with exact parity, report NOT_REPRODUCED_IN_FRESH_CHECKPOINT_PROCESS; this does not establish a fix or complete the partial formal trajectory. If SIGSEGV occurs, preserve native frames and the last stage, including debugger-detected inferior signal even when gdb exits0. An unavailable debugger/ptrace resource stops the single attempt with the exact error. No repeated debugger run, new storage/thread/backend guess or retraining is authorized.

D1 has already failed as a sufficient mitigation. Root cause remains unresolved; the25 complete runs, partial checkpoints/logs and existing full30 freeze/E gate must remain unchanged.

## 4. Focused correction review — C4-1 closed

Read the sealed correction and updated coder note. `_manifest_role_count` now sums the recorded underlying role components through `manifest['role_map'][role]`; the actual manifest maps V to `validation`. `_diagnose` computes/checks that manifest count before calling `data.labels('V','select')`, then independently checks returned label-ID order. The incorrect literal `roles['V']` lookup is removed.

Independently reran the specified project-Python `runtime_crash4_tool.py selfcheck`. Result: `PASS_CPU_SYNTHETIC_ONLY`, exact36-row logits and complete metrics against the existing evaluator, six7-row batches including the short final batch,76 stage events with correct order. The added metadata-only actual-manifest preflight resolved `V -> ['validation']` and count250767. Its inspected code reads config/manifest JSON only and opens no array, label or checkpoint. This directly closes the actual schema failure rather than relying only on synthetic arithmetic parity.

The prior all18 exact-R2-byte comparison and unchanged implementation-scope inspection remain applicable; no package modification or new workload is required for this correction. No CRITICAL/MAJOR/MINOR issue remains identified in this bounded implementation review. No reviewer implementation changes were made, and no real checkpoint/GPU/gdb operation was performed.

**Permitted next action:** one fresh D1/cuda:1 invocation under the supplied gdb batch command, selected checkpoint epoch2/step5064 full V once, selected model retained while the distinct fixed-final epoch3/step5486 model performs full V once. Strict checkpoint/source/config/data checks and ticket/temp restoration remain mandatory. Capture actual inferior outcome/native signal, stage log, metrics/reference equality and diagnostic cost. Any debugger launch failure consumes no authority for an automatic retry; stop with its exact message.

If both sweeps complete exactly, classify NOT_REPRODUCED_IN_FRESH_CHECKPOINT_PROCESS. That outcome alone does not complete the partial run or permit training/resumption. A crash or parity/identity error must be reported as such. Native root cause and any next recovery decision remain outside this pre-diagnostic PASS.
