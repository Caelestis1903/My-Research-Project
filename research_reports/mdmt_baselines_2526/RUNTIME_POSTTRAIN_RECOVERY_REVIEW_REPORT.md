# REVIEW REPORT — one AutoIFS posttraining completion recovery

2026-10-29. **Latest verdict: PASS — RCV-1/RCV-2 corrected and independently verified.** Only one fixed AUTOIFS/KuaiRand/2022 posttraining completion recovery may execute. No optimizer/training step, additional V sweep or E access is authorized. Remaining4 launches require successful actual recovery and its separate completion verification. Native root cause remains unresolved. The initial FAIL and all prior evidence are preserved below.

**Initial verdict: FAIL — two MAJOR blocking implementation gaps (RCV-1/RCV-2).** Initial review prohibited recovery, training, remaining queue launches and E pending the two corrections and focused re-review.

## 1. Design and actual prerequisite evidence

Read `DESIGN_POSTTRAIN_RECOVERY.md`, the approved Crash4 diagnostic implementation, actual `runtime_crash4/gdb_execution.json`, `diagnostic/diagnostic.json`, stage JSONL, saved V IDs/logits/metrics and debugger transcript. The single gdb invocation's inferior exited normally, with no observed SIGSEGV. Classify NOT_REPRODUCED_IN_FRESH_CHECKPOINT_PROCESS, not FIXED.

Independently parsed all stage records: each view has748 records comprising62 ordered batches of4096 rows with911-row final batch, plus aggregation/metric boundaries. Each saved logit array is finite float32[250767,2]. Saved V IDs equal the original prepared V IDs exactly. Both persisted metric JSONs and diagnostic summaries equal the corresponding complete reuse2/reuse3 validation log summaries field-for-field. No V sweep or model forward was repeated by the reviewer; these checks used existing CPU artifacts only.

## 2. Actual original training-state audit

Read all17 complete JSONL records: search_setup, search epochs0–8, search_rewind, search9, reuse_setup, reuse1–3 and actual reuse_pruning_summary. All13 scientific epochs have422 updates and3450542 valid-label exposures. Recomputed total updates5486 and epoch-duration sum121.66783051693346s. Search rewind log records start index9 before final temperature update. Reuse setup records fresh rewind model/base optimizer, no reused search optimizer, equal transplanted selector and frozen gate. Actual pruning summary has six cells and original train-domain row counts.

Recomputed earliest-tie1e-12 V macroLL selection: reuse1=0.4314012657897326 improves; reuse2=0.42889490414241793 improves; reuse3=0.430536384036409 does not improve. Thus selected2 and first non-improvement at3/patience1 match the required stop; no later epoch record exists.

Independently CPU-loaded the three trusted local checkpoints without model evaluation: each has26 state tensors/2436811 finite elements. Selected is reuse2/global5064/next3/position0, ticket true/temp100, bestLL0.42889490414241793, selected_epoch2, one Adam state with per-parameter step844. Final is reuse3/global5486/next4/position0 with the same best/selected/meta and Adam step1266. Rewind is search_rewind_start/index9/global3798/next9/position0/temp59.94842503189412 with two search Adam states at step3798 and registered learning rates/decays. Python/NumPy/Torch CPU/CUDA RNG fields exist in all three states. Further implementation-specific consistency checks await the sealed tool.

## 3. Remaining review gate and limits

Independently inspected the initial sealed `runtime_posttrain_recovery.py` and `CODER_POSTTRAIN_RECOVERY_REPORT.md`. Its CPU synthetic selfcheck passes, but the two implementation gaps below are not covered by that result. Required behavior remains complete actual evidence validation, zero optimizer/backward, strict existing checkpoint loading/ticket/temp, only selected inference timing, original TRAINING bytes archived unchanged, original embedded identity preserved in atomic canonical replacement, and null original training_seconds/total_elapsed_seconds/peak_cuda_bytes. Epoch-duration lower bound and wrapper wall time are separate provenance; recovery-process measurements must not substitute for missing original costs.

The native cause remains unresolved. Successful posttraining completion would recover this original trajectory's bookkeeping only; no additional training trajectory or future automatic salvage is authorized. Remaining4 and E require their separately specified subsequent gates.

## 4. Initial implementation findings

### RCV-1 — MAJOR: direct script worker cannot import the project package

The initial `_load_stack` reuses Crash3's library/thread helper but does not add repository ROOT to `sys.path`; `_recover_worker` then imports `experiments...`. A direct Python script starts with the report directory on its path, not the repository root. Independent direct-script CPU reproduction (`project-python /dev/stdin`, runpy-load actual tool, call actual `_load_stack`, then import the same package) produces `ModuleNotFoundError: No module named 'experiments'`. The selfcheck is standard-library-only and therefore misses this executable-path failure. The one-use recovery directory was not created and the worker was not entered by this probe.

Minimum fix: add ROOT to the worker import path before project imports, as the existing external Crash4 tool does, and verify a direct-script import-only smoke without executing recovery or GPU work.

### RCV-2 — MAJOR: optimizer/RNG presence is accepted as consistency

Initial `_validate_checkpoint` checks optimizer mappings/finite values and RNG keys/tensor presence, without validating the required stage-consistent Adam steps/groups or usable RNG structure. The specification explicitly requires optimizer/RNG consistency and rejection of wrong/missing evidence (§3/§9).

Independent CPU probe first accepted the genuine selected checkpoint. Mutating an in-memory copy to set every Adam step from844 to0 was still accepted. A separate copy with Python/NumPy RNG values set to None, an empty Torch CPU uint8 state and an empty CUDA uint8 state was also accepted. No checkpoint file was changed. `_validate_checkpoint_set` supplies no later optimizer/RNG check that would close these omissions.

Minimum fix: validate this fixed run's recorded optimizer/RNG schema and stage counters, with negative fixtures for the demonstrated cases; do not instantiate an optimizer, restore/consume RNG, load GPU state or introduce a generic recovery framework.

Actual CPU metadata supplied to coder: base param IDs0..15 (16 entries) but state IDs0,2..15 (15 entries; zero-element dense_vector ID1 has no state); search gate params/state IDs0..7. Adam fields step/exp_avg/exp_avg_sq; step float32 scalar, moments float32 with corresponding parameter shapes. Rewind both groups step3798, base lr0.0003/wd3e-5 and gate lr2.9999999999999997e-6/wd0; reuse selected/final step844/1266, lr0.0001/wd0.0003; betas0.9/0.999, eps1e-8. Python RNG tuple3/version3/internal tuple625/gaussNone; NumPy tuple5/MT19937/uint32 keys[624]/position624/has_gauss0/cached0; Torch CPU uint8[5056], CUDA list2 of uint8[16]. No RNG restoration is necessary for this inference-only recovery.

## 5. Other independently verified behavior

The actual17-record training audit passes the tool's pure validator; the original incident validator also passes with epoch lower-bound121.66783051693346 and original wrapper129.106386328s kept separate. The actual Crash4 evidence independently checked in §1 satisfies the finite prerequisite. The worker verifies original/current R2, original data/config identity, actual stage JSONL/IDs, and recomputes full V metrics from the existing logits and only V/select labels; it does not merely consume a PASS Boolean.

No training/optimizer/backward entry is present in the recovery call graph. Existing loaders strictly restore model state and AutoIFS ticket/temp, followed only by the specified selected-model inference timing. Original identity is copied and the three unavailable original costs are explicitly None; recovery environment/costs have separate provenance.

Independent CPU temporary-file checks passed: original receipt saved by exclusive creation is byte-identical; atomic commit succeeds in a temporary canonical fixture; an injected `os.replace` failure leaves original canonical bytes unchanged. Archive bytes stay unchanged in both cases. A completion manifest built as a temporary fixture from the actual original identity passes the existing package `_verify_checkpoint_run` against both actual selected/final checkpoints with their embedded TRAINING receipts; all three original cost fields remain null. No fixture was written to canonical research paths, and the real run.json bytes remained unchanged.

No reviewer implementation correction was made. Only this report changed. Initial review stopped reading the tool once coder began the requested two corrections. No additional optional protection or research matrix is requested.

## 6. Focused final re-review — both findings closed

Read the sealed two corrections and coder report appendices. Independently executed `project-python /absolute/path/runtime_posttrain_recovery.py importcheck` from `/tmp`: PASS_CPU_DIRECT_SCRIPT_IMPORT_ONLY, all six required project modules import, ROOT is on the path and D1 thread counts are1/1. `_load_stack` now inserts ROOT before package imports. No actual array/checkpoint/GPU work is in this import smoke. **RCV-1 closed.**

The fixed checkpoint validator now maps ownership names to state tensors, validates this run's one-group Adam parameter IDs/registered lr, decay, betas/eps, requires all and only nonempty owned parameter states and checks float32 scalar local steps plus finite float32 moments of exact parameter shape. The known zero-element dense_vector remains in the16-entry base group with no Adam state. Rewind base/gate local steps3798 and selected/final reuse steps844/1266 are explicitly required. RNG checks validate Python state tuple/internal structure, MT19937 uint32[624] metadata, Torch CPU uint8[5056] and two CUDA uint8[16] states without restoring RNG or creating an optimizer.

Independently reran `selfcheck` from `/tmp`: PASS_CPU_SYNTHETIC_ONLY including new optimizer/RNG negative cases. Then CPU-loaded and passed the actual complete three-checkpoint set through the corrected validator, including rewind/selector cross-consistency. Independently replayed both previously accepted negative cases: all selected Adam steps reset to0 now reject; null/empty RNG now rejects. Separate malformed Python, NumPy, Torch CPU and CUDA cases each reject. Missing nonempty-parameter Adam state and changed learning rate also reject. All mutations were in-memory copies; saved checkpoint files were not modified. **RCV-2 closed.**

The real canonical run.json remained byte-identical TRAINING and the one-use recovery directory remained absent after review. No reviewer optimizer, backward, model forward, GPU inference, recovery worker, source/hash change or canonical status write occurred. Prior actual Crash4/full-log/atomic-write/null-cost/checkpoint-identity evidence in §§1–5 remains valid. No CRITICAL, MAJOR or MINOR defect remains identified in the reviewed finite scope; no optional framework or research expansion is requested.

## 7. Final run boundary and risks

**One posttraining completion attempt is permitted for the original AUTOIFS/KuaiRand/2022 trajectory only.** It may preserve exact original TRAINING bytes, revalidate existing source/config/data/log/checkpoint/diagnostic identities, reuse the two actual complete V outputs, run the specified selected-model inference timing and atomically write truthful completion bookkeeping. Preserve all original checkpoint paths/bytes and embedded training environment/identity. Keep original training_seconds, total_elapsed_seconds and peak_cuda_bytes null; epoch lower bound121.66783051693346s, original wrapper129.106386328s and recovery measurements remain separate.

After actual process completion, verify exit status, unchanged archived original/checkpoint/source identities, exact selected/final metrics, zero optimizer steps, valid recovery provenance and canonical complete/test-locked status before releasing any remaining4. Those never-started AUTOIFS/KuaiRand seeds2023–2026 retain the original R2/D1 protocol; any new native/numeric/identity/postprocessing exception stops the queue and receives no automatic recovery authority from this report. E stays locked behind the existing complete30/mixed-R1-R2 freeze.

The diagnosis remains NOT_REPRODUCED_IN_FRESH_CHECKPOINT_PROCESS and the native cause is unresolved. This PASS concerns completing one intact training trajectory's postprocessing; it does not claim runtime reliability, crash elimination, missing training cost reconstruction or new algorithmic evidence.
