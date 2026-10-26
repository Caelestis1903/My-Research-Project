# S1 and resident execution/export independent review

Verdict: **PASS for the reviewed S1 implementation and its campaign/artifact integration.** This report does not replace the separate formation/mechanisms review and does not by itself authorize SCALE or an incomplete full implementation.

The reviewer read DESIGN_SPECIFICATION.md, authoritative plan05/04/08, actual baselines.py/readers.py, the current campaign.py/cli.py/artifacts.py and their S1/selection/export paths, CODER_IMPLEMENTATION_REPORT_PREFIX.md and CODER_IMPLEMENTATION_REPORT_CAMPAIGN.md, and VERIFY_PREFIX_S1.json. The S0 review separately covers protocol/state/prefix and full real-data identity audit. No formal experiment was run by this reviewer.

## Implementation findings

- B uses the fixed per-domain P_form folds, target-fit input normalization/std floor/clip, correct258 inputs, zero output weight and training-role positive-rate bias. Each held fold is predicted by a head trained on the other fold for20passes; full refit is separately trained20passes. The refit model and Adam are retained. OOF refers only to B.
- H_DIRECT copies the refit model and complete Adam state before10 P_read passes. ETA_DIRECT_MATCHED uses261 inputs and the corresponding20+10 schedule; H_DIRECT_EFFORT uses60+10. Frozen complete six-head pass0/5/10 families are selected solely by P_cal Macro LL with earliest ties. B and source pilots are not modified by these copies.
- PILOT_PROB uses three probability candidates derived from the corresponding logits,261-dimensional input, the pinned stable FP64 mixture BCE, and a whole-six-head pass policy. PILOT_LOGIT uses source-minus-local residuals. FULL_SOURCE/FULL_SOURCE_ALL/FULL_TARGET/RANDOM_FEATURE use the full prescribed64-dimensional content and target/group background; FULL_SOURCE_ALL explicitly receives P_form+P_read supervision. FULL coefficients have no per-coordinate box.
- Linear readers optimize direct alpha/beta coordinates and the actual composite correction penalty, with coefficient ridge and method-specific bounds. FP64 L-BFGS-B, registered300→600 same-start numerical retry, free-gradient/KKT receipts, complete-policy P_cal selection, exact OFF, and OFF/larger-lambda tie rules are implemented. Two RRR starts enter one joint start×lambda policy inventory, rather than being selected on Q.
- Fixed-content feature arrays can be released after predictions and numerical policies are retained; raw-X reconstruction uses frozen pilots/normalizers/column metadata. Formed and projection banks retain the functions/factors needed to reconstruct their content. No trainable model/optimizer snapshot is serialized by these paths.
- S0 workers retain PrefixBundle and later stages use that same object. The campaign requires actual reviewer authorization and sends only commands/small summaries to the controller. The run order includes all four common prefixes, full S1, and the mandatory S2 three arms before its two-seed decisions. Later controls and locked seeds are separately gated. Test/legacy metrics are deferred until the global freeze. Runtime source files must be frozen before resident-service import; subsequent code changes cannot be presented as if the already loaded code changed.
- Exports retain row/role/domain/y/group/base/pilot/function/request identity, predictions, selected/terminal values, support, numerical policies, formation records and metric tables. Full native H/Z and actual downstream callbacks are required for export acceptance; a missing callback now records NOT_RUN and fails acceptance, with no fake-PASS fallback. Models remain resident through acceptance. Owned raw-X verification arrays are input-derived research data, not checkpoints.

## Independently executed checks

Using the project interpreter, the reviewer reran `verify_readers_required(config)`:

- Analytical reader gradient versus central differences maximum error `5.843311945419316e-12`.
- Correct unbounded FULL and bounded ordinary coefficient masks; measured solver projected KKT residual `3.166984728957467e-10`.
- Balanced numerical fixture selected exact whole-six-head OFF, with bit-identical probabilities to the specified baseline sigmoid path.
- Real64 MovieLens native-input direct-head check: first output gradient `.5224574208259583`, first hidden gradient0, subsequent hidden gradient `.0061205425299704075`.

The reviewer independently reran `verify_simple_raw_callbacks(config)` on72 real MovieLens fit rows spanning all three domains, with deliberately nonzero pilot/direct output weights and no training: complete NATIVE, LOCAL and B_REFIT raw-X callbacks each reported maximum probability error0. These checks establish execution/permission/numerical correctness, not research performance. No full-prefix AUC, runtime, or complete trained-system raw-X acceptance is claimed.

The complete actual dataset audit is already recorded by S0 review and need not be repeated for these code-only changes. Subsequent RUN must execute all registered training and full-role raw-X/export verification, inspect actual reader convergence receipts, and preserve final mechanism data before releasing RAM. Those are run-stage completion requirements, not assumed PASS values.

## Launch and review boundary

The parent chose the minimal launch contract: finish formation-review corrections, freeze all actual launch-source bytes and reviewer tokens, then start the resident service. Under that contract Campaign's constructor source identity matches its loaded implementation; no additional speculative reload mechanism is required. If an actual runtime correction later becomes necessary, record affected identities/runs and reload/replay explicitly rather than claiming an old resident object used new source.

S0 scoped authorization is separately available. A full authorization token may be issued only after the independent formation/mechanisms review is PASS and remaining applicable scopes are reviewed. S3/S4 triggers are resource rules; the primary orchestrator must inspect their retained claims and all counterexamples, not infer scientific success from this engineering verdict.

## Remaining package-C SCALE scope — PASS

The reviewer additionally inspected scale_probe.py and its actually invoked historical data/model bridges against the pinned source contract. The22 restored SCALE files were independently rehashed against Git blobs and SHA256 manifest: all exact. The existing actual FP64 cold/corresponding-warm Adam numerical receipt reports PASS and errors0; this review does not repeat that already completed numerical run or treat it as performance evidence.

The wrapper recreates historical canonical Original initialization and then the D model/auxiliary initialization stream before strict backbone copying. It invokes the pinned natural train-only historical data preparation, not the RCF fit role. Dconst01_NO_AUG uses the D forward with no copy/noise arguments, constant.1 probability BCE within each domain/task mean, summed six-cell objective, coupled Adam .01/wd1e-5/eps1e-8, and the original main-loader seed stream. Selection is strict validation Macro AUC with earliest ties, patience10 and max50. It keeps the selected model only in RAM, exports selected/terminal paired predictions, and defers exposed-test metric calculation until its four-run inventory is frozen. Initial and conditional seed stages remain separately registered. These inspected semantics match the required missing historical cell; no scale trajectory was run by the reviewer. Actual initialization/data manifests and training outputs remain RUN-stage evidence and must be retained for historical comparison. Unavailable historical row predictions cannot support a fabricated paired CI.

The independent formation/mechanisms final PASS and its current four-file SHA256 map were read and matched before full authorization. Thus the combined current S0/S1/C/SCALE and package-B reviews contain no unresolved blocking finding. The full token covers the actual complete source/configuration identity; subsequent source edits invalidate it. This authorizes the registered execution and conditional branches, not any scientific effectiveness claim or untriggered extra experiment.
