# Independent static review — final aggregation helper revisions

Verdict: **PASS WITH FIXES**. All identified blocking fixes are present. This review inspected source only; no new quality records, prediction arrays or research outcomes were read, and no GPU or aggregation command was executed.

## Requirement verification

| Requirement | Actual implementation | Result |
|---|---|---|
| Independent cohorts | Metadata-defined development, P2, formal and historical cohorts; comparison groups include both candidate/reference stages. Formal/development pairs are excluded, duplicate seed units rejected, formal groups over four rejected. | PASS |
| Statistics and presentation | One training seed per paired difference; sample SD and Student-t interval use n-1; n=1 has no interval. Cells/tasks/epochs are not seed replicates. Plots and appendix headings preserve cohort/stage identity. | PASS |
| Checkpoint selection | Eligible validation epochs >=12, earliest tie. Fresh formal STGU's actual validation_macro_logloss_min checkpoint_selection now selects LL and RHD family. Old aux-baseline adaptation branch remains first and retains its original declared AUC selection. Selected/endpoint and checkpoint/prediction SHA identities remain distinct. | PASS AFTER FIX |
| P2 contrasts | Exact LPS_CAL_PROB_CLIP, LPS_OWNER_META, LPS_GEOM identifiers are registered as separate formation-attribution comparisons. Formal warmups are excluded from logical comparisons. | PASS |
| Physical cost and context | Diagnostic and selection-correction directories receive separate accounting identities despite inherited raw parent run IDs. Conditional/formal process journals are included. Nested contexts, duplicate events and inclusive parents remain nonadditive; process sum/span/union remain distinct. | PASS |
| Standalone boundary | Historical/partial-parent missing totals remain null, known components retained. Shared warmups are added per deployment while counted once physically. Research scoring, terminal simple feedback and diagnostic work excluded when identifiable; inseparable uniform/cache work stays conservatively included. | PASS |
| Information identity | Only saved role IDs, ID-only event/replay keys, deterministic registered plans and explicit complete-role proofs yield exact unions. Missing cached reuse remains UNKNOWN. Conditional CAL raw-head diagnostic BCE has an explicit UNKNOWN section and inseparable measured readfit cost. | PASS |
| Execution gates | Cost helper requires explicit --execute-final and FINAL_METHOD_FREEZE; quality aggregation checks all requested logical runs COMPLETE before numerical reads. Appendix validates source aggregate counts, selected epochs, six cells and seed identities. | PASS |

## Bugs/deviations and corrections

**MAJOR, corrected by helper owner:** formal STGU configuration uses checkpoint_selection rather than family/primary_metric. Previous helper defaulted that actual configuration to LPS/AUC. Current family() and metric_key() explicitly recognize validation_macro_logloss_min while preserving old bridge adaptation selection. No experimental source changed.

Earlier reviewed helper corrections (uppercase local predictor names, incomplete legacy total, missing information sections) remain present. No additional blocking deviation was found in the scoped revisions.

## Remaining risks and safe execution

PASS covers mechanical bookkeeping, not scientific interpretation. Component seconds and overlapping process wall are not GPU-hours or energy. UNKNOWN repeated label-use counts and missing historical/partial-parent totals cannot support exact efficiency claims. L0 family-level cost still needs its per-predictor/shared-cache boundary for a single deployed local predictor. Diagnostic wrapper wall may remain unjournaled and is explicitly disclosed.

These helpers are safe for parent-authorized final execution after their gates and run completion. This review does not authorize algorithm changes or premature quality inspection.

## Reviewed source identities

- `aggregate_results.py`: `3fa23c52bfad91acd5832c5aed0a55b40c847e06fe3dddce2f35d1dc30de5d08`
- `write_numerical_appendices.py`: `1c63d231020a8b3b87e339a2f7e656486df9e1162ebbc9235c9628978634d77f`
- `aggregate_cost_information.py`: `df157d1135f0a1b00da8cae2daf23fa8e7dbf742acabc8f7c747efccd620bbc9`
