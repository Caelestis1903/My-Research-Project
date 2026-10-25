# Independent review — accounting and information aggregation

**Verdict: PASS WITH FIXES. No blocking issue remains after the helper-only corrections below.** This was static review: no GPU work, aggregation execution, or research-quality array inspection occurred.

Reviewed helpers:

- `artifacts/mdmt_next_v3/v3_0/analysis_scripts/aggregate_cost_information.py`, SHA256 `e1fca6fc1251033e77e9778f96afd5771a037e8f068ca212329bd716c7aee749`.
- Limited optional review of `aggregate_results.py`, SHA256 `22b91e5232762fa71ef733865a0ab21d1c4ccdee90d0ac93bb868db2db5df58c`: seed statistics, selected/endpoint identity and old aux pairing only.

All71 previously reviewed experiment source files were independently rehashed during this review and still match source identity `3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2`. No experiment implementation was changed.

| Requirement | Actual source behavior | Result |
|---|---|---|
| Canonical actual component sums | Nonadditive probe/refresh/bridge parents excluded; duplicate timestamped journal events excluded; common inference constructor context rows retained nonadditively inside the measured benchmark section. Physical warmup journals appear once. | PASS |
| Process-wall distinction | START/EXIT segments, overlapping process-wall sum, active-interval union, event span and unattributed residuals reported separately. Open segments remain explicit. These are not relabelled GPU hours or energy. | PASS |
| Standalone reconstruction | Required shared warmup added per logical method; unnecessary RHD warm H1 probes excluded for BASE/STATIC/PB. Terminal CUM/EMA feedback and identifiable research scoring/deletion/inference work excluded. | PASS |
| Inseparable work | Uniform formation reference and untagged same-bank cache work remain conservatively included with explicit caveats. No invented subtraction or precision deployment-speed claim. | PASS |
| Historical prefixes | Reused old components and required ancestors are source-labelled addbacks, outside current physical research compute. Missing/partial legacy cost is not zero: true standalone totals now null while known recorded components remain numeric. | PASS AFTER FIX |
| Three ledgers | Research actual, standalone reconstruction and development/search are alternative views, explicitly nonadditive. Historical STATIC search and reused old prefixes retain their era and accounting limits. | PASS |
| Exact different-row proofs | Unions use actual saved role IDs, known full-role chunks, deterministic registered meta IDs or ID-only RHD event arrays. Full-domain cardinality requires recorded distinctness and matching complete pool sizes. Complete main-permutation evidence is separately labelled. Unresolved hash unions remain UNKNOWN with a known lower bound. | PASS |
| Access semantics | Raw row materialization and cached objective/selection reuse are separate channels; tasks are recorded rather than counted as independent replicates. Main/shadow/formation/read/calibration/validation/scoring permissions follow the actual source paths. | PASS |
| Missing label-use journals | Existing LPS cached meta support/two-background/uniform records are recognized when present. Missing cached-use records remain UNKNOWN. P0 fixed-query metrics and cached solver/selection reuse have explicit unjournaled sections rather than absent/zero totals. Legacy cached reuse also remains UNKNOWN. | PASS AFTER FIX |
| Inputs and execution gate | Cost/access/config/status/role metadata only; lazy NPZ access requests ID arrays, not stored loss/prediction arrays. Final execution requires explicit `--execute-final` and FINAL_METHOD_FREEZE. Parent controls final scheduling. | PASS |
| Optional statistical aggregation | One independent training seed per paired observation, sample SD and Student-t95 with n−1 degrees of freedom; n2 labelled descriptive, n1 interval unavailable. Selected and common endpoint remain separate. Old aux.1 references remain source-labelled and pair to resumed aux0 by dataset/seed/checkpoint view. | PASS |

## Corrections

1. **MAJOR:** numeric `standalone_total_seconds` could be emitted despite incomplete legacy cost, and a `MISSING_ANCESTOR` suffix escaped a leading-prefix test. The helper now retains the known recorded amount but emits null true totals and `PARTIAL_TOTAL_UNRESOLVED` whenever a historical/partial-parent total is not identified.
2. **MAJOR reporting completeness:** unjournaled P0 query/metric/cached-solver use needed explicit unknown sections. The helper now writes those sections and assesses cached formation records as journaled or UNKNOWN without guessing. Frozen LPS already records its actual repeated cached meta objectives; those records are preserved, not overwritten by a false missingness claim.
3. **CRITICAL for final aggregation execution:** `aggregate_results.py` used uppercase MONO/FEATURE internal keys while frozen LPS logs and filenames use lowercase. Only the outside helper was changed to lowercase internal keys; display method names remain L0_MONO/L0_FEATURE. The frozen experiment plan already has a lowercase filename fallback and required no change.

## Interpretation boundaries

The outputs reconstruct recorded operational components, including actual resumes/restarts. They are not independently measured minimum standalone implementations. In particular, the L0 logical run includes its local-predictor reporting family; a claim about an individually deployed OWN/MONO/FEATURE pipeline must respect the per-predictor component fields and shared-cache limits rather than silently treating the whole family as one minimal predictor. Historical timings also come from their original load and incomplete accounting.

Missing components and unjournaled repeated objective uses cannot support exact cost or label-efficiency ratios. Tables retain these limits so the parent can make bounded scientific claims. This review establishes bookkeeping/source consistency, not algorithm efficacy or a quality–cost advantage.
