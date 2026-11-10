# Independent reporting exporter review

Verdict: **PASS WITH FIXES**. Both identified reporting-identity issues are corrected; no blocking findings remain.

Reviewed exporter: `export_report_tables.py`, SHA-256 `2faaee4a193e185a6a17539554ae3673035adb147a2688a0559f64c742bed1b7`.

Scope: the completed reporting design, 07 statistical definitions, 08 section 7, producer schemas in `block.py`, `analysis.py` and `bridge.py`, the coder's implementation/compression report, actual exporter source and completed `/tmp/ema_report_preview_compressed`. This review concerns saved-scalar exports and plots, not research conclusions or experiment execution.

## Findings and corrections

1. Derived `delta_F` values originally retained the source row's `reference_id=O`. The corrected code explicitly sets the arithmetic reference ID/state/epoch and preserves original source reference metadata in separate fields. Exact pair records retain their actual original comparison identity.
2. Endpoint deltas originally shared one family without explicit independent-selection versus same-time identity. Corrected output separates endpoint values, `test_endpoints_independent_selected` and `test_endpoints_same_time_atF`. Comparison identities and both selected/reference epochs are retained; matched-at-F comparisons assert equal epochs. Independently checked these paths using scalar fixture rows with different O-selected and F-selected epochs; no Test evaluation or model was run.

## Aggregation and schema verification

- Six domain-task cells receive equal weights within each seed. Missing cell values invalidate the six-cell mean. The exporter keeps cell and seed tables, rejects duplicate scientific cells/seeds and separates dataset, objective, population, anchor and contrast identities.
- Cross-seed summaries use the sample SD and `2.7764451052 * SD / sqrt(5)` interval only for five available seed values. Early E5/E10, middle E20 and late E35/E50 are aggregated within seed; tF remains separate. Stage tables retain the available anchor coverage. AUC and LogLoss remain separate, with raw signed LogLoss differences.
- Independently recomputed the supplied ML-C late full-V F−O comparison from saved intervention cells. Five values are `[0.004042947416085019, 0.004488023481490355, 0.0035700740528696873, 0.0037411657742447146, 0.0033572887890878644]`, mean `0.0038398999027555282`, sample SD approximately `0.0004406700495881`, 95% t4 interval approximately `[0.0032927357670947, 0.0043870640384162]`. Exported statistics match, including after compression.
- Exact pair corrected/broken quantities remain the producer's already-normalized AUC contributions; there is no second division. Original 3×3 counts and pair denominators remain in compressed cell tables and the original source tables. Checked retained counts and corrected values directly against source records.
- Baseline lookup matches dataset/objective/seed/anchor/population/domain/task. Ensemble comparisons therefore use the matching O/F population and anchor. C_PRIMARY recent/old states are explicitly documented as deletions from F, distinct from recent-only/old-only interventions on O. Fusion deletion plots use modified-minus-F. The h/fused plot explicitly identifies its O→F contrast.
- Endpoint input column names match the actual saved producer schema. Existing ENDPOINTS is the only endpoint input; absent Test results are explicitly skipped. No model, guardian, dataset loading, bridge fitting or new Test evaluation is imported or performed.

## Final artifact verification

The coder's final full ML-C preview completed successfully. Independently verified its manifest binds the exact reviewed exporter hash, all seven required plot families exist, all 23 linked original source hashes match, and no uncompressed CSV or redundant raw JSON copy exists. Derived CSVs are gzip-compressed; source tables remain referenced by path and SHA-256. Counts and provenance remain present in the compressed cells. Coder reports 77 files totaling 65,811,904 bytes, largest file 6,691,842 bytes.

Seven families are generated from actual saved scalar data: O/F/intervention curves, corrected/broken/net pair contributions, group×time effects/interactions, h/fused margins/rank changes, fusion deletions, ensemble correction overlap, and bridge action values. The preview's skip list visibly names the three unfinished conditions and absent Test endpoints. The O/F plot was also visually inspected for separate metric panels, anchor labels and interval display.

No source-package, supervisor or attempt files were changed by this reviewer. Unfinished conditions and actual Test endpoints cannot yet be validated against completed exports; rerun this same exporter when their saved outputs exist. This PASS releases the reporting exporter, not a scientific conclusion or a claim that the ongoing experiment is complete.
