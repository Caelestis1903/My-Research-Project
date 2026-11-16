# Historical Evidence and mandatory NDP-A handoff

NDP-A evidence root E=/home/xzf/Code/phase0-3_auxiliary_20261115.
Report R=research_reports/auxiliary_objective_attribution_20261115.
Raw A=artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1.

Pass each fresh Astra E/R/REPORT.md in full; R/STAGE2_DESIGN_SPECIFICATION.md, STAGE2_FINAL_REVIEW.json/.md, STAGE2_DECISION.json, STAGE2_RUNNER_REPORT.json, SOURCE_MAP.md; A/run_manifest.json, source_receipt.json, movielens_data_manifest.json, kuairand_data_manifest.json, zero_counterpart_receipt.json, reused_clean_index.json; A/tables/{ENDPOINTS,PAIRED_SEEDS,PAIRED_SUMMARY,VALIDATION_CURVES,WEIGHTS,COSTS}.csv; A/runs/*/result.json, epoch JSON/epoch_commit_ledger.jsonl and launch_logs. Also R/tables/STAGE2_BATCH_AUDIT.csv and STAGE2_FIRST_BATCH_DIAGNOSTICS.csv. Pass current-cycle full report/raw endpoints/diagnostics/costs and actual updated main context commit too.

Preflight source main dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11.
Full report SHA256 4100b8ee0f97cf1f66cca888be9f8f1297b5d7b52c676995a418f9ede02a463a.
ENDPOINTS SHA256 22ddfcf68be74e3f54fdf9b4e3f2a17fcbdac5a48b2c1cee95824c950a538908 (160 rows).
PAIRED_SEEDS SHA256 8b3212f00080c097e9166ce13f749566ceba3cc04b6d1da7421924382e42caf6 (3960 rows).
NDP-A source SHA256 831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b.
Selection SHA256 c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89.

Reported evidence, seeds2022–2026:
- STRONG: zero-perturbation old auxiliary BCE reproduced most old AUC degradation; total auxiliary mass and support/label composition changed the objective.
- NDP-A fixed-EMA−CLEAN fixed-EMA validation Macro AUC: ML -0.00010125623089003443 (1/5 positive, CI[-.000292875,+.000090363]); KR -0.005775319390310641 (0/5, CI[-.013027732,+.001477093]). ML new ranking signal WEAK; KR degradation MEDIUM. Macro LL changes +.000552562/-.000212518.
- ML online AUC +.000359628 (4/5), LL -.003262382; total NDP-A EMA vs CLEAN online AUC +.002895193, while clean EMA already +.002996. Keep readouts separate.
- KR d0t0/d0t1 AUC -.017958850/-.021424504, d1t0/d1t1 -.001311194/-.001692366, d2t0/d2t1 +.005083339/+.002651658. d2t0 5/5 positive, d0t0 0/5 positive. MEDIUM observed cell trade-off.
- ML early/mid/late EMA AUC differences -.001094/+.000013/+.000308 but selected -.000101; KR -.000113/-.004439/-.006087. Selected vs common-stage signs differ.
- Natural population ML155949/318599/325619, KR172782/1743898/89452. KR another small domain d0 loses most, so size does not explain the observed pattern on its own.
- All17250 NDP-A batches preserved 2B support and total mass1; both task zero-direction fractions0. Zero counterpart exact model/Adam/EMA equality with CLEAN passed. Current main context must contain these substantive findings before first Astra.

Historical NDP-A candidate STOP is retained; it does not terminate the new unlimited research loop. Do not rerun NDP-A or treat reused seed trajectories as independent confirmation. Preserve all old reports.
