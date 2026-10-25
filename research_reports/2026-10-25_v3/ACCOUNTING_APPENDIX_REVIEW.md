# Independent static review — accounting appendix writer

**Verdict: PASS WITH FIXES. All blocking issues corrected.**

Static inspection of the actual helper only. No new quality data, arrays, inference measurements or research outcomes were read; no GPU, aggregation or appendix-generation command was executed. Frozen experiment sources remain untouched by this review.

## Requirement verification

| Requirement | Actual implementation | Result |
|---|---|---|
| Input boundary | Reads only declared cost/information CSVs, their ledger/adapter JSONs and both inference benchmark receipts/CSVs. No experiment-module import, label/prediction/AUC/loss-table path or research decision. | PASS |
| Complete CSV transcription | Preserves every CSV field and row; pagination only changes layout. Source path/SHA/bytes, total row count, page position and index inventory recorded. Empty sections explicitly avoid asserting zero missing work. | PASS |
| Complete JSON transcription | All cost/adapter receipts rendered in full. Each inference receipt header preserves all non-record fields, and all records retain original order across5-record pages, including raw repetitions, module/state identities and checks. | PASS |
| Current ledger provenance | Ledger schema and quality_files_read=False required. Both current benchmark_receipt.json and cost_components.jsonl SHA must match cost-ledger input_provenance; stale/missing benchmark accounting refuses generation. | PASS |
| Consistent benchmark scope | Requires base40 and conditional11 records and matching declared expected_measurements. Fixed batch4096/5warm/30timed at receipt and COMPLETE-record level, exact raw timing list lengths and nonempty isolated GPU0 checks. | PASS AFTER FIX |
| Unsupported versus measured | Only COMPLETE and UNSUPPORTED_WITH_REASON accepted; unsupported requires nonempty reason. Receipt completion status must agree with record statuses. Complete physical jobs and unsupported records counted separately; unsupported values stay missing rather than fabricated timings. | PASS AFTER FIX |
| Internal counts and unknowns | Additive component/research-actual row count and information record/group counts cross-checked. Original CSV strings and JSON UNKNOWN/null preserved; no new subtraction, apportionment, efficiency ranking or scientific conclusion. Three ledgers stay explicitly nonadditive. | PASS |
| Final report identity | Generation receipt contains helper/ledger/cost-helper hashes, every source identity and generated page identity. Combined inference table parallels original observations with method/seed/checkpoint/status fields; no measured-outcome winner is chosen. | PASS |

## Bugs and corrections

**MAJOR validation gap, corrected by helper owner:** original final_inputs accepted any non-COMPLETE record under an explicit-unsupported top-level receipt, so a FAILED/unknown record could be described as unsupported. Current source explicitly allows only COMPLETE/UNSUPPORTED_WITH_REASON, enforces a nonempty unsupported reason and consistency of top-level completion status.

**MINOR validation gap, corrected:** an empty isolation_checks list passed the original any(...) test. COMPLETE records now require nonempty retained checks, all isolated.

No algorithm source or scientific hypothesis changed. No further blocker found.

## Remaining risks and safe execution

This static PASS certifies mechanical transcription and source consistency, not numeric speed/quality outcomes. Cost/component/process-wall interpretation and UNKNOWN access limitations must remain visible. A failed generation may leave partial pages before its final count checks; publication should use only pages inventoried by a successful final ACCOUNTING_APPENDIX_GENERATION_RECEIPT.json, not a wildcard of stale report files. Raw model/data artifacts are referenced by identity, not claimed uploaded.

Safe for final mechanical execution after both reviewed benchmarks complete and the final cost ledger is regenerated with their current receipt/journal hashes. No research analysis or additional experiment is authorized by this receipt.

Reviewed helper SHA256: `0346bd39638973aceac78ce4130c3ccbd073733c0a3e588bcc4037d5a8e9d1a5`.
