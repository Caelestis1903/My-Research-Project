# Independent static review — final exposed accounting integration

**Verdict: PASS. No blocking issue found in the narrow integration.**

Reviewed the actual changed helper paths against previously reviewed canonical accounting and final launcher schemas. No new quality records, arrays or research results read, no helper execution or GPU test, no experimental source modified.

## Requirement verification

| Requirement | Actual code behavior | Result |
|---|---|---|
| Existing leaf accounting | Per-run canonical journals and prior additive/parent/duplicate rules remain the source of actual leaf sums. New final_exposed run wrapper journal keeps its producer additiveFalse; synthesized role wrappers hardcode reported_additive=False/additive=False. | PASS |
| No triple counting | Role-inclusive, run-inclusive and canonical leaf views remain separate. Master ledger explicitly forbids adding wrapper totals to leaves or each other; actual and standalone calculations include only additive rows, excluding both wrapper levels. | PASS |
| Role timing and status | Reads actual final_exposed role COMPLETE/ERROR events, start/end UTC/epoch and worker PID. Negative intervals rejected. Evaluated, hash-certified skip and failed wrapper sums separated; hash-only skips are explicitly not new test evaluation. Scheduler/PID/exit metadata and final execution receipt retained. | PASS |
| Exact journal ranges | Before/after paths must resolve to the canonical run/name, byte offsets nonnegative/monotonic and within file length. Hash reads exactly [start,end) bytes and rejects early EOF. Full file SHA also retained. Zero append explicitly means no journal append, not zero unrecorded work; missing end stays UNKNOWN. | PASS |
| Source and receipt provenance | Operation source from registered config; actual event source retained separately. Raw role journal, worker run receipt, metadata backup, raw CSV identity, scheduler, execution receipt and current canonical journal hashes preserved. Metadata provenance includes final journals, run receipts and stdout identities. No quality scores are parsed from worker run receipts. | PASS |
| Honest unresolved overhead | Old bridge exposed scoring lacks a full component timer; startup/restore/capture/IO separation remains UNKNOWN. No guessed subtraction, precision residual, zero-cost, minimal-deployment, GPU-hour or energy claim added. | PASS |
| Existing writer coverage | Full cost_components.csv now includes all wrapper fields and byte ranges. Complete master ledger JSON embeds final_exposed summary, role records, execution metadata and scheduler events; the reviewed writer already transcribes all CSV columns/rows and full master JSON. Convenience CSV/JSON carry the same records, so writer changes are unnecessary. | PASS |

## Bugs/deviations and corrections

No new result-affecting deviation found; no reviewer correction made. Existing source-scoped reviews remain the authority for algorithms and earlier helper behavior. This receipt covers only final_exposed integration and its final helper SHA.

## Remaining risks and safe execution

Inclusive wrapper sums are intentionally overlapping provenance, including retained failed/retry contexts; they are not exclusive physical compute totals. Pre-wrapper initialization/verification and any missing component decomposition remain explicitly unresolved. Hashes establish the retained append bytes and sources, not a retrospective exclusive resource estimate. Final accounting should run after scoring/benchmarks stop appending, and publication must use the successful current ledger/appendix generation receipts.

Safe for final authorized mechanical accounting after final inputs are complete. No further tests or scientific interpretation are required by this review.

- Final accounting helper SHA256: `0f04f1d9191333b33b45ed677c2ea6485f1cac71748e880b79ee4ffdc021f1f3`
- Existing reviewed appendix writer SHA256: `0346bd39638973aceac78ce4130c3ccbd073733c0a3e588bcc4037d5a8e9d1a5`
