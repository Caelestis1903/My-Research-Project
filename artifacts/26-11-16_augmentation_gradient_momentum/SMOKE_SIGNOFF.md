# SMOKE REVIEW SIGNOFF

1. Verdict: **PASS**. `formal_matrix_allowed=true`. Blocking issues: none.

2. Verification: current source/design and implementation review hashes match the prepared manifest and both smoke reports. Prepare/smoke exit 0; raw smoke stdout equals SMOKE_REPORT.json. The current protocol smoke gate is `validate_attempt(..., require_smoke=True)`; it passed. This protocol has no separate `check_smoke` function.

| Dataset | Branches × rows | Exact source/gamma0 state | SHUFFLE relative norm error | Peak reserved GPU bytes |
| --- | --- | --- | --- | --- |
| movielens | 5 × 40960 | model/Adam/EMA/RNG/loader equal | 4.11181303571e-09 | 2749366272 |
| kuairand | 5 × 40960 | model/Adam/EMA/RNG/loader equal | 3.11131044206e-08 | 3336568832 |

All ten recorded branches are finite, use one optimizer update and one EMA update, and retain natural full support. Cap/update-norm arithmetic and runner summaries match the raw records. Seed2022, FP32 and cuda:1 match the recorded invocation. Test remains sealed.

3. Bugs/deviations and corrections: none; no implementation changes made.

4. Source SHA256: `d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6`. Design SHA256: `67fdebf500e80611de2f0600e0847bf0697774bcb0fa107571154b1524fff338`. Exact report/log/manifest hashes are recorded in SMOKE_SIGNOFF.json.

5. Remaining runtime conditions: use this unchanged source/design and frozen matrix; retain existing admission, RAM guardian, stop and Test controls. Actual durable-parent events and future remote-main publication remain subsequent recorded stage obligations.

6. Formal matrix: **allowed** — 50 new trajectories, seeds2022–2026, 50epochs, batch40960, FP32, 86250 optimizer/EMA updates.

No tests or smoke rerun; no broad code audit, formal training, driver launch, scientific analysis, commit or push performed.
