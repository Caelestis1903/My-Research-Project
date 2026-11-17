# Cycle04 attempt2 four-case engineering smoke report

- Controller: PID 2477619, foreground tool session 56896, exit code 0.
- Controller result: `ENGINEERING_PENDING_INDEPENDENT_REVIEW`; all four case records are `ENGINEERING_VERIFIED_PENDING_REVIEW`.
- Identity: source `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`; design bundle `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`; seed 2022.
- Scope: engineering smoke only; zero scientific epochs; Test was not accessed. Raw records retain all case calls, proof/suffix calls, engineering commits, and measured profiles. Shared physical attempt ledger has 20 rows.

| Case | Device | Status | Calls | Diagnoses | Full process peak | Torch allocated peak | Torch reserved peak |
|---|---:|---|---:|---:|---:|---:|---:|
| case1_gpu0_movielens | 0 | ENGINEERING_VERIFIED_PENDING_REVIEW | 8 | 2 | 1,675,624,448 B | 1,093,158,400 B | 1,346,371,584 B |
| case2_gpu0_kuairand | 0 | ENGINEERING_VERIFIED_PENDING_REVIEW | 8 | 2 | 2,027,945,984 B | 1,277,969,920 B | 1,698,693,120 B |
| case3_gpu1_movielens | 1 | ENGINEERING_VERIFIED_PENDING_REVIEW | 8 | 2 | 1,675,624,448 B | 1,093,158,400 B | 1,346,371,584 B |
| case4_gpu1_kuairand | 1 | ENGINEERING_VERIFIED_PENDING_REVIEW | 8 | 2 | 2,027,945,984 B | 1,277,969,920 B | 1,698,693,120 B |

Each case record reports `source_lambda_zero_exact`, `diagnostics_neutral_exact`, `actual_gradients_activity_exact`, and `prefix_immutable` as true. Expected plan totals were recorded as 32 physical calls, 8 diagnosis calls, 24 extra autograd calls, and 32 virtual Adam steps. Batch logs contain 8 rows per case, with 2 diagnosis-marked rows per case.

Profile and raw-record paths are enumerated in `RUNNER_SMOKE_REPORT.json`. All engineering states remain retained for parent RAM acceptance; no release or formal RUN signoff was performed.
