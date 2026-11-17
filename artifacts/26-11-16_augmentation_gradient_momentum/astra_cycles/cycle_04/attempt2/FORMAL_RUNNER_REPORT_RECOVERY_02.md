# Formal runner report — recovery 02

**Status:** FORMAL_COMPLETE. Mechanical execution record only; no research analysis is included.

## Run result

- Controller: PID 2762603, tool session 6629; exit code 0; foreground runtime 1073.043 seconds.
- Source SHA-256: `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`.
- Completed 20/20 trajectories, 1,000/1,000 epochs, and 34,500 useful updates.
- Test access: false.
- Recovery 02 resumed five trajectories for 111 epochs and recorded 2249 returned training calls; unknown partial calls in this segment: 0.
- Recovery workers accumulated 3306.623 seconds of worker runtime. Maximum observed full-process CUDA residency was 1948254208 bytes; measured CUDA-usable total was 25307578368 bytes.

## Recovery 02 worker records

| Dataset | Arm | Seed | Device | Worker PID | Attempt | Returned calls | Worker seconds | Unknown partial |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kuairand | MASS-SCALE-025 | 2025 | 0 | 2762707 | 3 | 49 | 87.173 | False |
| movielens | COMPLETION-BCE-025 | 2025 | 1 | 2762708 | 3 | 480 | 678.444 | False |
| movielens | COMPLETION-BCE-025 | 2026 | 1 | 2763152 | 2 | 600 | 846.273 | False |
| movielens | MASS-SCALE-025 | 2025 | 1 | 2762927 | 3 | 480 | 667.037 | False |
| movielens | MASS-SCALE-025 | 2026 | 0 | 2763383 | 2 | 640 | 1027.696 | False |

## Historical terminal records

- Two unknown-partial terminal observations exist in earlier run segments: keys `926a61e2c285299f6c3433b05b917384f93b0811dc98759de243c3dd85b9a33a`, `2df253901fbd915c33bb3f5a2c6152666931bbd1bc093b16f9aee1828dd1932c`.
- These historical observations belong to earlier run segments associated with NVML failures, per parent confirmation. They are not recovery 02 failures; recovery 02 added zero failed or unknown-partial terminals.

## Handoff and tables

- Selected RAM handoff: [`RUNNER_HANDOFF.json`](RUNNER_HANDOFF.json), `READY_FOR_PARENT_SELECTED_RAM_ACCEPTANCE`, 1,000 committed epochs, 191,017,920 selected tensor bytes.
- Parent selected-RAM acceptance: [`PARENT_STORAGE_ACCEPTANCE.json`](PARENT_STORAGE_ACCEPTANCE.json).
- Table export: `EXPORTED`, 18 CSV entries in [`tables/TABLE_RECEIPT.json`](tables/TABLE_RECEIPT.json), plus [`tables/RESOURCE_EXECUTION_AUDIT.json`](tables/RESOURCE_EXECUTION_AUDIT.json) as the nineteenth numeric table/audit artifact.
- Full cost and admission evidence remains in `physical_attempts.jsonl`, `resource_observations.jsonl`, `admission.jsonl`, and the exported `PHYSICAL_ATTEMPTS.csv`, `PHASE_COUNTERS.csv`, `COSTS.csv`, and `RESOURCE_EXECUTION_AUDIT.json`.

JSON companion: [`FORMAL_RUNNER_REPORT_RECOVERY_02.json`](FORMAL_RUNNER_REPORT_RECOVERY_02.json).
