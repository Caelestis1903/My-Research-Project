# Formal Runner Report R1

- Run status: **COMPLETE** (process exit code 0).
- Completion recorded: `2026-11-16T15:44:58.427807+00:00`.
- Seeds: 2022, 2023, 2024, 2025, 2026; requested devices: 0 and 1.
- Jobs: 20 COMPLETE, 0 NOT_STARTED.
- Committed epochs: 1000/1000; committed updates: 34500/34500.
- Source SHA-256: `a7666566be5790a0e97613d11c37792122a4028f72716669f52be1ceffe99002`.
- Design bundle SHA-256: `fe784adac9005f29b230e019b5e0fcd9c98dc842c1ac4ebe12849ba90512f3bb`.
- Test access: false.
- Runner terminal errors reported: none.
- Raw epoch commit ledger rows: 1008; epoch JSON files present: 1008.

## Table export

- Export command exited 0; receipt: `attempt1/tables/EXPORT_RECEIPT.json`.
- Counts: 40 trajectories (20 new), 400 endpoints, 4000 curves, 24000 cell curves, 34500 useful batches, 207000 weights, 2300 diagnostics.
- Export files and hashes are listed in the JSON report and export receipt.

## RAM handoff

- Parent acceptance: **ACCEPTED** (`attempt1/PARENT_STORAGE_ACCEPTANCE.json`).
- Parent verified 40 selected hashes across 20 pairs and 1000 epochs.
- Retained RAM state: 191017920 bytes; all workers dead: true.
- Test access: false.

The earlier `FORMAL_RUNNER_REPORT.json` is preserved unchanged. This report records execution and storage/export facts only; it contains no research analysis.
