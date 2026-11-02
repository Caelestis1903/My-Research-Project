# CODER IMPLEMENTATION REPORT — Execution

Implemented the execution-owned portion of approved DESIGN_SPECIFICATION SHA256 `9e97e74cc00311cc9b9035a4dedb370b94c1a5324442d700dc78dd0049124617`. Core algorithms remain owned by the other coder. This report records implementation and focused checks; scientific interpretation belongs to the orchestrator.

## Files and behavior

- `data.py`: explicit relocated CSV paths; source raw/six-array/row/encoder/scaler checks; train joint-label counts and ML nesting assertion without test metric inspection. Original preprocessing calls the exact vendored source. W1 changes the MovieLens loader's explicit split dependency in a separate function namespace; KuaiRand uses the source categorical types/vocabulary helpers in a bounded two-pass 60/10/10 raw-row adapter. W1 fitting never uses already-encoded original arrays.
- `evaluate.py`: source metric arithmetic for defined six-cell populations; undefined AUC cells retain null six-cell macro plus LL/counts/positives/negatives. Evaluations use natural inputs and restore training mode.
- `guardian.py`: independent local-socket process, BytesIO-only tensor IPC, epoch ownership ACK, latest recoverable state, selected/E50 states, liveness/status, reconnect, final release. The scalar `guardian.json` contains a local IPC key and must be excluded from publication. No model, optimizer, EMA or RNG state is written to disk.
- `run.py`: reviewed source identity gate, frozen explicit dispatch, independent worker subprocesses, full 50-epoch training, per-epoch online/EMA selection, recovery at the last ACK boundary, one retry per necessary job, per-card current memory/power gate and measured physical memory fit. Controller attach reads existing guardian jobs before starting pending work. Retry workers wait for prior process exit; external jobs are never stopped. Admission reserves frozen remaining active work, the next job and final evaluation against the1.20 cap; a measured active overrun records BUDGET_STOP_UNMATCHED while preserving completed trajectory metrics.
- `stage0.py`: two full real batches per requested method/dataset, train-only joint-label diagnostics, complete input-direction checks, teacher gradient isolation, real memory/step/validation/IPC measurements and frozen reservations. Default methods are A0/A1/J0/J1/F0/F1. Additional registered families receive their own requested short Stage0 before dispatch. Measurements distinguish reused architecture validation timing from actual validation work; startup/data/initialization/IPC and failed Stage0 work are charged.
- `gates.py`: seed-level paired effects/t intervals, per-cell and LogLoss harm predicates, development/held gates, FR trigger, strongest completed control and simple preference. Helpers return mechanical observations; the orchestrator writes every approved stage dispatch.
- `tests/test_execution.py`: four initial focused CPU checks covering W1 boundaries/undefined metrics, seed uncertainty/harm, independent GPU placement/24-job initial matrix and RAM ACK/restore-next-step identity; one additional pure admission predicate test for the independent review's observed cap gap.

## CLI and dispatch contract

All commands run from the repository with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_view_design_20261102_v2.run`.

1. `source-manifest` returns the actual new-package source identity. `EXECUTION_SOURCE_MANIFEST.json` contains the current files/hash.
2. The independent reviewer creates a JSON receipt with `verdict: PASS` (or corrected PASS WITH FIXES), `blocking_issues: []`, and this `source_hash`. `stage0`, `run`, `attach` and `final-evaluate` require `--review RECEIPT` and reject a changed source identity.
3. `stage0 --attempt DIR --review RECEIPT --device cuda:0` executes acceptance only. `--arms FR,F_PR` or the registered TG/RG/AG arms extends measurements only when needed before a legal conditional dispatch.
4. `initial-dispatch` emits all 24 S1 jobs. `run --attempt DIR --review RECEIPT --dispatch FILE --devices 0,1` executes the orchestrator-approved JSON. `attach` uses the same dispatch and joins existing work. `status --attempt DIR` returns scalar status/epochs.
5. `recommend-next --attempt DIR` emits S1R and development predicates. Parent uses `gates.development`, `held_gate`, `strongest`, `select_winner` and `run.job` for the registered conditional matrices and freezes each stage in its dispatch JSON. A dispatch includes `approved_by: orchestrator`, `stage`, `path_choice` and `jobs`. No method is dispatched merely because a recommendation exists.
6. `freeze --attempt DIR --decision FILE` requires the orchestrator's final `approved_by: orchestrator`, `training_complete: true` decision and no live training workers. It freezes all retained selected/E50 hashes, including stopped development controls.
7. `final-evaluate --attempt DIR --review RECEIPT` opens the test/confirm window only after that global freeze, writes endpoint metrics and raw CSV tables, and releases retained state after receipts. `shutdown` then ends the empty guardian.

The attempt retains scalar job/config/epoch/batch/commit/launch/decision/failure/cost records, selected/E50 metrics and hashes. `export_tables()` creates run/trial, epoch/validation/test, physical-cost/failure and paired-effect CSVs. Retries do not increase seed n. W1 reservations transfer the original-family S0 cost estimate; actual W1 row/step counts and costs are recorded by its workers.

## Checks and revision

Executed with the project interpreter:

```text
python -m compileall -q experiments/ema_view_design_20261102_v2
python -m pytest -q experiments/ema_view_design_20261102_v2/tests/test_execution.py
4 passed in 4.20s
```

The first guardian integration check exposed an unimportable subprocess module under pytest's shortened package name. Subprocesses now use the full `experiments.*` module and explicit repository working directory; startup failure is detected promptly. The corrected focused checks passed. No GPU checks, formal training or research-result analysis were run by this coder.

Independent review identified missing remaining-work reservation and final-evaluation overhead. Corrected admission includes frozen active/next/final reservations; measured S0, training failures/partials, complete final endpoint overhead and partial evaluation invocations enter physical cost. Final evaluation caches loaded dataset bundles. Stage0 success/failure invocation journals charge only elapsed work not already represented by successful arm records. The affected pure budget test passed (`1 passed in1.54s`); no scientific configuration changed.

Skipped unrelated repository tests and refactors. Independent review and real CUDA Stage0 remain required before scientific dispatch.
