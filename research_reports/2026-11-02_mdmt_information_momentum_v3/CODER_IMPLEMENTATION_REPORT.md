# Coder implementation report

Implemented `experiments/conditional_information_momentum_20261102_v3/` against DESIGN_SPECIFICATION. Eight method arms, source A0/A1 and lambda0 bypass, true tower logits, observed within-domain pair sampling, ECCM/CV/collapsed-history updates, independent pair streams, fixed EMA, RAM restoration, and all selected/E10/E50 endpoints are wired into a staged command-line runner.

The source byte manifest pins the independent vendor to `38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27`. Adaptation code is outside vendor. Original CSV paths and source manifest comparisons are passed explicitly. The runner uses only seeds 2022–2026, development 2022/2024, 50 epochs and batch40960. No checkpoint files are written. Formal stage decisions require orchestrator-authored dispatch/decision JSON; this implementation does not infer scientific promotions.

CPU checks: 14 tests pass; all eight arms reproduce the next update after a BytesIO state roundtrip; lambda0 matches the source model/gradients/EMA through two updates; exact finite-system expectations, support, weights, hand-computed decay/history updates, active/selected clocks, empty conditions, None and zero gradients are tested. `CPU_TESTS.xml` records results. Compileall and the 24 unique initial-job check passed. Real full-batch S0 and formal training await independent review and runner execution.

No changes to historical files and no research-result analysis were made. The main agent owns interpretation, dispatch decisions, CSV aggregation/reporting and publication. A failed worker leaves its latest epoch state in RAM; its failure/logs require inspection before an explicit retry at the same batch and configuration. Loss of the RAM owner requires re-execution from seed.

## Commands

Use `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` as PY below and set ATTEMPT to an absolute new artifact directory. REVIEW is the independent review JSON, containing verdict `PASS` or `PASS WITH FIXES`, empty `blocking_issues`, and `source_hash` matching `IMPLEMENTATION_SOURCE_MANIFEST.json`.

```sh
$PY -m pytest -q experiments/conditional_information_momentum_20261102_v3/tests
$PY -m experiments.conditional_information_momentum_20261102_v3.run stage0 --attempt "$ATTEMPT" --review "$REVIEW" --device cuda:0
$PY -m experiments.conditional_information_momentum_20261102_v3.run initial --output "$ATTEMPT/S1_DISPATCH.json"
$PY -m experiments.conditional_information_momentum_20261102_v3.run schedule --attempt "$ATTEMPT" --review "$REVIEW" --dispatch "$ATTEMPT/S1_DISPATCH.json" --devices 0,1
```

Subsequent dispatch JSON is `{stage: S2|S2_EXTRA|S3, approved_by: orchestrator, jobs: [...]}`. Construct entries with `run.job(dataset, arm, seed, stage, lambda_pair=value)`. Existing logical identities omit stage so S3 can reuse matching completed development trajectories. Source hashes are part of identity.

After all triggered stages, final decision JSON must include `approved_by: orchestrator`, `training_complete: true`, and the recorded stage choices/reasons. Freeze and test:

```sh
$PY -m experiments.conditional_information_momentum_20261102_v3.run freeze --attempt "$ATTEMPT" --review "$REVIEW" --decision "$ATTEMPT/FINAL_DECISION.json"
$PY -m experiments.conditional_information_momentum_20261102_v3.run evaluate --attempt "$ATTEMPT" --review "$REVIEW" --devices 0,1
$PY -m experiments.conditional_information_momentum_20261102_v3.run shutdown --attempt "$ATTEMPT"
```

`status` returns RAM owner job states. `runs/<key>/result.json` contains selected/E10/E50 validation, epoch/hash identities, costs and counts; `test_endpoints/` contains the six final endpoint metrics per logical trajectory. All epoch and batch traces remain in `runs/<key>/attemptN/`. Stage0 supplies actual data manifests and peak memory admission measurements. Scientific CSV/report generation is based on these scalar records and is owned by the orchestrator.
