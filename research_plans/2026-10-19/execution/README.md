# Conditional/shared-path experiment execution

**Complete:**856 full-budget training runs,856 audited frozen tests and126 offline diagnostics. Delivery: [PR #8](https://github.com/Caelestis1903/My-Research-Project/pull/8).

Execution branch: `codex/conditional-paths-2026-10-19`, based on pinned main55afd053d7a65ea782487d703f189615e296fd3e. The matrix contains856 independent full-budget training runs plus72 exact-equivalence aliases (928 nominal configurations). Live and final completion are recorded in EXECUTION_STATUS.json and RUN_SUMMARY.json; scientific interpretation is separate from execution completion.

## Read the findings

- [FINAL_REPORT.md](FINAL_REPORT.md): Chinese scientific report, completion and seven required evidence questions.
- [OPTIMIZER_MECHANISM.md](OPTIMIZER_MECHANISM.md): functional inactivity, factor-group interventions, moment history and numerical decay boundary.
- [CROSS_MODEL_REPORT.md](CROSS_MODEL_REPORT.md): native model scope, complete structural controls, task interaction, size/difficulty and meaningful baselines.
- [PERFORMANCE_AND_NOVELTY.md](PERFORMANCE_AND_NOVELTY.md): frozen selected/final risk, strong references, contradictory endpoints and publication disposition.
- [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md): unresolved claims and one proposed falsification study, explicitly not counted as completed work.
- [MODEL_FIDELITY.md](MODEL_FIDELITY.md), [LITERATURE_LEDGER.md](LITERATURE_LEDGER.md), [COST_MEASUREMENT.md](COST_MEASUREMENT.md): implementation, prior-art and resource boundaries.

The G0_DEVELOPMENT_NOTE is a completed-stage interim analysis, not the full-matrix final report. A useful bounded mechanism finding does not establish a new generally beneficial algorithm. The repeated historical N19 test is exploratory, not fresh population confirmation.

## Trace each result

|Evidence|Location / interpretation|
|---|---|
|All actual/alias configurations|run_manifest.json, RUN_LEDGER.csv; aliases reference an actual run and are not extra independent repetitions|
|Frozen design and code workflow|DESIGN_SPECIFICATION.md, IMPLEMENTATION_REPORT.md, REVIEW.md, EVALUATION_REVIEW.md|
|Pre-test decisions and bindings|ANALYSIS_RULES.md, DEVELOPMENT_DISPOSITION.md, TEST_FREEZE.json|
|Per-run evidence|runs/<run_id>/{config,initialization,source_hashes,receipt,selection_frozen,test_receipt}.json; trajectory.csv, metrics_validation.csv, metrics_test.csv, path/gradient diagnostics and inference timings|
|Complete aggregated risk|development_readout/ and test_readout/: seed_cell_metrics, seed_task_metrics, seed_domain_metrics, seed_macro_metrics, condition_summary, paired comparisons/seed/cell deltas|
|Descriptive evidence|Each readout's evidence/: same-task fixed references, fixed/MMoE/PLE strong references, selected/final differences, probe activity, G4 target/other domains and profile-scaled costs; test also has neutral-minus-natural cell/macro risk|
|Common-state optimizer diagnostics|offline_readout/:48 local runs, actual checkpoint moments, hypothetical one-step updates only|
|Cost profiles|offline_readout/model_cost_profiles.csv:78 model/variant/population profiles; shared-host latency and supported-dense coverage, not isolated speed|
|Probe intervals|probe_conditional_intervals.csv:200 paired within-domain row-bootstrap replicates,9,984 rows; conditional on existing model/probe, not independent population/optimization replication|
|Numerical/selection checks|DECAY_ROUNDING_CHECK.json, LONG_SELECTION_CHECK.json, FINAL_AUDIT.json|
|Standalone figures|development_figures/ and figures/: PNG/PDF exports; path curves use development probes, risk panels identify their evaluated split|
|Heavy model/prediction files|ARTIFACT_INDEX.csv: absolute local path, size and SHA256 for checkpoints/probes/frozen test predictions; final completeness in artifact_index_receipt.json|
|Raw stdout|RAW_LOGS.tar.gz and RAW_LOG_ARCHIVE.json; lossless logs with archive SHA, not training tensors|
|Interrupted attempts and recovery|RESOURCE_RECOVERY.md and resource_failures/: two training OOM attempts, five training session interruptions, and old incomplete/invalid test artifacts affecting18 configurations; all affected outputs preserved and restored, never counted as negative results|

Under the user-invoked start.md§11, primary performed design, implementation, self-review and analysis; only luna_worker handled mechanical experiment execution. Independent agent review is not claimed. Core training files stayed frozen; later changes are queue recovery/reporting only. Historical N18/N19 branches were not merged into this execution branch.

## Local artifacts and reproduction

The actual worktree is `/home/xzf/Code/phase0-3-20261019-git`; the original workspace execution entry links here. Large states and per-row arrays remain under `/home_new/xzf_cm19_artifacts`, outside Git. The repository contains their concrete paths and checksums rather than hundreds of GB of tensors. A clone alone does not contain those local artifacts. DATA_LINEAGE.json and ENVIRONMENT.json record the frozen cached populations and environment; SOURCE_LINEAGE.json records pinned source ancestry. Python was `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`, PyTorch2.9/cu126, two24GB RTX3090 devices. The resumed test readout also observed twoRTX3090 devices.

All optimization seeds2022–2025 use the same frozen population/subsets. KR has50 epochs and ML100, batch40960, LR.003/.001; G4LONG uses2,350 updates. Every formal run completes its budget, and selection is the earliest strict maximum selection macro target-cell AUC. G4 matched exposures preserve original domain mixture/update count while replaying the smaller target subset. MODEL_FIDELITY and the design specify the objective and model-specific interventions.

With the recorded local data/artifacts restored, derived readouts can be reproduced without training or changing model selection:

```bash
python -B experiments/cm19_collect.py --split development
python -B experiments/cm19_evidence.py --split development
python -B experiments/cm19_collect.py --split test
python -B experiments/cm19_evidence.py --split test
python -B experiments/cm19_final_audit.py
python -B experiments/cm19_artifacts.py
python -B experiments/cm19_figures.py --split development
python -B experiments/cm19_figures.py --split test
python -B experiments/cm19_inventory.py
```

These commands require complete original receipts; they regenerate tables/indices/figures. TEST_FREEZE is a historical pre-test binding, retained unchanged. Fresh training requires separate output/state paths and the same declared source/data manifest; completed evidence is not overwritten to rerun in place. See CURRENT_WORK for recovery history, not as a substitute for the final scientific report.
