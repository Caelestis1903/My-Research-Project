# 2026-10-15 execution evidence (updated 2026-10-16)

Overall **PARTIAL**. No new research training. User dataset root `/home/xzf/data`; no subagents used. The active goal is not complete.

- [Execution report](execution_report.md): task matrix, environment, fixes, actual runs, recovery commands and remaining requirements.
- [Results](experiment_results.md): per-seed/cell AUC and loss, bootstrap, selected/grid boundary, figure.
- [Mechanism analysis](mechanism_analysis.md): scoped evidence, contradictions, size/difficulty, competing explanations and falsification.
- `execution_status.json`: machine-readable package states; `publication_status.json`: actual Git delivery status.
- `artifact_inventory.json`, `missing_artifacts.json`, `artifact_recovery_search.json`: hashed accessible inputs and explicit missing provenance.
- `dataset_preflight_*.json`, `dataset_prediction_binding.json`, `environment.json`: real data paths, hashes, split/label/schema and environment.
- [Callgraph](runner_objective_optimizer_callgraph.md), [audit specification](audit_design.md), [self-review](implementation_review.md).
- `audit_evidence.py`, `dataset_preflight.py`, `analyze_tables.py`, `bootstrap_ao04.py`, `make_figures.py`, `build_reports.py`, `run_data_tests.py`: reproducible evidence processing. Source input files retain their existing repository paths; full clone plus `/home/xzf/data` required to rerun.

Input snapshots are indexed, not guaranteed immutable by their local path. New evidence outputs are versioned, datasets/weights not duplicated. `output_checksums.json` verifies the deliverable snapshot except publication/checksum metadata. Publication identity is determined from Git commit and remote branch, not from an embedded self-referential commit hash.
