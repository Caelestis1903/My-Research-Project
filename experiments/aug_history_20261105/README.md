# AUG_HISTORY_20261105

Frozen B implementation for `research_plans/26-11-5/07_FINAL_CODEX_ENTRY.md`.
The probability forward and online Gaussian/Adam path are reused from the pinned source. Fifteen independent non-feedback shadows run on each online trajectory. Model/optimizer/shadow/selected/E10/E50 states remain in a separate RAM guardian.

Use `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` for every command below. Invoke module `experiments.aug_history_20261105.run`; all subcommands accept `--manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1`.

1. `preflight` verifies source, real raw CSV/cache identities, populations and the fixed published-reference table.
2. `validate --suite all` executes focused CPU acceptance checks and reports A/C as `DESIGN_COMPLETE_NOT_IMPLEMENTED`.
3. An independent reviewer writes `research_reports/2026-11-05_aug_history/REVIEW_ACCEPTANCE.json` with `status: PASS` (or `PASS WITH FIXES`), no `blocking_issues`, and the exact `source_hash` from preflight.
4. `validate --suite B --gpu-smoke --devices 0 1` checks original batch40960 source equivalence without performance selection.
5. `train --route B --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --resume` admits concurrent workers per eligible GPU and retains completed endpoints in RAM.
6. `select --route B --split validation`, then `decide --route B --basis validation` freeze all family/matched-control endpoints and validation dispatch.
7. `evaluate --route B --frozen artifacts/aug_history_20261105/attempt1/B/selection_frozen.json --device cuda:0` evaluates only the frozen test endpoints.
8. `continue --decision artifacts/aug_history_20261105/attempt1/B/DECISION.json --resume` emits resolved A/C commands and their full specification paths, or `STOP_AND_REPORT`.
9. `report` writes raw endpoint, paired seed/summary and cost tables. The orchestrator writes the scientific analysis from these tables.

`release` frees RAM and shuts down the guardian after `B/EVALUATED.json` exists. Before that point, retained state is necessary for selection/evaluation. If the guardian is lost, rerun `train --resume`; deterministic replay verifies already committed order/model hashes and records physical replay costs separately. A conflicting replay fails rather than overwriting scalar history.

After correcting a worker failure or reducing concurrency for an OOM, `train --resume` reopens each failed/dead-worker job once in that invocation and retains its last committed RAM epoch. A subsequent failure ends that invocation. A restarted coordinator leaves still-live guardian workers running and does not dispatch a duplicate owner.

Exit codes: 0 complete, 2 configuration/numerical identity error, 3 engineering blocker, 4 measured resource blocker, 5 incomplete five-seed/dependency state. A/C are implemented and independently reviewed only when their frozen decisions trigger them.

## Route A

A uses the frozen twelve-path grouping, four fixed profiles, four grouped controllers, four global controllers and the three U/F shadows. It replays the original G-D0 trajectory from initialization.

Use a separate output such as `artifacts/aug_history_20261105/attempt_A1`, and add `--route A` to preflight/validate/train/select/decide/evaluate/report/continue/release. `preflight` writes `GROUP_MAP.json`. Independent A acceptance is `research_reports/2026-11-05_aug_history/REVIEW_ACCEPTANCE_A.json`, bound to the new source hash. `validate --suite A --gpu-smoke` runs original batch40960 through step20 so the actual K20 q controllers are exercised; these are engineering checks.

`continue --route A --decision artifacts/aug_history_20261105/attempt1/B/DECISION.json --output artifacts/aug_history_20261105/attempt_A1 --resume` reads B's own stored PREPARED/selection identities and writes a continuation receipt only to the A output. It does not reinterpret B under the new source hash. Subsequent A commands write under the new output's `A/` directory. C remains `DESIGN_COMPLETE_NOT_IMPLEMENTED` until triggered.
