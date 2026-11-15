# Stage 1 auxiliary-objective attribution

Frozen design: `research_reports/auxiliary_objective_attribution_20261115/DESIGN_SPECIFICATION.md`.
Only B, C, Z-PAIR and Z-EXAMPLE are implemented. Forty new trajectories use two datasets,
seeds 2022–2026, 50 epochs and batch 40960. Ten audited CLEAN trajectories are reused.
The training model, Adam, EMA, data and evaluation utilities are imported from current main;
the B/C adapter originates from pilot implementation `bf36e454adc2d0c569218a7993834f6d56cfc97f`.

Run from the repository root with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

```bash
python -m pytest -q experiments/auxiliary_objective_attribution_20261115/test_core.py
python -m experiments.auxiliary_objective_attribution_20261115.runner prepare --attempt artifacts/auxiliary_objective_attribution_20261115/attempt1 --cache-dir /home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache --pilot-root /home/xzf/Code/phase0-3-20261115
python -m experiments.auxiliary_objective_attribution_20261115.runner smoke --attempt artifacts/auxiliary_objective_attribution_20261115/attempt1 --review research_reports/auxiliary_objective_attribution_20261115/IMPLEMENTATION_REVIEW.json --device cuda:0
python -m experiments.auxiliary_objective_attribution_20261115.runner stage1 --attempt artifacts/auxiliary_objective_attribution_20261115/attempt1 --review research_reports/auxiliary_objective_attribution_20261115/IMPLEMENTATION_REVIEW.json --devices 0,1
python -m experiments.auxiliary_objective_attribution_20261115.runner tables --attempt artifacts/auxiliary_objective_attribution_20261115/attempt1
```

The cache and pilot paths are explicit execution inputs, not runtime source imports.
Without a cache, the frozen CSV loader constructs the same data in RAM. `prepare` copies
only CLEAN metrics/logs and their receipts; it verifies SHA256 identities and preserves
historical files. Training/checkpoint/optimizer states remain in the imported RAM guardian.
`status` reads its job metadata; `release` frees completed states and closes that guardian
after the orchestrator decides that no later state use is required.

Per-epoch `batches_epoch_XX.jsonl.gz` logs contain candidate and accepted original row IDs,
tasks/domains/labels, pair memberships, masks, norms/caps, all effective weights and losses.
Skip codes: 0 accepted, 1 invalid direction (or either member's direction/RMS for a pair),
2 invalid example RMS, 3 near-zero pair margin, 4 near-zero predicted pair change.
The clean probability BCE follows the original reduction. Auxiliary BCE is twice the
accepted target-entry mean, added with coefficient .25. Its entry coefficient is .5/M,
relative to clean 1/(2B), hence B/M. Zero arms compute the normal validity mask before
replacing accepted perturbations with exact zeros.

`tables` verifies complete five-seed budgets, validation selection/state identities and
natural loader row order, then exports six-cell/Macro/Traffic/Pooled endpoints and paired
statistics, curves, costs, effective weights and accepted-support overlap. Both-empty
accepted support has Jaccard 1. Same keyed candidates can have different adaptive masks.
Route decisions and scientific interpretation belong to the orchestrator. This module
has no Test evaluator, Stage 2 implementation, HPO or performance early stopping.
