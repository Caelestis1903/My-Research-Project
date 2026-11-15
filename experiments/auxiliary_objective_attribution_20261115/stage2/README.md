# Stage 2 NDP-A

The frozen `STAGE2_DESIGN_SPECIFICATION.md` authorizes ten NDP-A trajectories:
MovieLens/KuaiRand × seeds 2022–2026, 50 epochs, batch 40960, alpha .03 and mixture lambda .25.
Ten audited CLEAN trajectories serve as clean references and canonical zero-delta counterparts
after CPU and both real full-batch zero-equivalence checks pass.

NDP-A uses every natural row for both tasks. Each task receives its own detached BCE-ascent
embedding view; only that task's BCE enters its view loss. The total objective is
`.75*clean + .25*.5*(view0_task0_mean_BCE + view1_task1_mean_BCE)`.
Near-zero directions receive exact zero delta while retaining their natural weight.
When both deltas are zero, the implementation returns the source clean BCE graph directly.
No sampler, rejection, projection, extra auxiliary mean or new optimizer is introduced.

From the repository root, use `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

```bash
python -m pytest -q experiments/auxiliary_objective_attribution_20261115/stage2/test_core.py
python -m experiments.auxiliary_objective_attribution_20261115.stage2.runner prepare --attempt artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1 --cache-dir /home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache
python -m experiments.auxiliary_objective_attribution_20261115.stage2.runner smoke --attempt artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1 --review research_reports/auxiliary_objective_attribution_20261115/STAGE2_IMPLEMENTATION_REVIEW.json --device cuda:0
python -m experiments.auxiliary_objective_attribution_20261115.stage2.runner stage2 --attempt artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1 --review research_reports/auxiliary_objective_attribution_20261115/STAGE2_IMPLEMENTATION_REVIEW.json --devices 0,1
python -m experiments.auxiliary_objective_attribution_20261115.stage2.runner tables --attempt artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1
```

`prepare` binds the final Stage 1 decision/review/source/support/weight provenance and indexes
the previously copied CLEAN evidence without copying data caches or rescanning Stage 1 traces.
The implementation engineer can call `runner.smoke` before final independent review; formal
dispatch requires final review PASS, matching source hash, full-batch smoke PASS and the
zero-counterpart receipt. CPU-check evidence is recorded as `CPU_CHECKS.json` before smoke.

Per-batch gzip JSONL contains the full-support row-order hash, B/2B, 12 cell-label natural
counts and coefficients, BCE numerators/denominator, task/cell zero-direction summaries,
delta norms and target/non-target drift distributions. It omits per-entry arrays.
CUDA synchronized probe/construction/training times and separate validation time are recorded.
Each epoch stores both validation readouts, independent earliest strict maximum selection,
state/order hashes, RAM/throughput and the batch-log hash.

`tables` exports 20-trajectory validation endpoints/curves, costs, natural weights and
five-seed NDP-A−CLEAN paired statistics for selected/E10/E30/E50. It separates online/online,
EMA/EMA and total EMA/clean-online comparisons and writes no scientific decision.

The imported RAM guardian owns model/Adam/EMA/RNG/loader/selected states. Stage 2 exposes no
release command: selected states are retained for the orchestrator's later conditional use.
There is no HPO, Test, NDP-B or protection-mechanism implementation in this package.
