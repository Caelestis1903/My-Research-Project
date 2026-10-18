# N19 reproduction and artifact layout

Run from repository root on `codex/execute-n19-2026-10-18`. The archived
source_start_utf8.md is a decoded copy of the GB18030 user entry file. The revised
entry, DESIGN_SPECIFICATION and numbered addenda govern this stage. N18 remains
in `research_plans/2026-10-18/execution/` with its original scope and results.

Environment: ENVIRONMENT.json. Actual executable:
`/home/xzf/miniconda3/envs/phase0-3-env/bin/python`. Full source-level data loading
and split/preprocessing definitions are in stage_S1_data.py. N19 reuses the fixed
seed2022 S1 populations from:

```
/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016/local_cache/kuairand_2022.pkl
/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016/local_cache/movielens_2022.pkl
```

These caches are prerequisite local data, not downloaded by the training queue.
See config.json/data_source for raw file identity, task definitions and frozen
row-ID hashes. Native KR tab0/tab1/tab4 map to domain0/1/2. ML domains correspond
to ages{1,18}/{25}/{35,45,50,56}; the intentionally imbalanced historical training
population is34500/3000/150000 before common5% train-meta exclusion.

An individual full run:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -u experiments/n19_train.py \
  --suite natural --dataset kuairand --method rank_protected --seed 2022 --device cuda:0
```

KR defaults50epochs/lr.003; ML100epochs/lr.001. Batch40960. A run refuses to
overwrite an existing folder unless --resume is explicit. Resume restores last
committed model/optimizers/random streams and truncates only uncommitted metric
rows. No early stopping; selected checkpoint search is independent of budget.
P3 rank selection is eligible from epoch10 only. --engineering truncates visits
and produces a distinct receipt; never mix it with formal runs.

Queue execution (do not launch a second queue over an active manifest):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -u experiments/n19_queue.py \
  --manifest research_plans/2026-10-18/N19/queue_manifest.json
```

The queue reloads reviewed additions atomically, admits each GPU independently,
and allows at most5 active jobs/device below90% device memory. A manifest with
sealed=false intentionally remains alive after the pending tasks drain. Do not
silently rerun failed tasks or remove negative seeds. Raw return codes, process
IDs, device assignments and commands are preserved in queue events/status/logs.

Development aggregation and stage analysis (read completed formal runs only):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/n19_summarize.py --phase development
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/n19_dynamics.py
```

Per-run files contain exact config, source_snapshot, all epoch trajectories,
per-cell validation metrics, selected/final predictions, completion receipt and
mechanism-specific diagnostics. Large full model/optimizer/random states reside
in `/home_new/xzf_n19_artifacts/<run_id>/`; record-specific config states the path.
Checkpoint names are selected.pt,last.pt and the declared stage snapshots.
Physical active-factor exports are produced at frozen final evaluation. They
do not replace the warmup supernet states required for exact resumption.

Test evaluation requires a finalized TEST_FREEZE.json listing the allowed runs.
Until all primary/alternative decisions and required extensions finish, this file
must not be created. n19_evaluate.py performs no optimizer steps; it evaluates
selected/final once, records cell metrics/mechanism diagnostics and a freeze hash.
Then n19_summarize.py --phase test aggregates and n19_figures.py --phase test
exports PNG/PDF figures. PROTOCOL_LIMITS.md applies to all interpretation.

This file documents execution; it does not assert that the still-active N19
matrix or final analysis is complete. See EXECUTION_STATUS.json.
