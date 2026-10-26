# M3oE supplied-method reproduction

This directory implements the frozen `DESIGN.md`. The experiment report and
complete statistics are linked below. All four first-round arms use the actual original
M3oE STAR/shared/domain/task expert model and a common train-only encoding port.
`O` retains learned sigmoid fusion and both historical Adam optimizers;
`B` fixes the four fusion values to 1; `D` adds the supplied specificity/copy/noise
bundle; `G` adds the supplied all-parameter amplitude balancing (tau=0.5).

`O` is a controlled original-model port, not a claim of a bitwise historical
training trajectory: train-only encoding, independent loader generators,
canonical paired initialization, and always restoring the validation best are
explicit protocol changes. Its main Adam still includes learned fusion tensors
because the historical `~bool` predicate includes all parameters; its second
Adam updates those same four tensors using a persistent **training** iterator.

The two requested directories have independent complete `MDMTRec` packages and
entrypoints. Root uploaded files remain byte-identical references and are also
archived in `sources/` with SHA256 in `source_manifest.json`. The grad model is
`paired-from-supplied-model / ported`. No original VUG/adaptive-mask source is
claimed or invented. `historical_search.json` records a bounded local history and
archive search. The archive's 252 Python files contain none of the explicit
VUG/soft-label tokens searched; old Git `*mask*.py` path hits are later N19 run
snapshot directories, not a recovered supplied companion.

**Current GPU policy:** subsequent training must use physical GPU0 only; GPU1 is excluded by the latest user instruction. Commands below follow this policy. Earlier raw logs retain the actual historical two-GPU execution. See [GPU_POLICY.md](GPU_POLICY.md).

Use the existing environment:

```bash
PY=/home/xzf/miniconda3/envs/phase0-3-env/bin/python
OUT=/home_new/xzf_m3oe_reproduction_20261021
$PY experiments/reproduction_20261021/smoke.py --dataset movielens --device cuda:0
$PY experiments/reproduction_20261021/smoke.py --dataset kuairand --device cuda:0
$PY experiments/reproduction_20261021/run.py --dataset kuairand --prepare-data
```

After independent review, the runner first executes actual one-epoch pipelines
with full batch size 40960 to validate end-to-end behavior and profile memory:

```bash
$PY experiments/reproduction_20261021/schedule_runs.py --profile --dataset movielens --gpus 0
$PY experiments/reproduction_20261021/schedule_runs.py --dataset movielens --stage seed2022 --gpus 0
$PY experiments/reproduction_20261021/schedule_runs.py --dataset movielens --stage remaining --gpus 0
$PY experiments/reproduction_20261021/schedule_runs.py --profile --dataset kuairand --gpus 0
$PY experiments/reproduction_20261021/schedule_runs.py --dataset kuairand --stage seed2022 --gpus 0
$PY experiments/reproduction_20261021/schedule_runs.py --dataset kuairand --stage remaining --gpus 0
```

Execute stages in this order to preserve first MovieLens seed2022, then remaining
MovieLens, then frozen transfer to KuaiRand. The scheduler refuses missing memory
profiles, allows at most five concurrent processes per GPU, reserves a 15% +
384MiB margin beyond measured peak, and requires projected GPU memory below 90%.
It budgets current GPU use plus each queued PID's outstanding reservation
(estimate minus its observed GPU memory), including pre-initialization jobs.
Unavailable or inconsistent PID accounting falls back conservatively. Unknown
first-profile footprints are measured one process at a time only when at least
2GiB headroom remains below the 90% limit; each preflight is recorded. That
minimum is a precheck, not a claim that the unknown profile must fit. A failed run
halts new launches; existing jobs finish and their artifacts remain intact.

Individual executable package entrypoints (default D/G respectively):

```bash
$PY experiments/data_generate/run.py --dataset movielens --seed 2022 --device cuda:0
$PY experiments/grad/run.py --dataset kuairand --seed 2022 --device cuda:0
$PY experiments/reproduction_20261021/run.py --dataset movielens --arm O --seed 2022 --device cuda:0
```

Shared training code is `run.py` / `protocol.py`; the complete packages' trainer
classes are documented configuration adapters to that same code, not their
historical positional APIs. Dataset construction is a frozen audited S1 source
copy, with only MovieLens role downsampling and associated size assertions
removed. Encoders/scaler fit on the real global training prefix, OOV=0; data split
preserves original CSV order. KuaiRand has no interaction timestamp and its
chronological provenance is not independently reverified. `data.py` verifies
both raw file SHA256 values and current training counts, and writes immutable
cached data under external `OUT/cache` using an exclusive lock.

Each new run gets `OUT/formal|pipeline/DATASET/ARM/seedSEED/attempt_TIMESTAMP`.
`--epochs 1 --smoke` and `--max-steps` always identify pipeline checks, never
formal outcomes. Formal runs use epoch limit50, strict validation six-cell
macro AUC, patience10, earliest tie and best checkpoint restoration at every
exit. A test evaluation occurs once on that selected checkpoint. Normal training
has no AMP/TF32, clipping, scheduler or batch reduction.

Artifacts: `config.json`, `data_manifest.json`, compressed ordered split row IDs,
`epoch_metrics.csv`, `mechanism_metrics.csv`, `test_metrics.csv`, raw last-epoch
metrics, `stdout.log`, `stderr.log`, `completion.json`, and best model + matching
Adam/RNG/loader-generator states in `best.pt`. Source receipts use absolute
source paths and SHA256 rather than an invented Git HEAD. Bulky data and
checkpoints stay external. Macro/per-cell and independent Bernoulli BCE metrics
are explicit; no multiclass normalization is applied to two-task predictions.

Mechanism observation uses the first ordinary batch each epoch, extra
`autograd.grad` calls on its existing forward graph, no optimizer step, no extra
noise draw, no RNG consumption and no parameter/grad mutation. The logged Adam
delta is the actual first minibatch update after observation. Shared-subset
membership follows DESIGN exactly. The tau0 audit is not assumed numerically
identical to balancing off: `smoke_*.json` records all parameter-level gradient
and update differences, including historical zero-vs-None complexity behavior.

## Results and additional frozen controls

Read [the Chinese report](REPORT.md), [scientific interpretation](INTERPRETATION.md),
and [all seed/cell/domain/task tables](analysis/RESULT_TABLES.md).
The Stage2 controls are specified in [DESIGN_STAGE2.md](DESIGN_STAGE2.md):
`Gtau0` retains the complete G accumulation/zero-gradient path at tau=0;
`Dconst01` retains D augmentation with constant non-differentiable weight0.1.
They use distinct `stage2` run identities, with exactly16 formal runs added to
Stage1's32. Stage1 training source files remain unchanged.

```bash
$PY experiments/reproduction_20261021/stage2_schedule.py --profile --dataset both --gpus 0 --output-root "$OUT"
$PY experiments/reproduction_20261021/stage2_schedule.py --dataset both --gpus 0 --output-root "$OUT"
$PY experiments/reproduction_20261021/analyze.py --include-stage2 --input-root "$OUT"
$PY experiments/reproduction_20261021/build_tables.py
$PY experiments/reproduction_20261021/export_results.py --input-root "$OUT"
```

Stage2's scheduler additionally reserves at least3GiB/MovieLens and6GiB/KuaiRand,
based on actual retained CUDA/process memory after a resource interruption.
See `scheduler_operational_amendment.json`; batch size and algorithms did not
change. Invalid/interrupted attempts remain in `raw_results`, excluded from
formal summaries and followed by a complete restart of the same seed.

`validation_decision_inputs/` preserves the original three hashed evidence files
used before the Stage2 decision; the current `analysis/` is the final combined
matrix. `ENVIRONMENT.json` and `requirements.txt` record the actual environment.
Compact raw logs/configuration/metrics are included under `raw_results/`;
checkpoints and row-ID arrays stay under the external runtime root, with
checkpoint paths and SHA256 in `raw_results/checkpoint_index.csv` and per-run
receipts. Dataset source files are not bundled. Re-running the trained models
requires those data/checkpoint files; the tracked code, raw metric tables and
plots are sufficient to audit the reported comparisons without them.
