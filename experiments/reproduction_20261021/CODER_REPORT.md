# CODER IMPLEMENTATION REPORT

Status: IMPLEMENT COMPLETE; independent review and runner end-to-end checks are
still required. No formal training or scientific result analysis was performed
by the coder. Frozen specification: `DESIGN.md`, SHA256
`3a1270cc7b434802c2a68f6d0dbcf281470cf63195f266f070fde816341ae65f`.

## Implementation and provenance

- Completed independent `experiments/data_generate/MDMTRec/` and
  `experiments/grad/MDMTRec/` packages (basic layers/features/initializers,
  models, trainers, utils) and executable `run.py` in each requested directory.
- Original backbone is copied directly from
  `m3oe/MDMTRec/models/multi_domain/model.py` into each package's `original.py`.
  `fixed.py` changes only learned fusion to fixed coefficients1 and device-local
  allocation. `model.py` is byte-identical to the supplied data-generation
  model, including the original MDMTRec and unused PLE class. Only MDMTRec is
  instantiated. Both packages' `layers.py` equal the supplied layers, whose
  diff against original adds only DBO auxiliary classes.
- Root uploaded files remain untouched. `sources/` stores 21 byte-for-byte
  original/uploaded/audited source files with read-only file permissions;
  `source_manifest.json` records original paths, snapshot paths and SHA256.
  `compatibility.diff` records complete source differences for the copied
  models, trainer adapters and natural MovieLens audited-data port.
- Shared `protocol.py` implements frozen mathematics, paired construction,
  diagnostics and metrics. `run.py` implements training/selection/artifacts;
  package CTRTrainer classes are explicit config adapters to it. Historical
  positional trainer APIs are archived, not claimed compatible.
- `audited_data.py` is the original audited S1 loader with only MovieLens
  role-downsampling/role-count assertions removed; no full-file encoder was
  substituted. `data.py` verifies raw checksums and caches the frozen natural
  arrays with an exclusive lock outside the repository.
- `schedule_runs.py` runs end-to-end one-epoch profiles and memory-bounded
  formal queues. It never changes batch size or scientific hyperparameters.
  It preserves separate attempts and halts new launches after any failure.
- Grad's compatible model is `paired-from-supplied-model / ported`.
  `historical_search.json` records bounded searches of three local Git histories
  and project.zip. No supplied VUG/soft-label/adaptive-mask companion was
  recovered. The archive contains252 Python files; explicit source token
  searches find none. Broad old `*mask*.py` path matches are N19 run directories,
  not a recovered original companion. No missing mechanism was reconstructed.

## Effective semantics

O retains four raw-initialized1 sigmoid fusion modules, the all-parameter main
Adam lr.01 (the historical `~bool` predicate), and a second Adam lr.001 on the
same four fusion tensors. The second update consumes the persistent shuffled
**training** iterator, isolated from the main loader generator. B/D/G fix all
four coefficients1. All use coupled decay1e-5, defaults betas(.9,.999), eps1e-8,
batch40960, no AMP/TF32/clipping/scheduler, max50 epochs, val macro AUC strict
improvement, earliest tie, patience10, and best restoration even at epoch50.

B/O loss is sum of six minibatch group-mean independent Bernoulli BCE terms,
without division by6. D/G use the supplied specificity-only w=.1+.5*q with
non-detached real weights; smallest real training domain copied once, retained
labels, N(0,.1²) noise over the entire appended embedding before STAR/DBO;
appended weights use detached mean real weight. Empty cells contribute graph-
safe zero. Complexity is computed/logged and does not multiply the loss.
G retains six per-cell all-trainable-parameter gradients, tau.5 norm-ratio
scales detached, summed final gradient, with explicit zero gradient assigned
to every disconnected trainable parameter. The root grad trainer's undefined
`logloss_list` empty-cell bug is bypassed by this graph-safe implementation.

All four arms copy the same original canonical backbone for a dataset/seed.
D/G share their complete initial state. Main/auxiliary loader generators and
noise draws are independent, and diagnostics use no optimizer step or extra
forward/noise draw. Unified train-only OOV0/scaler fitting and original row-
order splits apply identically to all four arms. O is a controlled original
model port, not a bitwise reconstruction of unidentified historical runs.

## Basic verification performed

Environment: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`, torch2.9 CUDA.
Commands completed:

```bash
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m compileall -q experiments/reproduction_20261021 experiments/data_generate/MDMTRec experiments/grad/MDMTRec
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/reproduction_20261021/smoke.py --dataset movielens --device cuda:0
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/reproduction_20261021/smoke.py --dataset kuairand --device cuda:0
/home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/data_generate/run.py --prepare-data --device cpu
```

Both dataset basic checks PASS, recorded in `smoke_movielens.json` and
`smoke_kuairand.json`: real-data 510-row balanced diagnostic batch, four model
module paths, train/eval output shapes/contracts, four one-step updates,
original optimizer overlap, canonical/copied state hashes, D/G full initial
state equality, Goff=D bitwise full one-step update equality, empty-domain G
update finite, diagnostic model/RNG/gradient nonmutation.

MovieLens raw SHA matches frozen expectation; natural real train counts are
{0:155949,1:318599,2:325619}. KuaiRand raw SHA matches expectation; actual train
counts satisfy the frozen {0:172782,1:1743898,2:89452} assertion. Both cached
manifests include all split cell label counts/rates, hashes and ordered row IDs.

Canonical backbone SHA for seed2022:
MovieLens `17cb34b972c6f5f921c49c9521af0e62808a69da49510e951140a9f82bd2e69b`;
KuaiRand `fa0217ff302115b24115540f7d9caf404b83e0f8c77bbcecb95c13bc640b462c`.
D/G complete initial state SHA:
MovieLens `1181ad42a7ff74374f1b057f25a98002baf6b9e220c9455adf52bb7a7e367aca`;
KuaiRand `f7f134b5d05621330b0dd78c7aea2fe50fd3cc37652c3611edf4ddb79f865c69`.

The tau0-vs-off check deliberately is **not** asserted bitwise equal. In both
datasets `dbo_complexity.W_Q.weight` and `dbo_complexity.W_K.weight` are
None-vs-zero gradients; historical G zero assignment causes roughly.00996
first Adam updates to those auxiliary tensors where D has no update. Per-cell
floating accumulation also gives small prediction-relevant numerical gradient
and update differences; full parameter-level values are in both smoke JSONs.
For MovieLens the largest recorded gradient difference is ~2.98e-8 and
prediction-backbone update difference is ~1.50e-5. This is an implementation
observation, not research-effect analysis. G−D must be described as the whole
supplied balancing implementation increment; pure amplitude attribution would
need a separately designed/reviewed semantics-matched tau0 control.

## Remaining reviewed-run acceptance work

No actual one-epoch end-to-end pipeline was run by the coder. After independent
review the luna-runner must execute `schedule_runs.py --profile --dataset ...`
using full batch40960 before formal jobs. These pipeline artifacts do not count
as formal results. README contains exact ordered commands: MovieLens profile →
MovieLens seed2022 four arms → remaining MovieLens → KuaiRand profile → KuaiRand
seed2022 → remaining KuaiRand. At most5 processes/GPU and projected memory<90%,
with measured peak memory safety margins. The scheduler does not kill or modify
unrelated processes. Existing formal completed keys are skipped, duplicates
raise, failures remain separate attempts.

Final artifact schema: `epoch_metrics.csv` has epoch/step,
train_optimized_group_sum, train_real_d{d}t{t}_bce, val_{metric}, fusion_* and
is_best; `test_metrics.csv` one selected-checkpoint row;
`mechanism_metrics.csv` one first-batch observation/epoch with JSON vector/
matrix/stat fields. `completion.json` includes valid/status/formal, identifiers,
selected/stopped epoch, steps/time/peak memory, test_metrics, best checkpoint
absolute path and SHA256. No Git commit was made; orchestrator owns delivery.

## REVIEW scheduler correction

Reviewer requested that external GPU use and outstanding reservations be added,
rather than combined with max(). `schedule_runs.py` now queries per-GPU memory
for owned PIDs, and projects current GPU use + sum(max(estimate-PID_current,0))
+ next estimate. Unknown accounting reserves the full amount; inconsistent
process-vs-total samples also fall back to full reservations. Newly launched
PID reservations enter this calculation immediately without double-counting a
synthetic addition to observed GPU use. The five-process and strict90% limits
remain unchanged. First unknown-profile runs require at least2GiB headroom
below90%, are measured sequentially, and write profile_preflight.jsonl; absence
of that headroom rejects the launch. This precheck cannot prove an unknown
first footprint fits, so the runner still measures each configuration.

Basic pure scheduling simulations PASS: external use + pre-initialized jobs,
partly/fully realized reservations, two pending PIDs, actual usage above the
estimate, inconsistent sample fallback, and no owned processes. A mocked
profile GPU above90% is rejected before subprocess.run, and its preflight
receipt is verified. Results: `scheduler_checks.json`. No training was launched
for these checks, and no scientific configuration or training code changed.
