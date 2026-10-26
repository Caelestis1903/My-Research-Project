# CODER IMPLEMENTATION REPORT — Stage2

Status: IMPLEMENT COMPLETE. Independent review and subsequent luna-runner
end-to-end profiles remain required. No formal or full-epoch pipeline run was
performed by the coder. No test result file/table was read or used to select
or implement controls, and no research-effect conclusion is made here.

Specification: `DESIGN_STAGE2.md`, SHA256
`97087224b3ca9983c75de45aa38a0d874f86b462ea7e33dd8aed0669405aae09`.
`DECISION_STAGE2.json` was read before implementation. Exactly the two specified
controls, two datasets and four seeds are implemented: 16 new formal runs.

## Files and preservation

New files only: `stage2_protocol.py`, `stage2_run.py`, `stage2_schedule.py`,
`smoke_stage2.py`, `smoke_stage2_movielens.json`, `smoke_stage2_kuairand.json`,
`stage2_base_hashes.json`, `stage2_source_manifest.json`,
`compatibility_stage2.diff` and this report.

Stage1 `protocol.py`, `run.py`, `schedule_runs.py` remain byte-identical; their
SHA256 values are checked against `stage2_base_hashes.json` by both real-data
smoke checks. No Stage1 snapshot, specification, prior report or formal output
was modified. The compatibility diff compares each new entrypoint/scheduler
against the corresponding frozen Stage1 file and includes the new protocol
adapter and smoke script. Actual new code hashes are in
`stage2_source_manifest.json` and are also captured in every run config's
absolute-path source receipts.

The new entrypoint intentionally inherits the entire existing training,
validation, metric, checkpoint, noise schedule and loader implementation.
Controls call the original model-construction and optimizer functions with
parent G/D identities, preserving original modules and complete initial states.
Each runtime verifies its initial-state receipt and all split/feature/array
hashes against the Stage1 parent's **config.json and data_manifest.json only**.
It never reads parent completion/test result files for this match.

## Exact controls and effective configuration

`Gtau0` maps to the G model module and full G gradient path. It uses dynamic
specificity weights and unchanged augmented forward. Its backward calls the
original G implementation with effective tau0, preserving separate per-cell
autograd, all-parameter norms, scales calculation, summation order, and
zero-filled assigned gradients for every trainable parameter. It does not
substitute ordinary backward. All six observed scales equal exactly1.

`Dconst01` maps to the D model and unchanged augmented forward, then replaces
all real/copied weights with `torch.full_like(parent_weights, 0.1,
requires_grad=False)`. The inherited tensors are float32. The six losses are
reconstructed as group means of constant-weight independent BCE. Ordinary D
summed backward retains grad=None for unused parameters. The specificity and
complexity modules stay initialized and computed for observation, but have no
loss gradient and no Adam state/update. The backbone still updates.

Config/completion explicitly identify `stage=stage2`, `arm`/`control_id` as
`Gtau0` or `Dconst01`, `parent_arm`, `gradient_path`, `effective_tau`,
`weight_mode`, and `constant_weight`. Config's `tau` is the effective value:
0 for Gtau0, null for the inactive Dconst01 gradient mechanism. The internal
unused D parent tau default remains.5, affecting no D computation. Unsupported
explicit tau overrides are rejected before a run starts.

Stage2 formal classification deliberately does not require tau.5: every
non-smoke epoch50/max-steps0 intended seed is formal. New formal output:

```
OUT/stage2/DATASET/CONTROL/seed2022/attempt_TIMESTAMP/
```

Profiles/short checks use `OUT/stage2_pipeline/...`; no Stage1 formal/pipeline
tree is reused. Each run creates a new attempt with exist_ok=False. Epoch,
test and mechanism schemas inherit Stage1. `mechanism_metrics.csv` adds a JSON
`auxiliary_parameters` field keyed by parameter name, recording has_gradient
and has_optimizer_state. Best checkpoints preserve that same receipt, and
completion records it as `auxiliary_parameters_selected`. Parent config/data
receipts with SHA256 and initialization/data equality checks are also saved.

## Minimal verification

Completed commands:

```bash
PY=/home/xzf/miniconda3/envs/phase0-3-env/bin/python
$PY -m py_compile experiments/reproduction_20261021/stage2_protocol.py experiments/reproduction_20261021/stage2_run.py experiments/reproduction_20261021/stage2_schedule.py
$PY experiments/reproduction_20261021/smoke_stage2.py --dataset movielens --device cuda:0
$PY experiments/reproduction_20261021/smoke_stage2.py --dataset kuairand --device cuda:0
```

Both datasets PASS. Each uses one deterministic real-training-only balanced
510-row batch and170 copied rows, seed2022 with the actual first-step noise
seed. No validation/test evaluation occurs in this script.

For both controls: model module and full initial-state receipt equal the
parent, actual frozen parent data/config receipts match, forward predictions,
augmented independent BCE and routing match, and a hook at skip_conn confirms
the entire post-noise embedding tensor is bitwise equal to the parent.
Real-row embeddings equal unperturbed lookup and appended rows contain noise.

Gtau0: weights and six cell losses equal G exactly; all scales exactly1;
every assigned gradient equals the manually summed per-cell gradients with
explicit missing-coordinate zeros, bitwise. The existing ordinary-backward
contrast is audited by backbone/specificity/complexity group, including
None-vs-zero names, one-step update differences and resulting prediction
differences. MovieLens max prediction difference after one step is
1.7881393432617188e-7, KuaiRand1.1920928955078125e-7. These are basic numerical
path observations, not scientific effect estimates; detailed gradient/update
group values are in the smoke JSONs.

Dconst01: every weight equals float32(.1)=0.10000000149011612, requires_grad=False;
all six loss formulas agree with.1 times unweighted augmented cell BCE within
float32 tolerance. Weight probe std0, min=max=float32(.1); mean reduction
rounding is explicitly retained. Auxiliary gradient norm0; all six auxiliary
parameter gradients and Adam states absent; auxiliary tensors unchanged and
backbone tensors changed after one optimizer step. Both control IDs pass
explicit formal-versus-smoke routing checks. Stage1 source hashes are unchanged.

## Reviewed runner commands remaining

After independent PASS or corrected PASS WITH FIXES, luna-runner should run:

```bash
PY=/home/xzf/miniconda3/envs/phase0-3-env/bin/python
$PY experiments/reproduction_20261021/stage2_schedule.py --profile --dataset both --gpus 0,1
$PY experiments/reproduction_20261021/stage2_schedule.py --dataset both --gpus 0,1
```

The four sequential actual one-epoch full-batch40960 profiles are mandatory
before the queue and are not formal runs. `OUT/stage2_profiles.json` is separate
from Stage1 profiles. Queue defaults to both controls×two datasets×all four
seeds and does not impose a second seed2022 gate. It inherits the reviewed
per-PID outstanding-reservation memory accounting, 15%+384MiB margin,
strict90% projected memory and at most5 owned processes/GPU, as well as unknown
profile preflight receipts. Failures halt new launches and preserve attempts;
completed Stage2 keys skip and duplicate valid completions raise. No batch or
other scientific setting is changed to obtain concurrency.

No resume path is introduced; reruns always create a new attempt. Parent
references must be accessible under OUT/formal for initial-state/data receipts.
The runtime cache is read from OUT/cache; bulky artifacts remain external.
