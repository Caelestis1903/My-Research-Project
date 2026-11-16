# Cycle03 temporal NQR exposure

The approved seven-file Cycle03 design fixes EARLY10 (epochs 1–10) and LATE10
(41–50). The adapter invokes the frozen C2 operator once per batch. Twenty
new natural-data trajectories use seeds 2022–2026, 50 epochs, batch 40960,
FP32, source Adam and fixed EMA. Twenty explicit numeric CLEAN/NQR controls
supply the registered comparisons. Test remains sealed.

All commands run from `/home/xzf/Code/phase0-3_tvg_rm_20261116`, with
`PYTHONDONTWRITEBYTECODE=1` and the project Python below. Preparation needs a
fresh independent implementation review binding all contributing coder IDs,
source/design hashes, CPU checks and zero blockers. Both implementation
reports must exist before the parent freezes source. The parent alone starts
the frozen TV `serve(str(N))` in a current-conversation foreground session,
after code PASS, and writes the root/source/design-bound STORAGE_OWNER receipt.
Clients never launch an owner. `guardian.json` is private 0600; exclude it
from reports/publication. Tensor states use CPU/RAM/BytesIO exclusively.

```bash
export PYTHONDONTWRITEBYTECODE=1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.checks
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner prepare --review /absolute/implementation-review.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner smoke --devices 0,1 --review /absolute/implementation-review.json
```

N is the sole allowed attempt root:
`/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1`.
Preparation creates numeric identities/cache receipts/jobs. Four real smoke
cases cover each dataset on each card with opposite arms; formal scheduling
has no fixed card assignment. Each case uses the true epoch1 full/tail
samples at its virtual 10→11 or 40→41 boundary: four main updates, four direct
C2 references, two uninterrupted suffix updates. Engineering RAM commits are
ordinals 1/2 at actual steps 2/4; no scientific epoch or completion is claimed.
A fresh `SMOKE_REVIEW.json` must bind the smoke/source/design hashes, all four
profiles, compatible devices/arms and zero blockers before formal launch.

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner run --devices 0,1 --review /absolute/implementation-review.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner handoff
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner tables
```

The foreground supervisor starts only mechanical workers. Independently per
card: current memory/power <85%, host RAM <90%, no pending first-allocation
ACK, and measured full-process physical fit within CUDA-usable capacity.
Startup/allocation resamples precede further admissions. Own compatible
on/off smoke/live maxima carry forward; KR is queued first with seed/arm ties.
No count cap, projected utilization gate or batch/precision change is used.

Epoch transactions write numeric records, RAM-commit, get/hash-verify, then
persist the source/design/endpoint-bound ACK. Restore deep-copies optimizer
and mutable state, then restores loader/RNG. New permutations are checked at
all 1000 epochs; EARLY's first 100 and LATE's first 400 epoch records also
match available reference fields exactly. Source counts have only Adam/EMA;
eight separate phase counters reconcile active/off rows and updates.

A technical failure quiesces workers at the last safe whole-epoch ACK and
retains every physical attempt's cost. `run --recover` attaches to the same
owner after all workers exit; a technical pause requires a source/design/
pause-bound `RECOVERY_REVIEW.json` with independent reviewer, usable verdict,
zero blockers and `same_protocol=true`. Loss of acknowledged RAM pauses.
Stop controls in both workspaces prohibit new admissions and allow the atomic
epoch to commit. Twenty selected online/EMA pairs remain in RAM through the
parent's storage acceptance; neither handoff nor export releases them.

Tables contain 40 trajectories, 400 endpoint rows (selected/E10/E30/E40/E50,
both streams), 2400 endpoint cells, 4000 curves, 24000 cell curves, 34500 new
batch/schedule rows and 207000 new weight rows. All five comparisons match
seed/stream; negative LL deltas are favorable. Historical unmeasured fields
are null with provenance. Smoke and physical replay costs are separate from
useful training counts. Exports report raw arithmetic; the parent evaluates
scientific outcomes after all twenty new trajectories finish.
