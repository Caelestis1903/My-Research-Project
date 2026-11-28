# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`three_directions_20261128/B1` completed all 30 frozen trajectories under the foreground supervisor. The B numerical implementation and the corrected shared runtime passed independent review. [B numerical review](B_RUNTIME_IMPLEMENTATION_REVIEW.md#2-requirement-verification) · [Final runtime review](B_RUNTIME_FINAL_REVIEW.md)

## 2. Git and worktree state

The run used `/home/xzf/Code/phase0-3_three_directions_20261128` at detached HEAD `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`. The reviewed study package was present as untracked worktree files. The approved GA31 callback amendment in the vendored `objectives.py` was a tracked modification. The frozen B1 source hash was `d6a1f51c5999e48440f40b26166dfa2f8b222c90d97527c3702b3a95342c2b7f`. The roster file SHA-256 was `93c530a5c45ebd08e73944bb2079febec68f9bf37a5ce34a2a51ff46baaeac49`. [Frozen roster](../../artifacts/three_directions_20261128/B1/ROSTER.json) · [Source audit](../../artifacts/three_directions_20261128/SOURCE_AUDIT.json)

## 3. Configuration

M3oE fixed1 ran in FP32 for 50 epochs per trajectory with natural Train batches of 40,960 rows and the original short tail. The CPU shuffle generator used `seed + 11000`. Source PAIR sampling used the static configuration (`dynamic=False`), with one registered condition and 2,048 replacement draws per valid event. The PAIR coefficient was .05. Adam used learning rate .01, β=(.9,.999) and ε=1e−8. The online model and fixed EMA were evaluated on the full Validation cohort every epoch. [Execution design](B_EXECUTION_DESIGN.md) · [Data audit](../../artifacts/three_directions_20261128/DATA_AUDIT.json)

| Dataset | Train rows | Steps per epoch | Tail rows | Coupled weight decay | Fixed EMA ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens | 800,167 | 20 | 21,927 | .0001 | .9828205985452511 |
| KuaiRand | 2,006,132 | 49 | 40,052 | .00001 | .995 |

| Arm | Frozen configuration |
| --- | --- |
| B0 | Source v3 `P00` mixed Adam. |
| B1 | Source EXP080 `A21` collapsed shared PAIR history, γ=0. B3 measures the γ=.5 old-state correction on the same trajectories. |
| B2 | Source EXP080 `A21` with γ=.5 correction on the shared PAIR gradient. |

Python 3.13 and PyTorch 2.9.0+cu126 ran on two RTX 3090s. Each card received 15 trajectories. All launches met the measured per-card limits of less than 85% GPU memory and power and the host limit of less than 90% RAM. Host telemetry came from `/proc/meminfo`. The largest simultaneous worker count observed was 26. The ten scoped stop controls were clear at dispatch and final verification. [Dispatch telemetry](../../artifacts/three_directions_20261128/B1/DISPATCH.jsonl) · [Stop controls](../../artifacts/three_directions_20261128/B1/STOP_CONTROLS.json) · [Capacity receipts](../../artifacts/three_directions_20261128/CAPACITY.json)

## 4. Seeds

2022, 2023, 2024, 2025 and 2026 ran with all three arms on both datasets. [Frozen roster](../../artifacts/three_directions_20261128/B1/ROSTER.json)

## 5. Run status

| Dataset | Complete / planned | Verified epochs | Logical Train updates |
| --- | ---: | ---: | ---: |
| MovieLens | 15 / 15 | 750 | 15,000 |
| KuaiRand | 15 / 15 | 750 | 36,750 |
| Total | 30 / 30 | 1,500 | 51,750 |

The first launch was 2026-11-28 at 20:32:33.908 CST and the verified stage status was written at 21:33:15.576 CST. Elapsed wall time was 3,641.67 seconds, and summed worker time was 55,262.60 seconds. The supervisor exited with code 0. Attempt receipts record 51,750 physical Train steps and 1,500 epochs with both Validation readouts. Test accesses were zero. [Stage status](../../artifacts/three_directions_20261128/B1/STAGE_STATUS.json) · [Attempt costs](../../artifacts/three_directions_20261128/B1/COST.json)

All ten dataset-seed groups shared initialization and matched exact epoch order, per-batch context, row hashes, selected PAIR condition and task, pair-index hash, proposals, weights and endpoint exposure across B0/B1/B2 through the natural tail. The final audit linked all 51,750 batch rows to 1,500 acknowledged epoch records. All 20 B1 and B2 trajectories recorded exact-zero first-step difference and correction with zero old-state forwards and autograd calls. At step two each recorded one old-state forward, one autograd call and a nonzero gradient difference. During execution, a separate process reopened still-retained epoch-1 packets on both datasets. [Full verification log](../../artifacts/three_directions_20261128/B1/RUNNER_VERIFICATION.log) · [Per-run logs and receipts](../../artifacts/three_directions_20261128/B1/runs/)

## 6. Raw metrics

| Artifact | Rows | Contents |
| --- | ---: | --- |
| [ENDPOINTS.json](../../artifacts/three_directions_20261128/B1/ENDPOINTS.json) | 180 | E10, E50 and selected Validation for both readouts. Macro, sample-weighted, pooled, domain, task and six-cell AUC and LogLoss. |
| [SELECTED.json](../../artifacts/three_directions_20261128/B1/SELECTED.json) | 60 | Selected epoch, model hash, AUC and same-checkpoint LogLoss for every trajectory and readout. |
| [SIX_CELL.json](../../artifacts/three_directions_20261128/B1/SIX_CELL.json) | 360 | Selected domain-by-task AUC, LogLoss and count. |
| [Per-run epoch and batch logs](../../artifacts/three_directions_20261128/B1/runs/) | 1,500 epochs and 51,750 Train updates | Train mixed loss and real BCE, PAIR exposure and margin, gradient diagnostics, timings, epoch Validation metrics and acknowledged RAM receipts. |

For each published selection, the epoch was the earliest occurrence of the maximum recorded six-cell Macro AUC on its readout. The complete metric row matched that epoch's evaluation, including LogLoss. A fresh process reopened and decoded all 60 compact selected packets and checked the payload, packet and model hashes against the journal and stage receipts. All 120 named A1 packets were present at their recorded sizes during final verification. [Verified B1 model readbacks](../../artifacts/three_directions_20261128/B1/VERIFIED_MODEL_READBACKS.json) · [Verification log](../../artifacts/three_directions_20261128/B1/RUNNER_VERIFICATION.log)

| Dataset | B3 observed valid events | B3 old-state forwards | B3 reported observer seconds | B2 old-state forwards | B2 reported probe seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens | 5,000 | 4,995 | 2,389.40 | 4,995 | 2,439.87 |
| KuaiRand | 12,250 | 12,245 | 4,883.25 | 12,245 | 3,915.44 |

B3 observer seconds are recorded separately from B1 scientific step time. The per-batch receipts contain raw norms, cosines, correction ratios, old-state forward and autograd counts, and timings. [Verification log](../../artifacts/three_directions_20261128/B1/RUNNER_VERIFICATION.log) · [Per-run batch logs](../../artifacts/three_directions_20261128/B1/runs/)

## 7. Log locations

- The dispatch log preserves 30 launches, 3,080 allocation acknowledgements and 30 exits. The supervisor emitted zero lines to its combined stdout and stderr log. [Dispatch log](../../artifacts/three_directions_20261128/B1/DISPATCH.jsonl) · [Supervisor stdout and stderr](../../artifacts/three_directions_20261128/B1/SUPERVISOR.log)
- Each dispatch directory contains combined worker stdout and stderr in `worker.log`, resource samples in `allocations/` and the worker outcome in `OUTCOME.json`. [Dispatch directories](../../artifacts/three_directions_20261128/B1/dispatch/)
- Each run directory contains `RAM_JOURNAL.json`, `COMPLETE_VERIFIED.json`, 50 epoch metric files and the attempt's manifest, identity, resume, batch and final attempt logs. [Run directories](../../artifacts/three_directions_20261128/B1/runs/)

## 8. Runtime errors

Zero failed attempts, CUDA OOMs, retries or worker tracebacks. Each worker emitted the inherited PyTorch TF32 API deprecation warning and the warning from wrapping a non-writable NumPy Validation array. All 30 attempt receipts record `COMPLETE`. [Worker logs](../../artifacts/three_directions_20261128/B1/dispatch/) · [Attempt receipts](../../artifacts/three_directions_20261128/B1/COST.json)

## 9. Missing runs

Zero. Every frozen key has verified E50 completion and both Validation-selected readouts. [Stage status](../../artifacts/three_directions_20261128/B1/STAGE_STATUS.json) · [Verified model readbacks](../../artifacts/three_directions_20261128/B1/VERIFIED_MODEL_READBACKS.json)
