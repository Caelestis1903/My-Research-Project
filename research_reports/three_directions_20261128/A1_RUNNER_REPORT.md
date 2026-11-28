# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`three_directions_20261128/A1` completed all 40 frozen trajectories under the foreground supervisor. The A implementation and shared runtime passed independent review. [A review](A_IMPLEMENTATION_REVIEW.md) · [Runtime review](B_RUNTIME_FINAL_REVIEW.md)

## 2. Git and worktree state

The study ran from `/home/xzf/Code/phase0-3_three_directions_20261128` at detached HEAD `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`. The reviewed `experiments/three_directions_20261128/` sources were present as untracked worktree files, and the approved GA31 `objectives.py` amendment was a tracked modification. The runner's frozen source hash was `8bccafc3853f5f1ef76c3acdf4380c3cabb520a1ffe97eee703a4d8e9058c2ca`. The roster file SHA-256 was `cb8b0cd85f3b128233b5c60be9941efda68cee21595080b211e1d0498992eecb`. [Roster](../../artifacts/three_directions_20261128/A1/ROSTER.json) · [Source audit](../../artifacts/three_directions_20261128/SOURCE_AUDIT.json)

## 3. Configuration

M3oE fixed1 ran in FP32 with the original natural Train batches and both online and fixed-EMA readouts on the complete canonical Validation split. Each trajectory used 50 epochs, batch size 40,960 with the natural tail, Adam learning rate .01, β=(.9,.999), ε=1e−8, and a CPU shuffle generator seeded with `seed + 11000`. Each update drew 2,048 Train-global macro6 pairs at τ=1. [A execution design](A_EXECUTION_DESIGN.md) · [Data audit](../../artifacts/three_directions_20261128/DATA_AUDIT.json)

| Dataset | Train rows | Steps per epoch | Tail rows | Adam weight decay | Fixed EMA ρ | Gaussian κ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MovieLens | 800,167 | 20 | 21,927 | .0001 | .9828205985452511 | .05 |
| KuaiRand | 2,006,132 | 49 | 40,052 | .00001 | .995 | .10 |

| Arm | BCE | Pair view | Pair coefficient |
| --- | --- | --- | ---: |
| B0 | Clean | Clean, forward only | 0 |
| P0 | Clean | Clean | .05 |
| G0 | GA31-R3 D0 half replacement | Clean, forward only | 0 |
| GP0 | GA31-R3 D0 half replacement | Independent Gaussian endpoints on D0, clean elsewhere | .05 |

The reviewed Python 3.13 environment used PyTorch 2.9.0+cu126 on two RTX 3090 GPUs. Measured host RAM came from `/proc/meminfo`. Each launch met the current limits on its GPU and host RAM. All ten scoped stop controls were clear at dispatch and final verification. [Dispatch telemetry](../../artifacts/three_directions_20261128/A1/DISPATCH.jsonl) · [Stop controls](../../artifacts/three_directions_20261128/A1/STOP_CONTROLS.json)

## 4. Seeds

2022, 2023, 2024, 2025 and 2026, each run with all four arms on both datasets. [Frozen roster](../../artifacts/three_directions_20261128/A1/ROSTER.json)

## 5. Run status

| Dataset | Complete / planned | Verified epochs | Logical Train updates |
| --- | ---: | ---: | ---: |
| MovieLens | 20 / 20 | 1,000 | 20,000 |
| KuaiRand | 20 / 20 | 1,000 | 49,000 |
| Total | 40 / 40 | 2,000 | 69,000 |

The first launch was 2026-11-28 at 19:35:56 CST and the verified stage status was written at 20:31:05 CST. Elapsed wall time was 3,309.8 seconds and summed worker time was 63,302.2 seconds. The supervisor exited with code 0. Physical Train steps and Validation epochs matched the logical counts. Test accesses recorded as zero. [Stage status](../../artifacts/three_directions_20261128/A1/STAGE_STATUS.json) · [Attempt costs](../../artifacts/three_directions_20261128/A1/COST.json)

The audit checked all 2,000 epoch receipts and the frozen tail sizes. Each dataset and seed matched BCE row-order hashes, pair hashes, conditions and weights across all four arms at every step. The first Gaussian BCE draw in all 20 G0/GP0 trajectories matched its archived key, noise SHA, D0 count and BCE/D0 ordered row hashes. [Per-run traces and epoch receipts](../../artifacts/three_directions_20261128/A1/runs/) · [Archived Gaussian replays](../../artifacts/three_directions_20261128/SOURCE_AUDIT.json)

## 6. Raw metrics

| Artifact | Rows | Contents |
| --- | ---: | --- |
| [ENDPOINTS.json](../../artifacts/three_directions_20261128/A1/ENDPOINTS.json) | 240 | E10, E50 and selected Validation for both readouts. Macro, per-domain, per-task, six-cell, sample-weighted and pooled metrics. |
| [SELECTED.json](../../artifacts/three_directions_20261128/A1/SELECTED.json) | 80 | Selected epoch, model hash, AUC and same-checkpoint LogLoss for each trajectory and readout. |
| [SIX_CELL.json](../../artifacts/three_directions_20261128/A1/SIX_CELL.json) | 480 | Selected domain-by-task AUC, LogLoss and count. |
| [Per-run epoch and batch logs](../../artifacts/three_directions_20261128/A1/runs/) | 2,000 epochs, 69,000 Train updates | Train total loss and real BCE loss, ordered exposure and diagnostics, epoch Validation metrics and RAM receipts. |

Each published selected readout matched the earliest epoch at the maximum recorded six-cell Macro AUC. Its complete metric row matched that epoch's Validation evaluation. The supervisor published 120 verified compact model receipts. A fresh process reopened and decoded all 120 live packets, matched each receipt and rechecked the payload hashes. Counts were 40 online selections, 40 fixed-EMA selections, 20 B0/P0 fixed-EMA E05 states and 20 B0/P0 fixed-EMA E10 states. [Verified RAM readbacks](../../artifacts/three_directions_20261128/A1/VERIFIED_MODEL_READBACKS.json)

## 7. Log locations

- The dispatch log preserves 40 launches, 4,080 allocation acknowledgements and 40 exits. [DISPATCH.jsonl](../../artifacts/three_directions_20261128/A1/DISPATCH.jsonl)
- Each dispatch directory contains the full worker stdout and stderr in `worker.log`, allocation receipts in `allocations/` and the worker outcome in `OUTCOME.json`. [Dispatch directories](../../artifacts/three_directions_20261128/A1/dispatch/)
- Each run directory contains `RAM_JOURNAL.json`, `COMPLETE_VERIFIED.json`, 50 epoch metrics files and the attempt's manifest, identities, resume receipt, final attempt receipt and `batches.jsonl`. [Run directories](../../artifacts/three_directions_20261128/A1/runs/)

## 8. Runtime errors

Zero failed attempts, CUDA OOMs, retries or worker tracebacks. All 40 worker logs retain the PyTorch TF32 API warning and the warning emitted when the Validation evaluator wraps a non-writable NumPy array. All 40 attempt receipts record `COMPLETE`. [Worker logs](../../artifacts/three_directions_20261128/A1/dispatch/) · [Attempt receipts](../../artifacts/three_directions_20261128/A1/COST.json)

## 9. Missing runs

Zero. All frozen keys have a verified E50 completion and both Validation-selected readouts. [Stage status](../../artifacts/three_directions_20261128/A1/STAGE_STATUS.json) · [Verified RAM readbacks](../../artifacts/three_directions_20261128/A1/VERIFIED_MODEL_READBACKS.json)
