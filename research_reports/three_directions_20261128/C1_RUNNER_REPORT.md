# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`three_directions_20261128/C1` completed the initial 20 C2/C3 trajectories and verified 20 exact linked controls. The foreground supervisor exited 0. [Runtime release](C1_RUNTIME_RELEASE_REVIEW.md) · [Stage status](../../artifacts/three_directions_20261128/C1/STAGE_STATUS.json)

## 2. Git and worktree state

The run used `/home/xzf/Code/phase0-3_three_directions_20261128` at detached HEAD `70455ed0ebc898ae80275d17f14e7be80f8cfd57`. The approved C1 runtime was present in tracked and untracked worktree files. The released and frozen C1 source hash was `f06680455940737f70e91f8054f57e3366ba1ebc610b80b1ff17a3936220d61f`. The roster hash was `5a197551bb066bba5f6a51860d5759736f7e84777678610eb9e4500f5e1ba457` and its file SHA-256 was `4aca2dbfcfe51bffb12b4f415c73c034b73e34ddf9f7d153c6d0bab3d12dff5e`. The historical audit verified 16 A1/B1 source and provenance blobs against the archive commit. [Archive receipt](../../artifacts/three_directions_20261128/A_B_SOURCE_SNAPSHOT.json) · [Frozen roster](../../artifacts/three_directions_20261128/C1/ROSTER.json)

## 3. Configuration

M3oE fixed1 ran in FP32 for 50 epochs per trajectory with natural Train batches of 40,960 rows and the original tail. The CPU shuffle generator used `seed + 11000`. Native Adam used learning rate .01, β=(.9,.999) and ε=1e−8. The clean pair objective used macro6 sampling, coefficient .05 and temperature 1. C2 and C3 ran the exact A1.P0 warmup through E10 and used soft targets from the corresponding fixed-EMA E10 teacher in E11–E50. Each active condition used up to 4,096 Train candidates, matched C2/C3 banks of up to 2,048 ordered pairs and 2,048 replacement draws per event. Target margins were clipped to [−5, 5]. The online and fixed-EMA readouts were evaluated on full Validation each epoch. [Execution design](C1_EXECUTION_DESIGN.md) · [Frozen configuration](../../artifacts/three_directions_20261128/C1/ROSTER.json) · [Capacity decision](../../artifacts/three_directions_20261128/C0/decisions.json)

| Dataset | Train rows | Steps per epoch | Tail rows | Coupled weight decay | Fixed EMA ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens-1M | 800,167 | 20 | 21,927 | .0001 | .9828205985452511 |
| KuaiRand-Pure | 2,006,132 | 49 | 40,052 | .00001 | .995 |

| Logical arm | Execution |
| --- | --- |
| C0 | Ten linked A1.B0 trajectories. |
| C1 | Ten linked A1.P0 trajectories. |
| C2 | Ten newly trained trajectories with soft targets on a random candidate bank. |
| C3 | Ten newly trained trajectories with soft targets on a historically correct candidate bank. |
| C4 | Frozen status `NOT_TRIGGERED`. |

Python 3.13.5 and PyTorch 2.9.0+cu126 ran on two NVIDIA RTX 3090s. Each card received ten jobs, and the largest observed simultaneous count was 13. All launches met current GPU memory and power below 85% and host RAM below 90%. The measurements recorded CUDA-usable capacity of 25,307,578,368 bytes per card and full worker residency between 1,426,063,360 and 1,692,401,664 bytes across both cards. All ten scoped stop controls were clear at preflight and final verification. [Preflight](../../artifacts/three_directions_20261128/C1/PREFLIGHT.log) · [Dispatch telemetry](../../artifacts/three_directions_20261128/C1/DISPATCH.jsonl) · [Capacity receipts](../../artifacts/three_directions_20261128/CAPACITY.json)

## 4. Seeds

2022, 2023, 2024, 2025 and 2026 completed in each of C2 and C3 on both datasets. The same seed inventory was verified for C0 and C1. [Frozen roster](../../artifacts/three_directions_20261128/C1/ROSTER.json) · [Linked controls](../../artifacts/three_directions_20261128/C1/LINKED_CONTROLS.json)

## 5. Run status

| Dataset | Complete / planned | Committed epochs | Logical Train updates |
| --- | ---: | ---: | ---: |
| MovieLens-1M | 10 / 10 | 500 | 10,000 |
| KuaiRand-Pure | 10 / 10 | 500 | 24,500 |
| Total | 20 / 20 | 1,000 | 34,500 |

The first launch was 2026-11-28 at 22:01:46.696 CST. The verified stage status was written at 23:09:11.290 CST. Elapsed wall time was 4,044.59 seconds and summed worker time was 24,435.13 seconds. Attempts recorded 34,500 physical Train steps and 1,000 physical epochs with both Validation readouts. Test accesses were zero. [Stage status](../../artifacts/three_directions_20261128/C1/STAGE_STATUS.json) · [Attempt costs](../../artifacts/three_directions_20261128/C1/COST.json) · [Verification log](../../artifacts/three_directions_20261128/C1/RUNNER_VERIFICATION.log)

All 20 E1 and E10 records matched the original P0 epoch order, training losses, Validation metrics and selected model identities. At E10, all 20 also matched the original online and fixed-EMA teacher hashes and every committed warmup batch receipt. Each dataset-seed pair matched C2/C3 optimizer, loader and RNG digests and all per-condition bank capacities. Independent trace checks matched the two arms on every E11–E50 event context, current-sampler pair identity, draw-position hash and unique-position count. All 20 first E11 receipts recorded 2,048 bank draws and one gradient-bearing pair forward over 4,096 rows. Fresh reads of a live MovieLens E44 packet and a live KuaiRand E10 packet verified the frozen bank. The E10 packet also verified the online, EMA, optimizer, loader and RNG state digests. [E10 boundaries](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [Verification log](../../artifacts/three_directions_20261128/C1/RUNNER_VERIFICATION.log)

## 6. Raw metrics

| Artifact | Rows | Contents |
| --- | ---: | --- |
| [ENDPOINTS.json](../../artifacts/three_directions_20261128/C1/ENDPOINTS.json) | 240 | E10, E50 and selected Validation metrics on both readouts. Includes macro, weighted, pooled, domain, task and six-cell AUC and LogLoss for 20 physical jobs and 20 linked controls. |
| [SELECTED.json](../../artifacts/three_directions_20261128/C1/SELECTED.json) | 80 | Selected epoch, model hash, AUC and same-checkpoint LogLoss per trajectory and readout. |
| [SIX_CELL.json](../../artifacts/three_directions_20261128/C1/SIX_CELL.json) | 480 | Selected AUC, LogLoss and counts for each domain-by-task cell. |
| [EXPOSURE.json](../../artifacts/three_directions_20261128/C1/EXPOSURE.json) | 640 | Condition-by-phase rows for 20 physical trajectories. Includes event counts, draws, unique realized pairs, current-sampler overlap and uniform-support expectations. |
| [E10_BOUNDARIES.json](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) | 10 | Matched prefix hashes, condition-level bank summaries, array storage and E10 RAM packet sizes for both arms. |
| [COST.json](../../artifacts/three_directions_20261128/C1/COST.json) | 40 | Physical and linked attempt receipts. |
| [VERIFIED_MODEL_READBACKS.json](../../artifacts/three_directions_20261128/C1/VERIFIED_MODEL_READBACKS.json) | 90 | 40 C1 selected, 40 linked selected and ten linked P0 E10 teacher receipts. |
| [Per-run records](../../artifacts/three_directions_20261128/C1/runs/) | 1,000 epochs and 34,500 updates | Train total loss and BCE, Validation results, diagnostic and allocation measurements, and RAM commits. |

Each new selected metric row matched the earliest occurrence of the maximum six-cell Macro AUC on its readout, including LogLoss from that epoch. All linked endpoints matched the original A1 completion. A fresh process reopened and decoded all 90 published compact RAM packets and matched their packet, model and generation identities against the journals and stage publication. [Independent verification](../../artifacts/three_directions_20261128/C1/RUNNER_VERIFICATION.log)

| Dataset | Warmup draws | E11–E50 memory draws | Misses | Bank array bytes per run | E10 RAM packet bytes per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens-1M | 4,096,000 | 16,384,000 | 0 | 835,584 | 27,022,623–27,031,455 |
| KuaiRand-Pure | 10,035,200 | 40,140,800 | 0 | 1,363,366–1,364,828 | 58,755,557–58,772,517 |
| Total | 14,131,200 | 56,524,800 | 0 | | |

## 7. Log locations

- [Roster freeze and live readback gate](../../artifacts/three_directions_20261128/C1/ROSTER_FREEZE.log) · [Preflight telemetry](../../artifacts/three_directions_20261128/C1/PREFLIGHT.log) · [Stop controls](../../artifacts/three_directions_20261128/C1/STOP_CONTROLS.json)
- The dispatch log records 20 launches, 2,080 allocation acknowledgments and 20 exits. Each invocation contains combined worker stdout and stderr, allocation receipts and the outcome. [Dispatch log](../../artifacts/three_directions_20261128/C1/DISPATCH.jsonl) · [Dispatch directories](../../artifacts/three_directions_20261128/C1/dispatch/)
- The supervisor's combined stdout and stderr was empty. Per-run directories contain batch, epoch, completion, attempt and RAM records. [Supervisor log](../../artifacts/three_directions_20261128/C1/SUPERVISOR.log) · [Run directories](../../artifacts/three_directions_20261128/C1/runs/) · [Independent verification](../../artifacts/three_directions_20261128/C1/RUNNER_VERIFICATION.log)

## 8. Runtime errors

Zero failed attempts, CUDA OOM exits, retries or worker tracebacks. Earlier host pressure intermittently blocked new admissions. The MovieLens 2023 and 2024 pairs experienced project library read waits and reached setup 445–503 seconds after launch. Each recovered during the same supervised attempt. Workers emitted the existing PyTorch TF32 API and non-writable NumPy array warnings. [Worker logs](../../artifacts/three_directions_20261128/C1/dispatch/) · [Attempt receipts](../../artifacts/three_directions_20261128/C1/COST.json)

## 9. Missing runs

Zero. All 20 frozen C2/C3 keys have verified E50 completion and both Validation-selected readouts. [Stage status](../../artifacts/three_directions_20261128/C1/STAGE_STATUS.json) · [Verified readbacks](../../artifacts/three_directions_20261128/C1/VERIFIED_MODEL_READBACKS.json)
