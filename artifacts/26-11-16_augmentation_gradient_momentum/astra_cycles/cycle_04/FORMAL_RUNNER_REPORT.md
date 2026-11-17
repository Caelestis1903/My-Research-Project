# Cycle04 formal runner report — initial paused handoff

> Initial paused report; retain unchanged when recovery proceeds. A later recovery/completion report must be separate.

## Status

- Status: `TECHNICAL_PAUSE` at 2026-11-17T00:57:56.781896+00:00 UTC.
- Committed: 275/1000 epochs; 7733/34500 updates.
- Jobs: `{"FAILED": 1, "NOT_STARTED": 0, "PAUSED": 19}`; actual CUDA workers: 0.
- Parent RAM owner PID 2429109 retains 20 states / 573265960 tensor bytes. No release or shutdown was performed.
- Test was not accessed. No research analysis was performed.

## Failed worker

- Key `2266bea2b8fcffcc1c79a6aa124027dbaab2488902056b5d42ddacf13ad70fbc`: kuairand seed 2023, arm `COMPLETION-BCE-025`, device 0, PID 2438112.
- Last ACK: epoch 8; returned train calls and partial-call duration are unknown. Supervisor exit 1 after 556.5374474525452 seconds; `unknown_partial_call=true`.
- Traceback: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/runs/2266bea2b8fcffcc1c79a6aa124027dbaab2488902056b5d42ddacf13ad70fbc/foreground.log` reports `TechnicalPause: NVML measurement failed code=3` at runner.py:703 (`profile.mark`), repeated at runner.py:737 (`profile.close`). Epoch 09 batch log is retained.

## Physical attempts and costs

- 20 distinct formal worker processes: 20 start records, 19 pause records, 20 supervisor-exit records. These are lifecycle events, not separate workers.
- Raw lifecycle/per-worker call and time records: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/physical_attempts.jsonl`. The JSON report lists every formal worker. Per-worker times overlap and are not additive.
- Formal batch logs: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/runs/<key>/attempt1/batches_epoch_*.jsonl.gz`; transaction receipts: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/transactions/<key>/attempt1/epoch_*.json`.
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/costs.jsonl` has 52 engineering-smoke cost rows. No formal diagnostic-cost export was generated for this incomplete run. Unknown partial work remains unestimated.

## Run and pause receipts

- Exact command: `PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.completion_weight_attribution_20261117.runner run --root /home/xzf/Code/phase0-3_tvg_rm_20261116 --attempt /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/ENGINEERING_SIGNOFF.json --devices 0,1`
- Pause: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/TECHNICAL_PAUSE.json`
- Progress: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/attempt1/CURRENT_PROGRESS.json`
- Source SHA-256: `00ce7925c3589deb239f57e8374673e5afe8e3fe155e159dd34c140ae08cd553`; design bundle SHA-256: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`.
- No recovery, retry, source patch, or next cycle was started by this runner. Parent recovery review and fresh `run --recover` dispatch remain pending. The latest user instruction is to complete this Cycle04 and report, then stop without Cycle05.
