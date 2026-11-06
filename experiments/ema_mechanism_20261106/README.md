# EMA mechanism study

Run from this worktree with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run`.
The sole scientific contract is `research_plans/26-11-6/00_EXECUTION_RUNBOOK.md`, its manifest, and the approved `research_reports/2026-11-06_ema_mechanism/DESIGN_SPECIFICATION.md`.

The workflow is `audit`, `preflight`, `validate`, two dataset `smoke` commands, `train`, `analyze --stage 1/2/3`, `controls`, `analyze --stage 4`, `freeze`, `evaluate`, `report`, `release`. All commands take `--manifest` and `--out`; use each command's `--help` for its specific arguments. Training requires a matching validation receipt, both full-batch smoke receipts, and the independent PASS review copied to `out/IMPLEMENTATION_REVIEW.md`.

Online, Adam, all histories and endpoints remain in the dedicated guardian's RAM. Disk holds immutable panel identities, bounded probability/logit NPZ observations, scalar epoch transactions and reports. Restart a failed worker with the same output directory and `train --resume`; its last committed state remains in the guardian. If guardian RAM was lost, the same logical job replays once from initialization and verifies each existing epoch's scientific identity. Completed jobs are retained for the frozen independent Test process; `release` frees only this task's guardian after report/Test artifacts exist.

The scientific budget is fixed at 20 trajectories, 50 epochs, batch 40960 and five seeds 2022–2026. The scheduler starts at most two workers and admits each on a GPU whose memory and power are below 85%, with host RAM below 90%. It does not change the scientific budget or stop other processes.
