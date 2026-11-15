# Scheduler and Resource Budget

Stage2 NDP-A measured ML about 1,205–1,437 worker seconds per 50-epoch trajectory; KR about 2,618–3,984 seconds. Peak allocated memory about 2.51 GB ML and 2.92 GB KR. TV-GRM computes clean/view gradients and adds state, so smoke must measure actual cost.

40 new runs: 20 ML + 20 KR, 2,000 epochs. Compute exact updates from loader lengths (ML 1,000 steps/epoch; KR 2,450), not a guessed constant. Historical CLEAN/NDP-A are reused, not new compute.

Use existing per-GPU admission (<85% memory/power; host RAM <90%). First allocating worker on unmeasured GPU runs alone. Never reduce batch/epochs/precision/seeds on OOM; mark ENGINEERING_BLOCKED and preserve logs. Finish current full matrix/report before optional work.
