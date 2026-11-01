# GATS02: augmentation placement, task delta projection and time averaging

Implementation follows [DESIGN_SPECIFICATION](../../research_reports/2026-11-02_gradient_aug_time_select/DESIGN_SPECIFICATION.md).
The unchanged GA31-R3/GAM01/AIGM01 packages supply natural preprocessing, fixed1 model, Gaussian view keys, metrics, Adam, RAM IPC and resource queries.

Use the project Python explicitly, from repository root:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_time_select_20261102/tests
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode check --device cuda:1 --code-commit REVIEWED_SHA --run-dir artifacts/gradient_aug_time_select_20261102/attempt_TIMESTAMP_check
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode all --devices 0,1 --code-commit REVIEWED_SHA --stage0-receipt artifacts/gradient_aug_time_select_20261102/attempt_TIMESTAMP_check/stage0_checks.json --run-dir artifacts/gradient_aug_time_select_20261102/attempt_TIMESTAMP
```

Run the real CUDA check only after independent review. `all` requires a PASS receipt matching the reviewed source manifest and execution SHA. Every attempt directory is new. CSV defaults are the relocated real paths in the pinned protocol; the existing loader checks six arrays, rows, encoders and scalers. Models, optimizer, shadows, RNG and selected/E10/E50 snapshots stay in RAM, including interprocess transport. The coordinator must remain alive until all training, global selection freeze and unified test finish; it releases states afterward. Disk artifacts contain only numerical data, configuration, hashes and logs.

Twenty complete development trajectories use seeds 2022/2024. Fixed EMA is always retained; A0 also has delayed5/SWA20; A1/B have adaptive and independent shuffled-rho replay. Replay recalculates the exact online path, verifies every step hash, and counts all recomputation. B1 per-sample probabilities and row IDs are compressed numerical NPZ files, with keyed mask reconstruction metadata in CSV; no model tensors are in these files. SWA starts at E31 and E10 is unavailable.

Matched controls consume the eight-run additional budget, ordered R2/R1/R3, only when the validation gate or registered loss/worst-cell trade-off needs them. The global cap is 40 unique trajectories. Failed scientific gates stop the candidate while the remaining registered queue continues. A default practical pass goes to confirmation. A still exploratory eta=.5/delay10 candidate can consume four remaining runs for its two seeds plus its own two-seed matching control; delay10 uses a frozen seed1729 random ten-off control with the same cumulative dose. Optional tau2 requires a reviewed design: the coordinator writes `conditional_design_request.json` and retains RAM while awaiting `conditional_design_resolution.json` containing `{"reviewed": true, "action": "STOP", "reason": "..."}` or a reviewed source revision handled by the orchestrator. There is no implicit tau formula. Already settled default/control decisions do not cause a new parameter grid.

Historical numerical baseline inputs default to `research_reports/2026-11-02_gradient_aug_time_select/historical_aigm_baseline/` (override with `--history-dir`). They are byte-preserved evidence from `26-11-01_v3@6f5742c2085c1e49d1767b0a7079bb9f28160eda`; their source receipt is beside them. Reuse requires current A0/A1 development checkpoint hashes and frozen configuration/data identity to match. Otherwise corresponding baselines must be newly trained within the cap. Existing test values are imported only after all current candidates and endpoints have frozen. Baseline numerical reuse does not restore historical weights or add seeds.

Root run artifacts include every run/pass ledger, epoch/batch/view/mask/projection/rho diagnostics, per-kind validation/test metrics, selected endpoint hashes, paired raw values and seed-level mean/sample SD/t95% intervals, source/data/environment receipts, decisions, costs and release receipt. The orchestrator writes the scientific report after the runner completes.
