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

## Recovery after the first coordinator failure

`attempt_20261102_0005` completed the 20 base trajectories but exited before controls/test while reading historical encoder/scaler identity from the wrong manifest level. Its weights were RAM-only and were lost. Recovery reads the actual nested `features.encoder_hash/scaler_hash`, rejects missing identities, and uses one numerical endpoint core for both old-log CPU preflight and later actual RAM verification.

First run the real CPU preflight against the preserved numerical attempt; it reads no test CSV and creates a fresh receipt directory:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode preflight --recovery-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005 --task-start-unix 1793549242.1426222 --run-dir artifacts/gradient_aug_time_select_20261102/attempt_TIMESTAMP_preflight
```

After independent recovery review, freeze the new code SHA, repeat the two-dataset CUDA Stage0, and run a fresh `all` attempt. Add the same `--recovery-attempt` and `--task-start-unix 1793549242.1426222` to both `check` and `all` commands above. `all` itself reruns the numerical preflight before training. The original 10h and 12h deadlines remain UNIX `1793585242.1426222` and `1793592442.1426222`; restart does not extend them. Original worker cost and original development forecast carry forward.

Fresh training reconstructs all 20 RAM trajectories, then compares initialization, effective configuration, every primary online step hash, order/view hashes, final optimizer, selected epochs and every retained endpoint identity/metric to the old attempt. A mismatch writes `cross_attempt_identity.json` and blocks promotion for independent review. Successful recomputation counts as the same scientific trajectories, with all additional physical passes charged. `execution_cost_continuity.json` separates old/current/combined costs; `recovery_execution_receipt.json` records origins, deadlines, source identities and preflight. Old attempt files remain untouched. The real-schema recovery regression test requires that preserved numerical attempt to be available at the path above.

The second coordinator attempt `attempt_20261102_recovery_0315` lost a worker with confirmed native exit `-11` before a full RAM result. For the next fresh attempt, retain `--recovery-attempt .../attempt_20261102_0005` as the identity reference and add `--additional-cost-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315` to preflight/check/all. This repeated argument accepts explicit extra raw cost sources and deduplicates their resolved paths. It reads their raw run/pass files, never their inherited prior-cost receipts.

Only confirmed `-11` after join/reap is newly recoverable. The existing limit remains two full retries. KuaiRand/B2/2022 carries its observed failure count: its first new launch is attempt2, exclusively after other scheduled workers finish, with no new jobs started until it ends. Other jobs retain their own allowance. Retry remains the identical full primary/replay configuration from initialization. The original hard deadline can cancel pending retries. Worker stderr enables all-thread faulthandler for native diagnostics; model states stay in RAM.

`prior_physical_cost_sources.json`/`prior_physical_cost_ledger.csv` describe all prior raw sources. Final `physical_cost_sources.json`/`physical_cost_ledger.csv` include those sources and every current attempt directory, including failed retries. Merged runtime_stats replaces its child primary/replay costs; incomplete replay costs use committed batch/epoch/validation logs and identify lower bounds and N/A fields. `execution_cost_continuity.json` provides combined known totals or labelled lower bounds, complete/partial pass counts and scientific deduplication. A successful results-only costs.csv excludes failed work, so use the physical ledger for total resource cost.
