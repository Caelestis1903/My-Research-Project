# Failed Recovery Runner Report

## Execution identity and configuration

- Experiment: `gradient_aug_time_select_20261102`
- Recovery run directory: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315/`
- Recovery Stage0 directory: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315_check/`
- Execution commit: `3a164c6d6e3847883d1f2ae23c7d8e76ee47b6db`
- Source manifest hash: `95c5d28204849e2e815ff33753339a1ef115fdbb2401886923435b6e9511299d`
- Runtime: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; mode `all`; devices `0,1`; batch `40960`; `50` epochs; FP32.
- The Stage0 check used `cuda:1`, passed for MovieLens and KuaiRand, and recorded `test_evaluated=false` for both. Its identity matches the full-run commit and source hash.
- Recovery was passed `--recovery-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005` and `--task-start-unix 1793549242.1426222`. The absolute task clock was not reset.

## Run status

At coordinator exit, 11 unique runs were complete: eight A0/A1 runs (including two A0 calibrations) and three B2 runs. One additional B2 run was partial and failed: KuaiRand, seed 2022. The failure ledger has that single entry, marked `oom=False`. The other completed seeds were 2022 and 2024. No scientific conclusion is recorded here.

The coordinator completion record is `FAILED`, exit code `1`, after B2 development became incomplete. The failure entry is:

`development / kuairand / B2 / seed 2022 / device 0 / attempt 1: worker exited -11 without RAM state`

The worker's primary run completed 50/50 epochs. Replay logged 42 complete epochs and then epoch 43 through batch 28, global step 2086. Its replay `runtime_stats.json` was not written. The final worker stdout line is epoch 43, batch 28, step 2086; the final epoch metric row is epoch 42, step 2058. Its batch trace has 2,086 rows and 2,086 distinct steps; projection trace has 8,344 rows; view trace has 2,086 rows.

No B2 follow-on stages, conditional queue, confirmation runs, unified frozen test, `selection_frozen` receipt, or `state_release` receipt were produced. `progress.json` recorded development completed 3/4, active 0, pending 0, failed 1, `test_accessed=false`. The coordinator process exited, so RAM-only coordinator state was no longer resident. No retry was started; the attempt and its files remain preserved pending the next reviewed execution identity.

## Failure record and read-only diagnostics

The coordinator traceback in `completion.json` and the outer stderr reports `RuntimeError: Incomplete stage` for the failed worker. The worker exited with signal-style status `-11`. The failure ledger marks `oom=False`; no OOM is reported.

Read-only kernel log output contains a contemporaneous record for worker PID 3719029 at `2026-11-02T04:17:07.635525+08:00`: a segmentation fault in `nvidiactl`. This is a recorded event; the root cause is undetermined. The filtered kernel log query returned no contemporaneous NVIDIA Xid line. A post-exit `nvidia-smi -q` observation reported `GPU Recovery Action: None` on both cards; ECC counters were `N/A`. These observations do not determine the cause. The narrow output excerpt is preserved at `research_reports/2026-11-02_gradient_aug_time_select/native_failure_diagnostics/worker_3719029.txt`.

The failed run's replay instrumentation values below are lower bounds from persisted records, not final runtime totals. The partial replay has 2,086 recorded steps, gradient calls sum to 10,430, and original batch rows sum to 85,404,424. The view trace records 2,086 draws and 3,808,127 target rows. The replay's per-step forward lower bound is 4,172 (one original-batch forward and one recorded view forward per logged B2 step). The 42 complete epoch records sum to 658.5821999190957 seconds; time for the unfinished epoch and any unrecorded final work is unknown. Replay GPU training-event time and shadow-update total are `N/A` because no replay runtime summary exists.

## Cost export and sources

[failed_recovery_costs.csv](failed_recovery_costs.csv) contains one row for each of the 11 complete unique runs, using a single combined runtime record per augmented run and a primary runtime record for A0, plus one partial-failed row. The failed row combines its complete primary runtime record with lower bounds from replay traces; it does not add primary/replay component files to a combined record. Unknown final totals are `N/A`.

Across those 12 rows, the CSV sums exact completed-run values plus the failed run's recorded lower bounds: training seconds `9862.585071863257`; completed full training passes `19`; optimizer steps `34136`; training forwards `61372`; gradient calls `87880`; original exposures `1388247324`; extra views `97495927`; view draws `27236`; validation forwards `11244`; validation wall seconds `735.8940833364031`; validation GPU-event milliseconds `540751.109719038`. These are raw sums, with the partial failed row remaining a lower bound. Training GPU-event milliseconds for the failed row are `N/A` in the CSV.

Key artifacts:

- Stage0 checks and matching execution identity: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315_check/stage0_checks.json`
- Recovery receipt and original absolute task start: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315/recovery_execution_receipt.json`
- Coordinator completion/progress/failure/launch/trial ledgers: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315/{completion.json,progress.json,failure_ledger.csv,launch_ledger.csv,trial_ledger.csv}`
- Full-run stdout/stderr: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315.stdout.log`, `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315.stderr.log`
- Failed worker logs and partial replay records: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315/runs/development/kuairand/B2/seed2022/attempt1/`
- Cost CSV: `research_reports/2026-11-02_gradient_aug_time_select/failed_recovery_costs.csv`
