# EXPERIMENT EXECUTION REPORT — B1 interim

1. **Experiment ID**: AO-04 Stage R2, B1, KuaiRand natural + MovieLens d0_large, seeds 2022/2023. Canonical run root: `research_R2/formal_runs_20261012/`.

2. **Git/worktree state**: Git metadata is unavailable from `/home/xzf/Code/phase0-3` (`git status` reports that this path is not inside a repository). The reviewed code hashes match `research_R2/preflight/formal_execution_mapping_20261012.json` for all seven `src` files and the three formal orchestration files.

3. **Config**: KuaiRand used `research_R2/configs/runtime_20261012/kuairand_natural_m3oe.json`; MovieLens used `research_R2/configs/runtime_20261012/movielens_d0_large_m3oe.json`. Frozen B1 settings, paired initialization root `research_R2/formal_init_20261012`, fixed 50 epochs, and physical GPU map seed2022→GPU0 / seed2023→GPU1. No config or code changes were made for this batch.

4. **Seeds**: 2022 and 2023, as authorized by the experiment contract.

5. **Run status**: MovieLens B1 seed2022 and seed2023 each completed 50/50 epochs. KuaiRand B1 seed2022 completed 50/50 epochs. KuaiRand B1 seed2023 reached epoch38 before a Python native segmentation fault; its one permitted exact resume from `last_resume.pt` exited with code 136 (`Floating point exception (core dumped)`). It remains incomplete. There are seven COMPLETE manifests overall (four B0 and three B1), eight formal attempts launched, and one exact resume used. No B2, decision, B3/B4, or final test was run because the B1 evidence set is incomplete.

6. **Raw metrics**: Completed B1 runs have six selected-validation domain-task rows each, including AUC, LogLoss, target-weighted AUC/LogLoss, and counts, in `research_R2/preflight/B1_raw_metrics_completed_20261012.json`. This file copies raw selected-epoch entries from `metrics_epoch.csv` and contains no aggregation or interpretation. Selected-epoch metadata are also in each run's `run_manifest.json` and `selection.json`.

7. **Log locations**: Complete-run `stdout.log` and `stderr.log` are in each canonical run directory. The original four launch records are in `research_R2/preflight/formal_B1_launch_state_20261012.json`; original launch log paths are under `research_R2/preflight/logs/formal_B1_*_launch.log`. The resume command output is in `research_R2/preflight/logs/formal_B1_kuairand_seed2023_resume1_launch.log`. Crash and retry details are recorded in `research_R2/preflight/B1_kuairand_seed2023_resume1_receipt_20261012.json`.

8. **Runtime errors**: Linux kernel `dmesg` recorded PID 1072336 segfaulting at 2026-10-12 14:39:09 +0800. Its last checkpoint remained at epoch38. The only permitted exact resume, PID 1080981, exited 136 with a floating-point exception; the checkpoint remained unchanged, with no run manifest. No OOM evidence was observed. No further retry was made.

9. **Missing runs**: KuaiRand B1 seed2023 is incomplete. The remaining B2 runs (both datasets × seeds2022/2023), the validation-only A→B decision, conditional B3/B4 runs, and final-test evaluations remain unscheduled. The runner stopped the next stage pending review of the repeated native process failure and adherence to the one-resume limit.
