# Cycle02 natural quadratic ranking

The approved seven-file Cycle02 design fixes three arms, two datasets, seeds 2022–2026, 50 epochs, batch 40960 and FP32. `core.py` owns the objective and train-step API. This adapter reuses the frozen model, data checks, Adam, EMA, evaluator and BytesIO client. It writes numeric records only beneath:

`/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1`

Run commands from `/home/xzf/Code/phase0-3_tvg_rm_20261116` using the project interpreter and `PYTHONDONTWRITEBYTECODE=1`. CPU checks:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.checks
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -c 'from experiments.natural_quadratic_ranking_20261116.runner import runtime_admission_checks; print(runtime_admission_checks())'
```

The parent integrates both coder reports and obtains a distinct implementation review before preparation. Review JSON requires `verdict` PASS or PASS WITH FIXES, empty `blocking_issues`, `cpu_checks_passed=true`, actual `source_hash`, `design_bundle_sha256`, `reviewer_id`, and the two distinct `coder_ids`. Source hashing reads bytes and discovers local imports; it invokes no previous cycle gate. The source identity includes all seven new files, the lock, the exact protected 79 paths, and any additional actual local imports. Source changes require fresh binding/review.

Later runner commands (not executed by coders):

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.runner prepare --review /absolute/code-review.json
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.runner smoke --devices 0,1 --review /absolute/code-review.json
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.runner run --devices 0,1 --review /absolute/code-review.json
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.tables
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.runner handoff
```

After code review and preparation, the **parent alone** starts unchanged `experiments.task_view_gradient_residual_momentum_20261116.runner.serve(str(N))` in its persistent foreground session, with the absolute N above. This package contains no owner launcher. Parent writes N/STORAGE_OWNER.json with `root`, `pid`, `socket`, `session_id`, `server_sha256`, `source_hash`, `design_bundle_sha256`, and `endpoint_hash=digest([str(N),pid,socket])`. Its source hash identifies C2 workers; server_sha256 separately identifies the frozen TV server. Private guardian.json remains mode 0600. Keep that session supervised through the final parent storage acceptance.

Smoke runs exactly six disposable dataset/arm cases, seed 2022. Each uses the true natural first full batch and true tail from one source permutation, real Adam/EMA, diagnostic on/off and source-clean identities on disposable restored states, complete natural online/EMA validation, BytesIO roundtrip, commit/get/hash verification, and pause/re-register/restore. Case keys bind `purpose=engineering_smoke`; they never equal formal job keys. A smoke snapshot commits at epoch 1 with two useful engineering updates; it never calls the server's epoch-50 complete operation. It releases only its own state after recovery verification. Smoke physical cost includes all disposable identity operations.

Before formal run, an independent reviewer writes N/SMOKE_REVIEW.json with PASS or PASS WITH FIXES, empty blockers, `reviewer_id` distinct from both coders, `source_hash`, `design_bundle_sha256`, `smoke_sha256`, and `resource_checks_passed=true`. Smoke and signoff bind the immutable run manifest separately. Missing capacity means supervised waiting. Actual OOM means technical PAUSE and corrected measured profile/review, retaining batch/precision.

Admission evaluates each card independently: current NVML memory <85%, measured board power <85%, current host RAM <90%, no pending allocation ACK on that card, and current used bytes plus incoming measured full-process residency <= measured CUDA-usable total. It resamples after allocation ACK. C1 full-process evidence bootstraps disposable smoke only: ML 4515168256 B; KR max(5540675584,5192548352+352321536)=5544869888 B. Formal workers use C2 full-process/torch/context profiles from smoke on matching hardware identity, including matching usable capacity; actual larger worker observations raise admission profiles. NVML PID monitoring covers lazy Adam, diagnostics, full/tail, evaluation, CPU snapshots, serialization and recovery. Context is added to torch reserved only, never to an already complete NVML process footprint.

Worker continuation is an epoch transaction. The numeric epoch file is written first; commit sends a CPU snapshot and absolute path to the same N owner. A get verifies the retained full-state hash and individual hashes before a local ACK binds the epoch-file hash. Snapshot costs end at capture; its own serialization/transfer costs live in that ACK and are carried into subsequent continuation. Ambiguous replies reconcile actual retained state and the commit journal. Uncommitted files cannot become recovery states. RAM loss pauses work.

After technical correction, use `runner run --recover --devices 0,1 --review /absolute/code-review.json`. Global recover first requires every recorded worker dead, the same endpoint identity, matching source/job/design/journal/state hashes, and no user stop. Epoch zero reinitialization requires proven no state/no ACK and a failed physical-attempt record. An acknowledged epoch 50 finalizes without training again. There is no retry-count cap. A live scheduler supervises attached mechanical subprocesses and exits only after they finish their supported atomic epoch. Both workspaces' controls govern every new launch.

If a worker dies before its final physical receipt, reconciliation keeps complete returned batch lines and ACK transfer costs. It records launch-to-observed-death time as an upper bound and leaves unreturned operation/evaluation costs null. This numeric reconciliation cannot create a continuation state; recovery still verifies the actual retained RAM against the last commit.

The server compacts completed trajectories to selected online/EMA states, epochs and validation records. Numeric results preserve the selected hashes and original job provenance before compaction. All 30 selected pairs remain in RAM until parent acceptance and later parent disposition. Handoff lists all keys, hashes, bytes and 1500 committed epochs, with the owner's foreground session identity. Parent must accept supervision before the runner context closes.

The exporter selects exactly 30 manifest jobs plus 10 explicit hash-bound CLEAN results. It verifies earliest strict-maximum selection and reads new epochs only through matching RAM journal/local ACK receipts. Outputs contain 320 endpoints, 4000 estimator-epoch rows, 24000 cell-epoch rows, raw coefficient/statistic ledgers, diagnostic groups, all physical attempts, costs and paired five-seed arithmetic. Missing historical costs remain null. Failed/replayed batch logs have a separate path/hash/count inventory. Diagnostic timing is an inclusive subset of training-step time. Parent performs scientific interpretation. Test stays sealed.
