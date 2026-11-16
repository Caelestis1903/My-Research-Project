# Runtime implementation report — Cycle02 NQR-025

Status: **IMPLEMENTED; FINITE CPU CHECKS PASS; DISTINCT REVIEW AND GPU/RAM SMOKE PENDING.**

Applied ponytail FULL and stop-that-shit CHANGE. Read W/AGENTS.md, O/start.md decoded as GB18030, W/buchong.md, the execution runbook, all seven Cycle02 design files, DESIGN_ACCEPTANCE.json, and the completed core coder report/check receipt. DESIGN_SPECIFICATION.md sections 5.3/6/7 remain authoritative. No scientific/API ambiguity required a design change. Core coder Dirac's final files were consumed unchanged.

Exclusive changed source paths:

- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/protocol.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/tables.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/README.md`

Exclusive reports written:

- This `RUNTIME_IMPLEMENTATION_REPORT.md`.
- `RUNTIME_CPU_CHECKS.json`, with exact commands, outputs, before inventory, final comparison and observed source closure.

`protocol.py` implements canonical config/job identities, the accepted seven-file design bundle, the original six-file Astra provenance, byte-only source closure, strict absolute C2 paths, both-workspace stop controls, environment lock checks, immutable numeric preparation, review/smoke gates and exact CLEAN reuse. It reads the 79-path RESOURCE_1 receipt without invoking any old source manifest/design gate. The new closure has 87 unique files: protected 79 (already including the lock), seven new package files, and the additional actual transitive import `experiments/task_view_gradient_residual_momentum_20261116/tables.py`. Its observed final hash is `d3f518b8d23cbafa9be0b954de995d0edfeaf12ebab3836cd4a723d3eda9d0a0`. No new source receipt or training manifest was frozen/written by this coder; parent performs final binding/review.

Exactly ten manifest CLEAN entries are verified as a subset of the existing 20-entry index. Their original job/result identities remain intact. The check validates result hashes, 50-epoch counts, earliest strict-maximum selection and same-state selected hashes/metrics, ten initialization identities and 500 order identities. The 52 shared conditional/model/data/setup/evaluator/EMA source files match the historical pilot source receipt byte-for-byte; the pilot core/protocol hashes and run manifest are checked. Pilot CLEAN bypasses view construction, uses mean probability BCE plus a zero auxiliary, source Adam and source EMA; its inactive probe/augmentation settings do not alter that path. The historical run uses setup once, an exact initial EMA copy, private seed+11000 loader and one randperm per epoch. Runtime calls the approved frozen helpers unchanged.

`runner.py` implements only prepare/smoke/run/worker/handoff. Parent alone supplies the new foreground TV serve owner and STORAGE_OWNER.json. Clients use absolute N, verify owner endpoint/server/source/design identities, and serialize tensor state only through CPU copies and BytesIO. There is no owner launcher, driver, AI session or research loop. Formal workers use the exact core train_step API, all natural rows/tails, initialization/order checks, independent online/EMA selection and 50-epoch counters. Numeric epochs precede RAM commits; get validates the complete continuation hash and component hashes. Local ACKs bind absolute epoch files and their byte hashes. Lost replies reconcile the retained state/journal. Same-owner recovery requires dead recorded workers and verified acknowledged RAM; epoch-zero restart requires no state/no ACK plus failed-attempt evidence. A recovered epoch 50 completes without training. Complete verifies the frozen selected-state compaction; all 30 formal selected pairs remain retained for parent acceptance.

The resource predicate independently checks each card's current memory <85%, measured board power <85%, current host RAM <90%, pending allocation ACK and physical fit <= measured CUDA-usable capacity. It resamples after allocation ACK. No predicted-power, projected-85 memory, projected-90 RAM, card-coupling or worker-count gate is added. Initial smoke uses the specified C1 bootstrap references. C2 PID-specific NVML profiling monitors full/tail/diagnostics/evaluation/CPU snapshots/serialization/recovery, combines measured torch peaks with compatible context overhead without double counting, and raises profiles from larger live observations across seeds. Formal profiles require matching C2 hardware/usable-capacity identity.

The later smoke command covers exactly six dataset/arm cases, seed 2022, true full 40960 and natural tail batches. Disposable restored states exercise source-clean lambda-zero/all-single-class and diagnostic identity, with those costs separately counted. It performs complete natural online/EMA validation, RAM roundtrip/commit/get, pause/re-register/restore including RNG/loader equality, and release after verified recovery. Smoke keys include engineering purpose; the two-update smoke never calls complete. The separate smoke/resource review binds source/design/run manifest/owner and is required before formal training.

Costs retain all physical attempts, including failed/replayed work, independently of useful update counts. Diagnostic seconds are an inclusive subset of training-step time. Snapshot costs end at capture; its own serialization/IPC costs reside in the ACK and carry into continuation. Abrupt worker death reconciles complete returned batch lines and ACK costs, records launch-to-observed-death time as an upper bound, and leaves unreturned operation/evaluation costs null. This reconciliation never constructs tensor state from numeric files. Recovery still requires the actual last acknowledged RAM state.

`tables.py` exports exactly 30 manifest results plus ten explicit hash-bound CLEAN results. New epochs are resolved from commit ledger and matching local ACKs; orphan numeric files cannot enter endpoints. The output contract is 320 endpoint rows, 4000 estimator-epoch rows, 24000 cell-epoch rows, all six cells, 51750 useful batch ledgers, coefficient/statistic denominators/null counts, diagnostic groups, physical/smoke costs, uncommitted physical-log inventory and five-seed paired arithmetic/descriptive summaries. Historical missing costs stay null. No scientific verdict or causal analysis is produced.

Checks were finite and CPU-only. Exact five invocation commands and outputs are in `RUNTIME_CPU_CHECKS.json`. The concise callable admission command is:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -c 'from experiments.natural_quadratic_ranking_20261116.runner import runtime_admission_checks; print(runtime_admission_checks())'
```

Executed checks passed: six Python-module in-memory syntax checks; 13 admission fixtures including the concrete 19118 MiB predecessor (ML fits, KR fails), strict thresholds, physical equality, independent other-card opportunity, pending ACK and updated measured peaks; all five unchanged core CPU groups; canonical configs/jobs and 51750-update budget; ten CLEAN references/500 epochs; shared step/ledger/CPU-snapshot fingerprint integration and rejection of a changed continuation; synthetic 320-endpoint paired arithmetic (5550 paired rows, 1110 five-seed summaries); environment lock; and dead-worker/truncated-footer/unknown-cost reconciliation. Frozen setup emits its existing PyTorch TF32 deprecation warning.

Before/after read-only disk inventories match: complete result path/hash collections for 50 prior TV and 30 C1 trajectories, whole journal lengths/hashes, owner/guardian/handoff receipts and exact reused-index hash. The protected 79-file aggregate remains `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. This task made no GPU query, IPC/owner contact, dataset training, Test evaluation, git/publication, dependency, root/design/state or frozen-source change, and spawned no descendants.

Actual GPU NVML behavior, full-batch/tail/evaluation residency, RAM transport/recovery and real-result export remain unexecuted RUN-stage checks. Next: parent final source binding and distinct implementation review; then parent-owned foreground storage, fresh runner smoke, separate resource signoff and the registered matrix. Parent closes this coder context permanently after the file handoff.
