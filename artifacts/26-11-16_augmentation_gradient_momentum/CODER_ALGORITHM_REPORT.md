# CODER ALGORITHM IMPLEMENTATION REPORT

Implemented the approved TV-GRM specification in `/home/xzf/Code/phase0-3_tvg_rm_20261116`. Scope: algorithm, protocol, workers, RAM adapter, tables, and the four CPU groups. No formal training, GPU smoke, scientific result analysis, commit, or push was performed.

Read `AGENTS.md`, `start.md` through `iconv -f GB18030 -t UTF-8`, `buchong.md`, the completed design, and native preflight PASS receipt. Applied ponytail full and stop-that-shit change. The current explicit assignment makes manual stop atomic at an **epoch** boundary: finish updates, EMA, validation, and the guardian commit, then return with selected states retained. Both execution and original control workspaces are watched through coder2's shared pure `autonomous.stop_requested(workspace, control_workspace)` helper.

## Files owned and changed

All paths below are under `experiments/task_view_gradient_residual_momentum_20261116/`:

- `__init__.py`: package identity.
- `core.py`: detached natural views, separate same-parameter clean/view gradients, source-clean step, residual channel, global cap, SHUFFLE and INSTANT, compact support/gradient diagnostics.
- `protocol.py`: exact full50 configurations, source/design/selection identities, independent-review gate, verified read-only historical control index and dataset manifests, shared stop adapter, current-source smoke gate.
- `runner.py`: prepare/smoke/run CLI; audited admission and RAM IPC adapted for per-epoch commits, pause, selected-state compaction and release; source-equivalence full-batch smoke harness.
- `tables.py`: 70 trajectories including the 20 reused controls; endpoints, curves, costs, natural support, diagnostics, five-seed paired statistics, and the preregistered arithmetic retention ranking.
- `test_core.py`: four compact CPU test groups.

This report is the only coder1 artifact added outside that package. `autonomous.py`, `test_autonomous.py`, and `AUTONOMOUS_MAIN_PROMPT.md` belong to coder2 and were not edited. Historical source/vendor/report files were not edited. Concurrent plan/context/state edits belong to the other agents.

## Numerical implementation

The optimizer is one unmodified `torch.optim.Adam`, with coupled WD, betas `(.9,.999)`, eps `1e-8`, lr `.01`, and `foreach=False,fused=False,amsgrad=False`. It receives **only clean data gradients**; torch performs the clean update and maintains its own per-parameter step/m/v. The separate channel maintains q by parameter name and uses that parameter's actual Adam counter for bias correction. A clean `None` skips the parameter, WD, m/v/q, counter, and all cap/random coordinates. Active zero gradients still update normally. A missing view gradient contributes mathematical zero, and view-only parameters are counted and skipped.

Clean/view gradients are computed before any parameter update. The imported audited NDP view constructor detaches delta; the forward hook adds delta to the trainable embedding. The clean and view losses each cover all natural `2B` entries, with the view taking task0 only from view0 and task1 only from view1. There is no `.75/.25` mixture in TV-GRM.

Residual q excludes WD. All direction/norm/cap calculations use the active parameter set; norm sums accumulate detached FP64 scalars while parameters and m/v/q remain FP32. SHUFFLE computes its own local true-q counterfactual capped/preconditioned injection norm, then injects a keyed Rademacher direction directly in that update space. It does not apply another preconditioner, momentum, or cap to the random vector. The random generator is independent of model/loader RNG and keyed by dataset, seed, epoch, batch, parameter name, and stream. The FP32 injection norm relative error must be at most `1e-5`, with exact zero checked separately. INSTANT uses current r instead of qhat while retaining the clean Adam beta1/beta2 and identical cap.

Gamma zero directly executes the canonical `zero_grad(set_to_none=True) -> probability BCE -> backward -> torch Adam -> EMA` path. It performs no probe/view/random work and leaves q empty. EMA runs once after the whole clean-plus-residual transaction.

## Execution and state lifetime

The frozen matrix has exactly MovieLens/KuaiRand × five specified arms × seeds 2022–2026: 50 trajectories, 50 epochs, batch40960 including tails, FP32, 86,250 optimizer updates and EMA updates. There is no seed screening or reduced-budget option. Independent online and fixed-EMA states use the earliest strict maximum of validation six-cell Macro AUC.

The workers reuse the verified cache at `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache`. Runtime Python imports come from this worktree, not sibling absolute paths. CLEAN/NDP-A scalar evidence is indexed read-only under `/home/xzf/Code/phase0-3_auxiliary_20261115`; 20 result identities and both historical data/split/encoder/scaler receipts were checked successfully. No reference training is silently appended.

The guardian adapter reuses the audited memory transport, request client, liveness check, and resource helpers. The local service adds the missing PAUSED and compact-selected lifetime operations. Model/Adam/q/RNG/loader and selected states travel only through RAM IPC. On completion, final online/EMA, optimizer, q, and RNG recovery copies are discarded, retaining the two selected readouts. `retain` keeps one complete candidate configuration (five seeds on each dataset) and releases other current-matrix states. `release` releases rejected cycles or a replaced incumbent. The parent must release the old incumbent after replacement and invoke retention/release after its candidate decision; temporary current-matrix selections are kept until that decision. A lost/released recovery state requires a frozen rerun in a new attempt, never reconstruction from a hash or replacement of selected with final weights.

Admission uses each GPU's current memory and power <85%, a projected memory check using measured smoke peak, and current/projected host RAM <90%. Bootstrap allocations are reserved until their first measured batch; after allocation, additional jobs on that device can launch whenever resource admission permits. There is no process-count cap. Smoke records include a RAM recovery/IPC allowance for admission. Worker failures pause launches and let existing workers finish/commit their current epoch. After `run` returns PAUSED and the concrete failure is repaired, `recover --attempt PATH --key KEY` explicitly rearms each failed/dead RUNNING job under the unchanged reviewed source and smoke identity. The guardian retains the job, attempt counter, and committed epoch state, and clears `PAUSE_TRAINING` only when no unresolved failure remains. Reinvoke `run` to continue. Recovery refuses manual stops, live workers, and lost committed RAM; an epoch-zero failure explicitly restarts the same frozen initialization. Ordinary `run` never automatically retries a failed job. A user stop suppresses launches and returns `MANUAL_STOPPED` after live epochs finish. The driver owns shared autonomous state writes and final termination/freeze.

Training, smoke, and tables never evaluate Test. The final user-stop/freeze Test phase is owned by the parent, as explicitly agreed. Candidate selected states remain accessible using the audited `request(attempt, 'get', key=KEY)` RAM client and are checked against their selected hashes by `retain`.

## Verification performed

Command, from the execution worktree:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/task_view_gradient_residual_momentum_20261116/test_core.py
```

Final output after the independent review's recovery fix: **5 passed, 1 warning in 5.35s**. The warning is the existing source setup helper's PyTorch TF32 API deprecation.

1. Natural clean/view losses and parameter gradients match explicit full-support target-task sums; embeddings receive view gradients, delta is detached, and tail/missing-domain cases retain the correct denominator/support.
2. Two parameters exercise active/absent/reentry counters, clean zero gradients, missing/view-only gradients, coupled WD and clean moments, q recurrence, and INSTANT's current-residual update.
3. Nonuniform preconditioning exercises the global cap, actual injected SHUFFLE norm, exact zero cases including C=0 with nonzero r, and independent/reproducible random streams.
4. Gamma zero matches source model/Adam/EMA across multiple steps without probing; both gamma zero and TV-GRM survive an actual guardian RAM commit/get/restore with model/Adam/q/EMA/RNG/loader/selected state matching uninterrupted continuation.
5. One reviewer-requested recovery fixture commits RAM state, exercises explicit recovery from FAILED and dead RUNNING records, verifies unchanged state/epoch and incremented registration attempts, preserves the pause until all failures are rearmed, and rejects missing committed RAM. Re-registration before explicit recovery is refused.

Six owned Python modules pass AST syntax parsing. The runner CLI help loads successfully. No GPU was used by these checks. The historical Jacobian/finite-difference suite was reused: the current auxiliary `core.py`, `checks.py`, and `stage2/core.py` hashes exactly match the audited NDP-A source receipt `831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`. Existing evidence is `stage2_attempt1/CPU_CHECKS.json`, `zero_counterpart_receipt.json`, `SMOKE_REPORT.json`, and `research_reports/.../STAGE2_FINAL_REVIEW.json` in the auxiliary sibling. No new finite-difference matrix was added.

## Exact CLI and review contract

All commands run in `/home/xzf/Code/phase0-3_tvg_rm_20261116` with the project Python:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner prepare --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --review PATH_TO_INDEPENDENT_REVIEW_JSON
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner smoke --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --device cuda:0
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner run --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --devices 0,1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner tables --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner status --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner retain --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --arm TV-GRM-010
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner release --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --key KEY
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.runner recover --attempt artifacts/26-11-16_augmentation_gradient_momentum/attempt1 --key FAILED_OR_DEAD_KEY
```

`prepare` optionally overrides `--cache-dir PATH` and `--reference-root PATH`; the defaults are the approved absolute locations above. `smoke`, `run`, and `recover` optionally take `--review PATH`, otherwise they read the review path frozen by prepare. `recover` requires a key and records its rearming event in `recovery_ledger.jsonl`; repeat for every key in its `unresolved` output, then invoke `run`. `release` without `--key` releases every inactive state in that attempt. `_worker --attempt PATH --key KEY --device cuda:N` and `_guardian --attempt PATH` are internal commands. CLI prints JSON.

The independent review JSON must contain `verdict` (`PASS` or `PASS WITH FIXES`), empty `blocking_issues`, `source_hash`, `design_sha256`, and `cpu_checks_passed: true`. Obtain the current source/design identities by calling `experiments.task_view_gradient_residual_momentum_20261116.protocol.source_manifest()`. The source receipt includes actual imported historical sources and coder2's shared stop helper. Hashes must be taken after concurrent work is finished. Prepare/smoke/run validate the source and design against that review. Formal launch additionally requires `SMOKE_REPORT.json` PASS for the current source/design with exactly two datasets, five single-step branches each, and source-clean/gamma-zero state equality. Any subsequent source change requires updated independent review and smoke.

## Outstanding gates

Independent static/CPU review and the runner's real full-batch GPU smoke are pending. The smoke harness is implemented but was not executed by this coder. Formal full50 training remains gated behind both. No concrete implementation or data blocker was observed in the permitted CPU/source checks. Final Test additionally needs the parent's user-stop/freeze gate and retained selected comparator states; historical scalar JSON cannot supply comparator weights.
