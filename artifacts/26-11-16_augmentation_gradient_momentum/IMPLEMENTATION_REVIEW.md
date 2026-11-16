# REVIEW REPORT

1. Verdict: **PASS WITH FIXES**

All identified implementation issues have been corrected by the original coders and independently rechecked. Blocking issues: **none**. This is the static/CPU implementation gate; runner GPU smoke is next.

Workspace: `/home/xzf/Code/phase0-3_tvg_rm_20261116`.

- Final `protocol.source_manifest()['sha256']`: `d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6`
- Design SHA256 from the same manifest: `67fdebf500e80611de2f0600e0847bf0697774bcb0fa107571154b1524fff338`
- Final autonomous.py SHA256: `19bbb5e075a3928997ba6118da9e420f27609554615556bc25c97a2cc0f819fc`
- Final main prompt SHA256: `96ee719eb022aac5dfbd9ded754f2bcea045b0715ddaf407db3f7e1728fd494b`

Read local AGENTS.md, start.md decoded as GB18030, buchong.md, approved DESIGN_SPECIFICATION.md, both completed coder reports, all new package source/tests/prompt, corrected plans/state/manifest, and context changes. Applied ponytail full and stop-that-shit minimal verification. Reviewer changed only these two review files.

2. Requirement-by-requirement verification

| Requirement | Result | Independent evidence |
| --- | --- | --- |
| Natural task views and gradients | PASS | core.py:127,153; imported sensitivity/make_views: same theta, detached delta, trainable embedding, only target-task BCE, all natural 2B entries including tails; CPU group a. |
| Clean Adam and residual state | PASS | core.py:48; source torch Adam receives clean data gradients only; WD once; actual per-parameter n_p; clean None skips m/v/q/WD/step/cap; zero active; view None equals zero; reentry preserved; CPU group b. |
| Global cap, SHUFFLE and INSTANT | PASS | core.py:65-104; FP64 norm accumulation, FP32 states; SHUFFLE uses own true q and capped/preconditioned local injection norm plus independent keyed Rademacher stream; no second preconditioner/cap; INSTANT changes residual history only; CPU groups b/c. |
| Gamma zero and EMA | PASS | core.py:145-176; canonical source path without probes/random/q; one EMA after full update; exact model/Adam/EMA/RNG/loader and real RAM continuation checked in CPU group d. |
| Frozen matrix and baseline preservation | PASS | protocol.py:34,72,118; exactly 2 datasets x 5 arms x 5 seeds2022-2026, 50epochs, batch40960 tails, FP32; 2500epochs/86250 updates; 20 read-only historical references validated; 67 frozen historical source/receipt entries match. |
| Selection, raw exports and paired statistics | PASS | runner.py:145; tables.py:10; independent online/EMA earliest strict validation Macro AUC maxima; selected/E10/E30/E50, six-cell/domain/task/Macro/traffic/pooled/worst-cell metrics; seed-level SD/CI and direct CLEAN/NDP-A/SHUFFLE/INSTANT contrasts; retention ranking matches design. |
| RAM lifetime and epoch recovery | PASS WITH FIXES | RAM IPC only; epoch state includes model/Adam/q/EMA/RNG/loader/selections; complete compacts recovery copies; retain verifies ten selected identities and releases other keys; parent releases rejected/displaced champion states. Explicit recover now rearms failed/dead workers without dropping committed state. |
| Resource admission | PASS | runner.py:29,262; independent per-GPU current/projected memory and current power <85%, current/projected hostRAM <90%; bootstrap reservations clear after measured allocation; no concurrent-process count cap. |
| Persistent native parent and Astra policy | PASS | autonomous.py:101,112,136,242 and main prompt; real exec then exact-ID resume, raw JSONL/stderr/exit, turn completion continues, technical error PAUSED; fresh native Astra required inside parent. Actual preflight parent/Astra model/effort/source metadata independently read. |
| Unbounded loop and stage order | PASS | 00-13 plans, manifest, state and prompt: all outer caps null; DESIGN/IMPLEMENT/independent REVIEW/RUN/ANALYZE; NO-GO/ASTRA_STOP lead to fresh exploration, USER_STOPPED only normal termination; future plans require new reviewed designs. |
| Manual stop and sealed Test | PASS | Both workspace flag/sentinel controls checked; locked merge preserves flag; runner checks launch and agreed epoch boundary; same-session FINALIZE path requires atomic completion/summary; Test policy requires USER_STOPPED, freeze and full five-seed selected RAM receipts. No autonomous Test evaluator/inference path. |
| NDP-A context and publication | PASS | EXP_AI_CONTEXT and evidence ledger contain substantive findings and full report/raw/source/data/selection/cost pointers; pointers exist. Main publication remains unverified and explicitly gated on actual commit/push plus remote-main evidence before Astra. |
| Runtime evidence gates | PENDING RUNNER | Static/CPU implementation accepted. Real two-dataset/ten-branch full-batch GPU smoke, live driver progress/dispatch evidence and remote publication receipts are subsequent stage gates; none claimed completed. |

The review followed requirement coverage, added behavior, formulas, shapes/masks/routing/weights, gradients, and preserved baselines. No research algorithm or hypothesis was redesigned. The agreed epoch boundary for manual stop/recovery is retained.

3. Bugs/deviations found

- **MAJOR R1 — fixed:** failed/dead workers could never resume. `schedule` classified retained FAILED/dead RUNNING records as failures on every new invocation and recreated PAUSE_TRAINING. This blocked continuation of the frozen matrix after a repair. Independently reproduced without training.
- **MINOR R2 — fixed:** the main prompt named a nonexistent first coder receipt. It now names CODER_ALGORITHM_REPORT.md.
- CRITICAL issues: none found. Unresolved implementation issues: none.

4. Corrections made

Original coder1 added `runner recover --attempt PATH --key KEY [--review PATH]` and a guardian action. Recovery validates the unchanged source/design/review/current smoke, refuses manual stop/live workers/missing committed RAM, preserves saved epoch/state, and clears PAUSE_TRAINING only after all failed/dead keys are rearmed. Normal run does not silently retry. One focused fixture verifies both failure states, exact retained state, next registration attempt, unresolved pause, and refusal of missing RAM.

Original coder2 corrected the report filename and added the recovery command to the main prompt. Its driver code stayed at the final hash above. Both final coder reports were reread. Reviewer edited no implementation/specification files.

Independent checks:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/task_view_gradient_residual_momentum_20261116/test_core.py experiments/task_view_gradient_residual_momentum_20261116/test_autonomous.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/task_view_gradient_residual_momentum_20261116/test_core.py::test_explicit_recovery_reuses_committed_epoch_and_rejects_missing_state
```

The first command passed **15 tests in 3.91s**: four numerical groups and eleven driver fixtures. After the recovery correction, the second command passed **one additional focused fixture in 2.52s**. Existing unchanged checks were reused. The single warning is the source setup helper's TF32 API deprecation. Final runner help includes recover; `git diff --check` passed.

Historical source verification matched all 67 entries of the frozen NDP-A source/receipt manifest (runtime code in this worktree; archived receipts in the evidence root). Read-only control validation accepted all 20 historical trajectories and exact data/selection identities. Required full report/raw/review pointers exist. Parent and Astra actual model/effort and native depth1 thread_spawn metadata were read directly from their recorded session logs.

5. Remaining risks and stage gates

- Actual full-batch GPU numerical/resource behavior awaits runner smoke: two dataset harnesses, five single-step branches each. Formal run checks this exact source/design smoke receipt.
- The driver has fixture evidence; real thread/session events, legal stage progression and native next dispatch must be recorded after launch.
- Context is integrated in the worktree. Actual commit/push and remote-main publication are pending. The earlier TLS dry-run supplies no publication/authentication verdict; the parent must verify the future remote receipt before Astra.
- Exact recovery/final selection requires live guardian RAM. Loss requires the frozen rerun described by the design. The current runner has no final Test command; a separately reviewed evaluator and live candidate/comparator identities are required after USER_STOPPED and final freeze.

6. Whether experiments are safe to run

**Runner prepare and the specified engineering GPU smoke may proceed.** Formal full50 training may proceed only after that smoke passes for this exact source/design and any smoke blockers are closed. No remaining static/CPU blocking fix is needed.

Exact next commands, from the workspace:

```bash
P=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
M=experiments.task_view_gradient_residual_momentum_20261116.runner
A=artifacts/26-11-16_augmentation_gradient_momentum/attempt1
R=artifacts/26-11-16_augmentation_gradient_momentum/IMPLEMENTATION_REVIEW.json
"$P" -m "$M" prepare --attempt "$A" --review "$R"
"$P" -m "$M" smoke --attempt "$A" --device cuda:0 --review "$R"
# Only after the current-source smoke passes and blockers are closed:
"$P" -m "$M" run --attempt "$A" --devices 0,1 --review "$R"
# After repairing a technical worker failure and quiescing workers:
"$P" -m "$M" recover --attempt "$A" --key FAILED_OR_DEAD_KEY --review "$R"
# Recover each unresolved key, then invoke run again.
```

No broad repository regression, GPU smoke, formal training, live driver launch, Test evaluation, scientific result analysis, commit or push was performed by this reviewer.
