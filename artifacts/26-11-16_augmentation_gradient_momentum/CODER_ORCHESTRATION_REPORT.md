# CODER ORCHESTRATION REPORT

Coder2 implemented the approved orchestration/plan/context scope in `/home/xzf/Code/phase0-3_tvg_rm_20261116`, source main `dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11`. Inputs: local AGENTS.md, GB18030 start.md, buchong.md, DESIGN_SPECIFICATION.md, PREFLIGHT_RECEIPT.json PASS, ponytail full and stop-that-shit change. Later user coordination supplies the confirmed runner interface, dual-workspace stop controls and remote-main publication gate.

## Changes

- `experiments/task_view_gradient_residual_momentum_20261116/autonomous.py`: stdlib detached driver around installed `codex exec --json --model gpt-6.1-sol -c model_reasoning_effort="xhigh"`; actual thread.started.thread_id persists and subsequent turns use `codex exec resume EXACT_ID`. Raw event JSONL, actual item/event IDs, stderr, exit code and CLI failure are recorded. Completed turns resume while ACTIVE with next_action. Missing/mismatched session identity and CLI/turn failures PAUSE; no replacement parent or fabricated completion. Duplicate driver lock and a still-live previous CLI prevent overlapping recovery.
- `test_autonomous.py`: eleven small event/resume/error/stop/Test-gate fixtures. The subprocess fixture emits CLI events only; it does not dispatch Codex or training.
- `AUTONOMOUS_MAIN_PROMPT.md`: inspects current completed receipts, completes static/CPU independent review -> runner smoke -> needed smoke signoff -> frozen formal run -> main analysis/report/context publication, then fresh native Astra >=3 mechanisms/six artifacts and the next reviewed experiment. Astra remains gpt-6-astra/xhigh inside the persistent parent; no standalone exec Astra. Uses confirmed prepare/smoke/run/tables/status/retain/release commands.
- `research_plans/26-11-16_augmentation_gradient_momentum/00_EXECUTION_RUNBOOK.md` through `13_AGENT_HANDOFF_PROTOCOL.md` and `experiment_manifest.json`: removed total cycle/revision/Test contradictions; corrected per-parameter clean-None/zero/re-entry Adam steps, detached view semantics, post-preconditioner/post-cap own-state SHUFFLE norm matching and mandatory INSTANT residual-history ablation. First round is 50 new runs, 2500epochs, 86250 complete parameter/EMA updates; ML20/KR49steps per epoch and1000/2450 per50epoch trajectory.
- `artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json`: preserves current counts, records completed design/preflight versus pending implementation/review/run, null outer caps and parent session, concrete next_action, sticky manual stop, pause/recovery, guardian/champion/final-freeze fields and pending main publication.
- `EXP_AI_CONTEXT.md` and `research_context/experiments/Experimental_Evidence_Ledger.md`: integrated existing full NDP-A report findings, exact published raw paired summary/seed/cell values, reported evidence strengths, Stage1/Stage2/readout/stage contradictions, source/data/selection/weight/review/cost evidence and full report/raw pointers. Historical text is preserved. This imports existing conclusions; Coder2 performed no scientific analysis. Canonical current question path is `research_context/experiments/Experimental_Open_Questions.md`.

## Stop, Test and RAM contract

All driver resume/exec boundaries and prompt-contracted child dispatches check manual_stop_requested/STOP_AUTONOMOUS in BOTH execution and original control `/home/xzf/Code/phase0-3`. Shared stop helper is adopted by Coder1's runner protocol prelaunch/pre-epoch/postcommit. `update_state` locks, merges current state and preserves latched stop; `stop` writes a durable sentinel too. Existing atomic parameter/EMA/RAM work finishes before finalization. Finalizer resumes only the exact parent, dispatches no new research and records actual quiescence/summary. Driver writes AUTONOMOUS_STOP_SUMMARY.md; technical failure PAUSES with stop still latched. USER_STOPPED is the only normal termination.

Test requires manual termination, final freeze, eligible complete five-seed selected identities for both datasets and live guardian verification receipt. ACTIVE/PAUSED/candidate freeze/Astra STOP cannot enable Test. No eligible candidate => NOT_TRIGGERED. Current runner has no final-Test CLI; after manual stop/freeze, main must commission a separately reviewed evaluator before final Test. Test cannot feed HPO or renewed exploration.

After complete validation analysis, main uses `retain --attempt PATH --arm ARM` for one champion's ten selected states (five per dataset), verifies identities, then releases rejected/displaced state keys. No-key release clears the entire attempt and must not target retained champion/current recovery. Current recovery/selection staging plus one champion bound RAM lifetime. All tensor/model/optimizer/EMA/checkpoint states remain in existing RAM guardian.

## Verification completed

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/task_view_gradient_residual_momentum_20261116/test_autonomous.py
```

Result: **11 passed in 0.31s**. Covered actual-shaped thread/item events, exact resume ID, mismatch/invalid parsing, CLI success/error/stderr/exit capture, both roots' flag/sentinel, stop-preserving state updates, no dispatch on preexisting stop, summary, missing-session PAUSED recovery, CLI spawn failure/recoverable finalization, and manual-stop/freeze/selected-RAM Test policy.

Review identities: autonomous.py SHA256 `19bbb5e075a3928997ba6118da9e420f27609554615556bc25c97a2cc0f819fc`; AUTONOMOUS_MAIN_PROMPT.md SHA256 `96ee719eb022aac5dfbd9ded754f2bcea045b0715ddaf407db3f7e1728fd494b`.

Installed `codex exec --help`/`resume --help` confirms the command syntax. Existing multi_agent feature is enabled; no configuration changed. Actual new runner `--help` confirms prepare/smoke/run/tables/status/retain/release and their arguments. Source inspection confirms shared stop helper and retain/release interfaces.

Follow-up bounded host check: both exact codex_command argument lists, with --help appended, returned exit0. Fresh accepts --json/--model/-c/--dangerously-bypass-approvals-and-sandbox/--cd and stdin `-`; resume accepts --json/--model/-c/--dangerously-bypass-approvals-and-sandbox AFTER the explicit SESSION_ID and stdin `-`. Resume has no --cd flag; subprocess cwd supplies the execution workspace. No real Codex session, driver or training was invoked. autonomous.py and fixtures are unchanged, so the reviewed driver source hash remains valid. Prompt now names the actual CODER_ALGORITHM_REPORT.md receipt; latest runner help confirms recover --attempt --key [--review], referenced for specific PAUSED-job recovery after cause repair.

One basic consistency check passed: JSON manifests parse;50runs/2500epochs/86250updates and null caps agree; old context text is intact; state records zero completed trajectories, null driver session, NOT_STARTED driver, no Test. `git diff --check` passed.

Read-only NDP-A source hash verification matches PREFLIGHT:

| File | SHA256 |
|---|---|
| full REPORT.md | 4100b8ee0f97cf1f66cca888be9f8f1297b5d7b52c676995a418f9ede02a463a |
| ENDPOINTS.csv | 22ddfcf68be74e3f54fdf9b4e3f2a17fcbdac5a48b2c1cee95824c950a538908 |
| PAIRED_SEEDS.csv | 8b3212f00080c097e9166ce13f749566ceba3cc04b6d1da7421924382e42caf6 |

## Parent commands after independent review

From the execution workspace:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.autonomous start
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.autonomous status
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.autonomous stop
```

After repairing a recorded technical failure, use the same start command with `--recover`; it retains exact parent_session_id. Never supply --last or a new Astra exec. Initial prompt contains the confirmed formal runner commands, actual read-only cache/reference roots and the required receipt checks. R must name the real independent review JSON, with source_hash/design_sha256/cpu_checks_passed and no blockers; it is assigned by main after review.

## Outstanding gates

1. Complete Coder1 receipt/basic CPU checks, independent review of design/actual code/BOTH coder reports, then parent driver launch. Hash current autonomous.py in review; protocol source_manifest includes it.
2. Runner real full-batch smoke and needed independent smoke signoff, followed by frozen full formal matrix. Coder2 has not run GPU smoke, formal training, driver or native research dispatches.
3. Main analysis/report/context publication. Earlier pinned starting-head push dry-run failed with gnutls_handshake transport error and made no write; this is not an authentication verdict. At real publication retry that transport once or use already connected GitHub tree/commit/update_ref with expected head, target main unchanged. Verify remote main before Astra; unresolved error => PAUSED. Coder2 made no commit/push or transport retest.
4. Final Test evaluator is intentionally absent from the training CLI; its reviewed implementation follows manual-stop/final-freeze eligibility.

No remaining Coder2 implementation blocker. The next action is independent review and completion of the recorded parent gates.
