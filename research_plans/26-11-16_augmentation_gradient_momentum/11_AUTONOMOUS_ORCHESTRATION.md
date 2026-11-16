# Autonomous Orchestration Contract

## Highest-priority current-conversation policy — 2026-11-16

This user policy overrides older driver, session-lifecycle and handoff instructions below. Preserve historical experiment details, scientific gates and source provenance.

- The original foreground orchestrator alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. All tasks and agents stay under the current conversation; only that orchestrator dispatches children.
- Each subtask gets a fresh native child with explicit `fork_context=false`. Coder and reviewer are distinct. Close each child on completion; never resume or reuse it. Handoffs use explicit files/reports only, without inherited conversation context.
- Independent/headless/background Codex sessions, `codex exec/resume/fork`, `tmux`, `nohup`, `systemd`, and `autonomous.py start/run/recover` are forbidden. Create no background agent/driver. The original 50/50 trajectories are complete; the old driver was PAUSED after treating a recovered reconnect warning as fatal and is retired. Preserve its source/receipts; do not repair or restart it.
- Concurrent mechanical training is permitted only under supervision of a runner in the current conversation. Keep the existing RAM guardian as storage; preserve live states and frozen source identities. No detached AI session or replacement guardian/driver.
- `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden total/revision cap. Only explicit USER_STOPPED normally terminates; candidate NO-GO/Astra STOP leads through report/context to fresh research, while technical failures PAUSE.
- Check both workspaces' stop controls before dispatch. On user stop, finish/checkpoint the atomic operation safely and produce the final summary. Test stays sealed until explicit user stop + final freeze + the existing eligible selected-RAM verification gate; ACTIVE/PAUSED/candidate outcomes never open Test.

## Preserved protocol and historical provenance

Parent gpt-6.1-sol/xhigh; coder and DIFFERENT reviewer inherit it. Native Astra gpt-6-astra/xhigh is the explicit user exception to start.md same-model policy. Existing PREFLIGHT_RECEIPT.json PASS records actual independent depth1/no-fork Astra metadata and fixed source reads; no duplicate preflight.

autonomous.py is a stdlib detached process using installed codex exec --json then codex exec resume EXACT_SESSION_ID. Capture thread.started.thread_id and actual event IDs when emitted, raw JSONL/stderr/exit diagnostics. Never --last, never separate exec Astra, never copy credentials/change global config. A turn.completed ends one turn only; ACTIVE+next_action resumes. CLI/turn errors PAUSE with termination_reason=null; recover exact session with --recover.

max_cycles=null; max_astra_cycles=null; max_experiments=null. NO-GO/NO_CONTINUATION_SIGNAL -> report/context -> fresh Astra. Astra STOP rejects its candidate -> fresh Astra; REVISE corrects missing requirements. No hidden revision/time/count cap. Real technical blockers PAUSE without lowering scientific gates. Only USER_STOPPED normally terminates.

Every cycle: completed DESIGN -> native coder -> different reviewer -> corrected blockers -> runner raw results -> MAIN ANALYZE/report -> substantive current-main context publication verified remotely -> fresh native Astra >=3 mechanisms/six artifacts -> complete new plan -> next DESIGN. Reuse valid first-round receipts instead of restarting completed stages. Static/CPU review precedes runner smoke, then needed smoke signoff, then frozen formal matrix.

Manual controls in BOTH /home/xzf/Code/phase0-3_tvg_rm_20261116 and /home/xzf/Code/phase0-3: artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json.manual_stop_requested or STOP_AUTONOMOUS. Check before each driver resume/exec, every child dispatch and runner launch/epoch RAM commit. All state writers use autonomous.update_state to merge current state and preserve sticky stop; request_stop also writes sentinel.

Stop new dispatch, complete current full gradient/parameter/EMA/RAM transaction, quiesce workers and write final research summary. Driver may resume SAME parent for FINALIZE ONLY, forbidding new research/training; parent records stop_finalization complete/atomic_operation_complete/summary_path. Technical finalization failure PAUSES with flag latched; recovery resumes finalization. Driver also writes AUTONOMOUS_STOP_SUMMARY.md from actual state. Stop must not disappear when parent yields.

Test allowed only USER_STOPPED + final_freeze.complete + final_candidate_eligible + verified live selected RAM for both datasets/seeds2022–2026, exact selected hashes and guardian receipt. All ACTIVE, technical PAUSED, Astra STOP or candidate-freeze states remain sealed. No candidate => NOT_TRIGGERED; final Test never feeds selection/exploration.

RAM: existing guardian only; current recovery/current selection staging + one champion's five seeds per dataset. Release rejected/displaced states; preserve eligible selected states at stop. No disk models/checkpoints/tensors. Lost RAM => PAUSED and frozen-protocol affected-trajectory rerun.

Run after independent review, from execution workspace:

    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_view_gradient_residual_momentum_20261116.autonomous start

Commands status, stop, start --recover use the same module and project Python. Driver launch is the parent stage; implementation coders do not start it.
