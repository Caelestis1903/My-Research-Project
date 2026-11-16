# Agent Handoff Protocol

## Highest-priority current-conversation policy — 2026-11-16

This user policy overrides older driver, session-lifecycle and handoff instructions below. Preserve historical experiment details, scientific gates and source provenance.

- The original foreground orchestrator alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. All tasks and agents stay under the current conversation; only that orchestrator dispatches children.
- Each subtask gets a fresh native child with explicit `fork_context=false`. Coder and reviewer are distinct. Close each child on completion; never resume or reuse it. Handoffs use explicit files/reports only, without inherited conversation context.
- Independent/headless/background Codex sessions, `codex exec/resume/fork`, `tmux`, `nohup`, `systemd`, and `autonomous.py start/run/recover` are forbidden. Create no background agent/driver. The original 50/50 trajectories are complete; the old driver was PAUSED after treating a recovered reconnect warning as fatal and is retired. Preserve its source/receipts; do not repair or restart it.
- Concurrent mechanical training is permitted only under supervision of a runner in the current conversation. Keep the existing RAM guardian as storage; preserve live states and frozen source identities. No detached AI session or replacement guardian/driver.
- `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden total/revision cap. Only explicit USER_STOPPED normally terminates; candidate NO-GO/Astra STOP leads through report/context to fresh research, while technical failures PAUSE.
- Check both workspaces' stop controls before dispatch. On user stop, finish/checkpoint the atomic operation safely and produce the final summary. Test stays sealed until explicit user stop + final freeze + the existing eligible selected-RAM verification gate; ACTIVE/PAUSED/candidate outcomes never open Test.

## Preserved protocol and historical provenance

Parent -> Astra: verified actual current remote main commit, full current report/raw hashes, paired seeds/endpoints/curves/diagnostics/costs/manifests, substantive updated context, canonical research_context/experiments/Experimental_Open_Questions.md, stopped-method list and output directory. Explicitly include full historical NDP-A report and every evidence pointer from 02 on EVERY fresh handoff. Fresh native context receives no transcript.

Astra -> parent: all six output paths/hashes, chosen full recurrence, >=3 explored mechanisms, assumptions/alternatives/novelty, falsification, full five-seed matrix, resources and blockers. Parent checks completeness then routes DESIGN -> native coder -> different reviewer -> luna-runner -> main ANALYZE. No duplication of completed receipt-bound first-round work.

Record actual parent thread.started.thread_id, native child IDs/model/effort/source, report/raw/context identities, concrete next_action. All writers merge through autonomous.update_state; check both manual stop controls before every child dispatch. Do not replace actual metadata with requested model names.

Report/main publication is complete only when the actual remote main includes the report/context commit, with expected-head protection. The pinned starting-head dry-run failed with gnutls_handshake transport error, not an auth verdict. Retry that transport once at the real publication stage or use already connected GitHub tree/commit/update_ref with expected head. No new branch/credentials/global config. Otherwise PAUSED with exact error before fresh Astra.

Recovery reads AUTONOMOUS_STATE and real JSONL/CLI errors/receipt hashes/guardian status. Resume exact parent_session_id and recorded next action; do not infer completion from directories, partial logs or a PID. Technical failure PAUSES with no normal termination reason. Candidate NO-GO or Astra STOP instead requires fresh research. Only USER_STOPPED normally terminates; finalizer produces summary at safe atomic boundary and retains eligible selected RAM.
