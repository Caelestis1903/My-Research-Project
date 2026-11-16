## Highest-priority current-conversation policy — 2026-11-16

This user policy overrides older driver, session-lifecycle and handoff instructions below. Preserve historical experiment details, scientific gates and source provenance.

- The original foreground orchestrator alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. All tasks and agents stay under the current conversation; only that orchestrator dispatches children.
- Each subtask gets a fresh native child with explicit `fork_context=false`. Coder and reviewer are distinct. Close each child on completion; never resume or reuse it. Handoffs use explicit files/reports only, without inherited conversation context.
- Independent/headless/background Codex sessions, `codex exec/resume/fork`, `tmux`, `nohup`, `systemd`, and `autonomous.py start/run/recover` are forbidden. Create no background agent/driver. The original 50/50 trajectories are complete; the old driver was PAUSED after treating a recovered reconnect warning as fatal and is retired. Preserve its source/receipts; do not repair or restart it.
- Concurrent mechanical training is permitted only under supervision of a runner in the current conversation. Keep the existing RAM guardian as storage; preserve live states and frozen source identities. No detached AI session or replacement guardian/driver.
- `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden total/revision cap. Only explicit USER_STOPPED normally terminates; candidate NO-GO/Astra STOP leads through report/context to fresh research, while technical failures PAUSE.
- Check both workspaces' stop controls before dispatch. On user stop, finish/checkpoint the atomic operation safely and produce the final summary. Test stays sealed until explicit user stop + final freeze + the existing eligible selected-RAM verification gate; ACTIVE/PAUSED/candidate outcomes never open Test.

## Preserved supplemental rules

1. AUTONOMOUS LOOP
   max_cycles = null
   max_astra_cycles = null
   max_experiments = null

2. TERMINATION
   Normal termination = USER_STOPPED only
   Algorithm NO-GO = launch fresh Astra
   Astra STOP = reject current candidate and explore again
   Technical failure = save state and pause/recover
   Never lower research quality gates to continue.

3. MANUAL STOP
   Add manual_stop_requested to AUTONOMOUS_STATE.json.
   Support a STOP_AUTONOMOUS sentinel.
   Check stop request before every new Astra/Codex dispatch.
   Finish or safely checkpoint current atomic operation.
   Produce final research summary.

4. TEST
   Keep Test sealed throughout all autonomous cycles.
   Open Test only after manual termination and final freeze.

5. CONTEXT
   Integrate NDP-A into main research context before
   the first Astra research handoff.
   Explicitly pass the NDP-A full report and raw evidence.

6. TV-GRM
   Explicitly compare with NDP-A.
   Correct SHUFFLE norm matching.
   Evaluate the contribution of residual momentum.
   Clarify per-parameter Adam step and detached views.

7. HANDOFF
   Every cycle:
   Codex report -> main context update -> fresh Astra
   -> research plan -> main agent -> Codex experiment.
