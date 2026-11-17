## Current user continuation instruction — 2026-11-17

Finish Cycle04 in full, including analysis, report, review and publication/delivery. Then execute exactly one additional full research cycle (Cycle05), following DESIGN -> IMPLEMENT -> distinct REVIEW -> supervised current-conversation RUN -> ANALYZE -> report/review/publication. Stop after the Cycle05 report/delivery; do not start Cycle06. This latest instruction supersedes the earlier stop-after-Cycle04 paragraph below. Ponytail FULL and Stop That Shit CHANGE apply; fresh independent native children only, no reuse or independent/background AI.

## Current user termination instruction — 2026-11-17

Finish Cycle04 in full: all twenty trajectories through fifty epochs, analysis, complete research report, evidence review and publication/delivery. Then terminate with USER_STOPPED. Do not start Cycle05 or another Astra research handoff. This overrides older uncapped continuation instructions below. During current-cycle recovery keep the immediate manual stop latch false; it is a stop-after-report request.

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


## C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

完整报告research_reports/cell_mass_bce_20261117/REPORT.md。500epochs/17250有用更新/20选中模型已直接验收；实际17333返回包含83次重放成本。原ownerPID2926581当前验收95,508,960字节；模型只在RAM，C04模型仍不可用。完成本轮报告/归档/独立最终证据复审/实际发布后USER_STOPPED，不启动C06，不自动开Test。
