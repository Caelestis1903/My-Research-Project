## Current user continuation instruction — 2026-11-17

Finish Cycle04 in full, including analysis, report, review and publication/delivery. Then execute exactly one additional full research cycle (Cycle05), following DESIGN -> IMPLEMENT -> distinct REVIEW -> supervised current-conversation RUN -> ANALYZE -> report/review/publication. Stop after the Cycle05 report/delivery; do not start Cycle06. This latest instruction supersedes the earlier stop-after-Cycle04 paragraph below. Ponytail FULL and Stop That Shit CHANGE apply; fresh independent native children only, no reuse or independent/background AI.

## Current user termination instruction — 2026-11-17

Finish Cycle04 in full: all twenty trajectories through fifty epochs, analysis, complete research report, evidence review and publication/delivery. Then terminate with USER_STOPPED. Do not start Cycle05 or another Astra research handoff. This overrides older uncapped continuation instructions below. During current-cycle recovery keep the immediate manual stop latch false; it is a stop-after-report request.

# Autonomous Research Runbook — TV-GRM

## Highest-priority current-conversation policy — 2026-11-16

This user policy overrides older driver, session-lifecycle and handoff instructions below. Preserve historical experiment details, scientific gates and source provenance.

- The original foreground orchestrator alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. All tasks and agents stay under the current conversation; only that orchestrator dispatches children.
- Each subtask gets a fresh native child with explicit `fork_context=false`. Coder and reviewer are distinct. Close each child on completion; never resume or reuse it. Handoffs use explicit files/reports only, without inherited conversation context.
- Independent/headless/background Codex sessions, `codex exec/resume/fork`, `tmux`, `nohup`, `systemd`, and `autonomous.py start/run/recover` are forbidden. Create no background agent/driver. The original 50/50 trajectories are complete; the old driver was PAUSED after treating a recovered reconnect warning as fatal and is retired. Preserve its source/receipts; do not repair or restart it.
- Concurrent mechanical training is permitted only under supervision of a runner in the current conversation. Keep the existing RAM guardian as storage; preserve live states and frozen source identities. No detached AI session or replacement guardian/driver.
- RUN dispatch correction from Cycle03: `start.md` explicitly requires all five seeds2022–2026. The legacy named `luna-runner` role has four seeds hardcoded in its developer prompt and refused that approved manifest before launch. Use a fresh native `default` child assigned only the mechanical runner task, with the user-requested `gpt-6-luna`/`high` and `fork_context=false`; preserve the full five-seed manifest and runner/reviewer separation. This path completed Cycle03 unchanged. Do not modify global role configuration, reuse the closed child, or reduce the seed budget. See `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/RUNNER_ROLE_RESOLUTION.json` and its verified publication receipt.
- `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden total/revision cap. Only explicit USER_STOPPED normally terminates; candidate NO-GO/Astra STOP leads through report/context to fresh research, while technical failures PAUSE.
- Check both workspaces' stop controls before dispatch. On user stop, finish/checkpoint the atomic operation safely and produce the final summary. Test stays sealed until explicit user stop + final freeze + the existing eligible selected-RAM verification gate; ACTIVE/PAUSED/candidate outcomes never open Test.

## Preserved protocol and historical provenance

Status: DESIGN COMPLETE; IMPLEMENTATION in progress. Independent review, smoke and formal training pending. Source main: dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11. Approved definition: artifacts/26-11-16_augmentation_gradient_momentum/DESIGN_SPECIFICATION.md.

NDP-A is complete and must not be rerun. TV-GRM tests a distinct optimizer route: clean Adam m/v plus a separately accumulated view-minus-clean first moment, preconditioned by clean v and globally bounded.

Read AGENTS.md, start.md (GB18030), buchong.md and the approved design. PREFLIGHT_RECEIPT.json is PASS: parent gpt-6.1-sol/xhigh; independent native gpt-6-astra/xhigh. Use that receipt, not another capability pilot. A concrete later dispatch failure is PAUSED/AGENT_DISPATCH_BLOCKED with a recoverable next_action.

First round: completed DESIGN -> coder/basic CPU checks -> different reviewer static/CPU REVIEW -> luna-runner real full-batch SMOKE -> needed reviewer smoke signoff -> frozen formal RUN -> main ANALYZE -> full report and substantive main-context publication. Resume hash-valid completed receipts; do not duplicate completed work. Five arms × two datasets × five seeds = 50 new trajectories/2500 epochs/86250 complete updates. Historical CLEAN/NDP-A each reuse ten trajectories.

The stdlib autonomous.py driver detaches from the terminal, starts installed codex exec --json as gpt-6.1-sol/xhigh, records actual thread.started.thread_id and resumes exactly that parent across completed turns. Astra is a fresh native child inside this parent, never a separate exec. The parent launches after independent review; this coder stage does not start it.

Every full report -> actual main-context publication -> fresh Astra >=3 mechanisms/six artifacts -> parent plan completeness -> next DESIGN/coder/different reviewer/runner/main ANALYZE. max_cycles=null; max_astra_cycles=null; max_experiments=null. REVISE fixes missing elements; candidate NO-GO/NO_CONTINUATION_SIGNAL and Astra STOP lead to fresh research. Technical failures PAUSE with diagnostics. Only USER_STOPPED normally terminates.

Check manual_stop_requested and STOP_AUTONOMOUS under artifacts/26-11-16_augmentation_gradient_momentum in BOTH execution workspace and original control /home/xzf/Code/phase0-3 before every driver resume and child dispatch. Finish the current complete parameter/EMA/RAM transaction; retain eligible selected RAM and write final summary. Test requires USER_STOPPED + final freeze + valid eligible five-seed selected RAM, and remains sealed in all autonomous cycles.

Keep natural data, M3oE fixed1, 3 domains × 2 tasks, seeds exactly2022–2026, 50epochs/batch40960/FP32. Online/EMA independently select earliest strict six-cell validation Macro AUC maximum; loss is judged separately. Domain size is not difficulty. No architecture, gate, fusion, sampling or split changes.

Main publication uses the original repository's current main, preserving target main and expected remote head. Initial pinned-HEAD dry-run hit a gnutls_handshake transport error; it was not an auth result. After the real reviewed report/context commit, retry that specific transport once, or use the connected GitHub tree/commit/update_ref tools with expected head. Verify remote main contains the context commit before Astra handoff. Otherwise PAUSED with the exact error. No new branch, credentials or global config changes.


## C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

完整报告research_reports/cell_mass_bce_20261117/REPORT.md。500epochs/17250有用更新/20选中模型已直接验收；实际17333返回包含83次重放成本。原ownerPID2926581当前验收95,508,960字节；模型只在RAM，C04模型仍不可用。完成本轮报告/归档/独立最终证据复审/实际发布后USER_STOPPED，不启动C06，不自动开Test。
