## MDMTRec 研究上下文

- AI 从 [实验索引](mdmt_research_context/01_AI_EXPERIMENT_INDEX.md) 检查既有干预、对照与结论。只有发现新设计与既有结论矛盾，或干预重合需要对照具体方案时，才读取 [02_AI_EXPERIMENT_DETAILS.md](mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md) 对应章节。
- 数值核对使用 [数据对照与性能](mdmt_research_context/03_EXPERIMENT_RESULTS.md)。`mdmt_research_context/04_HUMAN_RESEARCH_REVIEW.md` 仅供人类阅读。AI 禁止获取或读取该路径，包含递归搜索、批量加载以及自动上下文注入。
- 新结果用同一实验编号更新四个文件。先验收对应来源、分割、读出、seed、配对参照和六格，再更新索引结论。重复历史轨迹、派生评价和未运行阶段分开记。
- 正式性能比较使用 seeds 2022–2026。小域不是困难域，禁止以域规模替代困难度或验证这一结论。

## Current user continuation instruction — 2026-11-17

Finish Cycle04 in full, including analysis, report, review and publication/delivery. Then execute exactly one additional full research cycle (Cycle05), following DESIGN -> IMPLEMENT -> distinct REVIEW -> supervised current-conversation RUN -> ANALYZE -> report/review/publication. Stop after the Cycle05 report/delivery; do not start Cycle06. This latest instruction supersedes the earlier stop-after-Cycle04 paragraph below. Ponytail FULL and Stop That Shit CHANGE apply; fresh independent native children only, no reuse or independent/background AI.

## Current user termination instruction — 2026-11-17

Finish Cycle04 in full: all twenty trajectories through fifty epochs, analysis, complete research report, evidence review and publication/delivery. Then terminate with USER_STOPPED. Do not start Cycle05 or another Astra research handoff. This overrides older uncapped continuation instructions below. During current-cycle recovery keep the immediate manual stop latch false; it is a stop-after-report request.

# Research Experiment Orchestration

## Highest-priority current-conversation policy — 2026-11-16

This user policy overrides older driver, session-lifecycle and handoff instructions below. Preserve historical experiment details, scientific gates and source provenance.

- The original foreground orchestrator alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. All tasks and agents stay under the current conversation; only that orchestrator dispatches children.
- Each subtask gets a fresh native child with explicit `fork_context=false`. Coder and reviewer are distinct. Close each child on completion; never resume or reuse it. Handoffs use explicit files/reports only, without inherited conversation context.
- Independent/headless/background Codex sessions, `codex exec/resume/fork`, `tmux`, `nohup`, `systemd`, and `autonomous.py start/run/recover` are forbidden. Create no background agent/driver. The original 50/50 trajectories are complete; the old driver was PAUSED after treating a recovered reconnect warning as fatal and is retired. Preserve its source/receipts; do not repair or restart it.
- Concurrent mechanical training is permitted only under supervision of a runner in the current conversation. Keep the existing RAM guardian as storage; preserve live states and frozen source identities. No detached AI session or replacement guardian/driver.
- `max_cycles=null`, `max_astra_cycles=null`, `max_experiments=null`; no hidden total/revision cap. Only explicit USER_STOPPED normally terminates; candidate NO-GO/Astra STOP leads through report/context to fresh research, while technical failures PAUSE.
- Concurrency follows start.md and the user's 2026-11-16 correction: independently admit another worker on each GPU when CURRENT measured memory and CURRENT measured power are both below 85%, with current host RAM below 90%. Do not add predicted worker power or peak GPU memory to this predicate. Re-sample after each startup/allocation; pending startup protection is allowed, fixed active-worker caps and requiring both cards to fit are forbidden. This overrides projected-reservation admission in older plans/reviews; preserve historical receipts.
- Following an actual CUDA OOM, retain that current85% predicate and additionally require physical fit within measured CUDA-usable total memory using full-process residency measurements. This hardware-capacity check uses usable capacity, not an extra projected85% limit or a worker-count cap. The cycle01 PHYSICAL_CAPACITY_CORRECTION.md records the concrete failure and measurements.
- Check both workspaces' stop controls before dispatch. On user stop, finish/checkpoint the atomic operation safely and produce the final summary. Test stays sealed until explicit user stop + final freeze + the existing eligible selected-RAM verification gate; ACTIVE/PAUSED/candidate outcomes never open Test.

## Preserved protocol and historical provenance

This repository is an experimental research codebase.

The primary Codex agent acts only as the ORCHESTRATOR.

It must not directly implement research algorithms when a specialized
agent exists.

## Required workflow

For every new algorithmic experiment:

DESIGN
↓
IMPLEMENT
↓
REVIEW
↓
RUN
↓
ANALYZE

The stages must not be skipped.

## Stage 1 — DESIGN

Spawn `designer`.

Designer must produce a DESIGN SPECIFICATION.

No implementation may begin before design is complete.

## Stage 2 — IMPLEMENT

Pass the approved DESIGN SPECIFICATION to `luna-coder`.

Coder must implement only the specification.

Coder may perform syntax/basic checks but must not analyze research results.

## Stage 3 — REVIEW

Spawn `reviewer`.

Reviewer must independently inspect:

- DESIGN SPECIFICATION
- actual implementation
- CODER IMPLEMENTATION REPORT

Experiments may proceed only if reviewer verdict is:

PASS

or

PASS WITH FIXES

and all blocking issues have been corrected.

If verdict is FAIL:

return to CODER.

## Stage 4 — RUN

Spawn `luna-runner`.

For formal performance comparison, training seeds are 2022–2026. Use the approved design for any explicitly distinct historical or diagnostic protocol.

Runner records raw results only.

Runner MUST NOT analyze research conclusions.

## Stage 5 — ANALYZE

The primary agent performs scientific analysis.

The primary agent MUST NOT assume:

- Curriculum Learning is effective.
- Momentum mechanisms are effective.
- M3OE necessarily fails on difficult domains.

Small domains are not difficult domains. Keep domain size and difficulty separate, and do not design experiments to test that distinction.

The analysis must evaluate evidence independently.

For experiment analysis examine:

- AUC
- loss
- per-domain performance
- per-task performance
- seed consistency
- domain size
- domain difficulty
- baseline performance
- model
- training stage

Identify:

1. Stable repeated phenomena
2. Contradictory experiments
3. Evidence strength
4. Competing explanations
5. Whether domain size is a plausible main explanatory variable
6. Whether a small-domain improvement / large-domain decline trade-off exists
7. What experiment would best discriminate competing explanations

Evidence strength must be labeled:

STRONG
MEDIUM
WEAK

The orchestrator must prefer falsification-oriented experiments over
experiments that merely confirm the current hypothesis.

## Project Python environment

- The project environment is the Conda environment `phase0-3-env` at `/home_new/xzf/envs/miniconda3/envs/phase0-3-env`.
- It uses Python 3.13.5, PyTorch 2.9.0+cu126, and the exact pip package snapshot in `requirements-phase0-3.lock`.
- In new SSH terminals, activate it with `source /home_new/xzf/envs/miniconda3/etc/profile.d/conda.sh` followed by `conda activate phase0-3-env`.
- Codex must run project Python commands explicitly through `conda run --no-capture-output -n phase0-3-env python ...` or `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; do not use Conda `base`.
- Run tests with `conda run --no-capture-output -n phase0-3-env python -m pytest ...`.
- Keep this setting scoped to this repository; do not change global shell or Conda defaults.
- PyTorch includes the CUDA 12.6 runtime. This setup session had no NVIDIA device exposed (`torch.cuda.is_available()` was false), so check GPU visibility before starting GPU experiments.

## Local dataset paths

- MovieLens CSV: `/home_new/xzf/datasets/ml-1m/ml-1m.csv`.
- KuaiRand processed CSV: `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`.
- Both CSVs were verified against the existing project SHA-256 records after relocation.
- The legacy `/home/xzf/data` locations are no longer valid. Paths inside historical experiment records and pinned source snapshots record their original locations.
- Prepared data expected under `/home_new/xzf/datasets/mdmt_next_20261024_strict_v1` was not present in the migrated datasets or the local recovery directory. Restore and verify these artifacts before running workflows that require them; a corrected path alone does not make those workflows ready.
