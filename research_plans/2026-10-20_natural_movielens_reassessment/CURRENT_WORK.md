# 当前执行状态（整体任务尚未完成）

实际worktree `/home/xzf/Code/phase0-3-natural-git`，分支 `codex/natural-movielens-reassessment-2026-10-20`；原用户目录报告中的 execution symlink 指向此报告目录。ART=`/home_new/xzf_natural_movielens_reassessment`。

## 当前规范

用户补充的 start.md 第11条：设计/实现/自审/分析由主代理完成，只把机械训练/评价交 luna_worker。先前已完成的独立设计/review记录保留；新自审不称独立审查。五阶段仍完整记录。最小必要验证，完整预算；每GPU最多5本研究进程，总显存实际占用不超过90%，逐卡准入，不碰外部进程。

## 设计与放行

全部历史H01–H31审计、672原N19/CM19配置索引、完整数据及A–F设计完成。1044基础名义配置含16pilot、1028formal。现在全部训练族实现/审查放行：V2 16、core120、N17初始64及pilot后8quota模板、AO04 12、N19 316、CM19 508；初始1036可执行，另8required模板待共同LR。额外LR/H200及N3精度分支按结果前冻结判据执行，不按效果符号删项。没有应用历史alias减少实际训练。

V2与AC2D/S5/S6有先前独立PASS；V9按start主代理修正中断重跑策略后PASS。N17/transfer/AO04/N18主代理自审PASS，gate及源码已提交冻结。运行期间不可修改冻结wrapper/vendor；科学分析新增文件不在训练source集合。N19/CM19保留原模型、更新和合成机制，仅注入新人口/路径/预注册预算。

## 运行状态

机械agent `/root/luna_execution`（luna_worker）拥有统一RUN_LEDGER和ART/runner调度器；不要启动重复GPU队列。core协调器 `natural_core_adopt_coordinator.py` PID111227 接管现有children，保持每卡5进程，旧parent丢失的OS exitcode如实UNKNOWN_MONITOR_LOST，完整预算另核。新published_queue_plan包含N17/transfer/AO04后续配置，core继续不中断。实际最新计数见RUN_LEDGER；本文件不代替实时台账。

失败attempt完整保留，不当科研阴性；同seed完整重跑。未读test的阶段必须等待主代理对应closure及SHA冻结，不能因训练exit0自动放行。N17 pilot冻结、plateau、precision gate按预注册机械规则执行并通知主代理；不得遗漏quota/触发追加。

## 已完成科学分析

V2 16完整50epoch，selected/final32份预测逐cell复算，见V2_PHASE_ANALYSIS.md与analysis/v2。N18自然数据离线诊断完成，0训练，见N18_PHASE_ANALYSIS.md与analysis/n18；实际mask/audit哈希与同信息LP身份有据。两阶段完成不等于整项研究闭环。

## 必须继续

完成所有core/N17/AO04/N19/CM19及160core/143transfer/N17/AO04机制任务、对应test冻结/评价；主代理逐阶段科学分析，形成NUMERIC_COMPARISONS/MECHANISM_ANALYSIS/RESEARCH_CONCLUSION_REASSESSMENT/OPEN_PROBLEMS/FINAL_REPORT与完整原始索引。必须完整报告逐域任务、AUC/LL、选中/末轮、四seed、域规模/难度/基线和竞争解释，证据STRONG/MEDIUM/WEAK。最后提交推送独立分支并建对main的PR，不合并。当前尚无最终PR，不能冒报交付完成。
