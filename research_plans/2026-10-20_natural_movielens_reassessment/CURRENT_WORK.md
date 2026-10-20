# 当前执行状态（整体任务尚未完成）

实际worktree `/home/xzf/Code/phase0-3-natural-git`，分支 `codex/natural-movielens-reassessment-2026-10-20`；原用户目录报告中的 execution symlink 指向此报告目录。ART=`/home_new/xzf_natural_movielens_reassessment`。

## 当前规范

用户补充的 start.md 第11条：设计/实现/自审/分析由主代理完成，只把机械训练/评价交 luna_worker。先前已完成的独立设计/review记录保留；新自审不称独立审查。五阶段仍完整记录。最小必要验证，完整预算；每GPU最多5本研究进程，总显存实际占用不超过90%，逐卡准入，不碰外部进程。

## 设计与放行

全部历史H01–H31审计、672原N19/CM19配置索引、完整数据及A–F设计完成。1044基础名义配置含16pilot、1028formal。现在全部训练族实现/审查放行：V2 16、core120、N17初始64及pilot后8quota模板、AO04 12、N19 316、CM19 508；初始1036可执行，另8required模板待共同LR。额外LR/H200及N3精度分支按结果前冻结判据执行，不按效果符号删项。没有应用历史alias减少实际训练。

V2与AC2D/S5/S6有先前独立PASS；V9按start主代理修正中断重跑策略后PASS。N17/transfer/AO04/N18主代理自审PASS，gate及源码已提交冻结。运行期间不可修改冻结wrapper/vendor；科学分析新增文件不在训练source集合。N19/CM19保留原模型、更新和合成机制，仅注入新人口/路径/预注册预算。

## 运行状态

机械agent `/root/luna_execution`（luna_worker）拥有统一RUN_LEDGER和ART/runner调度器；不要启动重复GPU队列。core协调器接管现有children，当前controller PID以ART/runner的state/事件记录为准（已发生只换父进程、不重训children的调度修复），保持每卡5进程，旧parent丢失的OS exitcode如实UNKNOWN_MONITOR_LOST，完整预算另核。新published_queue_plan包含N17/transfer/AO04后续配置，core继续不中断。实际最新计数见RUN_LEDGER；本文件不代替实时台账。

失败attempt完整保留，不当科研阴性；同seed完整重跑。未读test的阶段必须等待主代理对应closure及SHA冻结，不能因训练exit0自动放行。N17 pilot冻结、plateau、precision gate按预注册机械规则执行并通知主代理；不得遗漏quota/触发追加。

## 已完成科学分析

V2 16完整50epoch，selected/final32份预测逐cell复算，见V2_PHASE_ANALYSIS.md与analysis/v2。AC-2D 64/64训练与测试完成、448份预测2688cell复算，性能分析见AC2D_PERFORMANCE_ANALYSIS.md；160固定点机制任务仍待执行，不能标机制完成。AC-V9 8/8完整训练，56份预测336cell复算，原定公共点/selected/final分析见ACV9_PERFORMANCE_ANALYSIS.md；按历史Q41固定窗口补读全部400个已训练冻结模型，0新增训练，尚待执行。AC-S5两预算32/32完整配置已分析、1344cell预测复算，ACS5_PHASE_ANALYSIS.md；两预算同seed初始化及每预算四臂实际批次一致，含预算×控制交互。批次/初始化身份、原公式、主/arch实际预算与累计支持及full-coordinate状态窗口已整理，analysis/controlled_updates。AC-S6两预算16/16完整已分析、672cell预测复算，ACS6_PHASE_ANALYSIS.md；selected两预算DH−F均四负，full final四正/LL四好，matched final四负/LL四差。core120与V2 16均已完成；后续队列已进入N17 pilots，两卡各5进程。N18自然数据离线诊断完成，0训练，见N18_PHASE_ANALYSIS.md与analysis/n18；实际mask/audit哈希与同信息LP身份有据。部分阶段完成不等于整项研究闭环。

历史S5/S6全部64原CSV已索引/聚合，analysis/controlled_history；原模型/optimizer bytes缺失不伪称旧预测重算。N17全部53问题原记录和N19/CM19全部遗留问题已完整复制并SHA绑定，analysis/question_registry，另有53项阶段性对应表。OPEN_PROBLEMS.md含原问题来源与23项当前问题/矛盾，仍需后续结果追加，不是最终清单。

历史N19/CM19已实际重新聚合1184 run/29856 cell、747完整四seed比较、28104逐cell配对，并复核单任务baseline口径；见HISTORICAL_NUMERIC_RECONSTRUCTION.md、analysis/{transfer,transfer_cells,neutral_interventions}。CM19 856旧run的69448固定探针汇总、model SHA身份和梯度尺度上下文在analysis/cm19_paths，大表以确定性gzip提交，原heavy tensor没有重读。N19 328旧run的审核支持/known-channel RMSE、实际mask预算、rank记录在analysis/n19_records，四优化seed不是四审核draw。MECHANISM_ANALYSIS.md目前只包含已完成权重算术、解释边界和待完成项，不是机制终报。

已补冻结人口的user/movie ID覆盖审计（FEATURE_SUPPORT_AUDIT.md），无训练/标签拟合；同N仍改变域1用户和同域电影支持，不能独立归因AUC。AO04旧两seedML/一seedKR数值已重新汇总并保留身份限制，analysis/ao04_history。NUMERIC_COMPARISONS.md目前合并已完成阶段210条四seed宏指标比较，明确阶段性、没有填补未运行值。S5/S6共享F/DH的16跨目标配对初始化/实际batch身份和目标×控制交互见analysis/controlled_objectives。

N17旧12,768条配对端点及371逐seed数组已重新汇总复算，含完整AUC/LL/阶段、quota×q及N2/N3原门控；见analysis/n17_history，未重载旧大工件，未与新优化seed跨人口配对。

旧B1自然KR重建的1360条selected及13600条同epoch指标已重新汇总，含主R0阴性、R1/R4同模型阳性、target LL代价和timing算术限制；analysis/b1_history。该重建不是历史exact初始化，新自然S6不冒称覆盖未迁移的全部敏感度规则。

后续分析入口已备（均由主代理执行）：analyze_core.py --stage AC-S5/AC-S6与analyze_controlled_updates.py只准入完整四seed cohort；compare_controlled_history.py合并新旧。N17 test闭锁前可运行analyze_n17_mechanisms.py检查N2/N3/precision；正式test后analyze_n17.py。AO04冻结评价后analyze_ao04.py。transfer冻结test后依次collect_transfer.py、analyze_transfer.py、analyze_transfer_cells.py、analyze_neutral_interventions.py；模型路径/记录另用analyze_cm19_paths.py、analyze_n19_records.py。这些准备不代表新训练/诊断已完成。

机械顺序已调整：core全部→N17 pilots/freeze→V9 8曲线只读任务→N17其余训练/预定H200→N2/N3/precision→通知主代理冻结N17→AO04训练/配对init/局部诊断→通知冻结AO04→160 core诊断→N19 316及scope CPU诊断→CM19 508及scope CPU诊断。主代理发布TEST_FREEZE和命令后，worker在下一批次/阶段边界优先接入同一GPU槽位，不能额外超5。transfer诊断receipt真实路径是scope/diagnostic_receipts，不含states；机械runner已修正，没有更改研究producer。

08:35 UTC机械日志记录过提前启动N17的准入检查：现有core和发布协调器的活跃PID登记独立，安全交错需要shared-slot/adopt改造；按最小调整原则保留原core完成barrier，不改producer、不停止健康children。证据在ART/runner/operational_admission_events.jsonl。阶段内部继续按每卡上限准入。

## 必须继续

完成所有core/N17/AO04/N19/CM19及160core/143transfer/N17/AO04机制任务、对应test冻结/评价；主代理逐阶段科学分析，形成NUMERIC_COMPARISONS/MECHANISM_ANALYSIS/RESEARCH_CONCLUSION_REASSESSMENT/OPEN_PROBLEMS/FINAL_REPORT与完整原始索引。必须完整报告逐域任务、AUC/LL、选中/末轮、四seed、域规模/难度/基线和竞争解释，证据STRONG/MEDIUM/WEAK。最后提交推送独立分支并建对main的PR，不合并。当前尚无最终PR，不能冒报交付完成。
