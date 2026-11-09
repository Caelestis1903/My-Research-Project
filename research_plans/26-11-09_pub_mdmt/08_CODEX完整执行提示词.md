# Codex完整执行提示词：PUB-MDMT

在GitHub仓库 Caelestis1903/My-Research-Project 执行本研究。先阅读本目录00–07文档。当前任务阶段为研究规划，不含训练实现；执行时先审计，再实现，不得直接开始长训练。

## 必读来源
main：EXP_AI_CONTEXT.md、experiments/Experimental_Global_Summary.md、MDMT_ALGORITHM_METRICS_LEDGER.md。
分支26-11-09：research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md、NUMERICAL_TABLES.md、IMPLEMENTATION_REVIEW.md及A0决策产物。
相关实现：experiments/aug_history_20261105/{group_registry.py,registry.py,model_adapter.py}；11/06 EMA机制实现与报告；clean/Gaussian损失实现。
论文arXiv:2410.05806；作者仓库yjdy/Multitask-Optimization-Recommendation-Library。

## 审计
1. 记录main和26-11-09 SHA、merge-base、ahead/behind；不合并实验分支。
2. 固定作者仓库commit SHA，定位PUB solver/optimizer代码，输出论文与源码对照。若无法取得源码，明确标记不可审计，不得编造。
3. 核验fixed1计算图、六格输出、12组坐标和autograd支持。
4. 证明六个L_dt求和逐值等于原始自然样本loss且梯度一致。
5. 确认Adam版本、bias correction、eps、weight decay、step计数和其他选项。
6. 输出审计报告、资源估算和测试计划；关键语义无法核实时暂停实现。

## 实现
新增独立包experiments/pub_mdmt_20261109/，不得修改历史vendor、现有EMA或原始算法。实现P0两任务、P1六任务全局、P2拓扑感知，以及同alpha原始梯度Adam控制。
候选更新必须共享同一更新前m/v/step；全部候选完成后才更新参数和状态。任务顺序不得污染状态；每坐标只写一次；自然样本目标必须精确恢复。

weight decay必须先核实；不得对每个任务重复加入同一正则项，不得无说明改AdamW。写出公式、与baseline单步对照和测试。

## 单元测试
损失/梯度重构、候选更新手算、任务顺序置换、状态无污染、坐标覆盖、DT普通Adam、WD、断点恢复、RNG/loader复现、solver数值和alpha残差。全部通过后才开始科学训练。

## 训练评价
MovieLens/KuaiRand，Seeds 2022–2026，M3oE fixed1；原预算、初始化、自然目标、数据顺序、batch和学习率保持一致。对照Adam、固定EMA、P0/P1/P2及对应同alpha梯度控制。PUB+EMA仅在PUB本身有收益后执行。
先每10 step更新alpha；CCP迭代{50,100,200}、tol{1e-4,1e-6}、Gram相对稳定项{1e-12,1e-10,1e-8}小规模搜索。最终配置五seed，少seed不得否决。Test只在选点和超参冻结后读取。

## 交付
产出审计、实现规范、单元测试、manifest、运行日志、逐seed结果、六格指标、Macro/Traffic/Pooled、alpha/Gram/更新诊断、训练成本和决策报告。实验结束后按项目规范回填main总上下文与指标台账。PUB不得列为正式发表模型baseline。

禁止改变采样或增加数据增强。小域不是困难域，不得对其进行任何假设性分析或实验。
