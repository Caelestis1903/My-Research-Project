# 可直接交给 Codex 的执行提示词

你现在负责实施 My-Research-Project 的完整研究任务。首先读取 main 中 research_plans/26-11-22_conditional_history_shared_update/00_EXECUTION_RUNBOOK.md，并读取同目录01–10、12–13与experiment_manifest.json。方案已完成研究设计；依据阶段门执行，不重新从零设计研究方向。

## 任务结果

完成方向A条件样本对/历史更新的独立收益判别、方向B共享参数更新机制判别、所有实际触发的有限算法候选和五种子评价，产出完整研究报告、原始数表、实现与上下文更新。允许最后没有合格新算法，但必须得出明确的保留/停止决定。

## 开始时

1. 核对git状态与研究快照875215c1e509ab55c1b6632d4f5d13e8b94782da，保留当前用户改动。
2. 阅读六个总研究上下文、AGENTS.md、AGENT.md、start.md。本轮附件/规划中的正式seed2022–2026、两方向完整执行、最终验证选点后Test与限定任务边界优先于旧四seed、旧二seed门及旧C04/C05循环。
3. 按12_SOURCE_AND_REUSE_REGISTER恢复main与必要固定来源SHA；保留旧结果，不把来源分支全部合并。
4. 核验数据hash、模型/参数布局、source loss、Adam/EMA、batch order、pair index、选点规则和已有复用身份。只补P00/P10缺失seed2023/2025/2026，共12条。
5. 预计B缺完整历史状态，准备10条同源clean重放。历史selected模型/收据不能冒充Adam可恢复状态。

## 工程与人员流程

由主agent负责调度与科学分析；coder和reviewer为不同的全新独立上下文子agent，使用与主agent一致的模型，机械RUN可用适合运行的上下文。不要创建独立后台AI会话，不用codex exec/resume/fork、tmux/nohup/systemd或旧autonomous driver启动研究。

在新目录 experiments/conditional_history_shared_update_20261122/ 实现06定义的接口与CLI。复用现有v3数据、模型、pair、ECCM和evaluate内核；外层不能照搬旧run.py里的二seed限制、固定并发/6GiB规则。真Adam反事实采用CWA源码中深拷贝真实模型/optimizer/None与私有step的模式，不借用PUB全flat更新替代原生Adam。

完整DESIGN规范 → coder实现及报告 → 独立review通过并修复阻塞 → supervised RUN → 主agent分析。必要测试仅覆盖会改变科学结论的公式、梯度分解、实际/虚拟Adam相等、RNG不污染、RAM恢复、选点和五seed门。

## 科学顺序

### A核心

完成A00/A10/A20/A30五seed矩阵，其中28条历史科学轨迹数值复用、12条新补全。比较相同监督下A20−A10；不要把A20+F相对online的总收益当作条件历史的独立收益。

若历史包有正向Validation均值，运行A21静态PCOLL与A22单份S_dense长Adam（两数据×5seed×2臂）；从简单替代中选择。现有PCOLL默认动态抽样，必须显式dynamic=False。

若保留历史且有已登记的六格代价，运行λ=.025的历史与普通Adam各十条，λ=.05复用；根据Validation选择。需要限制历史作用域时，先有本轮真实轨迹定位，才运行A50/A51，不能从旧Test坏格直接设计门控。

### B核心

无论A结果如何，执行B的十条clean状态重建及40锚点核心干预。必须保持原始自然BCE权重：L_dt按2×实际batch大小归一化，六格求和恢复原loss。作用组按真实坐标分EMBED、S_DENSE、N；domain embedding三行属于D。

每锚点真JOINT/ONLY/LOO/ZERO_DATA及匹配伪分组，总2080虚拟Adam、720额外autograd；等范数比较单列读出。检查真实当前更新的六格排序效用，不用负cosine代替负迁移。EMA缩放恒等式不是任务过滤结论。

按04/09的门，在完整五seed数据集上运行一次LOO脉冲、后续普通联合训练，并做有限窗口参数/预测平均。任务/伪分组的源选择只用训练面板，Validation用于评价，不能筛选有利seed或anchor。

B机制支持可操作EMBED残余时，再运行B_ADAPT/B_STATIC/B_UNIFORM ×η{.25,.5} ×两数据×5seed，总60条条件候选。B_UNIFORM按ADAPT日志重放名义质量缩放；三者都必须直接胜正常A00+F，不能以胜被劣化的诊断控制替代收益。

### 联合与最终评价

只有至少同一数据上A/B各自有效且06第11节正面桥接通过才运行固定AB十条；Gaussian背景只有能决定最终方法是否包含增强时才运行十条。AB/G严格执行06第11节通道公式，B只减弱自然gq、PAIR独自进入A条件历史、Gaussian残差归普通通道；不临场重新设计组合。既有三发表模型默认复用，不重新做已经完成的AUC选点统一。

所有HPO每配置五seed，不用单seed否决。通过本轮FINAL_FREEZE后，对冻结候选和必要匹配控制一次性Test，按最早严格最大Validation六格Macro AUC选点。Test不反向触发调参/新臂。历史Test暴露如实标注，旧C01–C05封存任务不自动打开。

## 资源和恢复

使用 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python。自然CSV路径在06/08。每卡当前显存<85%且当前功耗<85%、host RAM<90%时可准入；再检查实际CUDA可用物理容量，不设固定worker数、不要求两卡同时满足。每次启动/分配后重新测量。

严禁将模型、optimizer、EMA、RNG、loader或可恢复训练快照落盘。状态通过当前会话监督的RAM进程保存，保存完必须现场get/hash验证。磁盘只存允许的数值、日志、配置、代码、报告。owner丢失时只在必要时同源重放，单列物理成本，不增加n或挑选最好attempt。不要擅自减batch、改人口、学习率或AMP。

## 最终必须交付

- 完整五seed原始六格AUC/LL、Macro/Traffic/Pooled、Validation/Test与online/F身份；
- A独立效应、简单替代、六格保护与B实际更新/EMA/身份对照三机制结论；
- 阶段门输入、每个触发/未触发/技术阻塞原因；
- 源码/数据/运行清单、复用和状态重建账本、HPO全部值、总成本和恢复记录；
- 最终报告与独立审查、总上下文及算法台账回填；
- 按仓库当前实验分支规则提交实现和完整结果，回读远端commit，提供SHA和链接。

本轮完成所有已触发阶段与报告后停止，不开启无上限下一循环。用户发出停止时安全结束当前原子操作并交付已完成状态。A分支停止不能取消B核心；技术失败不能写成科学否决；不会为了产出新方法强行叠加两个无效组件。
