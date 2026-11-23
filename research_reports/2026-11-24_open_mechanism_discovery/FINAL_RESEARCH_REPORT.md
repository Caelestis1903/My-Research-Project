# 开放式机制发现与研究方向报告（2026-11-24）

**研究范围**：MDMTRec / M3oE 的共享—专有学习、训练动态、样本—任务依赖、负迁移与语义保持增强。  
**文献截止**：2026-11-24。  
**仓库核对**：\`Caelestis1903/My-Research-Project\`，main 当前提交 \`f66bf17cba2980618e6f2edd1c7381b73c35cac5\`；共享—专有协调实验分支 \`26-11-23_v2\` 相对该 main ahead 1、behind 0。  
**任务性质**：研究发现与机制筛选；不实现算法、不启动训练、不撰写 Codex 执行计划。

---

## 0. 执行摘要

### 0.1 当前结论

**不建议把下一阶段继续定义为“寻找更好的 EMA / 动量系数”，也不建议把 11 月 23 日静态共享—专有更新顺序实验的 NO-GO 理解成共享—专有协调问题已经被排除。更值得推进的是：研究共享更新如何改变各域任务专有分支的可学习性，并据此决定后续专有更新是否、何时、以什么范围执行。**

本轮开放检索与仓库审计后，形成四个优先机制：

1. **共享更新诱导的专有梯度一致性（最高优先）**：专有参数不只是共享参数更新后的被动接收者；专有头状态会反过来改变共享梯度。需要研究共享更新前后，域任务专有分支对共享表示产生的梯度是否出现有结构的变化，以及这种变化能否指导专有更新，而不是继续测试一个固定的 U→P 顺序。
2. **样本—任务条件影响调度（高优先）**：同一训练样本对不同任务、域任务单元的影响并不相同。相较一般任务权重或域配比，更有价值的是辨认哪些样本对某个任务有帮助、对另一个任务却形成干扰，并将这种条件影响映射到实际训练动作。
3. **阶段性、组合式训练影响归因（高优先诊断方向）**：样本或来源的影响可能随训练阶段改变符号，也可能存在单样本贡献无法相加解释的互补/冗余交互。应由阶段性和成组影响证据决定是否需要新的调度机制。
4. **域任务条件下的有害特征依赖干预（中高优先）**：SAF、MT-CCL、ORCA等说明负迁移可以来自不稳定特征依赖或任务捷径，而不只是梯度夹角。但本项目不能再把一般去相关、一般因果对比或样本重加权包装成新方向；必须定位到具体域任务单元的错误依赖，并用干预确认。

**增强仍可保留，但应从“扰动什么幅度”转向“哪些输入变化在多个标签下仍然语义合法”。** 如果当前数据特征无法定义合法变换，D4应降级，不应靠生成模型人为补造语义假设。

### 0.2 本轮重要修正

- 用户上传的《MDMTRec_2025_2026_完整研究报告.md》以 main 的旧提交 \`4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983\` 为研究上下文。当前 main 已前进到 \`f66bf17cba2980618e6f2edd1c7381b73c35cac5\`，其常驻科研上下文改为 \`experiments/reports/EXP_AI_CONTEXT_V2.md\`，包含截至 S6 的新裁定。不能把旧报告的研究状态直接当成当前最新状态。
- 当前 main 的最新上下文把早期“固定 EMA 仍是所有问题的中心”进一步收敛：Equal 下有条件性的 Adam history 现象；Sample 下两个已测最大域 target 均未通过 practical rescue gate，S6 停止 Phase B。这个新结论限制了“Adam history 是通用方法”的外推。
- \`26-11-23_v2\` 的 D2 共享—专有协调实验确实完成了 S1 诊断矩阵：10 条基线轨迹、30 个 anchor、570 个 fork，未使用 Test。但最终 \`SCOPE_DECISION.json\` 是 \`COMPLETE_NO_GO_S1\`：V_A 选择的 P_AXIS 在独立 V_B 上没有达到确认要求；MovieLens 与 KuaiRand 均未确认通过。它排除的是当前固定 scope / 固定更新动作在当前协议下的晋级，不是所有共享—专有协调机制。

### 0.3 研究决策

建议将下一轮机制发现聚焦在 **“共享更新—专有梯度响应—条件更新动作”**，并将样本影响归因作为并行诊断工具。不要同时堆叠任务图、因果去相关、动态 EMA、生成增强和新 MoE 架构。下一阶段首先要确认一个具体可观测的 MDMT 现象，再选择一个最小控制动作。

---

## 1. 研究范围、证据层级与仓库上下文

### 1.1 文献报告的作用与边界

上传的 2025–2026 文献报告收录68篇去重文献，其中2025–2026会议56篇，严格A类MDMT会议论文10篇，并区分相邻推荐/MTL工作与可迁移的通用机制。其最重要的知识结论仍然有效：

- 严格MDMT研究主体仍以结构/表示选择性共享为主；
- 相邻多任务学习、跨域推荐与训练数据归因提供了更丰富的过程机制；
- 梯度相似、表示相似、实际迁移收益不是同一概念；
- 固定参数 EMA 的已有收益应被接受，不需要在每个新方向中重新证明；
- 一般自适应 EMA、通用梯度投影、域大小驱动权重、一般采样/批次调度、简单 ranking loss 替换等已有较多重叠实验。

本报告不重复整理这68篇论文，而是在其知识地图上补充未纳入的2025–2026研究，并重新用当前 main 与 D2 实验结果校准研究优先级。

### 1.2 当前 main 的最新研究裁定

当前 main 的 \`experiments/reports/EXP_AI_CONTEXT_V2.md\` 是最新常驻上下文，不能再以旧的 \`EXP_AI_CONTEXT.md\` 路径或旧提交摘要替代。

当前可接受的关键事实：

| 主题 | 当前裁定 | 研究含义 |
|---|---|---|
| Equal / Sample | 是不同有效训练目标；Sample 在已测自然比例与协议下缓解 Equal 条件性损伤 | 不把 Equal 下的现象直接外推到 Sample |
| 域规模 | 不足以单独解释退化；域身份、目标、暴露、标签分布和训练阶段共同变化 | 不以“小域=困难域”构造算法动机 |
| Sample 下 residual negative transfer | S1 的72组没有达到稳定 practical NT；S2 的24组亦无达到阈值的单元 | 不再以普遍 Sample 负迁移为未经验证的前提 |
| Equal 下局部救援 | KuaiRand Equal/tab1 有明显条件性 target rescue；Sample 下两个已测最大域 target 未通过 practical gate | 保留目标、objective、共享访问之间的条件性 |
| Adam history | Equal 下 S5 将 d1 rescue 定位到 persistent Adam history 的整体载体；Sample 下 S6 未达到 practical rescue gate | 不把 Adam history 拆解或动量改造视为当前无条件主线 |
| 梯度方向/幅度 | raw-gradient norm 与即时 applied-step norm 匹配均未复现 Equal 下的 target rescue | 仅比较梯度大小或一步位移不足以解释该现象 |
| 跨架构 | P7 有限结构移植结果不能替代 S2–S6 的机制识别 | 需要区分机制复现与表面指标相似 |

其中最重要的是 C13–C14：Equal 下 KuaiRand d1 的历史介导救援证据较强；Sample 下 KuaiRand d1 与 MovieLens d2 两个最大域 target 均未达到 +0.003 practical threshold，且没有运行 HT/DH Phase B。因此“Sample 下不存在任何 history effect”过强，但“Equal 下机制可直接迁移到 Sample”也不成立。

### 1.3 11月23日 D2 实验的独立核对

核对对象：\`26-11-23_v2\`，其与当前 main \`f66bf17...\` 的 merge-base 相同，ahead 1、behind 0。实验工件位于：

- \`artifacts/mdmt_shared_private_coordination_20261123/S0/\`
- \`artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json\`
- \`artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json\`
- \`artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json\`
- \`research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md\`
- \`research_reports/mdmt_shared_private_coordination_20261123/DESIGN_ERRATUM.md\`

**实验范围**：MovieLens 与 KuaiRand；训练 seeds 2022–2026；固定1模型；三域×两任务；50 epochs；natural 两任务目标；batch size 40960；Adam 与参数 EMA 沿用该数据集既定参数。S1 在 E10/E25/E40 建立 anchor 并对相同训练状态 fork，测 U、P_CELL、P_AXIS、P_FULL 等 private scope 与不同更新次序/联合更新对照。

S0 工件显示：
- 模型为3 domains、2 tasks、4 shared experts，FCN [512,256,256,64]；
- U 组66个参数张量，P_CELL 48个，P_AXIS 20个，P_FULL 68个；
- fixed1 fusion constants 为普通数值常量，不存在可训练的四个 fusion scalar；
- MovieLens train/validation/test 行数为800167/100021/100021；KuaiRand为2006132/250767/250767；
- 数据预处理、模型概率和源实现的静态/运行验收工件为 PASS；Test provenance 标记 SEALED；
- S1 完成10条轨迹、30个 anchors、570个 forks，Test evaluations=0。

**最终 S1 gate**：
- \`SCOPE_DECISION.status = COMPLETE_NO_GO_S1\`；
- V_A 对 P_AXIS 仅 KuaiRand 入选；P_CELL 与 P_FULL 均无合格数据集；
- V_B 独立确认中，MovieLens 的 P→U 平均 Macro AUC +0.0000113、J_ACC −0.0000011；KuaiRand P→U +0.0000527、J_ACC −0.0000820；
- 两个数据集均没有复现至少两个 anchor 的双对照一致正向要求，\`confirmed_datasets=[]\`；
- 最终原因是 \`INDEPENDENT_V_B_DID_NOT_CONFIRM\`。

这些数值处于 \(10^{-5}\) 量级，远低于项目常用的 +0.003 实用增益门槛。V_A 的局部正向不能替代独立 V_B 确认。

**正确结论**：静态私有参数范围与固定先后顺序在该实验协议下未产生可晋级的持续排序收益。不能据此推出“共享更新与专有适配之间没有动态相互作用”。当前 D2 操作把 scope 与更新顺序预先固定，尚未测试由共享更新诱导的私有梯度变化能否预测哪类专有更新有效，也没有测试样本条件或训练状态条件的更新策略。

---

## 2. 新增文献与机制补充

以下工作在上传报告的论文索引中未见对应记录，作为本轮开放检索新增候选。分类上，除 MTCCM 外均为通用 MTL、数据归因或跨领域机制；不能冒充严格MDMTRec结果。

### 2.1 ConsMTL：专有参数会反向塑造共享梯度

**Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters**，CVPR 2025，Xiaohan Qin, Xiaoxing Wang, Junchi Yan。

- 作者指出，传统梯度冲突方法主要在共享参数上做梯度聚合，却忽略专有参数也会影响反向传播到共享表示的梯度。
- ConsMTL 把训练写成双层优化：上层在共享参数上寻找联合更新方向；下层通过专有参数附加目标，让各任务回传到共享参数的梯度逐步接近联合更新方向。
- 这不是“对专有参数做 EMA”，而是改变专有参数在共享梯度形成过程中的角色。
- 论文在 Cityscapes、NYUv2、QM9、CelebA 等任务上实验，任务数从2到40；官方补充材料报告每个实验3 seeds。作者代码为 [MTLlib](https://github.com/jianke0604/MTLlib)。

**对本项目的价值**：与 D2 静态 U→P / P→U 更新顺序相比，ConsMTL 指向更本质的问题：P 不仅在 U 之后适配，也会改变下一次 U 的梯度。值得把共享更新前后各 cell 的私有参数状态与其诱导的共享梯度联系起来。

**边界**：CVPR工作是通用视觉MTL，不是推荐；其“梯度一致性”目标不能直接等同六格 AUC 收益。若仅把 ConsMTL 公式移植到 M3oE，属于方法迁移，不足以构成独立创新。需要发现域×任务条件下的非对称响应，并证明它能改变有效更新动作。

来源：[CVPR正式论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.html)；[全文PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf)；[补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Qin_Towards_Consistent_Multi-Task_CVPR_2025_supplemental.pdf)。

### 2.2 DTJS：同一数据对不同任务的影响不同

**Adaptive Data and Task Joint Scheduling for Multi-Task Learning**，ICDE 2025，Zeyu Liu et al.

- DTJS 指出，任务梯度关系不足以表达样本如何塑造各任务梯度；同一训练样本对不同任务的影响程度可能不同。
- 方法包含 intra-task scheduling（根据样本在任务内的影响进行 easy-to-hard 调度）与 inter-task scheduling（学习任务冲突感知并调整任务训练过程），并以双层优化交替更新模型与冲突感知器。
- 作者公开 [DTJS代码](https://github.com/ZeyuLiu0706/DTJS)。

**对本项目的价值**：它提供一个比域权重/任务loss权重更细的研究单位：样本×任务的影响。对 M3oE，域身份、任务标签、样本特征与训练状态共同决定一次共享更新接收到什么信号。

**边界**：DTJS仍是通用MTL，且“样本难度”与“样本对目标排序的有益影响”不是同一概念。项目已有采样、配额、任务视图及增强方向实验；不能再次做一个按 loss 或梯度范数排序样本的 easy-to-hard scheduler。新增空间必须在于同一样本对不同域任务单元的条件影响差异，并证明这种差异改变更新选择。

来源：[IEEE正式论文页](https://ieeexplore.ieee.org/abstract/document/11113158)；[DBLP会议记录](https://dblp.org/rec/conf/icde/LiuCLWL25.html)；[作者代码](https://github.com/ZeyuLiu0706/DTJS)。

### 2.3 MT-CCL：因果任务亲和与双层对比约束

**Multitask Causal Contrastive Learning**，IEEE TNNLS 2025，Chaoyang Li et al.，DOI: 10.1109/TNNLS.2025.3555683。

- 该方法使用 task-aware causal intervention 处理特征中的混杂因素，并以 inter-task causal affinity 衡量任务关系。
- 训练目标包含 intra-task contrast 与 inter-task contrast，分别服务任务内有用特征学习和任务间干扰缓解。
- 实验包含 Multi-MNIST、NYU-v2、CityScapes、CelebA 等通用MTL设置。

**对本项目的价值**：补充“特征内容导致任务关系误判”的机制路径。即使梯度方向或任务亲和度看似合理，模型也可能依赖由共同混杂因素造成的表面关联。

**边界**：该工作没有在推荐任务中验证，且“因果”命名不意味着在 MovieLens/KuaiRand 上自动识别了真实因果特征。SAF 已经使用加权HSIC与样本重加权，因此本项目不能只换一种去相关或对比损失。需要可观测的具体域任务错误依赖作为入口。

来源：[PubMed正式索引](https://pubmed.ncbi.nlm.nih.gov/40293904/)；[TNNLS作者材料](https://hongsong-wang.github.io/TNNLS2025/)；[DBLP记录](https://dblp1.uni-trier.de/rec/journals/tnn/LiCJHL25.html)。

### 2.4 Distributional Training Data Attribution：训练随机性本身也是归因对象

**Distributional Training Data Attribution: What do Influence Functions Sample?**，NeurIPS 2025，Bruno Kacper Mlodozeniec et al.

- 传统训练数据归因常把样本影响当作单个模型参数或单次运行上的固定变化。
- d-TDA 将训练初始化与 batch 随机性纳入模型输出分布，研究训练数据改变如何影响跨运行的输出分布。
- 论文提出 distributional influence，并在视觉模型剪枝和扩散模型样本识别中展示用途。

**对本项目的价值**：提醒我们，某种样本或任务干预可能只改变特定seed轨迹，不改变平均性能，或者改变跨seed结果分布。对五种子正式实验而言，不能只报告均值而忽略配对方向、seed异质性和输出变化的分布特征。

**边界**：d-TDA并不意味着当前五seed足以稳定估计高维分布，也不适合直接作为当前主算法。它更适合作为诊断统计思想：区分平均收益、seed间离散程度与特定样本/组合导致的尾部变化。

来源：[NeurIPS 2025正式论文页](https://proceedings.nips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html)；[论文PDF](https://papers.neurips.cc/paper_files/paper/2025/file/0e8909cae8248c98279f6cd82074aa6d-Paper-Conference.pdf)。

### 2.5 Interaction-Aware Influence Functions：样本组影响不等于单样本影响之和

**Interaction-Aware Influence Functions for Group Attribution**，ICLR 2026。

- 传统 group attribution 常将组内单样本 influence 相加，忽略样本之间的互补与冗余。
- 该工作以二阶展开增加 pairwise interaction term，估计样本对联合改变 target function 的影响；报告在六组 dataset-model 配对上比一阶 influence 更接近 leave-group-out retraining，并在 Llama 指令数据选择中改善多个下游任务。

**对本项目的价值**：这是 D3“条件组合增益”更细的实例级形式。某一来源样本单独有益，并不代表它与其他样本/域任务来源共同训练时仍然有益；也可能存在单独影响很弱、组合影响很强的样本对。

**边界**：二阶 influence 对大规模推荐模型的计算成本可能较高；本项目不能一开始就做全样本对穷举。应先对少量已发现的争议样本组做验证性干预，且将其作为诊断而非直接上线调度器。

来源：[arXiv论文](https://arxiv.org/abs/2605.15675)。

### 2.6 Influence Dynamics and Stagewise Data Attribution：样本影响会随训练阶段改变

**Influence Dynamics and Stagewise Data Attribution**，ICLR 2026。

- 该工作指出静态归因忽略训练阶段；样本影响可能非单调变化，出现符号翻转或阶段性峰值。
- 作者在toy model和语言模型中分析训练阶段与token-level influence变化之间的关系。

**对本项目的价值**：与已有训练后期排序变化、EMA收益阶段差异以及 D2 的 E10/E25/E40 anchors 相联系。一个固定训练早期的样本/任务影响评分，不一定能代表中后期的有效贡献。

**边界**：其证据来自语言模型，不是推荐排序；不能直接把训练阶段变化等同于需要课程学习。项目已经有 EARLY/LATE 等阶段干预的阴性结果，因此只保留“影响信号可能阶段性变化”的诊断价值，不重新提出一般早晚切换。

来源：[ICLR 2026正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c1edb917edb6cec59bf3a0afc03150e-Abstract-Conference.html)；[OpenReview PDF](https://openreview.net/pdf?id=8epkNiuAQC)。

### 2.7 MTCCM：推荐领域的任务冲突控制，但任务定义较远

**MTCCM: A multi-task conflict control model for Web API recommendation with semantic enhancement**，Applied Soft Computing，2026。

- 该方法以语义增强模块处理非结构化 API 输入，并通过 conflict control module 选择冲突较小的辅助任务、调整梯度方向。
- 它是推荐应用，但任务是 Web API/mashup 推荐，不是典型多域CTR/CVR推荐。

**对本项目的价值**：可作为“推荐任务中的任务冲突控制”补充参考。

**边界**：其核心仍接近已有梯度冲突与辅助任务选择，问题域和指标也不同。当前不列为主候选，更不建议把它作为 PCGrad 失败后的直接替代。

来源：[Elsevier正式页面](https://www.sciencedirect.com/science/article/pii/S1568494626001973)。

---

## 3. 机制候选池与重叠审查

### 3.1 候选排序

| 排名 | 候选机制 | 新增信息 / 控制对象 | 与既有工作区别 | 当前优先级 | 关键否决条件 |
|---|---|---|---|---|---|
| M1 | 共享更新诱导的专有梯度一致性 | U 更新前后，P 对共享表示产生的梯度如何变化；专有状态反向塑造共享更新 | 不再固定 U→P 或 P→U；关注共享—专有反馈闭环 | 最高 | 若各 cell 的 P 梯度响应没有稳定结构，或协调动作不能改善排序 |
| M2 | 样本×任务条件影响调度 | 同一行样本对不同任务/域任务单元的增量影响 | 超出域配比、loss权重、样本难度或全局梯度范数 | 高 | 条件影响与普通loss/频率代理等价，或动作不改变六格排序 |
| M3 | 阶段性与组合式训练影响归因 | 样本/来源的影响符号随训练阶段变化；成组贡献存在互补/冗余 | 超出静态单样本归因、成对来源相似度、一般阶段调度 | 高（诊断） | 交互项没有稳定动作价值，或归因成本超过可获得信息 |
| M4 | 域任务条件有害特征依赖干预 | 某些特征/表示关系在一个域任务单元有益、另一个单元有害 | 超出全局HSIC、通用因果对比、总体样本重加权 | 中高 | 无法定位可干预依赖，或去除后损害真实偏好信息 |
| M5 | 语义合法的多标签保持增强 | 哪些字段变换能同时保持多个标签含义 | 超出高斯扰动、梯度方向扰动、一般条件mix | 中（条件保留） | 数据字段没有可信语义约束，或标签一致性不能保持 |
| M6 | 输出响应/敏感度条件化控制 | 同一参数变化对六个输出的响应程度及排序边界变化 | 超出梯度norm、loss幅度和参数变化率 | 中 | 输出敏感度不能预测动作增益，或仅改善校准 |
| M7 | 跨seed分布影响归因 | 样本/动作对训练输出分布而非单次模型的影响 | 超出只比较均值或单次影响分数 | 低至中（诊断） | 当前seed预算不足以估计分布，且额外运行成本过高 |
| M8 | 推荐任务冲突控制器 | 选择辅助任务并改变冲突方向 | 已有梯度投影、任务选择、任务权重工作密集 | 低 | 与 PCGrad/GradCraft/DTJS/ConsMTL 实质重叠 |

### 3.2 必须区分的概念

1. **专有头适配 vs 专有参数平滑**：前者改变共享表示变化后专有分支如何重新利用输入；后者只是对专有参数的历史轨迹做平均。此前专有参数 EMA 未产生稳定增量，不能据此否定适配反馈。
2. **样本难度 vs 样本对某任务的有益性**：高loss样本可能是噪声，也可能是关键边界样本；低loss样本可能已学会，也可能通过捷径获得低损失。DTJS的intra-task influence提供线索，但迁移到AUC必须直接定义排序目标。
3. **单样本影响 vs 组合影响**：两个样本的单独贡献不能简单相加；互动项可能代表冗余或互补。该问题不同于一般域相似度和单来源utility。
4. **阶段变化 vs 课程训练**：影响随epoch变化是经验现象；只有阶段条件信号能稳定改变动作并提高selected AUC，才构成课程或调度算法。
5. **任务冲突 vs 预测功能受损**：梯度冲突只是局部优化几何；真正要解释的是哪一类样本、哪个域任务单元的排序能力被破坏。
6. **特征去相关 vs 语义合法性**：独立性不是推荐预测的最终目标；一个统计上相关的特征可能是真实兴趣信号。必须通过任务条件干预而不是只看HSIC下降判断。

---

## 4. 优先方向的研究定义

### M1：共享更新诱导的专有梯度一致性

**核心问题**：共享参数发生变化后，六个域任务专有分支的梯度是否会出现系统性变化？这种变化是否能够解释为何某些私有更新顺序有时有效、有时无效？

**机制链**：

\[
\theta_U \xrightarrow{\text{shared update}} \theta_U'
\]

\[
g_{P,c}^{before}=\nabla_{\theta_U}L_c(\theta_U,\theta_{P,c}),\qquad
g_{P,c}^{after}=\nabla_{\theta_U}L_c(\theta_U',\theta_{P,c})
\]

比较的不是单纯梯度夹角，而是共享更新前后专有状态所诱导的共享梯度变化、对应输出响应以及后续专有适配能否恢复排序。ConsMTL说明专有参数参与塑造共享梯度；D2已表明静态更新次序没有持续收益。因此研究重点应转向**条件化反馈**，不是再比较一个固定顺序。

**潜在方法形态**：
- 用当前共享更新造成的 cell-level 梯度响应作为状态量；
- 区分“共享表示已改善，但专有头尚未适配”与“共享表示对该cell本身有害”；
- 只在有明确恢复信号时执行对应专有更新或协调项；
- 不引入新的EMA系数，不先训练复杂控制器。

**必须保留的对照**：普通联合更新、静态 U→P、反向 P→U、joint accumulation、相同总样本暴露和Adam状态；并区分额外optimizer step收益。

**停止条件**：若共享更新前后专有梯度变化没有可重复的域任务结构，或该结构与AUC恢复无关，则停止该机制，不继续扩张成复杂双层网络。

### M2：样本×任务条件影响调度

**核心问题**：同一训练行对任务0和任务1的作用是否具有稳定差异？这种差异是否在不同域中改变方向，并影响共享表示学习？

**研究变量**：行样本 \(x_i\)、域 \(d\)、任务 \(t\)、训练阶段 \(e\)、共享/专有参数状态。候选影响量必须直接连接验证排序变化或输出边界，而不能只取样本loss、梯度norm或标签频率。

**潜在方法形态**：
- 在训练内校准数据上估计样本对不同任务的局部增量影响；
- 识别任务间“互补监督”与“重复监督”；
- 只改变少量训练样本的进入顺序或共享更新参与度，保持各域任务的总体有效暴露可核对；
- 将任务特定影响作为输入，不把任务困难程度直接当成域规模或负例比例。

**与既有实验的边界**：已有Equal/Sample、q/quota、WR/PERM/ORDER、loss质量、任务视图扰动等实验意味着一般采样/权重策略不再新颖。DTJS也已提出任务内数据影响与任务间冲突联合调度。只有当新方案直接估计推荐排序目标下的样本×任务条件贡献，并形成不同于上述已测动作的策略时，才值得继续。

### M3：阶段性与组合式训练影响归因

**核心问题**：某个样本/来源对目标排序的影响是否会随训练阶段变号？若两个样本/来源共同训练，其组合收益是否不同于各自独立影响之和？

**潜在研究对象**：

\[
I_{ij}^{interaction}
=U(\{i,j\})-U(\{i\})-U(\{j\})+U(\varnothing)
\]

其中 \(U\) 必须定义为固定训练预算、相同起点和指定验证目标下的增量，不把梯度内积直接叫作utility。

**研究价值**：可以解释为什么静态来源相似度、单样本影响或早期诊断不能稳定预测最终AUC。若交互项显著且能改变正确动作，才值得学习稀疏组合选择器。

**成本控制**：先用小规模、预注册的争议样本组或少量域任务来源组合进行验证，不做全量 \(O(N^2)\) 样本对归因。stagewise影响作为诊断，不直接触发早晚课程训练。

### M4：域任务条件有害特征依赖干预

**核心问题**：是否存在某个特征或表示关系在任务A中被利用，但在任务B中形成预测捷径或排序退化？这种关系是否只在某些域、人群或训练阶段成立？

**机制依据**：SAF的weighted HSIC、MT-CCL的task-aware intervention与ORCA的错误依赖校正均提供先例。

**区别要求**：
- 不能仅最小化所有共享表示维度的相关性；
- 不能将任务错误较大直接等同于有害依赖；
- 需要先定位条件错误模式，再对具体路径或残差分量进行干预；
- 评价必须同时看目标任务格和未受干预格。

**停止条件**：若数据中没有可解释且可干预的特征依赖，或干预只降低统计依赖而不改善排序，则不开发大规模因果模块。

### M5：语义合法的多标签保持增强

**核心问题**：对一个输入样本做局部变换时，哪些变换能同时保持当前存在的多个任务标签含义？

该方向不是继续搜索高斯方差、方向扰动幅度或mix比例。需要在原始字段、物品/用户属性或序列片段上定义合法变化集合，并检查不同域任务的标签保持约束是否一致。

**决策**：只有当MovieLens或KuaiRand现有字段存在可信的可变/不可变语义时，才保留为主方向。否则不应强行构造反事实或引入生成模型。

---

## 5. 当前不建议启动的方向

| 方向 | 当前决定 | 原因 |
|---|---|---|
| 继续搜索自适应 EMA / rho | STOP | 多轮反馈指标与动作收益脱钩；固定EMA收益已有结论；Sample下Adam history未达到practical gate |
| 一般梯度投影 / PCGrad变体 | STOP | 已有PCGrad负向结果；ConsMTL的新价值在专有参数参与共享梯度塑造，不是再换投影公式 |
| 固定 U→P / P→U / P范围选择 | 当前版本 NO-GO | D2 S1独立V_B未确认，结果量级远低于实用门槛 |
| 仅按loss、梯度norm或域频率调样本 | STOP | 与既有权重、采样和梯度统计实验重叠 |
| 一般easy-to-hard课程训练 | 低优先 | 已有阶段与时间调度阴性结果；DTJS的价值是样本影响与任务冲突联合估计，而不是简单排序loss |
| 一般HSIC/去相关/因果对比 | 不单独启动 | SAF与MT-CCL已有先例；必须有特定域任务错误依赖 |
| 再做一套ranking loss替换BCE | STOP | NQR及EARLY10已覆盖相关方向；需要新信息源，不是再换损失形式 |
| 直接扩成大规模Transformer/Token骨干 | 暂缓 | MDL/MTFM提供工业异构输入路线，但当前3域×2任务数据未显示需要重建骨干 |

---

## 6. 研究路线选择与下一步决策

### 6.1 推荐顺序

**第一步：优先研究 M1。**  
原因是它紧接当前 M3oE 的真实参数结构，能利用 D2 的模型/状态审计和已有共享—专有参数分组，不需要先引入新数据或重建网络。D2 NO-GO 是对固定动作的有效反馈，正好促使研究从“哪一个静态顺序更好”转向“何时、何种状态下专有更新能修复共享变化”。

**第二步：将 M2/M3 作为诊断工具，而不是立即作为算法。**  
如果 M1 显示专有梯度响应差异主要由少数训练样本或标签组合驱动，则进一步进入样本×任务条件影响；如果响应随训练阶段变化或出现样本组交互，则采用 stagewise / group attribution。反之，不必为追求复杂性强行加入样本归因。

**第三步：M4/M5 作为内容层分支。**  
当证据显示问题来自共享表示中的具体错误依赖或数据增强破坏标签语义时，再进入特征干预或语义保持增强。不要把这两个分支与M1同时叠加，否则难以识别新增信息来源。

### 6.2 进入算法设计前必须明确的判断

1. 新信号究竟提供了什么旧方法没有的信息？
2. 信号能否改变一个可执行动作，而不仅是更好地解释历史结果？
3. 动作是否与已有 U→P/P→U、P范围选择、梯度投影、样本配额和任务权重实质不同？
4. 新增动作的收益是否在同预算、同样本暴露和匹配Adam状态下成立？
5. 是否能解释六格AUC中的获益格与受损格，而不是只报告Macro平均？
6. 是否将AUC与LogLoss分开报告，避免把校准改善当成排序改善？

### 6.3 正式评价边界

- 正式算法性能比较仍使用 seeds 2022–2026；
- Validation 六格 Macro AUC 选点，Test 在方案冻结后评价；
- 主要实用AUC增益门槛为 +0.003，单格平均下降保护线为 −0.001；
- LogLoss单独报告，不替代AUC主指标；
- M3oE、HyperGate、AutoIFS等正式发表模型与内部训练控制分开；
- 诊断性短程结果可以用于定义机制和筛选动作，但不能代替五种子完整性能结论；
- 对当前 D2 的 S1 NO-GO，应保留为正式阴性结果，不重跑同一静态动作以寻找偶然正值。

---

## 7. 最终研究决策

### 接受

- 当前 main 的 Equal / Sample 区分与 objective-contingent history 结论；
- D2 S1 的静态共享—专有协调 NO-GO；
- ConsMTL 所提示的专有参数对共享梯度形成具有反馈作用这一机制假设；
- DTJS 与近期数据归因工作所提示的样本×任务条件影响和组合影响值得作为诊断对象；
- 语义保持增强只有在真实数据字段提供合法操作时才有研究价值。

### 保留为首要研究方向

**共享更新诱导的专有梯度一致性与条件适配。**

它既能与既有共享参数收益、专有参数操作弱收益、D2静态更新顺序失败形成连贯逻辑，也能从“参数历史平均”转向“共享—专有反馈过程”。但当前只接受其作为值得进一步检验的机制方向，不宣称新算法已经成立。

### 暂缓或停止

- 自适应EMA系数搜索；
- 一般梯度投影；
- 固定私有参数范围与固定更新顺序；
- 一般样本重加权/采样配额；
- 一般课程训练；
- 无具体特征依赖证据的去相关/因果对比；
- 没有语义合法性定义的条件mix或生成增强。

**本阶段交付到此为止：研究报告与方向决策。下一阶段如需进入Codex规划，应以本报告的M1机制为主线，先审查 D2 原始工件、当前 main 最新上下文与模型计算图，再单独形成实验规划；本报告本身不授权实现或训练。**

---

## 参考文献与新增来源

1. Qin, X., Wang, X., Yan, J. *Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.html
2. Liu, Z. et al. *Adaptive Data and Task Joint Scheduling for Multi-Task Learning*. ICDE 2025. https://ieeexplore.ieee.org/abstract/document/11113158
3. Li, C. et al. *Multitask Causal Contrastive Learning*. IEEE TNNLS, 2025. DOI: 10.1109/TNNLS.2025.3555683. https://pubmed.ncbi.nlm.nih.gov/40293904/
4. Mlodozeniec, B. K. et al. *Distributional Training Data Attribution: What do Influence Functions Sample?* NeurIPS 2025. https://proceedings.nips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html
5. Heo, J. et al. *Interaction-Aware Influence Functions for Group Attribution*. ICLR 2026. https://arxiv.org/abs/2605.15675
6. Lee, J. H. et al. *Influence Dynamics and Stagewise Data Attribution*. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c1edb917edb6cec59bf3a0afc03150e-Abstract-Conference.html
7. Chen, K. et al. *MTCCM: A multi-task conflict control model for Web API recommendation with semantic enhancement*. Applied Soft Computing, 2026. https://www.sciencedirect.com/science/article/pii/S1568494626001973
8. Fei, K. et al. *Stable and Adaptive Fusion for Multi-domain Multi-task Recommendation*. AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/38491
9. Yang, C. et al. *Automated Information Flow Selection for Multi-scenario Multi-task Recommendation*. WSDM 2026. https://doi.org/10.1145/3773966.3777992
10. Liu, C. et al. *Causal Domain Clustering for Multi-Domain Recommendation*. SIGIR 2025. https://arxiv.org/html/2507.06877v1
11. ORCA: *Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling*. CIKM 2025. https://arxiv.org/html/2508.16573v1
12. Taesar: *Generative Data Transformation: From Mixed to Unified Data*. WWW 2026. https://arxiv.org/html/2602.22743v1

## 仓库来源与审计入口

- 最新 main 上下文：[EXP_AI_CONTEXT_V2.md](https://github.com/Caelestis1903/My-Research-Project/blob/f66bf17cba2980618e6f2edd1c7381b73c35cac5/experiments/reports/EXP_AI_CONTEXT_V2.md)
- D2 设计：[DESIGN_SPECIFICATION.md](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md)
- D2设计勘误：[DESIGN_ERRATUM.md](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/research_reports/mdmt_shared_private_coordination_20261123/DESIGN_ERRATUM.md)
- D2 S1运行状态：[S1_RUN.json](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json)
- D2 scope裁决：[SCOPE_DECISION.json](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)
- D2参数分组：[PARAMETER_MANIFEST.json](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/artifacts/mdmt_shared_private_coordination_20261123/S0/PARAMETER_MANIFEST.json)
- D2数据身份：[DATA_MANIFEST.json](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-23_v2/artifacts/mdmt_shared_private_coordination_20261123/S0/DATA_MANIFEST.json)
