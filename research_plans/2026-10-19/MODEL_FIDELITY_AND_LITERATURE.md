# 2026-10-19 模型定义、原文复核与可复现性边界

本清单供开始实施的科研 Agent 使用。**不能因模型同名就推定实现忠实**，不能把第三方实现宣称官方代码。只检查可能实际改变共享/调制机制、训练目标、归一化、优化器和评价人口的关键差异。

## 1. M3oE：三种互不可直接替代的协议

来源：
- M3oE / SIGIR 2024： https://arxiv.org/abs/2404.18465
- 作者实验代码： https://github.com/Applied-Machine-Learning-Lab/M3oE
- 本地源模型： m3oe/MDMTRec/models/multi_domain/model.py
- N19 训练： https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-n19-2026-10-18/experiments/n19_train.py

必须审计以下真实执行口径：
- **original 风格**：本仓库 N19 中保留可学习融合、按六组 Equal 求和并且通过额外架构优化器对重叠参数执行更新；该名称**不自动代表与论文每处超参数/训练细节严格一致**。建立源代码对照并注明忠实/修改项。
- **original_sample**：保留可学习融合，但统一为单优化器的 Sample 目标，以隔离原始风格目标/优化协议。
- **fixed**：N19 既有：融合为 sigmoid(1) 常数，单 Adam 或 AdamW，Sample 目标；此条件下 KR+Adam 路径失活已观测到。

诊断对象是 M3oE STAR-style 调制路径中的 W_shared、W_slot,d 的**有效乘积**和下游函数，而不是 M3oE 全部共享专家。原模型还有 skip_conn、star_mlp、shared experts、domain experts、task experts、gate 与 tower；移除某调制权重时必须明确这些部分仍然保留。

注意：gate(emb.detach()) 仅阻断部分梯度路径，不是完全前向失活；理论/机制分析要区分训练梯度与完整预测函数导数。除确实会改变该实验结论，否则不应为此大规模修订既有实现。

## 2. STAR：最直接的独立结构检查

原论文：**One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction**，CIKM 2021，https://arxiv.org/abs/2101.11427  
可审查的第三方 TensorFlow 实现（非作者官方认证）：https://github.com/QunBB/RecSys/blob/main/recsys/multidomain/star.py

必留组件：
- Star-Topology FCN：domain-specific 与共享矩阵的乘性组合。
- Partitioned Normalization（PN）：领域归一化差异。
- Auxiliary Network：对域相关信息进行单独处理的辅助预测分支。

不能只拿 M3oE 的单层 STAR+Skip 替代原始 STAR 后声称“STAR 已复现”，不能把 STAR 的 Auxiliary Network 直接等同 M3oE 的 skip_conn。首轮按论文是**多域单任务**，分别训练 task0/task1。若进一步做两任务共享 STAR 扩展，必须记为 STAR-MDMT-extended，人为改造需要自己的架构说明与等成本对照。

## 3. PEPNet：门控中性化不等于移除权重因子

论文：**PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information**，KDD 2023，https://arxiv.org/abs/2302.01115  
公开实现结构参考（阿里 TorchEasyRec 文档，不应冒充与论文训练管线完全一致）：https://github.com/alibaba/TorchEasyRec/blob/master/docs/source/models/pepnet.md

保留 EPNet（Embedding Personalized Network）、PPNet（Parameter Personalized Network）和各层 Gate NU 的先验输入。Gate 用于对 embedding/隐藏层的激活进行逐元素调节，机制不同于 M3oE 的 shared×slot 权重组合。

核心干预：让某一 gate 的乘性系数成为**其计算图的中性单位**，通常为 1，保留其余表征；分开测 EP/PP 贡献，并从头训练中性 gate 版本。必要时保留原任务专用 towers/参数，而非把模型简化成普通 MLP。不要用 gate 参数范数大小作统一失活指标。

## 4. HAMUR：hyper-adapter 的输出贡献与计算图

论文：**HAMUR: Hyper Adapter for Multi-Domain Recommendation**，CIKM 2023，https://arxiv.org/abs/2309.06217  
论文作者公开代码：https://github.com/Applied-Machine-Learning-Lab/HAMUR

其关键是 domain-shared hyper-network 动态生成 domain adapter 参数，并且 adapter 以适配网络主干。官方仓库包括 MLP/Wide&Deep/DCN 等 backbone。优先选**MLP+HAMUR 对照 MLP**，保持原文 adapter 与超网络行为，并记录选择该 backbone 的理由。

诊断应分：
1. hyper-network 产生的 adapter 参数的有效值/变化；
2. adapter 引起的特征/预测变化（adapter-off 的中性化）；
3. 从头训练移除 adapter 的结果。

HAMUR 原有任务结构和 N19 两任务不同：两任务独立的多域版本符合原任务定义；人为联合头版本须明确标注扩展，不得把未经过作者复现的头部更改当成原论文。

## 5. MMoE 与 PLE：非同构共享对照

本仓库已有基础模型的经过语义初始化审计的实现：experiments/stage_S1_models.py。模型所称 shared experts、expert gates 与 PLE progressive extraction 的共享/专用专家，需要按各自计算图观测实际输出或使用率，而不是套用 STAR 的 weight product 范数。

核心问题是更一般的 **effective use of shared module**。任何“关闭 shared expert”的干预必须标明是否让 gate 对剩余专家重归一化，并配合完整从头训练的无共享/等成本私有基线；若任务结构不允许合法同构对照，报告该限制。

## 6. 原理上可能已被现有工作覆盖的部分

- AdamW 与耦合式 L2 的区别：**Decoupled Weight Decay Regularization**，https://arxiv.org/abs/1711.05101；如果实验只是复现这种差别，创新性不足。
- STAR 已明确提出 shared center 和 domain-specific 参数的乘性结构，不能以“存在 STAR 乘法”作为新想法。
- PEPNet 已使用门控动态调整 embedding / 深层网络表示；HAMUR 已做共享 hypernetwork 与域适配。共享路径较弱或模型可选择少用某个模块，可能是正常冗余优化而不是独立病害。
- 有关多任务梯度竞争、Skip/残差旁路竞争、专家失活和乘性分解尺度不识别的理论需要专门排重；未核查论文不得断言先例缺失。

## 7. 模型验收不应成为防御性测试膨胀

每个导入/移植模型只做以下**结论相关**的有限检查：
- 数据域与任务路由是否对应真实标签和模型定义；
- 目标调制路径与中性化定义是否正确，干预前预测能复现；
- 目标梯度能进入被诊断分支，优化器组无错误重叠；
- 参数数/FLOP 及 checkpoint 实际可训练性是否真实；
- 样本/seed/选点一致、测试集冻结。

不得为无关代码风格、极端输入异常、通用软件工程可靠性不断新增测试；出现会影响科研结论的 Critical 问题应修复后只重跑受影响实验。对外引用时必须明确：官方来源/第三方参考/项目移植，原生任务 vs 人为 MDMT 扩展，原论文结论 vs 本轮待证假设。

## 8. 新颖性判断的硬边界

**跨模型现象本身不等于新理论**，**调制路径恢复不等于泛化收益**，**恢复范数不等于修复失活**。只有在：
1. 复现跨架构或识别真实适用边界；
2. 关键变量控制下发现之前没有清晰解释的触发机制；
3. 证明有害性/有用性或提出超出通用优化器切换的可重复可训练修复；
4. 与先行工作相比具有独立理论/实证价值

时，才能把研究方向升级为方法论文候选，否则完整报告负结果和明确的新问题，不夸大。
