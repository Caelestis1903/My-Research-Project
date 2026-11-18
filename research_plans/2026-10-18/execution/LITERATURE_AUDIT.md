# 直接竞争文献核查与排重

本轮重新访问原文，阅读下列直接影响判定的方法/公式；未宣称审计全部证明。
截至执行日 2026-10-18 做了一次有界的近期检索，没有将检索不到新结果
解释为新颖性证明。以下“差别/不足”是本轮推断，不能冒充原论文承诺。

| 方向 | 实际核查的一手来源 | 原方法提供什么 | 本轮必须超出的门槛及判定 |
|---|---|---|---|
| P1 | [SAF，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38491/42453)，Proposition 1、Eq9/14 | 两层 RFF-HSIC 权重和适应融合 | 有限经验矩为零与总体独立须区分；本轮只证离散支持边界，没有复现 SAF 或证明其实证失败 |
| P1 | [DVD，KDD 2020](https://pengcui.thumedialab.com/papers/Stable_DVD.pdf)，变量组与环境条件 | 选择性去相关，保留稳定组内结构 | 简单保留相关结构已是强对照；支持 LP 本身不是新稳定算法 |
| P2 | [ExtraDebias，SIGIR 2026](https://haoxuanli-pku.github.io/papers/SIGIR%2026%20-%20Debiased%20Recommendation%20Beyond%20the%20Positive%20Propensity%20Assumption.pdf) | 零 propensity 下用 Lipschitz 信息构造区间和最坏风险 | 本轮未提供新的跨域桥接关系；嵌套标签缩界被同信息 LP 完全复制 |
| P2 | [WCP-DR，KDD 2026](https://haoxuanli-pku.github.io/papers/KDD%2026%20-%20Causality-Based%20Conformal%20Imputation%20Correction%20with%20Non-Random%20Missing%20Labels.pdf)，§4.3 | 条件标签分布相同、协变量偏移下加权 conformal 修正 | 本轮 MNAR 反例不满足该条件，不能当成 WCP-DR 失败；也不把边际覆盖当 cell 覆盖 |
| P2 | [HiDe-ESMM，SIGIR 2026](https://haoxuanli-pku.github.io/papers/SIGIR%2026%20-%20Hierarchical%20Denoising%20Entire%20Space%20Multi-Task%20Model%20for%20Post-Click%20Conversion%20Rate%20Prediction%20with%20Noisy%20Labels.pdf) | 行为链测量与噪声辨识条件 | 当前 ML 人为阈值、KR 完整标签不能替代自然噪声通道和 clean audit |
| P1/P2 | [Feasible Fusion，ICML 2026](https://haoxuanli-pku.github.io/papers/ICML%2026%20-%20Feasible%20Fusion-Constrained%20Joint%20Estimation%20under%20Structural%20Non-Overlap.pdf)，问题定义与矩约束 | 结构不重叠下观测数据和随机实验融合 | 普通“约束＋区间＋随机审计”已有先例；本轮没有超出一般信息融合的识别定理 |
| P2 | [AEM²TL 作者机构页面](https://researchers.mq.edu.au/en/publications/an-adaptive-entire-space-multi-scenario-multi-task-transfer-learn/) | 整空间多场景多任务结构 | 仅核查作者机构概览；全文仍有缺口，不根据摘要推断具体可识别定理 |
| P3 | [DSMTL 原文 v2](https://arxiv.org/html/2111.10601v2)，Definitions 3/4、IL 与 JL | 私有回退、经验个体/平均安全及带残差的总体界 | 本轮不是首次收益保护；受限线性头证书不能推出神经模型总体非劣 |
| P3/P4 | [AutoIFS，WSDM 2026](https://arxiv.org/html/2512.13396v1)，§4 | 四类信息流、低秩参数化、任务相关选择/剪除 | 原模型能表示交互；不能把它误写成纯加性头。本轮新修复已等价普通额外 cell 头，故不启动不必要的 AutoIFS 重训 |
| P3/P4 | [M3oE，SIGIR 2024](https://arxiv.org/html/2404.18465v3)，§2 | 三类专家、双层融合、DT 输出与 AutoML | 本轮保留 DT 输出能力的冻结表征；固定融合是研究条件，原论文 AutoML 不属于本轮实验 |
| P4 | [MultiBalance](https://arxiv.org/html/2411.11871v1)，方法与计算考虑；[NTKMTL](https://arxiv.org/html/2510.18258v1)，方法 | 梯度开销、任务关系与 Jacobian 动态已有直接邻近工作 | ANOVA 分解/平铺六 cell 不是贡献；必须在实测决策信息成本上胜过普通展开 |
| P5 | [D3，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/download/28699/29354)，Eq12/13/18 | 双 softmax gate 的乘积融合与 specificity loss weight | 交换 gate 保留出版前向但改变权重；本轮给出同函数动作的风险反号构造，不称官方实现 bug 或总体性能失败 |
| P4/P5 | [GRAPE](https://arxiv.org/html/2505.20380v1)；[VAO v2](https://arxiv.org/html/2510.07760v2)，方法部分 | 多目标验证效用、源混合和目标相关权重 | 普通验证梯度替换 gate 已被覆盖。本轮构造也可由普通人口权重解决，不构成独立新方法 |
| P6 | [EMAIS/EMS](https://arxiv.org/html/2501.13296v2)，§5–8 | EMS trace 比、EMA 统计、logit 梯度代理及训练过程 | 本轮只测静态 logit 代理，未复现其完整在线更新或学习率规则，不能报告“击败 EMAIS” |
| P6 | [向量 MIS/OMIS](https://arxiv.org/html/2407.15525v1)，proposal、估计器与时间比较 | 向量梯度的重要性混合及估计成本 | 本轮域 proposal 的二次型最优分布属于常规广义 IS；静态混合不是完整在线 OMIS |

实际访问 SAF 论文给出的 [extended/code 地址](https://github.com/q1179897215/SAF)
得到 404；官方 D3 训练实现和忠实训练 checkpoint 仍未取得。因此，P1/P5
只保留明确的数学/信息边界。没有把仓库 DBO-inspired V1 冒充 D3。

本轮值得留下的内容是可复核的窄反例、同函数类等价证明、真实固定点
读数和停止依据；没有证据支持将它们包装为已优于上述方法的顶会算法。
