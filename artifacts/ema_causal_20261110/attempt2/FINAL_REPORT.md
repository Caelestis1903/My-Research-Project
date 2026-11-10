# 固定 EMA 因果与功能归因研究报告

主方向：C。

## Q1

晚期固定EMA的主要排序收益来自共享参数修正，EMBED及近期历史修正是稳定来源；recent与old在EMBED中存在正交互。ML的old单独作用可为负，但在联合状态中有益；KR的old单独作用更不稳定。四条件晚期全V六格均值均提高，私有修正并非稳定收益源。

证据强度：STRONG。

## Q2

同一X=近期修正的F−X→F四态实验表明，h及完整融合向量的上游变化重现主要AUC收益，改变真实正负pair的margin并同时产生纠正和破坏。task融合项解释部分纠正，Gaussian留出侧精细效应较弱。历史同权预测组合在V_mech上重现主要纠正且AUC高于参数F、LL更低。

证据强度：STRONG。

## Q3

ML的近期online更新中，固定训练目标下降与验证排序受损并存，近期EMA修正恢复这些排序；KR同向均值较弱。两Clean的EMBED同状态、同步幅对照中，保留vhat并改用当前numerator有小幅AUC收益，抹去坐标预调节明显损害AUC。一阶历史是可检验的局部来源，未解释全部EMA收益。

证据强度：MEDIUM。

## Q4

本轮EMA菜单没有训练可见规则对F及简单静态动作的稳定额外价值。所有条件rho预测器退化为统一all/fast；KR机制预测器C_design退化为F。ML有限菜单经验空间较小，KR存在oracle与部分跨验证侧空间但训练信号未取得该价值。旧版本须分别解释动作覆盖、时域/导数语义、信号迁移和实际价值。

证据强度：STRONG。

## Q5

主方向选择C：围绕Clean/EMBED的一阶历史开展最小短程反馈判别，保留v预调节、步幅对照和固定F内部对照。选择依据是实际同状态方向替换的正面证据，而非EMA菜单阴性。条件备选为D：若改善仅存在于单步online读出、长期消失或被F吸收，则停止扩展EMA/优化器控制器，转向目标与概率质量。

证据强度：MEDIUM。

## 五 seed 独立选点 Test

| 数据 | 目标 | 指标F−O | 均值 | SD | 95% CI |
|---|---|---|---:|---:|---|
| movielens | clean | macro_auc | 0.0038488 | 0.0003744 | [0.0033840, 0.0043137] |
| movielens | clean | macro_logloss | 0.0012486 | 0.0056480 | [-0.0057643, 0.0082616] |
| kuairand | clean | macro_auc | 0.0046164 | 0.0030488 | [0.0008308, 0.0084020] |
| kuairand | clean | macro_logloss | -0.0012253 | 0.0017754 | [-0.0034297, 0.0009791] |
| movielens | gaussian_d0 | macro_auc | 0.0038568 | 0.0006452 | [0.0030556, 0.0046580] |
| movielens | gaussian_d0 | macro_logloss | 0.0053343 | 0.0029248 | [0.0017027, 0.0089659] |
| kuairand | gaussian_d0 | macro_auc | 0.0077669 | 0.0042575 | [0.0024805, 0.0130533] |
| kuairand | gaussian_d0 | macro_logloss | 0.0003277 | 0.0028556 | [-0.0032180, 0.0038734] |

完整逐seed、六域任务指标及状态身份见ENDPOINTS.csv；同刻和独立选点分别保留。每条件n=5；各epoch/cell不扩充样本量。

本轮20条轨迹、1000epoch、34,500次真实Adam；源训练配置、自然人口和数据身份见RESOLVED_PROTOCOL.json与DATA_MANIFEST.json。参数历史仅保留于RAM。

功能/参数/时间/pair/桥接原始表见blocks各条件ARTIFACT_INDEX.json；统计和来源身份见同目录收据。
