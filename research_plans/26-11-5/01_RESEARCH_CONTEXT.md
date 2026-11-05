# 01 当前证据、源码语义与研究问题

## 1. 固定读取身份

main 研究上下文基准：`1aa8b411f607c07827b8304e6611aa84d16714f7`。已读取 `EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md` 及 `AGENT.md` 的当前规则；以 11/04 handoff 和 11/03 五 seed 最高优先级规则为准。

执行源码参照：`c8675ae2923c15b6965884b140b5b646690cd95e`。main 主要汇总结论，最新训练包仍在源提交；不能假装 main 已合并全部实现。具体源码与报告清单见 [10_SOURCE_INDEX.json](10_SOURCE_INDEX.json)。以下数值为归档结果，不是本轮重训或服务器 CSV 复算。

## 2. 当前结论 → 证据 → 边界 → 决策

| 结论 | 已有证据 | 边界与本轮决定 |
|---|---|---|
| 参数 EMA 已有可靠排序价值 | GAM01 五 seed，ML online→EMA AUC +0.004399，KR +0.004616；其他 ML 配置也正向 | 不重做“EMA 是否有效”确认；AUC 与 LL 分开 |
| Gaussian+EMA 在 ML 超过当前最强发表模型 | G-D0 Macro AUC 0.779357，M3 fixed1 0.775190，差约 +0.004167 | 不把全部收益归给 Gaussian；无增强固定 EMA 为 0.779039 |
| KR Gaussian 有增量但未超过最强发表模型 | G-D0 0.679180，固定 EMA 0.676523，HyperGate 0.682113 | 相对固定 EMA +0.002657；d1 两任务代价不能被 Macro 掩盖 |
| 扩大覆盖不保证更好 | KR G-D02 0.674604，G-D0 0.679180；共同 E50 在线也有代价 | 不继续默认扩到双域；本轮冻结 G-D0 |
| 复杂组件与 EMA 总收益必须拆开 | 11/03 五 seed 补全中 C/H/P01 存在完整方案收益，ML 新组件增量仍可为负；KR P01 有增量及单格代价 | 不因组件弱而抹去总收益，也不把 EMA 收益归给新组件 |
| SAD 所测长差分记忆没有建立独立收益 | GAM01 两数据五 seed A3 未胜普通增强 Adam | B 不修改真实动量，不重做 SAD |
| GATS 动态时间尺度、EMA02V2 视图等旧停止门不等于全方向失败 | main 指向 GATS 与 EMA02V2；后者仅 2022/2024 两 seed | 继承局部观察，不按旧两 seed 门否决新路线 |
| MASK、PAIR/ECCM、历史投影存在不同协议和完成度 | main 最新汇总区分旧 MASK/早期条件增强/PAIR 两 seed及后续 P01 五 seed | 不混合绝对排名；不重复全部旧候选 |

证据链接：
- [main 11/04 handoff](https://github.com/Caelestis1903/My-Research-Project/blob/1aa8b411f607c07827b8304e6611aa84d16714f7/EXP_AI_CONTEXT.md)
- [指标台账](https://github.com/Caelestis1903/My-Research-Project/blob/1aa8b411f607c07827b8304e6611aa84d16714f7/MDMT_ALGORITHM_METRICS_LEDGER.md)
- [Gaussian 覆盖报告](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)
- [G-D0 双数据配置与六格](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/G_D0_TWO_DATASETS.md)
- [GAM01 原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md)
- [EMA02V2 原报告](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md)

## 3. 当前固定训练协议

| 项目 | MovieLens | KuaiRand |
|---|---|---|
| train/validation/test | 800167/100021/100021 | 2006132/250767/250767 |
| train d0/d1/d2 | 155949/318599/325619 | 172782/1743898/89452 |
| 域含义 | age {1,18}/{25}/{35,45,50,56} | tab 0/1/4；d1 最大 |
| 任务 | rating>3 / rating>4 | is_click / long_view |
| 参数量 | 773470 | 1614254 |
| 每 epoch 步数 | 20 | 49 |
| Gaussian κ / λ | 0.05 / 0.5 | 0.1 / 0.5 |
| 目标域 | d0 | d0 |
| EMA ρ0 | 0.9828205985452511 | 0.995 |
| 半衰期 H0（步，公式计算） | 40 | 138.282572986 |
| coupled weight decay | 1e-4 | 1e-5 |

共同：fixed1、3域2任务、4 shared experts、embedding16、FCN [512,256,256,64]，四融合系数为1。Adam lr=.01，betas=(.9,.999)，eps=1e-8，foreach/fused/amsgrad=False，FP32、AMP/TF32=False。batch40960，50 epoch、不早停。FULLTRAIN_NATURAL80_10_10；原行号分割、train-only 编码缩放；KR 先全局分割再筛 tab。源 csv 路径见 manifest；执行时按原 checksum 校验，不把路径存在当作数据身份通过。

## 4. Gaussian 与梯度规范

可扰动字段为非路由 sparse embedding，ML6/KR33，每字段 E=16。噪声位置：embedding 输出之后，STAR 和 skip 之前。ID、domain embedding、dense 字段、标签不改。

\[
\tilde h_{if}=h_{if}+\kappa\,\operatorname{sg}\sqrt{E^{-1}\|h_{if}\|^2}\,\xi_{if},\quad\xi\sim N(0,I_E).
\]

对当前真实 batch 大小 N（避免与路线 B 混淆），目标域集合 S：
\[
L=L_0+\sum_t L_{a,t},\quad L_0=\frac{\sum_{i,t}\ell^0_{it}}{2N},\quad L_{a,t}=\frac{\lambda}{2N}\sum_{i\in S}(\ell^{aug}_{it}-\ell^0_{it}).
\]

这是监督替换，不是额外扩大某域/任务权重；尾批也用实际 2N。真实损失采用原 probability BCE。需要 logit 的只读探针直接读取 tower 原始输出，不从饱和 sigmoid 反推 logit。

## 5. EMA 与实际路径

EMA 从 θ0 复制、每次 Adam 后更新一次，参数全部参与；buffers 从在线模型复制。旧 Gaussian 的 EMA 不反馈训练。在线与影子独立按 validation 六格 Macro AUC 最早严格最大值选点。推理无噪声。

`reference.py` 中 gate 索引为 j=t*3+d；gate 输入 `emb.detach()`，gate 自身仍学习。fixed1 下域专家只在对应域监督中有效，任务专家只在对应任务中有效；共享专家、STAR shared、skip、普通特征 embedding 共同服务多个域任务。`MLP_N.domain_specific` 只是通用内部命名，不是参数归属依据。

A 必须进一步按坐标区分 domain-indicator embedding 的三个实际 lookup 行；普通特征表按架构归共享，不按某次观测中 ID 出现在哪个域来临时分组。Adam 历史状态和 coupled L2 仍能让“无当前数据梯度”的参数发生位移，不能把监督路径与实际位移等同。

## 6. 真正待判断的关系

A：路径差异是否提供比全局 EMA 更好的历史时间尺度。
B：实际增强响应方向是否比统一系数、合并响应、无信息同秩方向更有用。
C：历史功能漂移是否能指导扰动协方差，在同等能量下提高收益。

这些是设计假设，不是已证实的机制。Gaussian 的零均值输入噪声不意味着增强梯度均值为零；参数 EMA 不会自动抹去持续目标偏移。小域不是困难域，本轮不对该结论提出假设性检验。
