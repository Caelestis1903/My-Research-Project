# 01 — 研究上下文与已知证据

## 1. 输入报告与文件身份

第二阶段提示词指定的第一阶段路径：
research_reports/26-11-23_multi_mechanism_discovery/00_RESEARCH_REPORT.md

在当前 main 未找到该路径及其三个支持文件。实际第一阶段交付为：
research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md

本方案以实际存在的报告为准，并保留路径差异，不声称已读取缺失文件。

## 2. 第一阶段结论

本轮种子预注册为 2022–2026：这是第二阶段提示词明确指定的五种子集合，也与 D2 S1 的实际 seed 范围一致。该选择作为本计划的明确预注册，不沿用 main 上下文中未另行预注册时的默认 2022–2025 规则。

第一阶段保留的主线是共享更新—专有梯度响应—条件更新动作。新增研究给出的关键机制来源：
- ConsMTL (CVPR 2025)：专有参数参与塑造共享梯度，提示共享—专有反馈不只是更新顺序问题。
- DTJS (ICDE 2025)：同一样本对不同任务可能有不同影响；不能简化为样本难度。
- MT-CCL (TNNLS 2025)：任务条件特征依赖可能导致任务关系误判；需具体可干预证据。
- NeurIPS 2025 distributional attribution：训练随机性应纳入结果解释。
- ICLR 2026 interaction-aware attribution：组贡献可能非加性。
- ICLR 2026 stagewise attribution：影响可能随训练阶段变化。
- MTCCM (Applied Soft Computing 2026)：推荐中的任务冲突控制参考，但与当前 CTR/CVR MDMT 距离较远。

第一阶段候选排序 M1–M8 不等于本轮全部纳入。实际纳入 M1/M2；M3 有条件；M4/M5 审计；M6/M7 作为辅助；M8 排除。

## 3. main 最新上下文的关键事实

依据 experiments/reports/EXP_AI_CONTEXT_V2.md：
- Equal 与 Sample 是不同有效目标。
- Sample 在已测自然比例与协议下是更稳健参照；Sample 下普遍 residual NT / Safe-MTL 主线未达到稳定实用门槛。
- Equal 下 KuaiRand tab1 存在条件性大幅退化与 shared-access rescue。
- S5 将 Equal 下 d1 rescue 主要定位到 persistent Adam history；S6 切换 Sample 后两个已测最大域 target 均未达到 +0.003 practical rescue gate，HT/DH Phase B 未运行。
- raw-gradient norm matching 与 applied-step norm matching 均未复现 Equal 下 rescue。
- 小域不是困难域，size 不能独立解释全部现象。
- 固定 EMA 的收益已被接受，不再重复证明。

## 4. D2 最新结果及本计划边界

D2 分支 26-11-23_v2 的 S1 完成：
- 2 datasets × 5 seeds；
- 10 条 baseline trajectories；
- 30 个 anchors（E10/E25/E40）；
- 570 个 forks；
- Test evaluations = 0；
- SCOPE_DECISION = COMPLETE_NO_GO_S1；
- V_A 仅在 KuaiRand 对 P_AXIS 给出初筛资格；V_B 独立确认未通过；
- confirmed_datasets 为空；原因 INDEPENDENT_V_B_DID_NOT_CONFIRM。

V_B 的 P→U mean Macro AUC：MovieLens +1.13e−5，KuaiRand +5.27e−5；J_ACC 分别 −1.15e−6、−8.20e−5。该幅度远低于 +0.003 实用算法门槛。

因此当前静态 P 范围/更新次序不晋级。M1 与 D2 的实质差异是：M1 通过二阶 consistency objective 优化 P，使每个 cell 对 U 的梯度趋近自然联合共享梯度；不是再测固定 U→P/P→U，也不是再做 P_SCOPE 选择。

## 5. 不能误用的历史结论

- Equal 下的 history-mediated rescue 不可外推为 Sample 下通用方法。
- D2 的 S1 NO-GO 不能证明共享—专有反馈不存在。
- 专有参数更新弱收益不等于专有参数对共享梯度没有影响。
- AUC 与 LogLoss 分别评价排序与概率质量，不得相互替代。
- 代理梯度一致性改善不是排序收益；必须报告两者关系。
- 域大小不是难度，样本量不作为候选信号。

## 6. 研究问题

RQ1：P consistency update 是否能降低各 cell 的共享梯度不一致？
RQ2：一致性下降是否对应更好的六格排序，还是仅改变梯度几何？
RQ3：样本×任务的局部排序影响是否存在超出 loss/频率代理的异质性？
RQ4：样本组影响是否非加性，且是否随训练阶段改变？
RQ5：是否存在具备真实语义约束的特征干预/增强对象？

## 7. 结论范围

本计划只验证上述机制在 M3oE fixed1、两数据集和固定协议下的局部可观测性。S1/S2/S3 结果不得被称为完整算法有效性；S4 需新算法规格与独立审查。
