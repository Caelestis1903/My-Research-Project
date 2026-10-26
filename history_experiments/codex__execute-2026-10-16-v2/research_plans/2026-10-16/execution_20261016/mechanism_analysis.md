# 科研机制分析与证伪结论

主代理，2026-10-16。依据[结果报告](experiment_results.md)、[文献矩阵/理论](literature_and_theory.md)及各producer receipt。新增正式训练0；8个零训练case和既有训练复用分开计数。STRONG/MEDIUM/WEAK总是限定所支持命题，不是整个研究方向的评级。

## 1. 稳定重复现象

**STRONG：有限pool、初始化theta、所选坐标内的数值事实。** 同alpha、不同q的期望梯度相同，covariance不同；8个case通过，和旧oracle对应坐标32项一致。固定theta下Adam的E[vhat]反映二阶矩，平均应用更新能随q改变；C分区所选坐标的累计delta均值差在两个B、两个数据集、四model seeds均超过条件MC误差。SGD累计delta的期望不变，实测符合。这个结果验证一个局部预言，没有验证哪个q能改善AUC。

**STRONG：统计对象改变会反转方差结论。** KR Sample的合并坐标trace ratio≈0.505，而旧完整C trace≈2.497；二者对应坐标精确一致。少量坐标/参数分区的选择能够改变“Balanced降噪”的判断，不能把某个scalar trace当全模型、全阶段统一机制。ESS提高也不必然对应目标梯度方差或性能提高。

**MEDIUM：selection functional显著影响本地apparent gain。** 既有B1主要R0 test Δmacro≈+0.000363、2/4正，而预定R1/R4非目标收益4/4正；R1/R4选同一组点，不是独立复制。新排名分析显示target的val-test epoch秩关系比non-target更稳定，但共同早期学习趋势也是原因。需要分别报告训练轨迹与选点函数，不能只报一个selected均值。

**STRONG（限定模型/输入）：AO04 B1融合不是这些probe上的数值常量。** selected/last的rho均未饱和，专家输出差及非零prediction/logit Jacobian可核验；完整B1/2022 validation预测重放匹配。该事实否定“必然rho饱和/严格零敏感度”对这两个state的解释，不证明所有rho方向全局可辨识，也不证明方法有效。

## 2. 反例、不一致与未复现

- 旧S6-R2 KR R0 test非目标约+0.006757、4/4正，重建R0仅+0.000579、2/4正。原初始化/lineage缺失，不能把差别定位为某一个代码因素，也不能把重建当历史造假证明。
- B1 seed2022两臂同epoch仍val/test反号；不同选点时序不能解释全部现象。seed2024的局部阶段/seed反例及R1/R4阳性不能被R0均值抹去。
- 固定theta Adam差异在C/P明显，但B40960的A_arch最弱case只有max-coordinate z≈2.29；不声称每个分区/尺度同等稳定，更不能直接归因于shared C。这个replay没有原main/arch频率的正常训练循环。
- E[mhat]同均值（最大z≈3.35）；m样本路径不同不等于m均值偏移。新结果不能支持“主要是persistent first momentum”的叙事。
- AO04 seed2022 val宏提升约0.001494而test下降约0.008385；新val d2t1边际区间不含0，但test各cell旧区间均含0、只有一个完整配对seed。不能从点估计反号认定稳定generalization reversal，更不能以未完成seed2023充当阴性。
- 一开始按外部文件名认定“没有selected checkpoint”的取证过粗；内嵌selected state补回5份，3个run完整val绑定可升级。其余历史缺口仍在。这是证据更正，不是重建代替原件。

## 3. 证据强度与因果边界

| 命题 | 强度 | 理由/反证边界 |
|---|---|---|
| 同alpha不同q可改变梯度covariance | STRONG（给定pool/theta） | 代数与真实autograd/旧矩阵一致；非新理论 |
| 这种变化可改变coordinate-wise Adam平均局部更新 | STRONG（C投影、fresh固定场） | 数值与真实torch核对；不等于正常训练曲线或收益 |
| 其是S5历史损伤或S6反转的原因 | WEAK | 没有原状态闭环、没有训练中同theta干预、没有selected link |
| selection改变本地apparent gain | MEDIUM（描述性） | 多规则/seed/时序分解支持；选点与训练轨迹耦合，test已反复开发 |
| AO04的融合方向全局不可辨识 | WEAK | 局部非零Jacobian反对严格零敏感度；覆盖只有一seed两个state |
| Balanced/动量/课程学习普遍有效 | WEAK / 无本轮直接收益证据 | 不同alpha/坐标反例，未跑课程学习，未验证一般性方法 |
| 已具备新算法创新及实用收益 | WEAK / 条件不满足 | 15篇矩阵含直接同构先例，缺强Sample对比与新holdout |

## 4. 规模、难度、任务、baseline和训练阶段

KR train d0=172782、d1=1743898、d2=89452；target d1是最大域。既有B1 test baseline AUC：d0约0.627/0.655、d1约0.699/0.726、d2约0.655/0.673；AUC受任务、正例数和排序结构影响，不能仅凭这些绝对值给出可跨任务比较的纯难度排序。validation/test d1占比约98.2%，与训练约86.9%不同；val/test粗域比例接近，不能单靠该比例解释反号。

R0小域d0/d2的任务效应均有正负，d1 AUC近零但loss可能变差。不存在稳定跨seed/split/任务的“小域改善、大域下降”规律。某些selection下确有non-target AUC收益/target LogLoss代价，属于**特定规则与指标的trade-off（MEDIUM）**，不是按域规模的一般规律。

域规模作为主解释变量为**WEAK**。没有独立改变N、unique rows、exposure、q、alpha；本oracle每域pool都16row，仅p进入alpha/q定义。不能将重复曝光称新增独立数据，不能把小域等同困难域。ML嵌套rating标签与KR行为标签不同，任务相关性、物理域、label prior、baseline与训练阶段同时变化；单M3OE底座也不能外推MMoE/PLE或断言M3OE必然在困难域失败。

## 5. 竞争解释

1. **估计器几何与adaptive非线性（STRONG局部，WEAK下游）**：trace并不足以预测Adam应用更新，逐坐标均值/方差及接近零梯度坐标都会影响除法。当前没有theta反馈，训练后几何可能不同。
2. **选择时序/验证噪声（MEDIUM）**：seed2023的timing抵消明显，但seed2022同epoch反号说明其非充分解释；两锚点恒等式不是分离因果的实验。
3. **有限正例、窗口与校准（MEDIUM描述，WEAK单因果）**：小cell区间宽，target LogLoss代价未必伴随AUC变化；条件row bootstrap不含训练seed、用户/时间聚类或选择不确定性。
4. **C/A耦合与历史路径（WEAK已识别分量）**：历史DH同时影响C/A current来源，m/v与真实可达theta未分离。固定初始化replay既不证明C充分，也不证明A充分。
5. **融合本身可动但信号不泛化（WEAK）**：局部Jacobian不为零而selected收益不稳定，与过拟合/shift/冗余路径等多种解释相容；没有独立干预/全epoch覆盖，不能择一作为已证实机制。

## 6. 证伪优先的下一项工作与停止决定

先补齐明确缺失的S5/S6原source/config/init/batch/state/prediction链条与所需审查。没有证据时保持INCOMPLETE，不能用更多随意重训救援。2D暂不运行的原因是这些前置条件没有闭环，**不是要求在2D运行前就证明其收益**。

若原规划前置门可通过，最有区分力的正式实验就是预定Stage2D：KR-natural/M3OE，SN/SB/EN/EB×4seeds，冻结相同budget、init/评价与guards，仅以SB−SN和EB−EN识别同目标采样效应；记录曝光/状态并按R0及同epoch并列，允许无实用差异时停止Track A。这样直接检验局部state变化是否连接selected AUC/LL，而非再确认Adam公式。任何新方法确认另需未开发temporal/external holdout。

本轮不启动Stage4/4B、Stage6/E1，不重新打开CLOSED路线。已完成的法证和数值只推进observations→local mechanism一段，controlled normal-training intervention→independent replication仍未完成；保持这些缺口比宣称整条闭环成功更符合现有证据。
