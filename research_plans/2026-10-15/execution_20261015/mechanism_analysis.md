# 机制分析与证伪边界

状态：完成可访问历史证据的法证复核，另完成C2.1固定θ数值oracle；无新正式训练、无新方法有效性或创新性认定。证据强度STRONG/MEDIUM/WEAK描述支持范围，不能与L0–L5来源等级混为一谈。

## 稳定现象与矛盾

**STRONG（限定为当前CSV算术事实）**：S5 Equal/KR/d1/M3OE的selected target rescue约T−F +0.0414/+0.0423，HT−F +0.0411/+0.0405，四seed两任务均正；DH−F约+0.0009/+0.0008。S6-R2 KR selected test ΔN=+0.006757（4/4正），但同组validation ΔN=−0.003851（仅1/4正）。本次独立重算确认这些数字；没有独立重跑执行链。

**MEDIUM（历史状态整体解释，非本轮因果复现）**：S5 pattern与persistent history总体因子相关，但C与A_arch同步干预、m/v/step/路径未分离，原moment gate只15/16。不能把它解释成m单独致因，或把shared expert的名字当作梯度隔离证明。selected选点时序本身也跟history改变，必须做同epoch与独立选点的两种比较。

**MEDIUM（现有预测点估计；泛化解释WEAK）**：AO04 KR B1/2022 validation macro升0.001494，test降0.008385，两个d0 cell分别降0.022434/0.030011。原test逐cell指标可以从预测重算至浮点精度，但本次六个paired bootstrap CI均含0、只有一完整paired seed，故反转的普遍性和统计稳定性证据弱。KuaiRand B1/2023是执行INCOMPLETE，不是科学NO。MovieLens B1两seed平均validation效应约+0.000005765，也不足以主张结构性增益。

重要反例：S6-R2 MovieLens selected test effect近0；S5 seed2024 HT在d0的selected代价明显且T late target collapse；历史S8 selected continuation没有稳定超过原F best，late阶段局部rescue不能替代selected性能改善。S4 norm controls、S5历史pattern和S8阴性边界来自不同协议，不可简单合并为一个机制已被跨设定验证。当前S4/S8原始状态也缺失，此处历史描述不冒充新复算。

## 规模、难度和域/任务代价

规模作为主解释变量的证据为 **WEAK**。KR train d1=1,743,898，d0=172,782，d2=89,452；目标d1反而是最大域。S5改善这个最大域时d0下降，两任务T−F约−0.0273/−0.0374；d2 selected接近零或稍差，HT却同时改善d1与最小d2。因此现象不是统一的“小域获益、大域下降”规律；确有特定干预下的目标/非目标trade-off，但方向取决于物理域、objective、arm、seed与training stage。

域难度不能由n定义，也不能把不同任务的绝对AUC直接当作纯难度排序。应比较同域同任务的F baseline、matched reference、label结构和time window。这里缺失matched reference原预测/状态，不能给域规模因果归因。MovieLens任务为嵌套rating阈值，与KR两个行为任务并不等价；跨数据集效应差异无法单独识别任务相关性原因。

## 竞争解释

1. **选点/训练时序**（MEDIUM，可检验）：S5与S6 selected各自选epoch，不等于同θ/同epoch干预。S6旧test grid稀疏，无法恢复未保存epoch的exact alternative test。最优先的判别是B1 dense replication，但必须先恢复/明确原协议身份和完整状态约束。
2. **有限正例与时间窗口**（MEDIUM事实、WEAK因果）：KR test d0 long_view只有120正例，AO04配对区间宽；时间窗口的特征/条件关系或排名噪声可产生不同泛化方向。当前bootstrap按行分层未建模时间依赖；无法凭该CI排除时间相关性。
3. **domain/label marginal shift**（STRONG描述、WEAK反转解释）：原数据重新计数确认train d1占86.93%，val/test约98.2%，d1 click正率0.3778降至约0.210；但val/test域比例极近，因此粗域mix变化不足以单独解释val/test AUC反号。AUC、LL和校准应分开看，label prior变化不直接等于排序损伤原因。
4. **C/A耦合和optimizer history**（WEAK具体成分归因）：历史history bundle效应不说明共享C足够，也不说明architecture A足够。先做B2 C×A正交隔离，再决定C1 common-θ ghost m/v probes；禁止跨轨迹状态硬插后称分量因果贡献。
5. **融合不可辨识或低敏感度**（WEAK）：AO04现有rho均未接近0/1饱和，scalar derivative系数约0.048–0.049；这只否定“rho饱和必然解释一切”的简单说法。没有E_j−E_k方向与logit Jacobian、冻结θ预测干预，就不能从q变化/不变化断言可辨识或有效学习。gate entropy与branch norm仅日志描述，不是branch collapse证明。
6. **普通过拟合/隐式正则**（MEDIUM历史线索）：S8 late rescue而selected无收益支持这种竞争解释，但不能自动解释S6-R2 KR已选test反转。两条问题应继续分开。

## 最有区分力的后续工作与停机

先恢复S5/S6原始manifest、source/data/init/batch hashes、selected/common/final预测与完整optimizer，关闭A0链条缺口。若原工件不可恢复，需要明确承认不能exact复现，再确定新协议，不将重训结果无标签混入旧结果。

B1限定KR Sample F/DH×4seeds×50epochs，每epoch full C/P/A state、RNG/iterator，冻结validation R0–R4选择；用same-epoch Δpolicy与各自selected Δ并列、cell/label paired bootstrap及时间窗口分析区分timing和噪声。若差异消失或CI不足，停止该线，不开发补救方法。B2才用于Equal的C/A定位，不能与B1混成支持某个先验结论的总证据。

C2先证明固定目标alpha与不同q的权重oracle等价，再谈Adam噪声交互；本次尚未运行该oracle。D1/E1仍未触发。被多次研究过的test只能作forensic，任何新方法性主张必须用未开发的后置/外部holdout。

本次未进行文献全文或创新性查新；规划§16仅是已知背景，novelty保持OPEN。没有把课程学习、动量、小域困难、M3OE困难域失败作为分析前提。

## C2.1 数值证据追加

完整分析见 [固定参数oracle](C2_numeric_oracle_20261016/numeric_analysis.md)。**STRONG（有限pool/固定θ）**：同α不同q的期望loss/全梯度相同，协方差会改变；Balanced在当前Equal下减小方差、Sample下增大方差，两个数据集四seed的C/P/A均同向，构成“平衡抽样普遍降噪”的限定反例。跨θ阶段、全训练数据和selected性能的推广为**WEAK**，不能据此宣布C2训练GO。

**STRONG（条件性概率上界）**：小batch空域归一化偏差不能直接解释原40960大batch情形；考虑较小末批，KR最稀少域缺失概率小于10^-793，ML小于10^-165（均匀无放回batch边际的IID上界，取宽松整数界）。类别/时间/测试有限正例及选点时序仍是竞争解释。

Adam实测仅覆盖detached坐标副本的fresh step0，β经bias correction在第一步抵消，因此没有persistent momentum的因果证据。有限pool中每域row数相同，q效应不能转换为“真实域规模是主要原因”的证明；也没有小域/大域selected AUC权衡数据。下一项最有区分力的工作仍是B1正式dense训练；若随后评估C2，应先在真实阶段θ、匹配optimizer state和正式batch下核查差异，再决定训练矩阵。
