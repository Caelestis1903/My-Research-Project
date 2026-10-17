# 全部研究问题去向表

来源：[Caelestis1903/My-Research-Project @ 2d04c6de6c92](https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/question_registry.json)。本表为2026-10-17后续规划，未执行新实验。

原Q1–Q46全部保留：14 RESOLVED、1 FALSIFIED、22 BOUNDARY、9 NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN。原字段、证据、结论与不确定性完整保存在配套 QUESTION_DISPOSITION.json 的 original_registry_entry；下表的后续动作是本次建议。

新增问题不设数量上限；已解决问题不为追求统一结果而重复启动，结构不可识别问题不靠增加seed假装解决。

## 阶段索引

|阶段|内容|
|---|---|
|N0|已有结果评价函数、人口、选择规则与固定窗口复算|
|N1|对称q×LR校准，独立新种子和完整共同预算|
|N2|IID与随机整数域quota：相同目标与相同预期曝光，分离域计数通道|
|N3|双路径锚、多draw、C子块与norm×direction交互；条件瞬时Adam映射|
|N4|N3独立确认后，使用新seed的tied/untied与真实/固定随机分组总拓扑对照|
|N5|条件ORDER尾部、支持/重复、ML人口解耦与mask信息破坏边界|
|N6|定题文献覆盖与效应/机制/方法分层终局|

## 原Q1–Q46

|ID|原状态|问题|下一阶段|最小后续动作|
|---|---|---|---|---|
|Q1|RESOLVED|Objective-equivalent sampling 是否真的改变 training trajectory？|N0, N1|复算评价函数/人口后，以对称q×LR和独立确认种子界定条件效应。|
|Q2|RESOLVED|是否只是 checkpoint-selection artifact？|N0, N1|固定selection/evaluation人口和读数，复算共同轮次与自选差异；新确认使用冻结规则。|
|Q3|FALSIFIED|是否依赖 Adam？|冻结历史结论|封存无动量SGD反例，不重做Adam通用必要性检验。|
|Q4|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|若 optimizer-specific，是否为 m/v/history？|N3|先从已存状态反事实做逐块方向×幅度区分；只有新状态假说仍有证据才新设健康保留长期状态控制。|
|Q5|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|非 Adam 机制是 exposure/repetition/coverage/order/geometry 中哪一个？|N2, N3, N5|先IID/固定quota分离域计数波动，再多draw块干预；重复/coverage/order仅按条件侧支。|
|Q6|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|shared/private parameter 是否必要？|N3, N4|SN/SB双路径锚和32draw确认C局部作用，再条件执行tied/untied长期拓扑对照。|
|Q7|BOUNDARY|domain size 是否解释 effect？|N5|保留别名复制在WR下完全同规律结论；仅独立支持N成为目标时使用真实独立样本或受控生成分布。|
|Q8|BOUNDARY|domain difficulty 是否解释 effect？|N0, N5|保留mask阶段/人口反例；先重算评价函数，只有自然难度归因成为目标才多mask及独立受控因子。|
|Q9|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|task/domain imbalance 是否解释 effect？|N2, N5|N2在固定总体下识别计数波动；独立imbalance主张须N5受控支持/目标p/采样q因子。|
|Q10|BOUNDARY|validation selection 是否改变结论？|N0, N1|完整保留四selection、common-epoch及不同评价权重，N1预先选择主要estimand。|
|Q11|BOUNDARY|是否跨环境成立？|N5|固定ML训练行列表及优化seed，复制N1关键q对比；仅机制存活才复制该机制。|
|Q12|BOUNDARY|SAF/D3/AdaTask 是否覆盖？|N6|依据N2/N3/N4具体命题核对SAF/D3/AdaTask及新增相关文献；机制等价与性能基线分开。|
|Q13|BOUNDARY|是否存在未被已有工作覆盖的 mechanism gap？|N6|将拟声明机制写成可证伪命题后核对已有理论和实证覆盖。|
|Q14|RESOLVED|是否值得方法化？|N6|保持v3 NO_GO，新轮分别评估效应、机制、方法三种成功标准。|
|Q15|RESOLVED|SN/SB 的“same objective”在代码和 empirical gradient 上是否严格成立？|N2|沿用原同图同J身份，仅核验新增固定quota与新q权重的期望/数值身份。|
|Q16|RESOLVED|现象是否主要发生在 overfitting 之后？|N0, N1|保留操作性pre阶段已存在效应，新的主要早期窗固定于外部clock。|
|Q17|BOUNDARY|effect 是否只是 learning-speed / finite-budget effect？|N1, N2|对称q×LR、共同预算、更新/样本钟和阈值右删失；N2匹配步数/样本预算检验计数波动。|
|Q18|RESOLVED|在相同 sample multiset 下，仅改变 optimizer-step 顺序是否改变结果？|N5|保留same-multiset整体ORDER作用；纯内容顺序主张需要尾批固定与等大小batch置换。|
|Q19|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|repetition 是否是 exposure effect 的必要组成？|N5|以固定E,N的R=E−U给出原零重复干预不可行证明；可条件研究E≤N早期窗或同E/U/R重复时间结构。|
|Q20|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|coverage 是否独立于 exposure/repetition？|N5|保留coverage与repeat代数约束；研究同E/U/R下重复集中度/间隔，或另行真实支持实验。|
|Q21|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|estimator covariance 的 anisotropy 是否比 scalar variance 更有解释力？|N2, N3|N2先识别计数波动；N3必要时做同均值和trace匹配的局部shape噪声干预、逐块幅度/方向。|
|Q22|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|shared C/R 是否是必要 mediator 而非相关参数块？|N3, N4|与Q6共同执行C多draw/双锚和条件共享拓扑；R维持已私有事实。|
|Q23|RESOLVED|optimizer effect 与 sampling effect 是否存在真实 interaction？|N1, N3|保留校准配置interaction及固定reference反例；N1明确q×LR响应，N3区分更新幅度/方向。|
|Q24|BOUNDARY|第二环境失败时，boundary 究竟来自哪里？|N0, N5|先统一报告评价函数/人口与外部clock，再固定ML支持；需要唯一driver才独立操纵候选因子。|
|Q25|BOUNDARY|existing literature 是否已经覆盖“same objective + different q + MDMT trajectory”机制？|N6|同Q12/Q13，按新具体同目标/采样/MDMT命题核对通用与场景工作。|
|Q26|RESOLVED|D3/SAF 的 objective change 是否使其不能作为 mechanism-equivalent prior work？|冻结历史结论|保留D3/SAF完整学习权重一般改变J的公式边界。|
|Q27|RESOLVED|AdaTask 的 optimizer-state intervention 是否能消除 sampling effect？|N6|固定同目标AdaTask组件未消除q敏感性已回答；只有完整方法优劣问题才等预算调参对比。|
|Q28|BOUNDARY|routing 的 stop-gradient 边界|冻结历史结论|保持routing detach图，不静默改计算图。|
|Q29|RESOLVED|参数块名称与实际共享支持|N3, N4|沿用C/E共享、R gates/towers单元私有、P混合的真实支持定义。|
|Q30|BOUNDARY|Adam 学习率赢家位于网格下边界|N1|补Adam预定低邻点并对两q提供相同调参预算。|
|Q31|BOUNDARY|pilot 与 finalist 种子重合|N1|旧2022/2023仅作tuning，正式确认用不重合新seed；复用轨迹不计独立重复。|
|Q32|RESOLVED|历史固定 SGD 数字变化来自哪里|N0|保持原训练数值漂移已排除，仅在图表标清selection/evaluation人口。|
|Q33|BOUNDARY|只用 SN 调 LR 的估计对象|N1|同时估计同LR q干预与SN/SB等预算各自调优表现。|
|Q34|RESOLVED|一阶矩遥测混入二阶矩范数|冻结历史结论|只使用已修正corrected_m_norm与v_norm，旧mixed列如实保留。|
|Q35|RESOLVED|稀疏遥测 lag 与真实 step lag|N3|旧稀疏lag事实保留；若N3确需时间相关性则预设连续步窗口及真实step编号。|
|Q36|BOUNDARY|onset 程序不提供全族错误率保证|N0|新主端点用预定外部窗口；只有正式全局onset主张才另外校准同时程序。|
|Q37|BOUNDARY|IID 批次置换与课程策略|N5|相同batch集合下多个预定order seed，区分条件路径方差与总体策略平均。|
|Q38|BOUNDARY|ORDER 也移动短尾批次|N5|需要纯内容顺序解释时，尾批固定仅置换等大小batch，并单独比较尾部位置干预。|
|Q39|RESOLVED|GPU 跨预算前缀数值复用|冻结历史结论|沿用已核验前缀复用，对新产物只检查必要producer/model/pred来源。|
|Q40|BOUNDARY|状态消融的健康、幅度和学习进度|N0, N3|固定外部clock，逐块范数/方向和绝对AUC/LL/J共同报告；只有新证据才长期健康保留state控制。|
|Q41|BOUNDARY|自身早期窗口造成的环境表观翻转|N0, N5|跨环境使用预定共同窗口并列实际updates/examples；必要时固定训练支持。|
|Q42|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|全局等范数没有隔离角度与块尺度|N3|逐块native/candidate方向×native/candidate范数2×2，零块处理预设。|
|Q43|BOUNDARY|MovieLens 的 seed 同时改变训练子集|N5|先固定ML训练行列表，仅改变优化seed；需要方差分量时再独立subset_seed×optimization_seed交叉。|
|Q44|BOUNDARY|破坏性控制改变共同早期窗口|N0|主要早期窗口独立于候选消融失败；固定10/20/50等时钟与原窗口同时保留。|
|Q45|BOUNDARY|长时钟读数只有预算条件下的部分种子支持|N1|预先选择共同主要预算；如主张100轮则所有正式确认seed具100轮，否则限定50轮。|
|Q46|BOUNDARY|局部C的AUC阳性与loss/J冲突|N0, N3, N4|分别看排序/LL/J与评价人口；C多draw/双锚检验，条件长期拓扑。|

## 本次新增Q47–Q52

|ID|状态|问题|第一判别步骤|
|---|---|---|---|
|Q47|OPEN_PLANNED|宏评价、train-p与test-p评价的反转由哪些cell贡献？评价权重变化与域内人口变化如何区分？|N0, N1：固定预测逐cell算术分解、预定三套权重、严格单调校准；N1独立训练seed确认该现象。|
|Q48|OPEN_PLANNED|短pilot选出的LR赢家是否只优化早期速度，且双方等预算调参后采样差距是否保留？|N1：Adam/SGD各4点LR×2q×2tuning seed，H100；冻结共同与各自eta，用8个独立confirmation seed确认。|
|Q49|OPEN_PLANNED|域计数波动相对于域内样本噪声贡献多少，固定域配额是否改变downstream采样效应？|N2：IID/quota×SN/SB四臂；随机系统舍入配额、实际权重Bp_d/n_d；先16锚固定点，再按诊断残差触发8seed训练。|
|Q50|OPEN_PLANNED|C局部作用来自哪些子模块、幅度/方向及其交互？共享拓扑的长期作用是否成立？|N3, N4：双路径锚、多draw、C子模块、norm×direction四格及交互；独立seed确认后才做tied/真实untied/固定随机untied。|
|Q51|OPEN_PLANNED|在相同theta、m/v历史及step下，Adam瞬时非线性是否把不同梯度分布映射为不同平均更新？|N3：复用同theta梯度bank比较E[U(g)]、U(E[g])、SGD及共同滞后固定分母线性映射；仅当Adam残差重要时触发。|
|Q52|OPEN_PLANNED|EMS/variance-adapted LR与MIS/OMIS现有工作是否已覆盖最直接的候选解释和方法？|N6：逐式原文差异表；只有候选主张仍成立时执行论文规定的同目标基线。|

## 最终记录要求

每项最终记录：原始观察、估计对象、已做实验、结果与不确定性、排除/未排除解释、状态、最小后续。NOT_TRIGGERED不是阴性；INCONCLUSIVE说明精度或设计限制；新可行干预不冒充已实现原结构不可能的反事实。阶段门槛与预算以 NEXT_EXPERIMENT_PLAN.md 为准。
