# PUB迁移算法设计

## 统一符号
参数组g的活动任务集合C(g)，任务损失L_c按自然样本目标分解。更新前Adam状态m_g,v_g,step=t。任务梯度g_c,g=∇_{θ_g}L_c。候选u_c,g=AdamCandidate(g_c,g;m_g,v_g,t)，纯函数计算，不修改optimizer state。

## P0：两任务
Task0/Task1分别聚合三个domain的原始自然样本损失。S和D0/D1/D2上构造两个候选更新并求alpha；T0/T1只有一个实际任务梯度，不做二任务求解；DT专有参数普通Adam。优点是最接近论文、成本低；缺点是域差异被聚合。

## P1：六任务全局
在S上对六个L_dt构造候选并求alpha_S。D_d、T_t组沿用alpha_S在C(g)中的子集，不重新求解；DT普通Adam。优点是统一任务偏好；缺点是S上的权重未必适合D/T局部子空间。子集alpha不做softmax或归一化，PUB原版无simplex约束。

## P2：拓扑感知局部
S六任务；每个D_d两个task；每个T_t三个domain；DT单任务普通Adam。各多任务参数组独立构造Gram并求alpha。优点是适配局部参数参与关系；缺点是局部解噪声、额外自由度和成本更高，不预设最优。

## 梯度空间控制
使用相同alpha和原始任务梯度，形成gbar=Σalpha_c g_c后执行一次标准Adam。与PUB的Σalpha_c u_c对照，区分动态权重与更新空间组合贡献。

## Adam历史
所有候选读取同一更新前m/v/step。全部候选计算后才求alpha、更新参数。随后按论文用相同alpha合成原始梯度：
gbar=Σalpha_c g_c；
m_new=β1m+(1−β1)gbar；
v_new=β2v+(1−β2)gbar²。
论文把moment更新放在参数更新之后，并承认此策略理论未充分分析；第一阶段保留原始PUB，不增加独立moment变体。

## Weight decay风险
项目Adam配置含weight_decay，必须从冻结manifest确认具体值和coupled/decoupled语义。由于alpha不归一化，对每个任务候选重复加入相同λθ会让正则项按Σalpha缩放，不可接受。主方案先将数据梯度用于候选平衡；共享状态更新时按PUB权重合成数据梯度，再按原训练语义只加入一次公共λθ并更新m/v。必须写出逐坐标公式并与标准Adam一步对照。若无法严格匹配，先做所有对照同设定的zero-WD机制诊断，再做带WD正式比较；不得悄悄改AdamW。

## 训练伪代码
1. 一次forward得到两列任务预测。
2. 构造六个L_dt，验证损失和及梯度等价原始loss。
3. 对所需cell用autograd.grad取梯度并抽取参数组活动坐标。
4. 用同一更新前m/v/t计算候选Adam更新。
5. 分块点积构造Gram；求解正alpha并记录残差。
6. 每个参数组组合更新，每坐标写入一次。
7. 组合原始梯度更新对应Adam状态，step仅递增一次。
8. 单任务专有参数组执行标准Adam等价更新。
9. 记录alpha、Gram、候选范数/余弦、求解状态和成本。
10. 按原协议Validation，Test隔离至冻结后。

顺序：P0 correctness → P1 correctness → P2 correctness；性能优先P0/P1，P2完整保留但不预设优先晋级。
