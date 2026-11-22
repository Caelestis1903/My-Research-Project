# 全部实验矩阵、依赖与复用

## 1. 全局固定量

datasets={movielens,kuairand}；seeds={2022,2023,2024,2025,2026}；50epochs；batch40960；FP32；自然原行序80/10/10；M3oE fixed1；Adam lr=.01、β=(.9,.999)、eps=1e−8、coupled WD ML1e−4/KR1e−5；固定EMA ρ ML.9828205985452511/KR.995。A22只在S_dense使用β1=.99。不做少seed性能预筛选。

训练臂必须记录arm_id、dataset、seed、parent_source_sha、source_hash、data_hash、config_hash、pair_rng_schema、readout、selected_epoch/hash、physical_attempt、cost_role。所有表里online与EMA是同一训练的不同读出。

## 2. 方向 A 训练矩阵

| ID | 精确定义 | 既有科学轨迹 | 本轮新科学训练上界 | 依赖/启动门 |
|---|---|---:|---:|---|
| A00 | clean普通Adam，online与F | 10 | 0 | 必备强内部对照；B重放可恢复状态 |
| A10 | P00静态pair+普通Adam，λ=.05 | 4 | 6 | 核心必做 |
| A20 | P01静态pair+ECCM，λ=.05 | 10 | 0 | 必备历史候选 |
| A30 | P10动态条件抽样+普通Adam，λ=.05 | 4 | 6 | 核心必做 |
| A21 | PCOLL_STATIC_099，静态pair，真实m.9+单辅助h.99 | 0 | 10 | 任一数据A20−A10 Validation正均值 |
| A22 | ADAM_S99，静态pair，S_dense单混合m.99 | 0 | 10 | 同A21 |
| A40_HISTORY_L025 | 冻结获胜历史H*，λ=.025 | 0 | 10 | 独立增量成立；已登记历史或本轮六格代价 |
| A41_ADAM_L025 | 同静态pair与λ=.025，普通Adam | 0 | 10 | 与A40成对必做，防止换λ后丢失组件归因 |
| A50_STAR_SKIP_ONLY | H*仅在shared_weight/bias、skip_conn、star_mlp采用历史改造，expert回普通Adam | 0 | 10 | 剂量之后仍有本轮轨迹支持的子组代价 |
| A51_EXPERT_ONLY | H*仅在expert采用历史改造，其他S_dense回普通Adam | 0 | 10 | 与A50成对，保留同λ监督/采样 |
| A_STATE_REPLAY | 对需要真实轨迹归因的A20/H*恢复完整历史状态 | 不增加n | 0 | 最多10条物理重放；只在状态缺失且有实际归因需求时 |

A50/A51中的H*由A21/A22比较决定；若H*=A22，限制的是β1=.99的作用范围，不能称为ECCM条件表删减。对于λ=.025，A41是最终独立增量的必要匹配控制；不能借用λ=.05的A10进行纯历史归因。

A正式新科学上界72条。P11、PCV、其他λ、RMS归一化和新pair预算均不在默认矩阵。动态A30只是抽样对照；没有同时独立的两组件证据，不追加P11。

## 3. 方向 B 核心机制矩阵

| ID | 数据/种子/锚点 | 操作 | 物理成本 |
|---|---|---|---|
| B_REPLAY | 两数据×5seed×50epoch | 同源clean，恢复真实Adam/EMA，主轨迹不改 | 默认10条完整重放 |
| B0_GRAD | E5/E20/E35/E50末batch，更新前 | 六个真实cell梯度、两套label计数匹配的随机loss-entry分组 | 共40锚点；18额外autograd/点=720 |
| B1_STEP | 每锚点G∈{EMBED,S_DENSE,N} | 一份JOINT；每G的ZERO_DATA、6 ONLY、6 LOO；EMBED上2×6伪LOO | 52虚拟Adam/点=2080 |
| B1_NORM | 同锚点、共享组 | JOINT/LOO位移缩至共同最小范数；online/EMA读出 | 额外前向，不增加optimizer步 |
| B1_WINDOW | E20/E35起的正常父轨迹窗口 | 相同时间权重的参数平均和概率/logit平均 | 流式RAM累计；额外前向 |
| B2_PULSE | 触发数据集的5seed、E20/E35；每锚点真源+partition0伪源 | 一次LOO脉冲；其后正常联合更新并重算梯度 | 最大40逻辑段，共1380步；其中40首步可复用B1，增量实际1340次Adam |

参数组定义见06。S_DENSE是S去掉EMBED，且应与v3 shared_name的坐标一致；N=D/T/DT所有非全局共享坐标，并非必然没有跨格作用的零效应组。域embedding三行归D，不并入EMBED。

B1_WINDOW窗口长K=该数据每epoch更新数(20/49)，按global_step标号；它是有限窗口函数组合对照，不是旧全轨迹预测EMA的重建。B2也按K总步计，末batch锚点不能误写为“完整下一个epoch末”。

B2按数据集整体触发，入列后全部5seed、两个锚点都完成。train规则返回NO_ACTION的分支以零动作纳入，不筛掉负seed或无动作锚点。A是否晋级不影响B0/B1完成。

## 4. 方向 B 完整候选

仅当B门满足且EMBED中有可操作的残余排序代价：

| ID | 动作 | η搜索 | 训练数 |
|---|---|---|---:|
| B_ADAPT | 每epoch从训练面板选择一个源q或NO_ACTION，在EMBED减弱该源data gradient | .25、.5 | 20 |
| B_STATIC | 由训练面板跨seed冻结固定q；同样在EMBED减弱 | .25、.5 | 20 |
| B_UNIFORM | 重放对应ADAPT的动作时序与源质量，对EMBED全data gradient统一缩放 | .25、.5 | 20 |

三者使用相同clean目标基础、真实Adam状态、原样WD和全参数F。B_UNIFORM依赖同seed/η的ADAPT动作日志；缩放仅匹配被移除的名义数据质量，不声称actual update范数相同。六格、Macro、成本全部报告，不能用STATIC/UNIFORM的差值替代相对A00的绝对独立增量。

## 5. 条件联合与Gaussian补充

| ID | 条件 | 训练数上界 | 对照 |
|---|---|---:|---|
| AB_FROZEN | 至少同一数据A、B各自有独立增量，06第11节正面桥接通过；冻结双方参数 | 10 | A单独、B单独、A00；较强单方法为联合增量参照 |
| G_FROZEN | 已选主方法需要决定是否纳入既有增强；固定Gaussian D0背景 | 10 | 同身份G-D0+F；报告新增优化组件独立Δ |

AB/G唯一通道公式、原Gaussian替换目标及B原trace重放见06第11节；B只减弱原始自然gq，PAIR与Gaussian残差不随B减弱。AB只有一套冻结组合，不搜索额外融合系数；G只继承λ_aug=.5与kappa ML.05/KR.1，不再搜索覆盖域。若主方法为AB，则G_FROZEN使用该最终主方法，而非对A、B、AB分别跑三套。

## 6. 正式发表模型与最终Test

M3oE、HyperGate、AutoIFS已有统一自然人口/Validation AUC选点五种子结果，默认新增训练0。ML强发表参照为M3oE，KR为HyperGate；冻结它们的原协议、数据与端点身份。若身份验证显示本轮数据/选择语义实质不一致，标出具体差异并建立独立表，不能跨表相减；只有该不一致确实影响最终算法资格才补齐相应对照。

最终Test矩阵至少包含A核心四臂的五seed组件比较、最终候选及其准确匹配的普通更新/简单历史控制、A00+F、同协议三发表模型。已经存在的历史Test直接复用、标HISTORICAL_EXPOSED；新模型只在FINAL_FREEZE之后读取Test。未晋级而只完成机制研究时，报告完整Validation/机制结果，不为淘汰臂批量开放Test。

## 7. 规模账本

| 情形 | 完整物理轨迹 | 新科学轨迹 | 说明 |
|---|---:|---:|---|
| 核心必做 | 22 | 12 | 12缺失补全+10状态重建 |
| 核心+A简单历史 | 42 | 32 | 常见第一轮扩展 |
| 所有条件分支均触发的理论上界 | 172 | 152 | 含额外A状态重放10；非默认排满 |
| 正式发表模型重跑 | 0默认 | 0默认 | 有具体身份差异才另行记录 |
| 技术失败重试 | 按实际 | 不增加独立n | 不预分配“重跑seed”来取最好结果 |

172 = 22 +20(A21/A22)+10(A状态)+20(剂量匹配)+20(作用域)+60(B三形式两η)+10(联合)+10(Gaussian)。
完整轨迹均50epoch；全上界8600epochs、296700次主轨迹Adam/EMA（86条ML与86条KR）。核心1100epochs、37950次主更新。B的2080虚拟步和最多1340增量短段步单列，不伪装额外seed。

执行过程中逐门展开，分支停线即结束该路径；不能为了利用空闲GPU运行未触发候选。
