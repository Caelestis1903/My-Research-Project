# 07｜归因指标、数值闭合与统计协议

> 状态：**DESIGN_ONLY**。本文件定义 Codex 后续必须实现的计算与验收，不包含本轮新实验结果。本文公式是分析定义；已审计训练事实的出处见 [01_RESEARCH_CONTEXT.md](01_RESEARCH_CONTEXT.md)。执行顺序见 [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)，状态构造见 [05_IMPLEMENTATION_SPEC.md](05_IMPLEMENTATION_SPEC.md)，结论判定见 [08_ANALYSIS_AND_DECISION.md](08_ANALYSIS_AND_DECISION.md)。

## 1. 指标对象、独立单位与比较身份

### 1.1 不可混用的三种比较

| 比较身份 | 参数状态与评价时间 | 回答的问题 | 必须使用的字段 |
|---|---|---|---|
| `matched_anchor` | 同一物理轨迹、同一 optimizer step；E5/10/20/35/50 | 某组或某段历史修正对当前函数造成什么效应 | `trajectory_id, seed, step, state_id, reference_state_id` |
| `matched_at_F_selected` | 同 seed 的 F 所选 epoch，同刻 O/F/诊断态 | F 最佳验证读出时，修正的条件效应是什么 | `selection_owner=F, selected_epoch_F` |
| `independent_selected` | O、F 各自根据完整 Validation 选点，再评价各自 Test | 两个读出规则的最终性能差异是什么 | 各自 `selected_epoch, selected_step, selection_metric` |

正式主性能表仅对 O/F 使用第三种身份。全局冻结后额外允许在 F-selected 同刻评价 O、S-only、N-only，并与同刻 F 组成二因素 Test 边界检查；这不属于各诊断态独立选点的算法性能。最终 Test 的五种状态身份是 O-selected、F-selected、O-at-F-selected、S-only-at-F-selected、N-only-at-F-selected；后四者包含同刻二因素对照。诊断态不是本轮提出的新算法，不要求独立 50 epoch 选点，也不需要每个干预达到 +0.003。时间和接口主机制以 Validation 为证据，不在看过 Test 后追加新的时间或接口选择。

每个 dataset×objective 条件都有 seed 2022–2026，共五个真正独立训练单位。O、F、组替换、时间替换、pulse、同一 seed 的所有锚点均不增加独立样本量。相同 seed 的确定性恢复重放不计为第六个 seed。

Clean 是机制发现主条件，Gaussian D0 是预先规定的边界条件；不得看到 Test 后交换身份。两个数据集分别形成结论，不以跨数据集平均掩盖方向差异。

### 1.2 正式评价定义

- 六格 AUC：对每个有效 domain-task cell 的完整自然评价人口计算，保持原 FP32 概率输出与原 evaluator 的 ties 语义。
- `Macro AUC`：六格等权算术平均。`Traffic AUC`：按已审计人口权重加权的 cell AUC。`Pooled AUC`：按源定义跨域合并后评价；三者不能混称“总 AUC”。
- LogLoss：继承源概率转 float64 后的 `max(log(p),−100)`、`max(log1p(−p),−100)` 规则，不替换为新的 epsilon clamp；报告未缩放损失。
- 选点：完整 Validation 六格等权 Macro AUC 的最早严格最大值；不使用 epsilon 改变并列规则。若第 e 轮与已有最大值数值相等，保留更早轮次。
- 最终 O/F Test：五 seed 配对平均 ΔAUC≥+0.003 是算法收益标准；六格平均 ΔAUC≥−0.001 单独检查。单格不通过时保留整体收益并明确保护条件未满足。LogLoss 独立裁定，不用它直接否定 AUC 改善。

这些数值语义源于已审计的 [RESOLVED_PROTOCOL.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json) 与 [readouts.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py)。新统计代码应调用或复用其评价语义，不重新定义一个近似 evaluator。

## 2. 模块 A：由准确排序转移形成 AUC 增量

### 2.1 含 ties 的精确恒等式

一个 cell 内正、负样本数分别为 P、N；对真实正负样本对 i,j 定义

$$
m^{p}_{ij}=p(x_i^+)-p(x_j^-),\qquad
\phi(m)=\mathbb 1[m>0]+\tfrac12\mathbb 1[m=0].
$$

$$
AUC=\frac1{PN}\sum_{i=1}^P\sum_{j=1}^N\phi(m^p_{ij}).
$$

令排序状态 a,b∈{0,1,2} 分别代表错误、平局、正确；`N_ab` 是参考态到干预态的转移数量，则

$$
\Delta AUC=\frac1{2PN}\sum_{a=0}^2\sum_{b=0}^2(b-a)N_{ab}.
$$

等价地，记 `W/T/C` 为错误/平局/正确：

$$
\Delta AUC=\frac{N_{WC}+\tfrac12N_{WT}+\tfrac12N_{TC}
-N_{CW}-\tfrac12N_{CT}-\tfrac12N_{TW}}{PN}.
$$

必须同时报告完整纠正、半单位纠正、完整破坏、半单位破坏及未改变的 pair 质量，不能只展示被纠正的样本对。

### 2.2 全人口 O(N log N) 计算，而非枚举 PN

下列为待实现算法伪代码。`ref` 和 `new` 是同一批行、同一真实标签的两组正式概率；输出 int64 的 3×3 表。

```text
NEG = 所有负样本的 (ref_score, new_score)
POS = 所有正样本的 (ref_score, new_score)
对 NEG.new_score 做坐标压缩；相等数值使用同一个 rank
建立 ALL_NEG 的 new_score 有序频数，用于总计 <、=、> 查询
Fenwick LESS = 空频数树

按 ref_score 从小到大遍历 POS 与 NEG 的所有相等分数块 v:
    # 此时 LESS 只含 NEG.ref_score < v，不能提前插入等于 v 的负样本
    EQUAL = ref_score == v 的负样本 new_score 排序列表
    对每个 ref_score == v 的正样本，令其 new_score = q:
        L = LESS 中 new_score (< q, == q, > q) 的三个计数
        E = EQUAL 用 lower_bound/upper_bound 求出的相应三个计数
        A = ALL_NEG 中 new_score (< q, == q, > q) 的三个计数
        G = A - L - E
        # L: 参考态正确；E: 参考态平局；G: 参考态错误
        # 三个列计数(<,=,>)分别对应干预态正确、平局、错误
        N[C, (C,T,W)] += L
        N[T, (C,T,W)] += E
        N[W, (C,T,W)] += G
    将本块 NEG.ref_score == v 的全部负样本插入 LESS

assert sum(N) == P*N
assert min(N) >= 0
return N
```

时间复杂度 O((P+N)log(P+N))，内存 O(P+N)。每个 cell 分开处理；可以在 CPU 上对不可变预测数组并行，不能并行修改训练模型。至少使用 int64；先检查 PN 是否超出 int64 范围。排序和 binary search 使用真正保存的概率值，不增加 tie epsilon、不随机打破平局。

验收：

1. 小型人工数组含全平局、部分平局、完全纠正、完全破坏，结果与暴力枚举完全相同。
2. 每个真实人口转移表总数准确等于 PN。
3. 由转移表复算的参考/干预 AUC 与源 evaluator 差值绝对误差≤1e−12；不满足时停止该归因表生成，修复行顺序、标签、ties 或计数错误。
4. 同一状态与自身比较只有对角项，ΔAUC=0。

### 2.3 概率 margin、logit margin与预测变化

同时保留

$$
\Delta p_i=p_i^{new}-p_i^{ref},\qquad
\Delta m^p_{ij}=\Delta p_i^+-\Delta p_j^-,\qquad
\Delta m^z_{ij}=\Delta z_i^+-\Delta z_j^-.
$$

`score_changed`、`margin_changed` 与 `rank_transition` 是三个不同对象。排序归因使用正式概率；功能解释优先同时展示 logits，避免 sigmoid 压缩使真实函数变化难以观察。若概率饱和产生 ties，单独计数，不把 logits 的无平局排序悄悄替换成正式 AUC。

对每个主要干预，至少输出：正、负样本各自 Δp/Δz 的均值与分位数；Δmargin；3×3 转移；净纠正贡献。单纯每个分数改变更多或者 margin 均值变大，不构成 AUC 收益解释。

## 3. 面板、样本对和描述性分层

### 3.1 面板身份

| 面板 | 定义与用途 | 可支持的范围 |
|---|---|---|
| `V_full` | 原完整 Validation 自然人口 | 正式六格 AUC/LL、核心干预真实人口效应及准确转移 |
| `V_mech` | 每域固定最多4096条自然验证行；由有效行组成，保留原任务标签与mask | 逐 step 集成、margin与功能解释；结果仅称面板效应 |
| `V_A/V_B` | 沿用源规则：以 `SHA256(dataset\|row_id)` 的最低位分割有效验证行，与seed无关 | 选择与评分分离的跨行机制检查，不用于训练特征拟合；不宣称用户独立留出 |
| `P_A/P_B rank` | 继承互斥训练池；每cell最多256正+256负，记录不足与有效mask | 训练可见排序效用与对另一训练面板的迁移 |
| `P_A/P_B natural` | 每域固定最多512条自然训练行 | 函数幅度、自然训练目标与概率/校准分析 |

`V_mech` 的源取样规则、面板seed、稳定行ID、真实域ID、任务有效标签和hash在 Stage 0 冻结。源面板按dataset×seed拥有明确身份，不能为使五seed的pair相同而另行统一重抽。相同dataset×seed下的Clean/Gaussian及所有状态复用对应面板与pair；面板不随性能、错误样本集合或Test结果改变。P_A/P_B不能与Validation重叠。

固定面板某cell若缺少任一标签，输出`not_estimable`、真实正负人数与原因，不能擅自补行、重抽、填AUC=0.5或改变六格权重。受影响面板的六格Macro标`not_estimable`，仍保留其他cell；完整Validation人口的准确评价独立继续。真实源完整六格人口的标签可用性属于应继承的协议审计。

### 3.2 有界 pair 抽样

每个dataset×seed×面板cell从真实正样本集合×负样本集合中，以独立诊断RNG均匀无放回抽取`min(32768, P*N)`对；PN不足32768时全部枚举。将pair编号映射为`positive_index = pair_id // N`、`negative_index = pair_id % N`，从整数区间抽样，不先构造PN长的索引数组。固定索引在该seed所有状态以及同seed Clean/Gaussian之间复用；不同seed的源面板和pair可以不同。记录`n_draws=n_unique_pairs`与抽样算法/hash。旧预测器需要的4096对保留源身份与源抽样语义，不改成这套新规则，也不将其叠加成新的独立样本。

全人口准确AUC和3×3计数不由这些pair近似。32,768对用于三状态重叠、路径、分层等难以用二维计数完成的细节；其总体是**对应固定面板的正负配对分布**，不是自动等同于完整验证人口。训练rank面板与自然面板也分别报告。

不将32,768个pair当作32,768个独立训练重复，不提供掩盖五seed变异的pair级显著性。需要描述蒙特卡洛精度时可在固定已保存预测上用第二个预注册抽样seed重新抽pair，比较计算误差；这不是额外训练，也不是研究结论的新证据单位。

### 3.3 分层与选择规则

预设分层包括：参考态 margin 的绝对值分位区间、两端分数变化幅度区间、参考态 W/T/C、阶段、域任务格。分位边界只由对应参考态的 discovery 面板确定，再原样应用到 confirmation 面板。若希望严格跨阶段可比，使用首个预注册锚点边界并同时给连续分布，不每轮重划成“最佳”边界。

主分层表同时写`pair_mass`、层内净纠正率以及`pair_mass×net_rate`。不同层的贡献可以合成为对应面板贡献；不得仅展示纠正率最高的小层。事后挑选的典型纠正pair与破坏pair只作案例，不能作为总体量化证据。

禁止使用域样本量作为困难度变量、分层依据或解释；人口数量仅用于数据身份、配对分母和资源核算。

## 4. 模块 B/C：主效应、条件效应和交互

### 4.1 共享/非共享二因素

令 f(O)、f(S)、f(N)、f(F) 是同 seed、同 step、同人口上的某一个明确指标，分别对应在线、仅共享EMA、仅非共享EMA、完整EMA：

$$
E_S=f(S)-f(O),\quad E_N=f(N)-f(O),
$$

$$
I_{SN}=f(F)-f(S)-f(N)+f(O).
$$

$$
f(F)-f(O)=E_S+E_N+I_{SN}.
$$

同时报告：

- `add_S_online=E_S` 与 `add_N_online=E_N`；
- `delete_S_from_F=f(F)−f(N)` 与 `delete_N_from_F=f(F)−f(S)`；
- `interaction=I_SN`；
- `standalone_gap_to_F=f(F)−f(S/N)`。

删除效应与单独添加效应回答不同问题。某组混合态表现差，可能说明与其余参数共适配，不能自动解释成该组没有收益。

在真实分数、logits、pair margin、AUC、LL上分别计算交互。AUC交互同时含“网络函数交互”和“指标阈值效应”；只有margin层也出现的交互才直接说明分数函数非可加。不能仅凭AUC的交互项就断言网络内部产生了新的表示。

### 4.2 顺序归因不伪装成唯一贡献

顺序 O→S→F 的两段为 `f(S)−f(O)`、`f(F)−f(S)`；顺序 O→N→F 的两段为 `f(N)−f(O)`、`f(F)−f(N)`。两条顺序都报告；差异就是交互背景的影响。

允许把两条顺序的平均作为对称分摊的补充描述，但必须命名`two-order allocation`，不称为唯一因果份额。不默认计算所有参数组的完整Shapley值，也不开展所有组的全组合。

主要组选择依据是五seed的单独效应、删除效应、交互及六格来源的共同结果。该选择用于决定预先有界的细分分支，不使用单seed中途筛选。细分涉及验证数据选择时，后续区间明确标`post_selection_descriptive`；不把探索性区间包装成完全独立确认。

### 4.3 时间窗口

采用真实 optimizer step。已审计系数为 ML rho=0.9828205985452511、KR rho=0.995；每epoch分别20、49次更新。窗口按半衰期的0.5/1/2/4倍取最近整数：

| 数据集 | 半衰期steps | 计划h | 对应epoch数量约值 |
|---|---:|---|---|
| MovieLens | 40 | 20/40/80/160 | 1/2/4/8 |
| KuaiRand | 138.282572986 | 69/138/277/553 | 1.41/2.82/5.65/11.29 |

在步t使用`h_eff=min(h,t)`；相同h_eff只计算一次，保留alias。`t−h_eff=0`时R_0=0，old=0；记录初始化质量rho^t，不将此空旧历史态计为新的有效对照。每行都记录真实rho、h_requested、h_eff、rho^h_eff、origin_step和target_step。

$$
R^{old}=\rho^{h_{eff}}R_{t-h_{eff}},\qquad
R^{recent}=R_t-R^{old}.
$$

对 O、O+R_recent、O+R_old、F 计算与S/N完全同样的二因素主效应、条件效应和交互。组×时间交叉只使用方案中允许的主要组、半衰期窗口和锚点；背景固定在O或F时必须在state_id中写出。

时间窗口回答观测更新组合在当前函数上的作用；不能将“近期”直接命名成高频噪声，也不能将“旧历史”直接命名成有益知识。

## 5. 模块 D：功能解释所需的闭合量

### 5.1 历史预测集成

固定小面板每个真实 optimizer step 的 online预测用于

$$
q^p_t=\rho q^p_{t-1}+(1-\rho)p_{\theta_t},\quad q^p_0=p_{\theta_0},
$$

$$
q^z_t=\rho q^z_{t-1}+(1-\rho)z_{\theta_t},\quad q^z_0=z_{\theta_0}.
$$

主比较是概率集成q^p，logit集成sigmoid(q^z)为预先规定的补充。必须标`exact_all_step_on_panel`；没有完整人口逐step预测，不宣称完整人口的精确历史集成。

每个seed/锚点输出 O、F、q^p、sigmoid(q^z)的AUC/LL；F−q的同刻差；与O比较的纠正、破坏、重叠纠正及各自独有纠正。pair overlap在固定pair索引上计算，明确其面板总体。

若F与q取得接近的AUC，但纠正pair显著不同，不能只凭AUC接近说预测集成解释了相同机制。反之，参数平均与预测平均共同修复相同pair，才形成更具体的历史组合证据。

不把`(AUC_q−AUC_O)/(AUC_F−AUC_O)`称为严格解释比例；当分母绝对值<0.0002时不输出该比例。即使分母较大，该比值也仅是描述性收益对比，允许为负或超过1，不强制截断。

### 5.2 完整切口的四态函数重组

对 [03_CAUSAL_EXPERIMENT_DESIGN.md](03_CAUSAL_EXPERIMENT_DESIGN.md) 确定的完整接口，设z_a为上游状态a的真实输出，H_b为下游状态b：

$$
s_{00}=H_0(z_0),\quad s_{10}=H_0(z_1),\quad
s_{01}=H_1(z_0),\quad s_{11}=H_1(z_1).
$$

逐样本和逐pair margin计算

$$
E_U=s_{10}-s_{00},\quad E_H=s_{01}-s_{00},\quad
I=s_{11}-s_{10}-s_{01}+s_{00},
$$

$$
s_{11}-s_{00}=E_U+E_H+I.
$$

分解分别在logit、概率和margin上完成；四态各自的真实AUC与转移另算，不能将margin贡献百分比直接变成AUC贡献百分比。需要以同一pair证明哪一项或哪种组合使margin跨零，并展示破坏pair的对应机制。

完整端点s00和s11必须复现各自真实网络前向。接口必须包含所有必要旁路/门控输入；若只修补局部路径，记录`cut_complete=false`并将结论限为局部路径效应。主要解释不能依赖一个不完整切口却宣称完整中介。

### 5.3 专家融合的五项功能分解

按 [03](03_CAUSAL_EXPERIMENT_DESIGN.md) 的D2b，在O→F、E35和tF提取同一输入/cell的真实gate、shared/domain/task expert输出。fixed1的实际融合为

$$
u=\sum_j g_j e_j+d+t.
$$

令差分均为F−O，则

$$
\Delta u=c_{content}+c_{routing}+c_{interaction}+c_{domain}+c_{task},
$$

$$
c_{content}=\sum_jg_{O,j}\Delta e_j,\quad
c_{routing}=\sum_j\Delta g_j e_{O,j},\quad
c_{interaction}=\sum_j\Delta g_j\Delta e_j,\quad
c_{domain}=\Delta d,\quad c_{task}=\Delta t.
$$

先验收张量重构，再对五项逐一使用F的tower读取`u_F−c_k`，计算真实删除效应、同pair margin变化与排序损失/新增纠正。根据五seed V_A选择最多一个功能项k；将其与B/C和D2已经选定的重要修正X连接：设`theta_0=F−X, theta_1=F`，对这两个真实状态重新计算同一功能项`c_k^X`，再在F的融合口读取`u_1−c_k^X`，用V_B核对同一批由X恢复的pair是否失去恢复及其净AUC代价。不得拿O→F的原`c_k`直接替代这个新contrast的分量。每锚点最多5个原F背景删除态加1个连接态，共6个额外状态，E35与tF重合时复用。选择后的结果标明选择人口，不借用Test。

张量五项可以精确相加，**五个删除AUC不能相加**：它们都在F背景评价，tower非线性及排序阈值会产生交互。某项同时包含其参数和上游h改变的作用，因此不能将`content`直接等同于“shared expert参数的唯一贡献”。第六个连接态才用于检验“已定位的X修正通过该功能项影响这些pair”；其结果仍是指定背景的条件作用，不是唯一微观中介份额。如果X删除或功能项删除无实质作用，直接报告，不再搜索其他功能项或所有组合。

融合代数验收先用捕获的g/e/d/t转FP64，分别重算`u0_ref64/u1_ref64`；要求`(u1_ref64−u0_ref64)−sum(c_k)`的相对L2残差≤1e−10（分母max(norm(u1_ref64−u0_ref64),1e−12)）。实际源forward的FP32融合还含累加舍入，另记录`fusion_rounding_residual=(u1_actual−u0_actual)−sum(c_k)`的绝对L∞/相对L2；不能把它强制当零或增加一个有物理含义的“第六机制”。实际删除始终在真实u1_actual上进行。端点重构使用`u0_actual+sum(c_k)+fusion_rounding_residual`再送入真实下游，按源FP32重复前向容差核对logit/概率。若忽略舍入残差会影响关键pair符号，明确数值敏感范围，不以张量norm小自动通过。

### 5.4 路径积分

仅在划定的重要contrast上计算

$$
\Delta m=\int_0^1\nabla_\theta m(\theta_0+\alpha R)^\top R\,d\alpha.
$$

源训练图包含gate输入detach；真实参数替换会同时改变gate输入。路径导数必须使用独立scratch上的`full_functional_graph=True`，保持前向值相同，训练原图不变。用源detach导数算出的“积分”不能作为函数闭合证据。[原接口实现](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py)

实际alpha端点/中间前向与8→16→32点有界求积并存。输出每pair真实Δmargin、积分和、绝对残差、相对残差、节点数，以及主要分组积分。组积分是给定路径的分配，不能替代二因素真实干预。

初始数值验收标准：`abs_residual≤1e−5+0.01×abs(actual_delta_margin)`；关键pair若残差足以改变“是否跨零”的解释，则标`sign_unresolved_by_integral`，不由积分强行归因。达到32点仍不闭合时保留真实前向/patch结论并取消该pair的积分解释，不无限增加节点或启动新训练。

### 5.5 训练来源与同一pair的损害/恢复重叠

在同一固定pair上定义

$$
u_{ij}=\phi(m_{\theta_t})-\phi(m_{\theta_{t-h}}),\qquad
v_{ij}=\phi(m_{\theta_t+R^{recent}})-\phi(m_{\theta_t}).
$$

记录：

$$
recovered=E_{pairs}\min((-u)_+,v_+),
$$

$$
lost\_recent\_benefit=E_{pairs}\min(u_+,(-v)_+).
$$

还需报告v改善中未与近期损害重合的部分、v损害中未与近期收益重合的部分，以及所有九种u/v符号组合。这些是面板上的真实排序变化，不将其强行推广成完整人口唯一中介率。

同窗固定自然训练面板的未缩放BCE、实际训练目标近似与验证AUC/LL并列。Gaussian条件区分clean probe与固定噪声bank下的增强目标。若近期训练目标下降但验证排序损害，且R_recent恢复相同pair，可支持所测窗口的训练目标/验证排序不一致；没有任务来源干预就不命名负迁移，没有相应时间证据就不命名噪声。

## 6. H1/H2/H3桥接：真实动作价值、受限oracle和无泄漏预测

### 6.1 两个预注册七动作菜单

| 菜单 | 动作 |
|---|---|
| `rho7` | F、global fast、global slow、S fast、S slow、N fast、N slow |
| `mechanism7` | F、F−R_all_recent、F−R_all_old、F−R_S_recent、F−R_S_old、F−R_N_recent、F−R_N_old |

rho动作在e=5,10,…,50结束的一epoch pulse中从起点F初始化，fast=rho²、slow=sqrt(rho)；机制菜单在同终点使用半衰期窗口的观测修正。两类都是真实读出，不反馈主训练。菜单state、时点、背景、系数提前冻结，不因验证结果更换动作以制造可用空间。

两菜单动作数相同便于比较；它们的时域作用与位移幅度并不自动匹配。因此只能说“这两个已定义菜单的可用空间不同”，不能单凭菜单差异断言所有rho递推无法达到同一函数。补充报告参数位移范数、函数幅度及margin响应，但几何覆盖本身也不替代动作评价。

**H2作为方向B主要理由前的有界匹配检查。** 令`H_int=round(log(.5)/log(rho))`，仅预采E35/E50的额外H_int步fast/slow影子，从`t−H_int`的F出发，连续H_int步以rho²/sqrt(rho)更新；不必在十个窗口新增第二套H菜单。保存两个global向量，S/N候选通过同一坐标mask精确派生。每条件五seed后按 [08_ANALYSIS_AND_DECISION.md](08_ANALYSIS_AND_DECISION.md) §4的触发与选择规则，最多选择两个共同scope G∈{all,S,N}；每个G内仅用当前P_A分别选择rho fast/slow与机制recent/old候选，不能把S方向和N方向直接配平后称为相同执行对象。

令两候选相对F的位移为delta_rho、delta_mech，令`r=min(norm2(delta_rho),norm2(delta_mech))`，真实替换状态为

$$
\theta^{match}_{rho}=F+r\frac{\delta_{rho}}{\|\delta_{rho}\|_2},\qquad
\theta^{match}_{mech}=F+r\frac{\delta_{mech}}{\|\delta_{mech}\|_2}.
$$

每G仅两态、每anchor最多四态，V_B评分；同时保留原动作的真实效果。任一候选为F/零位移时r=0，标`matched_assay_uninformative`，不能当作H2阴性；若公共幅度造成的实际函数变化低于已审计数值分辨率，同样标明信息不足，不再搜索半径。匹配的是递推运行长度、共同参数scope和L2位移大小，历史权重结构仍有意不同；不能声称已匹配所有非线性功能尺度。

只有匹配后机制动作对F本身仍有有意义的正效用，且比匹配rho更好，才能把此检查用于支持所测作用方向的差别。两者都损害F、机制动作只是损害较少，不足以成为方向B的主要理由。差异消失时，将原差异归为时域/幅度/有限菜单混合，是否优先A还需已有训练信号及rho可用空间支持，不自动改判A。该检查是晚期同刻诊断，不等于H步rho的整条未来策略性能已获验证。

### 6.2 至少四种价值，不能只报oracle

对每个菜单、同一窗口和人口，用U(a)=AUC(a)−AUC(F)表示真实动作收益：

1. `empirical_oracle=max_a U(a)`：使用评分人口的结果挑选最佳动作，F在菜单内，故非负是构造性质。它是有限样本、有限菜单的事后参照，不是可部署性能或对总体最优策略的严格上界。
2. `validation_split_choice`：仅用V_A选择动作，在V_B评价；可对调后平均，但预先固定，不能只留有利方向。它衡量使用验证标签选择后能否迁移，不称训练可用策略。
3. `train_direct_choice`：用当前P_A选择动作，在P_B、V_A、V_B和允许的完整V锚点评价。它是当前训练信息可操作性的直接诊断。
4. `future_predictor_choice`：仅用当前/过去可见特征选择下一个预注册窗口动作，评价真实未来动作，不用未来结果选动作。

每类同时与F、同fold早期训练信息确定的最佳固定动作、carry/current-direct对照比较。报告动作频率、熵、完全退化为同一动作的fold数；若动态方法与统一固定动作完全一致，明确给出动态增量为零。

`regret=oracle_utility−chosen_utility`只对同一菜单/同一评分人口定义。预测MAE变小但真实动作价值未变，不算额外可操作收益。符号准确率只在预先定义的效应量分层中报告，并给覆盖率；不能事后删除近零动作来抬高准确率。

### 6.3 LOSO加时间阻断

每个dataset×objective单独执行五fold leave-one-seed-out。留出的seed完全不进入模型拟合、标准化、固定动作选择或阈值确定：

| 用途 | origin epoch | target epoch |
|---|---|---|
| 训练（其他四seed） | 5/10/15 | 10/15/20 |
| gap，不用于训练或评分 | 20 | 25 |
| 评价（留出seed） | 25/30/35/40/45 | 30/35/40/45/50 |

全部训练target在epoch20已经可得，首次评价origin为25。不得随机拆分同一seed的不同时点；不得把未来目标、Test、后期验证结果放进标准化或特征选择。

训练标签限P_A的实际未来动作收益；P_B和V_A/V_B只评分。精确feature列、动作聚合维度及映射以 [08_ANALYSIS_AND_DECISION.md](08_ANALYSIS_AND_DECISION.md) §4为唯一实现定义；本文件不授权额外feature。输入只能是规定的当前或过去训练可见量，禁止因为本次观察到好结果而扩展控制器。训练阶段变量按该定义保留，并与只用阶段/固定偏好的参照比较。

每个fold只有4seed×3训练origin=12个时序训练单元。使用预定ridge α=1，训练fold内标准化；常量列置零并记录；不进行新的HPO。所有feature字段、source timestamp、是否用了训练标签、是否用了验证标签都写入`FEATURE_MANIFEST.json`。源定义见[11/09分析报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)；若因新的菜单必须改变聚合维度，必须显式标`adapted_definition`，不冒用原模型的原始名字。

选择动作默认取预测收益严格最大的动作，预测最大值≤0时选F；数值并列按固定action_id且F优先。不得使用Test或目标窗口的真实效用调节动作阈值。

### 6.4 H1/H2/H3的统计支持范围

- **H1**：历史proxy对本次真实组/时间动作收益无稳定held-out价值，而对应真实机制的训练可见简单信号有价值；或者发现旧指标主导预测的是阶段/幅度而非净纠正。只用相关系数不足以接受动作结论。
- **H2**：真实机制菜单有可重复额外作用，训练可见信号能够取得相对简单固定动作的价值，而所测rho菜单缺少相应效果；结论限定已测执行器。作为方向B的主要理由时，还须经过§6.1的共同scope/H步/幅度匹配，且匹配机制动作对F仍有正效用。仅oracle强、实际训练信号无价值，或只胜过退化rho动作时，不接受“已找到可用信息但rho不会利用”的完整主张。
- **H3**：完整归因已解释固定F的主要作用，已测合理菜单的经验空间和可迁移选择价值均较小，支持当前条件内自适应投入回报有限。不是“所有自适应历史机制不可能改进”的证明。

三种解释可因组/阶段而并存。“oracle空间大但不能预测”与“oracle空间本来小”必须分开。未达到显著性不自动等于没有空间；以真实效应、配对区间与研究决策量级共同判断。

## 7. 五seed配对统计与报告规则

对每个预注册contrast，先计算每seed配对差d_s。报告五个原值、均值、样本SD、正/负/零seed数和95% t区间：

$$
\bar d=\frac15\sum_s d_s,\quad
SD=\sqrt{\frac14\sum_s(d_s-\bar d)^2},
$$

$$
CI_{95\%}=\bar d\pm2.7764451052\frac{SD}{\sqrt5}.
$$

区间是有限seed不确定性的描述，不作为“所有seed正向”或“区间下界必须大于0”的自动门。用户的平均收益规则优先。若资源故障尚缺seed，标`INCOMPLETE`，不以不完整结果否决任何方案。

阶段汇总先在每seed内平均再跨seed统计：early=E5/E10，middle=E20，late=E35/E50；逐锚点表必须保留。F-selected另外成表，不加入late平均。不能把五seed×五锚点当作25个独立样本。两个训练目标的交互采用同seed、同锚点的difference-in-differences，再以五seed为单位；不能混用各自selected点与同刻机制差。

本轮预设`epsilon_mechanism=0.0002 AUC`，只用于分层展示、细分优先级和“量级很小”的研究讨论，**不是数值计算容差、显著性阈值，也不是算法+0.003晋级门**。小于该值的真实效应仍保留，不四舍五入成0。

六格、Macro、LL分别报告。不以挑最有利cell、epoch、window或seed方式形成主结论。探索性细分和大量相关性不逐项贴未经处理的显著性星号；主报告以预注册contrast的效应和证据链为中心。可用已有可信结论作为前提，不为追求更多p值重跑已完成实验。

## 8. 数学与实现数值验收

### 8.1 参数状态及浮点递推

实数数学中R_t=rho(R_{t−1}−Δθ_t)。源训练EMA使用FP32，因此实际保存值满足

$$
R_t=\rho(R_{t-1}-\Delta\theta_t)+\eta_t,
$$

其中η_t是浮点运算舍入残差。`R_recent=R_t−rho^h R_{t−h}`与`R_old=rho^hR_{t−h}`的加法分解对观测R恒等成立；但将它解释成加权历史更新时需要量化累计舍入项。

active worker可维护一份RAM-only FP64理想R递推参照（或等价累计η参照），不改变实际FP32训练；在锚点输出观测R与参照的绝对/相对差。该参照及所有参数导出向量禁止落盘。若其差足以改变关键pair的margin符号，报告受影响pair并用真实状态解释，不把理论精确式当作数值逐位相等。

### 8.2 必须检查的闭合关系

| 检查 | 要求 | 失败处理 |
|---|---|---|
| group partition | 每个参数坐标恰好属于一个组；空组显式记录 | 修复registry，不能运行错误组干预 |
| `R_recent+R_old=R` | FP64算术相对L2残差≤1e−10（分母max(norm(R),1e−12)） | 检查clock、h、历史state与dtype |
| F完整端点 | 优先直接使用原F向量；重建态与原F逐参数/预测差记录 | 不用重建误差制造干预效应 |
| B/C metric identity | 总差−两主效应−交互的绝对误差≤1e−12 | 修复引用的state/metric身份 |
| rank 3×3 | 总计=PN、负计数=0、ΔAUC误差≤1e−12 | 禁止出表直到修复 |
| patch endpoints | s00/s11与对应真实前向一致，容差由源FP32重复前向上界冻结 | 不完整/错误接口不能形成主机制 |
| patch additive identity | 同一float64保存分数的代数残差≤1e−10 | 修复行索引/任务mask/公式 |
| fusion five-term identity | 从捕获g/e/d/t用FP64重算的融合差与五项相对L2残差≤1e−10；另记源FP32融合舍入残差，带残差重构下游端点 | 修复gate/expert索引、fixed1系数和上游状态；不得用五项AUC和代替该检查 |
| prediction EMA | 初始化、每step次数、rho与参数EMA相同；短人工序列与展开和一致 | 修复时间权重，不称精确集成 |
| derivative semantics | scratch全函数前向与原前向相同；有限差分检查导数 | 保留真实patch，暂停错误积分解释 |
| path closure | 见§5.4的残差与跨零检查 | 有界加点；不通过则标积分未解 |
| training isolation | 观察前后主参数、optimizer、RNG、buffer、数据游标身份保持 | 恢复同seed有效状态，不混入损坏轨迹 |

不存在一个通用于所有函数效应的“参数小于1e−6就没影响”规则。FP32重建误差、积分近似误差、pair采样误差、五seed变异分别记录，不能用同一个epsilon混合处理。

## 9. 必须实现的数据表与外键

所有CSV/Parquet行均含公共身份：`experiment_id, attempt_id, protocol_hash, code_sha, dataset, objective, seed, trajectory_id, state_clock, split, population_id, panel_hash`；不适用字段显式为空，不以字符串“0”代替缺失。每个state有稳定`state_id`，所有对比有`contrast_id`。输出AUC比例单位、LL未缩放；float64标量至少保存17位有效数字。

| 表/文件 | 主键或粒度 | 必要字段 |
|---|---|---|
| `STATE_MANIFEST.jsonl` | state_id | state_kind、step/epoch、parent_state_id、background、group_mask_hash、h/rho、state_digest、buffers_policy、creation_reason；无参数向量 |
| `EXPECTED_OUTPUTS.csv` | expected_key | 所需condition/seed/anchor/module/contrast、完成状态、缺失理由 |
| `ENDPOINTS.csv` | seed×readout×endpoint×split×cell | own_selected_epoch/step、selection_owner、auc、logloss、n_pos/n_neg、Macro/Traffic/Pooled、identity_kind |
| `INTERVENTION_METRICS.csv` | state×population×cell | auc/LL、n、score/logit摘要、actual_step、valid_mask_hash |
| `PAIR_TRANSITIONS.csv` | contrast×population×cell×a×b | int64 count、PN、fraction、weighted_auc_contribution、sum_check、auc_closure_error |
| `PAIR_STRATA.csv` | contrast×panel×cell×stratum | strata_rule_hash、n_draws/n_unique、pair_mass、corrected/broken/tie量、net_contribution、margin摘要 |
| `FACTORIAL_EFFECTS.csv` | seed×anchor×factorial×metric×cell | 四个state_id、两个主效应、两个删除效应、interaction、closure_error、背景与归因顺序 |
| `TEMPORAL_EFFECTS.csv` | seed×anchor×group/background×h×cell | h_requested/h_eff、origin_step、rho_power、initial_mass、recent/old效应及交互、FP舍入残差 |
| `ENSEMBLE_EFFECTS.csv` | seed×anchor×panel×ensemble_kind×cell | exact_all_step标记、step_count、initialization_hash、O/F/q指标、F−q差、纠正重叠/独有质量 |
| `FUNCTIONAL_PATCH.csv` | seed×anchor×contrast×cut×population×cell | cut_complete、四态ID、端点误差、上游/下游/交互score与margin摘要、四态AUC及真实flip |
| `FUSION_COMPONENTS.csv` | seed×E35/tF×contrast×component×population×cell | component_id/content-routing-interaction-domain-task、真实端点状态ID、FP64代数残差、源fusion_rounding_residual的L∞/相对L2、下游重构误差、F删除state_id与真实AUC/LL/margin/flip、top1选择人口、linked_X/state_0/state_1、重算c_k^X的连接删除state_id、与X原始纠正pair的重叠、V_A选择/V_B评分身份；不存无界激活张量 |
| `PATH_CLOSURE.csv` | seed×anchor×contrast×固定解释pair | pair_id、实际Δmargin、积分值、分组积分、残差、节点数、sign_unresolved；不存梯度/参数向量 |
| `SOURCE_OVERLAP.csv` | seed×anchor×h×panel×cell | online历史损害/收益、recent修正收益/损害、recovered、lost_recent_benefit、u/v联合类别、训练目标与V指标变化 |
| `ACTION_MENU.csv` | seed×window×menu×action×population×cell | origin/targetstep、action状态ID、实际U、action_horizon、位移/函数幅度标量、menu_hash |
| `H2_MATCHED_ASSAY.csv` | seed×E35/E50×共同scope×population | 触发原因、两原候选state_id、选择人口P_A、H步起止身份、两原范数、公共半径、匹配state_id、对F及两态差的AUC/LL/margin、V_B评分、是否零半径/数值无信息；不存位移向量 |
| `FEATURE_MANIFEST.json` | feature_name | 原定义/适配定义、source_clock、label_access、group聚合方式、source_code_path |
| `BRIDGE_FEATURES.csv` | seed×origin×menu×action | 08§4的固定动作级特征及各自timestamp，不含未来target特征 |
| `BRIDGE_FOLDS.json` | fold_id | holdout_seed、训练origin/target、评价origin/target、scaler拟合行hash、feature/menu hash |
| `BRIDGE_PREDICTIONS.csv` | fold×origin/target×menu×predictor×action | predicted_utility、真实P_A/P_B/V效用、selected_action、static_action、oracle_action、label_access |
| `BRIDGE_VALUE.csv` | seed×menu×selector×population | 先在该seed评价窗内汇总的MAE、真实选择收益、对F/静态增量、regret、动作熵/退化率 |
| `PAIRED_SEED.csv` | contrast×seed×cell/metric | 两端原值、paired_delta、identity_kind、是否探索性选择后 |
| `PAIRED_AGGREGATE.csv` | contrast×cell/metric | n、五个seed覆盖、mean/SD/CI、正负零数、stage聚合规则、缺失状态 |
| `NUMERICAL_QA.csv` | check_id×state/contrast | expected、observed、tolerance、pass、影响范围与修复状态 |

预测数组仅保存获准的有界面板/必要完整人口读出，采用行ID外键复用；不要为每个pair复制两个样本分数，更不要把每step全部面板预测或参数导出无界写盘。标量表可压缩；总观察输出遵守 [06_SCHEDULER_AND_RESOURCES.md](06_SCHEDULER_AND_RESOURCES.md) 的2GiB预算。需要释放时先删除可由已保存预测准确重建的冗余中间表，不删除正式五seed证据。

## 10. 报告生成与本文件验收

自动报告必须从上述冻结表生成，禁止手填主结论数值。每个主机制条目包含：同seed同step的真实参数/时间干预 → 完整函数接口效应 → 同一pair margin与排序转移 → AUC净贡献 → 实际训练窗口行为 → 下一步决策。关键数字应带表名、contrast_id、五seed覆盖与适用人口。

完成条件：

1. 正式四条件×五seed的O/F端点完整，主干预五seed覆盖完整；任何缺失明确显示，不悄悄取平均。
2. 核心B/C完整验证人口AUC与准确转移闭合；两个层次的指标身份没有混用。
3. 至少对主要修正完成D的真实接口重组或等价完整函数干预，并连接到同一pair的纠正；只给参数幅度、相关系数或表示距离不算完成。
4. H桥接区分oracle空间、训练信号可用性与实际动作价值；结果能支持A/B/C/D方向的有条件选择，不需要制造某个方向必胜。
5. Test只在全局机制与分析规则冻结后读取；Test不改变分组、窗口、指标、假设、门槛或菜单。最终读出性能与机制证据分表。
6. 本文件中的所有预期表在新实验完成前保持`not_run`或`planned`，不得用已有报告数字冒充新因果干预结果。

