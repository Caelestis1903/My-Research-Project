# 方向 B：共享更新、任务身份与历史读出的机制研究

## 1. 问题、边界和执行顺序

**本轮判断：真实域任务贡献在共享坐标上的更新，是否造成跨格排序损害；训练可见的任务身份是否提供超出普通更新分割和尺度的可用信息；固定EMA对这些作用的改变是否包含历史背景和函数组合的效应。**

固定EMA的整体排序收益、共享S尤其EMBED的主要贡献、recent/old互补均直接继承[研究上下文](01_RESEARCH_CONTEXT.md)。不重复四条件EMA有效性、共享删除、recent/old删除或历史窗口搜索。旧current-preconditioned单步与A_MATCH六轮反馈的差异也已成立；本轮改变的是源贡献，分别报告单步、一次脉冲传播、持续完整训练。

B0/B1核心必做，不受A_GATE影响。本版可操作scope固定为**共享EMBED**；其他共享参数及非共享参数用于定位和必要对照。若仅其他组阳性，保留该组机制，不据此触发未获对应身份控制支持的EMBED算法。

| 阶段 | 目的 | 范围 | 输出 |
|---|---|---|---|
| B0 | 真实Adam下的六格贡献效用和任务身份 | 10条Clean、40锚点、三个scope | 六源×六接收格矩阵、同范数、两套伪分组 |
| B1 | 区分位移缩小与历史背景 | B0相同真实状态，额外只读前向 | online/EMA、同位移跨背景、真伪比较 |
| B_PULSE | 一次源贡献干预的后续传播 | E20/E35真/伪脉冲，之后正常联合更新 | 未来效用、有限窗口预测平均、身份增量 |
| B_SEARCH | 有证据才检验持续规则 | B_ADAPT/B_STATIC/B_UNIFORM×η=.25/.5 | 60个完整五种子配置槽位 |

执行ID与[实验矩阵](05_EXPERIMENT_MATRIX.md)一致：B_REPLAY恢复轨迹，B0_GRAD分解梯度，B1_STEP执行52种虚拟Adam，B1_NORM执行同幅读出，B1_WINDOW做父轨迹有限窗口，B2_PULSE执行条件触发短段。本文B0/B1为这些核心步骤的研究分组，B_PULSE即B2_PULSE。

B_PULSE只估计一次更新的传播；B_SEARCH才回答持续算法的独立增量。阶段门以[09_DECISION_RULES.md](09_DECISION_RULES.md)为准。

## 2. 训练状态、面板和锚点

### 2.1 原生配置

两个数据集均为自然80/10/10划分、train-only预处理、M3oE fixed1、50epochs、batch40960、FP32；种子固定2022、2023、2024、2025、2026。Adam使用foreach=False、fused=False、amsgrad=False。

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| train / validation / test行数 | 800167 / 100021 / 100021 | 2006132 / 250767 / 250767 |
| 每epoch更新数 | 20 | 49 |
| 末batch实际行数 | 21927 | 40052 |
| lr / β1 / β2 / ε | .01 / .9 / .999 / 1e−8 | 同左 |
| coupled WD | 1e−4 | 1e−5 |
| 固定EMA每step的ρ | .9828205985452511 | .995 |

EMA初值等于online初值，每次有效Adam后更新一次，仅供读出，不反馈在线梯度。禁止改采样比例、batch大小、主干、原生目标或checkpoint规则。

固定锚点为**E5/E20/E35/E50末个真实batch更新前**，合计2数据×5seeds×4=40。记录epoch、global_step、batch_offset、实际B、row IDs/hash。末batch仍使用真实B；后续窗口按global_step定位。

### 2.2 恢复与复用

11/10、11/11报告明确模型/Adam/EMA等在报告和Test完成后释放；近期selected权重也不能恢复原Adam。默认重建10条Clean×50epochs，用于本轮新增机制干预，不增加统计seed数。

若A执行时有完全同身份Clean轨迹，挂B observer共用；否则按固定源重放。复用须现场核验真实模型、每参数Adam、EMA、RNG、loader及步号，不接受仅模型hash或历史存活收据。已有完整性能数字继续复用，不为再次确认EMA有效性重训。

### 2.3 固定面板与隔离

- P_A：训练行池，用于选源q、NO_ACTION和运行期动作。
- P_B：与P_A互不交的训练行池，仅确认另一训练人口上的效用。
- V_A/V_B：固定自然验证行按来源hash分侧；机制动作和静态源冻结均不读其标签，V_B作为确认侧。完整Validation继续用于正式训练选点。
- 训练ranking探针每格最多256正/256负、4096固定pairs；自然LL探针每域512行。
- 固定自然V_mech每域最多4096行，历史ML为[4096,4096,4096]、KR为[2692,4096,1843]。prepare时重新核验支持、行ID、manifest/hash，不用历史人数代替现场检查。

所有方法在同一population使用相同样本、pair和ties定义。P_A/P_B只观察，不改变自然loader。拟合或静态源冻结采用LOSO：其他四seed的E5/E20训练P_A作来源，留出seed的E35/E50评价；轮转五折。不能用V_B挑q后再在同一V_B宣称独立效用，不能只保留阳性seed/锚点。

## 3. 实际参数与六格目标

### 3.1 参数映射

有序格为
$$
q=(d,t)\in[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)].
$$
tower/gate索引为 $3t+d$，不要与domain-major表格顺序混淆。

| 组 | 实际名称规则 | ML坐标数 | KR坐标数 |
|---|---|---:|---:|
| S | shared_weight/shared_bias；skip_conn.*、star_mlp.*、expert.*、共享embedding | 483152 | 661920 |
| Dd | slot_weight.d、slot_bias.d、domain_expert.d.*、域embedding第d行 | 每组74960 | 每组295632 |
| Tt | task_expert.t.* | 每组16576 | 每组16576 |
| DTdt | gate.(3t+d).*、tower.(3t+d).* | 每组5381 | 每组5381 |
| 全部 | S+3D+2T+6DT | 773470 | 1614254 |

域embedding名称为 **embedding.embed_dict.{features.columns.index('domain_indicator')}.weight**，其0/1/2行各16维属于D0/D1/D2。不能按embedding前缀将它列为共享。

三个互斥scope：

- **EMBED**：S中的embedding，主机制和本版算法作用组。
- **S_DENSE**：S去掉EMBED；细分STAR_SKIP/EXPERT只记几何统计，与A的S_dense先核验坐标再桥接。
- **N**：D_ALL∪T_ALL∪DT_ALL。

三组覆盖可训练坐标；记录参数顺序、shape、dtype、flat ranges及group hash。分组来源是aug_history_20261105/group_registry.py与ema_causal_20261110/interventions.py，固定来源见[12_SOURCE_AND_REUSE_REGISTER.md](12_SOURCE_AND_REUSE_REGISTER.md)。main可直接复用completion_weight_attribution_20261117/core.py的原生clone-Adam模式。

### 3.2 自然权重分解

当次真实batch有B行：
$$
L_0=\frac1{2B}\sum_i\sum_{t=0}^1\ell_{\rm BCE}(p_{it},y_{it}),\qquad
L_q=\frac1{2B}\sum_{i:d_i=d}\ell_{\rm BCE}(p_{it},y_{it}).
$$
故 $\sum_qL_q=L_0$、$g_q=\nabla L_q$、$\sum_qg_q=g$。

真实源码使用概率BCE的real.mean(1).mean()；六格为sum/(2*len(y))。不能改成每格mean后等权，不能放大小格权重。

在同一次原生forward图上提取六个g_q，允许unused并保留图；live g仍由原L0.backward生成，不以另一浮点累加顺序的Σg_q替代。加和只作容差一致性检查。缺失格data贡献为0，None语义仍由原生图决定。

模型gate输入有emb.detach()。历史函数诊断的probe_input_full_graph=True会解除该detach；本轮真实训练梯度分解不得使用这个开关。

## 4. 冻状态原生Adam干预

### 4.1 冻结内容与精确顺序

更新前RAM对象：
$$
X=(\theta,\{m_p,v_p,\tau_p\}_p,\text{param groups},
 F_{\rm prev},\text{buffers},\text{RNG},\text{loader},\text{batch identity}).
$$
读取每参数私有step $\tau_p$，不能用一个global_step替代。每个干预从同一个X重置，不串接反事实。

对原生非None参数p、反事实data gradient h：
$$
z_p=h_p+\lambda_{{\rm wd},p}\theta_p,\quad
m'_p=\beta_1m_p+(1-\beta_1)z_p,\quad
v'_p=\beta_2v_p+(1-\beta_2)z_p^2,\quad
\tau'_p=\tau_p+1,
$$
$$
\theta'_p=\theta_p-\frac{{\rm lr}_p}{1-\beta_1^{\tau'_p}}
 \frac{m'_p}{\sqrt{v'_p}/\sqrt{1-\beta_2^{\tau'_p}}+\epsilon_p}.
$$
公式用于说明，实现调用深拷贝模型、原生optimizer构造器、load_state_dict及optimizer.step，保持实际FP32运算顺序。

**None和零张量：** native grad=None时所有干预保持None，参数/m/v/step/WD均跳过；native零张量仍执行历史和WD。q的None只在向量和中解释为0，不能新建原生None参数的.grad。记录native活动mask、每参数pre/post step及tensor_zero。

JOINT虚拟分支必须与随后live一步的模型/optimizer hash精确相同；observer不能改变live参数、optimizer、EMA、RNG、loader。不能将U(g)−U(g_q)当作U(g−g_q)，新的m和v均须重算。

### 4.2 菜单

令 $P_G$ 为组投影，g为native联合data gradient：
$$
\begin{array}{ll}
{\rm JOINT}:&h=g,\\
{\rm ZERO\_DATA}(G):&h=g-P_Gg,\\
{\rm ONLY}(q,G):&h=g-P_G(g-g_q),\\
{\rm LOO}(q,G):&h=g-P_Gg_q.
\end{array}
$$
G外保留native g，WD由Adam加一次。ZERO_DATA保留旧m/v和WD，是ONLY的历史底座。ONLY回答相对当前G data为0时q的作用，LOO回答其他源已存在时q的边际作用；非线性下不要求两者同号、六LOO可加。

## 5. B0：排序效用、尺度和任务身份

### 5.1 符号与实测判别

沿用[02](02_THEORY_AND_HYPOTHESES.md)：
$$
U_{q\to c}^{G,O}=Q_c(\theta^J)-Q_c(\theta^{-q,G}),\qquad
D_{q\to c}^{G,O}=-U_{q\to c}^{G,O}.
$$
U表示加入q的条件边际作用，负值表示降低接收格排序；D表示去除收益，选动作最大化D。EMA两后态同样定义 $U^{G,F},D^{G,F}$。

$$
U_{q\to c}^{G,\rm only}
 =Q_c(\theta^{\rm only(q,G)})-Q_c(\theta^{\rm zero(G)}).
$$
概率AUC使用
$$
Q_c(\theta)=\frac1{n_c^+n_c^-}\sum_{i:y_i=1}\sum_{j:y_j=0}
[\mathbf1(p_i>p_j)+\tfrac12\mathbf1(p_i=p_j)].
$$
固定pair正确率与全自然面板AUC分列；raw logit margin只辅助，不替换历史概率AUC及ties。

每组输出6源×6接收格、Macro、self、其他格均值、最差格及LL。可从同矩阵汇总同域异任务、异域同任务、异域异任务，不增加实验。

训练P_A识别后，独立面板上源的self U为正、其他格存在负U且LOO恢复其排序，才标记跨格有害更新。ONLY补充不同背景；分歧时注明条件边际/非线性依赖。负余弦、方差大或范数大均不直接等于负迁移。若更新降低所有格却无self收益，记录一般有害更新或训练目标与排序不一致。

必须记录纠正/破坏/平局转移并闭合净ΔAUC。

### 5.2 同范数控制

在G内相对同一θ_pre取实际位移d_J、d_L，令
$$
r=\min(\|d_J\|_2,\|d_L\|_2),\quad
\tilde d_J=r\,d_J/\|d_J\|_2,\quad
\tilde d_L=r\,d_L/\|d_L\|_2.
$$
两个共同半径状态均重新前向，G外保留JOINT后坐标。r=0记NOT_INFORMATIVE。原始和匹配结果都报告，不只挑有利半径。真/伪比较必要时也用共同min半径。

这些是只读状态前向，不改live optimizer，也不计新训练轨迹。

### 5.3 两套身份无关分组

将同batch的2B个loss entries(row_id,t,y)按y=0/1分层随机重分为6个伪bin。每bin正负entry数匹配对应真实q的 $n_{q0},n_{q1}$，成员混合真实域/任务；每entry恰属一个bin。因此保留loss质量、class support和六bin容量，六伪梯度仍和回g。

不修改输入、标签、loader或live训练；不能仅置换六个真实组的名字。两固定partition使用独立hash RNG，key含research_id、训练seed、global_step、partition_id，不消耗live RNG。

每partition独立六选一，与真六选一预算相同，不在12伪bin中挑一个最优冒充等预算。伪LOO仅作用EMBED；partition0供脉冲，partition1为B0重复控制。最终算法增量从不以退化伪组作主参照。

### 5.4 有限矩阵

| 项目 | 每锚点virtual Adam | 40锚点 |
|---|---:|---:|
| JOINT | 1 | 40 |
| 3组×ZERO_DATA | 3 | 120 |
| 3组×6 ONLY | 18 | 720 |
| 3组×6 LOO | 18 | 720 |
| 2伪partition×6 EMBED LOO | 12 | 480 |
| **合计** | **52** | **2080** |

每锚点真6+伪12=18额外autograd sweeps，总720。范数、EMA跨背景及函数读出只增加前向，单列readout数和时间，不追加optimizer数。

## 6. B1：历史效应与机械缩放

### 6.1 三种不同的历史干预

$$
F_t=\rho F_{t-1}+(1-\rho)\theta_t,\ F_0=\theta_0,\qquad
R_t=F_t-\theta_t=-\sum_{k=1}^t\rho^{t-k+1}\Delta\theta_k.
$$
实际FP32用真实F，理想式只审计舍入残差。

- k步改变参数e，未来**位移**保持原记录值：其后online差恒e，$\delta F_t=(1-\rho^{t-k+1})e$。
- 仅替换历史快照θ_k，下一步强制回原online轨迹：$\delta F_t=(1-\rho)\rho^{t-k}e$。
- 改写真实模型/Adam后重算未来梯度：实测δθ_j并核查
  $$
  \delta F_j=\rho\delta F_{j-1}+(1-\rho)\delta\theta_j.
  $$

不能混用后两种权重。已有R_old=ρ^hR_(t−h)、R_recent=R_t−R_old是历史修正拆分，不是任务收益的唯一可加份额。

### 6.2 同位移跨背景

同预态J/LOO差为δ=θ_L−θ_J，令c_ρ=1−ρ：
$$
F_J=\rho F_{\rm prev}+c_\rho\theta_J,\quad F_L=F_J+c_\rho\delta.
$$
测三个去除效应：
$$
D_{\rm online}=Q(\theta_J+\delta)-Q(\theta_J),
$$
$$
D_{\rm small}=Q(\theta_J+c_\rho\delta)-Q(\theta_J),\quad
D_{\rm EMA}=Q(F_J+c_\rho\delta)-Q(F_J).
$$
D_online与D_EMA之差同时包含幅度和背景。去掉机械幅度后的历史交互为
$$
I_{\rm hist}=D_{\rm EMA}-D_{\rm small}.
$$
对去除有害q可获益的格，I_hist<0表示历史背景减弱其边际损害；D_EMA>0表示EMA中仍有可恢复代价。

再将同一δ幅度放在online/F两个背景评价，报告有益当前更新的成本，与伪组作相同半径/人口/阶段对照。真伪相近则接受普通平滑或函数背景解释；不能仅因F变化小就称选择性过滤。

这是受控历史背景交互，不计算未经识别的“q解释EMA总收益百分比”，允许多种机制并存。

## 7. 训练选源、静态源与短段触发

### 7.1 选择规则

在P_A上：
$$
s_q=\tfrac16\sum_cD^{E,O}_{q\to c}(P_A).
$$
六q与NO_ACTION共同竞争，后者分数0；最佳严格>0才选q，否则NO_ACTION；q完全并列按固定domain-major顺序。partition0按相同规则六选一。

只用训练标签。所有NO_ACTION均参与预注册seed/anchor均值，不筛掉零动作。

### 7.2 静态源

留出seed s，用其他四seed E5/E20训练P_A的s_q平均值，加NO_ACTION=0，冻结 $q_{\rm static}^{(-s)}$。记录每数据、每fold的来源/hash。

留出seed E35/E50使用B0已算的六LOO，比较P_A当前规则与该固定源在P_B/V_B的一步效用。无需新增训练。它不等于固定源短段，持续固定源由B_STATIC正式检验；不能在仅真/伪两短段的预算中声称第三短段已运行。

### 7.3 数据集级触发

EMBED完整五seed诊断包含实际排序损害，且训练P_A规则能选出正净收益源时，触发该数据集全部5seed×E20/E35的真/伪脉冲。一旦触发，不按seed或anchor再筛阳性；NO_ACTION逻辑槽位保留。

另一数据集可记NOT_TRIGGERED；两者都触发时最大40个脉冲槽位。S_DENSE的阳性保留机制，不替代EMBED身份控制门。

等待五seed结门时，B0只保留E20/E35所选真源与partition0伪源post，包含model/Adam/F/RNG和batch游标，每锚点至多两份。也可按数据块结门后立即流式跑并释放；不得保留52反事实×40完整快照。

## 8. B_PULSE：一次干预后的真实传播

### 8.1 执行步骤

1. 首步直接复用B1_STEP训练选出的原始EMBED LOO post状态；伪分支用partition0。NO_ACTION与JOINT对应。
2. 首步已经更新本分支真实m/v/step及正常EMA，不能只改θ后沿用JOINT Adam。
3. 后续K−1步全部普通自然联合目标；每步在各分支模型上重算forward、gradient、Adam和EMA，不继续删除q。
4. K=ML20/KR49，包含首步；使用父轨迹同一未来row IDs、batch边界和随机数键。JOINT窗口直接从父轨迹复用。
5. 读出即时、半程和终点online/F的P_B/V_B六格AUC、LL及pair变化。末batch前的K步终点按global_step记，不误写成下一epoch末。
6. 核查实际δF递推，终点将δθ_K与δF_K缩至共同半径放在同一背景比较，区分幅度、方向历史和背景。

作用起点是EMBED，后续真实梯度可改变其他参数，终点必须评价完整模型，不能假定传播始终局限于E。

### 8.2 成本和零动作

| 口径 | ML | KR | 总计 |
|---|---:|---:|---:|
| 真/伪逻辑槽位 | 20 | 20 | 40 |
| 每槽K | 20 | 49 | — |
| 段内逻辑steps | 400 | 980 | **1380** |
| 已计入B1_STEP的首步 | 20 | 20 | **40** |
| 新增后续Adam调用 | 380 | 960 | **1340** |

2080virtual与1380logical不能再重复计首40。NO_ACTION可复用同身份JOINT窗口，记COMPLETE/NO_ACTION并保留零差值；物理调用据实扣除。表为两数据触发且分支均需执行的上界。

该实验估计一次更新传播，正值不直接证明持续controller有效；脉冲消失也不抹去已成立的局部排序效应。

## 9. 有限窗口预测平均：第三机制

已有全轨迹q_logit/q_probability证据保留。B1_WINDOW在全部10条Clean父轨迹的E20/E35窗口必做；B_PULSE触发时再扩展到对应真/伪分支。每个K步窗口做与参数EMA同权重的有限读出：

$$
\Pi_0^p=f(F_{\rm pre}),\quad\Pi_0^z=z(F_{\rm pre}),
$$
$$
\Pi_j^{p,a}=\rho\Pi_{j-1}^{p,a}+(1-\rho)f(\theta_j^a),\qquad
\Pi_j^{z,a}=\rho\Pi_{j-1}^{z,a}+(1-\rho)z(\theta_j^a).
$$
比较终点 $\Pi_K^{p,a}$、$\sigma(\Pi_K^{z,a})$ 和 $f(F_K^a)$。共同参数权重：
$$
F_K^a=\rho^K F_{\rm pre}+(1-\rho)\sum_{j=1}^K\rho^{K-j}\theta_j^a.
$$

这是**以F_pre预测为历史起点的K步平均**，没有更早online预测，不等于从初始化开始的全轨迹prediction EMA。日志固定命名WINDOW_PROB_EMA、WINDOW_LOGIT_EMA，不冒充已有全轨迹q。

只保存P_B/V_B的score bank，不存逐步θ。20个joint父窗口共690组step-panel读出为核心必做；两数据脉冲均触发时joint/真/伪最多2070组（首步可复用B1_STEP已有预测）。零额外Adam；实际panel数、rows、前向和时间独立记账。

- 预测平均保留主要收益与同pair纠正：支持历史函数组合。
- 参数读出独有差异：定位参数重心的非线性作用。
- 真伪在两种读出均相近：缺少任务身份独立信息。
- 身份增量与函数组合同时存在：分别接受，不要求二择一。

该窗口不参与η选择，不扩H菜单，不生成完整算法Test端点。

## 10. B_SEARCH准入

按[09](09_DECISION_RULES.md)须同时满足：

1. EMBED存在实际跨格排序损害。
2. 训练选择的真源在五seed未来短段Validation均值优于JOINT及等预算partition0伪源；B0 E35/E50留出一步效用优于仅训练冻结的固定源。
3. 完整EMA仍有可改进正均值，六格均值不低于JOINT−.001。
4. 面对正常JOINT/F的直接增量成立，不依赖ZERO/ONLY或被劣化伪源。
5. 效应低于.0001时记LOW_ACTION_VALUE，保留机制但不支付60条搜索预算。

不加局部+.003、CI必须全正或5/5全正门；五seed和所有零动作必须完整。一个数据集的有效条件信号可触发候选，正式候选仍在两数据五seed评价。

有真实损害而训练规则无增量，记MECHANISM_WITHOUT_TRAIN_SIGNAL；EMA已消除该动作余量，记FIXED_EMA_ALREADY_SUFFICIENT_FOR_THIS_ACTION。停止本版控制器，不再扩预测器网格；A停止也不取消B核心与报告。

## 11. 唯一持续候选和两个简单替代

### 11.1 共同协议

B_ADAPT/B_STATIC/B_UNIFORM均完整50epochs，E1–50采用各自下述规则；不设延迟启用或阶段切换。η只取.25/.5，不搜索启动epoch、scope、ρ、H、门网络或惩罚项。

只改变EMBED当前data gradient；保留真实Adam历史、WD和固定EMA。顺序为forward → 必要g_q → 原生联合backward → 构造h → Adam一次 → EMA一次。其他scope使用本分支原生g。该方法不称为自适应EMA。

### 11.2 B_ADAPT

每个E1–50首batch，对六q用当前分支真实状态构造
$$
h^{q,\eta}=g-\eta P_Eg_q.
$$
在任何live更新之前，从同一真实prestate分别构造**七个虚拟分支：一个JOINT和六个η候选**；各分支重置模型、Adam、EMA及RNG前态。实际clone Adam后的online候选在P_A与虚拟JOINT比较Macro AUC；六q和NO_ACTION取最大，最佳≤0则NO_ACTION，q并列按固定顺序。选定后回到未改变的live前态，执行恰好一次所选Adam更新和一次EMA；不能先把JOINT推进到live，再追加候选step。必须使用当前η的真实候选效用，不能把η=1的LOO排序直接当作η=.25/.5结果。

不增加多格惩罚、任意门网络或训练保护阈值；六格成本完整记录，正式Validation判断六格保护。selector只看训练P_A，EMA未来效用由B_CORE准入检验，不以单步EMA机械缩小替代selector。

q_e或NO_ACTION维持一epoch，每batch在当前模型重新计算g/g_q应用η；下一epoch首batch重选。当前batch无q数据则g_q=0，保留None/零张量语义。

小型ACTION_TRACE.jsonl记录dataset、seed、η、epoch、q_e/on/off、P_A候选分数、状态hash和理由；不落盘训练张量。

### 11.3 B_STATIC

固定 $q_{\rm static}^{(-s)}$ 只由B_CORE其他四seed的E5/E20训练P_A选取，不按η、Validation或Test改选。E1–50恒定执行
$$
h=g-\eta P_Eg_{q_{\rm static}}.
$$
冻结动作为NO_ACTION则保持普通训练。

STATIC去除动态选源；其持续作用次数可能不同于ADAPT的on/off，因此属于完整简单策略比较，不把差值孤立解释成仅身份一项的因果效应。实际动作次数、质量和成本分列。

### 11.4 B_UNIFORM

先完成同dataset/seed/η的B_ADAPT并冻结trace，再重放其q_e/on/off；本控制使用自己的模型、gradient、m/v。

NO_ACTION时κ=1；否则源格在当前batch的自然loss质量为
$$
m_{q_e,b}=\frac{n_{d(q_e),b}}{2B_b},\qquad
\kappa_b=1-\eta m_{q_e,b},
$$
$$
h=(I-P_E)g+\kappa_bP_Eg.
$$
它保持相同名义data loss质量减少量，取消指定源方向处理；不缩G外data gradient、不缩WD、不改lr。不声称exact norm matching，也不把它当成LR变换，因为Adam/WD/epsilon/history作用仍在。

trace只提供q/on/off标量，不复制候选模型、optimizer或梯度。顺序固定ADAPT→冻结trace→UNIFORM；STATIC不依赖ADAPT性能或Test。

### 11.5 五seed完整矩阵与选择

$$
2\text{ datasets}\times3\text{ arms}\times2\eta
 \times5\text{ seeds}=60
$$
个50epoch配置槽位，共3000epochs、103500原生Adam步骤。正常Clean+F由同身份对照复用并始终列为主参照。

ADAPT首batch每epoch需七个clone Adam（JOINT+六个η候选），20条ADAPT轨迹×50epoch共7000次；JOINT复用已计算的联合梯度，不增加一轮源梯度分解。后续active batch需一个源的EMBED梯度；STATIC active batch也需该梯度，UNIFORM直接缩放native gradient。可仅对E参数求g_q以降低开销，但不改公式；不能用纯Clean速度估60条成本，实测额外autograd、virtual Adam、panel forward、GPU/RAM及wall time。

每arm按每数据五seed完整Validation宏AUC和六格保护选择η，不按单seed选；每seed取完整Validation六格Macro AUC最早严格最大checkpoint。配置全冻结后按统一流程Test，同轨迹online/F不计两份seed。

- ADAPT不胜STATIC：优先固定策略。
- ADAPT/STATIC不胜UNIFORM：优先统一缩放，撤下身份必要性。
- 均不胜正常Clean+F：停止B，不以胜伪组/ZERO/ONLY宣称增量。
- 仅LL改善：单列损失，不授予AUC资格。
- 对发表模型总收益和对Clean+F组件增量分表。

与最优Macro差≤.0001且六格保护状态相同，按UNIFORM→STATIC→ADAPT优先简单者。这是复杂性选择分辨率，不是统计等效证明。

## 12. RAM、复核和交付

### 12.1 有界状态

禁止模型/Adam/gradient/EMA/checkpoint落盘。B0只保留当前prestate、一个clone、最优post；E20/E35等五seed结门时最多40份真/伪选中post，加必要JOINT终点参数与预测，不缓存全部反事实。

这些张量约为GiB量级，但真实峰值还包含optimizer、buffers、allocator及恢复副本，必须实测。有限窗口只需score bank，无逐stepθ历史。

活guardian恢复最后完整原子边界；guardian丢失则同源同seed重放，不能从CSV/selected权重伪造Adam。物理重放与科学n分表；资源规则见[08](08_RESOURCES_AND_SCHEDULER.md)。

### 12.2 最小必要检查

- 真/伪loss及gradient和回原生目标。
- JOINT虚拟/实际参数及optimizer hash精确一致。
- None、tensor_zero、空域、私有step与coupled WD正确。
- group覆盖、域embedding行、tower/gate次序一致。
- observer不污染live状态/RNG/loader，gate detach保留。
- JOINT短段与父轨迹一致，首步不重复计数。
- EMA三类历史干预及FP32残差正确。
- AUC纠正/破坏/ties闭合，固定pairs不漂移。
- η=0与正常Clean逐位一致；trace重放不复制候选状态。

仅检查影响本轮结论的项目，不展开无关大规模工程验证。

### 12.3 必须交付

| 表/文件 | 内容 |
|---|---|
| B_ANCHOR_MANIFEST | 40锚点身份、实际B、group/optimizer活动及hash |
| B0_EFFECTS | 全seed、源、接收格、scope、菜单、原始/匹配幅度AUC/LL |
| B0_IDENTITY_CONTROLS | 真/两伪容量、质量、选择及留出效用 |
| B1_EMA_INTERACTIONS | online/EMA、机械缩放、跨背景、真伪比较 |
| B_PULSE_EFFECTS | 所有触发槽位含NO_ACTION、即时/半程/终点及复用 |
| B_WINDOW_FUNCTION_READOUTS | K步预测平均、参数EMA、pair重叠及起点/权重 |
| B_STATIC_LOSO | 每fold固定源、训练来源、未来一步效用 |
| B_GATE.json | 按数据集的机制/可用性/预算及搜索结论 |
| 若触发搜索 | 完整候选端点、ACTION_TRACE、简单对照配对表及成本 |

统计单位始终seed：先seed内聚合既定锚点和格，再算五seed配对均值、样本SD、t4区间。报告区分经验效应、受控作用、机制解释和持续算法性能。机制可以成立而当前controller停止，本轮无需强制产出新算法。

