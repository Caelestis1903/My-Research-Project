# 四个模块与功能性因果识别设计

状态：**DESIGN_ONLY**。下列对照均为冻结训练轨迹上的只读干预，不是新增训练臂。数学定义见 [02](02_THEORY_AND_HYPOTHESES.md)，矩阵与时点见 [04](04_EXPERIMENT_MATRIX.md)。

## 1. 共同设计与研究分层

Clean 是主要机制发现条件；Gaussian D0 是事先指定的训练目标边界条件。两个数据集分别给结论，再判断共同性，不把四组混成一个总体。每条件必须完成五 seed。固定时点为 E5/E10/E20/E35/E50；另保留各 seed 的 **F 自身 Validation 选点时刻**作为 `matched_at_F_selected`，此时所有干预共用该时刻，不能与双方各自选点的性能差混用。

正式训练照旧满50 epoch。Stage 1、2、3是逻辑分工，物理上交错：采集过程中产生参数包；一个条件的五 seed 参数包齐全后，在仍驻 RAM 时完成分组选择、有限细化和功能干预，再释放。禁止先把参数释放、再让离线脚本重建因果状态。

所有干预共享：在线初始化、样本顺序、Adam m/v/step、Gaussian noise key、输入行身份、固定 buffer、原始数据与 loss 语义。参数替换仅作用于独立 scratch 模型。不能对 hybrid 做微调、BatchNorm 重估或再训练。

## 2. 模块 A：真实排序纠正的归因

### A1. 完整人口的准确核算

每个正式评价状态对完整 Validation 六格计算源 AUC/未缩放 LogLoss。O→F、O→S-only、O→N-only、recent/old 及必要功能干预均生成完整人口的3×3排序转移表：

| 在线状态 → 干预状态 | 错误 | 平局 | 正确 |
|---|---|---|---|
| 错误 | 未纠正 | +0.5 | +1 |
| 平局 | −0.5 | 不变 | +0.5 |
| 正确 | −1 | −0.5 | 保持 |

表内不是比例预期，而是每种转移对 AUC 分子的实际分值。用排序/Fenwick二维计数求九个整数计数，不枚举全人口 pair。计数总和必须等于 `n_positive*n_negative`，加权净变化必须与源 AUC 差闭合。macro 为六格等权，各格分母不同，不可把所有 pair 先合并再算“macro”。

### A2. 固定面板上的细粒度解释

复用旧协议的 P_A/P_B 训练池与 V_mech 自然验证面板身份，保留 KuaiRand 不足4096行域的有效 mask。细粒度 pair 从每格面板内真实正例×负例均匀、用独立 hash RNG 预先抽取最多32768对；不足时全部枚举。旧4096训练 pairs 另保留用于历史指标对齐。它们不是新的独立样本 seed。

在所有状态上使用相同 pair IDs，记录：

`p_positive, p_negative, z_positive, z_negative, margin_before, margin_after, delta_margin, rank_transition, group_intervention, time_window, stage`。

预设分层只包含可明确定义的变量：在线绝对 margin 分位组、在线正负分数区间、当前域/任务、epoch、相同/不同用户的描述性标签、训练可见的实体曝光计数（若数据字段存在）。不把域样本数作困难度，不以 Test 或干预结果选择特征分层。

报告三种分布：纠正 pair、破坏 pair、排序未翻转但 margin 改变的 pair。单样本分数变化只用于解释它在多个真实正负对中的作用，不能替代 pair 分析。若“近零 pair”承担主要净收益，应同时报告该层人口占比、纠正与破坏数，避免只展示成功案例。

### A3. 与 B/C/D 共享 pair，建立具体关联

对 F 纠正的每个固定 pair，检查：仅重要组、删除重要组、近期/旧历史修正、历史预测集成、四态接口重组是否保持其正确排序。用集合交集和净加权贡献描述“重现哪些纠正”“新增哪些破坏”。这些重叠率是功能证据，不称独立中介比例。

输出逻辑表 `PAIR_TRANSITIONS`、`PAIR_STRATA`、`PAIR_OVERLAP`、有界 `PAIR_CASES`，物理文件默认 `.csv.gz`，完整/面板由population字段区分。全人口转移不保存 pair 笛卡尔积；案例表每格/阶段/转移类别最多64个 hash 排序实例。

## 3. 模块 B：参数组的真实干预

### B0. 分组必须按真实图核对

复用 `experiments/aug_history_20261105/group_registry.py::build_path_groups`。以源布局为起点：

| 组 | 实际内容 | 支持 cell |
|---|---|---|
| S | 非routing embeddings、shared_weight/bias、skip_conn、star_mlp、4个shared experts | 全六格 |
| Dd | domain embedding 的第d行、slot_weight/bias[d]、domain_expert[d] | 域d的两任务 |
| Tt | task_expert[t] | 任务t的三域 |
| DTdt | gate[3*t+d]、tower[3*t+d] | 指定cell |

域/任务专家在 fixed1 下无其他域/任务混合系数。`domain_specific` 这一内部 MLP 命名不能决定组归属。Stage 0 复用已有 `test_single_cell_gradient_paths`，补 KuaiRand 及完整功能导数支持核对；结构零用真实 row/cell 输入验证，一次即可。

### B1. 必做 S/N 二因素

| ID | 参数构造 | 科学作用 |
|---|---|---|
| O | 实际在线 $\theta_t$ | 同刻参照 |
| BS | 在线背景，仅S坐标逐值复制F | S单独充分性 |
| BN | 在线背景，仅N坐标逐值复制F | N单独充分性；也是删除S |
| F | 实际固定EMA | 组合状态 |

每条件五 seed、五固定锚点与 F-selected 时刻，全 Validation 和固定面板均评价。计算独立作用、删除作用、score/margin/AUC三层交互，不做独立作用机械求和。

### B2. 五 seed 完成后的有限细化

主要发现的定位使用 Clean 的 V_A，确认效果在对应 V_B 单列；Gaussian 用同一规则作边界分析。所有 core B1 都必须先完成五 seed，不允许第一 seed 决定其他 seed 不采集。

选择规则是**分配诊断预算，不是算法淘汰门**：先在seed内对晚期E35/E50取均值，再对五seed取均值；使用V_A的Macro AUC删除效应及交互。S/N分别在平均删除效应绝对值达到 `epsilon_mechanism=0.0002` 时进入细化；若S/N平均交互绝对值达到该量级，则两个大类均进入，不论单组作用大小。若三者均更小，仍保留全部B1数值并转向D的全模型功能解释；不得写“该组一定无效”。最多两个大类，不留“同量级”之类需要运行中人工解释的条件。

S 细分为 `S_EMBED`、`S_STAR_SKIP`、`S_EXPERT`；N 细分为 `D_ALL`、`T_ALL`、`DT_ALL`。对进入的大类，每个子组都做：

1. 在线背景加入该组的完整 R；
2. F 背景删除该组的完整 R。

每个细化锚点最多12个额外状态，锚点限 E10/E35/E50 与 F-selected。不做所有组的2^12组合，不给单参数唯一份额。

由于 D/T/DT 叶组在该 fixed1 图中的支持不重叠，可以将族干预在相应 cell 的预测与父状态其他 cell 的预测精确拼回，得到仅 Dd/Tt/DTdt 的读出指标；这是已审计路由下的等价计算，不是预测集成。若支持审计不成立，禁止拼接，改为最多两个有决策价值的实际叶组替换并标明未覆盖项。

输出 `FACTORIAL_EFFECTS.csv.gz`、`GROUP_SELECTION.json`、`GROUP_SUPPORT.json`。`GROUP_SELECTION` 记录所有被选/未选大类及原始效果，不将选择结果写成研究结论。

## 4. 模块 C：历史时间来源

### C1. 每个窗口的二因素

对 [02](02_THEORY_AND_HYPOTHESES.md) 的四个 h，在同刻构造：

| ID | 状态 |
|---|---|
| O | $\theta_t$ |
| CR(h) | $\theta_t+R_t^{recent}(h)$ |
| CO(h) | $\theta_t+R_t^{old}(h)$ |
| F | $\theta_t+R_t^{recent}+R_t^{old}$，读取实际F核对 |

CR/CO 都是参数修正干预，不是对历史模型重新训练。四组 h 共最多8个额外状态；h_eff 重合时只评价一次并保留逻辑别名。记录参数误差、预测误差、six-cell AUC/LL、纠正/破坏及交互。

### C2. 与主要参数组连接

在 B2 已识别的最多两个重要组 G 上，主窗口取 `round(H)`，限 E10/E35/E50/F-selected。候选组为已展开大类的三个互斥子组，未展开大类保留其S或N父组；因此候选坐标之间不重叠。按上述晚期五seed V_A平均删除效应的绝对值降序取前两组，完全并列按冻结group_id；不因效应低于细化预算阈值丢弃已做核心数据。对每组在**其余参数为在线**和**其余参数为F**两个背景下分别执行 recent/old 二因素。O/F/已做单组状态可复用，明确每项背景和模型 hash。不能用 online 背景的一项效应与 F 背景另一项相加。

这可区分“某组本身需要旧信息”与“某组的旧信息需要其他组协作”。没有发现单组大效应、但跨组交互大时，保留联合机制，不继续逐参数拆分。

### C3. 近期不等于噪声，旧历史不等于有用

逐步在线预测在固定面板上已采集，因此对每个 h 直接核对实际 $\theta_{t-h}\to\theta_t$ 的同一批 pair 是否变差，以及 CR/F 是否修复。定义和报告两项不同量：

- 自然历史实际变化：原训练在窗口内如何改变排序与训练目标；
- 受控读出修正：在当前参数背景上添加加权历史修正的实际效果。

二者共同支持近期不利更新解释，但不声称删除那些历史更新后后续训练仍不变。若 old 贡献大，结合 D1 判断是否是历史预测组合优势，不能直接称“长期记忆保留了知识”。

频率分析默认不执行。只有当相同时间归因可由持续漂移或往复振荡解释，而两者将改变 A/B/C/D 选择时，才对已存且时间分辨率足够的标量score/margin摘要作有界频谱描述；不能据频谱给噪声贴有害标签，不新增训练或追补未采集的投影序列。若现有摘要不足以分辨，明确这一低优先级边界，不影响已有时间干预结论。

输出 `TEMPORAL_EFFECTS.csv.gz`（包含group/background字段）与 `SOURCE_OVERLAP.csv.gz`。表名及外键以07为准，物理压缩路径由 `ARTIFACT_INDEX.json` 统一登记。

## 5. 模块 D：参数修正怎样变成排序收益

### D1. 同步精确历史预测 EMA

每步在固定 P_A/P_B/V_mech 上评价在线模型，并维护概率、logit 两个流式历史读出。初始化与权重完全匹配参数 F，最多保存当前累积状态和锚点分数，不保存每步完整历史参数。逐步分数只在RAM流式使用；C3仅持久化最终需要的anchor窗口参考端点概率p和已归约统计，不能把全部scheduled epoch×h的p/z落盘。端点概率预算见06§5.4；不落盘任何参数/梯度向量。

在锚点比较 O、F、概率历史集成、logit历史集成，记录：

- cell AUC/LL与 exact pair transition；
- F纠正的pair中，预测集成也纠正的比例和加权净收益；
- F独有的纠正与破坏；
- 两个顺序的 score 层路径分解以及 a/b 交互。

此处 ensemble 的结论明确限定于固定面板；不能写成完整 Validation/Test 的集成性能。全人口的 O/F 与 B/C 干预 AUC 仍是准确的。无需稀疏 checkpoint ensemble 或额外训练。

### D2. 完整计算图的两处切口

**切口一：h = STAR+skip。** 用 donor 参数计算 h，随后 recipient 的所有 gate、shared/domain/task experts 都消费同一个 donor h，routing 仍来自原输入。gate 训练时的 detach 不影响本次前向值。

**切口二：u_dt = fused_dt。** 用 donor 计算所有将进入 tower 的64维融合张量，recipient tower 对正确 dt 张量读出；路由和任务列顺序保持源实现。

对 O↔F 必做四态 `00/10/01/11`；对 B/C 选出的一个主要组/时间修正X，再以 `theta0=F−X, theta1=F` 做同样四态。X候选限已评价的完整组修正、全参数主半衰期recent/old，以及C2所选组的同窗口recent/old；按晚期E35/E50的seed内均值、再跨五seed的V_A真实删除效应（F减去删除X态，取有符号值）选择最大正值，完全并列按冻结descriptor_id。该同一个X描述符应用于各诊断锚点，V_B单列确认。若没有正值候选，第二contrast与第六融合读出标 `NO_POSITIVE_COMPONENT_CONTRAST`，保留完整O/F接口机制，不临时扩大搜索。最多一个额外contrast，不能继续搜切口。

每个 contrast、每个 cut：两个端点必须等于原模型真实预测，另两个混合状态重新前向；评价完整 Validation 和固定面板，按相同 pair IDs给出：

`margin_00, margin_10, margin_01, margin_11, representation_term, downstream_term, interaction_term, rank_00/10/01/11`。

**最低解释要求：** 报告主收益范围中纠正与破坏各由哪一项使 margin 跨零；展示干预删除该功能项时这些纠正怎样减少，并与净 AUC核对。若只是中间表示相似度、RMS或相关系数，本模块未完成。

#### D2b. 在真实专家融合公式中进一步回答“怎样改变预测”

仅说“上游表示更好”仍然过于宽泛。对 O→F，在 E35 与 tF 做一次有界融合分解。对同一输入/cell，写 `g` 为shared expert gate、`e_j`为shared expert输出、`d/t`为domain/task expert输出，所有张量从真实源forward提取：

$$
u=\sum_jg_je_j+d+t,
$$

$$
\Delta u=\underbrace{\sum_j g_{O,j}\Delta e_j}_{c_{content}}
+\underbrace{\sum_j\Delta g_j e_{O,j}}_{c_{routing}}
+\underbrace{\sum_j\Delta g_j\Delta e_j}_{c_{interaction}}
+\underbrace{\Delta d}_{c_{domain}}+\underbrace{\Delta t}_{c_{task}}.
$$

这五项在融合张量上精确相加，但它们的AUC作用不相加。逐项在F的融合接口注入 `u_F-c_k`，保持F的tower及其他计算不变，重新预测完整V/面板，得到真实删除效应。用E35五seed V_A的最大正向平均删除效应选择最多一项k，完全并列按 `content,routing,interaction,domain,task` 顺序；同一k应用于E35及tF。无正向候选时不执行第六态，保留五项与完整cut结果。随后在D2选定的组/时间contrast `theta0=F−X, theta1=F` 上重新计算同一个成分 `c_k^(X)`，只做一次 `u_1−c_k^(X)` 删除，在V_B检查X原本纠正的同一批pair是否因此丢失。每时刻最多6个额外状态，不重新搜索其他成分或展开组合。

第六态用于直接连接“重要参数/历史修正X”与“功能成分k”；不能仅凭全F的某功能项重要与X参数组重要，就认定X通过该项起作用。若该连接不成立，分别报告两条证据，不强行拼成完整链。

最终将参数来源与功能来源交叉报告，例如“共享参数的近期修正主要通过改变shared expert内容，使哪些原本错误pair的正负logit间隔恢复”，或“gate重分配只有与expert内容共同改变才有收益”。示例不是预设结论。若domain/task项承担主要作用，按实际支持格报告；若只在tower变化后才有效，结合D2切口二交互说明。不存在足够影响的成分也直接报告，不再无限向每个神经元拆分。

每个分量必须有融合重构误差、真实删除后的预测、对应pair跨零/破坏数，才能称功能机制；只报gate熵或expert张量范数不达标。源fixed1的融合系数为1使上述式子成立；Stage0仍以实际forward核对，不在未来模型改变后静默复用公式。

数值上，五项的严格代数闭合对捕获张量以FP64重新计算的融合差验收；源FP32累加得到的实际fused差与五项之和还可能有舍入残差，必须单列。实际删除背景始终使用真实源u_F，不能为追求1e−10代数误差改写源前向。

### D3. 路径积分和 STAR 乘法项是针对性辅助

对 D2 的实际 contrast，在预先 hash 抽出的有界诊断行上使用完整函数导数的 JVP 积分；端点以及 alpha=0,.25,.5,.75,1都做真实预测。优先8点，闭合不达标时16、最多32点。输出残差、沿途 margin 曲线和 S/N 分量；不依赖首点线性近似。

若 S×D 对 logit/margin 的交互本身重要，增加一个 STAR cross-term 删除/注入对照，固定 embedding 和背景，完整下游重算。若交互仅出现在 AUC 而 score/margin近可加，优先解释为排序阈值组合效应，不启动 STAR 定向分析。

### D4. 训练来源：先给直接证据，再决定是否需要 Adam 探针

在同一固定训练面板、独立 P_B 与 V_mech 上，记录实际更新前后的 BCE、pair risk、AUC/margin变化，并与本次识别的 recent/old 修正对应。Gaussian 条件分别记录源 replacement loss 与 clean loss，固定噪声探针不消耗训练 RNG。

如果近期更新降低训练目标，却反复损坏验证排序，且 recent 修正实际恢复同一批被损坏pair，可以接受“近期更新中的训练目标与验证排序不一致”这一有边界的主要问题描述。它已经比“模型抖动”具体，无需强行命名为过拟合或负迁移。

只当该证据会实质改变是否转向 Adam（方向C）时，执行有上限的同状态虚拟更新诊断。预留 Clean E20/E35 最后一个实际 optimizer step 的 RAM-only 来源包，只实际存储四份FP32向量：pre-step theta、真实数据梯度、实际更新后的偏差校正mhat/vhat；另存步计数等标量。coupled-WD梯度由数据梯度和pre-step theta派生，实际位移由该theta与已保存的锚点O相减，不能另外常驻这两份向量。四向量预算与05/06一致。

对主要组，比较实际方向、用当前梯度替换一阶矩方向但保留实际二阶预调节、保留实际一阶矩而使用各向同性预调节；后两者分别与实际组位移做范数匹配，避免把大小变化冒充方向机制。所有其他组仍使用实际位移。只在 scratch 上构造候选参数并评价固定面板，**不调用真实 optimizer.step、不修改m/v、不继续训练**。

此结果可支持“当前状态下 numerator历史方向/预调节方向是损坏来源之一”，不能等同于长期重置Adam的效果。每触发数据集必须覆盖五 seed、两个固定锚点，不能单seed停止。最多两个数据集，不添加beta、LR、WD网格。触发与否写 `SOURCE_ASSAY_DECISION.json`；不触发不记阴性。

## 6. 从机制到旧自适应失败：有限动作桥接

这是针对 H1/H2/H3 的只读诊断，不是新自适应 EMA 算法。

每 E5/E10/…/E50 使用源一 epoch pulse（从该窗开始时 F 初始化，fast=rho²、slow=sqrt(rho)）。在同一结束时刻设两个等基数菜单：

| rho菜单（7项） | 机制菜单（7项，h=round(H)） |
|---|---|
| F | F |
| 全参数fast / slow | F−R_all_recent / F−R_all_old |
| 仅S fast / slow，其他组F | F−R_S_recent / F−R_S_old |
| 仅N fast / slow，其他组F | F−R_N_recent / F−R_N_old |

所有候选只评价相同固定面板，不为菜单再训练；核心锚点已存在的全 Validation 结果复用。F 本身始终是允许动作。菜单同为7项避免用大菜单的样本内最大值天然压过小菜单。

解释分四层：固定F；从其他四seed训练侧选出的统一静态动作；训练可见信号的动作；同面板oracle。后者只作为有限集合诊断上限。另报告 V_A选/V_B评、交换方向的效果，不能称为在线可部署。

只复用三组特征/方法：身份阶段简单参照、明确适配的 C_simple、当前机制动作在P_A上的真实效用（含无拟合carry）；固定列定义见08第4节。不增加新架构或HPO。LOSO且早期 origins5/10/15→targets10/15/20的P_A未来效用拟合，留出seed origins25/30/35/40/45→targets30/35/40/45/50评分；P_B和V_A/V_B只报告迁移。预处理只拟合训练折。候选动作已在各个实际未来锚点以真实模型评价，不用预测器数值充当动作收益。

一套信号可能有预测力却一直选相同动作；必须报告动作频率/熵、相对静态动作的增量、六格损害与LL。若机制菜单有跨行侧空间且训练信号能选中，而rho菜单没有，支持H2在所测操作类中的解释；若只有oracle好，保留“存在空间但当前不可操作”，不能写成新算法成功。

### 6.1 只有会改变研究决策时执行的H2匹配检查

原rho pulse长一epoch，机制菜单删除半衰期内的修正；两者时域和位移幅度不同。若要把H2作为选择方向B的主要理由，必须先做以下有限判别，不能仅比较两菜单最大值。

采集时仅在E35/E50额外维护H步fast/slow：从 `t−round(H)` 的F初始化，连续接下来的H个真实在线更新推进到t。只额外保留两个FP32向量/anchor；其他S/N读出由mask派生。无需十个窗口都加一套菜单。

全部五seed完成后，最多选择两个共同scope：all，以及B中删除效应较大的S或N。每个scope内用当前P_A分别从 `{F,H-fast,H-slow}` 与 `{F,drop_recent,drop_old}` 选动作，不能rho选S而机制选N。令相对F的位移为delta_rho/delta_mech，取 `r=min(norm(delta_rho),norm(delta_mech))`，两者各缩放到r（比例不超过1），真实替换后在V_B评价，并记录完整V AUC/LL。每anchor最多4个额外状态。任一动作选F/零位移，则该匹配比较记 `NOT_INFORMATIVE_ZERO_RADIUS`，不记阴性。

若匹配时域/作用组/半径后差异仍保留，可把“所测操作方向不同”作为H2的功能证据；差异消失则降低为时域/幅度或原有限菜单差异。是否转向A仍须有训练信号与rho动作价值支持，不能自动改判A。该对照不声称穷尽全部rho序列或证明下一代递推算法有效。

作为选B的主要理由时，匹配后的机制动作自身必须仍优于F；仅仅比一个退化rho动作损害更少不够。自动触发条件统一见08§6.1，所有判断使用本block五seed数据，Test不参与。

## 7. 必须关闭的识别漏洞

| 漏洞 | 本方案修正 |
|---|---|
| 只看单组AUC，无法解释预测 | D2完整接口四态、同pair margin跨零与删除恢复 |
| recent删除冒充历史训练因果 | 分开标记当前读出因果和自然历史来源；有必要时有限同状态更新 |
| 稀疏历史预测平均冒充参数EMA等权 | 每step流式同rho、同初始化，限固定面板 |
| 缺失非线性交互 | S/N、recent/old、图接口四态，各有score/margin/AUC三层 |
| 用梯度代理作为最终机制 | 真实替换与前向为主，完整函数积分作核对 |
| 五seed结束后丢了参数 | 条件block驻RAM完成Stage2/3，释放后只聚合数表 |
| 按Test挑组或选择解释 | 全局MECHANISM_FREEZE后才加载Test，预定5种端点读出 |
| 自适应失败归因过宽 | H1/H2/H3分别支持，限定有限动作类，不要求唯一解释 |

完成本模块不是要求所有假设出现阳性，而是要求从实际干预中得出可用结论：哪条链成立、哪条不起主要作用、哪些交互须整体保留，以及它们怎样改变下一阶段 A/B/C/D 的优先级。


