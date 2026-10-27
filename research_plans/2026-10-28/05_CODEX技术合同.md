# 05 BASP-A：Codex实现技术合同

状态：设计合同；本文件描述的是**需要新实现**的接口和验收，不表示这些类或CLI已经存在。数学基础见02—04，数值唯一入口为08。实现目录固定为`experiments/mdmt_basp_v1/`，结果目录不得覆盖任何旧实验。

## 1. 代码基线与实施顺序

将`26-10-27@03c338092008a71b2d360a61b06d985eac551ec7`的native、source、strict_data、指标和RAM状态实现视为可审计参考，不直接改旧包。复制/导入依赖时保存每文件blob SHA、内容SHA256及许可证；禁止依赖未来移动的分支名。原生固定融合与原版M3oE的AutoML协议分开。读取AGENT.md，不能把有意关闭的融合参数当bug恢复。

实施顺序：只读audit/plan → 数据角色核验 → 固定K的预测骨架 → 收益面板/蒸馏 → 明确矩阵/NULL → 梯度权限 → 单元测试 → 独立代码审查 → 固定K开发 → 结构事务实现及测试 → 有条件结构实验。当前交付不执行其中的训练。

### 文件接口

```text
config.py         严格schema，重复键/未知字段/非法组合报错
identity.py       代码、数据、角色、动作和完整状态身份
roles.py          标签权限与row_id/group_id计划
prefix.py         native/pilot与来源标准化
model.py          无标签forward、目标头、需求树、source函数树
benefit.py        ProbeSnapshot、TeacherCache、收益预测、校准诊断
relation.py       GainTable、稀疏mask、NULL、可选低秩拟合
losses.py         BCE、收益/聚类/对比、owner/anchor、shadow
permissions.py   各损失参数白名单及梯度合成
conflict.py       BCE-only梯度草图、OOF ridge、分区与支持
transaction.py    四分支短训、状态恢复、一次性S验收
metrics.py        六格/宏/自然/pooled/原型指标
trainer.py        训练时钟、teacher刷新、投影、V选择
campaign.py       阶段预算、Q决策、统一冻结与历史E释放
export.py         机制证据、失败/拒绝/成本完整导出
__main__.py       audit/plan/prepare/train/report/freeze/evaluate
```

`audit`与`plan`默认只读，不能导入触发训练的模块。`train`仅在实现验收与独立review收据齐全后允许。

## 2. 数据角色与信息权限

继承已有自然行身份，不重采样成均衡训练。T0=fit，F=P_label，S=P_cal∪P_update，Q=Q_explore，V=validation，E=test，legacy=Q_confirm。当前两数据三域两任务，所有原任务标签均观测；KuaiRand long_view不额外乘click。编码/缩放仅由T0拟合。旧E/legacy已暴露，不能改名为全新确认集。

|角色|可做什么|禁止什么|
|---|---|---|
|T0|native/pilot训练、source保持锚点、预处理|伪造foreign真实标签|
|F|推荐训练、收益标签、蒸馏、分区提议、影子读取、虚拟更新|将训练内重用称为全pipeline折外|
|S0/S1/S2|分别在第6/10/14轮一次性验收最多一个事务|做梯度、试多个阈值后重用同片、据此宣称测试泛化|
|V|固定规则选择完整checkpoint|训练收益预测器/拟合关系/挑分裂候选|
|Q|预定义阶段决策与冻结前有限研究判断|成为逐batch控制反馈或无上限调参集|
|E|所有方法/阈值/状态身份冻结后统一历史复评|选checkpoint、改算法或重启失败版本|

F_probe按每域固定无放回取min(1024,N_d)行；两个任务同row，不能拆fold。其域平衡只是辅助测量设计，L_rec保持自然人口。两固定子面板F_probe_A/B用于条件需求与CL资格诊断，**均是训练数据**。有可信user_id时面板/OOF/S按用户组划分；无可信ID则明确row级，不声称user独立。

S在读取质量前固定为三片；记录每片真实行数、正负例和有效支持。某小域支持不足就拒绝相应结构，不从Q/E借标签。角色权限API必须按`purpose`检查并导出访问日志。

## 3. 前缀、源函数与“端到端”的准确范围

共同前缀默认沿已审计协议：native固定融合24个自然T0 epoch、batch40960；V在12—24选macro LL完整状态；source-pilot在owner T0上12 pass，两任务各256→32 ReLU→1，拼为64维φ，经T0_s标准化（std下限.001）。保存从原pilot头到标准化φ坐标的精确仿射转换。前缀是所有方法共同投入，研究账只计一次，独立部署账必须计入。

`train_scope=ADAPTER_JOINT`：native/pilot冻结；推荐梯度同时进入可部署需求表示、路由、本地叶、读出和获许可的source后继。这是**适配子系统端到端**，不是整个embedding到输出都更新。

`train_scope=FULL_MODEL_JOINT`：复制一份可训练native，L_rec反传到当前h,z及底座；同时保留冻结参考native₀/pilot用于φ、来源教师和诊断坐标。当前h,z不得从静态缓存读取；teacher包含当前native的完整冻结副本。收益/CL等辅助梯度仍在h,z处detach，防止通过改底座伪造监督。最终部署通常需当前native与参考φ路径，必须计入双路径成本，不能声称与冻结版本同价。只有06规定的全模型比较才启用，与TARGET_MATCH使用完全相同的底座更新权限和学习率。

这个分层不是规避端到端目标：先隔离机制是否存在，再检查真实底座联训是否保留该价值。结构离散接受和teacher更新采用交替过程，不声称全套操作单次可微。

## 4. 张量与稳定身份

```python
@dataclass
class ForwardOutput:
    logits: Tensor       # [B,T]
    reference_logits: Tensor  # [B,T], 不含foreign显式读取
    need_mass: Tensor    # [B,K_N_max]，inactive=0
    source_values: Tensor | None  # [B,D,K_S_max,64]
    read_mass: Tensor | None      # [B,T,K_N_max,D,K_S_max]
    null_mass: Tensor | None      # [B,T,K_N_max]
    gain_prediction: Tensor | None # [B,T,A_d]，本域动作不进入
```

动态树用stable UUID管理，禁止用“列表第2项”当证据身份。`target_node_id/source_node_id/action_id/parent_id/version`全链保留；每个域可不同K。实际计算可按域紧凑分组，padding仅用于日志/批接口。

`GainTable`每条记录包含：目标域、任务、需求叶、source域/叶、动作定义及幅度、参考状态hash、teacher epoch、G原量纲、二阶矩、unique row count、ESS、支持mask、observed/prior/unknown、prior_parent_id、过期标记。方差可以用于诊断，但默认不做随意置信下界裁剪。重复epoch访问同一row不增加独立样本量。

`forward(x,domain)`绝不接受y、真实gain或梯度草图。标签只进入外部loss/probe/decision函数。

## 5. 初始化及确定前向

A默认K_N=K_S=2；所有source后继U=0，V独立N(0,.02²)。w初始0，L2半径2；目标a输出层0；本地l叶线性32→T，权重/偏置0；owner C/b=0产生均匀责任。需求trunk用Xavier、bias0；μ输出层0，中心在epoch2结束初始化。

源owner预测：

\[
z^{owner}_{st}=z^0_t+c_{st}+v_{st}^{\top}\sum_j u_{sj}(\phi_s) f_{sj}(\phi_s).
\]

v/c是冻结标准化pilot头；真实owner风险只用d=s行。源保持约束是此owner混合预测，不要求每个新叶单独在全部source支持上拟合相同函数。另报每叶与群体风险，不能以混合保持掩盖完全无支持的叶。

epoch1/2的foreign路由固定为α_a=m_a/2，NULL=.5；读出w为0所以初始预测严格等native。only-cluster/no-foreign方法始终不将该读取加入预测，但仍执行同一训练内probe工具以生成可比较需求标签。**这意味着“只收益聚类”仍可能通过收益监督间接利用来源先验，只是没有显式foreign读取；不能把它描述成完全不接触来源信息。**

epoch2结束：对μ预测而非原始带标签b做KMeans2（kmeans++，8次初始化，最多100次迭代，tol1e-4）。全部预测相同/中心距离≤1e-8则退化为均匀责任，标记NO_PREDICTABLE_PARTITION；仍可完成固定K研究，不强造分离。节点q的距离温度默认1，距离采用04的动作质量加权，不对每行归一成单位向量。

epoch3开始按实测G确定共享；负值截断到0，NULL始终合法。完整α与NULL逐行和1。本域来源严格mask，UNKNOWN不当负值。G是固定测量buffer，无直接推荐梯度；q和μ仍在最终预测图中，source内容可被允许的推荐梯度改变。

## 6. 收益刷新、冷启动与循环依赖

每个epoch开始，从上一轮**投影及结构提交后**状态建立teacher；eval模式、关闭dropout，clone后不共享可变参数。全部foreign单动作在同一F_probe上计算BCE风险差，记录相同OFF参考。测量可并行使用已经算出的c，不对每个动作重复运行整个native。

收益目标整体detach。G用该teacher自己的q加权，避免混合新旧责任。当前模型q可以随batch变化，故G最多滞后一轮；记录current/teacher q的平均L1变化与固定动作风险误差，不声称零延迟收益。默认不用无期限EMA；需要EMA时必须另立版本并保存有效状态窗口。

训练每step附加一个循环的probe minibatch256行，计算L_gain/cluster/CL；probe标签/收益缓存只来自F。cycle row plan固定并计入成本。主推荐batch2048保持自然无放回顺序及尾批。

为了避免被关闭路径永远未训练，每step按`target/source/root`轮转一个foreign读取动作，执行L_shadow=BCE(sg(z_ref)+wᵀsg(f),y)，**只更新该w**。两个任务同时处理；source后继不借此获得foreign写权，目标头也不被shadow改写。所有使用相同source库的对照具有同样shadow计划。GAIN_RELATION关闭不关闭诊断探针；仅在source完全不存在的目标基线中省去该计算，并同时提供其可用计算预算的强对照。

测量当前函数收益与训练机会是两回事。shadow只修复reader冷启动，不证明未来来源形成的潜力已被完整探索；若owner+主预测均不形成不同source函数，只允许06的一次固定机会分配诊断，而不是无限增加探索网络。

## 7. 全部loss的精确定义与权重

L_rec=Σ_valid BCEWithLogits / N_valid，保持原尺度；不再把全部loss整体除以模块数量。

L_gain是每域探针平均的加权MSE：任务权重1/T，初始动作1/A，分裂后动作坐标权重守恒；目标b/s_b。s_b在epoch1后用完整F_probe的b计算全数据集RMS，floor=.001，随后冻结。探针等域只影响辅助项，主目标不改域权重。

L_cluster按节点路径质量加权的预测收益到中心平方距离，每行再除参与节点质量和。中心可训练；μ校准损失防止以共同缩小表示轻易消除距离。没有熵均衡、强制相等原型份额或无条件正交。

L_owner是各真实source风险自然质量加权平均，π_s=|T0_s|/|T0|。每step每source取256个owner锚点无放回循环；L_anchor是同样π_s加权的KL[冻结pilot || 当前owner混合预测]，两个任务均权。实现用稳定logit形式。

L_contrast在收益预测空间作用，最多每step64正对+64负对，不构造全N²。按固定teacher硬需求组建立候选。正对为同组且teacher预测画像加权距离≤.2的两行；负对只来自同域两个组，在F_probe_A/B的**实测群体收益画像**上都距离≥.2且最佳动作（含OFF）不同。两组在任一半支持不足64则不设负对。负margin= min(.5×两个面板组间距离的较小值,1)。

\[
L_{contrast}=\operatorname{mean}_{pos}\|\mu_i-\mu_j\|_W^2+
\operatorname{mean}_{neg}[m_{ij}-\|\mu_i-\mu_j\|_W]_+^2.
\]

空集合相应项为0，teacher资格和margin全部detach。对比关闭则不维护配对队列。这里只直接约束**需求表示**；来源函数是否分化必须通过输出/风险交叉优势检验，不能从CL距离推出。

默认系数：gain .1、cluster .05、contrast .01、owner .1、anchor .1、shadow .05、size 0。epoch1只训练推荐头/reader及shadow；epoch2启用gain（系数1进行一个既有epoch预热，不额外训练）、其余结构项0；epoch3开始使用默认gain/cluster并开放owner/source；contrast从epoch5开始。所有系数在配置中记录，不依Q择优追加网格。

仅对共享需求trunk r的辅助梯度做上限：令g_R为L_rec梯度，g_A为已加系数的gain/cluster/contrast梯度，合成g_R+γg_A，其中γ=min(1,.25×max(||g_R||,.001)/max(||g_A||,1e-12))。epoch2 gain预热不采用此上限。gain专有头/中心保留自己的目标梯度；native完全不接受这些辅助梯度。导出γ分布：若长期触顶且校准差，只能使用一次预定义“去CL/固定辅助权重”的诊断，不追加动态控制网络。

## 8. 梯度权限合同

|loss|当前native|a/local leaf|need trunk/μ/centers|w|source U/V|owner router|
|---|---|---|---|---|---|---|
|L_rec|仅FULL_MODEL_JOINT|是|q相关路径是|是|仅SOURCE_WRITE且foreign非detach|否，foreign不使用owner路由|
|L_gain/L_cluster/L_contrast|否，h/z detach|否|是|否|否|否|
|L_owner/L_anchor|否，使用冻结z0/φ|否|否|否|是|是|
|L_shadow|否|否|否|仅轮转w|否|否|
|投影|无梯度|无|无|半径约束|source U缩放|固定不动|

a/local的L_rec梯度不是通过辅助loss绕行；r既供本地函数又供q，必须按实际图分解。SOURCE_WRITE=false时不能对全部f执行requires_grad_(False)，因为owner仍要训练；只在foreign路径detach。SOURCE_WRITE=true也不开放source-pilot/root/native₀的参数。

优化器为按稳定名字管理的Adam参数组：非center lr=.001，center lr=.0003；betas(.9,.999)、eps1e-8、weight_decay0。FULL_MODEL_JOINT当前native组lr=.0001，其余不变。预热冻结参数仍保留组身份，不重置已有Adam状态。新分裂残差参数首次加入时m/v/step=0；不激活的参数不能以零梯度偷偷执行weight decay或moment更新。

## 9. 来源保护与投影

每个epoch及事务候选末尾，检查每source owner两任务在完整T0_s上相对冻结pilot的ΔLL≤.001；每source所有active叶的函数残差RMS在T0_s及每个foreign F_d上分别≤.5。阈值不以域样本量改变。完整检查过贵时只能在固定配置中选择可验证分块累计，不能改成随机小batch合规声明。

冻结owner路由及各V，对同source全部可训练U乘共同γ∈[0,1]；父/子累加残差也全部乘γ。此时owner logit沿γ仿射，γ=0恢复原pilot，因此BCE凸子水平集包含0。RMS随γ线性；先分别求风险与RMS最大可行γ，再取最小。二分最多20次，返回可行下端，容差1e-8。保留Adam moments，明确这是投影式Adam，不是保持无约束轨迹等价。

若连续两次γ<.1，记录SOURCE_CONSTRAINT_BINDING并停止该source增长，**不**降低约束或给其他域造标签。可通过预定义单次去anchor但保留硬约束对照判断冗余；任何源风险预算放宽属于另一个方法。

## 10. 结构事务的接口与容量对照

```python
proposal = detector.propose(snapshot, F_probe, budget) # 不读S/V/Q/E
candidates = transaction.fork(snapshot, proposal,
    variants=("KEEP", "READER", "WIDE", "SPLIT"))
for arm in candidates:
    arm.train_exact_batches(shared_row_plan[:8], shared_owner_plan[:8])
    arm.project_and_refresh_inherited_policy()
decision = decision_rule.evaluate_once(candidates, S_window)
transaction.commit(decision.winner_complete_state)
```

提议/阈值/窗口见03和08，不能再给runner留“自行决定是否显著”的自由度。默认一次事务选一个全局父节点；同窗不为第二名重用S。

NEED_SPLIT保留父函数，新增两组32→T零输出线性残差及一个收益距离门。源分裂新增两个rank4残差，父保留；source子读先验由F编译的条件r(x)固定，owner新增φ64→1二叉门。读取w兄弟共享。

WIDE增加实际可训练的**无条件**残差，不使用提议群体的条件路由。NEED情况下使用r32→hidden→T的GELU（初始hidden=4，最后层0）；SOURCE情况下选择rank∈[4,16]使新函数参数接近SPLIT新增总参数。计入SPLIT新增gain-head列、中心、owner节点和路由参数，不能只匹配U/V。最接近配置须满足新增参数差≤5%、相关新增MAC差≤20%；若不存在则标记CAPACITY_MATCH_UNAVAILABLE，不作结构必要性结论，不能拿dummy参数凑数。

READER为任务私有小残差：r32→2→1，作用于提议父覆盖范围；source提议对受影响foreign域任务各建一支，need提议对本域各任务建支。新增输出0；它可能更小，真实成本另报，不强制扩到与SPLIT一样大。

候选8步期间teacher/G以及CL资格冻结，子G只用守恒父先验；每个分支消费同一F/owner序列。提交后生成新版本并在下一常规刷新重测G。SPLIT拒绝时可以选满足同样安全边界且风险最小的READER/WIDE，否则KEEP；若选WIDE，只报告普通容量增量，不叫成功分裂。

完整快照包含：模型参数/buffer、所有Adam状态、native₀和teacher身份、树结构/坐标映射、source标准化器、gain原始统计及版本、损失尺度、配对资格与队列、Python/NumPy/CPU/CUDA RNG、具名子stream、sampler游标、owner游标、主时钟、S已消费标记、selected/final选择历史。失败后不可只恢复权重。

选择发生在S上，因此不能把S最优分支风险再当无偏效果估计。只在Q/后续完整模型中验证研究主张。每次接受/拒绝的原因及四分支全部结果必须导出。

## 11. 六个开关与关闭语义

|开关|false时|仍应正常运行的机制|
|---|---|---|
|GAIN_CLUSTER|以冻结h0的固定32维投影做T0/F无标签KMeans责任；不训练μ/中心损失，或明确标记aux-matched诊断版本|G在固定责任上测量；CL可更新用于本地预测的r/需求投影，但必须启用其独立语义头并计参|
|GAIN_RELATION|`relation_fallback=NONE`彻底不显式读foreign，或`FIXED`固定α=m/2；配置必须指定|收益聚类可由训练内probe监督；分裂可研究本地需求；G只诊断不驱动政策|
|DYNAMIC_SPLIT|不创建事务、不读S；固定K继续所有正常训练|异质性诊断仍可输出，不伪造接受/拒绝记录|
|CONTRASTIVE|不计算该loss、不建配对队列|仍导出真实风险/函数分化，不能把无CL等同无分化|
|SOURCE_WRITE|foreign路径detach；owner监督和保持照常|读取、gain、需求分裂仍可用；来源只由owner学习|
|LOW_RANK|使用实测dense有向G及其mask|不改变source函数rank4、动作数或训练预算|

GAIN_CLUSTER关且CONTRASTIVE开是允许的独立实验：采用μ作为独立语义头接受L_gain以定义对比，但q仍固定，r仍供本地功能；这属于`aux_matched=true`配置，不可与完全无收益监督的普通聚类偷偷混称同一臂。主普通聚类对照`aux_matched=false`，去掉不被部署使用的μ参数，按实际参数做容量匹配。

`GAIN_RELATION=false & fallback=NONE`时不允许SOURCE_SPLIT事务，因为没有显式foreign服务路径；NEED_SPLIT仍合法。SOURCE_WRITE=true在没有foreign预测时不产生foreign写梯度，这不是实现错误，但必须显示`effective_write=false`。

LOW_RANK只指关系估计/部署压缩：对各任务有观测的G做带支持权重的rank2近似min Σ_valid ESS(G−ABᵀ)²+.001(||A||²+||B||²)，固定5轮交替ridge，deterministic初始化；同域mask、UNKNOWN以及子先验质量独立保留。G观测原表仍存以诊断，政策使用重构值；符号错误率、重构误差和决策损失必须报告。初始6需求×6source×2任务时，dense有效关系48，而rank2因子也是48，不预设有参数节省；更不减少source函数调用。

## 12. 训练伪代码、合法选点和成本

```text
固定角色、源码、seed及各方法配置；重建共同前缀
创建方法实例；state identity包含全部开关与预算
for epoch in 1..20:
  if epoch>=2: 冻结上一状态teacher，刷新同版本probe标签
  if epoch>=3: 用teacher q生成G/support/CL资格，应用当前继承规则
  for natural F batch(2048,保留末批):
    前向；主BCE；按阶段取probe256/owner每源256/轮转shadow
    白名单求梯度；仅r共享处辅助norm cap；一次Adam提交；w投影
  if epoch==2: 根据可部署μ预测初始化需求中心
  if epoch>=3: 完整source风险/RMS投影
  if epoch in {4,6,10,14}: F-only需求诊断
  if epoch in {6,10,14}且合格: 事务最多8个主更新并推进游标
  V评价；epoch>=5才参加macroLL最小选点，平局1e-12取最早
保留selected与epoch20 final完整状态，二者分别导出Q
```

事务8步是额外主更新机会：所有在同一配对结构比较中的非增长臂也在该预定窗口执行8个普通KEEP步，**不因是否提出候选改变实际训练主时钟**；增长方法的其他三分支另计搜索成本。04简写以此条为准。若无提议仍执行相同KEEP预算，不读S。主数据游标按预先生成的连续permutation流消费，不能丢弃或重放8步。

复杂度：初始A=(D−1)K_S，Q_need=T×A。每batch source残差O(BΣ_sK_Spr)；关系混合O(BTK_NA)；gain头O(B·32·T·A)；探针全动作风险O(N_probeTA)，共享前向；草图O(N_probe·pr·16)，只在少数窗口运行。dense统计O(TΣK_NΣK_S)，协方差草图O(16²/叶)。原型增长增加函数执行，低秩G并不消除它。动态内存使用实际叶，不预分配并把inactive假称免费。

初始完整A（冻结前缀、无结构、dense G）的可训练参数按定义为63680：目标头49926，need trunk8288，gain heads792，centers48，local396，w768，source U/V3072，owner390。六个width41目标残差头为63966，差286（0.45%），是一个强容量匹配入口。这些是**结构计算值**，不是运行测量；Codex须按真实代码复核，偏差即检查合同。无CL本身无额外参数；teacher、G、冻结φ前缀单独计内存/部署需求。

## 13. 必须通过的单元、数值及语义验收

1. **无标签部署**：随机替换/打乱标签后forward输出不变；任何label/gain外部输入被API拒绝。
2. **角色**：Q/V/E不能调用训练/probe；S每片最多一次事务验收；两个任务不得跨fold。
3. **收益**：手算BCE差、符号、一阶/二阶近似、OFF=0；同state所有动作；禁止softmax冒充G。
4. **权限**：逐loss autograd测试白名单；SOURCE_WRITE=false foreign梯度严格不存在，owner梯度仍非零；φ/native₀完全冻结。
5. **冷启动**：w=0初始等native；U=0且V非零可学习；全G≤0输出等z_ref，但shadow仍更新w且观测继续。
6. **分裂守恒**：非0.5路由、任意父G、含NULL、多任务情形下need/source split前后logit绝对误差≤1e-6（同shape FP32）；FP64≤1e-12；重复列权重/先验计数不增加。
7. **状态事务**：KEEP重放下一步loss/参数/Adam/RNG/游标一致；拒绝分支不污染母状态；新增参数step=0，旧参数不重置。
8. **小域/空支持**：空batch/空原型/单类AUC→NA而不是0.5；ESS正确；inactive/source-own mask严格为0。
9. **source约束**：γ=0恢复teacher；凸可行区间与RMS缩放正确；投影之后再选V；记录绑定原因。
10. **模块独立性**：六个单独关闭语义、指定pair交互、所有非法组合及缺配置拒绝；每臂实际主更新数相同。
11. **差异可识别**：CL-only距离不能触发分裂；只变read不记write收益；WIDE胜时不能记split成功。
12. **成本**：63680/63966初始计数，分裂含router/gain扩展计数，probe/失败/搜索另外计；低秩不假报函数调用减少。
13. **数值**：FP64 softplus logit LL、AUC以logit排序；同batch恢复严格，跨shape按atol1e-5/rtol1.3e-6检查并保留最差点，不能用过严纯绝对误差把有效运行当失败。
14. **候选B/C**：B虚拟Adam对真实一步及有限差分hypergradient；C simplex投影/QP残差/精确BCE误差；未实现不得标PASS。

本包11脚本只验证合同和少量解析性质；上述完整模型验收必须由Codex实现后实际运行，不能用静态检查收据替代。
