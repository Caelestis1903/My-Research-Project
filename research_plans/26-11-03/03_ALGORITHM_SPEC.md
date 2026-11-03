# 03 冻结算法规范

以下定义为Codex实现合同。H/T不组合运行；不新增需求D_c、动态q、条件moment表或可学习门。所有“改善”均为待实验判定，不是算法保证。

## 1. 记号、目标与参数范围

k=1,2,...为全局更新步；进入本步参数为θ_k，前一步进入训练时的完整参数为θ_(k−1)。每个真实batch有B行、两个二元任务、三个域。L0=Σ_iΣ_t BCE(p_it,y_it)/(2B)，尾批使用实际B。保留源模型的probability BCE计算，不改为另一种clamp/logit损失。

S仅包含：参数名恰为`shared_weight`、`shared_bias`，或前缀为`skip_conn.`、`star_mlp.`、`expert.`的参数。按外层归属判断，内部domain_specific子串不排除。embedding、domain_expert、task_expert、gate、tower等均不属于S。所有参数互斥且完整覆盖；每步每参数最多更新一次。

β1=.9、β2=.999、eps=1e−8；基础lr=.01；coupled weight decay w为ML1e−4/KR1e−5。对本步有效参数，g=数据梯度+wθ；S手工只加一次L2，其他参数由源Adam加一次。None不等于零：普通参数None不推进任何状态，显式零推进；S出现意外None是接线错误，不静默置零。S在源底座预期每步均有梯度；其s时钟与全局步一致，仍保存参数级step并断言。

固定非反馈参数EMA在全部online参数更新后只更新一次：ML rho=.9828205985452511，KR rho=.995，从θ初始值起步，buffer复制。它不参与probe、donor选择或梯度计算。

## 2. 可复现的域任务条件嵌入插值

### 2.1 Anchor与donor

每步每个出现的域d，按独立CPU随机流从本batch域内无放回抽M_d=min(256,n_d)个anchor。两个任务共享该域anchor集合，但donor分别抽样。原始batch不重采样、不删行。

对anchor i和目标任务t，u=1−t，候选优先级：

1. 同域、不同原行号、y_jt=y_it且y_ju≠y_iu。
2. 若1为空，同域、不同原行号、y_jt=y_it。
3. 若2也为空，j=i（identity，不伪造记录）。

在所用集合均匀抽一个donor；不同anchor允许复用donor。优先级由标签支持决定，不由loss、域大小或历史性能决定。不要求两种目标标签在batch中都出现；不制造ML的01联合标签，不删除KR已有01。记录各级使用率和自配比例。

U_MIX对照直接从第2级集合抽取，不测试第1级。H_REAL第二probe为同域独立无放回抽M_d个真实样本；不按标签平衡，可与第一集合重叠。

随机key精确定义为UTF8字符串`MDMT1103|dataset|seed|epoch|global_step|stream|domain|task`；SHA256前8字节按little-endian转整数，再模(2^63−1)。stream为anchor、donor或real_second；anchor的task固定−1。使用CPU torch.Generator生成索引。key不包含arm、λ、κ、τ；相同条件的方案共享索引。模型初始化与原loader随机流完全沿源实现，不接入本key。

### 2.2 插值位置与梯度

τ默认.1，固定每行同一τ，不额外抽Beta分布。将anchor和donor输入拼接为2M行；在源`model.embedding`输出处，前M行替换为(1−τ)e_anchor+τe_donor，后M行保持原值。所有非`domain_indicator`的字段共同使用同一τ；域字段逐位保持anchor值。使用源`field_slices`映射，不假设原始列顺序等于嵌入排列。

监督只使用前M行的目标任务t，label为anchor的y_it（donor同标签）。不把另一任务的插值输出强行配上anchor标签；另一任务不进入这份辅助loss。标签只用于采样和loss，绝不能写入x。

监督型M/TM中两端embedding均保留梯度，不能detach donor。H/C的probe只求S梯度，但同样保持真实forward/gate的语义，不替换原`emb.detach()`。hook用try/finally移除；不允许跨步图。源模型无BatchNorm/Dropout，仍在S0断言，不擅自修改train/eval状态掩盖差异。

插值是虚拟嵌入，不宣称原始离散字段生成了真实合法记录。ML的age虽由同域端点混合，仍只是虚拟特征，不改变实际域路由。两端同标签是训练约束，不是中间点标签已被观测。

### 2.3 监督型M的完整损失

L_mix,dt=mean_anchor BCE(p_t(e_tilde),y_it)，w_dt=n_d/(2B)。

L_M=(1−λ)L0+λ Σ_(d,t) w_dt L_mix,dt，λ默认.1。

所有出现域和两个任务都计算，Σw=1；无缺席域质量重分配。自然anchor保留目标标签边际，不将正负配对强行均衡。该式刻意改变优化目标，不声称其仍是原BCE的无偏梯度。

λ=0或τ=0时直接令L_M=L0，旁路所有增强抽样与forward；不是把抽样后的identity loss当成全batch BCE。对T而言这仅将监督切到T0，并不关闭仍为正的κ。普通M对所有参数调用普通Adam，固定EMA不变。

## 3. H：只修复旧历史，不改当前真实监督

### 3.1 当前梯度与probe

先在θ_k用完整真实batch计算L0及全部参数的数据梯度，保存；H所有版本的当前g均为∇L0+wθ。更新v=β2 v_old+(1−β2)g²，vhat=v/(1−β2^s)，D=1/(sqrt(vhat)+eps)。D是对角向量，后续按S整体拼接的数学定义计算。

轮转probe域d=(k−1) mod3。该域为空则按d+1、d+2循环找首个出现域；只处理一个域。该域两个任务分别得到：a_t=∇_S真实probe均值BCE，b_t=∇_S条件插值probe均值BCE。probe梯度不加L2、不写入`.grad`，参数仍在θ_k；当前真实g与v不被probe改变。H_REAL将b_t换成独立真实第二probe；H_U将donor规则换成U_MIX。

### 3.2 双证据与锥

令u_t=sqrt(D)*a_t，v_t=sqrt(D)*b_t。若任一L2范数≤1e−12，跳过该任务；否则当cos(u_t,v_t)≥γ=.2时，将u_t/||u_t||和v_t/||v_t||加入法向量集合N。阈值不调参。两个任务最多四个法向量；不同任务之间不要求正余弦。

定义h=sqrt(D)*m_old，以及C={z: n_j^T z≥0，对所有有效j}。

h_star=argmin_(z∈C) 0.5||z−h||²。

0总可行。无有效法向量则h_star=h；首次m_old=0同样不需投影。令r=[(1−κ)h+κh_star]/sqrt(D)，κ默认1，允许的搜索值见05。

m=β1 r+(1−β1)g；mhat=m/(1−β1^s)；θ←θ−lr*D*mhat。

其余参数正常Adam使用真实g。该偏差因子是与Adam一致的质量归一化；经过非线性历史修复后，不声称mhat是无偏梯度估计。锥只约束历史分量的一阶方向，不保证当前g、总更新或AUC同时改善所有任务。

### 3.3 低维精确求解

以单位法向量为列，G=N^T N，q=N^T h。枚举最多2^4个活跃集合A；λ_A=−pinv(G_AA)q_A，其他λ=0，要求λ≥−tol、q+Gλ≥−tol且活跃残差绝对值≤tol。候选h+Nλ中选距离h最小者；同成本按集合大小再字典序保留首个。

Gram与内积用FP64，pinv rcond=1e−12；tol=1e−9*max(1,||h||)。不构建P×P矩阵，不引入外部QP依赖。若无可行候选，记录矩阵、尺度与残差并终止该run为数值故障，不能静默改成清零moment。参考实现见design_reference.py。

实现时可流式计算S各张量的内积和Nλ，不必把模型所有参数复制到CPU；CPU只接收≤4维Gram/q。保留完整参数形状和manifest。

### 3.4 必要近邻控制

- H_REAL：上文第二probe换为独立真实样本，其他完全不变。测试增强是否必要。
- C_MIX：使用同一a/b和同一D/锥，但投影对象为sqrt(D)*g，不改m_old。得到g_repaired后m=βm_old+(1−β)g_repaired；v仍由未投影真实g更新，其他参数普通Adam。κ同样用于投影前后凸组合。测试当前梯度处理是否足够；这不是原版PCGrad复现。
- H_NORM：先计算本状态下H应产生的r，但仅使用c*m_old，c=||sqrt(D)r||/||sqrt(D)m_old||；分母≤1e−12则c=1。其他相同。它匹配历史范数，不声称跨不同训练轨迹匹配整个参数更新。
- H_U：与H_MIX相同，只取消donor的另一任务差异要求。
- κ=0：H完全旁路，直接源A0；无用probe不必计算。

## 4. T：同当前样本跨参数运输

### 4.1 两次梯度必须对应同一个随机目标

T0使用L0；TM使用L_M。先冻结本步batch、anchor/donor及τ，构成ξ_k。在θ_k与θ_(k−1)分别求同一L(θ;ξ_k)的梯度；旧点使用前一步完整参数/必要buffer快照，不能只替换S而把其他参数保留当前值。旧点不更新模型、优化器、EMA或其他状态。

g=∇L(θ_k;ξ_k)+wθ_k；g_minus=∇L(θ_(k−1);ξ_k)+wθ_(k−1)。旧点只须返回S梯度；其他参数仍只按当前L普通Adam更新。不得用缓存的上一步batch梯度代替g_minus，不允许新旧点重新抽donor。

k=1无有效旧点时设δ=0并跳过旧点forward。step结束前保存的是本步更新前θ_k，用于下一步；不能误保存更新后的θ_(k+1)导致下一次差恒零。

### 4.2 冻结递推

在S整体计算δ=g−g_minus，δ_clip=δ*min(1,||g||_2/(||δ||_2+1e−12))。仅一个全S系数，不逐层自选裁剪，不含非S参数范数。

m=β1*m_old+(1−β1)*g+κ*β1*(1−β1^(s−1))*δ_clip。

v=β2*v_old+(1−β2)*g²；mhat=m/(1−β1^s)，vhat=v/(1−β2^s)，θ←θ−lr*mhat/(sqrt(vhat)+eps)。κ默认.25；其余参数源Adam。保留一份m/v及一个旧参数快照，不创建δ的EMA。

κ=0严格退化到当前监督目标的普通Adam：TM退化M，T0退化A0。旧点等于当前点时δ=0。若不裁剪、κ=1且前一归一化历史恰等于本步样本在旧点的梯度，则本递推归一化后等于当前梯度；这解释质量因子，不是对随机非凸MDMT的收敛保证。

### 4.3 必要控制

T0与TM构成历史开关；M与A0构成监督开关。在完全相同λ、τ、κ、lr、wd下计算I=TM−T0−M+A0；先逐seed再汇总。正交互不是晋级必需条件，TM优于最佳简单近邻才是联合增量的关键。

T_U：TM使用U_MIX donor。

T_NORM：在自身当前状态计算m_adam=βm_old+(1−β)g和上述m_transport，以同一D的范数比c=||sqrt(D)m_transport||/||sqrt(D)m_adam||，只写m=c*m_adam。若分母≤1e−12则保留m_adam并记录degenerate计数；不凭空制造方向。v及其他参数不变。这是标量历史/步幅竞争方案，不声称相同轨迹重放。

## 5. 尺度、普通增强与身份控制

SCALE(λ)使用(1−λ)L0、同lr/wd/eps和固定EMA；λ=.1时为.9。仅当M/TM使用的λ改变时补对应尺度，不能把旧c=.1结果当作本控制。它只隔离缩小真实loss系数，不完全匹配梯度范数，所以有希望候选还需要H_NORM/T_NORM。

A1直接复用v3所指源Gaussian及其EMA02V2随机key，不能改study key后称同轨迹。A0/A1的online和EMA来自同一条训练，不计为两条科学重复。DUP/τ=0/λ=0/κ=0属于接线一致性测试，不另跑完整50epoch来重新证明已知事实。

## 6. 每步执行顺序与日志

固定顺序：源loader取真实batch → 生成独立训练probe/增强索引 → 当前目标梯度 → 必要的H probe或T旧点梯度 → 检查有限值 → 一次互斥分组optimizer更新 → 一次固定EMA更新 → 释放图/临时梯度 → 写标量计数。没有真实训练数据以外的信息进入控制器。

H记录有效约束数、真实/增强cos、projection距离、历史保留范数比、两任务支持覆盖、自配率；T记录||δ||/||g||、裁剪率、校正/普通moment范数比、新旧参数距离；M记录真实/增强loss及六格权重。每步记录optimizer/EMA次数、forward样本行数、autograd调用数。默认标量汇总每epoch，首步及每20步保存细粒度诊断，不保存梯度/参数数组到磁盘。

这些诊断用于识别接线失效、退化或简单替代，不作为AUC收益的替代终点。未出现干预不是必须补造更强干预的理由，按05的有限规则停止或选择简单方案。
