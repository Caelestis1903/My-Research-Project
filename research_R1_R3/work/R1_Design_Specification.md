# R1 AO-04 DESIGN SPECIFICATION

状态：R1 设计已完成，供独立Reviewer与Main Controller最终审核。本文没有实验结果，没有授权实施、smoke test、预测评估或训练。用户限定本次只完成R1，R2/R3均禁止；以下执行要求只约束未来另行授权的研究。最终冻结口径以 `R1_Experiment_Charter.md` 为准；任何修订必须在读取新实验结果之前留痕。

## 1. Research question

在 KuaiRand natural、3 域 × 2 任务、强 Sample-M3OE 的固定协议下，仅改变他域专家之间的相对融合比例，是否存在可重复、达到实用量级且没有不可接受逐格退化的 AUC 改善？该改善是否可被固定无意义权重、简单目标域权重或相同新增参数数目的共享门重参数化/优化变化解释？

分别回答三个问题，不互相替代：

1. 描述性机制：专家、融合系数、梯度、门控是否有非均匀现象？现象本身不是性能改善证据。
2. 冻结模型后处理：验证集选比例、独立测试集验证的后处理能否改善？这是训练好的均匀模型附近的机会。
3. 重训算法：静态可学习比例是否能改善？这才是 B1/B2 训练算法比较。

不使用 R0 的删除通道阳性/阴性规则；不实施 AO-09、在线亲和矩阵、EMA、选择性置零或稀疏部署。

## 2. Hypothesis being tested

- H0-opportunity：固定协议内，非均匀他域比例不能提供稳定且达到实用门槛的宏 AUC 收益。
- H1-opportunity：至少预注册的 B1 或简单 B2 相对匹配 B0 有稳定、实用且符合退化约束的收益。
- H0-specificity：B1 的表现可由固定无意义比例B3、相同新增参数数目的共享门重参数化/优化对照Bcap，或简单目标域比例B2解释。
- H1-specificity：B1 相比 B3/Bcap 仍有稳定净收益；如果还优于 B2，则支持额外任务条件化的必要性。

B1 vs B0 为唯一主比较。B2 是预注册的简单替代比较，不允许 B1 失败后把 B2 自动升级为主比较。B1 成功不证明亲和估计、来源语义、在线学习或论文创新有效；B2 足够时优先停止复杂化。

## 3. Variables changed

| 臂 | 唯一结构干预 | 新增可训练标量数 | 初始化 |
|---|---|---:|---|
| B0 | 原样均匀他域回收 | 0 | 历史适配器的原样初始化 |
| B1 | 每个目标域 × 任务一个他域相对比例 | 6 | ratio raw logit=0，比例=0.5 |
| B2 | 每个目标域一个比例，两任务共享 | 3 | ratio raw logit=0，比例=0.5 |
| B3 | 每格固定随机比例，跨训练 seed 复用同一表 | 0 | 独立随机表，不与 B0 前向等价 |
| Bcap | 每格共享专家门 softmax 的一个温度 | 6 | log-temperature=0，temperature=1 |

B3的目的只是排除一次预注册、无关系信息的非均匀结构扰动；一个随机表的阴性不能排除所有随机权重。Bcap检验六个新增活跃参数带来的门控尺度/优化变化这一竞争解释；温度可被原门线性权重/偏置吸收，因此它不新增六个可辨识函数类自由度，不是同有效容量的对照。相同参数数目不等于相同函数类，不得声称它排除了函数类容量混杂。

## 4. Variables held constant

以代码审计确认的 S1 强 Sample 适配器为行为基准，不直接使用可能存在优化器分组缺陷的原始 trainer 替代：

- KuaiRand natural；3 域、2 任务、4 共享专家、3 域专家、2 任务专家。域、任务、标签映射和顺序必须按数据审计 manifest 固定，不根据结果改名。
- 同一文件与全局行序 80/10/10 后按域过滤的 train/validation/test split；数据 hash、行 ID、特征编码、标签规则必须锁定。
- batch size=40960，ordinary shuffled concatenated train loader，drop_last=False，num_workers=0；所有臂使用相同 seed 的逐 batch 样本与顺序。
- embedding dim=16，fcn_dims=[512,256,256,64]，共享/域/任务专家、STAR 路径、六个塔、LayerNorm、共享门 `emb.detach()` 均不变。
- 原四个 global sigmoid raw scalar 均按历史初始化 raw=1；其优化器和更新节律不变。
- 主 Adam lr=0.01，weight_decay=1e-5；四 scalar 的 arch Adam lr=0.001，weight_decay=1e-5；betas=(0.9,0.999)，eps 等默认项必须从审计结果写入正式 manifest；无 scheduler。
- 50 epochs 全部训练；patience=10 仅历史 shadow early-stopping 记录，不实际截断。不能用 stage_r1 通用 patience 条款覆盖已确认的强 baseline 实际协议。
- 每 epoch 以 validation 六格等权 macro AUC 选 checkpoint；最大值相等选 earliest epoch；所有臂同样规则，最多 50 个候选 checkpoint。selected 与 final 分别保存。
- 不增加目标域采样、损失重加权、源替换、冻结专家、梯度投影、momentum 干预、EMA、curriculum 或额外正则项。
- 共同 backbone 的初始 state_dict、初始化 RNG 消耗、训练 batch RNG、architecture batch 序列在 paired seed 下相同。新增参数是常数初始化，不消耗 backbone/loader 随机流；B3 使用单独生成器。

Sample objective 精确定义为 `L=1/(B*T) sum_i sum_t BCE(p_i,t,y_i,t)`，即样本 × 任务平均，各域贡献随实际样本数变化；不能把它替换为六格 BCE 等权平均。目标域缺失的 batch 不人工补齐；架构更新也用同一个 train Sample objective，禁止用 validation 学参数。所有臂 train 参数的学习目标是 Sample BCE，选模目标是 validation macro AUC，二者区别要明示。

历史selected checkpoint被报告提及不代表实际存在。当前代码审计确认S1源码/data模块可读且hash匹配，缺的是checkpoint、optimizer/RNG等二进制state。本设计不把旧预测npz当checkpoint，也不从指标表重构权重。实际checkpoint缺失，所以未来预算按重新训练全部匹配B0计算；源码可复用不等于历史模型state可复用。

## 5. Mathematical/algorithmic definition

### 5.1 实际融合图

对目标域 d、任务 t、样本 x，令 j1(d)<j2(d) 为另两域按物理 ID 升序排列，域专家 `E_j(x)` 的输入仍是 x 的目标域 STAR 表示。E_j 是模块索引，不是一次 forward 从源域样本读入的信息。

`S_d,t(x) = sum_n softmax(z_d,t(x))_n * H_n(x)`。

`F_d,t = S_d,t + aD * [bD*E_d + (1-bD)*(rho_d,t*E_j1 + (1-rho_d,t)*E_j2)] + aT * [bT*U_t + (1-bT)*U_(1-t)]`。

`aD,aT,bD,bT = sigmoid(四个既有 raw scalar)`，均为跨格 global，不是逐域或逐任务参数。T=2 时他任务仅一个，任务侧不存在可重分配的多个他任务比例。

B0：rho=0.5。B1：rho_d,t=sigmoid(q_d,t)。B2：rho_d,t=sigmoid(q_d)。B3：固定 rho_d,t，不求梯度。

每格域融合的自域系数仍是 bD；他域系数之和严格为 1-bD。与 aD 相乘后为 aD*(1-bD)。这是系数总和约束，不保证专家输出向量范数、信息量或梯度大小相同。bD 的定义不变，但重训后的 bD 数值允许按既有训练路径自然变化；不能宣称其值被固定。

### 5.2 Bcap 的六个活跃重参数化/优化对照参数

保留每格原 `Linear -> Softmax` 的线性层参数与输入 detach，只在原 softmax 之前插入温度：

`tau_d,t = exp(theta_d,t)`，`theta_d,t(0)=0`。

`S^cap_d,t = sum_n softmax(z_d,t / tau_d,t)_n * H_n`。

域与任务融合保持 B0。六个 theta 都进入对应格共享专家门的 forward，不能用虚设、无梯度的 dummy 参数凑数量。温度作用于共享专家 logits，不能改为最终预测概率的温度。不存在新增裁剪、范围搜索、gate prior、温度正则或额外门网络；非有限 tau/梯度按统一数值故障规则处理。

theta=0时Bcap与B0前向等价；B1/B2的q=0时同样等价。归一化权重总和不变，但温度改变了共享专家相对路由，必须报告共享门熵。由于 `softmax((W*h+b)/tau)=softmax((W/tau)*h+b/tau)`，tau可吸收进每格既有gate Linear，Bcap与B0的可表示gate函数类相同，但Adam路径、尺度与weight_decay几何不同。因此这是同新增参数数目的活跃重参数化/优化对照，而非六个新增可辨识函数自由度的容量匹配。Bcap不主张可替代任意域专家相对比例，B1优于它仅排除这一所测优化/尺度替代。

### 5.3 新参数的优化器归属与节律（新预注册建议）

新增 q 或 theta 统一归入主 Adam：lr=0.01、weight_decay=1e-5、betas=(0.9,0.999)，每个普通 train batch 更新一次，与原主干共用 Sample objective。B1、B2、Bcap 完全相同规则。B3 的固定比例注册为不可训练 buffer，任何优化器不得持有它。

原四 global scalar 仍只归入既有 arch Adam：lr=0.001、每 epoch 一个训练 batch 更新一次。新 q/theta 不加入 arch optimizer，arch step 不更新它们；普通主 step 不更新四 global scalar。每次阶段切换清空全部梯度，保证没有上一步梯度积累到下一步。允许计算未更新组的梯度以保留历史 forward/backward 行为，但只能由指定 optimizer 写入参数。

理由：沿用主 Adam 不增加学习率候选，且所有新增活跃参数得到普通训练批次的完整更新预算。若把 q 放进仅 50 次、lr=0.001 的 arch 更新，从 q=0 出发可能只探索非常窄的 0.5 邻域，使阴性结果难以区分“无机会”和“优化器探索不足”。本条是未来设计选择，绝不是历史 baseline 已有事实；不得试跑比较两种归属后择优。采用此规则后，阴性也只限于这一预注册参数化/优化协议，不否定所有可学习共享方法。

### 5.4 B3 固定随机表

使用独立 `numpy.random.RandomState(10404)`（MT19937），NumPy版本2.1.3，一次按task-major `i=t*3+d` 顺序抽取6个 `Uniform(0.1,0.9)` ratio。本次仅生成并锁定配置常数，没有加载数据、执行模型或计算研究指标：

| task | target domain | rho（升序第一个他域的比例） |
|---:|---:|---:|
| 0 | 0 | 0.7502780865025745 |
| 0 | 1 | 0.8537642404691582 |
| 0 | 2 | 0.44953240522281324 |
| 1 | 0 | 0.2525338669196546 |
| 1 | 1 | 0.23434256306102627 |
| 1 | 2 | 0.765217885741144 |

canonical JSON是按上表顺序的float列表、UTF-8、无空格，SHA256=`182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957`。未来直接读取此固定表，不依赖再次抽取；两个训练seed复用。禁止重新抽取、挑选方向、按域规模翻转、从validation选随机表或用训练seed决定新表。

### 5.5 冻结比例扫描（独立后处理比较 P0）

仅使用每个B0 seed已按原均匀模型验证规则选好的selected checkpoint，冻结全部参数和模型状态，eval mode。开启扫描须先通过B0重建代码与协议的replay验收，包括state、loss、split、optimizer和数值检查；不是看新test能否复现旧数字后才选择开启。每目标域d使用一个比例rho_d，两任务共享；候选grid={0,0.1,...,1}。先限定可行集合：该域两个validation task相对同checkpoint原rho=0.5的AUC差各自均须>=-0.003；再在可行集合内最大化两任务validation AUC等权均值。无可行非均匀项则取0.5。原rho=0.5必须包含，作为始终可行的fallback。选checkpoint先于扫描；不得每比例重新选epoch。

因为 frozen forward 中目标域 d 的输出只依赖 rho_d，目标域选择可以因子化为 3×11 次目标域候选评估，但最终选择空间为 11^3=1331 个可组合向量。必须披露该多配置验证搜索；不能称为“零调参”。若实现显示 cross-target 耦合而无法因子化，须在实施前改设计并独立复审，不能悄悄改联合扫描。

扫描使用双精度聚合存储的validation AUC；与最大值差不超过1e-6视为并列，优先0.5，再选离0.5最近、数值较小。此容差只用于新比例扫描，不改历史checkpoint earliest-max规则。

每 seed 的三个所选 rho 拼成一个向量并冻结 manifest，然后独立 test 仅评估该向量与同 checkpoint 的 rho=(0.5,0.5,0.5)。不扫描 test，不汇报 test 最优比例，不以 P0 结果选择 B1 初始化、学习率、对照或臂。grid 端点保留总他域质量却让其中一专家系数为0，这是比例重分配而不是无补偿通道删除；不能从端点通道损失推出均匀最优。

## 6. Exact affected modules/files if identifiable

当前可辨认的事实来源：

- `m3oe/MDMTRec/models/multi_domain/model.py`：`MDMTRec` 类、共享门、域/任务专家、四 sigmoid scalar 和 task-major 融合映射；保留原文件。
- `experiments/stage_S2_run.py`：对 S1 runner/model/metrics 的 imports，以及主/架构更新的历史合同；这是审计参考，不是本实验的 Equal 训练入口。
- `experiments/stage_S2_protocol.py`：历史强适配器的固定值与 shadow patience；Sample 定义以 S1 审计为准，不沿用 S2 Equal loss。
- `research_R1_R3/work/R1_Code_Audit_Inventory.json`：当前checkpoint、文件清单及runtime可用性审计；S1源码/data模块可读且hash匹配，缺的是checkpoint等二进制state。
- `experiments/stage_S1_models.py`、`stage_S1_run.py`、`stage_S1_data.py`、`stage_S1_metrics.py`、`stage_S1_protocol.py`：实际存在且最终审计hash匹配的历史适配器/data/metrics/protocol，允许未来严格复用其已审计行为；缺的是模型与optimizer/RNG二进制state。

未来若获授权，建议隔离文件为 `research_R1_R3/src/ao04_model.py`、`ao04_protocol.py`、`ao04_runner.py`、`ao04_frozen_scan.py` 和固定JSON配置；这些是拟定文件，当前不得创建或实现。不覆盖S0–S8代码、报告、checkpoint或数据。源码/data模块严格复用与历史checkpoint不可直接复用须分别标记。

## 7. Implementation requirements

以下均是未来审核要求，本次不执行：

1. 定位并复核强适配器来源、代码 hash、data hash、依赖版本及损失规则；受阻就报告，禁止猜测恢复。
2. B0 精确保留所有原 forward/梯度 detach/设备 dtype 路径、参数分组和训练协议；B1/B2/Bcap 改动只能在本文指定位置。
3. 把原 backbone state 显式复制到 paired 臂，记录逐参数 hash。B1/B2/Bcap 初始化与 B0 全六格特征和输出 allclose：FP32 atol=1e-6、rtol=1e-5；更严格 bitwise 结果若可得同时记录。不能让附加参数改变初始化 RNG。
4. 小型固定输入包含所有3域、两任务有效标签；检查 task-major 索引、mask 路由与尺寸。真实 batch 检查范围有限、所有权重归一化、自域/他域总系数约束。
5. 以非退化的合成专家输出、gate logits 和标签分别验证 q/theta 有非零有限梯度；检查一次 main step 实际改变新参数且不改变四 arch scalar，一次 arch step 只改变四 arch scalar。某真实 batch 缺域导致对应 q 无梯度不能误判缺陷，也不能因此改变 sampler。
6. 原参数各归属唯一 optimizer，交集为空、并集为全部可训练参数；记录新增数量和有效梯度参数数。不得因参数名字匹配而将新 q/theta 同时加入两个 optimizer。
7. 一致保存 selected、final、共同 epoch checkpoint、两个 optimizer state、RNG state、配对 batch hash、完整配置。不得用最后 epoch 替代已选 checkpoint。
8. B3 固定表作为 buffer 保存、加载往返一致、优化器外且不调用新的训练 RNG。Bcap 的 tau 必须进入 gate forward 并实际可更新；报告温度而非只报告 theta。
9. 所有十个 run 的 validation 选模与 P0 所选向量在访问新 test 结果之前冻结，先写不可变 selection manifest，再解封 test。已经阅读的历史 test 表只能作为历史复现参考，不能选择新臂或阈值。

## 8. Ablations required

全部五臂为必要对照，不把 B2/B3/Bcap 留作结果好后才追加。无额外训练消融。P0 是 B0 的冻结后处理测量，不是第六个训练臂；不训练 ratio、不改变 checkpoint 或数据。

解释矩阵：

- P0 正、B1/B2 负：存在冻结后处理机会，但当前训练参数化/优化没有兑现；不推出在线亲和必要。
- P0 负、B1/B2 正：均匀 checkpoint 局部扫描不足以否定训练联合适配；不得称两者矛盾。
- B1 与 B3 相近：本随机结构干预可解释表现，任务/源关系解释弱。
- B1与Bcap相近：所测六个新增活跃参数的门控重参数化/优化解释未被排除。
- B2 达标且 B1 相对 B2 无预注册净收益：简单目标域权重足够；不开发任务维度或在线模块。
- 全部阴性且协议有效：停止本先导，不增加新 dataset、lr、扫描网格或训练时长；只限当前预算和协议。

## 9. Expected logging requirements

每 epoch、seed、arm 记录：

- train Sample BCE、validation 六格 AUC/LogLoss、六格 macro AUC/LogLoss；原 global raw 与 sigmoid，q/rho 或 theta/tau；他域完整系数、每格共享门熵；新参数梯度/更新 norm、主/arch step 数、每个新增参数实际更新数、batch hash、非有限事件。描述性饱和标记：rho<0.01或rho>0.99，tau<0.1或tau>10；标记不触发改lr、clamp或重启。
- validation 选模分数、selected epoch、earliest tie 判决、shadow patience 首次触发 epoch、final epoch；选模是否落在49 epoch上界的 censoring 标记。
- train/val/test 每域和每格样本数、positive数、positive rate；缺少正类或负类时 AUC 不可用标记，不能补0.5。
- 参数总数、原/新/活跃参数数、FLOPs估计口径、实测训练墙时、推理耗时和峰值显存；五臂使用同硬件环境与计时方法。不能将一次噪声计时当实质效率结论。
- 数据/源码/配置/依赖版本、所有checkpoint与输出hash；完整成功/失败状态与异常；故障也占尝试预算并留痕，不择优隐藏。

最终每格与每 seed 独立 test AUC/LogLoss、各臂相对 B0 的 paired difference；宏 AUC、宏 LogLoss、样本加权逐格 AUC、pooled task AUC 分开命名。

主指标 `M = (1/6) sum_d,t AUC_d,t`。guardrail所用加权逐格AUC定义 `W=sum_d p_train_d*(1/2)sum_t AUC_d,t`，其中 `p_train_d=n_train_d/N_train` 在数据manifest冻结、所有validation/test臂相同，必须称“训练域份额加权逐格AUC”，不能称混合总体AUC。如果另报告test份额加权结果，单列为 `W_testshare`、只作描述且不进入guardrail。若报告pooled task AUC，则每任务直接拼接三域标签/预测计算，并明确跨域排序影响。域宏、任务宏、最差格、最大退化格、改善格数量一并报告。LogLoss按历史指标实现，clip epsilon=1e-7；不偷偷改概率裁剪。

两 paired seed 报告均值、样本标准差（ddof=1）、两条原始差值与符号；不报告统计显著、不把 bootstrap test 样本区间当训练 seed 证据。可报告样本不确定性时必须另列、说明其未覆盖训练随机性，不能替代多 seed。

P0 记录完整 validation grid 曲线、所选 rho、tie 判决、候选数、空间1331、freeze时间/hash，仅所选配置的 test 表。

## 10. Acceptance criteria for reviewer

### 10.1 R1 审核与当前停止点

Reviewer 可在完全没有新实验结果的情况下判断规范是否自洽；确认架构公式、历史协议、控制公平性、泄漏边界、预算与阴性路径已固定。若历史代码/数据/checkpoint尚不可执行，应把 R1 文件标为审计完成但后续执行就绪性受阻。PASS 不等于训练授权。当前交付完 R1 后必须停止，不触发下面的执行预算。

### 10.2 未来实施审核门槛

所有第7节约束、参数分组、初始等价、归一化和梯度更新审核通过；任何阻塞项修复后复审。PASS WITH FIXES 只有全部 blocking fixes 验收后才能用于未来运行；INVALID 运行不能被纳入“稳定阴性”。

### 10.3 预注册运行预算

- 当前允许 full runs=0、smoke=0、预测评估=0。
- 未来单独授权后，最多五臂×paired seeds(2022,2023)=10完整训练attempt（失败也计入），每次最多50epochs，总上限500run-epochs；不授权其它seed或未来确认阶段。
- 首个 seed 在不同臂间不能择优停止、改配或追加；完整配对矩阵按冻结顺序执行。不能依 first test 结果决定是否做第二 seed。
- P0 最多两份 B0 checkpoint×3目标域×11比例 validation 候选，以及每seed一次所选vector的test；不做更多checkpoint/更密grid/逐任务扫描。
- 数值或基础设施故障标FAILED/INCOMPLETE，保留日志，无额外完整重试、禁止第11个完整训练attempt。只有同配置、同已保存完整model/optimizer/RNG/state精确续跑才作为同一次attempt，且基础设施resume全队列最多2次。没有完整state不能从头重跑却称resume；全部实际run-epochs合计不得超过500。无法完成矩阵时结论限于已完成证据，缺失不能算阴性。
- 任一参数、tau、预测、loss或梯度出现nonfinite，立即停止当前run及后续队列并标数值INVALID，交Reviewer排查；禁止性能触发温度clamp、事后降低新参数lr或删除故障seed后继续判正。外部基础设施中断与数值INVALID分开记录。

### 10.4 预注册性能与退化门槛（实用筛查，不是显著性）

必须明确分开 `delta M_val`（所选checkpoint的validation宏AUC差）与 `delta M_test`（预先冻结配置的独立test宏AUC差）。方向保留/停止只采用validation筛查，validation含选模乐观性，所以只能作为筛查信号。下面同一组门槛可在validation用于筛查，在test用于最终“是否独立验证”判决，两份结果分别报告，不能互相替代。全部十run选模及扫描向量冻结后才能解封新test；test判决不得决定增加臂、换dataset、改阈值、追加seed或进入R3。本合同不授权后续扩展，本次只制定规则，不作方向GO/STOP判断。

以下数值在看新数据之前固定，不从历史方差反推：

1. B1 机会信号：两seed `delta M(B1-B0)>0`，均值 `>=0.003`。0.003沿用任务文件明确讨论的实用筛查量级，不是通用统计标准。
2. 六格均为受保护格：每格paired AUC差的两seed均值 `>=-0.003`，任一seed单格 `>=-0.005`；每任务三域宏AUC差的两seed均值及W差的两seed均值各 `>=-0.001`。宏LogLoss差的两seed均值 `<=+0.005`，每格LogLoss差的两seed均值 `<=+0.010`。这些是与正式Charter统一的资源筛查阈值，非领域标准；任何越线禁止整体GO，不能因格样本小而豁免。
3. 排除本简单控制解释：B1 相比 B3、相比 Bcap 的宏AUC paired 差值各须两seed为正、均值 `>=0.001`。0.001是预注册的“控制可解释余量”，不是显著性。主比较达标但此项不达标，证据至多 CONDITIONAL，不能把微小控制差当机制证据。
4. 若主张任务条件化必要：B1 相比 B2 同样两seed为正且均值 `>=0.001`；否则即使B1达标也优先简单B2，不升级在线模块。B2 的收益按与B0相同机会/退化阈值判断，但独立列为次比较。
5. 无 seed 选择、无“任一格达到0.003即成功”、无把50epoch中test最好epoch当结果。五臂、多比较、多验证配置的存在必须披露；两个seed不支持 STRONG 或统计显著结论。
6. 训练/推理成本完整报告，不新增Charter之外的硬性耗时阈值。计时不可信/硬件不匹配时标成本证据不足，不把它偷换为AUC阴性。

预先给判决含义：validation GO只指“先导符合全部门槛的有限阳性信号”，仍不确认方法有效、论文贡献或授权扩展；test最终报告另写独立验证通过/未通过。CONDITIONAL用于有效但seed不一致、控制差不够、P0与重训不一致等，不授权补跑。STOP用于完整有效矩阵未达实用收益、退化越线、简单对照足够解释、或成本超预算；不宣称理论被永久证伪。INVALID用于实现、泄漏、数据或协议错误；INCOMPLETE说明未完成而不是无效/阴性。

未来训练算法效益证据最多WEAK（两个训练seed），与Charter统一；R1结构/文件事实可按其独立核实质量标更强，但不是算法有效性证据。域规模可由样本数描述，域难度用B0逐任务AUC、loss、标签分布及其限制描述；不把最小域自动叫最难域。natural只有3域，不能以三点相关确定域规模是主因，不能用小域改善掩盖大域退化。

## 11. Conditions that would falsify the hypothesis

- 有效完整预算中，B1/B2均没有重复实用宏AUC收益，直接削弱本预注册“静态非均匀可利用空间”假设，按预算停止。
- 改善只在一个seed出现，另一seed反向；不支持稳定收益。
- B1收益由B3或Bcap达到相当水平，削弱“源域关系具有额外信息”的归因；结构扰动、门控尺度或优化路径解释保留。
- B2与B1相当或更好，否定“额外任务条件化在当前设置有必要”的强主张，支持先用简单比例；不否定整个非均匀分享机会。
- 收益只靠不可接受的逐格AUC/LogLoss退化，否定所声明的整体实用收益。
- P0在validation好而独立test无收益，否定这一冻结后处理收益；不单独否定重训算法。P0/B1两路均阴性才更强地削弱当前机会，但仍限于协议与预算。
- ratio饱和、ratio约0.5或β接近1仅是描述性现象；没有性能对照不能单独判阳性/阴性。新增ratio与β、专家尺度存在共同适配/重参数化，不能从学得ratio直接认定真实域亲和。
- 数据/训练目标、共享门已有间接任务差异、专家共同重训、β漂移、验证搜索和同参数非等函数类均是竞争解释，必须在最终分析里保留；出现训练协议或泄漏错误只能 INVALID，不能当科学反证。

R1 最终应产出经独立审核的研究合同，并明确后续执行尚未开始。本文没有任何实测阳性、阴性或 GO 判决。
