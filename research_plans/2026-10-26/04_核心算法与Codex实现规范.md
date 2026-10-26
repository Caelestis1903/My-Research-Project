# 核心算法与 Codex 实现规范


规范 ID：`DFD-v4.0`；制定日：2026-10-26；状态：**设计已锁定，尚未实现或测量新方法**。本文件定义唯一主方法 `DFD` 和必要对照。运行数量、统计与停止规则见[05](05_完整实验矩阵与研究决策.md)。不得在看到结果后改写本文件的科学语义。

## 1. 问题、监督与计算对象


三域 `s,t∈{0,1,2}`，两任务 `q∈{0,1}`。样本为 `(row_id,x,d,y0,y1)`，每行保留原始实际域 `d`。任务顺序、列名和编码必须来自核实后的 manifest；不能用任务名称推断标签逻辑关系。

原生 fixed-fusion MDMTRec 给出该行实际域的两维 logit `z(x)∈R²`，四个共享专家输出 `E(x)∈R^(4×64)`。定义

$$
h(x)=\operatorname{concat}_{e=0}^3 \operatorname{LN}_0(E_e(x))\in\mathbb R^{256},
$$

`LN_0` 无可训练 affine，eps=`1e-5`。`h,z` 来自同一个已经冻结的原生 checkpoint；此后所有阶段都不能更新或重新估计它。原生模型中的既有 LayerNorm、gate 的 `emb.detach()` 和 fixed fusion 保持原义。

函数原型定义为一个**独立的 source/task 监督函数**，不是中心向量，也不是低秩分块。`F_{s→t,q}` 用来源 `s` 的任务 `q` 标签训练，`t≠s` 表示形成时的目标意图；同一 source/task 有两个函数。目标意图是否形成不同功能是待检验假设，而非命名即成立。

本方法的额外信息是来源监督形成的参数知识。固定 `h` 后各输出都是 `h,z` 的确定函数，不能称为新增的逐样本观察信息。需求差由普通训练标签导出，不是独立标注或潜在结果的因果识别。

## 2. 数据角色与共同协议


### 2.1 唯一允许的数据角色


沿用自然协议的原始行、fit-only 编码、外层 validation/test 及角色 manifest。禁止为本方法重新平衡领域、复制稀缺域或增加噪声。

| 角色 | DFD-v4.0 用途 | 禁止用途 |
|---|---|---|
| `fit` | 原生训练、pilot、source 函数形成；特征 normalizer 与几何中心 | 不可混入旧外层 validation/test |
| `P_form` | 条件概率/需求学习、两折诊断、全池需求 refit；所有软读取对照共同的20 pass预拟合 | 不直接反传原生、pilot 或 source 参数；不能成为“未见测试” |
| `P_read` | 固定库上的读取器/常量系数拟合 | 不更新库；不与 P_form 调换身份 |
| `P_cal` | 选择整组六个读取头的完整状态，或 OFF；属于模型选择 | 不逐 cell 拼接最优头；不训练需求或来源函数 |
| `P_update` | 主方法不用；旧 RHD 的有限诊断可按原角色使用 | 不为补样本量并入新方法 |
| `Q_explore` | 预注册的开发机制和晋级判断 | 不参与拟合、epoch 或策略选点 |
| `validation` | 原生 checkpoint 选择；最终预测器结果并列报告 | 不拟合函数、需求或读取器 |
| `Q_legacy_confirm_exposed`,`test_exposed` | 方法和运行清单锁定后的完整历史基准复评 | 不调参、不换 seed/指标、不宣称新数据确认 |

fit 行数：ML `[124759,254879,260495]`，共 `640133`；KR `[138225,1395118,71561]`，共 `1604904`。P_form/P_read 总行数分别 ML `24004/24006`，KR `60183/60184`。P_cal 为 ML `16003`、KR `40123`。这些是 v3 公开报告值，执行时需核实实际 hash 与逐格标签支持，不能直接当作已读取服务器数据。[v3 设计](https://github.com/Caelestis1903/My-Research-Project/blob/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0/research_reports/2026-10-25_v3/DESIGN_SPECIFICATION.md)

优先恢复既有 manifest。确需复建时采用已发布整数协议：原始外层按原行序 `a=(8*N)//10,b=(9*N)//10` 得 train/val/test；train 内每域 row_id 升序后 `PCG64(20261024+1009*d)` 排列，以相同 80/90 整数切点分 fit/probe/query。probe 再以 seed 偏移 `20000` 排列，60/80 整数切点分 P_label/P_cal/P_update；query 偏移 `40000` 排列，前 `n//2` 为 Q_explore。P_label 每域 row_id 排序后 `PCG64(20261025+60000+1009*d)` 排列，前 `n//2` 为 P_form，余为 P_read。若字节或行身份不能匹配旧版本，登记 `PROTOCOL_MISMATCH`，不得继续声称沿用同一自然协议。

### 2.2 目标聚合


原生训练与 pilot 均使用自然样本两任务平均：

$$
L_{\rm sample}=\frac{1}{2|B|}\sum_{i\in B}\sum_{q=0}^1\ell(z_{iq},y_{iq}),
\quad \ell(a,y)=\operatorname{softplus}(a)-ya.
$$

`Equal` 的每域组内均值与该目标不同，本轮不设 Equal 主臂，也不把二者等同于全局学习率。评价的 macro LL 为六个 domain/task 均值等权，另外必须报告自然总体 LL；评价目标与训练采样权重不同是显式设计。

逻辑 batch=`40960`，drop_last=False，ML/KR 每 epoch=`16/40` 步。允许 microbatch=`4096`，每个 microbatch 的和除以**原完整逻辑 batch 的分母**，累积后只 step 一次；末 batch 用其真实行数。不得每个 microbatch 单独求均值或独立 Adam step。

### 2.3 固定超参数和随机流


| 组件 | 次数与选择 | 优化器与参数 |
|---|---|---|
| 原生 | 固定 24 epochs；在 epoch12–24 按 validation macro LL 最低选，精确平局最早；保留 epoch24 | Adam lr=.001，betas=(.9,.999)，eps=1e-8，耦合 wd=1e-5 |
| 6 个 pilot | 固定 12 个完整 fit passes，无早停/选点 | 同上 |
| 每折及最终 η | 每个目标任务固定 20 passes，batch512；无结果驱动选点 | Adam lr=.001，betas=(.9,.999)，eps=1e-8，wd=0 |
| 来源形成 | 固定 12 个完整 fit passes，无早停/选点 | Adam lr=.001，betas=(.9,.999)，eps=1e-8，耦合 wd=1e-5 |
| 读取器 | 先 P_form 20 passes，再 P_read 10 passes，batch512；P_cal 只选预拟合后完整状态或 OFF | Adam lr=.001，betas=(.9,.999)，eps=1e-8，wd=0 |

所有梯度计算 FP32，无 AMP、无 TF32；风险标签/概率混合风险和汇总以 FP64 计算。不得根据吞吐改为半精度后继续沿用同一方法 ID。需记录软件与设备版本，但旧服务器的 torch/CUDA 路径是证据，不是当前环境已验证依赖。

开发 seeds=`5101,5102`；锁定复评 seeds=`6101,6102,6103,6104`。P0 检查这些 seed 是否已被用于本方法；若已运行，停止该身份的正式启动并记录冲突，不自动换 seed。不同数据使用同 seed 编号但分离 RNG。

所有派生 RNG 用 `SeedSequence([run_seed, dataset_id, namespace, source_id, task_id, component_id, pass_id])`，dataset_id 为 ML0/KR1；无关坐标填0。namespace 固定：native-init=10、main-plan=20、pilot-init=30、pilot-plan=31、η-init=40、η-plan=41、formation-perturb=50、formation-plan=51、shuffle-control=52、geometry-subset=53、geometry-fit=54、read-init=60、read-plan=61、bootstrap=70。主计划、缓存、诊断不得消费同一 RNG。两折身份独立于模型 seed：每域 P_form 排序后 `PCG64(20261026+70000+1009*d)` 排列，位置模2为 fold；两个任务共用 row fold。

## 3. 阶段 A：原生冻结与来源 pilot


### 3.1 原生身份


固定底座：三域、两任务、4 shared experts，`fcn_dims=[512,256,256,64]`，`exp_d=exp_t=bal_d=bal_t=1`，不恢复 learned fusion/AutoML/DARTS。该配置下 bal_d=1 不等于模型没有跨域共享；embedding、STAR/shared experts 等仍由多域训练。

原生训练 24 epochs 后读取预定 LL-selected 完整状态，保存其 model/optimizer/RNG/plan/selection 身份，然后冻结全部参数并设 eval。原生 epoch24 作为独立输出；不从多个 backbone checkpoint 分别训练整套下游、再选最优组合。

缓存各角色实际域的 `h[rows,256],z[rows,2]`、row_id、domain 和标签访问权限。train/eval mode 在缓存前后无隐式变化；cache key 包含 backbone 权重、encoder、输入行和特征提取函数 hash。`h` 是四个 shared-expert 输出的归一化拼接，不是 STAR 中间 256 维向量；两者维度相同不能混用。

### 3.2 六个来源监督函数


每个 `(s,q)` 独立 MLP：`f_s,q:256→32 ReLU→1`，含 bias，共 `8257` 参数，6 个为 `49542`。输入为冻结 h；隐藏层 Xavier、bias0，末层 weight/bias0。每个函数初始残差为0，但隐藏层非零，避免双零导致失活。无 dropout、无 BatchNorm、无新 embedding。

$$
a_{s,q}(x)=z_q(x)+f_{s,q}(h(x)),\qquad p_{s,q}(x)=\sigma(a_{s,q}(x)).
$$

训练时只在 `d_i=s` 行上更新 `f_s,q`，自然目标为

$$
L_{\rm pilot}=\frac{1}{2|B|}\sum_{i\in B}\sum_q\ell(z_{iq}+f_{d_i,q}(h_i),y_{iq}).
$$

不经过 foreign pilot，也不让梯度回到 z/h。12 passes 结束后永久冻结全部 pilot。当前目标域 t 的 `p_t,q` 是强 full-h 本地基线 `LOCAL`，原生 `σ(z_q)` 是 `NATIVE`；两者均必须独立报告，不能只选更差者作为参照。pilot 不利用 P_form/P_read/P_cal 标签，因此能在这些角色上定义没有本阶段直接标签拟合的读取差。

## 4. 阶段 B：预测目标需求


### 4.1 明确可观察的读取差


对目标 t、任务 q 和 foreign source s 定义固定半读取候选

$$
p_0=p_{t,q}(x),\qquad p_{a(s)}=\tfrac12p_{t,q}(x)+\tfrac12p_{s,q}(x),
\quad a_0=\operatorname{logit}(p_0),\quad a_1=\operatorname{logit}(p_{a(s)}).
$$

不以未经校准的纯 source 替换作为唯一机会，也不搜索读取强度。P_form 上可计算配对差

$$
d_{s\to t,q}=\ell(a_0,y_q)-\ell(a_1,y_q).
$$

正值表示这个固定状态/固定半读取动作对该行的 BCE 更低。它是模型输出的干预差，不是用户行为处理效应；一行的两个 source、两个任务差不算独立新样本。

### 4.2 用条件概率解析需求


每个 `(t,q)` 只拟合一个 η 头，两个 foreign source 共用同一个条件概率估计：

$$
v_q=[h_{256},z_0,z_1,f_{0,q}(h),f_{1,q}(h),f_{2,q}(h)]\in\mathbb R^{261},
\quad \eta_{tq}(v)=\sigma(\mathrm{MLP}_{261\to32\to1}(v)).
$$

每头 `8417` 参数，共 `50502`，仅训练期使用。隐藏 Xavier/bias0；末层 weight0，bias 为该拟合 fold 的任务正率 logit（clip 到 `[1e-4,1-1e-4]`）。使用普通 BCE；不以 oracle 动作作为分类标签，不引入效用网络的额外自由度。

v 的均值/std 来自**目标域 fit 的冻结 pilot 特征**，std floor=`1e-3`，标准化后 clip 到 `[-10,10]`；该 normalizer 对两折、最终 refit 及 source 外推完全相同。记录每维及每角色裁剪比例，不能根据 Q/test 改正则或归一化。

由 BCE 恒等式得到

$$
\widehat\mu_{s\to t,q}(x)=\operatorname{softplus}(a_0)-\operatorname{softplus}(a_1)
-\eta_{tq}(v_q(x))(a_0-a_1).
$$

μ 的监督信息仍是 P_form 的 y。所有概率/logit 转换与需求计算 FP64；为了避免概率数值恰为0/1，生成诊断 logit 时统一 clip 概率到 `[1e-7,1-1e-7]`，实际 d 与解析 μ 必须使用同一 a0/a1 定义。训练 reader 的 BCE 不使用这种诊断裁剪，见§6。

### 4.3 两折诊断、最终 refit 与中心化


对每个目标域按固定 row fold 划分 P_form。训练 fold0 的 η 后只在 fold1 产生 OOF μ，反向亦然；每次 20 passes。每个 P_form 行的 OOF 预测都来自没看该行标签的 η；pilot/backbone 也没有见 P_form 标签。用户/物品可跨 fold 重复，此过程只排除该阶段同行标签复用，不宣称用户隔离或总体独立。

诊断完成后，若数据集通过 P1，六个 η 从其规定初始化开始在全部 P_form 上再训练 20 passes；部署形成使用该全池模型。不得把 refit 的 in-sample μ 填入 OOF 记录。

对每个有序 pair `(s,t,q)` 保存

$$
m_{s\to t,q}=\frac1{|P_{\rm form,t}|}\sum_i d_{s\to t,q,i},
\quad b^{\rm OOF}_{s\to t,q,i}=\widehat\mu^{\rm OOF}_{s\to t,q,i}-m_{s\to t,q}.
$$

`m` 是全池已观测配对差的均值，仅用于后续形成中心化；OOF 策略评分不使用全池标签均值调动作。两个目标的 b 数组等权合并（每个目标总权1/2），取总体加权标准差定义

$$
\tau_{s,q}=\max(10^{-3},\operatorname{SD}_{\rm equal-target}(b^{\rm OOF}_{s\to t_0,q},b^{\rm OOF}_{s\to t_1,q})).
$$

其方差须围绕该合并数组的加权总均值计算，不把平方均值当方差。m 和 τ 一次冻结；不根据形成结果调整。中心化用于降低“某个目标整体更易改善”独占来源的风险，并不证明已经消除全部基率/难度混杂。

## 5. 阶段 C：需求条件来源函数形成


### 5.1 样本责任及自然总质量


对 source s 的实际 fit 样本 x，列出两个 intended targets `t0<t1, t≠s`。**保持 x 的实际 domain=s 不变**，用实际 h_s、native z_s 和全部 pilot 函数构造 v，再送入相应目标 t 的 normalizer/η。构造 a0 时用 `z_s,q+f_t,q(h_s)`，a1 为其与 `z_s,q+f_s,q(h_s)` 的半概率混合。不得偷偷将输入域改成 t 或使用 target 的真实标签。

$$
b_k(x)=\widehat\mu_{s\to t_k,q}(x)-m_{s\to t_k,q},\qquad
w_{s,k,q}(x)=0.1+1.8\,\operatorname{softmax}_k(b_k(x)/\tau_{s,q}).
$$

于是每行/任务 `w0+w1=2`，每项在 `[0.1,1.9]`，无额外 batch 或全池均值归一化。若两个中心需求相同，w=1。权重缓存一次并永久冻结；不由当前 F、当前 y 或训练进度重新计算。

这一步把 target 上学到的需求延伸到 source 的特征支持，是 **H2 的待证假设**。折外目标预测通过并不能证明此延伸可靠。记录 source 与 target 的 μ/输入支持分布、clip 比例、每原型质量、ESS=`(Σw)^2/Σw²`、两函数分配相关性；这些是边界诊断，不是通过改变门槛“修好”分布的许可。

### 5.2 独立来源函数


12 个 `F_s→t,q:256→32 ReLU→1`，每个 `8257` 参数，共 `99084`。初始化为对应 pilot `f_s,q` 的参数副本，并对末层 weight 加专用 RNG 的 `N(0,1e-3)` 微扰、末层 bias 不扰动。每个 slot 的微扰在 DFD/UNIFORM/GEOM/SHUFFLE 间完全相同。这样 UNIFORM 不会因被强制精确同初始同轨迹而人为退化为重复函数。

函数彼此独立，native、pilot、η、m/τ/w 均冻结。训练目标

$$
L_{\rm form}=\frac1{4|B|}\sum_{i\in B}\sum_{q=0}^1\sum_{k=0}^1
w_{d_i,k,q}(x_i)\ell(z_{iq}+F_{d_i\to t_k,q}(h_i),y_{iq}).
$$

每个样本/任务的责任总量固定，数据自然源质量不变；函数内分配发生改变。这里每个函数有自己的真实监督 loss，区别于旧 LPS 只监督多个块之和、再对其反向梯度重新分配。所有函数都经过 12 passes；没有只监督 owner sum 的替换实现。

单独的严格正权 weighted BCE 在无限表达能力和充分优化下仍具有相同的 source 条件概率最优解。因此本方法依赖有限函数容量、有限训练和正则化下的资源分配，**不保证渐近时自动产生不同知识**。不增加强制正交/熵最大化 loss 来掩盖这个限制。

训练结束永久冻结所有 F。不保留功能形成期间表现最好的中间库；固定末次库是唯一输出，P_read/P_cal 不倒流选择形成 epoch。

## 6. 阶段 D：普通连续读取


对目标 t/任务 q，候选0为本地 pilot：`a0=z_q+f_t,q(h)`。其余4项为两个 foreign source s≠t 各自两个 intended-target 函数 `a_j=z_q+F_s→k,q(h)`；按 `(source升序,intended_target升序)` 排序。不读取 source=t 的形成函数，也不只保留 intended_target=t 的两个函数。这个集合显式检验同一来源的不同功能是否有用。

六个读取头输入

$$
v^{read}_{tq}=[h_{256},z_0,z_1,p_0,p_1,p_2,p_3,p_4]\in\mathbb R^{263},
\quad \alpha=\operatorname{softmax}(\mathrm{MLP}_{263\to32\to5}(v^{read})),
\quad \hat p=\sum_{j=0}^4\alpha_j\sigma(a_j).
$$

每头 `8613` 参数，六头 `51678`。隐藏 Xavier/bias0；末层 weight0，bias=`log([.9,.025,.025,.025,.025])`。normalizer 来自 target fit 的固定库特征，std floor1e-3、clip±10，之后冻结。

所有使用软读取器的臂（包括 DFD、UNIFORM、GEOM、SHUFFLE、LOCAL_CAP）先在 P_form 上做20次完整预拟合，再在 P_read 上做10次完整拟合，Adam 状态连续继承。这样简单对照也实质利用相同的目标标签；不得只给它运行一个不参与预测的需求头来伪装信息匹配。此时 P_form 在 DFD 中被需求与读取两阶段复用，是训练信息，不再对最终模型具有折外性质。

每头按目标域行独立拟合普通 BCE；一 pass 是六头各自遍历当前角色池一次。整个六头组完成一 pass 后才形成候选状态。训练风险使用稳定计算：

$$
\log\hat p=\operatorname{logsumexp}_j(\log\alpha_j+\log\sigma(a_j)),\quad
\log(1-\hat p)=\operatorname{logsumexp}_j(\log\alpha_j+\log\sigma(-a_j)).
$$

不要先截断预测导致饱和区梯度错误。只有读取头有梯度；候选 logits、h、z 和概率均 detach。普通 gate 就是主读取器，DFD 不把它重新命名成新的收益决策算法。

候选包括 `LOCAL_ONLY` 整体 OFF、pass0（已完成20次 P_form 预拟合）的完整六头组、P_read pass1–10 全部完整组。P_form 内没有选点。P_cal macro LL 最低选一个组，精确平局顺序为 OFF、pass0、较早 pass。保存该组六头参数及 Adam 状态；禁止每个 target/task 单独按 P_cal 选 pass 再拼接。OFF 是独立的精确策略，不用极端 softmax bias 近似。选择 OFF 仍记录形成/训练成本，并标记内容读取退化。

最终 predictor 由一个 native LL-selected checkpoint、固定 pilot、固定函数库、normalizer 及一个完整读取策略组成。没有第二次 validation 选择不同流水线快照。本轮 `selected` 一词指该预定两层选择的最终 predictor；原生 epoch24 单独报告，不能把它当成 DFD 的共同训练终点。

## 7. 必须实现的对照及准确差异


### 7.1 六个核心流水线


| ID | 形成 | 读取 | 用途 |
|---|---|---|---|
| `DFD` | §5 的目标需求分配，12函数 | §6 软读取 | 唯一主方法 |
| `UNIFORM` | 12函数，所有 w=1 | 同 DFD 的读取和标签 | 同容量/同来源/同训练次数的形成对照 |
| `GEOM` | 每 source 固定两几何簇的软责任 | 同上 | 与非收益的支持分工比较 |
| `SHUFFLE` | 分层打乱 DFD 的二维责任向量 | 同上 | 排除权重分布、标签难度和一般正则化解释 |
| `CONST_UNIFORM` | 直接复用 UNIFORM 固定库 | 每 target/task 一组常量 simplex 系数 | 条件读取相对域任务固定读取的增量 |
| `LOCAL_CAP` | 每 target/task 4个只用本地标签的额外函数 | 5成分同架构读取器 | 完整表示、相同每次读取函数数的强本地容量对照 |

所有软读取器均实质使用 P_form20 + P_read10 的标签。DFD 的需求学习另有两折与 refit，**不宣称所有臂计算或标签访问次数相同**；独有标签行相同。SHUFFLE 的责任来自 DFD，也必须承担需求学习依赖成本。所有负结果必须保留。

`NATIVE`、`LOCAL` 从共同前缀导出，无需重新训练；仍是每个 seed 必须列出的主质量参照。不能把 CONST 的常量读取器当作 LOCAL，也不能用裸 native 代替本地全表示强对照。

此外，两个便宜但必要的**直接概率预测器**进入全部主比较：

- `ETA_DIRECT`：复制已经完成全 P_form20 refit 的六个 η 及其 Adam 状态，保存 pass0，再在各自 P_read 上普通 BCE 微调10 passes；P_cal 从完整六头 pass0–10 选择，平局最早。DFD 用于形成的原 η 不变。输出直接为 η 的概率，不经过 source形成或凸组合。它仍使用三个 source pilot 残差特征，应称“无形成的直接概率拟合”，不能称无 foreign 信息。
- `H_DIRECT`：六个 `[h256,z0,z1]258→32 ReLU→1` 概率头，每头8321、共49926参数，无任何 foreign函数输出；target fit normalizer、初始化/η普通BCE规则同§4。P_form20预拟合后 P_read10微调，P_cal从完整pass0–10选择。它是无 foreign pilot 特征的完整表示直接拟合。

两者不受 LOCAL_CAP 五成分凸包限制，必须报告并纳入05的晋级，不能仅以更多本地混合函数代替。ETA_DIRECT 独立部署只需 native、pilot、一个最终η，不需要 OOF/m/τ/source形成库；其成本按实际所需流程计算。H_DIRECT 独立部署不需要 pilot。P1无支持时不为这些导出开启额外正式训练；若模型已算出，则保留其完成状态。

### 7.2 GEOM


每 source 在 fit h 中按专用无标签计划最多取20000行，K-means++ K=2、n_init=5、max_iter=300、tol=1e-4。source 两个任务共用中心；中心按拟合返回顺序对应升序 intended targets，顺序与随机种子记录。不得用 P_form/P_cal/Q 选中心或 K。

`ρ=.05+.9*softmax_k(-||h-c_k||²/τgeo)`，`w=2ρ`。用该 source 的拟合子集令平均 `H(ρ)` 接近 `.5*log(2)`：base 为该子集对两个中心的全部非零平方距离集合的中位数，若不存在则1，floor1e-6；τ 范围 base×`[1e-4,1e4]`，log空间60次二分。不可达时选绝对差较小的端点，平局较小τ；记录退化和实际熵，不重抽数据/种子。中心/温度一次冻结。保留和 DFD 相同的函数初始化、12个完整形成 passes、损失分母和读取器。

### 7.3 SHUFFLE


对每 source/task，先用其冻结本地 pilot 概率在全部该源 fit 上取经验十分位边界（线性分位，重复边界可导致空箱）；按 `(y_q,概率箱)` 分层。每层按 row_id 升序后用专用 RNG 置换，将该层每行的**整个二维 w 向量**赋给另一行。两意图必须使用同一置换，保持每行和2以及每层的联合权重分布。单行层不改动并记录。

该控制使用 source fit 标签，属于合法训练，但只保留本地标签/预测难度分布，破坏需求与具体 source 特征的对应。row_id 到二维权重的置换只生成一次，整个12 passes固定，不按epoch重抽。记录实际改变行比例、箱计数、各意图质量与 ESS；若大量权重原本相同，SHUFFLE 自然与 DFD 近似，不得加噪声制造对比。

### 7.4 CONST_UNIFORM


在同一个 UNIFORM 库上，对每个 target/task 用 `P_form∪P_read` 的独有行一次拟合常量 `α∈Δ⁴`，预测仍是五成分概率凸组合。使用 FP64 convex BCE，直接优化5维 simplex：SLSQP，bounds[0,1]、sumα=1、解析梯度、maxiter500、ftol1e-12。起点为 own-only、均匀、`[.9,.025×4]`；以训练目标选择最优解，精确平局先较小 foreign 总质量、再字典序。不同初始化不是新的研究 seed。

为避免FP64概率恰为0/1，solver固定将各候选成分概率clip到`[1e-15,1-1e-15]`后再线性混合；不要对混合后的训练概率作会引入平坦区的截断再声称凸优化。记录该固定数值保护；其输出与原成分混合的概率差≤1e-15。实际P_cal/最终评价按05统一epsilon=1e-7。

六个常量最优解组成一个完整候选，P_cal 只比较该完整策略与 own-only，平局 own-only。不逐 cell 在 P_cal 改系数。分别记录可行性（sum误差≤1e-8、上下界误差≤1e-10）及近优性：令g为该凸训练目标对α的梯度，Frank–Wolfe gap=`gᵀα−min_j g_j`≤1e-7。可行性本身不能证明近优。记录迭代数和失败状态；求解失败属于实现故障，不能退化为几次 sigmoid 优化后称强常量对照。

### 7.5 LOCAL_CAP


每 `(t,q)` 创建4个 `256→32→1` 函数，均从本地 pilot `f_t,q` 复制并加相同规则的独立末层微扰。仅在 `d=t` 的 fit 行上监督，12 passes；无 foreign 标签，source 意图权重不用在本地头上。每个函数的 BCE 系数为 `1/(4|B|)`，与 UNIFORM 的单函数系数一致；因为每行有4头×2任务，其无共享参数的总标量 loss 是两倍两任务均值，须在日志中声明，不能误报为 native 的同一总 loss。native/pilot 全冻结，各函数独立，额外 loss 不改变 native 或其他共享模块。

读取 local pilot 加4个本地函数，输入依然263维，拟合规则完全相同。额外函数共24个=`198168`参数；加 pilot 与 reader 后新增部署参数为 `299388`，**高于 DFD 的200304**。它匹配每 target/task 的函数数和读取形式，是有利于本地解释的更强容量控制，不是严格全局参数匹配或同来源样本量控制。

### 7.6 通过核心比较后才做的两个 P3 对照


**`U_BUDGET`**：复用 UNIFORM 库，reader 在 P_form 连续预拟合60 passes，再 P_read10，其他与§6一致，P_cal 不选择 P_form 内的 pass。DFD 每个 target/task 的 η 两折总计相当于20次完整 P_form 标签访问、refit20、reader20，共60；该控制让这60次访问全部用于普通读取器。它匹配标签访问次数，不能称 wall time 完全匹配。研究诊断额外访问另列，不混入部署预算。

这里匹配的是优化训练的 label-pass，μ/m 的统计、OOF评分、P_cal/诊断及缓存的实际访问全部另计，不免费。核心 UNIFORM20 与 U_BUDGET60 两条完整结果都保留，05要求同时比较；不能只用可能过拟合的60路径作为简单方法。U_BUDGET可从UNIFORM已保存的P_form20前缀继续到60，但不能从其已经P_read训练的末状态继续。

**`POOL`**：先在全部自然 fit 上训练两个 task-specific pooled pilot `g_q:256→32→1`，12 passes，损失 `Σ_iq BCE(z_q+g_q,y)/(2B)`；再创建与 DFD 相同的12个函数 slot，从对应 g_q 复制并加微扰，在全部 fit 上各自训练12 passes，不用 source mask 或需求权重。损失为 `Σ_iqΣ_(s,k) BCE(z_q+F_s,k,q,y)/(12B)`，同一 task 有6个函数；slot 名仅保留读取位置，不再声称 source 所有权。仍保留原6个 source pilot中的本地 p_t 作为第0成分，以只替换 foreign 函数形成；读取与§6相同。其每函数覆盖行更多、成本更高，明确属于有利于 pooled 解释的强控制，不冒称相同标签访问次数或更新规模。若它匹配 DFD，source 特定组织的必要性不足。

二者按05的触发与种子执行，不增加 K、宽度、温度或读取强度搜索。

## 8. 梯度与 Adam 的准确合同


| 阶段 | native | pilot | η | F库 | reader |
|---|---|---|---|---|---|
| native | 更新 | 不存在 | 不存在 | 不存在 | 不存在 |
| pilot | 冻结/no_grad | 仅本源任务更新 | 不存在 | 不存在 | 不存在 |
| η folds/refit | 冻结 | 冻结 | 仅目标任务头更新 | 不存在 | 不存在 |
| formation | 冻结 | 冻结 | 冻结，w已缓存 | 仅source任务对应函数更新 | 不存在 |
| read | 冻结 | 冻结 | 不参与 | 冻结 | 更新 |
| evaluation | 全部 eval/no_grad | 全部 eval/no_grad | 不推理 | 全部 eval/no_grad | 全部 eval/no_grad |

target P_form 标签通过固定 η→w 间接影响 source 形成；不能宣称完全无 target 影响。没有 target label 对 F 的直接梯度；没有 source loss 反传 η；没有虚拟 Adam 或二阶梯度。

每阶段新建该阶段 Adam，复制 pilot 到 F 时**不复制 pilot 的 m/v/step**，以新函数训练状态0开始。不同方法形成本身的参数初值和 plan 一致，不能共享已经训练过的 F optimizer。读取器 P_form→P_read 连续继承 Adam；候选恢复包括 optimizer，尽管最终推理不用其状态。

每个函数采用独立参数对象，可由一个 optimizer 管理。逻辑 batch 中无来源样本的函数，参数 grad 必须保持 None；不要因全库前向或手工 zero tensor 推进其 m/v、step 或 weight decay。`zero_grad(set_to_none=True)`；只构建活跃源的训练 loss。耦合 Adam 更新为 `gbar=g+wd*θ`，再维护标准一二阶矩；不能换 AdamW。每层/函数报告实际 step 次数与末次 m/v 范数，权重变化不能自动被解释为真实 Adam 位移变化。

## 9. 从头执行的伪代码


```python
for dataset, seed in registered_development_rows:
    roles, encoder = verify_natural_protocol(dataset)
    native = train_native_from_scratch(roles.fit, epochs=24,
                                      select="validation_macro_ll", eligible=range(12,25))
    cache = freeze_and_cache(native.selected, roles)
    pilot = train_source_task_pilots(cache.fit, passes=12)
    demand_oof = crossfit_eta(pilot, cache.P_form, folds=2, passes=20)
    p1 = evaluate_registered_fold_out_policies(demand_oof)
    save_p1_rows_and_source_support_receipt(p1)
# P1按两个seed一起作数据集决定，不能边看第一个seed边改第二个
for dataset in datasets_passing_P1:
    for seed in [5101, 5102]:
        native, cache, pilot, demand_oof = load_verified_prefix(dataset, seed)
        eta = refit_eta_on_all_P_form_from_fresh_init(passes=20)
        weights = frozen_centered_target_responsibilities(eta, demand_oof, cache.fit)
        for arm in ["DFD", "UNIFORM", "GEOM", "SHUFFLE", "LOCAL_CAP"]:
            bank = train_frozen_input_functions(arm, weights, passes=12)
            reader = prefit_on_P_form(bank, passes=20)
            candidates = fit_on_P_read_with_full_state_candidates(reader, passes=10)
            final = select_whole_policy_on_P_cal(candidates, include_exact_OFF=True)
            evaluate_validation_and_Q_explore(final)
        const = solve_constant_simplex_on_union_P_form_P_read(uniform_bank)
        evaluate_validation_and_Q_explore(const)
freeze_development_decisions_and_triggered_P3_manifest()
# 所有旧test/legacy评分在方法、角色、对照和代码锁定后执行
# P4新seed逐个从scratch重复同一完整pipeline；不沿用开发bank/η/reader
```

该伪代码中的函数名是**待新建接口规范**，不是声称远端已有这些接口。P1 停止时只输出完成的结果及 NO_START；不擅自换一个新方向继续训练。

## 10. 与真实仓库代码的衔接


### 10.1 已核实与未核实的接口


本次实际读取了下列固定版本源码：

- [fixed.py，c5618e28](https://github.com/Caelestis1903/My-Research-Project/blob/c5618e281785a4b7063ff9bf2a61cf38d4faf1be/experiments/data_generate/MDMTRec/models/multi_domain/fixed.py)：`MDMTRec.forward(x, test_flag=False)` 输入 `[F,B]`，输出 `[B,2]` 概率；gate 使用 `emb.detach()`，共享专家张量可从实际前向流程机械抽取。
- [protocol.py，c5618e28](https://github.com/Caelestis1903/My-Research-Project/blob/c5618e281785a4b7063ff9bf2a61cf38d4faf1be/experiments/reproduction_20261021/protocol.py)：`make_model` 先建立原始模型、排除 `_weight_*` 后拷贝 canonical backbone；包含完整 RNG、指标和哈希工具。其历史 optimizer lr=.01、Equal-style loss 与本轮不同，**不能直接复用旧 trainer/optimizer**。
- [controlled.py，f0b1084b](https://github.com/Caelestis1903/My-Research-Project/blob/f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e/experiments/natural_ml_reassessment/controlled.py)：可参考自然目标、独立指标输出与 checkpoint 管理思想；其具体实验变体不自动成为本轮 baseline。

v3 报告提到服务器 `fixed1_adapter.py.encode(x,risk=False)`、LpsTrainer 等，但对应训练源码不在已读取的远端证据树中。`research_plans/26-10-25_v2/` 也不在该远端树中。不得把这两个不可读接口作为已验证依赖。依据是公开 v3 DESIGN_SPECIFICATION/SPEC_RESOLUTION，加上上述可读实际源码。[实现边界](https://github.com/Caelestis1903/My-Research-Project/blob/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0/research_reports/2026-10-25_v3/IMPLEMENTATION_REVIEW.md)

### 10.2 明确重建路线


未来 Codex 在当前 main 创建独立实现分支，不合并整个历史实验分支。先查服务器是否有 v3 源码/审计工件；可读取时核 hash 后建立本轮 adapter 并做前向与权限验收。不能读取时，**采用固定源码重建路线**：通过 `git fetch` 获取证据分支所含对象，以 `git show`/`git archive` 提取 `c5618e28...` 的 `experiments/data_generate/MDMTRec` 及其实际依赖到本轮包内的 `vendor/pinned_fixed/`，保留来源与文件 hash；不改远端旧文件，不恢复 AutoML。

实现原生 adapter 时机械复制固定模型的 forward，新增 pre-sigmoid logits 与 shared expert 输出，不通过 `logit(probability)` 反求主训练 logits；新接口输入 `[B,F]`，内部明确一次转置。forward 的训练 gate detach 保持不变。canonical 初始化按 pinned `make_model` 的先原始再拷贝方式实现，并核对原生概率。只要 parity、角色和核心 loss 验收通过，重建路线允许继续本轮新实验；不将其标成旧服务器逐字节复现。

### 10.3 新建模块和入口


目录 `experiments/mdmt_dfd_20261026/`，以下为未来实现清单：

| 模块 | 必须承担的接口/输出 |
|---|---|
| `data.py` | `RoleBundle`、raw/encoder/role hash、固定 row plans、自然 batch；fit-only support flags |
| `native_adapter.py` | `NativePacket(logits[B,2], shared[B,4,64], h[B,256])`，无语义变更的 fixed 模型 |
| `functions.py` | PilotBank、FunctionBank、单函数来源/任务/意图 owner map、各臂形成 loss |
| `demand.py` | η两折/refit、精确 risk difference、m/τ、固定二维责任缓存 |
| `readers.py` | 普通软 mixture、LOCAL_ONLY、常量 simplex、normalizer 与 whole-policy selection |
| `plans.py` | 确定性 fold/epoch/RNG、完整矩阵和依赖注册，输出 immutable manifests |
| `state.py` | 完整 checkpoint、严格 resume、source/config/hash 与 cache identity |
| `diagnostics.py` | P1折外策略、支持分层、函数删除/置换、RHD四叉（独立开关） |
| `metrics.py` | 六格、macro/自然总体、配对bootstrap、原始预测保存 |
| `costs.py` | 非重叠叶阶段计时、真实资源/标签访问、shared-prefix与standalone账 |
| `cli.py` | 下列所有可执行命令，不仅是打印计划 |

```bash
python -m experiments.mdmt_dfd_20261026.cli inspect --spec DFD-v4.0
python -m experiments.mdmt_dfd_20261026.cli build-plan --stage all --dry-run
python -m experiments.mdmt_dfd_20261026.cli verify-semantics
python -m experiments.mdmt_dfd_20261026.cli run-stage --stage p1 --resume
python -m experiments.mdmt_dfd_20261026.cli decide --stage p1
python -m experiments.mdmt_dfd_20261026.cli run-stage --stage p2 --resume
python -m experiments.mdmt_dfd_20261026.cli decide --stage p2
python -m experiments.mdmt_dfd_20261026.cli run-stage --stage p3 --resume
python -m experiments.mdmt_dfd_20261026.cli freeze-evaluation
python -m experiments.mdmt_dfd_20261026.cli run-stage --stage p4 --resume
python -m experiments.mdmt_dfd_20261026.cli evaluate-exposed --locked-only
python -m experiments.mdmt_dfd_20261026.cli summarize
```

`build-plan` 必须展开全部 run IDs、确切常量、依赖、角色、seed 和预算上界；`decide` 严格执行05的布尔规则，并写明每个未启动项原因。P3/P4未触发时命令正常输出 NO_START 并继续汇总，不因“队列为空”改方法。

推荐 run_id=`v4_{dataset}_s{seed}_{arm}_{stage}`；artifact root 可用 `/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0/`（先核可写与空间）。这是未来服务器建议路径，不是本次已生成运行数据的位置。

## 11. 状态、工件、成本与必要验收


### 11.1 完整状态和原始输出


checkpoint 保存：模型各组件、该阶段 optimizer 的 m/v/step、stage/pass/逻辑与microbatch游标、所有 RNG、row/fold/shuffle计划、native selected/endpoint与其选择历史、normalizers、w/m/τ/centers、候选完整策略、代码/config/数据/source哈希、累计成本与唯一运行身份。中途恢复不得重训已完成 fold、重复 reader prefit、重新初始化微扰或重选 backbone。microbatch 断点只在完整逻辑 step 后保存；不保存半累积梯度的伪完成状态。

必需工件：`SOURCE_MANIFEST.json`、`DATA_ROLE_MANIFEST.json`、`PLAN.json`、`SEMANTIC_RECEIPTS.json`、P1 OOF逐行预测与fold、每个seed/arm最终预测npz、逐格指标、形成责任与支持摘要、来源函数同状态干预、标签访问账、成本叶账、`DECISION_P1.json`、`DECISION_P2.json`、`FROZEN_EVALUATION_SPEC.json`、最终全文报告与失败清单。逐行文件含 row_id/domain/y/预测/角色/checkpoint_hash，预测标签不写进线上预测函数。

### 11.2 参数、缓存和计算来源


DFD 新增部署参数 `49542+99084+51678=200304`；训练 η 另 `50502`，不在推理。此数不含原生，原生实际参数数从 manifest 测量，不把旧报告的不同模型参数数硬贴过来。模型评价全部4个 foreign函数，没有稀疏推理速度承诺。

fit h 的 FP32 原始缓存约 ML `640133×256×4≈0.66 GB`、KR `1604904×256×4≈1.64 GB`（十进制，尚不含z/标签/其他角色与临时数组）；可分块内存映射，不能为了省空间默默改成FP16。一个 backbone forward 可以服务同一缓存的多个函数，不能把12函数错误计成12次完整 backbone 前向。

按 native训练、特征缓存、pilot、η folds/refit、几何/置换、formation、reader prefit/read/cal、评估、I/O 分非重叠叶计时。三本账：实际研究执行成本；独立部署重训成本（共享前缀对每方法都应计入）；开发/选型成本。共享前缀实际只算一次，不能在部署账省略；含子步骤的总计时不能与子步骤相加。记录 CUDA设备/负载、峰值显存、进程墙钟、并发活跃区间，未测GPU利用率/功耗不换算GPU小时或能耗。

### 11.3 四组必要验收


只做会改变科学结论的检查，不建无关防御测试体系。

1. **原生与数据**：两种真实schema各一批，adapter sigmoid 与 pinned fixed forward 最大绝对误差≤1e-6；h确为四shared专家；native gate detach；fit-onlyencoder及所有角色行不交叠；Sample逻辑batch/microbatch loss和梯度一致。FP32梯度 allclose atol1e-6/rtol1e-4，报告实测最大差。
2. **需求与责任**：FP64逐行 BCE 差与恒等式 max abs≤1e-10；η构造mu无y输入；OOF模型训练ID与该行互斥；w每行和2误差≤1e-6、范围正确；UNIFORM w1；SHUFFLE二维联合向量/分层边际精确守恒；GEOM仅fit特征。检查假设域t不修改实际输入域s。
3. **梯度与状态**：真实小batch中 native/pilot/η无形成梯度，外源函数不会因被读取而更新；缺源函数grad=None且m/v/step不动；复制函数参数相等但新Adam为空；相同checkpoint恢复的下一逻辑步与不间断执行参数/optimizer allclose atol1e-6/rtol1e-4。不得用权重文件相等代替完整状态。
4. **预测/选择**：精确OFF等于LOCAL；五概率凸组合与稳定logsumexp BCE一致（FP64 abs≤1e-10；FP32概率≤1e-6）；P_cal只选完整六头；常量simplex可行与近优；无测试标签到训练依赖图。导出预测能从最终完整predictor独立重建。

出现非有限 loss/预测、角色交叠、错误梯度、选点泄漏或无法重建最终预测，属于阻断问题，修复后重跑受影响项；不调松容差或只重跑失败seed。实现者提供变更和收据，独立审查者检查代码与关键数值后才能启动正式训练。审查者可以指出语义修复，不可借审查重新开放研究方向。
