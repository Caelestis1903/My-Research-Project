# 03｜首轮实验矩阵与 Codex 实施规范（修订 v2）

**本文件是执行合同，04_CODEX执行提示词.md 是交接入口。** 本次提交只修订规划，没有实现训练器或启动训练。后续交付 Codex 执行时，依次完成实现、必要检查、32 次正式训练和只读诊断；完成首轮报告后停止。SAGU/SAOS/SGVS 不在本轮训练范围。

源码基准：main@540171f865d3f80b7b46abdba69d0939cab49d80。下文 R 指 `history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021`，M 指 `history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/data_generate/MDMTRec`。机器配置见 `stage0_protocol.json`；正文与 JSON 必须一致，冲突时先修复合同，不自行选择版本。

## 1. 科学问题及范围

先回答：**在自然 Sample 人口权重下，固定轻量增强是否有超过统一损失缩放的 selected AUC 增量？** 四臂在两个损失尺度下分别测量增强效应，再测交互。探针辅助判断值得开发哪种梯度/历史控制，不将 cosine 当泛化收益。

保持 Sample 的原样本、领域和任务**权重预算**；增强开启后训练风险改变，不声称完整经验目标不变。首轮不改采样 q、不用 specificity、不平衡六格梯度、不开发新的优化器。P0 是本次显式建立的 canonical baseline，不冒称仓库已有唯一 canonical Sample 配置。

## 2. 唯一实验底座

### 2.1 模型、优化器与预算

| 项目 | 冻结选择 |
|---|---|
| 主干源码 | M/models/multi_domain/fixed.py 的 MDMTRec |
| 配置 | D=3、T=2、expert_num=4、fcn_dims=[512,256,256,64]；SparseFeature embed_dim=16 |
| 固定融合 | exp_d=exp_t=bal_d=bal_t=1，Python 标量；不是 sigmoid(1)≈0.731 |
| 前向语义 | 保留 STAR、LayerNorm、gate 的 emb.detach()、专家/tower 连接；输出 [B,2] sigmoid 概率 |
| 优化器 | 一个 torch.optim.Adam；lr=0.01、betas=(0.9,0.999)、eps=1e-8、weight_decay=1e-5、amsgrad=false |
| 数值/正则 | FP32；Adam foreach=false/fused=false；耦合 L2 在 optimizer 内加入；无 scheduler、AMP、裁剪、梯度累积、参数 EMA |
| GPU 设置 | TF32=false，cudnn.benchmark=false；记录确定性设置、硬件和软件版本 |
| 预算 | batch_size=40960；完整 50 epoch；patience=null（不早停）；每真实 batch 一次 Adam.step |
| 正式种子 | 2022、2023、2024、2025，同数据集四臂配对 |

选择 fixed1 是沿用既有固定研究主干，避免混入 .731 版本的结构差异。LR/batch/Adam 是本轮显式冻结的起点，不代表继承 legacy_group_sum 的性能结论；全部四臂重新训练。

新实现放 `experiments/gradient_aug_20261031/`。以 M 的固定模型和 basic 层建立命名隔离的本地副本，记录来源路径/Git blob/SHA256。增加“Embedding 后、STAR/skip 前”的可选扰动入口；关闭时保持参数键、顺序和前向数值。不直接运行 R/run.py：它仍有旧目标、早停及归档后的路径假设。不修改归档源码/历史报告。

不要通过顶层通用名 `data`/`protocol` 导入归档模块，以免误命中其它阶段；数据适配与必要 basic 依赖放入新包，用明确包名导入。不复制历史训练输出。

### 2.2 数据身份

复用 R/audited_data.py 的 `_load_movielens_base(...,'natural',2022)`、`_load_kuairand_base(...,'natural',2022)` 预处理。split 固定，不随训练种子重划；仅 train fit vocab/scaler，OOV=0，沿源文件行序全局 80/10/10；KR 先全局切分再按 tab 过滤。

| 数据 | train counts d0/d1/d2 | train / val / test 行数 | 任务、物理域 |
|---|---|---|---|
| MovieLens-1M | 155949 / 318599 / 325619 | 800167 / 100021 / 100021 | rating>3、rating>4；{1,18}/{25}/{35,45,50,56} 年龄组 |
| KuaiRand-Pure | 172782 / 1743898 / 89452 | 2006132 / 250767 / 250767 | stored is_click、long_view；tab0/tab1/tab4 |

文件路径可以重定位；内容必须匹配 `stage0_protocol.json` 中 raw/array/split/encoder/scaler 哈希，来源为 R/data.py、data_splits_and_labels.csv、INTEGRITY.json。不可调用根目录 stage_S1_data.py 的 MovieLens 角色重采样路径。身份不符先检查路径/预处理；无法恢复则停止受影响数据集并报告，不能静默换 split。开发不使用测试指标。

### 2.3 采样、初始化和选模

- 每 epoch 对全部 N 个原样本作一次无放回 shuffle；drop_last=false、num_workers=0；不用 WeightedRandomSampler、有放回 q=p 或固定配额 batch。初始边际人口权重 p_d=n_d/N，后续条件概率随剩余池变化，不强制每批比例等于 p_d。增强不进入 sampler。
- 独立 CPU loader generator：train=seed+11000、validation=seed+33000、test=seed+44000。每臂单独构造并保存状态，记录每 epoch 原始 row_id 顺序哈希。
- seed_all(seed) 后构造一个未增强 fixed1 模型，保存初始 state；P0–P3 严格加载同一 state/参数顺序和空 Adam state。无需复刻历史 O→B 移植 RNG，也不得混称历史初始化。
- 每 epoch 完整真实 validation；按六格 macro AUC 严格 > 更新 selected，平局最早；E0 不参与选模。四臂跑到 E50，不使用早停。
- ML 每 epoch 20 步、总 1000 步；KR 每 epoch 49 步、总 2450 步。保存 E1/E5/E10/E20/E50、selected 完整状态，另存 latest 断点。
- 32 个 run 的 selected 和共同 epoch 清单冻结后，执行一次统一 test 评估批次：selected、共同 E10、final E50，重合复用。不逐 epoch 查看 test，不据 test 改参数或 checkpoint。

## 3. 目标与增强的唯一数学定义

### 3.1 Sample

所有原样本的两个任务标签均可用。领域 d 的人口比例 p_d=n_d/N，任务数 T=2：

\[
L_r(\theta)=\frac1{NT}\sum_{i,t}\ell_{it}^r
=\frac1T\sum_t\sum_d p_d\bar\ell_{d,t},\qquad
\ell_{it}^r=\operatorname{BCE}(f_{\theta,t}(x_i),y_{it}).
\]

当前 batch B 使用 \(\hat L_r=(|B|T)^{-1}\sum_{i\in B,t}\ell_{it}^r\)。代码用概率 BCE reduction='none'，先任务均值再样本均值。不做六格 group-mean 求和或批内按域再归一化；缺域无需填不存在格的均值。评价 LogLoss 始终用真实输入和未缩放 BCE。

### 3.2 NR-Gauss-v1 增强

| 变量 | 首轮冻结值 |
|---|---|
| 增强对象 | 最小训练域 d*=argmin_d(n_d,d)：ML=d0，KR=d2 |
| 概率/视图数 | 目标域每次真实曝光生成 1 个视图，概率 1；其它域 0 |
| 混合 λ | 0.5，跨数据集/seed/epoch 不变，不调参 |
| 噪声强度 | κ=0.1，按每样本每 field Embedding RMS 作相对高斯扰动 |
| 不变部分 | 原始特征 ID、domain ID、标签、任务、row_id、domain Embedding、所有 dense 输入 |
| 禁用 | augmentation_enabled=false 或 λ=0 走真实损失；κ=0 为同输入视图 |

ML 可扰动字段为 `user_id,movie_id,gender,occupation,zip,cate_id`；age 和 domain_indicator 不扰动。KR 为 audited_data.py 的 KUAIRAND_SOURCE_COLUMNS 除 tab 外全部 sparse 字段：user_active_degree、is_live_streamer、is_video_author、follow_user_num_range、fans_user_num_range、friend_user_num_range、register_days_range、onehot_feat0–onehot_feat17、video_type、upload_dt、upload_type、server_width、server_height、music_type、tag、counts。机器配置展开 onehot 范围，不留通配符。

对于 field f 的 \(e_{if}\in\mathbb R^{16}\)：

\[
s_{if}=\operatorname{stopgrad}\left(\sqrt{\frac1{16}\sum_{j=1}^{16}e_{ifj}^2}\right),\quad
\xi_{ifj}\sim\mathcal N(0,1),\quad
\tilde e_{if}=e_{if}+\kappa s_{if}\xi_{if}.
\]

s=0 时噪声为 0，不设下限；不 detach Embedding 本身。反向遵循此 stopgrad 计算图，不微分噪声尺度/分布对参数的依赖。M/basic/features.py 的默认初始化标准差为 0.0001，相对强度避免绝对 sigma=0.1 在初期淹没表示。这是明确的待测局部平滑假设，不保证语义不变。

EmbeddingLayer squeeze 输出顺序为“所有 sparse 按特征遍历顺序拼接，再接 dense”，并非按原始列交错；按实际层构造 slice 映射并断言总维度。不能假设 ML 的 domain Embedding 就在最后 16 维。输入索引不改写，domain/dense slice 必须逐位相同。

噪声用独立设备 generator，每 batch 重置为
\(s_{\rm aug}=\mathrm{seed}\times10^6+\mathrm{global\_step}\)，global_step 从 1 开始，P2/P3 对齐。固定 field 顺序和目标样本 batch 内顺序；不消耗模型/loader/探针的全局 RNG。记录噪声种子、设备和 field 规范。

本协议保留“最小域产生一个扰动视图”的历史起点，但改为相对噪声、保护 domain/dense、固定权重预算、Sample 聚合且无 specificity。阴性只评价 NR-Gauss-v1，不能推翻旧 D 组合的已观测收益，也不否定所有增强。

### 3.3 固定原样本权重预算

令 \(a_i=\mathbf1[d_i=d_*]\)，\(\ell_{it}^a\) 为同一样本增强视图 BCE：

\[
\hat L_{\rm view}=\frac1{|B|T}\sum_{i\in B,t}
\left[(1-\lambda a_i)\ell_{it}^r+\lambda a_i\ell_{it}^a\right].
\]
\[
L_{P0}=\hat L_r,\quad L_{P1}=0.1\hat L_r,\quad
L_{P2}=\hat L_{\rm view},\quad L_{P3}=0.1\hat L_{\rm view}.
\]

每个原样本两视图系数和为 1，人口权重仍为 p_d。不得将增强行直接 cat 后做无权重 mean，也不能用未归一化 \(L_r+\lambda L_a\)。

默认两次 forward：原 batch 计算真实损失，目标域子 batch 计算增强损失，按同一 |B|T 分母合成 scalar，一次 backward/step。无目标样本时跳过增强 forward。fixed 模型启用 LayerNorm、无启用的 BatchNorm/Dropout；编码须断言这一条件。若实际出现批状态层，先修复来源差异，不能擅自用拼接改批统计。

同一参数点 κ=0 时，P2 的 loss/梯度/Adam 更新在 FP32 容差内退化为 P0，P3 退化为 P1；不要求长期浮点轨迹逐位相同。

### 3.4 Adam 尺度语义

数据梯度 \(g_t=c\nabla\hat L\)，c∈{1,0.1}，耦合 decay \(\omega=10^{-5}\)。对 grad 非 None 的参数：

\[
h_t=g_t+\omega\theta_{t-1},\quad
m_t=\beta_1m_{t-1}+(1-\beta_1)h_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)h_t^2,
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^{k_t}},\quad
\hat v_t=\frac{v_t}{1-\beta_2^{k_t}},\quad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

k_t 是该参数实际 Adam step counter；grad=None 按 PyTorch 语义跳过，不补零。c 不缩放 optimizer 内的 decay，故 P1/P3 可能改变相对正则强度与 epsilon 作用。四臂识别此协议下缩放效应，不识别其内部唯一原因；只有影响下一步选择才追加最小 decay/epsilon 匹配对照。

## 4. 方法、预算与正确配对解释

| ID | 增强 | c | optimizer | 意义 |
|---|---|---:|---|---|
| P0 | 无 | 1 | 普通 Adam | 本轮 Sample baseline |
| P1 | 无 | 0.1 | 同上 | 无增强的缩放效应 |
| P2 | NR-Gauss-v1 | 1 | 同上 | 正常尺度增强效应 |
| P3 | NR-Gauss-v1 | 0.1 | 同上 | 小尺度增强效应 |

开发仅 seed=2022：两个数据集各四臂 small-batch 检查和最多 1 epoch 工程检查，最多 8 个开发短 run；不看 test，不按性能选 κ/λ。冻结后执行 2×4×4=32 次正式训练，重新初始化。非有限值先定位；不得只为失败臂改参数，数学协议失效须记录并修订协议版本。

对每种指标 M，逐 dataset/seed/checkpoint 计算：

\[
A_1=M_{P2}-M_{P0},\quad A_{0.1}=M_{P3}-M_{P1},
\]
\[
S_0=M_{P1}-M_{P0},\quad S_1=M_{P3}-M_{P2},
\]
\[
I=(M_{P3}-M_{P1})-(M_{P2}-M_{P0})=S_1-S_0.
\]

AUC 正值有利，LogLoss 负值有利。\(I\approx0\) 表示该指标尺度上未见实质加性交互，**不能推出收益由缩放解释**。例：0.70/0.71/0.72/0.73 对应增强=0.02、缩放=0.01、I=0。不得将差分转为非线性训练的机制贡献百分比。

报告全部逐 seed 配对差、mean/std、方向计数及描述性 95% 配对 t 区间（df=3）。不因区间跨 0 删除稳定方向，不将未过筛选门槛当作已证明无效。

## 5. 只读机制探针

离线复用每个正式 run 的 E1/E10/selected 完整状态，重复锚点去重，不插入训练循环。每锚点使用 2 个固定训练 batch，每个最多 1024 个原样本，来自 seed=314159 的训练 row_id permutation。真实 validation 探针 batch 最多 4096 行，来自 seed=271828 的固定 permutation。缺少目标域时标 unavailable，不改成均衡采样。

为避免样本支持混淆，定义扩展视图损失 \(\hat L_a^B\)：目标样本取增强 loss，非目标取真实 loss，仍除以 |B|T：

\[
g_r=\nabla\hat L_r,\quad g_a^B=\nabla\hat L_a^B,\quad
\delta g=g_a^B-g_r,\quad g_{\rm mix}=g_r+\lambda\delta g.
\]

另报目标子集的真实/增强梯度，两者同除以 |B|T，防止非目标相同梯度掩盖差异。raw gradient 不含 c/decay，反事实 Adam 输入为 c g，包含实际 optimizer 的 decay/state/bias correction。

共享块固定为参数名 `embedding.*、shared_weight、shared_bias、star_mlp.*、skip_conn.*、expert.*`；其余分记 slot_weight/slot_bias、domain_expert、task_expert、gate、tower。此处“共享”是结构归属，不预设收益。记录 shared/all/各块的 norm、cosine、δg 与正交分量，以及反事实更新 norm/cosine、m/v 变化、目标子集对应统计。零范数 cosine 写 null/原因；None 只在统计向量中可视为零，不写回 param.grad。

每个反事实分支从同一快照**联合 deepcopy((model, optimizer))**，保留参数映射、buffers、m/v、per-parameter step，以真实 PyTorch Adam 各 step 一次。固定真实 validation batch 比较更新前后 BCE。不得临时改正式 state 再近似恢复。隔离并恢复 Python/NumPy/Torch CPU/CUDA RNG、noise generator、train/eval 状态；不访问正式 loader iterator。探针结果记录源 state、row_id 哈希、额外 backward/耗时；正式快照及 loader/RNG 哈希应不变。

探针只解释给定 θ/history 下的单步 BCE 与更新关系，不训练 selector、不调 λ、不换 checkpoint，也不证明双 moments 能提高 selected AUC。最多每数据集 4 seeds×4 臂×3 锚点×2 batch，不追加多步 rollout；探针与完整训练成本分开报告。探针噪声 generator 用 seed*1000000+100000+2*anchor_global_step+probe_batch_index（index 从 1 开始），与正式增强随机流隔离。

## 6. 指标和结果驱动决策

### 6.1 必报指标

- 主指标：各 run 自身 validation-selected checkpoint 的真实 test macro six-cell AUC；同时报告对应 validation。
- 六格 AUC/原始 LogLoss、每域两任务均值、每任务三域均值、worst-cell AUC（同一 checkpoint 六格最小）。
- pooled AUC：每任务合并所有域计算，再平均两任务，同时保留各任务；不展平不同任务标签混算。
- 流量加权 cell AUC/LogLoss：按所评估 split 的 p_d、任务等权；与 pooled AUC 分开。另报真实总体 LogLoss。
- selected、固定共同 E10、final E50；完整 50 epoch 真实 train/validation 曲线。仅 final 改善不等于 selected-best 提升。
- 原样本曝光、额外视图、Adam steps、forward/backward 次数、wall time、峰值显存、选点 epoch。首轮更新次数匹配，额外算力如实报告，不自动增加算力匹配臂。

### 6.2 预先冻结的研究筛选门槛

门槛用于研究投入决策，不是总体显著性/等效性检验。

1. 每数据集仅用 P0/P1 四种子 selected validation macro AUC 均值选工作尺度 c*，平局取 1；test 前写入 selection_frozen.json。
2. 同时完整报告 A1/A0.1。c* 下 mean ΔAUC≥0.001、至少 3/4 seed 正向，记“值得继续”；test 同对照达到同门槛，记“正式首轮支持”。另一尺度阴性不推翻条件性收益。
3. 若另一尺度过门槛而 c* 未过，标“尺度依赖增强候选”，保留既定对照进入后续合同；不据 test 自动换 c* 或宣称优于最佳简单 baseline。两尺度均未过则结束首轮，不默认追加种子。
4. 两尺度 mean 差均绝对值<0.0005，且每尺度至少 3/4 seed 绝对差<0.001，记“当前预算下效果很小”；不称严格等效。中间结果报告条件/分裂方向。
5. 自动方法 GO 的保护条件：相对同尺度无增强臂，最大训练域任一任务、pooled task-mean、worst-cell AUC 不出现 mean≤−0.002 且≥3/4 seed 下降；macro LogLoss 不出现 mean≥+0.002 且≥3/4 seed 恶化。触发则 TRADEOFF，保留真实宏收益并明确代价。
6. 一个数据集过门槛即可保留有边界研究；两个过门槛才称首轮跨数据集支持。I 单列，不用正交互代替绝对增强收益。

| 结果 | 当前结论 | 后续决策（本轮不运行） |
|---|---|---|
| A1/A0.1 很小，S0/S1 正向 | 此增强没有实用独立 AUC 增量，缩放更值得研究 | 暂停 NR-Gauss-v1 的 SAGU/SAOS；有必要才匹配 decay/epsilon |
| 两尺度增强正向、I 很小 | 增强正主效应、未见实质交互 | 固定增强+Adam 是强基线 |
| 仅一个尺度增强正向 | 存在条件关系 | 保留该条件，不说“全部收益是缩放” |
| macro 提升但大域/pooled/worst 明显下降 | 存在性能交换 | 报 TRADEOFF，先明确保护目标 |
| cosine 负而单步真实 BCE 有益 | 冲突不是负收益充分条件 | 不据 cosine 硬裁剪；考虑固定混合/一致性 |
| 几何量多锚点与效用同向 | 是值得检验的信号 | 后续可设计 SAGU，仍用完整训练判定 |
| 固定混合有收益、history 探针有来源差异 | 状态值得研究，未证明污染 | 可优先 SAOS；不以 SAGU 成功为先决条件 |
| 仅 late/final 改善 | 阶段/过拟合缓解现象成立 | 不称 selected-best 提升 |
| 仅 LogLoss 改善 | 概率预测质量改善、排序收益未见 | 不能仅凭 LogLoss 归因于校准 |

### 6.3 第二阶段边界

删除原重复 Q1：同固定比例混合+同 Adam 与 Q0 数学重复。后续可能比较 Q0 归一化固定混合、Q2 共享块动态替换门控、Q3 替换差梯度分解；必须另行冻结私有块处理、超参和预算。SAOS 可由增强+历史证据直接进入候选，不要求 SAGU 先成功。

**本轮完成四臂与探针报告后停止，不自动实现/运行 Q2/Q3、SAOS、SGVS 或统一 EMA 扩展。** 02 中为研究候选，不是续跑合同。

## 7. Codex 实现及验收

### 7.1 文件职责/接口

新目录 `experiments/gradient_aug_20261031/`，推荐文件 model.py、data.py、objectives.py、protocol.py、run.py、probe.py、analyze.py、tests/。来源优先 M/fixed.py/basic、R/audited_data.py；参考 R/data.py 的身份检查和 R/run.py 的独立 loader，不继承旧目标/早停。

run.py 须支持 `--mode audit|check|dev|train|evaluate`、`--dataset movielens|kuairand`、`--arm P0|P1|P2|P3`、`--seed`、`--device`、`--protocol`、`--run-dir` 和数据路径。evaluate 只接受 selection 冻结工件；train 不评估 test。probe.py 独立读 checkpoint，analyze.py 生成 32-run ledger/配对表/决策。以上为待实现接口，当前没有现成训练命令。

独立配置项：augmentation_enabled、loss_scale、aug_relative_std、aug_mix_lambda、gradient_probe_enabled。首轮 gradient_mix_enabled、gradient_gate_enabled、optimizer_state_split_enabled 固定 false；传 true 立即报不支持，不用空实现伪装完成。编码前输出 commit/工作树、来源绝对路径、实际 forward、固定系数、loss、数据身份、参数分组和 optimizer 数量。

### 7.2 必要检查

1. P0 对冻结 fixed1 参考：同 initial state/batch 的预测、mean BCE、梯度、一次 Adam 更新一致；不用 legacy loss 数值当 Sample oracle。
2. κ=0：P2↔P0、P3↔P1；λ=0/增强关闭走原损失。CPU FP32 atol=1e-6、rtol=1e-5；CUDA 超出先定位，不随意放宽。
3. 不均衡 batch 核对 Sample 总和/(|B|T)；每原样本两视图权重和为 1；覆盖 batch=1、缺域、无增强对象。
4. domain/dense slice 零扰动，其余 field mask 正确；标签/row_id 不变，无新增 trainable 参数。
5. 四臂 initial state/每 epoch row_id 哈希相同，P2/P3 noise seed 相同；probe 不改正式状态、loader/RNG；None 不被误写零梯度。
6. 检查 checkpoint 恢复后的下一步，包括 model、m/v/step、所有 RNG、loader generator、permutation/cursor；若仅支持 epoch 边界恢复，保存并声明此边界，从边界重跑，不虚称任意步恢复。
7. 工程检查不看 test；正式严格 > 选模/共同 E50 预算；评价用未缩放真实 BCE。

数据/环境可用且检查通过后，在已交付的首轮执行任务内直接完成 32 次训练，无需再次研究设计确认。硬件不足可串行排队；确实 OOM 则报告阻塞，不偷偷改 batch/步数。

### 7.3 工件与完成标准

使用新 ID `GA31-v2` 和 attempt 子目录，不覆盖旧结果。保存：
resolved_config.json、source_manifest.json、data_manifest.json、split_row_ids.npz、init_state.pt、完整 model/optimizer/RNG/loader checkpoint、epoch_metrics.csv、per_cell_metrics.csv、batch_trace.csv、augmentation_stats.csv、optimizer_state_stats.csv、gradient_probe.csv、runtime_stats.json、stdout/stderr、失败/中断 ledger、selection_frozen.json、paired_effects.csv、DECISIONS.json、REPORT.md。

只在固定锚点保存各参数块 raw gradient/m/v/update 汇总，不保存每步全参数大数组。报告 32/32 完成或具体阻塞、必要检查、selected/E10/E50、五个效应及域任务交换，按第 6 节裁决后停止。

**准备度：首轮算法和协议已冻结，可以交付 Codex 编码、验收并运行；训练器尚待实现，真实数据/GPU 可用性由执行环境审计。** 本次修订没有产生新 AUC 或机制实验证据。

