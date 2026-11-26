# DESIGN SPECIFICATION — history_information_20261124

本规范供 IMPLEMENT、独立 REVIEW 和当前会话 RUN 使用。研究计划中的 05 定义数学，06 与 manifest 定义矩阵，08 定义科学决策，04、07、09、10、11 定义诊断、实现、统计、资源和来源。以下明确本轮执行中需要选择的复用入口、调度与状态边界。以当前 `TEST_POLICY.json` 记录授权。完成全套已触发训练、阶段决策、全局冻结和合格的冻结评价后，停在撰写最终研究报告之前。

## 1. 研究问题

自然 MDMT 的 A21 已保存 PAIR 辅助一阶历史。改变共享 Adam 二阶矩中 BCE、coupled WD 与 PAIR 的交叉项，或者用梯度产生时的预条件器保存辅助历史，能否在相同监督、步数和完整选点下提升排序？增量能否超出普通 PAIR、原 A21、组级步幅和相邻通用优化器，且兼顾六格？固定参数 EMA 的读出是否改变增量和阶段分布？

## 2. 假设与竞争解释

- C1 假设共同二阶矩的跨来源坐标缩放限制了有用历史。比较 C1−H0、D2 同状态 C1−M、以及完整 C1−M/P；分解 BCE×PAIR 与 WD×PAIR 判断其实际来源。
- C2 假设当前 D 重解释过去的 PAIR 方向，使用历时 D 具有额外价值。比较 C2−H0、时变 D 的交换差及对应真实排序变化，再用 M/P 判断组尺度或通用预条件历史是否足够。
- S 检验额外收益是否依赖 .99 长辅助核。D4 区分排序纠正、破坏、保持及 EMA 叠加。分别估计在线增量、同配置 EMA 增量、各自选点交互和同进度交互。
- 自然域计数、标签支持和父模型格 AUC 是不同变量。能触发时以 N 对照定位作用范围，以 HyperGate 比较具体迁移协议，以 GA31-R3 判断固定增强的条件交互。

## 3. 改变的变量

C1 只改变施于共享参数的一套二阶矩，ζ∈{.5,1}。C2 只改变辅助历史与预条件的运算顺序，α∈{.5,1}。四个配置分别完整训练；共同零点为 H0。后续变化仅由第 8 节触发，配置携带完整算法 ID、参数、βP、读出和作用域。固定 EMA 是同轨迹的第二读出。

## 4. 固定变量

- MovieLens 与 KuaiRand 的已核对原自然 CSV，原行序 80/10/10。KR 先切原行再映射 tab0/1/4，ML 使用 rating>3 和 rating>4，KR 使用 click 和 long_view。复用原 train-only encoder、OOV、列序、行 ID。Train 合计 800167、2006132 行，三域原计数分别为 155949/318599/325619、172782/1743898/89452；主 batch 40960，尾批 21927/40052，20/49 步每 epoch。
- M3oE fixed1、embedding16、4 个共享 expert、FCN[512,256,256,64]、gate 输入 detach、FP32、AMP/TF32 关闭。固定 2022–2026 各五个 seed，每轨迹 50 epoch。初始化及 loader RNG 沿源，loader 的生成器种子为 seed+11000，尾批保留。
- lr=.01，βB=.9，βP=.99，β₂=.999，ε=1e−8，coupled WD 分别为 ML 1e−4、KR 1e−5。全图 PAIR 与训练批相同，静态 q、2048 对、λPAIR=.05。B0/B1/H0 匹配的共同主体与预算见第 6 节。
- EMA 初始化同 online，整模型步成功后恰更新一次，ρML=.9828205985452511、ρKR=.995，原 buffers 复制。每 seed 每读出用完整 Validation 六格 Macro AUC 选 E1–E50 最早严格最大点，使用未四舍五入值，LogLoss 绑定同 checkpoint。

## 5. 数学与事务

### 5.1 继承的目标、时钟和运算

主矩阵使用原模型一次 forward 取概率与 tower hooks 的原生 logits。tower 顺序 3t+d，日志格序 2d+t。自然 BCE 对 [B,2] 全部元素取 mean。用 `boundary_sampler.CONDITIONS/boundaries/sample`，固定 24 条条件及原浮点顺序。活动条件的 q=.25/|A|+.75 w/Σw，保持原质量 w。原 SHA256 键 `dataset|seed|epoch|global_step|stream`，按原 condition/pair_indices 流有放回配对 2048 对；同键不含 arm。辅助为平均 softplus(−原 logit margin)，完整 L=L_base+λ(w/q)R。对自定义范围先从同一 R 图取 detached raw r，保留图，再一次完整 backward。B1 和 P_C2 无需 raw sweep。J 外参数始终接收完整目标的原生 Adam 更新。

记 native 梯度存在的参数时钟为 τ，全局辅助合法事件为 e∈{0,1}，局部 k 在 e=1 且 native 非 None 时只增一次。主 S_DENSE 每步要求 native 与已采 PAIR 的 raw r 均存在，所以 τ=t，k 等于原 A21 全局事件计数。B1 的原 `pair_clock` 保持原实现语义，外部事件数由 provenance/selected 汇总。native None 整参数跳过 WD、更新和两个时钟。N/HG 合法 unused raw r 在 native 存在时置同形零，PAIR 活动零梯度仍衰减 Q/U 并推进 k。无 PAIR 时保持 Q/U/k 并省略其注入，正常推进 m/V/τ。

以原数值路径先算 `mixed=p.grad.add(p, alpha=wd)`，再在 PAIR 步算 `b=mixed-(lambda_pair*ratio)*r`；h=(w/q)r，缺事件 h=0。b 含唯一一次 WD。mB=.9mB+.1b，事件步 Q 用原 `Q.mul_(betaP).add_(r, alpha=complement*ratio)`。mB/V 用 τ 偏差修正，Q/U 用 k；无事件读取时避免 k=0 的除法。继承的 FP32 系数字面量 .1/.01/.001 和原乘加顺序应保留。当前 ε 放在 sqrt 外。

### 5.2 新候选及对照

```text
Vζ ← .999 Vζ + .001[(1−ζ)mixed² + ζ(b² + λ²h²)]
Dζ = 1 / (sqrt(Vζ/(1−.999^τ)) + ε)
U  ← βP U + complement × Dζ h          [仅活动 PAIR，用本步已更新的 Dζ]
u  = Dζ mBhat + eλ[(1−α)Dζ Qhat + α Uhat]
Δθ = −lr × u
```

- ζ=0 走原 mixed·mixed `addcmul_`。ζ>0 用非负凸组合。α=0 先合并分子后用共同分母 `addcdiv_`；α>0 读 U 后直接相加。C2 的四元状态包含 Q，即使 α=1。C12 用自己轨迹的 Dζ 构造 U，由初始化完整训练。候选状态冷启动为零。
- C1 的同轨迹 `V0−Vsep` 等于以固定 β₂ 递推的 `2λbh`，且 `Vζ=(1−ζ)V0+ζVsep`。C2 在固定轨迹上的相对 H0 差为 `αλΣ_j κ[k,j](D[t_j]−D[t])h[t_j]`，其中 κ 为归一事件 EMA 权重。固定 D 和首个 PAIR 事件的差在算术容差内为零。
- S 仅令 βP=.9、辅助偏差修正为 1−.9^k，C2 的 Q/U 同步修改。P_C1 保留 .99 Q，但用 native τ 的 V_B 和 V_P 分母分别处理基础和辅助历史，V_B 跟踪 b²，V_P 跟踪未乘 λ 的 h²，无事件照常衰减 V_P，方向为 D_B mBhat+eλD_P Qhat。P_C2 用 V0/mixed 与本步 D0，逐 native 步积累 T=.9T+.1D0·mixed，方向 T/(1−.9^τ)，无需 Q/U。
- M 在自身轨迹从零维护 H0 和冻结目标两套状态，用同一批前向/反向的 b/h、时钟与支持。按 STAR_SKIP 和 EXPERT 固定坐标顺序做 FP64 完整拟提交步组范数，保留 H0 方向并乘目标/参考范数。参考范数零时提交零，登记 `MATCHED_ZERO` 或 `UNMATCHED_ZERO`；目标范数零时对非零参考提交零。S 目标使用独立 Q99 与 Q9，U9 对应 S。记录 FP32 实际位移舍入误差。
- native 与自定义坐标互斥地各提交一次，EMA 随后一次。observer、虚拟步和 M shadow 使用 detached/no-grad 张量。检测到真实非有限或部分提交失败时回到上个已验收的 RAM 原子事务并重放，物理重放成本单列。

## 6. 受影响模块与直接复用

主证据锁定 `358d3a465747eda6af1d55acc937759fa651a10b`，发表闭包 `c8675ae2923c15b6965884b140b5b646690cd95e`。本地源码可优先直接使用，`../phase0-3_git` 是 Git 对象查询位置。新代码限定在 `experiments/history_information_20261124/`。按 07 §2.2 留下入口及科学职责，可在包内共用状态或训练辅助函数。旧源码不需要编辑。

| 现有文件/入口 | 直接复用及新适配 |
| --- | --- |
| `experiments/conditional_information_momentum_20261102_v3/{protocol,boundary_sampler,objectives,optimizers}.py` | 主矩阵 B0=`v3.config('A0',dataset,0)`，B1=`v3.config('P00',dataset,.05)`，均用原 `v3.make_optimizer` 和原 `objectives.train_step`。保留原模型、原 PAIR RNG、配置、完整反向图及普通 Adam 参数组。 |
| `experiments/conditional_history_shared_update_20261122/history_controls.py` | H0=`history_config('A21',dataset,.05)`、`make_history_optimizer`、同原 `train_step`。保留原 .99 Q、原六格 M9/Q shadows、`selected`、`pair_clock`、`parameter_manifest` 和完整序列化。用源 `on_pre_step(pair,r)` 取同一 live 梯度，将新观察者状态放在外部，借同前态计算本步 vmix/vsep/U0，勿写回原 optimizer。 |
| `…/history_runner.py:step_anchor`、`a_localization.py:replay`、`b_virtual.py:registry`、`state.py:capture/restore/commit` | 复用捕获、重放、参数登记和 put/get/hash 口径。适配观察者、新队列、前台 RAM store 与当前选点，旧 runner 的数据加载、旧队列、旧 guardian 启动和 E10/E50-only 逻辑均不用作本轮入口。 |
| v3 vendored `ema02/{evaluate,protocol}.py`、底层 `vendor_ga31/{audited_data,metrics}.py` | 复用 feature schema/vocab/hash、原概率推理/指标、EMA 和 RNG。新增 Train/Validation 窄桥，只编码原前 90% 数据行；ML 保留原 pandas dtype/转换与 scaler，KR 保留原 chunk dtype、先全局 80/10 位置再 tab 过滤与 train-only vocabulary。可使用先核对 hash 的合法 Train/Validation cache；原 `v3.load_dataset()`、原发表 `prepare_cache()` 会构建完整 Test，运行前封存时禁止调用。 |
| `experiments/overnight_ab_20261103/{published_models,published_training,data_bridge}.py` | 发表源核对与条件 HyperGate/AutoIFS。HyperGate 复用原目标和分组。对新 PAIR 可一次调用原 `objective` 时挂 `core.towers` 的只读 logits hook，核对与 `_forward_terms` 返回的两列一致且 dropout 同图。若确实需要接口适配，以原 objective 等价测试为准。 |
| `experiments/gaussian_domain_coverage_20261104/objectives.py` 和 `…/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/{objectives,views,protocol}.py` | 仅 Gaussian 触发时，恢复 c8675ae GA31-R3 D0 原噪声、覆盖与基础目标。复用通用 dataset 键，避免 D0 源 KR-only 包装硬编码影响 ML。 |

B0/B1/H0 保持原 `config` 精确序列化，新 arm label 放外层。`lambda_pair=0` 按背景跳转 B0、HG_B0 或 G_B0，保留本背景的基础目标。主范围 .99 候选 ζ=α=0 原样跳转 H0，N/HG/G 跳各自同背景同范围历史父入口。B1/P 和 βP=.9 的短核不按数值零点改写身份。新增 M3oE C1/C2/P/M/S 可用兼容 `mode/shared/parameters/demand/selected/active/pair_clock/history/zero_grad/step` 的适配器走同一 `train_step`，P_C2 的目标接口走无需 raw 梯度的等义分支。先确定有限性再提交自定义与原生互补块。

S_DENSE 精确复用 `shared_name`，STAR_SKIP=`shared_weight/shared_bias/skip_conn.*/star_mlp.*`，EXPERT=`expert.*`，保留含 `domain_specific` 子名。N 包含 D/T/DT，包括直接从 `features.columns.index('domain_indicator')` 定位的域 embedding 三行，排除普通共享 EMBED 和 S_DENSE。按原 registry 登记每个名称、shape、dtype、顺序、组、可达性及必要的 mask。总元素 ML 773470、KR 1614254，N 分别 290318、952334；其余共有 483152、661920。HyperGate 以真实模块调用/六格梯度可达性注册共享 dense，排除共享 Embedding。

### 6.1 实现文件所有权与接口

| 交接 | 独立所有权 | 对其他实现的接口 |
| --- | --- | --- |
| 训练内核 | `__init__.py`, `protocol.py`, `sources.py`, `groups.py`, `objectives.py`, `optimizers.py` | `RunSpec` 提供不可变任务身份及 `to_dict/from_dict`。`sources.load_train_validation(dataset)` 交付源 `features` 与 Train/Validation `x/y/row_ids` 及冻结身份。`objectives.build_training(spec,bundle,device)` 返回 `model,optimizer,ema`；`step(...,pre_step=None)` 执行一次真实更新，在真实 forward/backward 后提交前把 context、PAIR、按参数顺序 raw r 传给 observer，native grads 可由 model 读出。优化器提供 `state_dict/load_state_dict` 及虚拟步所需的纯方向运算。 |
| 机械运行 | `observers.py`, `diagnostics.py`, `state.py`, `runner.py`, `scheduler.py` | 消费 RunSpec、训练内核和不可变队列。`runner.Supervisor` 持有 RAM 并提供 `run(queue)`、`selected_get(job_key,readout)`（返回模型状态、身份、实测哈希）、`close()`。D1/D2/D4 只产原始记录，通过同一 pre_step 与评价接口采集。 |
| 编排接口与统计 | `selection.py`, `analysis.py`, `cli.py` | `materialize` 生成 `QUEUE.json`（stage、jobs 列表为 RunSpec 字典、依赖与触发理由、前置收据、queue hash），`analyze` 消费原始文件并生成决策。`cli supervise` 持有一个 Supervisor，解析独立命令，调用其 RAM 接口做全局冻结与授权评价。新 Test 数据函数只能由 gate 成功后的评价路径调用。 |

三组共享工作区，采用上述文件所有权交接。模块内部可用原工具函数，跨组共享只使用这里的 RunSpec、训练批事务、记录 schema 和 Supervisor 接口。实现先后可独立验证；正式运行前合并核验接口与完整离线审查收据。

## 7. 实现、来源和状态要求

### 7.1 已核对与需验收项目

- 设计时核对了本地 A21 的 `history_controls/history_runner/protocol/state/a_localization/b_virtual` 和 v3 的 `objectives/optimizers/boundary_sampler/protocol` 与 358d3a4 指定 Git blob 一致。发表 `published_models/published_training/data_bridge` 及两张审计端点表与 c8675ae 对应 Git blob 一致。主代理另已核定两个主核心包 81 个文件、两发表/历史包及 protocol 共 71 个文件、Gaussian 6 个文件，以及固定上下文 01–03。S0 记录实际导入闭包的 Git blob/SHA256，train/validation arrays、行 ID、encoder/scaler 仍须从实际入口核对冻结值。数据桥应一次准备每个数据的同身份只读 Train/Validation 输入，供 worker 复用。
- D0 直接复用 6,480 行来源端点及派生 1,050/210 行，原始 A_HISTORY 端点 Git blob 为 `d420cdf0043f88b66a6bb886e0c73d72175023a8`。H20/H22、EXP080 历史冻结数据以及 EXP081 SCALE 作为原身份数值证据，均不重训，也不计入新增重复。旧 KR A21 同 PAIR Test 正向与 Validation 负向分别保留。
- 旧 1122 训练 guardian 中的 PID 已退出，残留 socket 无监听。归档收据与最终数值可沿原身份引用；H0 的新 Vsep/U 历史、四锚点和 B1/H0 的共同 E25 概率无法由旧标量补齐。此次启用冻结后新评价，B0/B1/H0 还需两读出 selected RAM，按 S1 从冻结初始化每 dataset/seed/臂各一次重建，最多 30 条。同一次收集曲线、诊断、D4、selected；不为比较再重跑。若在 S0 实际发现身份完整的活态，按能力剔除相应重建。
- 发表 HG/AutoIFS 的 `ENDPOINTS.csv` 已含每模型两数据各五 seed 的历史 Validation/Test selected 行，与 c8675ae 文件逐 blob 相符。以人口、指标、config、E1–50 或 reuse E1–50 选点、selection_id/model_hash 核实实际 `PUBLISHED_BASELINE` 行，合格以 `NUMERIC_REUSE` 纳入事先冻结名单。`A_REUSE_AUDIT.csv` 的 `INCOMPATIBLE/REPLAY_MISSING` 是历史派发前的旧输入/待补记录。HG_B0 若作为迁移臂需要新的 EMA/selected，两读出无完整复用时按 S3d 补齐，与正式 HG 槽位按身份去重。仅修复确实不匹配的正式发表基线槽位。主代理已在 `/tmp/mdmt_sota_sources_2526/{RecStudio,AutoIFS-pinned}` 恢复固定版本，核实 HEAD 及六个外部文件的 SHA256/Git blob。
- GA31-R3 条件模块的 kappa ML=.05/KR=.1、λaug=.5。以同一次 clean forward 作为 PAIR 图，D0 view 增加 `.5*sum_D0(loss_view−loss_clean)/(2B)`。非域字段逐行 detached RMS、设备本地逐字段 torch.randn，固定 `GA31-R3|dataset|seed|train|global_step|gaussian|1` 前 8 bytes 小端并取 63-bit，step 从 1 跨 epoch 连续。核对原首/尾批适配和噪声身份。真实 HG batch 1024、eval 512、lr=.001、WD0，ML782/KR1960 步，原 objective 全 components、dropout 及默认 Adam flags。直接移植 βP/ρ 的每步数值，记录不同样本跨度。AutoIFS 如需恢复，原 batch4096 search10/reuse50、rewind、hard ticket、frozen gate、fresh reuse optimizer 与修正接口，选点只在 reuse。

### 7.2 前台监督及 RAM 生命周期

旧 guardian 缺席时，`cli supervise` 是由当前会话 RUN 子代理的工具托管、可读 stdout/写 stdin 的一个持久前台 PTY 进程。此进程从 S1 的首次 RUN 保存同一 `RamStore` 到冻结评价收据完成，负责当前会话的机械训练以及 worker 的 IPC。各 GPU worker 由它启动和收割，经父子进程 IPC 提交/读取 packet；主进程持有状态和 owner generation。支持一个作用明确的控制输入方式即可，避免另起 guardian、后台进程或 AI 会话。

`audit/check/materialize/analyze` 依旧可独立 shell 执行；状态无关。前台 `supervise` 接受逐条明确发出的原 CLI 子命令及 RunSpec/queue 文件身份，例如分别提交 `run --stage S1`、`run --stage S2`、`freeze`、`evaluate --split test`。每条命令返回自己的命令 ID、结束状态和产物路径，再等待后续命令；阶段授权来自主代理通过显式文件交接，运行器不自行分析或展开矩阵。若提供独立 CLI 客户端，仍只向该前台处理器提交明确请求，训练状态全程归该处理器。不能以多次各自退出的独立 CLI 替代持有 selected 状态的同一进程。RUN 子代理在完成最后一次 Test 收据前持续监督，不把 RAM 所有权交给退出后继续运行的服务。

原子 epoch packet 含在线参数/buffers/mode、原生 optimizer 私有时钟及组、m/V/Q/U/T、全局/局部时钟、新旧 shadows、EMA、Python/NumPy/CPU/实际 CUDA RNG、loader generator/permutation/cursor、双读出最佳状态和身份。每次 put 后实际 get 并重算 hash，核对 owner generation、job/config/data/source/group。恢复只从合格主状态，保持同设备同源重放身份。锚点 packet 用 `POST_BACKWARD_PRE_STEP`，cursor 是该批起始 offset，额外保留 pre-forward RNG/buffers、post-forward buffers、mixed/None mask/raw r、PAIR 和 live 后态三哈希。锚点消耗后及时释放，淘汰网格配置在确定不进冻结后释放其 selected，保留数值。训练张量禁止写入普通文件、临时文件或压缩包。

每次实际派发、admission 和原子边界检查用户命令与 S0 已登记的两个工作区当前停止控制。旧研究的终止收据按其原作用域记录，活动控制按本轮权限收据核对。按每张独立 GPU 当前显存 <85%、当前功耗 <85%、宿主 RAM <90% 派发，启动后实测再看下一 worker。真实 OOM 后追加完整进程驻留量和 CUDA usable 容量的物理可装入检查，保留原 85/85/90 条件与 batch。旧状态不可用、资源等待、源码不符、科学关闭分别记录。现场两张 RTX 3090 可见，runner 每次重新采样。

## 8. 消融、诊断和条件矩阵

| 阶段 | 执行与转移 |
| --- | --- |
| S0 | 产出 `SOURCE_MANIFEST.json`、`DATA_MANIFEST.json`、`PARAMETER_SCOPE_MANIFEST.json`、`REUSE_REGISTER.csv`、权限与必要 blocker。固定名单和基线修复按 §7。 |
| S1 | B0/B1/H0 已知状态缺口按需补齐。D0 数值直接复用。D1–D4 按下面细则执行；只因身份/数值/状态失效暂停受影响轨迹，局部科学结果任何方向皆不影响 S2。 |
| S2 | C1_Z050、C1_Z100、C2_A050、C2_A100 各两数据五 seed full50，共 40 条。online/fixed_ema 每轨迹双读出，0 点仅使用 H0，保留四配置完整曲线。 |
| S3a | 入门家族每数据选择一个同一家族参数，按每家族唯一读出派发 S_C1 和/或 S_C2；每家族 10 条。 |
| S3b | 两数据 S 完成后绑定家族 C/S 代表，派发其 M 和预定 P；每家族 20 条。 |
| S3c | 两个原 C 在同一数据、同一读出通过全部控制且均未被 S 替代，才创建 C12（10 条）；否则记录门未触发。完成 C12 或记未触发后定唯一 Cstar。 |
| S3d | 对合格 Cstar 按门执行 H0_N/Cstar_N（20）、HG_B0/B1/H0/Cstar（40）、G_B0/B1/H0/Cstar（40）。每模块仍两数据五 seed，已同身份复用槽位可减物理成本。 |
| 冻结与 Test | 所有已触发模块完成并完成阶段决策，冻结一次，Test 只评已冻结 selected，随后停在撰写最终报告之前。 |

D1 在 H0/S2 按 STAR_SKIP 和 EXPERT 记录 04 §4.1 的 FP64 标量，包含按自然格质量/PAIR 支持、交叉矩的 BCE/WD 分解、Q、DQ、U 和交换差，按 E1–10/11–25/26–50 汇总。H0 的 E5/20/35/50 末批，两个数据五 seed 共 40 锚点，每锚点独立 eval clone 一次 Train probe forward，至多六次 `autograd.grad`；每可支持格固定 2,048 个逆包含概率采样 pair，跨四锚点一致，缺类写原因。报告 raw 和自然质量 (n_d/(2N)) 加权探针梯度。

D2 每锚点先重放真实 H0 步，模型/optimizer/EMA 三哈希精确相同后，计算另外十个反事实。分支是 H0、四核心、对应四组半径 M、M9_V0、M9_V1，后两者用本轨迹 `mB+sum(M9_cell)` 与 τ 的 .9 偏差修正。J 外沿真实步，J 内同一前态/同批梯度变换各自一步，使用 post-forward buffers 和自己的后态 EMA。面板 `P_TRAIN,V_A,V_B` 按 04 §5.3 固定 SHA256 行/分半/分层键，每 domain×joint-label 最多 512，记录 π；每可支持面板格固定 8,192 对。评估 AUC/LL、逆 π 加权 AUC、纠正/破坏/ties、步幅和成本；逐分支释放。40 参考、400 替代和 2,640 整面板评价分别计成本，实际分块 forward 另计。全部诊断前后验证 live 哈希守恒。

D3 做第 10 节代数和关键事务验收。D4 直接收集 B1/H0/四核心的共同 E10/E25/E50 的完整 Validation FP32 概率，同格固定 32,768 对，用 04 §5.3 命名独立的 `purpose` 流，固定跨臂/epoch/读出。排序分数 (I(p^+>p^-)+.5I(p^+=p^-))；记录纠正/破坏、净值、保持/遗忘、A21/EMA 纠正及破坏集合重叠，缺支持与空并集为 null。分别做各自 selected 部署交互和同进度交互，同一候选交互用同参数。诊断行、pair、stage 和读出不增加 n。

### 8.1 决策执行口径

定义 parent increment 为同数据同读出的五 seed selected mean ΔMacro 对 B1 与 H0 都严格 >1e−4、对 B0 ≥0。安全是对 B0/B1/H0 每一参照的每格五 seed mean ΔAUC ≥−.001；缺失配对不能通过门。

1. S2 每家族/数据/读出从安全非零配置优先，按完整 Macro 最大选择；差≤1e−4 用较小非零 ζ/α。若无安全配置，仍保留最佳非零及损害。登记 H0 零点及两配置相对父的完整结果。家族只要其任一数据的 online 选择行安全且有 parent increment，控制读出固定 online。否则固定有安全 parent increment 的 EMA；若只有过均值门的 trade-off，允许控制并冻结出现该信号的读出，两读出均仅 trade-off 时优先 online，同时标记后续扩展不合格。若任何读出均无 parent increment，关闭该家族。独立审查应在跑前确认这条对 08 §3/§3.1 的 trade-off 补充，以及选参触发指冻结的选择行。
2. S 相对 C 的两数据等权 selected 均值差 ≥−1e−4、两数据都安全、至少一数据有 parent increment 时，跨两数据统一采纳 S。其余情况保留 C。M 绑定此次唯一代表，P 保持预定公式。代表相对 M 或 P 的平均差 ≤1e−4 属该数据控制覆盖。代表要在同一个数据/读出超过 M/P，原 C 还须超过 S，三个必要增量皆严格 >1e−4，并满足 parent increment 与安全；另一数据也需安全才能扩展。对照比较报告配对区间，覆盖对应本试验行动分辨率。
3. C12 用两个冻结原 C 的数据别参数与共同读出。其扩展资格要求同一数据对 B1/H0 与 `M_C1,S_C1,P_C1,M_C2,S_C2,P_C2` 均 >1e−4、对 B0 ≥0，两数据均安全，且相对最好合格单组件的两数据平均增量 >1e−4。组合激活的额外状态计入复杂度。
4. 从满足资格的 online 代表优先确定 Cstar；没有时才比较 EMA 依赖候选。同一读出下，以相对 B1 的两数据五 seed 增量等权平均排序，差≤1e−4 时先选实际活跃状态元素更少者，再用 `S_C1,C1,S_C2,C2,C12` 顺序。统一家族/算法形式，可以两数据取各自冻结参数。M/P 覆盖的家族不进入 Cstar。N/HG 在这个 Cstar 通过对应独立增量及两数据保护门时都触发。Gaussian 另需同配置 fixedEMA 至少一数据有 parent increment、两数据安全、GA31-R3 身份通过。未触发的项写 `NOT_TRIGGERED` 与对应实际判值，触发后遇身份故障写 `TECHNICAL_PAUSED`。
5. 冻结由所有已完成的 Validation 决策驱动，冻结输入包含全部触发结案状态、源/人口/算法/组 hash、配置、五 seed、selected epoch/model hash、读出、比较集和 Test 权限。新 Test 的槽位限 B0/B1/H0、每家族每数据在 S2 冻结的一个代表、触发 S/M/P/C12/N/HG/Gaussian，以及固定发表模型名单。新机制保留双读出，纯发表 baseline 用 online；匹配的历史发表 Test 数值按原来源复用，每个新评价任务必须在同前台 owner 做实际 selected get/hash。按当前 `TEST_POLICY.json` 在独立受控命令中启封，不继承旧周期 Test 访问；若现行活动 scope 使用显式 stop 门则同时核对其合格记录。历史已暴露 holdout 只评冻结 selected，一轮后禁止用结果调参、再选点、再加臂。
6. 正式每数据的强方法门是相对冻结发表名单中该数据五 seed Test Macro 均值最高者 Δ≥.003 且每格 Δ≥−.001，仍报告相对固定名单其他模型及匹配内部参照的增量。保留 08 的六种结论候选标签和必要的来源，只写阶段决策和数值，本轮停止点之前无最终研究报告和上传。

## 9. 预期日志

保留 07/09 定义的 RunSpec、完整曲线、端点、配对差与汇总、配置选择、决策、costs、state receipts、最终冻结与访问收据。每个条件臂保留机器可读终态，已触发结果含每 seed 完整数值。原始记录含逻辑轨迹 ID、attempt、seed、数据/算法/超参、作用域/组、来源/人口/预处理、读出、epoch/step、checkpoint、optimizer 和模型身份。步骤摘要附 PAIR 事件、条件/样本哈希、w/q、自然和辅助时钟、支持/None、交叉项符号与重构、历史范数、真实位移、诊断及成本。数据已无则写具体缺项原因。

六格按原模型概率与精确 ties 的 AUC 及概率 BCE LogLoss，以 score clamp log floor −100，格缺类 AUC 与整体 Macro 为 null。报告格 count/positive/negative、父 AUC、域/任务/最差格，等权 Macro、自然流量加权 Traffic、逐任务跨域 Pooled。seed 为独立单位，使用五 seed 配对差、ddof=1 SD、df4 95% t 配对 CI、正/零/负数。训练/验证/面板/诊断推理，主成功更新/物理重放，wall/worker 用时，RAM/完整 GPU 进程驻留峰值分别记录；诊断数或 readout 次数不计新训练。阶段诊断/选择可以保留机器产物，最终报告与远端发布在本次停止点之外。

## 10. 独立 reviewer 的最小验收

1. Check 原生闭包、Train/Validation hashes、五 seed、scope 总数和原首/尾批，核对固定 q/行/PAIR 以及 once-forward 全图梯度。主 B0/B1/H0 使用相同原路径；对原 A21/候选零点比较含源 shadow 与 optimizer config 的完整状态和 EMA，要求同设备精确一致。
2. 小张量验证 C1 正/负 cross 与非负凸组合，C2 恒定 D/首事件及变化 D 差，等 β 同时轴线性等价，活动零 r、合法 raw None、native None、单次无 PAIR 和冷态全无 PAIR。含 WD/zero 数据梯度的单次提交与 EMA；普通 Adam c=.1 冷/热尺度等价只验新适配。M/P/S 按第 5 节核对，C2/U 使用当前 D 且不重复预条件。
3. 两数据仅用 seed2022 的真实首/尾批，验证 PAIR 图和 scope 外梯度、canonical 路由/原 native 行为、V 和 clocks、原保留尾批；条件分支按触发才验 HG native objective/logits/dropout、N mask/时钟、Gaussian 首尾噪声及 base、AutoIFS 阶段。如果问题能改变科学结果，针对性修复与复审。
4. 验证 D2 锚点三哈希、诊断不改变 live 及 RNG、两类 packet 游标语义、前台库跨两个独立命令读回、同源中断恢复与连续运行状态一致、权限/角色隔离、最终 freeze/Test 的 hash 与单次性。D2 完整 40 锚点是 S1 输出，最小工程验收只需源真步执行路径样例。冻结后无评价权限或选中态读不回，拒绝新 Test。
5. FP64 数学 rtol=1e−10/atol=1e−12，FP32 新运算独立参考 rtol=1e−5/atol=1e−7，源六格 shadow 重构 rtol=1e−4/atol=1e−6。原生委托、实际回读、同设备 H0 重放采用精确 hash。阶段门用小型结构化行检查 >、≥、双读出、唯一控制组、C12 六个控制、缺项及 Test 无适应性，不追加科学训练。
6. 由 coder 提交修改文件、检查输入身份/实测误差与 hash 和 `CODER_IMPLEMENTATION_REPORT`，由独立 reviewer 按本文件与 04–11 检查并给 verdict。正式 RUN 需要 PASS 或已改正阻断项的 PASS WITH FIXES，现行停止与资源条件仍需实际核对。

## 11. 证伪条件

- C1/C2 四个配置均完成时，某家族对同 PAIR 和 H0 无可分辨 selected 增量，关闭该训练算法定义。局部变化证明运算产生后果，其 AUC 收益仍由 full50 裁定。
- M 覆盖代表时，支持组步幅作为当前充分解释。P 覆盖时，保留分源或通用预条件近邻；S 覆盖时，撤去该候选需要 .99 长核的主张。WD cross 主导时，按 coupled 正则化与预条件证据解释。没有在线增量而 EMA 满足条件时，只支持 EMA 背景组件。
- 共同进度的历史交换差或排序净修复不足、完整 Macro 改善但特定格越线，分别收窄机制和应用范围。N 结果允许否定共享独占解释，HG 失败限定本协议迁移，Gaussian 失败限定本背景。小域/大域是否发生代价由观测支持判断。
- 技术前提失效时记录受影响槽位与可恢复步骤。实际新源码、训练数据窄桥、运行时 GPU、前台 RAM 事务和训练指标由 IMPLEMENT/REVIEW/RUN 验证。

计划入口与冻结公式：[00](../../research_plans/26-11-24_history_information/00_EXECUTION_RUNBOOK.md)、[04](../../research_plans/26-11-24_history_information/04_MECHANISM_DIAGNOSTICS.md)、[05](../../research_plans/26-11-24_history_information/05_CANDIDATE_ALGORITHMS.md)、[06](../../research_plans/26-11-24_history_information/06_EXPERIMENT_MATRIX.md)、[07](../../research_plans/26-11-24_history_information/07_IMPLEMENTATION_SPEC.md)、[08](../../research_plans/26-11-24_history_information/08_DECISION_RULES.md)、[09](../../research_plans/26-11-24_history_information/09_METRICS_AND_STATISTICS.md)、[10](../../research_plans/26-11-24_history_information/10_RESOURCES_AND_SCHEDULER.md)、[11](../../research_plans/26-11-24_history_information/11_SOURCE_AND_REUSE_REGISTER.md)、[manifest](../../research_plans/26-11-24_history_information/experiment_manifest.json)。
