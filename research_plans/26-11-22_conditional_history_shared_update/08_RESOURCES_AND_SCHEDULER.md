# 资源预算、监督调度、RAM 状态与失败恢复

## 1. 当前结论与预算口径

本计划的必需完整训练是 **12 条 A 组件补全 + 10 条 B 同源 CLEAN 状态重建，共 22 条、1100 epoch、37950 次真实 Adam/EMA 更新**。已有五种子数值不因本轮再引用而增加训练成本，状态重建不增加独立统计单位。通常先完成 22 条；A 历史独立增量通过后，加 A21/A22 两种简单替代得到 42 条。其余分支只在 [阶段门](09_DECISION_RULES.md) 触发后展开。[完整矩阵](05_EXPERIMENT_MATRIX.md) 是唯一任务清单。

所有已登记条件分支全部触发且均覆盖两数据集时，理论合计 **172 条完整物理轨迹，其中科学新训练 152 条、状态重建最多 20 条**。这是有限矩阵的合计上界，不是默认队列、固定并发数、必须跑满的目标或失败次数上限。恢复产生的实际额外成本另记。当前交付只编写和提交规划，不启动这些训练。

以下成本必须分账：

| 成本类型 | 计量单位 | 是否增加正式 n | 典型用途 |
|---|---|---:|---|
| `NEW_SCIENTIFIC_TRAIN` | 新配置完整 50 epoch、实际 Adam/EMA、worker 秒 | 每配置仍只有五个指定 seed | A 缺项、通过阶段门的新候选 |
| `REUSE_NUMERIC` | 读取端点/身份/选点记录 | 否 | A00/A20、旧 P00/P10、正式发表模型 |
| `STATE_RECONSTRUCTION` | 同身份重放的物理 epoch/更新 | 否 | B 的真实 Adam 锚点，必要的 A 目标轨迹 |
| `VIRTUAL_ADAM` | 冻结状态克隆上的一次原生 Adam 调用 | 否 | B 单步反事实；不能计作 live 训练 |
| `COUNTERFACTUAL_SUFFIX` | 分叉短段新增的实际 Adam/EMA 调用 | 否 | 一次 LOO 脉冲的传播 |
| `DIAGNOSTIC_SWEEP/FORWARD` | autograd sweep、前向次数、样本数与秒 | 否 | 六格分解、伪分组、排序/背景读出 |
| `ENGINEERING_CHECK` | 原 batch 正确性/容量/恢复检查 | 否 | 工程准入，不作单种子性能淘汰 |
| `FAILED_ATTEMPT/RECOVERY_OVERHEAD` | 已发生的物理 epoch、调用、秒 | 否 | OOM、失联、失去完整状态后的重建 |

## 2. 数据规模与完整训练成本

源协议为两自然数据集、M3oE fixed1、FP32、batch 40960、50 epoch；保留原尾批、train-only 预处理和原始训练人口。不能减 batch、改精度或人口来绕过资源门。[R1]、[R2]

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| train 行数 | 800167 | 2006132 |
| validation / test 行数，各 | 100021 | 250767 |
| 每 epoch Adam/EMA 更新 | 20 | 49 |
| 自然尾批行数 | 21927 | 40052 |
| 每条 50 epoch 更新 | 1000 | 2450 |
| 模型参数数 | 773470 | 1614254 |
| 单份 FP32 模型张量字节 | 3093880 | 6457016 |
| 固定 EMA ρ | .9828205985452511 | .995 |
| coupled weight decay | 1e−4 | 1e−5 |

其他普通 Adam 设置为 lr=.01、betas=(.9,.999)、eps=1e−8；无 AMP/TF32。online/EMA 是同轨迹两种读出，不能各算一条训练。A 的历史处理额外计算单列，不改变每 batch 一次真实参数提交与一次 EMA 的语义。

### 2.1 必需工作

| 工作 | ML 完整轨迹 | KR 完整轨迹 | 物理轨迹 | epoch | 真实 Adam/EMA |
|---|---:|---:|---:|---:|---:|
| A10=P00、A30=P10，补 seeds 2023/2025/2026 | 6 | 6 | 12 | 600 | 20700 |
| B_CORE 同源 CLEAN 重建，五 seed/数据 | 5 | 5 | 10 | 500 | 17250 |
| 必需合计 | 11 | 11 | **22** | **1100** | **37950** |

A00 与 A20=P01 的既有五 seed、P00/P10 的旧两 seed 已有完整训练数值。先按 [来源登记](12_SOURCE_AND_REUSE_REGISTER.md) 校验身份，再补缺项。B 的十条不是重证 CLEAN/EMA 有效，而是恢复旧报告明确已释放的模型/Adam/EMA 状态，产生新的任务来源干预证据。若现场验收确有同身份、全字段、所需锚点可获得的完整 RAM 状态，则逐条扣除重建；旧 PID、历史收据或 selected-only 模型不满足条件。

如其他合法任务刚好正在从头执行同身份 CLEAN，可在其训练期间挂只读 observer，共用一次物理轨迹。不得额外训练 A00 后再独立重建完全同身份 B_CLEAN。

### 2.2 条件分支完整预算

| 分支 | 触发理由 | 新科学训练 | 状态重建 | 新增 epoch | 新增真实 Adam/EMA |
|---|---|---:|---:|---:|---:|
| A21/A22 | A20−A10 正向后，比较单份辅助历史与单份较长 Adam 历史 | 20 | 0 | 1000 | 34500 |
| A 目标轨迹真实状态 | 有效历史需要定位共享更新代价，既有所需状态已释放 | 0 | 最多10 | 500 | 17250 |
| λ=.025 匹配剂量对照 | 有独立收益但六格受损；历史胜者与相同 λ 普通 Adam 各两数据五 seed | 20 | 0 | 1000 | 34500 |
| 两种历史作用域 | 已定位作用组、降 λ 仍不足以保护；每臂两数据五 seed | 20 | 0 | 1000 | 34500 |
| B_ADAPT/B_STATIC/B_UNIFORM，η=.25/.5 | B 真实效用、身份信息及可部署性阶段门通过 | 60 | 0 | 3000 | 103500 |
| 一套冻结联合配置 | A/B 各自有独立增量且信息/作用位置互补 | 10 | 0 | 500 | 17250 |
| Gaussian D0 组件补充 | 已有最终候选，且匹配的冻结增强对照身份可复用 | 10 | 0 | 500 | 17250 |
| 条件分支合计 | 全部触发的理论情形 | **140** | **10** | **7500** | **258750** |
| 连同必需任务 | 不含失败恢复及诊断 | **152** | **20** | **8600** | **296700** |

λ=.05 的历史候选和普通 Adam 对照已存在时直接复用。新搜 λ=.025 必须同时训练匹配的普通 Adam：只补历史臂无法判断最终剂量下历史组件的独立作用，因此剂量阶段预算是 **20 条**，不能写成 10 条。以上 172 条包含该必要匹配控制。

各数据集的科学结论分别裁定；任一数据的合格信号可以触发后续阶段，而正式新配置一旦触发，均执行两数据×五 seed的完整登记范围，不能按某个数据或seed的早期效益裁掉配对项。实际消耗低于上界来自未触发分支和合法状态复用。若 A 目标是刚训练的 A21/A22 且完整 RAM 尚在，其目标重建直接减为零。若新增数据/协议确实破坏既有发表模型或 Gaussian 对照身份，先记录具体差异并更新控制矩阵；不能无声借用不同身份，也不能自动把全部旧基线重跑一遍。正式发表模型新增训练默认 **0**。

## 3. B_CORE 诊断与短段成本

四个预注册锚点为 E5/E20/E35/E50 的最后一个真实 batch 更新前；两数据×五 seed×四锚点=40。锚点按 live 次序流式处理，分支处理完立即释放，父轨迹完整可恢复状态继续保留。[B 设计](04_DIRECTION_B_DESIGN.md) 决定数学语义。

| 冻结状态工作 | 每锚点 | 40 锚点合计 | 资源/统计含义 |
|---|---:|---:|---|
| JOINT 克隆一致性 | 1 virtual Adam | 40 | 与随后原生 live step 精确校验 |
| EMBED/S_DENSE/N 三作用域：ZERO_DATA+6 ONLY+6 LOO | 39 virtual Adam | 1560 | 共用同一真实 Adam 前态；旧E/R/N按06映射 |
| 两套伪六分组，只对 EMBED 做 LOO | 12 virtual Adam | 480 | 两次独立六选一，与真任务选择预算相同 |
| 合计 | **52** | **2080** | 不改父轨迹；不增加 seed |
| 真六格及两套伪六格梯度 | 18 额外 autograd sweeps | **720** | 原生 live backward 另计，不以分解和替代它 |
| 同范数、同位移跨背景、排序/两态读出 | 实际额外前向 | 按实际次数与样本数记录 | 不伪装成零成本，也不虚构固定训练轮数 |

短段 `B_PULSE` 按 [04](04_DIRECTION_B_DESIGN.md) 的dataset门触发；一旦某数据触发，完整执行该数据五 seed×E20/E35×真任务/partition0伪分组两动作，不按seed或anchor挑选阳性。JOINT复用父轨迹未来段。两数据都触发时，共40个逻辑短段，是下表的上界。ML K=20、KR K=49，均**包含第一个 LOO 更新**；后面 K−1 步恢复普通联合目标。

| 短段计数 | ML | KR | 合计 |
|---|---:|---:|---:|
| 新增逻辑段数 | 20 | 20 | 40 |
| 含首步的段内逻辑 Adam/EMA | 400 | 980 | **1380** |
| 已由 B0 的2080 virtual Adam计算并保留的所选首步 | 20 | 20 | **40** |
| 复用首步后的新增实际 Adam/EMA 调用 | **380** | **960** | **1340** |

首步须保留完整模型、m/v/step和EMA后态，不能只保留参数再重新计算。若技术上未能复用首步而实际重算，物理账至多增加40次并注明原因；科学逻辑仍为1380步上界。不能把1380与2080完整相加，同时声称首步已经复用。四十个逻辑段也不是四十个新 seed。训练面板决定NO_ACTION时，该逻辑槽必须保留并按JOINT零效应报告，但可以合法别名JOINT、扣除重复物理计算。

B0/B1的单步及历史背景核心对比在两数据全五 seed必做，不随A停止取消。B_PULSE未触发的数据记NOT_TRIGGERED。两数据短段全部触发、首步复用且没有NO_ACTION别名扣减时，核心父训练37950次、B0克隆2080次、短段新增1340次，合计 **最多41370次物理 Adam 调用**，其中只有37950次属于22条完整主轨迹。总计不含后续候选选择器的克隆、A归因、工程检查与失败恢复。

B1_WINDOW的父轨迹20个窗口必做，包含 **690组逐步面板前向**。两数据脉冲均触发时，JOINT/真/伪三条路径的同权历史预测平均合计最多 **2070组逐步面板前向**，其中父窗口690组已计入，额外分支最多再增加1380组；不增加Adam调用。每组的实际panel/readout前向次数与样本数继续展开记录。不能把预测平均写成免费，也不因此保存全部参数快照到磁盘。B_SEARCH属于正式新算法训练，触发后仍为两数据×三臂×两η×五seed的完整60槽，与B_PULSE按dataset触发的机制短段规则区分。

### A 的历史归因额外成本

A目标轨迹的六格shadow必须从初始化开始维护，不能到锚点才置零。ECCM/COLL原step已经计算独立PAIR梯度，观察器主要增加张量运算；A22/S99普通Adam原路径没有这项独立梯度，如需完整六格历史分解，每个有效PAIR步额外一次shared autograd，两数据五seed单配置最多 **17250次额外sweeps**，按实际记录。A的锚点克隆、六格读出、同范数控制也单列，不包含在B的720/2080中。

新候选可在首次训练时预挂中立observer，直接复用完整历史；未预挂且后续确需归因时，用A总计最多10条的STATE_REPLAY额度恢复，不能每个候选另开十条重放。shadow及其可恢复状态仍只在RAM保存。

## 4. 时间估算与完成时间更新

| 已测历史成本来源 | ML 每50epoch worker秒 | KR 每50epoch worker秒 | 本次用途 |
|---|---:|---:|---|
| 11/03 P01 新增2023/2025/2026，`cost_role=new_training` | 557.108673 | 1349.980900 | A 配对家族初始代理，不冒充 P00/P10 已测耗时 |
| 11/15 CLEAN，后续 NQR 报告引用 | 407.738488 | 1155.568793 | B 重建纯训练底座 |

由此，A十二条约3.1785 worker-hours，B十条底座约2.1713 worker-hours，合计约 **5.35 worker-hours**。[R2]、[R3] 这不包含720次梯度分解、2080次虚拟Adam、1340次短段新增更新、验证面板、Test、报告与失败恢复。共享服务器负载、PAIR路径差异和观察器会改变实际耗时，所以不把5.35称为双卡墙钟或承诺数小时内完成。

首个原 batch 工程检查后，按每个配置分别记录：完整进程峰值、epoch秒、diagnostic秒、selected/full-validation秒、额外sweep/forward/clone次数。首轮真实完成一个epoch后更新剩余预算，不为估计效率另开多 seed 性能试验。之后用已完成同类job的实际时间更新 ETA；`elapsed_wall_seconds`、`worker_seconds_sum`、`gpu_busy_seconds`分别记录。并发墙钟包含等待外部任务和当前资源门的时间，不能用于宣称算法的隔离效率。

B_ADAPT/B_STATIC/B_UNIFORM按最终04定义在E1–50全程运行，不设置额外延迟启用期。60条候选全部触发时，主训练Adam/EMA仍为103500次；ADAPT的20条轨迹×50epoch×七个虚拟分支（JOINT+六个η候选）另增加 **7000次clone Adam**。七分支先从同一prestate独立评估，再执行一次live选中更新；不复用尚未执行的live JOINT，也不产生第二次live step。JOINT虚拟分支使用已计算的联合梯度，不额外增加源autograd。所有epoch都active时，ADAPT额外EMBED autograd上界为 **39500 sweeps**（首batch六源6000，其余batch33500）；STATIC上界 **34500 sweeps**，UNIFORM为0，合计最多 **74000次EMBED-only sweeps**。实际NO_ACTION可减少这些物理调用，保持逻辑槽完整。EMBED-only sweep不等价于全参数完整backward，耗时必须实测。

若最终冻结B_ADAPT，AB_FROZEN和G_FROZEN各10条轨迹仍沿用七分支选择，每个阶段另有最多3500次clone Adam；两阶段全部触发最多再加7000次。若冻结STATIC/UNIFORM则无该选择器开销。以上选择器调用单独计费，不计入296700次主训练更新，也不增加科学轨迹数。

完整训练数量相同不等于计算公平。候选报告同时列标准训练更新、辅助梯度、选择器克隆、评估前向、端到端秒及峰值，不只报告“50epoch”。

## 5. GPU/主机准入与当前会话监督

沿用当前生效的逐卡准入规则。对设备 g，每次启动前读取当前遥测：

~~~text
current_gpu_memory_fraction(g) < 0.85
AND current_gpu_power_fraction(g) < 0.85
AND current_host_ram_fraction < 0.90
AND no_unresolved_startup_for_this_device
AND current_used_bytes(g) + measured_incoming_full_process_bytes
    <= measured_cuda_usable_total_bytes(g)
~~~

最后一项继承已发生 CUDA OOM 后的物理容量修正；它是能否真正装入 CUDA 可用容量的检查，不能改成“预计显存小于85%”。不加入预计worker功耗，不设置固定worker数，不要求两卡同时满足。[R4]、[R5]

历史 RTX3090 测得 CUDA 可用总量25307578368B（24135.1875MiB），NVML名义总量25769803776B（24576MiB）；完整进程驻留曾为ML4306MiB/KR5284MiB，含约316MiB非torch部分。它们是历史容量参照，不代替本轮测量。PAIR旧torch reserved峰值ML1350565888B/KR1631584256B也不能作为完整进程用量。[R3]、[R5]

调度顺序：

1. 核验当前前台runner、stop controls、SOURCE_READY、review PASS、数据身份及本次job清单。
2. 逐卡读显存/功耗、读host RAM和实际可用容量。只有满足当前门的卡可以接收新worker。
3. 启动后设置该卡 pending；绑定job_id、attempt_id、owner_generation、新PID和设备。收到**该新进程首次完整分配 ACK**才清除pending，不能复用旧generation的ACK。
4. 原batch及自然尾batch的容量profile覆盖真实选中快照、A历史、B克隆和评估路径。未覆盖的新路径先做必要工程profile；不依据旧轻量路径猜测峰值。
5. 清除pending后重新读取遥测，再决定是否增加worker。每张卡独立判断；可同时承担多个机械训练，但所有调度由当前会话前台runner监督。
6. 完成/暂停后重新核验owner与完整状态，再释放该worker资源。不能为了腾显存销毁仍需B锚点或最终Test的唯一RAM状态。

禁止 `codex exec/resume/fork`、tmux、nohup、systemd、headless/background AI、旧 `autonomous.py start/run/recover` 或退役driver。本计划的 `advance` 仅展开已登记有限分支，不创建新的研究循环。已有仍有效的RAM owner可作存储，不能接管成独立AI调度器。[R4]

## 6. RAM_ONLY 状态与流式生命周期

### 6.1 允许与禁止存储

**模型参数/buffers、Adam m/v/step、EMA、A条件历史、RNG、loader游标、可恢复训练状态，只能驻RAM或内存BytesIO。禁止写checkpoint、pt/pth、pickle、tensor缓存、磁盘memmap、恢复用压缩包。** 代码、配置、来源/数据hash、标量、非恢复性预测评价数表、失败账和报告可以落盘。文件后缀不能决定许可；能恢复模型/训练的内容均属于禁止落盘状态。

不能从磁盘selected权重、CSV或旧hash猜测Adam；不能通过把恢复状态改名为“诊断数据”规避RAM_ONLY。数表中仅记录参数名、shape、dtype、hash及标量摘要，禁止逐参数状态大数组。数据cache遵循冻结预处理身份，不混入模型状态。

### 6.2 完整状态契约

每个可继续训练的RAM提交至少包括：

- online模型参数与buffers、train/eval模式；
- 完整optimizer param_groups、每参数m/v/私有step、None/tensor_zero活动语义；
- EMA参数、ρ与累计真实更新数；
- A的history、active、selected、pair_clock、demand、moments及parameter_manifest；
- Python/NumPy/CPU/CUDA RNG、独立loader generator、当前permutation与游标；
- dataset/seed/config/source/group hashes、epoch/global_step、最后已提交事务号；
- online/EMA各自selected epoch、Validation记录、模型hash及所需selected模型；
- owner endpoint、owner_generation、job/attempt身份和可直接get的状态ACK。

恢复时先核验内容，再恢复全部张量/控制状态，最后恢复RNG。临时clone、probe、排序面板使用独立RNG或保存恢复，不能污染父轨迹。B状态捕获在预注册更新前；真实JOINT与live精确一致才接受该anchor。

### 6.3 驻留量与释放规则

一份模型只有约3.09MB/6.46MB，并不表示训练显存仅数MB。online、EMA、Adam m/v是主要四份FP32张量，另有selected模型、梯度、激活、A条件历史、数据cache、Python/框架开销、CPU复制、BytesIO和B临时分支。十条完成轨迹的20份selected模型约95508960B，仅是选中模型张量，不含optimizer、历史或训练峰值。[R6]

B一次只保持当前父完整状态及正在处理的有限分支，不能同时留40个anchor×52个clone。用于短段的真/伪首步必须保留完整状态，其余分支产出数值与hash后释放；短段结束亦释放。完成B锚点后才允许把轨迹compact为selected-only；最终冻结评价需要的selected模型在评估及验收前保持RAM。

C05的历史10对20模型直接get收据只证明11/17当时可用；其complete压缩只保留selected_models/selected_epochs/selected_validation，不保留旧Adam/RNG/loader。即使当前owner仍活，也不能用它提供本轮历史锚点。[R6]

## 7. 恢复状态机与成本信用

| 观察到的情况 | 允许的操作 | 完成信用与账目 |
|---|---|---|
| owner当前可直接get，完整state和最后提交匹配 | 从最后ACK继续未完成suffix | 保留已提交完成量，不重复已完成轨迹 |
| 首次真实更新前失败，无有效ACK | 同seed/同源重新初始化新attempt | 记录失败启动成本；科学身份不变 |
| 已提交训练但owner退出/完整RAM失去 | 明确标STATE_LOST，新attempt从同源初态重建 | 丢失且不可恢复部分不冒充完成；物理失败成本仍计入 |
| 只有selected模型 | 可在身份与当前存活核验后做获准selected评价 | 不能续训、不能构造历史Adam锚点 |
| 已完成端点和数值身份完整，但模型状态已失 | 复用数值；仅为实际必需的状态/最终评估作有目的重建 | 不因“没有模型文件”重做全部已完成方法 |
| CUDA OOM | 当前85/85/90门保留，重测完整进程物理fit后恢复未完成项 | 不减batch/精度/人口；失败成本单列 |
| 遥测暂时不可用、control JSON不完整 | TECHNICAL_PAUSED，先修复实际工程阻塞 | 不转成科学NO_GO，不删除其他任务状态 |
| 用户明确停止 | 完成当前原子操作，保全可用RAM和已提交数值，报告USER_STOPPED | 未完成任务记未完成，不能记负结果 |

必须逐次现场验收：owner的PID字符串或历史ACK不等于当前可用。连接恢复后，核对endpoint身份、generation、配置、完整字段、最新事务及模型/Adam/EMA hashes；不凭旧端口曾响应就继续。新job/attempt只能持有自身pending，不能让旧pending永久占卡。

控制JSON使用同目录临时文件、完整写入后 `os.replace` 原子替换。数值完成记录与RAM事务ACK绑定：先state提交、验收ACK，再发布对应完成记录。遇到连接不确定，先查询事务，不盲目重执行同一步。C05已发生过空JSON与陈旧pending导致暂停，故这些检查是已知工程风险控制，不增加无决策意义的测试。[R6]

每个失败attempt记录最后返回call、最后RAM提交、最后可见epoch、失败触发、实际资源、恢复来源和重复工作。分别报告“有效完成epoch”“实际已计算epoch”“重复/退役epoch”，不得像过去遗失275epoch的attempt那样混入新完成信用。[R7]

## 8. 数值日志、磁盘与交付

优先保存seed×anchor×scope×source×receiver聚合表、运行身份、候选状态hash、固定panel身份和必要预测；避免逐参数/逐entry诊断膨胀。历史NDP逐entry日志曾达约5.77GB、compact Stage2约30.3MB；C05原始成员约1.263GB、压缩约677MB、81个不超过8MiB分片。容量规划必须基于当前实际增长率，不照抄旧compact大小。[R2]、[R6]

准备阶段重测工作盘及允许的数值staging目录。历史工作盘剩3.18GB、`/tmp`约188.97GB仅是11/17记录，不能作为本次可用空间承诺。若沿用已有打包方式，保留 `3A + 256MiB + 1GiB` 可用空间，A为本轮实际待归档输入大小；其他归档方式按真实临时副本数核算。不能删除无关任务文件或把RAM模型转储磁盘来解容量问题。

最终 `COST_ACCOUNTING.json/CSV` 至少包括：logical_method_id、physical_job_id、attempt_id、cost_role、source_config_hash、seed、dataset、有效/物理epoch、live/virtual/suffix Adam、live EMA、诊断sweep/forward、选择器成本、Validation/Test成本、worker/wall秒、峰值GPU/hostRAM、输入/归档字节、状态丢失与重建成本、复用parent_id。所有汇总都可追溯到物理job，重复引用同一CLEAN/JOINT/首步只计一次。

## 9. 来源

- [R1 五seed协议与状态释放](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)。
- [R2 CLEAN/NDP真实工作量](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/auxiliary_objective_attribution_20261115/REPORT.md)；[CLEAN成本引用](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/natural_quadratic_ranking_20261116/REPORT.md)。
- [R3 PAIR五seed成本原表](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/costs.csv)。
- [R4 当前执行规则](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/AGENTS.md)；[start.md](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/start.md)。本任务覆盖旧无限循环/四seed/Test停点的适用关系见00。
- [R5 实际OOM后的容量修正](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/PHYSICAL_CAPACITY_CORRECTION.md)；[容量原始测量](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/PHYSICAL_CAPACITY_MEASUREMENTS.json)。
- [R6 C05报告及工程成本](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/REPORT.md)；[执行历史](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/EXECUTION_HISTORY.md)；[运行实现](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/experiments/cell_mass_bce_20261117/runner.py)。
- [R7 C04完整报告与失去状态记录](https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/completion_weight_attribution_20261117/REPORT.md)。完整来源、是否在main及复用限制见 [12](12_SOURCE_AND_REUSE_REGISTER.md)。

[R1]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md
[R2]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/auxiliary_objective_attribution_20261115/REPORT.md
[R3]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/2026-11-03_mdmt_five_seed_completion/costs.csv
[R4]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/AGENTS.md
[R5]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/PHYSICAL_CAPACITY_CORRECTION.md
[R6]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/cell_mass_bce_20261117/REPORT.md
[R7]: https://github.com/Caelestis1903/My-Research-Project/blob/875215c1e509ab55c1b6632d4f5d13e8b94782da/research_reports/completion_weight_attribution_20261117/REPORT.md
