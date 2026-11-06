# 07｜历史复用、资源预算与恢复

## 1. 预算结论

**本次完整机制研究默认新增20条带观测的训练重放，共1000 epochs、34500次真实Adam更新。** 同一训练轨迹共享所有全程EMA、两份临时pulse和全部参数组读出；不为每个rho、每个参数组、每个指标再训练一条轨迹。

这些重放用于获取既有标量归档中不存在的参数历史干预、预测变化和指标前瞻信息。已有普通EMA/Gaussian+EMA经验收益直接继承，重放不提供额外独立seed，也不重开A/B/C算法搜索。

完整Validation有 **4500次模型读出**：ML每epoch为online+3份物理EMA，KR为online+4份。2250只对应一种训练目标的10条轨迹，不是本方案clean/G-D0合计预算。

| 数据集 / 真实训练目标 | 新逻辑轨迹 | seeds | 每轨迹epoch / step | 合计epoch | 合计Adam step | 每epoch regular模型数 | 完整V模型读出 |
|---|---:|---|---:|---:|---:|---:|---:|
| MovieLens clean | 5 | 2022–2026 | 50 / 1000 | 250 | 5000 | 4 | 1000 |
| MovieLens Gaussian D0 | 5 | 2022–2026 | 50 / 1000 | 250 | 5000 | 4 | 1000 |
| KuaiRand clean | 5 | 2022–2026 | 50 / 2450 | 250 | 12250 | 5 | 1250 |
| KuaiRand Gaussian D0 | 5 | 2022–2026 | 50 / 2450 | 250 | 12250 | 5 | 1250 |
| **合计** | **20** | 每条件5 | — | **1000** | **34500** | — | **4500** |

无early stop、无单seed筛选。全部原始训练样本曝光合计1,403,149,500行；Gaussian额外view合计82,182,750行。后者仍按原replacement loss计算，不扩大原样本损失分母。

## 2. 已有产物如何直接复用

完整数值证据见 [01_EXISTING_EVIDENCE.md](01_EXISTING_EVIDENCE.md)。以下是执行层复用规则。

| 已有资源 | 本轮复用方式 | 需要新增什么 | 是否新增完整训练 |
|---|---|---|---|
| 11/03五seed、overnight A/B、11/04域覆盖端点 | 重算配对均值、样本SD、六格、选点与固定epoch差异；复用已发表模型结果 | 明确source/data/seed/state身份，不能混协议排名 | 否 |
| 最新A/B的online/F/U_FAST/U_SLOW完整50epoch V曲线 | 先做阶段、时间尺度与收益差异分析；A/B同一父轨迹去重 | 无需再训练来确认已有曲线现象 | 否 |
| Clean旧rho=.995与当前F的同轨迹端点 | 继承已由online model_hash桥接的长/短历史现象 | 标明ML半衰期和旧Gaussian设置不同 | 否 |
| 最新A/B/C实现与测试 | 复用模型、loss、参数布局、路径组、指标、数据缓存、RAM事务组件 | 新的纯观察包及必要测试 | 否，工程smoke另计 |
| 服务器自然人口与预处理cache | 只读复用；检查raw/cache/row-order/encoder/scaler hash | 解析实际路径、冻结manifest与panel | 否 |
| 新1/2/4 epoch全程EMA和F | 同一online训练上同时维护 | 新观测所需完整轨迹状态 | 包含在20条内 |
| 新1 epoch pulse、局部分组反事实、M0–M3 | 两份临时shadow加一份scratch即时读出 | 旧标量中不存在的参数/函数信息 | 包含在20条内 |
| 温度诊断、预测指标交叉验证 | 已保存有界预测的CPU分析 | 固定统计代码 | 否 |
| 条件第二背景 | E20/E40已保留的F/fast/slow RAM向量 | 最多两个组的有限额外读出 | 否 |

最新报告明确A/B/C guardian已在冻结评价、报告后释放，模型/optimizer/shadow/checkpoint没有写盘。因此当前不能从已归档CSV恢复参数或补建过去的shadow。这里没有“先找一下不存在的checkpoint再反复等待”的阶段；阶段0核对现状后，按20条观测重放实施。若执行环境意外保留一份**完整且身份匹配**的未来观测状态，只能按真实内容减免相应工作，不能假定旧guardian尚在。

来源：[最新综合报告的资源与释放记录](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)、[实现README](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/README.md)。

### Test复用

所有regular模型的selected/E10/E50先冻结。源数据与指标函数一致、seed和state_hash一致的历史Test端点直接引用原行，不重复前向。否则只对冻结列表评价一次。

未去重的最大Test模型读出数为 `10*4*3 + 10*5*3 = 270`；selected恰为E10/E50、ML F/H2 alias、已归档同状态均减少实际次数。最大自然Test前向样本量49,617,570行。pulse和逐cell条件模型不加入Test列表。

## 3. 各阶段训练与观测成本

| 阶段 | 新训练 | 共用状态 / 额外工作 | 停止条件 |
|---|---|---|---|
| S0归档与实现审计 | 0 | 源报告、CSV、JSON、hash；CPU整理 | inventory、缺项和工程合同明确 |
| 工程实现与smoke | 0次科学训练 | seed2022原batch和尾批，两个dataset、clean/G-D0、完整pulse时钟；只查资源与数值 | 必要测试通过；失败为工程问题 |
| S1–S3核心观测采集 | 20条、1000epoch | 同时维护regular EMA、pulse、panel、M0–M3；4500完整V | 20条与所有五seed块完整 |
| S1–S3统计与预测 | 0 | 读标量/有限预测；五seed、前向时间及V行交叉验证 | 给出按06定义的研究决策 |
| S4温度诊断，触发时 | 0 | 已有概率，CPU最多960次一维有界拟合 | 预注册次数完成，不扩大温度网格 |
| S4第二背景，触发时 | 0 | E20/E40保留向量，最多160个新增组合状态读出 | 两背景、最多两组完成即停止 |
| 冻结Test与报告 | 0 | 不超过270次完整Test模型读出，可直接复用很多旧端点 | 五seed表、报告、hash完整；释放本任务RAM |

工程smoke为seed2022的独立数值/资源验收，全部数据集与clean/G条件合计最多400次真实Adam调用（包含观察开/关配对及源语义检查），单独计费，不并入1000科学epoch或五seed结果。超出该界限必须先修正工程问题，不用重复smoke搜索性能。

### Pulse和梯度成本

每轨迹10个pulse窗口，每窗口一个epoch。两份pulse只在10/50的训练时间内活跃；它们不产生额外Adam step。每anchor27种状态读出，20条轨迹共200个anchor，所以：

- 状态读出预算为 `200*27=5400`；涉及P_A、P_B、V_mech三个panel，最多16200个state-panel评价。
- F已由当epoch的regular输出计算，实际新增状态读出为5200、state-panel评价最多15600。每个训练panel最多4608个角色合并前的原始行，两个训练panel加V_mech最多21504行，因此新增pulse前向样本量上界111,820,800行。
- 排名梯度为 `200 anchors*6 cells*2 derivative_semantics=2400` 次小panel反向归约；每种语义可以复用一次前向图，依次提取六格梯度并立即释放P维结果。没有额外optimizer更新。
- 每步innovation仅12组RMS即时归约；一共34500个step，只持久保存epoch聚合及anchor最后一步瞬时量。M0不要求保存theta_prev/delta_prev的P维历史。

条件第二背景已保存全fast与全slow预测。给定组b，只有“其余组fast、该组slow”和“其余组slow、该组fast”是新增状态；全fast/全slow直接复用。上限 `20 trajectories*2 anchors*2 groups*2 new states=160`，三个panel最多480个额外评价；计算后只保留标量与混合状态hash，不持久保存S4的原始预测。

温度诊断限定E20/E40、online/F、每cell单个正温度T∈[.25,4]，V_A拟合V_B评价后互换。若四个dataset×objective条件全部触发，最多 `20*2*2*6*2=960` fits；每次最多100迭代、容差1e-8。使用源p的 `sigmoid(logit(p)/T)`，p=0/1按极限，T=1精确返回源p；LL保持原log-floor。AUC沿用原p排序，记录数值饱和计数，不把浮点新ties解释成排名机制。

## 4. 显存与RAM预算

### 参数体积与物理影子

参数量来自固定源码及11/05 manifest，不是本次实测GPU峰值。

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 参数量P | 773470 | 1614254 |
| 一份FP32参数，十进制MB | 3.09388 | 6.457016 |
| 一份FP32参数，MiB | 2.95055 | 6.15789 |
| 常驻EMA份数 | 3 | 4 |
| 活跃pulse份数 | 2 | 2 |
| GPU参数/Adam/梯度/scratch/临时观测保守预算16P，MB | 49.50208 | 103.312256 |
| GPU组编号int64[P]，MB | 6.18776 | 12.914032 |

16P是参数类工作缓冲的保守量级，**不含原batch40960的激活、算子workspace、CUDA context或allocator保留空间**。实际总显存必须通过smoke记录，不能把上表当GPU峰值承诺。功能梯度在小panel上计算，27个读出顺序复用一份scratch，不常驻27份模型。

### Guardian模型状态

每轨迹保留：最近可恢复online/Adam/regular影子/活跃pulse；每个physical regular readout和online的selected/E10/E50；仅E20/E40两组 `[F,fast,slow]`。不保留其他anchor参数，不保留逐步完整参数或梯度。

对上述payload及必要临时引用留余量后，采用保守上界：

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 每轨迹guardian参数类payload上界 | 30P个FP32标量 | 34P个FP32标量 |
| 每轨迹十进制MB | 92.8164 | 219.538544 |
| 两目标×五seed合计MB | 928.164 | 2195.38544 |

20条全部保留共约 **3.124GB（2.91GiB）** 参数类payload，低于 **4GiB guardian硬上限**。4GiB只约束guardian中本任务的模型状态；worker、IPC序列化瞬时复制、Python对象、数据mmap和预测缓冲另计RSS。预算余量不是允许新增参数历史缓存的额度。

源 `send_memory` 把 `torch.save` 写入 `io.BytesIO` 后经IPC发送，属于RAM序列化；不得改成 `.pt`、`.pkl` 或其他磁盘checkpoint。

`tensor_bytes()` 对guardian状态递归计数；每次commit前检查硬上限。最多2个worker并发，按smoke峰值和当前空闲GPU/RAM准入；不足时降低本任务并发或等待资源，不改科学配置，不结束其他用户进程。

## 5. 磁盘预测与日志硬上限2GiB

本研究明确允许保存有界预测观测。采用prediction-only NPZ，`allow_pickle=False`，白名单字段与shape见05；这不允许任何参数/梯度/optimizer状态落盘。

预测采用float32，row IDs/labels/pairs只每panel保存一次；F在pulse anchor复用已经保存的regular输出。以下为最大配额且不依赖压缩率的原始浮点字节估算：

| 观测 | 精确上界计算 | 字节 |
|---|---|---:|
| Pulse：26个新增状态；P_A/P_B排名与natural z+p，V_mech只p | 26×200×196608 | 1,022,361,600 |
| 每epoch V_mech online/F z+p | 2×1000×196608 | 393,216,000 |
| 每epoch V_mech其余H只p | (1000+1500)×98304 | 245,760,000 |
| 每epoch P_A/P_B的online/F z+p | 2×1000×98304 | 196,608,000 |
| **合计** | — | **1,857,945,600 ≈1.73035GiB** |

再预留256MiB用于panel身份、NPZ headers、完整标量、状态索引、图、日志和有限S4预测，合计约1.98035GiB，仍小于2GiB。NPZ压缩后通常更小，但执行准入不依赖“必然压到某个比例”的假设。

实现按trajectory/epoch或anchor分块，避免逐样本重复长SHA和重复row IDs。metadata存一次并用hash引用；不保存完整自然Validation/Test预测，不把日志扩展为每步P维张量dump。文件写入前预估大小、写入后统计实际目录字节；超限前停止下一次写入并记录 `RESOURCE_BLOCKED`，不静默丢弃科学记录。S4可只保留已定义控制标量及其receipt，原组合模型在RAM即时评价后释放。

## 6. 耗时估算的来源与适用范围

已有实测来自[最新综合报告成本表](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)：

| 历史阶段 | training秒 | probe秒 | validation秒 | 核心worker累计秒 |
|---|---:|---:|---:|---:|
| B：10条、500epochs、16个online/影子读出 | 8206.223 | 1117.640 | 11239.245 | 20563.108 |
| A：相同父路径与50epoch | 10457.704 | 1791.151 | 13722.035 | 25970.890 |

报告明确这些是**共享并发环境中各worker记录的计时之和，包含GPU等待**；不是独占GPU kernel秒、整机墙钟或可直接除以GPU数的时间。B单worker allocated峰值ML/KR约1.329/1.544GB、reserved约1.669/1.873GB；这是历史参考，不替代本研究完整功能梯度smoke。[B阶段成本与边界](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/B_STAGE_ANALYSIS.md)

用于排程的透明粗估：

1. 将B的10条Gaussian父训练量粗略放大为本轮20条，训练基底约 `2*8206.223=16412.446` worker秒；clean无额外Gaussian、影子更少，实际不能用此值当固定测量。
2. 历史B完整V样本前向量为1,403,152,000行；本轮为826,959,500行，比例0.58936。按相同吞吐缩放，完整V约6623.944 worker秒。
3. 两项合计约23036秒，即 **6.4个累计worker小时的训练+完整V基底估计**。本轮还有约1.12亿行pulse前向、2400小panel反向、IPC与标量分析。

在相近共享负载下，可先以 **8–12个累计worker小时** 作为排程量级，给新增观测和波动留预算；它不是完成时间承诺。正式队列启动前，用本轮smoke分别测得每dataset×objective的训练step、完整V每行、pulse每panel、功能梯度、IPC耗时，按冻结计数重算 `RESOURCE_ESTIMATE.json`。实际报告保留原始累计worker时间、整机开始/结束墙钟、并发数、GPU设备和等待时间，不把它们混成一个“GPU小时”。

## 7. 路径、数据与环境复用

解释器固定源项目的 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。原CSV默认已核验路径为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 与 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。不同挂载可通过CLI显式传入，但必须匹配源raw SHA及split/encoder/scaler身份。

cache root不猜测。优先 `--cache-root`；未给时先从已归档DATA_MANIFEST/RUN_MANIFEST的真实记录定位，检查路径存在且各数组hash一致。缺失时可从SHA匹配的raw CSV调用原 `prepare_cache(root,p)`，在新目录 `artifacts/ema_mechanism_20261106/data_cache` 重建完全相同的缓存，随后核验原数组/row/split/encoder/scaler身份。先检查磁盘，数据cache体积单列，不计2GiB观察cap。已有cache身份冲突不能覆盖。源代码中仍存在历史 `/home/xzf/Code/...` 常量，不得据此假定当前文件必然在原位置，也不得自行换一套预处理重新生成“近似数据”。

训练/观测bundle使用 `include_test=False`。preflight可以核对Test文件的字节hash和已归档元数据，但不将Test标签或指标提供给模型选择/指标拟合；新Test只有freeze gate通过后的独立评价进程可加载。

## 8. 资源失败、恢复与生命周期

### 正常完成

模型状态始终RAM；每epoch先提交guardian可恢复状态，再原子发布标量/预测receipt。完整五seed分析、有限控制、冻结Test和报告完成后，自动释放本任务guardian。E20/E40控制向量在S4决策完成且确定不再使用后可提前释放；selected/E10/E50保留到Test复用/评价完成。

### worker失败而guardian仍在

从最后已提交epoch恢复online、Adam、影子、pulse、loader/RNG、选点、保留anchor和观察事务状态。对本任务失败worker最多一次幂等重启；活跃owner不能重复派发。未完成epoch不计入科学训练次数，已发生的物理成本单独记账。

### guardian参数RAM丢失

同一逻辑job从theta0确定性重放。逐epoch核验既有order/init/model hash；一致则标记REPLAY_VERIFIED，仅补充物理成本，不新增seed。参数状态未落盘，不能从CSV“恢复到E20”。每条逻辑trajectory最多一次额外物理重放；再次丢失记录工程阻塞，不无限启动。

若所有20条都发生一次额外全程重放，物理训练上限为40条、2000epochs、69000次Adam、9000次完整V模型读出；这是故障上限，不是默认实验矩阵。支持在RAM内恢复的worker失败通常远小于这个成本。科学样本始终是20条逻辑轨迹、每条件五seed。

### OOM、磁盘上限或来源冲突

按顺序处理：释放本任务临时scratch/图；降低本任务并发；从已提交RAM恢复；若仍不满足，报资源阻塞。禁止减batch、减seed、删掉不利候选、缩短epoch或通过写模型checkpoint绕过RAM约束。hash冲突停止对应job并保留证据，不能覆盖旧记录。工程阻塞不作为科研方向否决。

## 9. 成本验收与结束条件

最终 `COSTS.csv` 每行包含dataset/objective/seed/trajectory_id、逻辑和物理attempt、原始样本/view数、Adam步数、regular/pulse评价次数、梯度probe次数、训练/probe/V/Test/IPC/analysis分项时间、GPU allocated/reserved峰值、worker/guardian RSS、预测磁盘字节及资源失败原因。

完成条件是：全部预注册五seed块有完整指标，20条与alias/重放计数正确，阶段结论按06形成，Test身份被冻结，报告和hash可复现，资源清理完成。没有出现新的决策性分歧时停止；不为了追求更漂亮的机制故事再启动A/B/C、新rho网格或新优化算法。
