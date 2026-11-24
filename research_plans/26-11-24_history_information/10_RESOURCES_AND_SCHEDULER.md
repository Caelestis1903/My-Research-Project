# 10 — 计算资源、RAM状态与任务调度

## 1. 本次与未来执行的边界

本次只提交研究规划及历史数表重分析，**不启动模型训练、GPU探针训练或新Test评价**。本章是后续明确启动实验时的调度合约。未来任务沿用当前有效仓库规则，不调用退役自主driver，不启动独立后台AI。

执行环境先核对：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，Python3.13.5、PyTorch2.9.0+cu126及锁定依赖；逐卡记录GPU UUID、可用CUDA总显存、power limit、驱动、设备映射和主机RAM。当前规划不假设旧GPU数量、空闲量或旧socket仍可用。

规则来源：[AGENTS.md](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/AGENTS.md)、[start.md](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/start.md)。旧发表模型protocol中的“一卡一个worker”、预测峰值×1.2门和旧轮询/重试设置均不覆盖这里的现行规则。

## 2. 有限矩阵的资源账本

| 模块 | 两数据完整逻辑轨迹 | 每seed训练预算 | 全局训练batch/主更新上界 |
| --- | ---: | --- | ---: |
| B0/B1/H0按需父轨迹重建 | 30 | M3oE50epoch | 51,750 |
| C1/C2四个非零配置 | 40 | M3oE50epoch | 69,000 |
| 默认物理上界 | **70** | **合计3,500epoch** | **120,750** |
| M/S/P六臂 | 60 | M3oE50epoch | 103,500 |
| C12 | 10 | M3oE50epoch | 17,250 |
| N作用域两臂 | 20 | M3oE50epoch | 34,500 |
| HyperGate四臂 | 40 | 原生batch1024、50epoch | 2,742,000 |
| Gaussian四臂 | 40 | M3oE50epoch | 69,000 |
| 所有条件触发的保守上界 | **240** | **12,000epoch** | **3,087,000** |

M3oE每个两数据五seed臂的更新数为5×50×(20+49)=17,250。HyperGate对应为5×50×(782+1960)=685,500，四臂为2,742,000。不能为了降低预算把HyperGate改成batch40960；模型原生配置是比较身份的一部分。

发表基线若需身份修复，另列至多HyperGate10条及AutoIFS10条，并与已完成HG_B0去重。HyperGate为500epoch/685,500更新；AutoIFS是search10+reuse50、batch4096，合计600epoch/205,800个全局batch、240,100次Adam调用。全部上界合并为**260条、13,100epoch、3,978,300个训练batch**；将AutoIFS search双optimizer分别计数时，合计**4,012,600次主优化调用**。这是一份保守容量账，不是要求预先全部运行的队列。

历史轨迹复用减少物理成本；技术重放增加物理成本。两者都不改变正式seed集合，online和EMA也不增加轨迹。四个核心候选无论诊断结果如何都完整full50；条件模块只由08的Validation与机制门触发。

### 2.1 额外计算的单列口径

- D1：40个H0锚点、40次独立Train探针模型forward，最多6格×40=240次来源autograd sweep，随父轨迹一次采集。
- D2：40个原H0参考步和400个替代虚拟步；11分支×40锚点×3面板×2读出=2,640次面板模型推理。这里的“次”是一整个面板评价，内部按eval batch分块，实际forward调用另计。
- D4：E10/E25/E50利用完整Validation已经产生的概率；固定pair分析在CPU完成，不为每个pair另做模型forward。selected与共同epoch预测按需复用。
- PAIR：C/H0等活动步在标准完整backward外有一次自定义scope的raw-r autograd；M使用自己的同次梯度维护shadows，不再为其反事实候选运行额外模型。
- EMA：同轨迹额外参数更新和评价前向、selected副本、诊断state拷贝都计成本。它不新增训练seed。

不要把autograd sweep、主更新、optimizer调用、实际forward、模型面板推理混成一个计量。09的costs表同时保存这些列。

### 2.2 墙钟估计

先用每个真实已批准配置的第一个正常epoch记录训练/评价/诊断成本，以`剩余epoch×该配置已完成epoch中位秒数`滚动估计worker时间；不添加专门速度训练、额外seed或缩短batch的试运行。总墙钟还受并发、其他用户进程、RAM与条件触发影响，应给区间并注明资源采样时间。

可用的历史参照是同自然人口HyperGate五seed：ML累计12,460.301 worker秒，KR53,326.508 worker秒，合计约18.27 worker小时/10条clean轨迹；四臂clean成本按比例约73.10 worker小时，新增PAIR/历史和双读出还有额外成本。这是已运行配置的容量参照，不是本次交付时间承诺。[原成本报告](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)

## 3. 逐卡准入：使用当前实测值

对每张可用GPU i，只有以下三个当前条件同时满足，才派发一个新的worker：

\[
\frac{M_i^{current}}{M_i^{total}}<0.85,\qquad
\frac{P_i^{current}}{P_i^{limit}}<0.85,\qquad
\frac{RAM^{used}}{RAM^{total}}<0.90.
\]

每张卡独立判断，不要求其他卡同时合格。读取真实显存与功耗；工具不可读时登记`RESOURCE_TELEMETRY_UNAVAILABLE`，不能用0冒充空闲。准入不加入预计worker功耗/显存，不设置固定active-worker上限。

启动/大分配期间可设该卡的`pending_startup`标志，防止几个派发同时读到相同旧值；worker完成初始化分配后立即重新采样，再决定是否继续增加并发。此标志随真实启动完成清除，不变成固定一卡一任务策略。等待资源时记录`WAIT_RESOURCE`，不算训练失败。

检查两工作区现行停止控制与用户指令，然后原子认领`logical_run_id`。每次启动和每epoch安全点再次检查。停止控制的实际路径由S0从当前有效执行配置登记，不能沿用失效历史路径或默认为“无停止”。

```text
while there are dependency-ready jobs:
  check current user stop and registered workspace stop controls
  harvest finished atomic epochs and verify RAM receipts
  for each GPU in rotating order:
    sample current memory, power, host RAM
    if current85/current85/host90 and not pending_startup[GPU]:
      claim one ready run; set pending_startup; launch supervised worker
      resample after actual allocation; clear pending_startup
  emit progress/blocked reasons; yield control while waiting
```

不使用`tmux/nohup/systemd/codex exec/resume/fork`来延续自主研究。机械训练进程属于当前会话监督的runner；科学判断仍由主代理负责。空闲等待不能阻断面向用户的进度更新。

## 4. OOM与物理容量

发生真实CUDA OOM后先记录：设备UUID、当前总进程驻留、失败分配、allocator allocated/reserved、CUDA-usable total、batch及最大域实际支持。清除该失败attempt的可释放显存，保留最后已提交RAM状态。

后续继续保留上述current85%准入，另做**物理容量**检查：用实际完整进程驻留测量判断新进程能否放入CUDA可用总容量。这不是“预计峰值必须低于85%”的新门，也不是固定worker上限。当前自然KR最大域为d1（1,743,898个Train行），内存推理不得写成d0最大。

按当前资源重排或减少同时驻留的虚拟分支、释放已用完的诊断副本，优先保持原训练batch/FP32。D2逐分支评价，不把11个克隆同时驻GPU。不可静默使用AMP、梯度累积或减batch来“等价”替换原Adam步。如果原协议确实不能在现有硬件执行，暂停依赖的臂，报告具体容量差；任何协议修改须成为明确修订并统一相应对照后再启动。

## 5. RAM-only训练状态

模型、optimizer、EMA、梯度/方向shadow、恢复RNG和训练游标状态均存RAM。可以复用实际存活的既有RAM guardian；若没有可用guardian，使用**当前受监督前台runner进程内的RAM对象库**。不为此启动替代后台guardian或旧driver。

磁盘可保存：源码、JSON/CSV日志、配置、哈希、预测与标量分析、报告；不可把训练state通过`.pt/.pth/.pkl/.npz`或伪装日志落盘。状态对象传输本身留在进程/现有RAM服务之间。

每次put后真实get，校验schema、model/optimizer/EMA/RNG/loader及影子态hash。只有`available=true`且get_hash=put_hash的收据可用于恢复或最终Test。旧socket地址、旧selected hash和历史“保存成功”日志只能证明过去保存过。

### 5.1 生命周期

1. 训练中保存current packet及online/EMA各自selected模型。epoch原子提交后更新last-good receipt。
2. H0的D2锚点与需从初始化累积的shadow一起保存，在同状态干预完成后释放锚点副本。最终v不能反推出历史U。
3. 未被选中的HPO配置仍保留完整数字与曲线；全局freeze前按预注册比较集合决定是否还需要selected状态。将进入最终Test的候选、父参照、控制和模型基线保留selected模型即可，不必永久保留全部optimizer。
4. 扩展选择后释放已经完成诊断且不再恢复的full packet，释放动作记录其不再承担的用途。选中模型必须留到授权评价和报告核对完成。
5. 用户停止或RAM不足时安全结束原子步骤，列出仍存活状态、可复用数字和会因进程退出而丢失的对象。不能用写盘绕开RAM-only。

### 5.2 内存预算

令N为参数数，J为自定义scope元素数。原Adam活跃状态为约8N bytes；固定EMA参数约4N bytes，online参数/梯度各约4N bytes。H0/C1相对Adam新增约4J bytes；C2/C12/P_C1及以原.99候选为目标的M相对Adam新增约8J bytes；以S为目标的M需要分开的Q99/Q9，相对Adam新增约12J bytes，详细分项见05。原A21 wrapper还保留12J个被动六格shadow，即48J bytes，不能只报告最小理论实现。

RAM中每个selected模型约4N bytes加buffers；在线/EMA同时保留两份，另有current完整state。需记录CPU副本、序列化临时副本和实际进程RSS，不以参数字节替代峰值内存。不同family只读分解所需的shadow也单列。达到当前host90%门时暂停新派发并按上述生命周期释放完成用途的对象。

## 6. 队列顺序、并发与依赖

| 优先级 | 可派发对象 | 必须已经具备 |
| --- | --- | --- |
| P0 | B0/B1/H0需要的唯一重建 | 数据/源码身份、原生adapter验收、可用RAM存储 |
| P0 | C1/C2四配置full50 | 各家族实现审查通过；不等局部诊断转正 |
| P1 | 已触发家族S | 全部S2配置完成，唯一selection_readout/逐数据参数冻结 |
| P1 | 已触发家族M/P | S完成并按06冻结C/S代表；M目标完整βP/参数已确定 |
| P1 | 符合条件的C12 | 原C1/C2控制完成，同一读出且无S替代 |
| P2 | N/HyperGate/Gaussian | Cstar及该模块的明确触发收据、源配置可用 |
| P2 | 必要发表基线身份恢复 | 审核确认不可复用，避免与HG_B0重复 |
| 结案 | 全局冻结、已授权Test、分析和交付 | 所有已触发工作完整/明确技术阻塞，selected实际RAM读回 |

在满足依赖的对象中按dataset/seed/config轮转，避免根据早期分数优先“看起来好”的候选。使用完整epoch作为恢复提交单位，训练过程可持续进行直到50epoch；调度的epoch quantum不是早停。条件模块可以并行，不能以新Test决定派发顺序或是否启动。

实现代码按家族依赖闭包冻结，已接受模块只读；开发未启动家族不会使已运行轨迹的source hash静默改变。任何影响公式/数据/随机流的修正生成新版本和审查收据，受影响旧attempt按身份说明，不选择性保留更高分结果。

## 7. 故障、停止和交付

故障区分：`WAIT_RESOURCE`是等待；`TECHNICAL_PAUSED`是有具体恢复依赖；`SCIENTIFIC_CLOSED`是完整证据已足以结束对应候选。记录基础设施、数据身份、非有限值、OOM和RAM丢失的不同原因，不将它们写成算法无收益。

丢失模型状态但数字完整时，保留合法历史指标；只有后续需要真实状态的动作才从同一初始化恢复该逻辑轨迹，并单列物理重放成本。若改了会影响轨迹的实现，则它是新版本，不能仍声称精确复用。恢复重复不增加seed数。

所有已触发臂完成或有具体技术阻塞说明、全部统计完成、独立结果审查通过后交付报告。未触发模块写`NOT_TRIGGERED`及规则；未获本轮Test授权写`TEST_PENDING_AUTHORIZATION`并提交可完成的Validation结论。正常研究终点由08的有限选择规则决定，不因未达到+.003而自动开启新的候选家族。
