# 08｜自动调度、RAM与恢复

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。唯一科学矩阵为 [05](05_EXPERIMENT_MATRIX.md)，所有进入/停止条件为 [09](09_EVALUATION_AND_DECISION.md)。

## 1. 复用现有基础设施

复用最新source的guardian、epoch commit、PID/start身份、资源测量及独立每卡准入。新包只适配本轮DAG、fork、不同长度job和候选消费者；不能直接调用旧因果controller并沿用20条/50epoch/四块的完成逻辑。

一个attempt只有一个scheduler owner lock、一个guardian身份；重启先检查已有进程和start time并收养活worker，不能为方便新建guardian丢失参数。无需人为固定两个worker上限。每次派发按当前数据块优先，同层五seed全部排队；可预取下一块，但最多一个未来块，并计入其将增长到的全部状态。

## 2. 调度状态和依赖

job状态：`PENDING/RUNNING/COMMITTED/COMPLETE/NOT_TRIGGERED/BLOCKED_ENGINEERING/BLOCKED_RESOURCE/FAILED`；阶段状态另有`WAITING_FOR_FIVE_SEEDS/FROZEN/FINALIZED`。门仅消费已COMMITTED且完整的record。未满五seed输出WAITING，不做部分均值淘汰。

优先队列：未完成最早块父轨迹 → A短支 → 条件RAW/续训 → 已注册KD → Gaussian边界 → 全局冻结/Test/报告。无SGD的CPU指标归约可与训练并行，但不能超出host预算。两个进程不能同时写相同job/epoch或覆盖同一scratch模型。

`execute --through report`自动生成并派发后继，不要求人工解释oracle空间或“感觉收益是否足够”。纯函数门产生DecisionRecord后，DAG仅实例化manifest中的模板。任务有负面结果也要写完整报告。

## 3. RAM-only生命周期

仓库start.md禁止任何模型/optimizer/checkpoint文件落盘。本轮继续执行：参数、m/v、F、历史重心、A恢复状态、selected对象只能留普通进程/guardian/内存IPC。禁止写pt/pth/npz/mmap/tmpfs保存模型。允许落盘的teacher软标签是按训练行组织的预测数据，需schema注明内容，不混入参数数组。

| 时间 | 保留对象 | 何时可释放 |
|---|---|---|
| 父训练中 | 活跃θ/m/v/F/RNG；稀疏F；各读出current-best；可作为教师的best真实父F；E20/E35必要fork包；E20独立锁住12个B及O/F prefix-best | 老best无引用即释放；稀疏F按最后消费者释放 |
| 五seed父完成 | B/C选择所需best；A两个起点及prefix-best；O/F端点 | 先计算B/C五seed选参与保留清单，再释放未选best |
| A短支 | 每alpha E20的E26完整状态及best；E35末端统计；冻结B附带曲线 | 门后释放未晋级alpha；E35只保留必要scalar，不留整条历史 |
| A续训 | 获选MATCH/RAW到E50；冻结AB读出 | 完整选点/比较后仅留正式端点 |
| KD | 教师两态、真实父F初值、soft targets、学生活跃optimizer及best | 所有KD/控制与全局Test完成后释放教师/选点 |
| dataset完成 | 正式候选+控制的selected包；Gaussian继承配置 | 无消费者参数释放；元数据/曲线持久化 |
| 全局结束 | 冻结Test端点 | Test、报告、索引校验后release并写收据 |

不存在“先跑完释放参数，再让汇总脚本执行C参数替代”的后处理路径；所有需要模型的B/C完整曲线在父训练期采集。也不能在选择教师后才发现没有该epoch真实F。

## 4. 资源准入

同时满足以下条件才启动下一个任务：

1. 选中GPU当前与预测峰值显存 <85%，功耗 <85%；另一张繁忙卡不否决此卡。
2. 当前host RSS+新worker预留+IPC/CPU预测排序预留 <90%总RAM，依据MemAvailable实时更新。
3. 原guardian内部commit predicate（当前size+旧row state+worker incoming等实际定义）≤4GiB。
4. 全进程实际参数tensor storage的保守峰值≤7GiB：包括worker、export clone、guardian旧/新commit、finalizer副本、尚未到来的selected/fork增长。共享storage去重，序列化bytes算host而不重复算tensor。
5. 预测/标量持久化预算≤2GiB，写前按uncompressed shape/dtype预估。

每次准入记录各项bytes及限额、GPU/host读数、拟派发job与拒绝原因。若4GiB predicate阻塞但7GiB通过，仍必须等待；不得修改内部预算绕过。先降低并发、消费已完成状态、缩小eval microbatch；训练batch/seed/epoch不变，不终止其他用户进程。

65P/seed只粗计current-best与A anchor核心，**不是内存上界**。未去重估算还必须加E20的12个B prefix-best至多24P和O/F前缀至多2P，即91P/seed（五seed KR约2.94GB），再加E20/E35稀疏F历史、活跃cache、buffer/RNG、旧/新commit、scratch、另一数据块已冻结端点与任何瞬时副本。所有这些对象都有实际consumer，不能遗漏或等第五seed快结束才发现超限。

prepare必须对全五seed父块的完整对象图模拟峰值及每次提交的旧/新并存，提前通过两重预算。按scope存共享S一次、共用N一次，C PARAM无需为KD额外pin父F；anchor只复制恢复必需状态与A/B前缀，不携带C current-best目录。不可变best/prefix/history包按引用管理，epoch事务只传递更新对象及引用，不把全部历史候选重复克隆进每次row commit；仍按真实storage和原4GiB predicate计数，不改变预算。

父五seed完成后立即做选择并释放未选状态，不能将两dataset全部候选网格一直留到Test。若完整未来对象图在最低并发下仍不能通过预算，prepare应在训练前报BLOCKED_RESOURCE，列明超额对象；不允许队列运行到无法释放的五seed barrier后永久等待，也不允许删seed、漏前缀或落盘模型绕行。

## 5. 2GiB落盘预算

| 内容 | 上限 |
|---|---:|
| 所有完整曲线、六格、门输入、HPO/端点/成本标量（gzip） | 0.45GiB |
| 预注册E20/E35/E50和最终selected的有限面板预测/pair账本 | 0.65GiB |
| KD教师train软标签（最多两教师、逐块复用） | 0.30GiB |
| 面板/数据身份、固定pair索引、源码配置摘要、日志报告 | 0.30GiB |
| 余量 | 0.30GiB |

完整V/Test预测只在当前评价RAM里归约，写scalar/hash后释放；不保存全epoch/H/scope logits。C拟合仅需当前P_A及下一epoch应用系数，P_B未来价值现场累积seed级统计；不依赖保留全部面板时序才能最终判门。固定pair一旦产生就复用，防止旧归因任务重复采样瓶颈。

写入前prepare展开每类array shape，超预算先去除可由已有标量复算的重复持久化，不按结果好坏删记录。日志采用每epoch/状态变化摘要，不打印逐参数张量。

## 6. 事务与恢复

每epoch事务顺序：预约瞬时资源 → 完成本epoch训练和注册观察 → 将新RAM对象提交guardian（旧commit仍活）→ 原子写预测/标量临时文件并rename → 写COMMITTED收据与hash → 释放旧commit/已消费引用。未COMMITTED的半epoch不进入科学汇总。

| 故障 | 自动处理 | 研究状态 |
|---|---|---|
| worker退出而guardian活 | 从最后完整epoch恢复全部状态，每job最多一次重启 | 未提交部分重放成本单列 |
| observer/指标失败 | 保留旧commit，修复相同数学实现后重做未提交epoch | 不混入半成品，不改门 |
| guardian丢失 | 同seed、同code/config/data/RNG从初始化重放到需要的最大epoch；fork先重放父前缀再实际分支 | 每逻辑job最多一次完整重放；n仍5 |
| selected丢失 | 从freeze列出的精确轨迹重建端点，逐epoch既有hash核对 | 不重新择点，不重读Test选恢复点 |
| OOM | 释放本任务临时图、降并发/eval microbatch，恢复旧commit | 不改训练batch，不算科学失败 |
| source/data/layout/hash不一致 | 停止受影响任务，记录具体差异 | SOURCE_MISMATCH，不能接着拼数据 |
| 同job重放限额耗尽 | 标明缺失状态和原因 | BLOCKED_ENGINEERING；不把4seed写成5seed否决 |

发生真正科学实现变更时另建attempt并锁新hash；原记录只读。纯scheduler顺序/计数修正独立记录hash，不重算科学配置或伪造原运行收据。必要修复完成后自动恢复；资源暂不可用则轮询待准入，不制造新的科研审批流程。

## 7. 时长和终止

任务没有按seed性能提前停机；只执行预注册科学STOP或明确工程阻塞。核心/全触发时长量级见05，由当前smoke测得`t_train_epoch/t_fullV/t_panel/t_IPC`和实际DAG更新ETA。分开记录worker累计小时、GPU分配/占用小时、整队列walltime，不能直接除GPU数作承诺。

用户中断时只暂停派发，保留guardian。进程或设备必须退出导致RAM丢失时写状态损失收据，下一次resume走有界重放；不偷偷落盘模型。`status`显示五seed完成度、剩余依赖、活对象、真实更新数、资源限制和具体下一动作。

最终交付必须包括报告、全部冻结比较、成本/恢复账本、artifact索引、源身份、ACCEPT/REVISE/STOP/NEXT；Test完成后release。未来代码/结果只提交新实验所需路径，不推参数文件、原始输入数据、认证文件或运行锁。
