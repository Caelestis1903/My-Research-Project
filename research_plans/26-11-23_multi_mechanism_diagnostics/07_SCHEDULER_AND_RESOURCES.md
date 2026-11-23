# 07 — 夜间调度、资源预算与持续恢复

## 1. 资源事实与重新测量

D2 曾记录两张 24 GiB GPU，fork 观测 MovieLens 约 1.41 GB、KuaiRand 约 1.85 GB，OOM=false。这些是历史测量，不代表本次当前可用资源。S0 必须重新读取 GPU UUID/总显存/已用显存/进程、主机 RAM、CPU、磁盘和其他进程。

M1 二阶梯度、M2 row-task virtual Adam、M6 Jacobian/trust-region、M8 六格梯度控制的成本可能显著不同。必须逐 family smoke，不能用普通 fork 的显存估算全部机制。

## 2. 全量任务规模

- baseline：最多 10 条 50-epoch trajectories；
- anchors：30；
- S1：2,130 个局部 fork slots；
- S2：全部 S1 endpoints 的 V_B readout；
- S3：所有可训练变体各自 2 datasets × 5 seeds × 5 epochs；
- S4：所有通过机制与短程门的变体各自 2 datasets × 5 seeds × 50 epochs；不设 Top-K。

S0 smoke 后，按实测每类动作的 GPU-seconds、CPU-seconds、peak VRAM、host RAM、wall seconds 计算总预算。报告最坏情况与预计情况；不得编造速度。

## 3. 不设研究优先级的调度

Scheduler 使用 family-round-robin：
1. 每轮从 M1–M8 和 X1–X5 各取下一个可运行 job；
2. 在 dataset、seed、anchor 间轮换；
3. 每个 family 保留公平资源配额；
4. 一个 family 暂时 OOM/技术阻塞时，跳过该 job 继续其他 family；
5. 不因某个 family 的结果更好而提高其队列优先级；
6. 不因某个 family 的 NO_GO 而取消其他 job。

并发数由实时 GPU/RAM 守护器确定。不得抢占、终止或修改其他用户进程。

## 4. 长程夜间运行

- 启动前生成完整 immutable manifest。
- 运行器持续消费全部预注册任务，不设“一晚到点即停止”的硬时限。
- 夜间窗口结束或主机中断时，所有 job 保持原状态；下次启动自动恢复 QUEUED/CHECKPOINTED。
- 不重新创建已 COMPLETE_VERIFIED 的 job。
- 不允许用重跑覆盖旧 receipt；每次 attempt 独立记录。
- 不需要人工逐个批准候选；只在全局协议/数据身份阻塞时停止对应 stage。

## 5. 失败和重试

状态：
QUEUED → ADMITTED → RUNNING → CHECKPOINTED → COMPLETE_VERIFIED
失败：
FAILED_RETRYABLE、TECHNICAL_BLOCKED、SCIENTIFIC_NO_GO、INCONCLUSIVE、NOT_APPLICABLE。

技术失败最多两次重试。OOM 后先释放本实验对象、重新读取资源、降低并发。不得改变 batch size、precision、optimizer、model、data split 或机制定义。

若某机制在当前硬件无法执行原始算子，记录 TECHNICAL_BLOCKED；继续其他机制。不得悄悄改成一阶 proxy 后保留二阶机制名称。

## 6. 持久化和恢复

每个 job/fork 原子保存 model/Adam/EMA/RNG/loader state、config/source/data hash、batch IDs、action receipt、metrics、costs。恢复必须核验 state identity 与 next batch IDs。所有 admission/exit/retry/oom 进入 append-only ledger。
