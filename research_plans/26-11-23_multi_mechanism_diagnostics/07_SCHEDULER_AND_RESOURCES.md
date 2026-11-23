# 07 — 调度、资源与恢复

## 1. 资源依据

D2 S0 RESOURCE_AUDIT 曾记录两张 24 GiB GPU；D2 S1 PHYSICAL_CAPACITY 记录 MovieLens fork 观测约 1.41 GB、KuaiRand 约 1.85 GB，且 OOM=false。该数据仅是 D2 当时的观测，不代表本次运行主机当前可用资源。M1 二阶图可能显著增加显存和时间。

S0 必须重新读取真实 GPU UUID、总显存、已用显存、功耗、进程、主机 RAM、CPU。不得在方案中硬编码当前资源空闲或固定 worker 数。

## 2. 资源准入

- 每个 GPU 独立准入；不得仅按设备总显存估算。
- 先执行一个 MovieLens、一个 KuaiRand 的 M1 second-order smoke。
- 记录峰值显存、GPU time、host RAM、二阶图大小、gradient count。
- 并发上限由实际测量和资源守护器决定；不得抢占或终止非本实验进程。
- M2 row-gradient score 先做 panel-size smoke，再确定每 worker 上限。

## 3. 估算方法

不虚构训练速度。完成 S0 smoke 后，按实测成本估算：
- baseline 10 × 50 epochs；
- M1 150 forks；
- M2 120 forks；
- M3 最多 150 forks；
- S2 每个晋级候选 2 × 5 个 seed confirmation jobs；
- S3 每候选 2 × 5 × 5 epochs；
- S4 条件矩阵 2 × 5 × 50 epochs。

每项成本分开记录 GPU-seconds、CPU-seconds、peak VRAM、host RAM、wall seconds、retry overhead。不能把并行 worker hours 直接相加当作墙钟。

## 4. Scheduler 状态

QUEUED → ADMITTED → RUNNING → CHECKPOINTED → COMPLETE_VERIFIED
失败分支：
- FAILED_RETRYABLE：可重试的瞬时 I/O/worker failure；
- TECHNICAL_BLOCKED：协议、数据、模型、状态、资源或实现不满足；
- SCIENTIFIC_NO_GO：实验正常执行但机制 gate 不通过；
- INCONCLUSIVE：支持不足或独立确认面板失效。

## 5. 重试与 OOM

- 技术失败最多两次重试；每次 attempt 独立记录。
- OOM 后释放本实验对象并重新测量资源；降低并发优先。
- 不得改变 batch size、模型结构、precision、optimizer 或训练配置来绕过 OOM。
- 如 M1 二阶目标在现硬件上无法执行，M1=TECHNICAL_BLOCKED；M2/M4/M5 可按原冻结计划继续。
- 不得将 OOM 后的一阶替代实现标记为 M1 原机制。

## 6. 状态恢复

恢复时检查 source/config/data/state hash、Adam m/v/step、EMA、RNG、sampler、next batch row IDs。checkpoint 必须在 epoch/fork 原子边界提交。部分写入视为无效，不得继续追加。
