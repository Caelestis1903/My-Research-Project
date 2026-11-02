# 独立实现审查 — EMA02V2

**判定：PASS WITH FIXES；阻塞问题均已修正。** 审查者为独立的 `/root/reviewer`，与两个实现代理不同。批准进入真实 CUDA Stage0；Stage0 通过后按协调者冻结的派发文件运行研究队列。

设计 SHA256：`9e97e74cc00311cc9b9035a4dedb370b94c1a5324442d700dc78dd0049124617`。本次实际审查的 40 文件源码哈希：`e524699526ae6e8eaf38950a0255eb359c7ca7705a64dab4f60bf9aa1ea93307`，与 `EXECUTION_SOURCE_MANIFEST.json` 一致。依据为完整 DESIGN、两份 CODER 实现报告、实际新包源码及固定来源代码。

## 语义检查

| 项目 | 独立检查结果 |
|---|---|
| 固定底座与来源 | 21 份 vendor 文件的 SHA256 与原 Git blob 逐项相符；源相对导入保持，执行不调用历史持久化数据缓存。固定融合、单 Adam、coupled decay、None/显式零语义均保留。 |
| 输入探测 | 专用前向返回真实 tower raw logits，使用 detached 参数值，仅在 probe 移除 gate 输入断图；真实监督训练保持原 gate detach。有限差分、参数状态与随机流隔离有针对性检查。 |
| 损失与反传 | A0/A1 复用源 callable；尾批按实际原 B×2 归一化。J/F 只替换目标域视图；TG/RG 两视图分别计各自任务；最终 δ 整体 detach。一次 backward、一次 Adam、随后一次固定 EMA。 |
| J0/J1/J_SUM | 同目标域、完整联合标签、排除自身、最多两不同供体；抽取后剔除零差方向；共同半径与 .2 插值上限；max-min/mean/random 分支、零梯度和并列随机处理符合设计。 |
| F0/F1/FR/F_PR | 教师使用更新前 EMA 和自身 embedding；两个 raw-logit 响应联合约束，FP64 小矩阵 SVD、规定截断、统一 cap、等模型退化与一次随机零空间回退。FR 匹配自身当前 F1 反事实行能量。 |
| 数据与评价 | 原始 loader 核对 raw、六数组、行号、encoder/scaler；W1 从原 CSV 前缀重新拟合，KR 全局切分后筛域。未定义格保留 null macro 和人口/LL；自然输入评价，online/EMA 独立最早最佳选点，E50 保留。 |
| RAM 生命周期 | 独立长寿命 socket guardian，BytesIO 传输，epoch 接收后 ACK；model/Adam/EMA/RNG/loader 与 selected 状态留 RAM。控制器重连检查现有 worker；恢复检查验证下一步状态一致。 |
| 阶段与测试访问 | S1 明确 24 条；后续 JSON 由协调者批准，机械门按 seed 计算。主/备路径固定；统一 freeze 保存所有 retained selected/E50 哈希后才开放 test/confirm。停止路径仍保留开发控制评价。 |
| 资源与成本 | 每卡独立判断当前显存/功耗 <85%，下一 worker 按实测峰值匹配物理剩余显存；不改 batch/精度/epoch。修正后的预算在派发前保留 active、next 和最终评价成本，已运行任务完成后明确记录超限状态。 |

半 κ 校准通过 `gates.development(..., kappa=...)` 与明确配置的派发文件评价；`recommend-next` 的默认 κ 汇总不替代协调者的校准比较。W1 成本预约使用原家族 Stage0 预测的迁移估计，实际行数、步数与耗时单列。

## 本次发现与修正

1. **预算准入缺口，已修正。** 原逻辑只在累计耗时达到 cap 后停止新增任务，因此剩余 1 秒也可启动一条预测长任务；pending 为空时还可能把 active 超限当作普通完成。现在准入包含冻结的 active 剩余、下一任务及已完成任务待评价成本；超限明确输出 `BUDGET_STOP_UNMATCHED`，不对训练算法作时间驱动早停。
2. **最终评价开销漏计，已修正。** 原计时仅覆盖 metrics forward，遗漏数据/hash、guardian IPC、模型构造和恢复。现在 endpoint 计时覆盖这些操作，数据 bundle 可复用，完整与 partial 评价、Stage0 成功/失败均计入物理成本，避免重复计账。

上述修正未改变数据、模型、增强公式、优化器或科学门槛；正式训练尚未启动。

## 独立执行的验证

使用项目解释器运行：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/ema_view_design_20261102_v2/tests
10 passed in 5.70s
```

这 10 组检查涵盖 core 公式/梯度、W1 边界/未定义格、seed 与 harm 门、逐卡放置、冻结预算准入，以及独立 guardian 的 ACK/恢复下一步一致性。另独立核验 21/21 vendor 文件和最终源码 manifest。未扩展到其他历史实验包测试。

真实数据 CUDA 输入导数、完整 batch backward、显存、耗时、联合标签计数与无 teacher 监督梯度，按设计留给紧随本审查的 Stage0。源码发生影响语义的更改时需更新审查哈希。
