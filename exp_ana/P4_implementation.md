# P4 实现报告：M3oE 三个新组件（全部开关化，默认 = 主基线）

> 一句话结论：三个组件已实现并通过全部等价性验收测试（5/5 PASS）；组件关闭时与主基线**逐位一致**（bit-exact）；组件 1 的 D 次反传开销实测仅 +6% 耗时 / +216 MiB 显存（远低于 3× 上限）。

## 0. 设计要点与前提声明

P1 的 D0.4 结论（小域 SNR 并不更低）已经削弱了组件 1 的原始动机，但按计划照常实现，让 P5 的数据来裁决其有效性。组件 1 的实现做了两处与"逐域朴素实现"不同的关键选择：

1. **梯度获取 = D 次反传而非 D+1 次**：完整训练 loss 本就是 6 项 BCE 之和，逐域 L_d 反传即可同时得到 (a) 共享参数的逐域梯度 g_d，(b) 私有参数（tower/gate/slot 权重）的完整梯度——每个私有参数只出现在一个 L_d 的图中；(c) 4 个架构标量出现在所有 L_d 中，其跨域累加值恰好等于完整 loss 的梯度。因此无需再做一次全量反传。实测开销见第 4 节。
2. **逐域梯度采集时不污染私有梯度**：每域反传后只把**共享参数**的 .grad 克隆进 g_d 并清零共享参数；私有/架构参数的累加不被打断。

## 1. 实现位置（`experiments/p4_run.py`，单一文件，未重构仓库代码）

| 组件 | 位置 | 说明 |
|---|---|---|
| 组件 2 模型变体 | `MDMTRecBeta`（:45-185） | 继承 `MDMTRec`；把 `_weight_bal_d`/`_weight_bal_t`（1 个标量 Weights）替换为 per-domain (D 维) / per-task (T 维) Parameter（`e_β` 初值 1.0，与原版一致）；`forward` 复制原版并改为 `bal_d[i]` 逐域取值。掩码 `.cpu()` 等细节与 `model.py:140-145` 保持一致 |
| 组件 2 课程 | `MDMTRecBeta.set_epoch`（:81-84）+ forward 内偏移 | 每 epoch 开始设 ρ(τ)=min(1, epoch/T_c)，T_c=总 epoch/2；forward 中 `sigmoid(e_β − κ_d·(1−ρ))`；κ_d = κ_0·log(p_max/p_d)/log(p_max/p_min)，p 全相等时 κ_d≡0；`curriculum_beta_t=0`（默认关，β_t 不加课程） |
| 组件 1 动量 | `PerDomainMomentum`（:187-241） | γ_d=clip(1−(1−γ_0)·p_d/p_max, [γ_0, γ_max])，p_d 取**训练集**逐域占比；`backward_accumulate`（:208）做 D 次反传并维护逐域 EMA `m_d ← γ_d·m_d + (1−γ_d)·g_d`；`apply_grads`（:233）把共享参数 .grad 替换为 **Σ_d m_d**（与 6 项等权 BCE 的共享梯度 Σ_d g_d 严格对应，drop-in）；私有参数走原累加梯度 |
| 组件 1 与 Adam 的关系 | `fit` 内重建优化器（:372-380） | `per_domain_only`：共享参数所在 param group β1=0，只用我们的 EMA；`per_domain_stacked`：保留 Adam β1=0.9，我们的 EMA 叠在其上；`none`（默认）：Adam 原样。β2 始终 0.999。注：Adam 的 β2 二阶矩会归一化梯度尺度差异，但不会消除方向噪声——本组件针对的是后者 |
| 组件 3 | `_arch_step`（:410-425）+ `_arch_update_at_step`（:472-497） | `arch_grad_ema`：架构参数梯度 EMA（γ_a=0.9，默认关）；`arch_loss_masked`：arch 更新 loss 改为与训练一致的 6 项掩码求和（默认关=原版未掩码整体 BCE）；`arch_update_every`：每 N step 更新一次架构参数（默认 0=每 epoch 一次，对齐基线）。**耦合提示**：原实现 50 epoch 只有 ~50 次架构更新，EMA 在这么少的样本上意义有限；开启 `arch_grad_ema` 时建议同时把 `arch_update_every` 设为 5-10（P5/P6 采用的组合），EMA 才有足够的样本支撑 |
| 数据下采样（P5 用） | `downsample_train`（:288-322） | 只动训练集；保持最大域不变、其余两域按公共因子收缩至目标 max:min；1:1 档（低于现状 1.9:1）数学上必须收缩大域，此时改为较大域收缩到最小域规模；`RandomState(seed)` 确定性可复现 |
| 单域 oracle（P5 用） | 复用原版 `--domain` 机制（未用） | P5 将直接调用原版 run_movielens.py 的单域模式，保持 as-published |

新增配置项及默认值（默认全部 = 主基线 as-published）：

| 开关 | 默认 | 说明 |
|---|---|---|
| `--momentum_mode` | none | none / per_domain_only / per_domain_stacked |
| `--gamma0` / `--gamma_max` | 0.9 / 0.99 | 组件 1 的动量上下界 |
| `--curriculum_mode` | none | none / domain_conditioned |
| `--kappa0` | 0.0 | 可为负（反向课程） |
| `--curriculum_beta_t` | 0 | 是否对 β_t 加课程 |
| `--arch_grad_ema` | 0 | γ_a 由 `--gamma_a`（0.9）控制 |
| `--arch_loss_masked` | 0 | arch 更新用掩码 6 项 loss |
| `--arch_update_every` | 0 | 0 = 每 epoch（对齐基线） |
| `--fix_optim_grouping` | 0 | P2 的 B10 修复开关，本文件复用 |
| `--downsample_ratio` / `--downsample_seed` | 0 / 777 | P5 数据档位 |

## 2. 等价性测试（验收标准，全部通过）

`experiments/tests/test_equivalence.py`（直接 `python` 运行，退出码 0）：

```
[PASS] (1) all-off val AUC stream == P0 for 3 epochs (bit-exact)
[PASS] (2) all-off per-step loss stream == P0 baseline for 60 steps (bit-exact)
[PASS] (3) curriculum degenerate: rho=1 forward is bit-identical to MDMTRec
[PASS] (4) rho=0 changes the forward (curriculum active)
[PASS] (5) analytic check: max offset at rho=1 is 0.0
```

- (1)(2) 用固定种子 2022 训练 3 个 epoch：关闭全部组件时 p4_run 走**原版** `CTRTrainer.fit/train_one_epoch`（只挂了 P1 验证过的无损观察器），逐 step loss 与 P0 基线 60 步逐位一致，逐 epoch 验证 AUC 与 P0 日志逐位一致。
- (3)(4)(5) 验证组件 2 的关键性质：ρ(τ)=1 时前向与原版**逐位一致**——任何增益只能来自训练轨迹，不可能来自模型容量增加。

## 3. 实现过程中发现并修复的坑（对后续阶段有用）

1. 模型 forward 返回的 `result` 在 CPU 上（model.py:198-206 未指定 device），原版 trainer 在调用侧做了 `y_pred.to(device)`——组件路径必须保留这一步，否则 CUDA 掩码索引 CPU 张量报错。
2. `arch_named()` 辅助：架构参数在原版是 `_weight_x.deep_weights`（Weights 模块），在 MDMTRecBeta 中是 `_weight_x`（Parameter），统一解析。
3. `--arch_update_every` 的 mid-epoch 更新会额外创建 train loader 迭代器，其 `_base_seed` 抽样会推后续 epoch 的 shuffle 序列（P1 已发现的同类坑）；该路径属于非默认实验配置，P5/P6 若使用将记录其 RNG 偏移。

## 4. 组件开销实测（seed 2022，2 epochs，GPU 与 P2 链共享争用，横向可比）

| 配置 | fit 耗时 | 峰值显存 | 相对 all-off |
|---|---|---|---|
| all-off（基线） | 95.2 s | 942 MiB | 1.00× |
| 组件 1（per_domain_only, fixgroup=1） | 101.1 s | 1158 MiB | **1.06×** / +216 MiB |
| 组件 2（domain_conditioned, κ0=0.5, fixgroup=1） | 94.9 s | 941 MiB | 1.00× / ±0 |
| 组件 3（arch_ema + update_every=5, fixgroup=1） | 110.1 s | 942 MiB | 1.16× / ±0 |

- 组件 1 的 D=3 反传使耗时仅 +6%——反传约占总步时的一半，3 次反传理论 ~1.5-2×，实测更低（小模型下 kernel 启动开销占主导），**远低于 3× 上限，无实现浪费**。显存增量是 3 份共享参数 EMA 状态（~58 万维 × 3 × 4B ≈ 7 MiB 理论值；实测 216 MiB 含采集图与临时量）。
- 组件 2 零开销（等价的前向 + 每 epoch 一次标量更新）。
- 组件 3 的 +16% 来自每 epoch 额外 4 次 arch 更新（update_every=5 × 20 step）。

## 5. 交付物索引

- `experiments/p4_run.py`：全部组件（单文件，~630 行）
- `experiments/tests/test_equivalence.py`：验收测试（5/5 PASS）
- `experiments/results/p4/P4_results.csv`：变体运行结果（本次为 bench 记录）
- `experiments/results/p4_bench/`、`experiments/bench_c{1,2,3}.log`：开销实测原始输出
