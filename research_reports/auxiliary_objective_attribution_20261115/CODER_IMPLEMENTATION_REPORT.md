# CODER IMPLEMENTATION REPORT — Stage 1

实现完成：2026-11-15。训练源码 SHA256：`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。
已按冻结 `DESIGN_SPECIFICATION.md` 实现 Stage 1；本报告记录实现与工程检查，不作科研结果裁定。

## 文件及设计映射

| 文件 | 已实现的责任 |
|---|---|
| `core.py` | 沿用 pilot B/C 的键控采样、完整数值 gate Jacobian、pair cap/example BCE 方向；Z 先执行同样有效性规则再将 accepted delta 精确置零；普通训练 gate detach 保持原样 |
| `protocol.py` | 固定 seeds、配置/源码/数据/选点 hash；只读数据 cache 完整身份核对；10 条 CLEAN 的原始证据复制及 SHA256 收据；复用 main 模型、RNG、评价与 RAM guardian |
| `runner.py` | 40 条 × 50epoch × batch40960；一次 backward/Adam/after-Adam EMA；独立 online/EMA 验证选点；每 epoch RAM 状态提交；逐卡资源准入；完整 trace 和有效权重 gzip JSONL |
| `checks.py` | 实际模型逐行 Jacobian、无关行独立性、同 FP32 初始化系数的 FP64 有限差分 |
| `tables.py` | 只导出原始指标与配对算术；核对完整预算、最早严格最大选点、state hash、自然 row order；输出六格/域/任务/Macro/Traffic/Pooled、E10/E30/E50、成本、权重、support overlap |
| `test_core.py` | 7 项紧凑 CPU 检查 |
| `README.md` | 完整执行入口、数据日志字段/skip codes、只保留 RAM 状态的生命周期 |

新实现只 import main 的 `conditional_information_momentum_20261102_v3` 公共 utility。必要 B/C adapter 来源为固定 pilot 实现提交 `bf36e454adc2d0c569218a7993834f6d56cfc97f`；运行时不 import 外部 pilot 工作区。未修改历史代码、vendor、计划、数据、报告或全局指标台账。

clean 和辅助 loss 的训练运算路径分别保持 `F.binary_cross_entropy(model(x), y)` 与 `2*F.binary_cross_entropy(target_probability, target_label)`，总目标为 `clean+.25*auxiliary`。ledger 在 detached 概率上另外计算逐项统计，不重排训练 reduction。M=0 时辅助项为同 device/dtype 的标量零。

每 batch 记录候选/accepted 原始 row ID、局部索引、task/domain/label、pair ID、keep mask、skip reason、方向/RMS/raw/capped/actual norm 和 cap；记录 12 个 domain×task×label 的数量、clean/aux numerator、系数质量和目标贡献。选中辅助 entry 系数为 `.5/M`，相对 clean `1/(2B)` 为 `B/M`。gradient norm/cosine 每 epoch 首 batch 固定记录，不参与选点或门控。

## 工程检查

项目 Python：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

1. `python -m pytest -q experiments/auxiliary_objective_attribution_20261115/test_core.py`：**7 passed，3.48 秒**。覆盖独立键控 RNG 和 support/label、两种 Z bitwise zero 与 forward/loss/gradient 等价、M=0 exact clean 及一次 Adam/EMA、Jacobian/FD/hook 清理、完整 RAM 继续训练与 selected identity、五 seed 算术及双空 support 的 overlap。
2. `runner prepare`：**PREPARED**。复制 10 条 CLEAN 的结果、500 份 epoch JSON、batch 日志及收据，共 **537 份原始文件**，逐文件 SHA256 留在 `attempt1/reuse_receipt.json`；原件保持只读。两个 cache 的 raw CSV、六个数据数组、split/row-ID、encoder/scaler 及人口身份均通过。
3. 工程 smoke 直接调用 `runner.smoke(attempt1, 'cuda:0')`：**PASS**，两数据 × B/C/Z-PAIR/Z-EXAMPLE 共 **8 次真实 40960 full-batch backward**。每项 loss/参数有限，计数均为一次 Adam 和一次 EMA；本次首 batch 各臂 M=1536。实际模型四行 Jacobian/有限差分合并在 B smoke 中，普通训练 FP32。

| 数据集 | 峰值 allocated bytes | 峰值 reserved bytes | FP64 FD 最大绝对误差 | FP32/FP64 梯度最大差 | 无关行梯度最大差 |
|---|---:|---:|---:|---:|---:|
| MovieLens | 1118085120 | 1352663040 | 2.776e-11 | 1.607e-7 | 1.490e-7 |
| KuaiRand | 1302201856 | 1635778560 | 2.870e-11 | 2.480e-7 | 2.384e-7 |

检查数据位于 `artifacts/auxiliary_objective_attribution_20261115/attempt1/CPU_CHECKS.json`、`SMOKE_REPORT.json`、`source_receipt.json`、`run_manifest.json`、两份 data manifest 和 `reuse_receipt.json`。源码 hash 覆盖新训练入口/core/protocol/checks/包初始化及 main 公共 `.py/.json`；纯统计导出、测试与说明不进入训练 hash，便于训练后生成报告而保持训练身份。

唯一警告来自继承 utility 使用的 PyTorch TF32 设置 API 弃用提示；数值设置保持源协议，未因此改动 vendor。

## 当前状态与交接

正式 40 条训练：**未启动，等待独立 reviewer 通过与主 agent 派发**。正式 `smoke`/`stage1` CLI 会验证 review verdict、阻塞项和训练源码 hash；`stage1` 还要求相同源码的 smoke PASS。每 GPU 显存与功耗均 <85% 且系统 RAM<90% 才准入，首批峰值未确认时每卡最多一个 allocating worker，之后按实测资源并发。

Stage 2、HPO、Test 和额外机制实验：**未实现/未运行，等待 Stage 1 科学决策**。实现中没有 Test evaluator 或性能提前停止路径。磁盘没有写入模型、优化器或 checkpoint 文件。正式训练、结果完整性复核和科学报告由后续指定 agent 完成。
