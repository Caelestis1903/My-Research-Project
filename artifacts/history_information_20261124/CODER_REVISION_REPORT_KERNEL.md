# CODER IMPLEMENTATION REPORT — 条件内核修订

本次修订处理独立审查 `d34b8b15` 的阻断 2、3、5 和阻断 4 的条件源验收。另在数据加载入口补上停止复核，供 CLI 的冻结 Test 竞态修复调用。

## 改动

- `protocol.py` 与 `optimizers.py` 为 N、HyperGate、Gaussian 的历史共同零点及 HyperGate、Gaussian 的 λPAIR=0 建立规范父优化器配置。直接父臂和旁路都使用同一父 `RunSpec` 序列化。训练外层继续携带原 arm、来源、组、依赖和背景目标。
- `optimizers.py` 在提交前拒绝 native 梯度为 `None` 且同参数 raw PAIR 非零的矛盾图。native `None` 配 raw `None` 或全零时跳过该参数的 WD 和时钟。native 梯度存在时，raw `None` 或全零仍记录活动事件并推进辅助时钟。
- `sources.py` 保留主干必需来源清单及身份，独立登记 Gaussian 专用来源、哈希和阻断原因。共享 Gaussian 基础设施仍由主干清单核对。Gaussian 专用缺件使该模块处于 `UNRESOLVED_IDENTITY`，S0 主干可以继续。冻结 Test loader 在人口核验及真正 Test 编码前检查当前停止状态。
- `groups.py` 保留 HyperGate tower 的完整调用序列，使重复调用和乱序均触发检查失败。
- `kernel_checks.py` 提供 `conditional_contracts(module, root)`。成功返回 `source`、`support` 和 `result=PASS`，使用 S0 登记的身份、两数据 seed 2022 和已核对的 Train/Validation 缓存。该函数由条件队列在触发时调用。

## 条件验收内容

| 模块 | 派发前源对照 |
| --- | --- |
| N | 每个数据固定自然 Train 256 行。核对三行域 embedding 注册、N 与 S_DENSE 两种作用域各自的原生 Adam 补集逐参数值和状态哈希、共同 PAIR 身份，以及局部 `None`、零值和无 PAIR 的梯度支持与时钟。 |
| HyperGate | 每个数据首个原生 1,024 行批。核对原 `_forward_terms` 与同图 tower logits、原目标四个 components、训练 dropout 的 RNG 消耗、直接发表 Adam 的模型与状态哈希及默认 flags。核对 B1/H0 的同批 PAIR，以及原生 batch/尾批和 epoch 步数。 |
| Gaussian | 两数据完整首尾批的固定 Train 行身份、D0 每行覆盖、clean 概率与 logits、GA31-R3 基础目标和源噪声。KR 对照固定 D0 coverage 入口，ML 对照带显式数据键的固定通用入口。另按字段独立复算 SHA256 种子及噪声张量并检查 PAIR 取样支持。 |
| AutoIFS | 每个数据 search 最后一阶段和 reuse 第一阶段各一批 256 行机械事务。核对原 stage RNG、rewind、search 双 Adam、发表 hard ticket 和 selector 移植、冻结 gate、fresh reuse Adam、reuse 单 Adam 及状态加载回环。 |

## 已执行检查

- 项目 Python 3.13 对 `protocol.py`、`optimizers.py`、`sources.py`、`groups.py`、`kernel_checks.py` 执行 `py_compile`，通过。
- `python -m experiments.history_information_20261124.kernel_checks` 通过。FP64 方向代数最大误差为 `7.771561172376096e-16`。非零 raw 与 native `None` 的负路径拒绝并保持整个模型与优化器状态哈希。合法 raw `None`/零值及无 PAIR 路径通过局部时钟核对。伪造冻结 Test claim 被拒绝。
- 五组固定小张量旁路各执行两步，模型、优化器和 EMA 三哈希逐组精确相同。外层候选与父臂使用不同 job key。下表给出各组精确优化器哈希。

| 路由 | 相同的优化器 SHA256 |
| --- | --- |
| `Cstar_N → H0_N` | `0c405f025a11361f6bccb85e621cbf496939ec6c95d8b1f35ec95fdb13beadf4` |
| `HG_Cstar → HG_H0` | `fa79f5f631383f45b96ff206bec856d93a676523d6a00dbff6e4102868a1d4ea` |
| `G_Cstar → G_H0` | `7bb91cbf220b60e8e825501b59193898ef1dbfe6f98f5e1d9f64655b11b4c26c` |
| `HG_H0` λPAIR=0 → `HG_B0` | `e8b2a8654a33d47fa87df711819eb47526fd4eef3d912c4d3beedcc78597ee1c` |
| `G_B1` λPAIR=0 → `G_B0` | `96d48329ae80675296c25f84e68f9ebff5491950517ffd67579e8b98f40f196c` |

- 源隔离检查在文件读取处分别模拟 D0 coverage `objectives.py` 和通用 GA31-R3 `objectives.py` 缺失。两次均返回 `UNRESOLVED_IDENTITY`，主干 175 个来源记录与哈希完全一致。实际文件齐备时另核对 7 个 Gaussian 专用来源。检查对 runtime 遥测使用替身。

## 待验收

本轮条件模块尚未触发。上表四项真实条件源对照等待各自触发，当前通过项为语法、小张量回归和来源隔离。主代理将刷新 S0 来源清单，再交独立审查。正式训练及 Test 访问均未发生。
