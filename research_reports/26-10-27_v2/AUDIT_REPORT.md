# Relation-Basis v2 阶段 0 审计

日期：2026-10-27。状态：`AUDIT_COMPLETE / PREPARED_REBUILD_REQUIRED`。本次审计读取源码、原始 CSV、角色行计划与设备信息；未启动训练、未评分模型、未生成 prepared 数据或模型状态文件。

## 当前可运行条件

两个原始 CSV 的实测 SHA-256、行数和自然域计数均符合锁定记录。GPU 可用，依赖已安装。缺失的是历史 prepared 数据与模型缓存，需要在新实现中完成精确数据重建和内存共同前缀重建，再通过验收及独立审查，才能启动阶段 1。

| 数据 | 实测行数 | 自然域 0/1/2 行数 | 原始身份 |
|---|---:|---|---|
| MovieLens | 1,000,209 | 210747 / 395556 / 393906 | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` |
| KuaiRand | 2,507,666 | 178087 / 2236414 / 93165 | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` |

真实路径分别为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 与 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。MovieLens 使用 `rating>3`、`rating>4`；KuaiRand 直接使用存储的 `is_click`、`long_view`。两任务全观测。ML 特征 8 列，其中 age 为 dense；KR 特征 34 列，其中 domain_indicator 为映射域标识。具体 schema、逐角色逐格正负标签计数与行身份见 `AUDIT_DATA.json`。E/legacy 的读取用于 schema 与计数审计，未产生任何模型风险或选择决策。

## 数据角色与重建

`/home_new/xzf/datasets/mdmt_next_20261024_strict_v1` 在审计时不存在。历史 strict_data 的 `prepare(dataset, destination, raw_path=...)` 已具有从原始 CSV 完整重建的代码：按自然行序 80/10/10 划分外层，在首 80% 内按域使用固定 PCG64 行计划；只在 fit 行上拟合词典与 age scaler，之后编码全部行，保留 OOV=0。执行重建时必须显式传入当前 raw_path，不能使用其旧路径默认值。

本次只在内存重算行计划，未调用 `prepare`。重算的所有角色域计数与 v1 锁定表一致：

| v2 角色 | 历史行角色 | ML 域 0/1/2 | KR 域 0/1/2 |
|---|---|---|---|
| T0 | fit | 124759 / 254879 / 260495 | 138225 / 1395118 / 71561 |
| F | P_form ∪ P_read | 9357 / 19116 / 19537 | 10366 / 104634 / 5367 |
| S | P_cal ∪ P_update | 6238 / 12744 / 13025 | 6912 / 69756 / 3578 |
| Q | Q_explore | 7797 / 15930 / 16281 | 8639 / 87195 / 4473 |
| V | validation | 27111 / 35523 / 37387 | 2692 / 246232 / 1843 |
| E | test，后续标记 exposed | 27687 / 41434 / 30900 | 2613 / 246284 / 1870 |
| legacy | Q_confirm，历史暴露 | 7798 / 15930 / 16281 | 8640 / 87195 / 4473 |

原始角色 seed=20261024，P_label 二分 seed=20261025，域步长=1009；这些属于数据协议，不因训练 seed 变化而改变。两个任务共享每条交互的划分。prepared 重建后的完整数组、fit-only 预处理和 row-id 验证仍是正式训练前必需验收。

## 源码身份与复用范围

本地目录没有 `.git`，`git status` 返回 not a git repository；不能把本地目录描述为某个 Git checkout。审计固定远端 main commit `8eb461f5e0abc9c06385a4e8239e9621cae87ef8`，DFD 源码实际位于 `history_experiments/26-10-26/experiments/mdmt_dfd_20261026/`。已通过 GitHub 工具在该 commit 直接获取并核对 NativeAdapter、functions、strict_data、fixed 与 original 五个关键文件的 Git blob。另对本地 RCF vendor_snapshot 中的 26 个历史文件重算 Git blob，全部与其记录的 `dc32f60a6aafce98a25838a485e02cead5425fc6` 身份一致。

可复用 NativeAdapter、PilotBank、strict_data 和其 pinned fixed1 依赖。NativeAdapter 输出 `h[B,256]` 和 `logits[B,2]`；PilotBank 的每来源两任务隐藏层各 32 维，可拼接为来源 64 维函数，按 T0_s 拟合标准化器。fixed1 保持 exp_d/exp_t/bal_d/bal_t=1，不启用 learned fusion/AutoML。

`SOURCE_MANIFEST.json` 列出候选依赖的来源 commit、路径、实际/记录 Git blob、SHA-256、imports 与许可证检查。当前仓库根目录和 supplied snapshot 未包含许可证文件，所检源码头未含版权声明；保留现有来源信息。最终实际导入闭包由 coder 完成，候选清单不等于全部已复用。

不要直接调用旧 RCF/DFD build_prefix：它们包含本实验不需要的 local bank/OOF 等训练，以及对暴露角色的过早前向路径。不要复用旧 probability clipping LL 作为本版 logit LL；v2 要求 float64 softplus(logit)−y·logit。

## 环境、资源与预算

Python 3.13.5、PyTorch 2.9.0+cu126、CUDA runtime 12.6；CUDA 可见设备 2。NumPy 2.1.3、pandas 2.2.3、scikit-learn 1.6.1、SciPy 1.15.3、pytest 8.3.5、PyYAML 6.0.2 已安装。无需新增依赖。完整 `pip freeze` 与设备命令输出见 `ENVIRONMENT.txt`。

审计时两张 GPU 均为 RTX 3090 24GiB。GPU0 已使用约 17.6GiB、约 221W/350W；GPU1 约 18MiB、约 26W/350W。主机约 56GiB 内存可用，工作区磁盘约 237GiB 可用，数据盘约 2.0TiB 可用。这些是时点观测，runner 启动每条进程前重新查看单卡占用；不结束或修改他人进程。

按用户最新训练 seed `2022,2023,2024,2025`：阶段 1 为 **2 数据集×6 方法×4 seed=48 条适配轨迹**，每条上限 20 epoch、含 1 epoch warmup、batch2048；共同前缀 **8 个唯一身份**，每份 native24 epoch、physical batch40960，owner-only pilot12 pass，实际成本分别入账。formal 仅按晋级条件执行。首批真实前缀运行将给出耗时和峰值显存，不把未测成本填写为零。

`start.md` 禁止本地模型及检查点文件；selected/final/optimizer/RNG 状态必须常驻 RAM，磁盘只保留代码、数据、配置、指标、日志和报告。恢复验收也应在内存完成。

## 审计发现与下一步

| 级别 | 发现 | 对下一步的影响 |
|---|---|---|
| Critical | prepared 缺失，现成 RoleBundle 直接加载会失败 | 正式训练前重建并验证自然行数据；仅改路径不足以运行 |
| Critical | 历史权重缓存未恢复 | 按锁定 native24/pilot12 协议在内存重建，计入共同前缀成本 |
| Critical | 旧 build_prefix/指标/角色权限与 v2 预算及数据权限不同 | 只复用已验证组件，由新实现落实 v2 角色与 logit 指标 |
| Relevant | 本地无 Git 元数据 | 以固定远端 commit 与逐文件 blob 建来源清单；交付由主 agent 处理 |

下一步是设计闭合后由独立 coder 实施上述精确路径、CPU 验收及数据重建，再由另一 reviewer 审查。GPU 训练尚未开始。
