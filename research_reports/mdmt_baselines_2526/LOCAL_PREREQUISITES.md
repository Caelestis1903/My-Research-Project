# 本地前置条件审查（2026-10-28）

本轮任务来源为根目录 `baselinestart.md`（GB18030 编码；只读解码，未改原文件）。用户要求在进入 RUN 阶段前批准。本轮授权范围为源码审计、设计、实现、独立审查和 CPU 合成正确性测试；真实数据试运行、开发训练和完整五种子训练均等待批准。

## 仓库与既有研究

- 工作目录 `/home/xzf/Code/phase0-3` 缺少 `.git`，`git status` 返回 128，无法声称其具有可核验的本地 HEAD 或干净工作树。
- GitHub 连接可访问目标仓库，权限包含 push；2026-10-28 读取的 `main` 为 `8eb461f5e0abc9c06385a4e8239e9621cae87ef8`。
- 最新实验分支 `26-10-28` 为 `97151c74544c3651feb805d1898bc9ce1b8d12d3`。远端 `experiments/mdmt_basp_v1/config.py` 与本地均列明开发种子 2022/2023、保留种子 2024/2025/2026。
- `research_plans/26-10-27_v2` 的原始 YAML 种子被其实际执行配置覆盖，不能用规划值代替已执行身份。
- 本次统一比较的五个种子为 2022、2023、2024、2025、2026。已有 BASP 报告只完成开发种子 2022/2023，Relation-Basis 报告为 2022–2025，均不能直接作为新的五种子结果。
- M3oE 当前研究身份为 fixed fusion、single Adam、无 AutoML/DARTS；不恢复已移除的机制，不修改既有算法和结果。
- 独立交付 checkout 位于 `/tmp/mdmt_sota_baselines_delivery_20261028`；不在原目录执行 `git init`，也不覆盖历史分支。

## 数据

- 原始 MovieLens CSV：`/home_new/xzf/datasets/ml-1m/ml-1m.csv`。
- 原始 KuaiRand CSV：`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。
- 原始 CSV 的既有迁移核验记录在 `AGENTS.md`；本次未重新计算文件哈希。
- 旧 `/home/xzf/data` 路径已失效，`/home_new/xzf/datasets/mdmt_next_20261024_strict_v1` 不存在。
- 最新代码指向的 `artifacts/mdmt_relation_basis_v2/data/movielens` 和 `artifacts/mdmt_relation_basis_v2/revision1/data/kuairand` 只剩 `DATA_MANIFEST.json`、`PREPROCESSOR.json`、`VALIDATED.json`；`x.npy`、`y.npy`、`rows_*.npy` 缺失。旧 VALIDATED 收据不能证明当前数组可用。
- 可复用冻结工具：`experiments/mdmt_relation_basis_v2/vendor_snapshot/mdmt_dfd_20261026/vendor/strict_data/data_roles.py` 的原始行序、年龄/tab 映射、标签构造和角色分配。应在新输出目录重建并与旧预处理器、清单和固定行身份核对，不能改旧文件或只修路径后宣称就绪。
- 角色为 T0=fit，F=P_label，S=P_cal/P_update，Q=Q_explore，V=validation，E=test，legacy=Q_confirm；标准基线的具体训练权限由设计规格明确冻结。V 选点，E 只作冻结后评估；E 已历史暴露，报告必须标明。

## Python 与计算资源

实际检查使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`：Python 3.13.5，PyTorch 2.9.0+cu126，CUDA 可见，设备数 2。

2026-10-28 18:24（Asia/Shanghai）`nvidia-smi` 只读快照：

| 设备 | 显存使用 / 总量 | GPU 利用率 | 已有计算进程 |
|---|---:|---:|---|
| RTX 3090 / GPU 0 | 17650 / 24576 MiB | 100% | PID 826260，python，17624 MiB |
| RTX 3090 / GPU 1 | 23883 / 24576 MiB | 0% | PID 616369，VLLM::EngineCore，23860 MiB |

不停止或修改上述进程。批准后的 runner 需要重新检查资源；本快照不能用于承诺可用 GPU 或运行时长。

## 状态

此文档是前置审查记录，不是实验结果或审查 PASS。真实训练和 AUC/LogLoss 均未测；是否进入 RUN 由实现、测试、独立审查、数据就绪情况及用户批准共同决定。

后续准备回执：两份数据已在新的 `artifacts/mdmt_baselines_2526/data/` 下恢复，MovieLens 1000209 行、KuaiRand 2507666 行；原始转换公式、T0 词表/缩放、写出数组与角色完整性逐值检查通过。八种原始角色的六格样本/正负数均与历史清单一致；见 [DATA_READINESS.json](DATA_READINESS.json)。历史数组本身已缺失，未声称做历史数组逐字节比较，也未重新计算哈希。空 checkout 缺少的四份历史预处理 JSON 已纳入新包 `reference_data/`，不包含原始样本或数组。
