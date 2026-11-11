# 实验来源与复核入口

本报告对应 `ema_next_20261111/attempt1`，实现基于分支 `26-11-11` 的提交 `5ef45b18b973ed11be6fbb5f654562349998de60`。规划来自 main 的提交 `01979f6588935f713029b2b5e04557de9b8c153f`，目录为 `research_plans/26-11-11_ema_next_phase/`。19 个规划文件逐字节保留，目录 SHA-256 为 `b9dbce949c81cbe9fee961402048945f18753de59561cc31b26129c37b09f069`；逐文件 Git blob 与 SHA-256 见 [PLAN_PROVENANCE.json](PLAN_PROVENANCE.json)。

科学实现位于 [experiments/ema_next_20261111](../../experiments/ema_next_20261111)，运行绑定 SHA-256 `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`。调度适配器位于 [ema_next_admission_20261111.py](../../experiments/ema_next_admission_20261111.py)，最终版本 SHA-256 `1755e97d6bc087d555f48145cb3bc3c5a0c70b60afa5bfb263b678ae82efc346`。源模型、损失、数据转换与历史报告未修改。设计、两份实现报告、独立审阅报告与哈希绑定收据均在本目录。

## 环境与数据

实际解释器为 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，Python 3.13.5、PyTorch 2.9.0+cu126、NumPy 2.1.3；训练使用两张 24 GiB RTX 3090 的实时可用资源。完整环境与依赖来源记录见 `evidence/SOURCE_RECEIPT.json`。

| 数据 | 原始文件 | SHA-256 |
|---|---|---|
| MovieLens | `/home_new/xzf/datasets/ml-1m/ml-1m.csv` | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` |
| KuaiRand | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` |

复用并审计自然分布缓存 `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache`。数组内容、行 ID、划分、编码器、scaler、P_A/P_B/V_mech 与参数分组的身份分别见 `evidence/DATA_MANIFEST.json`、`PANEL_MANIFEST.json`、`PROBE_MANIFEST.json`、`GROUP_MANIFEST.json`。未使用缺失的 `mdmt_next_20261024_strict_v1` 数据。

## 发表模型对照

从提交 `c8675ae2923c15b6965884b140b5b646690cd95e` 的 `research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv` 复用发表模型端点；原文件 Git blob 为 `a3d7aa3e2035f0b7ff55406e260fc59c880ba254`。本机读取文件位于 `/home/xzf/Code/phase0-3_gaussian_coverage/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv`，内容已与固定 Git blob 验证一致。

过滤条件为 `role=PUBLISHED_BASELINE`、`checkpoint=selected`、`estimator=online`、`protocol_id=AUC_SELECTED_V1`、`data_protocol=FULLTRAIN_NATURAL80_10_10`、`split=test`；方法为 `M3OE_FIXED1_ADAPTED`、`HYPERGATE`、`AUTOIFS`，每个数据集/方法含 seed 2022–2026。主要 AUC 对照固定为 ML M3oE、KR HyperGate，逐 seed 配对时不更换基线。HyperGate 源超参的 `UNVERIFIED_PROPOSAL` 标签与 AutoIFS search/reuse 身份保持不变。

## 实际执行顺序

1. 设计完成后，两位 coder 按分工实现；独立 reviewer 检查设计、代码与实现报告。
2. 执行 16 项科学/执行聚焦检查、两数据集真实 GPU smoke；绑定最终科学实现哈希后运行。
3. 对每个数据集依次执行 Clean 父轨迹、B/C 冻结、20 条 A_MATCH、实际触发分支、Clean 主候选与唯一 Gaussian 家族。
4. 两数据集全部完成后，生成全局端点冻结，再计算新 Test、数表与释放收据。

本次使用的命令入口如下，详细配置由 PREPARED 与 JOB 记录确定：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_next_20261111.run prepare --out artifacts/ema_next_20261111/attempt1 --cache-root /home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_next_20261111.run smoke --out artifacts/ema_next_20261111/attempt1 --dataset movielens --device cuda:0
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_next_20261111.run smoke --out artifacts/ema_next_20261111/attempt1 --dataset kuairand --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_next_admission_20261111 --out artifacts/ema_next_20261111/attempt1 --devices 0 1 --previous-controller 1946919
```

最后一条是本次已校准 attempt 的最终控制器恢复命令；实际前身进程、启动时间、接管 worker 和显存高水位在执行收据中。适配器的 KuaiRand 前块基线绑定本次 guardian、首个准入记录和 MovieLens 完成记录，不能把旧进程号/身份当作新机器的校准信息。

## 运行修正与状态生命周期

初次 smoke 在 CUDA context 初始化前重置显存统计而失败，没有产生训练结果；修正后重新审阅并完成两次 smoke。正式训练期间科学实现哈希保持不变。调度修订解决待启动 worker 的显存预留、父状态重复计数、JSON seed 键的恢复比较，以及 A 短程误用完整父轨迹峰值。精确旧源码与 diff 保存在 [scheduler_revisions](scheduler_revisions)，每次由独立 reviewer 复核后仅替换控制器，保留 guardian 与训练 worker。

C 风险并列规则的 manifest 少列了“距 .5 最近”一级；运行时依据算法规格 14 补齐为“距当前旧质量 a 最近、距 .5 最近、较小 alpha”。原规划文件不变，修正记录与解析配置分开保存。

模型参数、Adam、EMA、历史状态、恢复状态与选中端点均只存在 RAM。发布产物包含代码、标量记录、允许的预测/面板身份与日志，不包含模型文件、原始数据缓存或 IPC 授权密钥。释放后可复算已发布数值统计；重新生成模型预测需要重跑训练。

## 数值复核

`ENDPOINTS.csv.gz` 给出每个冻结端点的 seed、epoch、状态哈希和全部指标；`PAIRED.csv.gz` 给出同 seed 的候选/对照原值、差值、样本 SD 和 t4 区间。`CURVES.csv.gz` 保存完整验证曲线，`A_SHORT_PAIRED.csv.gz` 按两 anchor 末两轮的预设公式归约 A。完整阶段门、配置选择和正式判定分别以 `DECISIONS.json`、`HPO_SELECTION.json`、`FORMAL_DECISIONS.json` 为准。

文件内容校验由最终 `ARTIFACT_INDEX.json` 列出。运行资源记录是定时采样；CUDA 分配峰值、每轮训练/观察时间另见 `COSTS.csv.gz`，不得把资源采样最大值写成连续监测峰值。

最终执行状态为 95 COMPLETE / 0 FAILED，实际 1415 epoch、46280 Adam 更新；1450 条标量 epoch 记录含 35 条 epoch 0。160 个冻结端点对应 147 个唯一 Test 前向对象，释放 tensor 669716384 bytes。

`analysis_tables.py` 只读已发布标量，补充同一冻结端点的 scope、域/任务配对和 E20/E35/E50 阶段统计；新增行以 `comparison_class=report_derived_same_frozen_endpoints` 标记。重算命令：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-11_ema_next/analysis_tables.py`。TEST_COMPLETE 中的 endpoints_sha256 对应原始发布文件 ENDPOINTS.csv.gz；该文件逐字节保留，其他发布文件的哈希以 ARTIFACT_INDEX 为准。
