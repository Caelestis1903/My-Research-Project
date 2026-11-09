# PUB-MDMT CODER IMPLEMENTATION REPORT

状态：IMPLEMENT COMPLETE，2026-11-09。依据已批准的 `DESIGN_SPECIFICATION.md` 实现；等待独立 reviewer 判定后由 runner 执行数值校准、低预算选择和正式训练。

## 实现交付

新增独立包 `experiments/pub_mdmt_20261109/`，没有修改历史 vendor、EMA 或原模型源码。最终实现身份：

`0eb54f049abef69e735b90af2dbca4fb88dd876687cea3ab2719472730069e4d`

该 hash 由包内 Python/JSON 源文件构成，排除 tests；三份检查/完整 batch smoke 收据均绑定此身份。模型、特征、数据加载和12组坐标来源固定到历史 `1391affa`，原始路径、source/destination SHA256 与仅 import 调整记录在 `SOURCE_MANIFEST.json`。概率指标实现原样复制同源 `vendor_ga31/metrics.py`。本包运行不依赖相邻 checkout 源码。

| 模块 | 实现内容 |
|---|---|
| `vendor/`、`registry.py`、`group_registry.py` | 冻结 fixed1 模型/自然数据构建、初始化注册顺序、gate detach 和12组互斥坐标 |
| `data.py`、`common.py` | 原 CSV/数组/行指纹/编码器/scaler/自然人口身份，独立 RNG、loader 和状态 hash |
| `optimizer.py` | 六格自然损失、2/6次 autograd、纯 Adam 候选、P0/P1/P2、公共 coupled-WD 差分、普通 Adam 回退、真实 alpha/mask 重放 |
| `solver.py` | 论文 Eq.(4) CCP、CLARABEL、原始 Gram 尺度恢复、正 alpha/utility 与双残差、可行初始化和明确失败原因 |
| `training.py` | 完整自然 epoch、A/F同轨迹、逐step系数与诊断、只读短步BCE探针、RAM选点/恢复、先冻结Validation选择再读取Test |
| `run.py` | audit/check/smoke、18配置数值校准、40条2epoch频率比较、70条50epoch正式轨迹及Validation触发组合、逐GPU资源准入 |
| `report.py` | 所有六格/任务/Macro/Traffic/Pooled原值、seed配对均值/SD/95%t区间、成本、门值和频率100离线系数陈旧度数表 |
| `manifest.json`、`README.md` | 固定来源/五seed/有限矩阵与执行命令 |

候选全部读取同一旧 m/v/t；状态在所有组候选完成后统一提交。物理参数通过原布局每坐标写一次。P1局部系数直接取S子集；P2局部独立求解；DT、P0的T和0/1活动任务组使用普通coupled Adam。G从对应C的逐step纯标量trace重放系数和fallback，核对初始化、loader顺序、step和完整系数hash。

所有选中/E10/E50模型状态、optimizer及RNG/loader恢复状态只保存在进程RAM。磁盘输出为JSON/JSONL/CSV日志和指标、代码及报告。中断job从原seed重跑，旧JSONL日志按attempt保留；不落盘训练状态。

## 规定检查结果

`artifacts/pub_mdmt_20261109/attempt1/CHECKS.json`：**PASS，16 passed，5.64秒**。日志 `CHECKS.log`。

覆盖真实ML/KR六格loss及梯度重构、空域、12组覆盖和非所属组零梯度；非零旧状态候选手算和不变性；disabled/fallback在WD=0/1e-4/1e-5下多步与原Adam逐元素对照；2/6任务solver解析/独立数值根及重排序/尺度；P0/P1/P2系数语义、DT与原Adam一致；公共WD手写公式；任务顺序、真实G重放、RAM optimizer/RNG/loader恢复；0/1活动分支、最早严格最大值、Test入口隔离、真实概率LL和频率100离线统计。

`AUDIT.json`：**PASS**。两个原CSV SHA256、缓存文件及数组内容、ordered row fingerprints/IDs、train-only编码器/scaler身份、split与domain人数均匹配冻结 source 协议。

完整batch GPU smoke在 `cuda:0` 执行，每个数据集每个拓扑一次forward/所有任务梯度/更新，batch始终40960，六次均 **PASS**，无求解回退。

| 数据 | 拓扑 | 单步墙钟秒 | peak allocated bytes | peak reserved bytes |
|---|---|---:|---:|---:|
| MovieLens | P0 | 2.596788346 | 1100982272 | 1350565888 |
| MovieLens | P1 | 0.315117220 | 1116999680 | 1365245952 |
| MovieLens | P2 | 0.395553176 | 1116999680 | 1365245952 |
| KuaiRand | P0 | 0.280047910 | 1294278656 | 1642070016 |
| KuaiRand | P1 | 0.450359325 | 1328884224 | 1644167168 |
| KuaiRand | P2 | 0.502124112 | 1328884224 | 1644167168 |

首个MovieLens P0步骤包含CUDA/solver首次初始化开销。完整Gram、alpha、残差、范数、utility、RSS和时间原值见 `SMOKE_movielens.json` / `SMOKE_kuairand.json`；它们用于执行成本与正确性检查。

## 环境与修改历史

项目 Python3.13.5 / PyTorch2.9.0+cu126 不变，安装cvxpy1.9.3、CLARABEL0.11.1及其依赖；精确新增列表 `requirements-added.lock`，完整包版本 `ENVIRONMENT_FULL.lock`。没有更改全局Conda或shell配置。

实现检查中修复了批任务候选 m/v 扩展的broadcast形状错误；原始状态现用expand后clone计算，保留旧状态。P1子集测试最初使用不可行冲突toy Gram，已换为有正utility可行解的fixture，求解失败的普通Adam回退仍单独覆盖。没有因测试或smoke调整科学目标、batch、预算或seed。

## Runner交接命令

从仓库根目录，设置项目Python：

```sh
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
# reviewer须在本目录REVIEW_REPORT.md写明 Verdict: PASS 或 Verdict: PASS WITH FIXES。
$PY -m experiments.pub_mdmt_20261109.run calibrate --device cuda:0
$PY -m experiments.pub_mdmt_20261109.run pilot --devices 0,1
$PY -m experiments.pub_mdmt_20261109.run train --devices 0,1
$PY -m experiments.pub_mdmt_20261109.run report
```

`train`先完整执行A/C0/C1/G0/G1，再执行C2/G2；最后以五seed selected Validation裁定组合门，触发时执行对应PUB+EMA五seed。既有完成job只在实现/配置身份一致时复用。公共CLI在科学执行前检查独立review、当前CHECKS和两数据full-batch smoke。调度按各GPU实际功耗与显存、host RAM和实测batch显存余量准入，不使用固定两worker上限，不缩batch。

数表输出目录为 `artifacts/pub_mdmt_20261109/attempt1/tables/`。原值与数值汇总由代码输出，科学结论和最终报告由orchestrator撰写。
