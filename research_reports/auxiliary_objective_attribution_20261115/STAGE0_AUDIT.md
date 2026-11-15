# Stage 0 只读审计

结论：**Stage 0 PASS；复用10条CLEAN，B/C各10条重跑。Stage1新增40条、2000epochs。** 本阶段没有训练和Test预测/指标。

## 仓库与源身份

计划是 main 的 `research_plans/26-11-15_auxiliary_objective_attribution/` 目录；当前 main HEAD=`e5738e52294f064c23a9b953fe74abb84de713df`。历史来源分支 `26-11-15` HEAD=`52b48c9aab066d56fa91a060864c7c9abf12e50c`，相对当前 main 为 main-only14 / pilot-only2 个提交；未合并来源分支。新工作区为独立 sparse worktree，当前 main 无 AGENTS.md/start.md，用户提供的原工作区指令持续适用。

来源实现提交 `bf36e454adc2d0c569218a7993834f6d56cfc97f`。重新计算 source_receipt 中59个 `.py/.json` 文件及有序文件清单 SHA256，均等于 `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`。其中52个 `conditional_information_momentum_20261102_v3` utility 文件与新 main 逐字节一致。其余7个为来源实验适配代码；新实现不能依赖来源工作区运行时 import。

已阅读本次11份计划、main AGENT/最新AI handoff/Global Summary/Stable Findings/Open Questions及相关实验/证据/指标/架构上下文、11/14的15–17修订和manifest、来源 source map、implementation/final review、completion、protocol/core/worker和正确性测试。既有EMA/PUB/PAIR结论作为背景，不增加重验矩阵。

## 数据与软件

| 数据集 | 实际CSV SHA256 | train/val/test | train域计数 |
|---|---|---|---|
| MovieLens | f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4 | 800167/100021/100021 | 155949/318599/325619 |
| KuaiRand | a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f | 2006132/250767/250767 | 172782/1743898/89452 |

独立读取 relocated raw CSV 计算hash；只读反序列化既有data-only cache，重新计算所有六个 train/val/test x/y 数组hash、split人口fingerprints和原始行ID hashes，全部匹配main resolved_protocol和来源manifest；encoder/scaler hash及自然人口配置相同。允许核验Test数据身份，没有Test推理。原始CSV路径为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 与 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。无需缺失的mdmt_next prepared数据。

实际环境与pilot记录一致：Python3.13.5；torch2.9.0+cu126；numpy2.1.3；pandas2.2.3；scikit-learn1.6.1；scipy1.15.3；pytest8.3.5。项目Python为 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

## 轨迹、协议与选点核验

已定位全部60条pilot结果；针对CLEAN/B/C的30条逐条核对50份epoch记录，共1500份：完整50epoch、ML1000/KR2450个optimizer/EMA更新、源码hash、配置和Test=false一致。对每条online/EMA重算最早严格最大val六格Macro AUC，所选epoch、完整指标与对应状态hash均一致。用main模型factory及同seed在CPU重建10个dataset/seed初始模型，30条结果的initial_hash均匹配。

固定1模型、训练gate detach、自然loader、Adam和EMA的具体实现见SOURCE_MAP和设计。没有BatchNorm/Dropout；普通训练不启用historical PAIR/ECCM/dynamic EMA。

## 复用裁定

| 对象 | 数量 | 裁定 | 依据 |
|---|---:|---|---|
| CLEAN，两数据×五seed | 10 | REUSE_VALIDATION | 公共52文件、真实数据/配置/初始化/预算/选点/指标/软件相同；无辅助support依赖 |
| B，两数据×五seed | 10 | RERUN_MATCHED | 旧日志缺失候选/accepted行ID、mask、标签组成和有效权重分解 |
| C，两数据×五seed | 10 | RERUN_MATCHED | 同上 |
| Z-PAIR/Z-EXAMPLE | 20 | NEW | pilot没有零扰动轨迹 |

扫描来源B/C共34500条batch记录，只保存accepted计数和部分cell计数/范数，没有逐行候选/accepted索引、mask或逐标签权重。候选sampler虽已键控且可由batch context复现，后续accepted取决于当时模型；旧状态未保存磁盘，汇总计数不能重建接受行。按本次计划02/05的精确复用门，B/C原结果仅用于历史解释，新归因比较使用有完整支持日志的匹配重跑。

CLEAN原始result、50epoch验证及必要batch日志和收据由实现阶段复制到新reused目录并记录SHA256，原文件只读。新增40条取代原20条新增预算，已在派发前向主agent说明；不增加A/RANDOM/GAUSSIAN等科学矩阵。

## 问题分级

- **Relevant，已处理：** 来源B/C缺失support身份会阻止本轮有效权重归因；只重跑必要匹配臂。
- **Relevant，已处理：** 计划审计时main HEAD为79fbd44，当前已到e5738e5；52个公共源码逐字节一致，当前commit与分支关系重新记录。
- **Irrelevant：** 历史报告保留原路径/旧计划seed池；本次执行直接使用已迁移数据路径及2022–2026，不改历史文件。

Stage2/HPO/Test均待Stage1决策，当前NOT_TRIGGERED。Stage0未发现阻止Stage1设计实现的数据或数值语义差异。
