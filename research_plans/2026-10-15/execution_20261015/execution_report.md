# 2026-10-15 规划执行进度（更新于 2026-10-16）

状态：**PARTIAL / 目标仍未完成**。本次未运行新训练、未加载模型 checkpoint、未恢复 optimizer；完成的是路径修复、真实数据加载验证、A0/A1/A2 可访问证据审计及统计复算。

## 授权与工作流

按用户最新明确要求全程不派发子 agent。旧 AGENTS 中的子 agent 分工不执行；已完成审计设计、实现、运行后复核与分析，不把证据处理工作称为已走完新算法实验审批链。自审不标作独立审查，不能借此声称达到规划的 independent review 条件。

原规划 A0/A1/A2 优先；B1/B2/C1/C2/D1/E1 均为条件性研究。保留 F0 历史 OPEN/FAIL、NOT_GRANTED、NO_GO；当前操作不使用其 production queue/训练/恢复路径，也不修改旧 hold/authorization。用户的端到端任务授权支持常规路径修复、审计和发布，但不能用缺失原状态替代科学配对。

## 任务清单

| 节点 | 当前状态 | 已完成 / 未满足 |
|---|---|---|
| 环境/数据路径 | PASS（限定范围） | 两套数据真实加载；哈希、split、类别计数与 schema 已保存。MovieLens 验证 d0_large/2022，KuaiRand natural/2022 |
| A0 | PARTIAL / INCOMPLETE | 全本地科研目录工件与源码 SHA；缺失列表；静态调用链。未恢复 S4/S5/S6/S8 全原始 run 工件，未获独立复核 |
| A1 | PARTIAL | AO04 KR 三个完整 run ×7 checkpoint×6 cell=126 测试指标复算；4 个选点检查；现有 aggregate CSV 的 ML/KR 配对；rho 日志；KR 单 seed 配对 bootstrap。ML 预测覆盖不足，validation 缺行 ID，完整 Jacobian 未执行 |
| A2 | PARTIAL / INCOMPLETE | 两数据集四 seeds 的 selected CSV、各域/任务和多规则审计；无 S6 原预测/checkpoint/optimizer，不能做 S6 paired bootstrap 或 exact 新重选 |
| B1 | NOT_RUN_UNMET_PREREQUISITES | 原 S6 配对/初始化/批次/状态缺失，无法证明原协议 fidelity；dense state dump 与独立审查未完成 |
| B2 | NOT_RUN_UNMET_PREREQUISITES | S5 原始状态与独立复核缺失；C/A 新因子不能自动承接旧 HT/DH |
| C1 | NOT_RUN_UNMET_PREREQUISITES | 无可核验的 common-θ ghost state；禁止跨轨迹硬插状态代替 |
| C2 | NOT_RUN_UNMET_PREREQUISITES | A0/A1 科学门尚未通过；未新增 oracle/训练，不宣称同目标等价已通过 |
| D1 | NOT_TRIGGERED | 无新复制机制和预注册边界矩阵 |
| E1 | NOT_TRIGGERED | 无新科学缺口确认、独立 holdout 或 novelty review |
| Git 发布 | 见 publication_status.json | 原目录没有 .git；独立 checkout 专用分支保存成果，未覆盖原工作目录 |

## 环境、代码与实际运行

环境记录：`environment.json`。实验依赖已有 `phase0-3-env`（Python 3.13.5）；审计表格/绘图还使用 base 环境，版本来源分别记录于命令/环境。GPU0 首次观测约90.8%占用，GPU1约57.1%；GPU1 可独立满足显存门槛，无需等待 GPU0。未启动训练的原因是科学工件/复核条件未满足，不是要求两卡同时空闲。外部用户进程未改动。

数据实际路径固定：

- `/home/xzf/data/ml-1m/ml-1m.csv`，SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`。
- `/home/xzf/data/kuairand/processed/kuai_rand.csv`，SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。

只修改 `experiments/stage_S1_data.py` 与 `stage_S1_orchestrator.py` 的四个路径常量/fallback。没有修改目标函数、抽样、标签、split、模型或超参数。公共加载器变化须作为新源码哈希记录，旧 config 保留其历史身份，不将新路径源码声称为旧源码 bytes。

验证：两套数据完整 CPU loader preflight 通过；现有数据测试 2/2；weighted tied-AUC bootstrap 实现与 sklearn 的100个加权用例一致（1e-14）。正式科研训练 0 runs，历史预测统计是 forensic analysis。

首次直接 pytest 被旧 conftest 以 exit75 拒绝（缺 reserved slot FD）。仓库未保留其指名的 run_f0_python_task.sh；新增 bounded `run_data_tests.py` 使用现有 acquire/adopt slot API，真实持有 parent/child 两个 slot 后传 FD，原 conftest 不改，重跑2/2通过。该 launcher 只用于数据测试，不是科学 RUN 许可。

首次逐行比对发现 NPZ physical_domain 字段使用 local ID，随后核对 loader 映射及原 tab，确认0/1/2对应tab0/1/4。先停下诊断再修正审计 mask，未改历史 NPZ。

## 恢复调查与保留原则

当前本地所有 origin tracked files 均存在，远程没有更多可恢复的 tracked 工件。project.zip 中没有 S5/S6/S6ML 对应原始 config/manifest/checkpoint/预测；同用户 Code/封存 中无 S4–S8 stage 路径。详见 `artifact_recovery_search.json`。此结论仅覆盖明示搜索范围，未扫描其他用户磁盘。

缺失工件为 INCOMPLETE，不判作科学 NO；AO04 KR B1 seed2023 的历史 manifest 明确 INCOMPLETE（38 epochs，native faults，已用一次 resume），不重复恢复或补造结果。B3/B4 不计为阴性。

## 恢复命令（均从原工作目录执行）

```bash
python research_plans/2026-10-15/execution_20261015/audit_evidence.py inventory
OPENBLAS_NUM_THREADS=1 python research_plans/2026-10-15/execution_20261015/audit_evidence.py metrics
OPENBLAS_NUM_THREADS=1 python research_plans/2026-10-15/execution_20261015/analyze_tables.py
OPENBLAS_NUM_THREADS=1 python research_plans/2026-10-15/execution_20261015/bootstrap_ao04.py
OPENBLAS_NUM_THREADS=1 python research_plans/2026-10-15/execution_20261015/make_figures.py
/home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-15/execution_20261015/dataset_preflight.py movielens
/home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-15/execution_20261015/dataset_preflight.py kuairand
CUDA_VISIBLE_DEVICES='' /home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-15/execution_20261015/run_data_tests.py
```

后续优先恢复缺失原始 run 工件并核对 provenance；若决定原工件确实无法恢复，需要明确制定新协议复制及其比较边界，不能静默把不可配对的新训练当作原协议 exact replication。通过这些科学前置检查之后，才使用每卡≤5进程、单卡显存≤90%独立准入的持久化训练调度。
