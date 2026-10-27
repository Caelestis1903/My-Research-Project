**Relation-Basis v2：结果与复现入口**

本次已完成48条适配轨迹、8份共同前缀、960轮适配和96个selected/final历史E结果。最终研究结论见 [FINAL_REPORT.md](FINAL_REPORT.md)，E解锁前的决策见 [STAGE1_REPORT.md](STAGE1_REPORT.md) 与 [STAGE_DECISION.json](STAGE_DECISION.json)。

原始逐seed/域/任务指标、选点epoch和模型身份在 [SEED_METRICS.csv](SEED_METRICS.csv)；全部训练轮次在 [EPOCH_METRICS.csv](EPOCH_METRICS.csv)。全指标均值/SD/区间包括单列的pooled AUC，见 [METRIC_SUMMARY_ALL.csv](METRIC_SUMMARY_ALL.csv)。其他对照配对差、任务汇总和训练阶段差分别在 PAIRED_ALL_CONTROLS.csv、PER_TASK_SUMMARY.csv、TRAINING_STAGE_COMPARISON.csv。参数/时延/显存/MAC与成本边界见 RUN_PARAMETERS_AND_COSTS.csv、COST_REPORT.md、COST_LEDGER.json。

原始数据路径、SHA与自然分割核验见 AUDIT_REPORT.md、AUDIT_DATA.json。项目环境是 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，Python3.13.5、PyTorch2.9.0+cu126；完整版本见 ENVIRONMENT.txt。原始CSV分别为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 与 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。不需要恢复已删除的历史模型缓存。

在仓库根目录重新生成本次已完成结果的统计表：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/analysis_tables.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/aggregate_costs.py
```

运行全新的复现实验时，使用新的artifact目录，避免覆盖本次结果：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 audit
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 plan
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 prepare --dataset movielens
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 prepare --dataset kuairand
CUDA_VISIBLE_DEVICES=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 train --dataset movielens --seed 2022 --device cuda:0
```

最后一条为常驻worker命令；分别对两数据集、2022/2023/2024/2025启动八条。按每卡实际显存与功耗低于85%追加进程，保持24 native、12 pilot、每方法20适配和原physical batch；保持全部worker存活到READY。模型、优化器和selected/final仅在RAM，不生成模型文件。

全新复现使用最终R2源码，八个worker具有同一全包source hash，可用原CLI直接冻结。先report检查Q门槛，完成相应阶段决策，再freeze和evaluate：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 report --report-root research_reports/26-10-27_v2/reproduction
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 freeze --decision research_reports/26-10-27_v2/STAGE_DECISION.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_relation_basis_v2 --artifact-root artifacts/mdmt_relation_basis_v2/reproduce_20261027 evaluate --dataset movielens --seed 2022
```

evaluate对八个dataset/seed分别释放，导出六方法selected/final后worker自然退出。本次保存的STAGE_DECISION对应本次锁定配置与Q结果；若复现的Q事实改变阶段门槛，先记录其实际决策，再解锁E。

本次实际运行的ML诊断R1与KR诊断R2保留了各自身份。其共同训练源码证明在 SOURCE_REVISIONS/REVISION_EQUIVALENCE.json；RUNTIME_FIX2_REVIEW.md已独立核验。实际冻结命令为：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/freeze_compatible_revisions.py --artifact-root artifacts/mdmt_relation_basis_v2/revision1 --decision research_reports/26-10-27_v2/STAGE_DECISION.json --review-record research_reports/26-10-27_v2/RUNTIME_FIX2_REVIEW.md
```

该专用工具只接受本次两份白名单源码、完整八worker与48份真实身份；已完成目录的冻结文件采用独占写入，再次执行会拒绝覆盖。新复现实验不需要双版本接线。

本地 `artifacts/mdmt_relation_basis_v2/revision1` 保留真实配置、数据角色、逐行预测与日志；原失败、中止、启动失败与诊断重放保留原文件并单独计账。仓库交付包含代码、分析报告、聚合指标、配置和轻量运行收据；原始CSV及大型行数据缓存/逐行预测保留在本地。用户要求不保存模型权重，worker退出后需要按同配置重建才能再次运行模型前向。
