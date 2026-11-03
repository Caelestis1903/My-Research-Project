# 四候选五 seed 补全入口

本包仅编排原训练实现。冻结矩阵为 `research_plans/26-11-03_v2/PHASE1_MATRIX.csv`：40条候选轨迹，10条复用、30条新训练；新增1500 epoch、51750次optimizer更新。全部新训练结束后冻结配置与选点，再统一评价180个新test端点。模型和optimizer状态由两个原生guardian保存在RAM，标量记录写入新attempt。

依赖项目环境 `phase0-3-env`，不新增包。必须同时具备以下原代码、计划与标量历史；仅下载本目录不能直接运行：

- `experiments/mdmt_history_information_20261103/`、`research_plans/26-11-03/experiment_queue.yaml`，源分支 `26-11-03_history_information`。
- `experiments/conditional_information_momentum_20261102_v3/` 及其原冻结计划，源分支 `26-11-02_v3`。
- 两者各自包内固定vendor依赖；原算法文件不得改动。新包prepare校验原source hash。
- `artifacts/mdmt_history_information_20261103/attempt1/` 和 `artifacts/conditional_information_momentum_20261102_v3/supplement_20261103/` 的原标量结果、test端点与逐epoch JSON。
- `artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720/` 的selected/E10/E50控制表、逐epoch表与冻结身份，源分支 `26-11-01_v3`。

从仓库根目录执行：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run prepare
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_five_seed_completion_20261103/tests -q
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run run --devices 0,1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run summarize
```

`run` 要求新报告目录中的 `INDEPENDENT_REVIEW_RECEIPT.json` 给出PASS或无blocking的PASS WITH FIXES，并绑定 `manifest` 命令返回的source_hash。每卡显存与功耗均低于85%且有2GiB可用显存、系统剩余RAM超过6GiB时追加进程；两个family共享队列，首批分配确认后再追加。原batch保持40960；OOM或worker意外终止只按原身份恢复一次，其他失败保留日志并退出供检查。已完成轨迹不重跑。

`summarize` 只读取已完成标量，输出至 `research_reports/2026-11-03_mdmt_five_seed_completion/`。`endpoints.csv`平铺 `dataset,arm,seed,model_kind,checkpoint,split,epoch,model_hash` 和六格/域/任务/macro/traffic/pooled指标，另含来源路径、哈希与复用标志；`paired_seed.csv` 保留原值及配对差，`paired_summary.csv` 和 `absolute_summary.csv` 给seed均值、样本SD和描述性t95%区间；`validation_epochs.csv`、`costs.csv`、`RUN_MANIFEST.json`、`DECISIONS.json`保留阶段、成本和裁定。

裁定只使用test fixedEMA selected五seed均值：总收益对A0 online，组件增量对A0 fixedEMA；宏AUC≥.003、六格各≥−.001，LL均差<0单独判断。方向数与CI不参与淘汰。第二阶段正式发表模型协议见计划目录，当前入口不派发该矩阵。
