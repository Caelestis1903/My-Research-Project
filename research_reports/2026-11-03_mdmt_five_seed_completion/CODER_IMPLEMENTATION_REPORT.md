# CODER IMPLEMENTATION REPORT

实现完成，未启动正式训练，未分析科研结果。规格：[DESIGN_SPECIFICATION.md](../../research_plans/26-11-03_v2/DESIGN_SPECIFICATION.md)。新增代码：[run.py](../../experiments/mdmt_five_seed_completion_20261103/run.py)、[最小测试](../../experiments/mdmt_five_seed_completion_20261103/tests/test_completion.py)、[运行说明](../../experiments/mdmt_five_seed_completion_20261103/README.md)。历史算法包、vendor与历史报告未修改。

审查绑定source_hash：`db8d3385f3acabb11e2f10207e6adafb88cc949b51b7b9553a5117434366b7d6`。该hash同时覆盖新Python源码、设计/矩阵/复用审计和两原生算法source hash；完整文件表为 [IMPLEMENTATION_SOURCE_MANIFEST.json](IMPLEMENTATION_SOURCE_MANIFEST.json)。history/pair哈希分别保持 `7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685` / `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`。

`prepare` 已建立 `artifacts/mdmt_five_seed_completion_20261103/attempt1/PREPARED.json`：40逻辑身份、10条历史结果checksum与原生配置匹配、30个原生RUN job；未启动guardian或CUDA训练。原生worker保留40960 batch、50epoch、优化器/EMA/RNG/数据读取与完整逐epoch标量记录。两个family共用逐卡85%资源门；OOM/worker意外终止保留原身份及RAM epoch恢复一次，不增加seed。30条全部完成后才建立全局冻结及各原评价器所需冻结文件，随后评价180个新test端点并释放RAM。其他worker异常保留原日志并退出供处理。

统一端点表为扁平schema：`dataset,arm,seed,key,configuration,model_kind,checkpoint,split,epoch,model_hash,historical_reference,source_result,source_result_sha256,d0t0_auc,d0t0_logloss,...,macro_auc,macro_logloss,sample_weighted_cell_auc,sample_weighted_cell_logloss,pooled_task_mean_auc,pooled_logloss,...`。原六格/域/任务指标均平铺。最终660行含四候选480个validation/test端点与控制/Gaussian180个历史端点；候选test为240行，其中180新增、60复用。逐seed配对保留候选/对照原值、epoch/hash及差值；均值/样本SD/t95%区间按seed计算。selected fixedEMA的总收益和组件增量各自裁定，LL独立；未满五seed返回INCOMPLETE，无seed方向数或CI门。全部validation epoch、训练loss、成本和失败记录另表保留。

最小检查：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_five_seed_completion_20261103/tests -q` → **3 passed in 4.70s**。检查矩阵与原生配置、10条历史端点/60个冻结身份、控制去重、prepare可重复、五seed/阈值等号、LL独立、负seed及跨零CI不额外淘汰。新增训练预算为1500epoch、51750次optimizer更新；历史控制与复用成本均不计新增。

reviewer需要在本目录写入 `INDEPENDENT_REVIEW_RECEIPT.json`：`verdict` 为PASS或PASS WITH FIXES、`blocking_issues`为空、`source_hash`为上述hash。之后runner从仓库根目录执行：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run run --devices 0,1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run summarize
```

`run`完成前不运行summarize；`progress.json`、`launch_ledger.jsonl`及各family原日志提供机械运行进度。运行中不更改已审源码/设计。源码的原分支依赖及标量恢复输入见README，由主agent随发布安排。原算法现有CPU/S0证据继续使用，未重复GPU机制验证或写入任何模型/optimizer/checkpoint文件。
