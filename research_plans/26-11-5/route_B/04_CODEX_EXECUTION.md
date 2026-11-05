# B4 Codex首轮执行入口

先读上级07/09与B1–B3。实现目标是保持在线训练完全不变，添加只读反事实响应和15影子估计器。无需再次征求研究方向确认；数学/候选值已冻结。

## 实现与运行顺序

1. 准备固定源提交、数据身份和计划；显式覆盖KR target_domain=0，禁止旧A1默认d2混入。
2. 从source train_step拆出loss_terms，不改原total计算顺序。实现纯反事实Adam及clone-Adam真值测试。
3. 实现列规范化、FP64 Gram、Q、M和S方向控制，完成空响应/秩/时间时钟检查。
4. 实现完整15影子bank和每影子独立validation选择器；选中快照与E10/E50留RAM。
5. 执行公共+B测试和资源smoke，检查online hash不被新增组件改变。
6. 同时冻结两数据五seed队列，完整训练50epoch；不读取test，不用当前排名裁撤候选。
7. 汇总五seed validation选超参数，写HPO_SELECTION和selection_frozen；生成validation决策并锁定next_action。
8. 统一评价冻结的test端点、输出总收益与组件增量、成本和六格。
9. 调用continue执行已完整定义的A/C或停止，依据预定义决策，不另开无边界设计任务。

## 命令合同

```bash
PLAN=research_plans/26-11-5/experiment_manifest.json
RUN=artifacts/aug_history_20261105/attempt1
python -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --suite B --output "$RUN"
python -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route B --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output "$RUN" --resume
python -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route B --split validation --output "$RUN"
python -m experiments.aug_history_20261105.run decide --manifest "$PLAN" --route B --basis validation --output "$RUN"
python -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route B --frozen "$RUN/B/selection_frozen.json" --output "$RUN"
python -m experiments.aug_history_20261105.run continue --manifest "$PLAN" --decision "$RUN/B/DECISION.json" --output "$RUN" --resume
```

新module尚未实现，以上是实现后必须支持的命令，不是当前可直接执行的既有脚本。

## 验收

数学与纯性测试PASS；10条online完整；每数据每候选五seed齐；150个影子实例有完整50epoch V曲线；所有test端点身份冻结；源online与对照一致；输出decision/report/cost；未运行A/C明确NOT_RUN或DEFERRED而非失败。因RAM丢失重放时保持原seed、数据/噪声顺序及候选集；不把重放算新增seed。
