# A4 Codex独立执行入口

本文件定义实现和运行任务，不是已经存在的脚本。先读上级07、09与本目录A1–A3。无需重新设计路线，不需要B运行成功；独立运行也使用同一父Gaussian配置。

## 实施顺序

1. 按source index准备固定源码、最新计划、数据身份和运行环境。
2. 实现坐标GroupRegistry，输出`GROUP_MAP.json`（参数名、形状、坐标范围、组、来源格、hash），完成互斥/覆盖与域表lookup检查。
3. 实现固定分组EMA；F退化和online一致性通过后实现virtual_parameters、raw logits、训练排序probe及有界q更新。
4. 实现全局自适应和全部15影子配置，禁止只实现四个分组候选而漏掉简单替代。
5. 跑公共及A工程测试；少量batch smoke仅判资源/数值，日志标ENGINEERING_ONLY。
6. 派发两数据各五seed完整50epoch；候选在线轨迹共享，但状态和每epochV评价独立。保存全部候选RAM selected/E10/E50，直到五seed选型结束。
7. 计算五seed HPO，冻结选择；test入口解除后评价约定端点，生成报告。
8. 依总决策更新NEXT_ACTION；只回填已完成真实结果，未运行路线保留DEFERRED_NOT_RUN。

## 命令合同

```bash
PLAN=research_plans/26-11-5/experiment_manifest.json
RUN=artifacts/aug_history_20261105/attempt1
python -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --suite A --output "$RUN"
python -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route A --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output "$RUN" --resume
python -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route A --split validation --output "$RUN"
python -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route A --frozen "$RUN/A/selection_frozen.json" --output "$RUN"
python -m experiments.aug_history_20261105.run report --manifest "$PLAN" --output "$RUN"
```

若决策只触发一个数据集，datasets只列该数据集，但仍5seed；不能把另一数据集NOT_RUN写失败。

## 完成标准

A1–A3所有接口有实现；公共及A测试PASS；10条（或被选择单数据5条）在线轨迹完整；每候选五seed齐全；HPO与冻结记录hash一致；六格/总体AUC及LL、成本齐全；未污染Adam/RNG；RAM恢复或重放通过；生成`A/DECISION.json`及报告。控制器数值错误或资源不足分别工程/成本阻塞，不以低AUC替代错误解释。
