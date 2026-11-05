# C4 Codex独立执行入口

先读上级07/09和C1–C3。C可在B未取得增量时独立执行，不需要重新设计。固定EMA反馈到增强是本路线定义，不得当成“不小心破坏了B非反馈规则”而删除。

## 实施步骤

1. 准备固定source及G-D0父协议，核对目标域d0和字段列表。
2. 新adapter返回raw logits，提供只读完整输入导数模式；训练路径逐值对齐源forward，保持gate.detach。
3. 实现128行训练probe、u/A、稳定有符号矩阵指数、共享ξ的两个任务视图和原2N替换loss。
4. 实现γ0精确退化、MERGED和ROTATED控制，完成teacher/RNG隔离与恢复测试。
5. 工程smoke测资源，不以性能选择γ/K。
6. 运行C1/C2两数据五seed；冻结每数据validation决定后运行C3五seed，再C4五seed。四候选完成后冻结C*，运行两个匹配控制各五seed。
7. 父Gaussian对照身份合法则复用；否则补齐五seed。所有配置test访问在搜索和选点冻结之后。
8. 输出HPO过程、每格及总体指标、完整/组件比较、成本和研究决策，记录已外露历史test的边界。

## 命令合同

```bash
PLAN=research_plans/26-11-5/experiment_manifest.json
RUN=artifacts/aug_history_20261105/attempt1
python -m experiments.aug_history_20261105.run validate --manifest "$PLAN" --suite C --output "$RUN"
python -m experiments.aug_history_20261105.run train --manifest "$PLAN" --route C --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output "$RUN" --resume
python -m experiments.aug_history_20261105.run select --manifest "$PLAN" --route C --split validation --output "$RUN"
python -m experiments.aug_history_20261105.run evaluate --manifest "$PLAN" --route C --frozen "$RUN/C/selection_frozen.json" --output "$RUN"
python -m experiments.aug_history_20261105.run report --manifest "$PLAN" --output "$RUN"
```

train子命令负责内部四点依赖调度和每一步五seed聚合；select对最终选择和完整性再次审计。不是一次性把未解析的C3/C4参数传给训练器。启动后未知参数只能由manifest规定的引用解析器计算，不能让Codex临时猜。

## 完成与异常

数学/数值/损失/RNG/恢复测试PASS；每已调度数据30条新轨迹完整，父控制合法；各候选五seed、HPO选择和test冻结hash一致；报告online与EMA但不把两者当独立重复。某阶段工程阻塞只中止其依赖，不将未运行的C3/C4标性能失败。源码、参数和统计不落盘；RAM丢失按原seed重放。
