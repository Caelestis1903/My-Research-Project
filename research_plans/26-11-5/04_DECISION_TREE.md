# 04 结果驱动的有序决策与机器合同

所有规则逐数据集执行。输入必须是完整五seed、同协议、按V选点的结果。参数赢家按05选，调度默认只读取V与已经归档的研究证据；新test只用于冻结后报告。

## 1. 定义

F=原G-D0固定EMA；D/U/M/S*=各家族五seed验证选出的赢家。记v(X)=该候选五seed selected V Macro AUC均值；cell(X,Y)为六格V配对均差。保护阈值−.001；正式AUC资格+.003和LL均差<0独立报告。

`positive_increment = v(D*) > v(F)`。
`aggregation_tradeoff = any(cell(D*,F) < -.001)`：这里只说明改变影子估计及其合法选点带来新增代价，不声称已排除所有训练机制。
`direction_gain = min(v(D*)-v(S*), v(D*)-v(S_at_D_hK))`。
`multi_response_gain = min(v(D*)-v(M*), v(D*)-v(M_at_D_hK))`。
`simple_near = v(D*) - max(v(U*),v(M*),v(S*)) <= .0002`。

.0002是资源选型中“近似效果”的固定容忍度，不是新增性能合格门；任何正增量都按实际值报告。不同候选不能拿test最优值填入v。

`online_cost_supported`仅由已有同协议共同E50的G-D0 online与无增强online的六格对比或已归档明确观察填写，附source及比较对象。不得仅因G-D0输给另一架构就认定在线增强导致代价。没有相应材料就为unknown，不凭域规模补推断，不自动新开大型因果实验。

## 2. B之后的有序规则（第一条匹配即执行）

| 优先级 | 条件 | 下一步 |
|---:|---|---|
| 0 | 某必要候选少于5有效seed、HPO未冻结或源码/数据身份不合法 | WAIT_OR_REPAIR；不得科学否决 |
| 1 | 资源/数值问题使B不可运行 | COST_BLOCKED/ENGINEERING_BLOCKED；保留A/C，不能伪装B性能阴性 |
| 2 | positive_increment=false | RUN_C；B只改历史，C改变增强生成 |
| 3 | aggregation_tradeoff=true | RUN_A；先比较更简单路径历史配置；若A仍无可接受改进再RUN_C |
| 4 | D保留增量但尚有相关单格代价，且online_cost_supported=true | RUN_C；保留B已成立局部收益，不自动组合 |
| 5 | simple_near=true 或 direction_gain<=0 或 multi_response_gain<=0 | RUN_A；检验更便宜的分组/全局时间尺度是否足够 |
| 6 | 以上均不满足 | RETAIN_B_AND_REPORT；A/C为DEFERRED_NOT_RUN |

第4条的“尚有相关单格代价”必须写明相对F、无增强内部参考或冻结最强发表模型中的哪一个；默认使用相对无增强内部参考的同格差，且代价阈值−.001。无合法参考则该条不触发，按后续规则继续。已经借助历史test产生的研究动机在decision_basis.historical_test_exposure中透明记录。

方向/多响应均值优势只支持当前配置经验判断，不称完整机制因果证明。B成功不自动启动C，也不为了“证明没有更简单替代”无限追加实验；A是否值得运行已由上述资源比较决定。

## 3. A与C之后

A将A_group*、A_global*、A_fixed*、U*按同一validation规则排序。若简单候选与最优差≤.0002且更便宜，选简单候选，报告实际差。若选定A相对F有正增量且六格保护通过，RETAIN_A_OR_B_AND_REPORT；若没有，且C未运行，RUN_C。A、B的当前结果都保留，不因A更优删除B。

C完成四点搜索和两个匹配控制后，选已冻结C*或验证更优的简单控制作候选，正式报告所有关键对照。默认STOP_AND_REPORT，不按test反复调γ。没有增量仅停止当前定义的C，不否定增强历史联系的全部可能性。

每路线每数据默认最多执行一次完整矩阵；有故障的重放不算新科学配置。两数据结果不同则next_actions可以一个A、一个C，不能按“困难域”安排系数。

## 4. DECISION.json输出结构

必须输出真实数值，初始模板null不是0。必填字段：

```json
{
  "schema_version": 1,
  "study_id": "AUG_HISTORY_20261105",
  "route": "B",
  "dataset": "movielens",
  "status": "DECIDED",
  "manifest_hash": "runtime_sha256",
  "selection_hash": "runtime_sha256",
  "seed_ids": [2022,2023,2024,2025,2026],
  "all_required_candidates_complete": true,
  "winners": {"D": "resolved_id", "U": "resolved_id", "M": "resolved_id", "S": "resolved_id"},
  "validation": {
    "macro_auc_by_winner": {},
    "six_cell_auc_deltas": {},
    "direction_gain": null,
    "multi_response_gain": null,
    "simple_near": null,
    "aggregation_tradeoff": null,
    "online_cost_supported": "unknown",
    "online_cost_sources": []
  },
  "test_report": {"status": "NOT_YET_EVALUATED", "formal_gains": {}, "component_gains": {}, "six_cell_deltas": {}},
  "decision_basis": {"split": "validation", "test_consulted": false, "historical_test_exposure": true, "source_hashes": []},
  "cost": {"online_trajectories": 5, "estimator_seed_instances": 75, "gpu_seconds": null, "validation_seconds": null, "peak_gpu_bytes": null, "peak_ram_bytes": null},
  "matched_rule": "resolved_rule_id",
  "next_actions": [{"route": "A", "dataset": "movielens", "matrix": "A_FULL", "seeds": [2022,2023,2024,2025,2026]}],
  "deferred_routes": [],
  "reason": "filled_from_frozen_results"
}
```

运行器拒绝仍含必要null的DECIDED记录。测试报告后来追加时不得回写已冻结next_actions；确需人工调整须保留原记录，另建decision_revision并标test_consulted。

## 5. 人工继续合同

自动continue不可用时，运行报告给出已解析命令，例如：

```bash
python -m experiments.aug_history_20261105.run train --manifest research_plans/26-11-5/experiment_manifest.json --route A --datasets movielens --seeds 2022 2023 2024 2025 2026 --output artifacts/aug_history_20261105/attempt1 --resume
```

命令必须引用完整A/C规范和已有预算，不让下一代理重新探索算法。未触发路线只标未执行。
