# INDEPENDENT REVIEW REPORT A

Verdict：**PASS**。Blocking issues：**无**。

审查对象为 `DESIGN_SPECIFICATION_A.md`、`CODER_IMPLEMENTATION_REPORT_A.md`、四份冻结 A 规范、共同合同及实际 A 新增/修改源码。计划提交 `2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，骨干提交 `c8675ae2923c15b6965884b140b5b646690cd95e`；B 首次 FAIL、修订和既有 acceptance 记录保持原样。

## 实际实现核对

| 范围 | 独立核对结果 |
|---|---|
| 完整坐标与路径 | `GroupRegistry` 复用去重布局，完整互斥覆盖 S、D0/1/2、T0/1、六个 DT；普通特征表归 S，domain_indicator 按动态列对应表的 lookup 行归 Dd；gate/tower 按 `index%3,index//3` 映射，即 `3*t+d`。两数据冻结坐标数量有运行断言，未知参数拒绝 |
| 固定与自适应 EMA | F/U、四固定 profile、四 GROUP、四 GLOBAL 共15个独立 φ，从 θ0 初始化。q0 使用源原地 mul/add 次序；固定 profile 的半衰期映射正确；GROUP q12维、GLOBAL q1维 |
| q 代理与梯度边界 | 代理系数为 `rho0**(2**(-q)*K)`；旧 φ 和 Adam 后 θ 均 detach，只有 q 可微。functional raw logits 保留训练 gate detach；只调用 `autograd.grad(loss,q)`。无穷范数归一化、[-1,1]、空/零梯度保持及非有限拒绝符合规格 |
| 训练探针 | 只读 train_x/y/raw IDs，排除当前 batch raw IDs；每格每类最多16、无放回；缺类跳过；局部 SHA256-key Generator，不消费全局 RNG。同一步待刷新的所有候选共用抽样，各自保留 φ/q |
| 正式时序与隔离 | A 分支执行源 loss→一次 backward→一次 Adam，再在 K20/40 整数倍更新 q、各候选一次 EMA。跳过 B 的辅助梯度/Q。源模型无 BN/Dropout；探针不改 mode、参数、buffers、在线 .grad 或 Adam；临时 logits hooks 按既有 finally 清理 |
| 状态与恢复 | A 快照保留 route/job/source/manifest/data/layout/group 身份，全部 φ/q/η/K、刷新时钟、probe 身份、成本及既有 online/Adam/loader/RNG/选择器端点。恢复核对 A 路线和分组身份，复用已修正的 guardian owner/重试/epoch 事务 |
| 完整训练与选择 | 保留两数据各五 seed、每候选50次 validation 与 RAM selected/E10/E50。A_FIXED 集合包含 F，F 只维护一次；家族独立选型、保护规则、静态偏离与完整五seed门保留；GROUP 赢家的同 η/K GLOBAL 进入冻结端点 |
| A 决策 | 对家族赢家按 validation、保护和成本选型；≤.0002 且更便宜的简单版本可优先。选中版本对 F 有正增量且保护通过则保留，否则 RUN_C；test 不进入决策 |
| 跨阶段身份 | A 使用新 attempt/source/review 身份；continue 按旧 B 自己的 PREPARED 与冻结文件校验，写收据到新输出，不要求当前源码等于旧 B 源码，不回写 B |
| CLI/评价/资源 | 路线传至 private worker、guardian、选择/评价/报告/release；A 使用独立 acceptance gate。冻结后才评价 test；参数/代理/优化器状态只经 RAM/BytesIO/IPC；逐卡显存和功耗85%、系统 RAM90% 门保留 |

## 验证证据

复用 coder 对最终实现的有效结果：A 全套 `8 passed, 1 warning in 161.04s`，包含两数据路径、41步源 online/.grad 逐位一致、K20/K40 及 q/φ/RAM 恢复；共享/B `11 passed, 2 warnings in 16.57s`，CLI help exit0。未重复运行这两套完整检查。

Reviewer 独立执行以下定向检查：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/aug_history_20261105/tests/test_a.py -q -k 'fixed_profile_q0 or probe_train or old_b_continuation'
3 passed, 5 deselected in 4.68s
```

另用临时 `ENGINEERING_ONLY` 标量夹具调用真实 `selection.select` 与 `report.decide_A`，结果 PASS：覆盖两数据15候选×五seed、固定家族中的 F、独立 GROUP/GLOBAL 赢家、不同于 GLOBAL 赢家的匹配 η/K 控制、近似且更便宜的固定方案选择、无受保护正增量时 RUN_C、validation-only 决策。夹具不产生训练性能结论。

用真实 B attempt 的 PREPARED、selection_frozen、DECISION 执行只读继续检查，结果 PASS；冻结 selection hash 为 `fce00de767c5e95b0cb627f53eb5851d43a6c1e64a8c4786db73e57c0ff202a4`，允许两数据集 A_FULL；三个源文件检查前后 SHA-256 均不变。

## Acceptance 与 RUN

- 当前 source hash：`afecc8dcef7f1687d61383af0ca69176eb7b7ec26d0700eb43836a827b5e2ff9`。
- manifest SHA-256：`76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`。
- `REVIEW_ACCEPTANCE_A.json` 绑定上述身份；C 保持 DESIGN_COMPLETE_NOT_IMPLEMENTED。

可以进入 A 的 RUN 阶段：记录 implementation commit、新 attempt preflight、A 工程 gate，以及两数据 batch40960、实际触发 K20 q 探针的 GPU smoke。运行门通过后执行完整矩阵。GPU smoke、正式 A 训练和研究分析尚未执行。

成本字段中 `sampling_seconds` 是 `probe_seconds` 的子项；合计时避免重复相加。逐候选 controller/EMA/validation 成本与共享抽样成本分别保留。
