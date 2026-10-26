# 实际 300 epoch 预算判定与状态 seed 2025 最终预算

全部十组 V5/V8 seed 2025 实际训练已完成，主 agent 核验 60 项关键物理哈希，归档 70 份 compact 文件；整组状态由五个真实队列无损派生，整体模板没有作为训练队列启动。

|条件|q|近期相对目标下降|操作性平台|
|---|---|---:|---|
|clean|SN|0.0025561014|True|
|clean|SB|-0.0014572709|True|
|PERM|SN|-0.0043623093|True|
|PERM|SB|-0.0003604377|True|
|ORDER|SN|0.0038636946|True|
|ORDER|SB|-0.0020594665|True|
|mild|SN|-0.0093164171|True|
|mild|SB|-0.0080745726|True|
|strong|SN|0.0077350067|False|
|strong|SB|0.0033980411|True|

唯一不满足平台的 strong SN，gain=0.007735006730925001 > 0.005。精确选择集确认重新计算全部 300 个物理预测文件、1,800 个域任务单元，按冻结算法对需要的 8 个 epoch 使用 1,000 次同权重 bootstrap，8,000 条原始结果已归档。候选窗口与上界如下：

|起始 epoch|参照 peak|三个 AUC 差异的 97.5% 分位数|通过|
|---:|---:|---|---|
|91|90|[-8.015493112194602e-05, 0.0009168300486965508, 0.0034641695052683016]|False|
|294|219|[0.007711247900028455, 0.003522320702103047, -0.0013236622777686529]|False|

因此按冻结规则，全部十组匹配条件必须继续到 **MIN350**；不能把其余九组的平台视为整组最终停止。该确认只用于预算，没有计算初次显著退化、也不是全轨迹家族错误率保证。当前结果不支持在 300 epoch 判定最终机制或提前结束任务。

保留完整 300 epoch 五对读出，并保留全部较早阶段；从最终父读出导出的前缀义务现为 22 对（旧 17 对加 5 对 300 epoch）。350 epoch 不是硬上限。V4 自身原始最终 100 epoch 不被覆盖。

## 状态 seed 2025

六组状态干预实际训练均完成至 200 epoch，主 agent 核验 36 项关键哈希及 42 份归档。

|状态/q|近期相对目标下降|
|---|---:|
|V11_beta1_zero_v_only__kuairand__adam__SN__seed2025__continuation200|0.0010631161|
|V11_beta1_zero_v_only__kuairand__adam__SB__seed2025__continuation200|-0.0022561357|
|V11_beta2_zero__kuairand__adam__SN__seed2025__continuation200|0.0001018550|
|V11_beta2_zero__kuairand__adam__SB__seed2025__continuation200|0.0001296499|
|V11_m_only__kuairand__adam__SN__seed2025__continuation200|-0.0004453792|
|V11_m_only__kuairand__adam__SB__seed2025__continuation200|0.0017904736|

六组均满足 gain ≤0.005，因此 **FINAL200**。这不是性能健康、稳定最优或状态机制有效的证据。三对完整 1000/200 读出、九对 50/100/150 前缀、四 seed 比较及最终测试仍为必需工作。native seed 2025 自身最终 50 与这些状态最终 200 的比较必须标明预算不匹配；机制比较保留共同固定阶段。

## 可追溯产物

- `V5V8_seed2025_all10_actual300_primary_inputs_and_budget_check.json`
- `V5V8_seed2025_all10_actual300_archive_primary_verification.json`
- `V5V8_seed2025_strong_SN_selection_budget_until300_primary_verification.json`
- `V5V8_seed2025_extension350_required_design.json`
- `V5V8_seed2025_extension350_implementation_self_review.json`
- `V11_state_seed2025_actual200_final_budget_certificate.json`

尚未读取新 v3 测试性能；没有 `test_readout_frozen.json`。历史测试暴露的边界维持原记录，不能称为从未访问过的 holdout。
