# 固定读取干预与reader续训独立复审

**Verdict: PASS WITH FIXES。blocking_open=false。允许执行已批准附录的有限诊断。**

独立核对 `ADDENDUM_READ_INTERVENTIONS.md`、`interventions.py`、`test_interventions.py` 和 `CODER_INTERVENTIONS_REPORT.md`，并检查campaign在核心矩阵完成后等待独立收据的调用点。该模块不修改主模型训练文件或原selected/final状态。

实际实现符合附录：全部有效路径的固定OFF/ROOT、ALL_NULL、FIXED关系使用同一状态缓存；ROOT明确换为冻结pilot phi；F_probe以完整自然F域比例排序严格正OFF依赖，稳定并列选择一个路径。计划先写入，随后三分支从同一完整RAM快照及原Adam历史开始，共用下一游标八个自然F batch。只更新原有reader w，不使用shadow或其他辅助损失；q、alpha、G、source、目标/需求参数和native保持冻结。OFF分支直接移除该路径的图，避免被删路径以零梯度推进独占reader。

KEEP/OFF/ROOT三分支全部八步完成并锁定后，模块一次读取Q诊断标签，统一评分固定动作和三条续训分支。相对KEEP_READAPT报告OFF/ROOT风险差。母Trainer不被load或advance，结束时恢复调用前RNG并逐项比较完整RAM状态；相同selected/final训练状态可复用一次诊断。预测、责任、路径分量、行计划和成本落盘，模型/Adam仅留RAM。

审查发现的F_probe指标口径已修复：等域面板的GLOBAL natural_logloss及其变化按完整自然F域比例加权，并保留probe_mean_logloss及明确权重标签；Q继续使用真实Q请求均值。路径选择和分支训练未因此改变。

独立执行：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_basp_v1/tests/test_interventions.py -q
3 passed in 2.04s
```

测试覆盖OFF质量归NULL、ROOT在八步期间持续生效、自然F权重与并列选择、三个原Adam起点及同八批计划、reader梯度白名单、非reader参数与Adam不推进、锁定后单次Q访问、母完整状态一致、selected/final复用及无磁盘模型文件。

阻断项：**0**。核心训练沿用 `REVIEW_REPORT.md` 与 `RUNTIME_FIX1_REVIEW.md`；本次仅放行附录诊断。reviewer未执行自然数据诊断，未分析实验结论。
