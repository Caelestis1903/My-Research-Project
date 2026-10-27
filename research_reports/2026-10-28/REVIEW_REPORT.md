# 独立实现审查

**Verdict: PASS WITH FIXES。核心实现的阻断项已全部修复，允许进入固定P1训练。**

审查者：独立 reviewer 子agent，与两名coder分离。审查时间：2026-10-27 15:58 UTC。依据为 `DESIGN_SPECIFICATION.md`、原05技术合同、06阶段决策、08/09配置，以及 `CODER_MODEL_REPORT.md` 和 `CODER_PIPELINE_REPORT.md`。实际读取了新包模型、收益、关系、损失、权限、冲突、事务、数据、前缀、完整RAM状态、训练时钟、resident流程、导出和对应测试。

本次放行覆盖BASP-A核心预测与训练、常规F/V/Q导出、固定P1六臂，以及已实现的结构/开关接口。`ADDENDUM_READ_INTERVENTIONS.md` 的新增干预模块另行审查；核心worker在六臂训练完成后须等待独立 `INTERVENTION_REVIEW_RECEIPT.json`，再执行干预，随后才能写阶段READY。B/C及其他尚未实现的特殊候选被明确拒绝，不纳入本次通过范围。

## 关闭的阻断问题

1. **动态Adam分组与center学习率**：新增参数按稳定名字分类；新center保持0.0003；RAM恢复按快照的原分组、名字和元数据重建，保留旧m/v/step。
2. **结构S评价覆盖**：四臂统一使用冻结母状态的F定义覆盖；NEED使用真实收益/几何距离门及父责任，SOURCE排除owner域并使用foreign父读取质量。局部LL、自然质量、硬支持和ESS据此计算。
3. **FULL训练/评价坐标**：当前native实时产生h/z，评价保留冻结h0/z0，固定分区与phi参考坐标一致。
4. **结构提议证据**：source各target保留独立动作身份；风险轮廓按父质量加权；拒绝父冷却一个结构窗口。对比不再额外要求所有任务同时改变最佳动作。
5. **非对称source增长**：各target的动作宽度分别裁剪，新增坐标质量守恒，子观测数初始为0；source分裂后的CL teacher刷新已覆盖。
6. **容量对照**：计入实际foreign仿射门及新增中心坐标。初始完整A的SOURCE_SPLIT新增1229参数/1490相关MAC；WIDE rank10为1280参数/1280 MAC，匹配判断使用完整口径。
7. **诊断状态与可追溯性**：当前selected/final的F群体注册、Q student校准分别重新前向；上一teacher的预测器fold诊断明确标版本；原始收集器跳过身份元数据。未实现候选不能静默运行A。
8. **S消费的轨迹作用域**：同resident内各方法独立维护used/read集合；Trainer快照与恢复包含该集合；四候选完成后仍仅一次读取本窗S，再捕获赢家完整状态。

## 独立验收

指定项目Python实际执行：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_basp_v1/tests/test_model.py experiments/mdmt_basp_v1/tests/test_pipeline.py -q
28 passed in 1.86s
```

这些检查覆盖真实loss图的梯度白名单、SOURCE_WRITE关闭时foreign隔离而owner仍可训练、冷启动与全NULL影子训练、63680/63966实际计数、FP32/FP64非0.5分裂守恒、ESS/UNKNOWN/同域mask、完整支持源约束、source梯度草图对真实autograd、六开关、容量/事务判决，以及真实Trainer下一步的loss/参数/Adam/RNG/main/owner/probe/shadow游标重放。管线测试还检查新增参数组恢复、每trajectory单次S、自然尾批、F群体注册、Q导出及原始收集器。

已读取两数据集 `PASS_DIRECT_VALUES` 收据与实现：raw标签、域、编码值、T0词表/scaler、原有role数组均按chunk直接比较。MovieLens 1,000,209行，KuaiRand 2,507,666行；主训练保持原自然F人口。模型、teacher、优化器与selected/final均仅保留RAM，未加入磁盘训练状态保存路径。

当前核心阻断项：**0**。按预设阶段门运行；出现影响训练语义、指标或选择的新修改时审查对应变化。新增读取干预的独立验收尚待其实现完成，本收据不允许提前执行该模块。

本审查未启动研究训练、未分析AUC/LogLoss实验结论，也未追加合同外测试。
