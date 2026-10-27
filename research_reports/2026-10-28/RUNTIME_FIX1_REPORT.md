# Runtime Fix 1 — 结构诊断重复epoch导致运行终止

故障类型：Critical。首个MovieLens/2022 worker在进入非TARGET方法第4轮结构诊断时抛出 `TypeError: dict() got multiple values for keyword argument 'epoch'`；另外三个同任务worker随后由runner终止并保留日志。

## 原因与修复

`conflict.propose`正常返回的诊断字典本身包含 `epoch`，`Trainer.structure_window`再次调用 `dict(epoch=epoch, **diag)` 会在构造字典时冲突。TARGET方法跳过结构诊断，已有单batch/状态测试也未经过此记录语句，因此此前没有触发。

唯一生产代码修改：`trainer.py` 的记录改为 `{**diag, 'epoch': epoch}`。结构窗口调用的epoch作为记录值，既处理正常诊断已有epoch，也处理NO_PROBE等早返回没有epoch的情况。batch、20轮训练预算、loss、参数更新、结构阈值与选点规则均不变。

已用rg检查同一训练/报告调用链的同类字典展开，并核对实际生产者字段：梯度诊断不包含epoch/step；source投影记录不包含epoch/step；事务decision不包含epoch；fold诊断不包含teacher_version；指标/成本收集器的身份字段与度量字段分离。未发现同路径另一处实际重复键，因此没有批量重写这些代码。

## 验证

- 新增 `test_structure_window_records_proposal_diagnostic_epoch`：使用真实非TARGET BASP模型，调用实际 `Trainer.structure_window(4)`；将propose返回设为包含epoch的正常诊断，检查诊断、previous_diagnostic和提议记录完整产生。旧写法在此用例会抛出同一TypeError。
- `python -m pytest experiments/mdmt_basp_v1/tests/test_pipeline.py -q`：10项通过。
- 原完整验收 `python -m pytest experiments/mdmt_basp_v1/tests -q`：29项通过，耗时1.80秒。

本修复未启动重训。失败尝试日志保留；独立增量审查放行后由runner重新启动。原worker已终止，RAM中的模型/Adam/selected/final状态需要随共同前缀重建。
