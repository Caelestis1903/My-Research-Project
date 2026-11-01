# GATS02 恢复实现报告 R3：native -11重试与物理成本

依据已批准 [RECOVERY_DESIGN_ADDENDUM_R3.md](RECOVERY_DESIGN_ADDENDUM_R3.md) 实现。修改限于本轮 `run.py` 调度/worker诊断、`recovery.py` 多来源成本/重试额度、对应CPU测试及README。`objectives.py`、`shadows.py`和历史包未修改；科研配置、随机协议、模型、数据、batch40960、50epoch、FP32及候选门保持原样。未运行研究训练或新CUDA Stage0。

## 已观测故障的处理

无完整RAM结果的Pipe EOF或已退出worker，现在先join/reap，再用稳定exitcode分类。仅确认 `exitcode=-11` 的此类事件新增为recoverable；保存exitcode、signal=11、job/GPU/attempt/time、run_id、failure_count、retry_index与先前native失败数。Python assertion及非-11 native退出保持不可恢复，原OOM/OSError分类沿用。

每job仍为初始执行加最多两次完整retry。额度从显式旧成本来源的原始failure/trial ledger读取，按dataset/seed/完整config/数据/RNG身份匹配；0315的KuaiRand B2 seed2022已有一次失败，其新首次派发为attempt2、retry_index1；若再次同类失败，只剩attempt3。其他因RAM丢失重建的job从各自额度开始。

native重试保持相同job配置并从初始化重算完整primary+replay，使用新的attempt目录。serial待派时等待已有active完成，运行期间阻止一切本队列新派发，结束后恢复正常资源调度。原始hard deadline同样阻止pending retry。现有外部GPU任务继续保留。

worker打开自己的stderr后、训练前启用 `faulthandler.enable(file=stderr, all_threads=True)`，文件生命周期内保持；正常Python退出时清理。该修订提供native诊断和有界重试，SIGSEGV根因仍未定位。

## 显式来源与完整计账

身份参考继续为完整20轨迹的0005。CLI新增可重复 `--additional-cost-attempt`，第三轮同时指定0315。所有来源按resolve后的绝对路径去重，只读取各自原始run/pass文件，不把0315 receipt中继承的0005成本再相加。

每run目录有合并runtime_stats时，仅计该合并记录；其primary/replay子记录由它覆盖。没有合并记录时，分别计已完成pass和未完成pass。未完成pass按已提交batch/epoch/view/validation日志提供可测数值与下界；缺GPU event、shadow更新等字段记N/A，不按E42比例外推。新attempt的失败retry也通过同一原始目录枚举进入最终计账，成功results的costs.csv只保留其原语义。

实际CPU来源核算如下，完整pass包括失败job已完成的primary：

| 来源 | 完整job | 完整pass | partial pass | worker秒 |
|---|---:|---:|---:|---:|
| attempt_20261102_0005 | 20 | 36 | 0 | 21367.9183922371 |
| attempt_20261102_recovery_0315 | 11 | 19 | 1 | ≥9862.585071863257 |
| 合计 | 科学轨迹按身份去重 | 55 | 1 | ≥31230.50346410036 |

0315失败B2的primary为完整50epoch，replay已提交42个完整epoch及2086个训练step；replay worker秒下界658.5821999190957，连同primary为1906.0140764860553。partial replay的optimizer更新2086、梯度遍历10430、样本曝光85404424来自已提交trace；training GPU event和shadow更新数为N/A。最后未提交工作和未完成epoch时间不纳入精确值。

原始forecast 17870.690975639038及起点1793549242.1426222继续使用；10h/12h deadline仍为1793585242.1426222/1793592442.1426222。全部已完成成本和partial worker秒下界进入原预设超成本取消顺序。科学唯一轨迹预算、附加8条与确认新增12条上限不变。

## 产物接口

- 启动：`prior_physical_cost_ledger.csv`、`prior_physical_cost_sources.json`，逐原始来源记录完整/partial项；`recovery_execution_receipt.json`记录旧失败quota、原始时钟、下界字段和源列表。
- 完成或协调器失败：`physical_cost_ledger.csv`、`physical_cost_sources.json`覆盖所有旧来源和当前每个attempt目录；`execution_cost_continuity.json`给旧/当前/合计的known totals或标明的lower bounds、N/A组成、完整/partial pass数和科学去重数。
- `failure_ledger.csv`保留每次失败及额度；`launch_ledger.csv`记录serial_retry、retry_index与prior_native_failures。失败未被成功重试覆盖。

## 实际必要验证

`python -m pytest -q experiments/gradient_aug_time_select_20261102/tests --junitxml=research_reports/2026-11-02_gradient_aug_time_select/R3_CPU_TEST_RESULTS.xml`：**12 passed，1.73s**。

其中CPU事件注入直接执行实际schedule：EOF/-11及已退出/-11在join后分类；同配置retry成功；首次失败后最多attempt1/2/3；带入一次历史失败后仅attempt2/3；历史失败job首次fresh独占、其它job额度不扣；serial等待已有active且运行时无新job；assertion与-15不retry；原deadline取消pending retry。没有启动native stress或科研训练。

真实成本回归读取0005和0315，并重复传入相同路径，确认仅两来源、36+19完整pass与1 partial、旧receipt不递归相加；确认实际失败job key对应原KuaiRand/B2/2022且累计native失败为1。原CPU公式、48端点历史接口与确认12/总40回归继续通过。

最终真实CPU preflight目录为 `recovery_r3_cpu_preflight_final/`：两数据集verified=true、48/48端点一致；其中 `prior_physical_cost_sources.json`和CSV保留上表逐项来源及下界/N/A。预检未读取test或权重。

运行参数在原R2恢复命令上增加：

```text
--additional-cost-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315
```

identity仍传0005，task-start仍传1793549242.1426222。独立review通过并冻结新source/commit后，须用该source重新完成两数据集CUDA Stage0，再派发新空attempt。
