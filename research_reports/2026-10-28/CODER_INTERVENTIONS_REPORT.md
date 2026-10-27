# CODER IMPLEMENTATION REPORT — 固定读取干预与reader再适配

依据：主agent批准的`ADDENDUM_READ_INTERVENTIONS.md`。新增实现只在`interventions.py`、`tests/test_interventions.py`；未修改trainer或campaign。

`run_read_interventions(trainer,which)`支持V-selected和epoch20-final。每个实际foreign路径按稳定(target,source,leaf ID)完整测量固定OFF/ROOT；同时测KEEP、ALL_NULL和质量守恒FIXED关系。ROOT明确使用冻结标准化pilot phi，包含动态树时移除整条叶路径的全部可训练source残差。缓存原q/alpha/NULL/目标reference/函数值，不重测关系或重分配权重。

只用F_probe的固定OFF损害、完整自然F域比例和两个任务valid标签均值选择严格正收益最大路径；并列用稳定路径顺序。无正依赖记录`NO_POSITIVE_FIXED_DEPENDENCE`，不创建短训。选择计划先落盘。

被选路径从同一完整RAM快照创建KEEP/OFF/ROOT三分支，继承原reader Adam状态。三者共同消费快照主sampler下一游标的八个自然F batch；只更新原有全部reader w，无shadow或辅助loss，保持原w半径2。通过直接构造保留路径的图，删除路径不借零梯度推进独占reader。source/owner/target/gain/native均冻结，q/alpha/teacher/G保持原状态。检查非reader参数及其Adam状态均未变化；三分支的完整原Adam初态直接相等。

三分支八步全部完成并锁定后，才读取一次本状态Q标签，统一评分全部固定动作与三分支。分支模型和Adam在评价结束前保留RAM，随后释放。Q不决定路径、步数或赢家。

输出逐F_probe/Q、逐动作、全局macro/natural与cell、原责任加权need-leaf风险/AUC及相对KEEP差。OFF/ROOT再适配统一以KEEP_READAPT为参照。`F_probe_COMPONENTS.npz`和`Q_COMPONENTS.npz`保存行ID、标签、mask、原q、reference/KEEP、path系数和leaf/root修正，可复算全部固定动作；另存三分支Q预测。原始模型、权重、优化器均不落盘。

独立review修订：F_probe为每域固定抽样面板，其GLOBAL natural_logloss及相对KEEP差按完整自然F域比例加权各域valid LL；另保留probe_mean_logloss并明确FULL_F_DOMAIN_WEIGHTED_PROBE_ESTIMATE口径。Q继续采用真实Q请求均值。现有自然域比例测试同时核对两种口径；路径排序和三分支训练均未改变。

每次诊断保存母Trainer完整snapshot前后直接相等收据，覆盖模型/buffer、Adam、全局RNG、全部sampler/owner/probe游标、teacher/G及版本。函数不load或advance母Trainer，只操作独立副本；调用前RNG在finally恢复。selected/final训练状态完全相同（扣除纯wall-clock账）时复用一次测量并保留两个状态标签。

输出目录：`trainer.root/interventions/{selected,final}/`。包括INTERVENTION_PLAN、FUNCTION_INTERVENTIONS、组成数据、三分支预测、STATE_PARITY和独立COST_LEDGER；方法根目录FUNCTION_INTERVENTIONS.csv累计。成本分别记录fixed component forward、H8准备、三分支训练、评价、parity和总耗时，以及实际/尝试reader更新数及诊断过程GPU峰值。

最小验收：**3 passed**，覆盖手算OFF质量归NULL、ROOT始终为phi、八步期间干预持续、自然F权重与稳定并列选择、三分支原Adam初态/同八batch计划、仅reader梯度、非reader参数与Adam不推进、Q在分支锁定后一次访问、完整母RAM无污染、selected/final相同状态复用、无磁盘模型文件。新文件语法检查通过。

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_basp_v1/tests/test_interventions.py -q
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m py_compile experiments/mdmt_basp_v1/interventions.py
```

未运行自然数据干预或解释研究结果；待独立review放行后由pipeline统一调用。本实现不增加完整适配轨迹、seed或选参预算。
