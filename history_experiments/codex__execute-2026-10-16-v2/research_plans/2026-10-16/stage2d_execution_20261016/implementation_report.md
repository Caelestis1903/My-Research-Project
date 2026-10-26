# IMPLEMENTATION REPORT — Stage2D

实现仅新增experiments/stage2d_20261016.py与stage2d_queue_20261016.py，未修改旧训练源码或历史结果。

worker复用DataBundle、StageS1M3OEAdapter、原cell指标与预测收集函数；独立loop实现四臂alpha/q。两种q均用相同uniform域/域内映射、有放回抽样；SN/EN与SB/EB的同seed batch配对可逐hash核验。model/schema/init、50epochs、49main+1arch/epoch和Adam超参数固定。

真实参数分区C/P/A经旧审计函数和参数总量断言；main/arch互斥。每epoch保存完整resume state（model/两optimizer/RNG/两sampler位置/unique exposure/step），selected/common/final完整state；训练结束写selection_frozen后才读test。原始预测含row IDs/labels/domains，test仅forensic。记录loss/曝光/unique rows、每epoch起末main与arch的gradient/m/v/applied delta范数；完整m/v与bias correction所需step保留在state。

按原计划需要精确共同更新/计算预算，不用wall-clock早停；不同GPU共用资源导致wall time差异如实记录。自然采样与旧无放回shuffle-S1不同，SN明确是新独立协议的标准Sample对照，不冒充历史exact复用。

第一轮CPU完整工程流程通过（3main/1arch，选择/预测/state读回）；CUDA40960三main+一arch通过，reserved peak1262MiB。复核发现两项记录问题：common checkpoint指标的step应取该state，progress最终bytes应在manifest hash之前冻结。已局部修正并对最终source重跑最小检查，未开始正式训练。

队列每GPU独立≤90%与≤5进程，用实测reservation准入、持久日志/状态、source commit绑定；一个run失败不会停止其他独立run。Runner只记原始结果，科学判读由主代理执行。完整状态已保存，但没有宣称resume入口已验证；如需恢复，将针对实际失败做最小必要处理。
