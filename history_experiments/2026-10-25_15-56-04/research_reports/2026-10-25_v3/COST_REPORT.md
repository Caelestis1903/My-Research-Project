# 成本与信息访问报告

## 口径

研究实际账记录本轮真正执行的物理工作，公共warmup/首块只记一次，失败/重复startup和诊断仍保留。独立部署账为每个方法补回其独立必要的完整前缀，并剥离可识别研究评分、删除诊断及推理测量；无法分离的uniform元参考等工作显式保留。开发搜索账包含为有限方法选择支付的开发/历史记录，是另一观察视角。这三账有重叠，严禁相加。

成本组件只相加非重叠leaf；inclusive event/refresh父计时另列。多进程组件wall秒或重叠process-wall总和不是GPU-hours。已记录进程活跃区间并集、从首到末事件的跨度、单进程墙钟相加和主agent任务总elapsed回答不同问题。没有独立利用率/能耗测量，不声称精确GPU小时或能源优势。工程语义检查、审查与部分管理IO也没有完整统一计时，已记录研究分量不能冒充全部项目资源消耗。

单方法已记录分量与共享前缀补回是可核查估计。历史外部前缀的纯startup/validation/IO和部分缓存重复访问缺失，不能称完整精确minimum deployment成本；未知总量保留null，已知分量作为lower bound。STGU的实际group optimizer步数与native逻辑batch数分别列出。老aux完整结果不能因新续接目录而被重新计成两条独立实验。

## 冻结预算不是实际耗时

P1完整续跑预留12113.822554秒；旧aux/external预留1759.644884秒；P2预留2884.747753秒；正式初始预留26591.016908秒；32步回放诊断另预留87.200281秒。均基于真实保留组件率、次数、公开迁移假设和调度余量，且不是24-native等价超时。正式初始STGU显式用3倍保守容量迁移；初始预算在任何正式warmup之前冻结。组件计时保留后续接完整终点，完整比较不因达到预测秒数而终止。

预算提案与批准收据分别存档；P1独立部署澄清、P2 missing-pid退出配对修订保留原版。冻结输入的append-only成本日志按原SHA重建精确前缀，P2共77份已保留，formal初始同样保留，不因后续追加而丢失预算来源。实际数值与统一推理结果见终稿下表及完整附录。

## 信息角色与重复使用

fit更新主模型，P_update/P_cal用于控制探测和验收；P_form用于形成外查询，P_read拟合读取/局部修正，validation只按固定规则选checkpoint。Q_explore用于开发性比较，旧test/Q_confirm在最终方法冻结后评分，角色明确已暴露。参数更新、选择、诊断、实际原始行materialization、cached objective重复访问和不同row_id数量各自记账。

P_form=P_structure、P_read=P_value的确定性行身份已逐位核验；没有重造数据或角色。GEOM聚类仅用fit特征，0 formation meta/P_form查询；其主训练仍使用正常fit监督。FEATURE和CAL的额外监督不能被称为无标签方法。CAL raw-head附加P_cal诊断复用未逐次单列，P0部分缓存复用也未完备记录，明确UNKNOWN；即使能精确数出原始Data.batch行访问，也不能声称全部目标函数访问次数精确。唯一标签行与重复复用不可混同。

## 统一部署推理

基准固定物理GPU0单实验进程、FP32、AMP/TF32关闭、每次4096条相同fit特征输入、5次warmup、30次同步计时，记录全部latency、P50/P95、吞吐、loaded/peak allocated与reserved显存。先恢复既定selected完整身份，再释放训练优化器、梯度、phi/GEOM/控制器/rewind等部署不需状态；仅保留实际前向模块。参数数量为部署驻留模块的唯一参数，不声称每个权重对每个样本均算术活跃。总峰值包含权重、输入和临时量，增量峰值相对加载后基线，二者不能互换。

基础40条pipeline和条件/正式11条pipeline分表，真实CAL生产路径包含当前z0/foreign残差和min-L2投影，不以普通head或两参数gate近似。GPU1的外部vLLM工作始终未被使用或终止；GPU0隔离检查逐次留存。未支持或失败项不得伪装成测量完成，不以单次最优延迟选模型。

20条正式计时片段均完整保留到epoch11，4条预热均到10。按实际率修订剩余预留为26804.153988秒，比初始预留显式增加213.137080秒（约0.80%）；原冻结预算和101份SHA匹配的日志前缀全部保存。不可变方法spec不改写，预算修订另有收据，继续全部固定50epoch。

## 最终完整账与测量

已完成全部训练、178角色评分及51/51成功推理测量，再执行最终成本聚合和附录生成，输入SHA核验通过。共53732条成本记录、346791条信息访问记录；已知可相加leaf研究分量为**18715.766543秒**。该值不含未逐项计时的全部工程/审查管理成本，不是整个项目精确资源总量。

规范执行journal中各进程wall相加19226.915540秒，活跃区间并集9972.533860秒，首末事件跨度13112.449982秒；未归属到组件的进程wall residual为719.152670秒。它们不同口径，不能相加。最终exposed wrapper内部包wall另为260.526852秒（外部启动器计时260.937907秒），role/run wrapper与leaf重叠，仅披露、不重复加进研究组件账。P0、推理及无法独立拆分startup的边界另有完整收据。

开发搜索当前视图18715.766543秒，与研究实际重叠；另披露复用历史已记录1124.035054秒，不能说旧前缀免费，也不能把两本账相加作为总量。信息940组中885组可核实已记录集合的distinct，55组UNKNOWN；CAL raw-head额外P_cal cached诊断和P0部分cached重复仍是PARTIAL_UNKNOWN_UNJOURNALED_USES，不伪造精确次数。

### 正式50epoch独立部署组件重建

包含每法必要warmup与实际重启，去除可单独识别的研究评分/诊断/测量；所见分量并非独占GPU的最小部署测时。表为4101–4104，HISTORY部分父前缀不完整，总量保留UNKNOWN。

| method | known_recorded_seconds | mean_known_seconds | total_status |
| --- | --- | --- | --- |
| R0_BASE | [+318.589538906, +321.133261427, +323.570974936, +326.259434259] | +322.388302382 | RECORDED_COMPONENT_RECONSTRUCTION_WITH_LIMITATIONS |
| R5_PB_H1 | [+364.467567003, +365.080366197, +374.364997546, +365.994893834] | +367.476956145 | RECORDED_COMPONENT_RECONSTRUCTION_WITH_LIMITATIONS |
| R6_PB_HB | [+963.150488291, +981.613031588, +1014.112672741, +1011.856047766] | +992.683060097 | RECORDED_COMPONENT_RECONSTRUCTION_WITH_LIMITATIONS |
| R7_PB_HISTORY | [+922.811694242, +939.459668425, +965.345759295, +679.361674817] | +876.744699195 | UNKNOWN_PARTIAL_PARENT |
| STGU_DOMAIN_ADAM | [+664.041245879, +409.720810526, +672.697005107, +419.178485374] | +541.409386721 | RECORDED_COMPONENT_RECONSTRUCTION_WITH_LIMITATIONS |

HB均值为BASE约3.08倍、H1约2.70倍；这是本次记录的操作分量比，不是精确GPU能耗比。HB到50每条4900 shadow逻辑步，H1为245；HISTORY suffix4740步加共享初始化160步，独立必要总量仍4900。HB主selected无稳定优势，额外成本未获得主目标支持；晚期风险减损必须同时与更便宜的合法selected模型比较。

[质量—累计已记录成本](APPENDIX_QUALITY_COST_OBSERVATIONS.md)只使用既定selected/50检查点，明确给出已知累计组件和不完整父前缀，不做额外等成本续训，不把两个离散点间连线视为可用效率前沿。

### 同设置部署推理

40基础pipeline与11条件/正式pipeline均成功；固定4096行、5warm、30同步计时、FP32/noAMP/TF32，GPU0隔离。KR L2/L4的部署参数均1634942，P50分别5.511/5.523毫秒；形成phi无需部署，不能把训练元学习的成本误加到其推理参数，也不能因此忽略训练成本。FEATURE（L0）P50为3.999毫秒、1634876参数，CAL为6.049毫秒、P95 10.812毫秒，包含真实来源残差与min-L2求解；L5 P50为4.956毫秒。

正式五个RHD方法均部署相同native前向架构、1614254参数，所见P50约3.223–5.738毫秒、相同peak allocated约60.492MiB。控制器在推理中不执行，此延迟散布不证明控制机制本身改变了推理算量；只有一个既定检查点和短顺序测量，仍受整机调度/时钟影响。完整51项30次原始延迟、P95、吞吐、参数/缓冲、baseline/peak allocated/reserved及隔离检查均在[推理附录](APPENDIX_INFERENCE_COMBINED_NUMERICAL.md)。

全部89方法的独立部署汇总、三本账JSON收据、进程wall、信息汇总及UNKNOWN另见[成本附录索引](APPENDIX_COST_INDEX.md)。逐次操作CSV/journals和其完整分页转录在服务器保留并附SHA；公开报告保留所有科学数值和汇总，不把日志未细分费用视为0。
