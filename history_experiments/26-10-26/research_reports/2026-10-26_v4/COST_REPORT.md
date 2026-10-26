# 成本与信息使用

## 实际研究执行账

四条必需native24轨迹、四套pilot/P1以及ML二十个P2逻辑输出全部完成，无P3/P4。记录4062个不重叠训练/选择/开发leaf，共 **5674.226502秒**；其活跃区间并集1505.709423秒。按36个实际进程首末记录计算，活动跨度和5956.257387秒、跨度并集1522.515206秒。并发值定义不同，不能相加或称GPU小时/能耗。

主要leaf：native训练2029.606604秒、原生validation24.929963、缓存131.458023、pilot104.264690、η/LOGIT423.246784、P1_DIRECT224.112090、形成371.622553、reader P_form1334.271729、P_read514.452157；所有组件/数量在[CSV](tables/COST_training_components.csv)。冻结后184项raw-X预测/重建计26.581971秒，metric/I/O计25.192602秒，另记51.774572秒。分析bootstrap为CPU信息处理，另有实际秒数收据。

部分首次setup、责任/候选缓存、审查、包装器与I/O未独立journal；不将缺失当0。进程跨度覆盖部分间隙，但仍不是完整启动/收尾wall；精确研究总量为UNKNOWN，而非伪造一个精确总耗时。训练前包装器错误保留日志，均在训练前重试原配置；不是额外seed或负结果重跑。

## 独立部署重训的已记录组件账

同一prefix在实际账只计一次；下表按每方法必要依赖重新加回native/cache/pilot/η/库/reader的已记录组件，不加无关诊断。H_DIRECT不需要pilot；ETA_DIRECT不需要OOF/m/τ/形成；CONST_UNIFORM需要UNIFORM库但不需要其神经reader；DFD/SHUFFLE承担η folds/refit。表中组件是在共享GPU负载下测量，**不是各法独占GPU最短重训时间**，未记录部分仍UNKNOWN。

| 方法 | 5101已记录秒 | 5102已记录秒 |
| --- | --- | --- |
| NATIVE | 91.834 | 177.863 |
| LOCAL | 133.302 | 248.828 |
| H_DIRECT | 262.587 | 412.853 |
| ETA_DIRECT | 170.164 | 340.512 |
| UNIFORM | 168.888 | 496.159 |
| GEOM | 380.529 | 494.734 |
| SHUFFLE | 468.777 | 590.978 |
| CONST_UNIFORM | 145.437 | 296.929 |
| LOCAL_CAP | 386.909 | 497.662 |
| DFD | 470.367 | 590.692 |

## 开发/选型与标签账

P1有144个神经fold-head拟合和48个常量target/task/fold拟合；每行只进入一折训练，η两fold20对应每个标签20训练passes。DFD P_form训练label-pass为η-fold20+refit20+reader20=60；SHUFFLE同样依赖需求。UNIFORM/GEOM/LOCAL_CAP/H_DIRECT为20，ETA_DIRECT refit20；各神经readout另P_read10。常量在PF∪PR的唯一行做凸拟合，不伪称同SGD更新数。统计/选择/重复读取另记，不是新增独立标签。开发账与独立部署/研究实际账重叠，不能相加。

## 完整模型推理实测

GPU0 RTX3090，seed5101，自然fit前4096行编码X已在GPU，完整raw-X native+下游模型，FP32网络、锁定概率精度，无AMP/TF32；5warmup后30次CUDA同步。ML10结构与KR2前缀均测，30次仅为延时重复。

| 数据 | 方法 | P50 ms | P95 ms | 实际注册参数 |
| --- | --- | --- | --- | --- |
| movielens | NATIVE | 1.9288 | 1.9556 | 772974 |
| movielens | LOCAL | 2.1416 | 2.1779 | 822516 |
| movielens | H_DIRECT | 2.5347 | 2.7465 | 822900 |
| movielens | ETA_DIRECT | 2.7044 | 2.7281 | 873018 |
| movielens | UNIFORM | 5.6937 | 5.8008 | 973278 |
| movielens | GEOM | 5.7322 | 5.8233 | 973278 |
| movielens | SHUFFLE | 5.7062 | 5.8065 | 973278 |
| movielens | CONST_UNIFORM | 4.5386 | 4.6285 | 921600 |
| movielens | LOCAL_CAP | 5.7086 | 5.8105 | 1072362 |
| movielens | DFD | 5.7092 | 5.7869 | 973278 |
| kuairand | NATIVE | 2.4918 | 2.5203 | 1614254 |
| kuairand | LOCAL | 2.5206 | 2.7357 | 1663796 |

ML native实测772974参数，KR1614254；DFD额外200304，训练期η另50502，LOCAL_CAP额外299388。DFD P50约5.709ms，H_DIRECT约2.535ms；该实现的较高延时没有获得完整风险增量支持。参数数本身不作速度证据。完整sample/peak/设备负载与predictor hash均保留。

候选贡献数与实际执行需分开：LOCAL实际计算3个source pilot/task再gather，贡献1；soft OFF仍评5候选、跳gate。表中的静态candidate_functions字段不能当实测活跃量，亦无稀疏推理承诺。常量simplex系数作为固定数据保存，不全由nn.Parameter计数表示。

## 最终存储

按用户要求，模型/Adam与大中间缓存删除，原生数据及其他`/home_new`目录不动。清理时8,954,541,940→634,400,615字节，移除8,320,141,325字节；之后追加的分析表仍为信息文件。没有.pt/.pth或模型symlink。保留行级预测、日志、hash、normalizer/选择/支持信息和两个约10MB的固定责任数据包（不是神经参数）。最终预测可直接重算指标；重新前向需重训，不能声称权重永久留存。见[清理收据](evidence/INTERMEDIATE_CLEANUP.json)。
