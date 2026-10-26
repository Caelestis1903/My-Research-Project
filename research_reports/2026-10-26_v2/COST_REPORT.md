# 训练、参数与推理成本

所有比较条件已有四独立训练seed。以下训练时间来自八个实际开发context的delivery原始成本字段；推理证据保全与补测已经完成：135/136方法实例MEASURED，所有72个必需方法实例齐备，唯一KR5201 RANDOM_FEATURE的融合parity真实失败。没有把epoch/任务/十次计时重复算作训练种子。

## 共享研究成本与独立方法成本

共同prefix包含native、source及两套local pilots、各角色特征数据创建；B_OOF/refit随后拟合。它们在研究矩阵中共享，不能为每个方法再加一次并求“总实验时间”。PREFIX_WALL为包含性时间，不能再加其内部NATIVE_EPOCHS/PILOT/LOCAL_PILOT之和。后者保留作为分解诊断。

| 实际组件，四seed平均秒 | MovieLens | KuaiRand |
|---|---:|---:|
| PREFIX_WALL（inclusive） | 128.743 | 326.336 |
| B_OOF_AND_REFIT（共享） | 86.437 | 247.871 |
| DIRECT_FAMILY_COMBINED | 119.048 | 383.911 |
| PILOT_PROB gate拟合/选择/预测 | 33.606 | 125.002 |
| FULL_SOURCE_ALL reader拟合/选择/预测 | 94.746 | 51.613 |
| RCF形成，终态每函数累计时间仅计一次 | 53.661 | 135.872 |
| RCF reader拟合/选择/预测 | 2.584 | 21.339 |
| POOLED_TARGET形成 | 102.385 | 255.107 |
| LOCAL_ACTIVE形成 | 44.123 | 49.638 |

DIRECT_FAMILY_COMBINED是真实H/ETA/EFFORT三臂合计，没有各臂单独秒数，不按三分之一拆分。形成每pass日志是累计时钟，不能逐pass相加；只加每个唯一实际训练函数的最终elapsed。K1复用原K1/K0的槽不重新计训练；重新训练的宽头按实际计。

这些是共享设备、部分并发下观测的进程墙时，并非独占GPU时间。它们不能直接相加当总elapsed，也不能用不同拥塞时段小时间差宣称加速。额外单函数诊断reader在导出阶段运行，其单独耗时未记录，不能填0。原始审计、I/O、hash/压缩、bootstrap及修复重复前向同样不能全部塞进“训练速度”。原始服务/阶段时间与日志保留实际执行范围。

独立部署NATIVE只需要native，不需要研究中所有source/local/OOF。独立训练RCF需要native、source pilot、OOF/refit背景、需求、后继、reader和选点；FULL_TARGET/LOCAL_ACTIVE需要各自local pilots。当前测得prefix包含所有研究控制，**没有测得每种独立方法从零启动的隔离端到端墙时**；不能把包含无用组件的研究prefix当精确部署训练成本，也不伪造删除这些组件后的秒数。报告保留依赖账与实际组件时长，缺测范围明确。

## 参数与标签访问

| 融合部署保存的参数标量 | MovieLens | KuaiRand |
|---|---:|---:|
| NATIVE | 772974 | 1614254 |
| H_DIRECT | 822900 | 1664180 |
| PILOT_PROB | 923340 | 1764620 |
| FULL_SOURCE_ALL | 873036 | 1714316 |
| RCF | 923148 | 1764428 |
| POOLED_TARGET | 923148 | 1764428 |
| LOCAL_ACTIVE | 1021836 | 1863116 |

四seed在各方法对应结构下相同。native计一次，normalization/group阈值buffer另列；source隐藏标准化和固定输出/readout可经数值验真后融合，不能为了让RCF便宜而只对它融合。存储参数、每target/task实际活跃参数、训练阶段参数实例是不同量。

RCF每prefix形成12个32隐层头共99084参数；KR K1为6个64隐层头共99078，Random与RCF结构相同。两者不是每参数暴露和FLOPs完全一致。KR四seed这三方法target query标签访问均11555136，Adam steps为22800/22752/22800；source辅助访问/保留误差还需另看记录，不能只凭总标签数称完全公平。

被丢弃的两折OOF head是额外训练实例，不能当部署容量。FULL的每个非OFF完整政策有12×(2+64)=792个优化变量，两λ均拟合即1584个变量实例；GROUP_CAL每政策24，RCF/K2每政策72，K1每政策48，OFF不需要优化但此前候选拟合仍计成本。候选真实数量以`selection.candidate_history`为准，不能用清空后的RAM `candidates`列表所导出0当成未做λ搜索。三个direct checkpoint是同一梯度轨迹的选点机会，不算三次独立拟合。

## 真实推理计时与缺测

计时协议固定自然Q_explore原序前4096行，CPU FP32 raw X→native device→所需冻结组件→CPU p，CUDA同步，3次warmup、10次真实重复；编译/放置在计时外单列。所有方法相同设备/输入/协议，不使用标签。必须先与独立raw参考和retained p同时满足1e-6，才标MEASURED；原始十次时间、行hash、环境及内存分解保留。

发现普通summary会覆盖先前计时收据后，runner在runs之外原字节快照，持久保留及仅缺项补测已完成独立设计/实现/审查PASS，实际补测已通过正常RPC完成。没有原字节的ML5201旧时间不能从日志重造，只能对同一冻结模型重新测量并标补测。其余七份有效快照经独立审查按prefix/政策/row/环境核真后保留原时间。ML5201实际对全部16个已登记方法各补测一次，避免保留PILOT_PROB时缺少其第四个成本样本。最终必需九方法必须两数据×四seed完整；非必需控制若失败则不平均只通过的seed。

KR5201 RANDOM_FEATURE的融合parity为5.7095e-6，高于不变1e-6，真实状态NOT_MEASURED。它的完整原模型raw-X预测另外通过，不影响已存性能比较；它的融合延迟不能公平比较。遵循最小必要原则，本轮不为非关键控制新增compiler修复或训练。其余控制状态以最终完整表为准，不能将required九项齐备写成所有方法都测得。

实际补测后，八份preservation receipt均COMPLETE且完整冻结指纹不变；135项真实MEASURED，唯一失败保留原错误。每方法每seed先取10次的median，下表为四seed median的均值±样本SD，单位ms/4096自然行：

| 方法 | MovieLens | KuaiRand |
|---|---:|---:|
| NATIVE | 1.965±0.003 | 2.555±0.007 |
| H_DIRECT | 2.672±0.110 | 3.184±0.125 |
| PILOT_PROB | 4.441±0.128 | 5.016±0.052 |
| FULL_SOURCE_ALL | 5.638±0.116 | 6.175±0.045 |
| RCF | 7.432±0.029 | 8.314±0.041 |
| POOLED_TARGET | 7.442±0.033 | 8.088±0.069 |
| LOCAL_ACTIVE | 7.426±0.037 | 8.046±0.075 |
| K1_WIDE | 未触发 | 6.214±0.066 |

这些是批量4096的端到端稳态时间，不能冒称单请求延迟或生产P99。编译/放置另计；显存区分进程allocator峰值、相对resident allocation增量和整卡实际占用，不能将其互称或把各进程不同时点峰值相加当实际峰值。

RCF相对H约2.78×/2.61×，相对PILOT_PROB约1.67×/1.66×，相对NATIVE约3.78×/3.25×。它与同活跃容量的LOCAL/POOLED推理接近，但没有超过它们的稳定完整主指标增量，且形成有额外来源监督/预算成本。KR的K1部署参数接近而实测更快；不同宽度/头数量的执行图不能只按总参数称等成本。当前数据不足以支持为RCF复杂性付费的普遍结论；ML特定test的LL改善仍真实，不能否定所有成本—质量取舍。

[逐seed训练组件](tables/COST_four_seed_training_components.csv)、[均值/SD](tables/COST_four_seed_training_component_summary.csv)、[参数](tables/COST_four_seed_parameters.csv)、[活跃参数](tables/COST_four_seed_active_parameters.csv)、[候选优化量](tables/COST_four_seed_reader_candidates.csv)、[逐seed推理](tables/COST_four_seed_inference.csv.gz)、[10次原始值](tables/COST_four_seed_inference_repetitions.csv)、[四seed汇总](tables/COST_four_seed_inference_summary.csv)与[COST核算收据](COST_FOUR_SEED_ANALYSIS_RECEIPT.json)完整保留。Random的KR只有3/4成功测量，因此不报告删去失败seed后的平均延迟。

## 中断与物理重放成本

首服务1974641丢失后同4prefix重放，随后主机OOM使旧durable工作集不可继续，最终新四seed服务训练8prefix；主线native物理轨迹至少4+4+8=16，而独立开发统计重复仍为2数据×4seed。旧S1失败/已执行部分形成等物理成本不能填零；旧v1原始日志和完成/失败收据保留，不将新成功矩阵成本冒称整个会话总成本。

部分中断片段没有完整结束时钟，因此全会话完整GPU-seconds/单方法隔离总墙时为NOT_MEASURED；可观察日志分量及新增重复前向成本保留，不能从进程CPU时间估造精确GPU成本。SCALE八条独立训练成本另见`tables/SCALE_four_seed_costs.csv`，不与目标、角色及优化设置不同的RCF绝对时间作方法速度排名。

原collector的`module_sha256`是在调用时读取磁盘文件，不是加载函数字节证明。evaluate期间磁盘补丁与缓存旧collector可短暂不同；原源码副本及SHA已保留，实际编译器/benchmark/成本数值AST未变，不能把动态磁盘SHA倒写为早期加载身份。详见`DELIVERY_SOURCE_PROVENANCE.json`与独立审查。
