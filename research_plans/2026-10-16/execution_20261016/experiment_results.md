# 实验结果：Stage 2+（2026-10-16）

新增正式训练 **0**；新增零训练oracle **8/8**。以下严格分开新数值、历史预测法证和既有B1训练。所有差值注明比较方向；AUC越高越好，LogLoss越低越好。

## Stage 2B/2C：同目标、不同q的固定参数检验

KuaiRand/natural、MovieLens/d2_large；M3OE初始化；seeds2022–2025；每域16个既有真实训练row。每case保存48×K实际梯度（C/P各至多32坐标，A全部4坐标），alpha∈{Equal,Sample}、q∈{Natural,Balanced}、B∈{16,40960}；4096条独立replay×64步。不是训练：没有沿新theta重新求梯度或产生新的selected AUC。

| 核验 | 实测 |
|---|---:|
| 新case / 退出码0 | 8 / 8 |
| 新producer产物hash核对 | 56/56一致 |
| 与旧C2.1的B16坐标covariance比较 | 32/32一致，最大相对残差4.09e-16 |
| 同alpha不同q精确梯度相对残差上限 | 2.40e-16 |
| MC梯度均值最大标准化残差 | 3.35 SE（设计上限8） |
| 64步SGD/Adam/AdamW与torch公式最大绝对残差 | 6.19e-15 |

完整表：[估计器](estimator_statistics.csv)、[分区/seed汇总](numeric_seed_summary.csv)、[逐步状态](state_trace.csv)、[配对状态差](state_contrasts.csv)；输入/逐坐标差在numeric_oracle/<dataset>/seed<seed>/。凭据：[numeric_analysis_receipt.json](numeric_analysis_receipt.json)。

下表为64步累计delta的ensemble mean差向量范数，比较Balanced−Natural，**仅C所选坐标**，四model seed均值。SGD和Adam学习率相同，但范数不能作为性能优劣；范数的非负性也不是方向一致性。

| 数据集 | alpha | B | Adam差范数 | SGD差范数 |
|---|---|---:|---:|---:|
| KuaiRand | Equal | 16 | 0.500921 | 2.18e-5 |
| KuaiRand | Equal | 40960 | 0.034624 | 4.34e-7 |
| KuaiRand | Sample | 16 | 0.298626 | 1.04e-5 |
| KuaiRand | Sample | 40960 | 0.005168 | 2.09e-7 |
| MovieLens | Equal | 16 | 0.760025 | 9.75e-6 |
| MovieLens | Equal | 40960 | 0.217735 | 3.01e-7 |
| MovieLens | Sample | 16 | 0.315386 | 8.89e-6 |
| MovieLens | Sample | 40960 | 0.005590 | 2.07e-7 |

固定场SGD累计delta的均值与解析期望符合MC误差；全部分区最大配对坐标z为3.26，不是零效应等效性检验。Adam C累计delta的每个case/B/alpha至少一个坐标可辨别（最大坐标z的最小值9.18）；A_arch在B40960的最弱case仅2.29，不称全部分区同等稳定。E[mhat]的最大z为3.35，与同均值相容；E[vhat]含方差，不能把整个差异归于一阶动量。所有z是条件MC数值诊断，不是独立训练seed推断或校正后的p-value。图：[state_divergence.png](state_divergence.png)。

ESS同样依赖alpha/q：B40960时KuaiRand Equal/Natural平均ESS/B=0.255783，Equal/Balanced=1；Sample/Natural=1，Sample/Balanced=0.435699。MovieLens相应为0.130152/1及1/0.494477。较高ESS不保证更小的实际梯度covariance或更好selected性能。

重要反例：新表的**合并所选坐标**Balanced/Natural trace在KR Sample平均0.505344，而旧C2.1的**完整C分区**平均2.497026；ML Equal的合并所选坐标比值范围0.405829–1.207615，旧完整C分区却四seed均<1。32个新旧对应坐标矩阵数值一致，差异来自统计对象不同，不是实现矛盾。不能用任意少量坐标代表全模型方差或统一“噪声更小”结论。旧完整C结论及B16/B128、batch-cell归一化偏差见[旧oracle](../../2026-10-15/execution_20261015/C2_numeric_oracle_20261016/numeric_analysis.md)。

## Stage 3：既有B1 reconstruction的selection×trajectory

复用KR natural / M3OE / Sample、F/DH×4seed×50epoch，原数值审计400state/800预测；本次新增表格没有新增训练。其init不匹配旧S6历史pin，故不是historical exact replication。主R0按validation six-cell macro AUC选点，test仅法证。

| R0差DH−F，四seed均值±SD | validation | test |
|---|---:|---:|
| macro AUC | +0.000637±0.004224 | +0.000363±0.003997 |
| non-target AUC | +0.001134±0.006140 | +0.000579±0.005486 |
| target AUC | −0.000356±0.001093 | −0.000069±0.001159 |
| macro LogLoss | −0.000581±0.005595 | +0.001053±0.002420 |
| non-target LogLoss | −0.002630±0.004735 | −0.000424±0.000866 |
| target LogLoss | +0.003516±0.007319 | +0.004007±0.007150 |

R0 test macro仅2/4正向、1/4达到+0.003；没有稳定实用selected增益。敏感度R1/R4 test non-target +0.011584、4/4正，两个规则选中同一组checkpoint，不是两次独立复制，且有target loss代价。保持R0主规则，不按test提升R1。

| R0 test cell，四seed均值 | F AUC | DH AUC | ΔAUC | ΔLogLoss |
|---|---:|---:|---:|---:|
| d0t0 | 0.627278 | 0.632260 | +0.004982 | −0.002958 |
| d0t1 | 0.655482 | 0.652703 | −0.002779 | −0.001726 |
| d1t0 | 0.698790 | 0.698471 | −0.000318 | +0.004199 |
| d1t1 | 0.726464 | 0.726644 | +0.000180 | +0.003815 |
| d2t0 | 0.654703 | 0.653147 | −0.001556 | +0.001658 |
| d2t1 | 0.673419 | 0.675086 | +0.001667 | +0.001328 |

task0/task1 test宏ΔAUC=+0.001036/−0.000311，各2/4正向。旧bootstrap六个test AUC边际区间均含0，d0 long_view仅120正例。完整seed/阶段/baseline/loss及校准表复用[既有科学分析](../../2026-10-15/execution_20261015/B1_reconstruction_20261016/scientific_analysis.md)，其输入/结果本次hash通过。

新val-test 50epoch Spearman：macro均值0.7824（0.6201–0.8693），target均值0.9922（0.9869–0.9959），non-target均值0.6183（0.3733–0.7899）。d0两任务均值0.3917/0.3614，明显低于d1两任务0.9919/0.9900。这是轨迹描述，早期共同学习趋势也会提高相关性，不能把高相关当作near-best选点泛化保证。[逐seed排名](val_test_epoch_rank_agreement.csv)、[逐epoch六cell排名](val_test_cell_rank_agreement.csv)、[轨迹图](selection_trajectories.png)。

时序双锚点恒等式最大残差<1e-12。[逐seed分解](r0_timing_decomposition.csv)：seed2023的test macro同F-selected epoch policy差+0.007023，被DH自身timing−0.005323抵消，selected只+0.001701；seed2022两臂同epoch32，selected test−0.001639而val为正，证明不同选点时序不能解释全部反转。分解是恒等式而非独立因果识别。

## WP2 / Stage 5：AO04恢复、bootstrap与融合敏感度

恢复五个last_resume内嵌selected state。KR B0/2022、B0/2023、B1/2022完整validation预测重放最大残差分别2.295e-6、1.997e-6、1.773e-6；全部labels及loader row顺序绑定。KR B1/2023停在epoch38、config缺失，不能计完成；ML B0/2022有权重但旧validation预测为0byte下载占位。见[恢复凭据](ao04_recovery_attempt3/receipt.json)。

唯一完整KR B1−B0 seed2022：validation macro AUC +0.001494，test −0.008385（旧test逐行复算已核验）。新validation paired bootstrap1000次中，d2t1 ΔAUC+0.022437，边际95%区间[+0.005263,+0.038915]；其余5cell区间均含0。旧test六cell区间均含0。新[六cell AUC/LogLoss及区间](ao04_validation_bootstrap.csv)严格条件于既有validation-selected模型，无选择修正、多重比较修正或用户/时间聚类，不能视作确认性泛化证据。

B1/2022 selected/last-observed两个state、每域前16validation row，共192个row×task probe。selected rho范围0.4452–0.5364，last为0.4462–0.5433，未接近0/1。selected的aD(1−bD)=0.19596，平均|dprob/dq|=0.005889，最大0.024425；last平均0.007191，最大0.031307。解析chain最大残差2.18e-8；多尺度有限差分最大最小残差4.67e-6。可测非零敏感度不意味着AUC可获益，也不以未经预注册阈值称“高/低敏感度”。[逐row原始诊断](ao04_recovery_attempt3/fusion_sensitivity.csv)。

规划公式中的rho(1−rho)属于对raw q的导数；对rho的导数没有该因子，两者范数均已记录。expert difference、branch norm、gate entropy不足以证明全局collapse/identifiability；本次没有所有epoch/seed的Jacobians。

## 尚未测得的结果

2D四臂训练的selected AUC/LogLoss、Stage4 C×A、4B正常轨迹m/v分量、Stage6外部边界及E1方法全部**未测**。不能把数值PASS、恢复权重或已生成报告计为这些实验成功；逐项原因见[gate_decisions.md](gate_decisions.md)。
