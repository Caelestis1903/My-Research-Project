# EMA 收益机制实验报告

完成日期：2026-11-07。研究标识：EMA_MECHANISM_20261106。交付分支：26-11-07。

## 研究结论与下一步

**固定 EMA 的排序收益及其阶段性代价得到支持；简单指标已能提供部分小幅动作价值，复杂收益指标的独立必要性证据较弱。** 四组设置的晚期 F 同时优于当前 online 和理论年龄匹配的旧 online。逐格常驻时间尺度的交叉收益只在 KuaiRand Gaussian 条件下为正。未来收益大小预测中，C_design 在四组均优于 M2/M3；单组动作的整体收益较小，少数局部范围保留进一步定向研究价值。

本轮完成全部20条轨迹、1000 epoch和34,500次正式 Adam，S1–S4、冻结 Test、图表及成本记录均齐全。预注册规则在四组中均找到至少一个 ACTION_USEFUL 局部范围，机器可读状态为 READY_FOR_TARGETED_DESIGN。实际研究优先级应结合下列效应量：MovieLens 的整体单组动作增益约0.000035 AUC；KuaiRand 整体增益接近零且 seed 间波动较大。完整有利和不利范围保存在 [ACTION_DECISIONS.csv](ACTION_DECISIONS.csv)。

| 主要判断 | 证据强度 | 下一研究决定 |
| --- | --- | --- |
| 同轨迹 F 的晚期排序收益超过本轮理论延迟对照 | STRONG | 接受历史组合与滞后成本共同存在的解释 |
| KuaiRand Gaussian 的逐格常驻时间尺度交叉空间为正 | MEDIUM | 保留该条件的局部定位价值 |
| M2 能近似同刻训练排序风险变化，未来数值预测仍弱于身份/阶段参照 | STRONG（本轮比较） | 优先简单参照，结束本轮指标扩张 |
| MovieLens 的小幅单组动作价值 | MEDIUM，LOW_PRIORITY_SMALL_EFFECT | 保留经验值，降低复杂控制器投入优先级 |
| KuaiRand Gaussian d0t1–S 的 M3 局部增量 | MEDIUM（动作净益）；WEAK（复杂信息独立必要性） | 冻结该范围与简单参照，优先做针对性否证 |
| Gaussian 与 EMA 的交互随阶段和选点改变方向 | MEDIUM | 按条件报告交互，不设置统一协同规则 |

## 完成范围与比较协议

模型为 M3oE fixed1：embedding 16、FCN [512,256,256,64]、4个 shared experts。两个数据集各有 clean/Gaussian D0、seeds 2022–2026，每条50 epoch，batch 40960，coupled Adam lr=0.01、betas=(0.9,0.999)、eps=1e-8；MovieLens/KuaiRand 的 weight decay 分别为1e-4/1e-5，Gaussian kappa分别为0.05/0.1、replacement lambda=0.5。FP32、源数据顺序、源噪声 key 和源损失归约均保留。

自然训练/验证/测试人数分别为 MovieLens 800167/100021/100021，KuaiRand 2006132/250767/250767。训练人口与比例保持原样。P_A/P_B 来自互不交的训练行池；排名探针每格256正、256负和4096固定pairs，自然探针每域512行。V_mech 每域取最多4096个真实验证行：MovieLens为[4096,4096,4096]，KuaiRand为[2692,4096,1843]。无效容器槽由固定掩码排除，逐seed行数见 [VALIDATION_PANEL_COUNTS.csv](VALIDATION_PANEL_COUNTS.csv)。

每个常驻读出独立按完整 Validation 六格等权 Macro AUC 的最早严格最大值选点。选定端点、比较规则和分析身份冻结后，由独立 Test 进程处理270个逻辑端点、254个唯一状态：139次新增评价、115条归档复用、16条重复状态复用。模型参数及端点始终驻RAM。主统计单位是五个 seed；报告样本SD及配对区间 mean ± 2.776445×SD/√5。完整配置、数据和冻结身份见 [RESOLVED_PROTOCOL.json](RESOLVED_PROTOCOL.json)、[DATA_MANIFEST.json](DATA_MANIFEST.json)、[FREEZE_RECEIPT.json](FREEZE_RECEIPT.json)。

## Q1：收益来自什么，是否只是滞后或选点？

selected Test 的同轨迹 F−online Macro结果如下。LogLoss差为正表示损失上升。

| 数据/目标 | AUC差均值 ± 配对SD | AUC 95% CI | 正向seed | LogLoss差均值 |
| --- | --- | --- | --- | --- |
| MovieLens clean | +0.003848845 ± 0.000374396 | [0.003383971,0.004313719] | 5/5 | +0.001248644 |
| MovieLens Gaussian | +0.003856802 ± 0.000645238 | [0.003055633,0.004657971] | 5/5 | +0.005334282 |
| KuaiRand clean | +0.004616399 ± 0.003048773 | [0.000830848,0.008401951] | 4/5 | −0.001225342 |
| KuaiRand Gaussian | +0.007766927 ± 0.004257500 | [0.002480543,0.013053312] | 5/5 | +0.000327687 |

同刻和年龄匹配结果进一步区分选点与历史组合。下表为晚期epoch31–50的五seed Validation均值；年龄匹配仅使用预注册理论年龄和最近整数epoch。

| 数据/目标 | F−当前online AUC | F−年龄匹配online AUC ± SD |
| --- | --- | --- |
| MovieLens clean | +0.003630 | +0.002993 ± 0.000422 |
| MovieLens Gaussian | +0.003624 | +0.003153 ± 0.000325 |
| KuaiRand clean | +0.004637 | +0.004503 ± 0.001214 |
| KuaiRand Gaussian | +0.005156 | +0.003746 ± 0.000840 |

四项年龄匹配差的95% CI下界均大于0。历史组合具有超过这个固定延迟参照的功能价值，证据 STRONG。与此同时，early epoch1–10的F−online AUC为MovieLens约−0.0030、KuaiRand约−0.0203/−0.0206。固定pair的晚期排序时间变化率从online的0.0731/0.0730降至F的0.0121/0.0122（MovieLens），从0.1129/0.1156降至0.0140/0.0148（KuaiRand）；早期同样更平稳，却伴随AUC代价。实际证据支持“排序稳定与学习滞后共同决定效用”，单独的低波动量不足以指定收益方向。

依据：[VALIDATION_GAINS.csv](VALIDATION_GAINS.csv)、[TEMPORAL_RANK_STABILITY.csv](TEMPORAL_RANK_STABILITY.csv)、[SELECTION_DIAGNOSTICS.csv](SELECTION_DIAGNOSTICS.csv)、[时间曲线](figures/TIME_CURVES.png)。

## Q2：收益如何分布，是否存在可交叉利用的偏好？

每格最优与统一最优采用相同H候选；交叉评价由其他四seed的V_A选取、留出seed的V_B评分，再交换行侧。固定epoch10/20/30/40/50先在seed内平均。

| 数据/目标 | 样本内逐格空间 | 交叉评价增益均值 ± SD | 正向seed |
| --- | --- | --- | --- |
| MovieLens clean | +0.000030 | −0.000047 ± 0.000060 | 1/5 |
| MovieLens Gaussian | +0.000028 | −0.000082 ± 0.000049 | 0/5 |
| KuaiRand clean | +0.000238 | −0.000152 ± 0.000459 | 1/5 |
| KuaiRand Gaussian | +0.000439 | +0.000375 ± 0.000331 | 5/5 |

额外选优机会解释了前三组样本内空间的一部分；KuaiRand Gaussian保留了跨seed/行侧的正均值，证据 MEDIUM。常驻时间尺度选择与短窗单组动作是两个估计对象，下面的动作价值另行评价。

selected Test 六格F−online AUC如下，所有格的五seed均值均为正。

| cell | ML clean | ML Gaussian | KR clean | KR Gaussian |
| --- | --- | --- | --- | --- |
| d0t0 | +0.004095 | +0.002609 | +0.001374 | +0.012254 |
| d0t1 | +0.004425 | +0.003673 | +0.004037 | +0.018562 |
| d1t0 | +0.003891 | +0.004814 | +0.006473 | +0.005856 |
| d1t1 | +0.003473 | +0.005111 | +0.006726 | +0.005942 |
| d2t0 | +0.004065 | +0.003773 | +0.002417 | +0.002949 |
| d2t1 | +0.003145 | +0.003160 | +0.006672 | +0.001039 |

按F的五seed平均AUC识别，MovieLens最差格为d2t0，clean/Gaussian绝对AUC为0.756329/0.756629；KuaiRand最差格为d0t0，绝对AUC为0.636991/0.643921。其相对online增益对应上表。每任务、每域、Traffic、Pooled及全部LL值分别保存于 [ENDPOINT_ABSOLUTE_SUMMARY.csv](ENDPOINT_ABSOLUTE_SUMMARY.csv) 和 [PAIRED_SUMMARY.csv](PAIRED_SUMMARY.csv)。本轮研究变量为域任务身份、参数路径、阶段和训练目标。

## Q3：几何、功能方向和实际效用有什么联系？

每五epoch采集一个恰好一epoch的pulse，fast使用rho_F²，slow使用sqrt(rho_F)，均从窗口起点F初始化。12组与全组读出只改变历史估计器。完整72格保留，主预测使用冻结计算图支持的24格。

M2_full与同刻P_A精确排序风险降低量的五seed平均相关为0.9975–0.9991；source-detach代理约0.9949–0.9953。完整导数在这个局部风险近似问题上更准确。转向独立V_mech AUC时，相关降至MovieLens 0.365/0.390、KuaiRand 0.504/0.465（clean/Gaussian）。M3直接AUC的P_A/P_B相关为MovieLens 0.441/0.503、KuaiRand 0.719/0.685，动作一致率约0.50–0.57；M1幅度的跨面板相关约0.994–0.995。

这些结果区分了三个环节：函数幅度测量稳定；训练风险的一阶近似准确；带符号AUC效用的样本迁移和未来迁移更弱。数值导数可用性与未来动作可用性应分别判断。四项无需拟合的M2/M3 raw动作，在四个condition整体的相对F平均AUC净益均为负。

依据：[PULSE_METRICS.csv](PULSE_METRICS.csv)、[PULSE_SUMMARY.csv](PULSE_SUMMARY.csv)、[METRIC_CORRESPONDENCE.csv](METRIC_CORRESPONDENCE.csv)、[PANEL_STABILITY.csv](PANEL_STABILITY.csv)、[脉冲图](figures/PULSE_EFFECTS.png)。

## Q4：能否预测未来收益，是否值得定向算法设计？

固定160次ridge拟合，截距不罚、SSE加1倍系数平方和。特征来自e时刻P_A；目标为e+5新pulse的V效用。其他四seed的origins5/10/15拟合，留出seed的25/30/35/40/45评分，V_A/V_B交换。下表为condition整体均值；AUC动作值是同一单组读出机会的平均。

| 数据/目标 | C_design MAE_D | M2_full MAE_D | M3_direct MAE_D | C_design−F | M0−F | M3_direct−F | UNIFORM_ACTION−F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML clean | 0.000991 | 0.001866 | 0.001865 | +0.00002253 | +0.00003475 | +0.00003525 | −0.00009180 |
| ML Gaussian | 0.000984 | 0.001862 | 0.001854 | +0.00001821 | +0.00003497 | +0.00003356 | −0.00009495 |
| KR clean | 0.004157 | 0.006569 | 0.006457 | +0.00000250 | −0.00000736 | −0.00001238 | +0.00001522 |
| KR Gaussian | 0.004127 | 0.006611 | 0.006501 | +0.00000536 | −0.00000080 | +0.00000086 | −0.00000814 |

C_design的收益数值预测在四组均更好，复杂有符号信息未带来condition整体的稳定优势。MovieLens的M0/C_simple与M2拟合器给出的动作收益接近，约0.000035，整体价值小于0.0002，按协议标 LOW_PRIORITY_SMALL_EFFECT。KuaiRand的condition整体效应更小，且C_design动作对应的−LL效用均值为负（clean约−0.000020，Gaussian约−0.000027），存在排序/损失取舍。

一个值得保留的具体范围是KuaiRand Gaussian、d0t1、共享组S：M3_direct的AUC动作净益为+0.000346，相对统一动作+0.000277；C_simple相对F为+0.000298。M3相对C_simple的五seed增量为[−0.000147692,0,−0.000044046,+0.000379231,+0.000054284]，均值+0.000048355、SD0.000199264。该局部动作价值有经验依据，复杂信息独立必要性的证据为 WEAK。完整预注册范围、逐seed值、全部简单参照skill和regret见 [PREDICTION_SEED.csv](PREDICTION_SEED.csv)、[PREDICTIVE_COMPARISONS.csv](PREDICTIVE_COMPARISONS.csv) 与 [ACTION_DECISIONS.csv](ACTION_DECISIONS.csv)。

下一轮最能区分竞争解释的检验，应冻结这个局部范围与M3/C_simple/统一动作规则，在另留的自然时间窗口比较，并同时记录共享组读出对全部六格的联动。检验重点是“M3局部增量能否重现、是否超出稳定身份偏好”，优先挑战当前局部正结果。本轮按S4停止，不追加候选或训练。

## Q5：Gaussian交互与概率尺度解释

selected Test的Gaussian×EMA交互AUC均值为MovieLens +0.000007957（SD0.000880733），KuaiRand +0.003150528（SD0.007058189），两者均3/5 seed为正。相同对比在Validation selected为−0.000023978/−0.001006570，在Test E50为−0.000173473/−0.001279139。交互方向取决于阶段、评价集和选点，证据支持条件性解释。

预注册C_CAL在四个condition均触发，共960次温度拟合。E40的跨行侧自然LL比较如下，AUC始终沿用原概率排序。

| 数据/目标 | 原F−online LL | 双方分别校准后的F−online LL |
| --- | --- | --- |
| ML clean | +0.004593 | −0.000255 |
| ML Gaussian | +0.003986 | −0.000451 |
| KR clean | −0.002250 | −0.001497 |
| KR Gaussian | +0.000434 | +0.000214 |

MovieLens E40的五seed平均LL代价在该诊断中转为小幅负值，概率尺度解释了这个锚点LL差的主要部分；KuaiRand仍有剩余差异。正式Test数值使用源原概率，保持原选点和LL定义。C_BG在E20/E40未达到预注册的q>0.5且效应幅度≥0.0002触发条件，记录为NOT_TRIGGERED；全组与单组残差保存在 [GROUP_INTERACTIONS.csv](GROUP_INTERACTIONS.csv)，本轮不分摊单组贡献比例。

既有同人口台账中，MovieLens M3oE/HyperGate/AutoIFS的selected Test Macro AUC约为0.775190/0.771412/0.772425，KuaiRand约为0.671907/0.682113/0.678597。Gaussian+F在MovieLens相对M3oE约+0.004167，LL约+0.000931；KuaiRand为0.679180，仍低于HyperGate约0.002932。该参照沿用各模型既有训练协议，来源与状态见 [既有证据](../../research_plans/26-11-6/01_EXISTING_EVIDENCE.md)。

## 修订、成本与交付身份

| 修订 | 实际问题与处理 | 影响 |
| --- | --- | --- |
| R1 | 观测FP32原子归约改为固定组序float64归约 | 正式采集前闭合 |
| R2 | 补guardian实际RSS与峰值成本 | 正式采集前闭合 |
| R3 | KuaiRand验证域人数不等，改为min人口和固定有效行掩码 | 首轮preflight失败，0次Adam；修复后保留ML原面板身份 |
| R4 | 337.3MB派生对比表导致分析输出超2GiB，改为12.9MB gzip | 解压字节SHA完全一致；采集hash保留，分析修订单独登记 |

独立审查的采集版本27项检查通过，R4受影响14项检查另行通过；两数据集真实batch smoke共276次工程Adam。正式运行20次worker启动、无重启或重放。完整审查和修订见 [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md)、[EXECUTION_REPORT.md](EXECUTION_REPORT.md) 及 [ANALYSIS_REVISION.json](ANALYSIS_REVISION.json)。

实际计数为4500次完整Validation读出、5200个新增pulse状态、2400次小面板M2梯度。累计worker计时：训练13226.080秒、完整V/regular观测2893.289秒、pulse553.674秒、M2梯度301.436秒、IPC2411.616秒；正式队列墙钟9980.239秒。GPU allocated/reserved峰值约1.134/1.481GB，worker/guardian RSS峰值约3.923/3.633GB；guardian参数payload为2.551GB。报告生成时观察输出约1.97GB，低于2GiB。分项时间与模型状态字节分别记录在 [COSTS.csv](COSTS.csv)，二者使用不同口径。

采集提交为05f75a73f3036694ee414b4483c878eac832f5b1，采集实现hash为e371a4c00352a9eba8f76b11c4f50f790c8c80cd02552feb4eb5a86502810aaf；R4后分析包hash为d2c4dda8525991e65902dbd011648b910893e86da8333cf3c8bbdfcc0b9253d7。原始CSV、cache、参数布局、panel、逐epoch和冻结端点的完整hash见相应JSON收据。原始有界预测保留在本地 artifacts/ema_mechanism_20261106/attempt1/predictions；仓库交付代码、报告、全部标量表、身份面板及日志。

## 完整数值附录

以下为冻结流程从实际数表生成的端点、逐seed、分层决定与成本明细。

研究范围：M3oE fixed1，MovieLens/KuaiRand，clean/Gaussian D0，每条件五 seed（2022–2026），每轨迹50 epoch。
完成20条逻辑轨迹、1000 epoch、34500次Adam；常驻shadow、pulse与观测均无训练反馈。

## 核心端点与逐 seed 配对

下表为各读出独立以完整 Validation Macro AUC 最早严格最大值选点后的 Test 差。完整 selected/E10/E50、六格、Macro、Traffic、Pooled、绝对值与seed原值见 [ENDPOINTS.csv](ENDPOINTS.csv)、[PAIRED_SEED.csv](PAIRED_SEED.csv)、[PAIRED_SUMMARY.csv](PAIRED_SUMMARY.csv) 和 [ENDPOINT_ABSOLUTE_SUMMARY.csv](ENDPOINT_ABSOLUTE_SUMMARY.csv)。

| dataset | objective | metric | mean | sample_sd | ci95_low | ci95_high | positive_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | clean | auc | 0.00384884 | 0.00037440 | 0.00338397 | 0.00431372 | 5 |
| movielens | clean | logloss | 0.00124864 | 0.00564805 | -0.00576433 | 0.00826162 | 4 |
| movielens | gaussian_d0 | auc | 0.00385680 | 0.00064524 | 0.00305563 | 0.00465797 | 5 |
| movielens | gaussian_d0 | logloss | 0.00533428 | 0.00292478 | 0.00170269 | 0.00896587 | 5 |
| kuairand | clean | auc | 0.00461640 | 0.00304877 | 0.00083085 | 0.00840195 | 4 |
| kuairand | clean | logloss | -0.00122534 | 0.00177535 | -0.00342973 | 0.00097905 | 2 |
| kuairand | gaussian_d0 | auc | 0.00776693 | 0.00425750 | 0.00248054 | 0.01305331 | 5 |
| kuairand | gaussian_d0 | logloss | 0.00032769 | 0.00285562 | -0.00321803 | 0.00387341 | 3 |
| movielens | interaction | auc | 0.00000796 | 0.00088073 | -0.00108562 | 0.00110153 | 3 |
| movielens | interaction | logloss | 0.00408564 | 0.00635751 | -0.00380826 | 0.01197954 | 4 |
| kuairand | interaction | auc | 0.00315053 | 0.00705819 | -0.00561337 | 0.01191443 | 3 |
| kuairand | interaction | logloss | 0.00155303 | 0.00366917 | -0.00300284 | 0.00610890 | 4 |

| dataset | objective | seed | metric | value |
| --- | --- | --- | --- | --- |
| movielens | clean | 2022 | auc | 0.00410669 |
| movielens | clean | 2022 | logloss | -0.00825984 |
| movielens | clean | 2023 | auc | 0.00412084 |
| movielens | clean | 2023 | logloss | 0.00394083 |
| movielens | clean | 2024 | auc | 0.00327803 |
| movielens | clean | 2024 | logloss | 0.00171634 |
| movielens | clean | 2025 | auc | 0.00408561 |
| movielens | clean | 2025 | logloss | 0.00223529 |
| movielens | clean | 2026 | auc | 0.00365306 |
| movielens | clean | 2026 | logloss | 0.00661060 |
| movielens | gaussian_d0 | 2022 | auc | 0.00271611 |
| movielens | gaussian_d0 | 2022 | logloss | 0.00700948 |
| movielens | gaussian_d0 | 2023 | auc | 0.00400204 |
| movielens | gaussian_d0 | 2023 | logloss | 0.00645220 |
| movielens | gaussian_d0 | 2024 | auc | 0.00414925 |
| movielens | gaussian_d0 | 2024 | logloss | 0.00112576 |
| movielens | gaussian_d0 | 2025 | auc | 0.00413613 |
| movielens | gaussian_d0 | 2025 | logloss | 0.00364950 |
| movielens | gaussian_d0 | 2026 | auc | 0.00428048 |
| movielens | gaussian_d0 | 2026 | logloss | 0.00843447 |
| kuairand | clean | 2022 | auc | -0.00022393 |
| kuairand | clean | 2022 | logloss | -0.00312160 |
| kuairand | clean | 2023 | auc | 0.00823451 |
| kuairand | clean | 2023 | logloss | -0.00110939 |
| kuairand | clean | 2024 | auc | 0.00495508 |
| kuairand | clean | 2024 | logloss | 0.00053576 |
| kuairand | clean | 2025 | auc | 0.00477926 |
| kuairand | clean | 2025 | logloss | -0.00293281 |
| kuairand | clean | 2026 | auc | 0.00533707 |
| kuairand | clean | 2026 | logloss | 0.00050133 |
| kuairand | gaussian_d0 | 2022 | auc | 0.01504549 |
| kuairand | gaussian_d0 | 2022 | logloss | 0.00073854 |
| kuairand | gaussian_d0 | 2023 | auc | 0.00543668 |
| kuairand | gaussian_d0 | 2023 | logloss | 0.00365598 |
| kuairand | gaussian_d0 | 2024 | auc | 0.00489555 |
| kuairand | gaussian_d0 | 2024 | logloss | -0.00401333 |
| kuairand | gaussian_d0 | 2025 | auc | 0.00536903 |
| kuairand | gaussian_d0 | 2025 | logloss | -0.00045452 |
| kuairand | gaussian_d0 | 2026 | auc | 0.00808789 |
| kuairand | gaussian_d0 | 2026 | logloss | 0.00171177 |
| movielens | interaction | 2022 | auc | -0.00139058 |
| movielens | interaction | 2022 | logloss | 0.01526931 |
| movielens | interaction | 2023 | auc | -0.00011881 |
| movielens | interaction | 2023 | logloss | 0.00251137 |
| movielens | interaction | 2024 | auc | 0.00087122 |
| movielens | interaction | 2024 | logloss | -0.00059058 |
| movielens | interaction | 2025 | auc | 0.00005053 |
| movielens | interaction | 2025 | logloss | 0.00141421 |
| movielens | interaction | 2026 | auc | 0.00062742 |
| movielens | interaction | 2026 | logloss | 0.00182387 |
| kuairand | interaction | 2022 | auc | 0.01526941 |
| kuairand | interaction | 2022 | logloss | 0.00386013 |
| kuairand | interaction | 2023 | auc | -0.00279783 |
| kuairand | interaction | 2023 | logloss | 0.00476537 |
| kuairand | interaction | 2024 | auc | -0.00005954 |
| kuairand | interaction | 2024 | logloss | -0.00454909 |
| kuairand | interaction | 2025 | auc | 0.00058977 |
| kuairand | interaction | 2025 | logloss | 0.00247829 |
| kuairand | interaction | 2026 | auc | 0.00275082 |
| kuairand | interaction | 2026 | logloss | 0.00121044 |


## Q1–Q5：分层研究决定

所有区间以五个训练 seed 的配对差计算；同轨迹多个 epoch、cell、panel、shadow 不增加独立样本数。

### Q1 历史读出、训练阶段与有效滞后

- movielens / clean：`HISTORY_READOUT_GAIN`，证据 **STRONG**。
  五seed均值 +0.003630215，样本SD 0.000281045，95% CI [+0.003281251, +0.003979178]。
  证据：[表](VALIDATION_GAINS.csv)；Common-epoch comparison separates the readout effect from independent checkpoint selection; age-matched online and temporal rank tables assess lag. Retain the measured conditional readout effect; interpret lag and pulse evidence jointly.
- movielens / gaussian_d0：`HISTORY_READOUT_GAIN`，证据 **STRONG**。
  五seed均值 +0.003624202，样本SD 0.000218083，95% CI [+0.003353416, +0.003894987]。
  证据：[表](VALIDATION_GAINS.csv)；Common-epoch comparison separates the readout effect from independent checkpoint selection; age-matched online and temporal rank tables assess lag. Retain the measured conditional readout effect; interpret lag and pulse evidence jointly.
- kuairand / clean：`HISTORY_READOUT_GAIN`，证据 **STRONG**。
  五seed均值 +0.004636873，样本SD 0.001711644，95% CI [+0.002511586, +0.006762161]。
  证据：[表](VALIDATION_GAINS.csv)；Common-epoch comparison separates the readout effect from independent checkpoint selection; age-matched online and temporal rank tables assess lag. Retain the measured conditional readout effect; interpret lag and pulse evidence jointly.
- kuairand / gaussian_d0：`HISTORY_READOUT_GAIN`，证据 **STRONG**。
  五seed均值 +0.005156018，样本SD 0.001387857，95% CI [+0.003432765, +0.006879270]。
  证据：[表](VALIDATION_GAINS.csv)；Common-epoch comparison separates the readout effect from independent checkpoint selection; age-matched online and temporal rank tables assess lag. Retain the measured conditional readout effect; interpret lag and pulse evidence jointly.

### Q2 六格、时间尺度与路径偏好

- movielens / clean：`NO_AVERAGE_CROSS_EVALUATED_PREFERENCE_GAIN`，证据 **MEDIUM**。
  五seed均值 -0.000047299，样本SD 0.000059690，95% CI [-0.000121414, +0.000026817]。
  证据：[表](HISTORY_PREFERENCES.csv)；Same candidate count, leave-one-seed-out and opposite Validation-row selection/scoring compare conditional against uniform history preference. Combine with independent future action utility; keep all cell responses.
- movielens / gaussian_d0：`NO_AVERAGE_CROSS_EVALUATED_PREFERENCE_GAIN`，证据 **STRONG**。
  五seed均值 -0.000082320，样本SD 0.000048540，95% CI [-0.000142590, -0.000022050]。
  证据：[表](HISTORY_PREFERENCES.csv)；Same candidate count, leave-one-seed-out and opposite Validation-row selection/scoring compare conditional against uniform history preference. Combine with independent future action utility; keep all cell responses.
- kuairand / clean：`NO_AVERAGE_CROSS_EVALUATED_PREFERENCE_GAIN`，证据 **MEDIUM**。
  五seed均值 -0.000152137，样本SD 0.000458829，95% CI [-0.000721848, +0.000417575]。
  证据：[表](HISTORY_PREFERENCES.csv)；Same candidate count, leave-one-seed-out and opposite Validation-row selection/scoring compare conditional against uniform history preference. Combine with independent future action utility; keep all cell responses.
- kuairand / gaussian_d0：`CROSS_EVALUATED_PREFERENCE_SPACE`，证据 **MEDIUM**。
  五seed均值 +0.000375282，样本SD 0.000330564，95% CI [-0.000035167, +0.000785731]。
  证据：[表](HISTORY_PREFERENCES.csv)；Same candidate count, leave-one-seed-out and opposite Validation-row selection/scoring compare conditional against uniform history preference. Combine with independent future action utility; keep all cell responses.

### Q3 变化量、功能方向与实际收益的对应

- movielens / clean：`DESCRIPTIVE_CORRESPONDENCE`，证据 **MEDIUM**。
  证据：[表](METRIC_CORRESPONDENCE.csv)；Simultaneous train-to-Validation correspondence is separate from forward prediction; P_B and source-detach diagnostics identify sample and derivative differences. Use the registered future folds to judge action value.
- movielens / gaussian_d0：`DESCRIPTIVE_CORRESPONDENCE`，证据 **MEDIUM**。
  证据：[表](METRIC_CORRESPONDENCE.csv)；Simultaneous train-to-Validation correspondence is separate from forward prediction; P_B and source-detach diagnostics identify sample and derivative differences. Use the registered future folds to judge action value.
- kuairand / clean：`DESCRIPTIVE_CORRESPONDENCE`，证据 **MEDIUM**。
  证据：[表](METRIC_CORRESPONDENCE.csv)；Simultaneous train-to-Validation correspondence is separate from forward prediction; P_B and source-detach diagnostics identify sample and derivative differences. Use the registered future folds to judge action value.
- kuairand / gaussian_d0：`DESCRIPTIVE_CORRESPONDENCE`，证据 **MEDIUM**。
  证据：[表](METRIC_CORRESPONDENCE.csv)；Simultaneous train-to-Validation correspondence is separate from forward prediction; P_B and source-detach diagnostics identify sample and derivative differences. Use the registered future folds to judge action value.

### Q4 未来预测和动作价值

- movielens / clean：`READY_FOR_TARGETED_DESIGN`，证据 **MEDIUM**。
  证据：[表](PREDICTION_SEED.csv)；Actions compare against F and a fit-fold uniform action on the same single-group opportunities; all five simple references are retained. Targeted algorithm design is supported in the reported useful scopes only.
- movielens / gaussian_d0：`READY_FOR_TARGETED_DESIGN`，证据 **MEDIUM**。
  证据：[表](PREDICTION_SEED.csv)；Actions compare against F and a fit-fold uniform action on the same single-group opportunities; all five simple references are retained. Targeted algorithm design is supported in the reported useful scopes only.
- kuairand / clean：`READY_FOR_TARGETED_DESIGN`，证据 **MEDIUM**。
  证据：[表](PREDICTION_SEED.csv)；Actions compare against F and a fit-fold uniform action on the same single-group opportunities; all five simple references are retained. Targeted algorithm design is supported in the reported useful scopes only.
- kuairand / gaussian_d0：`READY_FOR_TARGETED_DESIGN`，证据 **MEDIUM**。
  证据：[表](PREDICTION_SEED.csv)；Actions compare against F and a fit-fold uniform action on the same single-group opportunities; all five simple references are retained. Targeted algorithm design is supported in the reported useful scopes only.

### Q5 Gaussian 与 EMA 交互

- movielens / two objectives：`CONDITIONAL_GAUSSIAN_INTERACTION`，证据 **MEDIUM**。
  五seed均值 -0.000006013，样本SD 0.000289677，95% CI [-0.000365695, +0.000353669]。
  证据：[表](VALIDATION_GAINS.csv)；Clean and Gaussian are distinct online objectives; the difference is between within-trajectory EMA gains, with selected/fixed and split-specific evidence kept separate. Report the interaction by stage and cell; preserve the frozen objective scope.
- kuairand / two objectives：`CONDITIONAL_GAUSSIAN_INTERACTION`，证据 **MEDIUM**。
  五seed均值 +0.000519144，样本SD 0.001205301，95% CI [-0.000977435, +0.002015723]。
  证据：[表](VALIDATION_GAINS.csv)；Clean and Gaussian are distinct online objectives; the difference is between within-trajectory EMA gains, with selected/fixed and split-specific evidence kept separate. Report the interaction by stage and cell; preserve the frozen objective scope.

## 预测比较与局部研究范围

固定八类ridge（SSE＋1×系数平方和、截距不罚）和四项raw动作诊断，与同折UNIFORM_ACTION比较。拟合只用其他四seed早期origins5/10/15与V_A未来标签；评分用留出seed晚期origins25/30/35/40/45与V_B，再交换。共160个ridge fit。
动作净益是单组读出机会的平均。各condition/cell/group/cell×group范围、MAE、相对全部简单参照skill、方向率、regret、AUC/LL动作效用在 [PREDICTIVE_OOF.csv](PREDICTIVE_OOF.csv)、[PREDICTION_SEED.csv](PREDICTION_SEED.csv)、[PREDICTION_SUMMARY.csv](PREDICTION_SUMMARY.csv) 与 [PREDICTIVE_COMPARISONS.csv](PREDICTIVE_COMPARISONS.csv)。

| dataset | objective | model_id | metric | mean | sample_sd |
| --- | --- | --- | --- | --- | --- |
| kuairand | clean | C_M01 | MAE_D | 0.00656406 | 0.00058377 |
| kuairand | clean | C_M01 | chosen_minus_F | -0.00000808 | 0.00003049 |
| kuairand | clean | C_M01 | chosen_minus_uniform_action | -0.00002330 | 0.00007840 |
| kuairand | gaussian_d0 | C_M01 | MAE_D | 0.00655136 | 0.00062181 |
| kuairand | gaussian_d0 | C_M01 | chosen_minus_F | -0.00000134 | 0.00004825 |
| kuairand | gaussian_d0 | C_M01 | chosen_minus_uniform_action | 0.00000680 | 0.00007817 |
| kuairand | clean | C_design | MAE_D | 0.00415719 | 0.00017414 |
| kuairand | clean | C_design | chosen_minus_F | 0.00000250 | 0.00006284 |
| kuairand | clean | C_design | chosen_minus_uniform_action | -0.00001273 | 0.00007627 |
| kuairand | gaussian_d0 | C_design | MAE_D | 0.00412671 | 0.00017522 |
| kuairand | gaussian_d0 | C_design | chosen_minus_F | 0.00000536 | 0.00005970 |
| kuairand | gaussian_d0 | C_design | chosen_minus_uniform_action | 0.00001350 | 0.00008413 |
| kuairand | clean | C_simple | MAE_D | 0.00663628 | 0.00051309 |
| kuairand | clean | C_simple | chosen_minus_F | -0.00000755 | 0.00003021 |
| kuairand | clean | C_simple | chosen_minus_uniform_action | -0.00002277 | 0.00007762 |
| kuairand | gaussian_d0 | C_simple | MAE_D | 0.00667569 | 0.00053237 |
| kuairand | gaussian_d0 | C_simple | chosen_minus_F | -0.00000015 | 0.00004869 |
| kuairand | gaussian_d0 | C_simple | chosen_minus_uniform_action | 0.00000799 | 0.00007997 |
| kuairand | clean | M0 | MAE_D | 0.00644855 | 0.00039207 |
| kuairand | clean | M0 | chosen_minus_F | -0.00000736 | 0.00003091 |
| kuairand | clean | M0 | chosen_minus_uniform_action | -0.00002258 | 0.00007948 |
| kuairand | gaussian_d0 | M0 | MAE_D | 0.00643760 | 0.00044910 |
| kuairand | gaussian_d0 | M0 | chosen_minus_F | -0.00000080 | 0.00005222 |
| kuairand | gaussian_d0 | M0 | chosen_minus_uniform_action | 0.00000734 | 0.00008159 |
| kuairand | clean | M1 | MAE_D | 0.00421614 | 0.00023245 |
| kuairand | clean | M1 | chosen_minus_F | -0.00000131 | 0.00004883 |
| kuairand | clean | M1 | chosen_minus_uniform_action | -0.00001653 | 0.00007225 |
| kuairand | gaussian_d0 | M1 | MAE_D | 0.00418039 | 0.00024246 |
| kuairand | gaussian_d0 | M1 | chosen_minus_F | 0.00000441 | 0.00006413 |
| kuairand | gaussian_d0 | M1 | chosen_minus_uniform_action | 0.00001255 | 0.00008435 |
| kuairand | clean | M2_detach | MAE_D | 0.00653436 | 0.00046455 |
| kuairand | clean | M2_detach | chosen_minus_F | -0.00000808 | 0.00003049 |
| kuairand | clean | M2_detach | chosen_minus_uniform_action | -0.00002330 | 0.00007840 |
| kuairand | gaussian_d0 | M2_detach | MAE_D | 0.00657886 | 0.00047963 |
| kuairand | gaussian_d0 | M2_detach | chosen_minus_F | -0.00000218 | 0.00004652 |
| kuairand | gaussian_d0 | M2_detach | chosen_minus_uniform_action | 0.00000596 | 0.00007710 |
| kuairand | clean | M2_detach_raw | MAE_D | NA | NA |
| kuairand | clean | M2_detach_raw | chosen_minus_F | -0.00001905 | 0.00005554 |
| kuairand | clean | M2_detach_raw | chosen_minus_uniform_action | -0.00003427 | 0.00004875 |
| kuairand | gaussian_d0 | M2_detach_raw | MAE_D | NA | NA |
| kuairand | gaussian_d0 | M2_detach_raw | chosen_minus_F | -0.00002349 | 0.00006144 |
| kuairand | gaussian_d0 | M2_detach_raw | chosen_minus_uniform_action | -0.00001535 | 0.00004822 |
| kuairand | clean | M2_full | MAE_D | 0.00656894 | 0.00046208 |
| kuairand | clean | M2_full | chosen_minus_F | -0.00000808 | 0.00003049 |
| kuairand | clean | M2_full | chosen_minus_uniform_action | -0.00002330 | 0.00007840 |
| kuairand | gaussian_d0 | M2_full | MAE_D | 0.00661125 | 0.00048012 |
| kuairand | gaussian_d0 | M2_full | chosen_minus_F | -0.00000208 | 0.00004645 |
| kuairand | gaussian_d0 | M2_full | chosen_minus_uniform_action | 0.00000606 | 0.00007697 |
| kuairand | clean | M2_full_raw | MAE_D | NA | NA |
| kuairand | clean | M2_full_raw | chosen_minus_F | -0.00001781 | 0.00005420 |
| kuairand | clean | M2_full_raw | chosen_minus_uniform_action | -0.00003304 | 0.00005025 |
| kuairand | gaussian_d0 | M2_full_raw | MAE_D | NA | NA |
| kuairand | gaussian_d0 | M2_full_raw | chosen_minus_F | -0.00002096 | 0.00005635 |
| kuairand | gaussian_d0 | M2_full_raw | chosen_minus_uniform_action | -0.00001282 | 0.00004772 |
| kuairand | clean | M3_auc_raw | MAE_D | NA | NA |
| kuairand | clean | M3_auc_raw | chosen_minus_F | -0.00000376 | 0.00004469 |
| kuairand | clean | M3_auc_raw | chosen_minus_uniform_action | -0.00001898 | 0.00004409 |
| kuairand | gaussian_d0 | M3_auc_raw | MAE_D | NA | NA |
| kuairand | gaussian_d0 | M3_auc_raw | chosen_minus_F | -0.00000787 | 0.00004464 |
| kuairand | gaussian_d0 | M3_auc_raw | chosen_minus_uniform_action | 0.00000027 | 0.00002732 |
| kuairand | clean | M3_direct | MAE_D | 0.00645702 | 0.00043544 |
| kuairand | clean | M3_direct | chosen_minus_F | -0.00001238 | 0.00003673 |
| kuairand | clean | M3_direct | chosen_minus_uniform_action | -0.00002761 | 0.00008021 |
| kuairand | gaussian_d0 | M3_direct | MAE_D | 0.00650134 | 0.00045616 |
| kuairand | gaussian_d0 | M3_direct | chosen_minus_F | 0.00000086 | 0.00004461 |
| kuairand | gaussian_d0 | M3_direct | chosen_minus_uniform_action | 0.00000900 | 0.00007372 |
| kuairand | clean | M3_rank_raw | MAE_D | NA | NA |
| kuairand | clean | M3_rank_raw | chosen_minus_F | -0.00001127 | 0.00005800 |
| kuairand | clean | M3_rank_raw | chosen_minus_uniform_action | -0.00002649 | 0.00005365 |
| kuairand | gaussian_d0 | M3_rank_raw | MAE_D | NA | NA |
| kuairand | gaussian_d0 | M3_rank_raw | chosen_minus_F | -0.00001158 | 0.00005175 |
| kuairand | gaussian_d0 | M3_rank_raw | chosen_minus_uniform_action | -0.00000344 | 0.00004127 |
| kuairand | clean | UNIFORM_ACTION | MAE_D | NA | NA |
| kuairand | clean | UNIFORM_ACTION | chosen_minus_F | 0.00001522 | 0.00006452 |
| kuairand | clean | UNIFORM_ACTION | chosen_minus_uniform_action | 0.00000000 | 0.00000000 |
| kuairand | gaussian_d0 | UNIFORM_ACTION | MAE_D | NA | NA |
| kuairand | gaussian_d0 | UNIFORM_ACTION | chosen_minus_F | -0.00000814 | 0.00004734 |
| kuairand | gaussian_d0 | UNIFORM_ACTION | chosen_minus_uniform_action | 0.00000000 | 0.00000000 |
| movielens | clean | C_M01 | MAE_D | 0.00227393 | 0.00018613 |
| movielens | clean | C_M01 | chosen_minus_F | 0.00003493 | 0.00001560 |
| movielens | clean | C_M01 | chosen_minus_uniform_action | 0.00012673 | 0.00004870 |
| movielens | gaussian_d0 | C_M01 | MAE_D | 0.00226768 | 0.00018050 |
| movielens | gaussian_d0 | C_M01 | chosen_minus_F | 0.00003497 | 0.00001897 |
| movielens | gaussian_d0 | C_M01 | chosen_minus_uniform_action | 0.00012992 | 0.00005053 |
| movielens | clean | C_design | MAE_D | 0.00099145 | 0.00008295 |
| movielens | clean | C_design | chosen_minus_F | 0.00002253 | 0.00001102 |
| movielens | clean | C_design | chosen_minus_uniform_action | 0.00011433 | 0.00004011 |
| movielens | gaussian_d0 | C_design | MAE_D | 0.00098363 | 0.00008565 |
| movielens | gaussian_d0 | C_design | chosen_minus_F | 0.00001821 | 0.00001527 |
| movielens | gaussian_d0 | C_design | chosen_minus_uniform_action | 0.00011315 | 0.00004544 |
| movielens | clean | C_simple | MAE_D | 0.00219840 | 0.00016018 |
| movielens | clean | C_simple | chosen_minus_F | 0.00003493 | 0.00001560 |
| movielens | clean | C_simple | chosen_minus_uniform_action | 0.00012673 | 0.00004870 |
| movielens | gaussian_d0 | C_simple | MAE_D | 0.00219376 | 0.00017729 |
| movielens | gaussian_d0 | C_simple | chosen_minus_F | 0.00003497 | 0.00001897 |
| movielens | gaussian_d0 | C_simple | chosen_minus_uniform_action | 0.00012992 | 0.00005053 |
| movielens | clean | M0 | MAE_D | 0.00210018 | 0.00017234 |
| movielens | clean | M0 | chosen_minus_F | 0.00003475 | 0.00001548 |
| movielens | clean | M0 | chosen_minus_uniform_action | 0.00012655 | 0.00004856 |
| movielens | gaussian_d0 | M0 | MAE_D | 0.00209101 | 0.00017207 |
| movielens | gaussian_d0 | M0 | chosen_minus_F | 0.00003497 | 0.00001897 |
| movielens | gaussian_d0 | M0 | chosen_minus_uniform_action | 0.00012992 | 0.00005053 |
| movielens | clean | M1 | MAE_D | 0.00100965 | 0.00010147 |
| movielens | clean | M1 | chosen_minus_F | 0.00002583 | 0.00001030 |
| movielens | clean | M1 | chosen_minus_uniform_action | 0.00011763 | 0.00004130 |
| movielens | gaussian_d0 | M1 | MAE_D | 0.00100421 | 0.00009914 |
| movielens | gaussian_d0 | M1 | chosen_minus_F | 0.00002002 | 0.00001655 |
| movielens | gaussian_d0 | M1 | chosen_minus_uniform_action | 0.00011497 | 0.00004763 |
| movielens | clean | M2_detach | MAE_D | 0.00187866 | 0.00015722 |
| movielens | clean | M2_detach | chosen_minus_F | 0.00003475 | 0.00001548 |
| movielens | clean | M2_detach | chosen_minus_uniform_action | 0.00012655 | 0.00004856 |
| movielens | gaussian_d0 | M2_detach | MAE_D | 0.00187481 | 0.00017950 |
| movielens | gaussian_d0 | M2_detach | chosen_minus_F | 0.00003497 | 0.00001897 |
| movielens | gaussian_d0 | M2_detach | chosen_minus_uniform_action | 0.00012992 | 0.00005053 |
| movielens | clean | M2_detach_raw | MAE_D | NA | NA |
| movielens | clean | M2_detach_raw | chosen_minus_F | -0.00005089 | 0.00004330 |
| movielens | clean | M2_detach_raw | chosen_minus_uniform_action | 0.00004091 | 0.00001400 |
| movielens | gaussian_d0 | M2_detach_raw | MAE_D | NA | NA |
| movielens | gaussian_d0 | M2_detach_raw | chosen_minus_F | -0.00005941 | 0.00002071 |
| movielens | gaussian_d0 | M2_detach_raw | chosen_minus_uniform_action | 0.00003554 | 0.00002102 |
| movielens | clean | M2_full | MAE_D | 0.00186643 | 0.00014833 |
| movielens | clean | M2_full | chosen_minus_F | 0.00003475 | 0.00001548 |
| movielens | clean | M2_full | chosen_minus_uniform_action | 0.00012655 | 0.00004856 |
| movielens | gaussian_d0 | M2_full | MAE_D | 0.00186175 | 0.00016968 |
| movielens | gaussian_d0 | M2_full | chosen_minus_F | 0.00003497 | 0.00001897 |
| movielens | gaussian_d0 | M2_full | chosen_minus_uniform_action | 0.00012992 | 0.00005053 |
| movielens | clean | M2_full_raw | MAE_D | NA | NA |
| movielens | clean | M2_full_raw | chosen_minus_F | -0.00005246 | 0.00003494 |
| movielens | clean | M2_full_raw | chosen_minus_uniform_action | 0.00003934 | 0.00000822 |
| movielens | gaussian_d0 | M2_full_raw | MAE_D | NA | NA |
| movielens | gaussian_d0 | M2_full_raw | chosen_minus_F | -0.00006101 | 0.00001996 |
| movielens | gaussian_d0 | M2_full_raw | chosen_minus_uniform_action | 0.00003394 | 0.00002047 |
| movielens | clean | M3_auc_raw | MAE_D | NA | NA |
| movielens | clean | M3_auc_raw | chosen_minus_F | -0.00004078 | 0.00002519 |
| movielens | clean | M3_auc_raw | chosen_minus_uniform_action | 0.00005102 | 0.00001372 |
| movielens | gaussian_d0 | M3_auc_raw | MAE_D | NA | NA |
| movielens | gaussian_d0 | M3_auc_raw | chosen_minus_F | -0.00005176 | 0.00002217 |
| movielens | gaussian_d0 | M3_auc_raw | chosen_minus_uniform_action | 0.00004319 | 0.00001961 |
| movielens | clean | M3_direct | MAE_D | 0.00186493 | 0.00013759 |
| movielens | clean | M3_direct | chosen_minus_F | 0.00003525 | 0.00001468 |
| movielens | clean | M3_direct | chosen_minus_uniform_action | 0.00012705 | 0.00004764 |
| movielens | gaussian_d0 | M3_direct | MAE_D | 0.00185379 | 0.00016377 |
| movielens | gaussian_d0 | M3_direct | chosen_minus_F | 0.00003356 | 0.00001923 |
| movielens | gaussian_d0 | M3_direct | chosen_minus_uniform_action | 0.00012850 | 0.00005114 |
| movielens | clean | M3_rank_raw | MAE_D | NA | NA |
| movielens | clean | M3_rank_raw | chosen_minus_F | -0.00004371 | 0.00003392 |
| movielens | clean | M3_rank_raw | chosen_minus_uniform_action | 0.00004809 | 0.00000905 |
| movielens | gaussian_d0 | M3_rank_raw | MAE_D | NA | NA |
| movielens | gaussian_d0 | M3_rank_raw | chosen_minus_F | -0.00005248 | 0.00002864 |
| movielens | gaussian_d0 | M3_rank_raw | chosen_minus_uniform_action | 0.00004246 | 0.00001656 |
| movielens | clean | UNIFORM_ACTION | MAE_D | NA | NA |
| movielens | clean | UNIFORM_ACTION | chosen_minus_F | -0.00009180 | 0.00003342 |
| movielens | clean | UNIFORM_ACTION | chosen_minus_uniform_action | 0.00000000 | 0.00000000 |
| movielens | gaussian_d0 | UNIFORM_ACTION | MAE_D | NA | NA |
| movielens | gaussian_d0 | UNIFORM_ACTION | chosen_minus_F | -0.00009495 | 0.00003285 |
| movielens | gaussian_d0 | UNIFORM_ACTION | chosen_minus_uniform_action | 0.00000000 | 0.00000000 |


预注册范围中满足正向动作净益且胜统一动作的记录数：897。全部有利和不利范围保留于 [ACTION_DECISIONS.csv](ACTION_DECISIONS.csv)，效应小于.0002者标 LOW_PRIORITY_SMALL_EFFECT。

## 有限控制与竞争解释

C_CAL/C_BG 的触发依据和冻结组见 [STAGE3_DECISION.json](STAGE3_DECISION.json)。温度仅重算源LL、AUC保留原p排序；第二背景只读取E20/E40保留状态，均不新增训练。
[CONTROL_RESULTS.csv](CONTROL_RESULTS.csv)、[CALIBRATION_SUMMARY.csv](CALIBRATION_SUMMARY.csv)、[BACKGROUND_SUMMARY.csv](BACKGROUND_SUMMARY.csv) 保存实际控制数值或 NOT_TRIGGERED。
固定历史组合与较合适旧online的竞争解释由 [VALIDATION_GAINS.csv](VALIDATION_GAINS.csv) 的年龄匹配对比约束；低波动与延后改进由 [TEMPORAL_RANK_STABILITY.csv](TEMPORAL_RANK_STABILITY.csv) 分开检查。
单组可加性由 [GROUP_INTERACTIONS.csv](GROUP_INTERACTIONS.csv) 与第二背景读出检查。P_A/P_B样本敏感性和full/source_detach差异保留在 [PANEL_STABILITY.csv](PANEL_STABILITY.csv)、[METRIC_CORRESPONDENCE.csv](METRIC_CORRESPONDENCE.csv)。

## 冻结、来源与修订

source commit：`40705dfe0bb4fe614bdca835c0bae567348c87d0`；collection implementation hash：`e371a4c00352a9eba8f76b11c4f50f790c8c80cd02552feb4eb5a86502810aaf`；manifest hash：`b65ae44febf828cb0cf844bf8d0226567921d6e675396a8b6beed7e50c3640dd`。
plan hash：`d9cecac2ab0fc87f53c425b355192ee6198354ddfe8cf22a7dc65595129534da`；analysis hash：`b9f314370ad16b365523a7532ec10e92bf7cea66d417da9d94ea3106bb8640ca`。
采集后派生表采用无损gzip，当前分析实现hash为 `d2c4dda8525991e65902dbd011648b910893e86da8333cf3c8bbdfcc0b9253d7`；原采集身份和本次分析修订分列于 [ANALYSIS_REVISION.json](ANALYSIS_REVISION.json)。
每个端点身份见 [SELECTION_FROZEN.json](SELECTION_FROZEN.json)；与源、数据、面板、分析、阶段决定的绑定见 [FREEZE_RECEIPT.json](FREEZE_RECEIPT.json)。Test由独立进程读取，归档同state复用单列，pulse/条件最优/温度不进入Test。
数据身份与原CSV/cache/row/split/encoder/scaler：[DATA_MANIFEST.json](DATA_MANIFEST.json)、[PROBE_MANIFEST.json](PROBE_MANIFEST.json)。既有证据和A/B去重：[AUDIT_INVENTORY.json](AUDIT_INVENTORY.json)、[EVIDENCE_REUSE.csv](EVIDENCE_REUSE.csv)、[ARCHIVED_FACTORIAL.csv](ARCHIVED_FACTORIAL.csv)。
V_mech每域有效自然行数为min(4096,该域Validation人数)，[VALIDATION_PANEL_COUNTS.csv](VALIDATION_PANEL_COUNTS.csv)记录逐seed有效人数及V_A/V_B人数。固定容量容器以indices≥0解析真实行；指标、排序pair、偏好、校准和未来标签均按真实行计算。完整Validation选点与Test仍使用源完整人口。
设计、实现、审查和运行修订记录由研究交付目录中的 DESIGN_SPECIFICATION、CODER_IMPLEMENTATION_REPORT、IMPLEMENTATION_REVIEW 与原始运行日志提供。

## 成本与产物

本轮实际逻辑Adam数：34500；完整V模型读出：4500；新增pulse状态：5200；M2小面板梯度：2400。
训练累计worker秒 13226.080；完整V/regular观测累计秒 2893.289；pulse累计秒 553.674；M2累计秒 301.436；IPC累计秒 2411.616。
分项worker时间包含共享设备等待；墙钟、GPU峰值、RAM、预测字节及故障物理成本分别保留于 [COSTS.csv](COSTS.csv)、[ANALYSIS_COSTS.csv](ANALYSIS_COSTS.csv)、[TEST_COSTS.csv](TEST_COSTS.csv)、[REPLAYS.jsonl](REPLAYS.jsonl)。
标准PNG/PDF图及底表：[时间曲线](figures/TIME_CURVES.png)、[历史偏好](figures/HISTORY_PREFERENCE_HEATMAP.png)、[脉冲效应](figures/PULSE_EFFECTS.png)、[排序与LL](figures/RANK_STABILITY_AND_LL.png)、[排序翻转](figures/RANK_FLIPS.png)、[预测比较](figures/PREDICTIVE_COMPARISON.png)。

本轮到S4、冻结Test和报告后结束；后续定向设计按上面的具体适用范围决定。
