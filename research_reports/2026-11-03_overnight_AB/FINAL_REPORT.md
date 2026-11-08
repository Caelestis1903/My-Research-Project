# 完整实验报告：统一发表模型比较与 C_MIX＋P01 组合

> Main-branch report archive created 2026-11-08. Source branch: research-plan/26-11-03-overnight-ab-auc. Supporting CSV/JSON files, figures, execution artifacts and implementation files remain on the source branch; relative links below resolve there. No source code or raw training artifacts were copied. Sections outside the current research scope are omitted from this main-branch archive; source-branch originals are unchanged.

完成日期：2026-11-04。实验 `MDMT_OVERNIGHT_AB_20261103`，评价协议 `AUC_SELECTED_V1`，数据协议 `FULLTRAIN_NATURAL80_10_10`。**状态 FINAL_COMPLETE：A端点30/30，B端点40/40，新增30条训练、1600epoch，缺项0。**

**CP的合格比较是MovieLens对HyperGate/AutoIFS，以及KuaiRand对M3 fixed1。MovieLens对M3的AUC增益为+.002335，低于+.003；KuaiRand对HyperGate/AutoIFS的AUC均差分别−.003945/−.000429。CP相对P01在两数据的AUC均值都下降，完整2×2没有支持把CP升级为排序优先组合。** LL及域任务取舍分别保留：ML对固定EMA的LL五seed全降；KR对HG的LL五seed全降，但对AutoIFS的Macro LL上升。

## 1. 完成范围、审计与实验身份

A使用两个数据集×三个发表模型×2022–2026，共30个合法online selected端点：M3复用10条近期A0 online轨迹，HG/AutoIFS新增各10条。B使用A0/C_MIX/P01/CP的固定EMA完整2×2，共40个槽位，复用30条、新训练CP10条。A0 online/EMA同源轨迹按身份去重；核心为60条逻辑轨迹，30复用、30新增。历史Gaussian固定EMA端点作为内部参照保留。默认CG组合关闭，未启动新增替代训练。

旧10/29三模型30条因训练人口/预处理不一致，全部不进入新主表。真实审计发现13条可用历史同epoch端点重选，17条缺选中状态，均保留在独立历史分类中。[A审计](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/A_REUSE_AUDIT.csv)、[B审计](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/B_REUSE_AUDIT.csv)、[旧A审计](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/LEGACY_A_AUDIT.md)记录路径、checksum、选点身份和原因。最新五seed补评输入按本地真实完成状态接入，控制组未重训。

数据为真实自然人口、原行序80/10/10划分、train-only编码和缩放，无域重采样。MovieLens任务为rating>3、rating>4，域按固定年龄规则划分；KuaiRand任务为is_click/long_view，域为tab0/1/4。原始CSV及六数组、row_id、encoder/scaler哈希绑定在DATA_MANIFEST及身份记录中。

所有模型最终阶段完整50epoch；AutoIFS另含search10，E10/E50均指reuse阶段。每估计器独立按validation六格等权Macro AUC严格最大且精确并列最早选点；V/Test绑定同一model hash。配置与矩阵先冻结，逐seed选点封存后自动test；历史控制使用既有同一test人口。主裁定用五seed配对均值：ΔAUC≥+.003，六格各≥−.001，ΔLL<0独立判断。SD为ddof1，配对t95%用df4系数2.776445；方向数和CI作为描述。

## 2. 发表模型及统一适配

| 正式模型 | 发表登记 | 本轮适配与固定来源 |
|---|---|---|
| M3oE fixed1 | *M3oE: Multi-Domain Multi-Task Mixture-of Experts Recommendation Framework*，SIGIR2024，[DOI](https://doi.org/10.1145/3626772.3657686) | frozen history.make_model；融合系数全1、无DARTS；embedding16、4共享专家、FCN[512,256,256,64] |
| HyperGate | *HyperGate: Hierarchical Perceptive Gating Network for Multi-domain Multi-task Recommendation*，WWW2025 Companion，[DOI](https://doi.org/10.1145/3701716.3715236) | 上游fdc82e3，RecStudio f08cca6；4专家、expert[256,256]、gate[256]、tower[256,128]、dropout.3，三门及原生辅助目标保留；超参来源标记UNVERIFIED_PROPOSAL |
| AutoIFS | *Automated Information Flow Selection for Multi-scenario Multi-task Recommendation*，WSDM2026，[DOI](https://doi.org/10.1145/3773966.3777992) | 上游65fc27c；MLP[1024,512,256]，rank ML2/KR16；原生search、rewind、selector移植、hard ticket和reuse；保留已审查的四处bias接口修正 |

M3/CP的batch40960、lr=.01、coupled wd ML1e−4/KR1e−5；HG的batch1024、lr=.001、wd0；AutoIFS batch4096，search base lr=.0003/wd3e−5，gate学习率为base×.01，reuse新Adam lr=.0001/wd3e−4。全部FP32、无AMP/TF32；发表模型保留原生目标、优化器分组及阶段随机流。模型预算、batch和更新数差异单独计成本。M3的online端点来自同时收集非反馈EMA的旧轨迹，成本保留co-collected shadow事实。

CP只投影P01分离出的真实含L2分量u；v由未投影混合梯度b²更新，PAIR条件历史H独立加入分子。共享和ordinary参数各更新一次，EMA一次。PAIR λ=.05、2048对、静态选择；C tau=.1、kappa=1、gamma=.2。完整数学、共享参数清单及源码对应见[设计](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/DESIGN_SPECIFICATION.md)和[03规范](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_plans/26-11-03_overnight_AB/03_TASK_B_IMPLEMENTATION.md)。

## 3. 统一selected test主表

均值±样本SD，n=5。前三行为发表模型，CP为候选方法。

| 数据 | 模型/估计器 | Macro AUC | Macro LL | selected epoch（2022→2026） |
|---|---|---:|---:|---|
| movielens | M3 fixed1 online | 0.775190 ± 0.001136 | 0.494891 ± 0.004690 | 29,17,24,26,17 |
| movielens | HyperGate online | 0.771412 ± 0.001267 | 0.498554 ± 0.002287 | 5,6,5,4,7 |
| movielens | AutoIFS online | 0.772425 ± 0.000868 | 0.496094 ± 0.001845 | 18,10,17,19,12 |
| movielens | CP fixedEMA | 0.777525 ± 0.000603 | 0.494212 ± 0.001505 | 21,20,21,21,19 |
| kuairand | M3 fixed1 online | 0.671907 ± 0.006145 | 0.437172 ± 0.002751 | 44,18,40,50,43 |
| kuairand | HyperGate online | 0.682113 ± 0.001449 | 0.440590 ± 0.003848 | 21,10,43,28,30 |
| kuairand | AutoIFS online | 0.678597 ± 0.003397 | 0.432992 ± 0.003248 | 6,32,2,6,39 |
| kuairand | CP fixedEMA | 0.678168 ± 0.003205 | 0.434505 ± 0.002073 | 34,39,50,45,40 |

CP分别对三个发表模型的配对比较：

| 数据 | 参照 | ΔAUC [配对t95%] | AUC正seed | ΔLL | LL降seed | 宏AUC标准 | 六格保护 |
|---|---|---|---:|---:|---:|---|---|
| movielens | M3 online | +0.002335 [+0.000242,+0.004428] | 5/5 | -0.000679 | 1/5 | 未达 | 满足 |
| movielens | HG | +0.006113 [+0.004974,+0.007253] | 5/5 | -0.004342 | 5/5 | 达标 | 满足 |
| movielens | AI | +0.005100 [+0.003645,+0.006556] | 5/5 | -0.001882 | 5/5 | 达标 | 满足 |
| kuairand | M3 online | +0.006261 [-0.000677,+0.013200] | 5/5 | -0.002667 | 4/5 | 达标 | 满足 |
| kuairand | HG | -0.003945 [-0.008744,+0.000855] | 1/5 | -0.006085 | 5/5 | 未达 | 未满足 |
| kuairand | AI | -0.000429 [-0.007579,+0.006720] | 2/5 | +0.001513 | 1/5 | 未达 | 未满足 |

**STRONG：ML对HG/AI的本协议优势。** CP对两模型AUC/LL均5/5改善，六格AUC均差全部正。相对M3同样5/5 AUC为正，均值+.002335248低于规定效应门槛；总LL均值下降主要由seed2022贡献，其余四seed上升。

**MEDIUM：KR对M3的总收益。** AUC五seed全正，均值+.006261247，六格保护满足，LL均值−.002667161、4/5下降。CP对HG的AUC为4/5下降，LL却5/5下降；对AI的AUC均值−.000429159、LL+.001512835。三个参照各自独立裁定。

![逐seed增益、均值与配对t95%](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/figures/paired_selected.png)

黑菱形为五seed均差，误差线为描述性t95%，彩点对应seed。AUC绿虚线为+.003；LL以负值为改善。

## 4. 总收益、组件增量与组合交互

CP fixedEMA相对A0 online、A0 fixedEMA、单组件及既有Gaussian的结果如下；内部参照按消融/控制命名。

| 数据 | 内部参照 | ΔAUC | ΔLL | AUC正seed | LL降seed |
|---|---|---:|---:|---:|---:|
| movielens | A0 EMA | -0.001513597 | -0.001927772 | 1/5 | 5/5 |
| movielens | C | -0.000917604 | -0.000499362 | 1/5 | 3/5 |
| movielens | P | -0.000889861 | -0.001339842 | 1/5 | 4/5 |
| movielens | Gaussian | -0.001831685 | -0.001610477 | 0/5 | 5/5 |
| kuairand | A0 EMA | +0.001644848 | -0.001441819 | 2/5 | 3/5 |
| kuairand | C | +0.002541401 | -0.000889605 | 3/5 | 3/5 |
| kuairand | P | -0.002440878 | -0.000963701 | 2/5 | 2/5 |
| kuairand | Gaussian | +0.000168557 | +0.000221775 | 2/5 | 2/5 |

**MEDIUM：组合未提高P01的平均排序结果。** CP−P的ML均差−.000889861、4/5下降，区间[−.001725837,−.000053886]；KR均差−.002440878、3/5下降，区间[−.008839985,+.003958229]。KR相对C的+.002541401低于+.003，d0t0均差−.001909228。当前选择不把CP替换P01作为排序优先配置；KR对M3的完整方案资格保留。

2×2交互以每seed的 `I=CP−C−P+A0_EMA` 计算，selected的AUC交互ML−.000293869（SD .000762199）、KR−.001544325（SD .013659958）；LL交互分别+.000088568/−.000411488。**WEAK：交互符号和方向机制。** 配对区间均跨0，未观察到可靠的正向叠加信号；交互不设性能门。完整四格和逐seed交互见[FACTORIAL_SEED.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/FACTORIAL_SEED.csv)、[FACTORIAL_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/FACTORIAL_SUMMARY.csv)。

## 5. 域任务、人口、traffic/pooled与最差格

![六格AUC与LL均差](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/figures/six_cell_deltas.png)

每格为CP−所列参照的五seed均差，单位1e−3；黑框表示AUC均差低于−.001。完整绝对值与所有seed见[数值附录](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/NUMERICAL_APPENDIX.md)及[ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv)。

| 数据/域 | train行数 | test行数 | M3任务0/1 AUC | CP任务0/1 AUC |
|---|---:|---:|---|---|
| movielens d0 | 155949 | 27687 | 0.770298 / 0.785173 | 0.772428 / 0.788066 |
| movielens d1 | 318599 | 41434 | 0.778130 / 0.797449 | 0.780659 / 0.799565 |
| movielens d2 | 325619 | 30900 | 0.752263 / 0.767825 | 0.755160 / 0.769271 |
| kuairand d0 | 172782 | 2613 | 0.635617 / 0.645909 | 0.636471 / 0.662738 |
| kuairand d1 | 1743898 | 246284 | 0.698833 / 0.726663 | 0.702990 / 0.731046 |
| kuairand d2 | 89452 | 1870 | 0.653260 / 0.671157 | 0.656792 / 0.678971 |

**STRONG：取舍随参照改变。** KR相对A0 fixedEMA，CP的d1两格AUC为−.002316/−.002344，traffic AUC−.002215920且5/5下降，pooled−.002126434；宏增量+.001644848由d0t1及d2两格支撑，d0t0−.000520。相对HG/AI时，CP的最大域d1两格均改善，traffic AUC分别+.003106521/+.003022009，但相对HG的d0两格为−.023043/−.008010，相对AI的d0t0和d2两格为−.005258/−.004360/−.004436。因此相同方法可在不同对照中呈现相反的域收益分配。

KR对AI的Macro LL+.001512835，但traffic LL−.005255473；对HG的Macro LL−.006084897、traffic LL−.012708207。ML对HG/AI的Macro LL下降也包含局部LL代价：d1t1对HG+.002282，d0t1对AI+.000929。LL按既定宏均值独立报告，六格AUC保护不扩展成额外LL门。

KR对EMA的任务0/1域等权AUC增量为−.000573605/+.003863300；相对M3则为+.002847393/+.009675101。ML对EMA的两个任务均下降约.00150/.00153。任务平均与traffic分别保留，防止单一宏值掩盖分布。

最差平均格ML为d2t0：M3 .752263170、EMA .756328580、CP .755159840；KR为d0t0：M3 .635616977、EMA .636990695、CP .636470632。KR逐seed先取最差格再平均则为M3 .634324636、CP .636470632，口径另列。

## 6. 训练阶段、loss与矛盾证据

CP固定EMA的同阶段对比：

| 数据 | 阶段 | 总ΔAUC | 总ΔLL | 对EMA ΔAUC | 对EMA ΔLL | 对P ΔAUC |
|---|---|---:|---:|---:|---:|---:|
| movielens | E10 | +0.001558 | -0.006304 | +0.000413 | +0.000884 | -0.000431 |
| movielens | selected | +0.002335 | -0.000679 | -0.001514 | -0.001928 | -0.000890 |
| movielens | E50 | +0.004866 | +0.001076 | +0.000079 | -0.002225 | -0.000105 |
| kuairand | E10 | -0.008379 | +0.007212 | +0.000747 | -0.006106 | -0.000737 |
| kuairand | selected | +0.006261 | -0.002667 | +0.001645 | -0.001442 | -0.002441 |
| kuairand | E50 | +0.005669 | -0.005760 | -0.000021 | -0.000048 | -0.003352 |

ML的E50总AUC+.004865916达到效应标准，LL+.001075792；主selected则AUC低于标准、LL均值下降。KR E10总AUC−.008379249，到selected/E50转正。E10/E50为固定阶段诊断，主裁定使用validation-selected。每seed完整阶段差见[STAGE_EFFECTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/STAGE_EFFECTS.csv)。

CP在线训练的真实BCE在E10→E50由ML .475060降到.418593，KR .519271降到.489358；E50混合目标分别.434385/.514724。PAIR辅助项包含在训练objective中，test LL采用统一未缩放定义。训练损失下降与selected排序优势分开评价，原值见[B_TRAINING_LOSS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/B_TRAINING_LOSS.csv)。

**STRONG：新旧协议的模型排序不同。** 旧10/29表的ML HG/AI相对M3 AUC分别+.007379/+.003551；本轮分别−.003777913/−.002765080。新表改变了完整自然train及train-only拟合、M3 lr/50epoch预算、AUC选点窗口和HG/AI固定训练阶段。数据/预处理、优化与选点共同变化，单一原因尚未被分离。旧表与新表按各自协议存放。

## 7. 机制证据与下一项判别实验

**STRONG：C实际改变了共享当前项。** CP全部17250步日志中，ML有38.7%–45.3%步骤projection_distance>0，KR18.0%–47.4%；平均预条件坐标范数比约ML .99027–.99261、KR .99377–.99808。[B_DIAGNOSTICS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/B_DIAGNOSTICS.csv)逐seed保存统计。固定M3无BatchNorm/Dropout，probe的autograd没有累加进p.grad。

**STRONG：ML概率损失与排序存在重复取舍。** 相对EMA的LL 5/5下降、均差−.001927772、区间[−.002759749,−.001095795]；相对Gaussian为AUC 5/5下降、LL 5/5下降。**MEDIUM：KR总体排序收益可重复但幅度异质。** 对M3五seedAUC均正，SD .005588025，区间[−.000677202,+.013199696]；按冻结规则总收益合格。

竞争解释包括：固定EMA/选点贡献较多总收益；C投影同时改变方向与范数；PAIR进入v后改变修复几何；不同域任务的排序/概率损失权衡。日志未保存real-moment/H相对范数、v范数及最终update·probe，故“H抵消C”等定向机制解释维持**WEAK**，记作Relevant机制证据缺项。

最有区分力的下一实验是CP范数匹配控制：PAIR、v/h、EMA及探针/RNG保持，沿u方向缩放到C投影后的相同预条件范数，以五seed与CP/P01配对。若它复现CP的LL/排序变化，方向约束的独立价值解释被削弱。本轮矩阵已经完成，不追加该实验。

研究取舍：排序优先时不以CP替换已有P01；保留ML对HG/AI、KR对M3的已达标比较，以及ML/KR相应LL与大流量域代价。后续机制试验按上述可证伪问题单独预注册。

## 8. 检查、修订与实际成本

设计→独立实现→独立review PASS WITH FIXES→六项真实原batch smoke→完整RUN→本分析均完成。15项检查包括CP的m/v/h数值、一次更新/L2/EMA、原生分支、HG目标、AutoIFS rewind/selector/fresh optimizer、RAM恢复和真实双卡CUDA RNG流映射。跨卡RNG、直接run-job认领与训练后评价恢复路径在正式训练前修复；记录见[审查](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/REVIEW_REPORT.md)和各coder报告。

attempt从00:30:08建队列到03:47:53最终汇总（北京时间2026-11-04），墙钟11864.184秒。训练失败0、状态丢失重放0，另有12次WAIT_RESOURCE owner资源门禁拒发，随后正常派发。1600个epoch提交身份唯一且连续。全部端点封存后，协调器和本attempt guardian正常退出，RAM会话释放。

| 数据 | 新训练模型 | n | 累计worker秒 | batch/全局更新数 | 发表模型Adam调用 | 标签暴露 | 单worker峰值allocated MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| movielens | HyperGate | 5 | 12460.301 | 195500 | 195500 | 400083500 | 69.1 |
| movielens | AutoIFS | 5 | 5984.488 | 58800 | 68600 | 480100200 | 157.9 |
| movielens | CP | 5 | 2136.643 | 5000 | 共享自定义＋ordinary Adam | 400083500 | 1037.5 |
| kuairand | HyperGate | 5 | 53326.508 | 490000 | 490000 | 1003066000 | 90.1 |
| kuairand | AutoIFS | 5 | 35779.360 | 147000 | 171500 | 1203679200 | 351.8 |
| kuairand | CP | 5 | 5707.918 | 12250 | 共享自定义＋ordinary Adam | 1003066000 | 1170.8 |

30条逻辑轨迹worker时间合计115395.220秒，1600次物理epoch launch合计119283.414秒；全局batch/参数更新908550。A的Adam调用925600包含AutoIFS search每批两个互斥optimizer；CP另有17250次自定义共享＋ordinary更新。CP细分训练7326.894秒、验证517.664秒；A只记录训练+验证合计。预算、batch、阶段及共享负载不同，成本按原记录口径列示。[NEW_TRAINING_COST_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/NEW_TRAINING_COST_SUMMARY.csv)、[COSTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/COSTS.csv)和[运行报告](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/RUNNER_REPORT.md)保留完整账目。

## 9. 数值、代码与复建入口

- [完整逐seed六格附录](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/NUMERICAL_APPENDIX.md)、[ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv)：全部660行V/Test×selected/E10/E50端点，含epoch、model hash、协议、配置、来源checksum；110行selected test覆盖所有登记估计器视图。
- [PAIRED_SEED.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/PAIRED_SEED.csv)、[PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/PAIRED_SUMMARY.csv)、[DECISIONS.json](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/DECISIONS.json)：配对原值、SD/区间和冻结规则裁定；[ABSOLUTE_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/ABSOLUTE_SUMMARY.csv)为绝对统计。
- [VALIDATION_CURVES.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/VALIDATION_CURVES.csv)、[FACTORIAL_SEED.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/FACTORIAL_SEED.csv)、[FACTORIAL_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/FACTORIAL_SUMMARY.csv)：完整选点曲线和2×2交互。
- [B完整阶段报告](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/B_REPORT.md)、[ANALYSIS_VALUES.json](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/ANALYSIS_VALUES.json)、[B_ANALYSIS_VALUES.json](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/B_ANALYSIS_VALUES.json)：研究解释与全部派生原值。
- [源码](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/experiments/overnight_ab_20261103)、[冻结入口](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_plans/26-11-03_overnight_AB/00_EXECUTION_RUNBOOK.md)、[设计](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/DESIGN_SPECIFICATION.md)：实现与实验合同。
- [原始标量归档与复建](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/RAW_ARTIFACTS.md)、[清单](https://github.com/Caelestis1903/My-Research-Project/blob/research-plan/26-11-03-overnight-ab-auc/research_reports/2026-11-03_overnight_AB/RAW_ARTIFACTS_MANIFEST.json)：原始运行记录、历史复用输入和逐文件/分块校验。图表可直接由本目录`analyze.py`从已发布ENDPOINTS.csv复建。

训练源码基于main 92a592418d3f11a588f1e628fbe9364ceab71e18，冻结规划c551dd95bb5c24fda0d5c2c1f79481bded23482c；每家族实际源码锁、数据身份与环境记录随归档保存。发布增量保留main新增的652ee0235fec2d0d4cfc67b4a6ef9977519619bf五seed补评提交，再合入原规划分支。

