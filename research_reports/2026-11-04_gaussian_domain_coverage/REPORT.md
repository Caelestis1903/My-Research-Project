# KuaiRand Gaussian领域覆盖：五seed完整研究报告

> Main-branch report archive created 2026-11-08. Source branch: research/26-11-04-gaussian-domain-coverage. Supporting CSV/JSON files, figures, execution artifacts and implementation files remain on the source branch; relative links below resolve there. No source code or raw training artifacts were copied.

完成日期：2026-11-04。状态 FINAL_COMPLETE：新增10/10条轨迹、500/500epoch、24500次Adam及EMA更新；全部最终评价完成。主结果为固定EMA的validation六格Macro AUC最高、精确并列最早checkpoint的test指标。seeds为2022–2026。

**当前选择：排序优先保留G-D0，概率损失优先保留G-D2；不优先追加G-D02实验。** G-D0的Macro AUC为0.679180，比G-D2高0.001181；G-D02为0.674604，比G-D0低0.004576。三种Gaussian均低于HyperGate的0.682113，均未达到本轮候选算法或强算法标准。G-D0对M3oE的总AUC收益为+0.007274，六格保护满足，保留这项完整方案收益。

MovieLens的既有GAUSSIAN_A1/fixed_ema已核对为G-D0，2022–2026五seed全部复用，本次补充新增训练0条。来源、逐seed结果及发表模型比较见[MovieLens补充报告](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md)，两数据集的六格AUC/LogLoss与各自超参数见[G-D0双数据集对比](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/G_D0_TWO_DATASETS.md)。下文保留KuaiRand原实验范围及其10条新增训练计数。

## 1. 统一协议与完成范围

仅改变Gaussian增强覆盖集合：[0]、[0,2]；[2]直接复用历史五seed。原fixed1 M3oE、1614254参数、embedding16、4共享专家、FCN[512,256,256,64]；Gaussian κ=.1、λ=.5、固定EMA ρ=.995、普通Adam lr=.01、coupled wd=1e-5、betas=(.9,.999)、eps=1e-8、batch40960、FP32、50epoch保持冻结。原GA31-R3局部Generator和逐字段RMS detach规则保留；单域直接调用原函数，双域分别产生视图并统一计算 (ΣLreal + .5Σselected(Laug−Lreal))/(2B)。每batch一次backward、Adam和EMA更新。

数据协议FULLTRAIN_NATURAL80_10_10：原行序80/10/10、train-only编码和缩放，先切分再筛tab0/1/4，完整自然人口，不重采样。train共2006132行，validation/test各250767行，每epoch49步。任务0=is_click，任务1=long_view。

| 域 | 原tab | train N | validation N | test N | V正例 t0/t1 | test正例 t0/t1 |
| --- | --- | --- | --- | --- | --- | --- |
| d0 | 0 | 172782 | 2692 | 2613 | 245/101 | 228/120 |
| d1 | 1 | 1743898 | 246232 | 246284 | 51644/27460 | 51847/27415 |
| d2 | 4 | 89452 | 1843 | 1870 | 1063/829 | 1100/827 |

复用150个端点（G-D2、固定EMA、M3oE、HyperGate、AutoIFS各五seed×三个checkpoint×V/Test），新增120个端点（两臂×五seed×online/EMA×三个checkpoint×V/Test），合计270。A0 online与固定EMA共享原训练轨迹；本轮复用20条历史轨迹的5个配置视图，新增10条轨迹。正式模型结果来自11月4日[A+B统一协议](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)。固定EMA属于内部控制。全部数据、配置及来源文件checksum见[RUN_MANIFEST.json](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RUN_MANIFEST.json)。

## 2. 五seed主结果与性能裁定

均值±样本SD，n=5。Macro为六格等权平均，Traffic为按test人口加权的cell指标，Pooled为各任务合并域后计算的指标再平均。所有指标的均值、SD、区间见[ABSOLUTE_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/ABSOLUTE_SUMMARY.csv)。

| 配置/模型 | Macro AUC | Macro LogLoss | Traffic AUC | Traffic LL | Pooled AUC | Pooled LL |
| --- | --- | --- | --- | --- | --- | --- |
| G-D0 | 0.679180 ± 0.003475 | 0.435518 ± 0.002752 | 0.715563 ± 0.003065 | 0.424828 ± 0.004252 | 0.720768 ± 0.003002 | 0.424828 ± 0.004252 |
| G-D02 | 0.674604 ± 0.003506 | 0.436172 ± 0.001675 | 0.717108 ± 0.002521 | 0.423602 ± 0.002474 | 0.722413 ± 0.002502 | 0.423602 ± 0.002474 |
| G-D2 | 0.677999 ± 0.005117 | 0.434283 ± 0.002164 | 0.713542 ± 0.004274 | 0.424782 ± 0.006431 | 0.718855 ± 0.004171 | 0.424782 ± 0.006431 |
| 固定EMA（内部控制） | 0.676523 ± 0.005227 | 0.435947 ± 0.001323 | 0.718165 ± 0.001129 | 0.423121 ± 0.004092 | 0.723348 ± 0.001113 | 0.423121 ± 0.004092 |
| M3oE fixed1 | 0.671907 ± 0.006145 | 0.437172 ± 0.002751 | 0.711621 ± 0.004825 | 0.426397 ± 0.007843 | 0.716990 ± 0.004704 | 0.426397 ± 0.007843 |
| HyperGate | 0.682113 ± 0.001449 | 0.440590 ± 0.003848 | 0.712842 ± 0.003019 | 0.433824 ± 0.003486 | 0.718065 ± 0.002873 | 0.433824 ± 0.003486 |
| AutoIFS | 0.678597 ± 0.003397 | 0.432992 ± 0.003248 | 0.712927 ± 0.001073 | 0.426371 ± 0.006287 | 0.718171 ± 0.001183 | 0.426371 ± 0.006287 |

性能等级以五seed平均Test Macro AUC相对最强发表模型HyperGate裁定：增益≥.003为强算法，≥.002为候选算法，>0且<.002为正向胜出。六格平均ΔAUC≥−.001独立裁定；LogLoss以平均差<0独立裁定。seed方向数和置信区间用于描述重复性，分类使用完整精度均值。

| Gaussian配置 | 对HyperGate ΔAUC | 性能等级 | 六格保护 | 对HyperGate ΔLL |
| --- | --- | --- | --- | --- |
| G-D0 | -0.002932373 | 未超过最强发表基线 | 不满足 | -0.005071886 |
| G-D02 | -0.007508582 | 未超过最强发表基线 | 不满足 | -0.004417283 |
| G-D2 | -0.004113262 | 未超过最强发表基线 | 不满足 | -0.006306671 |

G-D0相对AutoIFS的平均AUC高0.000583，对HyperGate仍差0.002932。G-D02对HyperGate、AutoIFS的AUC分别低0.007509、0.003993，均为5/5 seed下降。三种Gaussian相对HyperGate的平均LogLoss均下降；相对AutoIFS则均上升。完整分类及各比较对象的六格裁定见[DECISIONS.json](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/DECISIONS.json)。

## 3. 每seed绝对值与validation选中epoch

以下为主估计器；三种Gaussian/固定EMA使用fixed_ema，三个发表模型使用online。每seed的全部六格AUC/LL、域/任务、Traffic/Pooled、epoch、model hash及V/Test同状态身份完整保存在[ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv)。

| 配置/模型 | seed | selected epoch | Test Macro AUC | Test Macro LL |
| --- | --- | --- | --- | --- |
| G-D0 | 2022 | 25 | 0.676894620 | 0.437744548 |
| G-D0 | 2023 | 44 | 0.674363944 | 0.439193356 |
| G-D0 | 2024 | 43 | 0.682976051 | 0.433233039 |
| G-D0 | 2025 | 38 | 0.680801706 | 0.433898094 |
| G-D0 | 2026 | 50 | 0.680864568 | 0.433520231 |
| G-D02 | 2022 | 48 | 0.679436092 | 0.434520812 |
| G-D02 | 2023 | 50 | 0.669616057 | 0.436861560 |
| G-D02 | 2024 | 36 | 0.674634116 | 0.434587581 |
| G-D02 | 2025 | 45 | 0.673984654 | 0.438509434 |
| G-D02 | 2026 | 33 | 0.675348927 | 0.436382898 |
| G-D2 | 2022 | 28 | 0.672436024 | 0.434603615 |
| G-D2 | 2023 | 22 | 0.678318871 | 0.437290047 |
| G-D2 | 2024 | 36 | 0.686041082 | 0.432189473 |
| G-D2 | 2025 | 47 | 0.678196076 | 0.435155898 |
| G-D2 | 2026 | 36 | 0.675004394 | 0.432176311 |
| 固定EMA（内部控制） | 2022 | 33 | 0.673472090 | 0.436204447 |
| 固定EMA（内部控制） | 2023 | 50 | 0.671147085 | 0.437828245 |
| 固定EMA（内部控制） | 2024 | 50 | 0.684871430 | 0.434500840 |
| 固定EMA（内部控制） | 2025 | 50 | 0.677389268 | 0.436329239 |
| 固定EMA（内部控制） | 2026 | 47 | 0.675735118 | 0.434870540 |
| M3oE fixed1 | 2022 | 44 | 0.673696016 | 0.439326044 |
| M3oE fixed1 | 2023 | 18 | 0.662912576 | 0.438937635 |
| M3oE fixed1 | 2024 | 40 | 0.679916345 | 0.433965079 |
| M3oE fixed1 | 2025 | 50 | 0.672610006 | 0.439262046 |
| M3oE fixed1 | 2026 | 43 | 0.670398052 | 0.434369215 |
| HyperGate | 2022 | 21 | 0.684181065 | 0.437831422 |
| HyperGate | 2023 | 10 | 0.682633383 | 0.435689350 |
| HyperGate | 2024 | 43 | 0.680227337 | 0.441762283 |
| HyperGate | 2025 | 28 | 0.681615204 | 0.445444303 |
| HyperGate | 2026 | 30 | 0.681905766 | 0.442221342 |
| AutoIFS | 2022 | 6 | 0.681692759 | 0.428069219 |
| AutoIFS | 2023 | 32 | 0.673516570 | 0.436171077 |
| AutoIFS | 2024 | 2 | 0.676996108 | 0.433046654 |
| AutoIFS | 2025 | 6 | 0.679448825 | 0.432043548 |
| AutoIFS | 2026 | 39 | 0.681330762 | 0.435629541 |

## 4. 配对收益：覆盖增量、总收益和组件增量

配对单位为seed；t95%区间使用df=4的2.776445系数，SD采用ddof=1。下表ΔAUC>0、ΔLL<0有利；保护列列出低于−.001的平均AUC格。全部逐seed配对原值和差见[PAIRED_SEED.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/PAIRED_SEED.csv)，均值/SD/区间见[PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/PAIRED_SUMMARY.csv)。

| 比较 | ΔAUC [t95%] | AUC正seed | ΔLL [t95%] | LL降seed | 保护越线格 |
| --- | --- | --- | --- | --- | --- |
| G-D0−G-D2 | +0.001180888 [-0.004339750, +0.006701527] | 3/5 | +0.001234785 [-0.000761728, +0.003231299] | 1/5 | d2t0,d2t1 |
| G-D02−G-D2 | -0.003395320 [-0.012500040, +0.005709400] | 2/5 | +0.001889388 [-0.000672928, +0.004451704] | 2/5 | d0t0,d0t1,d2t0,d2t1 |
| G-D02−G-D0 | -0.004576209 [-0.009799153, +0.000646736] | 1/5 | +0.000654603 [-0.003509200, +0.004818406] | 2/5 | d0t0,d0t1,d2t1 |
| G-D0−M3oE fixed1 | +0.007273579 [+0.002351596, +0.012195561] | 5/5 | -0.001654150 [-0.004354170, +0.001045870] | 4/5 | 无 |
| G-D0−固定EMA（内部控制） | +0.002657179 [-0.000645935, +0.005960293] | 4/5 | -0.000428808 [-0.002637538, +0.001779921] | 3/5 | d1t0,d1t1 |
| G-D0−HyperGate | -0.002932373 [-0.008749113, +0.002884366] | 1/5 | -0.005071886 [-0.013057189, +0.002913416] | 4/5 | d0t0,d0t1 |
| G-D0−AutoIFS | +0.000583173 [-0.004219639, +0.005385984] | 3/5 | +0.002525846 [-0.002984189, +0.008035881] | 1/5 | d2t0,d2t1 |
| G-D02−M3oE fixed1 | +0.002697370 [-0.003379458, +0.008774199] | 4/5 | -0.000999547 [-0.004250107, +0.002251013] | 3/5 | d0t0 |
| G-D02−固定EMA（内部控制） | -0.001919029 [-0.009160261, +0.005322202] | 1/5 | +0.000225795 [-0.001791880, +0.002243469] | 2/5 | d0t0,d0t1,d1t0 |
| G-D02−HyperGate | -0.007508582 [-0.011559390, -0.003457773] | 0/5 | -0.004417283 [-0.008737688, -0.000096879] | 4/5 | d0t0,d0t1,d2t1 |
| G-D02−AutoIFS | -0.003993036 [-0.006125971, -0.001860101] | 0/5 | +0.003180449 [-0.000558651, +0.006919549] | 0/5 | d0t0,d0t1,d2t0,d2t1 |
| G-D2−M3oE fixed1 | +0.006092690 [-0.001338440, +0.013523821] | 4/5 | -0.002888935 [-0.004656749, -0.001121121] | 5/5 | d0t0 |
| G-D2−固定EMA（内部控制） | +0.001476291 [-0.002649498, +0.005602080] | 3/5 | -0.001663593 [-0.002737583, -0.000589603] | 5/5 | d0t0,d1t0,d1t1 |
| G-D2−HyperGate | -0.004113262 [-0.012090056, +0.003863533] | 1/5 | -0.006306671 [-0.012891680, +0.000278337] | 4/5 | d0t0,d0t1 |
| G-D2−AutoIFS | -0.000597716 [-0.010020603, +0.008825172] | 2/5 | +0.001291061 [-0.003435527, +0.006017648] | 2/5 | d0t0,d2t0 |

| seed | D0−D2 ΔAUC / ΔLL | D02−D2 ΔAUC / ΔLL | D02−D0 ΔAUC / ΔLL |
| --- | --- | --- | --- |
| 2022 | +0.004458596 / +0.003140934 | +0.007000068 / -0.000082803 | +0.002541471 / -0.003223736 |
| 2023 | -0.003954927 / +0.001903309 | -0.008702813 / -0.000428488 | -0.004747887 / -0.002331797 |
| 2024 | -0.003065031 / +0.001043566 | -0.011406965 / +0.002398109 | -0.008341935 / +0.001354542 |
| 2025 | +0.002605630 / -0.001257804 | -0.004211422 / +0.003353535 | -0.006817052 / +0.004611340 |
| 2026 | +0.005860173 / +0.001343921 | +0.000344533 / +0.004206587 | -0.005515641 / +0.002862667 |

**STRONG：G-D0相对M3oE的整体排序收益。** AUC均差+0.007273579，5/5正，t95%[+0.002351596,+0.012195561]，六格平均全部上升；平均LL下降0.001654150，4/5下降。G-D2对M3oE的既有平均AUC+0.006092690、LL−0.002888935仍保留，d0t0低于−.001保护线。G-D02对M3oE的AUC+0.002697370，d0t0下降0.006645151。

**MEDIUM：双域相对D0的宏排序下降。** D02−D0为−0.004576209，4/5下降；平均LL+0.000654603。区间[−0.009799153,+0.000646736]保留seed异质性。双域相对D2为AUC−0.003395320、LL+0.001889388，未取得Macro指标的覆盖扩展收益。

**WEAK：D0替代D2的稳定宏增量。** D0−D2的AUC+0.001180888、3/5正，LL+0.001234785、4/5恶化；共同E50的AUC差缩至+0.000124049。当前AUC均值最优为D0，D2保留更低的Macro LL。

## 5. 六个域任务、最差格与性能取舍

下表为五seed均值，每格格式AUC / LogLoss；逐格SD在ABSOLUTE_SUMMARY.csv。

| 配置/模型 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
| --- | --- | --- | --- | --- | --- | --- |
| G-D0 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 |
| G-D02 | 0.628972 / 0.290649 | 0.648625 / 0.182832 | 0.704167 / 0.503186 | 0.732477 / 0.344576 | 0.656041 / 0.650127 | 0.677342 / 0.645664 |
| G-D2 | 0.634346 / 0.289394 | 0.661911 / 0.180681 | 0.700652 / 0.504592 | 0.728460 / 0.345690 | 0.658848 / 0.645452 | 0.683779 / 0.639891 |
| 固定EMA（内部控制） | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 |
| M3oE fixed1 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 |
| HyperGate | 0.659514 / 0.289436 | 0.670749 / 0.182902 | 0.699529 / 0.514035 | 0.727857 / 0.354503 | 0.656476 / 0.654395 | 0.678551 / 0.648268 |
| AutoIFS | 0.641729 / 0.286408 | 0.657481 / 0.179697 | 0.700239 / 0.507256 | 0.727575 / 0.346357 | 0.661152 / 0.640774 | 0.683406 / 0.637460 |

| 比较 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
| --- | --- | --- | --- | --- | --- | --- |
| G-D0−G-D2 | +0.009575 / -0.001742 | +0.000898 / +0.000356 | +0.001799 / +0.000031 | +0.002262 / +0.000011 | -0.003245 / +0.004208 | -0.004205 / +0.004545 |
| G-D02−G-D2 | -0.005374 / +0.001256 | -0.013285 / +0.002151 | +0.003515 / -0.001405 | +0.004017 / -0.001113 | -0.002807 / +0.004675 | -0.006437 / +0.005773 |
| G-D02−G-D0 | -0.014949 / +0.002998 | -0.014183 / +0.001795 | +0.001715 / -0.001436 | +0.001755 / -0.001124 | +0.000437 / +0.000467 | -0.002232 / +0.001227 |
| G-D0−HYPERGATE | -0.015593 / -0.001784 | -0.007940 / -0.001865 | +0.002923 / -0.009413 | +0.002866 / -0.008803 | -0.000872 / -0.004735 | +0.001022 / -0.003831 |
| G-D02−HYPERGATE | -0.030542 / +0.001214 | -0.022123 / -0.000070 | +0.004638 / -0.010849 | +0.004620 / -0.009927 | -0.000435 / -0.004268 | -0.001209 / -0.002604 |

**STRONG：双域对D0的d0t0损失重复出现。** D02−D0的d0t0均差−0.014949244、5/5下降，t95%[−0.029018980,−0.000879508]；d0t1为−0.014183380、4/5下降。D02相对D2的d2t1也为5/5下降（均差−0.006436994）。目标域被直接增强后，排序收益仍随域任务变化。

增强d0相对G-D2的d0t0/d0t1平均改善+0.009575380/+0.000898182，分别4/5、2/5 seed为正；但相对HyperGate仍落后−0.015592887/−0.007939898。双域这两格相对HyperGate的缺口扩大到−0.030542131/−0.022123279。

未直接增强的最大域d1也发生变化。相对G-D2，D0的两任务AUC为+0.001799451/+0.002262097，D02为+0.003514683/+0.004016655。D02相对D0进一步提高d1约0.0017，却主要损失d0。其Traffic AUC比D0高0.001545409、Traffic LL低0.001225706；六格Macro AUC却低0.004576209，Macro LL高0.000654603。test中d1占98.2123%，解释了两种汇总权重下的方向差异。

**MEDIUM：相对固定EMA的小域收益/大域代价。** G-D0相对固定EMA的d0两格+0.006930376/+0.012862348，d1两格−0.002854581/−0.002666910，后两格均5/5下降。G-D0的Traffic AUC比固定EMA低0.002602037。相对G-D2，双域呈现d1改善、d0及d2多数格下降。取舍方向随比较对象变化。

| 配置 | 任务0 Macro AUC | 任务1 Macro AUC | 任务0 Macro LL | 任务1 Macro LL | 每seed最差格AUC的均值 |
| --- | --- | --- | --- | --- | --- |
| G-D0 | 0.667325 | 0.691035 | 0.480645 | 0.390391 | 0.643921 |
| G-D02 | 0.663060 | 0.686148 | 0.481321 | 0.391024 | 0.628972 |
| G-D2 | 0.664615 | 0.691383 | 0.479812 | 0.388754 | 0.634346 |

G-D0相对D2的任务0 AUC+0.002710107、任务1−0.000348330；D02相对D0两任务均下降（−0.004265568/−0.004886849）。三种Gaussian的平均最差格均为d0t0，D0提高至0.643921，D02降至0.628972。

**WEAK：域规模作为主要因果解释。** d0训练人口约为d2的1.93倍，M3oE的d0两任务AUC却更低；HyperGate下最小平均格转为d2t0。域身份、标签分布、特征支持、增强曝光量与N共同变化。d0t1的validation/test正例仅101/120；本轮保持人口固定，直接识别覆盖集合的影响，未分离N的独立贡献。

## 6. 训练阶段、验证/测试与在线模型

E10和E50为冻结阶段诊断，主裁定使用validation-selected。完整阶段配对见[STAGE_EFFECTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/STAGE_EFFECTS.csv)，所有种子及在线对照见[ANALYSIS_VALUES.json](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/ANALYSIS_VALUES.json)。

| 阶段 | 配置 | Test Macro AUC | Test Macro LL |
| --- | --- | --- | --- |
| E10 | G-D0 | 0.647112 | 0.448465 |
| E10 | G-D02 | 0.648271 | 0.451275 |
| E10 | G-D2 | 0.644901 | 0.446311 |
| selected | G-D0 | 0.679180 | 0.435518 |
| selected | G-D02 | 0.674604 | 0.436172 |
| selected | G-D2 | 0.677999 | 0.434283 |
| E50 | G-D0 | 0.678936 | 0.434961 |
| E50 | G-D02 | 0.673888 | 0.436669 |
| E50 | G-D2 | 0.678812 | 0.434159 |

**MEDIUM：阶段均值方向发生反转。** D02−D0在E10为AUC+0.001159284，到selected为−0.004576209、E50为−0.005048548。D02−D2在E10为+0.003370022，E50转为−0.004924499。D0−D2在selected的正均值，在E50接近零；选点与训练阶段对该微小增量的贡献需要共同解释。

validation-selected：D0−D2的V均差+0.002976529、test+0.001180888；D02−D2的V均差+0.000736884、test−0.003395320。D02−D0则V/test均负（−0.002239645/−0.004576209），均4/5下降。seed2022的D02−D0 test为正，保留为宏趋势的反例。

| 估计器/阶段 | D02−D0 Test ΔAUC | Test ΔLL | V ΔAUC | V ΔLL |
| --- | --- | --- | --- | --- |
| online/selected | -0.003043207 | +0.002577731 | -0.003587402 | +0.000436320 |
| online/E50 | -0.007889939 | +0.004916048 | -0.001880486 | +0.003151320 |

**MEDIUM：双域代价在在线模型共同末轮已经出现。** E50 online的D02−D0为AUC−0.007889939、LL+0.004916048；LL为5/5恶化。固定EMA与各自选点可以改变幅度，训练轨迹本身也包含差异。

| 配置 | E10优化loss / clean BCE | E50优化loss / clean BCE |
| --- | --- | --- |
| G-D0 | 0.519679 / 0.519678 | 0.489461 / 0.489444 |
| G-D02 | 0.518888 / 0.518880 | 0.488713 / 0.488661 |

双域训练目标和clean BCE更低，同时selected与E50的Macro排序及概率损失劣于D0。竞争解释包括共享表示/Adam优化轨迹的非加性交互、EMA与选点、有限域任务评估支持和自然train/test分布差异。共享梯度方向、表示变化及噪声耦合的独立机制证据为WEAK，本轮日志保留损失/视图/状态身份，未新增机制干预。

## 7. 研究问题裁定与下一步

1. 同时增强d0+d2相对仅d2：Macro AUC下降0.003395、Macro LL上升0.001889，未提高本轮Macro主指标。
2. 双域相对仅d0：Macro AUC下降0.004576、Macro LL上升0.000655；优先D0。
3. D0相对D2改善d0、d1排序，损失d2两格；D02相对D2改善d1，损失d0与d2四格。
4. D0缓解d0相对HyperGate的差距，d0t0/t1仍落后0.015593/0.007940；D02扩大缺口。
5. d1虽未直接增强，两个任务随覆盖集合改变；相对D2两新配置均改善，相对固定EMA则仍下降。
6. 最强发表模型仍是HyperGate。D0超过AutoIFS的AUC均值0.000583，LL上升0.002526；D02低于HyperGate与AutoIFS。
7. 当前单域配置足以取得更好的Gaussian结果。AUC优先选D0，Macro LL优先选D2；不优先继续双域扩展。

最有区分力的下一项研究是固定G-D0/G-D2，在新增的完整自然留出批次进行同一五seed配对，判别D0的局部排序改善能否跨自然人口保留，同时检验其d2和LogLoss代价。它直接影响两个单域方案的取舍；本轮已完成规定矩阵，不追加新训练配置或超参数搜索。

## 8. 工程检查、修订与实际成本

DESIGN→独立CODER→独立REVIEW→RUN→ANALYZE完成。14项定向CPU检查、两臂batch40960 GPU smoke及RAM往返通过。审查结论PASS WITH FIXES。正式训练失败0、OOM/状态丢失重放0；513次派发中13次资源/owner门禁拒发，随后自动重试，500次epoch提交完整。guardian正常SHUTDOWN，RAM状态在评价后释放。

正式运行前修订两项恢复问题：RAM整体丢失重放采用追加账本中的最新连续提交链并保留旧物理记录；已封存DONE先校验checksum再释放残余RAM。源码、数据、算法公式、训练矩阵和选点规则保持冻结；记录见[CODER_IMPLEMENTATION_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/CODER_IMPLEMENTATION_REPORT.md)、[REVIEW_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/REVIEW_REPORT.md)及[RUNNER_REPORT.md](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RUNNER_REPORT.md)。

| 配置 | n | 训练秒合计 | 验证秒合计 | worker秒合计 | Adam/EMA步数 | 增强行曝光 | 峰值reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G-D0 | 5 | 6068.886 | 510.688 | 6579.575 | 12250 | 43195500 | 1392.0 |
| G-D02 | 5 | 6867.107 | 515.012 | 7382.121 | 12250 | 65558500 | 1388.0 |

首次派发至完成墙钟3761.7秒（62.70分钟）；逻辑worker时间13961.696秒，500次epoch物理worker时间16529.491秒。物理worker计时覆盖注册后的训练、RAM IPC与该launch评价；协调器墙钟覆盖完整队列。正式原始行曝光1003066000、标签曝光2006132000、增强视图108754000；每个臂Adam/EMA各12250步。历史复用成本标记为historical_reuse，新增训练时间为0。两张RTX3090与既有任务共享，逐卡按显存和功耗85%及峰值/RAM余量派发。环境Python3.13.5、PyTorch2.9.0+cu126、FP32；完整成本见[COSTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/COSTS.csv)。

## 9. 源码、原始记录及复算

原CSV SHA256：a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f。冻结实现SHA256：c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d。基于A+B提交9e682f3e3f608c7d63c2e9306b8657b16d822541；同一实现的源码归档提交de7947e01c49aaecbd9d438c5ef662de46cca8f9。初始训练基准HEAD与源码归档提交在[运行报告](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RUNNER_REPORT.md)区分记录。

实验分支：[research/26-11-04-gaussian-domain-coverage](https://github.com/Caelestis1903/My-Research-Project/tree/research/26-11-04-gaussian-domain-coverage)。实现见[实验包](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/experiments/gaussian_domain_coverage_20261104)，数学和接口合同见[DESIGN_SPECIFICATION.md](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/DESIGN_SPECIFICATION.md)，完整数据/源码/配置/选点身份见[RUN_MANIFEST.json](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RUN_MANIFEST.json)。

原始标量归档含1581个文件，见[RAW_ARTIFACTS.md](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RAW_ARTIFACTS.md)及[清单](https://github.com/Caelestis1903/My-Research-Project/blob/research/26-11-04-gaussian-domain-coverage/research_reports/2026-11-04_gaussian_domain_coverage/RAW_ARTIFACTS_MANIFEST.json)。checkpoint身份保留epoch、model hash、冻结validation与test端点；模型/optimizer/EMA状态仅驻RAM，评价后释放。运行本目录analyze.py可从已发布ENDPOINTS.csv复算阶段、在线/EMA、域任务及逐seed对比；build_report.py从这些冻结数表重建本报告与台账片段。
