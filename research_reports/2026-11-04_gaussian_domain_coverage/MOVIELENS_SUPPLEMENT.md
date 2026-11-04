# MovieLens G-D0五seed结果补齐

本次补齐复用MovieLens G-D0的5条既有轨迹及M3oE、HyperGate、AutoIFS的15条既有轨迹，新增训练0条。KuaiRand继续引用本分支已经完成的结果。两数据集各自n=5，seed固定为2022–2026。

MovieLens原GAUSSIAN_A1 / fixed_ema的增强域是d0，因此对应本分支命名G-D0。五seed均已完整训练50epoch；selected epoch依次为23、20、25、23、22。此次完成名称映射、来源核对和双数据集汇总。原KuaiRand的270行端点、训练记录和研究结论保留。

## 1. 配置、数据及选点身份

两数据集均为自然人口、原行序80/10/10划分、train-only预处理、fixed1 M3oE、batch40960、Adam lr=.01、λ=.5、50epoch、FP32。G-D0仅增强原域索引0，沿用各自源Gaussian随机流及损失预算。

| 项目 | MovieLens | KuaiRand |
| --- | --- | --- |
| G-D0来源 | AB表GAUSSIAN_A1 / fixed_ema | 本分支G-D0 / fixed_ema |
| 增强域 | 原年龄域d0 | 原tab0映射d0 |
| train总行数 | 800167 | 2006132 |
| train d0/d1/d2 | 155949 / 318599 / 325619 | 172782 / 1743898 / 89452 |
| validation / test行数 | 各100021 | 各250767 |
| Gaussian κ | .05 | .1 |
| coupled weight_decay | 1e-4 | 1e-5 |
| 固定EMA ρ | .9828205985452511 | .995 |
| 每epoch更新数 | 20 | 49 |

每seed按validation六个域任务Macro AUC的严格最高值选点，精确并列取最早epoch；对应test沿用同epoch、同model hash。来源审核覆盖原配置、源文件checksum、数据/模型身份、完整50epoch验证曲线及V/Test端点。逐项记录见[MOVIELENS_SOURCE_AUDIT.json](MOVIELENS_SOURCE_AUDIT.json)。三个发表模型使用[A+B统一协议结果](../2026-11-03_overnight_AB/FINAL_REPORT.md)，保留各自原生训练设置。

## 2. 双数据集G-D0对比

以下为五seed平均Test AUC，Macro LogLoss使用同一AUC-selected状态。六格的完整AUC/LogLoss、标准差和超参数同时保存在[G_D0_TWO_DATASETS.csv](G_D0_TWO_DATASETS.csv)，独立阅读表见[G_D0_TWO_DATASETS.md](G_D0_TWO_DATASETS.md)。

| 数据集 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC | Macro LogLoss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MovieLens | 0.774415 | 0.789658 | 0.782567 | 0.801390 | 0.756629 | 0.771482 | 0.779357 | 0.495822 |
| KuaiRand | 0.643921 | 0.662809 | 0.702451 | 0.730723 | 0.655604 | 0.679574 | 0.679180 | 0.435518 |

MovieLens G-D0的Macro AUC为0.779356602 ± 0.000393344，Macro LL为0.495822265 ± 0.001550074。KuaiRand分别为0.679180178 ± 0.003474732、0.435517854 ± 0.002752362。SD均为ddof=1。

数据集的任务、域定义和κ、weight_decay、ρ分别按各自源协议固定；跨数据集比较使用相对该数据集发表模型的配对增量。

## 3. MovieLens逐seed及发表模型比较

| seed | selected epoch | G-D0 Test Macro AUC | G-D0 Test Macro LL | 对M3oE ΔAUC | 对M3oE ΔLL |
| --- | --- | --- | --- | --- | --- |
| 2022 | 23 | 0.779117309 | 0.494864985 | +0.003685572 | -0.008337085 |
| 2023 | 20 | 0.779169675 | 0.497435308 | +0.005990658 | +0.004180968 |
| 2024 | 25 | 0.780018162 | 0.493898744 | +0.004218572 | +0.000565070 |
| 2025 | 23 | 0.779063832 | 0.495564772 | +0.003389265 | +0.002631529 |
| 2026 | 22 | 0.779414034 | 0.497347516 | +0.003550599 | +0.005616258 |

MovieLens四组配置的40行selected V/Test端点见[MOVIELENS_ENDPOINTS.csv](MOVIELENS_ENDPOINTS.csv)。所有域任务/总体指标的均值与SD见[MOVIELENS_ABSOLUTE_SUMMARY.csv](MOVIELENS_ABSOLUTE_SUMMARY.csv)，三个比较的逐seed原值及差见[MOVIELENS_PAIRED_SEED.csv](MOVIELENS_PAIRED_SEED.csv)，配对统计见[MOVIELENS_PAIRED_SUMMARY.csv](MOVIELENS_PAIRED_SUMMARY.csv)。

| MovieLens参照 | 参照Macro AUC | G-D0 ΔMacro AUC [配对t95%] | AUC正seed | ΔMacro LL | LL降seed | 六格保护 |
| --- | --- | --- | --- | --- | --- | --- |
| M3oE fixed1 | 0.775189669 | +0.004166933 [+0.002843384,+0.005490482] | 5/5 | +0.000931348 | 1/5 | 满足 |
| HyperGate | 0.771411757 | +0.007944846 [+0.006156237,+0.009733455] | 5/5 | -0.002731649 | 5/5 | 满足 |
| AutoIFS | 0.772424589 | +0.006932013 [+0.006069073,+0.007794954] | 5/5 | -0.000271961 | 3/5 | 满足 |

MovieLens最强发表模型是M3oE。G-D0对其平均AUC增益超过.003，六格平均增益均为正，按本分支幅度规则为强算法且满足六格保护；Macro LL平均上升0.000931348。性能分类、六格保护和LogLoss独立裁定，详见[MOVIELENS_DECISIONS.json](MOVIELENS_DECISIONS.json)。

KuaiRand最强发表模型为HyperGate，G-D0平均AUC仍低0.002932373，d0两格未满足保护线；对M3oE的整体平均AUC则提高0.007273579且六格保护满足。因此两数据集对最强模型的性能等级不同，同时保留各自在M3oE上的收益。

## 4. 六格、任务和训练阶段

MovieLens G-D0相对M3oE的六格AUC均差依次为+0.004116419、+0.004485174、+0.004436419、+0.003940951、+0.004365995、+0.003656641；两个任务在三个域的平均排序均改善。最小平均AUC格仍为d2t0（0.756629），该域是MovieLens最大训练域。

六格LL均差依次为+0.000787261、-0.000367619、+0.001185282、+0.004269054、-0.001114642、+0.000828753。d1t1的LL代价最大。Traffic AUC/LL为0.780599801/0.494372285，Pooled AUC/LL为0.780950041/0.494372285；相对M3oE的Traffic AUC提高0.004164924，Traffic LL上升0.001143658。

| MovieLens固定EMA阶段 | G-D0 Macro AUC | G-D0 Macro LL | 对同阶段M3oE ΔAUC | ΔLL |
| --- | --- | --- | --- | --- |
| E10 | 0.768154614 | 0.491252159 | +0.001368649 | -0.007459172 |
| validation-selected | 0.779356602 | 0.495822265 | +0.004166933 | +0.000931348 |
| E50 | 0.776586521 | 0.506189959 | +0.005136495 | +0.003871616 |

固定阶段原值来自AB表的同阶段端点；主裁定维持validation-selected。LL在E10改善，到selected及E50转为上升，排序和概率损失需要分别评价。

**STRONG：MovieLens相对三个发表模型的排序增益。** 三个比较均5/5 seed为正，六格保护满足，对最强M3oE的平均增益为+0.004166933。对HyperGate的LL也5/5下降。**MEDIUM：相对M3oE的排序/LL取舍。** 其LL为4/5 seed上升，保留AUC性能等级和实际LL代价。

**WEAK：用域规模解释跨数据集差异。** MovieLens最大域d2与KuaiRand的d0分别是G-D0的最低平均AUC格，域身份、任务、基线强度及既定超参数共同不同。本次补齐识别历史结果的对应关系；Gaussian独立组件与EMA贡献沿用原内部消融记录。这里的发表模型比较评价Gaussian＋EMA整体。

## 5. 复用来源、检查和复算

补齐程序读取本分支AB端点及其原始来源，核对MovieLens增强域0、κ=.05、λ=.5、weight_decay=1e-4、ρ=.9828205985452511、batch40960、lr=.01和50epoch预算。完整验证曲线重选结果、数据/编码器/划分、源码及model hash记录在来源审核中。MovieLens复用20条历史轨迹；KuaiRand的既有20条比较轨迹只进入跨数据集视图；新增训练与新seed均为0。

补齐前后原KuaiRand端点和训练包的冻结身份保持一致。实现与独立审查分别记录在[CODER_MOVIELENS_SUPPLEMENT_REPORT.md](CODER_MOVIELENS_SUPPLEMENT_REPORT.md)和[REVIEW_MOVIELENS_SUPPLEMENT.md](REVIEW_MOVIELENS_SUPPLEMENT.md)。

在仓库根目录运行以下命令，可重新审核来源并生成补充数表：

    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/supplement_movielens.py

更新主报告入口：

    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-04_gaussian_domain_coverage/build_report.py

本次沿用同一实验分支，保留原历史结果，补齐MovieLens G-D0已完成五seed的来源和双数据集对比。
