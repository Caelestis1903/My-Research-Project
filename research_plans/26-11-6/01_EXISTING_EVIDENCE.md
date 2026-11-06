# 01｜既有证据、可复用产物与本轮研究决策

审计日期：2026-11-06。**固定 EMA 的排序收益已足够成立，本轮进入收益机制研究。** 本文从已归档逐 seed CSV 重算四格及阶段效应；未运行训练，也未重新调用模型访问 Test。配套 [evidence_snapshot.json](evidence_snapshot.json) 保存本节关键数据的全精度值、seed 配对、样本 SD、选中 epoch、来源提交和 Git blob 身份。

## 0. 证据范围、版本与统计身份

冻结来源为 main=`2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，最新研究分支 `26-11-06`=`40705dfe0bb4fe614bdca835c0bae567348c87d0`。两者已分叉：研究分支相对 main 为 ahead 13 / behind 2，共同祖先 `652ee0235fec2d0d4cfc67b4a6ef9977519619bf`。最新 A/B/C 的完成状态以研究分支为准；本方案写入 main，不等于实验包已合并。实施时从固定来源 SHA 创建隔离检出。[分支比较][branch-compare]

### 0.1 必读总文件的结论继承

| main 固定来源 | 本轮采用的事实或处理 |
|---|---|
| [EXP_AI_CONTEXT.md][main-context] | 采用 11/04 更新的当前数据人口、模型、发表模型比较与 Gaussian 域覆盖；更早配置按自己的协议保留 |
| [Experimental_Global_Summary.md][main-global] | 继承优化器历史、训练目标、参数 EMA 和增强实验的进展，避免把已解决的基础问题重新列为待证命题 |
| [Experimental_Stable_Findings.md][main-stable] | SF032 已接受固定参数 EMA 的排序收益；SF034 记录“更新/门非零”不自动带来独有收益；本轮不把工程启用等同于科学机制 |
| [Experimental_Open_Questions.md][main-open] | OQ029 明确无需重问 EMA 能否改善 AUC；OQ030–031 的算法设计问题暂转为本次机制与直接收益观测；OQ032 对应的最新同人口发表模型比较已在 11/04 台账补齐 |
| [MDMT_ALGORITHM_METRICS_LEDGER.md][main-ledger] | 采用最新五 seed 原始来源和去重声明：M3/A0、ML Gaussian/G-D0、KR 旧 Gaussian/G-D2 是来源别名；不把重复表格计作独立证据 |

### 0.2 样本量和训练路径不能混计

| 证据组 | 每候选统计样本 | 路径身份与本轮用途 |
|---|---|---|
| 当前 Clean / G-D0 四格 | 每数据集、每目标 n=5；seeds 2022–2026 | 每个目标的 online 与 EMA 共用一条训练路径；Clean 与 G-D0 是不同目标的路径，以同 seed 做差分配对。四种读出不是四倍独立训练 |
| 最新 B 与 A | 两数据各 5 seed；各 10 条物理训练 | A 重放 B 的 10 条父路径，500 个 epoch 的 online/order hash 匹配；不增加科学独立重复 |
| 最新 C | ML 六配置各 5 seed，共 30 条新路径 | 每配置有自己的 online/Adam/teacher；KR C 为 `DEFERRED_NOT_RUN`，不是负结果 |
| 11/04 域覆盖 | KR G-D0、G-D02 各 5 seed；G-D2 为已有 5 seed 锚点 | ML G-D0 复用已有 5 seed，新增训练 0；同名、同 seed 不保证不同目标的参数路径相同 |
| 11/01_v2 Clean 长 EMA | 两数据各 5 seed | 与当前 Clean 30 个 online 端点身份完全一致，仅增加历史估计器比较，不增加独立 seed |
| 旧 PolyakAdam | 旧 ML n=6，seeds 2022–2027 | 旧模型、数据处理和选点协议；只保留历史经验，不并入当前 n=5 的效应统计 |

本节表格的 SD 为 seed 间样本标准差（ddof=1），不是标准误；差分 SD 先在同 seed 内做差再计算。六格、多个 epoch、多个影子均不当作独立样本。来源为 [11/03 五 seed 报告][five-report]、[11/04 域覆盖报告][coverage-report]、[最新总报告][abc-report]及[旧 Polyak 记录][polyak].


## 1. 可立即接受的结论与下一研究决策

1. **固定EMA的排序收益已经成立，且不依赖必须存在Gaussian增强。** 当前协议Clean轨迹上ML/KR的selected Test Macro AUC净益分别+0.003848845/+0.004616399；D0 Gaussian轨迹上分别+0.003856802/+0.007766927。前者即普通A0 online与同轨迹A0 fixed_ema，而不是Gaussian+EMA与不同训练目标普通模型的总差。
2. **收益随训练阶段改变；早期滞后代价也已有直接事实。** KR同一E10，Clean/G-D0的EMA净益分别−0.009126465/−0.009499234，均五seed为负；到E50分别+0.005689981/+0.004410842，均五seed为正。ML E50两轨迹也都五seed正。这已经排除了“EMA收益全部由checkpoint搜索统计假象造成”的解释，但仍需区分时间平均与较好历史单点/有效滞后机制；后者用延迟online/窗口单点对照，不应再重复确认EMA是否有收益。
3. **AUC改善不意味着概率损失同步改善。** ML G-D0的同轨迹EMA selected LL增加0.005334282，五seed全部上升；E50也增加0.005038092，五seed全部上升。不能用“高频噪声被消除”直接解释成整体预测质量所有方面提高。排序、概率尺度、校准可分开研究。
4. **Gaussian与EMA不应预设普遍正向协同。** 当前ML selected AUC四格交互仅+0.000007957（SD0.000880733）；KR selected为+0.003150528（SD0.007058189），但validation selected为−0.001006570、Test E50为−0.001279139。存在设置与阶段依赖，当前证据不支持一个稳定普遍的协同命题。
5. **域任务收益差异已存在于归档结果，但“可在线利用的历史偏好”尚未识别。** 下列同轨迹六格净益表可作为存在性依据，不能直接升级为某个域必须更强/更弱平滑的算法规则。
6. **最新A/B/C新增复杂机制未建立超越固定F的稳定增量，工程上分支确已启用。** 应停止本轮算法堆叠，先研究收益信息；不能把这解释为固定EMA失效、MDMT差异不存在，或A/B/C所有可能实现已被否决。
7. **新增参数历史机制分析确需带观测的轨迹重放。** 最近实验的模型、optimizer、Q/统计矩阵、全部checkpoint仅驻RAM，正式评价后guardian已释放、未落盘；现有数表不能恢复参数与逐样本预测。重放的目的应明确是补齐新统计/反事实读出，不是重新确认已成立性能。

本节主要依据为 [Clean 原始端点][clean-endpoints]、[G-D0 原始端点][b-endpoints]、[最新总报告][abc-report]；机制强度与后续动作是本方案在这些事实上的研究判断。

## 2. 冻结协议和名称纠正

- seeds=`[2022,2023,2024,2025,2026]`，每轨迹50epoch，batch40960，普通coupled Adam lr=.01、betas=.9/.999、eps1e-8；ML wd1e-4，KR wd1e-5；FP32、AMP/TF32/clipping关闭。
- FULLTRAIN_NATURAL80_10_10；原行序80/10/10，train-only编码/缩放；KR先划分再筛tab0/1/4。ML train/val/test=800167/100021/100021，KR=2006132/250767/250767。
- 验证每 epoch 六格等权 Macro AUC，选择最早达到全程最大值的 epoch；各估计器独立选点；E10/E50同epoch单列。Test用于冻结后评价，已有Test为历史研究证据，不能再次当独立盲测。
- 当前模型fixed1 M3oE：embedding16、FCN[512,256,256,64]、4 shared experts；参数ML773470、KR1614254。
- 当前F（11/05 A/B/C参照）两数据都为G-D0，κ分别ML.05/KR.1、λ=.5。
- 旧11/03五seed补全中名为Gaussian的ML就是G-D0；KR旧Gaussian是G-D2。勿把KR旧G-D2和最新G-D0当同一配置或独立重复实验。
- 最新B的`config_json.parent.target_domain=0`才是本轮实际增强域；嵌套`parent.source.augmented_domain=2`是继承旧来源元数据。这个字段需保留溯源而不能覆盖实际target_domain。实施时按实际 loss target 保留此区分。
- 当前F：MLρ=.9828205985452511（半衰期40 optimizer steps=2epoch）；KRρ=.995（138.282573steps=2.822093epoch）。不是早期对话中记忆的ρ=.9。
- 域映射为 ML 年龄 {1,18}/{25}/{35,45,50,56}，KR tab0/tab1/tab4；t0/t1 按来源任务定义。小域不是困难域，域大小不进入本次机制假设或干预规则。

协议及参数来源：[最新总报告][abc-report]、[11/03 五 seed 报告][five-report]、[11/04 ML 来源审计][ml-coverage]。同数据集不同模型的发表比较虽然共用当前人口与 AUC 选点规则，仍有原生 batch、优化器、预算差异；HyperGate 部分超参标为 `UNVERIFIED_PROPOSAL`，不可称为原论文严格复现。[main 指标台账][main-ledger]

### 2.1 发表模型和增强覆盖已提供研究参照

| 数据集 | 当前发表模型 selected Test Macro AUC（每模型 n=5） | G-D0 + EMA 的已有位置 |
|---|---|---|
| MovieLens | M3oE 0.775190；HyperGate 0.771412；AutoIFS 0.772425 | F=0.779356602，对该协议最强 M3oE 总收益 +0.004166933，六格保护满足；LL +0.000931348 |
| KuaiRand | M3oE 0.671907；HyperGate 0.682113；AutoIFS 0.678597 | F=0.679180178，对 M3oE 总收益 +0.007273579；仍低于最强 HyperGate 约 0.002932 |

以上“总收益”同时包含训练目标变化与 EMA，必须与下一节的同轨迹 EMA 净益区别。KR 已完成 G-D0/G-D02/G-D2 各 5 seed；D02−D0 的 Macro AUC 为 −0.004576209。覆盖更多域未在此配置提高排序，因此本轮固定既有 G-D0 来源做机制观测，不额外扩展增强域搜索。上述比较无需重训发表基线或域覆盖矩阵。[统一发表模型报告][formal-report] [域覆盖报告][coverage-report] [ML 来源审计][ml-coverage]

## 3. 已归档四格重算：无需新训练

数据源：[最新 B `ENDPOINTS.csv`][b-endpoints] 的 F/online（同 `trajectory_id`）与 [11/03 五 seed 补全 `endpoints.csv`][clean-endpoints] 的 A0 fixed_ema/online（同 `key` 与 `configuration`）。数据人口、模型/训练及评价协议相同。Gaussian与Clean之间是不同训练目标的两条路径，交互采用同seed配对；不把路径差异归因为EMA单独作用。

定义：G_clean=U(CleanEMA)−U(CleanOnline)，G_gauss=U(GaussianEMA)−U(GaussianOnline)，I=G_gauss−G_clean。LL表使用原LL差，正值表示代价；AUC正值表示收益。selected包含估计器作用与各自合法选点差异；E10/E50为固定时刻读出。

### 3.1 selected Test绝对值（mean ± sample SD，n=5）

| 数据集 | Clean online AUC / LL | Clean EMA AUC / LL | G-D0 online AUC / LL | G-D0 EMA AUC / LL |
|---|---|---|---|---|
| movielens | 0.775189669 ± 0.001136047 / 0.494890917 ± 0.004690271 | 0.779038514 ± 0.001002948 / 0.496139561 ± 0.001543373 | 0.775499801 ± 0.000616550 / 0.490487983 ± 0.002056324 | 0.779356602 ± 0.000393344 / 0.495822265 ± 0.001550074 |
| kuairand | 0.671906599 ± 0.006144530 / 0.437172004 ± 0.002750710 | 0.676522998 ± 0.005226758 / 0.435946662 ± 0.001323028 | 0.671413250 ± 0.006326915 / 0.435190167 ± 0.002223915 | 0.679180178 ± 0.003474732 / 0.435517854 ± 0.002752362 |

### 3.2 selected/E10/E50：EMA净益与Gaussian交互

| 数据/评价集/端点 | G_clean AUC | G_gauss AUC | 交互I AUC | G_clean LL | G_gauss LL | 交互I LL |
|---|---:|---:|---:|---:|---:|---:|
| movielens/test/E10 | +0.001145725 | +0.001094519 | -0.000051206 | -0.007187354 | -0.004249110 | +0.002938244 |
| movielens/test/selected | +0.003848845 | +0.003856802 | +0.000007957 | +0.001248644 | +0.005334282 | +0.004085638 |
| movielens/test/E50 | +0.004787348 | +0.004613875 | -0.000173473 | +0.003301121 | +0.005038092 | +0.001736970 |
| movielens/validation/E10 | +0.000230135 | +0.000465342 | +0.000235206 | -0.005848808 | -0.003242050 | +0.002606759 |
| movielens/validation/selected | +0.002846555 | +0.002822577 | -0.000023978 | -0.000531423 | +0.001899167 | +0.002430590 |
| movielens/validation/E50 | +0.003822263 | +0.003277809 | -0.000544453 | +0.000871185 | +0.003035508 | +0.002164324 |
| kuairand/test/E10 | -0.009126465 | -0.009499234 | -0.000372770 | +0.013317647 | +0.005031373 | -0.008286274 |
| kuairand/test/selected | +0.004616399 | +0.007766927 | +0.003150528 | -0.001225342 | +0.000327687 | +0.001553029 |
| kuairand/test/E50 | +0.005689981 | +0.004410842 | -0.001279139 | -0.005711648 | -0.000415708 | +0.005295940 |
| kuairand/validation/E10 | -0.019906509 | -0.018706344 | +0.001200165 | +0.014392867 | +0.006219861 | -0.008173006 |
| kuairand/validation/selected | +0.002176179 | +0.001169608 | -0.001006570 | -0.000903787 | +0.000175754 | +0.001079541 |
| kuairand/validation/E50 | +0.003826161 | +0.004762021 | +0.000935860 | -0.004967984 | -0.000402363 | +0.004565621 |

### 3.3 selected Test配对不确定性（seed是统计单位）

| 数据 | 量 | 均值 | 配对差sample SD | 描述性t95% CI(df4) | AUC正seed |
|---|---|---:|---:|---|---:|
| movielens | g_ema_clean | +0.003848845 | 0.000374396 | [+0.003383971, +0.004313719] | 5/5 |
| movielens | g_ema_gaussian | +0.003856802 | 0.000645238 | [+0.003055633, +0.004657971] | 5/5 |
| movielens | interaction | +0.000007957 | 0.000880733 | [-0.001085617, +0.001101531] | 3/5 |
| kuairand | g_ema_clean | +0.004616399 | 0.003048773 | [+0.000830848, +0.008401951] | 4/5 |
| kuairand | g_ema_gaussian | +0.007766927 | 0.004257500 | [+0.002480543, +0.013053312] | 5/5 |
| kuairand | interaction | +0.003150528 | 0.007058189 | [-0.005613371, +0.011914427] | 3/5 |

不把“所有seed同向”或CI排除零升级为新增硬门槛。现有四格的Macro AUC+0.003评价规则也不应移植为机制是否成立的硬阈值。

### 3.4 六格selected Test同轨迹净益

| 数据集 | cell | Clean EMA−online AUC | G-D0 EMA−online AUC | 交互I AUC |
|---|---|---:|---:|---:|
| movielens | d0t0 | +0.004094543 | +0.002609411 | -0.001485132 |
| movielens | d0t1 | +0.004425141 | +0.003673471 | -0.000751670 |
| movielens | d1t0 | +0.003890656 | +0.004813939 | +0.000923284 |
| movielens | d1t1 | +0.003472647 | +0.005111461 | +0.001638814 |
| movielens | d2t0 | +0.004065410 | +0.003772944 | -0.000292466 |
| movielens | d2t1 | +0.003144673 | +0.003159585 | +0.000014912 |
| kuairand | d0t0 | +0.001373717 | +0.012254220 | +0.010880503 |
| kuairand | d0t1 | +0.004037304 | +0.018562308 | +0.014525003 |
| kuairand | d1t0 | +0.006472511 | +0.005856228 | -0.000616283 |
| kuairand | d1t1 | +0.006725991 | +0.005941513 | -0.000784478 |
| kuairand | d2t0 | +0.002416765 | +0.002948760 | +0.000531995 |
| kuairand | d2t1 | +0.006672108 | +0.001038535 | -0.005633573 |

这些是同路径对照在各自selected端点的净差，不能直接等同于“域任务在共同epoch上的最优半衰期”。配套 [evidence_snapshot.json](evidence_snapshot.json) 保存两 split×三端点的四格 Macro 绝对值和配对效应；每个量包含全精度 mean、样本 SD 与按 2022–2026 排序的五个值。六格 selected 保存 Val/Test 的 AUC/LL 效应 mean/SD，并保存 selected Test AUC 的五个配对值。交互逐 seed 等于 Gaussian 净益减 Clean 净益。所有值由原始 CSV 全精度计算；正文仅为阅读而舍入。

## 4. 已有时间尺度证据

### 4.1 当前Gaussian路径的三个非反馈全参数EMA

B 完整训练已维护 `U_FAST(h=.5h0)`、`F(h=h0)`、`U_SLOW(h=2h0)`，每数据5seed、50epoch；A再次重放同online只为其他观测，不能算新增科学seed。B/A全部500对epoch online/order hash逐一相同。

| 数据集 | 估计器 | 五seed V-selected Macro AUC | SD | selected epochs |
|---|---|---:|---:|---|
| movielens | U_FAST | 0.759495345 | 0.000463945 | [23, 18, 17, 19, 17] |
| movielens | F | 0.759522014 | 0.000473087 | [23, 20, 25, 23, 22] |
| movielens | U_SLOW | 0.759255378 | 0.000373574 | [33, 25, 28, 28, 29] |
| kuairand | U_FAST | 0.690861175 | 0.005599744 | [22, 40, 42, 33, 50] |
| kuairand | F | 0.690802423 | 0.005352210 | [25, 44, 43, 38, 50] |
| kuairand | U_SLOW | 0.690114019 | 0.004681094 | [29, 50, 49, 41, 50] |

当前 U_FAST 相对 F 的六格 V-selected 变化既有正又有负；U_SLOW 两数据六格均值均更低。[B 超参统计][b-hpo]全曲线仍可在原 CSV 或逐 epoch JSON 离线读取。本轮已完成端点复算，执行阶段先做这些既有曲线的零训练分析；无需为重复该三档对比单独训练。selected值涉及不同epoch，不能直接宣布时间偏好具有可泛化预测价值。

### 4.2 Clean A0长/短EMA旧结果可复用

11/01_v2 的 A0 shadow 使用 ρ=.995（ML 约 6.914 epoch 半衰期）；当前 Clean A0 使用 h40（ML 2 epoch）。对[旧 test_metrics][old-ema-test]与[当前 Clean 端点][clean-endpoints]的核对覆盖两数据×五 seed×selected/E10/E50 全部 30 个 online 端点，model_hash、Macro AUC、Macro LL 均完全相同。它们可桥接为同一个Clean父路径的读出比较，不能当新训练seed。

ML `长ρ=.995 EMA − 当前h40 EMA`：

| 端点 | ΔAUC | ΔLL | 结论 |
|---|---:|---:|---|
| E10 | −0.003389640（5/5负） | +0.047603711（5/5升） | 早期长历史代价明显 |
| selected | +0.000550201（4/5正） | +0.006550953（5/5升） | 排序与概率损失分离，含选点差异 |
| E50 | +0.003020306（5/5正） | +0.007140782（5/5升） | 晚期排序更好，但LL代价仍在 |

当时ML Gaussian用κ=.1，而当前ML为κ=.05；Gaussian之间不应照搬同路径身份。KR旧Gaussian为D2，当前为D0，亦不应混为单纯时间尺度差异。

以上长/短 EMA 比较复用同一父训练，补充了阶段与排序/概率损失分离的事实；不把当前协议中的 n=5 累加成 n=10。[11/01_v2 报告][old-ema-report]

## 5. 最新 A/B/C 已判定事实

| 路径 | 当前事实 | 新机制研究应继承的边界 |
|---|---|---|
| B Adam增强响应定向EMA | 两数据各5seed，D−F selected Test AUC为ML +0.000000104、KR −0.000004946；共同E10/E50也近零。ML500/KR1225次刷新全部D秩2/M秩1，state hash前后相同 | B已运行且修正非空；未记录innovation投影能量，故“响应方向信息无价值”与“仅改动极小innovation能量”仍未区分。不能为确认启用再重复B完整矩阵 |
| A 实际路径分组EMA | GROUP赢家ML−0.000749775、KR−0.000174743；KR FIXED_S_FAST V +0.001009861但Test −0.000217263。ML GROUP E10/E50 +0.000849077/+0.000548266，selected反负 | 分组/时间尺度实际改变预测；train排序探针未形成稳定selected增量。合理的下一问是直接收益指标与跨时间可预测性，不是再做相同controller HPO |
| C 历史漂移条件增强 | ML六配置×5seed，C3_WEAK、MERGED、ROTATED相对F selected AUC均负；最终MERGED−0.000667190（5/5负）。其online本身也下降 | 变化已进入训练轨迹，不能全归因EMA读出。KR C未运行，不作性能失败。既有建议“固定谱真实/旋转方向控制”属于后续算法路线，不应成为本次EMA机制主矩阵默认项 |

来源：[B 阶段分析][b-report]、[A 阶段分析][a-report]、[C 阶段分析][c-report]。

B/A/C实际50次物理完整训练、2500epoch、64500次Adam、19000次模型validation；因A/B同父路径复跑，只有40条唯一训练路径；每候选统计始终n=5。B/A/C运行成本均为共享并发环境的各worker记录training+probe+validation之和，不是独占GPU小时或整机墙钟。

## 6. 历史控制应保留、不可过度推广

- 旧PolyakAdam(ρ=.995)在旧ML协议6seed配对mean AUC+0.004022±0.001266；它本身就是Adam后的参数EMA shadow，不是新的优化器动量。旧模型/人口/选点不同，只作为参数历史平均有用的历史经验，不能合并进当前五 seed。来源：[旧 Polyak 机制与六 seed 记录][polyak]。
- 11/01_v2（40训练、5seed）的SAD βδ=.99没有建立超越普通增强Adam的selected收益；代数等价βδ=.9对照短步误差近浮点精度却可能长期轨迹分离，因此切勿把任何微小数值差直接解释成新机制。已有 Adam/目标控制不能重新包装为新未解决基础问题。[11/01_v2 报告][old-ema-report] [main 稳定结论][main-stable]
- 11/01_v3的A0/A1正式五seed完成；AIG等仅开发部分不满足当前五seed新结论要求。旧少 seed 停止记录不等于当前科学否决。[11/01_v3 报告][aigm-report] [main 指标台账][main-ledger]
- 11/03 C/H/P01及11/03夜间CP已有完整五seed；ML多数组件相对固定EMA的AUC下降而LL改善，KR可有宏与逐格取舍。这些证明已有指标分离/收益分布，不要求本轮重跑其算法。[11/03 五 seed 报告][five-report] [统一 A+B 报告][formal-report]
- 严格不从域大小推断任务难度或EMA需求；历史报告的相关叙事不纳入本次机制假设。

## 7. 可复用产物与明确缺项

| 资源 | 已有内容 | 本次直接可用 | 不能从它恢复 |
|---|---|---|---|
| 最新B/A `VALIDATION_CURVES.csv`及runs/*/epochs/E*.json | 全50epoch、online和15影子的六格AUC/LL、Macro/Traffic/Pooled、model_hash；源数据/顺序/配置hash | 固定epoch曲线、time-scale选择敏感性、六格历史偏好描述、局部峰值/选点稳定性；A/B身份去重 | 参数向量、预测logit、排名变化的逐样本/逐pair贡献 |
| 最新B/A/C `ENDPOINTS.csv` | 冻结selected/E10/E50 V/Test，五seed绝对值与hash | 完整重算均值、sample SD、配对差、效应分解 | 未冻结candidate的任意Test评估 |
| B `batch_trace.response` | rank、column_norm、orthogonality、state hash、响应耗时；Gaussian noise hash | 确认反事实无污染及方向分支启用 | r在Q子空间的投影能量、预测收益方向 |
| A controller日志 | q_before/q_after、实际probe调用、恢复链 | 规则实际行为与阶段关联 | 直接holdout反事实收益（未测量） |
| C条件统计日志 | 非零漂移/trace、时钟、noise/order hash | 启用事实、轨迹来源、反向因果边界 | 参数轨迹或逐样本预测 |
| 11/03、11/04标量归档 | 配置、数据/源码身份、逐epoch/batch scalar、冻结端点 | 解压后离线重建旧曲线/同协议配对 | 模型/optimizer/EMA checkpoint（从未入档） |
| 历史数据数组缓存 | 服务器只读已有cache，hash可核验 | 复用数据人口、train-only preprocessing、row order | 这里本地没有原始数据；不得声称已加载样本 |

最新REPORT明确：B/A/C RAM guardian在冻结评价、报告完成后释放；C释放返回RELEASED；所有模型、optimizer、Q/A/root/R与checkpoint未写盘。11/03 五 seed 与 11/04 域覆盖归档也明确相同策略。[最新总报告][abc-report] [11/03 归档说明][five-raw] [11/04 归档说明][coverage-raw]因此离线数表阶段=0新训练；若要直接干预参数历史/预测函数，本轮主预算固定为两 dataset×Clean/G-D0×5 seed=20 条带观测重放、1000 epoch。无需为每个EMA影子再训练一条。

## 8. 资源参考（已有测量，不作为本轮承诺）

B两数据10条500epoch累计worker20563.108秒：training8206.223、probe1117.640、validation11239.245。A同计25970.890秒（training10457.704、probe1791.151、validation13722.035）。C ML30条1500epoch累计15449.698秒。B单workerpeak allocated ML/KR约1.329/1.544GB、reserved1.669/1.873GB。验证多影子已经大于或接近训练成本，新方案应降低全量观测频率、以固定train诊断bank筛选指标、全量V保留必要边界，不能只预算EMA递推浮点操作。

来源：[最新总报告成本表][abc-report]。参数 FP32 一份约 ML 3.094 MB / KR 6.457 MB（十进制，由参数数×4 bytes 计算），多影子本身不贵；每epoch全量V和反事实分组forward才是主要成本。Adam一二阶状态及临时图、模型副本、RAM缓存另算。

## 9. 本轮因此要补什么、不再重复什么

**已经回答的经验问题：** EMA 在当前两个数据集的 Clean 和 G-D0 路径上有稳定排序收益；收益随训练阶段改变；AUC 与 LogLoss 分离；六格收益分布不同。四格 selected/E10/E50 已重算，完整曲线有现成标量来源。Gaussian 是否为 EMA 收益的必要条件、固定 epoch 是否有 EMA 收益、分组/方向分支是否实际启用，不再作为新的大规模确认任务。

**仍会改变研究决策的缺口：** 历史参数中哪部分变化带来哪些样本/pair 的真实排序或概率收益；这种收益是否能由直接局部反事实或其他指标刻画；较好历史单点与参数平均各贡献多少；六格偏好是否在固定 epoch 与隔离诊断中可重复、能否预测未来时刻或未参与拟合的 seed。现有 CSV 缺参数与预测级观测，不能通过更多汇总表替代。

**本轮主预算：20 条观测重放 = 2 数据集 × Clean/G-D0 两个固定目标 × 5 seed × 50 epoch，总计 1000 epoch。** 每条父训练可共享多个无反馈读出和诊断，观测不得改变原 online 更新。重放用于收集这些缺失观测；已归档端点/曲线直接复用，新增训练为零；不重训已发表模型基线，不重复 A/B/C 搜索，不为每个 EMA 影子新开训练。若未来在相同冻结来源找到真实可恢复状态，可据身份核验减免相应重放；目前证据明确历史 checkpoint 未留存，不以假想缓存减少预算。

## 10. 数据快照的读取与溯源

[evidence_snapshot.json](evidence_snapshot.json) 是紧凑证据索引，不复制 MB 级原 CSV。`macro[dataset][split][endpoint]` 含 `absolute` 和 `effects`；数值向量为 `[mean, sample_sd_ddof1, seed2022, seed2023, seed2024, seed2025, seed2026]`。`six_cell_selected` 为 `[mean, sample_sd_ddof1]`；`six_cell_test_auc_paired` 为按同一 seed 顺序的五个配对差。`selected_epochs` 对齐 seed 顺序。`sources` 给出固定提交、CSV 路径、过滤条件、配对键和可用的 Git blob ID，避免将报告舍入值反算为原始精度。

原始 B 端点 600 行，Clean 来源表 660 行；本轮只使用相应来源的所需读出，不把表格总行数当作科学样本量。其余阶段曲线通过固定来源的 `VALIDATION_CURVES.csv` 或 `runs/*/epochs/E*.json` 读取；复算时按数据集、seed、训练身份、估计器、split、端点或 epoch 明确唯一性。

### 固定版本来源

- main 总文件：[上下文][main-context]、[全局摘要][main-global]、[稳定结论][main-stable]、[开放问题][main-open]、[指标台账][main-ledger]。
- 11/05–06 最新研究：[综合报告][abc-report]、[A 分析][a-report]、[B 分析][b-report]、[C 分析][c-report]、[B 原始端点][b-endpoints]、[B 超参统计][b-hpo]、[A 超参统计][a-hpo]。
- 11/03–04：[五 seed 补全报告][five-report]、[Clean 原始端点][clean-endpoints]、[五 seed 归档说明][five-raw]、[发表模型统一报告][formal-report]、[域覆盖报告][coverage-report]、[ML 来源审计][ml-coverage]、[域覆盖归档说明][coverage-raw]。
- 更早历史：[参数 EMA / SAD 报告][old-ema-report]、[原始 test_metrics][old-ema-test]、[AIG-M 报告][aigm-report]、[Polyak 机制和六 seed 记录][polyak]、[10/31 控制与梯度统计][ga31-report]。

[branch-compare]: https://github.com/Caelestis1903/My-Research-Project/compare/2645f1f724f52f5ed5dea4b479335c0dc12fabcd...40705dfe0bb4fe614bdca835c0bae567348c87d0
[main-context]: https://github.com/Caelestis1903/My-Research-Project/blob/2645f1f724f52f5ed5dea4b479335c0dc12fabcd/EXP_AI_CONTEXT.md
[main-global]: https://github.com/Caelestis1903/My-Research-Project/blob/2645f1f724f52f5ed5dea4b479335c0dc12fabcd/experiments/Experimental_Global_Summary.md
[main-stable]: https://github.com/Caelestis1903/My-Research-Project/blob/2645f1f724f52f5ed5dea4b479335c0dc12fabcd/experiments/Experimental_Stable_Findings.md
[main-open]: https://github.com/Caelestis1903/My-Research-Project/blob/2645f1f724f52f5ed5dea4b479335c0dc12fabcd/experiments/Experimental_Open_Questions.md
[main-ledger]: https://github.com/Caelestis1903/My-Research-Project/blob/2645f1f724f52f5ed5dea4b479335c0dc12fabcd/MDMT_ALGORITHM_METRICS_LEDGER.md
[abc-report]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md
[a-report]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/A_STAGE_ANALYSIS.md
[b-report]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/B_STAGE_ANALYSIS.md
[c-report]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/C_STAGE_ANALYSIS.md
[b-endpoints]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/artifacts/aug_history_20261105/attempt1/B/ENDPOINTS.csv
[b-hpo]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/artifacts/aug_history_20261105/attempt1/B/HPO_SUMMARY.csv
[a-hpo]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/artifacts/aug_history_20261105/attempt_A1/A/HPO_SUMMARY.csv
[clean-endpoints]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv
[five-report]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md
[five-raw]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-03_mdmt_five_seed_completion/RAW_ARTIFACTS.md
[formal-report]: https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md
[coverage-report]: https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md
[ml-coverage]: https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md
[coverage-raw]: https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-04_gaussian_domain_coverage/RAW_ARTIFACTS.md
[old-ema-report]: https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md
[old-ema-test]: https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/test_metrics.csv
[aigm-report]: https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md
[polyak]: https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/PROJECT_ARCHITECTURE.md
[ga31-report]: https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md
