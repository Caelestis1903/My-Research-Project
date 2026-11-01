# AIG-M：有限搜索与增强创新量门控设计规格

设计日期：2026-11-01。执行入口为 `start.md` 和 `research_plans/26-11-1_v2/06_Codex完整执行提示词.md`。已阅读规划00–06、GAM01设计/实施/审查/报告及实际优化器、目标、状态、运行、分析、检查实现，GA31-R3实际模型、视图、数据、指标、状态依赖和数据manifest。历史数值复算与来源核验由协调者写入本目录，本文不覆盖其文件。本文只冻结设计，未实施、未训练。

## 1. 范围、身份与合同解释

- 只新增 `experiments/gradient_aug_aig_m_20261101/`、本报告目录和独立 `artifacts/gradient_aug_aig_m_20261101/attempt_<timestamp>/`。旧GA31-R3/GAM01源码、历史报告、原始数据和计划不改。
- 本轮计划来源为 `main@0973fa5f1a804aec34f88ccb2ffdd8c0231b602b`；核验的GAM01发布HEAD为 `26-11-01_v2@ee4cf0512a7723624c9ee6a1ef192f0f7f17686b`，历史执行代码为 `89edb99a389c56fde86460af677a48e14fbcf116`。真实核验时间、blob/文件hash见协调者 `SOURCE_FETCH_RECEIPT.json`，不能把来源SHA当作本轮实现commit。
- 本轮是新计划执行，发布到当天尚未占用的日期新分支；读取历史分支作证据不构成续接该分支。目前26-11-01及_v2已占用，协调者拟用26-11-01_v3。发布以main树为基础，带入核验的未修改GA31-R3/GAM01依赖及GAM01历史报告子树，再加本轮代码/报告。独立审查后的不可变执行代码commit与最终附报告的publication commit分别记录，不能以历史SHA替代。
- 03的最终显式公式与紧邻说明决定 `v` 使用当前差分 `d`，分子使用平滑差分 `q`。同节一次“r+λq”文字与显式 `v` 不一致，按其明确的“二阶矩沿用GAM01 SAD语义，以当前混合梯度r+λd更新”执行。已检查GAM01实际SAD与此一致；不存在需要暂停的实际优化器冲突。
- 03中基础Adam的一阶矩简写省略wd，实际标准Adam和同节说明规定coupled wd同时进入m/v；本轮保留实际语义。
- 02的动态β、未写偏差修正的概念公式由03/06更具体的第一版合同替代：**仅动态λ，βδ固定**。不实施动态β、不维护时变β乘积。
- 04文字称ρ=.995对应ML69/KR28 epoch，为10倍算术误差。按冻结换算公式，半衰期约138.28 step，即ML6.91/KR2.82 epoch。新报告注明勘误，保留原计划。
- 训练自然人口、标签、样本比例不改；本轮明确授权的Gaussian只改变训练forward中的目标域非路由sparse embedding，不产生新数据文件。
- 所有权重、模型/optimizer/q/shadow/RNG/loader快照仅在RAM，IPC使用现有BytesIO/Pipe。不得写模型或checkpoint文件，包括临时文件。磁盘只存代码、配置、数值数据、hash、日志和报告。

## 2. 数据、模型与随机流

直接复用 `experiments.gradient_aug_20261031_v2.vendor_ga31.data.load_dataset`、`views.make_model/draw_view`、目标函数、`metrics.metrics/evaluate` 和状态工具；完整包名导入，不改 `sys.path`、不全局monkey patch、不新增依赖。可复用GAM01 `SADMomentum` 及目标拆分代码；不要调用旧study固定40run或旧P2门。

| 数据 | train / validation / test | train域d0/d1/d2 | 增强目标 | 参数量 | step/epoch |
|---|---|---|---|---:|---:|
| MovieLens | 800167 / 100021 / 100021 | 155949 / 318599 / 325619 | d0 | 773470 | 20 |
| KuaiRand | 2006132 / 250767 / 250767 | 172782 / 1743898 / 89452 | d2 | 1614254 | 49 |

CSV分别为 `/home_new/xzf/datasets/ml-1m/ml-1m.csv` 和 `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。继承GA31-R3的raw、六数组、split/行号、encoder/scaler hash与特征顺序；ML原行序80/10/10，KR先全局行序划分再按既有规则过滤。train拟合编码器/scaler，OOV0。不依赖缺失的prepared artifacts。

fixed1 MDMTRec/M3oE，3域×2任务、4 shared experts、embedding16、fcn_dims=[512,256,256,64]、exp_d/exp_t/bal_d/bal_t均1。ML标签rating>3、rating>4；KR存储is_click、long_view。Sample损失预算、loss_scale1、batch40960、FP32、50epoch、无AMP/TF32/早停。Adam β=(.9,.999)、eps1e-8、foreach/fused/amsgrad=false。训练每epoch自然无放回shuffle，drop_last=false。种子只允许2022–2026；开发仅2022/2024。

沿用GAM01的初始化与loader独立generator offset、GA31-R3视图key `GA31-R3|dataset|seed|train|global_step|gaussian|1`。相同dataset/seed的全部配置须初始化hash和每epoch原行顺序hash相同；κ变化使用同一噪声视图，λ/优化规则不消耗额外随机数。shadow、诊断、选择与阈值计算不能改变训练随机流。

## 3. 目标、固定SAD与AIG-M

同θ同batch，真实梯度 `g=grad(mean_tasks_then_mean_samples BCE)`；目标域差分

`d=grad((sum target_aug_BCE − sum target_clean_BCE)/(B×2))`。

普通增强A1直接对源标量目标 `Lreal + λ(Ltarget_aug−Ltarget_clean)/(B×2)` 做一次backward并标准Adam；保持源求和顺序。SAD/AIG做两次 `autograd.grad` 得g/d，全部完成后更新一次参数。Gaussian K=1、独立view、RMS尺度stopgrad，dense/domain ID/标签不扰动。增强关闭、κ0、λmax0或无目标样本不抽view；配置关闭的AIG不做创新量计算。

对活动参数：`r=g+wd·θ`，coupled wd只进入真实通道；`m←.9m+.1r`，`q←βδq+(1−βδ)d`，`v←.999v+.001(r+λd)^2`。同一逐参数活动step n：

`θ←θ−lr·[m/(1−.9^n)+λq/(1−βδ^n)]/[sqrt(v/(1−.999^n))+1e−8]`。

固定SAD使用常数λ；AIG在每次更新前、使用旧q计算参数组G的

`u_G = sqrt(sum_G ||d−q_old||²)/(sqrt(sum_G ||q_old||²)+c·sqrt(sum_G ||g||²)+eps_u)`，

`s_G=sigmoid(a·(u_G−tau_G))`，`λ_G=λmin+(λmax−λmin)s_G`。

固定 **βδ=.95、c=1、a=4、eps_u=1e−12、λmin=0**，不搜索这些值。u中的g为不含wd的full-real梯度。组标量使用detach，不对门求导；模型及optimizer状态保持FP32，范数平方聚合可用FP64避免标量求和误差。u使用未偏差修正的旧q，更新分子使用偏差修正的新q。每组λ同时用于当前v与新q分子；不得把v改用q，也不得把λ乘入q的历史累积。

### 3.1 参数组：已由实际named_parameters枚举和forward核对

设计阶段用实际 `views.make_model` 与历史manifest特征实例化两种模型，参数量773470/1614254吻合。下面规则来自 `vendor_ga31/reference.py:MDMTRec.forward` 的实际连接：target输出只使用target slot、全部共享expert、target domain expert、两个task expert、索引 `i=t*3+target` 的gate/tower；gate输入emb.detach，但其本身仍接受target损失梯度。

- **E（embedding）**：所有 `embedding.embed_dict.<feature_index>.weight`。ML索引0/1/2/4/5/6/7，KR0–33。路由字段本身不加噪，但其参数可因目标预测改变获得差分，所以保留在实际target梯度组。
- **P（目标路径）**：精确名 `shared_weight/shared_bias`，前缀 `skip_conn.`、`star_mlp.`、`expert.`、`task_expert.`；精确 `slot_weight.<target>/slot_bias.<target>`；前缀 `domain_expert.<target>.`、`gate.<target>.`、`gate.<3+target>.`、`tower.<target>.`、`tower.<3+target>.`。
- **U（其他）**：其余非目标domain_expert/slot/gate/tower参数，标准Adam，λ=0，不分配q。训练前写所有实际参数名→组映射；必要真实batch检查U的d为None或精确零，有非零则属于影响算法定义的Critical问题，修正规格/实现并重新审查后再训。

E/P是**静态结构组**。不因某一步d=0临时换组：已有q在当前d=0时仍衰减并按冻结门参与更新，保持SAD状态语义。始终无差分且q=0的参数严格退化Adam。g/d都None时不更新step/state；一侧活动另一侧按0；显式零仍为活动。无目标batch中不增加增强forward，但活动真实通道上的既有q按固定β继续衰减。配置关闭增强/AIG或λmax0时走标准Adam，不抽view、不更新q/门状态。

### 3.2 阈值预注册

S3的固定SAD β=.95、λ=.5、κ=.1使用S1选定lr/wd，在每个开发seed**E1全部训练step、更新前**计算同一E/P定义的u；其q为该SAD自身历史。每dataset/组将2022与2024的E1观测合并，按标准线性分位数分别取q50/q75，写训练统计原值和 `thresholds_frozen.json`。不读取validation/test拟合阈值，不进行额外pilot训练。M1/M2和全部后续seed共享这两个开发seed给出的dataset专属阈值；不为每seed重新估计。阈值只选训练分位数，门斜率/分组/统计窗口不再改变。

M1：q50、λmax=.25；M2：q75、λmax=.5。两者κ=.1、β=.95，基础lr/wd沿S1。训练中记录每组u/s/λ、q/m范数、注入比及epoch汇总；至少保留E1全部校准原值与每epoch最后真实batch的实际诊断，不增加probe训练。

### 3.3 退化与因果范围

- use_aig_m=false且use_sad=false、use_augmentation=false、κ0或λmin=λmax=0时走对应标准Adam分支；无用组件不产生额外视图/梯度遍历。
- λmin=λmax=λ0且β固定时与GAM01固定SAD连续多步一致；β=.90恒λ时与普通混合Adam实数代数等价，保留最小数值检查，不加完整训练臂。beta_delta=0只作普通混合Adam分支哨兵，不直接代入分源公式。
- 全程d=0、q初始0时严格真实Adam；当前d=0而旧q非0不删除历史。
- S3两条固定控制λ都是.5。M2与SAD.95有相同κ/β/最大剂量及基础配置，可比较门控后的净差；动态平均剂量仍与门控信号一起变化。M1没有同预算固定λ=.25训练对照，不能把M1收益唯一归因于创新信息。此缺口记录在报告，不追加训练。

## 4. 组件、模块与状态

新包保持最小职责：配置/ledger/选择规则；AIG optimizer；目标/一步更新；多shadow与RAM状态；阶段运行调度；少量必要检查与数值汇总。可合并小模块。数值分析脚本只生成表格、seed配对统计、预设门与成本账本，不解释研究结论；runner执行并提交原始结果，最后交协调者分析。显式序列化 `use_augmentation/use_sad/use_aig_m/use_shadow_ema/use_aig_swa`；AIG依赖增强且与普通SAD模式互斥；本轮use_aig_swa恒false，拒绝启用，不实现备选。

每条online轨迹从θ0维护4个固定shadow：ρ=.995、半衰期H=2/8/20 epoch，`ρ=exp(log(.5)/(H*S))`，S为20/49。每次online step后更新一次，shadow不反馈训练；浮点参数EMA，buffer逐名copy，非浮点直接copy。选择/评价须恢复model.training。各shadow和online各自按完整validation六格macro AUC严格提升选epoch，平局最早，E0排除。

全部shadow仅增加状态、更新与validation成本，不增加online训练run。每epoch记录online与4shadow完整validation六格及汇总；保存RAM中各自selected和共同E10/E50模型身份与状态。最小快照包括online、optimizer m/q/v/step、4shadow、固定阈值及组映射身份、Python/NumPy/Torch/CUDA RNG、loader状态、epoch/global step、最佳值/选点、训练/视图hash与计数。恢复不重新初始化shadow或阈值。只保留当前selected和E10/E50所需快照，过时状态释放。

## 5. 有限矩阵与逐阶段选择

先写schema/config hash及下面**完整有限候选列表**，再启动训练。所有trial完整50epoch，E10不能剪枝。每阶段全部开发seed完成后用validation写决策并汇报，再进入下一阶段。S1–S4中的基础lr/wd对同数据集统一，故本轮提供同配置机制效应与有限基础调参后的表现；没有逐方法独立lr/wd搜索，不宣称已完成等成本的各方法独立优化器调参。

| 阶段 | 每数据集候选 | seed | 两数据集online数 |
|---|---|---|---:|
| S1 Adam | A:lr.005/wd1w；B:lr.01/wd.3w；C:lr.01/wd1w；D:lr.02/wd3w | 2022,2024 | 16 |
| S2 普通增强 | G1:κ.05/λ.5；G2:κ.1/λ.5；G3:κ.1/λ.75 | 2022,2024 | 12 |
| S3 固定SAD | S1:βδ.95/λ.5；S2:βδ.99/λ.5；均κ.1 | 2022,2024 | 8 |
| S4 AIG | M1:q50/λmax.25；M2:q75/λmax.5；均κ.1/βδ.95 | 2022,2024 | 8 |
| S5 正式 | 已选A0、A1，及通过该dataset门的AIG | 2022–2026 | 20 / 25 / 30 |

w为ML1e−4/KR1e−5。A0 online和A0+EMA共享同一轨迹；每dataset固定一个基础配置，后续不因方法单独改变。S1在四配置×四shadow中，以两个开发seed的selected validation macro AUC均值选最强固定shadow参考及其lr/wd；A0在线输出来自该同一配置。online表现、所有候选及相对历史C/.995的保护项同时报告，不能把校准参考选择写作新算法晋级。

S2使用该基础配置，对每G×shadow算两seed均值；选最强普通增强+EMA，保留其G/ρ作正式A1参考。另保留G2作为κ=.1/λ=.5的同配置机制对照，即使G1/G3更优。S3/S4固定同一基础配置与κ=.1，全部shadow的选择预算相同。S3固定路线若不超过A1+EMA停止其后续晋级，但仍完成合同内S4，不自动取消AIG检验。

**近似平局统一定义**：先取最高开发均值，距最高值严格小于.0005的候选视为平局，优先更少训练forward/梯度遍历、较少组件；同结构同成本则按冻结顺序：Adam C/A/B/D，增强G1/G2/G3，SAD.95/.99，AIG M1/M2；shadow顺序.995/H2/H8/H20。不使用逐seed最佳超参，不用GPU负载引起的耗时噪声打破平局。主增益门使用实际数值，不以近似平局抹去保护项。

**开发门（协调者批准冻结）**：逐dataset，将候选选定shadow与最强S2 A1+EMA配对；mean Δselected validation macro AUC≥.001，且2/2 seed正向，且下面保护为空，才晋级。S2的增强增量对A0+EMA也按同门描述，但失败仍保留最佳A1作正式对照。AIG只以最强A1+EMA为主门，A0+EMA关系并列报告，不追加第二主门。

保护：macro LL mean增加≥.002且2/2增加；worst-cell AUC或最大训练域（MLd2/KRd1）的任一任务AUC mean下降≥.002且2/2下降。任一即不晋级。pooled/traffic反向保留TRADEOFF说明，按规划不单独新增开发拒绝门。S4在满足门的候选内选最佳；都失败则该dataset AIG不进正式五seed，不自动启用AIG-SWA、不追加搜索。

每dataset独立选择一个M配置；若仅一个dataset过门，正式25条、名义总69；两者都失败正式20/总64；都通过正式30/总74。基础44条必须完成。不同dataset配置不同须写明，不用test挑公共赢家。预算为上限，不能把省下的候选预算转移到额外配置。标准98条、边界局部追加、温度校准、AIG-SWA、新自然划分均不在本次默认执行。

## 6. 缓存、正式冻结与统一test

历史GAM01权重已释放，且只维护.995 shadow；不能由数值日志恢复另外三个shadow，因此旧训练结果不替代本轮候选。只复用已核验数据数组/预处理接口。

本轮开发产生的完整RAM轨迹如数据/训练代码/完整配置/seed/初始化、loader与view流/指标/全部shadow语义完全一致，可复用为正式2022/2024输出；ledger必须链接原run并计为缓存，不伪造新训练。若完整身份不能证明则重跑。S1–S4加S5的64/69/74是无缓存最坏名义预算，最终分别报告逻辑输出、真实完整训练数、缓存数和所有失败尝试；不得同时把一次轨迹计为两条唯一训练。

S5前冻结dataset基础lr/wd、A1 κ/λ/ρ、AIG配方/组/阈值/ρ及是否晋级、代码commit/每文件hash、数据身份、完整run清单和选择规则。五seed阶段不能再选ρ或方法配置；每个固定模型的epoch仍仅按其validation选取。

搜索和S5全部有效运行完成、所有selected/E10/E50身份写入 `selection_frozen.json` 后，才统一调用test。只评估冻结的正式A0 online、A0固定shadow、A1固定shadow、获准AIG固定shadow；开发机制分析使用完整validation，不为被拒候选临时追加test。相同model/epoch去重forward，但输出保留selected/E10/E50各语义。test输出之后不改配置、阈值、门或追加trial。完成数值表后释放RAM状态。

## 7. Ledger与成本

先定义试验级hash（完整方法/超参/shadow/阈值来源/组映射）、运行级hash（另含dataset/seed/数据hash/代码hash/RNG语义），stage/device/attempt作为执行元数据单列；缺阈值的后续候选在开始前解析为具体值再hash，不以占位hash冒充最终配置。

trial ledger最少包含：stage/dataset/trial/seed/attempt/status、源码commit及manifest hash、raw/split/encoder/scaler hash、resolved config/hash、初始化/epoch顺序/view身份、预定与完成epoch/step、selected模型/epoch/hash、配置选择及门的输入/输出、缓存来源/理由、失败/OOM/重跑原因。有限候选表预先写入，failed trial保留。

逐run计：自然样本曝光、增强目标视图曝光与view数、训练forward、backward/autograd遍历、optimizer updates、shadow数/更新次数、每shadow和online validation forward/时间、选择次数、test forward/时间、worker训练秒和总wall秒、实测GPU CUDA-event训练/validation时间及计时边界定义、峰值allocated/reserved显存、RAM快照bytes。共享GPU上的worker总秒和event区间均受并发影响，不能当作独占GPU加速结论。SAD/AIG每增强batch两次梯度遍历，普通Adam一次，成本不能只按trial数比较。

复用现有逐卡调度。每张卡自身显存和功耗都低于85%，且剩余显存容纳原batch峰值时继续增加worker；首次真实batch分配回执后再读资源，不要求两卡同时满足。保留外部训练进程。OOM优先降并发、原batch重试，不能缩batch/epoch/seed。数值失败修复具体原因，保留失败账本并按影响重跑；不以任意失败/到时自动削减矩阵。

## 8. DESIGN → IMPLEMENT → REVIEW → RUN验收

coder仅按本文实现，输出 `IMPLEMENTATION_REPORT.md`，可做语法与必要CPU检查，不作研究性能结论。独立reviewer必须核对本文、实际源码、coder报告，PASS或修复全部blocking后的PASS WITH FIXES才能交runner。修改涉及公式/随机流/选择/成本时必须复核，不能边正式训练边改代码。

必要小张量FP64/FP32连续多步检查：coupled wd/m/q/v/bias/step手算、动态λ与常λ退化、λ0/关闭/β.90/β0哨兵、None/显式零、静态组与旧q在当前d0的衰减；验证阈值只取指定训练统计且冻结。沿用有效旧checks，只增加新逻辑所需assert，不扩展无关测试。

review通过后runner执行真实模型CUDA Stage0：两dataset seed2022、原batch40960，最多一epoch等效更新预算，无test。核对raw/数组/模型身份；源A0预测/BCE/梯度/Adam一步；同view混合梯度与拆分一致；SAD一步、AIG公式与U组差分；4shadow递推与buffer、on/off不影响online/RNG；完整RAM snapshot/restore后下一步online/optimizer/q/shadow/阈值/loader/view hash一致；峰值显存与成本计数。Stage0通过才正式S1。Stage0短检查不作性能证据、不作开发筛选。

## 9. 数值分析与交付

最终主端点是validation各自选epoch后的test六格macro AUC。输出每seed原值、五个配对差、mean/sample SD及df4配对t95%区间；开发两seed表标注n=2，不与五seed混用或把cell/epoch当独立重复。五seed包含开发2022/2024，必须注明复用开发seed；历史test也用于先前研究判断。报告共同E10/E50、macro LL、pooled（各任务跨域AUC再平均）、traffic加权cell、worst、所有域任务AUC/LL/样本量/正例率、模型身份与训练阶段。

正式AIG对最强A1+EMA：mean Δtest macro AUC≥.001且至少4/5正、开发方向过门、双方预设保护为空才GO。正式保护延续mean阈值.002及至少4/5不利seed，最大域任一任务/worst下降与macro LL增加单列。pooled/traffic稳定方向交换也标TRADEOFF。未达主门记NO_PROMOTION；有限搜索无独立增量或只剩调参解释时STOP。若AIG未获准正式，明确开发筛选停止和实际缺少五seed候选test，不把其写成已完成阴性正式结果。

协调者独立分析：稳定重复现象、矛盾实验、STRONG/MEDIUM/WEAK证据、优化/增强/EMA/新门控各自效应、规模与基线难度区别、小域改善/大域下降交换、竞争解释与最有区分力的下一实验建议。不能假定动量/课程有效、小域困难或M3oE在困难域失败。机制不明不抹去可靠效应，固定配置失败也不否定整个家族。M1剂量匹配缺口、两seed开发选择及缺少独立新留出评价具体写明。

交付本目录的证据审查/复算、设计/实施/独立审查、Stage0、逐阶段搜索、正式及最终中文报告；source/data/config/threshold/selection冻结身份，trial/run/launch/failure/cost ledger，全部validation原值与正式test原值、逐域任务/配对/训练阶段/门控诊断，运行入口及完整配置。导出最小可复现发布包：本轮代码/配置、实际依赖、完整原始数值CSV/JSON与stdout/stderr/批次视图和行序日志；体量较大时可压缩原始数值/日志并附文件清单/hash，包内不含任何权重或checkpoint。最终由协调者完成报告与授权的新分支推送；所有工件只含数值/日志/代码/报告，不含模型权重。
