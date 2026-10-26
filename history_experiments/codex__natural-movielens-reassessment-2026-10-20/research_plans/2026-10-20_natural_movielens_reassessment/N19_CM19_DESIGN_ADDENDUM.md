# DESIGN SPECIFICATION — N19 六方向及 CM19 条件路径自然人口复评

2026-10-20，独立 designer。**DESIGN COMPLETE；NOT IMPLEMENTED / NOT RUN**。只读查验了 N19 原设计及全部14份addenda、FINAL_REPORT、实际RUN_LEDGER/config/队列，CM19原设计、全量manifest、FINAL_REPORT及训练/数据实现。历史ML正式配置已逐行索引于 `HISTORICAL_ML_RUN_CONFIG_INDEX.csv`，**236 N19 +436 CM19 =672条**，含原config完整JSON、来源路径与SHA。这些是历史配置索引，不是本轮RUN_LEDGER或新结果。

## 1. Research question

自然ML下，N19六方向的真实方法/强对照/阴性边界是否复现？P3容量作用是否仍被乘性路径失活混淆？CM19耦合Adam-L2、局部因子正则、旁路、独立STAR/其它条件结构的机制与selected/final性能关系能否外推？哪些现象由总样本量和累计steps解释、哪些在匹配后仍存在？

## 2. Hypotheses

H-E：历史六方向的条件实证边界可迁移到相同模拟机制作用于自然训练支持时；不假设候选会改善。H-F1：M3oE路径/耦合衰减现象不只取决于人工极小d1。H-F2：恢复路径并不保证selected AUC与LogLoss同时改善。H-F3：独立STAR保持活跃的反例在自然人口仍成立。通过完整阴性方法和结构/优化对照允许逐项拒绝上述外推。

## 3. Variables changed

主迁移仅替换基础人口为完整自然ML；四seed不改变人口。P1合成Z/任务、P2噪声/审核、P6观测机制沿**原生成算法和固定参数**作用于自然ML行，标记为 `movielens_natural__stability/noise/sampling`，不能把合成任务结果写作两个native任务。原研究所有model/method/optimizer/variant/noise_interaction/backbone/tag保持逐条对应。仅匹配预算控制改变曝光/更新；仅匹配N控制改变训练支持。

## 4. Variables held constant

统一main/meta和selection/development规则：按域，`default_rng(202219)`对完整自然base-train逐域permutation，meta=floor(.05*N_d)，main为其余；`default_rng(191019)`对原validation逐域permutation，selection前floor(n_d/2)、development剩余。所有比较共用这些固定ID。明确记录全训练基础人口800167与普通main约760161之间的差，不能把meta排除误报成人工域比例采样。

全局base-train拟合schema来自F0自然loader；meta属于允许训练数据，拟合scope原样记录；validation/test绝不拟合。正常自然100epoch覆盖完整main、原B40960、ML LR.001、Adam(.9,.999)、eps1e-8、WD1e-5、float32，无scheduler/early stop。除原original臂外各方法保持原optimizer图。原新/旧source差异逐个报告，不把N19与CM19 original arch-iterator差异抹平。

## 5. Exact mathematical / algorithmic definitions

### E：N19完整主矩阵（236自然迁移训练）

读取原 `research_plans/2026-10-18/N19/queue_manifest.json` 中所有ML `n19_train.py` 正式任务，排除 `--engineering`，与实际236行ledger逐一联结；**保留完整command flags**，以配置中的method/backbone/noise_interaction/optimizer/tag区分run，不能对method简单去重。新配置另写dataset_config=movielens_natural，源码的dataset仍可用movielens以保留原dispatch。所有100epoch、2022–25；训练state/output转至本次独立root。

|研究模块|实际ML配置数|必须保留的完整比较|
|---|---:|---|
|native +P3/P4/P5|92|23个四seed组，见下文；92不等于92个独立假设|
|P1 stability|40|M3oE七方法×4；MMoE fixed/task_mask/task_dro×4|
|P2 noise|72|14个主方法×4 +gamma1非加性4方法×4|
|P6 sampling|32|八mask/CV方法×4，完整标签fixed参考来自native|

native方法原集合为fixed/original/original_sample/mmoe/ple、rank_uniform/rank_protected/domain_codebook/cell_codebook、domain_mix/axis_mix/full_mix/interaction/reliability_interaction、utility_router/regret_router/coalition_router/entropy_router，18个Adam组；fixed及四容量方法另有AdamW共5组，总23组。不要把原 `suite=natural` 当作自然ML：旧缓存仍为187500人工人口。

**P1** 原生Y0=rating>3，合成Y1~Bernoulli(.2+.6Z)。训练Z正确率按随机E为.95/.75、val .1/test .05，环境固定随机化；保留固定模拟seed191901+split。七M3oE臂fixed/global_cf/original_sample/task_cf/task_dro/task_mask/vrex，另MMoE三臂。task_cf只给task0加一致性；DRO=.5[max(BCE0(x),BCE0(flipZ))+BCE1(x)]；mask只去task0输入Z。反向任务不兼容global_cf必须保留。参数、loss scale与原实现相同，不改成普通augmentation。

**P2** 原native嵌套clean labels，a=.35 sigmoid(A_d+B_t+gamma*u_d*v_t)，b=.35 sigmoid(C_d+F_t−gamma*u_d*v_t)，A=(-1,.2,.8),B=(-.3,.5),C=(.5,-.5,.2),F=(.4,-.4),u=(-1,0,1),v=(-1,1)。原191902随机流、2%训练行audit、gamma0/1固定；相同自然人口内两gamma保持clean IDs/audit/uniform draws，只有flip阈值变。所有非oracle方法不可用未audit clean train标签。clean validation仍原条件，明确不是自然MNAR。

14主方法naive/audit_only/ind_channel/axis_channel/nested_ind/nested_axis/joint_axis/clean_oracle/original_sample/audit_fixed/profile_axis/profile_nested/profile_joint/audit_joint全部保留；gamma1只audit_fixed/profile_axis/audit_joint/profile_joint四臂，clean-oracle可严格目标身份复用。joint与composite、nested与非nested、axis与independent、共同拟合与audit-only冻结拟合必须分别对照。profile参数按原Jeffreys伪计数、.35边界、L-BFGS-B设置，不能用simulator真rate拟合或把相同audit的一次估计报成四独立审核。

**P3** 原rank32预热10epoch，SVD canonicalization后统一48active rank、≥4/domain；uniform选每域top16，protected用原worst-cell signed deletion score在同谱基筛选，原moment-reset仅限变换因子。只允许剪枝后epoch≥10选模，必须保留warmup。域字典与task字典均为rank47原构造，task字典两次STAR计算及真实物理存储计数保留。两optimizer完整五方法都必须有；不以路径初期已失活为由取消四seed。

**P4** domain_mix/axis_mix/full_mix/双中心interaction/replicated reliability全部端到端训练。confidence在epoch5/10/...两互斥meta半集逐系数±.1风险斜率，证据一致1否则.25，EMA .9；effective=P(c*P(v))双投影。保留confidence-off=interaction和axis-additive强对照，不把18raw参数当18有效自由度。不得用frozen头代替router+完整backbone训练。

**P5** 同一当前tower逐来源有限函数BCE，individual regret softmin温度.1、KL权重.1；coalition保留原leave-one-out定义；utility为原epoch meta更新；entropy是固定uniform KL，同初始化/架构/scale。后者用于否证“有限信用的独立收益”，不能省。source permutation应一起置换输出和targets，不把单源one-hot风险当完整utility一步。

**P6** 每任务边际pi=.55，joint r∈[.1,.55]，完整八方法mask_independent/mask_positive/mask_antithetic/mask_copula/mask_independent_cv/mask_copula_cv/mask_antithetic_cv/mask_residual_cv。使用原可预测teacher EMA .99、原policy EMA .95、mask/p修正与共同row批；未观测标签在loss和策略前zero-fill。residual policy用BCE−prediction-baseline的表征梯度，仅joint revealed labels及r校正。保留all-label固定参考；1.1 labels/visit为模拟访问数，不等于真实新标注节约；稠密计算成本如实报告。

### E-P3：必要人口/进度鉴别（新增80配置）

对固定/均匀秩/保护秩/域字典/任务字典 ×Adam/AdamW ×四seed，分别增加：

1. 完整自然支持但每macro-epoch178125 ordinary行（原人工187500扣meta的main总数），100macro-epochs、500main步，40runs。每macro-epoch独立permutation完整自然main，截取178125行；短尾按公式178125−4×40960=14285计算，保持40960×4+14285；Sample仍普通样本均值，完整自然main p为期望人口。原rank预热/剪枝在macro-epoch10，明确其steps与自然完整epoch10不同。
2. 核心自然等N187500人口按相同5%meta规则再分main，完整100epoch，40runs；保存实际main总数与domain数，由floor取整可能与178125相差1–2，不篡改ID凑数。如差值改变匹配曝光，预算组按目标实际数量再冻结；不得把近似样本量说成严格相同。

上述控制与旧人工P3结果分别呈现，人口/信息量与优化进度可部分解耦。只有一个固定子集，不声称跨子集普遍因果；人工N19旧四seed同人口，可在source/hash/算法相同时直接作为历史条件参考，但大checkpoint缺失不冒称重验旧prediction。

### F：CM19全矩阵（436自然迁移训练）

从原 `/home/xzf/Code/phase0-3-20261019-git/research_plans/2026-10-19/execution/run_manifest.json` 取所有ML独立runs；与436行ledger及source config比对。使用 `cm19_manifest.py` 原笛卡尔积，不从报告挑好方法。

|组|自然ML独立训练数|原配置|
|---|---:|---|
|G0|80|fixed/original/original_sample/star0/star1/pepnet/hamur0/hamur1/mmoe/ple ×Adam/AdamW×四seed|
|G0M|16|star_joint/hamur_joint ×两optimizer×四seed|
|G0MC|16|star_private_joint/hamur_private_joint ×两optimizer×四seed|
|G1|60|fixed/star0/star1 ×Adam×wd0/shared_wd0/slot_wd0/both_wd0/local_adamw ×四seed|
|G2|96|fixed/star0/star1 ×no_bypass/additive/bypass_scale/no_modulation ×两optimizer×四seed|
|G3|104|PEP三个neutral；HAMUR0/1各no_adapter/static/private_hyper；MMoE/PLE各private/private_matched；两optimizer×四seed|
|G4|48|fixed/star0/star1 ×Adam×d2独立支持fraction .25/.5 ×normal_epochs/matched_updates×四seed|
|G4LONG|16|fixed/star0 ×两optimizer×四seed，原max_steps2350|

100epoch普通组。G4LONG保持完整2350main updates；自然main约19steps/epoch，现约1900步完整100epoch，原人工500步；max_steps组另外训练到2350，不以自然100epoch已有更多步为由取消。原训练最后一个不完整epoch照原实现评估并保留实际曝光。G4的target仍原物理d2，不因为缩小后d1成为最大域就变更target；对应角色跨越必须在分析指出。matched_updates按原自然100% main逐域计数曝光，小target以replacement抽取，非target无放回；保留mix/updates但改变独立支持及重复噪声，不能叫纯N操作。

独立STAR保留PN domain moving moments、每层shared⊙domain矩阵、additive bias及独立aux branch；任务0/1分别单任务，不当联合MDMT。联合扩展与双独立主干同均值任务loss，但容量不等，保持原两个控制。PEP中性gate=1，HAMUR adapter=0、保留author-style batch normalization/eval batch依赖；MMoE/PLE expert output0不renormalize gate。各中性化量纲不同，不按Δp跨模型排名。原HAMUR并非严格论文domain-specific DN，必须保留此边界，不暗中“修论文”。

original：原Equal六cell sum及重叠fusion optimizer为明确source-style协议，保留CM19 cycle auxiliary iterator；N19的original每epoch重复first-shuffled-batch不能静默换为该cycle协议，二者不能alias。original_sample桥接保留统一Sample但不声称只改变一变量。

G2 additive与capacity_replacement原为完全相同图，可名义alias但不得算新训练。Adam/AdamW WD0等价和ML名义AdamW WD1e-5实际收缩float32舍入问题，须先保存同kernel/dtype/device实测有效系数，并在本轮逐run核对完整轨迹/selected/final SHA后才能alias；单步公式同名或10step近似不是全路径证据。**保留lr=.001、WD1e-5与原数值精度，不暗中增大WD/切float64来制造正解耦收缩**。报告名义AdamW与有效decay强度分别是什么。

### F-M：必要人口/进度控制（新增72配置）

fixed/star0/star1 ×Adam/AdamW×四seed，分别：人工d2固定2022（24，若新source完全等价且历史配置/原始指标可核实，可作为历史复用但不能称新训练）、自然等N187500（24）、完整自然支持但178125行/macro-epoch100周期的匹配曝光（24）。普通全自然G0的24父配置已包含，不重复计。与E-P3固定组严格等价可能alias，但至少要求模型state字典、loss/BN/RNG/每批次/step/各参数WD/selected规则全一致；N19/CM19计算图近似不充分。

F-M保持实际Sample目标。完整自然每macro-epoch独立全main permutation截取178125，保证预期q=自然p；固定100macro-epochs，500updates，与原100epoch人工main预算一致。必须记录mode=support_full_exposure_matched，不误写完整自然epoch。自然等N按本轮公共meta规则实际取整计数，必要时匹配曝光按实际人工参考main数而非硬猜178125。

## 6. Exact affected modules/files

建议独立 `experiments/natural_ml_n19/`、`experiments/natural_ml_cm19/` 两子树；不碰core/extensions所有权。原完整算法源优先vendor：N19 `n19_{data,models,objectives,train,evaluate,dynamics,rank_diagnostics,routing_diagnostics,credit_diagnostics}.py`、实际依赖model modules；CM19 `cm19_{legacy_data,legacy_models,models,hamur_reference,train,evaluate,local_mechanism,offline_readout,probe_intervals,costs}.py`及`cm19_sources`。绑定c440350…与eab817…源提交、每文件SHA。独立dataset config/paths注入；数据pop函数底层换F0，suite模拟器保持原实现。不得运行时从未经snapshot的可变sibling import算法。

## 7. Implementation requirements

完整逐配置映射OLD→NEW，保留历史flags；源config的train_rows/hash/param计数等派生值在自然人口重算，不照抄成假证据。各组初始化/普通训练/审核或meta/选模人口边界清楚；仅相关训练正误检查：计算图/梯度、mask身份、秩剪枝后资格、group参数归属、neutral intervention及恢复、eval批次和实际WD收缩。源算法如果发现结论相关bug必须独立修订spec，不把修复称严格迁移。所有正式配置完整预算，不使用早期probe否决剩余seed。

必须生成可恢复运行manifest，phase=N19_NAT/CM19_NAT/…，status从DESIGNED→IMPLEMENTED→REVIEW_PASS→QUEUED→RUNNING→TRAIN_COMPLETE→EVAL_FROZEN→EVAL_COMPLETE→ANALYZED，不将exit0自动当全预算完成。失败保留attempt，重试不换seed，不覆盖旧工件。测试必须完成设计/模型选择/development处置冻结后零更新评价；本次不再根据旧test扩大调参。

## 8. Required ablations

原236/436逐条全部必做或严格身份复用，含P1负对照/P2非加性/P3两optimizer/P4axis/P5uniform/P6静态CV，以及CM19G0–G4。E-P3与F-M匹配控制必做；不能用冻结module-off代替no_modulation/no_adapter从头重训。无需新造所有理论构造的自然版本、无需扩展KR训练；原natural KR结果保持独立参考。

## 9. Logging / mechanisms

所有组selected/final完整six-cell（单任务three-cell）AUC/LL、train p/eval p加权cell、macro、正负数、常量风险/skill、模型/任务/阶段/steps/examples/unique行/重复/成本。统一保存原命名下的完整state、逐batch/split/source/config hash、原始预测npz并索引大小路径SHA。不能把weighted cell AUC叫pooled。

P3保存warmup与剪枝前后有效谱/active rank/SVD重构误差/Adam reset对象、选轮≥10、紧凑导出同预测误差；P4slopes/confidence、effective interaction与贡献；P5两互斥meta半集finite/derivative/leave-one-out风险及复制；P6真实labels/joints/endpoint次数、EMA及策略额外成本，未观测label不记录进策略；P2audit行数/类计数/拟合收敛/rate RMSE及gamma。

CM19原step0/1/5/10/25/50/100/200/500/1000/1500/2000/2350（达到才记录）、epoch1/2/5/10、selected/final：modulation neutral Δp/Δlogit/P50/P95、effective product/activation、factor双精度norm、data-grad/L2-grad/m/v/step/applied delta。probe固定meta/development512行/域，原数组和rows保存；factor范数小不等于功能死，要求跨step功能零/近精度证据。whole-main-off与input-weight-slot-off分开，不能称整个M3oE死。单步同state仅切target规则、其它参数固定；完整训练则未来所有梯度可变，两种证据分列。初态数据梯度比值不能当充分失活定理。

## 10. Reviewer acceptance criteria

236+436原配置无遗漏；原预算未缩；所有匹配控制及aliases可机器计数；数据meta/选择hash固定且种子不变人口；算法源与addenda忠实；noisy/synthetic/label-mask真实信息边界；original参数重叠仅原有臂有，不能全局“修复”；选轮/末轮和测试冻结完整；实际AdamW浮点系数与别名证据正确；所有声明参数存储/compute区分物理导出与supernet。reviewer必须读设计、代码、coder report后独立PASS，blocking修复后才RUN。

## 11. Falsification and scientific decisions

自然native selected候选若不优于匹配强控制，即保留或加强原有限范围阴性；P1相对ERM大幅收益若消失则反证其人口迁移；P2轴假设误设损伤若不复现须按审核支持/类数解释；P3调制活跃但秩收益消失，否证“恢复路径即可保护”；自然M3oE与STAR都活跃会缩小KR失活普遍性；自然Adam-L2失活且局部移decay无效则挑战旧机制推广。单步恢复无效≠不可恢复；完整恢复但selected损失变差≠通用修复。匹配steps/支持改变路径边界时优先报告原叙事被削弱，不找有利超参。

每方向结论：保留/缩限/修改/反证/未识别，分别标STRONG/MEDIUM/WEAK，给原数值、新四seed、逐task/domain、baseline难度、训练阶段、有限正例、人口与预算；明确有无“小域增/大域降”及反例。所有新矛盾登记OPEN_PROBLEMS，不限数量。基于已反复使用test只做探索性复评，不把四优化seed一致称跨总体确认。
