# DESIGN SPECIFICATION — 自然 MovieLens，数据与 A–C 核心阶段

设计者：独立 designer。日期：2026-10-20。状态：**DESIGN COMPLETE FOR CORE；尚未实现、审查、运行**。本文件只授权此处精确定义的核心设计进入 IMPLEMENT；N17/N18/N19/CM19 扩展另见主设计与矩阵。设计不包含新结果。

## 1. Research question

完整自然 MovieLens 上，Equal/Sample 的目标变化、同目标 SN/SB 的采样变化、以及 S5 历史状态/当前更新变化，各自是否仍出现原有现象？差异能否由人口比例、独立样本数量和累计更新数区分？

## 2. Hypotheses

H-A：人工 d2_large 下目标聚合的逐域贡献/风险现象可迁移至自然人口。H-B：同 Sample 目标下 SN/SB 的有限预算差异在自然人口仍存在。H-C：S5-ML 的 history/current 因子图在自然人口具有同方向作用。均可被反证；不预设 Equal 有害、动量有益或小域困难。

## 3. Changed variables

人口条件：`movielens_artificial_d2_fixed2022`、`movielens_natural_matched_n187500`、`movielens_natural`。完整自然人口为主；其它仅作明确控制。目标 alpha、采样 q、history/current 因子按组变化，禁止组间捆绑归因。单独控制原 epoch 预算和匹配曝光/step 的预算。

## 4. Held constants

原物理域年龄 {1,18}/{25}/{35,45,50,56}，标签 rating>3/rating>4，原全局顺序80/10/10切分，原特征和训练期编码，四优化种子2022–2025。每组保留其实际模型、fusion、routing detach、优化器、WD、arch update、batch、原完整 epoch 预算。新实验是自然人口迁移，不冒称原始初始化/历史工件的精确重放。

## 5. Mathematical / algorithmic definition

`stage_F0_data.load_movielens_natural` 是数据唯一基础入口：全局 base-train 全保留，不调用 role_reversal_sample；DataBundle 仍可用 dataset=movielens/setting=natural，新 manifest 必须有 dataset_config=movielens_natural。预期800167训练行、155949/318599/325619逐域仅为核对，正式按原始bytes实际计算。原始csv、每split row IDs/特征标签、编码器及scaler哈希必须保存。seed只改变优化，不能改变此人口。

固定控制人口：从同一自然bundle派生；人工对照严格复用 `role_reversal_sample(..., setting=d2_large, seed=2022)`，不使用优化seed；自然等N对照按自然p采用最大余数整数分配总187500，逐域 `RandomState(20221020+d)` 无放回取样，排序回原row顺序。所有变体共用全局训练期schema与自然val/test。若实际源不足以实现指定数量，设计需修订，不静默改变总N。

损失：L_d=域内两个任务BCE均值，J_alpha=Σ alpha_d L_d。Sample alpha=p，Equal alpha=(1/3,1/3,1/3)。SN/EN q=p；SB/EB q=uniform；IID with replacement 按原 Stage2D Sampler 抽域及域内行；每样本乘 alpha_d/q_d，固定实际batch长度分母。禁止自归一化、权重裁剪或把 batch 内等域均值当作这些估计器。所有p由**实际普通训练人口**重算。

核心正式组：

|ID|原代码与算法|原完整预算|自然配置与种子|
|---|---|---|---|
|AC-2D|`stage2d_20261016.py` 的实际 StageS1M3OEAdapter、learned four fusion scalars、main/arch两不重叠Adam|50epoch，B40960，main LR .01，arch LR .001，WD1e-5，每epoch独立arch批|SN/SB/EN/EB ×2022–25：16runs|
|AC-V9|v3 `v3_train_20261016.py` 的 fixed_model，无DARTS，SGD .3，momentum0|50epoch，B40960，WD1e-5|SN/SB ×2022–25：8runs；完整自然和原人工结果对应，不替换AC-2D|
|AC-S5|`stage_S5ML_run.py`原受控Adam步骤，C/A的history/current来源与P完全按原实现|50epoch，B40960，main .01/arch .001，betas .9/.999，eps1e-8，WD1e-5；原Equal实际系数/scale|F/T/HT/DH ×2022–25：16runs；target固定物理d2|
|AC-S6|`stage_S6ML_run.py`/`stage_S6ML_protocol.py` 的Sample F/DH实际受控步骤（须核实导入链）|原50epoch及原LR/arch参数，不沿用S5 Equal loss|F/DH ×2022–25：8runs；target物理d2|

S5的F=(FULL history,FULL current)，T=(TARGET,TARGET)，HT=(TARGET,FULL)，DH=(FULL,TARGET)。每步只执行原规定的持久optimizer.step，再按原选择的实际delta覆盖共享参数；私有参数始终原FULL更新。不得用冻结参数、置零moment、第二次optimizer.step或简单loss重加权冒充HT/DH。原模型源hash校验若因封装路径改变，需要冻结新的移植身份，不能改掉核心梯度/更新公式。

AC-2D的人工固定、自然等N控制各完整16runs；完整自然16runs已列。另对完整自然AC-2D作匹配预算16runs：每macro-epoch恰好187500 main draws，批大小序列为40960×4+23660，50macro-epochs，恰好250main/50arch步；q/alpha仍由完整自然人口定义。它与完整自然50epoch(约1000main步)分表，绝不称自然完整训练。总AC-2D四条件64runs。这在共同源/特征/评估下交叉比较：人工vs等N自然主要改变支持组成与比例；等N自然vs全自然匹配预算改变独立支持；全自然匹配vs全自然原epoch改变曝光/更新。子集选择仍是一次实现，不可宣称纯比例普遍因果。

AC-S5/S6必须保留自然全预算。比较机制的同step轨迹、250main步前缀与末态；若需要同main+arch步因果比较，新增与AC-2D同样187500 draws/macro-epoch的S5/S6全部臂配对预算（24runs），不得用前缀不同arch步数代替。此匹配组列为必做，不因自然结果为阴性删除。

## 6. Exact affected modules/files

新文件放 `experiments/natural_ml_reassessment/`，只增加数据/协议/封装/队列/评估，历史文件只读。必须从原分支提交获取代码，而非只从main拼凑：当前本地Stage2D/S5/S6；v3来源 `/home/xzf/Code/phase0-3-20261016-v3-git/experiments/`。重现依赖完整归档于本次source tree，避免运行时隐式读可变sibling工作树。基础loader复用当前 `experiments/stage_F0_data.py` 与其 `stage_S1_data.py`。输出本目录、独立 `/home_new/xzf_natural_ml_reassessment_artifacts/`。

## 7. Implementation requirements

先冻结source/config/manifest，再独立review PASS才训练。封装可替换数据入口、输出root和科学协议标签，不能重写原算法为统一训练loop。S5/S6若源码冻结检查不支持自然dataset，需要明确适配参数化，保留旧路径不变。每组逐batch核对loss、梯度和实际Adam一步、参数组归属及none-gradient处理；只做影响科学结论的检查，不进行无关测试工程。coder只报告实现，不分析结果。

## 8. Required ablations

AC-2D alpha×q全部四臂、三人口及全自然匹配预算不可缺；S5四臂不可只留F/正结果，S6保留Sample阴性对照；V9保留无动量SGD反例。S5/S6附匹配预算。角色轮换是历史受控实验，保留其原证据而不机械构造“自然d0_large”。

## 9. Logging

逐epoch与selected/final/共同 E10/20/30/40/50 六cell AUC/LogLoss、宏/按train p与eval p加权cell指标（非pooled AUC）、target/non-target、实际main/arch steps、曝光/逐域unique与repeat、raw梯度、WD梯度、m/v/step、bias correction、applied delta分区摘要。S5保留原moment probe及关键state。selected仅validation原规则最早严格最大；test在配置及选点冻结后评价。保存完整state/RNG/sampler并建立checkpoint/prediction SHA索引，训练次数与离线诊断次数分开。

机制固定点：初始化、E10、E40、selected/final，SN/SB双路径锚；按实际full C/P/A分区，固定同theta/optimizer state，512独立batch Monte Carlo估计均值/全坐标协方差trace和block trace；exact有限分层pool oracle至少32真实行/域，同目标identity及MC误差报告。不得拿少数任意坐标冒充全模型方差。Adam诊断复制同一state/gradient，仅切换当前梯度与m/v来源；跨轨迹硬拼不能宣称history因果。原routing detach的训练导数字段与预测风险Jacobian分别标记。

## 10. Reviewer acceptance criteria

全48个基础自然配置、AC-2D其余48控制及S5/S6匹配24配置完整列入可续跑manifest；重复严格等价才允许alias。原数据/模型/优化器/epoch逐组对应明确；p/q/coeff重算；优化seed不改变人口；val/test无人工域重采样；schema在允许训练数据拟合；S5 persistent state和实际overwrite与原代码同batch oracle一致；所有正式比较四seed、full budgets、selected/final六cell；输出不覆盖历史。满足后方可PASS，阻断问题修正才可RUN。

## 11. Falsification conditions

若自然Equal/Sample逐域贡献或性能方向不保持，缩小历史外推；若SN/SB自然差消失/反转、或匹配N/step控制解释其差，不声称普遍采样损害；若HT/T相对F的历史图在自然不保持，反证自然泛化；若只有final或某task/cell变化，结论限定阶段/任务。机制差异≠AUC改善，路径或moment范数≠因果。报告STRONG/MEDIUM/WEAK、全部seed原值与竞争解释，不以四seed同号冒充跨独立总体确认。
