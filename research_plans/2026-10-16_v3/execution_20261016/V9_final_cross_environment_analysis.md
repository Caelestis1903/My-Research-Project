# V9：完整基线预算的跨环境边界与窗口对齐

KuaiRand 与 MovieLens 的 SGD .3、无动量、Sample objective、M3OE SN/SB 四种子2022–2025全部正式训练及1000/200完整配对读数已完成，主代理直接 point 数组复核 PASS，最大误差1.96e−16；`independent_review=false`。KuaiRand 最终预算50/50/50/100，MovieLens 全为50。每一环境内部两采样臂匹配；不同环境的输入、标签、行数、参数输入维度和每轮更新数不同，不能将其成组变化当单因素随机干预。test 仍待全部条件冻结后读取。

## 主结果与必要反例

| diagnostic macro AUC | KuaiRand 差值 SB−SN | MovieLens 差值 SB−SN | 解释 |
|---|---:|---:|---|
| 各自共同退化前窗口 | −.000493（1–37） | +.001391（1–16） | 窗口不同，不能直接当环境反转 |
| 两环境统一1–16 | +.000078 | +.001391 | 均三正一负，区间均跨零 |
| 固定50轮 | −.003841 | −.000223 | 前者三负一正，后者两正两负 |
| 各自最终预算 | −.001307 | −.000223 | 前者50/50/50/100，后者全50 |
| 各自完整 selection AUC 选轮 | −.003907 | +.000241 | 前者三负一正，后者三正一负，均CI跨零 |

统一1–16时 KuaiRand SN/SB=.559605/.559683、CI [−.001366,+.001046]；MovieLens=.538007/.539398、CI [−.000131,+.003849]。这说明用各自早期窗口得到的相反均值，不能单独作为“数据环境导致早期符号翻转”的证据。该新观察记录为 Q41/BOUNDARY，证据 MEDIUM。它不证明两个环境相同，也不证明所有差异由窗口造成。

KuaiRand每轮49步，MovieLens每轮5步；统一16轮分别为784/80步。统一轮次不能控制优化进度或样本量。因此既不能仅比较自身不同窗口宣称环境因果翻转，也不能把共同轮次相近符号当作机制相同。原固定参考 SGD .1 的负向早期效应与校准 .3 的阶段反例并存，优化器/学习率/阶段边界必须先于数据规模叙事。

完整 selected 的 KuaiRand CI [−.008249,+.000221]，MovieLens [−.001222,+.001127]；KuaiRand最终预算 CI [−.003619,+.000762]。MovieLens共同早期 CI 未同时覆盖−.003与+.003，末轮/selected区间更窄，冻结规则未触发阻塞性关键零效应扩种子；没有据此宣称精确等价、所有域均无效或唯一环境驱动原因。KuaiRand某些交互不确定，也没有用其关闭零效应因果问题。

统一1–16 macro logloss gap：KuaiRand +.000613、CI [−.000907,+.001702]；MovieLens −.0000132、CI [−.0000576,+.0000103]，两者都三正一负，MovieLens均值由一个负种子拉动。固定50/selected的 MovieLens logloss gap 约−.000407/−.000425，区间跨零。不能由 AUC 均值符号推广到所有损失指标。

## 域、任务、规模与难度

MovieLens训练域 d0/d1/d2 的 N 为34500/3000/150000，训练份额 .184/.016/.800；两个任务为 rating>3 与 rating>4。KuaiRand训练比例约 .08613/.86928/.04459，任务为存储 click 与 long-view。域编号和任务编号只在本环境内有意义，未跨环境强行匹配同编号cell。

MovieLens早期1–16最小域 d1 gap+.005270，两个任务+.004253/+.006288，均四种子正向；最大域 d2 −.001162，三负一正、CI跨零；中等 d0 +.000064，三负一正、CI很宽。最小域收益在本配置早期有 MEDIUM 证据，但“小域改善/大域下降”稳定交换只有 WEAK 证据。固定50时 d1变为−.001368、两正两负；完整 selected d1 +.003392、三正一负、CI [−.002532,+.010335]，持续性未证实。

KuaiRand完整 selected d0/d1/d2 gap约−.008972/−.002048/−.000702，各三负一正；task0/task1 −.002258/−.005557。中等 d0 的 SN 基线 .558936 低于最小 d2 .625781，因此大小不能当困难度标签。MovieLens早期最小 d1 的 SN .532333虽略低于其他两域，标签、支持量和分布仍混杂；三物理域不是足够的独立单位来识别 size 主效应。

MovieLens训练份额与验证份额明显不同，d1验证全集35523行而训练仅3000行。按训练 p 加权的早期cell AUC gap −.000834，而按实际验证行数加权+.001455，与macro+.001391不同；这些是cell AUC均值，不是pooled AUC。max/min训练份额比给两环境约19.5/50，权重ESS比例却约.436/.494，不同标量的“失衡程度”排序不同。不能选一个标量解释环境差异而忽略另一个。

## 边界归因与复制门槛

| 候选来源 | 被控制的部分 | 残留边界 |
|---|---|---|
| objective | 同为Sample目标、任务半和、p/q重要性校正 | 经验总体、p及标签定义不同 |
| optimizer | 同SGD .3、无动量、相同耦合WD | .3来自主环境校准；有效进度、梯度尺度不同，未证明各自全局最优 |
| architecture | 同M3OE家族与计算图规则 | 特征空间、输入维度和数据作用不同，不能唯一排除架构交互 |
| dataset/labels | 每seed内SN/SB同数据；KR训练人口固定，ML训练子集随seed变化，验证/编码固定 | 跨环境成组变化，未隔离特征或标签驱动因子 |
| imbalance/size | 实际支持、标签率、batch/ESS与cell均已审计 | 三域多因素混杂，不能唯一归因某个size/失衡标量 |
| overfitting/stage | 同冻结分类器，完整early/fixed/selected及实际平台预算 | ML未确认过拟合；KR仅B0 seed25 SB diagnostic38确认；不同窗口与step数影响比较 |

这里没有复现一个跨环境稳定、同方向的负向 macro primary 效应。原定 V9 仅在 E2 复现且发现关键 mediator 时再复制那个 mediator；当前无此触发，不新增未经门控的大矩阵。其状态应为明确配置/阶段边界，不能宣称 dataset、size、难度、失衡、architecture中的某一项被唯一识别。若唯一驱动归因是核心必需结论，最小可识别设计须一次只改变一个数据/标签/支持因素并控制其余路径与可实现的更新预算；当前结论保留不可识别边界，不把未做的这一实验称已完成。

跨环境总表中的 repetition/order/geometry/shared-block 第二环境项应诚实标为 NOT_TRIGGERED/未作复制，理由是上述预定门槛；不填入由主环境推断的结果。所有完整 test 仍将按最终冻结清单执行，再更新总体边界。V5/V7/V11等主环境独立必做流程继续。

依据：`V9_final_cross_environment_primary_verification.json`、`analysis/V9_final_cross_environment_{aggregate,joint16_marginals}`、`V9_diagnostic_analysis.md`、`V4_final_budget_and_learning_speed_analysis.md`；Q41保留全部问题字段，不删除原 Q11/Q24。


## Q43：MovieLens种子同时改变训练子集

实际配置再次核对显示，ML四个seed的训练行指纹不同；每域N、p及validation/encoder/scaler相同，每seed内SN/SB数据完全相同。既有role_reversal_sample用RandomState(seed*10+domain)选训练行，故四seed区间包含数据子集与优化随机性的联合变动，不是固定训练人口下纯优化种子方差。该边界不会破坏同seed配对q干预，也不改变上面的实测CI或冻结追加seed规则；它限制对方差来源和跨环境差异的唯一解释。元数据证据STRONG，方差组成在当前对角设计不可分。详见Q43_MovieLens_seed_population_scope_audit.json及baseline_domain_context_analysis.md。原始规划、数据和训练均未改，不为非阻塞的纯方差分解新增矩阵。
