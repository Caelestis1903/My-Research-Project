# DESIGN SPECIFICATION — N17/N18 与前序融合边界的自然人口补充

2026-10-20，独立 designer。**DESIGN COMPLETE；NOT IMPLEMENTED / NOT RUN**。可独立交付 coder，代码专属目录 `experiments/natural_ml_extensions/`；核心 coder 的 `experiments/natural_ml_reassessment/` 不可覆盖。主数据必须调用已冻结核心自然loader及人口配置。产物根采用 `/home_new/xzf_natural_movielens_reassessment`，每研究独立子目录。

## 1. Research question

自然 MovieLens 上，N17 对称调参、相同 Sample 目标采样、选轮/末轮方向矛盾和域计数解释是否保持？v2固定融合协议及AO04融合自由度的人工人口结论能否外推？N18哪些数学/固定KR诊断不依赖ML比例、哪些ML经验量需要重算？

## 2. Hypotheses

可被否证的主张是：N17 selected SB−SN为负而末轮可为正的阶段边界不只是人工3000行小域的结果；同目标估计器的计数项并非唯一机制；AO04增加非对称融合自由度没有稳定收益的结论可跨人口。N18普通LP/等价性等理论命题不因换数据自动变成新理论；不把其DATA-BLOCKED或未触发项写成方法阴性。

## 3. Changed variables

仅数据人口由固定人工d2_large（AO04历史d0_large）迁移至完整自然人口。各自原算法和预算不变。N17追加q×LR四格及匹配N/曝光预算用于分开LR、人口、进度；quota是独立机制变量。所有实验四正式优化种子2022–2025；两pilot种子2030/2031仅用于调参，单独列账，不计四正式种子。

## 4. Held constants

自然数据、split、schema和row-ID固定；N17 fixed_model（sigmoid(1)融合、原routing detach）、单优化器、mean-row mean-task BCE×p/q、B40960、WD1e-5、无clipping/scheduler/early stop。original-style AO04不能换成fixed_model。N18原KR-only teacher/数学实验原样保留，不因为日期早/候选阴性删除。

## 5. Exact algorithms, budgets and comparisons

### D-V2：前序 fixed-fusion 对照

复用 `/home/xzf/Code/phase0-3-20261016-git/experiments/v2_20261016.py`、其模型/采样器实际实现，完整50epoch。ML原四种子 SN/SB×Adam .01 / SGD .1、momentum0、WD1e-5，共16自然runs。其 fixed fusion/单optimizer不能被核心AC-2D learned fusion/两optimizer替代。若与其它新组完全同初值、逐batch、loss和实际step一致，可alias；同名字/同LR不充分。v3唯一正式人工ML组SGD .3/H50已在核心AC-V9覆盖；KR-only AdaTask、ORDER、mask和破坏性beta2=0结论不依赖人工ML，不机械新增所有ML负对照。

### D-N17：N5-C 对称调参与阶段反证

源 `/home/xzf/Code/phase0-3-20261017-git/experiments/n17_ml.py`，继承 `n17_capture.Runner` 和原 `n17_train.py`。保持原逐步训练/保存逻辑与 `n17_common` 的固定模型，替换bundle/split/output路径；每epoch完整自然训练人口IID-WR曝光，100epoch，尾批保留。

1. **16 pilots**：Adam LR=.0003/.001/.003/.01 ×SN/SB ×2030/2031，完整100epoch。按两q两seed各自1..100 selection宏AUC严格最大值平均选共同LR；各q分别选独立LR；精确平局较小LR，完整响应面保存。不按早期排名淘汰。
2. **16 required formal runs**：q×LR(.0003,.001)完整2×2 ×四seed。该四格回答历史下一项明确反证问题，不因pilot选别的LR删除。如共同或独立赢家在.003/.01，追加对应q的赢家四seed，完全重复配置复用；最多另16runs（两q可有两种额外LR）。共同LR与各自LR的估计对象分开。
3. **16 exposure/update matched runs**：对上述四格，完整自然支持/p不变，每macro-epoch187500 draws，40960×4+23660，100macro-epochs，500main步。保留新自然100epoch约2000main步完整结果。这是外部预算控制，不称完整自然epoch。
4. **16 matched-N runs**：核心 `movielens_natural_matched_n187500`，同四格完整100epoch，500main步。保持与完整自然相同schema/val/test；p按此固定子集实数重算。原人工固定2022的旧N17结果保留为第三人口；历史8seed与新4seed不伪配对。若主结论需要直接相同seed人工比较，可在本轮2022–25完整四格重训人工固定人口16runs，必须另列新增而不冒充历史复用。此补充只在要给人口条件的配对因果差时必要；纯历史对照可保持不同seed估计范围。
5. **8 quota runs**：共同LR冻结后SN-quota/SB-quota×四seed×完整100epoch；IID父轨迹复用。quota由原N17 `n17_quota.py`系统随机舍入，n_d=floor(BQ_d+U)−floor(BQ_(d−1)+U)，每样本实际权重B p_d/n_d。每短尾按实际B重算，不能沿人工系数。自然数据计数机制需有完整训练阴性/阳性证据，不只比较初始化oracle。

100epoch为主预算。保留N17原训练J平台分支作为预注册次级：每臂rho=(mean J81:90−mean J91:100)/max(abs(mean J81:90),1e-12)，共同LR任一q平均rho>.005且≥3/4 seed正、全部八父轨迹完整有限时，两个q全四seed延长200。原N17为八seed/6同号，此次4seed/3同号是明确迁移，不能隐瞒。H100 selected结果原样冻结，H200另列，不能以test触发。

主选轮1..100 selection六cell宏AUC最早严格最大；保留E10/20/50/100、AULC1..20/1..100和selected/final。不得用final正号替代selected结论。eval同batch及连续顺序按原实现。

### D-N17机制诊断（正式四seed，非额外训练seed）

在共同LR的SN/SB各自E10/E40双路径锚做固定点（16锚）。N2四估计器每锚512独立batch，保存全坐标ALL与C/P/R/E trace、域内项及计数项：Cov_IID=(Σp²/q Sigma + Σp²/q mu_d mu_d^T−mu mu^T)/B，quota条件协方差Σp² Sigma/n_d；有限pool及MCSE明确，不能以任意256坐标替代ALL。

N3每锚32独立配对draw，同theta、同原生m/v/step、同采样时钟，测native SN/SB、整体C替换、C范数×方向四格及C子块embedding/shared STAR/skip/star_mlp/shared experts、P参考。比较hybrid−SB、hybrid−SN、相对未更新状态AUC/LL/J；attenuation只在|gap|≥1e-4辅助展示。每子块预先测完，无最大块选择，因此无需在同一4seed重新选择再称独立确认。方向范数<1e-12标未定义。使用独立draw bank重复首个bank，报告bank与seed层次；必要MC加到64draw仅按精度规则，不能追正号。

同state Adam即时诊断另包括实际分母与滞后一拍共同分母映射、无动量SGD映射；不硬拼异轨迹m/v当成因果history。不能把全局update等范数当逐块等幅。历史N4长期拓扑在原设计未触发：本次**不是历史遗留未完成训练**，不自动铺开private-C。若自然诊断产生新的拓扑问题，应先由designer补规范、再实现review训练。

### D-AO04：历史融合阴性边界

源码当前 `research_R2/src/ao04_v12_{model,runner,protocol}.py`，实际config为ML/d0_large；B0/B1/B2，原50epoch、B40960、mainLR.01、archLR.001、betas(.9,.999)、eps1e-8、WD1e-5、每epoch一个persistent-shuffled-iterator训练arch batch，12自然runs（四seed）。保留原模型、rho参数及B2控制，不恢复从未触发B3/B4。自然数据/新初始化按原B0→B1/B2配对映射，维度变化明示；不引用人工初始化hash要求自然完全相同。

补固定theta rho扰动、a_D(1−b_D)、E_j−E_k、logit Jacobian、branch/gate entropy和selected/final逐cell风险；未保存checkpoint的历史量不能凭新模型填补。任务/聚合与本轮核心范围相同。B0/B1/B2完整训练后才能判断“额外自由度无收益”的自然范围，不用rho小变化替代性能。

### D-N18：可复用数学与需要自然人口重算的证据

完整审查来源 `/home/xzf/Code/phase0-3-20261018-git/research_plans/2026-10-18/execution/{DESIGN_SPECIFICATION,FINAL_REPORT,AMENDMENT_01..03}.md`。

P1支持LP、P2同观测世界/普通LP/嵌套逻辑等价、P3负环与额外cell头函数类等价、P5 D3产品gate对称不依赖ML比例，保留证明和构造，不重做假训练。P3冻结头、P4功能Jacobian、P6方向IS测量父teacher是KR/SN/E10、不是ML；这些结论仍限定原KR，不因ML人工比例失效，也不自动增加自然ML教师训练。

需要重算的是N18 P1真实ML schema/时间环境可用性、P2 native ML joint-label分布、每seed MAR(.4)与MNAR(.7正/.2负)遮蔽、嵌套逻辑缩界、相同信息普通LP及同预算random/task0-stratified audit。严格复用 `n18_theory_data.py` 与 `n18_common.py` 中原定义，只替换population自然loader，不引入额外干净label信息；固定优化/遮蔽seed2022–25，四seed是mask重复而非四独立人口。原timestamp若仅在原csv存在而编码输入未保存，应审计原始对应row，不伪造时间；无合法环境仍DATA-BLOCKED。此组为离线诊断，0端到端训练；不借此删去N19实际P1–P6完整自然补充。

## 6. Affected modules/files

专属新目录 `experiments/natural_ml_extensions/`，包含冻结历史source vendor、N17/V2/AO04/N18独立data injection封装、manifest、runner dispatch、原始collector；不修改核心目录或历史分支。完整依赖按pin归档，source_path/commit/SHA写入manifest。自然数据从核心bundle接口读取；若接口变更双方协调，不能各自重做不同split。

## 7. Implementation requirements

复用实际算法，不用统一简化网络替代。N17 capture保存/恢复完整model/optimizer/sampler/RNG；自然数据seed无人口变化；quota实际权重与短尾correct；AO04原算法和单独arch参数组/更新频率保留；原有F0控制/hold不修改。coder完成 IMPLEMENTATION_REPORT，独立reviewer核对原设计、actual代码与report，PASS或修完blocking后方可luna-runner运行。

## 8. Ablations required

V2固定Adam/SGD全对照；N17四格含历史缺SN.0003，不能只训练SB赢家；对称完整pilot、IID/quota、matchedN/预算；AO04 B2不能省；N18 ordinary同信息基线不可换弱对照。历史未触发N4/B3/B4不记阴性。严格数值同轨迹才alias，日志明确名义配置数与实际训练数。

## 9. Logging

共同核心指标外，N17每epoch all-model/state索引、selection/diagnostic row hashes、q/LR/预算/seed分列，quota counts与权重、完整gradient/Adam probes；AO04rho/fusion记录；N18有限population/audit/mask/hash及oracle信息边界。test读取在方案/config/checkpoint选点冻结后，由机械runner零更新评价。沿用已反复使用test的探索性标签。

## 10. Reviewer acceptance criteria

D-V2 16、D-N17 72基础配置（16pilot+16formal+16匹配曝光+16匹配N+8quota，额外LR/H200条件另列）、D-AO04 12完整列账；N17formal2022–25与原8seed明确区分。每个组保留原预算/算法，无法匹配原source/权重的缺口不能伪造。固定点分析区分训练导数字段与功能Jacobian；P3/P4/P6 KR-only历史不被误归为ML训练。N18自然数据诊断可复算且不虚报正式训练。

## 11. Falsification conditions

自然N17的same-LR四格若不保留SB晚期优势，反驳纯q推广；若matchedN或matchedbudget改变现象，缩小人口/阶段归因；quota若显著改变gap且绝对风险健康，反驳计数不重要的跨人口外推；反之仅trace减少不支持性能修复。AO04若自然B1相对B0/B2稳定超过原实际界且cell/LL无害，原“无额外收益”仅保留人工范围。N18普通LP身份继续成立不能提升创新性，DATA-BLOCKED条件不得靠合成N19替代。
