# DESIGN SPECIFICATION — RCF-v1 执行设计

设计日期：2026-10-26。设计状态：DESIGN_COMPLETE；尚无本轮训练或性能结论。

本设计把已发布方案变成实施合同，不另行设计算法。完整数学与训练语义以 `research_plans/2026-10-26-v2/05_核心算法与完整实验规范.md` 为准，分支以04为准，验收以08为准。本设计已阅读AGENTS.md、AGENT.md、GB18030编码的start.md及该目录12份文件；10/11是历史核算与来源收据，不能移作本轮验收证据。计划取得于 `research-plan/2026-10-26-v2-mechanisms@92a73cec187838c2e8baea577620abe32dda31b5`。

## 1. 决策问题及范围

检验：在冻结自然比例fixed-fusion M3oE、强目标基准B及最多两个source/task后继函数的条件下，真实目标群体剩余风险组织的函数是否提供普通来源读取、统一锚定微调及充分容量target-only拟合以外的增量。来源作用分为隐藏初始化、需求特征和来源行为约束；不得将其中一条路径的存在当作有用性的证明。

S0、S1、S2对两个数据集、开发seeds5201/5202完整执行。S1负结果仍触发一次有界S2三臂。S3/S4严格按已登记条件触发；未触发写NO_START，数据或源码缺失写BLOCKED_ARTIFACT/PROTOCOL，不能写零效应或算法失败。独立SCALE分支先数值桥，再按真实历史身份执行有限补齐。不得因单一seed负向而停跑、换seed、扩K或换指标。

## 2. 开始实施前的事实与优先级决议

工作区 `/home/xzf/Code/phase0-3` 是恢复的文件树、没有.git；不能编造本地branch/HEAD或初始化后冒充历史checkout。父编排器已核对远端当前main `19620d91f517fce01d4b162eef20ae4749d845e7`；实施身份应分别记录远端参考、计划来源和实际本地逐文件hash。之后远端变化重新记录，不覆盖本地未提交文件。

父编排器报告：两份raw数据SHA与规范相符，strict_v1数组位于 `/home/xzf/data/mdmt_next_20261024_strict_v1/{dataset}`；26个固定v4纯源码/vendor文件已恢复到新目录，SOURCE_MANIFEST逐文件核Git blob与SHA256。以上是依赖可用报告；正式audit仍须实际核角色、编码、标签及数组身份，不能直接把父报告填作本轮全部PASS。GPU0空闲24GB；GPU1有其他进程使用约21GB，不终止/修改该进程。按单卡显存及功耗均低于85%才增加训练并发；不以两卡同时空闲为条件。所有Python使用项目环境绝对解释器或conda run。

**存储决议 STORAGE_MEMORY_ONLY_V1：** 最新start.md第14条优先于旧05/08的“临时模型/optimizer文件”和目录示例。不在任何本地路径保存模型、检查点、optimizer/RNG训练快照或等价权重归档，包括临时文件和改扩展名后的NPZ。完整状态只以CPU/GPU内存中的深拷贝持有；允许保存实验数据、逐行h/z/φ与预测、日志、代码、配置、身份hash和分析报告。读者系数/dual/约束值可以作为研究数值表导出，不导出可加载模型或训练恢复包。

因此按prefix设长驻内存worker或同一进程campaign：native rolling-last/selected/epoch24、pilot和后续必要参数在已登记分支与outer导出完成前不能释放。多CLI命令通过长驻worker或同一进程API交接；不能每条命令退出并把必要模型丢失后声称resume成功。完整恢复验收在内存中保存model/Adam/RNG/row计划/dual后恢复并比较下一步；进程中断只准同身份确定性重放，不假称磁盘恢复。通常执行不得重复构建四条native来代替正常状态交接；意外重放单独记成本与原因。若硬件不能维持内存合同，列明确阻塞，不擅自写检查点。

**源码决议 BRIDGE_V1：** 冻结vendor字节保持不变。新包桥接可复用RoleBundle、NativeAdapter、PilotBank、稳定BCE/KL及指标/hash工具；不得调用旧pipeline/full_plan/decide/cleanup。RoleBundle旧purpose名单不足以表示RCF的P_form形成/约束权限，必须由新protocol显式授权与记录，不能偷偷通过semantic_check绕过账本。旧make_native内部硬编码旧plans.torch_seed；新桥接直接按原构造顺序建立Original、排除_weight_*的canonical backbone、复制到fixed模型，再包装NativeAdapter，只替换为RCF登记seed流并记录初始hash。保持native所有实际运算和gate-input detach。来源缺license时保留实际已取得许可证及来源链接；不存在的文件不可伪造。

09初始配置不被静默改写。实施前产生effective config及SPEC_RESOLUTION/identity记录上述存储、桥接与新stage RNG注册表，明确不改变算法目标；若发现05与09实质冲突，先修订有效配置并登记版本再运行。

## 3. 数据、角色、RNG及指标合同

原始SHA：ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`，KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。沿用v4整数行划分，fit-only词表/age缩放/OOV，不重新拆外层，不重采样域人口。fit域数量ML[124759,254879,260495]，KR[138225,1395118,71561]。ML任务rating>3、rating>4；KR实际is_click、long_view。

| 角色 | ML行数 | KR行数 | 本版允许用途 |
|---|---:|---:|---|
| fit | 640133 | 1604904 | native、source/local pilot、normalizer、群体切点、来源KL输入 |
| P_form | 24004 | 60183 | B折外/refit、需求/分配、形成目标、来源约束；FULL_SOURCE_ALL |
| P_read | 24006 | 60184 | 最终冻结内容reader；直接头规定续训 |
| P_cal | 16003 | 40123 | 完整六头政策选择、选择期来源保留审计 |
| P_update | 16004 | 40123 | 主线不使用 |
| Q_explore | 40008 | 100307 | 开发评价、机制与登记分支决定 |
| validation | 100021 | 250767 | native选点；后续只伴随评价 |
| test_exposed | 100021 | 250767 | 所有清单冻结后的统一复评 |
| Q_legacy_confirm_exposed | 40009 | 100308 | 所有清单冻结后的历史复评 |

audit核对原始→prepared标签/域/编码、角色不交叠且穷尽、每格正负支持、每角色row hash及schema hash。ML恢复原始user用于支持与配对bootstrap；KR缺可信user时明确只能row-bootstrap，不能造user。所有训练/选择访问按purpose、role、task、row hash/unique数、重复次数、方法登记LABEL_ACCESS_LEDGER。test标签不得用于选择；支持计数audit与性能评估权限分开。

额外RNG使用SeedSequence([seed,dataset_id,91000+stage,source,task,unit,pass])，dataset_id为ML0/KR1。新protocol在build-plan时冻结唯一整数stage registry，并分离初始化、query和aux row计划；不得使用进程顺序或Python hash决定随机流。P_form两折按固定v4域内行协议、两任务同折，不重新随机划分。

主指标六格等权Macro LL，LL_dt为该域该任务行均值；同时输出六格AUC/LL、Macro AUC、自然总体LL、每任务pooled AUC、固定群体/支持、selected/终态。ΔLL=方法−参照，负有利。统计重复单位为seed；群体/任务/fold/bootstrap不增加seed数。原始配对预测重算指标；ML user配对bootstrap1000次、KR row配对bootstrap1000次并限定相关性解释。开发n=2仅资源决策，不能声称强统计确认。

## 4. S0共同prefix及强基准

Native固定3域2任务、4 shared experts、fcn_dims[512,256,256,64]、exp_d/exp_t/bal_d/bal_t均1、无learned fusion/AutoML/额外融合optimizer。自然Sample两任务目标sum(BCE)/(2*真实逻辑batch行数)。batch40960、短末批、24epochs完整跑；validation Macro LL仅在12–24选完整状态（平局最早）。coupled Adam lr.001/betas(.9,.999)/eps1e-8/wd1e-5，FP32、AMP/TF32关闭。OOM先降低并发/排查分配，不能悄改batch。

NativePacket的H为四个64维shared expert各自无仿射LayerNorm(eps1e-5)后拼接256维；Z为实际native两logit，不能用shared均值/STAR替代。保存各角色H/Z及raw-X一致性收据，模型留内存。

每source/task pilot为256→32 ReLU→1，在source fit监督12个自然fit passes；a_s0=Z_q+f_s0(H)，source-owned BCE sum/(2*logical B)，同native优化设置，缺源head在前向前跳过并保持grad=None。之后冻结。保留原随机隐藏层供RANDOM_FEATURE；矩阵φ用source fit隐藏激活均值/std(floor1e-3)标准化，但delta复制原始隐藏参数。

每target/task B输入[H,Z]258→32 ReLU→1，直接logit。target fit归一化、std floor1e-3/clip±10；隐藏Xavier/bias0、末层weight0、bias为本训练角色正率logit（clip[1e-4,1−1e-4]）。另一折P_form训练20passes/batch512/Adam.001/wd0生成B_OOF；全P_form独立refit20passes。不以外层选epoch。B_OOF独立性仅限B，A与delta已经使用全P_form标签。

H_DIRECT复制B_refit与Adam、P_read续训10passes，P_cal整组六头在pass0/5/10选、平局最早。ETA_DIRECT_MATCHED输入[H,Z,f0q,f1q,f2q]261维，其余相同，不继承旧eta状态。H_DIRECT_EFFORT在P_form60passes再P_read10，候选0/5/10，多出的标签访问和成本公开。

固定群体g=1[Z_q>target-fit中位数]，相等归0；所有方法共用冻结切点，不能用B漂移重分组。每个target两个独立local pilot按同12pass本地fit训练，供FULL_TARGET/LOCAL_ACTIVE；单独记重复监督和容量。

## 5. S1固定内容比较

完整执行NATIVE、LOCAL（本地pilot加native offset）、H_DIRECT、ETA_DIRECT_MATCHED、H_DIRECT_EFFORT、GROUP_CAL、PILOT_PROB、PILOT_LOGIT、FULL_SOURCE、FULL_SOURCE_ALL、FULL_TARGET、RANDOM_FEATURE。

PILOT_PROB三个概率分别sigmoid(B_refit)、sigmoid(Z+f_source1)、sigmoid(Z+f_source2)，稳定混合BCE使用对应logits；gate输入[H,Z,三个概率]261→32→3，P_read10passes，P_cal整组选择0/5/10。PILOT_LOGIT内容是f_source−f_target，以相同组内αB+β为背景。FULL_SOURCE为两来源各32维标准化φ；FULL_SOURCE_ALL同内容但P_form+P_read训练，以B_refit为offset，明确非OOF。FULL_TARGET用两个本地pilot64维，RANDOM_FEATURE用source pilot初始随机隐藏层64维。

所有线性reader按第8节；FULL类的特征系数不设逐维box，不能人为削弱以制造谱/RCF优势。S1状态只描述固定内容，四个prefix均继续S2，不继承旧KR关闭门槛。

## 6. 需求与分配

对每source/task在真实foreign target的P_form上建立至多四个(t,g)。n≥256、正负各≥20才有效，否则LOW_SUPPORT。定义e=y−sigmoid(B_OOF)、v=p(1−p)、nuisance n=(1,B_OOF)，所有b_phi=E[eφ]、b_n=E[en]、H_phi_phi=E[vφφᵀ]、H_phi_n/H_nn均为组内条件均值。

rho_n=max(1e-8,1e-4 tr(H_nn)/2)；profile score b_tilde=b_phi−H_phi_n(H_nn+rho_n I)^−1 b_n。pi_tg=.5*n_tg/n_t，剔除无支持组后在有效组重归一化，质量只乘一次。C0=sum(pi H_phi_phi)+.1 Cov_source_fit(φ)，rho=max(1e-6,1e-3 tr(C0)/32)，C=C0+rho I。FP64、先对称，负特征值超相对1e-8或condition>1e10时报实际数值失败，不暗改ridge。

r=C^−1/2 b_tilde；至多四点按pi加权枚举非空二分、首群体固定slot0，按加权平方误差最小，平局成员ID字典序。无有效组或全部norm≤1e-10则K0/NO_ESTIMABLE_REQUEST；单有效组K1；槽聚合支持不足则合并K1，不补采样。A、K、π、metric/score、支持与成员在形成前冻结。LOW_SUPPORT与无需求分开。

## 7. S2形成目标、计划与投影

RCF每(s,q,k) delta为256→32 ReLU→1，8257参数，最多12槽/99084。隐藏复制source pilot原层+配对N(0,1e-3)噪声，末层严格0；source/native/B/A/normalizer全冻结，delta隐藏与末层可训练。首步隐藏0梯度预期，末层应可学。形成目标是B_OOF+delta；最终只读取delta，f0+delta仅用于来源保留。

每槽Z_k=sum(pi A)；T=sum(pi A E_group BCE(B_OOF+delta,Y))/Z_k；K=E_source_fit KL(Ber(sigmoid(a_s0)) || Ber(sigmoid(a_s0+delta)))；S=E_source_P_form[BCE(a_s0+delta,Y)−BCE(a_s0,Y)]；R=.5 sum_foreign_target E_target_P_form delta²，覆盖全部可能foreign读者而非仅assigned组。

损失T+.1K+nu(S−.001)+xi(R−.25)。fresh Adam .001/(.9,.999)/eps1e-8/wd0，每槽独立，不继承pilot状态。nu=xi=1，每完整pass全量S/R后以clip(dual+10*(constraint−budget),0,100)更新，detach；固定12passes，不挑中间函数。

每槽I_k仅含A>0的assigned query行，N_k=|I_k|，每pass独立固定排列、batch512/真实短末批。omega_i=N_k*pi_tg*A/(n_tg*Z_k)；query mean(omega*BCE)对应T。每query step独立循环排列取source fit4096、source P_form256、两个foreign各256，分别估K/S/R；耗尽重排、访问次数记账。不能用全foreign batch无assigned则skip来改变正则质量。K0/无query不建optimizer、不创建图，grad=None。ASSIGN_SHUFFLE用行级u_i=pi/n_tg、Z=sum(u A)、omega=N*u*A/Z，在同target有效行内置乱保持槽计数，不引入LOW_SUPPORT行。

终态完整P_form验S/R。c=1可行直接保留，否则40步二分最大可行保守c∈[0,1]；S(c)为凸BCE差，R(c)=c²R(1)，c=0精确可行。c<1e-6置精确0/COLLAPSED_BY_CONSTRAINT。仅输出乘c，不修改source锚。导出投影前后S/R/K/c/T及constraint mask；不得在P_cal/Q上调c。ANCHOR_OFF/LOCAL只投R，S只审计。

POOLED_TARGET独立于RCF请求及K：每source/task固定名义2槽，每槽所有合格foreign整域12passes；整域n≥256、正负≥20即可，不要求组内达标。J合格域，pi_pool=n_tg/(J*n_t)、A=.5、Z=.5、omega=N/(J*n_t)。source/RMS预算同RCF；R覆盖全部foreign。没有合格域才LOW_SUPPORT_POOLED。它常有更多步数、标签访问和退化后容量，必须如实记账。

LOCAL_ACTIVE每target/task固定4个本地32隐层delta，来自两个独立local pilot配对复制；每组两个独立槽，各在该组query上12passes、T为该组均值（不额外除2/4）。无K/S/nu，只T+xi(R−.25)，R为该target全部P_form delta²，独立aux512，dual/投影同规则。总24头/198168参数，最多活跃4，不随RCF退化而弱化。空组精确0/EMPTY_LOCAL_GROUP；小支持限制解释，不重分组。

## 8. 共同reader、功能导出及S3技术合同

冻结内容后仅P_read拟合：a=alpha_g*B_refit+beta_g+sum(w*delta_projected)。alpha[.25,4]直接优化、beta[−4,4]、普通单元w[−1,1]；objective为BCE+lambda E(实际合成修正)²/2+1e-6||w||²/2，lambda∈{.001,.1}。FULL_SOURCE/FULL_SOURCE_ALL/FULL_TARGET/RANDOM_FEATURE的特征w无逐维界，仅alpha/beta有界。FP64 L-BFGS-B max_iter300/gtol1e-8，数值未收敛才同起点扩600；记录自由梯度/有界KKT，不额外搜索。GROUP_CAL同alpha/beta且w0。

P_cal只在完整六头候选与精确OFF=B_refit之间按Macro LL选，平局OFF优先再大lambda，不逐格拼接。P_form额外监督的FULL_SOURCE_ALL独立记账。P_cal/Q外层源保留只审计，不授权重选OFF或改预算。

每函数×每固定目标组导出GROUP_CAL背景下固定+1/−1动作风险；单函数P_read训练冻结reader的Q风险；完整reader删除/替换配对风险；K1重训/重读风险。跨seed槽对齐只能用P_form profile，不能按Q最优置换。导出OOF→refit预测/残差及单函数效用符号漂移。删除有损不证明多功能必要。

S3组件只能在04分支触发：

| 控制 | 精确定义 |
|---|---|
| ASSIGN_SHUFFLE | 同target有效行内固定置乱A，计数保留；其余同RCF |
| K1_WIDE | 仅原有效K2换256→64→1（16513参数），source32隐藏复制两份加匹配微扰；原两槽有效query并集、原π归一化，12passes同源/RMS预算，reader系数[−2,2]；原K1复用、K0不建；无K2则NOT_APPLICABLE_MULTIFUNCTION |
| ANCHOR_OFF | 保持初始化/A/目标训练，去K/S/nu，仅R及其投影；源S审计 |
| SOURCE_INIT_RANDOM | 只改delta隐藏为固定随机；真实source锚不变 |
| SOURCE_LABEL_SHUFFLE | source任务标签整行成对置乱，pilot/φ/需求/初始化随之形成；S仍用真实source P_form标签，不能叫全无source监督 |
| SPECTRAL_K2 | M=sum(pi*b_tilde*b_tildeᵀ)，共同C广义top-K；输出Vᵀφ，reader按统一规则 |
| RRR_K2 | 每source/task V32×2，多target/group共同P_form拟合；a=alpha*b_OOF+beta+sum(cᵀVᵀφ)，六格等权BCE+.1/2各格实际合成修正均方+1e-6/2 sum||V||²；20passes/query512/Adam.001/wd0，alpha1/beta=c0，V随机N(0,.1)及谱warm两个start；丢临时alpha/beta/c，P_read重读，每start两λ共四候选+OFF整组选择 |
| UNGROUPED_RRR | 仅形成合并g，最终reader仍两组；其余同RRR |

PCA/RANDOM_K2仅在已有RCF优势需组织归因时由相同φ导出，不触发native。完整支持时至多五个新增非线性形成臂（shuffle/K1/anchor/random-init/label-shuffle）；谱/读出小头另记。POOLED更新剂量如有明确混淆，最多一项预登记的同有效更新数POOLED重放，不能扩全面网格。

## 9. 自动分支、锁定与SCALE

实用门槛对每个登记比较：两个开发seed的ΔMacroLL均<0、均值≤−.0005、平均ΔMacroAUC≥−.0005、同一cell跨seed平均ΔLL最大值≤.003。完整晋级要求RCF相对H_DIRECT、H_DIRECT_EFFORT、FULL_SOURCE_ALL、POOLED、LOCAL均通过，再由S3限定可保留主张。

S1所有结果都做一次S2。S2完整实用增量→完整S3归因；RCF与POOLED相当但胜强头→仅必要K1/source-init，保留普通锚定微调解释；仅胜弱头不胜FULL_ALL→SIMPLE_READER_SUFFICIENT、不做全S4；LOCAL匹配/胜→仅预登记source-init一项归因，保留目标拟合解释；仅预定群体两seed交叉优势→K1、已完成POOLED、群体profile，MECHANISM_ONLY、不做四seed全矩阵；大量delta归零→先最小区分无需求/约束/数值，正确实现后可METHOD_REDESIGN_REQUIRED；三臂无支持→NO_INCREMENT_IN_TESTED_SPACE。相当仅未过实用规则，不能声称统计等效。

S4通过数据×6201–6204重新共同prefix，并完整执行H_DIRECT、ETA_DIRECT_MATCHED、H_DIRECT_EFFORT、GROUP_CAL、FULL_SOURCE_ALL、LOCAL_ACTIVE、POOLED_TARGET、K1_WIDE、RCF及S3锁定的一个source干预。冻结后不得负seed停跑或改方法。未通过数据NO_START。外部AutoIFS-project-adapt/STEM-project-adapt独立忠实度验收，不可得NOT_RUN_EXTERNAL、不阻断内部结论，不能称SOTA。

SCALE严格固定c5618e281785a4b7063ff9bf2a61cf38d4faf1be的10-21数据、Equal-style目标、lr.01及实际增强/None语义，选择性恢复到新vendor_snapshot/scale_20261021。先验证coupled Adam(cL,wd,eps)与Adam(L,wd/c,eps/c)同lr/betas、m'=c*m/v'=c²*v/同step的冷/对应暖状态下一步等价；不是lr乘c。随后只新增Dconst01_NO_AUG，2022/2023×两数据最多4条；必要真配对baseline每数据最多2条。无原始历史p仅公开汇总比较，不造配对CI。已知尺度桥足够解释则停止新重加权主张；增强有独立重复增量才2024/2025。仅终点改善不能替代selected。此分支不拖延RCF结论，缺历史数据/源码单列阻塞。

## 10. 实施模块、可审查交付及最少验收

coder只实现本规范，不分析科研结果。实现新protocol/prefix/baselines/demand/formation/readers/mechanisms/state/artifacts/cli/scale_probe，冻结vendor不改。CLI覆盖06列全部合同：audit、build-plan、verify --scope required、prefix --resume、read-probe、form、decide、diagnose、freeze、run-locked、evaluate、finalize、scale-probe；不得用打印计划/空函数冒充已完成。未触发算法可按阶段先实现，但触发前必须实际可用并经审查。

| 必要验收 | 防止的实际结论错误 |
|---|---|
| 手算小batch的Sample/短批、条件组均值、π/Z/omega/LOW_SUPPORT重归一及POOLED fractional A：同时比较数值与梯度 | 聚合质量或训练目标改变 |
| delta精确0、首步末层可学、隐藏首步0/后续可学；真实小batch检查冻结owner/B/native/A无更新 | 形成未执行或越权反传 |
| K0及未参与head grad=None、Adam step/m/v无增长；reader只更新自己 | 零tensor代替None改变优化历史 |
| profile Schur/对称metric/广义方向与显式矩阵结果一致 | 需求/谱计算错位 |
| S/R分别及共同违反时投影最大保守端、c0、只改delta；LOCAL/ANCHOR只投R | 伪造来源可行保证 |
| stable BCE/KL在真实小batch与稳定解析式一致；形成/最终均B+delta | 错读f0或clip改变训练梯度 |
| 连续两步vs一步后内存完整state恢复下一步，比较model/Adam/dual/RNG/query/aux/预测；身份异拒绝复用 | 伪resume或状态混用 |
| reader objective/bound mask/梯度/KKT及整组六头选择、OFF精确等于B | 做弱FULL或逐格test选择 |
| 同raw-X与H/Z/缓存完整预测误差≤1e-6，导出p复算FP64指标 | 缓存、部署与报告不一致 |
| outer score/请求、候选delta、组切点及角色输出完备，label/成本账齐全 | 清理后无法进行已登记归因 |

只做这些影响当前结论的必要核验，不扩大泛化防御测试。合成手算可用于数值验收，严禁作为真实性能数据。审查者必须读本设计、实际代码及CODER_IMPLEMENTATION_REPORT；PASS或全部blocking已修正的PASS WITH FIXES才能正式run。FAIL返回coder。运行者记录原始事实，不分析科研结论。

产物：IMPLEMENTATION_IDENTITY、RUN_MANIFEST、DATA_ROLE_MANIFEST、LABEL_ACCESS_LEDGER、effective config、不可覆盖的row计划hash、实际verify收据、代码审查报告。所有尚未执行条目标NOT_RUN，不预填PASS/AUC/时间。

## 11. 导出、成本、分析及真正完成条件

全部触发方法/角色永久保留row_id/domain/y、合法user、role/schema/hash、native Z/B_OOF/refit/pilot、每函数逐行delta及reader贡献、群体切点/ID、A/score/metric/normalizer/支持、源S/K/foreign R/c、selected/终态p、政策与状态hash、六格/群体指标、标签unique/repeats/Adam步数、参数及GPU/时间成本。旧eta/mu未使用则NOT_APPLICABLE，不能由别的模型替代。H/φ是研究缓存数据，可按允许磁盘空间保留；不写模型state。所有已登记outer机制与raw-X核验在释放内存前完成，不能先释放后把缺项合理化。

完整系统独立端到端成本与共享prefix研究成本分开，避免重复相加；FULL等允许系数融合后公平计推理。重放、双start、额外local pilot、更多P_form标签都实际计账。

最终报告至少06指定FINAL_REPORT、IMPLEMENTATION_REVIEW、DATA_AND_IDENTITY、SOURCE_INCREMENT_REPORT、FUNCTION_FORMATION_REPORT、SCALE_REPORT、COST_REPORT、OPEN_QUESTIONS、tables、ARTIFACT_INDEX/EXPORT_ACCEPTANCE/CLEANUP_RECEIPT；清理收据应说明从未写训练模型、释放哪些已完成依赖的内存、保留哪些数据。禁止动其他目录/进程。

主编排者分析AUC、LL、逐域/逐任务、seed一致性、域规模与baseline难度、模型及阶段；分别识别稳定现象、反例、竞争解释、大小域权衡及最有区分力的后续实验。证据强度标STRONG/MEDIUM/WEAK，不预设课程/动量/小域困难或M3oE必败。回答来源增量、普通读取/初始化/锚/额外目标拟合、K1/K2交叉优势、OOF-refit迁移、约束跨角色保留、成本以及暴露基准边界。缺失数据只限制可得结论。

任务完成是全部已触发阶段、最低验收、导出复算、分析及交付实际完成，或真实不可替代依赖阻塞已完成其余可做任务并明确记录；不是只创建计划或启动后台进程。上传遵从start.md：代码/配置/完整报告与小收据更新到原取得分支research-plan/2026-10-26-v2-mechanisms，不另开日期分支，不上传大缓存或模型，不改旧实验。当前无.git可经连接器基于远端最新head构建带父提交的正常更新，不能覆盖远端新增工作。

设计阶段不可替代阻塞：目前未发现；数据和源码已有父编排器可用性证据，仍待正式audit。磁盘模型禁令已通过内存执行设计解决，不构成要求用户放宽限制的理由。

## 12. 实施分工与冻结Python接口

下述接口是本次新实现合同；不用已有名字相近的旧pipeline替代。数值数组默认numpy，训练时局部转换torch；所有行顺序显式以row_id绑定。函数可以另有内部helper，不得改变这里的含义。跨模块先按此合同实现，不通过修改对方文件修整接口。

`protocol.py`提供以下dataclass（元数据可以附加字段但不能删改必需字段）：

```python
@dataclass
class RoleView:
    row_id: np.ndarray       # [N] int64，角色内固定顺序
    domain: np.ndarray       # [N] int64
    y: np.ndarray            # [N,2]，取得前通过授权/账本；禁止提前训练读test
    h: np.ndarray            # [N,256] float32
    z: np.ndarray            # [N,2] float32
    group: np.ndarray        # [N,2] int64，冻结native切点

@dataclass
class PrefixBundle:
    dataset: str
    seed: int
    views: dict[str, RoleView]
    base_logits: dict[str, np.ndarray]  # B_refit，[N_role,2]
    base_oof: np.ndarray               # P_form顺序，[N_P_form,2]
    pilots: Any                       # 内存对象，见下面PilotAccess
    native: Any                       # 内存native selected完整状态
    base_models: Any                  # 内存B与其Adam，直接头续训使用
    normalizers: dict
    identity: dict
    ledger: Any

@dataclass
class FeatureBank:
    values: dict[tuple[str,int,int], np.ndarray]
    # key=(role,target,task)，仅该role该target行，顺序等于view中domain==target
    # value [N_role_target,M_target_task]；空内容为[N,0]，不伪造一个零参数槽
    columns: dict[tuple[int,int], list[dict]]  # 每列function_id/source/task/slot等
    bounds: dict[tuple[int,int], list[tuple[float|None,float|None]]]
    identity: dict

@dataclass
class PredictionBundle:
    method: str
    probabilities: dict[str, np.ndarray]   # 角色完整顺序[N,2]，selected最终p
    terminal_probabilities: dict[str, np.ndarray]
    selection: dict
    diagnostics: dict
    identity: dict

@dataclass
class FormationResult:
    method: str
    features: FeatureBank
    functions: dict[str, Any]  # delta函数与投影c，仅内存，含source/task/slot键
    requests: Any             # DemandResult或明确NOT_APPLICABLE元数据
    records: list[dict]       # 每slot/pass目标、约束、c、支持与label/step账
    identity: dict

@dataclass
class DemandResult:
    units: dict[tuple[int,int], dict]
    # (source,task): effective_k、groups列表(target/group/rows/pi/score/r等)、
    # assignment[有效组数,K]、metric[32,32]、phi normalizer及support/status
    identity: dict
```

PilotAccess为新桥接对象：`logits(role, source, task)`返回角色完整顺序的`Z_q+f_source`一维数组；`residual(role,source,task)`返回f_source一维数组；`hidden(role,source,task, *, standardized=True, initial=False)`返回[N_role,32]；`clone_hidden(source,task)`返回**内存**隐藏weight/bias副本；`local_hidden(role,target,task,replica)`供两个独立local pilot；`local_clone_hidden(target,task,replica)`返回相应内存参数。查询必须经prefix实际角色合同，不能通过测试标签。所有输出不依赖调用顺序改变模型或RNG。

### 实施包A：protocol/state/prefix/baselines/readers

负责上述dataclass、共同语义及内存对象；持有这些文件所有权，其他coder只导入。对外提供：

```python
load_config(path: str|Path) -> dict
derive_rng(seed:int,dataset:str,stage:str,source=0,task=0,unit=0,pass_id=0) -> np.random.Generator
audit_dataset(dataset:str, config:dict, output_root:Path) -> dict
build_prefix(dataset:str, seed:int, config:dict, output_root:Path,
             *, device:str, resume:bool=True) -> PrefixBundle
fit_base(bundle:PrefixBundle, config:dict) -> None
# 就地填base_logits/base_oof/base_models；使用前必须完成，重复同身份幂等
fit_direct_baselines(bundle:PrefixBundle, config:dict) -> dict[str,PredictionBundle]
fixed_feature_bank(bundle:PrefixBundle, method:str, config:dict) -> FeatureBank
fit_reader(bundle:PrefixBundle, features:FeatureBank, method:str, config:dict,
           *, train_roles:tuple[str,...]=('P_read',)) -> PredictionBundle
fit_probability_reader(bundle:PrefixBundle, config:dict) -> PredictionBundle
capture_memory_state(objects:dict, rng_state:dict, positions:dict) -> dict
restore_memory_state(snapshot:dict, objects:dict) -> dict
```

state内存快照必须deep-copy tensor/optimizer，不能引用仍会被下一步改变的state_dict；不提供磁盘save_model helper。`fixed_feature_bank`涵盖GROUP_CAL/PILOT_LOGIT/FULL_SOURCE/FULL_SOURCE_ALL/FULL_TARGET/RANDOM_FEATURE；FULL_ALL的额外训练角色在调用fit_reader时明确指定。direct模型和probability reader的P_cal整组候选也返回PredictionBundle。缓存H/Z可用磁盘研究数据；跨进程缺失native/pilot模型时必须拒绝伪恢复并交campaign重放，不从最终p反推参数。

### 实施包B：demand/formation/mechanisms

```python
compute_demand(bundle:PrefixBundle, config:dict) -> DemandResult
form(bundle:PrefixBundle, demand:DemandResult, method:str, config:dict,
     *, paired:FormationResult|None=None) -> FormationResult
diagnose_functions(bundle:PrefixBundle, formed:FormationResult,
                   predictions:PredictionBundle, config:dict) -> dict
spectral_features(bundle:PrefixBundle,demand:DemandResult,config:dict) -> FeatureBank
fit_rrr(bundle:PrefixBundle,demand:DemandResult,config:dict,
        *, ungrouped:bool=False) -> list[FeatureBank]
```

`form`支持RCF/POOLED_TARGET/LOCAL_ACTIVE与触发后的五项非线性干预；paired用于K1保持原K1/0、复用匹配初始化/请求及ASSIGN_SHUFFLE，不授权读取Q标签选参数。SOURCE_LABEL_SHUFFLE应在内存构建专用pilot/需求分支，不能原地改bundle真实pilot。形成完成后为登记的所有角色生成FeatureBank并保持函数对象供raw-X/投影核验；预测本身不等于允许提前性能选择test。fit_rrr返回两个start的冻结内容，最终四λ候选及OFF应由共同reader的整组选择入口执行，不按start各自选后再用Q挑。

### 实施包C：CLI/campaign/artifacts/scale

此包只调A/B API及登记流程，不另写算法或私改A/B文件。可增`campaign.py`以保持长驻内存。负责接收每个prefix的PrefixBundle、形成和预测对象、冻结stage清单、取所有seed后调用确定性的门槛决策。阶段决策要记录每个比较逐seed六格与门槛失败原因；科研解释留给主编排器。

```python
build_run_manifest(config:dict, audit:dict) -> dict
decide_stage(stage:str, predictions:dict, config:dict) -> dict
export_bundle(bundle:PrefixBundle, results:dict, output_root:Path, config:dict) -> dict
verify_exports(index:dict, config:dict) -> dict
run_scale_numeric(config:dict, output_root:Path) -> dict
run_scale_training(config:dict, output_root:Path, *, device:str) -> dict
```

artifact路径由config新root派生，无硬编码旧路径。导出先运行manifest/label授权；test/legacy仅全局freeze后计算指标，虽可预先无标签前向候选，不提前对其输出科研分析。CLI的各命令使用同campaign内存，或持久worker API；独立进程命令若模型已不在内存，必须清楚报告需要重放，不以已有p文件伪装完整模型恢复。collector必须聚合所有prefix/触发方法，不因为单prefix成功而写FINAL_COMPLETE。

SCALE补充：父编排器已恢复22个固定c5618源码文件并有SCALE_SOURCE_MANIFEST。其历史stage2_run的选点是**validation Macro AUC、patience10、max50epochs**，不是主线native的LL12–24；必须保持真实历史配置。若恢复代码与历史run身份的其他字段矛盾，记录后作最少身份核对，不把主线配置套用。

以上三个实施包允许并行，但均在本设计完成后开始；共享合同由包A维护，新增字段通过消息同步，不覆盖别人实现。每包写各自CODER_IMPLEMENTATION_REPORT并列出实际basic checks，禁止科研分析。联合review审查实际集成实现之后才RUN。
