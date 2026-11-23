# DESIGN SPECIFICATION — 条件历史与共享更新

日期：2026-11-23。阶段：DESIGN。

执行依据为远端 `main@4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983:research_plans/26-11-22_conditional_history_shared_update/` 原15文件。入口为00_EXECUTION_RUNBOOK.md，科学定义见02/03/04，任务矩阵见05，工程合约见06，统计/阶段门见07/09，资源与来源见08/12。本规格是独立实现交接，不修改原计划或manifest。

## 1. Research question

方向A：在完全相同的自然条件PAIR辅助监督下，P01历史更新包是否超出普通Adam；24条件分离是否优于单份辅助历史或较长Adam历史？

方向B：自然联合训练中，真实域任务贡献在共享坐标上的实际Adam更新是否造成跨格排序损害；训练信息能否识别比匹配伪分组、静态源和统一缩放更有效的源，并改善固定EMA仍未消除的排序代价？

分别识别完整训练增量、同状态一步效应、一次脉冲传播和持续策略性能。A结果不决定B核心是否执行。

## 2. Hypothesis being tested

- A-H1：A20−A10存在五seed独立增量，且相对A21/A22保留复杂条件历史的实际价值。
- A竞争解释：收益来自PAIR监督、较长单一历史或两通道时间尺度；历史只修复了被PAIR降低的底座。
- B-H1：真实源存在self收益和跨格损害，训练选源的未来效用超过等预算伪分组和固定源，且F仍有可行动余量。
- B竞争解释：普通更新分割、幅度/平滑、函数组合解释现象，或损害虽可描述但不能从训练信息控制。

不预设动量或课程有效，不把小域等同困难域。

## 3. Variables changed

| 阶段 | 改变量 | 目的 |
|---|---|---|
| A_CORE | 无PAIR/静态PAIR/动态q；普通Adam/原ECCM | 分开监督、历史包与动态抽样 |
| A_HISTORY | A21单辅助Q；A22单混合m=.99 | 判断复杂历史必要性 |
| A_DOSE | λ=.025/.05，历史与同λ普通Adam成对 | 分离剂量与历史贡献 |
| A_SCOPE | STAR_SKIP-only/EXPERT-only历史 | 真实定位后检验作用位置 |
| B_CORE | 真/两套伪loss分组；EMBED/S_DENSE/N；JOINT/ZERO/ONLY/LOO | 识别源贡献、任务身份与幅度 |
| B_PULSE | 一次真/伪LOO后正常联合训练 | 检验未来传播 |
| B_SEARCH | ADAPT/STATIC/UNIFORM，η=.25/.5 | 检验持续策略与简单替代 |

AB和Gaussian为原矩阵的条件分支，仅在对应门触发后注册，不增加额外搜索维度。

## 4. Variables held constant

两个原始自然CSV、原行序80/10/10、train-only编码/缩放；KR先切分再保留tab0/1/4。M3oE fixed1，embedding16、4个共享expert、FCN[512,256,256,64]，保留gate输入embedding.detach。种子仅2022–2026；每配置两数据完整50epoch；batch40960，保留自然尾批；FP32，AMP/TF32关闭。

Adam lr=.01、β2=.999、eps=1e−8、coupled WD ML1e−4/KR1e−5，foreach/fused/amsgrad=False；普通β1=.9，A22仅指定范围β1=.99。EMA初始等于online，每live Adam后一次，不反馈；ρ ML=.9828205985452511/KR=.995。loader独立Generator(seed+11000)，每epochrandperm，ML/KR每epoch20/49步、尾批21927/40052行。

主读出fixed_ema，每seed选择E1–50完整Validation六格Macro AUC最早严格最大值；同点LL绑定，online独立登记。E10/E50和机制锚点仅作次级分析。

## 5. Mathematical/algorithmic definition

### 5.1 自然目标与条件PAIR

对实际batch行数B：

`L0=(1/(2B)) Σ_i,t BCE(p_it,y_it)`；
`L_dt=(1/(2B)) Σ_i:d_i=d BCE(p_it,y_it)`；
`Σ_dt L_dt=L0`。

禁止改为各格mean再等权。六格按domain-major登记，tower/gate索引仍为`3t+d`。

联合标签`j=2y0+y1`，边界`i=(d,t,a,b)`同域且目标任务在a正、b负，最多24个；ML无01，最多12个有效条件。当前batch同时存在两端才活动。

`w_i=n_d/(2B) * n_da/n_dt+ * n_db/n_dt-`，活动集A、`Z=Σ_i w_i≤1`。静态`q_i=.25/|A|+.75*w_i/Z`；动态q完整复用源boundary_sampler中旧需求D的公式，不读取当前误差后再采样。

每步抽一个活动边界，从原batch两端有放回抽2048对：
`R_i=mean softplus(-(z+−z-))`，`r_i=∇R_i`；
`L=L0+λ*(w_i/q_i)*R_i`，核心λ=.05。

z来自一次真实源forward的tower hooks；PAIR梯度沿原图覆盖全部可达参数，无RMS归一化。PAIR key为dataset/seed/epoch/global_step/stream，不含arm/λ；静态臂逐步pair索引匹配。动态P10的D在当前optimizer与EMA完成后更新为`.9D+.1*mean sigmoid(-margin)`。无活动边界时真实训练继续；λ=0完全旁路PAIR和额外RNG。

### 5.2 A历史定义

S_DENSE={shared_weight,shared_bias,skip_conn.*,star_mlp.*,expert.*}；不含EMBED。expert内部含domain_specific子串仍归共享EXPERT。fixed1融合常数不创建可训练FUSION组。

对原生非None参数p，τ_p为私有step；mixed含一次coupled WD：
`a_p=g0_p+λ*(w_s/q_s)*r_s,p+WD*θ_p`。

A10/P00采用普通Adam；A30/P10仅改变q。

A20/P01：被选条件`H_s←H_s+(.01/q_s)*(r_s−H_s)`，每个当前活动条件的a_i加一，其余历史保持。`mR←.9mR+.1*(g0+WDθ)`；`v←.999v+.001*a²`；更新分子为：

`N=mR/(1−.9^τ_p)+λ*Σ_active w_i*H_i/(1−.99^a_i)`。

分母为`√(v/(1−.999^τ_p))+eps`。活动次数和抽中次数分别登记，不能混用。无PAIR时不推进辅助历史/活动时钟且不注入辅助分子，真实mR/v/τ继续。S_DENSE以外普通Adam处理完整mixed目标。

A21/PCOLL_STATIC_099：相同静态q和mixed v，只保留一份`Q←.99Q+.01*(w_s/q_s)*r_s`，pair存在时aΣ加一；分子`mR/(1−.9^τ)+λQ/(1−.99^aΣ)`。源PCOLL默认dynamic=True，新适配器必须显式False。A20存未加权条件r后按当前w重组，A21记住过去w/q；差值包含重组与时钟差别。

A22/ADAM_S99：S_DENSE对完整mixed梯度使用原生β1=.99，其他β1=.9；真实梯度和WD历史也改变。λ=0由工厂返回全参数β1=.9的A00 optimizer。范围限制只改变历史作用范围，其余参数仍普通Adam处理同一个完整PAIR目标；每参数仅一次step、全模型EMA仅一次。

A真实轨迹归因按03§9：从初始化维护六格辅助普通Adam shadow。ECCM/COLL实际mR加六格shadow重构同轨迹普通Adam一阶状态，共用当前实际v；S99额外分别维护β=.9/.99真实+WD残差和六格PAIR残差。不能把分叉完整P00轨迹状态移植为P01 shadow。对STAR_SKIP/EXPERT替换全部历史或单源残差只作只读更新探针；同时做原设计整组RMS尺度控制，不反馈训练。

### 5.3 B梯度、坐标与精确Adam干预

一次源forward图上求真实6源和两套伪6源，共18额外autograd/锚点。live g必须由原L0.backward产生；Σgq≈g仅作数值验收。

坐标G={EMBED,S_DENSE,N}互斥覆盖。域embedding名字由features.columns.index('domain_indicator')确定，其0/1/2三行分别归D，排除在共享EMBED之外。N=D/T/DT，不能假设N必然无跨格作用。

从同一真实prestate深拷贝模型、原生Adam，load_state_dict后重算分支：G外保持g；G内JOINT=g、ZERO_DATA=0、ONLY(q)=gq、LOO(q)=g−gq。保留每参数m/v/step、param groups、buffer、F、RNG、loader和batch身份。WD由原生Adam统一加一次；native grad=None保持None并跳过WD/step，native非None变零仍step。不能以update相减近似Adam分支。

每锚点`1+3*(1+6+6)+2*6=52`虚拟Adam；40锚点为两数据×5seed×E5/E20/E35/E50末batch更新前，总2080虚拟步、720额外sweeps。JOINT虚拟后态与随后真实一步的模型/optimizer hash同设备必须精确一致。诊断不改变live模型、optimizer、EMA、RNG、loader或PAIR键。

伪分组将2B个(row,task,label)loss entries按0/1分层分为6bin，匹配真实cell正负entry计数，每entry仅一次；不改标签、行、目标、loader。两套独立确定性keyed partition各自六选一+NO_ACTION，不在12bin中共同选最优。

### 5.4 排序效用、历史背景与读出

`U_q→c=Q_c(JOINT)−Q_c(LOOq)`；去除收益`D=−U`。Q用源概率AUC及tie=.5，固定pair准确率另报。self收益和跨格代价通过实际前向评价识别，余弦/范数只作解释量。ZERO/ONLY用于条件背景，不作最终算法强对照。

G内applied update dJ/dL同时缩到`min(||dJ||,||dL||)`，G外保留JOINT位移；零半径记NOT_INFORMATIVE，不补随机方向。

`F_t=ρF_(t−1)+(1−ρ)θ_t`。同prestate一步`δF=(1−ρ)δθ`，同时比较online全位移、online上(1−ρ)小位移、F上同小位移，区分机械缩放与背景交互。真实分支传播核验`δF_j=ρδF_(j−1)+(1−ρ)δθ_j`。固定后续位移和只换历史快照两种公式按02/04分开记录。

窗口概率/logit平均从f(F_pre)/z(F_pre)初始化，按同ρ逐步递推，与同起点参数EMA比较；命名WINDOW_PROB_EMA/WINDOW_LOGIT_EMA，不称全轨迹prediction EMA。只保存score banks，不保存逐步θ磁盘快照。

### 5.5 面板、脉冲和持续候选

P_A训练选源，P_B不交训练确认，V_A/V_B固定自然Validation分侧；面板IDs及pairs在准备时冻结。训练ranking每格最多256正/256负、4096pairs，自然LL每域512行；V_mech每域最多4096行，现场核验支持数。面板只观察，不改变训练loader。

P_A以EMBED去除Macro效用六选一+NO_ACTION，最佳严格>0才选源，平局按domain-major。固定源由其他四seed E5/E20的P_A冻结，留出seed E35/E50评价，不读Validation选q。

数据集级B_PULSE门触发后，该数据五seed×E20/E35×真/partition0伪全部完成，NO_ACTION计零。首步复用完整LOO post，包括Adam和F；后K−1步在分支状态重新正常联合训练，K=ML20/KR49。最大1380逻辑步，40首步已计B1，新增1340次Adam。JOINT复用父轨迹。20个父窗口共690组逐步面板读出为核心必做。

B_SEARCH正式臂ID为B_ADAPT、B_STATIC、B_UNIFORM。只在09规定的真实EMBED损害、训练选真源未来Validation优于JOINT/伪源、留出一步优于固定源、F正余量与六格≥−.001等门均通过时启动；低于.0001标LOW_ACTION_VALUE。ADAPT每epoch首batch从共同前态评JOINT+6个实际η候选共7clone，选P_A后仅一次live更新/EMA；q维持一epoch，gq每batch重算。STATIC使用LOSO固定q。UNIFORM重放同seed/η ADAPT标量trace，EMBED data梯度乘`1−η*n_d(q)/(2B)`，其他data与WD不改；它匹配名义目标质量，不匹配Adam位移。

### 5.6 条件组合与Gaussian原定义

同数据A/B均独立有效、在获胜A的E20/E35真实状态上通过训练选B动作的五seedValidation正面边际和六格保护，才注册唯一AB。数据梯度：

`gAB=g0+gG+λP*rP−η*P_EMBED*gq`。

B只减自然BCE源，不减PAIR或Gaussian残差。A辅助历史只接收PAIR，Gaussian残差进入普通真实通道；同一prestate、一次live提交。完整路由按06§11。

G_FROZEN只在最终候选需要决定是否采用既有Gaussian背景时触发，固定D0、λ_aug=.5、κ ML.05/KR.1，沿用原每稀疏字段embedding RMS噪声且不扰动域字段。目标是原D0损失部分替换：

`LG=[Σ C + .5*(Σ_D0 A−Σ_D0 C)]/(2B)`。

使用同一原始完整batch forward的pred/real/logits构造自然目标和PAIR；hooks在其后解除，不被D0 view覆盖，不为PAIR额外forward。Gaussian键与PAIR键独立；B减弱的gq始终来自原始自然BCE。该背景不额外建立Gaussian条件历史，不搜索覆盖/噪声/ρ。

## 6. Exact affected modules/files

新增包为`experiments/conditional_history_shared_update_20261122/`，按06§2接口实现：

- S0与核心外层：protocol.py、source_adapter.py、pair_adapter.py、reuse.py、state.py、runner.py、scheduler.py、analysis.py、cli.py。
- B核心：groups.py、probes.py、virtual_adam.py、utility.py、rollout.py。
- 条件后续：history_controls.py、controllers.py、composition.py，保持已冻结公式，只有门通过才生成训练任务。
- 最小必要检查：tests/test_scientific_contracts.py；日志与报告放本轮artifact/report根。

复用v3 protocol/boundary_sampler/objectives/optimizers及完整vendor closure；CWA core.py nested virtual_steps作原生克隆模式参考；按06真实坐标构建组适配或闭包恢复历史registry依赖。旧v3源码、旧结果、旧run.py生命周期保持原状，不复用其两seed/调度门。

## 7. Implementation requirements

### 优先顺序与科学队列

1. S0先做来源、数据、复用、软件/GPU、RAM状态审计，输出SOURCE_MANIFEST、REUSE_MANIFEST/reuse_inventory和具体BLOCKERS；核验身份后才SOURCE_READY。旁边git仓库可用于git blob/commit核对，当前工作目录无.git时使用实际文件SHA256，记录各自身份。
2. A_CORE仅新增A10/A30×两数据×{2023,2025,2026}=12条训练；A00/A20五seed和A10/A30旧2022/2024共28条数值逐身份复用。缺项先恢复来源证据，不自动全量重跑。
3. B_CORE默认10条同源clean重建，训练中流式处理40锚点；其统计n仍为5。若现场有同身份完整可恢复RAM状态才扣减重建。selected-only不能恢复历史Adam。
4. 按有限DAG逐门展开后续。候选未触发保留NOT_TRIGGERED，技术失败记TECHNICAL_BLOCKED/PAUSED，不记科学阴性。全部规定核心和触发任务完成后，执行原S5最终流程、报告交付并结束本目录任务。

### 工程不变量

StatePacket包含online参数/buffers/mode、完整optimizer与私有step、F、A历史/时钟/需求/shadow、Python/NumPy/CPU/CUDA RNG、loader permutation/cursor、selected模型/epoch/Validation及source/data/config/group身份。所有可恢复训练张量RAM_ONLY；代码、数据身份、日志、标量、预测评价记录和报告可落盘。RAM提交后实际get/hash验收再登记完成；状态丢失只同源有目的重放，成本单列，不增加n或挑选最好attempt。

每卡独立按当前显存<85%、当前功耗<85%、host RAM<90%准入；启动/分配后重新测量，允许当前启动pending，不设固定worker数、不要求两卡同时满足。保留已有OOM后的CUDA-usable物理容量检查，使用完整进程实测驻留；不把预计功耗或projected85%显存加为新门。不能无声减batch、改精度/人口。

coder仅实现和做syntax/basic正确性检查，交CODER_IMPLEMENTATION_REPORT；不同的新reviewer独立读取设计、实际源码、coder报告。PASS或PASS WITH FIXES且阻塞全部修正后，由当前会话runner监督科学训练，runner只记录原始数值，主agent分析。

### 选点、冻结与Test

方法、λ/η/scope和selected只看Validation。原FINAL_FREEZE锁定候选方法/所有参数/主比较/匹配控制/selected epoch与模型hash/六格保护参照/统计脚本版本；核验当前selected RAM后一次评价冻结最终主表。Test不能反向触发调参或新臂。历史Test标HISTORICAL_EXPOSED；未晋级臂不批量开放新Test，旧封存任务不因本方案自动开放。

### S0现场证据与剩余准入

| 项目 | 2026-11-23设计阶段观察 | 实施要求 |
|---|---|---|
| 计划来源 | 父orchestrator已从`../phase0-3_git`读取origin/main=4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983 | 原15文件按该提交恢复，单独记录本规格 |
| 科学来源 | 计划历史内核M=875215c1e509ab55c1b6632d4f5d13e8b94782da；P等来源见12 | 新适配器与旧数值内核分别hash，不复用旧review许可 |
| v3清单 | 原IMPLEMENTATION_SOURCE_MANIFEST SHA256=7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0；所列52/52文件逐项匹配，无缺失/差异 | coder登记完整closure和运行版本 |
| CSV | ML=f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4；KR=a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f；现场重算均匹配 | 从冻结loader加载后继续比对array/row/encoder/scaler/split身份 |
| 历史输入 | supplement/endpoints SHA256=2807c327e73c73d7660f4d37e781be9667c922e75b9aeed95e2c6ef4a76a9d80；selection_frozen=7629abc8baf04f856cccfe1fc42ed8ea0ddd3b93d36902134276ee388ebe081d；五seed endpoints=c01d1c01b09e74d7b3c74589482c1c7b8182a3351a125bda31ecc4a00a532a84 | 核验复用映射/端点身份，重复表行不增加n |
| prepared目录 | 缺失；v3 vendor/ema02/data.py直接从已核CSV调用冻结audited loader并校验数组/编码身份 | 本路径可直接重建，不能因旧prepared缺失误判CSV不可用 |
| 软件/GPU | 项目Python存在，两张3090可见 | 正式RUN前记录torch/依赖、当前遥测、CUDA usable与完整进程驻留 |
| RAM | 原guardian PID1025015/socket已消失，父orchestrator现场Client ENOENT；当前用户已授权原代码重建仅RAM存储guardian | 新存储实际get/hash及generation核验前RUN为技术PAUSED；旧存活收据不作当前验收 |
| 新代码 | 新包尚待实现 | 可先完成S0和不依赖guardian的实现/审查准备 |

以上是只读来源/环境观察，不含新研究训练结果。

## 8. Ablations required

核心必做A10−A00、A20−A10、A30−A10及A20/A30对A00总增量；B真源/两伪源、ZERO/ONLY/LOO、原始/同尺度、online/F/机械缩放背景、父窗口参数/概率/logit读出。

条件矩阵严格按05/09：A21/A22共20条；λ=.025历史与普通Adam共20条；有当前真实定位后A50/A51共20条；B三策略×两η×两数据×五seed共60条；正面桥接后唯一AB10条；原Gaussian门通过后G10条。A专用状态重建最多10条，B默认10条。

默认22条完整物理=12新科学+10状态重建，共1100epoch、37950主更新。原全部条件触发上界172物理/152新科学、8600epoch、296700主更新，不默认跑满。B2080虚拟、最多1340新增pulse、选择器和评价前向独立计账。B_SEARCH选择器7000clone，AB/G若采用ADAPT各最多3500，合计最多14000。无新seed池、β网格、P11、ρ/H/增强覆盖搜索。

## 9. Expected logging requirements

按10模板输出逐seed原始六格/宏AUC与LL、Validation selected epoch/hash、readout/source/data/config/group/attempt身份和配对差。Macro/Traffic/Pooled分开；缺预测不猜pooled。统计单位seed，报告绝对均值/样本SD、五个配对差及95%t4区间`mean±2.776445*SD/sqrt(5)`；不把epoch/锚点/读出当独立n。

B记录40anchor身份/实际B、scope/source/receiver/action效用、真/伪容量与partition hash、所有NO_ACTION、P_A选源、纠正/破坏/ties闭合、gradient/update norm、None mask与step hash、虚拟/实际JOINT hash、EMA背景、pulse首步复用和窗口读出。ADAPT保存标量ACTION_TRACE，UNIFORM校验对应seed/η trace hash。

成本分NEW_SCIENTIFIC_TRAIN、REUSE_NUMERIC、STATE_RECONSTRUCTION、VIRTUAL_ADAM、COUNTERFACTUAL_SUFFIX、诊断和失败恢复；记录有效/物理epoch、main/virtual/pulse/sweeps/forward、worker/wall秒、峰值GPU/RAM、当前准入遥测、RAM事务/get/hash、失败与重建来源。

分析审视AUC/LL、per-domain/task、seed一致性、域规模、由正常基线表现支持的困难度、模型与训练阶段；标稳定/矛盾现象及STRONG/MEDIUM/WEAK，判断规模解释和小域改善/大域下降trade-off，列竞争解释与最有判别力的后续问题。后续问题记录入报告，不自动扩张本目录矩阵。

## 10. Acceptance criteria for reviewer

- 源公式/自然数据身份正确，旧v3未改；静态PAIR匹配、q校正/活动时钟与S_DENSE范围正确；A21静态、A22β与λ0正确；辅助梯度仍沿完整源图。
- 原生Adam clone保留私有step、None/零/WD一次，JOINT与live同设备hash精确一致；detach/RNG/loader/F无污染。真/伪loss梯度守恒，坐标组互斥全覆盖。
- 五seed、50epoch、最早strict-max及同点LL落实；缺seed/状态丢失归技术状态；读Test前必须FINAL_FREEZE和当前selected RAM身份验收，Test不进HPO。
- 只执行原计划已触发DAG，B核心独立、NO_ACTION完整，未触发不记阴性；AB/G严格遵06§11原路由与触发。
- RAM_ONLY、当前资源准入和物理容量、状态提交/恢复正确；原guardian失去不伪装现场可恢复状态。coder/reviewer独立，旧文档PASS不替代当前源码审查。
- 最小检查只覆盖首/尾batch源回归、λ0/新历史、None/零/空域、分组/梯度守恒、virtual JOINT、RNG中立、EMA/短段、RAM roundtrip、选点/五seed/Test门；不为未修改旧模块扩张测试矩阵。

## 11. Conditions that would falsify the hypothesis

A20−A10五seedValidation均值≤0，结束本版历史继续信号；A10有益而历史不增益支持PAIR解释。A21/A22复制或超过A20（含相同保护状态.0001简化带）撤下24条件表必要性。历史不胜正常A00+F或同λ普通Adam，则无本轮新增算法资格；仅LL改善独立报告。

真/伪源在等尺度及未来留出效用相近，或UNIFORM复制/超过身份策略，反驳当前任务身份必要性。训练规则不能预测收益、F已消除余量、脉冲效应未传播，分别限制可控制性、可行动残余与传播主张。完整B不胜正常clean+F，结束本版持续控制器，不以胜ZERO/ONLY/伪组救援。AB不胜较强单法选择单法。

宏收益伴随六格代价标MACRO_GAIN_WITH_CELL_TRADEOFF。最终相对发表最强模型+.003为强总结果、+.002为候选总结果且六格平均≥−.001；独立组件增量另报，LL单列，不能用包含EMA的总收益替代组件归因。局部机制可成立而完整策略失败，两者同时保留。完成规定矩阵、最终冻结评价和报告后按原执行入口结案。
