# DESIGN SPECIFICATION — BASP-A，2026-10-28执行版

设计阶段完成。本规范授权实现，不授权跳过独立代码审查或预先认定研究结论。代码必须位于新目录 `experiments/mdmt_basp_v1/`；输出位于新目录 `research_reports/2026-10-28/` 及独立数据/日志目录。旧代码、历史规划和历史结果保留。

## 1. 权威、修订及范围

已完整阅读规划00、02、03、04、05、06、07、08、09及AGENT.md，检查26-10-27参考实现的data/native/source/state接口。实现数学与验收以05为准，阶段决策以06为准，原始数值以08为准；以下显式修订来自当前用户或developer，优先于历史规划。

|冲突|执行决定|影响|
|---|---|---|
|start.md写`research_plans/26-10-28`，实际目录为`research_plans/2026-10-28`|读取实际目录，历史文件不改名|仅入口修正|
|规划开发7101/7102、正式8101—8104；start.md强制只允许2022—2026|开发2022/2023，冻结复评2024/2025/2026，互不重用|P1—P4仍两seed，P5实际n=3；P5预算24而非32，总前缀最多10而非12|
|规划描述DESIGN_ONLY、不上传；用户要求完成实验并上传26-10-28|本轮为实际实施；最终推送指定新分支|历史DESIGN_ONLY状态不改写|
|历史要求保存完整checkpoint；start.md禁止本地保存训练模型|完整训练状态、selected/final、teacher、候选全部保存在RAM|磁盘只保存数据、预测、统计、日志、代码和报告；进程终止需重建共同前缀|
|合同要求新hash身份；当前developer设置hash=deny|不新加、不运行hash计算；使用既有远端commit/blob标识、显式版本、完整row数组与直接比较|运行身份为可检查的结构化元数据；state/源码一致性按逐tensor/逐字节比较|
|legacy prepared路径不存在，但已找到本轮之前的prepared数据|优先复用现有prepared并逐值验收；缺失或不符才按原算法恢复|不重复生成数十GB数据|

原始CSV路径：MovieLens `/home_new/xzf/datasets/ml-1m/ml-1m.csv`；KuaiRand `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。其历史SHA及迁移验证已记录在AGENTS；本轮不重新计算摘要。历史数据划分算法常量20261024及有序随机流标识保持原协议；它们不是新增实验重复seed，不得为满足训练seed范围而改变历史roles。

可用prepared入口：ML `artifacts/mdmt_relation_basis_v2/data/movielens`，KR `artifacts/mdmt_relation_basis_v2/revision1/data/kuairand`，均包含VALIDATED、DATA_MANIFEST、PREPROCESSOR、x/y及rows。不要直接调用会重新算hash的旧RoleDataset；新roles以mmap读取并按第3节直接校验。数据约67GB，当前可用RAM约56GB、磁盘约204GB；按chunk验收和按batch上GPU，禁止全量复制。源码直接比对收据见`REFERENCE_SOURCE_AUDIT.json`，34份非空源码/JSON与固定远端commit内容相同。

代码参考是`26-10-27@03c338092008a71b2d360a61b06d985eac551ec7`，main为`8eb461f5e0abc9c06385a4e8239e9621cae87ef8`。主agent负责核对远端源码字节及Git提交；允许复制所需固定底座并标明来源，禁止运行依赖旧hash函数的prepare/state路径。无新增第三方依赖。

本轮先完整实现与验收BASP-A六开关及结构事务，再跑规定固定K矩阵。B/C和P3替代只在预定义证据触发后实施；未触发写NOT_TRIGGERED，不伪造通过记录。不得增加无关防御测试或新控制器。

## 2. 文件与责任

主agent只承担编排、远端审计、研究决策和分析。coder与reviewer必须是不同子agent；除机械runner外所有子agent与主agent同模型。coder不得解释研究结果，runner只记录原始结果。

新包按05接口职责实现：config、identity、roles、prefix、model、benefit、relation、losses、permissions、conflict、transaction、metrics、trainer、campaign、export、`__main__`；职责可在现有模块内合并，避免纯转发包装。测试在包内tests。执行配置从原08/09加载后应用本规范唯一override，并导出EXECUTION_CONFIG与完整实验矩阵；未知字段、重复键、非法组合必须拒绝。审查收据必须覆盖实际最终实现，不能只审一份规划。

两个coder可并行，但不得跨写对方文件：`coder_model`拥有model/benefit/relation/losses/permissions/conflict/transaction和tests_model；`coder_pipeline`拥有config/identity/roles/prefix/metrics/trainer/campaign/export/CLI和tests_pipeline。双方先在消息中确认以下接口：

- 缓存Packet用普通dict，`h:[B,256]`、`z:[B,T]`、`phi:[B,D,64]`、`domain:[B]`；`y/mask/row_id`仅外部训练/评价持有，不传forward。
- `BASPModel(config, teacher_w, teacher_b, seed)`，`forward(h,z,phi,domain)`返回05的ForwardOutput；TARGET_MATCH可为同模块独立类。D/T来自执行config。
- model提供稳定名字参数类别与树状态；benefit从冻结teacher和合法probe生成detach标签/teacher q/G；losses返回具名标量字典；permissions接收loss字典、阶段及模型，按白名单和cap设置grad；trainer负责一次Adam step与投影后的时钟。
- `roles.Data`提供`roles`有序row数组、`domain`、`schema`、`inputs(role,ids)`、`labels(role,purpose,ids)`；prefix提供resident native/source及按row批量encode，返回上述Packet。model不得反向import trainer。
- transaction接受trainer注入的capture/restore/train_exact_batches/evaluate/project回调；pipeline负责完整状态，model方负责树变换/新参数/守恒。仅为本次四分支需要提供回调，不扩成通用插件层。
- 两coder首次确认时可统一具体函数名并写入IMPLEMENTATION_REPORT；不能改变数学、权限、角色或预算。

## 3. 数据恢复与权限

优先复用既有prepared，按参考strict_data中`assign_roles`、`_raw_schema`、`_chunks`及fit-only编码计算直接验收；仅缺失或不符时恢复，去掉运行hash依赖。保持自然CSV行序：前80%划T0/P/Q，后10%为V、末10%为E；前80%每域固定原协议无放回划fit80%、probe10%、query10%；probe再按60/20/20分P_label/P_cal/P_update，query各半分Q_explore/Q_confirm。不改变域组成、不做增强或平衡训练。

角色映射严格为T0=fit，F=P_label，S=P_cal∪P_update，Q=Q_explore，V=validation，E=test，legacy=Q_confirm。encoder/scaler只见T0；ML标签rating>3和rating>4，域映射原age；KR域tab 0/1/4映射0/1/2，标签stored is_click/long_view，不将long_view乘click。

重建验收逐chunk对照raw标签与encoded y、域、每个encoded特征；重新按T0原值集合计算vocabulary和age min/max，与实际预处理对象逐值比较；角色数组逐项比较固定划分函数输出；检查完整覆盖、无重复和互斥。直接导出每角色每域任务真实正负数及预处理配置。不得用仅存在manifest代替数据验收。

F_probe每域固定无放回min(1024,N_d)，两个任务同行。ML raw user_id可用于F子面板、ridge folds、S三窗的用户分组；使用原始ID，不能把encoded OOV 0视为一个真实用户。KR所用CSV没有可信user_id，明确row级。S窗口在读取质量前固定，按域尽量分层且ML用户不跨窗；某窗某域支持不足按真实数拒绝候选，不搬Q/E标签。F子面板、OOF只属于训练内预测器诊断。

labels(role,purpose,ids)按角色及行成员校验，输出访问日志。T0仅native/pilot/normalizer/owner约束；F可推荐/probe/gain/partition/shadow；S仅一次结构验收；V仅select；Q仅冻结状态后的阶段诊断；E需所有要报告的方法/状态/阈值统一锁定；legacy不参与本轮训练与选择。模型forward只接受输入和domain，API拒绝y/gain/草图。

## 4. 共同前缀与状态生命周期

固定融合M3oE：3域2任务4共享expert，fcn_dims=[512,256,256,64]，固定exp_d/exp_t/bal_d/bal_t=1，无learned fusion，无AutoML/DARTS，单Adam。native h是4个expert分别LayerNorm后拼成256维，与参考NativeAdapter保持逐值前向/梯度语义。

每dataset-seed共同native训练T0共24个自然epoch，batch40960、Adam lr=.001、weight_decay=1e-5；V macroLL在epoch12—24最小，1e-12内并列选最早。source pilot对真实owner T0训练12 pass，两任务各256→32 ReLU→1，lr=.001/weight_decay=1e-5/batch40960；两隐藏拼64维，按T0_s人口std（floor=.001）标准化。把原head转换到标准化phi坐标，验证同shape FP32绝对误差≤1e-6，跨shape采用08容差并保留最差点。

每共同前缀仅构建一次供所有配对臂。ADAPTER_JOINT冻结native/pilot；缓存h、z0、phi是数据，可落盘。FULL_MODEL_JOINT复制可训练native且每batch实时产生h/z；参考native0/pilot保留冻结phi路径，禁止读取当前native静态缓存。

selected/final完整RAM状态至少包括模型、buffers、Adam全部组m/v/step、train/eval和requires_grad、树/稳定节点、teacher、G及原始统计、scale、prior/action映射、pair资格/队列、Python/NumPy/CPU/CUDA RNG、具名子stream、sampler/owner/probe/shadow游标、主时钟、S消费标记、选择历史。不得调用torch.save/pickle保存模型。一个resident worker可以顺序训练多臂，完成时保留所需selected/final和共同前缀于RAM，等待全局freeze后一次评价E。先验收并保存可复算预测/风险/统计，再释放RAM。进程丢失须记录失败成本并从共同前缀重建，不凭metadata声称模型可恢复。

## 5. BASP-A预测、收益与六开关

目标头六个258→32 GELU→1输出层0；need trunk为258→32 GELU；每域初始两需求叶及两source叶。local为32→T且全0；source f_sj=phi_s+U_sj GELU(V_sj phi_s)，U=0，V~N(0,.02²)，rank4，无bias；owner64→2线性初始0；读出w_dt,s为64维、初始0，兄弟共用，投影L2≤2。mu每域输出[T,(D−1)K_S]，中心在epoch2末由预测mu KMeans2初始化（kmeans++、8次、100iter、tol1e-4）。全相同或中心距离≤1e-8则均匀q并记NO_PREDICTABLE_PARTITION。

参考logit z_ref=z+a+Σq local；最终logit=z_ref+Σq alpha c，c=wᵀf。need q用带任务1/T、动作质量权重的mu到中心距离softmax，温度1，分裂树在父责任内计算。严禁逐样本单位化收益画像。

teacher为上一轮完成投影/结构提交后的完整冻结副本，epoch2起每epoch刷新；同teacher同OFF、完整F_probe所有foreign原子计算b=BCE(y,z_ref)−BCE(y,z_ref+c)，允许正负，全部detach。epoch1后完整probe全数据集RMS冻结为s_b，floor=.001。目标是b/s_b；G按teacher自身q加权原量纲b，保留任务/方向/动作幅度/epoch/版本、二阶矩、unique行数、ESS=(Σq)²/Σq²及observed/prior/unknown。重复访问同row不增加独立n。

epoch1/2固定alpha=m/2且NULL=.5，初始m均匀和1。epoch3起u=[G/s_b]+，alpha_a=m_a u_a/(1+Σm_bu_b)，NULL=1/(1+Σm_bu_b)。本域mask严格0；ESS<64未测路径UNKNOWN并OFF；子守恒prior仅沿用一次refresh，不把缺测填负收益。G是固定buffer而非BCE学习参数。每step固定轮转target/source/root动作shadow只更新w，两个任务同行，source/off全部detach；路径OFF仍继续测量。

|开关关闭|必须保持的语义|
|---|---|
|GAIN_CLUSTER|冻结h0固定32维投影的无标签T0/F KMeans责任；普通对照删掉不用mu；aux_matched=true才保留独立gain/CL语义头|
|GAIN_RELATION|显式relation_fallback=NONE无foreign预测，或FIXED alpha=m/2；G/probe仍可诊断|
|DYNAMIC_SPLIT|不创建事务、不读取S；固定K且保留异质性诊断|
|CONTRASTIVE|loss=0，不维护pair队列|
|SOURCE_WRITE|仅foreign source值detach，owner仍可训练；不冻结整个source模块|
|LOW_RANK|使用实测dense G及mask；不改变rank4 source函数|

GAIN_RELATION=false/NONE不得做SOURCE_SPLIT；NEED_SPLIT合法。没有foreign预测时effective_write=false。LOW_RANK=true按有效ESS加权、rank2、ridge=.001、确定初始化5轮ALS，保存实测G与重构G；unknown/同域mask/prior独立保留。报告误差/符号错率/政策损失，不宣称减少source调用。

## 6. 损失与梯度权限

L_rec=所有自然有效标签BCE总和/有效标签数，禁止整体再除模块数。默认L=L_rec+.1gain+.05cluster+.01contrast+.1owner+.1anchor+.05shadow，size=0。gain为等域probe平均、任务/动作加权MSE；cluster为树路径责任加权mu中心距离，每行除参与节点责任和。owner/anchor按T0真实π_s加权，两任务均权；anchor为冻结pilot到当前owner混合的稳定logit KL。

epoch1只rec/reader/shadow，source/owner及未启用参数grad=None；epoch2 gain系数1预热，无额外epoch，其他结构项0；epoch3起默认gain/cluster/owner/anchor，source开放owner及被允许rec写；contrast epoch5起。每step自然F batch2048保留尾批，额外循环probe256与每source owner256；计入研究成本。

contrast最多64正+64负对：同teacher硬组且预测距离≤.2为正；同域不同组只有在F_probe_A/B各组支持≥64、实测收益画像距离两边均≥.2且含OFF最佳动作不同才为负；margin=min(.5×两面板较小组距,1)。空集合项0；资格/margin detach；不能用CL距离提议split。

逐loss权限严格按05表：rec→a/local、r/mu/q、w以及被允许foreign U/V，FULL scope可更新当前native；gain/cluster/CL仅need系统且h/z detach；owner/anchor仅source U/V和owner router；shadow仅轮转w；native0/pilot永远冻结。foreign不调用owner router。共享r的辅助梯度cap：g_rec+γg_aux，γ=min(1,.25 max(||g_rec||,.001)/max(||g_aux||,1e-12))；epoch2预热不cap；mu专头/中心保留目标梯度。通过分别autograd合成实施，禁止不小心让辅助梯度写入native/source。

Adam非center lr=.001，center=.0003，FULL native=.0001，betas(.9,.999)/eps1e-8/weight_decay0；稳定名字组，不重置旧状态，inactive参数grad=None。每次w投影半径2。每epoch/候选末，完整T0_s逐任务owner风险相对pilot≤.001，所有active残差在T0_s及每foreign F_d RMS≤.5；流式完整累计，禁止用随机batch替代。固定owner/V，共同比例γ缩全部U，20次二分返回可行下端、容差1e-8，风险与RMS分别求界取小；保留Adam moments。连续两次γ<.1记录SOURCE_CONSTRAINT_BINDING并停止该source增长。

## 7. 结构提议、守恒和事务

诊断epoch4/6/10/14，事务6/10/14，每窗最多一全局父、每run最多3次，每域need/source叶≤4，拒绝父冷却一窗。只用F提议：BCE-only gain及source U逐样本梯度，不包括CL；rank4×64的U草图用固定Rademacher16×256（±1/4）。两折ridge=.01，输入冻结h0,z0,taskonehot，在train fold标准化；比较fold常量skill>0。共享task的need分区使用任务预测草图拼接再聚类，部署不接收task标签；source proposal的任务轴同样保留，禁止把相反任务先平均。全probe重拟合后将分区编译为无标签仿射sigmoid，balanced agreement≥.75；每子硬n≥100、ESS≥64、父mass≥5%；相邻诊断交叉agreement≥.70，或两固定F半面板风险交叉优势方向一致。按可预测异质性×父质量×预计改善排序，F-only取一。支持不足只记INSUFFICIENT_SUPPORT。

NEED：父函数保留，两零输出32→T residual，责任q_parent*r及q_parent*(1−r)，G行继承。SOURCE：父保留，两rank4零U/非零V残差，m_parent按固定F可部署r守恒拆分，owner父责任拆分；兄弟w共享。新增mu列和center列复制父值，动作距离权重守恒拆分；子observed count=0，只是prior。FP32同shape前后logit≤1e-6、FP64≤1e-12；任意非.5路由、负/正父G、NULL及多任务都须成立。

四分支从同完整RAM状态fork：KEEP、任务私有READER(r32→2→1)、WIDE无条件可训练残差、SPLIT。need WIDE为32→hidden→T GELU最后层0；source WIDE从rank4—16找最近实用容量。计入split新增router/gain/center/owner，参数差≤5%、相关MAC≤20%；不可匹配则CAPACITY_MATCH_UNAVAILABLE，不作结构必要性判断，不用dummy参数。

四分支同8个连续F/owner批计划、冻结teacher/G/CL资格、具名候选新增RNG，旧流不被初始化扰动；投影后同时且仅一次看本窗S。SPLIT相对KEEP及WIDE覆盖组LL各改善≥.001，自然加权贡献≥.00005，macroLL损害≤.0002，任一cellLL损害≤.003；不能被更便宜READER在该组超过.0002。若不满足，只选满足同安全边界且风险最小READER/WIDE，否则KEEP。WIDE胜不记split成功。提交赢家全部状态、游标、S消费标记；失败分支不能污染母状态。

所有配对臂在6/10/14各有8次额外主更新，即使无proposal仍KEEP且不读S。提交路径8步已计主时钟，不重放；另3分支各8步记额外搜索成本。

## 8. P0验收和放行

P0必须输出SCHEMA/PERMISSION/PARITY/STATE/COUNT真实测试收据，review独立阅读设计、最终代码和CODER_IMPLEMENTATION_REPORT。只有PASS或PASS WITH FIXES且blocking已修复才能训练。CPU合成仅用于工程数学验收，研究性能必须真实自然数据。

必要测试：forward无标签且改变y不变；角色/purpose/row校验与S消费；手算精确BCE正负及OFF、一阶/二阶；逐loss梯度白名单和SOURCE_WRITE=false owner仍有梯度；w/U零初始化及shadow冷启动；G全非正时等z_ref；need/source守恒含动作质量与unique计数；KEEP恢复下一步loss/参数/Adam/RNG/游标逐项一致，拒绝无污染，新参step0/旧参不重置；空支持与单类AUC=NA、ESS及mask；source γ0恢复pilot、凸风险与RMS缩放及投影先于V；六单关与规定交互/非法配置；所有臂主更新数一致；CL-only不触发split、read不冒write；实际参数计数63680、TARGET width41为63966；分裂动态计数无dummy；FP64 softplus LL与logit AUC排序；同batch重放严格，跨shapeatol1e-5/rtol1.3e-6。四个合成语义例为同x相反标签、平均掩盖冲突、相同子函数、全部OFF。

B/C未触发未实现不得标PASS；触发B必须补真实Adam虚拟步及有限差分hypergradient；触发C必须补simplex/QP残差/精确BCE近似检查。不要为未触发候选预先扩大全套实现。

## 9. P1—P5矩阵、门和预算

P1a按顺序TARGET_MATCH、STATIC_LOCAL、CLUSTER_ONLY；P1b FIXED_READ、RELATION_ONLY、CLUSTER_RELATION。每两数据×两dev seed，共24条适配；NATIVE_RAW只评共同native，不新训练。P1a阴性不能跳过P1b。20 F epoch、2048自然batch、V从5—20最小macroLL选最早；selected与epoch20 final均保留。每配对臂主时钟相同。STATIC等prototype臂同source/owner/probe/shadow机会；真正source-free目标臂省instrument成本并如实记账。

P1输出gain预测相对常量MSE skill、符号balanced accuracy（|b|<1e-4单列）、best-action regret、校准斜率/尺度；F predictor folds与冻结Q分别报；固定F语义分区下收益异质性和政策偏好；GAIN对FIXED/NONE/NULL；固定状态移除与同预算reader重适配分开；四格CG-C与G-STATIC交互。NONE/GAIN四格改变整条source路径，不可当纯C×收益矩阵交互。

P2每数据最多6变体×两seed，总≤24：优先1 C_SPLIT（C中支持且可部署需求差异），2 C_CONTRAST（gain可预测但需求分化退化），3 CG_WRITE（局部读价值或明确未覆盖形成需求），4 CG_SPLIT（稳定局部差异且普通读不足），5 FULL_DENSE（至少两组件支持或一次事前注册全组合交互试验），6 FULL_MINUS_CONTRAST（FULL跑过且pair非空），7 FULL_MINUS_WRITE（FULL且foreign写活跃），8 FULL_LOW_RANK（实际规模增长或有决策压缩机会）。按优先级取触发前6，未触发不补预算。任何FULL有限交互试验必须在看其Q前注册原因和比较。

P3每数据最多2变体、总≤8，与B/C共享池。优先0 CG阳性需要纯归因的C_FIXED_READ；随后依事实选：B需要可预测写需求+形成瓶颈；C需要多动作成本瓶颈且二阶Spearman≥.7、非近零改进中位数相对误差≤20%；CONTINUOUS需要gain连续可预测但离散分区无特定价值；GAIN_SHUFFLE用于已有收益的语义归因；TARGET_SUPERVISED_LIBRARY用于来源归因且真实标签预算可匹配；UNIFORM_OPPORTUNITY仅明确冷路径训练机会瓶颈。一次必要去CL/固定aux、去anchor保硬约束、初始化/路径顺序互换等诊断也必须在既有P3预算竞争，不能在预算外追跑。不得把A macro阴性当自动启动B/C理由。

P4仅保留最简A与TARGET_MATCH，同FULL_MODEL_JOINT权限/初始化/native lr=.0001/20epoch时钟，2臂×2数据×2seed≤8。若没有保留A则NOT_TRIGGERED。报告gain漂移、固定phi增量、owner保护和双路径部署成本。

开发总适配≤64，P2≤24/P3≤8/P4≤8但总上限同时约束。共同前缀另计。每数据独立决策，不平均ML/KR掩盖方向。

C*从TARGET_MATCH、FIXED_READ、最强非收益容量控制按Q macroLL固定选，仍报告各原始对照。全局门：两dev seed相对C*的Q macroLL均<0，均值≤−.0005；macroAUC平均差≥−.0005；任一cell平均LL损害≤.003。局部门：同一F预定语义条件两seed Q组LL各改善≥.001、自然加权贡献各≥.00005、macroLL各损害≤.0002、macroAUC平均下降≤.0005、最差cell平均LL损害≤.003；seed间按域/任务/动作语义匹配而非簇编号。F存在但Q局部不成立最多一次P3有价值诊断，无新证据停止相应版本。

P5仅达到全局或局部门的数据触发。先冻结配置/方法/阈值，再用2024/2025/2026三个独立训练seed，最多4臂：retained A、C*、最强容量/结构竞争者、一必要消融。每数据最多12轨迹，两数据总≤24；共同prefix最多两数据×五seed=10。Q只锁定报告，不再选组件；所有方法selected/final、身份和计划齐全后统一一次释放历史E。seed统计实际n=3，报告paired均值/sampleSD/方向数/95% t interval（df=2），不得写n=4、不得把开发seed补作确认重复。没触发P5也要冻结完成的开发矩阵再统一历史E报告，明确开发选择人口与历史E曝光属性。

停止当前版本必须有规定证据：gain不能超过常量且一次有理由判别无信息；split持续不胜WIDE/READER；只有CL距离而无risk差；source硬约束将变化持续压近0；或目标适配匹配收益且更省成本。仅单任务异向、S不足、无proposal/CL资格、低秩无效不能当整个方向被否定。阶段决策由主agent分析原始数据后记录，runner不得裁决。

## 10. 指标、导出、运行与交付

性能最少每seed每域任务AUC/FP64 logit LL、macro有效cell平均及分母、natural LL、每task pooled AUC、正负支持、最差cell变化；selected与final分开；相对NATIVE_RAW/TARGET_MATCH/C*配对差。单类AUC=NA。原型soft AUC按正负责任乘积加权，并报硬argmax；不把soft质量、任务、epoch、fold当独立seed。

必须保存RUN_MANIFEST、DATA_ROLE_MANIFEST、LABEL_ACCESS、EPOCH/SEED/CELL/PROTOTYPE_METRICS、GAIN_CALIBRATION、GAIN_TABLES与ACTION_MANIFEST、FUNCTION_INTERVENTIONS、GRADIENT_SKETCH_DIAGNOSTICS、SPLIT_PROPOSALS、TRANSACTION_RESULTS、STATE_PARITY、CONTRASTIVE_DIAGNOSTICS、SOURCE_CONSTRAINTS、GRADIENT_PERMISSIONS、PARAMETER_AND_MAC_LEDGER、COST_LEDGER、FAILURES、FROZEN_SELECTION、HISTORICAL_E_RECEIPT及最终报告；未产生字段写MISSING/NOT_TRIGGERED/UNMEASURED而非0。状态身份用显式稳定名字、版本及epoch/step，checkpoint身份指RAM快照标识，不产生新hash。

保存预测/标签/行ID、实际动作风险与责任数据供复算；不保存任何模型tensor或optimizer checkpoint。记录训练/部署/冻结工具参数、allocated/active叶、真实source调用、显存峰值、有效训练/研究搜索/失败秒数、raw-X batch1/256 P50/P95，缓存前缀与端到端分开。每个GPU按自身显存及功耗均<85%时继续并发，先查询实际余量；尊重GPU0现有他人进程，不干预。OOM先在原batch下排查并发，不默改batch。

所有项目Python命令使用`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`或指定conda环境。先最小smoke、单epoch，再完整多seed；smoke不冒充研究结果且只用允许seed。完整20epoch不得按当前性能擅自提前截断。

主agent最终按证据独立分析AUC/LL、每域/任务、seed一致性、域规模、基线难度、model/scope和训练阶段；逐条区分稳定现象、矛盾、STRONG/MEDIUM/WEAK证据、竞争解释、域规模主解释可行性、小域收益/大域损害交易，以及最有区分力的下一实验（建议不等于自动追加执行）。完成后把新增实现、配置、全部原始数值/机制证据和中文分析报告推送My-Research-Project新分支26-10-28，保留历史记录。

本设计跳过未触发备选的预先实现、自动合并、额外超参网格、无关防御测试；RAM丢失会导致相应状态需重新训练，故resident进程必须活到统一评价完成。
