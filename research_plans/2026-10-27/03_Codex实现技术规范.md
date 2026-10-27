# Codex实现技术规范

本文件是新包的实现合同，不表示下述类和CLI已经存在。读取基准为main@cc954f86410e887105fe39c6425c3fe93ab60c7a。实现只能新增包和新报告，不修改历史归档。

## 1. 路径、依赖与复用

新包：`experiments/mdmt_adaptive_prototypes/`。建议模块：config.py、contracts.py、data.py、native.py、pilots.py、reference.py、prefix.py、functional_bank.py、readers.py、constraints.py、demand.py、structure.py、rowplans.py、state.py、costs.py、metrics.py、runner.py、analysis.py、cli.py、tests/。

选择性vendor来自`history_experiments/26-10-26/experiments/mdmt_dfd_20261026/`：native_adapter.py、functions.py的ResidualMLP/PilotBank、plans.py的历史namespace、state/metrics中的无副作用原语，及vendor/strict_data/{data_roles.py,artifacts.py}、vendor/pinned_fixed/basic/{features,layers,activation,initializers}.py、utils/data.py、models/multi_domain/{fixed,original}.py及父__init__.py。strict_data/artifacts依赖祖先state，不能只复制单文件。保留来源SHA、原版权和相对包层级；旧prefix.py用于核对而非直接驱动新训练，旧cli的dry-run不能作为安全审计入口。

依赖torch、numpy、pandas、scipy、scikit-learn、tqdm、PyYAML。audit读取实际Python/CUDA/cuDNN和库版本，成功环境固化requirements与ENVIRONMENT.json；历史环境不等于当前可用环境。cuda:0指当前可见设备空间中的逻辑设备，保存CUDA_VISIBLE_DEVICES及物理UUID。不得占用/终止别人的服务，不默认物理GPU0空闲；不改/home_new。

## 2. 类型和基础API

```python
@dataclass
class Batch:
    x: Tensor       # float32 [B,F_raw]，类别索引在embedding内转long
    y: Tensor       # float32 [B,2]
    mask: Tensor    # bool [B,2]
    domain: Tensor  # int64 [B]
    row_id: Tensor  # int64 [B]，原CSV 0-based行号

@dataclass
class NativeFeatures:
    h: Tensor       # float32 [B,256]
    z: Tensor       # float32 [B,2]，logit，不是概率
    shared: Tensor  # float32 [B,4,64]
    domain: Tensor  # int64 [B]
    row_id: Tensor  # int64 [B]

@dataclass
class FrozenFeatures:
    h: Tensor       # [B,256]
    z: Tensor       # [B,2]
    hz_normalized: Tensor  # [B,258]
    phi: Tensor     # [B,3,64]
    base_logits: Tensor   # [B,2]
    domain: Tensor
    row_id: Tensor
```

旧NativePacket仅logits/shared/h，没有domain/row_id；新wrapper从Batch附加，不调用不存在的packet.domain。X最后列与domain逐行一致，两个任务为独立Bernoulli，不做softmax二分类。六格顺序cell=3*task+domain。

```python
DataRegistry.discover(config) -> DataLocations   # 只读，不推测缺失文件
DataRegistry.prepare(locations, config) -> DataContract
DataRegistry.ids(role: str, slice_id: int | None = None) -> np.ndarray # 排序int64
DataRegistry.batch(ids, role, purpose, device) -> Batch
SourceLabelView.get(ids, kind: Literal['real','joint_permuted']) -> Tensor[B,2]
NativeAdapter.encode(x: Tensor[B,F_raw]) -> NativeFeatures
FrozenCache.get(row_ids, device) -> NativeFeatures
SourceRoot.from_pilots(pilots, T0_features) -> FrozenSourceRoot
FrozenSourceRoot.forward(h) -> Tensor[B,3,64]
FrozenSourceRoot.teacher_logits(phi, z) -> Tensor[B,3,2]
ReferenceGroup.forward(hz_normalized, z, domain) -> Tensor[B,2]
```

DataRegistry按purpose控制标签访问，输入缓存与标签分离。SourceLabelView置乱视图只影响指定source训练路径，不能改F目标标签；PERM具体规则见04。

## 3. 数据来源、特征和精确行角色

历史位置是定位线索而非当前存在承诺：

| 数据 | 原CSV | raw SHA256 | 行数 |
|---|---|---|---:|
| MovieLens | /home/xzf/data/ml-1m/ml-1m.csv | f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4 | 1000209 |
| KuaiRand | /home/xzf/data/kuairand/processed/kuai_rand.csv | a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f | 2507666 |

旧prepared根为`/home/xzf/data/mdmt_next_20261024_strict_v1/{dataset}`。可复用相同编码/行身份，但本轮权限overlay与模型身份必须另建，不能修改旧manifest伪装同协议。数据找不到则该dataset BLOCKED_DATA，不重采一份近似CSV。

ML列序：user_id,movie_id,gender,age,occupation,zip,cate_id,domain_indicator。cate_id取genres按`|`分的第一个；age是唯一dense，T0 fit minmax，不clip；其他sparse embedding16。年龄原值在缩放前映射{1,18}→0、{25}→1、{35,45,50,56}→2，未知报错。标签rating>3、rating>4，mask均True。ML稀疏词表大小含OOV：5401/3632/3/22/3201/19/3，分别对应user/movie/gender/occupation/zip/cate/domain；合并embedding维113。原user ID可用于聚类bootstrap，编码OOV=0不是用户ID。

KR原tab0/1/4映射0/1/2，任务为原is_click、long_view，不人为创建click条件mask。列序为user_active_degree,is_live_streamer,is_video_author,follow_user_num_range,fans_user_num_range,friend_user_num_range,register_days_range,onehot_feat0…17,video_type,upload_dt,upload_type,server_width,server_height,music_type,tag,counts,domain_indicator，共34 sparse、embedding维544。字符串/float/int解析严格复用固定strict_data的原dtype：onehot0…17、server_width/server_height/music_type float32后字符串化，user_active_degree及各种range、video_type/upload_dt/upload_type/tag按string，其余指定列按原int32，tab/标签int8。缺失统一原实现的`<NA>`，词表仅T0拟合、排序字符串、编码从1、OOV0；domain保持0/1/2而非加1。该processed CSV缺可信原user/item ID，不发明身份或宣称user-disjoint。

原CSV前floor(0.8N)为base-train，至floor(0.9N)为V，剩余E；不能自动解释为严格时间外推。每域base-train行用PCG64(20261024+1009d)排列，前floor(.8Nd)为fit、至floor(.9Nd)为probe、剩余query。排序probe后PCG64(20261024+20000+1009d)，切floor(.6Nprobe)、floor(.8Nprobe)为P_label/P_cal/P_update；排序query后PCG64(20261024+40000+1009d)，一半Q_explore、一半Q_confirm。排序P_label后PCG64(20261025+60000+1009d)，一半P_form、一半P_read。各角色最终排序，P_label只是父池。

本轮T0=fit；F=P_form∪P_read；S=P_cal∪P_update；Q=Q_explore；V=validation；E=test_exposed；legacy=Q_confirm_exposed。同一行两任务共同划分。

| 角色 | ML总数（域0/1/2） | KR总数（域0/1/2） |
|---|---|---|
| T0 | 640133（124759/254879/260495） | 1604904（138225/1395118/71561） |
| F | 48010（9357/19116/19537） | 120367（10366/104634/5367） |
| S | 32007（6238/12744/13025） | 80246（6912/69756/3578） |
| Q | 40008（7797/15930/16281） | 100307（8639/87195/4473） |
| V | 100021（27111/35523/37387） | 250767（2692/246232/1843） |
| E | 100021（27687/41434/30900） | 250767（2613/246284/1870） |
| legacy | 40009（7798/15930/16281） | 100308（8640/87195/4473） |

S按每域排序行以PCG64(20261027+80000+1009d)排列，np.array_split为5份，再各份合并域并排序。依次绑定pass6/8/10/12/14。每窗消费一次；无候选记SPENT_UNUSED而不把机会后移。S不是最终测试，Q是明确的开发选路集合。历史E/legacy均已暴露，重新随机seed或切S不构成新独立确认数据。

## 4. 前缀训练、身份和缓存

native沿原make_native谱系，先按历史namespace构造Original，再剔除_weight_*，strict载入fixed，保持初始化RNG消耗；不能同seed直接新建fixed冒充逐位同初始化。固定融合全1、3domain/2task/4shared experts，fcn_dims=[512,256,256,64]，训练gate输入保持emb.detach。h为四个shared expert各自无仿射LayerNorm(eps1e−5)后concat，z还包含domain/task专家。risk=True不是合法训练路径。

native：自然T0训练24epochs，physical=logical batch40960，尾批保留，BCEsum/(2B)，Adam lr.001、betas.9/.999、eps1e−8、coupled wd1e−5，FP32无AMP/TF32；V macroLL在12…24选最小，平局最早，完整24轮执行。OOM报BLOCKED_NATIVE_BATCH，不静默microbatch/累积或改lr。新LL口径可能改变旧selected，身份保存metric_version。

pilot：六个独立256→32ReLU→1，hidden Xavier/bias0，output weight/bias0。只用所属source T0真实标签（PERM另见04），12自然passes，batch40960、Adam同native；固定pass12。某批缺source则该head不构造无效损失，grad=None，不用零grad推进状态。之后按02转64维root/teacher。

B_REFERENCE：六个258→32ReLU→1残差head，输入02的N_d([h,z])，输出z_t+residual；hidden Xavier/bias0，末层weight/bias0，pass0等native。T0训练20passes，batch40960，Adam lr.001、betas.9/.999、eps1e−8、wd0。V候选0/5/10/20，选完整六头组，不逐格拼接。B不使用F/S/Q标签，不读取source函数。

缓存按原row_id寻址h/z，float32，只存冻结输入特征，不含标签。绑定raw/编码/row顺序、选中native state、提取器代码、precision和角色合同hash。缓存phi/B时还要绑定pilot/normalizer/B state hash。旧清理收据表明模型及h/z可能已经删除；没有真实模型不得凭CSV汇总恢复。仅同seed或超参相同不够复用。

前缀顺序：native训练/选择→h/z缓存→pilot训练→source标准化/teacher转换→B训练/选择→PREFIX_COMPLETE。保存native selected和epoch24、pilot12、B selected/20、normalizers与全部hash。多个方法共享同一前缀；独立部署成本仍计前缀，不视为免费。

## 5. 稳定随机流和行计划

历史native/pilot保留SeedSequence([seed,dataset_id,namespace,source,task,component,pass_id])，所有缺省尾项均0；native init10、native行20、pilot init30、pilot行31，torch_seed取generate_state(1,uint32)[0]。dataset_id ML0/KR1。

新组件seed函数：

```python
def component_seed(dataset, seed, component, event='0', branch_stream='common'):
    text=f'MDMT-AP-v1|{dataset}|{seed}|{component}|{event}|{branch_stream}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest(),'big') & ((1<<63)-1)
```

不允许字段含`|`；torch/PCG64使用63位整数，sklearn使用低32位；不使用Python hash()。组件固定命名：base.init.d{d}.t{t}、base.rows、adapter.trunk.H{H}、primitive.s{s}.{leaf_id}.V、private.s{s}.t{t}.{leaf_id}.V、formation.rows、anchor.s{s}、warmup.rows、refresh.rows、window.rows、window.anchor.s{s}、monitor.rows、monitor.folds、monitor.J、kmeans.s{s}.{leaf_id}、geometry.s{s}.{leaf_id}、source.cluster.s{s}、source.permutation.s{s}、intervention.rows、bootstrap。初始化event0，行计划event为明确pass或window ID，KMeans event0以跨窗固定seed。源anchor cycle号存为event；window anchor另带window与cycle字符串。各方法共同reader、primitive、row计划不加入arm ID；私有不同架构初始化有独立明确component。所有初始化CPU fork_rng后移到device。

F0两个V按leaf_ID生成；WIDE rank8初V依相同顺序纵向stack，不复制训后参数。FIXED_PROFILE/WIDE_PROFILE同理。TASK私有V用private流。source labels置乱双任务使用同一索引，不独立置乱每task。

```python
make_natural_plan(sorted_rows, batch_size, named_stream, pass_id) -> RowPlan
make_anchor_stream(T0_source_rows, batch_size, named_stream) -> AnchorStream
make_paired_trial_plan(F_rows, source_rows, event_id, steps) -> TrialPlan
```

每主pass自然无放回shuffle，保留尾批。anchor源内循环并记录cycle/cursor，末尾补下一cycle。KEEP/ALT共享同一个只读TrialPlan，不各自重抽。monitor从F具名排列取min4096行并排序，fold由另一流打乱后交替0/1；同一行任务同fold。恢复保存当前排列hash及下一cursor，不能靠循环次数猜。

## 6. 模型、梯度和结构API

```python
PrototypeModel.from_spec(arm_spec, prefix_ref, init_plan) -> PrototypeModel
PrototypeModel.forward(features: FrozenFeatures) -> Tensor[B,2] # logits
PrototypeModel.predict_raw(x: Tensor[B,F_raw]) -> Tensor[B,2] # probabilities
PrototypeModel.registry() -> TreeRegistry
PrototypeModel.trainable_named_parameters() -> list[tuple[str, Parameter]]
FunctionalBank.forward(phi: Tensor[B,3,64]) -> dict[str, Tensor[B,64]]
ConditionalReader.forward(features, leaf_outputs, registry) -> ReadPacket
# ReadPacket: logits[B,2], rho[B,2,4], leaf_mass dict[id]->[B,2], router[B,2,H]
target_loss(logits, y, mask) -> scalar
source_kl_loss(bank, teachers, anchor_batches, label_view) -> scalar
source_supervised_loss(bank, teachers, source_batches, assignments, label_view) -> scalar
project_reader_vectors(reader, radius) -> ProjectionReceipt
project_bank_to_constraints(bank, prefix, data, label_view, config) -> ConstraintReceipt
```

源KL只bank，F目标一般bank+reader，SOURCE_K2目标bank.detach，DIRECT/READ/TARGET_ONLY无可训练bank。source batch缺席不禁止合法foreign目标梯度写后继；仅没有任何合法监督路径的参数保持grad=None。native/B/root/teacher均requires_grad=False、eval。

```python
collect_demand(snapshot, monitor_rows, J, data) -> dict[LeafID, DemandBatch]
# DemandBatch: row/task/domain IDs[N], router[N,32], sketch[N,16], mass[N], fold[N], scale, snapshot_hash
fit_cross_fitted_ridge(batch, config) -> DemandFit
# OOF[N,16], full A[32,16]/bias[16], input mean/std[32], skill, fold scores
compile_binary_node(fit, centers[2,16], temperature) -> NodeParameters[33]
propose_one(snapshot, fits, history, capacity, cold_state) -> Proposal | None
split_preserving_function(branch, leaf_id, node_parameters) -> TreeChange
capture_state(model, optimizer, controller, plans) -> AdaptationState
clone_branch(snapshot, proposal_or_keep, immutable_prefix) -> BranchState
run_local_trial(branch, paired_plan, config) -> CompletedBranch
accept_candidate(keep, alt, S_slice, config) -> Decision
commit_winner(keep, alt, decision, permanent_ledger) -> MainState
```

所有提议函数只用F。S在完整固定候选后评价，不反传、不挑第二候选。rank4/32维需求诊断用于F1、NO_GROW；GEOMETRIC另用32维几何。READ H43、WIDE高rank及TASK私有rank2不调用rank4梯度公式，静态对照只执行相同刷新/KEEP/REACTIVATE，不伪造需求监测成本。

split先实际删除父参数引用、登记独立子U/V，内部父保留router节点。新leaf含source/parent/rank/birth_event/alpha/coordinate_version；children不可alias父、兄弟或KEEP。所有可训练模块与optimizer需按稳定名字注册，不能依赖插入顺序或Python id。前半冻结项也提前进入optimizer，grad=None，解冻后仍使用同一optimizer。

## 7. 训练与状态事务

主训练及局部冻结/重启规则完整按02。stage1预热后fresh全轻量Adam；pass4仅替换刷新reader的named Adam状态、保留bank状态；五窗口KEEP/ALT全轻量fresh，获胜分支完整optimizer接续。不允许只提交weights却接旧moment。局部步不改变main pass编号，真实optimizer_step/exposure另记。

```python
NamedAdam = {
 'hyperparameters': {...},
 'groups': [{'names': [...]}],
 'state': {stable_name: {'step': ..., 'exp_avg': ..., 'exp_avg_sq': ...}}
}
Checkpoint = {
 'identity': ..., 'prefix_ref': ..., 'arm_spec': ..., 'tree_registry': ...,
 'model': ..., 'optimizer_named': ..., 'requires_grad': ..., 'modes': ...,
 'gradients_or_none': ..., 'main_pass': ..., 'next_main_step': ...,
 'event_id': ..., 'phase': ..., 'next_local_step': ..., 'row_plans': ...,
 'anchor_cursors': ..., 'rng': ..., 'monitor': ..., 'J': ...,
 'history_partitions': ..., 'reactivate_attempted': ..., 'S_spent': ...,
 'selection': ..., 'projection_receipts': ..., 'ledger_watermarks': ...
}
```

无梯度从未更新的参数无state条目，区别于zero moment/正step。RNG含Python、NumPy、CPU Torch、实际可见GPU RNG以及所有named stream位置；scheduler/AMP均null，不暗加。

恢复顺序：验证数据/代码/config/prefix/metric身份→按registry真实重建树→strict载入model→按稳定名建optimizer并加载→恢复冻结/mode/gradNone→控制、计划/RNG→核对每参数恰好一次→下一完整step。仅在完整optimizer step、完整投影或完整子阶段边界原子保存，不保存半个累积梯度。

事件状态机OPEN→REFRESHED→SNAPSHOT_READY→KEEP_RUNNING→KEEP_READY→ALT_RUNNING→ALT_READY→EVALUATED→COMMITTED。候选身份、S片、attempt先记持久账。崩溃后只能完成同一candidate、同一S片和行计划；已评价则复用同hash结果，不换seed/候选。拒绝/失败的费用不能回滚。ALT数值失败可拒绝并保留合法KEEP；主状态身份破损、冻结参数改变或gamma0失败应停止run，不解释成科学阴性。

## 8. CLI与可执行顺序合同

统一入口`python -m experiments.mdmt_adaptive_prototypes.cli COMMAND --config research_plans/2026-10-27/07_EXECUTION_CONFIG.yaml`。先实现后使用，不能把本规划目录当训练包。

| COMMAND | 行为 |
|---|---|
| audit | 只读源码/配置/数据存在性/hash/环境；JSON stdout；不构造模型、optimizer，不mkdir/cache |
| plan | 只读展开冻结网格、依赖、成本与NOT_TRIGGERED；stdout，不写工件 |
| prepare | 新prepared/角色overlay/编码、S片与DATA_CONTRACT；不改旧数据 |
| prefix --phase development/formal | 训练或严格身份匹配复用native/pilot/B，formal受FORMAL_LOCK约束 |
| run-stage --stage stage1/stage2/attribution/bridge/formal | 仅执行已登记run IDs；--resume同身份恢复 |
| analyze --stage ... --role V/Q/E/legacy | V仅选点，Q仅开发决策；E/legacy须EVAL_LOCK；输出状态与STAGE_DECISIONS |
| freeze --phase formal/evaluation | 固化已决定方法或实际checkpoint/诊断群体；不是额外选择器 |
| report | 从真实工件与ledger输出新报告，不训练，不根据E改方法 |

严格只读入口用`python -B`，延迟导入训练模块，设置sys.dont_write_bytecode，避免解析命令之前生成__pycache__。旧runner的dry-run不可信。--dataset/seed/arms只缩窄已冻结矩阵；缺必需对照只能INCOMPLETE，不能据子集宣布完整阶段成功。SMOKE独立身份，不进入正式统计。退出0=完成或诚实NOT_TRIGGERED；2=合同错误；3=资源/数据阻塞；4=已有不匹配工件；5=运行故障。不得把非零退出当效果阴性。

## 9. 产物、指标与自动分析

每run目录保存CONFIG、ENVIRONMENT、DATA_CONTRACT、RNG_MANIFEST、PREFIX_REF、ArmSpec、checkpoint/selected、rowplans、events、cost_ledger、constraint_log、validation_table、predictions、DECISION及report。run_id编码protocol/dataset/seed/arm/revision/profile_hash，不包含可变选点分数。

完整predictor hash覆盖native、B、root/teacher坐标、normalizers、bank、树、router/w/a、普通部署开关、ArmSpec和指标口径。预测导出row_id/domain/task_mask/y/logits/p/role/model/data hash，仅在该角色标签合法后导出。原始数据、逐行预测、权重和私人服务器路径详情不上传GitHub，服务器保留；仓库提交代码、配置、聚合表、执行状态和hash。

统一float64 stable logit BCE，不clip概率；AUC固定用float64 logits排序，避免sigmoid饱和产生额外ties。单类AUC=NA，不能填0.5；macro/natural/weighted-cell AUC/pooled AUC名称区分。旧clip LL可另列legacy，不与本版配对相减冒充同口径。

计时使用进程独占工作区间的同步墙钟秒，GPU边界synchronize；包含CPU ridge、数据加载、投影、S评分等算法内部开销，不把GPU利用率当计算秒。prefix、主训练、refresh/KEEP、所有ALT/拒绝、干预refit、通用VQE评估分别分项；避免重复计共享prefix，同时独立部署总成本包含必要prefix全部训练/选点。完整20pass后才能选点，不能只记到最终选中pass的费用。

raw-X端到端推理batch1与256，50warmup/200测量，报告p50/p95、峰值VRAM、实际执行并集及设备；缓存轻量时延另列。完整方法训练成本与仅适配时间的等预算视图区分，详见04。
