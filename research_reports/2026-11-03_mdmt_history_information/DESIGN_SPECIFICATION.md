# DESIGN SPECIFICATION — MDMT_HISTORY_INFORMATION_20261103

设计日期：2026-11-03。阶段：DESIGN COMPLETE；尚未授予正式训练审查通过状态。

本文件将已冻结方案映射到实现和运行责任。数学定义以 `research_plans/26-11-03/03_ALGORITHM_SPEC.md` 为准，工程以04为准，阶段决策以05和 `experiment_queue.yaml` 为准；本文件不新增算法、网格、seed或晋级门。`start.md` 写成26-11-3的目录实际为26-11-03。当前用户授权执行实验并提交新分支，覆盖规划来源中“只规划”的会话状态。

## 1. 责任与范围

新代码仅位于 `experiments/mdmt_history_information_20261103/`；交付位于本报告目录；标量运行工件位于对应新artifact目录。执行分支为 `26-11-03_history_information`。保留旧代码、报告、原始指标原样。流程为设计、编码、独立审查、运行、主agent分析；coder和reviewer必须是不同代理，非runner代理继承主agent模型。

实现A0f/A1/SCALE、M/U_MIX、H_REAL/H_MIX/H_U/H_NORM/C_MIX、T0/TM/T_U/T_NORM。F、PAIR、D/q、ECCM、PCV/PCOLL、多条件历史表、梯度差分EMA不属于本次实现。H和T不组合。只执行被冻结阶段门实际触发的轨迹。

当前可复用底座为本地 `experiments/conditional_information_momentum_20261102_v3/`，但复制前必须对固定提交 `a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288` 核对实际源字节。历史A0/A1必须以该提交原始完整数值和身份复用，不从报告舍入值重建，也不因当前RAM缺少其参数而重训。

## 2. 最小代码布局与实际依赖

| 新模块/责任 | 源代码复用与必须改动 |
| --- | --- |
| `protocol.py` | 显式读取本轮queue和固定`resolved_protocol.json`，构造数据、模型、配置、身份；复用源setup、EMA、CPU复制、RNG、IPC、hash。不能调用隐式读取旧计划的vendor `load_protocol()`。 |
| `mixing.py` | StepContext、MixBatch、独立随机key、自然anchor与donor、functional embedding hook；无新持久状态。 |
| `optimizers.py` | 新HistoryOptimizer；S命名规则参考PairOptimizer，但不继承其PAIR/ECCM/CV路径。普通参数继续源Adam。H小Gram活跃集求解；T全S运输与标量控制。 |
| `objectives.py` | 当前真实/混合目标、H probe、T旧点、一步更新；源A1直接走原loss路径和EMA02V2 key。 |
| `run.py` | prepare/check/stage0/dispatch/decide/freeze/evaluate/report及内部worker/guardian入口；基于已有v3 worker与guardian生命周期适配。 |
| `stage0.py`及小测试文件 | CPU代数、实际接线、RAM恢复、完整batch成本检查；不增加发现性研究。 |
| 报告辅助 | 确定性阶段规则、数据表、配对统计和成本导出可在新包少数模块内组织；不建立通用框架。 |

接口可以按04作一一对应调整，必须在DESIGN_IMPLEMENTATION_MAP记录。实现前沿这些入口逐一检查所有调用处、配置、测试和导出；无需修改历史调用方。

Vendor原字节依赖闭包从固定v3提取：`vendor/ema02/data.py`、`protocol.py`、`evaluate.py`、`rng.py`、A1的`objectives.py`及其实际导入闭包（input_probe/views/views_joint_label/views_ema_field等）；其下GA31 views/objectives/protocol，以及vendor_ga31的audited_data/source_protocol/model/reference/basic、metrics/protocol与resolved_protocol。即使某些导入分支本轮不用，直接导入的依赖仍要满足；不复制训练结果或整个旧分支。SOURCE_MANIFEST记录源commit、路径、blob、SHA256、字节数与新路径。

源v3 `guardian.py` 的serve/request可复用；`ensure_guardian`硬编码旧模块入口，需要新包适配。若复制作vendor，应保留原字节，新入口在外部适配，不修改vendor。源`ema02.protocol.load_protocol`内部ROOT指向旧计划，所有本轮数据加载必须显式传p，源A1配置显式构造。

## 3. 协议、数据和身份

两自然CSV按04所列绝对路径使用；检查raw SHA256及完整array/row/encoder/scaler/split identity。ML训练域计数155949/318599/325619；KR为172782/1743898/89452。原行序80/10/10，KR先切分再保留tab0/1/4，train-only编码和scaler、OOV0。ML标签rating>3/rating>4，KR is_click/long_view。训练抽样不变，不能改变域比例或联合标签支持。

固定fixed1 M3oE：3域2任务、embedding16、4共享专家、FCN[512,256,256,64]，真实x[F,B]、y[B,2]、domain=x[-1]。字段嵌入位置来自field_slices。原gate的emb.detach语义保留。实际模型无BatchNorm/Dropout要断言。

每条正式轨迹50epoch、batch40960、尾批实际B、无早停；FP32，AMP/TF32关闭。Adam lr=.01，β=(.9,.999)，eps1e-8，coupled wd为ML1e-4/KR1e-5，foreach/fused/amsgrad=False；只有A0有界网格允许lr/wd变化。固定非反馈EMA rho分别.9828205985452511/.995，theta0初始化，每步所有online更新结束后更新一次，buffer复制。初始化与loader保持源随机流，loader seed+11000。

科学身份canonical JSON SHA256包括data/split/source/model、arm、seed、λ/τ/κ/lr/wd/β/eps/rho/epochs/batch/RNG与协议hash；stage和attempt不进入身份。A0与A0f是同一科学arm，默认A0b若同参数亦去重。无作用的参数不得仅因字面默认值不同制造重复轨迹。历史引用登记旧key和原commit/文件/选点/数据/模型/配置身份，不改写成新训练。

## 4. 混合信息与目标

随机字符串精确为 `MDMT1103|dataset|seed|epoch|global_step|stream|domain|task`，SHA256前8字节little-endian模(2^63−1)，独立CPU torch.Generator。stream=anchor/donor/real_second；anchor task=-1，两个任务共享anchor；donor与real_second按task=t分流。key不含arm、λ、τ、κ，初始化和loader不消费此流。

每出现域自然无放回抽min(256,n_d) anchor。donor优先同域、不同原行号、同目标标签且另一任务标签不同；无支持退至同域同目标不同原行，再退自身，集合内均匀抽取。U仅从第二级开始。H_REAL第二真实probe按real_second独立自然无放回抽样，允许与第一集合重叠。保留support_level、自配率和index_hash。

对每域每任务拼接anchor/donor为2M行；embedding hook只把前M行非domain字段改为(1−τ)e_anchor+τe_donor，后M行不动。domain字段逐位保持anchor，按field_slices排布。只以anchor目标任务的label监督前M行；另一任务标签只用于donor条件，不输入x。M/TM两端embedding有梯度，hook在finally移除。functional旧点hook使用该次旧embedding输出。

L0=全部真实probability BCE之和/(2B)。L_M=(1−λ)L0+λΣ_dt[n_d/(2B)] mean_anchor BCE_t；出现域两个任务全部计入，权重和1。默认λ=.1、τ=.1。λ=0或τ=0时直接L0并旁路增强采样/forward；T仍可执行非零κ运输。SCALE=(1−λ_scale)L0。A1保持源lambda_aug=.5、ML target0/kappa.05、KR target2/kappa.1及EMA02V2 Gaussian随机key；不套本轮混合参数。

## 5. 参数集合与历史更新

S恰为`shared_weight`、`shared_bias`或`skip_conn.`、`star_mlp.`、`expert.`前缀，内部domain_specific子串不排除。按named_parameters固定顺序记录shape，S与其他参数互斥完整，每步每参数最多更新一次。普通参数None不更新，显式0推进；S意外None直接报接线错误。S保留参数级step并核对全局时钟。

当前有效g=数据梯度+wθ；S手工L2一次，其他参数源Adam一次。普通A0/M/U/SCALE/A1及κ=0旁路优先直接用源Adam，确保状态语义一致。HistoryOptimizer只替换S更新；外部当前`.grad`不被probe覆盖。

H所有版本当前监督L0。先得全部真实数据梯度和S的g，v=.999v+.001g²，D=1/(sqrt(v/(1−.999^s))+eps)。probe域为(k−1)%3，空域循环后继取首个出现域。两个任务各得真实a和条件混合b，只求S梯度、不加L2、不写`.grad`。H_REAL改b为独立真实probe，H_U取消另一标签条件。

用sqrt(D)a、sqrt(D)b，范数均>1e−12且余弦≥.2才把二者单位方向加入N（最多4）。h=sqrt(D)m_old，投影至Nᵀz≥0。无约束或m_old=0恒等。以FP64 Gram/q枚举活跃集合：λ_A=−pinv(G_AA)q_A，rcond1e−12，tol=1e−9 max(1,||h||)，检查λ非负、所有约束与活跃残差；取最小距离，等价按集合大小/字典顺序。仅小Gram/q可移CPU，不构建P×P矩阵；无可行解记录数值故障并停止该run，不清零回退。

r=((1−κ)h+κh*)/sqrt(D)，m=.9r+.1g，θ减lr D m/(1−.9^s)。H_NORM在自身状态算同一个r，却仅用c m_old，c=||sqrt(D)r||/||sqrt(D)m_old||，小分母取1。C_MIX投影对象改成sqrt(D)g，v仍用未投影g，m_old不修复，m=.9m_old+.1g_repaired。κ=0完全旁路H。H的τ=0仅使probe identity，不等于关闭历史。

T当前目标T0=L0，TM/T_U/T_NORM=L_M。冻结当前batch/索引/τ，在θ_k与前一步完整pre-update θ_(k−1)分别求同一目标。旧点用strict functional_call替换全部参数和buffer，旧点只有S需requires_grad，不原地覆盖online、不更新EMA/optimizer。g_minus包含旧点wθ_(k−1)。第一次无旧点跳过旧forward，δ=0。

δ=g−g_minus，以单一全S系数min(1,||g||/(||δ||+1e−12))裁剪。m=.9m_old+.1g+κ*.9*(1−.9^(s−1))*δ_clip；v及Adam偏差归一化按当前g。T_NORM在自身状态以D范数比缩放普通m_adam，不写运输方向，小分母保留m_adam并计数。κ=0：TM退M、T0退A0。步末保留本步更新前完整θ_k作为下一步previous_state；旧点算完释放更旧快照。不建δ EMA。

## 6. RAM生命周期、运行与最小验收

顺序固定：源batch → 独立索引 → 当前目标梯度 → H probe或T旧点 → 有限值 → 一次互斥optimizer更新 → 一次EMA → 释放图 → 标量日志。记录完整forward行数、autograd/probe/old-point次数和optimizer/EMA次数；首步及每20步细粒度诊断，其余按epoch汇总。H记录约束、cos、投影距离、范数保留及支持；T记录δ/g、裁剪、校正比、新旧距离；M记录真实/混合loss和六格权重。

保持现有独立guardian RAM持有与epoch ACK。状态包括online/EMA、optimizer/S时钟、T previous_state、RNG/loader、config/source/data/parameter身份、epoch/global_step、best/selected、E10/E50、计数。worker完成50epoch后可退出，guardian仍保留online/EMA的selected/E10/E50共六份模型状态，直到所有数据全部配置和选点冻结且最终test结束。不能将其写pt/pth/ckpt、参数JSON或磁盘pickle。RAM状态恢复使用cpu_copy/BytesIO与源IPC。RAM丢失记录失败；每科学身份至多一次有原因基础设施重放，不能因结果不利重跑。

最小验收按04第6节逐项对应：CPU参考13项；A0/Hκ0/T0κ0/Mλ0或τ0及TMκ0=M的loss、gradient、一步Adam/EMA和恢复下一步FP32 atol1e−6/rtol1e−5；数据/标签/donor三级；domain不变；M双端embedding梯度；H probe不改当前grad；S/L2/None时钟；H参考KKT；T全参数旧点（含非S变化）、同索引、pre-update快照与质量因子；各NORM/C控制；内存恢复；最早严格最大选点；冻结前拒绝test。测试组织可合并，但这些实际语义不可省略。

独立reviewer必须检查设计、实际代码、CODER_REPORT，对Critical/Relevant问题做直接验证。结论PASS或PASS WITH FIXES且blocking问题清空，才准runner做正式队列。测试和审查只因实际影响本轮结果的修订而重跑。

真实S0每数据完整40960 batch，A0/M/H_MIX/TM各2步预热+3步测量，共20步/数据；计入全部计算，CUDA同步计时、峰值显存。其耗时中位数>A0四倍则具体方案COST_BLOCKED，转简单近邻，不称性能失败。S0不作泛化结果。逐卡同时显存/功耗<85%且空闲能容纳实测峰值即准入新worker；不要求两卡同满足，不杀外部任务，OOM先降并发不改batch。显存/成本阻塞记录具体证据，不把未测方案统称性能失败。

## 7. 阶段规则落地

所有决策使用validation fixed_ema selected六格macro AUC；每epoch完整validation，严格大于更新，平局留最早，E0不选，online独立选择。所有数据全部开发/确认和选点冻结后统一test。历史test数值不得进入本轮选择代码路径。

| 阶段 | 请求与决策 |
| --- | --- |
| D0 | 两数据seed2022共9arm：A0f/A1/M/T0/TM/H_REAL/H_MIX/C_MIX/SCALE，最多18槽。核验A0/A1历史先复用；A0f2024也复用。正ΔAUC或联合arm近失条件（vsA0f≥−.0005、vs最强简单≥.0003），且无AUC_TRADEOFF才触发家族。每数据最多最高一家，差≤.0002按M/H/T优先。其余DEFERRED_BY_BUDGET；全部不触发则该数据停止。 |
| D1 | A0组、选中家族主组、D0确定简单组，各严格4个YAML配置，seed2022；网格差≤1e−6按先列配置。先去数值/成本阻塞及AUC_TRADEOFF，再排序；全tradeoff仅描述。H简单H_REAL/C_MIX差≤.0002选前者；T简单T0/M差≤.0002选M。 |
| REPLICATE | 仅各组冻结胜者补2024；主或简单任一对A0f和A0b都两seed均值≥.0005、最差≥−.0002，才D2。不能用2024重选配置。 |
| D2 | 只按05表补固定匹配控制，每数据至多6arm×2seed。T交互使用同λ/τ/κ/lr/wd的A0/M/T0/TM。候选对两个基线都均值≥.0005且2/2正、无AUC_TRADEOFF才合格；复杂H/T还须逐个匹配控制≥.0003且2/2正。距最高V≤.0003按YAML简单性顺序选择。无合格简单替代则保留A0。 |
| CONFIRM | 每数据至多A0f/A0b/所选/最强其他比较者4个去重arm，补2023/2025/2026。比较者只在已两seed、无AUC_TRADEOFF家族候选选，差≤.0003按简单性。不凑数、不再调参。 |
| FREEZE/TEST | 配置与selected身份全局冻结，再一次统一test全部已执行端点。若数据在开发停止，保留A0决定，不能凭test翻案。 |

网格与D2控制逐项从YAML/05读取或忠实常量实现，不能runner临时选择。正式请求上限18+24+6+24+24=96，按科学身份去重，历史引用和S0不算新训练，失败重放单列成本。

AUC_TRADEOFF：开发对任一A0基线，同cell/traffic/pooled均值降≥.002且所有现有seed同向；确认五seed至少4/5下降。macro LL均值升≥.002且同向计AUC_WITH_LL_TRADEOFF，可作为排序候选继续但保留代价。单seed保护仅资源排序，不描述为稳定现象。最终接受对A0f和A0b分别五seedtest均值≥.0005、至少4/5正；2023/2025/2026单列均值>0且至少2/3正。联合H/T还对确认最强近邻≥.0003且至少4/5正。完整状态使用05枚举，未跑不记阴性。

## 8. 交付与分析边界

必须有SOURCE/DATA/PARAMETER_MANIFEST、DESIGN_IMPLEMENTATION_MAP、CODER_REPORT、独立REVIEW_REPORT、CPU_TESTS/STAGE0、RUN_MANIFEST、各阶段DECISION、CONFIGURATION_FROZEN/SELECTION_FROZEN、逐epoch validation、endpoints/paired_seed/paired_summary/six_cell/costs CSV、REPORT.md与QUESTION_DISPOSITION.md。逐seed六格、域/任务均值、macro/traffic/pooled/worst、AUC/LL、online/EMA、selected/E10/E50及状态hash齐全。traffic用评价人口权重，pooled按任务先合并域再取均值；undefined沿源规则。

coder只报告实现和检查；runner只记录原始结果；orchestrator分析经验与解释，seed为统计单位，报告均值、样本SD、95%配对t区间、方向数，开发2seed/本轮未开发3seed/合计5seed分开。训练步GPU时间、worker wall time、加载/验证/test/失败成本分列，不能把并发worker总时长称独占GPU时间。

报告直接回答接受哪个方案、对固定/有界基线、增强必要性、当前梯度或旧历史位置、方向性和另一任务条件增量、LL/traffic/cell代价、简单方案删掉何种组件。标证据STRONG/MEDIUM/WEAK，记录重复现象、矛盾、替代解释、域规模与基线难度的区别及是否小域收益/大域下降；不另开域大小实验。下一步建议仅提出最能区分当前替代解释的实验，不自动执行额外矩阵。

## 9. 设计检查发现与交接

已阅读冻结01–06、queue、SOURCE_INDEX、代数reference，并检查本地源objectives/protocol/optimizer/guardian/worker/data/model/evaluate及依赖入口。当前未发现需要重新决定算法的阻塞歧义。orchestrator已核对固定v3根源/vendor共50文件Git blob无差异，见PINNED_SOURCE_INDEX.json；原始归档SHA256及1535个本地原始标量文件与归档一致，见HISTORICAL_RAW_VERIFICATION.json。coder仍需从其中匹配A0/A1科学配置与所需端点。GPU/Python可用；orchestrator重算CSV SHA256为ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`、KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。完整处理后数据manifest、CPU/S0留待实现验收。

关键工程接线点是旧vendor隐式计划路径、新guardian模块入口、T完整旧点与pre-update快照、H probe保持当前梯度、历史引用去重以及全局冻结前RAM端点存活。它们会直接改变轨迹或评价，因此属于本轮验收范围。未增加新算法、库、网格或广泛工程测试。
