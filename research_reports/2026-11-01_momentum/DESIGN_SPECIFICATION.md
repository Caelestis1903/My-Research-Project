# GAM01：参数影子平均与增强差分动量设计规格

设计日期：2026-11-01。任务入口为 `start.md`、`research_plans/26-11-1/README.md` 与 `05_Codex完整执行提示词.md`；六份规划文件已通读。设计完成后由协调者审批，coder只实现本规格，独立reviewer验收通过后runner方可运行。分析结论由协调者给出。

## 1. 范围、来源与合同冲突的处理

- 本轮只新增 `experiments/gradient_aug_momentum_20261101/`、`research_reports/2026-11-01_momentum/`、`artifacts/gradient_aug_momentum_20261101/attempt_<timestamp>/`。
- 规划读取来源为 `Caelestis1903/My-Research-Project:main@e65ceb4735d533f3a9035a9199c30f68b4bde98d`。GA31-R3发布分支为 `26-11-01@a6f496b54f467cdf7b814b6bb8defdbdd457bc38`，该实验所引用底座为 `ee6e81fdfd37284e6447e8149cb93d04736dba02`；这些身份分别记录，不将底座SHA冒充本轮代码commit。协调者已核对远端源并保存 `SOURCE_FETCH_RECEIPT.json`，设计/实现不得覆盖该收据。
- 已阅读GA31-R3实际 `protocol.py/objectives.py/views.py/run.py`、vendor数据/模型/状态/指标接口、GA31-R3规格和报告；历史Polyak线索来自 `PROJECT_ARCHITECTURE.md` §0/§2.3与10月31日证据文档。历史收益不并入新实验统计。
- `start.md` 禁止本地模型/checkpoint文件优先于规划中的“保存checkpoint”：所有模型、Adam、q、shadow、RNG快照仅保留CPU/GPU RAM，通过BytesIO/Pipe传递；磁盘只写数值、哈希、代码、配置、日志及报告。恢复验收为同进程RAM恢复，不写临时权重文件。
- 自然数据人口、标签、原始行、采样比例均保持GA31-R3；本轮具体合同授权的Gaussian仅扰动训练forward中的非路由sparse embedding，不生成新样本、不改CSV/标签/域ID、不改变自然原始样本曝光。
- 03/05的 `beta_delta=0` 退化简写与分源通式不相容。明确将0作为标准混合梯度Adam分支；不得直接代入通式并声称等价。`beta_delta=.9` 保留真正的分源实现，用于与A1核对等价。
- P2存在预算冲突：K=2且固定λ=.25/.5/.75与RDI各5seed需20条/数据集，而文本上限为10条/数据集。Stage1完全可执行。P2若触发，必须在任何P2训练前由协调者冻结解决后的预算；K=1的A1/A2不可冒充K=2同预算对照。不得先实现P2或默默删减对照。

## 2. 最小实现与复用边界

不修改任何GA31-R3旧包、历史报告或其他实验。新包使用完整包名导入，禁止修改sys.path后导入顶层同名 `protocol/data/model`，禁止全局monkey patch。

| 新模块 | 职责与既有接口 |
|---|---|
| `protocol.py` | 解析本轮resolved config、四臂配置、来源与状态identity；导入GA31-R3命名空间下的setup/cpu_copy/rng_state/restore_rng/write_json/append_csv/IPC工具 |
| `optimizer.py` | 独立真实一阶矩、差分一阶矩、当前混合梯度二阶矩及逐参数step；不增加第二个v；普通路径复用torch.optim.Adam |
| `objectives.py` | 同θ同batch取得full-real、target-clean、target-Gaussian梯度；原B×T分母；复用GA31-R3视图构造与模型 |
| `run.py` | audit/check/all、RAM snapshot/restore、online/shadow分别选点、逐卡调度、40run、validation门冻结与统一test、数值工件 |
| `checks.py`和小型测试 | 本规格第7节必要代数/真实batch/恢复与RNG检查，不扩大工程测试 |
| 可合并的数值汇总模块 | 按seed生成配对数值与门结果，禁止代替协调者作科学解释 |

直接复用 `experiments.gradient_aug_20261031_v2.vendor_ga31.data.load_dataset`、`metrics.metrics/evaluate`、`audited_data.row_ids_hash`、`protocol`中的内存状态工具，以及 `experiments.gradient_aug_20261031_v2.views.make_model/draw_view`、`protocol.view_key`。可复用旧GPU状态/调度实现，但不要调用旧study_id/35–45run/ANTI门/基线选择等硬断言。历史Polyak实现源码限定检索未找到，本轮按已冻结P0公式重验，不声称逐行复现历史实现。

发布必须包含实际依赖的GA31-R3源目录或确认目标分支已存在对应文件；记录每个依赖Python/配置文件SHA256和来源commit/blob。新主模块保持最少职责，可合并小模块；不新增第三方依赖、不复制旧输出。

## 3. 数据、模型、优化器与随机流冻结

数据身份继承 `experiments/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json` 的全部raw/六数组/split/encoder/scaler hash和field顺序，只允许CSV路径重定位。

| 数据 | CSV | train/validation/test | train域计数d0/d1/d2 | 目标域 | 参数量 | steps/epoch |
|---|---|---|---|---:|---:|---:|
| MovieLens | `/home_new/xzf/datasets/ml-1m/ml-1m.csv` | 800167/100021/100021 | 155949/318599/325619 | 0 | 773470 | 20 |
| KuaiRand | `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv` | 2006132/250767/250767 | 172782/1743898/89452 | 2 | 1614254 | 49 |

调用既有loader验证全部身份。无需缺失的 `mdmt_next_20261024_strict_v1` prepared artifacts。split_seed=2022、train拟合词表/scaler、OOV0，源80/10/10行序split及KR过滤顺序保持原样。

fixed1 MDMTRec/M3oE：3域、2任务、4 shared experts、embedding16、fcn_dims=[512,256,256,64]，exp_d/exp_t/bal_d/bal_t均为Python标量1；LayerNorm、gate-input detach、sigmoid[B,2]不变。新模型state_dict键/参数顺序应与源完全一致，不新增网络参数。源模型无BatchNorm/Dropout。

所有正式run使用seed2022–2026、50epoch、batch40960、FP32；无早停、AMP、TF32、scheduler、clipping或gradient accumulation。自然无放回shuffle、drop_last=false、num_workers0。一个参数更新/原batch，ML1000步/run、KR2450步/run。全矩阵共69000次正式参数更新、2000epoch。

lr=.01、β1=.9、β2=.999、eps=1e-8、coupled L2；ML使用L2-ML：loss_scale1/wd1e-4，KR使用B-KR：loss_scale1/wd1e-5。标准Adam的amsgrad/foreach/fused=false。不要恢复旧ML loss_scale=.1。

初始化前调用源setup；同数据集同seed四臂初始张量、buffer、键和参数顺序匹配。CPU loader generators使用seed+11000/+33000/+44000，train顺序由原 `torch.randperm` 产生；每epoch和batch写row_id哈希并跨臂配对核验。

Gaussian复用R3独立device Generator与键 `GA31-R3|dataset|seed|train|global_step|gaussian|1`，step从1开始。这样四臂之间可配对同step噪声，且不推进全局Torch/loader RNG。只有目标域存在、增强开、λ非0、κ非0时抽view；关闭增强/λ0/κ0/目标缺失不得抽view。不引入持续的可变view RNG；恢复所需view身份由seed+step重建并记录key/hash。

Gaussian目标为ML d0/KR d2；ML6/KR33个 `perturbable_sparse_fields`，实际顺序沿源field_slices。每目标行/字段使用 `e + .1*stopgrad(RMS(e))*N(0,I)`，RMS无floor。dense、domain embedding、路由、ID、标签、行顺序不变。LayerNorm affine参与参数EMA。

## 4. 四臂、梯度与状态递推

| 臂 | 增强 | 差分状态 | shadow |
|---|---|---|---|
| A0 | 无 | 无，标准真实Adam | ρ=.995 |
| A1 | Gaussian K1，λ=.5/κ=.1 | beta_delta=0哨兵：标准混合梯度Adam | ρ=.995 |
| A2 | 同A1 | 分源SAD beta_delta=.90 | ρ=.995 |
| A3 | 同A1 | 分源SAD beta_delta=.99 | ρ=.995 |

A2在实数代数上与A1等价，是分源状态/数值实现对照；A3才改变差分时间尺度。不能将A2与A1的FP32累积差写成独有机制收益。A2仍按合同完整训练，不因代数等价删臂。

令B为原始batch行数、T=2、mask为指定目标域：

`L_real = sum(BCE(model(x), y))/(B*T)`

`L_target_real = sum(BCE_real[mask])/(B*T)`

`L_target_aug = sum(BCE(model(x[:,mask], view), y[mask]))/(B*T)`

`g_real=grad(L_real); delta=grad(L_target_aug)-grad(L_target_real)`。

target-clean项可复用full-real forward的逐项BCE计算图，不能再除目标域行数。A1使用源同语义标量 `L_real + λ*(L_target_aug-L_target_real)` 交给标准Adam。A2/A3用同一θ、同原batch、同view取得两条分源梯度；在所有梯度计算完成前不更新θ。允许一次full-real forward与一次target-aug forward、多个必要autograd.grad；如用梯度线性组合，必须通过同图scalar梯度一致检查。记录实际autograd.grad调用数，不能把多次梯度遍历计成一次backward。

对每个参数p，令k为其独立step计数；活动参数的分源更新为：

```
r = g_real + wd*theta_pre
m_real = .9*m_real + .1*r
q_delta = beta_delta*q_delta + (1-beta_delta)*delta
h = r + lambda_aug*delta
v = .999*v + .001*h*h
k += 1
mhat = m_real/(1-.9**k)
qhat = q_delta/(1-beta_delta**k)
vhat = v/(1-.999**k)
theta -= .01*(mhat + lambda_aug*qhat)/(sqrt(vhat)+1e-8)
```

状态初始化为0，全部detach/no_grad，不保留跨step图。decay只进入真实通道与混合v一次，δ不含decay。βδ=.9时 `m_real+λ*q_delta` 对应标准Adam exp_avg；v和step相同。βδ=0或use_delta_momentum=false直接走标准混合梯度Adam；λ=0或增强关闭直接走标准真实Adam，无view/noise消耗。正式配置冻结不在轨迹中动态切换优化器类型。

None语义以梯度图连接性为准：target两项均None时δ为None；仅target一侧None时该侧按零作差。full-real和δ都None的参数所有状态/step均跳过；至少一条非None则缺的分量按零，coupled decay正常加入、共享同一step_p。显式零tensor仍为活动梯度，不应和None混淆；抵消后数值为零但图连接存在仍更新。λ0分支不能让只有增强图连接的参数被激活。目标域缺失但真实梯度存在时δ=0，已有q按βδ衰减；若真实与增强图都未连接，则所有状态不动。

至少显式配置 `use_shadow_ema/use_augmented_view/use_delta_momentum/use_reliability_gain/beta_delta/rho_shadow/lambda_aug/noise_scale`。P2未触发时use_reliability_gain=true报未启用；其合法语义必须依赖K=2，不静默退化。

## 5. P0、选点、RAM快照与统一test

shadow通过deepcopy在线模型初始化，不调用消耗RNG的新随机初始化；参数requires_grad=false。φ0=θ0，每次真实参数更新后执行 `φ=.995φ+.005θ`，所有Parameter（含LayerNorm affine）参与，buffer复制在线值。无任何shadow反馈在线步骤。关闭P0不得分配shadow/额外validation。

每epoch分别对online和shadow完整真实validation；各自六格macro AUC严格`>`更新selected，平局最早，E0排除。保存两套独立selected epoch/validation/step/model hash，不让shadow覆盖online选点。online training loss为更新前真实batch预测，标明training-online。

每run RAM保留：online-selected、shadow-selected、共同E10和E50（重合epoch去重）的online/shadow模型状态；必要Adam/q/step、RNG/loader/shadow_count状态通过同一完整snapshot接口取得。selected按模型各自保存对应epoch。被替代且不属于固定锚点的状态释放。恢复快照包含online、shadow、optimizer m/q/v/step、epoch/global_step、shadow_count、Python/NumPy/Torch CPU及所用CUDA RNG、loader generators、数据与row-order身份。Stage0校验epoch边界恢复后下一步与不中断一致。

Stage1全部40run完成后，先仅用validation与D1冻结P2进入判断和配置选择；写 `phase1_validation_decision.json`。任何P2选择不得读取Stage1 test。若P2触发，Stage1 RAM快照在P2执行/预算解决期间保持；P2代码需独立设计补充、实现与review通过才训练。

Stage1与实际获准的P2训练/选择均完成后，写全体 `selection_frozen.json`（有效run清单、两种selected、E10/E50、epoch/step/hash、validation值、源身份），才进入单次统一test阶段。若P2未触发，则Stage1冻结后直接统一test。统一test逐模型/epoch去重评估，但selected/E10/E50各语义记录完整；不按test重选ρ/β/λ/seed或追加P2。test结束并生成数值表后释放所有RAM状态。

磁盘checkpoint身份为 `{dataset,arm,seed,attempt,model_kind,checkpoint,epoch,global_step,model_hash,optimizer_hash,shadow_updates}`，不包含权重。冷启动不具备磁盘恢复能力；状态丢失则重跑受影响完整run并保留失败attempt。

## 6. D1/D2、预冻结门与P2触发规则

只做合同指定两类诊断，不追加新probe训练。D1固定在每epoch最后一个真实batch的实际更新中计算必要范数/点积，每行标 `measurement=last_actual_batch`：||qδ||、||m_real||、λ||qhat||/(||mhat_real||+1e-24)、cos(qhat,当前δ)、与同状态标准Adam局部对照更新的cosine/相对L2差。局部标准Adam方向由 `m_mix_next=.9*(m_real_prev+λ*q_prev)+.1*(g_real+λδ+wdθ)` 和相同v/step构造，不执行第二次optimizer.step，不创建持久新v；它是同状态映射的一步对照，不代表A1完整训练轨迹。零范数cosine记录null和原因，不伪造0。阶段统计来自这些预定batch观测，不称全batch总体均值。

D2按epoch及各自selected记录online/shadow AUC、LL、选点epoch差；所有臂都维护shadow，P0主对照仅A0 shadow-selected减A0 online-selected，A1–A3组合另列。D1只解释更新，不能取代selected评价。

seed级效应mean/sample SD、全部五个差值、df4配对t95%区间；主继续门 `mean Δselected macro AUC >= .001 且 >=4/5 seed >0`，validation与test分列。P1必须同时超过A0和A1；只超过A0不能归因差分记忆。A2作为等价对照，A3为主要非等价检验；不事后挑有利epoch。只E50改善不晋升selected方法。

保护门沿GA31-R3冻结：最大训练域（ML d2/KR d1）的任一任务AUC、pooled任务平均AUC、worst-cell AUC，若配对mean<=-.002且至少4/5下降；macro LL若mean>=+.002且至少4/5恶化。任一触发记TRADEOFF，保留实际宏收益和具体代价。P0对照同轨迹online，P1分别对A0/A1；不使用测试保护项来决定P2。

P2只使用A3相对A1的validation和现有D1，进入定义如下，逐数据集执行：

1. **条件正向但不稳定**：A3−A1 selected validation mean>0、至少3/5正向，但未通过主继续门；且A3的50个epoch末batch注入比均值在五seed间的变异系数>=.5，或在预定阶段E1–10/E11–30/E31–50三个阶段观测均值的最大/最小比>=2且至少4/5seed成立。分母均值<=1e-12不算有效证据。
2. **重复阶段失配**：A3相对A1未通过主继续门，但D1显示E1–10与E31–50的epoch末batch平均cos(qhat,δ)由正变负（至少一段均值>=.1、另一段<=−.1），且对应两阶段平均注入比都>=.05，在至少4/5seed重复。该门识别显著且重复的滞后失配，不因“轨迹有所改变”就触发。

两门均不通过则P2 `NOT_TRIGGERED`，不实现、不训练；这不是P2阴性结果。门1/2阈值为本轮对“正向趋势/明显阶段变化”的事前操作定义，写入resolved config。若A3只是稳定负向且没有上述诊断，不触发。P0是否通过不自动触发P2。已达到稳定实用收益的A3且没有上述失配时不额外加入P2。

触发时，第二阶段必须先冻结预算冲突的解答、K=2匹配对照、chosen beta、RDI r/n/gain公式与eps能量尺度，再实现。预期公式为03所列两视图差分均值/V/S、τ=.9标量EMA、a∈[0,1]、λ=.5；S负值原样记录，不作为“可靠性有效”的证据。当前设计不授权提前编码RDI，不运行额外ρ/β/κ/λ网格。

## 7. 最小验收（Stage0）

Python仅 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。先CPU FP32/FP64小张量，再真实模型CUDA检查。开发seed固定2022；每数据集主开发轨迹最多1epoch，不访问test指标，所有分支forward/autograd/step开销单独记账。先语法/必要单测，reviewer独立检查通过后runner执行真实Stage0；失败只修具体实现。

1. A0同初始化同真实batch与GA31-R3对应L2-ML/B-KR预测、概率BCE、参数梯度及实际Adam一步匹配；参数量/state_dict/field_slices/data身份一致。真实batch40960显存检查，无额外性能调参。
2. 小张量至少连续2步验证λ0与标准真实Adam，βδ=.9与同混合梯度标准Adam，βδ0标准分支与增强Adam；覆盖coupled decay、bias correction、共享per-param step。FP64 atol1e-12/rtol1e-10，FP32 atol1e-6/rtol1e-5；记录实际最大误差，不能扩大容差掩盖错误。
3. target差分验证 `g_real+λδ` 与源同view标量梯度匹配，原B×T分母与非目标域贡献保持；缺目标/关闭/λ0/κ0不抽view。None两侧缺失、仅一侧连接、显式零、抵消零和不同参数活动历史用小张量验收。
4. 同seed同序列短轨迹比较shadow on/off：online参数、Adam/RNG/loader顺序完全相同；shadow参数按手工EMA、buffer复制、计数一致。shadow deepcopy不消耗RNG，eval恢复training标志。
5. RAM保存/恢复（含q/shadow/optimizer/step/global RNG/loader）后下一步与不中断状态一致；view由同seed/step重建，key/hash一致。不得写torch模型文件。
6. 选点小型assert：online/shadow分别取严格最大、平局最早、E0不候选、E10/E50/selected去重；完整有效run清单冻结前不得访问test。恢复计数使用真实selected step。

只保留能改变本轮公式、更新或结论的检查。Critical修复后重跑影响run；Relevant最小核验；Irrelevant只记录，不扩展矩阵。

## 8. 运行、成本、工件与交付

入口建议 `python -m experiments.gradient_aug_momentum_20261101.run --mode audit|check|all --run-dir <new-attempt> --devices 0,1`。all保持一个RAM coordinator生命周期；协调者父进程不初始化CUDA，独立worker接管设备。来源和配置冻结后记录代码commit及每文件hash；未有git HEAD时如实记录，不冒称已冻结commit。

逐卡读取显存/功耗，两者各自均低于85%且剩余显存容纳原batch时增加worker；无需两张卡同时满足。启动后等首次真实batch分配回执，再读资源继续并发。外部作业不停止。可复用旧调度器，真实OOM先降低并发并以原batch重试；不缩batch/删seed/改epoch。失败attempt和真实进度保留。

记录训练forward、autograd/backward实际遍历、optimizer.step、额外目标视图曝光、online/shadow validation forward、训练/评价worker秒、总walltime、峰值显存、RAM快照字节。40run并行时worker总秒与elapsed分列，不能凭共享GPU计时归因效率。shadow不是额外训练run。

至少交付 README/IMPLEMENTATION_REPORT/REVIEW_REPORT/REPORT、resolved_config/source_manifest/data_manifest、run_ledger/launch_ledger/failure_ledger、epoch_metrics/per_cell_metrics/selected_metrics/paired_effects、optimizer_state_diagnostics/shadow_metrics/decision.json/costs.csv、formula_checks/checkpoint_restore_checks。逐run包含超参、init/order/view hash、selected/E10/E50 online/shadow身份、全部域任务AUC/LL与训练loss；数据身份可聚合引用逐数据manifest，不写大预测矩阵。

所有六格AUC/LL、域均值、任务均值、macro、worst、traffic-weighted cell、逐任务跨域pooled AUC均保留。报告由协调者独立分析是否有效、seed/阶段稳定性、矛盾、STRONG/MEDIUM/WEAK证据、域规模与难度/基线区别、大小域交换、竞争机制和唯一最有区分力的后续实验；不预设课程/动量有效、小域困难或M3OE必失败。无触发不追加算法；完成规定阶段、统一test和报告后停止并按用户授权发布新分支。
