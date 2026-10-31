# GA31-v2：设计与执行规格

设计日期：2026-10-31。任务入口为 `research_plans/26-10-31/03_首轮实验与Codex实施规范.md`，冻结来源为 `540171f865d3f80b7b46abdba69d0939cab49d80`；规划包取自 `main@60c45e60e6b9aa35eccaa2bd3d0898b72ac72acc`。本文在任何本轮实现和正式训练之前冻结。

## 1. 本轮问题和覆盖规则

研究问题：自然 Sample 人口权重下，NR-Gauss-v1 是否在相同损失尺度上改善 validation-selected 的真实 test AUC/LogLoss？四臂分别识别增强效应、缩放效应和交互。只读单步探针解释给定参数与 Adam 历史下的差异，不据其调参。

最新版 `start.md` 对旧规划有两条明确覆盖，执行配置必须写入这些修订：

| 项目 | 本轮冻结 |
| --- | --- |
| 种子 | 2022、2023、2024、2025、2026；两个数据集 × 四臂 × 五种子，共 40 次正式训练 |
| 状态存储 | 所有 model/optimizer/RNG/loader 快照只存 RAM；禁止写 `.pt`、`.pth`、pickle 模型、临时模型文件或共享文件形式的训练状态 |
| 置信区间 | 五种子配对 t 95% 区间，df=4；样本标准差使用 ddof=1 |
| 方向计数门槛 | 原 3/4 的 75% 要求统一移植为 `ceil(0.75×5)=4/5`，适用于收益、很小效果及保护规则 |

原 2022–2025 四种子结果另作敏感性表，使用原 3/4、df=3 规则；不增加训练，不替代五种子主分析。效应量门槛和其它冻结超参不变。所有修订在看本轮效果前记录到 `resolved_protocol.json`；旧规划文件保留原样。

自然数据要求解释为不改变原始数据人口、标签、比例、划分或采样；本轮唯一增强是执行合同明确要求的模型内 Embedding 视图。不得另造样本人口或更换 q。

## 2. 来源与影响范围

只新增 `experiments/gradient_aug_20261031/`、本轮 `research_reports/2026-10-31/`、本轮 `artifacts/gradient_aug_20261031/GA31-v2/attempt_*` 下的代码、数据记录和日志。归档模型、历史报告、其它实验入口不改。协调者负责恢复冻结来源并最终提交；coder 不执行研究分析。

新包的职责可以合并到较少文件，必须覆盖以下接口：

| 文件/模块 | 必须实现的职责 |
| --- | --- |
| `model.py` 与命名隔离的必要 basic 依赖 | fixed1 模型、feature slice 映射、Embedding 后扰动入口；关闭时参数键/顺序不变 |
| `data.py` 与隔离的审计数据依赖 | 冻结 natural 预处理、实际路径、数据身份、split row_id 与 schema |
| `objectives.py` | Sample、归一化视图替换、独立噪声随机流；不创建额外 optimizer |
| `protocol.py` | 解析并校验本轮覆盖配置、seed、四臂、单 Adam、快照/RNG 与日志工具 |
| `run.py` | audit/check/dev/train/evaluate；一个长期存活的批次协调入口；RAM 状态转移与 test barrier |
| `probe.py` | 独立 CPU 快照输入、离线只读探针、实际 Adam 单步反事实 |
| `analyze.py` | 只处理已写出的数值工件，产生配对表、决策与报告素材 |
| `tests/` | 第 7 节必要验收；不扩展无关测试 |

R=`history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021`；M=`history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/data_generate/MDMTRec`。采用 M 的 `models/multi_domain/fixed.py`、`basic/features.py`、`basic/layers.py`，以及它们实际需要的 activation/initializers/utils 辅助；采用 R 的 `audited_data.py` 自然加载语义。依赖须显式包导入，不能借助顶层 `data`、`protocol` 模块名碰运气。

不运行 R 的旧 trainer，也不继承其 legacy_group_sum、patience10、原 O→B 初始化迁移或直接访问 test 的行为。新 fixed1 CPU 模型先由 `seed_all(seed)` 构造，再把同一初始快照严格加载到四臂；feature 使用源模型的整数列位置命名以保持 state_dict 键一致。

复用已有数据审计和状态身份函数完成合同要求，不新增可选哈希机制、额外审计层或依赖。来源以服务器返回的 commit/blob 元数据记录；数据的 raw/六数组/split/encoder/scaler 身份按原 JSON 验收。路径可改，身份不可改。探针和状态比较优先直接逐张量/数组相等检查；必要 row-order 身份复用已有 `array_hash`/`row_ids_hash`。

## 3. 数据、模型、目标和优化器

数据路径固定到现有 CSV：

- MovieLens：`/home_new/xzf/datasets/ml-1m/ml-1m.csv`；train/validation/test=800167/100021/100021；训练域计数=155949/318599/325619；任务为 rating>3、rating>4；增强 d0。
- KuaiRand：`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`；train/validation/test=2006132/250767/250767；训练域计数=172782/1743898/89452；任务为 stored is_click、long_view；增强 d2。

使用源 natural、split_seed2022、全局源行序80/10/10、仅 train fit vocab/scaler、OOV0 的预处理。KR 先全局切分再映射 tab0/1/4；ML 禁止旧角色重采样。身份不符先定位路径与预处理，未解决不能把替代数据算作正式结果。

模型为 D=3、T=2、expert_num=4、fcn_dims=[512,256,256,64]、sparse_embed_dim16 的 MDMTRec；exp_d/exp_t/bal_d/bal_t 均为 Python 标量1。保持 STAR、LayerNorm、gate 的 `emb.detach()`、expert/tower 连接、sigmoid 概率输出 [B,2]。无 learned fusion/DARTS/新可训练参数；确认无启用的 BatchNorm/Dropout。

原 batch 大小 B、任务数 T=2，`ell_r=BCE_probability(pred,y,reduction='none')`；`L_r=ell_r.mean(1).mean()`。增强目标掩码 a=1[d=d*]，λ=0.5：

`L_view = (sum(ell_r) + λ*(sum(ell_aug_target)-sum(ell_real_target)))/(B*T)`。

每个原样本视图系数和为1。P0=L_r；P1=0.1 L_r；P2=L_view；P3=0.1 L_view。默认完整真实 batch 和目标子 batch 两次 forward，合成一个 scalar 后一次 backward/Adam.step；没有目标样本时跳过增强 forward；增强关闭或λ=0走真实路径。不得通过 cat 后无权 mean 改尺度。

Embedding 输出按所有 sparse 原特征顺序拼接，再拼 dense。按实际 feature 类与 embed_dim 构造 field slice，断言总宽度。每个目标样本每可扰动 field 的16维向量 e 使用 `e + 0.1*detach(sqrt(mean(e**2)))*xi`；RMS无下限；e 本身保留梯度。ML 可扰动6 sparse字段，KR为JSON展开的33字段；domain Embedding、ML age dense、输入ID、标签、row_id 不变。固定 field顺序和目标样本在batch内顺序；每field产生目标子batch形状的iid高斯。

正式噪声使用独立设备 `torch.Generator`，每步重置为 `seed*1000000+global_step`；global_step从1开始；P2/P3的噪声种子配对。禁止消耗模型/loader全局随机流。

一个 `torch.optim.Adam`：lr=.01、betas=(.9,.999)、eps=1e-8、weight_decay=1e-5、amsgrad=false、foreach=false、fused=false。c只乘数据目标，耦合L2由Adam内部加入；不缩放decay，不把grad=None改成0。FP32、TF32=false（matmul和cudnn）、cudnn.benchmark=false；无AMP、scheduler、裁剪、累积或参数EMA。

## 4. 训练与 RAM 生命周期

batch40960、完整50epoch、不早停、drop_last=false、num_workers0。每epoch自然无放回打乱全部原样本。CPU loader generator独立：train=seed+11000、validation=seed+33000、test=seed+44000。每个epoch记录原始row_id顺序身份，配对臂完全相同；记录实际曝光与step。ML每epoch20步/每run1000步；KR49/2450。开发最多8个seed2022短run、每个至多1epoch，无test、无性能调参；正式重新初始化。

每epoch完整真实validation，macro六格AUC严格`>`选模，平局保留最早，E0排除。真实train曲线至少保存在线原始输入、未缩放BCE及其六格累计值，并明确为training-online；若报告train AUC同样标明在线预测或checkpoint eval，不能混称。validation每epoch记录完整指标。增加真实train固定checkpoint evaluation可以直接满足曲线口径，但不作为新的比较实验。

状态仅RAM，不通过文件恢复：

1. 一个长期存活的协调进程管理40个run。GPU子进程执行训练，epoch边界CPU深拷贝 model/Adam per-param m/v/step/Python、NumPy、CPU/CUDA RNG/loader generator/epoch/global_step/next permutation边界；快照不引用继续训练的可变tensor。
2. 保留E1/E5/E10/E20/E50和selected；相同epoch去重，latest可作为E50别名。selected被后来更优epoch替换且不属于固定锚点时释放旧对象。每个run完成后离线执行E1/E10/selected探针。
3. 子进程向协调进程通过Pipe `send_bytes`传递 `io.BytesIO` 中的状态序列化字节；或者采用等价的纯内存IPC。不得触发torch multiprocessing Tensor共享文件、临时checkpoint或磁盘swap式自建状态存储。进程结束前确认协调进程已收齐状态。
4. 协调进程保留所有完整CPU快照至全体40run选点与c*冻结、test与探针完成，再释放。测试可启动GPU子进程，接收RAM快照并返回纯指标。`evaluate`阶段不从磁盘加载模型；独立冷启动evaluate若没有RAM快照必须报告缺少状态。
5. `selection_frozen.json`只存dataset/arm/seed/attempt/epoch/global_step/selection值及工作尺度、来源身份、修订；不含权重或optimizer张量。40run均有已冻结selected后才能开始test，统一评估selected/E10/E50，重合复用。
6. 若进程/服务器重启丢失快照，从原seed重新完整训练受影响run；不声称支持磁盘续跑。开发只验证epoch边界RAM恢复后的下一步，不声称任意batch断点恢复。

容量实测：2026-10-31使用本地既有fixed模型和历史schema仅构造CPU模型，ML=773470参数、state_dict=3093880字节；KR=1614254参数、6457016字节。完整Adam快照主体约3倍模型宽度，分别9.28MB/19.37MB。20个ML run与20个KR run、每run最多6个去重快照，共约3.44GB（3.20GiB），加上初始state和RNG仍远低于本机当时49GiB可用RAM。coder须据实际隔离模型核对参数量和快照实际tensor字节；不依赖磁盘落盘省RAM。

按每GPU独立的显存/功耗占比控制并发：两者均低于85%时尝试启动下一个任务，不等待两卡同时满足；记录启动时显存、功耗、worker数。每次启动后等待该进程完成CUDA分配，再看新读数；不能一次读数批量超发。也须保留数据预处理/快照所需的实际CPU RAM。现有GPU0其它进程不可停止。原batch真实OOM先查并发与实际占用，减少并发/排队重试；不减batch、不改预算。若单独运行仍真实OOM，则写具体失败ledger并交付阻塞。

## 5. 只读探针

每run训练结束后，在不再训练的快照副本上进行E1/E10/selected，去重。每锚点2个固定训练batch，每个最多1024行，来自seed314159的训练row_id permutation前两个batch；验证probe最多4096行，来自seed271828固定permutation；所有run使用同一组数据行。缺目标域标unavailable，不重采成均衡batch。噪声seed=`seed*1000000+100000+2*anchor_global_step+probe_batch_index`（index从1开始）。

同一原batch与B*T分母得到raw `g_r`、扩展视图`g_a^B`（非目标保持真实）、`δg=g_a^B-g_r`、`g_mix=(1-λ)g_r+λg_a^B`。另算目标子集真实/增强梯度，分母同样B*T。raw不含c和decay。不能将目标子集均值与完整batch均值直接混合。

结构分组：shared=`embedding.*`、`shared_weight`、`shared_bias`、`star_mlp.*`、`skip_conn.*`、`expert.*`；另外slot_weight/slot_bias、domain_expert、task_expert、gate、tower；all为全部。报告各向量norm、real/aug cosine与norm ratio、δg及其相对g_r正交分量；目标子集同报。零norm cosine为null并记录原因。

real/extended_aug/mix三个反事实分支都从同一源 `(model,optimizer)` 联合deepcopy，保留param→state绑定。分别将c乘对应raw梯度，按None原义写grad，然后各执行一次真实Adam.step。比较更新norm/cosine、m/v变化、固定真实validation probe BCE前后差。可以只保留单个分支副本并顺次释放，但每次都从同一源重建。None仅在统计向量运算中视零，不能改变实际optimizer输入。

保存/恢复所有全局RNG与train/eval模式；probe不使用正式loader iterator或正式noise generator。源快照model/optimizer/所有RNG/loader状态在probe前后直接比较相等，借此完成只读验收。记录锚点身份、行身份、分支、额外forward/backward和耗时。每数据集最多5×4×3×2=120个probe batch，不多步rollout、不训练selector、不改λ/κ/选点。

## 6. 指标、配对分析与决策

每dataset/seed/selected、E10、E50，报告六格AUC/未缩放真实LogLoss、六格macro、每域两任务均值、每任务三域均值、worst-cell AUC；pooled逐任务跨域AUC再任务等权平均；流量加权cell AUC/LogLoss使用被评估split的域份额、任务等权；同时给真实总体LogLoss。不同任务标签不展平算AUC。指标名和聚合公式固定。

以seed为独立单位，对所有规定指标计算A1=P2−P0、A0.1=P3−P1、S0=P1−P0、S1=P3−P2、I=A0.1−A1；逐seed值、mean、sample std、方向数、配对t95%区间均保留。AUC正向有利、LogLoss负向有利。I接近0不能解释为“增强收益由缩放导致”。

每dataset用P0/P1五seed自身selected validation macro AUC均值选c*，平局取1，test前冻结。五seed主决策使用：

- c*下mean增强ΔAUC≥.001且≥4/5正向：validation“值得继续”；同对照test也通过为“正式首轮支持”。另尺度单列，不能据test更换c*。
- 另一尺度通过而c*没通过：尺度依赖候选；两尺度均没通过则结束本轮。
- 两尺度mean绝对差均<.0005，且各≥4/5 seed绝对差<.001：当前预算效果很小。
- 相对同尺度无增强臂，最大训练域任一任务、pooled task-mean、worst-cell AUC出现mean≤−.002且≥4/5下降，或macro LogLoss mean≥+.002且≥4/5恶化：TRADEOFF，保留宏收益及明确代价。
- 两dataset通过才称跨数据集支持；一个通过保留条件性结果。原四seed敏感性表单独按旧规则输出。

报告须区分selected与late/final现象，AUC与LogLoss，域大小与基线难度；分析稳定现象、矛盾、STRONG/MEDIUM/WEAK证据、竞争解释、大小域交换及最能区分解释的下一实验。不得预设课程、动量、困难域假设成立。下一实验只建议，不自动运行SAGU/SAOS/SGVS、EMA或额外decay/epsilon臂。

## 7. 审查和必要验收

coder提交实施报告，列实际修改文件、来源、接口、运行命令、测试结果、未解决阻塞；不能分析科研结果。独立reviewer同时读本文、实际实现、实施报告。只有PASS，或PASS WITH FIXES且所有blocking项修好后，才正式RUN。

必要验收只覆盖会改变当前比较的路径：

1. 同state/batch对冻结fixed1参考比较预测、Sample loss、梯度与一次Adam更新；参考loss使用Sample均值，不能拿旧group-sum当oracle。
2. κ=0下P2/P0、P3/P1的loss、grad与Adam一步一致；λ=0/关闭增强真实路径；CPU FP32 atol1e-6、rtol1e-5，CUDA超出先定位。
3. 不均衡batch、batch=1、缺域、无目标域覆盖；手算B*T分母、原样本权重和1。
4. domain/dense逐位不变；field顺序和slice正确；标签/row_id/参数键、顺序、数量不变。
5. 四臂init张量相等、epoch原row_id顺序一致；P2/P3相同步噪声seed相同；增强generator不改变global/loader RNG。
6. 实际Adam保留None语义；probe各分支param-state绑定正确且源状态/RNG/loader完全不变。
7. RAM epoch边界snapshot恢复后的下一batch/噪声/grad/Adam状态与未中断路径一致；不创建模型文件。
8. train不调用test；strict `>`选点、E0排除、50epoch完成、step/曝光匹配；全体冻结后统一test。
9. 冻结模型、数据身份与软件/GPU设置可核对；新包不能误导入根目录同名模块。正式前一批原batch40960确认单进程可容纳，不据性能选参数。

有具体Critical问题则修复受影响路径并重跑受影响run；Relevant只做最小验证；无关工程问题不扩展测试矩阵。没有新增依赖或额外防御层。

## 8. 工件和完成条件

日志/数据：resolved配置、source/data manifest、split_row_ids.npz、epoch/per-cell/batch/augmentation/optimizer/probe/runtime记录、stdout/stderr、失败ledger、selection_frozen、40run ledger、paired_effects、DECISIONS.json、REPORT.md。模型状态身份使用dataset/arm/seed/attempt/epoch/global_step/来源和必要一致性凭据；磁盘工件不含权重、m/v大张量或RNG训练快照。机制原始大向量不写盘，只写固定锚点各块汇总。

完成定义：40/40正式run各50epoch（或合同规定的真实数据身份/单卡OOM/不可修复非有限阻塞已具体记录）、统一冻结与测试、全部规定探针、完整seed/域/任务/阶段/成本分析、独立审查记录、代码与报告按协调者确认的分支规则推送。无阻塞时不提前停止。推送以实际来源分支规则为准，禁止覆盖历史结果。
