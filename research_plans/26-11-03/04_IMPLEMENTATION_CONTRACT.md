# 04 可实现接口、源代码与执行合同

## 1. 固定来源与新目录

总研究上下文固定 main `37065a52f083ddb196efa3dacccb5f7c793edfb6`；执行底座固定 v3 `a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288` 中 `experiments/conditional_information_momentum_20261102_v3/`，其 vendor/ema02 来自 `38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27`。本轮新建 `experiments/mdmt_history_information_20261103/`，报告新建 `research_reports/2026-11-03_mdmt_history_information/`。不修改旧包、原报告或旧原始指标。

Codex从固定提交读取/提取实际依赖，不合并整个研究分支；把复用文件以原字节放入新包vendor，记录路径、源commit、Git blob、SHA256、字节数。只复制实际导入的模型、features/layers、data、源目标、protocol、metrics、RAM guardian等依赖；不能假定main已经有v3包。新适配器不得隐式读取当前根目录旧计划。

现有 `PairOptimizer` 是参数分组/状态组织的参考，不直接作为新算法类。禁用其PAIR、D/q、ECCM、CV、PCOLL路径。源A0/A1调用必须保留原数学与随机流，不能改名后换噪声。

## 2. 不变训练协议

| 项目 | 冻结设置 |
|---|---|
| 数据 | MovieLens / KuaiRand原行序80/10/10；KR先切分再保留tab0/1/4；train-only编码和scaler，OOV0 |
| 训练域计数 | ML 155949/318599/325619；KR 172782/1743898/89452，以固定manifest逐项核对 |
| 标签 | ML rating>3、rating>4；KR is_click、long_view；保留真实联合标签支持 |
| 模型 | fixed1 MDMTRec/M3oE；3域2任务，embedding16，共享专家4，FCN [512,256,256,64] |
| 训练 | batch40960、50epoch、无早停、尾批实际B、真实BCE sum/(2B) |
| 优化 | Adam默认lr=.01、betas(.9,.999)、eps1e-8；coupled wd ML1e-4/KR1e-5；仅05的A0公平搜索允许指定lr/wd变化 |
| 数值 | FP32；AMP/TF32关闭；foreach/fused/amsgrad=False |
| 参数EMA | ML rho=.9828205985452511、KR .995；theta0初始化、buffer复制、无反馈，不搜索rho |
| 随机 | 源setup(seed)与loader generator seed+11000；增强/probe用03独立key |
| 估计器 | online、fixed_ema；两者各自validation选点，主决策仅fixed_ema selected |

CSV：`/home_new/xzf/datasets/ml-1m/ml-1m.csv`，`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。Python：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。这些是已读执行记录提供的路径，Codex启动时检查存在及hash，不能把本次规划写成已在该服务器再次核验。

特征来自源bundle schema，不增添CSV中没有的用户/视频ID。源x为[F,B]、y为[B,2]；域为x[-1]。先在S0断言实际布局，不自行转置后假定仍与原loader一致。嵌入字段位置来自field_slices，不能按输入列顺序硬编码。

## 3. 必须实现的接口

```python
@dataclass(frozen=True)
class StepContext:
    dataset: str
    seed: int
    epoch: int
    global_step: int  # 从1开始

@dataclass
class MixBatch:
    anchor_idx: Tensor       # 当前batch的列索引，CPU long
    donor_idx: Tensor
    target_task: int
    domain: int
    support_level: Tensor    # 1=另一任务对照，2=同目标回退，3=self
    tau: float
    index_hash: str

build_mix_batches(x, y, row_ids, context, mode, tau, max_anchors=256) -> list[MixBatch]
forward_mixed(model, x, mix_batch, functional_state=None) -> Tensor  # [M,2]
real_loss(model, x, y, functional_state=None) -> Tensor
mixed_objective(model, x, y, batches, lambda_mix, functional_state=None) -> Tensor
shared_parameter_manifest(model) -> list[dict]
collect_history_probes(model, x, y, context, batches, mode) -> list[tuple[GradList,GradList]]
project_history(grad_m, preconditioner, probes, kappa, gamma=.2) -> (GradList, dict)
transport_correction(current_grads, old_point_grads, kappa, parameter_step) -> GradList
HistoryOptimizer.step(current_data_grads, probes=None, old_point_data_grads=None) -> dict
train_step(model, optimizer, ema, previous_state, batch, config, context) -> dict
state_dict_ram(...) -> dict
load_state_dict_ram(state) -> None
```

接口名可以在实现映射中作一一对应的工程调整，但不得把关键数学交给实现者自行选择。目标、共享集合、donor回退、投影度量、运输质量因子、裁剪、二阶矩和更新顺序以03为准。

`GradList`按冻结的S manifest排序，保留shape；不能因某个梯度None而改变顺序。H probe只求S梯度，其他参数的当前梯度保持不动。M/TM监督型插值需要两端embedding梯度。多次autograd不会增加optimizer更新次数；不能在probe后调用额外Adam step。

## 4. 新旧参数求值的唯一语义

T使用 `torch.func.functional_call(model, (old_parameters, old_buffers), (x,), strict=True)` 或语义完全相同且通过测试的适配器。必须替换全部参数与buffer，不仅S。旧参数以detach克隆保存，求导前仅S设requires_grad=True；无原地覆盖在线模型，无EMA参与。若使用嵌入hook，则hook必须覆盖该次functional_call并在finally移除，不能引用当前embedding权重代替旧点输出。

官方接口依据：https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html 。以服务器安装的PyTorch版本验证此接口；功能缺失只做等价工程适配，不升级环境后偷偷改变训练数值语义。源模型没有BatchNorm/Dropout，仍断言这一点。

每步开始时的θ_k才是下一步需要的previous_state。为避免引用被optimizer原地改变，更新前clone；旧点梯度计算完成后释放更旧快照。只保留一份前一步完整状态，不创建第二套optimizer。状态切换测试必须验证：参数完全相同则delta为零；只改变一个非S参数也能影响旧点求值，而不是被漏替换。

## 5. 状态与随机流

RAM state包括：online/EMA全部参数和buffer，普通Adam与S的m/v/step，T的previous_state，完整config/source/data/parameter manifest，loader generator与global RNG，epoch/global_step，最早最佳validation分数与epoch，online/EMA的selected/E10/E50状态，所有计数。H没有额外长期条件状态，probe/normals只活当前步。

禁止把模型、optimizer、梯度或参数数组写入磁盘；不生成.pt/.pth/.ckpt或变相参数JSON。允许配置、标量CSV/JSON、哈希、运行日志与源代码。源guardian/RAM协调器可复用；不新造持久化服务。恢复测试用内存BytesIO/cpu_copy，不以硬盘checkpoint规避限制。

RAM丢失：记录物理失败/重放，不能声称恢复成功。每个科学身份最多一次有原因的基础设施重放；重复数值故障停止该配置。禁止为一个不利结果换seed或重跑后挑最好。

## 6. 最小正确性验收，不重做历史机制研究

本次规划自带design_reference.py，13项CPU代数检查已PASS；它仅验证小向量投影与运输公式，不验证真实模型接线、性能或GPU成本。Codex仍需以下针对性测试：

1. **旁路一致性：** 同初始化/真实batch，A0、H κ=0、T0 κ=0、M λ=0或τ=0，在loss、gradient、一步Adam、EMA、恢复后下一步达到FP32 atol1e-6/rtol1e-5。TM κ=0匹配M。H τ=0不等于关闭历史，不做错误的A0断言。
2. **数据/标签/特征：** 原manifest完全匹配；同域、同目标标签、优先另一标签不同、回退和self都可构造；domain字段不变；ML不生成01；不使用validation/test建donor池；anchor自然抽样，无正负均衡。断言λ/τ零分支不消费增强随机流。
3. **梯度路径：** 监督型M两端embedding均有应有梯度；H probe不改变当前`.grad`；原gate detach保持；同一模块无残留hook，参数无重复step。
4. **历史数学：** S集合完整互斥；coupled wd只一次；None/显式零时钟；H Gram/KKT与CPU参考一致；T新旧点相同delta0、同样本/索引一致、保存的是更新前参数；运输归一化因子与03手算一致。
5. **控制语义：** H_NORM匹配本状态的D历史范数；T_NORM匹配本状态的D moment范数；C_MIX不修改旧moment；所有对照使用相同probe索引定义。不能宣称不同训练轨迹间的逐步更新已完全匹配。
6. **恢复与评价：** 两步运行与RAM恢复后的第二步一致；validation最早严格最大选点；selected/E10/E50身份hash；test入口在冻结前拒绝执行。
7. **真实S0：** 每数据一个完整40960 batch，A0/M/H_MIX/TM各预热2步、测量3步，共每数据20步。复用同batch仅是成本/接线smoke，不作科学训练或泛化证据。记录完整forward/backward/probe/old-point/optimizer/EMA计数、同步训练步耗时和峰值显存。

只对实际改变接线的修复重做受影响项，不重新验证EMA有效性、loss尺度现象或小域难度。实现审查必须由未写该实现的reviewer完成；无独立reviewer可用则明确记录，不伪造审查身份，并停止正式GPU队列在IMPLEMENTED_WAITING_REVIEW。

## 7. 调度、成本与选择

GPU使用沿用逐卡检查显存/功耗<85%并结合实测峰值的准入，不要求两卡同步满足，不杀外部任务。OOM先减少并发，不能缩batch改变协议。S0若某算法同步训练步中位耗时>A0的4倍，标COST_BLOCKED：当前预算下不进入正式队列，转其简单近邻；不称性能阴性。显存超出空闲能力只调整并发或记录资源阻塞，不改变算法。

成本包括数据加载、全部probe和旧点计算、validation、最终test、失败与重放。记录worker wall time与GPU训练步时间，不把并发worker时长相加称独占GPU耗时。不做精确未来耗时保证。

每epoch完整validation；按fixed_ema macro六格AUC作主选点，严格大于才替换，平局保留最早epoch，E0不选。online独立选点。训练不中途早停；阶段是否继续只在50epoch完整轨迹之后判断。所有数据集本轮开发及确认配置、选点都冻结后才运行test；不能每epoch提前test再封存。

## 8. 统一CLI与交付字段

实现下列入口（本次规划未声称这些命令已存在）：

```bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
$PY -m experiments.mdmt_history_information_20261103.run prepare --plan research_plans/26-11-03/experiment_queue.yaml
$PY -m experiments.mdmt_history_information_20261103.run check --cpu
$PY -m experiments.mdmt_history_information_20261103.run stage0
$PY -m experiments.mdmt_history_information_20261103.run dispatch --stage D0
$PY -m experiments.mdmt_history_information_20261103.run decide --stage D0
# 后续D1/REPLICATE/D2/CONFIRM仅按05与签署decision显式dispatch
$PY -m experiments.mdmt_history_information_20261103.run freeze
$PY -m experiments.mdmt_history_information_20261103.run evaluate --split test
$PY -m experiments.mdmt_history_information_20261103.run report
```

run identity至少包含dataset/split/source/model/arm/seed/λ/τ/κ/lr/wd/β/rho/epochs/batch/RNG定义/hash。logical_run_id由canonical JSON SHA256生成；stage和attempt不进入科学身份。调度前去重，旧轨迹复用单独登记源身份，不能以目录名不同算新重复。

必须输出：SOURCE_MANIFEST、DATA_MANIFEST、PARAMETER_MANIFEST、DESIGN_IMPLEMENTATION_MAP、CODER_REPORT、独立REVIEW_REPORT、CPU_TESTS、STAGE0、RUN_MANIFEST、各阶段DECISION、CONFIGURATION_FROZEN、SELECTION_FROZEN、逐epochvalidation、endpoints.csv、paired_seed.csv、paired_summary.csv、six_cell.csv、costs.csv、REPORT.md、QUESTION_DISPOSITION.md。state hash仅用于身份，不将其包装为性能证据。
