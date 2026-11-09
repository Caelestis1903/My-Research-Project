# PUB-MDMT DESIGN SPECIFICATION

状态：DESIGN COMPLETE，2026-11-09。本规范在本轮实现和科学训练前冻结。用户已授权执行完整实验；来源计划中“规划完成、未训练”描述其编写时状态。

## 1. 研究责任与固定来源

检验同一更新前 Adam 状态产生的任务候选更新，在更新空间组合是否优于普通 Adam、固定 EMA F 及真实同 alpha 的梯度空间控制；比较两任务、六任务、局部拓扑。依据本目录 SOURCE_AUDIT.md 和计划 `research_plans/26-11-09_pub_mdmt/00–08`。

- main：`40c8afafedab422bdf36c02b38aff19114d95a9b`；历史 26-11-09：`1391affa5f08556c79b4cb2bfad15228a8b9129f`；不合并历史实验分支。
- 作者库 `yjdy/Multitask-Optimization-Recommendation-Library`：`e234c149a54d0f84748d5b9de94f45ba541e1e0a`，原文件在 `author_source/`。
- 论文 [arXiv:2410.05806v2](https://arxiv.org/html/2410.05806v2)，Algorithm 1、V-A–V-C。本轮采用论文 CCP 式(4)，修正作者实现中的导入、求解器、旧 step 和重复 WD 问题；不是照搬有缺陷代码。
- 五 seed 固定为 **2022、2023、2024、2025、2026**；两数据集所有正式候选均完成 50 epoch。

无 Gaussian 或其他新增增强、重采样、域权重先验、候选向量归一化、alpha simplex、LR ablation 或额外 moment 变体。域规模只作真实人口描述；不构造“域规模=困难度”的假设或实验。

## 2. 最小独立源码范围

新增包仅为 `experiments/pub_mdmt_20261109/`，不得修改历史 vendor、EMA 实现、原模型算法。main 未合入历史执行包，最终新包必须独立可执行，禁止依赖相邻 checkout 的绝对源码路径。

最小冻结 vendor 来自历史 26-11-09 的 `experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/`：`reference.py`、`model.py`、`basic/{layers,features,activation,initializers}.py`、`utils/data.py`、`audited_data.py`、`source_protocol.py`、`resolved_protocol.json`，以及必要空 `__init__.py`。原样复制；只允许调整新包内必要 import。新模型用原 `make_model`，clean forward 调用源 reference，保持原注册/初始化顺序、gate detach、LayerNorm、fixed1 和输出语义。

12 组注册逻辑复用 `experiments/aug_history_20261105/{group_registry,registry}.py` 的必要部分；稳定 hash、数据 cache 装配、RNG 和指标等小工具按已有行为实现于新包，避免把历史 guardian、pulse、预测控制框架带进来。登记来源文件原路径、commit、源 sha256、新文件 sha256 和 import 差异。作者源码仅作可审计参考，不直接 import 有缺陷的 PUB optimizer。

代码责任范围包括 loss 分解、纯候选、CCP solver、P0/P1/P2 更新、同 alpha 重放、训练/评价入口、最小调度、必要测试、manifest 和数表生成。CLI 可采用 `python -m experiments.pub_mdmt_20261109.run {audit,check,smoke,calibrate,pilot,train,report}`，允许等价更简单的入口。报告统计程序按本规范预先实现，科学解释由 orchestrator 完成。

## 3. 自然数据、模型和训练协议

沿用已审计 FULLTRAIN_NATURAL80_10_10：ML train/val/test=800167/100021/100021；KR=2006132/250767/250767。训练三域人数 ML=[155949,318599,325619]，KR=[172782,1743898,89452]。原始 CSV：

- ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`。
- KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。

可复用 `/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache` 的只读自然数组，必须核对源 resolved_protocol 的 raw SHA、数组内容 hash、split row IDs/指纹、encoder/scaler、split人数和三域人数；cache 文件 SHA 与其 manifest 一致。cache 不在时只用冻结 `_load_movielens_base` / `_load_kuairand_base` 重建同人口并校验相同身份，不能依赖缺失 strict_v1。Test y/预测不进入调参或选择函数。

M3oE fixed1：embedding16，FCN=[512,256,256,64]，4 shared experts，exp_d=exp_t=bal_d=bal_t=1；输出 `[N,2]`，tower index=`3*t+d`。参数数 ML=773470，KR=1614254，无 BatchNorm/Dropout。

batch=40960，eval batch=40960，epochs=50，无训练早停，lr=.01，betas=(.9,.999)，eps=1e-8；ML/KR coupled weight decay=1e-4/1e-5。FP32，AMP/TF32/AMSGrad/foreach/fused/maximize/capturable/differentiable/gradient clipping 均关闭。使用项目 Python3.13.5/PyTorch2.9.0+cu126；必要增加 cvxpy 与可用 conic solver（优先 CLARABEL），记录实际版本与新增依赖快照，不换 Conda 环境。

同 seed 同初始化（Python/NumPy/Torch/CUDA seed）、同 Loader：独立 `torch.Generator().manual_seed(seed+11000)`，每 epoch `torch.randperm(N)`，尾 batch 保留。每 epoch 记录初始化 hash、loader order hash、真实 step 和累计样本数。观察/求解不使用训练 RNG；eval 恢复原 train/eval 状态和 RNG。新 baseline 与历史协议的差异只按实际身份披露。

## 4. 六格损失与实际支持

对 batch 实际人数 N，`b_it=BCE(pred_it,y_it)`：

`L_dt = sum_{i:domain_i=d} b_it / (2N)`；`L_t=sum_d L_dt`；自然目标 `L=sum_dt L_dt=mean_i mean_t b_it`。

不要把六格各自均值相加；空域损失是与计算图相连的零标量。图审计已经确认所有非所属组的梯度精确为零。FP32 原 mean 和分组 sum 的归约容许浮点误差（审计24自然行的 loss 差5.96e-8、梯度差0）；测试用紧 allclose，不要求不同归约顺序的标量 bitwise 一致。

组为 S，D0/D1/D2，T0/T1，DT00/DT01/DT10/DT11/DT20/DT21。domain embedding row d 属于 Dd；其余映射严格使用源 registry。ML/KR 每组坐标：S=483152/661920；每 D=74960/295632；每 T=16576；每 DT=5381；inactive=0。每物理坐标恰好一组。

活动 cell 是当前 batch `n_d>0` 的真实 cell；不为缺席域的零梯度制造 PUB 任务。某组活动任务数0时，以自然总数据梯度0执行一次普通 coupled Adam（保留历史与WD行为）；数1时直接普通 Adam，不解单任务 NBS，不缩放其梯度。保持 step 每真实 batch 一次。

## 5. 纯 Adam 候选与 coupled WD 迁移选择

令更新前参数 θ、状态 m/v、计数 k；本步 t=k+1。定义数据或总梯度 z 的候选：

```
m+(z) = beta1*m + (1-beta1)*z
v+(z) = beta2*v + (1-beta2)*z*z
U(z) = [m+(z)/(1-beta1**t)] / [sqrt(v+(z)/(1-beta2**t))+eps]
```

候选全部读取同一更新前 m/v/t，纯计算，不调用 optimizer.step、不写状态；eps 在修正后的 sqrt 外，bias correction 只一次。实现尽量沿用 PyTorch single-tensor Adam 的 `lerp_`/`mul_`/`addcmul_`/`addcdiv_` 运算顺序，标准退化逐坐标与 `torch.optim.Adam` 对照。

对平衡组，数据候选 `u_c=U(g_c)`，只用数据梯度构造 Gram。令 `gbar=sum(alpha_c*g_c)`、`r=lambda*theta`，公共 coupled-WD 增量：

```
delta_WD = U(gbar+r) - U(gbar)
PUB: theta_new = theta - lr*[sum(alpha_c*u_c)+delta_WD]
G:   theta_new = theta - lr*U(gbar+r)
m_new = m+(gbar+r)
v_new = v+(gbar+r)
step_new = t
```

这是本项目为匹配原 coupled Adam 选择的 **coupled-WD PUB 迁移定义**，论文未规定这个差分项。公共 λθ 只加入一次；不在每个任务候选里加 WD，不使用 AdamW。禁用 PUB、回退组、单活动任务组、P0 的 T 组及全部 DT 组直接使用自然原始总数据梯度+一次 WD 的普通 Adam，必须与源一步参数/m/v/step一致。所有候选、alpha 与新状态先完成，再每物理坐标写一次参数和状态。

不为已解决的 WD 语义再增加 zero-WD 科学矩阵；若实现不能通过标准退化测试，返回 coder 修正，不能带缺陷启动科学训练。

## 6. P0/P1/P2 与真实同 alpha 控制

| 组 | P0 / C0 | P1 / C1 | P2 / C2 |
|---|---|---|---|
| S | L0/L1 两候选，独立解 | 六 Ldt，解 alphaS | 六 Ldt，独立解 |
| Dd | 对应域两任务，独立解 | 沿用 alphaS 中同域两项 | 同域两任务，独立解 |
| Tt | 三域聚合单任务普通 Adam | 沿用 alphaS 中同 task 三项 | 同 task 三域，独立解 |
| DTdt | 普通 Adam | 普通 Adam | 普通 Adam |

P1 子集 alpha 不能归一化。候选梯度可一次 forward 后按2或6个 loss 逐个 autograd.grad；没有实际图支持时填零，不能省掉共享状态中的历史项。每个平衡组以其活动候选更新；缺席域处理见§4。大 batch graph 如需重算必须保持同样数据与源 RNG，不减 batch。

G0/G1/G2 **重放对应 C0/C1/C2 的实际逐 step alpha、活动 cell 及回退 mask**，不在自身轨迹重新求 alpha。梯度来自 G 自己当前参数与状态，数据序列/初始化/预算和 C 匹配；单任务组同 C 普通 Adam。C 先运行并保存纯标量系数 trace，G 后运行；日志核对每 step coefficient trace hash。PUB vs G 是同一外生 alpha 计划下的完整优化轨迹差，包含更新及由此产生的后继状态/梯度差；不把二者称为相同梯度的单步对比。

## 7. 论文 CCP solver 与失败处理

用未作候选向量归一化的实际候选更新构造原始 FP64 Gram `G_ij=dot(u_i,u_j)`，分块累加后对称化。令 `s=trace(G)/n`（有限且正），`H=G/s`；稳定项为 `H_reg=H+ridge*I`，ridge∈{1e-12,1e-10,1e-8}。数值变量 a 与物理 alpha 关系 `alpha=a/sqrt(s)`，此映射恢复原始 G 的尺度，不能直接把 a 应用于模型。

论文 CCP：`phi_i(a)=log(a_i)+log((H_reg*a)_i)`；每外迭代最小化 `q_prev^T*a`，`q_prev=1/a_prev + H_reg.T@(1/(H_reg@a_prev))`，约束 `-log(a_i)-log((H_reg@a)_i)<=0`、a>0。使用 cvxpy conic solver 求这个凸子问题；不带作者代码额外的 `sum(H*a)` 项，不改成 simplex/softmax。省略与 a 无关的常数。前一解不在 log 域时，先以相同约束最小化 sum(a) 获得可行初始化；这只是 CCP 可行点初始化。

外层最大迭代{50,100,200}，tol∈{1e-4,1e-6}；内部 conic 容差设为 `min(1e-7,tol/10)`、max_iter=200。默认起点全1（或可行前解），只接受新解计算残差后返回，不返回旧迭代。停止看 `||H_reg*a-1/a||/(||H_reg*a||+||1/a||)<=tol`；alpha变化小只表示停滞，不能代替残差通过。

记录原始/稳定 Gram 的对角、谱、条件数、候选范数/夹角，alpha最小值/sum，CCP迭代、conic状态/异常与耗时、regularized residual、原始无 ridge residual `max|alpha*(G@alpha)-1|`，以及原始 utility `G@alpha`。接受要求所有量有限、alpha>0、原始 utility>0、原始 residual<=1e-4，且 solver 报告成功。近零候选、不可行/非有限、不可接受残差、solver失败时 **该组** 回退自然总梯度普通 Adam，并明确标记原因；不把 ones-alpha 更新称为普通 Adam。P1 S 解失败时所有继承组一并普通 Adam；局部单组失败只影响该组。任务重排序须同时重排 Gram、warm alpha、任务身份，恢复后物理更新一致。

alpha refresh 从真实 step1开始，再每 frequency 步（1,1+f,...）；其间保持物理 alpha。组活动集合改变时立即重解；回退只针对当步，并于下一步重试。不能沿用作者 `step%frequency==1` 导致 frequency1永不求解的分支。

## 8. 必要验证与有限搜索

独立 reviewer 对规范、实际实现、CODER_IMPLEMENTATION_REPORT 给出 PASS/PASS WITH FIXES，blocking 全闭合后，runner 才执行科学训练。coder 只做语法和以下必要正确性检查，不分析科研结果：

1. 真实 batch 六格 loss/梯度恢复自然 clean 目标，含空域；输出、tower、12组坐标覆盖/零泄漏。
2. 非零旧 m/v/step 的候选手算，候选前后状态 hash 不变；任务重排后 Gram/alpha/参数/m/v一致。
3. P0/P1/P2 共享集合与 P1 子集、P2 独立系数；DT与普通 Adam；零/单活动任务分支。
4. disabled/回退的参数、m/v、step、coupled WD 与 PyTorch Adam 多步逐元素对照，WD=0与源ML/KR WD；每坐标恰写一次。
5. 对角 Gram、非对角2任务解析根、正定6任务数值解/残差，退化/失败明确回退；正确恢复原始 alpha 尺度。
6. RAM state_dict 深拷贝恢复后下一步、alpha warm状态、RNG、loader顺序一致；没有训练状态写文件。G replay 系数精确一致，G不读自己的solver。
7. 原 full batch40960 的 GPU forward/backward/step，ML/KR各一次；Validation选择和Test入口隔离、最早严格最大值、真实 LL 指标。

以现有小型 pytest/assert 手算检查覆盖这些行为，不引入与当前结论无关测试。

有限搜索分两步，所有选择只看数值和 Validation：

- **Solver数值校准**：两数据×五seed，在源自然首两批/两次普通Adam工程步采集所需 S/D/T Gram；覆盖零状态和非零状态。离线对18个(niter,tol,ridge)组合执行相同 CCP，记录全部残差/状态/时间。候选须通过手算fixtures且所有可审计非退化 real Gram 满足§7原始残差门。数值退化 Gram单列且检查明确回退。合格组合中按 median wall time最小选一个共同配置；时间完全相同时依次选较小ridge、较少niter、较严格tol。若没有合格项则修具体实现/数值问题，不能写成算法性能失败或换科学目标。
- **共同频率低预算验证**：两个数据×五seed×C0/C1×frequency{1,10}，每job完整2 epoch（40短job），不读Test。使用上步同一solver配置；分数为 epoch2完整Validation Macro AUC，在数据集×拓扑×seed的20个匹配条件上等权均值。选分数更高的共同频率；绝对差<=1e-12时选10。每个seed、cell和LL原值保留；不以短job表现淘汰任何拓扑或少seed否决方法。frequency100只在已有记录上离线列alpha陈旧度/成本诊断，不新增长训练。

冻结 `SEARCH_DECISION.json`（配置、全部Validation输入和规则、hash）；随后全部正式候选用一个共同solver和频率，从相同seed初始化重新训练50epoch。短job不计正式seed或正式预算。

## 9. 完整正式矩阵、选点和条件阶段

| 物理训练 | 每数据seed数 | epoch | 读出 |
|---|---:|---:|---|
| A 普通 Adam | 5 | 50 | online A、同步固定EMA B/F |
| C0/C1/C2 | 每拓扑5 | 50 | PUB online |
| G0/G1/G2 | 每拓扑5 | 50 | 对应C实际alpha重放的普通Adam |

两数据合计 **70条正式轨迹，3500epoch，120750个 optimizer step**（ML20step/epoch、KR49step/epoch）。不为F重复训练A轨迹。F初始为θ0，每真实Adam步后 `F=rho*F+(1-rho)*theta_new`，ML rho=.9828205985452511，KR=.995；缓冲区按源模型语义复制。新A/F同协议重跑，不以未验证身份的历史绝对值替代本轮配对。

每条轨迹独立按完整 Validation 六格等权 Macro AUC 最早严格最大值选择RAM状态，留 selected/E10/E50 的参数身份和指标；每epoch记录完整Validation六格/聚合。配置在正式训练前冻结；job训练结束先落盘纯标量 `SELECTION_FROZEN`（选中epoch、状态hash、协议、配置、loader身份），再在同进程RAM中评价这三个状态的Test。Test结果不能供搜索或后续方法选择函数读取。

**PUB+EMA唯一触发门**：正式五seed的 Ck−A 在 selected **Validation** 的平均 Macro AUC>=.003，且每格平均差>=−.001。对每个数据集，符合门的C中只选择Validation Macro增益最大者（并列C0<C1<C2）；五seed资格及选择函数只读Validation。决定写 `COMBINATION_DECISION.json`。若该数据集有合格C，则从其原冻结配置重跑同五seed/50epoch的C并同时维护同rho的F，测试PUB+EMA与原C/A/F；无合格C则 `NOT_TRIGGERED`。不根据Test挑拓扑或触发组合。原计划并未要求PUB先超过F才允许互补实验。

完成C0/C1/C2/G0/G1/G2五seed两数据及已触发组合后结束。无收益时按计划停止当前迁移，不继续扩展solver、alpha先验或LR。实现/数值/资源故障按其原因记录并修复，不能当成科学阴性、减少seed或缩短正式epoch。

## 10. 成本、诊断和执行

所有项目Python显式使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。模型、optimizer、checkpoint和序列化训练状态 **只在进程RAM**；磁盘仅代码、配置、数据、数值trace、日志和报告。恢复测试用RAM深拷贝；中断job没有有效RAM状态时从同seed起点重跑，并保留失败尝试日志。完成每job评价后释放其RAM，无需历史guardian框架。

GPU逐设备独立准入：该卡显存占用<85%且功耗<该卡上限85%，同时host RAM<90%时，继续加入新训练job；一张卡可用不等待另一张卡。禁止历史固定2worker上限；记录真实显存/RSS和当前其他任务占用，按实际可用资源填充。OOM先在原batch下核查并发/显存，不能静默减batch。

静态payload估算：ML/KR一个FP32模型3.09/6.46MB，m/v=6.19/12.91MB，6梯度矩阵18.56/38.74MB；同尺寸候选矩阵另加18.56/38.74MB。实际full-batch activations和CUDA上下文占主要空间，必须用本轮smoke记录峰allocated/reserved/RSS、step耗时和吞吐。

历史机制smoke仅用作初估：ML/KR clean两epoch训练4.366/8.163秒，峰GPU约1.024/1.267GB，历史硬件负载与本轮不同。按 `35条/数据集 * 25个两epoch * (4.366+8.163)`，未计PUB时70条正式轨迹的训练约10963 worker秒；若全部按源训练成本6倍作粗估则约65777 worker秒，另加完整Validation、solver及Test。runner用本轮P0/P1/P2实测更新估算，不以论文16%成本当本地事实。

每step保存实际应用alpha和回退mask供G重放；每10step/每个refresh保存Gram、utility（数据组合Galpha及含公共WD后实际应用方向的内积分别记录）、残差、候选范数/余弦、sum/min/max alpha、真实PUB更新范数、同刻自然Adam更新范数及比值、额外autograd次数、求解时间。solver接受门只用于新求解的alpha；两次refresh之间沿用既有alpha并记录当刻utility变化。保留每epoch loss和总运行成本。

每epoch首batch，用其前最多1024个自然训练行做只读短步诊断：组S在更新前参数的原始候选/组合utility，对应scratch候选步和PUB组合步的六格BCE变化；缺席cell记NA。只做现有候选的功能观测，不产生训练样本、不反馈决策；scratch结束恢复RNG。比较utility排名与实际短步loss方向/排名，按seed、训练阶段、数据集报告。

## 11. 指标、分析和交付验收

主端点是Validation选点的Test六格等权Macro AUC/LogLoss；保留Validation/Test、selected/E10/E50、每cell、每task三域等权、六格Macro、Traffic（各域人数加权cell指标后平均task）和Pooled（逐task合并真实行后AUC/LL，再平均task）。Traffic AUC不同于Pooled AUC，不混称。真实LL未经训练系数/alpha缩放。

以seed为统计单位：n=5，原值、均值、样本SD、逐seed配对差、配对95%t区间 `mean±2.776445105*SD/sqrt(5)`、方向数。比较 C/G vs A、C/G vs F、Ck vs Gk、P1−P0、P2−P1；选择身份、最差cell及变化完整保留。AUC合格逐比较标记：平均ΔMacro>=+.003且所有六格平均Δ>=−.001；LL独立以平均ΔLL<0裁定。不加seed同向数或CI硬门。

最终报告由orchestrator完成：算法有效性、seed/阶段/数据/任务稳定性、相互矛盾实验、STRONG/MEDIUM/WEAK证据、更新空间独立增量、固定EMA比较、utility与BCE/AUC区别、有效步长解释、数值回退/成本、替代解释、未解决问题、最有判别力的后续实验。仅描述真实域人口与观测结果，不以域大小推断困难度。

交付：SOURCE_AUDIT、DESIGN_SPECIFICATION、CODER_IMPLEMENTATION_REPORT、独立REVIEW_REPORT、测试结果、冻结manifest/环境/数据身份、搜索决策、完整运行日志、alpha/Gram/更新诊断、逐seed全部六格/聚合/配对数表、成本、最终报告及必要图表。回填本分支 `EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md` 的当前实验结论。PUB称候选方法/内部控制，不列正式发表模型baseline。代码与报告按用户最终确认的目标分支提交推送；不合并历史实验分支。
