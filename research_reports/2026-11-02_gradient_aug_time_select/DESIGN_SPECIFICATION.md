# GATS02 DESIGN SPECIFICATION

设计日期：2026-11-01。状态：设计完成；协调者已裁决下述合同冲突，交独立 coder 实现、独立 reviewer 审查，PASS 后由 runner 执行 Stage0，再正式训练。本文只定义本轮方法和验收，不包含新实验性能判断。

## 1. 任务与来源

唯一入口为 `research_plans/26-11-2/06_Codex完整执行提示词.md`，同时读取 00–05、`experiment_queue.yaml`、根 `AGENT.md`/`AGENTS.md`、`PROJECT_ARCHITECTURE.md`、GB18030 编码的 `start.md`、前置 `research_plans/26-11-1_v2/00–06` 与现有 AIGM/GAM/GA31 代码和研究报告。

协调者核验 main 为 `0bf3e970ca47da44b129e67bb4c84a653ba6a20b`；最新证据分支 `26-11-01_v3` 为 `6f5742c2085c1e49d1767b0a7079bb9f28160eda`，历史 AIGM 执行代码 commit 为 `2cec25ab5b32411855a13ab3143678a5e980ea3f`。来源身份与实际文件 blob/hash 由本轮 source receipt 保存，不能将历史执行 SHA 当成本轮执行 SHA。

研究问题：B1 难度相关字段覆盖、B2 同任务增强冲突分量移除、B3 等剂量延迟增强，是否超过普通 Gaussian 及必要匹配控制；固定 EMA 的启动/窗口与梯度稳定性 adaptive EMA 是否有额外价值。旧 MASK 只作为方向来源；未找到的旧原始配套代码/日志记录为缺口，不阻塞新合同。普通 EMA 不称为创新，不恢复旧课程、DBO、VUG、SAD/AIG-M、网络结构或其他优化器。

新增范围仅为 `experiments/gradient_aug_time_select_20261102/`、`research_reports/2026-11-02_gradient_aug_time_select/` 和独立 artifacts attempt。复用历史数据、model/view/metric、RAM IPC 和调度模式；不修改历史包、报告、数据、CSV、规划。coder 报告列出复用接口、改动文件、验证和未实现的条件分支；reviewer 与 coder 必须为不同代理，除机械 runner 外沿用协调者模型。

## 2. 已裁决的冲突与执行优先级

1. 04 第 4/8 步仍把 adaptive/replay 放在 A0；遵从唯一入口 06、最新 03 与 YAML：A0 为 fixed/delayed/SWA，A1/B 为 fixed/adaptive/replay。任何 online 轨迹最多 4 个独立 shadow；默认每臂 3 个，不增加可选第四槽。固定 EMA 始终保留。
2. 三套完整匹配控制需要 12 条新轨迹，不能同时容纳于最多 8 条附加预算。严格保留 20 基础 + 最多 8 附加（控制优先）+ 最多 12 确认、总上限 40。先按 dataset 评估主增量/保护，仅仍可能晋级或有预登记 trade-off 需判别者派发匹配控制；B2 > B1 > B3，每 dataset 一对开发 seed 占 2 条。无预算的未匹配者记 `BUDGET_STOP_UNMATCHED`，不得晋级、不得写成与控制相当。受干预 mask/schedule 不能假装同一 online 轨迹回放。
3. AIGM 最终 A1 Gaussian κ 为 ML .05、KR .1，均 λ=.5。本轮“κ 与 A1 相同”使用这两个值，不因历史 G2 默认 .1 把 ML 改回 .1。
4. B2 单任务 target loss 用原 batch B 归一化，两个任务均值一次得到原 B×2 归一化。禁止先除 B×2 再除 2。
5. SWA 是 E31–E50 的 epoch 末参数等权平均；E31 前未形成该估计器，E10 记 N/A，不能用早期 online 模型代替它选点。延迟 EMA 在 E5 末以 online θ 初始化，E6 起更新，E1–5 不作为 delayed EMA 有效选点。
6. 完整 rho 序列依赖未来训练信号，必须先完整采集，再从相同初始化/RNG 确定性重算同 online 轨迹产生独立 replay shadow。逐 step online 参数 hash 一致才可称同一唯一轨迹；所有重算曝光、forward/backward、更新、验证和墙钟全计成本。
7. B2 固定缩放按 §6 的训练统计 ratio-of-summed-norms 冻结，协调者已批准。B1 τ=2 的公式与平均能量匹配未充分定义，不先行猜测或实现；只有明确触发且预算有余时另产 DESIGN 附录并实现/复审。未触发时无需追加设计或训练。
8. 确认必须具有相同 seed/config 的 A0/A1 数值配对。可严格核验并复用历史完整数值 baseline；若不满足身份，则新增 baseline 也占 40 上限，并据预算取消候选，不能借无配对确认扩大预算。

以上为协调者对规划冲突的显式裁决。其他新公式冲突必须先报告协调者，coder 不自行变更。

## 3. 固定数据、模型、随机流与优化器

真实 CSV 路径：ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`，KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。复用 `gradient_aug_20261031_v2.vendor_ga31.data.load_dataset` 及 audited_data；显式传 raw path。按冻结 GA31/AIGM manifest 检查 raw SHA256、train/val/test 六数组 hash、split 行指纹/行号、encoder/scaler hash、域人口。现有缺失的 mdmt_next prepared 目录不在这条数据加载路径，不另造 prepared 数据替代。

自然原行序 80/10/10，KR 先全局 split 后筛 tab 0/1/4；只在 train 拟合 encoder/scaler，OOV=0。ML 标签 rating>3、rating>4，KR is_click、long_view。ML train 域计数 155949/318599/325619、总 800167、20 step/epoch；KR 172782/1743898/89452、总 2006132、49 step/epoch。目标增强域分别为 ML d0、KR d2。没有重采样、复制样本、改变标签或人工人口比例。B1 只改变原合同允许的 Gaussian 覆盖位置。

模型复用 `gradient_aug_20261031_v2.views.make_model`，fixed1、3 domain×2 task、4 shared experts、embedding16、fcn_dims=[512,256,256,64]，固定融合系数均 1；参数量 ML 773470、KR 1614254。forward `[F,B] -> [B,2]`；路由 domain 和 clean 输入保持原状。learned fusion/AutoML/DARTS/DBO 均关闭，只有一个标准 Adam。

每轨迹完整 50 epoch、batch=40960、末批自然大小、FP32、loss_scale=1；Adam lr=.01、betas=(.9,.999)、eps=1e−8、coupled wd：ML 1e−4、KR 1e−5；foreach=False、fused=False、amsgrad=False。权重衰减只进入 optimizer 的实际标准 Adam 通道，同时影响一/二阶矩；不进入下述 clean target 参考梯度或 delta。关闭 AMP/TF32；沿用源 deterministic setup、loader offsets 和 keyed Gaussian view，保持源原始参数顺序及 None/显式零的活动语义。

seed 仅 2022–2026，开发 2022/2024，确认 2023/2025/2026。loader、view、mask、replay 各自独立；mask key 至少含 study/dataset/seed/step/role，独立 generator，不能消费 loader/view RNG。组件关闭时不抽取其 RNG、不做其额外梯度。诊断、shadow、validation 不改变 online 随机流。记录初始化、每 epoch 行序、逐 step online 参数、view/mask key/hash；完整 RNG/loader 状态只留 RAM，磁盘仅写 hash/标量身份。

## 4. 共同目标及 A0/A1

令原 batch 大小 B、任务数 T=2、目标域样本索引集 S。clean 每样本每任务 BCE 为 ℓ_it，增强为 ℓ̃_it。模型概率 BCE 沿用冻结源，不自行改 logits loss 或裁剪。

`L_real = Σ_i,t ℓ_it/(2B)`；`L_delta = Σ_i∈S,t (ℓ̃_it−ℓ_it)/(2B)`；`g=∇L_real`，`δ=∇L_delta`。target 分母始终为原 B，不用 |S| 替代。

A0：标准 Adam 使用 g；A1：标准 Adam 使用 `g+λδ`，λ=.5。A0/A1 保持源单次 backward 的标量实现，以便关闭退化严格对齐。A1 单 Gaussian iid view，κ=ML .05/KR .1，仅目标域非路由 sparse embedding。每字段每行 clean embedding RMS stopgrad 后乘 Gaussian，沿用原 view key 和 draw 次序。不存在目标域、κ=0 或 λ=0 时直接 A0，不抽 view。B1/B2/B3 关闭时直接走 A1 参考路径，不维护备用梯度图。

训练日志始终区分真实 BCE、A1 参考混合 scalar 和实际注入梯度；B2 投影更新不能把参考 scalar 称为其精确可积优化目标。

## 5. B1 与随机字段匹配控制

同一 clean forward 得到目标域样本难度 `l_i=(ℓ_i0+ℓ_i1)/2`，detach。目标域 batch 内采用 average-tie rank，归一化 `(rank−1)/(n−1)`；n=1 时 r=.5，n=0 时走无增强分支。高 r=难。

默认 `p_i=clip(.05+.90*(1−r_i),.05,.95)`。按每目标域样本、每非路由 sparse 字段独立 Bernoulli(p_i)，把原 Gaussian 扰动乘该字段布尔 mask。未选字段 clean，dense/routing/labels/full-real loss 不变。难度不回传，不再额外对 target 样本求均值。无需新增 model 主干；沿用原 embedding hook 的最小扩展。记录每目标域样本原始 row_id/epoch/step、p_i、字段 mask 或可精确重建的 key、难度/概率分位数、选中字段率、实际扰动 RMS/能量及 epoch 分布。

随机控制 R1 必须是真实独立训练：从 B1 日志冻结每个 `(dataset,seed,step,field)` 的选中数量 k_f 与实际总扰动平方能量 E_f，先按该数据集两 seed 完整训练数值冻结，再运行 R1。R1 在当前 batch 目标域 n 行中用独立 keyed 均匀无放回选择恰 k_f 行，不读取难度/验证/测试；Gaussian 标准噪声仍复用 A1 view key。用控制当前 clean RMS 生成原始扰动，对每字段用 detached 单一尺度乘至总能量 E_f。k_f=0/E_f=0 时该字段零扰动；非零源能量但控制原始能量为零时必须报实际身份问题，不替换人口或静默注入别的噪声。

这样 R1 逐 batch/field 精确匹配覆盖数和总能量，切断难度与位置关联。它不等于固定 κ 的简单稀疏增强，报告其能量再尺度语义；无需再训练另一套固定覆盖率控制。不得只在已训练 B1 权重上改变 mask 并把 forward 当作控制训练结果。若未派发 R1，B1 仍能报告相对 A1 的结果，难度选择独立效用记未判别且不得晋级。

## 6. B2 任务条件投影与固定缩放控制

同 θ、同原 batch 计算：`L_clean,t=Σ_i∈S ℓ_it/B`，`L_aug,t=Σ_i∈S ℓ̃_it/B`，`g_t=∇L_clean,t`，`δ_t=∇(L_aug,t−L_clean,t)`。四个 task-separated autograd.grad 加 full-real 梯度，共最多 5 次梯度遍历；可复用同 clean/aug forward 图，但不可先 optimizer.step 再求任一梯度。日志记录实际遍历次数，不把 autograd.grad 当零成本。

参数分组原样复用 `gradient_aug_aig_m_20261101.optimizer.parameter_groups`：E 为 `embedding.embed_dict.*`；P 为共享路径、target slot/domain expert/gates/towers 与 task experts；U 为其他参数。保存每 named_parameter 的组和元素数，不能按经验另分共享层。

对 t=0/1、G=E/P，先在整个组全部张量聚合 `dot=<δ_t,G,g_t,G>` 和 `norm2=||g_t,G||²`，用 FP64 reduction、epsilon=1e−12：`a=dot/(norm2+epsilon)`，`δplus_t,G=δ_t,G−min(a,0)*g_t,G`。标量 detach，输出保持 FP32；禁止逐层归一化或把任务聚合后再投影。U 增强差分为零。Stage0 检查每任务 U delta 为 None 或精确零；若非零，暂停该分组并报告来源，先修正设计/复审，不静默丢弃非零 U。

合成 `δplus_G=(δplus_0,G+δplus_1,G)/2`，optimizer 输入 `g+λδplus`；λ=.5、标准 Adam，不新增 delta momentum、不改 exp_avg beta。无冲突时完整退化 A1；开关关闭直接源 A1 路径，不多算任务梯度。real/delta 均 None 的参数仍 None；单侧有梯度时按相应零合成，保留源活动语义。U 的 real gradient/Adam 保持原语义。

保存各 task/group 的 clean/delta/retained norm、dot/cosine、a、投影移除 norm，以及聚合注入 norm。epsilon 使理论投影点有微小负残差；检查与原式预期和 FP32 数值误差相容，不要求不可能的绝对零。

R2：对每 dataset、G=E/P，合并两开发 seed 全部训练 step/task 的 detached 范数，冻结 `q_G=Σ||δplus_t,G|| / Σ||δ_t,G||`，分母全零取1；每组原理上 q∈[0,1]，保存原始分子/分母、seed/step数与冻结 hash。不要用两任务聚合后有抵消的范数比，不用 validation/test 调 q，不逐 seed 选 q。R2 以 A1 原始 δ 为基础，E/P 乘对应固定 q、U0，注入标准 Adam。它是组固定减量对照，真实独立训练；报告与 B2 完整投影对比。随机方向投影不默认启用；若固定减量仍不能判别，记录其限度，不自行扩展轨迹。

η=.5 附加配置只有注册触发且预算允许才可做：`δeta=δ−η*min(a,0)*g_t`；需新配置身份及相应同配置控制解释，不用 default η=1 的控制冒充 η=.5 的剂量匹配。控制/调参预算不足时停止追加。

## 7. B3 与随机关闭日程

B3 E1–5 λ=0，不画 Gaussian；E6–50 λ=25/45，使用数学精确比例，显示小数 .5555556。A1 Σepoch λ=25，B3 同为25；由于每 epoch 相同 batch 大小/组成总曝光，记录按 epoch、step 和样本权重的累计预算，末批不能漏计。

R3 用独立 NumPy Generator seed1729 从整数1..50无放回预先取5个关闭 epoch，保存排序列表和库版本/hash；其余45 epoch λ=25/45。两个数据集/所有研究 seed 共用同一冻结 epoch 列表，不重新抽“有利”日程。R3 为独立完整训练；更改关闭时程会改变 online 权重，不能作为同轨迹 shadow/replay 省掉训练。

delay10 附加配置仅满足注册触发且有预算时使用 E1–10关闭、E11–50 λ=25/40=.625；记录同剂量和需相应匹配对照的归因边界。E10 只观察，不用于早停/剪枝。

## 8. 参数平均与 adaptive/replay

所有 shadow 从 online 参数独立复制、requires_grad=False，不反馈训练。每次 online optimizer.step 后 fixed/adaptive/replay 更新浮点参数，buffer 复制源值，eval 后恢复 training 标志；检查不消费 online RNG。固定 rho：ML .982820598545、KR .995，来源为已选 AIGM 强基线。

A0 有 fixed EMA、delayed5 EMA、SWA20 三 shadow。delayed5 在 E5 末复制 θ5，E6起逐 step 以固定rho更新；E1–5不参与其选点，E10/E50有效。SWA20 在每 epoch 末收集 θ_e，e=31..50，`s_e=s_previous+(θ_e−s_previous)/count`，首次直接θ31；没有逐step参数队列。E31–50每 epoch评估该累计窗口模型，selected只从此有效区间选；E50是20个epoch末权重等权平均，E10=N/A。无需额外4th shadow。

A1/B1/B2/B3 及其控制有 fixed EMA、adaptive EMA、replay-rho 三独立 shadow，不维护 delayed/SWA。C2 的正式独立时间候选预先锚定 A1 online 轨迹，B臂 adaptive作为同轨迹诊断，不从所有 B×shadow事后挑出另一个C2。A0窗口结果提供跨轨迹背景，不能把 A0 delayed/SWA 与 A1 adaptive的差异全归于时间规则。

adaptive 信号为当前实际 optimizer 输入梯度（含本臂增强处理、wd前）的 global L2 norm；不额外 backward。用最近100个范数的 population std(correction=0)/mean 得CV。step1–99使用数据集fixedrho；在step100、200…将包括当前step的最近100观测映射为 `rho_min+(rho_max−rho_min)*CV/(CV+.5)`，当前step起保持到下一次更新；ML [.95,.995]、KR [.98,.999]，epsilon1e−12。记录每step gradnorm/CV/实际rho，不能读取validation/test调rho。

replay：首遍完成后冻结该轨迹完整逐step rho 值数组，用独立seed1729在全长step轴做随机排列，值的多重集合精确相同。重放从相同θ0、Adam、RNG/loader开始，逐step重新算相同online更新；重放维护独立replay shadow，同时固定EMA始终保留，adaptive有独立状态（首遍已有快照不得覆盖或冒充replay）。最多3个shadow。首遍/重放每steponline hash完全相同并核验final optimizer/行序/view hash，确认同一唯一轨迹；否则该重放FAILED，禁止配对成时间机制证据。重放仅replay shadow需完整epoch validation/独立selected；首遍已有效证据不重复算成新的unique轨迹，实际额外计算全计成本。排列打破时序也改变rho自相关，报告这项竞争解释。

## 9. RAM 状态、test 冻结与配对基线

磁盘禁止任何 model/optimizer/shadow/RNG tensor snapshot、checkpoint/pth/pt/权重压缩包。RAM epoch状态至少含 online、Adam、所有shadow及SWA计数/和、adaptive norm deque/currentrho、RNG、loader generators、epoch/globalstep、独立selected/best、cost counters、配置/源码/数据hash。source snapshot/IPC可复用 BytesIO/Pipe；不得经磁盘或临时文件中转。mask概率/能量、rho、metrics、hash等数值实验日志可落盘；它们不包含模型状态。

每个有效online/shadow按validation六格macro AUC严格提升选epoch、平局最早、E0排除；保留RAM selected、E10、E50端点，旧未用快照释放。全队列开发、控制、确认完成且所有候选/超参/选择/模型身份冻结后，统一写selection_frozen含每个要测试端点的run/config/data/source/model/optimizer hash与epoch/step，再由单独test阶段打开模型测试。Stage0/训练期禁止test forward/metric。加载test数组仅作来源hash核对，不用于选择。

统一test可描述本轮完成的开发/控制与晋级确认，测试清单在打开前冻结；不能因test改变候选、配置、门槛或再训练。每种有效估计器独立selected/E10/E50，端点同kind+epoch去重forward但保留全部端点行。SWA E10写N/A原因。原始test已参与历史研究，报告固定划分与既有seed池这两个具体边界。

确认采用2023/2025/2026：候选与同seed A0/A1固定配置配对。严格复用历史baseline数值需核验：原始数据/六数组/split/encoder/scaler、模型架构/参数、loss、Adam、κ/λ、loader/view随机协议、固定rho、选点、完整50epoch/E10/E50身份均一致；开发A0/A1的重跑online/fixed模型hash与历史相同是直接证据。导入数值保留history ref/commit/run/model hash，记`NUMERICAL_BASELINE_REUSED`、本轮无RAM权重/无新增训练；历史baseline不增加seed n也不重复计本轮成本。缺identity或hash不符则不可复用，新增基线占总预算；不足时取消相应确认。只复用数值不宣称旧权重可恢复。

## 10. 队列、门与预算

顺序：source audit → coder → independent reviewer PASS →真实CUDA Stage0 PASS → A0四条完整开发校准 → A1/B2/B1/B3各四条完整开发 → validation-only分阶段决策/必要控制 →触发时有限trial → C1/C2裁决 →条件确认 →全清单冻结→统一test→分析/归档/推送。

基础20条为 `{ML,KR}×{2022,2024}×{A0,A1,B1,B2,B3}`；不因B2失败取消B1，不按E10淘汰。主档不减少这20条。每run真实50epoch，ML1000更新、KR2450更新。

每dataset独立先比候选fixedEMA与A1fixedEMA：开发macro AUC两seed均正、均值≥.001、LL均值恶化<.002，worst-cell/最大域两task均值下降不超过.002；出现两seed同向且达到.002的保护损害即拒绝。探索门为均值≥.0003、2/2正、无保护。C2在A1上相对fixedEMA及replay两者按同门，不能只优于online。matched-control也须两seed正且平均增量>0；`.0005`内近似平局时优先更简单/更低成本控制，不能宣称选择信息独立有效。所有关键原值/保护列表写decision JSON。

控制筛选按“仍可能实用晋级，或注册的LL/worst-cell稳定trade-off需区分”派发；明显失败者接受相对A1停止，不浪费控制训练。R2→R1→R3优先，每dataset两seed成对，每对2条；总附加（控制+超参）≤8。只有完成对应匹配对照并超过它的候选才能调参/晋级。未匹配者明确缺口、不得写控制相当。未触发控制与trial列出条件和状态。

最多一个B（B2>B1>B3的成本/近似平局顺序）进入C1，C1就是该候选已有fixedEMA，不新增online轨迹。C2独立；最多两个B/C进入确认，每dataset触发才跑对应3seed，最大12条。分配必须先满足配对baseline/必要control；40唯一完整轨迹上限不能因确认或补baseline突破。固定预算不是必须跑满；全部候选STOP且问题已回答时结束，不发明候选。

只有已注册且预算有余时增加一个候选配置，最多两个候选：B1 τ2须新DESIGN附录；B2η.5、B3delay10按注册条件。不同文件较宽触发不覆盖05的“超过匹配控制、保护为空”要求。超参收益与机制增量分别记账。

首次完整A0训练实测吞吐后决定并发，不缩batch/epoch/seed/精度。10h停止低优先级派发、12h停止新任务并完成在跑任务/汇总；超预估1.5倍取消超参trial，2倍取消第二确认，继而取消确认，保留A0/A1及B1/B2最低开发覆盖。4h缩减档仅真实吞吐触发、需协调者记录模式；不自行用它避开默认20条。

## 11. 实现必要验证与 Stage0

只做能影响结果的最小测试。CPU/FP64手算及FP32小模型覆盖：B2同任务独立投影/组聚合/无冲突退化/零梯度/epsilon/除2/None/weight decay分离；B1rank方向、tie/singleton、clean不变、mask RNG独立、控制count/energy匹配；B3与随机日程累计预算；EMA/delayed/SWA/adaptive及rho排列多重集合/不反馈online；RAM恢复下一步一致。使用源接口避免测试无关模块。

reviewer独立读取本文、实际代码与CODER IMPLEMENTATION REPORT，核对公式、分组、数据、shadow上限、RNG、RAM、队列预算/控制触发、基线复用/test隔离和成本；blocking修正后才PASS。

真实CUDA Stage0每dataset用真实原batch40960覆盖 clean/aug两个task gradient、B1真实mask、B2组投影/U、B3预算、标准Adam一步、三shadow独立更新、完整RAM restore后下一步/model/optimizer/shadow/RNG/loader一致、κ0/开关关闭对源A0/A1的梯度/参数一致。可复用同真实batch和初始化，禁止test和性能结论。记录每项max error/容差/参数组/峰值allocated/reserved与PASS；Stage0源码manifest/executioncommit须与正式一致。U非零、归一化错误、状态/RNG不一致属于Critical，修复→复审→重测受影响项，不展开无关测试。

## 12. 资源与账本

实际环境显式 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`，锁版本Python3.13.5、torch2.9.0+cu126。记录驱动、CUDA、依赖hash、GPU状态。GPU0既有他人任务保持；逐卡单独判断当前显存与功耗均<85%且剩余显存可容原batch峰值时增并发，不要求两卡同时满足。先单worker完整训练校准，按实际allocated/reserved/RAM限并发；OOM只降低并发原配置重试，每run可恢复异常最多2次。

run_id按dataset/arm/seed/config/source/data身份唯一，ledger去重，状态PLANNED/RUNNING/COMPLETE/FAILED/CACHE_REUSED及numerical-baseline-reuse单列；研究失败是COMPLETE+STOP，不是FAILED。记录时间/GPU/attempt/异常/stdout/stderr。每run及重放保存训练曝光、增强样本/视图/field数/能量、forward、autograd/backward遍历、optimizer/shadow更新、逐kind validationforward/选点/耗时、testforward/耗时、训练worker秒/总wall/CUDAevent、RAMtensorbytes、峰值allocated/reserved。unique trajectory与重算passes、逻辑shadow输出、历史数值复用分别计数。

最后测试/表格完成释放全部RAM状态，保存state_release数值记录。归档只含源码、配置、身份、指标、日志、报告，不含原始CSV、模型或检查点。

## 13. 统计与交付验收

seed为重复单位，开发n=2与确认n=3/合并n=5分列；report逐seed原始AUC/LogLoss和配对差、mean/sampleSD、配对t95%区间、正向seed数，不把epoch/cell充作重复。六格macro、逐task pooled均值与traffic-weightedcell独立命名；全部domain×task AUC/LL/count/positive、worst、最大域/增强域两task、E10/E50/selected及hash齐备。

最终报告逐候选按“当前结论→核心证据→适用边界→研究决策”，区分调参、增强、投影、时间平均及交互。检验稳定现象、矛盾、STRONG/MEDIUM/WEAK、竞争解释、域规模与难度、small-domain改善/large-domain下降交换，提出最能证伪竞争解释的下一实验建议而不自动加跑。AUC提高但LL不改善需报告校准/选点交换，不能用几何信号/非零mask/rho等代替效用。

交付DESIGN、coder report、独立review、source/environment/data receipt、Stage0、全部run/trial/cost/failureledger、rawvalidation/test数值、配对表、decisionJSON、REPORT和README可复现入口。只新增本轮文件。按start.md向My-Research-Project日期分支提交/推送；既有当日分支加_vn，不forcepush/覆盖main新提交。任务结束不追加无关审计。
