# DESIGN SPECIFICATION — MDMT_INFORMATION_MOMENTUM_V3

设计日期：2026-11-02。设计阶段完成；实现、独立审查、S0 和正式训练尚待执行。实现仅限本规范及所引用的 v3 公式。

## 1. 依据、范围与优先级

研究规范为 `research_plans/26-11-2_v3/00–08`、`experiment_queue.yaml`，来自 `2c4ab338798e4bd98d3263bfd719b14345edc27e`。数学定义以02/03/05为准；阶段门以06为准，下面对未完全确定的执行细节作冻结解释。

已阅读本地入口 start.md、AGENTS.md、AGENT.md；本地旧 EXP_AI_CONTEXT/Global Summary 缺少最新段，最新历史结论使用本报告目录 `source_context/` 中固定 `96d21fccff4e9304589e82527ec7a859e8c3f05c` 的版本。继承参数EMA排序收益、损失尺度影响和旧具体方法停止结论，不重跑对应历史命题，不预设本轮pair/ECCM/CV有效。

本次用户要求实现、完整执行、分析并发布，覆盖设计文档的 DESIGN_ONLY 状态。start.md 覆盖性规则仅允许2022–2026：S1/S2使用2022/2024；若触发S3，则覆盖2022–2026五seed，其中2022/2024是开发重叠，2023/2025/2026为本轮未用于开发的训练种子。分别报告五seed和三个未开发seed；不称五seed独立确认。开发重复不得通过重新运行计为新增科学重复。

start.md 禁止落盘模型/优化器/checkpoint。本轮完整恢复状态和评价快照仅驻内存，磁盘只写代码、配置、数据指标、日志、哈希和报告。非有限值保存数值与故障日志，不落盘参数“现场”。不修改历史实验文件，不新增研究候选、网络结构、视图或超参。

## 2. 固定底座与最小复用

执行底座为 `38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27` 的 `experiments/ema_view_design_20261102_v2/`。协调者已在 `SOURCE_REMOTE_INDEX.json` 核对本地40个源文件的Git blob；使用的源文件必须在新包vendor内保留原字节、路径/commit/blob/SHA256/字节数映射。适配代码放在新包，不改vendor。若原字节模块有相对根路径依赖，适配层显式传入配置，不能隐式读取当前根目录的旧计划。

新包：`experiments/conditional_information_momentum_20261102_v3/`。必要模块可以合并，但职责必须覆盖：

| 职责 | 直接复用或最小适配 |
|---|---|
| 数据 | v2 `data.load_dataset(..., split_id='original', p=explicit_protocol)`；vendor audited_data 与 resolved_protocol.json，保留原manifest比较 |
| 模型 | v2 vendor `gradient_aug_20261031_v2.views.make_model`、其 model/reference/basic；不改forward/gate |
| 原目标/A1 | v2 `objectives.train_step/losses` 的 A0/A1 分支及其 source_losses/rng依赖 |
| EMA/环境 | v2 `protocol.update_ema/make_optimizer`、vendor setup/cpu_copy/rng_state/restore_rng |
| 评价 | v2 `evaluate.evaluate/metrics`，保留六格、pooled、traffic和undefined语义 |
| 新逻辑 | boundary_sampler、logit_capture、parameter_groups、objectives、optimizers/state、protocol、run、最小tests |
| 内存持有 | 优先复用已存在guardian的RAM IPC语义和epoch提交；只改新包入口/命名与新增状态，不引入新的持久化服务 |

不能直接复用旧runner的矩阵、费用门或旧F/J/TG路径。旧 `input_probe.probe_logits` 冻结参数且改gate输入导数，禁止作为本轮训练logits来源。

A0/A1复用先检查已有数据、初始化、loader顺序、A1 RNG key、50epoch、EMA、selected身份完全一致。A1需保留源 `study='EMA02V2'` 的Gaussian RNG上下文，不能因新study名改变噪声后仍声称同轨迹。开发日志可直接复用相符validation；已有selected/E50 test只在本轮最终冻结后导入分析。旧runner不保存E10状态，旧guardian进程当前已不存在；若找不到E10预测或状态，不能冒充E10前向。完整交付需要E10时，按原配置重放这些基线并计入物理重跑成本，科学轨迹去重。不能用旧A1超参下其他study的随机噪声替代。

## 3. 数据、模型和训练

真实CSV已存在：ML `/home_new/xzf/datasets/ml-1m/ml-1m.csv`（75,514,352 bytes）；KR `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`（416,370,411 bytes）。S0通过源load_dataset核对raw/数组/行号/encoder/scaler身份；本设计未重新计算哈希，不把文件存在当作数据核验完成。

原行序80/10/10，KR先切分再筛tab0/1/4，train-only编码/scaler和OOV0。ML训练域计数155949/318599/325619；KR172782/1743898/89452（以固定manifest运行核验）。原标签ML rating>3/rating>4，KR is_click/long_view。不得改变自然训练行、比例、标签或划分。无需恢复缺失的mdmt_next预处理目录。

fixed1，3域2任务，embedding16，4共享专家，fcn=[512,256,256,64]。batch40960、50epoch、尾批实际B，无早停。Adam lr=.01，betas(.9,.999)，eps1e-8，coupled wd ML1e-4/KR1e-5；foreach/fused/amsgrad=False，FP32，AMP/TF32关闭。每batch一次逻辑optimizer更新和一次固定参数EMA；ML rho=.9828205985452511，KR .995，theta0初始化，buffer复制，无反馈。

初始化使用源setup(seed)；源loader generator种子seed+11000，epoch randperm和尾批与源一致。新增条件/pair随机流独立于初始化、loader和A1，以固定hash(dataset,seed,epoch,global_step,stream)生成；不得用Python进程随机hash。所有PAIR臂共享相同流定义，不含arm或lambda。条件顺序固定(domain, task, positive_joint_label, negative_joint_label)字典序。

## 4. 条件、目标、采样

当前真实batch统计联合标签支持。c=(d,t,a+,a−)，同域、y_t正负且两类非空才有效。ML最多12条件，KR最多24；保留KR真实01，不制造ML01。

`w_c=n_d/(B*2) * n_(d,a+)/n_(d,t,+) * n_(d,a−)/n_(d,t,−)`，Z=sum(w)<=1。缺失域/任务不给其他域任务补质量。D_c初值.5；动态q使用旧D：`.25/K+.25*w/Z+.5*w*(.05+D)/sum(w*(.05+D))`；静态q=`.25/K+.75*w/Z`，不更新D。条件全局稳定编号，当前有效子集上计算w/q。

每步选择一个c，当前batch正负类别内分别均匀、有放回抽2048行索引，`R=mean softplus(-(z_pos,t-z_neg,t))`。一次真实forward返回源probability；六tower只读hook取得带训练图的raw logits，t*3+d组合成[B,2]，不改gate emb.detach，不使用logit(clamped probability)。hook在finally移除，不跨步保留图。

真实loss=`BCE(prob,y,reduction=sum)/(B*2)`。当前混合loss=`L_real+lambda*(w_s/q_s)*R`。原BCE系数始终1。lambda仅.05/.2；lambda=0整条pair/history路径旁路，调用源A0普通Adam，不抽pair或更新D/H。有效条件为空则本步辅助零，无伪样本。

动态D在本步结束用采样前forward的 `e=mean sigmoid(-(z_pos-z_neg))` 更新被选项 `.9D+.1e`。e、D、H均detach；未选D不更新。

## 5. 参数集合与优化公式

共享历史集合S严格为参数名 `shared_weight`、`shared_bias` 和前缀 `skip_conn.`、`star_mlp.`、`expert.`。按外层归属，不按内部domain_specific子串。其他embedding、slot、domain_expert、task_expert、gate、tower使用当前混合梯度的普通Adam；所有合法参数均可接受pair梯度。导出参数清单/shape/hash与互斥完整覆盖检查。

普通PAIR-Adam直接混合loss backward +标准Adam。ECCM/CV/PCOLL在同一图上先 `autograd.grad(R,S,retain_graph=True)` 得r，再混合loss.backward得到当前梯度。共享真实梯度用减去lambda*w/q*r还原，或独立g_real缓冲；两种路径须S0核验，wd只在optimizer中加一次。S意外None为接线故障；其他参数None不推进step/wd/m/v，显式零正常推进。

ECCM共享参数：所有当前有效c的active clock n_c每步+1；仅选中H_s更新 `H_s += .01/q_s*(r_s-H_s)`，未选H保持。`m_real=.9m_real+.1*(g_real+wd*theta)`；`a=g_real+wd*theta+lambda*w_s/q_s*r_s`；`v=.999v+.001a²`；分子 `m_real/(1-.9^step)+lambda*sum_active w_c H_c/(1-.99^n_c)`；分母 `sqrt(v/(1-.999^step))+1e-8`。step为参数级真实梯度时钟。H零初始化，使用更新后H；缺席条件冻结且不注入。q>0、alpha/q<=.96、w/q<=4，不裁剪。

CV用旧H计算 `C=sum_active wH-(w_s/q_s)H_s`；混合backward后共享grad加lambda*C，全部参数普通Adam。optimizer完成后H_s=.9H_s+.1r_s。无ECCM active bias correction、无1/q历史刷新。

PCOLL共享参数采用与ECCM相同的m_real/v及当前a；单个历史Q更新 `.99Q+.01*(w_s/q_s)*r_s`；分子 `m_real_hat+lambda*Q/(1-.99^n_pair)`。冻结解释：n_pair仅在本步存在有效pair时+1；无有效pair时Q/n_pair冻结且本步不注入Q，真实m/v仍正常更新。显式零pair梯度计一次有效pair更新。Q为全条件合并历史，q为抽样概率，命名不得混淆。PCOLL动态q/D、lambda、betaA与P11相同。

一个optimizer可分组dispatch标准/定制更新，但同一参数不能两个optimizer重复step。EMA在所有参数更新后一次执行，再更新D及统计。参数级step、active clock、selected count、PCOLL pair clock分别记录。

## 6. 状态、评价和不落盘执行

每条轨迹内存state包含model、EMA、optimizer（m/v/H/Q/所有时钟）、D、参数清单、独立随机流/loader/global RNG、epoch/global_step、最佳指标/epoch、selected/E10/E50 online/EMA状态、source/data/config hash。恢复验证通过cpu_copy/BytesIO或现有RAM IPC，不写.pt/.pth/.ckpt或参数数组文件。

每epoch评估完整validation online和EMA，各自以macro六格AUC严格>更新最佳，最早平局保留，排除E0。保存E10/E50和各自selected快照到RAM。主要决策只用fixed_ema selected，online为附加端点。六个逻辑端点若同epoch/同state可复用预测，但保留逻辑端点身份。

多进程训练由现有RAM owner保有已完轨迹状态，全部开发决定和S3配置/选点冻结后才统一test；若某数据S1停止，其最终配置和选点冻结后仍等全局决策冻结再test。验证日志足以驱动协调者，runner不读test作晋级。test完成并写齐指标后释放RAM状态。数据加载可以构造test数组，不调用模型test评价。

最小实现可在同一长期协调进程保有RAM状态；若用现有guardian只适配入口和state字段。禁止为避免RAM持有而每epoch预先test后封存结果。RAM丢失后从seed重跑，明确物理重跑，不捏造恢复成功；不能静默用缺状态的旧记录宣称全部端点完成。

## 7. S0和独立审查验收

先CODER_IMPLEMENTATION_REPORT与本设计交给独立reviewer；PASS或已修正阻塞的PASS WITH FIXES后才能真实S0及正式训练。coder只做必要CPU/basic检查，不分析研究结果。S0包含：

1. 同一初始化/真实batch下源A0与lambda0的probability、loss、梯度、一个Adam step、EMA、RAM恢复后下一step一致；FP32 atol1e-6/rtol1e-5，若同调用路径期望逐位一致并报告最大误差。
2. sigmoid(hook logits)==源probability；gate detach保持；每参数分组正确，shared expert内部domain_specific仍在S。
3. 微型标签系统精确枚举w/q界、CV期望、ECCM频率校正、先读后写；FP64 atol1e-12。ML无01，KR01保留，同域正负配对，缺类别/空条件和None/零/active/selected时钟。
4. 一个手算非零wd小向量测试ECCM/CV/PCOLL，检查wd一次、bias correction与单step；所有模式RAM恢复下一步一致，无跨步图。
5. 两真实dataset加载身份通过；全batch40960短smoke覆盖P11及CV/PCOLL分支，记录loss、梯度有限、更新次数、EMA次数、峰值内存及用时。无需完整epoch的smoke，不当正式重复；CPU小系统结果不代替真实batch测试。

只处理能改变当前训练/指标/结论的错误。修正后重做受影响检查；不增加无关测试或性能研究。

## 8. 有界阶段与冻结决策

S1：每数据×2022/2024，P00/P11各lambda .05/.2，16新候选轨迹；A0/A1共8逻辑参考，优先可证明身份的复用。各数据独立计算相对A0的配对validation fixed_ema selected差。若所有候选配置平均差<=0且均非2/2正，停止该数据。本设计冻结S2触发为存在平均差>0的配置；零附近2/2正本来蕴含均值>0，不另设事后“接近”阈值。

保护优先选lambda：相对A0，macro LL平均恶化>=.002且2/2不利，或任何同一cell/traffic/pooled AUC均值下降>=.002且2/2不利，标TRADEOFF。存在无保护触发配置时只在这些配置中按两seed validation macro AUC均值选，平局较小lambda；否则记录最佳tradeoff配置但不作为无代价总体方法晋级。

S2：对S1继续的数据，以P11所选lambda跑P10/P01/PCV/PCOLL各两seed（最多16条）。同时比较该lambda P00和P00自身最优lambda。强简单为已运行P00/P10中最高validation均值且无保护者，平局P00。PCV/PCOLL另一lambda仅在该臂对A0均值>0且与当前最佳候选差距<=.0005时补一次，两臂共最多8条；此条将“可能改变排序”冻结成有限可机械判断边界，不根据test补跑。每臂全局最多两个lambda。

主要历史候选开发晋级：相对A0均值>=.001且2/2正；相对best simple>=.0005且2/2正；相对最近历史控制>=.0003且2/2正；无保护触发。ECCM最近控制为PCOLL，CV/PCOLL可互为历史竞争路线，均须强简单对照。ECCM不胜PCOLL则停止ECCM。多个候选合格时选validation最高，平局优先CV、PCOLL、ECCM；简单方案同等/更好则选简单。简单方法可在对A0>=.001且2/2正、无保护时独立进入S3，不要求它超过自己或产生历史增量。全部不合格则保留A0，S3 NOT_TRIGGERED。

S3每个继续数据最多4个去重臂：A0、best simple、chosen primary、strongest other history competitor；冻结method/lambda/model/rho/全部超参后覆盖2022–2026。完整相符开发轨迹复用，新增仅缺seed/臂，分别记逻辑与物理计数。确认门沿06的效应阈值和至少4/5正，保护为4/5同向；三个未开发seed单独列出，不因三seed区间宽而追加训练。若chosen primary为simple，不强迫留复杂候选为主贡献；竞争控制最多保留已有开发最佳历史近邻。

完整开发<=48逻辑轨迹，通常<=40；S3<=40逻辑端点轨迹。复用开发后实际唯一轨迹更少。每条正式轨迹仍50epoch，不因阶段负趋势中途截断。物理重跑另计，不用“88”掩盖重放成本。阶段未触发写NOT_TRIGGERED及具体冻结门，不写阴性结果。

## 9. 执行、产物与分析责任

runner只按已审查配置执行并记录原始结果；协调者根据validation作阶段选择并签署DEVELOPMENT_DECISION/CONFIGURATION_FROZEN/SELECTION_FROZEN。GPU按每卡独立检查显存、功耗均<85%时尽量增加并发，并以真实smoke峰值防止已知OOM；不要求两卡同时满足，不杀外部任务，不改变batch。OOM先调整进程并发或等待空余，不静默缩batch。

显式使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。记录环境、GPU、代码hash、进程/attempt、数据/参数hash、各epoch loss、validation、step/forward/backward/EMA计数、抽样条件/w/q/D/H范数/时钟、pair原行数与索引指纹、时间与峰值显存。耗时包括额外autograd、数据加载、验证、最终test与失败/重放，不能仅报最后backward。

必交付SOURCE_MANIFEST、DESIGN_IMPLEMENTATION_MAP、CODER_IMPLEMENTATION_REPORT、REVIEW_REPORT/receipt、CPU_TESTS、STAGE0、RUN_MANIFEST、开发/配置/选点冻结文件、逐seed/六格/paired CSV、成本表、REPORT。统计单位seed，均值、样本SD、配对差95% t区间，开发n=2和确认五seed/三个未开发seed分开。主要分析EMA selected，同时给online/E10/E50与worst、traffic/pooled、各域各任务AUC/LL。身份记dataset/method/lambda/seed/epoch/estimator/state_hash，模型参数不落盘。

协调者报告稳定与矛盾现象、STRONG/MEDIUM/WEAK、替代解释、domain大小和baseline难度（分别度量）、是否小域收益/大域损失、训练阶段差异、机制未解决处，以及最能区分解释的下一实验建议；不为下一建议自动追加本轮矩阵。当前对分条件历史的反证优先由PAIR-Adam/PCOLL/CV完成。

## 10. 设计时就绪状态

GPU和PyTorch可见性由协调者已检查；两CSV存在，固定源40文件已核对；数据manifest和真实S0仍需运行。没有已知数据缺失阻塞。实现开始前本规范可冻结；正式训练前必须完成独立review和S0。基线E10状态缺失若无其他有效工件须重放，已明确成本与路径；不需要新增研究阶段。

## 11. 2026-11-03 独立补充机制实验执行规范

本节依据用户 `buchong.md`，仅补齐原2×2中的P01/P10。原S1停止决定、原S2/S3的NOT_TRIGGERED记录和历史结论保持不变。本次是停止后的独立补充研究，由用户明确授权固定lambda=.05；既有test已经暴露，报告须说明这一时间顺序。原八臂矩阵、搜索、PCV/PCOLL和S3均不进入本次队列。

### 11.1 唯一新增矩阵与代码复用

新增且仅新增：`{movielens,kuairand} × {P01,P10} × {2022,2024} × lambda_pair=.05`，共8条，每条50epoch。P01=静态条件采样+ECCM；P10=动态条件采样+普通Adam。原数据、模型、loss、EMA、初始种子、loader/pair随机流、batch40960及其他超参全部不变。

已检查现有 `protocol.py::config` 和 `run.py::job/schedule/freeze/final_evaluate`：P01/P10已实现；schedule允许S2显式dispatch且不要求同attempt先运行S1，因此无需新增算法或改变原runner。新建独立RAM attempt目录（例如 `artifacts/conditional_information_momentum_20261102_v3/supplement_20261103`），使用原guardian和原入口；不复用已释放模型状态的attempt1，不向旧dispatch添加任务。

最小执行配置为 `stage='S2'`、`approved_by='orchestrator'`，jobs逐一调用 `job(dataset,arm,seed,'S2',lambda_pair=.05)`。S2这里只是既有执行接口的合法阶段标签；补充配置记录 `study_role='independent_post_stop_supplement'`、用户任务来源和原attempt标识，报告中不得把它写为原S1晋级。coder责任仅生成/核对这8条dispatch及必要分析适配；没有实际缺口时不修改训练代码。

### 11.2 最小检查与旧证据复用

沿用原独立审查、14项CPU测试及真实S0的有效证据。正式启动前核对当前 `source_manifest()` 与原 `INDEPENDENT_REVIEW_RECEIPT.json`、`IMPLEMENTATION_SOURCE_MANIFEST.json` 一致；核对P01/P10仍按上述模式映射、8个job身份无重复、seed/epoch/lambda固定。将原PASS的 `STAGE0_REPORT.json` 复制进新attempt作为调度需要的已完成证据，补充配置记录其原路径、hash和reuse理由；不得改写时间或称本次新跑了S0。原峰值作为现有调度准入值，保持现有逐卡显存/功耗和RAM规则。

coder完成执行配置后仍由另一个reviewer核查本补充规范、dispatch、实际代码身份与coder实施说明，给出PASS后运行。训练代码未改不重新跑整个S0，不重新证明ECCM/CV代数；如有必须修复的具体接线变化，只重做受影响检查并重新审查。

A0、P00 .05、P11 .05直接复用原attempt1已冻结的数值记录，禁止重训/重放。复用前按dataset/seed/arm/lambda关联旧job、CONFIGURATION_FROZEN、SELECTION_FROZEN、result/test endpoint，核对source、data raw/arrays/row IDs/encoder/scaler、初始化和loader协议、50epoch、EMA参数及selected规则一致。新worker沿用 `load_dataset` 自动核对原固定manifest；新旧manifest比较身份字段，attempt/path不同不构成数据不一致。若缺少某条旧指标，先从原raw记录提取，不生成替代基线或凭报告四舍五入值计算。

### 11.3 冻结、测试与内存生命周期

启动前冻结8个配置及其source hash；所有新轨迹每epoch完整validation，online/EMA各自macro六格AUC严格提高才更新，保留最早平局，排除E0；selected/E10/E50状态只驻RAM。8条全部50epoch完成后，由协调者冻结所有新选点与状态hash，再调用原freeze/evaluate入口，得到8×2估计器×3端点=48个新test逻辑端点。阶段decision写明 `training_complete=true`、只结束本补充、`S3=NOT_AUTHORIZED`、无搜索；不重新调用原S1门决定是否跑这8条。

本次新test不得提前运行或参与配置/选点，旧test数值只用于冻结后的对照分析。新48端点全部落数值记录后释放RAM并关闭本次guardian。A0/P00/P11不需要也不声称有本次前向复核。物理新增=8条；复用逻辑轨迹=两数据×三方法×两seed=12条；2×2机制分析四方法共16条逻辑轨迹，另有4条A0参照。不得把旧轨迹计为本次新增。

### 11.4 冻结统计口径与问题

主口径fixed_ema selected；对每数据、每seed、每指标分别计算：

- 静态下ECCM简单效应：`P01−P00`。
- 动态下ECCM简单效应：`P11−P10`。
- 普通Adam下动态选择简单效应：`P10−P00`。
- ECCM下动态选择简单效应：`P11−P01`。
- 交互：`I=P11−P10−P01+P00`，先在同seed内计算，再跨seed汇总。
- 各PAIR方法相对强基线：`Pxy−A0`。

AUC正差为改善，LogLoss负差为改善；LogLoss交互正值表示组合比可加预期更不利，不与AUC符号混称。可附ECCM和动态选择各自两简单效应的均值作为边际主效应，但不能替代上述四个简单效应。对全部对比提供逐seed原值/差、均值、样本SD、n=2的95%配对t区间（df=1），方向计数；不能将六格、epoch、不同lambda或重复endpoint视为独立样本。

同时保留validation与test、online/EMA、selected/E10/E50的六格AUC/LogLoss、macro、traffic/sample-weighted、pooled、worst和checkpoint身份。selected交互包含方法各自验证选点策略的作用；共同E10/E50辅助区分阶段与选点。域大小沿用原train counts，难度分别用A0性能描述；检查是否小域收益/大域损失及宏/traffic反号。

报告回答静态ECCM是否有独立增量、动态选择是否有独立增量、交互正负/稳定性、值得保留的组件；每项结合A0、LL、六格/traffic保护与两个seed一致性，标STRONG/MEDIUM/WEAK。正交互本身不等于组合胜A0，两个seed正值不自动触发新阶段。原晋级阈值可作为量级参照，不能事后修改原停止决定或新增S3。未解决的条件历史是否胜PCOLL、CV是否足够仍如实保留，不为解释它们补跑。

### 11.5 同报告、同分支交付

仅在既有 `research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md` 末尾追加独立章节“2026-11-03：P01/P10补充机制实验”，结构为补充结论、矩阵与继承身份、四简单效应/交互/A0证据、域任务与阶段差异、成本、未解决问题和下一步决定。原正文的“S2未执行”等历史叙述保持其当时语境，新增章节明确本补充发生在其后；可在原报告开头加一行补充章节导航。

原表格/原raw不覆盖；新数值CSV/JSON、冻结记录和运行日志放既有报告目录下明确的supplement子目录或带supplement前缀文件。不新建另一篇REPORT或补充报告Markdown。coder/reviewer/runner的实施与审查说明追加到各自既有记录，保持此前证据。最终代码与原报告更新推送原 `26-11-02_v3` 分支，不创建新研究分支。
