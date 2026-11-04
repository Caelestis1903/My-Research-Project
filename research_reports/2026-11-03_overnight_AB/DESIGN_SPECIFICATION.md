# DESIGN SPECIFICATION — MDMT_OVERNIGHT_AB_20261103

设计阶段状态：COMPLETE。本规范把冻结规划落实为实现与验收合同；数学、矩阵、训练预算和评价阈值沿用已有决定。

## 1. 来源、优先级与范围

- 用户任务入口：`start.md`，GB18030，SHA256 `3386e3cb5a63c4641e6219e7e4d60c9cb41c2a2ed130d4bacb13947a4787376e`。
- 冻结规划：分支 `research-plan/26-11-03-overnight-ab-auc`，commit `c551dd95bb5c24fda0d5c2c1f79481bded23482c`，目录 `research_plans/26-11-03_overnight_AB/`，以00→01→02/03→04和`protocol.json`为实施依据。
- main代码基准：`92a592418d3f11a588f1e628fbe9364ceab71e18`；发表适配器固定来源：`d2fdd272a2cdf7da626e29b97f61555d25271f89`。
- history固定源码hash：`7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`；pair固定源码hash：`5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`。融合包另建source hash。
- 本轮所有科学种子仅为2022、2023、2024、2025、2026。无1–4 seed性能停止门。
- 设计不增加新模型、超参搜索、科研筛选、域重采样或第三个组合。自然train人口保持原样；C的既定内部probe及已授权的组合目标按03实现，不生成替代数据集。
- 新实现仅在`experiments/overnight_ab_20261103/`；新报告仅在`research_reports/2026-11-03_overnight_AB/`。history/pair/vendor/旧manifest/旧decisions/schedule/旧报告只读。
- 先恢复独立工作树；原目录无Git元数据时，从远端建立Git管理目录和基准工作树，记录实际历史目录与执行目录。历史原始产物仍从`/home/xzf/Code/phase0-3`只读接入，不能把新clone空artifacts误当历史缺失。
- 提交代码、规划、审计、标量报告到本次拉取的规划分支；不另创远端日期分支。执行分支可在本地用于隔离。

### 调度覆盖

`start.md`第12条覆盖protocol的`max_workers_per_device=1`和04的单worker限制：每张可用GPU独立检查，显存和功耗占比均低于85%时，应在RAM和已测峰值允许的条件下增加训练进程。不能等待两张卡同时符合门限，也不能以固定单worker上限限制已获准的并发。

保持冻结`protocol.json`原字节和科学配置hash，另存`RUNTIME_OVERRIDES.json`，记来源start路径/hash、原限制和生效策略。此覆盖只改变资源调度，不改变batch、精度、采样、优化步骤或seed。每卡用短时原子派发锁串行检查资源与登记worker；每trajectory独占owner锁防止重复训练。卡内可有多个本队列worker；已测峰值×1.2及RAM预留继续有效。首次未知峰值先完成smoke再逐个加进程。旧补评明确预留的设备继续保留；先识别外部owner，不用瞬时低负载抢占旧协调器将派发的GPU。

## 2. 科学矩阵与问题

| 集合 | 必需身份/端点 | 默认新增 |
|---|---:|---:|
| A：M3OE_FIXED1_ADAPTED、HYPERGATE、AUTOIFS×两数据×五seed | 30个online selected | 20条HG/AI |
| B：A0、C_MIX、P01、CP×两数据×五seed | 40个fixedEMA selected槽位 | 10条CP |
| A+B去重 | 60条逻辑轨迹、70个用途槽位 | 30条 |
| CG替代：A0、C_MIX、GAUSSIAN_A1、CG | 40个fixedEMA槽位 | 仅激活后10条 |

M3是近期A0同轨迹online端点的正式模型别名；A0 fixedEMA是内部参照，二者不增加独立n或重复成本。30条预计新增不是硬性重训数：完全同身份合法结果直接复用，缺失控制的必要重放另记原因与成本。

问题是：CP完整方案相对A0 online是否有宏收益和逐格保护；相对A0 fixedEMA、C/P各自的增量如何；两个组件的2×2交互如何；完整方案与三个发表模型分别如何比较。A与B互相独立准入，不能把另一条线的全部训练作为启动条件。

CG默认关闭。只有CP经有限工程测试确认不可执行/内存阻塞且一次针对性修复后仍阻塞，才激活两数据五seed完整CG；CP已完成部分保留。性能差不触发切换。核心完成后追加CG仍需事先授权，本次默认不追加。

## 3. 审计与复用

`inventory.scan`读取01列出的本地补评prepare/progress/launch ledger、guardian和完整报告，以及history、pair supplement、A0/A1原始记录。核对进程PID/start time/boot_id/cwd和现有owner，不以“远端没有结果”推断“服务器没跑”。

每条候选分配以下动作：

- `REUSE_DONE`：轨迹、选点、指标语义均匹配，完整V曲线或可信冻结选点、同一model hash的V/Test六格及聚合端点齐全，源checksum已记录。
- `REUSE_EVAL_ONLY`：合法selected状态仍在RAM或既有可读历史存储，训练完整，只缺统一评价。只预约评价；不复制/落盘模型，不release外部guardian。
- `WAIT_EXTERNAL`：外部活跃或已准备/认领，登记owner/PID/boot/job及状态，轮询接入。未派发但仍被外部拥有的条目不转新训。
- `REPLAY_MISSING`：无活跃owner、无同身份合法结果、无可恢复状态，具体缺失原因已查明。原外部队列条目需明确退出及放弃/失败记录后，才原子认领同seed重放。
- `INCOMPATIBLE`：保留独立协议组，不能混入本轮五seed。

`trajectory_id`绑定数据/六数组/row_ids/encoder/scaler、模型初始化、完整目标/优化器/预算/随机流及算法源码；device、路径、量子、attempt不属于科学身份。`selection_id`再绑定估计器/rho/初始化/buffer/阶段窗口/选点函数和指标。`endpoint_id`再绑定阶段/epoch/model hash/split。已有等价桥接记录为identity alias，不伪装源hash。

A审计至少60行：旧三模型×两数据×五seed30行＋新主表30行。旧30行全部`eligible_main=false`，原因`TRAIN_POPULATION_AND_PREPROCESS_MISMATCH`。旧曲线只能形成独立历史附录：完整曲线且选中状态/同epoch完整端点存在可标`RESELECT_ONLY_LEGACY`；缺状态或缺曲线分别记录，不为附录重训。

A/B审计字段遵守01，源不可访问写实际缺项。保留既有bridge证据，只核对本次导入文件checksum和身份，不重做整套控制验证实验。

## 4. 数据、模型与评价

只调用history的完整自然人口loader和原split。ML CSV `/home_new/xzf/datasets/ml-1m/ml-1m.csv`，raw SHA256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`；KR CSV `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`，raw SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。train域人口ML=[155949,318599,325619]，KR=[172782,1743898,89452]。路径别名必须验证相同内容，不修改旧包。

直接使用`bundle.features.as_dict()`；不重编码、不重划分、不调用旧baseline T0/F loader。M3/B输入x.T；HG/AI输入x。domain_indicator保留0/1/2，HG内部映射一次1/2/3，外部不加1。mask为y同shape全True；train函数只收TrainView，validation与test分离。

选点`AUC_SELECTED_V1`：每估计器各自于最终stage epoch1–50按validation六格等权macro AUC严格最大且并列最早；Auto只允许reuse1–50。best初始负无穷，不设epsilon，不读LL/test，不允许E0/search/跨seed共用选中epoch。六格缺失时macro为null，阻塞该评价，不nanmean。

HG/AI logits只做一次FP32 sigmoid，再使用同一history指标定义；M3/B保持源概率评价。eval/no_grad、无增强，保存并还原model training状态。每个端点V/Test对应同一真实model hash，保存完整buffers/stage metadata。E10/E50是预定义诊断，不能替换selected。输出六格AUC/LL/count/positive/negative、macro、样本加权格均值、各domain/task、pooled task mean AUC/LL。

### A固定训练

M3：当前history.make_model，fixed coefficients全1，embed16、4专家、FCN[512,256,256,64]、batch40960、lr=.01、50epoch、Adam(.9,.999),eps1e-8；wd ML1e-4/KR1e-5。正式身份写M3OE_FIXED1_ADAPTED。

HG：固定adapter及其原生objective，embed16、4专家、expert[256,256]、gate[256]、tower[256,128]、relu、dropout.3、无BN；三门全开，aug_dim16/aug_dims[64]/aug_dropout0，domain/task penalty均.1；batch1024、eval512、lr=.001、wd0、50epoch。配置来源标签保留UNVERIFIED_PROPOSAL。

AI：固定adapter、MLP[1024,512,256]、rank ML2/KR16、dropout0、无BN、gate_penalties=[0,0]；batch/eval4096。search10，base lr=.0003/wd=.00003，gate lr×.01/wd0，两组互斥且覆盖trainable参数；reuse50，lr=.0001/wd=.0003，新建optimizer。

HG/AI保留原生objective及辅助项，不能替换统一BCE；保留源优化器语义与初始化。HG epoch argument1..50；AI search0..9、reuse101..150；PCG64 SeedSequence=[seed,dataset_index,method_index,formal_phase_index=1,epoch_argument]，ML/KR和M3/HG/AI索引沿源定义。尾批保留，无早停。M3/B继续torch.Generator(seed+11000)源顺序。

AI阶段顺序必须是：进入零基search9之前捕获rewind→完整search9→导出最终selector→新build模型load rewind→import最终selector（含gate embedding）→ticket=True,temp100,freeze gate→新reuse Adam。search温度1到100、乘100**(1/9)。阶段转换与search最后epoch一起原子RAM提交；commit_index单调跨60epoch。恢复同时重建ticket/temp/requires_grad/各optimizer，不能只load state_dict。

### B唯一融合

共享集合S严格为name等于shared_weight/shared_bias或以skip_conn./star_mlp./expert.开头，保持named_parameters顺序。S和ordinary互斥且完整覆盖，记录manifest/hash。

主组合每真实batch先按原P01采样和forward/backward，再按原C产生probe。context固定study=EMA02V2,split_id=original，沿用原donor/PAIR随机流，日志不消耗训练RNG。PAIR静态lambda=.05、2048 pairs、alpha=.01、bias beta=.99；不补造联合标签状态。C轮转一个非空域、两个任务clean/mixed probes、tau=.1、kappa=1、gamma=.2，直接用源project_history，不把probe加进loss或p.grad。

对共享参数，记r为未乘lambda的重要性PAIR梯度，a=lambda*(w_i/q_i)*r，b=混合目标p.grad+wd*theta，u=b-a。先按原P更新active clock和选中h；仅选中h更新，所有active n递增；pair=None时H=0且旧表保留。随后：

```
v <- .999*v + .001*b*b
D <- 1/(sqrt(v/(1-.999**t))+1e-8)
c <- source.project_history(u,D,probes,kappa=1,gamma=.2)
m <- .9*m + .1*c
H = lambda*sum_active(w_j*h_j/(1-.99**n_j))  # pair=None时0
theta <- theta - lr*D*(m/(1-.9**t)+H)
```

ordinary只用原混合p.grad做一次Adam（coupled wd一次）；全模型只update_ema一次，global_step一次。v不得使用c²，m不得重复加PAIR，H及最终numerator不再投影。源w/q和alpha/q界限、active时钟、浮点计算次序保持。

关闭组件时直派原生A0/C/P入口，不经融合近似。固定EMA初始theta0、参数EMA/buffer源复制、非反馈；rho ML=.9828205985452511/KR=.995。两数据50epoch、batch40960、FP32无AMP/TF32。Gaussian替代按03既定目标函数与HistoryOptimizer(current_gradient_repair)，配置分离c_mix.kappa=1和gaussian.kappa=.05/.1；默认不运行。

低频统计step1及每20步只用已有tensor记录投影、梯度/历史范数、pair比率/时钟、更新范数、EMA距离和成本，不新增forward或性能门。

## 5. 最小实现分工与依赖

以下都是现有02–04要求，不扩展新公共框架。

| 文件/责任 | 最小实施方式 |
|---|---|
| config.py | 加载冻结protocol/plan，身份hash与runtime override；拒绝错seed/arm/rho/预算/阈值 |
| data_bridge.py | 调用history loader，路径别名、只读views与schema；复用原数组hash检查 |
| vendor/sota2526/、published_models.py | 按固定commit获取HG/AI及真实依赖，manifest登记path/blob/SHA256；直接factory映射，不导入旧BASP训练入口 |
| published_training.py | 原生目标/optimizer/shuffle，create_session及advance_epoch，AI阶段转换 |
| optimizers.py、worker_b.py | 唯一SerialCPOptimizer和融合训练；原生单组件与有条件CG入口 |
| selection.py、evaluation.py | 单一严格AUC选点、冻结、同hash概率评价、原子标量写入 |
| state.py、guardian.py | 完整Session CPU RAM快照、BytesIO IPC、RNG恢复、authenticated AF_UNIX |
| inventory.py | 01的逐轨迹审计、外部认领与合法端点别名/复用 |
| queue.py、run.py | 04全部CLI、资源/owner/公平epoch调度、自动评价汇总 |
| reporting.py | 04规定全部表、五seed配对/2×2、成本、进度及最终报告/台账增量 |
| tests/ | 下节限定检查，复用已有机制fixture，不扩展科研矩阵 |

CLI必须提供prepare/smoke/run/run-job/status/pause/resume/retry/report和内部_guardian。run-job也需owner、设备派发与具体family工程通过，不提供绕过入口。prepare幂等；身份不一致不能续同attempt。run自动完成逐seed评价、汇总和报告。

## 6. RAM、队列与恢复

权重/模型/optimizer/RNG张量只驻RAM；不调用torch.save、磁盘pickle、参数memmap等。数据数组缓存沿原许可只读；持久化只允许代码、配置/数据身份、训练标量数据/日志和报告。guardian authkey仅本地0600元信息，不推送。

guardian独立进程，register的launch_index不限；正常每epoch重新register/yield不增加failure_count。commit_index恰为前值+1，绑定job/source/config/data，完整RAM快照替换后写标量ACK再应答。支持allocated/commit/yield/failed/train_complete/get/release/status/shutdown。train_complete验证HG/M3/B final50和AI search10+reuse50。

snapshot包含online/EMA、m/v/h/clocks/ordinary optimizer、全RNG、loader顺序、global_step、阶段、selected/E10/final/buffers、参数manifest和实际成本；AI另含rewind、selector、ticket/temp、冻结状态。相同model hash可在RAM内部去重。恢复时先build model/optimizer，再restore RNG，之后才训练。

每job从READY→CLAIMED→RUNNING，整epoch→RAM提交→PAUSED_RAM，最终TRAIN_COMPLETE→SELECTION_FROZEN→EVAL_RUNNING→DONE。额外状态为WAIT_EXTERNAL/WAIT_RESOURCE/RETRY_PENDING/BLOCKED_ENGINEERING/STATE_LOST_REPLAY/CONDITIONAL_NOT_ENABLED。失败只阻塞具体job，其他已通过family继续。

A/B在单卡按一个完整epoch轮换；多卡/卡内并发优先给两lane各实际机会，lane内优先恢复已开始job，限制驻RAM模型数。另一lane READY时最多等待当前epoch；避免重复开启过多seed堆积RAM。RAM预留max(6GiB,total*10%)，计入guardian快照、worker及提交时新旧双份峰值。

coordinator恢复先对账仍活跃worker，不能重复派发；worker死亡从最近完整epoch重放并累计实际费用；评价失败只重试评价。主机/guardian失去RAM后，仅缺失身份同seed从头重放，已DONE的签名标量仍可复用。基础设施首次失败最多2次自动重试；普通yield不算失败。OOM不变batch/精度/目标，等资源恢复。pause/SIGTERM停止新派发，当前epoch完整提交，guardian保留；status/report不训练。

## 7. 独立审查与最低验收

coder和reviewer必须不同子agent；非run阶段继承主agent模型。coder只实现本规范，写IMPLEMENTATION_REPORT和逐family准入，不分析科研结果。reviewer独立阅读规范、实际实现和coder报告。只有PASS或已修完blocking的PASS_WITH_FIXES才交runner。

逐family依赖闭包冻结IMPLEMENTATION_LOCK；A_HYPERGATE/A_AUTOIFS/B_CP/NATIVE_REPLAY分别可独立审查与开跑，B_CG仅条件启用。已通过家族的共享代码只读；另一家族开发不能失效已运行身份。先完成一个可运行家族及共享调度，不等全部可选部分才给GPU机会。

必须检查且只做这些检查：

1. plan self-test/materialize：80全身份、60核心、30预期新增、A30/B40槽位、两数据五seed、替代默认关闭。
2. 数据/schema/domain映射一次、train-only、自然人口与hash、训练入口不读test。
3. 严格AUC argmax/最早并列/排除E0及search/LL不介入/六格全需；同model hash的V/Test、一次sigmoid、未缩放LL。
4. 小张量B数值检查：四开关原生分支；共享/ordinary更新各一次、L2一次、v=b²、m无PAIR重复、h及n语义、pair=None、C空约束、probe不污染p.grad、kappa隔离。
5. 连续两epoch与1epoch→RAM→1epoch的model/optimizer/EMA/RNG/selection/数据顺序hash等价；AI覆盖search9→search10→reuse1；正常量子无限register不耗重试；重启不重复claim。
6. 无权重落盘、外部预留/trajectory owner、单卡资源门并发、epoch公平与STOP、调度不读test性能。
7. 两数据各B、HG、AI真实完整原batch smoke：forward/backward有限、原batch可运行、显存/RAM实测。smoke状态丢弃，正式seed重置；不形成性能选择。
8. 汇总门边界+.003/−.001、负seed不增硬门、LL独立、缺seed INCOMPLETE、重复seed/协议不混合、CI不作门。

每family ENGINEERING_ACCEPTANCE绑定新源码hash、命令、结果和阻塞；审查FAIL返coder。配置冻结后逐seed训练完成即冻结合法selected并评价，DONE写文件checksum后才能release RAM；外部补评仍用其原全队列test冻结规则。

## 8. 运行与交付准则

新训练前CONFIGURATION_FROZEN包含核心/替代矩阵、启用规则、数据/模型/超参/预算/选点/指标/阈值/source和科学配置hash。runner只记录原始结果和状态，不给科研结论。运行完成后由主agent分析。

报告依04完整输出ENDPOINTS/ABSOLUTE_SUMMARY/PAIRED_SEED/PAIRED_SUMMARY/FACTORIAL_SEED/FACTORIAL_SUMMARY/VALIDATION_CURVES/COSTS/A_REUSE_AUDIT/B_REUSE_AUDIT/DECISIONS/PROGRESS/FINAL_REPORT/LEDGER_APPEND，逐端点带完整身份、六格和聚合量。成本去重原始训练/本轮新增/失败重放，区分search/reuse、真实optimizer调用/标签暴露、GPU/worker时间、峰值显存/RAM及附带EMA成本。

以同协议匹配五seed逐seed相减，SD用ddof=1，描述性t95区间=均差±2.7764451051977987*SD/sqrt(5)。mean macro ΔAUC>=.003判宏达标；每格mean ΔAUC>=−.001判保护；两者同时满足才写“AUC合格且满足逐格约束”。mean macro ΔLL<0独立改善。不足五seed只写INCOMPLETE，正向seed数、CI、单格单seed、traffic方向不增门。交互逐seed先算11−10−01+00，含六格和LL，不把交互正值变额外合格门。

完整方案总收益、fixedEMA组件增量、相对单组件增量和相对三个发表模型比较使用不同comparison_id；不因组件增量未达.003抹去总收益，不把只胜M3写成胜三个模型。报告保留实际域人口、域/任务结果及训练阶段，但本任务不提出或检验域大小等于难度的命题。

主agent分析需区分重复稳定现象、矛盾结果、竞争解释和当前证据缺口，证据强度标STRONG/MEDIUM/WEAK；不预设课程/动量有效或M3困难域必败。依据本轮配对和2×2决定是否保留组合；仅提出最能区分竞争解释的下一实验，不自动扩大本次矩阵。

全部核心端点齐备且A/B五seed配对完整才写FINAL_COMPLETE；存在工程/外部缺项则列PARTIAL_ENGINEERING/WAIT_EXTERNAL和确切身份。夜间时长不构成性能停止条件。最终发布同拉取分支的代码和完整报告；main台账如需回填先读最新版本，仅增量合并，不覆盖并行成果。


## 9. 三个coder的共同API与文件所有权

实施前共同采用以下接口，全部job/config/记录使用普通dict，不新增框架。运行时coder拥有公共模块；A/B coder只依赖已固定公共函数签名，不各写第二套guardian/selection/hash。

### 公共数据和session合同

`data_bridge.load_bundle(dataset, cfg=None) -> (bundle, manifest)`保持源bundle字段，`data_bridge.training_view(bundle)`返回仅含train/val/schema及原row ids的view，不含test数组。`schema_of(bundle)`直接features.as_dict()。评价器另取原bundle的test view。

`job`至少包含job_id、trajectory_id、dataset、seed、arm、method、lane、config、config_hash、source_hash、data_hash。`job["config"]`是归一化的该job科学配置；完整原protocol另存在job["protocol"]，避免A/B各自寻找配置。

A的`published_training.py`与B的`worker_b.py`各实现相同四函数：

```python
create_session(job, bundle, device) -> dict
advance_epoch(session) -> dict             # 原地推进恰好一个完整stage epoch
export_session(session) -> dict            # 完整CPU RAM快照，禁止文件IO
restore_session(job, bundle, device, snapshot) -> dict
```

create/restore中的bundle是上述无test的training view。advance_epoch不启动子进程、不访问guardian、不写权重、不执行test，不因性能提前结束。runtime负责调用、RAM提交、标量持久化和最终评价。各family可在自己的session中保存私有字段，不依赖另一个family。

所有live session共同键为job、device、stage、next_stage_epoch、completed_search、completed_final、commit_index、global_step、model、optimizers、rng、selection、snapshots、cost、complete；B另有ema。optimizers用dict：HG={"base":...}，AI search={"base":...,"gate":...}/reuse={"base":...}，B={"main":融合或源optimizer}。stage为train/search/reuse；commit_index为实际已完成stage epoch总数。complete仅在HG/B final50或AI search10+reuse50时True。

`advance_epoch`返回EpochRecord普通dict，至少含job_id、stage、epoch、commit_index、completed_search、completed_final、global_step、validation、model_hash、selection_hash、train_objective、cost、complete。validation对HG/AI是online指标；B为{"online":指标,"fixed_ema":指标}。cost为本epoch增量且各数值有限；record不含权重张量。search日志保留，selection不接纳search。完整snapshot在export_session中，含model/optimizers所有state_dict、RNG和家族阶段元信息；restore必须保持与连续运行等价。

`session["snapshots"]`统一为{"online": {"selected":snapshot,"E10":snapshot,"E50":snapshot}, "fixed_ema": {...}}，A仅online。每个端点snapshot包含model_state（CPU state_dict）、stage、epoch、model_hash、validation、model_meta。AI的model_meta至少含ticket/temp/gate冻结和selector恢复所需信息。evaluator使用家族factory重建该端点，核验hash后统一评价。selected快照在严格V改进时替换，E10/E50按final阶段计数；同hash允许runtime内部去重。

### 公共函数签名

runtime提供：

```python
state.cpu_copy(value)
state.state_digest(value) -> str
state.capture_rng() -> dict
state.restore_rng(state) -> None
selection.consider(tracker, stage, epoch, validation, model_state, model_meta=None) -> bool
selection.freeze(tracker) -> dict
evaluation.evaluate_model(model, bundle, split, method, batch_size) -> dict
```

tracker为dict，创建时由family设{"final_stage":"train"或"reuse","best_score":-inf,"selected":None}；consider仅接受最终合法stage，使用validation六格等权AUC，严格大于更新best和CPU selected snapshot，返回是否替换。validation统一采用history.metrics返回结构；runtime编写一个共享适配函数读取六格，禁止A/B各自解析成不同macro。A的evaluate_model负责layout和一次sigmoid；B直接复用history.evaluate。若现有source指标结构需要附加字段，公共适配可加字段，不改源含义。

### 三方文件所有权

- coder_A：vendor/sota2526/、published_models.py、published_training.py、A专属tests、CODER_A_IMPLEMENTATION_REPORT.md。
- coder_B：objectives.py、optimizers.py、worker_b.py、B专属tests、CODER_B_IMPLEMENTATION_REPORT.md。
- coder_runtime：config/data_bridge/state/guardian/selection/evaluation/inventory/queue/run/reporting、共享tests、CODER_RUNTIME_IMPLEMENTATION_REPORT.md。
- 三方均不覆盖其他所有者文件；接口调整先发主agent协调。每方按实际新增依赖最小vendor，不引入新Python依赖。

### 已由主agent确认的审计输入

本地`artifacts/mdmt_five_seed_completion_20261103/attempt1/COMPLETE.json`存在，原30条新增补评训练和180测试端点已完成、旧guardian已SHUTDOWN；`research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv`完整660行，优先直接核对身份后复用，不重放已齐全控制。三个外部固定源码根均已存在且commit核验匹配。runtime仍逐导入文件记录checksum/别名，审计表填写已确认完成状态。


