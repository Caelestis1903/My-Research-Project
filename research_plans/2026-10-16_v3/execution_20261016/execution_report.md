# v3 实验执行报告（RUNNING，非终局）

## 当前已核实状态：V11 诊断分析完成，全部实验继续执行

V11 四种子状态训练最终预算为 **250/150/100/200**；全部12对最终状态和27对历史读数已完成并通过156000次选择索引检查。与native最终50/100/50/50的16父读数已完成聚合、联合边际、支持交互和原始六单元数值重建（最大误差8.89e−15）。[完整状态分析](V11_fullstate_final_diagnostic_analysis.md)同时报告绝对AUC/loss/J、逐域逐任务、种子CI、训练阶段与预算限制。主控自审，independent_review=false。

原始22轮窗口的v-only AUC交互 +.012912（四seed正），同时SN绝对变化 −.012157（四seed负）、SB仅 +.000755（2正2负）；共同E50两臂AUC均四seed改善，交互却不稳定。β2=0最终AUC=.5且loss/J很高；m-only长期有学习，AUC与loss不总同向。以上支持整个更新状态映射的条件影响（MEDIUM），不识别唯一、必要的长期历史机制。新增Q44（BOUNDARY/MEDIUM）记录联合窗口从原始native22缩到4后，绝对收益估计随窗口改变；不能将不同窗口混为同一健康恢复证据。own_final_epoch仅为selected_fixed_budget标签别名，不能双计独立结果。

V5/V8 seed25十个实际350训练现均COMPLETE并满足平台，最终匹配预算正式认证为FINAL350（root60关键哈希/70归档PASS，不是350硬上限）。四对完整350读数已核验16000次选择，strong完整350由GPU81/PID505078执行；四CPU前缀队列17对及最终sampler/weight队列已启动。最终六个V8单元审计采用每批2个GPU worker，首批clean GPU82/PID507092实际执行。V5 SGD四seed最终匹配50/50/50/350的12父性能已聚合并复核230400值，最大误差3.80e−16；[分析稿](V5_SGD_final_matched_analysis.md)保留ORDER第50轮四正但长程seed25反向、自选差距主要伴SN下降。最终采样身份/22前缀仍需闭合后继续V7。

问题登记共44项；最终Newly Exposed Questions至少覆盖Q28–Q44共17项，六字段工作附录已保留。Q40的诊断最小补充已完成、定为BOUNDARY/MEDIUM，最终新测试仍待全条件冻结。test_readout_frozen.json不存在，本次新test性能未计算；历史test已暴露，不声称pristine holdout。V5/V7/V8、最终遥测、V12/V13、冻结测试和最终报告/归档推送仍有义务。下文初始快照和过程更新仅为历史，不覆盖本段和实际收据。

任务继续执行，未达到全部流程完成或最终提前停止条件。完整计划来自 origin/main@d18f194，工作分支 `codex/execute-2026-10-16-v3`。最近核实的远程里程碑为 `c79412e99c86744734c6ad33c5a011e9b9c21404`；之后仍有持续实验产物，该提交不表示最终验收。

当前汇总优先于下文保留的早期执行记录：V1–V4、V6、V9、V10的规定diagnostic读数与分析均已完成；V5 Adam全部最终匹配读数与104,064值重建PASS；V11四种子影子历史统计及1,740行方向补充PASS。V8 seed23最终200及六个100/150前缀全部核验。V11完整状态seed23最终150、seed24最终100已满足冻结预算条件，seed23全部150配对与三个100前缀读数PASS，seed24三个100配对也已全部核验12000次选轮。seed22实际150已完成/root36critical+archive42PASS，但β2=0/SB末段非平台且无eligible窗口，故全六条件必须至少200；复制PID402370实际运行，保留三个150前缀读数义务。seed25六个150轮条件已实际训练。

V5/V8 seed25的clean250两臂均未满足selection过拟合确认且训练目标仍有>.005改善；各250预测、48,000条必要epoch bootstrap行及全部CI主核验PASS。因此整组十条件必须共同至少300轮，规则没有300硬上限。十个200→250复制全部完成，五对启动前核验和整批复制时间快照对照通过。clean/PERM两对300副本均复制核验PASS并已实际启动协调队列（PID392716/399314）；ORDER实际250/root12critical+archive14PASS且300副本PID398619复制中。mild/strong仍在250训练，之后也按全组匹配预算准备300。全部旧100/150/200前缀和新增五对250前缀共17对保留为强制读数。V7仍等待完整SGD V5最终分析，不能因NO-GO直觉跳过。

`test_readout_frozen.json` 仍不存在；尚未计算新v3 test性能。V12/V13、终局报告、最终test与归档推送继续执行。新增Q42已记录全局等范数不能独立隔离块间尺度与方向的识别边界；最终Newly Exposed Questions须覆盖Q28–Q42。以下较早数字和“正在执行”段落保留为过程历史，最新逐项状态以 `orchestrator_checkpoint.json` 和实际producer/analysis收据为准。

## 规范、工作区与环境

`start.md` 按 GB18030 解码，较新且明确的 §11 要求主 agent 完成设计、实现、自审、科学分析，仅机械运行委派既有 luna-runner。因此全部新实验保留 DESIGN→IMPLEMENT→SELF REVIEW→RUN→ANALYZE 记录，`independent_review=false`，没有虚称独立 reviewer。规划与启动文件末尾旧 v2 路径冲突时采用明确指定的 v3 目录，并保留原文件。

用户原工作区 `/home/xzf/Code/phase0-3` 无 HEAD、文件未跟踪；旧 v2 worktree 也有已有改动，均保持原状。本轮另建 `/home/xzf/Code/phase0-3-20261016-v3-git`。复用 Python3.13、PyTorch2.9.0+cu126、CUDA12.6、2×RTX3090 24GiB、32 CPU/64GiB 主存及原有数据；未重装环境或重下载数据。完整产物通过 `runs/` 指向 `/home_new/xzf_phase0_3_v3_artifacts_20261016`。原始权重、逐行预测、batch 索引、完整梯度不上传 Git，以生产者 SHA 索引本地保留。

GPU 跨所有队列合计每设备最多5进程、显存<90%，每个新队列显式引用全部当前前序状态并去重实时 PID。不能只计本队列或等两卡同时满足容量。CPU 经逐次实际资源复核，现上限12个双线程数值 readout +1个CPU cell/sampler；最多两个 GPU cell worker 占用现有 GPU 五进程配额。等待训练或其他队列的 stdlib 等待器不提前导入 numpy/torch。`CPU_readout_capacity_review_v4_SGD100.json` 和各 activation review 记录了配额归属，防止把已释放槽位重复分配。

## 已完成并验证

| 阶段/对象 | 实际状态 | 证据入口 |
|---|---|---|
| V0 实现/历史审计 | COMPLETE | V0_decision.json |
| V1 同目标、512批完整梯度、确定性对照 | COMPLETE，主agent数值复核 | V1_decision.json、V1_analysis.md |
| 八组完整 domain-task 梯度 | COMPLETE，独立数值复算身份 | V1_task_direction_decision.json |
| 16个LR pilot | COMPLETE，selection-only冻结 | calibration_frozen.json |
| V2固定参考16条50轮训练 | COMPLETE | trajectory_manifest、archive/formal |
| V2固定Adam四配对1000/200读出 | COMPLETE，已聚合/主agent复算 | V2_fixedAdam_analysis.md |
| V4初始24条50轮训练 | COMPLETE | V4_training_manifest_attempt2 |
| V4校准Adam四配对初始50读出 | COMPLETE，已聚合/主agent复算 | V4_calAdam_until050_analysis.md |
| V4 Adam seed23两臂100轮 | COMPLETE且均平台，最终100 | V4_Adam_seed2023_extension100_train_budget_check.json |
| V5前三种子SGD PERM/ORDER训练与操纵审计 | 12条训练COMPLETE，600 arm-epoch身份PASS；性能读出尚待完成 | V5_sampler_first3seeds_primary_verification.json |
| V8精确别名复制2×/4× | 16个耦合映射/梯度/更新PASS | V8_duplication_decision.json |
| V8 clean逐cell审计 | 8/8完成、主agent复算 | analysis/V8_clean_cell_primary_until050/ |
| V8污染seed22–24初始训练 | 12/12初始50轮COMPLETE；需另行预算匹配 | archive/formal/V8*、V5_V8_completed_initial_train_budget_audit_v2.json |
| V9 MovieLens | 8条训练、4配对读出及主agent分析COMPLETE，test仍未读 | V9_diagnostic_analysis.md |
| V10 AdaTask初始训练 | 8/8初始50轮COMPLETE，原始配对读出继续 | V10_initial50_train_budget_verification.json |
| V6观测读出 | 固定θ几何+320条移动θ阶段读出COMPLETE，干预因果部分仍待 | V6_geometry_observational_analysis.md |

## 正在执行与依赖

V2剩余SGD、V4校准SGD及2B、V5全部配对、V8污染和V10 AdaTask的1000次selection行bootstrap/200次diagnostic读出仍按冻结命令运行。行重采样、epoch重新选择和四训练种子不确定性分别报告，不能用其中一类替代另一类。已完成的点轨迹不等于完整 bootstrap 已完成。

训练预算只用 train objective 与 selection 规则。V4 Adam seed22/24/25最终50，seed23最终100；后者两臂最近5轮相对前5轮改善 −.00240046/+.00453707，均满足<=.005。SGD seed25原B0 SN改善.00708179，全部可能的selection AUC下降窗口都不满足LogLoss恶化必要条件，故按原规则要求B0/2B×两q至少100。四条副本现已完成100，改善率依次+.00337749/+.00132855/+.00284449/−.00105820，全部达到冻结平台标准，最终共同预算100，无需150；50/100原始产物均保留，100轮完整读出继续。

V8轻度污染 seed23 SB改善.00581024，全部四个候选下降窗口均不满足selection LogLoss恶化，故clean/mild/strong×两q六条配对轨迹至少100。这里的clean100是V8共同污染预算条件；V4该seed自身平台50的判定仍保留，V5初始匹配50的对照不与clean100错配。V10 seed23因匹配Adam最终100而两臂延长；其余AdaTask七条> .005的条件仍需完成selection判定。未从diagnostic决定预算。

为保持正在读出的50轮输入不可变，使用 `v3_prepare_continuation_20261016.py` 完整复制prefix到新目录：SGD25四副本676个产物SHA/4.484GiB、AdaTask23两副本338 SHA/2.729GiB、V8seed23六副本1014 SHA/6.737GiB。每个文件核验不同inode、生产者SHA、原收据未变；主agent另核对config/trajectory/selection/resume关键SHA后才批准GPU恢复。恢复模型、优化器、RNG、sampler和计数，50轮原始config作为历史保留，实际100由新的extension descriptor/launch/receipt说明。这些副本延续同一轨迹，不增加独立seed。

V5最后SGD seed25四个条件与V8 seed25四个污染条件已按最终100预算全部审查并入队，因此原始V5共24个、V8污染共16个条件均已接纳；部分仍按全局GPU guard排队或训练中，不能记作全部完成。V7局部反事实依赖V5分析；V11只有真实四种子 primary interaction 门控触发后才设计状态分解，不能由初始固定horizon或某个seed激活。V12反事实图、V13方法门控、终局报告、全部条件/问题冻结和最终test尚未完成。

## 异常、修正与可追溯性

- V4首次8个启动在模型初始化前因日志先建立run_dir触发FileExistsError。原目录/状态/日志保留，修复仅移动日志路径，24条科学命令逐项一致；第二次全部训练完成。见V4_attempt1_postmortem及startup_fix_review。
- 聚合器首跑重复arm关键字导致异常。单点修复后重跑并以原六cell表交叉核验；未改变训练或冻结readout。
- 原V5九对读出误按Adam先行而等待未完训练。确认无实际数值读出/输出后只停止等待器及调度器，保存原始启动记录，科学命令逐项不变地划分为SGD优先与释放槽位后读出。见archive/dispatch_corrections及对应自审。
- Q31核验80个pilot/finalist epoch前缀一致，明确pilot和正式种子重叠；启动器未保存某次wait返回码则记录null，不反推exit0或重复计算。
- Q32历史SGD同batch/参数/预测身份确认，旧新效应差主要由评分总体改变加小量重选构成，未发现训练漂移。
- Q34只读遥测一阶矩标签混入二阶矩范数；原值保留，80,000条平方差恢复及真实状态八块oracle通过。最大相对修正22.51%；实际训练/性能未变，后续只用corrected_m_norm。
- Q35跨epoch稀疏观测lag并非实际steplag；原始值保留，轮内batch_index≥lag读出完成，数量5/4/1明确。真实轮首lag1跨44step已核验；不改训练、不声称整轮自相关。
- 混合50/100预算图只在全部seed均有观测的epoch画粗均值，保留所有实际seed细线并输出支持数量表，防止一条延长轨迹被误看成四seed均值。

所有数值来自真实产物。8个环境×seed validation拆分在新训练前冻结；历史全集validation/test已暴露，因此新diagnostic只对本轮调参独立，不称历史完全untouched。当前 `test_readout_frozen.json` 尚不存在，未执行新v3 test性能读出。恢复入口为 `orchestrator_checkpoint.json`、逐阶段 `execution_ledger.json` 与各manifest/status/launch；全局状态持续为RUNNING，不以保存里程碑结束任务。

Q36新增推断边界：冻结onset程序使用观察到的prefix peak并扫描多窗口，没有构造同时置信带或证明familywise错误控制。标签与预算充分性仅按既定操作性规则解释；全部fixed epoch与AULC10/20保留，不修改阈值、不新增未经预设的结论。详见onset_inference_scope_audit.json。最终test汇总器已补齐AULC读出，并要求所有汇总清单SHA在test开封前冻结；test仍未读。

V8 seed23的100轮六条轨迹现已完成归档，但clean SN改善.00579043、mild SB改善.00664922且所有候选selection下降窗口仍不能满足LogLoss恶化。因此按原规则要求全六条至少150；与冻结readout一致的NumPy mean/argmax复算再次确认零可确认窗口。第二段前缀复制保留旧continuation_source.json、另写continuation_source_to150.json，1944个生产者文件/11.1035GiB在新metadata写入后由主agent逐一复核通过，150轮GPU恢复已实际启动。旧50/100输入与其完整读出均保留，不把中间100预算误称终点。

AdaTask前两种子的1000/200完整初始50轮配对读出与主agent六cell复算均通过，最大误差2.22e-16。seed22按selection过拟合规则最终50，seed23匹配Adam100且自身两臂100均平台，最终100；seed24/25仍待。无四seedAdaTask机制结论。


更新：校准SGD四组初始50完整读出及正式V11门槛已完成，见 `V4_calSGD_initial50_and_V11_actual_analysis.md`。共同pre22交互−.01391006、4/4负、CI排除0并超过.003，实际触发状态实验；这是LR特定optimizer交互，尚无状态因果结论。SGD早期1–10小正、1–20不稳定、固定50/selection-selected转负的阶段反例完整保留。V5/V8 seed25因非平台且不能确认selection过拟合，整组最低150；100产物不覆盖，V4自己baseline100仍独立。所有新test仍未读取，任务继续。


最新执行修订（持续 RUNNING）：V2固定SGD全部四组1000/200读出及主agent核验已完成，固定参考interaction未触发而校准interaction实际触发的矛盾被保留。V11三种状态训练均通过两步float64公式、fresh/resume轨迹与native真实epoch1状态重放核验，24条正式训练和4条完整shadow回放已经分两段队列启动。原GPU/训练/readout冻结源未改。

GPU bootstrap等价工程实际同一完整50轮1000/200 oracle通过：数组最大误差3.33e−16，全部4000重选epoch、onset与预算类别一致，174MiB实际峰值、167秒。基于该测量，pending-only CPU读出迁移为同一严格1000/200 GPU引擎；7个等待队列归档且没有终止其已运行数值工作。当前原CPU仍10个数值readout继续，GPU跨训练/readout合计保持每卡5进程及90%显存限制，并增加主存预约。PERMSGD24仅pending迁移，23仍计算；父调度器SIGSTOP防止重复派发，等待23自然退出再清理父进程，详见dispatch_corrections。

预算最新状态：V8 seed23实际150六条完成后strongSB改善.00922897且无eligible selection确认窗口，故全六条最低200；seed25的V5/V8实际150共十条完成后cleanSN改善.00634126756且无eligible窗口，故共享clean的两个六条件factor共十条最低200。不存在200硬上限，需按实际产物继续检查。V5 AdamPERM23实际100完整读出确认selection过拟合，匹配WR/PERM最终100无需150；AdamPERM24初始50 SN仍未确认过拟合，匹配四条最低100，copy676文件/5.339GB及主agent24关键文件SHA/inode通过后已启动队列。任何中间100/150均保留，V4自己基线最终预算不被stress实验改写。

统一早期窗口的cell/domain/task交互计算及图已实际生成核验：`V11_joint_early_domain_task_interaction_analysis.md`；`figures/initial50_trajectory_comparison/`包含初始50全部个体seed轨迹及PDF/PNG，图中22虚线仅标共同描述窗口，各面板纵轴刻度须逐一阅读，logloss带为观测seed范围并非置信区间。终局仍需完成后续匹配factor、V7与状态分析，再冻结所有条件后才读test。


迁移GPU首批16配对现已全部实际COMPLETE、主agent核验PASS。新增Q39跨预算数值复用问题经actual AdaTask23 GPU100→原CPU50完整1000/200 oracle解决：最大误差3.33e−16，全部4000次重选epoch与onset/selected/budget完全一致，原点值和cell完全一致。归档在archive/engineering_evidence/prefix_GPU100_to_CPU50_actual_oracle；允许严格父缓存复用，不能称所有CPU/GPU浮点位完全相同。


实际新进展：原生Adam23完整100轮读数已完成exit0、主agent全部输出/生产者哈希及六cell点值复核PASS。V10四种子最终匹配预算50/100/50/50全部完整，统一早期14、固定50、最终预算、AUC/LL选轮与所有域任务已完成分析，见V10_final_vs_calAdam_analysis.md。V4 B2四种子初始50统一pre37及精确1225步/全阈值首次达到分析也已完成，见V4_B2_initial50_analysis.md。V11训练part1全部8项COMPLETE，其余状态、matchedstress200等继续执行。未读取新test，不代表全部任务终局。


V4全部实际正式训练、完整配对读数与最终速度表完成，主agent核验PASS；V9最终KR/ML共同1–16窗口揭露自身窗口37/16符号差并非单独环境因果证据，新增Q41/BOUNDARY。状态shadow原实现漏存设计要求的方向余弦，独立只读补充已设计/实现/自审，尚不RUN、不得跳过实际四seed复核。


V3七基线28对完整112000次重选记录与点值/文件身份通过主agent复核，见V3_final_baseline_selection_analysis.md。V8seed23最终200六个cell全部完成，主agent物理重哈希全部1200个批次文件，三corruption同q逐epoch字节一致，标签/支持/暴露一致，J分解误差最大1.03646e−8。V11seed23实际100全六完成后beta2SN仍>.005且零eligible确认窗口，必须全六最低150；1920文件11.424GiB副本与主agent36关键SHA通过后队列313608/第39已接入，原100保留并计划由未来最终parent精确派生100读数。


## 四种子影子性能反事实完成，完整状态实验继续

四个原始影子回放及其19,836行绝对指标、71,820行对比、17,955组四种子统计均实际完成、主代理核验PASS（最大4.1e−16）。局部第10轮m/full-shadow改善两臂绝对AUC，但当前q的AUC差距仍四负；范数与阶段对loss/AUC影响不同。详见`V11_shadow_local_statistics_analysis.md`及图表，局部影响MEDIUM，长期唯一必要机制未识别。方向补充4回放已注册第41GPU队列，匹配延长22到100/23到150/25到100与24初始选择判据仍待完成。Q40已保留四种子beta2饱和和m-only学习不足，也保留v-only两臂较好的反例。Adam24四个actual100输入PASS，CPU2完整读数进行中；WR_SN尚须selectionbootstrap确认预算。新test仍未读取，未达到终局停止条件。


## 实际预算继续延长：seed25最少250、state24最少100

Seed25全部10个匹配条件actual200/exit0及60个关键文件SHA已核验。九个满足平台，mildSN改善率.005263345；完整1,000次selection确认的三个窗口均未通过，因此按冻结规则整组最少250，复制新目录进行中。为避免额外完整中间预算计算，新增只缩减bootstrap轮次的精确工程读数：所有200轮point/1200cell先从真实预测重算，实际完整CPU缓存四轮×1000与单臂子集逐位相同，100个合成确认分类与原函数一致，实际8轮缓存的三个量化窗口已独立复核。它不替代最终完整配对读数，100/150/200共12份中间前缀义务保留。State24三对actual50完整读数均PASS，v-only两臂均需延长，整6条件最少100复制中。四种子Adam抽样审计已实际完成并核验600个配对arm-epoch、3600覆盖行。新test仍未读取，V7尚待V5最终分析，不能停止整轮任务。


## 250轮十条件已完成，匹配300与状态后续继续

Seed25全部10条actual250已由五个真实队列完成exit0，主agent复核60个关键生产者文件及70个紧凑归档文件。整组状态是五队列实际task记录的无损汇总，queue_pid=null且明确不是实际整队列，来源SHA及真实PID均保留。Clean SN/SB相对目标改善.00780423/.00506054，实际1000次selection确认均未通过，故整组最低300；ORDER、两档corruption各自平台不豁免匹配时长。前四对300已实际排队，最后strong对300正在独立复制。没有300硬上限，17个初始/中间预算配对读数义务仍保留。

State23最终150、state24最终100均已完成全部3个完整配对及主agent12000次重选核验；state23原100三个严格前缀也全部完成。State22 actual150中beta2zero SB目标损失晚期从40.744降至16.902、selection AUC仍为.5，不能称波动或永久失效，零eligible窗口迫使整六条最低200；副本及36个关键SHA/全部metadata/inode核验PASS，实际200队列已启动。State25 actual150仍在完成，不能先认定终点。

V11四种子方向补充1740行与12个检查点已全部核验，图figures/V11_shadow_epoch10_geometry_and_auc.png将原始范数/余弦与已核验范数匹配后的AUC区间共同展示。Q32历史SGD核对的selection半区−.00021110与独立diagnostic半区−.00484873明确分开；训练数值相同不要求不同评价人群的差值相同。所有新test性能仍未读取，V5 SGD、V7、完整状态、test和最终闭环尚未完成，持续RUNNING。


State25 actual150六条现已全部完成，主agent36关键生产者SHA和42归档文件复核通过。beta2zero SB改善率.01470619、零eligible确认窗口，按原规则整六条最少200；续训清单与独立副本要求已设计/实现/自复核，机械复制已授权。原50/100/150共九份配对读数将从最终完整parent严格派生，旧输入保持不变。该结果不构成最终200充分性声明。


新进展：actual300 clean/PERM/ORDER六条均COMPLETE/exit0，逐对12关键生产者SHA与14紧凑归档文件全部复核PASS，均满足平台；corruption两对仍执行，不能认定全十条终点。clean/PERM全300完整1000/200读数已由固定工具真实启动，严格限制同时最多两项GPUbootstrap。State25六个150->200副本已完成，主agent36关键SHA/全部继承metadata与inode核验PASS，第61队列在全部60前驱后授权。

已有七基线固定10/50轮的域规模/正例率/绝对AUC/LL补表与图已完成：672个cell、126组汇总独立误差最大2.22e−16，见baseline_domain_context_analysis.md。由真实配置新登记Q43：MovieLens每seed训练子集不同，但每seed内SN/SB相同，validation/encoder/scaler固定；ML种子区间包含子集及优化随机性，不能称纯优化方差。16个KR/ML配置及当前生产者SHA、实际抽样源代码完成核对，BOUNDARY/STRONG，不阻塞有限配对/环境边界结论。全43问题保留，最终NewlyExposedQuestions须包含Q28–Q43共16项。任务继续，新v3 test性能尚未读取；历史test已暴露。


State22 actual200全六条/root36关键SHA/archive42均PASS，但β2SN改善.09206735，其唯一候选窗口18的精确1000选择确认上界[−.000836,+.001123,+.004320]未通过；主agent独立重算200个真实预测文件、1200个cell和4000条原始重采样记录后确认必须全六条至少250。新六副本清单与训练命令已设计/实现/自复核，实际copier439384执行中；150/200共六份前缀义务保留。24个selection阶段健康读数见V11_state_seed2022_T200_budget_and_health_note.md，既记录β2长期非单调损失，也记录m-only随训练改善和v-only后期反例，不将50轮状态永久化。

Clean seed25 actual300完整1000/200读数已COMPLETE/exit0，主agent4000重选及onset/预算/点值核验PASS，最大2.22e−16；ORDER完整300读数队列444340在全部61前驱后实际启动，仍与PERM合计最多两项GPUbootstrap。旧17份seed25前缀仍遵守实际最终父要求，已另存parent_scope_clarification，不以提前完成300读数冒称全组最终300。

工作附录Newly_Exposed_Questions_WORKING.md已包含Q28–Q43共16项的六个规定字段，Q40/最终遥测/最终尾部审计等未完成部分明确标注；它不是终局报告。全部规定流程继续执行，新v3 test性能仍未读取。


### 实际 300 epoch 复核及状态 seed 2025 最终预算

V5/V8 seed 2025 十组实际 300 epoch 已完成，60 项关键哈希、70 份归档均通过主 agent 复核；strong SN gain=0.007735 仍超阈值，两个候选退化窗口的精确 1000 次选择 bootstrap 均未确认，整组按规则需 MIN350。前两对独立副本已启动复制，22 对前缀义务保留，350 不是硬上限。clean/PERM/ORDER 的完整 300 epoch 读出已通过复核，mild/strong 正在运行，当前不形成最终 V5/V8 结论。状态 seed 2025 六组实际 200 epoch 均满足操作性平台，FINAL200 已核验；完整三对及九对前缀读出仍待完成。状态 seed 2022 已启动六组 250 epoch 续训。详见 [预算分析](V5V8_actual300_and_state25_final200_budget_analysis.md)。所有本轮必需流程尚未闭合，不能将此阶段判为最终停止。


测试执行入口 `v3_launch_frozen_native_tests_20261016.py` 已完成独立工程设计、实现和主 agent 自审（非独立 reviewer），仅作未来冻结后的机械启动。它固定原始测试引擎、校验全局冻结/来源清单及已授权输入，记录真实 PID 与字节哈希；缺少冻结文件的实际拒绝检查已通过，没有读取数据或计算测试性能。新增冻结字段详见 `frozen_native_test_launcher_implementation_self_review.json`。


状态 seed 2025 的 β1=0/v-only、β2=0 两对最终 200 epoch 完整读出已完成，主 agent 核验 8,000 次选择重采样。β2=0 两臂最终 selection/diagnostic 宏观 AUC 均为 0.5，诊断 LL 为 SN 9.932614 / SB 6.087876，训练 J 仍很高；不能把该性能下限上的零差距解释为成功介导。该 seed/阶段绝对健康观测为 STRONG，长期必要机制仍 WEAK/未识别。见 [实际健康边界](V11_state_seed2025_beta2zero_final200_health_note.md)。m-only 最终读出、九对前缀及 seed22 最终预算/读出仍在执行，Q40 尚未关闭。


### 完整状态训练预算已确认，读出仍在闭合

V11 四 seed 的六组匹配状态训练最终预算为 **2022:250 / 2023:150 / 2024:100 / 2025:200**。seed22 的五臂满足操作性平台，唯一非平台 β1=0/v-only SB 由实际完整250的精确1000选择重采样确认退化（起点104、参照峰值24）；首次AUC下降48当时不满足训练目标下降，不将二者混同。全部250生产者36关键哈希、42归档已核验，最终预算证书已记录。seed25的三对完整200及九对严格50/100/150前缀全部验证完成（累计48000次选择检查，前缀不是独立重复）；seed22剩余m-only完整250、六对150/200前缀及全四seed联合分析仍在执行。native最终50/100/50/50与状态最终预算不同，受控比较须保留共同固定阶段，最终或各自选择轮次不能一概标为匹配。
