# v3 实验执行报告（RUNNING，非终局）

任务继续执行，未达到全部流程完成或最终提前停止条件。完整计划来自 origin/main@d18f194，工作分支 `codex/execute-2026-10-16-v3`。最近核实的远程里程碑为 `63e8a3d7869e34cd7479d565a396447a63353e01`；之后仍有持续实验产物，该提交不表示最终验收。

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
