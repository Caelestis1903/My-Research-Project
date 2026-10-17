# 当前执行上下文（持续工作，不是停止通知）

用户要求完整执行start.md和2026-10-17规划，最终分析/报告/Git push。原start为GB18030；最后用户要求严格遵守其§11，非机械任务仅主agent，designer/reviewer阶段由主agent书面设计/自审，明确independent_review=false。RUN由持久机械队列，另有luna-runner n1_mechanical_runner只监督原始状态。不要派发设计/代码/分析子agent。

原工作区/home/xzf/Code/phase0-3无HEAD/remote、全部未跟踪；不要改其Git。实际checkout/home/xzf/Code/phase0-3-20261017-git，branch codex/execute-2026-10-17-run，origin HTTPS fetch/SSH push已成功。最新已推送里程碑156e035（首8条完整pilot元数据归档）；此前2cdd683/edf116a/05b9445/7dcfb59已提交capture、N2/N3、固定ML适配、门槛统计准备代码。源底座2d04c6de，计划从main@a56639ab取回，七规划文件已复制到原工作区对应位置。

输出research_plans/2026-10-17/execution_20261017；runs链接/home_new/xzf_phase0_3_n17_artifacts_20261017（845GB余量）。用既有Python /home/xzf/miniconda3/envs/phase0-3-env/bin/python；原始KR缓存与历史预测只读取自/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016。2×3090每卡有其他项目进程约9GB，保留不杀；每卡本轮4任务+原项目1=5。

N0/N6训练前已完成；N0 4068行三权重读数，672校准/2016评价行，12个预测引用按原认证SHA恢复，额外两组E100资格，不增加独立seed。ML raw CSV可按row ID恢复user_id，1209 test用户，200次配对用户bootstrap已补；KR处理数据无ID。N0图均值x轴已显式alpha修正并视觉查看。

N1 pilot32任务已经真实运行，绝不重复启动。queue PID660163，launch/manifest/status在输出目录；pilot都是H100，任何短期排名不淘汰。读取当前status和progress确认实际状态；不要把前缀标完成。tune2030/2031；Adam .0003/.001/.003/.01、SGD .03/.1/.3/1.0，qSN/SB。n17_train.py和n17_common.py自启动以来冻结，不可修改，否则运行中的源SHA检查会失败。

pilot全部终态后，主agent调用 experiments/n17_n1_control.py select，检查原始响应面/赢家，再启动所生成N1_confirmation_manifest。共同eta按双q双seed的selection best均值；单臂独立赢家同规则，平局较小LR。正式2032..2039，32–64条，完全相同复用。新command使用n17_capture.py逐epoch存model，供统一test AULC；capture层逐tensor检查已通过。pilot只用全selection/diagnostic曲线，不读取其test调参；无需重训pilot以补其test模型。

N1确认H100后调用n17_n1_control.py gates，只用diagnostic selected及train J：原文rho触发某optimizer全部16条续200；H100选点保留。N1.6 macro/任一cell |均值|≥.003且≥6/8同号只作资源触发。优先Adam，Adam未触发而SGD触发用SGD，否则固定点默认Adam。N2必做16锚2032..2035×两路径×E10/E40×四臂512draw；quota全程只有触发新增16条。source文件n17_quota.py（继承capture）、n17_geometry.py已实现并小规模核验/自审。正式N2仍未运行。

N3 n17_local.py已实现，16锚×32draw，整体C、5子块norm×direction四格及P参考；原生optimizer一步，完整train J和val两半，actual delta范数和预测SHA。n17_local_readout.py由主agent归纳，开发4seed选一子块/阶段（不添加开发显著性门槛），冻结预期方向；另2036..2039独立8锚确认；MCSE原文阈值.0001/4，最多64。N4仅N1.6+独立N3.6满足触发；N4具体拓扑尚未实现，须先补设计后做。N5-C若KR独立确认有实用q效应，至少ML固定subset_seed的对称调参+8optimizer seed；其他N5按原触发，不全铺开。

所有新模型test仍封存。未来确认及机制/环境正式训练必须保留每轮纯model；n17_test_readout.py有全局freeze硬门，n17_performance.py已准备但尚未正式使用/最终审查。所有条件/选点/分支闭合后主agent签global freeze，再统一test，主比较seed t CI+exact signs+bootstrap，两个optimizer主要共同eta比较Holm，七项Bonferroni方法保护界按原文。

不要为了赶进度改种子、预算或阈值。不要终止健康长期实验。持续每60秒内给简短进度，保持必要checks最小；源码实现优先复用，不做无关工程测试。最终交付计划的九项报告/表/问题终态和三常规报告，并commit/push回读远程；当前报告明确IN PROGRESS，不能当最终成果完成。

新增准备入口：n17_followups.py在N1 gates后生成H200/quota/fixedpoint/N3dev/SGDwholeC各清单并原样保留H100状态；不自动启动。n17_ml.py与n17_ml_control.py已按N5_C_DESIGN_ADDENDUM实现/自审，固定subset2022，N187500、每epoch5步、尾23660，触发器沿用N1重要残差，优化器沿用N2优先规则；任何重要KR残差即资源触发，无新显著性门槛。n17_local_readout.py --confirm接受原candidate_frozen文件，检查独立2036..2039及MC精度；补开发draw时不得重新--nominate。n17_method_guards.py只给数值收益/七项保护条件，不自动授权GO-METHOD；合成算术检查单独ENGINEERING。n17_performance.py保存原segment时钟，并重建累计active elapsed/epoch compute秒，H200需独立组名。

当前原工作区的research_plans/2026-10-17/execution_20261017已安全链接到实际输出，用户可从原路径找到报告。N0新补672个cell/seed/readout人口baseline上下文及168组摘要，源为历史表，未新增训练/test。N1 current active进度应以状态文件为准；不要依据本交接的8/32快照判断后来完成数。

最新实际状态：2026-10-17 UTC10:42左右，pilot32/32全部H100完成、无失败、完整预算/配对核验PASS；N1_control select已执行，Adam共同和own全.01，SGD全1.0，均网格上边界。H10赢家Adam.001/SGD.3，N1_pilot_surface128行和图已生成并视觉核对。3ae4fca已推送，随后真实启动32条confirmation queue PID736337（不要重复启动），新status=N1_confirmation_manifest.status.json。前8为Adam seeds2032–2035×q。capture配置逐epoch模型True、testFalse，commit3ae4fca已核对。mechanical runner已收到续任务，只监控confirmation原始状态，写confirmation_mechanical_*.json/jsonl，原pilotcompletion保留。N1_PILOT_ANALYSIS和AUTHORIZATION已归档。n17_archive现新增每epoch域曝光求和及本地batch/quota trace SHA，不上传大trace；所有pilot32 metadata已归档，待下一个Git里程碑提交最新归档和启动状态。

N4_DESIGN_ADDENDUM已提前冻结但未实施：随机hash必须按训练p累计阈值划三组，不是mod3/等分；KR无userID按固定rowID、seed2048。若N4选E40作为阶段，最终统计须另保留E40（现性能脚本固定端点只有10/20/50/100/200，需条件补40），不能用其他阶段替换。N3/N4/N5均未正式运行。

最新确认里程碑（2026-10-17 UTC11:53左右）：confirmation COMPLETE16/RUNNING8/PENDING8/FAILED0，全部Adam16条完成，partial16_validation通过。原queue PID736337继续，禁止重复。前4条SGD约E35，新增4条启动。后续仍必须等全部32再gates。最新已推送b309fbd，1597144为最终N1图代码准备，b309fbd为域大小/基准/难度上下文代码准备，均未读新test。已有8条capture真实身份核验通过；此快照不是停止通知。

最新关键里程碑（UTC2026-10-17 13:20左右，持续工作不可结束任务）：N1 confirmation32/32全部H100完成无失败，full audit N1_confirmation_validation.json PASS（4900更新/200613200行每次）。主agentgates已冻结，两优化器important=True：Adam宏diagnostic -0.01059792 CI[-.02285727,.00166144]、d1两任务8/8负/d2两任务7/8负；SGDmacro+.00152046 CI跨0，d0t1+.01038031 7/8正触发。两optimizer全部rho组mean<.005，H200均不触发，禁止放宽。优先Adam .01，新增quota16、N2geometry16、N3dev16、SGDwholeC16共64机械任务；N5-C触发已prepare 16pilot，但正式训练等待N2/N3阶段闭合，N4仍需独立N3门槛。
cc2fc2b已推送：完整审计/诊断门槛/64联合清单和授权，N1 original46问题条目严格比较保持完全相同。已真实启动N2_N3_ready_manifest queue PID815642，launch=N2_N3_launch.json，禁止重复启动。为保持GPU利用率，H200未触发且N1不可变，联合任务只读锚、新目录写结果，所以与N1只读归档并行，不必等归档IO。N1归档现已完成return0，index64，child confirmation_mechanical_completion.json。主要base源n17_train/common/capture/quota/geometry/local都必须继续冻结：运行中有SHA guard，改动会使健康运行失效。
新队列有1个实际失败：N2geometry_adam_SN_seed2033_E40，原首draw分域 vs整批forward梯度identity断言；原config/log/FAILED均保留，不可抹除。不是OOM，实际残差没被原脚本保存。其余8运行健康，firstarms progressing，仍55pending。主agent新增engineering-only n17_identity_recheck.py和N2_IDENTITY_FAILURE_DIAGNOSIS.md，不改任何正式源码。CPU同一40960行draw：samegraph residual4.65e-9，separate32 8.66e-8<tol6.83e-6，separate64 1.69e-16；CPU不能证明CUDA原因。preflight/N2_failed_identity_CPU.json已完成。
下一步最小CUDA复核：在首批N2最后arm quota draw约400时，只对queue PID815642 SIGSTOP暂停准入（不停止任何健康固定点/训练子进程），等nvidia实际GPU槽位空出，再跑same recheck --device cuda:<free> --out preflight/N2_failed_identity_CUDA.json；诊断结束立即SIGCONT queue，记录机械事件，保持总compute进程每卡≤5含外部YOLO。尚未SIGSTOP、尚未跑GPUrecheck、尚未放宽阈值或修改formal算法。依据实际CUDA值再决定最小修复/同seed重试，原失败不能当完成。不要在源码仍被健康N2/N3导入并SHA guard时改n17_geometry.py；N3也检查geometry SHA。若需修复，使用隔离的新文件并证明估计器未改，或待全部相关健康进程结束再改；只有验证/记录修正不应重跑无影响条件。当前source readout n17_geometry_readout要求source_signatures一致，若验证-only修复导致版本不同，需显式证明/记录估计器相同再调整，不能无条件忽略版本。
机械child n1_mechanical_runner已接新followup，只监控64队列，每45秒，文件N2_N3_mechanical_observations.jsonl，已知1失败不重报，新失败立即报，不做科学选择/自动archive/新队列；它知道计划SIGSTOP只是准入暂停。函数store n17_poll现指向64联合queue的计数/进度而非旧确认。
新增n17_geometry_readout.py及N2_READOUT_ADDENDUM/IMPLEMENTATION_REVIEW在81a3250已推送，正式N2尚未readout：16锚完整四臂×512，源/锚身份，raw trace/domain within/between, quota/IID/q contrasts, Adam mean update/fixedD/Jensen向量范数和MC RMS尺度，按path/phase分4seed均值范围，不把非零noisy norm当均值漂移显著。尚未运行新test。N1_DIAGNOSTIC_GATE_ANALYSIS.md保持diagnostic资源与test主分析区别。

最新覆盖诊断状态（UTC13:35左右）：CUDA工程复核已完成且queue已恢复SIGCONT，不再暂停。GPUfloat32 separate-forward res1.8875e-5>6.8345e-6，同graph5.333e-9，float64 separate5.936e-17，支持不同batch形状f32数值差异而非聚合代数错误；不声称具体算子原因。另一SN2034E40也触发同旧检查失败；所有原N2任务已启动，失败恰2，其余14条件不重跑，N3当前已运行。GPU rawjson、N2_identity_GPU_mechanical.json已保存；日志实际在checkout/logs/N2_failed_identity_CUDA.log（不是OUT/logs），机械json已补绝对路径和SHA。
最小隔离修复已DESIGN→IMPLEMENT→主agent REVIEW PASS→CPU工程smoke PASS：n17_geometry_identity.py为原geometry逐字副本，仅新增同graph身份检查helper/调用/来源metadata。原source完全未改，保证健康N2/N3 geometry SHA guard不失效。N2_validation_only_equivalence.json证明移除三项就逐字恢复原全部源码；status PASS_SELF_REVIEW_VALIDATION_ONLY。新检查保留原tol，原跨forward误差逐block记录而非假标PASS。smoke192行、B512、4臂各2draw，preflight/geometry_identity_smoke receipt COMPLETE_ENGINEERING，不能作科研数据。n17_geometry_readout新增 --validation-equivalence，接受仅该明确证明的两个源，base/quota必须一致，并记录一draw数值误差与MC RMS的比较；AST/selfreview通过，尚未正式readout。
已生成N2_identity_retry_manifest.json（2任务），N2_fixedpoints_resolved_manifest.json（原14+新2），N2_IDENTITY_RETRY_AUTHORIZATION.json。新输出名oldname_identityfix，原FAILED保留。重试与原seed/checkpoint/k512/streams完全相同。当前尚未真实启动retry！下一步先提交这套修复/证明/授权及CPU/CUDA产物，再通知机械child在下一批N2最后arm draw>=400时暂停主queue准入，启动2项retry queue，待两项均已准入且GPU PID可见、retry无PENDING再恢复主queue。然后两个scheduler可并存：retry无后续准入，主queue计其PID为外部。异常时先暂停retry准入再finally恢复主queue，避免两个有PENDING队列同时入场。不得停止健康子进程。更新retrylaunch/provenance/status，完成后在resolved manifest生成原始status投影，正式N2readout必须用resolved+equivalence。
最新已推送441b26d（完整64个N1元数据+原1失败CPU诊断）；本段新修复/证据仍待commit/push。原主要训练源码n17_train/common/capture/quota/geometry/local继续冻结。不要提前读取任何新test或跳过N3独立确认/触发N4/N5。

覆盖上一段“未启动”：retry已真实启动PID830855，N2_identity_retry_launch.json、N2_retry_admission_mechanical.json在OUT。主queue815642继续。协调有一次status文件未创建被误当PENDING0提前恢复旧queue的机械错误，child修正并完整记录；主agent核实所有8活动PID均GPU可见，每卡5总compute（4本轮+1外部），现retry2已全RUNNING/0PENDING所以无后续竞争准入。原N2其余14全部COMPLETE，原失败2保留；新2仍RUNNING。源n17_geometry_identity.py也已冻结（运行中SHA guard）。seed2033重试首draw row SHA与原复核一致。
新必要分析入口n17_quota_readout.py已按N2_FULLTRAIN_READOUT_ADDENDUM/REVIEW实现/AST自审，未执行、未读test。最终performance comparison必须包括N1_common_adam和N2_quota_adam（H100、同8seed）才能读；输出IID gap/quota gap/I_quota/两个单臂lift及4臂真实数值，次级pointwise区间，lift表generic SN/SB=IID/quota并保留real_q，可供method_guards单独作健康保护界。无新训练/新显著性门槛。
最新已推送19b689b（验证-only修复/证明/两锚授权），当前新分析入口及retry启动机械记录待下一里程碑commit。全局test仍封存，N3候选尚未提名，不可提前N4/N5训练。


最新覆盖（UTC14:15左右，持续执行）：N2两validation-only重试均COMPLETE，原14+新2的resolved16正式readout及最小完整性验证通过。N2_fixedpoint_summary/receipt.json及N2_FIXEDPOINT_ANALYSIS.md记录实际结果：ALL IID SB/SNtrace2.008914–2.584112，quota1.994669–2.550850；域计数预测占0.001297%–0.105936%，固定点不支持计数主导。Adam meanqcontrast/MC RMSsum .479189–.624439，fixedD/native .910502–1.054373，INCONCLUSIVE；不声称非线性漂移已确认/历史必要。四E40路径锚的两个STAR子块采样data梯度0但原生Adam更新非零，零分母保留未定义。N2_figures PNG/PDF已实际生成并视觉核对；新n17_geometry_context只读产物生成上下文表，无新抽样/test。主队列815642继续，14complete+2原failed+8N3devrunning+40pending；N3当前首批draw约24/32，不可部分提名。retry830855已退出，不可再写其原生status。之前误写仅outer metadata，已恢复PID并记录metadata_restoration，原outer terminal写时间不可恢复；科学记录未改。当前最新已推送08bcad0，后续N2结果里程碑待提交；全部训练源码继续冻结。仍须完成N3全dev→一次提名→独立8确认及必要64补draw、SGD16、quota16、MLpilot16及正式8seed、按门槛N4，最后globalfreeze/test/报告/push，绝不能现在final。


覆盖至UTC14:53（持续工作）：最新推送e009b4d；a670d97已归档全部N2固定点文本/summary/PNG-PDF/真实分析，9e2908c新增n17_local_control.py +N3_CONFIRMATION_CONTROL_ADDENDUM/REVIEW，c440af6为N3实际更新范数/坐标数，e009b4d为N3正式结果图入口（还没运行）。N3开发原队列目前7/16 COMPLETE，8活动，最后seed2035已开始。全部16完成后用n17_local_control audit完整性检查，再n17_local_readout --nominate只一次冻结；不得读取部分开发作选择。若candidate.need64，则prepare development_supplement（原4seed×双路径，唯一阶段/块，draw32..63）；独立confirmation prepare另8锚2036..39×双路径，draw0..31。二者可同一新增机械清单并行但不能重选。确认精度不足才confirmation_supplement8锚，最多64；readout合并各不重叠目录，--confirm original candidate。新增local_control只准备/audit，不自动run，运行源码全部仍冻结。新n17_local_readout还会输出parameter_counts、actual_block_update_norms及每seed均值范围、预测SHA同SB次数、实际与目标范数相对误差，目标0/None不填假0；没有改变nominate或confirm公式。该最后相对误差补充与METRIC_AND_POPULATION_DECOMPOSITION.md（已完成N0部分，新test部分明确pending）待同次提交。N3图按完成readout绘整体C四格/交互及冻结候选dev/confirm，训练seed n4 t95，MC非训练重复，实际作图后需view_image。N2_READOUT结论仍只是固定点；quota16仍必须完成。

后续派生状态不要再次覆盖任何原生queue status：N3 audit不依赖status，只要原主队列全16对应COMPLETE再调用；需要归档则写N3_development_terminal_snapshot.json。quota audit需--run-dir-only清单与状态投影时写新的N2_quota_completed_audit_manifest[.status].json，明确DERIVED_READONLY_PROJECTION，原815642.status和830855.status不可由主agent投影改写。ML select使用其未来真正原生队列status。未来如需在主queue有PENDING时优先准入确认，只SIGSTOP原scheduler，子进程健康继续；新增status必须存在、任务集合完全匹配、全部8或16预期任务已准入且RUNNING PID都GPU可见（终态任务有exit），再恢复旧scheduler。确认只5cases每draw比开发21cases快，仍守每卡外部YOLO+本轮≤5。

最终八seed主CI若跨越±.003的有益/有害界，规划要求新增具体精度问题（可Q53，或按现注册新增顺序），不加seed追阳性。旧46entry永远不覆盖。N5 A/B/D和shape只有核心主张仍需相应辨别时触发，不为强行唯一归因扩矩阵；C已资源触发必须按准备MLpilot16→对称winner→正式8seed完成。global test仍SEALED，不得提前。
