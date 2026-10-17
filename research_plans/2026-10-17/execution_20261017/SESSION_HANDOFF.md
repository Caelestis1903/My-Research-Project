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
