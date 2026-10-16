# v3 执行记录（进行中）

本轮完整规划来源main@d18f194；独立分支codex/execute-2026-10-16-v3。start.md为GB18030，按其§11由主agent设计、实现、自审和科研分析，仅机械运行委派luna-runner。independent_review=false；未伪称独立审查。

V0 已完成。V1 的16组完整经验梯度/生产批次审计均已完成并通过主agent数值复核，每臂512批、每批40960行、全模型1614254坐标。四种子确定性对照、参数单元支持审计、八组完整 domain-task 梯度及其独立数值复算也已完成。8个环境×种子 validation 拆分已在训练前冻结。详见 V1_decision.json、V1_task_direction_decision.json 和 V1_analysis.md。

PyTorch2.9.0+cu126，CUDA12.6，2×RTX3090 24GiB；复用phase0-3-env和现有KuaiRand/MovieLens数据。GPU最多5进程/设备，按该设备90%显存预算准入；跨队列使用前序状态和实时PID共同计数。32个可用CPU、约64GiB主存；经资源复核，最多6个双线程 readout 计算进程另加1个轻量诊断进程。原始训练产物保存在 /home_new 本轮专属目录，逻辑路径为 runs/；完整梯度矩阵留在本地，由散列索引追溯。

16个规定学习率 pilot 和16个固定参考学习率正式训练均已完成。LR pilot 仅使用 selection validation，冻结 Adam=.003、SGD=.3；Adam 是规定网格的下边界 winner，不能声称全局最优。V4 正式 finalist 与2B batch对照正在运行，V2/V3/V4 的1000次逐行重新选择与200次 diagnostic 重采样同步进行。固定参考组50epoch训练已结束，正式泛化结论仍待四种子完整 readout 和冻结后 test。所有扩展只使用 train objective 与 selection 判定，不能使用 diagnostic 决定训练预算。

V4 首次启动有8个 FileExistsError，原因是队列日志目录抢先建立了训练器要求不存在的 run_dir；均发生于模型初始化之前。仅停止该调度器，原失败目录、状态和日志完整保留。第二次尝试仅把日志移到 OUT/logs，24条科学命令逐项相同，修复自审通过后重启。详见 V4_startup_fix_review.json 及对应 postmortem；此处不把启动失败算作完成训练。

历史 SGD 差异审计已完成：8个 run 的19600个 batch、在线 loss、epoch10/50参数和同epoch预测一致。旧全集验证与新 selection 半集的效应变化主要由评分样本变化产生，epoch重选贡献很小；不是训练实现漂移。证据见 historical_SGD_reconciliation_decision.json。

V8 的2×/4×精确别名复制已完成，四种子、两个q共16个映射及实际梯度/更新数值检验全部通过。它保持 p/q、预算和独立支持量，按完全相同输入序列复用完整训练；不是16次独立重训。结论仅限 WR 下名义别名池大小不改变轨迹分布，不推断真实独立样本量无作用。详见 V8_duplication_decision.json。

Q31 的80个 pilot/finalist epoch前缀数值核验完成，确认 pilot seed 与正式种子重叠；新的运行目录不会增加独立种子。该短诊断的 detached launcher 未保存实际 wait 返回码，记录为 null，完整 PASS 收据和原始日志保留，不反推 exit0，也不重复运行。

当前整体状态为 RUNNING。V5及后续对照仍须按依赖门控实际执行；V4可能需要50epoch增量延长。test 尚未授权或读出。完整待办与恢复上下文见 orchestrator_checkpoint.json；机器状态见各 manifest.status.json，不能把“已启动”写成“结果验证完成”。

固定参考 Adam 四种子的完整 readout 已完成并由主agent复算：共同pre27 AULC差−.006980、4/4负，固定50为2正2负，尚不能形成稳定最终泛化差距的结论。见 V2_fixedAdam_analysis.md。聚合器首跑因重复arm关键字退出，已作单关键字修复，失败输出保留，实际重跑与原始point交叉检查通过，未修改冻结训练或readout源码。

MovieLens四种子两臂8个基础训练全部完成，均达到自身冻结的平台标准；V9独立readout已启动。AdaTask原定8条科学命令拆成两个队列进行安全并发，未新增条件或选择种子；四种子完整分析和与校准Adam匹配预算仍待完成。资源复核允许8个双线程readout和1个cell审计，观测总CPU约19.7/32核、可用内存44.7GiB后增加两个readout；GPU合计限制仍为5进程/卡、显存<90%。
