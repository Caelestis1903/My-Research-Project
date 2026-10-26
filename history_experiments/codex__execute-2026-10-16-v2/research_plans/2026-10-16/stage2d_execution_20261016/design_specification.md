# DESIGN SPECIFICATION — Stage 2D continuation

2026-10-16，主代理。用户明确纠正上一轮提前结束，要求执行完整条件流程或达到真实最终停止条件。原始最终规划不变；此前execution_20261016的停止解释被本续执行判定取代，旧数值/报告保留作为历史快照。

## 准入纠正

WP1/WP2的审计已确定可访问证据和缺失边界。“历史原件无法恢复”阻塞依赖其身份的Stage3B/4/4B，不能自动阻塞独立初始化的新Stage2D。2A查明通用interaction已有先例，这阻止空泛创新主张，不阻止经验检验。2B同alpha期望匹配、无数值bug；2C在C/P分区跨两数据集四seed均有可辨别的Adam平均局部更新差，SGD期望相同；A局部弱case限制范围，不构成全部Track A硬STOP。原规划没有要求先有2D性能收益才允许运行2D。

最新用户要求端到端真实执行构成当前操作授权。start.md§11要求除机械运行外主代理负责，因此本轮主代理完成DESIGN→IMPLEMENT→SELF_REVIEW→RUN→ANALYZE，独立review字段如实为false；不能再用被该指令替代的代理职责安排自行制造全局停止条件。旧F0受管队列/hold/状态不改，不调用其生产入口。本研究独立新路径与新study身份，不声称historical exact或独立审查PASS。

## 研究矩阵、冻结变量

KuaiRand natural全训练split，M3OE StageS1M3OEAdapter，不是V1/V2；4共享/3域/2任务专家，fcn=[512,256,256,64]，embedding16。现有4fusion raw scalars保留，不是恢复有意删除的V1/V2参数。main C/P Adam lr=.01，A_arch Adam lr=.001；betas=(.9,.999)、eps1e-8、coupled WD1e-5、无clip、无scheduler/实际early stop。

| arm | alpha | q | weight |
|---|---|---|---|
| SN | p_train | p_train | 1 |
| SB | p_train | (1/3,1/3,1/3) | 3p_train |
| EN | (1/3,1/3,1/3) | p_train | 1/(3p_train) |
| EB | (1/3,1/3,1/3) | (1/3,1/3,1/3) | 1 |

4arms×seeds2022,2023,2024,2025，共16runs，50epochs，batch40960，num_workers0。每epoch main曝光N_train=2006132（48full batch+40052末batch），49steps；每epoch1个独立train arch batch40960，50arch steps。所有arm的main/arch步数、批量大小、评价次数、参数容量相同；记录实测wall time，不声称GPU wall-clock耗时严格相同。

为纯q干预，两种q均为域内均匀**有放回IID**，完整数据支持；不让Natural无放回/Balanced有放回混入另一个因素。该SN是新协议的强Sample/标准Adam对照，不能当旧shuffle-S1相同执行身份。同seed同q的SN/EN、SB/EB必须同init、每个batch row hash及全部数据身份一致；不同q不要求相同row，因为那会消除采样干预。使用独立numpy main/arch持久RNG，按相同uniform(0,1)耦合域和域内index（common random numbers），不消费模型RNG。main/arch均采用对应arm相同alpha/q，检验整个原main/arch训练底座的估计器策略。

## 指标、选点与停止门（训练前冻结）

Primary endpoint为validation-selected six-cell macro AUC（validation最大，严格>，tie最早）；primary contrasts仅SB−SN、EB−EN。不能把Equal−Sample称sampling-only。每epoch完整validation；selected/common9,19,29,39,49/final及最后state完整保存；训练结束、selection写盘冻结后才评估已有test，明确仅forensic，不用来决定阈值/选择/重试。

保存每cell/domain/task AUC/LL、macro、sample-weighted domain AUC（非pooled AUC）、worstcell/worstdomain、selected/common/final与same-epoch。训练日志含loss、domain exposure、unique row exposure/重复率、batch指纹、每参数分区raw gradient/m/v/applied delta范数及m/v/step摘要，full model+两optimizer+RNG+sampler位置用于恢复。分区范数统计不改变优化；不采集全模型per-cell梯度额外网格。

沿原探索门：Δmacro≥+.003为practical、≥+.005为strong，至少3/4seed同向。nonharm guard提前固定：四seed平均任何cell AUC≥−.003、单seed任何cell≥−.005；每cell平均LL差≤+.010。这些是探索门，不是p-value或部署保证。若两个同目标primary contrast均未达到稳定实用selected收益，并且无稳定相关state/gradient差，则Track A STOP。若只有局部state差但selected无实用收益，停止方法开发；不把已有理论成立无限升级成边界训练义务。若有满足收益/guard的候选，再严格按Stage6逐项设计独立setting验证，不能自动新算法。

## 实现与最小必要验证

隔离新worker与持久队列，不改旧S1/S5/S6代码。复用现有数据/model/metrics函数；独立训练loop以避免旧S1禁止sampler的协议标签被误复用。记录新的source/config/init/split/batch/checkpoint hashes、code commit、真实attempt id。

依次验证：语法；真实行上的loss公式/autograd与Adam一步；同q配对row和初始化；小规模CPU完整train→selection→state/预测保存/加载（测试工程路径，非正式结果）；CUDA40960实际batch短profile，测显存reservation。通过后立即提交冻结代码并启动16run。

每卡独立准入：≤90%总显存、最多5个本研究进程；按实测profile预留初始化空间，尽可能并行，不要求两卡同时空闲。不动外部作业。持久化scheduler保存状态；某run失败保留原attempt，诊断必要修复并继续独立任务，不整队停止。源代码变化不得暗中进入活动run。主代理持续检查，不以“启动/归档”替代结果分析或最终科学停止判定。
