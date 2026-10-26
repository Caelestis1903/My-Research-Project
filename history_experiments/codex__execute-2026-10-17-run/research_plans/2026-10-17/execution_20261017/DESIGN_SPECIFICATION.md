# 2026-10-17 执行设计冻结

依据：NEXT_EXPERIMENT_PLAN.md（原文SHA见source_and_seed_receipt.json），历史底座2d04c6de。新结果尚未读取。各阶段依次DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE，任何新算法条件先更新本设计再实现审查。

## 指令冲突的处理

用户最后明确要求严格执行start.md。其第11条要求除机械运行外仅主agent工作，优先于此前AGENTS消息的设计/审查派发要求。主agent完成设计、代码、逐项自审和科学分析；自审不冒称独立review。训练由持久机械队列执行，不由队列做科学选择。start旧10-15入口及归档路径被明确新10-17入口和规划输出约定替代。原工作区无HEAD/remote，所有原文件保留，执行在隔离checkout `/home/xzf/Code/phase0-3-20261017-git`，分支codex/execute-2026-10-17-run。

## N0（先执行）

复算七家族已冻结逐cell表，原seed/预算/选轮人口逐项索引。固定E10/20/50/实际具资格E100，AULC1..20和原early窗口分开。三套cell权重均保留：uniform、各训练p、各评价人口p；加权cell AUC不等于pooled AUC。calAdam E50按cell和域列出贡献及alpha∈[0,1]插值。历史test暴露明确保留。

原产物盘存在时，对七家族四seed两臂的宏AUC选轮和E50固定预测，在selection半集拟合各cell严格单调sigmoid(a logit(p)+b)，a=exp(s)>0。固定平均BCE + 1e-4*(s²+b²)/2，s∈[-8,8]、b∈[-30,30]；L-BFGS-B起点(0,0)，不使用test拟合。以双精度logit得分计算单调变换前后AUC，概率截断eps=1e-12只用于logit/LL，记录截断与浮点tie。报告selection/diagnostic/test的AUC、LL、Brier和(a,b)，优化失败如实记录。先前存表E50与预测指标作必要核对。split计数、正负例、joint-label、user簇可用性记录；处理文件无user_id则只能row条件解释，不伪造用户bootstrap。

## N1（N0/N6/manifest完成后）

固定M3OE、Sample J、B40960、每epoch2006132行（49更新、尾批40052）、Adam(.9,.999,eps1e-8)或无动量SGD、coupled WD1e-5。保留原forward(x,test_flag=False)、固定fusion有效值sigmoid(1)、无DARTS/DBO/gradient balancing/augmentation。训练q固定：SN=p、SB=1/3；权重p/q。初始化seed与sampler seed*1000+17独立流，同seed两臂数据和初值严格相同。旧KuaiRand缓存人口仅一份共享只读；每run登记训练seed及population seed，selection/diagnostic沿用历史row hash分半规则，test不参与本轮选择。

32 pilot = Adam[.0003,.001,.003,.01] / SGD[.03,.1,.3,1.] × SN/SB × seeds2030,2031 × H100。共同eta按两q两seed各自1..100 selection宏AUC最大值均值，单臂eta按该臂两seed均值；精确平局取较小eta。所有完整健康配置跑满，不按早期排名淘汰。

在2032..2039确认共同eta两臂两优化器32条，独立赢家不同才补该臂8条；完全相同复用。冻结H100各run earliest strict-max选轮，E10/20/50/100及AULC1..20/100。H200只按N1.5训练J公式及完整有限16轨迹条件触发；H100选轮状态/结果保留。N1.6 diagnostic宏或六cell均值绝对值≥.003且≥6/8同号，只触发资源分支，不称确认显著。

最小验证：loss同图身份、同seed数据/初值/采样流、3步smoke、原实现与新训练step参数更新一致、完整epoch的更新/样本和保存恢复。不得扩展无关测试。数值失败保留失败条件，不换有利seed；OOM属于可恢复工程问题，保持定义重试到新attempt。

## 持久执行及存储

每GPU最多5个计算进程（包含已有其他项目），逐设备显存预测≤90%时尽量填满，不要求两卡同时空闲。每新进程初始预算约1900MiB，最终以smoke实测调整。限制CPU线程，记录外部负载；并行wall time是资源条件下实测，不能伪称独占吞吐。

每epoch存selection/diagnostic预测、六cell指标、全训练J、步数/样本/耗时、逐batch域计数与行ID SHA、sampler RNG及config/source/initial SHA。完整state含optimizer/RNG/step在E0/E10/E40/E100、selected与last；纯model在E20/E50；H200延长保留H100原件。不保存每epoch巨大row列表，sampler确定性与epoch起始RNG可精确重建。checkpoint/预测大文件不上Git；关键表/索引/代码/配置上Git。

## 后续条件与统计（原规划不改）

N2固定点与N3首批必须做，优先Adam；Adam未触发而SGD触发用SGD。16锚=2032..2035×SN/SB路径×E10/E40。N2四臂每512draw，quota严格系统舍入且实际权重B_t*p/n，在线全坐标分块矩。完整quota仅N1.6触发，通常新增16run。N3每锚32draw、CPRE/C子块、norm×direction四格及相对SB/SN/未更新绝对指标，独立2036..2039确认筛选结果，MC规则最大64draw。后续N3.4/3.5、N4/N5仅按原文触发，实施前补充具体设计。delta_local=.0001，N3.6选择及门槛按原文；N4用新2040..2047且先排重。不能将NOT_TRIGGERED当阴性。

训练seed为重复单位。配对seed双侧95%t区间、精确符号随机化及seed bootstrap敏感性；两优化器共同eta主要比较Holm。delta=.003；GO-METHOD主均值≥.003且CI下界>0，六cell非劣界.003及train-p LL相对恶化1%的七项Bonferroni单侧保护界。历史46问题不改写，新问题逐项登记，证据标STRONG/MEDIUM/WEAK，不以规模代替难度。

N6逐式核对8篇原文、目标/估计器/optimizer/度量/假设与本图边界，先形成差异表再启动pilot。没有候选主张不额外实现EMS/MIS。最终test统一在本轮所有设计/模型/选点冻结后读取，保留历史暴露事实。

状态：DESIGN COMPLETE，等待实现及自审；非独立设计/审查。
