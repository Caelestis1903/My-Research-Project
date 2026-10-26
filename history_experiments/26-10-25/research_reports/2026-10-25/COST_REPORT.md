> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`a9c8585963c98c1975b8e1a64ac1cb14ee3e9bb4863b58f3620f94022f0f31f9`。

# 成本与执行偏离

三类账保留，但本轮不足以提出严格同部署计算预算的优势主张。

**研究执行账。** `run_costs.csv`记录各实际执行段的main、elapsed、snapshot、probe、cache、actor；`cost_components.csv`保留组件，`queue_time_spans.csv`记录准入到退出观察跨度。S2核心事件跨度2896.676s，约48.28分钟，含调度/暂停/回收延迟；并发进程累计epoch wall为ML6101.588s、KR16626.297s，不能相加后称GPU运行小时。S1原始8条跨度216.959s；STATIC队列含16校准分支及2条mean修正重跑，跨度298.073s。AUX与核心重叠，不再把两个跨度相加为端到端时间。

实际完整主训练epoch合计1188：S1含两条保留的旧mean尝试100，STATIC80，S2新增执行912，aux30，baseline66。主逻辑S2仍是48×24=1152；seed3101共享warmup节省实际执行，不从独立部署中抹去它。原始H1/H5逻辑分支步数另见logical_probe_steps.json，不把独立P_update/P_cal两次风险评分算成两份训练分支。

**算法部署账。** `deployment_cost_components.csv`给每个S2逻辑24epoch加回其必要warmup的实测组件。BASE/STATIC不需要研究shadow；动态策略不应承担其他shadow learner。原S2的probe计时把这些研究控制器计算包含在一起，无法精确分离每个控制器的纯部署时间；早期bank warmup还存在共享源actor工作。新正式配置已提供关闭BASE/STATIC shadow、仅拟合活跃控制器的开关，但本轮没有晋级正式训练，不能把未实测的节省写成结果。没有CUDA kernel专用计时，main是同步后的wall组件，不是纯GPU计算。

**开发搜索账。** 16条STATIC校准、4条aux尝试、6条近期方法pilot及2条保留的mean修正前尝试全部计入。静态winner的成本不能替代整个8动作搜索；AutoIFS搜索和reuse均计费；STGU账本记录真实组optimizer steps。共同模型fork减少物理执行，不改变逻辑定义。

**预算偏离与纠正。** 首批9条S2的累计预算漏计首次fork actor刷新、CAL_OFF验证或snapshot，后39条已修正。旧budget_best保持原样且标为不完全成本，不能事后换一个更有利checkpoint来声称公平预算优势。S1早期actor时间包含cache，不能与cache重复相加。

必要aux默认各数据144 native等价epoch，ML cap161.297s、KR461.791s。原子epoch检查实际用到ML193.396s、KR495.314s，分别越界32.099s与33.524s；这是明确执行偏离，不称严格遵守硬上限。四条均停止，未补齐24epoch、未把未完成轨迹当完整析因。CAL/容量/额外H5等可选包停止为PENDING_BUDGET。后续baseline采用包含启动/I/O的进程deadline，6条均到限，无重试。

硬件为共享RTX3090，GPU1因显存长期超过80%未获准；GPU0按显存≤80%、功率<85%逐次准入，没有固定进程数。负载随并发改变，首轮native参考时间ML1.1201s/KR3.2069s不能被解释为每个后续进程应达到的速度。因此预算耗尽反映本次执行环境和额度，不证明对应算法在所有部署环境都更贵。所有质量主结论以完整24epoch/预定selected比较为依据；不从这些受共租和计费偏离影响的时间推出精确效率倍数。

以上限制对成本主张属于Critical；对完整S2的主训练数据、动作和预设质量选点没有直接改变路径。没有为修饰账本而重跑核心、延长reserve或删除失败尝试。
