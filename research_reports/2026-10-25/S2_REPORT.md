> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`2477b922a01936f70439b07cb7de47a25d1b0113e1a2f57b4ad1929d21988112`。

# S2 完整开发报告

## 结论与晋级

48条核心轨迹全部完成固定24epoch，exitcode全部0；两数据、3101/3102、每条线六个方法，无质量早停。全部48条同时取得主selected和共同epoch24的Q_explore评分。主选点严格保持RHD按validation macro LL、LPS按validation macro AUC；validation另表，未用Q挑checkpoint。

六项预定晋级判断全部为False：LAST、HIST、ridge、NN、LPS51、CONTENT_DIRECT。因而不启动正式4101–4104/50epoch包；这不是“未显著就补seed”。统计方向只是开发预算门槛，不能当总体显著性检验。

| 预定比较与指标，候选−参照 | ML3101 | ML3102 | KR3101 | KR3102 | 判断 |
|---|---:|---:|---:|---:|---|
| LAST−STATIC，macro LL | +.000295 | +.000161 | +.004879 | +.003046 | 0/4改善 |
| HIST−STATIC，macro LL | 0 | 0 | 0 | 0 | 无增量 |
| ridge−HIST，macro LL | 0 | 0 | 0 | 0 | 无增量 |
| NN−HIST，macro LL | 0 | 0 | 0 | 0 | 无增量 |
| LPS−GLOBAL，macro AUC | +.000023 | +.000004 | −.003897 | +.004418 | 虽3/4正，两数据均值均不足+.0005 |
| LPS−DIRECT，macro AUC | −.003369 | −.000056 | −.000447 | −.000326 | 0/4改善 |
| K4 DIRECT−K1 DIRECT，macro AUC | +.000864 | −.005242 | −.009105 | −.002170 | 1/4改善 |

RHD最强简单参照在每数据集按两个seed均值选择；STATIC/HIST在此相同。状态版本也没有通过共同BASE后14锚点实际改变HIST及正效用条件。无需用MSE或某个有利次要指标替代失败的主目标。成本预算表的局限另见COST_REPORT，不存在借成本结果把未通过质量门槛的方法晋级的情况。

## 绝对主选点表现

表中AUC/LL是Q_explore；每个单元为两个seed均值，完整每seed、选中epoch、natural LL及六格均在机器表。CAL_OFF使用自己独立的validation AUC选点，沿OWN_OFF共享主模型轨迹，不多计训练arm。

| 方法 | ML AUC | ML LL | KR AUC | KR LL |
|---|---:|---:|---:|---:|
| BASE（RHD的LL选点） | .808289 | .475736 | .699706 | .465850 |
| STATIC / HIST / ridge / NN | .808145 | .475950 | .699706 | .465850 |
| LAST | .808223 | .476177 | .694170 | .469812 |
| K1 DIRECT7 | .812412 | .474176 | .706312 | .464021 |
| K4 OWN_OFF | .808613 | .475886 | .701070 | .466911 |
| GLOBAL51 | .808498 | .475666 | .700028 | .467682 |
| SIM51 | .808527 | .475843 | .700885 | .467009 |
| DIRECT51 | .810224 | .474825 | .700675 | .466854 |
| LPS51 | .808511 | .475642 | .700288 | .466563 |
| CAL_OFF | .816814 | .467444 | .701550 | .466485 |

BASE在本表是RHD主选点，不能把它与LPS的AUC选点直接当完全匹配的ranking对照；BASE的AUC-best、LL-best与epoch24均单独保留。核心归因使用K1/OFF/GLOBAL/DIRECT等本线匹配对照。

## 稳定现象、矛盾结果与训练阶段

**MEDIUM：当前RHD决策版本没有独立收益。** HIST和两种状态模型的主结果与静态相同，LAST的主风险指标4/4变差。KR的LAST selected AUC平均也下降约.005536，而ML平均略升.000078，所以“风险更差”和“所有ranking都更差”不是同一句话。共同epoch24，LAST的LL仍两数据均变差（ML+.000828，KR+.009508）；KR的AUC均值却略升+.000686，进一步说明主风险失败不能被次要ranking局部改善改写。

**MEDIUM：当前LPS收益树没有超过同信息直接门控的ranking增量。** selected AUC四配对均低于DIRECT，ML/KR均值差分别−.001713/−.000386。LPS−GLOBAL存在明显KR seed反转，ML两次微小正值不足预定实用门槛。完整BASE重放又显示简单历史规则有正的同状态H1机会，不能把本轮负结果扩大成“历史或局部信息本质无用”。

**WEAK支持多单元优势；MEDIUM支持此K4配置不稳定。** K4 DIRECT只在ML3101胜K1，其他三配对下降，两数据均值分别−.002189/−.005638。模型、seed和训练阶段必须同时限定。共同epoch24，K4−K1的AUC均值差为ML−.000118、KR−.003578；ML的macro LL反而改善−.004243，KR则恶化+.002047。这是风险、ranking与选点阶段之间的矛盾，不是K4具有稳定ranking优势的证据。

LPS−DIRECT在共同epoch24的KR AUC均值转为+.000225，ML为−.000297；KR LL均值转为+.000374。保留这个阶段反转，仍按预定selected主目标判断不晋级。没有使用更好看的终点来替换主结论。

## 逐域、逐任务与域规模

fit域规模：ML d0/d1/d2为124759/254879/260495；KR为138225/1395118/71561。比较native共同epoch24的两seed均值：ML两任务平均AUC约d0=.81686、d1=.82057、d2=.80839；KR约d0=.66199、d1=.74771、d2=.68821。ML两个大域规模接近但表现不同；KR最小域不是AUC最低域。LL还明显受标签基率影响，例如KR d0的低LL与低AUC并存。不能把小域直接命名为困难域。

K4 DIRECT−K1的selected域平均AUC在两seed平均后，ML三个域分别−.001985/−.001291/−.003291，KR为−.011973/−.001225/−.003716。KR最大损失出现在中等规模d0，并非最小d2。因此域规模是可能影响支持量的变量，但不是已识别的主要因果解释，证据为WEAK。当前三域自然比较同时混合标签基率、表征、native基线难度和训练阶段。

任务维度也没有救回K4主张：K4−K1两任务的域宏AUC均值，ML分别−.002504/−.001874，KR为−.005001/−.006274。LPS−DIRECT的两任务域宏AUC在两数据均为负；但KR按自然行混合域计算的任务pooled AUC均略正。这是聚合权重不同的真实矛盾，不能用pooled替换预定macro。

**小域改善/大域下降确实出现在特定对照中，但不是普遍定律。** DIRECT−OWN_OFF在selected AUC上有3/4配对出现该模式：ML3102及KR两个seed；ML3101大小域同时改善。到共同epoch24，仅ML3101保留该模式，其余三个不满足。该现象依赖模型对照、选点和训练阶段；K4−K1只有1/4 selected配对满足它。ML的K4−K1在域均值LL上呈小域降低、最大域升高，但AUC没有同样改善。完整六格、任务、AUC及LL方向分别存表，不混称一个统一trade-off。

## 强简单对照与未完成归因

CAL_OFF相对OWN_OFF的selected AUC：ML两个seed提高+.008475/+.007925，同时LL降低−.010384/−.006500；KR为+.003634/−.002674，LL方向也有差异。校准解释在ML有重复支持（MEDIUM），在KR并不稳定；KR的K1 DIRECT仍强于这两个CAL_OFF结果。不能把校准普遍有效作为先验，也不能声称LPS已胜过完整CAL_PROB_SELECT闭环。

必要aux0四轨迹从epoch5分叉，但均BUDGET_EXHAUSTED：ML K1/K4止于22/10，KR止于9/9；因此没有完整24epoch析因结果，`AUX_ATTRIBUTION_PENDING_BUDGET`。CAL_PROB_SELECT闭环、K1_H31容量、更多H5/FLAT/IDENTITY等没有剩余reserve，均不假装已排除。单位参数预算匹配不等于actor容量匹配；K1与K4输出动作数不同，内容归因须保留这一限制。

## 最有判别力的后续工作

RHD优先做冻结简单反馈规则对STATIC/HIST的完整24epoch闭环，检验共同trace的小收益是否真的转化为训练收益；不先扩大NN。LPS优先在完整预算下补齐aux析因，并以CAL_OFF及容量匹配K1作为强对照，尝试推翻“多单元本身带来收益”的解释。域规模的因果检验需要固定域身份和评价支持、操纵fit独立样本量并控制训练曝光；当前自然域关联不能代替它。这些属于明确待验证问题，不在本轮预算外追加。

原始工件：`../tables/S2_Q_explore.json`及cells表、`S2_selected.csv`、`S2_validation_selected.csv`、`S2_all_contrasts.csv`、`S2_cell_contrasts.csv`、`S2_task_contrasts.csv`、`S2_size_tradeoffs.csv`、`S2_size_risk_tradeoffs.csv`、`native_epoch24_difficulty.csv`、`promotion.json`。曲线和规模图在`../figures/`。时间重放另见REPLAY_REPORT。两开发seed只反映有限优化随机性，不提供新数据确认或正式四seed置信区间。

训练动作补充：`RHD_actual_actions.csv`核实HIST、ridge、NN各自覆盖4条轨迹、共1568个epoch11–24主训练逻辑batch，非静态fallback动作数均为0。这里对“当前实现实际没有采用额外动作”的证据为STRONG；对任何状态模型普遍无效的主张没有相应证据。BASE在ML因fallback为action5而自身为action0，不能把其nonfallback计数误作状态使用。

LPS策略快照补充：`LPS_policy_depths.csv`记录四条主selected和四条epoch24快照的深度cap均为0；可用的两个独立epoch10快照也为0。这里说明实际被评分的策略退回域/任务根动作，不能称已证明域内局部分区有增量。没有逐个保存所有中间刷新结构，故不把这个快照结论扩成“训练每一步都没有局部动作”。
