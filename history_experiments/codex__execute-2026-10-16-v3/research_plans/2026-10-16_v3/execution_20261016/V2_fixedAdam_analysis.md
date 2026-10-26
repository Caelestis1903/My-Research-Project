# 固定参考 Adam 的四种子结果：阶段性分析

本文件仅覆盖 KuaiRand、既有 M3OE、Sample objective、Adam LR=.01、WR、50epoch、seed2022–2025。校准 Adam=.003、SGD、其他对照与冻结后的 test 尚未闭环；本文件不是整体终局报告。主agent分析与自审，independent_review=false。

四组配对 readout 已实际完成，每组1000次 selection 逐行重采样重新选择、200次独立 diagnostic 重采样。共同 pre-degradation 窗口为1–27，取四种子全部臂中最早的预注册 AUC degradation onset 之前。该窗口由 diagnostic 事后分类，不决定训练、LR或checkpoint。验证集曾被历史工作使用，独立性仅限新的 v3 调参和选择。

| Diagnostic readout | SB−SN mean | 四个seed | 配对seed bootstrap95%CI |
|---|---:|---|---|
| pre-window AULC1..27 | −.006980 | −.012437, −.003372, −.002096, −.010016 | [−.011226, −.002734] |
| AULC1..10 | −.005711 | −.012875, −.006914, −.000693, −.002360 | [−.010246, −.001526] |
| AULC1..20 | −.008390 | −.013620, −.007499, −.002784, −.009658 | [−.012090, −.004503] |
| common epoch10 AUC | −.019667 | −.036075, −.021732, −.005535, −.015325 | [−.030888, −.009584] |
| common epoch50 AUC | −.003304 | −.016960, +.019653, +.011826, −.027734 | [−.022347, +.015740] |
| selection-AUC选择后的diagnostic AUC | −.009319 | −.009710, +.008719, −.011474, −.024812 | [−.021036, +.003670] |
| selection-LogLoss选择后的diagnostic AUC | −.008287 | −.039765, +.015088, +.006478, −.014950 | [−.028204, +.010783] |

前期指标4/4同向，配对sign-randomization p=.125，这是n=4时能达到的最小双侧值；不能写成严格显著。四个种子的pre27 LogLoss也均较差，平均差+.001383，seed CI[+.000646,+.002293]。该配置存在早期轨迹性能差异的证据为 **MEDIUM**，与 V1 同期望、不同 estimator 的 **STRONG** 数值/代数证据相容。它否定该配置的“仅在 late overfitting 才有差异”和“仅 selected 指标产生差异”解释，但尚未证明长期泛化劣势或唯一机制。

所有八个固定参考训练的末5epoch对前5epoch目标相对改善为.001284–.003399，均小于冻结的.005阈值。因此该组不要求延长；这是操作性平台而非全局优化证明。common50有两正两负且区间覆盖实际有害和有益效应，不能写成双方等效、SB已追上或长期差距存在。固定参考组本身尚不足以把整体研究归为“仅训练速度”或“稳定最终泛化差距”；须结合校准配置与剩余核心对照。若终局核心判断确实依赖此处的null，按冻结规则对相应匹配对照追加2026–2029，而不为每个次要终点机械扩展。

Diagnostic检测到的过拟合：seed2023 SN首次AUC degradation为41、confirmed overfit为43；seed2025 SB两者均为28。其余六臂未检测到符合规则的onset，不能据此证明没有过拟合。Selection侧判定不同，例如seed2025 SN在26已confirmed，而其SB未检测到；两半验证人口的差异不能被混作同一个分类，也不能用diagnostic修改训练预算。

共同pre27的分域与加权结果如下；baseline difficulty仅用该冻结配置的SN AUC描述，不等同于域的内在困难程度。

| 域 | Train N | SN AUC | SB AUC | 差均值 | 符号 |
|---|---:|---:|---:|---:|---|
| d0 | 172782 | .592097 | .587639 | −.004458 | 2正2负 |
| d1 | 1743898 | .679702 | .668582 | −.011120 | 4负 |
| d2 | 89452 | .630795 | .625431 | −.005363 | 1正3负 |

d1两项task分别平均−.010912与−.011328，均为4/4负。所有域的均值均为负，较小域的种子方向不稳定；本窗口没有稳定的“小域改善、最大域下降”交换。最大域在此baseline的AUC最高，中等域最低，三个域不支持将域大小直接当作困难程度。只有三个域且多重属性混杂，域大小的独立因果作用证据为 **WEAK**。

实际diagnostic行数加权的cell AUC差为−.011006、按原训练p加权为−.010289，两者均4/4负，macro为−.006980。这些是cell AUC的加权平均，均不是跨域pooled AUC。Task0/1的pre27差为−.005264/−.008696，均3负1正；不能用macro覆盖反向单元。

Checkpoint selection是明显的modifier：例如seed2024 common50为+.011826，而各臂独立选择后为−.011474；seed2022在同一个SB-selected epoch比较时也可改变方向。Selection-bootstrap量化的是给定已训练轨迹的行/epoch选择不确定性，不能替代训练种子、外层LR选择或历史泄漏的不确定性。Diagnostic-at-reselected分位数仅反映选择epoch的变化，也不是diagnostic行置信区间。

竞争解释仍包括：较低的有限预算学习速度、训练噪声与步长的交互、暴露/覆盖/重复与顺序共同变化、验证人口敏感性，以及优化器配置差异。固定theta geometry与AUC的关联未识别唯一媒介。下一项有区分力的证据是已冻结的校准配置/2B预算结果及WR/PERM、exact-batch ORDER对照，而非继续搜索有利epoch。

原始来源：analysis/V2readout__adam__seed2022..2025__until050/；四种子汇总和图：analysis/V2_fixedAdam_aggregate_until050/。主agent从原始point数组复算共同27差值、与汇总在1e−15内一致，见 V2_fixedAdam_primary_verification.json。pre27_seed_cells.csv、pre27_cell_summary.json、pre27_seed_marginals.csv和pre27_marginal_summary.json保留以上分解；原始50epoch训练compact数据已在 archive/formal/ 封存。
