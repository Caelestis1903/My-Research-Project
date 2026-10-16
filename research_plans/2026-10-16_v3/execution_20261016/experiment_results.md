# 实验结果报告（进行中：只纳入已完成并复核的读出）

当前仅有下列配置完成四种子初始50轮的全部数值读出和主agent复算。本报告持续更新，不能将它视作全任务或充分预算的最终结果。表内差值统一为 SB−SN；AUC正值更好、LogLoss负值更好。CI为训练种子重采样95%区间，selection行重采样与diagnostic行重采样另见各配对文件。n=4时双侧精确符号随机化p最小为.125，CI和同号不等于高功效显著性。原始逐seed数值、两臂绝对值、SD、中位数、CI、符号p均保留在 `analysis/completed_primary_summary_snapshot.csv` 及原聚合目录。

## 已完成配置：共同pre窗口

共同pre窗口由冻结的AUC下降规则对各arm/seed分类后取交集，不以偏好的效果选择。以下窗口因配置不同而不同；本表不是V11的共同全局窗口interaction估计。

| 配置 | 初始窗口 | SN宏AUC | SB宏AUC | ΔAUC [seed95%CI] | 正/负seed | ΔLogLoss [seed95%CI] |
|---|---|---:|---:|---|---|---|
| KuaiRand / Adam .01（固定参考） | 1–27 | 0.634198 | 0.627218 | -0.006980 [-0.011226,-0.002734] | 0/4 | +0.001383 [+0.000646,+0.002293] |
| KuaiRand / Adam .003（初始50） | 1–22 | 0.648158 | 0.634207 | -0.013951 [-0.018133,-0.009769] | 0/4 | +0.005218 [+0.004387,+0.006547] |
| MovieLens / SGD .3 | 1–16 | 0.538007 | 0.539398 | +0.001391 [-0.000131,+0.003849] | 3/1 | -0.000013 [-0.000058,+0.000010] |

两个KuaiRand Adam配置在共同pre窗口均四种子AUC下降且LogLoss变差，支持受当前配置限定的早期效应，证据MEDIUM；它不只由独立checkpoint选择或规则所定义的晚期过拟合产生。MovieLens的宏AUC区间跨零，不能以本表声称跨环境稳定同方向。

## 固定50与独立selection-selected readout

| 配置 | readout | SN宏AUC | SB宏AUC | ΔAUC [seed95%CI] | 正/负seed |
|---|---|---:|---:|---|---|
| KuaiRand / Adam .01（固定参考） | common_epoch50 | 0.659421 | 0.656117 | -0.003304 [-0.022347,+0.015740] | 2/2 |
| KuaiRand / Adam .01（固定参考） | selected_macro_auc | 0.663948 | 0.654629 | -0.009319 [-0.021036,+0.003670] | 1/3 |
| KuaiRand / Adam .003（初始50） | common_epoch50 | 0.636728 | 0.630150 | -0.006578 [-0.014455,+0.001299] | 1/3 |
| KuaiRand / Adam .003（初始50） | selected_macro_auc | 0.667745 | 0.655343 | -0.012402 [-0.025439,-0.002588] | 1/3 |
| MovieLens / SGD .3 | common_epoch50 | 0.557809 | 0.557586 | -0.000223 [-0.001553,+0.001107] | 2/2 |
| MovieLens / SGD .3 | selected_macro_auc | 0.561441 | 0.561682 | +0.000241 [-0.001222,+0.001127] | 3/1 |

固定参考Adam在固定50为2正2负、selected为3负1正且区间跨零；校准Adam也有固定50反向seed与selected微正seed。不能从早期四种子一致下降跳到“最终必然更差”。校准Adam seed23最终预算100，其余三seed为50；最终共同预算/selected解释必须等待100轮完整读出，不能把本表初始50视为全部充分训练。

## 域、任务、大小与阶段

校准Adam共同pre22：三个域的平均ΔAUC分别为d0 −.021447、d1 −.011156、d2 −.009251，各域均4/4负；两个任务分别−.011782/−.016120，均4/4负。最大训练域d1与最小d2未形成稳定一正一负交换。d0的SN基线AUC低于最小d2，不能将小域等同困难域。五个domain-task cell为四seed同负，另一个cell有反例。详细绝对值与种子统计见 `V4_calAdam_until050_analysis.md` 和对应marginal目录。

MovieLens最小训练域d1在pre16平均ΔAUC +.00527041，四seed均正，CI[+.00174322,+.00916846]，两个任务也均四seed改善；该域固定50平均−.00136779且符号不一致。最大域d2 pre16平均−.00116193、CI跨零，故“稳定小域改善、大域退化”的联合证据WEAK。MovieLens每轮5step、KuaiRand每轮49step，标签定义及域定义也不同；相同epoch不等于相同optimizer step阶段。见 `V9_diagnostic_analysis.md`。

## 正在计算的数据不得作为完成结果

校准SGD初始50的固定horizon点轨迹已完整，暂存 `analysis/V4_calSGD_fixed_horizons_provisional/`：AULC1–10平均+.00079036（4/4正），AULC1–20平均−.00009218（3正1负），固定50平均−.00384147（3负1正）。这些仅为预先规定horizon的训练种子统计；1000/200行重采样、共同pre分类和seed25充分预算未结束，未用于触发V11、宣称SGD早期必要性或判定最终预算。

V5已经验证前三种子600个arm-epoch的操纵身份，但四seed性能仍未完整。V8污染12条初始训练完成，seed23要求同clean/mild/strong六臂延长100；seed25条件已匹配最终100预算入队。AdaTask初始八条训练完成、完整selection读出尚待，seed23匹配100已运行。局部参数干预、条件状态分解、最终test均未完成，缺失数据不会用估计数填补。

## 数据与图表入口

- `analysis/V2_fixedAdam_aggregate_until050/`：全部逐seed/规则/epoch、六cell、domain-task轨迹与图。
- `analysis/V4_calAdam_aggregate_until050/`：校准Adam初始50的同类数据。
- `analysis/V9_aggregate_until050/`：MovieLens完整初始配对结果。
- `analysis/V1_full_coordinate_geometry.csv`、`V1_paired_geometry.csv` 及对应谱图：固定参数完整坐标几何。
- `analysis/V6_geometry_synthesis_initial50/`：真实轮内lag、更新与纠正m/v范数，含逐seed和主agent核验。
- `archive/formal/`：各已完成预算的紧凑生产者文件与append-only SHA索引；完整大产物留在本地runs。

所有新v3 test readout仍未读取。统计解释与限制见 `bootstrap_uncertainty_scope.md`。后续混合50/100预算图保留所有个体轨迹，但粗均值只显示全seed都有观测的epoch，支持数量单独输出。

推断边界Q36：共同pre窗口由冻结操作性classifier产生；没有同时峰值/窗口检验的familywise保证。不能把未检测到onset等同没有潜在过拟合，亦不把“confirmed”写成强于既定程序的统计证明。固定horizon反例和一致结果继续完整报告。
