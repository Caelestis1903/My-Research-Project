# 校准 SGD 初始50轮及正式 V11 交互检验

状态：初始50轮八组配对读数已全部 COMPLETE，并由主agent复算/核验；不是最终全预算或test报告。independent_review=false。证据文件：V11_all8_initial50_primary_verification.json、V11_actual_gate_primary_verification_and_analysis.json、analysis/V4_calSGD_initial50_aggregate/、analysis/V4_calSGD_initial50_marginals/、analysis/V11_calibrated_initial50_interaction/。

## 冻结门槛的实际结果

Adam .003 与 SGD .3 的全种子/双q共同pre窗口为1–22。每seed交互 (SB−SN)_Adam−(SB−SN)_SGD 分别−.01876930、−.01665721、−.00848100、−.01173272；均值−.01391006，种子bootstrap95%CI[−.01771325,−.01010686]，4/4同负。预定 >=3/4同号、CI排除0、|均值|>=.003 三条件均满足，实际门槛TRIGGERED。四seed exact sign randomization p=.125，不能写成常规双侧p<.05。

该窗口Adam平均ΔAUC−.01395116，SGD−.00004111。对这组冻结LR/模型/阶段的optimizer-configuration交互，证据STRONG；不因此推出Adam状态是必要或充分根因，亦不剥离LR与参数轨迹差异。初始50为最终baseline前缀的有效性有600个模型/预测文件及轨迹的实证SHA一致证书；后续全预算selected仍必须读取。

## SGD的稳定性与反例

| 初始50读数 | 平均ΔAUC | 4seed符号 | 种子95%CI |
|---|---:|---|---|
| AULC1–10 | +.00079036 | 4正 | [+.00054878,+.00112369] |
| AULC1–20 | −.00009218 | 3正1负 | [−.00164890,+.00093622] |
| SGD自己共同pre1–37 | −.00049257 | 1正3负 | [−.00094825,−.00003690] |
| 共同epoch50 | −.00384147 | 1正3负 | [−.00749053,−.00043945] |
| selection-selected macro AUC | −.00550079 | 4负 | [−.00876549,−.00300914] |

1–10的四seed改善量小，logloss平均+.00065545、方向混合且CI跨零；不能称为综合改善。1–20区间跨零，既不证明必然下降也不证明严格相等。SGD共同pre37的负效应虽符号较一致且该seed-bootstrap区间排除0，但远小于预定.003量级；区分方向与实际量级。固定50和selection-selected的负效应否定“只有Adam才会出现任何负sampling效应”的强叙事，证据MEDIUM，范围限此初始预算与操作性端点。种子2025最终baseline100读数尚未结束，不能把该selected行冒充充分训练终局。

## 域、任务、大小与难度

SGD共同pre37的d0/d1/d2平均ΔAUC为−.00098389、+.00001357、−.00050739；d0/d1区间跨零，d2为3负1微正且区间负。最大训练域d1接近零，最小d2略负，无稳定小域改善/大域下降交换。两个任务分别−.00001893/−.00096621，均有种子反例，区间跨零。train-p加权cell AUC平均−.00009557，区间跨零；不是pooled AUC。

固定50的d0四seed均退化，平均−.01101261；d1平均+.00135730却仅1seed正，均值受单seed影响，不能声称大域稳定受益。d2平均−.00186909、CI跨零。任务1四seed均退化，平均−.00666823；任务0有反例。selection-selected时d1、d2及双任务均四seed负，d0为3负1正；但此规则与共同epoch估计对象不同。既有绝对SN指标中d0比最小d2更难；域大小仍不足以充当主要解释变量，证据WEAK。

## 可证伪的下一步

V11_state_decomposition_design.json 在状态结果出现前冻结三种实际训练更新：beta1=0/v-only（精确别名，只训一份）、beta2=0（瞬时g²分母，不是SGD）、m-only（偏差校正一阶矩、分母1）。固定Adam LR .003，明确学习速度和有效步长混杂，不据消融单独断言必要性。全状态shadow在相同SN参数路径上重算SB历史梯度，再固定当前梯度分别交换m/v/全部状态；同时读取原始及全局范数匹配的一步反事实。它比仅看state范数相关性更能区分历史方向与标量步长解释。训练阶段1/10/40、全四seed均保留；与完整训练和AdaTask若相互矛盾必须报告。

V5、V7、V8、V10、V11状态矩阵、最终test及最终决策仍未完成。本文件不是终止或完成声明。
