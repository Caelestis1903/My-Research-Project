# 09｜评价、有限阶段门与最终研究决策

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。本文件与 [manifest](experiment_manifest.json) 的同名门共同锁定自动判断；运行中不得增设正向seed数或显著性门。

## 1. 三个判断层次

| 层次 | 比较对象及阈值 | 允许的结论 |
|---|---|---|
| 机制进入门 | 完成五seed；本文件的短程/未来效用门，典型ΔAUC≥1e−4 | 足以分配下一阶段预算 |
| 新组件研究价值 | 完整曲线own-selected V对固定F或规定更简单控制，ΔAUC≥2e−4且六格保护 | 冻结为正式Test候选；不声称达到论文性能标准 |
| 正式AUC合格 | 五seed own-selected Test，完整方案对预先指定发表模型平均ΔAUC≥.003，每格平均Δ≥−.001 | 该数据/条件/比较对象下AUC合格 |

1e−4与2e−4是本轮有限预算的效应大小决策线，不是统计显著性，也不替代用户的+.003正式标准。恰好等于阈值算通过。任何性能否决前相应配置都须齐全五seed；工程smoke仅验证实现。

## 2. 统一指标和选点

每epoch完整V计算六格AUC/原始LL、Macro、源定义Traffic/Pooled。AUC/LL选用相同AUC-selected预测对象；不为LL另找最优epoch。每配置每seed从完整曲线选最早严格最大Macro AUC：`if score > best`。训练曲线完整，不做性能早停；A保留E1–20前缀，KD e0–e5范围预注册。

定义 \(U_d(j)=\frac15\sum_s AUC_{V,d,s}(j,e^*_{j,s})\)。选超参以U为主，先筛相对指定控制六格平均均不低于−.001的配置，再排序U；没有可行配置则记录最优的真实宏增量及被损格，不运行后续放大。HPO并列容差为1e−6，仅用于配置选择，**不用于checkpoint选点**。不同数据集各自选一个五seed共用配置，不跨数据平均抵消。

所有差先在同seed内配对；一个seed内多个epoch、cell、pair先按定义归约，再用n=5均值、样本SD和t4区间，\(\bar\Delta\pm2.7764451052\,SD/\sqrt5\)。区间与正向seed数完整报告，不是硬门。LL独立裁定：平均ΔLL<0为改善；≥0报告代价/无改善，不取消已达到的AUC资格。

## 3. G_A：短程真实反馈

对dataset d、direction_alpha α，令

\[
D_{s,a,\alpha}=\tfrac12\sum_{e\in W_a}(AUC^F_{candidate,s,e}-AUC^F_{parent,s,e}),
\quad W_{20}=\{25,26\},\;W_{35}=\{40,41\}.
\]

主统计 \(M_{d,\alpha}=\frac15\sum_s\frac12(D_{s,20,\alpha}+D_{s,35,\alpha})\)。逐cell同样先平均两末端epoch与两anchor。**G_A**要求：当前dataset两个alpha合计20条预定短支完整（不等待另一dataset）；就当前dataset/α的10个fork均完成6epoch和5seed；M≥1e−4；六格均值≥−.001；E20那一项五seed均值严格>0。

每数据在通过者中按M排序，差≤1e−6优先α=.5，仅一个A_MATCH晋级。若合并门满足但E20≤0，记`REVISE_TIMING`，本轮不把正式启动点改到35。若两个alpha均未通过，记录A STOP并转B。

通过后才运行A_RAW的同样两anchor六epoch五seed；RAW完整性要求为当前dataset自身10条短支齐全，数值门与G_A相同，不再套用MATCH两个alpha合计20条的完整性要求。若通过则RAW也进入E26→E50续训。MATCH与RAW各自完整曲线再按U比较；差≤1e−6优先RAW。原生Adam控制来自已完成父轨迹，不另训。

解释使用已记录online差：若online主均值≥1e−4而F主均值<1e−4，记`ONLINE_GAIN_WITHOUT_EXTRA_F_GAIN`。当|F主均值|<1e−4时，结果与F已覆盖该类改善相容；F主均值≤−1e−4则直接记录F代价，不写成吸收。动作真实而online/F均未达到门，记`LOCAL_GAIN_NOT_SUSTAINED`。混合结果保留具体均值。

## 4. G_B与共享一遍前向替代

B_SPLIT所有12个逻辑配置在五seed完整V曲线比较；B_SHARED_ONLY是一个独立1次前向候选。先分别得到最佳B_SPLIT和B_SHARED_ONLY。B_SPLIT内部并列顺序：S优先all、logit优先probability、主H优先短H再长H。它们与F的比较每个都记录，不能把一个无收益配置的失败扩展为整个历史预测方向失败。

**G_B**：最佳B_SPLIT的U−U(F)≥2e−4、六格≥−.001。B_SHARED_ONLY适用相同门但无需蒸馏。若两者都通过且U差≤1e−4，则优先B_SHARED_ONLY作为部署主候选；仍保留B_SPLIT机制结果。

若B_SPLIT未过G_B，停止其蒸馏与更大状态数搜索；C的预定无SGD读出仍完整裁定。若仅LL下降而AUC未到门，记录`LL_ONLY`及其量值，保留结果，不自动扩展训练矩阵。

## 5. G_C：差异必须变成实际价值

C仅使用Hbase/S/logit。全部策略使用前epochP_A拟合参数，e+1部署；未来审计窗口固定为e=21–50，权重来自e−1。对s=.5/1及PRED/PARAM分别比较同mode全局策略。P_A拟合与P_B AUC/pairwise风险均使用源rank_x/rank_y/rank_row_ids人口；P_B LL使用natural人口，继承source readouts.pulse_rows口径。PARAM的风险/AUC必须由实际插值参数前向产生的logits/probability计算，不能复用PRED混合margin。缺任一格风险则门输入不完整，记BLOCKED_PANEL，不填零。

机制门G_C_MECH必须同时满足：

1. 五seed的全部完整曲线和未来窗口齐全；平均\(|\alpha_c-\alpha_G|>10^{-6}\)，固定P_B pair的预测排序不同率>0。
2. 在上述未来窗口，先按cell/epoch、再seed平均，P_B pairwise风险(C−GLOBAL)<0，P_B Macro AUC(C−GLOBAL)>0。
3. 同窗口完整V的五seed平均Macro ΔAUC≥1e−4，六格平均≥−.001。

第1项只确认动作没有退化；第2项检验训练可见信号的未来效用；第3项检验泛化排序价值。P_B没有进入系数学习或HPO。没有4/5、CI>0附加要求。

完整候选进入正式冻结还需 **G_C_SELECTED**：U(C)−U(同mode GLOBAL)≥2e−4、逐格保护，且相对本轮已得到的最佳统一读出（F、B_SHARED_ONLY、最佳B_SPLIT、C_GLOBAL_PRED、C_GLOBAL_PARAM）的U差≥2e−4及逐格保护。若机制门过而selected门不足，记`MECHANISM_GAIN_SMALL`，不扩大控制器。

s按五seedU选择；同分优先.5。PRED与同α PARAM比较：如果参数版性能差距≤1e−4，优先推理成本更低者；只有PRED明确优于同α参数版，才将增量解释为条件函数组合。**G_C_GLOBAL**：C_GLOBAL_PRED或C_GLOBAL_PARAM的U−U(F)≥2e−4且六格保护通过，即使C_CELL不通过，也各自保留为简单统一替代。两GLOBAL在相同性能容差≤1e−4时优先PARAM；PRED真实更好时可以进入正式Test、主候选和教师池，不要求MDMT特异增量。C未通过时分别标记`NO_ACTION_CHANGE`、`NO_FUTURE_SIGNAL`、`NO_INCREMENTAL_VALUE`，停止新的分域网络/指标扩展。

## 6. G_AB：有限组合

父B在五seed父轨迹完成后即可按G_B冻结；A短程期间只附带该**一个固定B配置**的曲线。A达到G_A且B达到G_B才在A续训保留AB。完整后比较U(AB)与U(A_F)、U(B)。

G_AB要求U(AB)−max(U(A_F),U(B))≥2e−4且相对两者六格保护均通过。否则保留较佳单组件。报告独立selected的差中差与E50同刻差中差，不将二者混用。组合不产生新HPO、SGD或蒸馏家族。

## 7. 蒸馏进入与退出

教师池为满足G_B的B_SPLIT、满足G_C_GLOBAL的C_GLOBAL_PRED、以及满足G_C_MECH和G_C_SELECTED的C_CELL_PRED。按U选择一个；差≤1e−6按B_SPLIT、GLOBAL、CELL顺序选简单者。还要求教师相对最佳**1次前向**统一替代的U增量≥2e−4；否则优先已存在的一遍前向算法，取消本轮KD成本。

四个B_DISTILL配置（2λ×2LR）与两个LR的KD_LABEL都完成五seed×5epoch；以学生F的U选择一个λ/LR，差≤1e−6优先较小λ，再较小LR。保留online完整曲线并独立选点，但不把online/F当另一个未注册搜索维度。用赢家λ/LR补KD_FTEACHER五seed×5epoch。教师与学生初值均按13。

KD历史信息增量通过条件：

- 学生F相对父F、同LR的KD_LABEL_F、KD_FTEACHER_F分别U增量≥2e−4且相应六格保护；同时报告它对两个LR中更好的LABEL控制的差。
- 保留教师超F收益的比例 \(r=[U(KD)-U(F)]/[U(teacher)-U(F)]\ge.5\)，分母已由进入门限定≥2e−4。

若满足，记`COMPRESSED_HISTORY_GAIN`，单模型部署进入主候选。若只胜父F而不胜简单控制，保留总收益、标记新增历史知识未形成独立增量；若label或F教师更好，将其作为更简单方法候选，不归功B/C历史。若KD未保留收益而两态教师有效，保留两态性能选项及原F低成本选项，不再扩大蒸馏网格。

## 8. Gaussian边界与主候选选择

每dataset从已满足对应门的完整Clean候选选择U最高者；差≤1e−4时按实测推理前向数、参数字节、训练额外步骤、固定ID词序依次选择更简单者。候选集合包括A、B、通过门的C、通过门的KD和有增量的AB；简单LABEL/Fteacher若对父F与六格有≥2e−4增量也可入选，但标注其来源。

**G_G**：所选主候选对Clean父F的U增量≥2e−4且六格≥−.001。每dataset至多一条算法族进入Gaussian，沿用其Clean超参/起点/scope/mode，不重新搜。A/B/C/AB在Gaussian父轨迹上照旧完整V选checkpoint并五seed评价。若赢家为KD_B、LABEL或Fteacher，固定执行“Gaussian教师/初始化 → Clean学生5epoch”的边界：在Gaussian父轨迹上用Clean已冻结的教师配置合法选点，取同epoch真实父F初始化；学生硬标签与软标签都用未扰动的原train输入及源Clean BCE，三个匹配臂统一使用Clean主赢家的student_lr；KD_B和Fteacher使用Clean阶段已冻结KD赢家的非零distill_weight，LABEL固定为0。若主赢家是LABEL，其LR可以来自两个LABEL候选中的获胜者，不把LABEL的λ=0复制给软教师臂；教师family仍为Clean阶段已选教师。该唯一映射写入GAUSSIAN_TRANSFER_FREEZE，运行三个臂各五seed×5epoch，无新增HPO。实验condition标为`gaussian_d0_parent__clean_student`，不得写成学生继续Gaussian训练。若没有候选过门，直接复用已知Gaussian F结论，不新增Gaussian训练。

两数据集独立决策；单数据有效可以继续该数据的边界评价。Gaussian结果只能改变适用范围和最终研究裁定，不反向选择Clean参数。

## 9. Test冻结及四种强制比较

在任何新Test读出前一次性保存`GLOBAL_SELECTION_FREEZE.json`：所有已触发任务完成、候选/控制ID、五seed配置、每seed端点epoch/state hash、比较清单、研究门结果和发表模型身份。未完成五seed、RAM端点不存在、未决工程错误时冻结失败。不同块可提前冻结选择收据以释放未选状态，但不能提前读Test影响后续块。

必须分别报告：

1. **完整方案 vs 三个已发表模型**，主要裁定对固定最强发表模型（ML M3oE、KR HyperGate）；逐seed基线原值从源ENDPOINTS读取。
2. **完整方案 vs 相同数据条件普通在线O**，总收益。KD还另列同初值/同额外步数LABEL控制；55阶段预算不能伪装50epoch等预算。
3. **新增组件 vs 无该组件的固定F**；A/B/C/KD对照身份不同，参照表写明。候选轨迹内F−online是EMA在该轨迹的作用，不替代此项。
4. **复杂方案 vs 更简单替代**：MATCH−RAW、B−S-only、C−GLOBAL及同αPARAM、KD−LABEL/Fteacher、AB−较佳单组件。

新Test只计算冻结列表。B网格只进入一个赢家及其同H/mode另一scope必要控制，不把12个Test当搜索表。C只测试冻结s的PRED/PARAM与GLOBAL两模式；KD测试赢家与对应两控制的online/F。A只测试已完整训练候选的online/F；AB仅触发时。O/F端点若hash/协议相同可复用历史物理读出，逻辑记录仍齐全。

Test上的+.003门分别相对每个清楚标记的对照输出；论文主门必须对最强发表模型，不用内部F冒充。新增组件即使达到总收益门也只记真实Δcomponent。不得读取Test后再增一轮HPO或更换最终候选。

## 10. 输出状态和研究收敛

结果文件分开保存`mechanism_decision`、`component_decision`、`formal_auc_decision`、`logloss_decision`、`cost_decision`与总行动：

- `ACCEPT`：该层次门满足；若正式AUC合格但LL升高，写`AUC_PASS_LL_COST`并给代价。
- `REVISE`：宏收益达标但单格保护不满足、E20/E35时机分化、或压缩损失；具体限定下一问题，不自动新增本轮实验。
- `STOP`：预定五seed动作/价值门完成后不足，结束该版本追加。
- `NEXT`：已冻结的后继阶段尚待执行，必须有实际依赖ID。
- `NOT_TRIGGERED`：条件分支未进入；`BLOCKED_ENGINEERING/RESOURCE`：执行受阻；两者不能作完整算法阴性。

当A反馈、B有限读出、C信息价值和被触发的KD/Gaussian全部按门裁定，且新Test冻结/评价/报告完成，本轮机制探索结束。无收益时保留固定F是明确研究决定；有简单替代时以该替代进入下一阶段，不要求形成MDMT特异复杂算法才算完成。
