# 局部推进门统计口径澄清

本口径在primary读取P1候选Q结果和作推进判断前固定，解释原03、05、06已要求的局部统计，不增加阈值、训练臂或实验。

## 1. 决策用hard，soft独立诊断

06的“F预先定义的语义条件”采用**部署责任q的hard argmax分组**。对目标域d的请求，使用已由V按既定规则选择的候选模型冻结路由，`leaf(x)=argmax_k q_dk(x)`；精确并列按稳定叶身份排序取首项，inactive不参与。分组不接收标签。选定的条件及路由状态先登记，再对Q评分。

选择这一口径的依据：05的CL资格明确采用teacher硬需求组，03同时约束硬支持与soft ESS，06要求“共同分区”且分别导出soft与hard。因此硬组用作可复算的语义条件；原模型预测、训练和G统计仍使用原soft q，不改变算法。

同一seed内，对候选、指定对照和C*一律使用**同一冻结条件及相同行ID**。不能各用自己的最有利原型，也不能看Q后换hard/soft、换leaf、换任务组合或换状态。局部门主判定用V-selected；final沿该selected定义的条件单列训练终点诊断，不用final结果替代未通过的selected门。

soft责任质量、ESS、soft加权LL/收益、soft原型AUC按原06完整导出，字段明确为soft诊断，不参与此次局部门。对比soft与hard可揭示边界模糊，但不产生额外推进机会。

## 2. F确定条件与跨seed对应

在读取Q前为每个待判局部条件登记：dataset、目标domain、任务集合J、候选方法及selected状态、冻结路由、叶集合、F中真实动作收益画像、偏好来源动作和指定对照。任务集合是一个指定task或事前登记的两task全集，不允许在Q上挑有利组合。

F偏好来自该硬组的**实测带符号读取收益**，动作含OFF=0，保留task及来源动作身份；不能按mu距离或随机cluster编号赋语义。每task按F组均值收益最大动作标注，精确并列按既有稳定action顺序。跨seed按相同目标domain、task与F偏好source动作的可解释身份对应；source子叶需保留其根/父谱系，随机need leaf编号不作对应依据。

若同一语义键在一个seed内落到多个叶，在Q开放前将这些硬叶的并集登记为该语义条件；不能在Q中再选其一。若两个seed的F动作偏好没有可解释对应、条件未预先登记、或一边没有该条件，则该跨seed局部门为不可判定，并记录具体原因。原有F可预测性与支持条件照常执行，不新增Q样本量阈值。

此登记并不要求不同seed选中相同样本，而是要求各seed都由F确定同一可解释条件。03的分区稳定性、部署一致性和结构支持门保持原值。

## 3. 自然贡献以全Q有效标签数为分母

令A为候选、C为指定对照，`delta_it = LL_C(i,t) - LL_A(i,t)`，正值为改善；`m_it`是原有效标签mask；`h_i`是本节固定hard条件的0/1成员标记，已包含domain条件；J是预登记任务集合。

```text
M_Q       = sum_{i in Q, all t} m_it
M_group   = sum_{i in Q, t in J} h_i * m_it
group_gain = sum_{i in Q, t in J} h_i * m_it * delta_it / M_group
group_mass = M_group / M_Q
natural_contribution = group_mass * group_gain
                     = sum_{i in Q, t in J} h_i * m_it * delta_it / M_Q
```

`M_group=0`时记UNASSESSABLE，不能填0或通过。全Q分母包含两task及所有domain，不能用目标域总量或等域F_probe样本量替代。

当前两task全部观测，设Q总行数N、该条件行数n：

- 单task条件：`natural_contribution = (n/N)/2 * group_task_LL_gain`。
- 两task条件：`natural_contribution = (n/N) * mean_over_two_tasks(group_task_LL_gain)`。

因此用户所述单task“组行占总Q行比例/T×该task LL改善”是正确口径。单task也不能以另一task未计入条件为理由免除其整体安全检查。

## 4. 原局部门原值保持

在两个开发seed均有同一已登记F语义条件、且满足原可预测收益差异要求后，分别要求`group_gain >= .001`及`natural_contribution >= .00005`。安全门在**完整Q全domain/全task预测**上计算：每seed macro LL损害≤.0002，跨seed平均macro AUC下降≤.0005，任一domain-task跨seed平均LL损害≤.003。安全门不缩到局部组内。

对于机制结论，同时保留06要求的相应模块匹配对照；相对C*的部署价值与相对模块对照的增量分列，不能换成最弱对照。不同seed按同一对照定义配对。报告明确区分通过、未通过与不可判定，并保留分子、分母、逐task差和原始支持。

本口径不从Q重新挑组，不使用soft门挽救hard门，也不把单task局部收益当两task整体收益；未满足条件时按06既有触发/停止规则处理。
