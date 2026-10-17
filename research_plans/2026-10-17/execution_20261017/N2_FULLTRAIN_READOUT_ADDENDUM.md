# N2完整训练交互读数（test封存期间冻结）

原N2.4要求的必要统计汇总，不新增实验或选择。输入是全局test解封之后完整的n17_performance输出和同一comparison manifest；当前不得提前执行test。

固定组名：N1_common_adam为复用IID父轨迹，N2_quota_adam为新quota；若原门槛选择SGD，后缀按冻结门槛改为sgd，不根据结果挑优化器。每组同8个2032–2039训练seed、H100。

每个已冻结readout/metric在同seed内计算IID gap=SB_IID−SN_IID、quota gap=SB_quota−SN_quota、I_quota=quota gap−IID gap；另算SN_quota−SN_IID和SB_quota−SB_IID。保留四臂绝对值、所选epoch以及原始seed。selected为各轨迹同一既定selection规则下的读数；固定epoch与AULC照常分别报告，不合并阶段。

对每个效应用既定paired()生成8训练seed的t区间、exact sign randomization、20k配对seed bootstrap；这些是预设次级读数，点区间/未校正p不替代两个N1共同eta主比较的Holm族，不按有利readout决定成败。真实q标签保留；供保护界读取的绝对quota lift表中SN/SB仅为generic baseline/candidate字段，分别表示IID/quota，须注明固定真实q。

全部结论仍需同时检查六cell、域/任务、加权LL、两臂绝对变化和成本；只缩小gap不表示修复，未证明健康也不能称健康。配额改变高阶矩/批内依赖，I_quota不单独证明trace唯一中介。RUN仅限全部真实输入已完成、源/comparison SHA核对后。
