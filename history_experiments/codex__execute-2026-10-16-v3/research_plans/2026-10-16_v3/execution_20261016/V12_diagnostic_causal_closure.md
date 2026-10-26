# V12 诊断阶段反事实闭环（测试规则冻结前）

所有规定诊断训练、配对读出、操纵身份和条件分支均已完成。测试仅作为全部条件冻结后的指标确认；不再从test结果改变训练、选择或方法候选。主控完成设计、实现、自审和分析，机械子agent只执行进程，遵循start.md §11；independent_review=false。

## 同目标与候选路径

J(θ)=Σ_d p_d N_d⁻¹ Σ_{i∈d}(ℓ_i0+ℓ_i1)/2。同一个既有stop-gradient图导数字段记为G̃_d，WR固定θ下E_q ĝ_q=Σ_dq_d(p_d/q_d)G̃_d=Σ_dp_dG̃_d=G̃*。routing emb.detach保留，不能把它误写成完整普通全导数；共同weight_decay=1e−5也不等于只优化无正则J。目标、图梯度期望相同不意味着协方差或更新路径相同。

q → 实际曝光/重复/覆盖/时序与估计器分布 → CPRE更新/优化器状态 → 后续θ → 阶段AUC/loss → selection规则选轮 → 固定test评估。数据/标签/模型/域任务基线同时影响该路径；时间展开的反馈不等于已单独识别的每条因果箭头。

## 六项规定反事实的实际结论

|CF|完成的控制与观察|判断|剩余限制|
|---|---|---|---|
|CF1 同J改变q|V1全部精确/MC身份、16θ×512批、完整经验图梯度、可控exact对照；V2/V4全四seed轨迹/选轮完成|同实现下q可改变路径和性能，MEDIUM；同图身份STRONG|不唯一识别中介；MC点状区间不是同时覆盖保证|
|CF2 同q改变optimizer|fixed与calibrated Adam/无动量SGD；LR/batch/预算；V10/AdaTask和V11所有触发状态/影子控制完成|Adam通用必要性被SGD反例否定，MEDIUM；calibrated interaction四seed负|固定参考interaction不触发；LR/映射/进度共同变化|
|CF3 域内sampler|Adam最终50/100/100/50；SGD最终50/50/50/350；原始WR/PERM身份、40训练4200臂轮205800批权重PASS|方案整体干预有条件作用；必要重复/coverage不唯一可识别|固定E,N有R=E−U≥max(0,E−N)，SB小域E>N不可能零重复同配额对照；宽PERM CI不证明等效零|
|CF4 同multiset改order|四seed实际完整batch集合/row次数不变；全部最终性能与22规定前缀完成|SGD共同E50交互+.005816四正；final均值−.006792，seed25反向；MEDIUM阶段路径敏感性|964/1000臂轮短尾移动；content与size-clock耦合，IID置换不是curriculum|
|CF5 同θ替换CPRE|4seed×1/10/40共12局部、7条件全部exit0；4788读数/12996对照/3249统计组重建PASS|C的selection AUC筛选E10/E40四seed≥50%为阳性；MEDIUM局部路径证据|C40 selection loss与trainJ各2正2负、平均衰减负；原联合一致性门槛未满足，条件full-training NOT_TRIGGERED；不作长期必要性或机制不存在结论|
|CF6 batch/确定性|2B全足预算、20精确步数读数/2448首次hit与censor；可控exact/stochastic身份|速度与时钟是配置/阶段边界，MEDIUM；exact同路径为身份控制|2B同时改变步数/clock；toy不能替代真实非凸全模型泛化|

## 必须保留的反例与证据

固定SGD.1共同早期50 AUC gap−.003043四负；固定Adam.01早期27−.006980四负；calibratedSGD早期37仅−.000493、1正3负。calAdam与calSGD的共同22交互−.013910四负，但固定参考对照不触发。不能把学习率或阶段边界删掉。

V11全部状态最终预算250/150/100/200、12最终+27历史配对和156000选择检查完成。v-only原始22轮交互+.012912四正，SN绝对−.012157四负；联合4轮交互仍正但绝对健康估计不同。共同E50两臂AUC均四seed改善，gap交互不稳定；β2=0破坏性floor不构成必要性，m-only长期仍学习。global等范数不保持CPRE各块幅度（Q42），因此不识别纯方向/唯一长期历史。

V8三强度最终50/200/50/350；seed23全部1200、seed25全部2100批物理SHA相同，24来源576单元上下文/384稀疏范数已完成。目标小域SN平均AUC随mask增强下降，但E50 mild loss稍好、q差距非单调；mask改变特征规律/J_c且只有一个固定嵌套实现，不是自然困难度唯一操纵。真N主要解释不可识别；名义池2x/4x精确别名不引入新支持。没有普遍的小域改善/大域损害规律，WEAK。

ML第二环境四seedSGD完成；联合16与各自窗口不同，负向宏观效应未稳定复现，预设深层复制条件未触发。其seed同时改变训练子集（Q43），环境配置束不能归于唯一数据集/N/困难度。AdaTask固定组件.003仍有q差异，长期两臂绝对AUC改善但相对gap更负；不能把差距缩小等同健康，也不能声称完整论文方法普遍失败。

Q34最后254归档快照756800遥测行已刷新，原mixed m/v范数保留并输出corrected m；前缀与克隆不是独立seed。Q35仅前6batch移动θ稀疏lag；不能宣称全epoch梯度自相关。Q45固定100有预算资格n2，不能写四seed；Q46明确保留C局部AUC阳性与loss冲突。

## 竞争解释与可证伪判断

目标实现不一致已被同图身份反证（STRONG）；“全部只由selection/late/Adam产生”均有配置内共同早期反例（MEDIUM）。速度、信息/表示变化、coverage/repetition耦合、short-tail clock、优化状态映射与多通路共享效应仍是竞争解释。当前不声称任何单一中介必要，也不把不显著当等效零。没有核心关闭依赖覆盖±.003的关键null，因此额外2026–2029分支不触发。

最有判别力的后续按问题分别是固定batch大小位置的顺序控制、可行E≤N的支持/重复控制、预注册健康/进度匹配的状态映射，以及新AUC-only假设下容量/初值/预算匹配C私有化。这些是当前强外推的边界，不能用未来工作掩盖本轮未跑项；本轮所有原规定且已触发诊断项已完成。

[完整V5](V5_SGD_final_matched_analysis.md)、[V7](V7_local_counterfactual_analysis.md)、[V8](V8_final_matched_diagnostic_analysis.md)、[V11](V11_fullstate_final_diagnostic_analysis.md)、[文献](V10_existing_work_coverage_boundary.md)、[问题全集](question_registry.json)。最终test待冻结后实际运行，最终总分类尚未签发。
