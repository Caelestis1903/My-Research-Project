# V13 方法化十项条件工作表（尚未终局）

本表按原规划的十项并列必要条件记录现有证据与缺口，不选择最终类别、不颁发 GO/NO-GO，也不触发提前停止。任一方法化判断必须等待仍规定的实验、测试和问题闭环。观察到 baseline 差值，并不等于已有新方法满足 endpoint、四 seed 稳定性和不损害其他单元的要求。

|原条件|已验证证据及强度|当前缺口/边界|
|---|---|---|
|1. pre-overfit 阶段稳定 effect|固定 SGD .1 的四 seed 早期共同窗口 AUC gap 均负；calibrated Adam 的共同早期 interaction 也四负。MEDIUM|模型、LR、阶段和评价人口限定；不是全部配置的恒定损害，最终测试尚未完成。|
|2. 非 selection artifact|共同固定轮次和共同早期已经存在差值，且四种选择规则、共用所选轮次均保留。MEDIUM|不能把 selected 差值当成纯 trajectory effect；选择重采样区间与训练 seed 区间不同，冻结后的 test 待完成。|
|3. 非单纯训练速度|2B 精确时钟及适足预算已有方向改变和反例。WEAK 的唯一机制排除证据|实用平台不是全局收敛；部分配置可追上，不能把早期损害自动解释成永久性能下限。V5 最终预算与读出仍待完成。|
|4. 至少两个环境或可解释 boundary|KR/ML 联合窗口 16 与各自窗口比较已核验，显示显著窗口及环境配置束边界。MEDIUM|环境中数据、标签、维度、模型时钟等共同变化，不能唯一归为 N/难度；ML seed 同时改变训练子集。没有把第二环境未重复的负向宏观效应写成跨环境复现。|
|5. 至少一 mediator 有反事实支持|局部 whole-state 替换确实改变下一步指标，完整状态训练与共享块局部干预协议均已冻结。MEDIUM（仅局部干预）|局部直接效应不是长期必要中介。V11 完整状态最后预算/全套读出、V7 规定的 12 个局部实验及条件触发完整训练尚待完成；纯方向/块间尺度不能分离。|
|6. 排除主要 alternatives|目标/计算图身份审计 STRONG，泛指“只有 Adam/只有 selection/只有 late”有阶段限定反例 MEDIUM|repetition/coverage/exposure、谱 trace/shape、content-order/tail-clock 等仍存在联合干预或结构不可识别性；不能声称所有主要解释均已唯一排除。|
|7. SAF/D3/AdaTask 不能直接覆盖|D3/SAF 公开目标与本轮目标不同 STRONG；预定 AdaTask 组件仍有 q 敏感性 MEDIUM|这不等于全部现有方法不能覆盖，也不是新颖性证明。通用 sampling/order/optimizer-memory 已有先例，Q12/Q25 保留 BOUNDARY。|
|8. 改善真正 primary endpoint|尚无本轮新方法的完整 endpoint 证据|不能用局部 one-step AUC、late rescue 或缩小 gap 替代绝对 SN/SB 主指标改善。必须依据条件触发与实际完成项判断，不为了取得 GO 新增任意方法。|
|9. 方法在四 seed 稳定|baseline 和已经完成的干预按四 seed 配对保留，矛盾符号均报告|baseline 现象的四 seed 重复不是新方法四 seed 收益。区间较宽不证明等效零；只有核心关闭实际依赖覆盖 ±.003 的关键零效应时才扩 seed。|
|10. 不退化原 objective/其他 task/domain|绝对 AUC、loss、J 和六个 cell 均纳入预定读出；baseline 域难度背景与规模、正例率已实际归档|完整状态与局部干预存在指标/阶段分歧；不能仅看宏观差值缩小。最终方法若触发，须按同一目标及所有域任务实际验证。|

现有证据不足以签发 GO。这个阶段性事实不构成最终 NO-GO 停止证书；后续 mandatory 流程保持不变。最终报告必须逐项重写本表为终局证据，并且只选择一个规定的总分类。

证据入口：`V12_pretest_causal_synthesis_IN_PROGRESS.md`、`V3_final_baseline_selection_analysis.md`、`V4_final_budget_and_learning_speed_analysis.md`、`V5_Adam_final_matched_analysis.md`、`V9_final_cross_environment_analysis.md`、`V10_final_vs_calAdam_analysis.md`、`V11_shadow_direction_and_norm_analysis.md`、`V5V8_actual300_and_state25_final200_budget_analysis.md`。

四 seed 的同号重复、配对 bootstrap 区间与精确符号随机化检验必须分别陈述：n=4 时双侧精确符号随机化最小 p=0.125，不能把四个同号写成该检验 p<0.05。ML 的四 seed 还包含训练子集差异，遵循 Q43 的方差解释边界。
