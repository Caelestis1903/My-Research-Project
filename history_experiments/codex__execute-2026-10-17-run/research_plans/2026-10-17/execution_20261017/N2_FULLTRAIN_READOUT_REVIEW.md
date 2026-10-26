# N2完整训练读数实现/自审

实现experiments/n17_quota_readout.py；规格N2_FULLTRAIN_READOUT_ADDENDUM.md。此为原N2.4的必要分析代码，不新增采样/训练。

主agent核对实际n17_performance的paired_seed_endpoints列名、merge后epoch后缀、两原始group名称和八seed/H100、四臂差和I_quota符号、缺项/重复键拒绝、同seed合并、保护表generic字段与真实q标签、点区间/次级p作用域。分组budget显式转换Python int，避免结果JSON序列化失败。paired()直接复用既定统计函数，不另选有利阶段或seed。

AST PASS；independent_review=false。主agent自审PASS。当前仅完成实现/静态核对，未运行正式读数、未新增工程训练/假结果、未读新test；只有全局test冻结及真实performance输入完整后才可RUN。健康判定仍须六cell及LL保护和两臂绝对变化，不能从交互单独认定收益。
