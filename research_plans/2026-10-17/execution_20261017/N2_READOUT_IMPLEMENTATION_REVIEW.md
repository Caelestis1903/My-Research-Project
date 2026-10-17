# N2固定点汇总实现与主agent自审

规格：N2_READOUT_ADDENDUM.md；实现：experiments/n17_geometry_readout.py。

## IMPLEMENT

只读正式16锚的config/receipt/geometry/moments/counts，核对锚SHA、完整四臂/10块、统一源指纹、512draw和预设seed×路径×阶段。输出原始矩、quota/IID与q的trace对比、同锚向量均值/Adam映射差以及每路径/阶段4训练seed描述性摘要。完整原始向量和counts保留本地并索引SHA；不创建新实验。

SGD线性映射差仅在optimizer=sgd时给出，明确derived；Adam未触发mapping时不伪造映射数据。MC RMS两臂之和为允许相关时的保守RMS尺度，不是置信界。Jensen和平均更新仅为参数存储舍入前的已冻结映射对象。

## REVIEW

independent_review=false，按start§11由主agent完成。逐项检查输入实际字段、参数坐标顺序、四臂差和交互符号、路径/阶段不混合、训练seed不被draw替代、缺省/零分母不补0、正范数MC偏差和科学解释边界。审查中将无Adam-map时的SGD派生分支限制为optimizer=sgd，避免对未触发的Adam配置误标，发生在任何正式N2数据产生前。

AST解析PASS；未增加工程训练、未生成假研究数据、未执行正式汇总、未读取test。静态自审结论PASS_SELF_REVIEW。正式16锚完成后的执行/输出核验仍待完成，不能将此文作为N2结果完成证据。

验证器修复补充：N2_validation_only_equivalence.json证明新/旧geometry只有同图验证和来源元数据差异。汇总入口新增显式--validation-equivalence，核对两源码SHA、反向恢复原SHA、修复配置原source指针；无该证明时仍要求同一source，base/quota始终须一致。原始矩/向量有限值作结果读取验收。首SN-IID验证draw的跨forward误差逐块范数（旧source仅全向量上界）相对MC单梯度和均值RMS单独报告，不能解释为所有draw的误差界。AST及主agent逐式审查PASS，未改任何统计端点或触发阈值，正式汇总仍待真实全部结果。
