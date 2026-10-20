# N19 / CM19 主代理自审

**Verdict: PASS。Blocking issues: 0。** 按用户显式补充的 start.md 第11条，由主代理审阅设计、归档源码、转接实现和实现报告；不是独立子代理审查。

检查设计覆盖：N19原236及80控制、CM19原436及72控制全部一一列入，历史flags/config保留。源Git提交和每文件SHA可核。自然训练人口固定2022，优化seed不改变人口；原meta及selection/development代码保留。800167基础训练人口与760161普通main明确分列。matchedN的178126与曝光控制178125实际差1行如实记录。

检查算法：原构造、fit、step/update、优化器、损失、模型、mask/channel/EMA、秩warmup/prune及source-style重叠arch规则均原样调用。N19 first-shuffled arch与CM19 cycling arch保留差别。匹配曝光仅影响指定fixed/容量/STAR控制的原permutation，不干预其它随机调用。G4与G4LONG保持原机制和2350step预算。

检查结果记录：CM19 fit结束会把self.steps切回selected，故预算从完整receipt的final_steps核验；N19 trajectory的examples纠正为实际行数，未改训练。发现并已在运行前修复一次wrapper progress构造重复status参数错误。按实际rows核验预算并保存曝光/源/state索引。检查点test前后SHA固定；选模定义与development处置先关闭，测试后不再扩调参。

检查数值与解释边界：保留float32 AdamW名义WD及原kernel，不制造有效收缩；容量/路径与人口解释交后续实际证据。synthetic stability任务明确Bernoulli标签，不当native task1。局部counterfactual与完整训练分别记录，weighted cell不当pooled。旧人工checkpoint缺失不冒报重验。

源文件冻结后批准824完整训练和143原始诊断任务。测试仍须每scope的主代理closure，不因训练PASS自动放行测试。未运行实验不产生科研结论。
