# N2/N3 主agent自审

依据原规划、N2/N3设计补充、实际代码、实现报告与必要smoke；遵循start第11条，非独立review。

quota：随机系统舍入使用独立batch U，所有实际配额正、和为B_t；域内WR，权重B_t*p/n。新增perbatch权重与源码记录，尾部位置/更新数沿用N1。resume恢复同一单流的完整bit-generator状态；不把p/q误用为quota实际权重。

geometry：每臂512独立draw、全部1,614,254坐标；同图导数字段μ来自完整人口，IID按n/B*p/q、quota按p合成。域内trace用条件无偏n||h−μ_d||²，计数项按公式，quota E[1/n]用实际舍入分布读数。mean/trace的MC尺度、模型不变和梯度身份分开记录；不以512draw冒充512训练seed。Adam数学更新的parameter-rounding差有单独数值检查，不宣称其已证明长期状态机制。

local：同checkpoint原生optimizer state加载，每q梯度只在anchor计算；候选完整复制C/子块，范数方向四格保留其他SB坐标；zero方向不定义。完整J不由batch loss替代；两个validation人口分别报告，不读新模型test。开发/确认seed、draw及阈值按设计记录，筛选不得进入训练队列自行决定。

必要检查已通过；geometry的工程小样本不要求MC均值数值完全相等。添加的trace MC输出/CLI配置属于运行记录，不改变已核验sampler/gradient/optimizer映射身份。正式运行前源码须提交并保持冻结。

Verdict: PASS。仅批准依原规划与已冻结N1门槛启动，不意味着N2/N3已执行或已支持机制。
