# B1 重建测量的分析合同（正式训练尚未开始时冻结）

主 agent 编写，2026-10-16。实现与科学解释由主 agent 顺序完成。这里冻结数值处理；代码执行不自动形成机制结论或触发新方法。

1. 默认只接受正式 `runs/seed2022–2025/{F,DH}` 的8/8完整输出；检查新study身份、source/config/prediction/full-state hashes、50epoch覆盖、真实选点与同seed初始化/数据/main及arch批次配对。工程数据必须显式使用单独入口、目录和非正式标签，不能填入正式矩阵。
2. 从每epoch保存的validation/test逐行预测独立重算AUC（sklearn）和float64 BCE（epsilon=1e-12），对照原epoch记录；严格核对row IDs、labels、domains，不能以数组长度代替身份。完整state按实际schema、model/optimizer参数名称覆盖、m/v/step和计数检查，不把工程PASS当作400个状态已核验。
3. 两split均输出每seed×arm×epoch×domain×task的AUC/LL、类别计数；另列macro、target d1、non-target d0/d2、各domain/task、worst-cell、worst-domain和按evaluation行频率加权cell均值（不是pooled AUC）。training loss和selected/common/final并列。
4. R0 macro是主要比较，R1–R4为原规划的法证敏感度。独立复算validation选点，与冻结selection.json一致。分别比较各臂自身选点，以及所有同epoch差；选点时序分解保留F锚点与DH锚点两个版本并检查恒等式。
5. 训练随机性按4个配对seed报告均值/标准差/范围/同向个数，探索门槛ΔAUC+.003和+.005；不把4/4同向当作显著性检验。以各cell损失+.003 AUC（下降）作为披露成本，不把它声称部署安全标准。
6. 选中R0模型的预测使用同row、同标签分层paired bootstrap，固定RNG=20261016，1000次。每split/cell重抽一组row权重，并同时应用于所有seed模型，输出各seed与固定模型集合平均效应的边际95%区间。这个区间只描述给定训练模型的数据抽样变异，训练seed分布另报；不同任务独立分层，不拼造macro CI，不做未经预注册的显著性筛选。
7. 保存选点分数分布/固定10bin校准、Brier/ECE和按全局row-order四个窗口的分域任务指标。窗口标签明确为row-order proxy，不能未经原时间字段证明称作真实日期。bootstrap未建模时间依赖，不宣称全部泛化不确定性已消除。
8. 科学解释必须检查稳定现象、相互矛盾的seed/模型/阶段、STRONG/MEDIUM/WEAK支持范围、训练域规模和baseline难度，检验“小域收益/大域代价”是否真存在。比较timing、有限正例、条件关系变化、C/A共同干预及过拟合等竞争解释，给出最有区分力的下一项实验。
9. 只有实际8/8、配对与状态核验全部通过后才生成正式数值分析。工程分析用于验证处理代码，始终formal_runs_completed=0。后续B2/C1/C2/D1/E1按原规划逐项判定；全目标完成还要求三份最终报告、关键表图归档及远程push确认。
