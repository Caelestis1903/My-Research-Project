# N2 实现细则（任何N2数据读取前）

原方案N2定义不改。此阶段仍等待N1，不启动固定点或quota正式训练。

quota每batch独立U，系统舍入n，域内独立WR；按真实batch B_t权重B_t*p_d/n_d，尾批40052，所有域n>0。保存n与权重；配置明示条件权重，原p/q仅为IID参照。沿用N1训练代码和保存协议，新增quota模块源码SHA单独记录；基座文件冻结不修改。

固定点16锚（N1 gate确定的一个optimizer，2032..2035两条路径E10/40），每四臂512draw。完整参数坐标，不用投影代替。对每锚完整扫训练人口算各域导数字段μ_d。每个draw逐域算组内mean梯度h_d；按IID的(n_d/B)*p_d/q_d或quota的p_d合并得到实际估计器g。每锚一次原全batch loss autograd与逐域合并数值身份检查。

在线保存每臂全坐标mean/M2及block trace（ALL、C/P/R/E、5个C子块），不保存不需要的512×完整残差矩阵。完整均值向量与精确μ保留NPZ本地。域内trace用条件无偏量n_d*||h_d−μ_d||²在线估计Σ_d各块trace；按各臂实际draw给出估计及MC标准误，不把有限样本恒等误差当bug。

IID预测within=(1/B)Σp_d² trΣ_d/q_d，between=(Σp_d²||μ_d||²/q_d−||Σpμ_d||²)/B。quota预测within=Σp_d² avg(1/n_d)trΣ_d、between=0。使用同锚、同B，随机舍入的E[1/n]保留，不替成1/(Bq)。实测total covariance trace及均值误差和MC尺度同报。block trace不等于下游机制解释。

N3.5若触发，N2同时对各完整梯度在同一原生optimizer state下在线计算Adam U(g)、共同滞后分母线性映射和SGD映射的均值及方差，最后与U(mean(g))比较；不必保留巨大bank。N2的512独立draw不当训练seed重复。是否触发此映射仍以N1/N2条件为准。

新增quota的必要验证只做系统舍入/实际权重loss及同图梯度身份、实际尾批、RNG保存恢复。既有IID图恒等不重复全矩阵审计。正式N2运行前留下实现报告与自审PASS。
