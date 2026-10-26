# N17 精度门槛补充设计（主代理，结果前冻结）

2026-10-20。N17 尚未开始正式训练、未产生 N3 新结果。按 start.md 由主代理补全原设计“必要时64draw”的数值含义，保留所有原32draw双bank及完整坐标。

每锚在两个独立32draw bank均结束后，对 development（原名 diagnostic）宏 AUC 的 C_full−native_SB、C_norm−native_SB、C_direction−native_SB 三个预注册配对差分别计算样本标准差/sqrt(32)。任一 bank 的任一可定义差 MCSE > 0.000025，或因方向未定义/非有限而不足32个配对读数，触发该锚两个bank均增加32draw。门槛是原1e-4辅助效应分母阈值的四分之一，只检查精度/可定义性，不检查效果符号、显著性或是否支持旧假设。每bank最多64draw，无第三轮加量；最终仍不够精度明确报告 WEAK，不无限追加。

独立性：原bank0=draw0..31，bank1=32..63。额外bank0=64..95，额外bank1=96..127，各自只与本bank拼接，不能把bank0扩成0..63而重用bank1。所有子块/逐任务域原读数均保留，额外draw运行相同原local全比较，统计以bank和seed层级列示。

该分支在原始N3结果后由机械命令按冻结门槛生成manifest；主代理负责科学解释。它不改变训练、选模、test冻结或LR/H200规则。
