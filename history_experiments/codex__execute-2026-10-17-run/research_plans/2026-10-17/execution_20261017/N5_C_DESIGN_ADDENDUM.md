# N5-C固定MovieLens人口（条件分支，结果产生前冻结）

原规划N5-C要求关键KR现象经独立确认并达实际量级时，至少执行固定人口、对称调参、八optimizer seed的第二环境。为避免观察结果后决定是否值得外推，本执行采用N1.6已定义的重要残差规则作为保守资源触发：任一优化器触发即启动；该规则本身不当作统计显著性。只做一套最小第二环境：优化器沿用N2的优先规则（Adam触发优先，否则SGD）。不新增优化器筛选。

固定历史movielens_2022缓存的train/val/test行列表和历史seed2022 selection/diagnostic分半。subset_seed=2022与初始化/采样optimizer_seed显式解耦；不重抽子集。模型沿用对应环境的原fixed_model，Sample目标及Adam/SGD定义、B40960、WD1e-5与KR相同。每epoch暴露实际训练人口N，完整尾批，更新数ceil(N/B)，并报告与KR不同的samples/steps/相对进度。

所选优化器沿用N1四点LR网格，SN/SB × LR × tuning seeds2030/2031，共16条H100；共同eta和各自赢家的selection规则及较小LR平局规则与N1完全相同。所有pilot完成后才选。正式optimizer seeds2032–2039，两q共同eta16条；各自赢家不同则补相应八条，最多再16，完全同配置复用。主要共同配置与等调参预算性能分别保留。所有正式轨迹逐epoch保留model以支持全局解封后的test AULC；pilot只用selection/diagnostic。

H100主预算不因KR49步与ML约5步/epoch而视为同优化进度；本最小分支不另行增加ML训练到KR步数。若需要匹配step，会是不同样本暴露估计对象，必须另列，不能偷换。

本分支只回答固定ML人口下优化随机性的稳定性。三个subset交叉设计仅在完成此分支后，原耦合subset方差仍决定当前主张边界时启动；若仅报告两环境条件效应差异，不自动扩大。不给三个域做高自由度因果回归。不默认启动ORDER、重复、mask分支，它们各自仍需原规划的剩余解释条件。

实现策略：复用冻结Runner的step/fit/objective/state/save及capture保存，不改其源码；单独ML子类显式初始化固定ML bundle/split/config与优化seed。必须核对同subset跨optimizer seed的train/val/test行SHA完全相同、配对q初值相同、对应p/q身份、ML实际epoch步数、capture原样留存。仅做这些直接影响结论的必要检查。源码提交、自审PASS且资源触发后才可正式运行。

DESIGN COMPLETE；N5-C尚未触发、尚未运行。
