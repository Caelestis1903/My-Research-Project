# N4条件拓扑实现设计（未触发、未实施）

本文件只细化原N4，不预判N3结果。N1.6与N3.6同时满足后，才实现和审查一个已冻结C子块/阶段对应的长期对照。子块由原独立确认选择，不从N4结果反选。

## 路由与初值

三拓扑为原tied、三份真实域untied、三份固定随机组untied。每个副本完整复制同seed原selected C子模块tensor，其他参数对象与原图语义一致。原始域与任务路由、fusion、routing detach不变。初始函数在浮点精度内核对；untied有效自由度增加是处理的一部分，不能称容量不变。

KR处理文件无稳定user_id的事实已在N0/训练前环境阅读中记录，按原方案使用稳定原始row_id，不能后验改成用户或换hash seed。固定hash_seed=2048：UTF8文本`2048:<integer row_id>`做SHA256，前8 bytes按little-endian无符号整数除以2^64得到u∈[0,1)，按固定目标训练p的累积阈值分三组。**绝非hash mod 3或三等份**。所有split沿用同一公式，记录实际train π、各split组人数、域×组×task支持；一个hash只支持该映射下结论。π取实际完整train人口组比例，独立于q。

可能的最小实现：训练/评估数据附加一个仅供路由的group索引列，仍保持domain为最后列；wrapper先移除该列再送原网络，原schema、编码器和learned feature不得增加group ID。这样冻结的预测与objective函数仍逐batch取同一行路由，不靠重复feature反查row ID。tied直接用原模型/原数据，不需该控制列。若selected模块可包装则在其输入行维分组、运行相应副本并按原顺序重组；若为shared STAR直接参数，只局部替换该算子的路由实现并保留后续原forward。实现只覆盖实际选定的一个模块，不无条件扩展全部模块。

## 正则与optimizer时钟

选定模块R=λ/2 Σ_k π_k ||θ_k||²，λ=1e-5；其余参数沿用原coupled WD。可等价采用每副本optimizer param group的coupled WD=λπ_k，避免显式R重复添加；若采用显式R，对应optimizer WD必须0。tied只有一个有效Parameter及一份state，梯度直接求和，R退化原一次正则，不能各副本Adam后平均。

每global step都为无数据副本设显式零data grad，再按既定正则/Adam或SGD更新；记录每副本state clocks及group曝光，不因grad=None跳步。不额外改变其他参数的更新定义。full train数据J与R/total分别报告，不能将π正则悄悄混入原数据J读数。

## 矩阵与复用

优化器取N3已独立确认者；核心共同eta严格N1该优化器的共同值。三拓扑×SN/SB×新seeds2040–2047×H100=48条。tied若严格退化原初始化/forward/loss/sampler/optimizer正则与更新，就复用N1所选优化器16条pilot；须书面恒等说明及最小同图/更新核对。否则tied也重新做16条，不能只凭输出近似就复用。

真实/random untied各做4LR×2q×tune2030/2031，共32新增pilot，完整H100；各拓扑/q独立赢家依同selection最优均值、平局较小LR。核心共eta48条与各自赢家性能对比分开；赢家不同才补相应2040–2047轨迹，完全同配置复用。所有正式轨迹逐epoch留model，test继续封存到全局冻结。

## 必要检查与结论边界

仅核对初始输出、tied聚合data/regularization梯度恒等、选定副本参数支持/零贡献时钟、hash人口固定、一个小规模保存恢复及真实完整epoch更新/尾批。无需无关架构/边界测试。实现后另写IMPLEMENTATION REPORT、自审PASS，才授权机械队列运行。

分析分共同eta topology×q交互与等调参预算绝对性能；独立训练seed是重复。保持AUC/LL/J、六cell、samples/steps和成本。真实/随机组差仍包含曝光、支持、局部优化尺度及额外自由度，不能单凭日志声称隔离纯域语义或唯一中介。gap消失且两臂受损须报告破坏性阻断，无方法收益。原始问题和未触发项均保留。

DESIGN COMPLETE；是否触发仍等待N1/N3，不是N4已实现或已运行。
