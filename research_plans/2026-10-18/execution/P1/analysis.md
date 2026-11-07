# P1 首轮判定：DATA-BLOCKED

科学问题真实存在：合法稳定目标在给定支持和信息保留条件下可能不可行。
本轮两个支持控制分别给出最小绝对交叉矩 1 与 0；pooled 抵消和共同权重
冲突也经精确 LP 检查。边界的证据为 **STRONG（限声明的离散模型）**。
普通支持 LP 和选择性保留已完整表达这些结果，没有新增于 DVD/一般
可行性分析的算法收益。未声称 SAF 已被实验否定，相关条件见
[`../LITERATURE_AUDIT.md`](../LITERATURE_AUDIT.md)。

数据尝试不是只看字段：已读取 KR/ML 原表与缓存 schema，并把 ML 评分
时间戳按保留 row_id 接回训练样本，核对 rating 标签，计算 3 域×4 时间
分位的两任务正率。ML 最小域 task0 正率从第 1 分位 .6051 到第 4 分位
.5100；其余域也有非单调时间差异。这支持“评分人口随时间变化”的描述，
不识别哪些变量是可去除 shortcut、哪些应保留。KR `upload_dt` 是视频上传
信息，缺曝光时间与策略，不能把行序 split 改称政策干预。

数据证据为 **MEDIUM（描述性时间环境）/WEAK（稳定机制与方法有效性）**。
本轮未训练稳定网络，因为尚未获得可合法指定的 nuisance、目标 shift 和
独立外部环境；训练 HSIC 下降不能回答该问题。未测新方法 AUC/LogLoss。

停止的是当前数据上的新稳定算法开发；问题并未被否定。最小下一步是
取得一项明确的环境/策略变化及其不应改变的任务机制，再比较普通条件
平衡或 DVD 是否已达到可行界。若它们解决，则 NO-GO；只有证书还能改变
外部环境风险，才进入稳定学习 pilot。当前不具备顶会方法证据。

代码：`experiments/n18_theory_data.py`。证据：
[`support`](../theory_data/P1_support.json)、
[`ML 时间分组`](../theory_data/P1_ML_rating_time_environments.csv)、
[`时间定义`](../theory_data/P1_ML_time_definition.json)、
[`数据单元统计`](../theory_data/data_cells.csv)、[`推导`](derivation.md)。
