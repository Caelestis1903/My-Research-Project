> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`109f19f66f06f6054dc5b6f8ff3c9b2716b8fff69f8f151421cb0f53b6aa38b2`。

# 成本计费修正

S2 首批9条已启动后，发现首次fork actor刷新、CAL_OFF独立验证和checkpoint序列化未全部进入累计预算时间。对39条未启动配置增加纯计时与checkpoint费用journal，保留来源hash变更收据。未改变损失、梯度、动作、随机流、actor拟合或AUC/LL主选点。已启动9条保留原始源码身份与账本，不用事后更有利checkpoint替换其budget_best。

早期预算选点因此只能视为不完全成本下的辅助结果，不用于宣称同预算优势。主selected及共同epoch24比较不受此计时修正影响。研究执行账额外列snapshot及首次refresh，S1早期actor计时包含cache，禁止与cache重复相加。

必要aux0包在已存在的epoch5快照分叉，新增实际费用全部进入reserve；独立部署仍含warmup。每数据集144 native等价epoch，按已测组件先分配K1 45%、K4 55%；完成后余额才可支持完整CAL闭环包。预算在原子epoch结束检查，任何至多一epoch越界单列，不延长预算来补齐未完成轨迹。

实际aux原子epoch检查造成超额：ML 193.396s /161.297s，KR 495.314s /461.791s。这是执行偏离，不能标记为严格守住硬上限。为防止后续重复，baseline及正式有界队列使用 `analysis_scripts/bounded_schedule.py` 的进程期限，包含启动、训练、策略、验证和写盘；到限保留最后完整checkpoint并标BUDGET_EXHAUSTED。轮询日志的exit观测时间可能晚于timeout发信号时间，不能据轮询延迟声称多执行了计算。此改动不改变模型算法，也不补跑已超限aux。
