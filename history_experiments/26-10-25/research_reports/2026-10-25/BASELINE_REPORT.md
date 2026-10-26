> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`9622c158c10246f56f75e8ddaa8552c99888ff67334555a27c2c000656b2c134`。

# 近期方法有限适配pilot

六条预定pilot均已执行并进入BUDGET_EXHAUSTED；没有24epoch完整外部基线结果，因此全部标记BASELINE_UNRESOLVED，不提出外部最优或超过原论文方法的结论。主模型数据角色、batch40960、FP32、无AMP/TF32保持共同协议；方法特有的搜索、分组更新和优化器步骤计费。

| 方法 | ML最后完整epoch | KR最后完整epoch | 状态 |
|---|---:|---:|---|
| AutoIFS适配 | 13 | 7 | 预算停止；KR尚未进入reuse |
| STEM_DOMAIN_TASK | 13 | 7 | 预算停止 |
| STGU_DOMAIN_ADAM适配 | 19 | 7 | 预算停止 |

每数据集预留72个实测native等价epoch，运行前固定每方法分配24。ML每条硬期限26.8829s、KR76.9651s，包含进程启动、数据、训练、验证和I/O。退出码均124；调度器每8秒观察一次退出，所以收据中的32.15s/80.3s等为退出观察上界，不是实际超时计算时长。旧timeout状态因baseline runner未逐epoch写RUNNING而暂填epoch0，已依据完整last.pt元数据校正；原记录保存在RUN_STATUS_timeout_raw.json，没有补训或改模型。

AutoIFS保留官方四条减法信息流、低秩变换、退火稀疏搜索、rewind和冻结选择器/输入embedding后的reuse；本协议固定search10、rewind至search epoch10起点、随后reuse，属于数据和训练协议适配。来源：[AutoIFS作者代码](https://github.com/ChaohuaYang/AutoIFS)，固定commit见METHOD_ADAPTER_RECEIPT。

STEM按domain×task构成六个头，保留借用专家的detach语义与共享/自身专家梯度。来源：[STEM作者代码](https://github.com/LiangcaiSu/STEM)，固定commit见收据。

STGU采用论文Algorithm1的真实逐组更新与包含活跃私有参数变化的loss比率，历史平滑和异号负效应处理均保留。三域Adam与五种分区的穷举互正亲和分组是显式适配；作者视觉代码使用归一化AffinityPropagation及延后loss记账，不能称逐行官方实现复现。[STGU论文](https://arxiv.org/html/2502.11986v2)、[作者代码](https://github.com/wooseong97/sel-update-mtl)。本实现不是只有EMA动作选择的inspired模块。

六个真实schema小批检查全部通过：finite logits、一步更新与完整状态恢复；AutoIFS还检查search→reuse及选择器冻结，STGU检查未激活私有参数grad=None。该检查支持结构适配可运行，不等于完整24epoch质量验证或复现原论文分数。没有按外层test挑选替代对手；PRP/M2R等未核实核心材料不冒充已复现。

原始结果、最后validation、实际完整epoch、checkpoint路径：`../tables/baseline_pilot.csv`。固定来源及偏离：`../manifests/METHOD_ADAPTER_RECEIPT.json`；语义检查：`baseline_smoke.json`；状态校正：`baseline_status_reconciliation.json`；退出：`BASELINE_PILOT_QUEUE_bounded_execution.jsonl`及每run的TIMEOUT.json。
