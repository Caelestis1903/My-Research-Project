# IMPLEMENTATION REVIEW — Stage 1

**Verdict: PASS。Blocking issues: 0。** 审查由独立于 coder 的 reviewer 完成，范围为冻结设计、实际实现、最终 `CODER_IMPLEMENTATION_REPORT.md` 及对应工程证据。允许执行冻结的40条 Stage 1 训练；Stage 2、HPO、Test继续等待结果门控。

审查源码身份：`a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`，共57个训练/公共源码文件。当前 `source_manifest()`、source receipt、7项CPU检查和8项full-batch smoke的hash一致。

## 设计与实现核对

| 核对项 | 实际实现与结论 |
|---|---|
| B/C公式及计算顺序 | 对照固定提交 pilot 的 `core.py`：键控候选采样、target-logit Jacobian、B的pair有效性/截断、C的BCE方向、概率BCE reduction及单次更新一致；未加入A投影/回溯规则 |
| 零扰动干预 | Z先计算对应B/C的假想方向、cap和keep，再对accepted delta执行 `zeros_like`；标签、target task和辅助归一化保留 |
| RNG及候选支持 | sampler使用独立SHA256键流，不消费全局numpy/torch RNG；B/Z-PAIR和C/Z-EXAMPLE共用各自候选流；训练后accepted mask允许因模型状态分岔而不同 |
| 训练目标与权重 | `clean+.25*(2*aux_mean)`；M=0精确clean；ledger按domain/task/label记录分子、数量、系数质量和贡献，辅助单项系数`.5/M`、相对clean比值`B/M`正确 |
| 支持可核对性 | 日志有候选原始row ID/局部索引/task/domain/label/pair ID、keep/skip原因及accepted ID/task；raw/capped/actual norm与cap齐全。接受行标签和域可由候选字段及mask恢复，跨task重复计数有记录 |
| 探针与训练图 | 临时hooks在finally移除；探针接回gate数值导数，普通训练仍沿用原gate detach；delta detach；探针和诊断用autograd.grad，不写参数grad或更新状态 |
| 训练预算与评价 | 两数据集、seeds2022–2026、50epochs、batch40960；单Adam和after-Adam固定EMA；每epoch只做clean validation，online/EMA独立选最早严格最大六格Macro AUC；共同E10/30/50保留 |
| Test及状态存储 | 新实现没有Test推理入口；模型/Adam/EMA/RNG/loader/best/selected状态通过既有RAM guardian提交，没有磁盘模型checkpoint路径 |
| 恢复续跑 | snapshot/restore覆盖模型、Adam、EMA、Python/numpy/torch/CUDA RNG utility、loader、epoch/global_step、selected/best和计数；CPU连续/恢复检查通过 |
| 数据与CLEAN复用 | 数据cache只读核验；537份复制文件逐文件重算SHA256均匹配收据，10条CLEAN完整且Test=false；源分支固定commit留存，新代码不运行时import外部pilot包 |
| 调度与身份门 | 每卡独立检查显存/功耗<85%、host RAM<90%；每卡未知首批峰值时最多一个allocating worker，之后继续资源准入；正式入口校验review verdict/hash和smoke PASS |
| 原始导出 | 导出完整预算/选点state hash/自然loader行序核对、六格/域/任务/Macro/Traffic/Pooled、五seed配对算术和accepted-support overlap；双空support的Jaccard定义为1；导出不作科学决策 |

## 使用的验证证据

- 最终coder记录：7项CPU检查通过，3.48秒；本review复用其最终相同源码证据，没有重跑整套检查。
- 两数据集×四臂共8次真实40960 full-batch backward通过；每次恰好1次Adam和1次EMA更新，loss及参数有限。
- 实际模型FP64有限差分最大绝对误差：ML `2.776e-11`、KR `2.870e-11`；FP32/FP64梯度最大差分别`1.607e-7`、`2.480e-7`；无关行梯度最大差分别`1.490e-7`、`2.384e-7`。均满足冻结容差。
- GPU reserved峰值：ML `1352663040` bytes、KR `1635778560` bytes；调度使用这两项实测峰值。
- reviewer另做一次针对性CPU同初态对照：分别对B/C比较历史pilot与新实现（开启首批gradient诊断），loss、clean/aux loss及model/EMA/Adam state digest逐项精确一致。两臂loss均为`1.0831468105316162`。这验证新增账本/诊断没有改变该步训练语义。
- reviewer核验最终57文件source hash及537份CLEAN复制文件hash；10条复用结果均为50epochs、CLEAN、Test=false。历史数据/协议/原始选点身份沿用已通过的Stage 0审计。

本次审查未发现需修正的阻塞项。PyTorch继承TF32设置的弃用提示不改变当前数值设置。正式运行按冻结矩阵执行，结果和科学门控由后续阶段处理。
