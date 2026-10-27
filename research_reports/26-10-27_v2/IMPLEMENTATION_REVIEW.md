# 独立实现审查 — Relation-Basis v2

日期：2026-10-27。Reviewer：`/root/reviewer`；Coder：`/root/coder`。

**Verdict：PASS。Blocking findings：0。**

实现符合已批准的 `DESIGN_SPECIFICATION.md`，可进入数据重建及阶段 1 运行。Runner 应先成功完成实际 prepared 验证，再启动锁定预算的真实训练；每条进程启动时核对 GPU 资源。

## 审查依据

已独立读取 AGENTS.md、GB18030 编码的 start.md、v2 全部规划文件、DESIGN SPECIFICATION、CODER IMPLEMENTATION REPORT、SPEC_RESOLUTION、实际新包代码、两份测试文件、CPU stdout 和实现来源清单。用户最新的四 seed 指令覆盖入口文档中的旧 seed；当前有效矩阵为 2022–2025、两数据集、六方法，共 48 条适配轨迹和 8 份共同前缀。

本地目录没有 Git 元数据，来源按审计固定的 main commit 与逐文件清单判断。独立逐字节比较了新包 vendor 的 22 份文件与已审计原件，全部相同；40 份 Python 源码均可解析。新包导入闭包使用自身 vendor，相对导入不依赖 main 中缺失的本地顶层 DFD/RCF 包。

## 设计与实现核对

| 当前结论相关路径 | 实现检查 | 结果 |
|---|---|---|
| 数据身份与自然行角色 | pinned strict_data 从真实 raw 重建；完整编码、标签、fit-only 词典/scaler 与原 assign_roles 再验证；F 为 P_label=P_form∪P_read，S=P_cal∪P_update；两个任务共用 row_id | PASS |
| 标签权限 | native/source 仅 T0；适配 optimizer 仅 F；V 只选点；Q/S 只用于既定开发与诊断；E 标签在全局冻结之前被拒绝 | PASS |
| 共同前缀 | fixed1、原初始化后复制非融合参数、native24×batch40960、V epoch12–24；owner-only pilot12×batch40960；h256、两任务隐藏拼 phi64、T0_s population 标准化与教师转换检查 | PASS |
| 六方法 | 六个域任务私有头；B_REFERENCE ReLU32、DIRECT GELU32、TARGET_ONLY GELU40；READ 固定根函数与常量关系；LOWRANK/FULL 共用函数库与 reader，仅关系参数化不同 | PASS |
| 有向读取 | 本域来源不列入候选；foreign 四项与固定 NULL=0 softmax；同来源跨基底共享 w；rank4 U/GELU/V 残差、U=0/V 独立非零；每步 w 半径2投影 | PASS |
| 适配训练与选点 | 自然有效标签平均 BCE；缺失标签在计算前排除；20轮含第1轮 warmup；同一 Adam 连续使用；相同 F 行计划；候选仅5–20、V macro LL、1e-12取最早；继续训练至final20 | PASS |
| 指标 | FP64 logaddexp logit LL、原 logit 排序 AUC、六格 macro 与自然平均区分、单类 NA、逐任务 pooled 单列、逐seed配对和t区间 | PASS |
| 完整 RAM 状态 | model/Adam/RNG/训练模式/梯度权限/游标/行计划与config/data/source/native身份均保留；恢复先检查身份；selected/final 独立存在；新执行路径不调用磁盘模型保存 | PASS |
| 常驻生命周期与冻结 | 每前缀六臂全部完成后 CPU offload；全8worker/6方法/config/source身份冻结；各worker验证精确身份并等单独释放后才缓存和评分 E；结束前导出 selected/final | PASS |
| 固定状态诊断 | Q ALL_ROOT、F无标签均值关系替换、基底输出距离/U范数、关系熵/NULL、S来源教师风险；不重新训练reader或据诊断换checkpoint | PASS |
| 成本 | 唯一前缀秒数、适配秒数/峰值显存、诊断秒数、raw-X与缓存时延、参数与明确范围的MAC分别记录；无稀疏执行声明 | PASS |

实际新增参数核对：B_REFERENCE=49,926；DIRECT=49,926；READ=50,718；TARGET_ONLY=62,406；LOWRANK_RELATION=62,166；FULL_RELATION=62,846。关系参数分别112与792，680的差额未混入函数参数。

## 验证证据

Coder 的 `CPU_ACCEPTANCE_STDOUT.txt` 为 **10 passed，2.09秒**；已逐项检查测试确实调用配置、实际模型、warmup/联合 backward、mask、指标、RAM下一次更新、native/教师转换、数据权限、安全CLI、小型20轮适配链以及48轨迹汇总/八worker冻结条件。直接复用该有效证据，未重复扩大测试范围。

独立补查了原测试未调用的 `diagnostics.latency` 实际链：用固定小CPU输入、真实 NativeAdapter/PilotBank/SourceFunctions 与六种 Model，逐臂执行 raw-X、cached-prefix、batch1/256、50次warmup/200次测量。六臂均输出4条时延记录和部署参数账；raw/cached 最大绝对差为 `2.384185791015625e-07`，通过实现阈值 `1e-6`。此调用未训练模型、未写模型文件。

复用路径中的旧 `vendor_snapshot/.../state.py` 保留原 `save_checkpoint` 定义以维持原字节身份；新执行路径仅导入 JSON/哈希工具，全部实际状态捕获使用新包的 RAM `capture/restore`，没有调用旧保存函数。

## 非阻塞呈现缺项

`report.py:56` 和 `report.py:71` 的汇总字段筛选未纳入 `pooled_auc_task0/1`，因为这两个字段不以 `_auc` 结尾。逐seed的 pooled AUC 已由 `trainer.py:30` 导出到全部原始指标和 `SEED_METRICS.csv`，主选择指标、配对主指标与阶段门槛均不受影响。主agent已确认在最终汇总中从该逐seed表补四seed均值、SD与区间，并分别标明两个任务；无需为呈现补全重训。

## 运行准入与后续边界

本次通过的是当前实现的阶段 1 运行准入。实际 prepared 完整验证、native/source GPU训练、48条适配轨迹、真实成本和冻结后的 E 复评尚由 runner 执行并记录。阶段2仍按已批准设计依实际门槛触发，触发新算法时另走设计、实施和独立审查。

未追加网格、动态分裂或无关工程检查；未执行真实 GPU 训练。
