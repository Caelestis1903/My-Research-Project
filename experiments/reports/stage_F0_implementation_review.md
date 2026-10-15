# F0-B1-ML 独立 IMPLEMENTATION REVIEW REPORT

日期：2026-10-12。审核依据：`research_F/F0/F0_Training_Design_Addendum.md`、`research_F/F0_Design_Specification.md`、实际 F0 实现及测试、Coder implementation report。Reviewer 独立检查源码，不以 Coder 声明代替验证。

## 1. 最终 Verdict

**PASS。** 首轮 FAIL 中的 MAJOR 实现问题及随后 PASS WITH FIXES 中的两项 MINOR 已关闭。本结论通过实现审核门，不自动通过数据、GPU、存储或实验工件冻结门。无未修正的 CRITICAL / MAJOR / MINOR 实现问题。

## 2. Requirement-by-requirement verification

| 要求 | 独立核验与结论 |
|---|---|
| ML natural、80/10/10、train-only encoder/scaler、物理域/构造标签 | 代码及合成数据测试符合；完整 base-train 前缀无历史 rotation 抽样。真实数据来源、重复事件及用户交叠仍需数据门审计。 |
| SDMT 数据/ID配对与全局 schema | 过滤相同物理行和标签，local routing=0，保留全局 embedding shape，并有显式 pairing assertion。 |
| M3OE、MMoE、terminal CGC-1、task-major | 结构、融合、gate detach 和 routing 符合；CGC 删除 unused shared-output gate，前 K cell 输出与旧实现等价。 |
| D/T=1 | 删除不存在的 beta，保留 alpha；额外独立合成解析验证 D1/T1 融合值与两个 alpha 梯度路径正确。 |
| Q 参数匹配 | 六项实际非 embedding 参数为 576482、576104、581300、573474、579290、573490，与冻结表一致。 |
| 初始化语义配对 | 所有 scope 共享同 seed/backbone/setting 的 MDMT source；physical domain embedding 行映射及 source/target hashes 可审计。 |
| Sample/Equal loss | 分母按行及本地任务数，Equal 使用固定 train shares；非满 batch 按实际行数合并。B1 无缺失标签。 |
| search/formal/追加 | 288 search、48 first-four formal；全搜索覆盖后 validation 选配置，earliest-max；追加为完整 12-unit × seeds2026–2029 的48条，loader支持48/96条。 |
| 统计 trigger | n4 使用 df3 的 t.975=3.182446305284263；先验证五项完整有限向量，再判断跨零及应用 margin。分析阶段的主/保护统计族仍须按设计执行。 |
| checkpoint/resume | 权威 epoch commit 加 selected/fixed 可恢复投影；七个提交点故障验证通过，并核对权重与 optimizer hashes。 |
| formal routing probes | selected/final × train/validation 固定 hash 选行；gate/entropy/top1、融合、相似性及 dense 执行数齐备；重复输出与完整状态 no-op 审计通过。 |
| test 隔离 | 训练不计算 test 指标；selected API 接受路径与冻结身份并内部恢复，验证 snapshot role、run/config/epoch 及文件/model/optimizer hashes。 |
| 并发与身份 | canonical flock、独占 GPU lock、原子 status；共享五个 process slots；两个 owner 保留相同 GPU index/UUID。启动端仍负责同时存活进程总数。 |
| 资源/hash/矩阵 | 资源 schema 诚实保留 pending measurements，epoch记录墙钟、吞吐、CPU/GPU峰值与存储；28文件依赖inventory；3416必需轨迹及未实现block明确保留。 |
| baseline | 历史 S1 实现未改；新 reduced-scope 归一化身份与旧参照分开。 |

## 3. Bugs/deviations 与关闭证据

- **MAJOR-01，关闭：** `slot_bias.N` 在 width 不匹配时初始化异常。已显式零初始化；生产 builder 测试覆盖全部模型/物理域。
- **MAJOR-02，关闭：** runner 的 scope-dependent root 导致假配对。已统一 source root，embedding/物理行/hash 测试通过。
- **MAJOR-03，关闭：** n4 使用错误 t 临界值。已改 df3，独立边界测试通过。
- **MAJOR-04，关闭：** latest commit 后 selected/fixed 未写导致恢复丢失。已实现投影恢复；七个故障点及 Reviewer 原始“首次选中后中断、后续同分”复现均通过。
- **MAJOR-05，关闭：** 必需 routing probes 未实现。已接入 formal 完成路径并验证完整 no-op。
- 上轮 selected 身份、全块扩 seed、资源字段、dependency inventory 与 ledger 缺口已关闭实现部分；真实资源测量和共同 manifest 仍是外部门。
- **MINOR，关闭：** trigger 短路掩盖后续坏输入。现在先验证全部五项；首项触发而末项 NaN/长度不足均拒绝。
- **MINOR，关闭：** canonical owner 重写丢失 GPU UUID。共同 `gpu_identity` 写入 canonical/GPU owners，实读两份 lock 文件验证一致。

## 4. 独立检查与工件身份

Reviewer 在项目环境独立运行三组并发 tests：17+7+12=**36 passed**，最多3个 Python 进程；额外独立验证 D1/T1 解析融合/alpha梯度及原始 plateau checkpoint 中断案例。最终两项 MINOR 修复后最小定向检查 **6 passed**（含两个新增测试），最多2个 Python 进程。

Coder 报告最终完整 suite 为 **38 passed**。Reviewer 未再次完整重跑最终38项；独立验证范围是前轮完整36项加最终新增两项及相关定向回归，不能把 Coder 的38项运行写成 Reviewer 同次完整运行。

- 最终 Coder report：`experiments/reports/stage_F0_implementation.md`。
- report SHA-256：`6132ca5cbcca720747c6eab471827a9941df990dac4ad54e2853690a38568305`，Reviewer 独立核对。
- 实现及依赖 inventory：28文件。
- inventory 聚合 SHA-256：`b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`，Reviewer 独立重算，与 Coder 最终报告一致。

## 5. Corrections made 与 Remaining risks

Reviewer 未修改源码或算法；仅新增本审查报告。未运行真实数据训练、实际 test 模型评价或科学结果分析；合成 optimizer/resume fixture 属于单元验证。

CPU 合成验证不能证明 CUDA 确定性算子、batch40960 显存或实际资源预算。process lease限制持槽执行数，启动端仍须保证最多5个同时存活的本任务 Python 进程，不得另启动过量等待进程；直接 API 调用也计入上限。GPU锁须使用共同 output root 和一致设备映射。

## 6. 是否安全进入 RUN

**实施审核 PASS；可进入 Runner preflight 与冻结工件阶段。** 真实 search 启动必须另完成：ML数据审计PASS、GPU/目标batch显存与确定性检查、首块至少1.3倍估计存储空间、设计/协议/矩阵/数据/代码/队列共同manifest及run gate冻结。当前报告不证明这些门已过。

门通过后先执行冻结 search；第一批开发轨迹按3模型×2batch实测成本并保留其search身份。formal须等待288条search完整及12个配置冻结。test须等待validation扩seed/P决定和selected checkpoint身份冻结，并满足相应完成门。每卡最多一个训练worker，总Python进程最多5。

本报告只审核 F0-B1-ML；完整F0仍PARTIAL，其他未实现block、KuaiRand provenance与独立确认边界继续保留。
