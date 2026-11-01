# 06｜Codex完整执行提示词（唯一入口）

你负责在 Caelestis1903/My-Research-Project 中实现、审查并按条件执行本目录的夜间研究队列。**本规划授权范围内允许自动执行，无需阶段间询问用户；不允许任意扩展研究方向。**

## A. 开始前核验

1. 读取本目录 00–05、experiment_queue.yaml、根目录 AGENT.md、PROJECT_ARCHITECTURE.md。
2. 核验 main 最新 HEAD；本规划写入时基线为 0973fa5f1a804aec34f88ccb2ffdd8c0231b602b。若 main 已前进，以当前 main 为基础增量创建文件，不 reset、不 force push、不覆盖新提交。
3. 核验最新证据分支 26-11-01_v3 HEAD 6f5742c2085c1e49d1767b0a7079bb9f28160eda；读取 AIGM01 REPORT、SEARCH_REPORT、DESIGN_SPECIFICATION、IMPLEMENTATION_REPORT、EVIDENCE_REVIEW、REVIEW_REPORT、RUNNER_REPORT、decision JSON、原始数表和代码。
4. 核验前置计划 research_plans/26-11-1_v2/00–06 与 GAM01/GA31-R3 依赖；读取 MASK 历史文档与可找到源码、Polyak 旧实现及日志。旧 MASK 或 Polyak 原始代码缺失不阻塞其他候选，必须记录缺口，不得宣称逐行复现。
5. 核验真实数据 CSV、六数组、行号、encoder/scaler、自然 split、模型结构、Sample 目标、Adam 配置和实际 GPU 外部任务。禁止改变人口、标签、采样、loss 权重预算、batch=40960、50 epoch、FP32。
6. 输出 source receipt：ref/commit SHA、实际读取路径、blob SHA、数据与源码 hash、环境、资源占用、已存在结果与缓存复用边界。AIGM01 RAM 权重已释放，不可宣称可继续训练；历史 test 已参与判断。

## B. 实现边界

新建独立实验包和报告目录，建议 experiments/gradient_aug_time_select_20261102/、research_reports/2026-11-02_gradient_aug_time_select/、artifacts/gradient_aug_time_select_20261102/attempt_<timestamp>/。不修改历史实验包、研究报告、原始数据、旧计划或历史 CSV。

严格遵守 02 的公式与 03 的队列。禁止新增未登记候选、全开关笛卡尔积、全参数网格、无关优化器竞赛、网络结构变更或新域关系模型。组件关闭时严格退化到 A0/A1/固定 EMA，且不消耗关闭组件的 RNG/梯度计算。

### Shadow 数量和分配

任何单条在线轨迹最多 4 个 shadow：
- A0 轨迹：固定 EMA、延迟 EMA、SWA W20；不维护 adaptive/replay。
- A1/B 轨迹：固定 EMA、adaptive EMA、replay-rho；若还需第 4 个，只允许 SWA，不运行延迟 EMA。
- 固定 EMA 必须始终保留。Replay 与 adaptive 必须有独立 shadow 参数状态，不能复用同一状态后声称同时比较。
- 每个 shadow 的验证 forward、选点次数、内存和耗时都计入成本。shadow 不反馈 online。

## C. 公式与实现核验

1. B1 clean per-example loss 仅用于计算难度 rank；只控制增强 Gaussian 在非路由 sparse embedding 字段上的覆盖率。真实 loss、label、domain ID、clean embedding 不改变。mask RNG 独立 keyed。
2. B2 分别得到 target task0/task1 的 clean gradient 与 augmented-clean delta；同 θ 同 batch。只删除 delta 相对于同任务 clean gradient 的反向分量。投影按 E/P 参数组聚合；U 组语义需真实 batch 核验。weight decay 只进入真实 optimizer 通道，不进入 delta 或投影参考 g_t。
3. B3 关闭 E1–E5 增强，E6–E50 lambda=.5555556；随机关闭控制用 seed 1729 预先冻结五个关闭 epoch，后续 lambda 同为 .5555556。实际增强总权重与 A1 相同。
4. C2 adaptive EMA 的 CV/rho 公式按 02 固定；rho 序列每 100 step 更新。为避免每步额外 backward，CV 使用当前训练步已计算的 optimizer 输入梯度 global norm（Adam 预条件及 weight decay 前）；A0 中该量等于 clean full-real gradient norm，A1/B 中则包含该臂实际训练目标的增强处理。Replay 对同一 rho 值序列做固定随机排列，使用独立 shadow。不得读取 validation/test 调整 rho。
5. Adam 一阶/二阶矩、bias correction、coupled weight decay、epsilon、step 计数沿用当前 AIGM/GA31 已核验语义；不把 EMA shadow 说成 optimizer momentum。
6. 记录 RNG 状态、loader generator、视图 key、mask key。诊断、shadow、validation 不得改变在线训练随机流。

## D. 实现审查与 Stage0

- 独立 reviewer 逐条核对 02 公式、03 矩阵、数据协议、参数分组、RNG、开关退化、shadow 上限、RAM restore、成本账本和队列自动分支。
- CPU 公式/单步测试与真实 CUDA Stage0 必须通过。Stage0 每数据集至少覆盖 clean/aug task gradients、B1 mask、B2 projection、B3 lambda budget、Adam update、shadow update、RAM snapshot restore 下一步一致性、U 组梯度和显存。
- Stage0 不打开 test、不计性能结论。Review 或 Stage0 不通过时只修复阻塞问题并重新审查；不得开始正式训练。
- 公式歧义先停止并写明冲突来源，不允许 coder 自行猜测。

## E. 自动执行队列

严格按 experiment_queue.yaml 和 04 顺序运行：
1. Audit / source receipt。
2. Build + independent review。
3. Stage0。
4. 20 条开发轨迹：A0/A1/B1/B2/B3，开发 seed 2022/2024，完整 50 epoch。
5. 分阶段分析、匹配控制回放/必要时额外轨迹。
6. 仅按注册触发条件执行最多 8 条附加配置轨迹。
7. A 组和 C2 先行裁决；最多一个 B 候选进入 C1，最多两个 B/C 候选进入确认。
8. 确认 seed 2023/2025/2026，完整 50 epoch。
9. 候选和 checkpoint 身份冻结后统一 test，完成报告归档。

第一候选失败时自动运行下一个已批准候选；失败不是停止整个队列的理由。全部候选失败时仍完成 A 组、匹配控制和结论归档。不得放宽门槛或添加新候选。

## F. 资源与可靠性

- 现有 GPU 任务不得终止或抢占。启动时检查每卡显存/功耗，先单 worker 校准吞吐，再按真实资源调整并发。
- 不缩 batch、epoch、seed、FP32 或改变数据划分。资源不足只降低并发。
- 每个 run 有唯一 run_id、config/source/data hash、状态 ledger、时间、GPU、epoch/step、validation 次数、forward/backward 次数、stdout/stderr 和失败原因。
- 可恢复故障最多重试 2 次；OOM 只降低并发后用原配置重试。
- 遵守 RAM-only checkpoint 规则。不得将模型权重/optimizer/shadow/RNG 快照写磁盘；无法跨进程恢复时准确报告。
- 10 小时停止派发低优先级任务；12 小时硬停止新任务并汇总。超预算时按 04 顺序取消附加 trial、第二候选确认、低优先级候选；不得取消 A0/A1 与 B1/B2 的最低开发覆盖。

## G. 评价与交付

开发只用 train/validation；候选、超参、选点和模型身份冻结后统一 test。报告 validation-selected macro AUC、macro LogLoss、pooled AUC、traffic AUC、worst cell、全部域×任务、最大域两任务、增强目标域两任务、E10/E50、逐 seed 配对差、均值/SD/区间、正向 seed 数、成本和复杂度。

区分调参收益、增强收益、梯度修正收益、时间平均收益与交互收益。方向一致/门非零/梯度变化不等于效用。

交付完整源码、独立 review、Stage0、配置与 source receipt、run/trial/cost/failure ledger、原始 validation/test 指标、配对分析、阶段 decision JSON、最终研究报告和可复现入口。只新增本轮文件，不修改历史报告。

最终逐候选按“当前结论 → 核心证据 → 适用边界 → 研究决策”组织；明确停止、条件保留和晋级候选。全部失败时也必须给出有决策价值的收敛结论。
