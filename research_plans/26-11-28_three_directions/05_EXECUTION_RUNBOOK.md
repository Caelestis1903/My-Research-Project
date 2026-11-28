# 05 — 执行手册、资源管理与恢复

## 1. 执行阶段

1. S0：main HEAD、源文件 SHA、数据 split hashes、optimizer manifest、EMA、Gaussian 和 A21 identity audit。
2. A0/B0：独立单元测试与首尾 batch 数值等价检查。C0 仅做可用 artifact audit。
3. REVIEW：由独立 reviewer 检查 02–04、实现和 tests；阻塞问题修复后才能进入训练。
4. RUN：仅由当前受监督会话中的 runner 执行；不得使用独立后台 AI、headless Codex、tmux、nohup、systemd 或旧 autonomous driver。
5. ANALYZE：完成每个阶段 Validation 选点、配对统计和触发决策。
6. FREEZE：锁定候选、配置、seed、source hash 和 selected model identity。
7. TEST：只在授权和冻结状态均满足时执行。
8. REPORT：生成报告、指标表、成本、失败状态和上下文更新。

## 2. 队列

job key 至少包含 study/dataset/seed/arm/config hash/source hash/data hash/readout。每个 job 有 attempt_id；attempt 仅表示物理恢复，不改变科学轨迹身份。队列顺序：
- S0 audit；
- A0/B0 implementation checks；
- A1 与 B1 可在独立资源下并行，但不能共享可写模型状态；
- C0 offline audit；
- A2/B2/C1 仅按 03 的 trigger 生成；
- S4 仅在两个独立机制通过后生成；
- Freeze/Test/Report。

任何 NOT_TRIGGERED 项必须在 manifest 中保留原因，不能从报告中消失。

## 3. GPU / RAM

- 每次启动任务前检查当前 GPU memory、power 和 host RAM，遵循仓库当前受监督 runner 约束。GPU 资源准入需使用实时观测值；启动后重新采样。若发生 CUDA OOM，按仓库已记录的 physical capacity correction 执行，不使用静态 worker 数量猜测。
- 不引入新的后台 guardian/daemon。优先使用仓库现有 RAM guardian；如当前 guardian 不可用，先报告阻塞并暂停，不能用磁盘保存模型状态替代。
- 每个训练 epoch 完成后提交 RAM state，并立即 get/readback/hash 验证。
- 训练中断时只从最后一个验证成功的完整 epoch transaction 恢复；不得从半个 batch 状态静默续跑。
- 恢复后重新核对 job identity、owner generation、data/config/group hash、RNG、loader cursor、optimizer、EMA、PAIR history、memory bank 和 selected metadata。

## 4. 失败策略

- Scientific failure：按决策规则关闭该方向或阶段，生成报告。
- Technical failure：PAUSE，保留失败 traceback、最后 verified state、已消耗成本和恢复原因。
- Source mismatch：只暂停受影响 arm；不把不同 Gaussian identity、PAIR sampler 或数据 split 混成一个结果。
- OOM：记录物理显存、进程驻留、当前 power 和 batch，按既有容量修正规则恢复；不修改 batch size 来掩盖问题，除非重新冻结完整协议并重跑匹配臂。
- Missing checkpoint：C0 标记不可用；不得假设 RAM checkpoint 仍存在或从未授权来源恢复。

## 5. 报告产物

建议：
- 数值和 manifest：`artifacts/three_directions_20261128/`
- 报告：`research_reports/three_directions_20261128/`
- 主文档：本 `research_plans/26-11-28_three_directions/`

至少包含：
- SOURCE_AUDIT.json
- DATA_AUDIT.json
- RUN_MANIFEST.json
- IMPLEMENTATION_REVIEW.md
- A/B/C 阶段 endpoints.csv
- paired_summary.csv
- per_seed_selected.csv
- six_cell_selected.csv
- costs.csv
- decisions.json
- FINAL_REPORT.md
- CONTEXT_UPDATE.md

不得将模型 checkpoint、optimizer state、EMA weights、gradient history 或 memory bank 的模型张量写入磁盘；只保存允许的数值/元数据和受控 RAM state receipt。

## 6. Test 授权

Test 不用于超参数、候选或阶段选择。所有 Validation 决策完成后冻结 selected models、readout、source hashes 和 config。Test 入口必须验证：
- explicit authorization；
- freeze hash；
- selected model hash；
- correct split identity；
- no post-Test selection or tuning。

若 Test 权限缺失，完成 Validation 报告并将 TEST_PENDING_AUTHORIZATION 明确写入终态。

