# 09｜Codex Handoff Contract

## 1. 你的任务

你是本轮唯一工程执行者。请严格按 00–08 执行，目标是完成 D2 的完整有限实验链：
- 确认共享表示更新是否产生域任务异质响应；
- 确认专有路径在共享更新后是否能恢复/利用该变化；
- 确认该响应能否转化为 full training 的独立 AUC 增量；
- 若通过，完成冻结后的 Test 评价和完整交付；若未通过，完成有证据的 NO-GO 报告。

不要自行提出第二个核心方向，不要重新进行大规模文献搜索，不要改写 D2 成为 EMA 或通用梯度优化研究。

## 2. 执行优先级

优先级从高到低：
1. 研究方案的科学正确性与原协议保真；
2. 真实 M3oE 参数/梯度支持审计；
3. 可恢复状态与 RNG/loader 纯度；
4. 五 seed 完整性；
5. AUC 与六格保护；
6. 成本与可复核性；
7. 代码简洁性。

如果实现便利性与科学定义冲突，以 00–07 为准。

## 3. 实施要求

- 先读完整方案与源文件，再写代码。
- 先建立 SOURCE_MANIFEST 和 DATA_MANIFEST，再实施。
- 新增独立 package，不修改历史源码。
- coder 完成后必须有独立 reviewer；reviewer 不得是同一 agent。
- Reviewer PASS 后才可正式训练。
- 先跑 CPU correctness tests，再跑 GPU/RAM smoke，再 dispatch S1。
- 所有参数状态 RAM_ONLY；磁盘只保存标量、配置、hash、row IDs、日志和报告。
- 每个 run 记录 source/data/config hashes。
- 训练种子只允许 2022–2026；不得用 1–4 seed 提前淘汰整个方向。
- 所有五 seed 必须完成后再做科学决策。
- 不因不利结果改 seed、改 epoch、改选点或选择性报告 cell。

## 4. 不确定实现点的处理

若源模型实际结构与本方案参数组定义不同：
1. 不允许直接按名称猜测；
2. 生成真实 forward/gradient support map；
3. 提出最小修订说明，列出对 H1/H2 estimand 的影响；
4. 暂停正式训练，交独立 reviewer 审核；
5. 只有当 estimand 保持一致且修订通过后继续。

若某 arm 无法做到 optimizer state 完全一致，必须记录技术阻塞，不得用近似 SGD 或重置 Adam 状态代替。

若 RAM-only 状态不能恢复，不得以仅有 epoch 数值作为可恢复模型状态。

## 5. 禁止的“优化”

- 不得为了显存而缩小 batch 或启用 AMP。
- 不得为了更好 AUC 而改 EMA rho、Adam beta、weight decay、训练 epoch 或 validation selection。
- 不得在 Stage 1 结果不理想时追加新 scope、新评分器或新模型头。
- 不得以更多计算量解释方法收益；必须报告每参数更新次数与总 forward/backward。
- 不得把 fixed1 融合系数改回可学习融合。
- 不得将 P_CELL、P_AXIS、P_FULL 的结果混为一个参数组。
- 不得在 Test 后调参或继续训练。

## 6. 最终交付与关闭

最终将：
- 所有源码和测试放入本轮新包；
- 所有标量工件放入本轮 artifacts；
- S1/S2/S3 报告和审查放入本轮 reports；
- 更新本轮研究计划中的执行状态记录（只允许在本目录增加 append-only execution log，不改写科学设计）；
- 在 GitHub 推送本轮代码、报告、数表和必要的上下文索引更新；
- 给出 commit SHA、文件清单、S1/S2/S3 最终状态、Test 是否访问、所有 seed 完成情况。

本目录任务到 COMPLETE_GO、COMPLETE_NO_GO 或 TECHNICAL_BLOCKED 后终止。不得启动无次数限制的自主研究循环。
