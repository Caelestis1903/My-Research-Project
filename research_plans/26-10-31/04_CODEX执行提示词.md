# Codex 首轮执行提示词｜GA31-v2

请在 My-Research-Project 中完成本轮实现与实验。先读取根目录 AGENT.md，再依次读取本目录 00、01、02、03 和 stage0_protocol.json。**03 是执行合同，JSON 是冻结参数；02 的候选算法不授权自动训练。**

目标是分离自然 Sample 协议下增强、统一损失缩放及其交互；完成 P0–P3 的两个数据集、四个配对种子，共 32 次正式训练与规定的离线探针。不要重新开展方向探索，也不要直接运行历史 legacy_group_sum trainer。

请连续完成：

1. 审计当前 commit/工作树、冻结来源与实际路径，核对 fixed1、单 Adam、自然 split 和数据哈希。
2. 在 experiments/gradient_aug_20261031/ 实现独立训练入口、NR-Gauss-v1、固定原样本预算的混合损失及四臂。源文件以 source_commit 固定，不覆盖历史。新输出使用 GA31-v2/attempt。
3. 实现必要检查：Sample oracle、κ=0/λ=0 退化、路由/dense 零扰动、配对 RNG、None 梯度、只读探针、checkpoint 恢复；仅 seed2022 工程短 run，不看 test、不调强度。
4. 检查通过且数据/GPU 可用后直接完成 32 次 50-epoch 正式训练，不因单种子正负中途选臂，不自动改变 batch/预算。
5. 根据 P0/P1 的 selected validation 均值冻结 c*，冻结所有 checkpoint，再统一评估 selected/E10/E50 test；只做规定的离线反事实探针。
6. 生成完整配对差分、逐域逐任务与成本报告，按 03 门槛输出 DECISIONS.json 和 REPORT.md，列出全部完成/失败/阻塞项。
7. 按项目已有 GitHub 提交习惯提交本轮代码、协议、汇总与可复核的轻量工件；不上传原始数据、大 checkpoint 或覆盖历史结果。报告执行 commit 和工件路径。

首轮 gradient_mix/gate/optimizer_state_split 均关闭，若请求开启须明确报本合同不支持。完成四臂与探针报告即停止；SAGU/SAOS/SGVS 的下一轮需另行冻结公式、对照和预算。本任务不要求再次索取研究设计确认。

如遇数据身份不符、真实 OOM、源语义冲突或非有限值，先定位可修复实现错误；确实阻塞就交付已完成内容、失败 ledger 和具体原因，不虚构结果，不偷偷改协议继续。

这是后续执行提示词；当前这次规划修订本身没有启动训练。

