# 当前执行状态

任务进行中；尚未完成正式训练与最终结论。

- 当前阶段：DESIGN，独立 designer 正在审计全部历史执行分支并制定完整补充设计。
- 原始 workspace Git 未有提交；原文件保持。正在 fetch 全部上游分支，随后建立独立执行 worktree/branch。
- Python：/home/xzf/miniconda3/envs/phase0-3-env/bin/python。
- 原始数据：/home/xzf/data/ml-1m/ml-1m.csv。
- 新大型工件根目录：/home_new/xzf_natural_movielens_reassessment。
- 当前机器不存在历史 /home_new/xzf_cm19_artifacts；历史源码、配置、逐run指标、冻结记录及部分预测保留在 sibling worktrees。不能将历史索引视作checkpoint仍可用。
- 默认优化种子：2022、2023、2024、2025。
- 正式执行须经过 DESIGN→IMPLEMENT→REVIEW(PASS或修复后的PASS WITH FIXES)→RUN→ANALYZE。

## 执行工作树与分批放行

实际代码与工件索引位于 `/home/xzf/Code/phase0-3-natural-git`。独立分支 `codex/natural-movielens-reassessment-2026-10-20`，已推送核心设计提交7f05f3a9。V2的16配置实现已完成，review确认原训练/测试顺序；共享数据loader/source hash绑定为blocking修复，修完复审前不运行。核心120配置实现及D-F审计设计仍在进行。
