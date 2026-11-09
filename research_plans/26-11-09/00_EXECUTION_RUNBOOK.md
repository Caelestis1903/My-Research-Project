# 00｜执行总入口：MDMT 自适应 EMA

**规划日期：2026-11-09｜状态：PLAN READY；未执行训练**

## 1. 研究目标与不可变边界

本计划承接固定参数 EMA 收益机制研究。固定 EMA 的排序收益、早期滞后代价、晚期收益、参数路径/任务收益异质性均作为已有事实；本轮不重新验证 EMA 是否有效。目标是检验训练时可获得的更新收益信息能否预测后续历史吸收收益，并据此控制共享参数 S 的 EMA 时间尺度。

仅规划和实现文档，不在本轮启动 GPU 科学训练，不修改任何既有训练代码、实验报告或总上下文文件。执行时先做 A0 离线审计，再按阶段门决定是否新增训练。

## 2. 必须阅读的来源与身份

- main 上下文：`EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md`、`experiments/Experimental_Stable_Findings.md`、`experiments/Experimental_Evidence_Ledger.md`、`experiments/Experimental_Hypothesis_History.md`、`experiments/Experimental_Open_Questions.md`、`experiments/Experimental_Contradiction_Register.md`。
- 机制来源分支：`26-11-07`，报告目录 `research_reports/2026-11-06_ema_mechanism/`。来源分支与 main 分叉，禁止假定其代码、产物已合入 main。执行前必须重新记录分支 head、main head、merge-base、ahead/behind 和文件 blob SHA。
- 增强/历史控制来源：`26-11-07` 的 `research_reports/2026-11-05_aug_history/` 及 main 的 `research_plans/26-11-5/`。
- 机制规划来源：main 的 `research_plans/26-11-6/`，尤其 01–09 和 `experiment_manifest.json`。

已核对：`experiments/ema_mechanism_20261106/worker.py`、`analysis.py` 与 `research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md` 在 `26-11-07` 可读取，但在 `main` 返回404，说明该机制分支内容尚未合入main。Codex必须从 `26-11-07` 实际head建立隔离worktree，并在SOURCE_AUDIT记录其真实head SHA；不能从main直接启动，也不能把该分支的文件误认为main已有。

11/07机制报告固定 EMA 的 F−online selected Test Macro AUC：ML Clean +0.003848845、ML Gaussian D0 +0.003856802、KR Clean +0.004616399、KR Gaussian D0 +0.007766927，均为同轨迹五seed配对值。晚期 F 还优于按理论平均历史年龄匹配的旧 online；早期则有滞后损失。ML 的 AUC收益伴随 LogLoss代价。不能把所有收益简化为 checkpoint 延迟，也不能把低通性质直接当成性能机制。

11/07收益指标比较中，M2对局部训练排序风险变化的近似很准确，但迁移到独立验证 AUC 较弱；M3直接 AUC 跨面板稳定性有限；C_design在四种条件下的未来数值预测误差优于 M2/M3。复杂指标整体动作价值小，KuaiRand Gaussian、d0t1、共享参数 S 的 M3_direct相对 C_simple 仅约 +0.000048 AUC。11/05 A/B/C 未建立超越固定 F 的稳定独立 Test 增益。以上是本轮设计约束，不是要求重新跑这些实验。

## 3. 关键术语和阶段

- `F`：固定 EMA；`Fast`：`rho_f = rho_F^2`；`Slow`：`rho_s = sqrt(rho_F)`。
- `P_A/P_B`：互不重叠、固定身份的训练人口观测面板。可用于训练时控制信号；不得从 Loader 删除或筛除样本。
- `V_A/V_B`：独立 Validation 行侧，用于审计信号迁移和完整模型选点；不得参与控制器动作选择。它们不是独立 Test。
- `C_persist`：收益持续性信号；只有 A0/A1 支持时才启用，不预设成立。
- `A0`：已有数据离线收益持续性审计，零 GPU。
- `A1`：新seed前瞻性信号验证。
- `B`：A/B/C三种闭环控制器的同轨迹比较。
- `C`：跨数据集正式性能评价；仅在 B 有明确增量价值时执行。

## 4. 执行顺序

| 顺序 | 命令/任务 | 新正式训练 | 完成条件 |
|---|---|---:|---|
| 0 | `audit`：确认分支、文件、原始产物、代码身份 | 0 | 产生来源审计和哈希收据 |
| 1 | `analyze --stage A0`：重算相邻脉冲窗口持续性、跨面板一致性、未来收益预测 | 0 | 明确 C_persist：支持/不支持/数据缺失 |
| 2 | `analyze --stage A0` 决策 | 0 | 只允许按 06 的阶段门选择 A1、静态偏好替代或停止 |
| 3 | `train --stage A1`：KR Clean/G-D0，新seed 2027–2031 | 最多10条唯一在线轨迹 | 所有轨迹五seed完整；控制信号仅用训练面板 |
| 4 | `analyze --stage A1` | 0 | 信号跨seed、跨P_A/P_B可迁移 |
| 5 | `train --stage B`：A/B/C影子控制器，同在线轨迹 | 复用 A1轨迹；若A1无新轨迹则不运行 | 完整验证曲线、冻结配置、5seed |
| 6 | `evaluate --stage C`：ML Clean/G-D0及KR确认 | 仅按阶段门 | 完整六格指标与发表模型对比 |
| 7 | `report`、`release` | 0 | 报告、决策、成本和释放收据齐全 |

**重要：** 若 A0 未发现收益持续性，不能把 C_persist 改名后继续运行。若稳定的任务级静态偏好存在，允许转入 06 定义的低成本静态偏好分支；若无，则保留固定 EMA，停止自适应控制路线。

## 5. 最终交付物

运行目录与本规划目录分离。规划文件位于 `research_plans/26-11-09/`；运行结果放在 `artifacts/ema_adaptive_20261109/<attempt_id>/`，正式报告放在 `research_reports/2026-11-09_ema_adaptive/`。不得覆盖 11/05、11/06 或 11/07 的历史产物。

Codex 必须先输出 source audit、manifest解析结果、资源估算和实现审查；审查未通过不得启动训练。任何 Test 访问必须在算法、超参数、Validation选点规则及候选集合全部冻结后进行。
