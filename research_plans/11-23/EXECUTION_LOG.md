# D2 执行记录

## 2026-11-23 · S0 通过，S1 启动

本轮从 `main` 的规划提交 `f66bf17cba2980618e6f2edd1c7381b73c35cac5` 建立独立分支 `26-11-23_v2`。冻结设计保留全部 19 项诊断干预和批次映射，每个数据集五个种子、每个种子三个锚点，共 570 次诊断更新。 [设计规格](../../research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md) · [计数勘误](../../research_reports/mdmt_shared_private_coordination_20261123/DESIGN_ERRATUM.md)

两套数据的 Train/Validation 行序、特征与划分哈希均复现。CPU 和 CUDA 的 canonical 首批及自然尾批通过精确一致性检查。实现审查的四项问题已关闭，复审结论为 `PASS WITH FIXES`。首次烟测的 RAM owner 随工具会话退出，第二次在同一前台会话完成验收和活体读回。 [实现审查](../../research_reports/mdmt_shared_private_coordination_20261123/IMPLEMENTATION_REVIEW.md) · [S0 CUDA 与 RAM 验收](../../research_reports/mdmt_shared_private_coordination_20261123/S0_SMOKE_REVIEW.md)

S1 按冻结矩阵开始执行 MovieLens 和 KuaiRand 的 10 条 J_ORIG 轨迹。Test 保持封存。

## 2026-11-23 · `COMPLETE_NO_GO_S1`

两套数据的五种子均完成 50 个 epoch。10 条轨迹在首次尝试完成，30 个锚点和全部 570 个诊断 fork 的收据通过身份复核。V_A 上唯一达标的范围为 KuaiRand 的 `P_AXIS`。V_B 上两套数据的 J_ACC 比较均值为负，KuaiRand 仅 E25 同时超过反向顺序与 J_ACC。因此未进入 S2 或 S3，Test 评价为零。独立科研复核重新计算了冻结判据并确认终态。 [阶段完成收据](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json) · [冻结决策](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json) · [最终报告](../../research_reports/mdmt_shared_private_coordination_20261123/FINAL_REPORT.md) · [独立复核](../../research_reports/mdmt_shared_private_coordination_20261123/S1_REVIEW.md)
