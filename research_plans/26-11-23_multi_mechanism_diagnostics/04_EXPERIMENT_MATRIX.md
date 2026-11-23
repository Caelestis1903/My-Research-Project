# 04 — 实验矩阵

## 1. 固定维度

| 维度 | 取值 |
|---|---|
| Dataset | MovieLens、KuaiRand |
| Model | M3oE fixed1 |
| Seeds | 2022–2026 |
| Baseline epochs | 50 |
| Batch | 40960，保留 natural tail |
| Anchors | E10、E25、E40 |
| Validation | V_A screening；V_B confirmation |
| Test | sealed，S4 freeze 后才允许评价 |

## 2. S0 feasibility audit

M1 二阶计算、M2 样本梯度读取、M4/M5 语义可行性分别审计。无训练。若二阶图 smoke 超出显存或资源约束，M1=TECHNICAL_BLOCKED，不影响 M2 审计。

## 3. S1 baseline

若 D2 state live recovery 通过，复用；否则重建：
2 datasets × 5 seeds = 10 trajectories，50 epochs each，anchors E10/E25/E40。

## 4. M1 local forks

每 dataset × seed × anchor：
- M1-0 Joint × 2 steps；
- M1-1 P-consistency × 1 + Joint × 1；
- M1-2 P-BCE × 1 + Joint × 1；
- M1-3 coordinate-shuffled consistency × 1 + Joint × 1；
- M1-4 consistency-only（readout-only diagnostic）。

共 5 arms × 30 anchor states = 150 fork jobs。M1-4 不计入公平预算主比较。

## 5. M2 local forks

每 dataset × seed × anchor：
- M2-0 natural uniform；
- M2-1 task-specific impact weighting；
- M2-2 task-agnostic collapsed score；
- M2-3 within-domain shuffled score。

共 4 arms × 30 = 120 fork jobs。样本 panel size、每格支持和计算上限在 S0 冻结。每 domain 每 seed/anchor 从同一 canonical batch B 抽取最多 128 行，按实际联合标签组合分层；每行分别计算两个任务的局部 score。不足则全取并记录，不补造。未入 panel 的 batch rows 权重固定为 1，panel weights 在 domain-task 内重归一化，保持该 task 的总 loss mass 不变。

## 6. M3 conditional forks

触发条件：M2 top/bottom group 至少 2 seeds × 2 anchors 同向，且支持检查通过。

每 dataset × seed × anchor 对一个冻结 group pair：
- A-only；
- B-only；
- A/B equal mixture；
- A/A mixture control；
- B/B mixture control。

最多 5 × 30 = 150 forks。若未触发，M3=NOT_TRIGGERED，不补做探索性 pair sweep。

## 7. S1 总量上限

- baseline trajectories：最多 10 条；
- anchors：30；
- M1：150 forks；
- M2：120 forks；
- M3：最多 150 forks；
- 总 fork 上限 420；
- M1/M2 共享 baseline/anchor，不重复训练 baseline；
- 所有 fork 只执行预注册局部更新与 Validation readout，不执行 Test。

## 8. S2 confirmation

最多 2 个候选（M1、M2 各一个）；M3 仅当有独立可解释交互时最多一个。使用 V_B；仍为 5 seeds × 2 datasets。候选在 V_B 解封前冻结。S2 不新增超参数搜索。

## 9. S3 short trajectory

每个通过 S2 的候选：2 datasets × 5 seeds × 5 epochs。对照与处理组各一条。最多 3 candidates。若多个候选通过，独立轨迹运行，不做全因子组合。

## 10. S4 formal algorithm

只有独立设计审查 PASS 才执行：2 datasets × 5 seeds × 50 epochs。S4 的算法动作必须由 S1–S3 证据定义，不得在本方案中留给 Codex 自行猜测。未触发则报告 NOT_TRIGGERED。


## 11. S3/S4 的冻结补充

S3 primary trajectory 从 E25 anchor 开始，连续 5 epochs；E10/E40 仅作为 S1 局部阶段读数，不再增加短程轨迹网格。每个通过的候选为 2 datasets × 5 seeds，处理与匹配控制独立恢复。

S4 只可能晋级 CGPA（M1）。训练矩阵：
- CGPA：P-consistency step → same-batch joint step → one EMA tick；
- P-BCE-prestep control：P-BCE step → same-batch joint step → one EMA tick；
- canonical joint-only：原训练步骤与 EMA tick，不伪称 optimizer-step 匹配。

每个新超参 consistency_lr_multiplier 搜索 0.1/0.3/1.0；inner steps 默认 1。所有选择以五 seed Validation 综合结果完成后 freeze。M2/M3 不自动组合进 CGPA。
