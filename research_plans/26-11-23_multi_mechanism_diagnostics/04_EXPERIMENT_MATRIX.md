# 04 — 全候选实验矩阵与数量

## 1. 固定维度

2 datasets × 5 seeds × 3 anchors = 30 anchor blocks。
Datasets: MovieLens、KuaiRand。
Seeds: 2022–2026。
Anchors: E10、E25、E40。
Validation: V_A screening + V_B confirmation。
Test: S0–S3 sealed。

## 2. S1 局部 fork slots

| Family | Registered arms per anchor | Fork slots |
|---|---:|---:|
| M1 | 3 consistency objectives + 3 coordinate-shuffled controls + P-BCE + joint baseline = 8 | 240 |
| M2 | 4 task/sample impact actions + natural + task-agnostic + shuffled controls = 7 | 210 |
| M3 | 2 group-pair types × 5 interaction arms = 10 | 300 |
| M4 | 4 feature-dependence actions × intervention/sham = 8 | 240 |
| M5 | 5 augmentation families × augmented/matched control = 10 | 300 |
| M6 | 3 output-sensitivity controls × candidate/matched control = 6 | 180 |
| M7 | 3 distributional attribution actions × robust/mean or shuffled control = 6 | 180 |
| M8 | 4 conflict controllers × candidate/matched control = 8 | 240 |
| Cross-family | 5 registered interactions × combination/matched control arms = 10 | 300 |
| **Total** |  | **2,190** |

每个 slot 都必须进入 RUN_MANIFEST。某个变体在某数据集不适用时，仍创建 slot 并以 NOT_APPLICABLE + reason 终结，不静默减少总数。

## 3. M1 arms

每 anchor：
- M1-A directional consistency；
- M1-B magnitude-aware consistency；
- M1-C pairwise conflict energy；
- 三个各自 coordinate-shuffled controls；
- P-BCE-only private step + joint；
- two-step joint baseline。
所有 arm 使用同一 A/B batch mapping、同一 anchor、相同两次 optimizer steps；consistency-only 另作 readout，不参与预算主比较。

## 4. M2 arms

每 anchor：
- task/cell-conditioned influence weighting；
- influence-ranked task-gradient accumulation；
- negative-impact suppression；
- cell-response weighting；
- natural uniform；
- task-agnostic collapsed score；
- within-domain shuffled score。
所有组 row exposure、batch size、loss mass、optimizer steps、Adam state 匹配。

## 5. M3 arms

每 anchor固定两个 group pair：
- score-defined high/low influence pair；
- domain + joint-label support + group size 匹配的 random pair。

每个 pair：
A-only、B-only、A/B mixture、A/A mixture、B/B mixture。固定 batch size、总样本数、optimizer step、Adam 起点。M3-A stagewise attribution 与 M3-C stage×group interaction 使用 E10/E25/E40 全部读数，M3 不设条件触发。

## 6. M4/M5/M6/M7/M8 arms

M4 按真实字段组执行 conditional dependence map、conditional permutation、task-conditioned residualization/HSIC penalty；每个动作都有 sham/neutral control。

M5 对 field masking、embedding perturbation、within-stratum categorical substitution、dense perturbation、multi-view consistency 分别执行变换审计和训练 arm。具体字段不适用时保留 NOT_APPLICABLE 记录。

M6 执行 sensitivity scaling、worst-cell trust region、parameter-group response gating，并与 global clipping / random scaling / unmodified update 对照。

M7 执行跨 seed influence distribution、mean-variance robust score、leave-one-seed-out transfer；使用 M2 的五 seed score receipts，但独立写出 M7 decision。

M8 执行 PCGrad reference、conflict-aware weights、common-descent projection、conflict-aware cell gating。

## 7. Cross-family interactions

X1 M1×M2；X2 M2×M3；X3 M4×M5；X4 M6×M8；X5 M1×M8。每组包含 component arms、combined arm、matched-budget control。所有五组都执行。

## 8. S2：全候选 V_B 确认

2,130 个 S1 fork endpoints 全部计算 V_B readout。不得只选择 S1 positive variants。V_A 与 V_B row/pair IDs 完全互斥。S2 不额外训练，不用 V_B 调参。

## 9. S3：所有可训练变体的短程轨迹

所有可执行训练动作均进入 5 epochs continuous trajectory，不受 S1/S2 Top-K 选择限制。按变体、数据集、seed独立恢复：
- 2 datasets × 5 seeds × 5 epochs；
- E1/E3/E5 readouts；
- 对应匹配控制同样运行。
纯测量型机制完成跨阶段和跨 seed 重复测量，不伪造训练动作。

## 10. S4：全量合格变体正式训练

所有同时通过机制确认和短程轨迹 gate 的变体逐一执行：
2 datasets × 5 seeds × 50 epochs。
不设候选数量上限、不做 Top-K。每个候选独立选择 Validation 六格 Macro AUC checkpoint，冻结后 Test 一次性评价。

## 11. 成本与估算

S1 固定 2,130 fork slots + 最多 10 条 baseline trajectories。S3/S4 数量取决于实际可训练变体数量，但不得按研究优先级删减。S0 GPU smoke 后，按每个 family 的实测 forward/backward/optimizer cost、peak VRAM、host RAM 计算 GPU hours 与墙钟。成本估算必须同时列出全矩阵最坏情况和实测准入后的实际情况，不得编造训练速度。
