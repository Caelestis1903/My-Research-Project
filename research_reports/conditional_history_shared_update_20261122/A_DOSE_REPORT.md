# 配对剂量与真实状态干预

按上一阶段冻结的 A21 更新规则，比较 λ=0.025 的历史方法 A40 与同剂量普通 Adam A41。两臂在两个数据集均跑完五个种子；相同自然样本对、模型和预算。固定 EMA 的 selected Validation 结果如下。

| 数据集 | A40 − A41 ΔAUC | 正向种子 | 配对 95% 区间 | A40 − A41 ΔLogLoss | A40 − clean 最差格 |
|---|---:|---:|---|---:|---|
| movielens | -0.000233 | 2/5 | [-0.000691, +0.000226] | +0.001084 | d1t0 -0.000602 |
| kuairand | -0.004366 | 1/5 | [-0.009234, +0.000502] | +0.001115 | d0t0 -0.012322 |

两数据集的低剂量历史增量均为负，LogLoss 均上升。KuaiRand 的最大格损害比原 λ=0.05 更大，降低 PAIR 权重未修复迁移代价。目标数据集 MovieLens 冻结为 A21、λ=0.05、S_DENSE。KuaiRand 的局部最优选择为普通 Adam，跨数据集比较仍使用同一 A21 配置。

在 λ=0.05 和 0.025 各取 2 数据集 × 5 种子 × 4 个真实更新锚点，共 80 个锚点。STAR_SKIP 与 EXPERT 的原尺度和等尺度干预生成 320 条门控记录。MovieLens 在较弱剂量下，两组均没有满足原尺度和等尺度共同确认条件的受损格；KuaiRand 在剂量选择时已偏好匹配普通 Adam。按 09 的作用域门，A50 与 A51 未触发。详见[剂量表](../../artifacts/conditional_history_shared_update_20261122/tables/A_DOSE/)和[实际干预](../../artifacts/conditional_history_shared_update_20261122/tables/A_TARGET/localization.json)。
