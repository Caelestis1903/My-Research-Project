# 04｜Experiment Matrix

## 1. 固定协议

| 项目 | 固定值 |
|---|---|
| 数据集 | MovieLens、KuaiRand |
| 数据划分 | 原始自然 80/10/10；train-only preprocessing；保持历史 row order / split identity |
| 域/任务 | 3 domains × 2 tasks |
| 模型 | M3oE fixed1 canonical implementation |
| shared experts | 4 |
| embedding | 16 |
| FCN | [512,256,256,64]，以当前 canonical source 核验为准 |
| epochs | 50 |
| batch | 40960，drop_last=False，保留自然尾批 |
| precision | FP32；AMP/TF32 关闭 |
| optimizer | Adam lr=.01, betas=(.9,.999), eps=1e-8；coupled WD 按当前数据集 canonical 协议 ML=1e-4、KR=1e-5 |
| EMA | 固定历史 rho：ML=.9828205985452511，KR=.995；不搜索 |
| seeds | 2022, 2023, 2024, 2025, 2026 |
| 主目标 | 自然 BCE；每 batch 按实际行数和两任务平均 |
| 主评价 | 六格 Macro AUC |
| 选点 | Validation 六格 Macro AUC 最早严格最大 |
| 次级指标 | 六格 LogLoss、Macro LL、Traffic、Pooled |

若 Stage 0 发现当前 main canonical source 与上述设置有任何差异，必须暂停并报告差异；不得静默修改固定协议。源配置是最终权威。

## 2. S1 基线轨迹

| arm | 数据集 | seeds | epochs | 用途 |
|---|---|---|---:|---|
| J_ORIG | ML/KR | 2022–2026 | 50 | 原始联合 Adam + 固定 EMA；提供 E10/E25/E40 fork anchors |

总计 2 × 5 = 10 条完整轨迹，500 epochs。仅当精确身份可核验的历史状态同时含 model、Adam、EMA、RNG、loader 和 epoch state 时才可复用；只复用数值端点不等于复用状态。

## 3. S1 诊断臂

每个 dataset × seed × anchor（E10/E25/E40）运行下表。每个 fork 都从同一 anchor clone 开始；每个 sequential arm 运行 AB 和 BA 两种 batch 映射。

| ID | 更新 | 目的 |
|---|---|---|
| J2 | A、B 各一次完整联合 Adam | 原始两步参照 |
| JACC | A/B 同 pre-state 梯度累积，一次联合 Adam | 相同数据/计算、一次参数更新控制 |
| U_ONLY | A/B 累积，仅 U 更新 | 共享更新单独响应 |
| P_CELL_ONLY | A/B 累积，仅 Gate/Tower 更新 | cell-specific 路径单独作用 |
| P_AXIS_ONLY | A/B 累积，仅 domain/task experts 更新 | 域/任务轴专家作用 |
| P_FULL_ONLY | A/B 累积，仅 P_CELL+P_AXIS 更新 | 全专有路径单独作用 |
| U→P_CELL | U(A) 后 P_CELL(B)，并执行反向 batch 映射 | 共享后 cell head 适配 |
| P_CELL→U | P_CELL(A) 后 U(B)，并执行反向 batch 映射 | 更新顺序控制 |
| U→P_AXIS | U(A) 后 P_AXIS(B)，并执行反向 batch 映射 | 共享后轴专家适配 |
| P_AXIS→U | P_AXIS(A) 后 U(B)，并执行反向 batch 映射 | 更新顺序控制 |
| U→P_FULL | U(A) 后 P_FULL(B)，并执行反向 batch 映射 | 完整专有路径适配 |
| P_FULL→U | P_FULL(A) 后 U(B)，并执行反向 batch 映射 | 更新顺序控制 |

S1 诊断不增加完整训练轨迹，不新增随机 seed。统计单位仍为五个训练 seed；三个 anchor 是同一 seed 内的重复测量，不充当独立样本。

## 4. S2 完整训练臂

仅当 S1 选出唯一 scope (P^*) 后执行：

| ID | 方法 | 是否新增完整轨迹 | 用途 |
|---|---|---:|---|
| J_ORIG | 原始每 batch 联合 Adam | 若 S1 baseline 可精确复用则否 | 强内部基线 |
| J_ACC | 两 microbatch 梯度累积，一次联合 Adam | 是 | 更新次数/批预算匹配 |
| P→U | 先 P* 再 U 的 block schedule | 是 | 顺序反向控制 |
| U→P | 先 U 再 P* 的 block schedule | 是 | 唯一候选算法 |

每个新 arm：2 datasets × 5 seeds × 50 epochs = 500 epochs。最多 3 个新 arm = 1500 epochs；若 baseline 状态不可复用，再加 500 epochs。不得因候选结果不理想再扩展 scope 搜索。

## 5. S3 Test

仅在 S2 validation gate 通过后冻结：
- J_ORIG
- J_ACC
- P→U
- U→P
- 正式发表模型 M3oE / HyperGate / AutoIFS 中身份和协议可比的对象

所有 test scoring intents 预先登记；不允许 scoring 后增删 arm。若某正式基线无法在相同人口/选点协议下获得可核验结果，标记 NOT_COMPARABLE/UNAVAILABLE，不编造数值，也不以内部模型代替正式发表基线。
