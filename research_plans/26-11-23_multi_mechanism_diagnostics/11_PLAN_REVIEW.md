# 11 — 独立方案审查

## 1. 审查结论

**PASS WITH CONDITIONS**：方案具备可交付 Codex 的完整阶段结构、候选机制区分、统一状态快照、配对统计、资源调度和决策规则。S0 的 M1 二阶可行性与 M2 逐样本梯度计算成本是关键技术门；未通过时必须按机制分别标记 TECHNICAL_BLOCKED，不得改写成低成本代理而保留原机制名称。

## 2. 关键审查问题与实际修正

| 审查问题 | 初始风险 | 修正 |
|---|---|---|
| 第一阶段路径不一致 | 第二阶段 prompt 指定 26-11-23_multi_mechanism_discovery，但当前 main 实际存在的是 2026-11-24_open_mechanism_discovery | 在 README、01、10 明确登记差异；以实际存在的第一阶段报告为输入，不伪称读取缺失文件 |
| D2 是否已否定 M1 | 固定 U/P scope 和更新顺序 NO-GO 可能被过度外推 | M1 改为 P consistency objective（二阶塑形），与 D2 static sequence/scope 对照明确区分 |
| ConsMTL 移植是否足够创新 | 直接移植通用梯度一致性公式可能只是方法搬运 | 将其定位为机制候选；要求 cell-specific response、排序映射和独立 V_B 证据，不预先宣称创新 |
| 二阶计算成本 | create_graph 可能超显存，直接一阶替代会改变机制 | S0 必须二阶 GPU smoke；失败即 M1 TECHNICAL_BLOCKED，不能静默降阶 |
| M2 是否退化成 loss/grad norm sampling | 现有研究已覆盖通用采样/权重 | M2 score 定义为样本任务更新对 Train-only calibration pairwise margin 的局部影响预测，并设置 task-agnostic/shuffled controls |
| M2 是否使用 Validation 选择权重 | 可能产生验证泄漏 | score 只从 Train calibration 计算；Validation 只做独立读数 |
| M3 是否被全量 O(N²) 放大 | 交互归因成本高 | 只在 M2 产生稳定候选组后触发一个 group pair；固定 batch size |
| M3 的 batch exposure 不公平 | A+B mixture 可能使用双倍样本 | A-only、B-only、A/B mixture 均固定同一 batch size；A/B 各半 |
| M1 多一步更新是否混淆 | P-consistency step 会增加优化器更新次数 | P-BCE matched private step、two-step joint baseline、coordinate-shuffled consistency control；所有主 arms 两次 step |
| 参数组是否互斥 | M3oE 的 domain/task expert 参数可能跨多个 cell 支持 | 要求运行时 support registry；不假设每 cell 有互斥私有头 |
| 锚点是否仍可用 | D2 state receipts 属 RAM-only | live readback；失败则重建完整轨迹，不能把 endpoint 当状态 |
| seed/anchor 虚增样本量 | 30 anchors/数百 forks 容易误统计 | seed 为独立单位；anchor/fork 为 seed 内重复，先 seed 内归约 |
| Test 泄漏 | 诊断期可能提前用 Test 选候选 | S0–S3 不加载 Test；S4 freeze 后才可一次性评价 |
| 小域困难度误判 | 可能按域大小分组 | 明确禁止；不使用 domain size 作为困难度代理 |
| M4/M5 语义臆造 | 数据字段可能没有合法反事实 | 仅审计；缺语义依据就停止，不生成伪标签/伪反事实 |
| Codex 自行发明算法 | S4 需要从机制结果形成新动作 | S4 必须另交 DESIGN_SPECIFICATION 并独立 PASS；未触发则 NOT_TRIGGERED |

## 3. 仍需 S0 确认的真实不确定性

1. D2 RAM state 当前能否完整 readback；若不能，需重建十条轨迹。
2. M1 全 U 二阶 consistency objective 是否能在实际 GPU 上完成。
3. M2 逐样本虚拟 Adam 与 calibration margin gradient 的计算成本。
4. 当前可用数据字段是否存在可被业务语义支持的合法增强变换。

这些是实际影响机制可执行性与 estimand 的问题，必须在 S0 解决；不要求为了理论完整而增加其他无决策价值的实验。

## 4. 方案最终裁定

本方案允许 Codex 从 S0 开始，按 gate 顺序执行 S1–S3。S4 需要新的算法规格和独立审查。没有通过候选则终止，不追加无上限机制或组合。方案没有把研究报告中的所有候选机械地做笛卡尔积。


## 5. 第二轮审查补充：S4 可执行性

初版 S4 仅写“通过后另交算法规格”，不足以让 Codex 在研究门通过后自动执行。本次已将唯一允许的条件晋级算法冻结为 CGPA：
- P consistency-only step；
- 同 batch canonical joint step；
- 每 batch 仅一次 EMA tick；
- P-BCE-prestep 为两步预算匹配控制；
- consistency lr multiplier 轻量搜索；
- M2/M3 不自动混入。

该补充避免 Codex 自行发明算法，同时保留机制未通过时 S4 不触发的停止规则。CGPA 是本项目待检验候选，不预先认定其有效或创新成立。
