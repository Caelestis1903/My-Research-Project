# R0 Phase 2B 主张修正清单（Contradictions and Corrections）

**Stage R0 交付物**。列出 Phase 2B 八份最终报告中被证据核对推翻、需要修正或降级的具体主张。每条给出原主张、修正依据（含文件与位置）、修正后口径。分三类：A. 事实性错误（须修正文本）；B. 判决/定位修正（须改判定或表述）；C. 内部一致性残留（上游文件未跟随下游修正）。

---

## A. 事实性错误

### C-1：AO-05 的参数复杂度主张与 M3OE 原文结构矛盾【最严重】
- **原主张**：AO-05 §7(a)："参数量从 O(D·T·P) 降为 O(P+(D+T)·M)"（Phase2B_Algorithmic_Opportunity_Candidates.md；Ranking §二 AO-05 条以此作为 SUPPORTED 的成本叙事基础）。
- **修正依据**：M3OE 原文（`论文/M3OE.pdf_os_db45l62lb0pc7398j2jg/M3OE-normal.md`）§2.4.1–2.4.3：共享专家 N 个 + 域专家 D 个 + 任务专家 T 个，专家总数为 **N+D+T（加法）**；D×T 乘法项只出现在门（每 (d,t) 格一个 N 维 dense softmax linear，Eq.4）与预测塔（每格一个两层 MLP，Eq.8）。LIM-12（Phase2B_Method_Limitation_Ledger.md）与 Phase1 卡片原本登记正确，错误在 AO 候选写作环节引入。
- **修正后口径**：对三族专家做掩码只是常数因子压缩，"乘法变加法"叙事不成立；掩码分解若要有实质参数收益，作用对象应为 D×T 预测塔（但与 P005 邻近，须重新划界）。AO-05 的 SUPPORTED 判定失去成本叙事基础。

### C-2：AO-02 的负载均衡类比在 M3OE 上无作用对象
- **原主张**：AO-02 §11："两级门控天然产生域×任务×专家三维负载统计""偏置控制只作用于 top-k 选择、不碰梯度"（Phase2B_Algorithmic_Opportunity_Candidates.md）。
- **修正依据**：M3OE 原文核实——唯一门控是共享专家模块内的 N 维 dense softmax（Eq.4，第 217–220 行）；域/任务融合是标量 β+均匀回收（Eq.5/6，第 230/240 行）；α/β 是无输入的一维可训练标量（§2.6.1，第 281–289 行）；**全文无 top-k、无路由分配、无负载概念**。候选引用的证据 P040（M2R 簇路由）/P071（LLM 稀疏 MoE）/P093（LLM top-k）全部来自稀疏路由族。
- **修正后口径**：AO-02 当前形态在 M3OE 上不可执行；其证据（均衡损失有害）只支持"稀疏 MoE 文献"的问题存在性，不支持 M3OE 上的问题存在性。

### C-3："两级门控 = 域门 × 任务门"的共享架构误读（影响 AO-02/AO-09/AO-10 三处）
- **原主张**：AO-02 §11、AO-09 MVQ（"域门×任务门 k-sweep 损失矩阵"）、AO-10（"门控条件输入是标准接口、三路可插拔"）共同假设 M3OE 存在域门与任务门两级可寻址门控。
- **修正依据**：M3OE 的两级融合实际是"共享专家模块内 softmax 门 × 模块间 α 标量融合"（Eq.4 + §2.6 融合权重标量）；域/任务维度无门，α/β 无输入条件接口。另 §3.7（第 444 行）：N=1（MovieLens）/N=4（KuaiRand），MovieLens 上共享门退化为标量 1，top-k 无对象。
- **修正后口径**：AO-09 的 MVQ 测量对象修正为"共享专家门 k-sweep（仅 KuaiRand，k∈{1,2,3,4}）"；AO-10 的语义条件注入属于结构改动而非接口插拔；AO-02 见 C-2。

### C-4：AO-01 的 P112 退化证据未声明 objective 偶然性
- **原主张**：AO-01 以 P112 KuaiRand d2t1 66.03 vs 单域 MLP 69.28 为立项直证 [A]，未注明该结果的 objective 口径（Phase2B_Algorithmic_Opportunity_Candidates.md；Ranking §二 AO-01 PC=5）。
- **修正依据**：P112（M3OE）原文 Eq.12（第 302 行）：L = Σ_d Σ_t BCE(ŷ_{d,t}, y_{d,t})——逐格无权重之和，即 Equal 型目标。本团队实验（experiments/Experimental_Stable_Findings.md SF001/SF002；E004/E009/E010）：Sample objective 下 72/72（S1）与 24/24（S2）单元无 practical NT；Equal 下 KuaiRand tab1 click 才有 −0.0450±0.0006（4/4）的可重复退化。现象 objective-contingent。
- **修正后口径**：AO-01 的立项依据应改写为"Equal 型目标或极端比例下的大域保护"；在 Sample 自然比例主线上，问题存在性证据为阴性（0/96）。不构成 CONTRADICTED（极端比例 OQ001 未测），但 SUPPORTED 判定不成立。
- **附带事实失真**（skeptic 发现）：AO-01 §1 把 RelaImpr 表述为训练目标（实际仅评测口径）；§8 把 P112 §3.4 的现象自述引用为机制自述。

## B. 判决/定位修正

### C-5：AO-01 与 AO-05 的 SUPPORTED 判定均不成立，"两 SUPPORTED 并行"建议须撤销
- **原主张**：Phase2B_Global_Conclusion.md §二"AO-01 与 AO-05 并行推进"；Ranking 排名 1/2。
- **修正依据**：C-1（AO-05 成本叙事为假）+ C-4（AO-01 立项证据 objective 错位）。"共同前置实验（公平协议复现 d2 失效格）"的紧迫性同步下降：本团队 S1/S2 已在 M3OE+KuaiRand 上给出等效信息（Equal 退化可复现、Sample 无退化），剩余问题只是极端比例。
- **修正后口径**：R0 无 SUPPORTED 候选；先导按测量类闸门组织（见 R0_Revised_Opportunity_Ranking.md）。

### C-6：AO-03 的"未被触及的新空间"隐含叙事须下调
- **原主张**：AO-03 以"何时需要梯度干预无判据"为缺口（PO=2 已是全候选最低之一）。
- **修正依据**：S7-A（EXP029，E023/E024）已完成 Domain×Task 梯度几何测量；S7-B（EXP030，E025）same-θ 干预改变梯度指标但无 validation 收益（anchor39 step5 AUC 均值 −0.001226，3/4 为负，经 C012 修正后口径）；S8（EXP031，E026/E027，STRONG）full-state continuation 无 selected/best 收益；Dead End D010（梯度统计→性能因果）已 CLOSED；最近邻统计量 U_k 跨数据集不可靠（SF008：KuaiRand 负 U_k 无性能后果）。
- **修正后口径**：AO-03 未达 CONTRADICTED（其精确 estimand"逐参数组冲突量×逐格退化秩相关"未测），但收益假设的证据基调为负；其 MVQ 须先证明所用度量携带 U_k 之外的信息。

### C-7：AO-06 的诊断组件已被实验纪律吸收，须如实降级定位
- **原主张**：AO-06 以"逐格 AUC/LogLoss 背离诊断面+约束式控制"为贡献。
- **修正依据**：SF010（experiments/Experimental_Stable_Findings.md）已把"同时报告 target/non-target × AUC/LogLoss × checkpoint"制度化——诊断组件成为报告纪律而非算法贡献；剩余贡献全押损失形状控制，外部 JRC/BBP 成熟；候选自定位防御性（PO=2，不承诺 AUC 增益）。本团队数据中背离反复出现（E014/E019/E025）但全部为干预臂小幅度背离，正常训练自发背离未测。
- **修正后口径**：AO-06 主要属校准侧工具，不列为 AUC 算法机会。

### C-8：AO-07 与当前数据/任务不兼容，须明示而非仅标薄证据
- **原主张**：AO-07 以"证据底座最薄但问题表述有新意"维持 PROMISING。
- **修正依据**：KuaiRand/MovieLens 每样本只属一域、无跨场景重复打标结构（M3OE §2.1 数据定义）；标定需 Ali-CCP 式含场景切换数据；团队管线不做标签复用/伪标签迁移。
- **修正后口径**：NOT-CURRENTLY-FEASIBLE（数据/任务结构不匹配），非"证据薄"。

## C. 内部一致性残留（上游未跟随下游修正）

### C-9：Phase2B_Incremental_Innovation_Map.md 链 5 小结仍含被 SKEPTIC 证伪的"CDC/SDSP 无随训练校正机制"主张
- 修正只写入 AO-04 修订版四点划界，未回写阶段二交付物（lit reviewer 内部一致性检查第 1 项）。**修正**：IIM 链 5 小结应加注该主张已被原文证伪。

### C-10：Phase2B_Method_Capability_Map.md F7 Q9 把 P013"无遗忘指标"归入 [A] 条目
- LA 原判 [B]，Ledger 与 AO-08 已对齐 B，CM 残留字面拔高（lit reviewer 内部一致性检查第 2 项）。**修正**：CM F7 Q9 改标 [B]。

### C-11：P040 均衡损失的表号不一致
- Ledger 作 Table 2、SKEPTIC 回查作 Table 6（数字一致，AO 修订版已不指表号，Ledger 未同步）。**修正**：Ledger 统一为 Table 6 或去表号。

### C-12：Ranking 的证据等级全部转引、未独立回查原文
- Phase2B_Algorithmic_Direction_Ranking.md §五.6 自我声明"未独立回查任何原文；若前序环节存在残余等级拔高，PC 维分数连带高估"。R0 的抽验证实该风险实际发生（C-1/C-2/C-4 均未被 Phase2B 内部流程捕获）。**修正**：凡 R0 已核实的条目以本文件为准；未核实条目的 [A]/[B] 等级视为转述等级。

## D. 口径差异记录（非 Phase2B 错误，但须留痕）

### C-13：S6-R2 状态口径矛盾
- stage_r0.md 任务书称"尚未完成的 S6-R2 不作为任何方向的支持或反证"；档案（Experimental_Experiment_Index.md EXP032、Evidence Ledger E028–E032、Stable Findings SF011/SF012、Contradiction Register C021–C023，整理截止 2026-10-09）一致登记为**已完成的 evaluation-only 实验**（复用 16 个既有 F/DH checkpoint、无新增训练、integrity 检查通过、源报告 KuaiRand gate 判 scoped GO）。
- **R0 执行口径**：遵循任务书，S6-R2 不用于任何 AO 的支持或反证。隔离影响："Sample 下 selected test-side optimizer-policy 收益""validation/test 反转""参数分区可组合性"三类判断在 R0 中无任何可用证据；AO-03 的 WEAKENED 判决不依赖 S6-R2（E023–E027+D010 足够）。

---

**修正责任说明**：A 类与 C 类为文本级修正，建议在 Phase2B 源文件加勘误注记（本阶段未改动任何 Phase2B 文件，仅在此登记）；B 类为判决修正，已由 R0_AO_Hypothesis_Matrix.md 与 R0_Revised_Opportunity_Ranking.md 吸收。
