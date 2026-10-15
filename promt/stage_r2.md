# Stage R2 — AO-04 最小先导实验正式执行

## 一、任务目标

Stage R0 和 R1 已完成，R1 v1.2 实验设计经过独立 Reviewer 审核并获得 PASS。

现在正式授权执行 **Stage R2**，包括代码实现、独立实现审查、baseline 重建、R2-A 训练、条件性 R2-B 训练、最终测试与结果分析。

本阶段只研究 AO-04：在 Sample-M3OE 上，学习非均匀他域专家融合比例是否能够获得实际 AUC 收益，以及这种收益是否能被简单替代解释覆盖。

**本任务是执行已审核的实验合同，不是重新设计研究方向或算法。**

## 二、必须遵循的文件

优先完整读取：

1. `R1_Experiment_Charter_v1.2.md`
2. `R2_Execution_Plan.md`
3. `R1_to_R2_Independent_Review.md`

然后根据需要读取：

4. `R1_Architecture_Audit.md`
5. `R1_Baseline_Protocol.md`
6. `R1_Charter_Change_Log.md`

上述 v1.2 Charter 是科学实验定义的主要依据；Execution Plan 规定具体运行流程；独立审核报告规定已发现的风险及实现审核要求。

如文件间存在实质冲突，停止相关操作并记录，不能自行更改研究目标、实验臂或判定条件。

无需重新开展 Phase 2B、R0 文献分析，也不允许根据历史结论预设 AO-04 有效。

## 三、Agent 分工

使用以下角色，可以并行完成相互独立的准备工作，但必须遵守审核依赖关系：

- **Main Controller**：控制阶段进度、检查执行门槛、整合结果和研究结论；不直接修改算法代码。
- **Designer**：依据 v1.2 文件核对完整数学定义和实现要求，不重新设计研究方向。
- **Coder**：实现 B0–B4 与隔离训练程序，完成基础检查。
- **Independent Reviewer**：独立检查实现与数学合同是否一致，重点审查梯度、优化器、数据与测试集隔离。
- **Runner**：在审核通过后执行固定实验队列，记录完整原始结果，不根据性能调整方案。
- **Experimental Analyst**：完成全部预定实验后，分析结果、竞争解释和下一阶段研究价值。

## 四、执行顺序

### Step 1：准备与 baseline 核验

严格按照 R2 Execution Plan 进行一次有界历史 checkpoint 搜索。

若无法恢复符合要求的 checkpoint，使用已锁定的当前 runtime 重新训练匹配 B0，不得将历史 CSV 指标直接作为新实验的配对 baseline。

检查：

- 实际 M3OE 架构与四个全局融合标量。
- Sample objective 的具体实现。
- KuaiRand 数据文件及划分、标签、特征、encoder 指纹。
- 参数初始化与随机种子。
- 主 Adam / 架构 Adam 的更新规则。
- 50 epochs 完整训练与 validation 选模协议。
- 模型状态、优化器和随机状态保存及恢复。

### Step 2：模型实现与独立审核

在独立目录实现 B0、B1、B2、B3、B4，保持原始 S0–S8 代码和报告不变。

严格采用 v1.2 的公式与超参数，不自行新增结构、调参搜索或正则化项。

特别注意：

- B1/B2/B4 的初始前向行为必须与 B0 等价。
- B4 使用原始未经过 sigmoid 的 raw_bD。
- 新增 q/delta 必须进入主 Adam，原四个架构参数保持原更新协议。
- B3 的固定比例表不能根据实验结果修改。
- 各方法使用配对的原始参数初始化和数据曝光顺序。
- 不得裸调用旧 S1 中会自动执行测试集评价的 fit/write_outputs。
- 训练/验证与最终测试必须隔离，checkpoint 保存不能依赖测试集结果。

由 Independent Reviewer 验证实现、参数量、梯度、优化器分组、前向等价、测试隔离和运行状态保存。

**实现审核未通过，不允许正式训练。**

### Step 3：执行 R2-A

按照已审核队列执行：

- B0：seeds 2022、2023。
- B1：seeds 2022、2023。
- B2：seeds 2022、2023。

每次训练 50 epochs，遵循原始 Sample-M3OE 协议。

先完成两个 B0，并验证 selected checkpoint 可以严格重新加载及重放验证集结果。

记录全部验证集逐格 AUC、LogLoss、macro、加权指标、selected epoch、训练轨迹、模型参数与 checkpoint。

六次实验有效完成后，进入 Step 4。

### Step 4：执行 R2-A → R2-B 判决

严格使用 Charter v1.2 的验证集判定规则。

定义：

D_s = ValMacroAUC(B1,s) − ValMacroAUC(B0,s)

T_s = ValMacroAUC(B1,s) − ValMacroAUC(B2,s)

执行已预注册的实用信号、探索性信号、guardrails 和 B1−B2 额外收益判定。

使用完整精度数值，不从显示时四舍五入的表格计算。

生成不可变的阶段 decision manifest，记录 YES/NO、所有差值、门槛、保护条件、失败原因及剩余预算。

不得根据测试集决定是否追加实验。

### Step 5：条件执行 R2-B

仅当 Step 4 满足全部追加条件时执行：

- B3：seeds 2022、2023。
- B4：seeds 2022、2023。

如果不满足条件，按预注册规则停止追加，并保留已经观察到的 B1/B2 结果及其证据边界。

不得使用剩余预算临时增加新的方法、种子或超参数实验。

Hpost 保持默认 disabled。

### Step 6：最终测试与结果分析

完成所有需要执行的实验后，先冻结全部选模、配置、checkpoint 和阶段决定，再统一评价测试集。

实验报告必须包括：

1. B0/B1/B2，以及条件执行的 B3/B4 的完整 AUC 与 LogLoss 表。
2. 每个 seed 的配对指标差值、均值和不确定性。
3. 六个 domain-task 单元的收益与退化。
4. Macro AUC、训练域份额加权逐格 AUC 与各任务指标。
5. Validation 与 Test 收益是否一致。
6. 各实验臂的融合参数、训练轨迹及 selected epoch。
7. B1 相对 B0、B2、B3、B4 的实际表现。
8. 是否存在简单方法已经能够取得类似收益的情况。
9. 现有结果能否支持继续研究 AO-04。
10. 进入 R3 的 GO / CONDITIONAL / STOP 建议，以及尚未排除的竞争解释。

结论分为描述性现象、初步性能信号、实用性能通过和剩余算法研究空间四层，不得混淆。

两个训练 seeds 不能证明算法具有稳定、普遍的性能收益。

## 五、执行约束

- 不得重新设计 B0–B4。
- 不得修改已冻结的 objective、学习率、epoch、数据或选模方式。
- 不得根据测试集调整实验。
- 不得以验证集阴性结果为理由临时更换研究方向。
- 不得添加在线亲和矩阵、EMA、动量、课程学习等新方法。
- 总训练预算最多 10 次完整 attempts，R2-A 最多 6 次，R2-B 条件追加最多 4 次。
- 基础设施异常按照原 Execution Plan 的精确恢复与停止规则处理。
- 对 INVALID、INCOMPLETE 和科学阴性分别记录。
- 不得自动进入 R3 或 R4。

## 六、阶段交付物

请在独立 `research_R2/` 目录中保存：

- 代码实现及配置。
- Implementation Report。
- Independent Implementation Review。
- Baseline Reconstruction / Validation Report。
- 全部运行 manifest、训练日志、checkpoint 和指标。
- R2-A Decision Report。
- 条件性的 R2-B 实验记录。
- `R2_Experimental_Analysis.md`。
- `R2_Global_Conclusion.md`。

报告使用中文，数学公式、实验变量和代码标识保持原名。

如在运行过程中遇到 hard blocker，按原协议停止，输出完整阻塞原因及已完成工件，不伪造后续训练结果。

**本任务正式授权执行 Stage R2。请按照已审核方案完成可执行的全部工作，完成后停止，不自动进入 R3。**