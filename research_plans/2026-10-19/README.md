# 2026-10-19：STAR/条件调制共享路径失活的跨模型研究

**状态：研究方案已制定，尚未运行本方案新增训练。** 研究面向推荐系统与机器学习顶会顶刊，目标不是证明 AdamW 一定更好，而是找到**失活的适用范围、触发机制、与泛化性能的因果联系、潜在新颖性**。

## 推荐阅读与执行顺序

1. [FULL_EXPERIMENT_PLAN.md](FULL_EXPERIMENT_PLAN.md) — **完整主方案**。科学问题、现有证据、模型、功能失活定义、全过程训练、机制隔离、统计/选模/泛化、结论与交付。
2. [RUN_MATRIX_AND_DECISIONS.md](RUN_MATRIX_AND_DECISIONS.md) — **具体训练矩阵**。原论文/改造模型、两数据集、Adam/AdamW、种子、WD、Skip/辅助路径、加性对照、更新数控制，以及判定依据。基础 G0 共 160 个名义正式网络（10模型任务配置×2数据集×2优化器×4种子），另有明确机制、结构及 MDMT 扩展组；严格相同的历史有效训练和数学等价组允许有证据地复用，不能虚报。
3. [MODEL_FIDELITY_AND_LITERATURE.md](MODEL_FIDELITY_AND_LITERATURE.md) — **原模型与代码忠实性**。M3oE 原论文训练 vs N19 fixed、独立 STAR、PEPNet、HAMUR、MMoE/PLE，以及原理覆盖风险。
4. [START_EXPERIMENT_PROMPT.md](START_EXPERIMENT_PROMPT.md) — **可直接发给 Codex 的启动指令**：完整执行、有意义的方法必须训练、最小必要科学检查、不滥用防御性测试。

## 研究对象

- **核心复现**：N19 fixed M3oE、原论文风格 M3oE、独立 STAR，重点测 shared × domain-specific 的乘性权重路径。
- **跨类机制**：PEPNet 的 embedding/DNN gate、HAMUR 的动态 adapter，注意中性化干预与 M3oE 不同。
- **非同构共享对照**：MMoE/PLE，区分共同的优化路径竞争和 STAR 特有的参数化问题。
- **因果机制**：Adam/AdamW 耦合/解耦权重衰减、仅调制参数组干预、Skip/辅助支路竞争、乘性 vs 加性参数化、优化更新次数与样本量。
- **最终价值**：不仅看参数路径是否活跃，还要对比无模块从头训练、最佳 checkpoint 性能、cell AUC/LogLoss、模型成本与强基线，并区分训练末期过拟合效应。

## 为什么要做

[此前 N19 报告](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md) 指出：KR + Adam 下 M3oE 的 STAR 调制权重路径 Epoch 5 即在 float32 中无可观测贡献；ML + Adam 不是完全失活；AdamW 恢复 KR 的路径功能贡献，却没有稳定提高 AUC。**这些只是具体训练协议下的现象，不证明原始 M3oE、独立 STAR 或所有 MDMT 模型有相同缺陷。**

历史研究 N18/N19 的执行 PR 尚可能未合并 main。本文件夹是**独立的新规划**，不会隐式合并那些分支；执行应显式引用经过验证的 N19 实验 SHA、代码和本地 checkpoint/缓存谱系，避免遗漏关键数据或重做无关实验。

## 重要纪律

本轮用户明确说明资源足以支持完整训练，不允许重新退回“只做固定点测量后就整体 COMPLETE”；同样不得预设 AdamW、动量、课程、采样和任何共享修复有效。关键阴性/冲突结果原样报告；未知数据限制如实记录，不伪造模型复现。**仅做可能改变训练或科研结论的必要测试，杜绝与研究问题无关的防御性工程检查。**

完成本方案不是“必须找到一个正结果”，而是实际完成各个可执行对照并能回答：何时失活、为何失活、是否有害、适用哪些模型、是否存在新论文价值。
