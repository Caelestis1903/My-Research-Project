# 最终结果与报告独立审查

**结论：PASS。阻塞项：0。**

训练实现 SHA256：`a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`；实现提交 `bf36e454adc2d0c569218a7993834f6d56cfc97f`。原实施审查及 R1 修订历史保持不变。

## 身份与完成量

- 独立重新计算源码收据中59个文件的 SHA256，以及设计规范与运行清单中的设计 hash；全部匹配。
- 60个唯一 `(dataset,arm,seed)` 配置恰好覆盖两数据集、六臂、seeds2022–2026。每条均为50epoch，仅存在 `attempt1`，没有失败记录；唯一派发清单为 PILOT。
- 逐一核对3000份epoch记录和3000条唯一RAM提交。120个online/EMA选点均为各自验证六格Macro AUC的最早严格最大值；selected epoch、完整指标及对应原始epoch模型hash一致。
- 同数据集/seed/epoch的六臂行序hash完全一致。103500条batch记录各含一次Adam和一次EMA更新，与每条训练的步数和最终计数一致。
- 两数据集各六臂GPU smoke记录为PASS，源码身份一致。没有freeze或Test端点文件；360条导出端点均标记validation。

## 数值、表格与成本

从原始selected端点独立重算全部4686行配对统计：五seed差值的均值、样本SD、95% t区间及正向seed数全部一致，最大浮点差为 `5.551115123125783e-17`。统计单位保持为seed，阶段曲线先在seed内求阶段均值。

指标台账新增24行的六格及Macro/Traffic/Pooled AUC、LogLoss均与原始五seed端点在九位小数舍入内一致。报告、汇总表和逐seed附录明确使用validation；台账总说明也已区分该分组与历史Test分组。Macro、按评价人口加权的Traffic和逐任务跨域池化的Pooled没有混称。

worker墙钟合计 `20.79278644905102` 小时，包含数据载入、训练、验证和内存提交；pilot墙钟 `5674.28845836292` 秒。GPU峰值allocated为 `1.0110588073730469 GiB`，reserved为 `1.345703125 GiB`。按原始launch到最终epoch提交区间重建的最大重叠数为23，与报告的并发观测一致。

## 研究结论与停止门

A/B/C相对clean固定EMA的selected validation Macro AUC，在两个数据集均为0/5正向；因此均无法同时胜过clean、RANDOM与GAUSSIAN。`NO_CONTINUATION_SIGNAL` 与预设阶段门一致，HPO、正式确认、Test及条件消融标记NOT_TRIGGERED正确。本轮按既定停止条件结束。

报告保留ML C的LogLoss改善、KR C相对泛化扰动控制的优势、早期与晚期反号以及online/EMA总收益与组件增量的区别。当前配置的验证结果、投影数值有效性和机制因果解释分别赋予证据强度；没有将未触发的Test阶段写成正式性能否决。

域人口、标签先验和参考表现分开呈现。报告没有用域大小替代困难度，也没有把辅助目标组成解释写成已完成的独立因果分解。建议的零扰动对照只作为下一项判别设计，没有越过停止门追加实验。

本次审查未修改训练源码、原始结果或历史报告，未启动训练或Test评价。报告、数表、完成状态和台账可进入提交发布流程。
