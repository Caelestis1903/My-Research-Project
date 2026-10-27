# 独立复审：R2 FP32 比较与两份诊断源码的兼容冻结

日期：2026-10-27。Reviewer：`/root/reviewer`；Coder：`/root/coder`。

**Verdict：PASS。Blocking findings：0。**

可保留四个存活的 MovieLens R1 worker，归档实际失败且已失去 RAM 模型的 KuaiRand worker，再以 R2 原 seed、原预算重跑对应组。八组均 READY、48 条 receipt 齐全且主agent关闭阶段决策后，可使用专用工具生成兼容冻结 view，再按原入口逐组释放 E 评估。

## 审查范围与直接证据

已读取 `DESIGN_RUNTIME_NUMERICS_ADDENDUM.md` 第5–9节、`CODER_FIX2_REPORT.md`、两份 CPU 验收 stdout、实际比较分支与测试、`freeze_compatible_revisions.py` 及其 toy quorum 测试。

独立调用了工具的只读 `source_equivalence()`：

| 身份 | 实际核对结果 |
|---|---|
| R1 full source | `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`，完整快照41文件与清单/摘要匹配 |
| R2 full source | `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`，完整快照41文件与清单/摘要匹配 |
| 共同 training source | `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`，排除且仅排除 diagnostics.py 与 tests/ 后37文件集合及逐文件字节相同 |
| 实际变化 | 仅 diagnostics.py 与 tests/test_runtime_numerics.py |

已逐项查看 diagnostics diff：仅将跨 batch helper 的 atol 从1e-6改为1e-5，并添加逐元素误差/允许量比值最大点的证据；移除该 helper 后，新旧 diagnostics 的全部导入、科学诊断、raw预测和时延函数 AST 完全相同。模型、native/source、训练、data/cache、指标、选点、RAM恢复及既有 Data 授权实现保持原字节。

## R2 数值修复

当前 helper 按明确 cached 参考端执行逐元素 all：

`abs(raw-cached) <= 1e-5 + 1.3e-6*abs(cached)`。

shape 与两端 finite 检查仍先执行；NaN/Inf不通过容差分支。原 max_abs/max_index 继续记录，同时保存 max_normalized_error、worst_index、该点两端值/真实误差/允许量，能定位“最大绝对差点通过而另一近零点失败”的情况。失败信息继续进入 ValueError 严格 JSON，并由原 worker FAILED 路径保存。

`CPU_FIX2_ACCEPTANCE_STDOUT.txt` 为 **1 passed，1.29秒**。已确认测试覆盖原ML点、KR近零点通过；明显超过新阈值、NaN/Inf、shape错误拒绝；两元素反例使 worst_index 指向实际违规近零位置。该改动仅用于跨 batch final-logit 部署 parity，teacher/RAM绝对1e-6不变，50/200时延次数与实际batch定义不变。

## 兼容冻结授权

专用工具与当前设计相符，保持以下实际限制：

- 仅接受写死的 R1/R2 两个完整 source hash；不接受“任何训练摘要相同”的其他修订。
- 先重新核验两份完整快照、相同文件集合、唯一允许的diff、37文件训练等价及非比较诊断AST等价。
- 读取全部两数据集×四seed的 READY、source清单、effective config、data manifest和六方法receipt；要求 READY状态、实际dataset/seed/config、六个selected/final、精确source清单、逐数据的角色/预处理身份和receipt状态hash一致。
- stage2必须对两个数据集实际关闭；方法、阈值、decision、训练等价证明和全部8个真实worker条目写入同一 canonical core，先计算其摘要。
- 两view包含相同 canonical identity 与完整真实worker表，仅顶层 source_hash 分别服务 R1、R2；不重写任何 worker 原身份，不修改或绕过已有 Data.authorize_final。
- canonical索引另记录两个最终view的实际文件hash、view摘要及逐worker接线路径，避免循环hash。
- 输出使用新建且拒绝已有文件；不写任何 EVALUATE release。E仍需主agent完成全局冻结后通过原 evaluate 入口逐组释放。

旧 worker 读取默认 `GLOBAL_FREEZE.json`，重跑的 R2 worker须通过既有 `--freeze-file .../GLOBAL_FREEZE_FP32_STANDARD.json` 接线。原 release 函数从默认view取出的每组entry与另一view完全相同，因此可释放两组真实身份。接线按source hash判定，未把dataset或seed硬编码成授权依据。

`CPU_FREEZE_COMPATIBILITY_STDOUT.txt` 为 **1 passed，0.78秒**。已检查测试实际加载两份快照里的 Data validator：分别对正确view授权、错误view拒绝；两view同一8worker/canonical身份；缺一个READY即拒绝。复用这些有效测试证据，未追加GPU或大范围测试。

## 执行接线

冻结工具的 `--review-record` 应指向本文件 `research_reports/26-10-27_v2/RUNTIME_FIX2_REVIEW.md`。工具只在八组真实READY且主agent阶段决策已完成后执行。失败目录先归档保留真实身份、日志和成本；保留存活ML进程及其RAM模型，不能为了工程revision一致而重训。重跑不增加独立seed重复数，最终报告应分别列出R1/R2实际来源与额外成本。

未修改实现、未终止进程、未运行GPU训练或额外研究实验。
