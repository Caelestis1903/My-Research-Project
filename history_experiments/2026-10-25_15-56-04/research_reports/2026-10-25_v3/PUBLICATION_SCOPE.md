# 发布范围、身份与服务器工件

本报告对应`start.md`及`research_plans/26-10-25_v2/`规定的MDMT-NEXT-v3完整实验。全部已触发训练、最终评分、统一推理及科学分析均已完成。

## 远端目标与历史保留

- 仓库：`Caelestis1903/My-Research-Project`
- 新时间分支：`2026-10-25_15-56-04`（任务起始时间命名，Asia/Shanghai）
- 固定父提交：`619371d9c4d57f6f0b3e8a71b686498f9e5f36d4`
- 固定父tree：`36aee99d71f52e5c559001d263abbf49c2311b81`
- 本轮报告位置：`research_reports/2026-10-25_v3/`

用户明确授权将完整实验报告推送到时间新分支。新提交基于上述固定父提交，仅增加本轮报告、汇总数值和PNG/PDF图；保留父提交中的既有研究成果及阴性结论。服务器工作区原有未提交文件不纳入发布操作。

## 完整报告内容

包含11个规定主报告、索引、完整设计/实现审查及修订记录；89逻辑轨迹和8额外本地预测器的selected与共同终点Q数值；两个历史暴露角色的全部388个预测器检查点结果、六格、域/任务、配对统计和身份；194个验证AUC/LL别名身份；P0、固定状态删除、形成元步、来源权重、CAL/GEOM诊断；三本成本账JSON收据、89方法部署成本汇总、进程wall、信息汇总/UNKNOWN及全部51项推理测量原始重复记录；预定检查点质量—已记录成本观察及图。

575页逐操作日志转录（组件、逐次原始行materialization、研究实际leaf、部署组件明细、开发搜索操作）与原始CSV/journals保留服务器。它们不增加科学重复数；公开成本索引给出每页服务器路径与原SHA256，所有汇总数值和未知项仍公开。源报告、完整本地分页及成功生成收据均原样保留；暂存副本只将这些服务器日志页链接改为路径/hash文字，发布manifest记录每次改写及原/暂存SHA。

原始数据、逐行标签/预测、支持/查询row_id数组、完整权重/Adam/RNG、训练源码归档和执行日志保留在服务器。公开图与表为汇总结果；它们不能替代取得原始工件权限后的独立重跑。新seed只提供固定旧样本下的随机性复评，不提供新未见数据。

## 服务器复核根目录与收据

服务器根目录：`/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/`。checkpoints位于其受保留的符号链接目标`/home_new/xzf_mdmt_next_v3_2026-10-25_15-56-04`，算法源码归档位于`source_versions/`；原始数据/角色/编码器身份见协议附录。以下SHA256对真实本地文件字节计算。

| 收据 | 相对工件路径 | SHA256 |
| --- | --- | --- |
| 最终方法冻结 | `FINAL_METHOD_FREEZE.json` | `8a0698db50ed4d129909b9decdd400e1ee8db89cfa142591f942282f9405e382` |
| 正式不可变锁 | `FROZEN_CONFIRMATION_SPEC.json` | `697c9f3250986d7687a26a21e1f4bc57b4a8077c62e1ff5208537d91d96c934b` |
| 89轨迹Q聚合 | `analysis_final/AGGREGATION_RECEIPT.json` | `ad6eedaec09dc6b5165c5636a082aa0d194d67dde7eb48227eb8e89008ae1d4e` |
| 两个exposed角色聚合 | `analysis_exposed_final/EXPOSED_AGGREGATION_RECEIPT.json` | `100f364b265f252f9f00f19705adc0af0f547beb1435e4a910af27c7bfbf0650` |
| Q数值附录 | `analysis_final/APPENDIX_GENERATION_RECEIPT.json` | `9668203fded32ce981cf682521361e46d809b77402743699f62fe980fdbdf4e1` |
| 成本/推理附录 | `ledgers/ACCOUNTING_APPENDIX_GENERATION_RECEIPT.json` | `2db23ff94418bc5c114dba00b2114b4fd408a0cdeabb62eb97dca00403455d84` |
| 194别名核验 | `analysis_validation_aliases/RECEIPT.json` | `22008fabf4fbbf40dbff65c1c45a0c87343649cf0839aac0ecadb2c30727a157` |
| 81源码完整性 | `FINAL_SOURCE_INTEGRITY_CHECK.json` | `a170695462c6d6b34f4327f36166aabf0b651aef1d681b950e2cc61fa650339c` |
| 正式训练执行 | `campaign/FORMAL_bulk_execution_report.json` | `7c4712275e3eebe19e39005c339f9004fa58eec9136291bc3173340da7e13a73` |
| 最终评分执行 | `final_exposed/EXECUTION_RECEIPT.json` | `699def2f85507b4438004340488f8a3d9c531ec33403b05fcd5dd2ab1b3327bc` |

发布暂存清单与上传回执保留在服务器`artifacts/publication/`，列出每个发布路径、大小、原/暂存SHA256、Git blob SHA1及排除日志页SHA。远端分支head/tree/blob身份逐项核验；最终commit身份记录在发布回执及交付答复中，避免在同一提交内容中写入自引用commit SHA。
