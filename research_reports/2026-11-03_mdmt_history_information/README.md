# MDMT条件信息增强与真实历史更新

本轮完成13条新增50epoch训练，复用固定历史A0/A1；两数据均在预定D0门停止，统一冻结并完成126个test端点评估。无失败或重放。

- [实验报告与研究决定](REPORT.md)
- [逐seed数值附录](NUMERICAL_APPENDIX.md)
- [研究问题处置](QUESTION_DISPOSITION.md)
- [全部端点](endpoints.csv)、[六格指标](six_cell.csv)、[域/任务汇总](domain_task_summary.csv)
- [配对原值与差](paired_seed.csv)、[配对统计](paired_summary.csv)、[完整训练阶段指标](validation_epochs.csv)、[成本](costs.csv)
- [运行与历史来源身份](RUN_MANIFEST.json)、[独立审查](REVIEW_REPORT.md)、[执行记录](RUNNER_REPORT.md)
- [原始标量归档与恢复](RAW_ARTIFACTS.md)
- [实现源码](../../experiments/mdmt_history_information_20261103/)、[冻结方案](../../research_plans/26-11-03/)

当前保留A0f。KuaiRand C_MIX具有单seed宏收益，但验证单格退化触发保护门；候选没有多seed确认，A0b和匹配U/NORM控制未触发。具体收益、代价与未解问题见报告。
