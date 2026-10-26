# DFD-v4.0 报告与工件入口

本轮已按注册门槛正常结束：4个native24前缀与全P1，ML20个P2输出完成；KR P2及两数据P3/P4 NO_START。184/184冻结导出、12/12完整推理测量和用户要求的临时模型清理完成。外层固定μ分组有明确MAJOR缺项，不能声称全部机制信息完整。

- [最终结论](FINAL_REPORT.md)
- [数据与角色](DATA_AND_ROLE_AUDIT.md)
- [P1完整结果](P1_DEMAND_REPORT.md)，[ML详情](P1_MOVIELENS_ANALYSIS.md)，[KR详情](P1_KUAIRAND_ANALYSIS.md)
- [完整P2与冻结伴随角色](FULL_PIPELINE_REPORT.md)
- [归因与证据强度](ATTRIBUTION_REPORT.md)
- [成本/标签/推理/存储](COST_REPORT.md)
- [开放问题与缺项](OPEN_QUESTIONS.md)
- [过程修订](PROCESS_REVISIONS.md)，[旧RHD有界诊断](RHD_BOUNDED_DIAGNOSTIC.md)
- [设计规格](DESIGN_SPECIFICATION.md)，[训练独立审查](IMPLEMENTATION_REVIEW.md)，[最终化独立审查及缺项补审](FINALIZATION_REVIEW.md)
- [184个冻结预测、4个P1 OOF及形成责任信息的路径/身份](ARTIFACT_INDEX.json)
- [完整聚合数值、所有六格、seed/cluster区间、角色与终态](tables/)
- [冻结、实际阶段决定、审计和清理机器收据](evidence/)

服务器信息产物根：`/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0/`。大原始数组、行级预测和模型不上传GitHub；模型已按用户要求清理。Git分支只发布新源码、精简报告、聚合数值与审计身份。原workspace中的新源码/报告为独立有效checkout的链接，原失效.git未修改。

分析可用源码 `analysis_p1.py`、`analysis_pipeline.py`、`analysis_final.py`、`analysis_support.py`、`analysis_costs.py`、`build_reports.py` 从保留的预测和信息复算。训练入口在 `experiments/mdmt_dfd_20261026/cli.py`；本次冻结/评分/清理由独立 `finalize.py` 执行。所有重推须先恢复/重训模型，不会因摘要或hash生成虚假预测。
