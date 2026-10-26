> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`b500a52392bc0d77d9ad9c9f4e1d278dccfe0041ba285502d049ade44aa71905`。

# DESIGN SPECIFICATION — MDMT-NEXT-v2.0

Authoritative design: research_plans/mdmt_next_v2_20261025/01–04 copied byte-for-byte from the user-provided 26-10-25 directory. Equations/interfaces: 02; matrix/budget: 03; execution overrides: 04.

Implementation is independent of historical runners. Reuse only checked raw preprocessing, mechanical native forward and full-state primitives. Native references retain natural sample/task BCE, Adam .001 coupled wd1e-5, batch40960, FP32. New C8 shared ownership is explicit by module object identity. New LPS bank uses separate content and key counts, exact A51 logits and owner/RO gradient boundaries. All policies obey prefix-only feedback and whole predictor/actor checkpoints.

Run order: P0 semantics/timing → S1 eight warmups and diagnostics → STATIC calibration and chronological bootstrap → S2 48 core trajectories plus mandatory aux reserve → recent-method audit/pilot → promotion and immutable formal lock → all locked training → exposed outer evaluation → analysis. No quality early stopping. User-specified seeds override historical defaults.

Acceptance is limited to actual graph/roles/state/actions/selection/cost correctness. No claims about scientific effectiveness before completed comparisons. Review is performed by the primary agent before run delegation, per 04 explicit override. Mechanical training only is delegated to luna_worker.
