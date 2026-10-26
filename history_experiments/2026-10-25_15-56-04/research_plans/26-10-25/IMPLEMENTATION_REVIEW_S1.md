> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`9f7ce0ab40c16cf8bd9d75263d1e746c97384e0a3c627dca0b29993cb4d87dc5`。

# Primary-agent implementation review — S1 admission

Scope: S1 native C8 exact branches and mean/concat K1/K4 warmups. Review is not independent; the user explicitly disallowed non-run delegation.

Real ML/KR smoke passed: native max absolute probability difference 0, embeddings 113/544, exact A51 forward/cache identity, 51→7 mapping, risk/BCE argmax equivalence, RO donor/native gradient isolation, aux native isolation, private natural C8 gradient equality, full model/Adam/grad/RNG restoration digest, gamma0/HIST fallback. Data SHA/role reconstruction validated against existing exact row IDs.

Review fixes: identify shared parameters by module objects (including task experts); clear unused foreign reader grad to None so OFF does not advance its Adam state; keep scalar utility and per-row task pairing; restore actor state and optimizer binding together; use pure support/panel plans. No H1 gradient cache is enabled: all eight candidates recompute the real forward and Adam update, so the conditional cached-vs-recompute acceptance does not apply.

Verdict: PASS WITH FIXES; listed fixes are applied. First native C8 anchor additionally asserts full restoration on actual support40960 and dual1024 panels. Failure blocks the affected run; not a scientific null. LPS selectors and later stages require their own review before admission. S1 diagnostics do not unlock outer roles. Source is frozen per resolved configuration, and exact source receipts remain attached to each attempt.
