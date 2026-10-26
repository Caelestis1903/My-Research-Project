# 2026-10-24 实验报告

本次提交仅发布实验报告及报告内使用的统计图。没有纳入数据表、机器可读结果、配置、日志、源码、原始数据、逐行预测或模型权重。正文内的数值表属于报告内容。

- [完整最终研究报告](FINAL_REPORT.md)：全流程、四seed正式比较、selected/final、逐域任务变化、成本、研究决策和证据限制。
- [S1 初步机制探索报告](S1_REPORT.md)
- [S2 算法开发与pilot报告](S2_REPORT.md)
- [I3/I5 有限重放报告](REPLAY_REPORT.md)
- [正式selected配对统计图](figures/paired_selected.png)

本轮完成44条主pilot、4条基线适配pilot、40条正式训练及40个统一评价任务；正式seeds为2024–2027。当前I1_PERM未获得预注册selected ranking优势：相对固定RW，test macro AUC差为MovieLens −0.000334、KuaiRand −0.000513，描述性95%区间均跨零。KuaiRand final LogLoss的改善保留为限定副指标结果，未替代主结论。

为避免失效链接，报告发布副本将未上传工件的相对链接改为服务器归档说明；未调整实验数值或结论。原始报告内容SHA256分别记录在各报告开头。
