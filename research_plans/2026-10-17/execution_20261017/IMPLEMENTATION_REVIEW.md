# N1 实现自审

按start.md §11由主agent逐项比对DESIGN_SPECIFICATION、实际n17实现与IMPLEMENTATION_REPORT；independent_review=false，不冒称独立审查。

核对：Sample loss=(两任务BCE均值)*p/q后按行均值；固定融合和detach图复用；初始化与NumPy sampler独立；两臂共享数据、seed和配置；全H100不按短期指标淘汰；selection仅固定半集、earliest strict-max，test延后；每epoch总examples和尾batch与原实现一致；Adam/SGD及WD一致；E10/E40完整状态供机制锚点；源hash冻结；每卡外部进程计入并发上限。批次RNG可重建，但不承诺GPU浮点跨硬件bitwise。

发现并在正式实现前修正的文档录入问题：尾batch实际40052；N3 delta_local原文为.0001，均已按原文修正。未读取本轮训练结果，无阈值更改。

Verdict: PASS WITH FIXES。阻塞条件为必要smoke及完整GPU epoch通过、N0完成、N6差异表和N1 manifest冻结。满足后才启动32条pilot。
