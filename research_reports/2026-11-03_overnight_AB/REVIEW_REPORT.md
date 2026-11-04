# 独立实现审查报告

裁定：**PASS WITH FIXES**。A_HYPERGATE、A_AUTOIFS、B_CP的阻塞问题已修正，可交runner执行两数据原batch smoke，再按冻结矩阵正式训练。正式派发代码同时要求家族准入和对应SMOKE.json的PASS。

## 审查依据

独立读取[设计规范](DESIGN_SPECIFICATION.md)、冻结01–04、实际新包源码、原history/pair更新公式、固定发表模型训练流程，以及[Coder A](CODER_A_IMPLEMENTATION_REPORT.md)、[Coder B](CODER_B_IMPLEMENTATION_REPORT.md)、[Coder runtime](CODER_RUNTIME_REPORT.md)三份实施报告。审查者为`/root/reviewer`，与三个coder分别为不同agent。

A保留HG原生目标/辅助项、初始化及PCG64 shuffle；AutoIFS按search0–9执行，在进入search9前保存rewind，最终selector含gate embedding移植至rewind模型，hard ticket、冻结gate后建立全新reuse Adam并训练reuse1–50。B保留原PAIR采样和C probe随机流，先更新h/active，再以未投影b²更新v；仅投影u=b−lambda*(w/q)*r进入m，H单独加入最终分子；shared/ordinary各一次参数更新、coupled L2一次、固定EMA一次。关闭组件直派源入口。

统一选点仅看合法最终阶段V六格macro AUC严格最大值、精确并列最早；同一完整模型状态用于V/Test，发表logits只做一次FP32 sigmoid。完整Session与优化历史经BytesIO IPC驻RAM，磁盘保留标量和数据缓存。队列按epoch量子公平分配A/B，各GPU独立按85%显存/功耗门派发；已识别共享负载的精确owner快照由RUNTIME_OVERRIDES记录，未知计算PID和旧MDMT保留设备继续阻塞。

## 已修问题

| 级别 | 问题 | 实际修正与核验 |
|---|---|---|
| Critical | 换卡恢复沿用CUDA列表序号，HG dropout/AI随机门会使用错误流 | A/B显式传会话device，活动CUDA流映射目标GPU；真实双卡最小随机张量断言通过 |
| Relevant | 新包source身份缺实际原生/外部依赖核验 | 实际核对history/pair固定hash及6个发表模型依赖文件；最终导入根均来自独立执行树 |
| Critical | 直接run-job缺协调器认领与GPU资源检查 | 加活跃coordinator claim、trajectory锁、按host/UUID派发锁和资源/owner复核；smoke使用同一GPU门 |
| Relevant | 首次分配等待跨卡耦合，旧A审计统一占位且B表漏CP | 等待按设备分组；逐轨迹接入旧A审计，A60/B40槽位已核对 |
| Critical | 正常yield的worker=None窗口可触发.get异常 | None owner显式归一为空dict |
| Critical | TRAIN_COMPLETE后评价中断无法重新派发 | 保留评价worker身份；owner死亡计基础设施重试，恢复complete Session直接补archive，训练循环推进0epoch |

上述修订在新训练启动前完成；冻结数学、五seed矩阵、预算和阈值保持设计定义。

## 独立验证与准入身份

在原生执行树，使用项目Python实际运行：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/overnight_ab_20261103/tests -q
```

结果：**15 passed in 5.17s**。覆盖A/B真实小batch backward、目标与单次更新数值、AutoIFS阶段边界、连续与RAM恢复轨迹、严格选点/统计边界、epoch公平/资源门、guardian正常量子不耗重试及真实双卡RNG映射。固定源TF32旧API产生1条弃用warning。

原生history实际hash为`7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`，pair为`5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`，均吻合冻结来源。独立核对prepare：80条身份=30 REUSE_DONE＋30 READY＋20 CONDITIONAL_NOT_ENABLED；两数据raw SHA及自然train域人口吻合protocol；A审计60行、B审计40行；prepared与当前家族hash一致。

| 家族 | 裁定 | source hash |
|---|---|---|
| A_HYPERGATE | PASS WITH FIXES | `ea9b66f27060450627b8ea996120938679fdbea444a6fa4e1bb2d0773e783f39` |
| A_AUTOIFS | PASS WITH FIXES | `ea9b66f27060450627b8ea996120938679fdbea444a6fa4e1bb2d0773e783f39` |
| B_CP | PASS WITH FIXES | `7e78497b1b5483147c3848f424c1980f58aca7c9ef93ae880a8497e75daf7a7f` |

B_CG保持默认关闭，本次准入为A与主组合CP。runner下一步为两数据HG1024、AI4096、CP40960原batch smoke；各对应记录PASS后执行完整2022–2026队列。审查未进行科研性能裁决。
