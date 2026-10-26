# 独立审查：驻留阶段并发调度

Verdict：**PASS**。Blocking issues：无。本审查仅授权已登记的运行调度补充，不构成科研效果分析。

已独立阅读 DESIGN_PARALLEL_SCHEDULER_ADDENDUM.md、CODER_IMPLEMENTATION_REPORT_PARALLEL_SCHEDULER.md、实际 phase_scheduler.py、readers.py 末尾 hook、原 Campaign.phase/reload_readers/worker 消息接口及 coder mock 收据。Coder 与 reviewer 为不同 agent。本 reviewer 没有连接、安装、重启或训练真实 service。

安装入口只允许固定 PID2088990、工作区、artifact 根、config hash 与 Campaign.phase 目标；worker/client 在第一条 PID 判断即 NO_OP。授权必须为独立 PASS/PASS WITH FIXES、无 blocking，逐项绑定旧/新函数指纹及两个源码文件 SHA。它只取已加载的 campaign 模块，不 reload campaign。按编排者明确批准的附加边界，仅在当前同步调用栈中匹配精确 Campaign.reload_readers code object，严格检查 typed self、固定 config/root、八个指定 worker 全部存活且 idle、未冻结及原 s0 COMPLETE 状态。没有 gc/其他线程扫描、模型/Adam/RNG 访问或任意 eval/exec。全部校验通过后唯一类属性赋值是 Campaign.phase；安装前后写固定收据。前置不满足则 FAIL_CLOSED，不能为吞吐重新训练 prefix。

并发函数保持原 S1/S2/S3 的 worker、device、训练方法及 methods/decision 消息；只对原开发种子派发，S3 空子集跳过。每卡每轮重新读取显存/功耗，任一比例达到85%或无可靠读数均不准入；主机另保留8GiB，并扣除全部 pending 尚未体现到 RSS 的保守工作集。它轮询所有 active 连接，按原 key 绑定完成摘要，最终结果按原清单顺序返回。出错停止新派发，收齐其余已派任务而不关 worker、不丢模型，明确 FAILED_DRAINED；不会误写 COMPLETE。locked、freeze、export、bootstrap 与推理计时接口不变。

独立实际检查：

- 原 readers.py 全部字节仍为连续前缀，其 SHA 精确等于上次授权的 `1a9f578f0dc7bb7732b4c61a13320955f71c4340e6a4232c96f5dfd9a12d6493`；全部原函数/类 AST 一致。
- 逐一对比原串行 phase 与新调度在 S1/S2/S3 mock 的训练消息和返回结果，完全相同；config 及 worker/process/pipe 对象身份不变。
- 同卡两个合格任务在首次训练 recv 前均已发出；乱序完成仍正确绑定且原序返回。
- 单卡显存85%、功耗85%或缺测均阻止派发。独立重执行 coder 的 pending/host 预留检查通过。
- 失联 worker 在任何派发前拒绝；任务失败后保留句柄并 drain 其他结果，不写 COMPLETE。
- 所有固定授权字段任一改变均拒绝；reviewer 普通过程调用返回 NO_OP，Campaign.phase 未被修改。

收据：PARALLEL_SCHEDULER_INDEPENDENT_CHECKS.json。所有 mock 均未接触真实 service，不生成科研性能结论。

本次授权刷新全源码/S0文件映射、正常 reader reload token 及专用 scheduler token。实际安装仍须由编排者通过唯一正常 reload-readers RPC 执行，核安装收据、父 PID 保持及八 worker 前后模型 hash 相同。实际并发日志、旧种子 Q 指标及 native/pilot hash 一致性仍属 RUN 验收，不从本 PASS 推定。任一前置条件失败时继续原串行科学矩阵，不再开第二条安装路线。并发墙时不得冒称独占 GPU 的推理加速。
