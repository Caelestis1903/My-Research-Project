# CODER IMPLEMENTATION REPORT — resident parallel phase scheduler

状态：IMPLEMENTATION_COMPLETE，待独立reviewer PASS。没有连接、重启、修改或训练真实service，也没有分析任何实验性能结果。

依据：`DESIGN_PARALLEL_SCHEDULER_ADDENDUM.md`。parent额外明确批准：仅在当前同步正常RPC调用栈寻找code object精确等于已审查 `Campaign.reload_readers.__code__` 的frame，取严格typed Campaign self只读核config/root/八worker alive且pendingNone；不扫描gc/其他线程、不保存frame、不取任意对象。

修改仅有：

- `experiments/mdmt_rcf_20261026/phase_scheduler.py`：并发S1/S2/S3的普通Python函数、资源准入、固定安装入口。
- `experiments/mdmt_rcf_20261026/readers.py`：原文件末尾追加固定的import/call hook；原有文件字节全部作为连续前缀保留，所有科学函数/类AST逐一相同。

并发合同：

1. 原 `require_review` 仍先执行。所有resident worker必须存活/无pending；依原正常release-data RPC释放纯数据页。
2. 保持原workers队列顺序及S3空子集跳过。每张卡每轮只新派一个命令，下一轮重新采样单卡显存和功耗，二者均小于85%才派发；任一缺读数或达到85%拒绝。
3. 主机以8GiB底线加所有pending尚未反映到RSS的工作集和新任务新增工作集准入；用原资源配置、原prefix实际峰值及该worker已观测峰值中保守上界，绝不按性能结果决定预算。
4. 原worker已有device/本地seed/RNG/模型/训练函数都不变；发送phase/methods/decision消息与原实现一致。不新建worker、不更改batch、不重训已完成method。
5. poll任意完成的pending worker，按key调用原recv/remember；最后返回结果按原key顺序。错误时停止新派任务，drain其余已派任务并保存成功结果，写FAILED_DRAINED而非COMPLETE，不清空句柄/模型。空闲且主机预算无法启动时写BLOCKED并返回明确错误，不永久等待自己的不存在任务。
6. 10秒至少更新调度进度；stage barrier保留。locked/export/benchmark/bootstrap/freeze/finalize不并发化。

固定安装：

- 非PID2088990立即NO_OP，不访问token、不写安装收据。
- 固定root/工作区/config/serviceID、token独立PASS且无blocking、scheduler/reader文件SHA、旧/新phase代码指纹、target必须精确匹配。加载的campaign只从已存在sys.modules获取，不reload campaign。
- 只接受当前正常reload-readers RPC frame的严格typed self，八个指定prefix alive/idle，原status为s0 COMPLETE，且未freeze。所有校验在唯一类属性赋值前完成。
- 唯一目标 `Campaign.phase`。写VERIFIED_BEFORE_ASSIGNMENT和INSTALLED收据；同函数重复安装返回ALREADY_INSTALLED，其他旧方法身份不符FAIL_CLOSED。安装失败不尝试其他注入路线，保留原串行实现。没有模型/Adam/RNG运输或序列化。

最小基本验收：

- 同卡两个eligible命令在第一个recv前已发出：PASS。
- 内存或功耗任一达到85%均不派发：PASS。
- 主机8GiB和pending尚未实现工作集预留：PASS。
- 乱序完成正确绑定key、返回原key序：PASS。
- S3空子集跳过且methods/decision消息精确相同：PASS。
- 一任务失败drain另一任务、无COMPLETE：PASS。
- 错verdict/config/source/oldcode指纹授权拒绝；普通进程安装NO_OP：PASS。
- 修改前reader前缀SHA与原独立reviewtoken一致，所有原FunctionDef/ClassDef AST一致：PASS。

收据 `PARALLEL_SCHEDULER_BASIC_CHECKS.json`；可复查mock脚本 `verify_parallel_scheduler_mock.py`。本次检查没有接触真实service，实际一次normalRPC安装、父PID保持、八worker前后模型hash/真实并发日志由reviewer通过后root/runner执行。独占推理benchmark仍在原训练barrier之后。

代码身份（旧phase从与service相同路径正常import的原campaign取得，未安装）：

```json
{
  "old_phase_code_hash": "1f408a25cf5ba665cb5a3e7cc9edaaabbc1341accd3602fa0e4c7c0106fd2ea8",
  "new_phase_code_hash": "791c4f75e6964d7102f1414629b465110729143e1f6efdf1b37721ed7259a2db",
  "scheduler_sha256": "f74bb061c8d1d52cc869135e00082b35a3dc9e09ced158ed54fd72068b61e2db",
  "readers_sha256": "455652b3bf43d83fb517a1a8c459a37fcac942ae4e89d8fe2f1184208f5edb36"
}
```

专用授权字段名：`verdict`、`blocking_issues`、`service_pid`、`config_hash`、`workspace`、`artifact_root`、`target`、`old_phase_code_hash`、`new_phase_code_hash`、`scheduler_sha256`、`readers_sha256`。reviewer还需刷新现有全源码授权和reader reload授权（new reader SHA），之后root只通过正常reload-readers命令安装。
