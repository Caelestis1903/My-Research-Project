# CODER REVISION REPORT — 编排门修复

本次修复独立审查 `d34b8b15` 的阻断项 1 和 4，覆盖 Test 启封停控和条件模块派发验收。

## 修改

- `cli.py` 在全部 selected RAM get/hash、owner 和 GPU 预检完成后，紧邻一次性 `TEST_ACCESS.json` claim 重新读取两个已登记工作区的当前停控。claim 后进入评价前及调用每个数据集的 Test loader 前也重新核对。停令在 claim 前到达时不会消耗 claim。停令在 claim 后到达时保留既有一次性 claim 语义，在 `TEST_RECEIPT.json` 记录被阻止的评价。
- `selection.py` 从已实例化的 `QUEUE.json` job arm 识别 N、HyperGate、Gaussian 和 AutoIFS。`REPAIR_HG` 需要 HyperGate oracle，`REPAIR_AUTOIFS` 需要 AutoIFS oracle。纯主矩阵和空队列直接通过该门。
- `cli.py` 在前台 `run` 释放 selected 或调用 `Supervisor.run` 前执行已触发模块的 `kernel_checks.conditional_contracts(module, root)`。各模块通过才写入不可变的 `<stage>/CONDITIONAL_CHECKS.json`。复用时核对 study、stage、queue hash、source hash、实际触发模块集合和各模块结果。Gaussian 额外核对已审计的可选源 hash 和闭包文件的当前 SHA256。失败写 `<stage>/CONDITIONAL_CHECKS_FAILURE.json`，记录阻断模块并返回 `TECHNICAL_PAUSED`。验收完成后重新读取停控，再进入派发路径。

## 检查

- 项目 Python `py_compile` 通过。`python -m experiments.history_information_20261124.tests.orchestration_check` 返回 `PASS`。
- 在两个隔离工作区登记真实停控路径。模拟 selected RAM 预检期间另一工作区写入 `USER_STOPPED`，确认 `PermissionError`、零 Test claim、零已发表 Test 端点读取调用和零 Test loader 调用。模拟 claim 后、loader 前的停令，确认 loader 无调用且失败收据记录停令。使用同一临时收据调用内核组新增真实数据桥停控门，确认触发停令时 `_population` 无调用。
- 以已实例化的合成 S3d 队列和替身 oracle 验证 N、HyperGate、Gaussian 按实际 job 触发。一个 oracle 失败时，未释放 selected、未调用 `Supervisor.run`、未写 PASS 收据。成功重试写入 source/queue 及逐模块凭证，后续同身份命令复用凭证。篡改 Gaussian 文件字节或凭证 queue hash 时，派发暂停。独立检查了两个发表修复臂的模块映射和主 S1 无条件模块。
- 主矩阵两数据 seed 2022 的真实首/尾批技术验收入口为 `python -m experiments.history_information_20261124.kernel_checks --real`。既有 [训练内核检查收据](CODER_IMPLEMENTATION_REPORT_KERNEL.md) 记录了四批的 PAIR hash 及与原 A21 精确匹配的 model、optimizer、EMA hash。本次编排修复沿用该证据。

## 复审交接

`cli.py` SHA256 为 `7ea8888d2d03f7aa46b49ed2cd58439f8f46ec55eff324a7a550a99579a1d4ad`，`selection.py` 为 `844a112bcd5f446bebece7ca962b9207860850d48c9b5b43bb5dac44b0a8464b`。本轮源码汇合后应刷新 S0 源清单供独立复审。真实条件 oracle 依实际正式队列触发。本次检查使用临时合成队列和停控夹具，正式训练、真实 Test 读取和新 Test claim 均为零。
