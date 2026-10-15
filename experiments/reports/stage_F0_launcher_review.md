# F0 search launcher 独立 REVIEW REPORT

日期：2026-10-12。范围：仅复审 launcher 控制修复、shell wrapper、控制文档及 focused tests；不重新审核研究算法。

## Verdict

**PASS。** 未发现 launcher 修复的 blocking 缺陷。旧版本审查中的 detached trajectory child、控制/训练进程分离、控制进程未计槽及启动信号竞争窗口均由单进程控制模型消除。本结论不修改或自动释放 run gate。

## 检查对象与冻结身份

以下 SHA-256 由 Reviewer 只读核对，均与 Coder 提供值一致：

| 文件 | SHA-256 |
|---|---|
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `21e05692bbe6b43ea68c6e14f6797d9f7f277d6cc7459c2d467fe16a4beb278e` |
| `experiments/stage_F0_outputs/runs/WORKER_CONTROL.md` | `e018fe711fe5033bd061ba1d2c890c52075e402875b75d1817e42ce076fcbdbc` |
| `experiments/tests/test_stage_F0_launcher.py` | `778d082f9260d6b6802cacecf3c106a5c1ee97264419323e2c0dc9dbfd7229d4` |

同时只读检查 `experiments/stage_F0_outputs/runs/run_search_worker.sh`、既有 runner 的 main/cleanup、process-slot及GPU/canonical lock路径。

## 逐项发现

- **进程所有权符合。** wrapper用`exec`替换shell；launcher在同一Python进程内调用既有runner。worker PID即PGID，没有可独立继续训练的trajectory child。
- **SIGSTOP语义符合。** 对该worker PID或整个worker进程组发送STOP会同时停止controller与训练。只停止无关的外层祖先进程并不等同于停止worker；控制目标须采用日志中的worker PID/PGID。
- **INT/TERM/HUP清理路径符合。** handler抛出KeyboardInterrupt，in-process runner异常/清理路径释放canonical/GPU locks，launcher结束并释放worker-ID/process-slot locks。该模型没有child wait，旧child等待超时问题不再适用。
- **共享进程计槽符合。** controller先非阻塞占一个共享五槽，runner执行时另占一个槽，属于保守计数。worker-ID flock仅接受0/1，阻止同ID重复启动；既有GPU lock保证每卡最多一个训练worker。
- **总上限仍需启动端执行。** 所有同时存活的本任务Python进程，包括监控、测试及Conda控制进程，均计入5上限；不得启动过量等待槽的进程。两个controller最多对应两个GPU training workers。
- **gate/review fail-closed符合。** runner调用前要求gate中的launcher SHA与当前文件完全一致，且launcher-specific reviewer verdict为PASS/PASS WITH FIXES。审查时旧gate的hash仍是旧版本，因而新launcher会被阻止；须由主控随后更新并冻结。
- **失败后不自动重试符合。** GPU contention和其他runner错误均记录并退出，重新启动前需检查lock和canonical状态，未改变原研究轨迹的恢复规则。

## 验证与证据边界

Coder报告py_compile、bash -n及focused pytest **3 passed**。Reviewer独立阅读实现和三项测试，确认覆盖进程组self-test、两worker-ID独占锁、stale hash及failed/missing review拒绝。本轮按主控要求仅只读，未启动Python、未重跑tests、未发送信号、未运行实验。

focused tests没有实际执行活进程STOP/CONT/TERM测试，也没有模拟两个controller与其他Python同时竞争共享槽。源码支持上述控制语义，但不能把这三项tests写成已经验证实际GPU信号行为。Python信号handler可能等待正在执行的native/CUDA调用返回；需要强制结束时，由外部控制端终止同一worker PID/PGID，并核对退出与陈旧claim后再恢复，不存在本launcher创建的独立训练child。

## Corrections made 与 RUN边界

Reviewer未修改代码、gate或manifest；仅新增本审查报告。launcher-specific实施门可以登记为**PASS**。重新启动须先把本报告、精确launcher及控制工件身份绑定到新冻结gate/manifest，并保持既有数据、GPU/存储、实验恢复及formal/test阶段门。
