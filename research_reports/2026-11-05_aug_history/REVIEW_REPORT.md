# INDEPENDENT REVIEW REPORT

Final verdict: **PASS WITH FIXES**。

首次审查 verdict 为 FAIL；下列三个生命周期问题已由独立 coder 修复，并于 2026-11-05 10:22:58 UTC 完成定向复核。Blocking issues：**无**。首次发现和修订记录保留如下。

审查对象：`DESIGN_SPECIFICATION.md`、`CODER_IMPLEMENTATION_REPORT.md` 和 `experiments/aug_history_20261105/` 实际实现。冻结计划为 `2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，父训练源码为 `c8675ae2923c15b6965884b140b5b646690cd95e`。本轮只审查 B 与必要公共运行流程。

## 首次 FAIL：已修复的 blocking issues

1. **失败后不能按合同恢复 RAM 状态。** `run.py` 的 `train` 不接收 `resume`，main 调用也不传入已解析的 `--resume`。`execute_job` 失败后 guardian 将任务标为 `ENGINEERING_BLOCKED`；调度器遇到该状态直接报错，guardian 无 retry/reset 命令。即使完整 epoch RAM 状态仍在，`train --resume` 也不能继续。需让显式恢复在原 owner 已退出时一次性重新排队，保留已提交状态与失败记录。

2. **协调器重启会重复启动仍存活的 worker 并清除其 owner。** `run.py` 的 `train` 每次从空 `running={}` 开始；ready 条件只排除本地 running 和 TRAIN_COMPLETE，未排除 guardian 已登记的存活 worker。重启后的协调器因此会重复派发同一 job；新 worker 的 register 被拒绝后，异常处理仍调用 guardian `failed`，把原 worker 的 owner 清空并标失败。需复用/等待 guardian 的存活 owner，失败上报须限于成功登记的实际 owner。

3. **被拒绝的 replay 会替换 RAM 中已提交状态。** `guardian.py` 的 commit 在调用 `persist_epoch` 校验既有标量记录前，先更新 `state/record/commit_index/record_hash`。若重放 hash 冲突，标量被拒绝，但 RAM 已安装冲突快照；随后恢复可能从冲突 epoch 之后继续。需在替换前完成冲突校验，或在持久化失败时恢复原 RAM 状态，保持 RAM 先于标量发布的提交顺序。

以上问题影响规定的恢复、所有权和实验结果身份。Coder 按审查限定修复并增加一个聚焦生命周期验收，未改变 B 数学、候选或实验矩阵。

## 已独立核对的实现

| 范围 | 静态审查结果 |
|---|---|
| 源模型与 loss | 实际 dataset 传给源 loss/RNG；源 total 运算顺序保留；两个增强 cell 共用前向；实际 2N 分母 |
| 只读 Adam | θ/m/v/逐参数 step 深复制；固定正式 active mask，区分 None 与零；四次 functional 普通 Adam 预测返回实际 FP32 位移 |
| 响应与 EMA | FP64 非零列归一化/Gram/秩；K10/K20 从 step1 刷新；固定独立符号；旧 innovation；15 个独立 φ、每批一次更新；online 只执行一次正式 Adam |
| 数据与对照 | raw、缓存字节/数组/行号/人口与预处理身份校验；发表模型固定 CSV blob；训练 bundle 不加载 test |
| 选点与 HPO | 每 epoch 完整验证 online+15影子；最早严格最大六格 AUC；所有候选五 seed 完整门；数据集级家族选择、逐格保护、同 h/K 控制 |
| 冻结与调度 | 冻结选点 hash 后 test；validation 顺序决策；历史 KR E50 online 对比有固定来源；A/C 明确未实现并输出既定继续命令 |
| 磁盘与资源 | 参数状态经 BytesIO/IPC 留 RAM；磁盘只写标量/日志；逐卡显存、功耗和 RAM 门；原 batch GPU smoke 在 RUN 阶段执行 |

实施报告记录首次 10 项针对性 CPU 测试、CLI help 和 compileall 已通过；其数学路径在本次修订中保持不变。本次未启动正式训练或性能分析。

## 修订复核与验收

| 首次问题 | 修订与独立复核结果 |
|---|---|
| 失败后无法恢复 | `--resume` 已传入 `train`；`ready_jobs` 对失败/死亡 owner 使用 guardian `retry`，每 job 每次调用至多重新开放一次。`retry_job` 保留原 RAM state/index，并记录重试；再次失败结束当前调用 |
| 存活 owner 被重复派发/清除 | `ready_jobs` 排除 guardian 的存活 worker；`execute_job` 仅成功登记后上报失败；注册拒绝写独立日志；`fail_job` 核对当前 owner 的进程身份 |
| 冲突快照先替换 RAM | `commit_epoch` 保持 RAM 先于标量发布；`persist_epoch` 失败时恢复原 row。冲突 replay 后原 snapshot/index/hash 不变，合法重放和相同提交可幂等通过 |

独立执行：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/aug_history_20261105/tests/test_b.py -q -k 'resume_ownership or ram_export_restore'
2 passed, 9 deselected, 1 warning in 8.49s
```

该检查覆盖 RAM 恢复一致、显式且单次恢复、存活 owner 排除、注册失败隔离、非 owner 失败拒绝、冲突事务回滚和合法幂等提交。唯一 warning 为父源码沿用的 TF32 setter 弃用提示。

已签发 `REVIEW_ACCEPTANCE.json`：

- source hash：`9baac157d4574ccc2ec02a9db42cd365ec918346c32b1b3ace2dcc6f65893f2a`。
- manifest SHA-256：`76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`。
- 范围：B 与当前公共运行流程；A/C 为 DESIGN_COMPLETE_NOT_IMPLEMENTED。

可进入 RUN：先记录 implementation commit、执行 preflight、完整 B 工程 validate 与两数据原 batch40960 GPU smoke；通过现有运行门后启动正式矩阵。GPU smoke、正式训练和结果分析由后续阶段完成。
