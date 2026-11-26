# CODER IMPLEMENTATION REPORT — 机械运行

按获批 [DESIGN SPECIFICATION](DESIGN_SPECIFICATION.md) §6.1 的机械运行职责实现。前台 `Supervisor` 接收已冻结的 `RunSpec` 队列，保持一个 RAM owner generation，经进程内存 IPC 管理同设备 worker。`run(queue)` 输出阶段收据与原始文件，`selected_get(job_key, readout)` 返回实际重算的模型哈希、身份与 RAM packet 哈希。`release_selected(retain)` 和 `close()` 管理状态保存期。

## 修改文件

| 文件 | 变更 |
| --- | --- |
| `runner.py` | 按 epoch 原子提交，完整 Validation 独立选点并保留严格最早并列。单批来源、局部时钟、成本、六格端点和两读出概率记录交给编排分析。真实故障重放从最近已验收 packet 恢复，同源、同 GPU。epoch 的原始日志未完成时，回退本次追加，沿用上一份已验收 RAM 状态。 |
| `state.py` | 管理 epoch、post-backward/pre-step anchor 及 selected 三种 RAM packet。每次写后从实际 RAM readback 解码，核对写入哈希、owner、身份与游标。selected 离开 complete packet 后独立保留。 |
| `scheduler.py` | 分卡按当前显存、功率与宿主 RAM 三个严格门准入，实际分配 ACK 后重新采样。真实 OOM 后增加 CUDA usable 容量与全进程驻留检查；首次见到的工作负载可在观测到无计算进程且拥有完整可用 CUDA 容量的卡上监督实测。登记两工作区当前作用域的 `USER_STOPPED`、`PAUSE_TRAINING` 和归档中的 cycle04/05 控制。 |
| `observers.py` | 按活跃 PAIR 和 native 时钟维护同轨迹 (V_{mix},V_{sep},Q,U)。记录两个组的 BCE/WD cross、重构误差、预条件交换差和位移。 |
| `diagnostics.py` | 固定分层面板和确定性成对索引。D1 使用更新前独立 eval clone，D2 重用 A21 真步重放与共享 FP64 半径运算完成 11 分支。D4 使用相同固定 pair 规则，B1/H0/S2 留存共同 E10/E25/E50 和 selected 的完整 Validation 概率及身份。 |
| `tests/test_mechanical_contracts.py` | 五项小型工程合同检查，含真实源 A21 更新入口的一次合成张量锚点重放。 |

阶段原始产物位于 `artifacts/history_information_20261124/<stage>/`。核心文件为 `RUN_RECEIPT.json`、`runs.csv`、`curves.csv`、`endpoints.csv`、`steps.jsonl`、`diagnostics.jsonl`、`costs.csv` 和 `state_receipts.jsonl`。`predictions.jsonl` 记录实际概率文件和哈希。CUDA 尚无合格驻留证据时，新工作负载等待可测资源，诊断或 optimizer 状态张量仅经 RAM IPC 转移。

## 实施检查

使用指定项目 Python `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

- 五个生产模块及合同测试通过 `py_compile`。
- `python -m pytest -q -s experiments/history_information_20261124/tests/test_mechanical_contracts.py` 通过 5 项。验证逐卡阈值和物理 fit 边界、活动与归档停止控制、跨重复调用 selected 回读、失败日志回退、CPU 同源恢复的 optimizer/EMA/loader/RNG 一致性、C2 首事件及历时算术。
- H0 锚点来自 60 行合成两任务输入、三个域及真实源 A21 `train_step`。完成 1 次参考步、10 次虚拟步、66 次面板评价和 6 次探针求导。三类 live 后态的哈希逐字匹配，诊断前后 live 与 RNG/loader 哈希相同。合成输入的 cross 重构最大绝对误差为 `3.9968028886505635e-15`。

| 锚点证据 | SHA256 |
| --- | --- |
| 实际回读 anchor | `a902873874d991ad84a1e043ca12b34495d1d92401676441bbd1945df4b75ed8` |
| 后态 online | `f1d535560bd577c958d3df916eeff282727ce7b17d3df14ff2a9e29bbba4e31c` |
| 后态 optimizer | `1b240f7211c0ede09ba8844c95ebb7a31d9ca7f386d203da23079c2ceb4420d3` |
| 后态 EMA | `02bcbb697cdad0d4c745faa5a0787f8be14f4ce16ace623c5989f71a66593977` |

## 独立审查交接

- CUDA 并发、现场实际 OOM 与 S1 的 40 个锚点由独立审查和依序获准的 RUN 核验。此报告证据来自语法检查和 CPU 合同检查，正式轨迹和 Test 评价尚未启动。
- S0 报告本轮父轨迹无可用旧 RAM。当前队列由新前台 owner 创建；如果队列引用外部 owner 的既存父状态，Supervisor 会提示需要经实际回读核验的 owner 交接。
- 新工作负载在实际 OOM 之后的首个观测由完整 CUDA usable 容量的卡承接。后续按同数据、配置、来源和 scope 的实测全进程驻留准入，并继续遵守原 85/85/90 条件和已冻结批次。
