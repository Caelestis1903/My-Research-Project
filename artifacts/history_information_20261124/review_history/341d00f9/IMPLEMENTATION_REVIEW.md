# IMPLEMENTATION REVIEW — history_information_20261124

## 1. Verdict

**FAIL.** 前轮三项阻断已修复，条件队列已接入源验收。Gaussian 来源隔离仍会阻断 clean 路径。当前冻结来源文件均齐备。

- 主干来源 SHA256 `341d00f92ccdc414d620d633a5fec13555f9e746b652e84445315abdc837d7b6`
- Gaussian 来源 SHA256 `fae0dc51d0ba7114045260cec3949547ea8627c869293f8add48354bd0da7483`
- 设计 SHA256 `2c9c802c335d6ca226ecb314b76d934df733187665bfc9b70609d17569a60dd0`

## 2. 逐项核验

| 前轮审查要求 | 当前结果 |
| --- | --- |
| Test 停控、预检与实际加载 | 通过。两工作区真实停止文件夹具重跑通过。在 selected RAM 预检中写入 `USER_STOPPED`，未创建一次性 claim、未读取发表 Test 端点、未进入新 Test loader。在 claim 后写入停令，评价失败收据记录停令，未进入 loader。独立夹具在真实 loader 的人口预检中写入另一工作区 `PAUSE_TRAINING`，随后拒绝并保持编码入口调用次数为零。 |
| 条件父旁路的精确身份 | 通过。`optimizer_spec` 规范化优化器配置，外层 `RunSpec` 保留各自的 job key。用冻结 ML seed 2022 Train 输入的 256 行，在真实 `build_training` 和 `step` 对照 `Cstar_N→H0_N`、`HG_Cstar→HG_H0`、`G_Cstar→G_H0`、`HG_H0(λ=0)→HG_B0`、`G_B1(λ=0)→G_B0`，每对模型、optimizer、EMA 哈希、loss 和 PAIR 身份精确相同。 |
| native None 与 raw 支持 | 通过。活动 PAIR 的矛盾非零 raw 在提交前被拒绝，模型和 optimizer 哈希守恒。合法 None/零、活动零梯度与无 PAIR 的原时钟契约通过内核检查。FP64 方向检查最大误差 `7.771561172376096e−16`。 |
| 条件源验收实际进入队列 | 验收接入通过。实际 CLI `execute(run)` 在 selected 释放和 `Supervisor.run` 前执行条件 oracle。结构化队列夹具验证按实际 arm 提取模块，匹配 study/stage/queue/source、校验 Gaussian 文件身份、拒绝错绑和复用合格收据。N、HyperGate、Gaussian、AutoIFS 分别有自然 Train 来源入口。实际四模块 oracle 随科学触发调用，Gaussian 故障隔离见下。 |
| Gaussian 可选来源与当前冻结身份 | 部分通过。当前 S0 收据为 `COMPLETE`，20 个历史发表槽位匹配。独立逐条核对 175 个主干与 7 个 Gaussian 文件的 SHA256、Git blob 和闭包哈希。两种专用目标缺件已隔离，但固定 Gaussian 的 views/protocol 缺件会阻断主干。 |
| 继承的主要科学与事务验收 | 复用归档独立审查的主矩阵数学、原 A0/P00/A21、两数据首尾批、诊断、选点、状态及资源证据。当前回归重跑了与前轮阻断直接有关的真实入口。 |

## 3. 阻断问题与最小修法

1. **MAJOR — S0 对两份 GA31-R3 噪声源仍按全局必需处理。** `sources.py:144–172` 仅把目标文件 `objectives.py` 移出主干。模拟发表闭包内 `views.py` 或 `protocol.py` 缺失，`audit_sources()` 逐项抛 `FileNotFoundError`。此处使用的是设计 07 §7.2 要求的固定噪声模块。`objectives.py:17–19` 以及发表原生 `state/evaluation` 经历史 protocol 的导入链会在 S0 导入阶段先遇到相同模块。

   最小恢复路径是复用已有冻结副本。Git 已核对主干 358d3a4 与发表 c8675ae 各自路径下 `views.py` 同为 blob `ecc354eba1c1d1e8b21694e8dbfb3c2b7221d254`，`protocol.py` 同为 `7e59458c3473717f9c07e48dca984e2c19380bd7`。需要在 S0 导入依赖这些文件的模块前，按相同 blob 恢复确实缺失的副本，再核对实际载入文件与清单。保留两处原 Git 来源和实际使用路径。该方式无需改旧算法或重新训练。独立于共享导入的 D0 coverage 和通用 Gaussian objectives 缺件仍沿现有 `UNRESOLVED_IDENTITY`。
2. **MAJOR — Gaussian 专用来源技术暂停会阻断已完成的 clean 分析。** 在 S0 已标记 `UNRESOLVED_IDENTITY` 的情形，S3d 可仅派发 N/HG，`analysis.py:143–147` 却在读取其合格收据前对任何 `TECHNICAL_PAUSED` 臂抛错。S0 后发生 Gaussian oracle 或文件身份失败时，`cli.py:302–343,663–674` 在派发前让整 S3d 返回暂停。这和 06 §5.4、07 §7.2 的仅暂停 Gaussian 并先完成 clean 结论不符。

   首先按冻结源恢复可恢复专用文件后重试同一 queue hash。对仍未恢复的 Gaussian，沿既有 `TECHNICAL_PAUSED` 状态接受并出具已验收 N/HG 的结果，失败状态须带模块、源和原队列身份。需要保持仍未完成的 Gaussian 不进入全局冻结或 Test。无需重选配置或变更五 seed、模型和比较。

## 4. 更正

更新独立审查报告和 verdict，绑定当前源码和设计。生产代码未编辑。

## 5. 剩余验证

条件 oracle 按触发执行。CUDA 同设备恢复、现场资源派发和完整 S1 锚点属于获准后的实有运行验证。本轮进行了 CPU 机械检查和短批源对照，未启动正式训练或访问真实 Test。

## 6. 运行权限

当前源码的独立审查结论为 FAIL。待纠正两项阻断并取得合格复审后开始正式 RUN。
