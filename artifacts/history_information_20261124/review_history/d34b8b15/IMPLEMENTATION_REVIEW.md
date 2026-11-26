# IMPLEMENTATION REVIEW — history_information_20261124

## 1. Verdict

**FAIL.** 当前源码存在冻结 Test 停止门缺口，以及条件算子身份、梯度契约和派发前置验收问题。正式 RUN 应在 coder 修复并取得新的独立通过收据后开始。

- 审核源码 `d34b8b150a2e82686afbfecb5c2726b86556ff816a4befe80cbd30eb07c6a918`
- 设计规范 SHA256 `2c9c802c335d6ca226ecb314b76d934df733187665bfc9b70609d17569a60dd0`

## 2. 逐项核验

| 设计要求 | 核验结果 |
| --- | --- |
| 两个固定自然人口、原切分和行身份、冻结来源及复用 | S0 CLI 重跑成功。当前闭包 182 条实际文件记录全部匹配。ML 和 KR 的 Train/Validation 分别为 800,167/100,021 与 2,006,132/250,767 行。Train-only 编码、行 ID、数据哈希、参数坐标计数及 20 个 HyperGate/AutoIFS 历史发表槽位通过。D0 原表与两份派生表为 6,480/1,050/210 行。Gaussian 源身份缺失时的隔离失败见问题 5。 |
| 主基线和一次完整反传 | B0、B1、H0 确实委托冻结 A0、P00、A21。主矩阵 .99 候选零点走原 A21 config 与 shadow。原 logits hook、固定 24 条条件与带放回抽样、静态 PAIR 比率及步键保留。raw 梯度 detached 后保留原图，完整 loss 反传一次。原生补集由同一 mixed 梯度提交，整体 EMA 最后更新一次。两数据 seed 2022 首尾批的原生精确哈希检查有现成收据与实际对应代码。 |
| C1、C2、C12、S、P、M | 核对了 `b/mixed/h` 来源、唯一 coupled WD、ε 在 sqrt 外、C1 非负凸组合、当前已更新 D 构造 U、C2 保留 Q、局部事件轴偏差修正、S 的 .9、P_C1 双分母、P_C2 mixed LaProp 及 M 自身轨迹双状态与固定两组 FP64 半径。M 的零半径与 FP32 位移诊断保持。条件零点和异常梯度支持仍有偏差，见问题 2 和 3。 |
| D1–D4 及诊断与 live 隔离 | D1 使用真实梯度和分源交叉矩。D2 从先前模型及 post-forward buffers 以原 A21 真步重放，对 online/optimizer/EMA 三哈希做精确核对，随后执行 10 个替代分支。面板行和 pair 流的键、包含率、缺支持和逐分支 EMA 符合设计。独立重跑五项 CPU 机械合同通过，其中真实源 A21 入口的合成锚点通过 1 次参考重放、10 次替代、66 次面板评价及诊断前后哈希守恒。D4 在固定人口上保存 FP32 概率并核对文件与端点，重用同格 32,768 对，区分 selected 和共同 E10/E25/E50。 |
| 五 seed、预算和阶段选择 | RunSpec/队列固定 2022–2026，主训练 E1–E50。完整 Validation Macro AUC 严格优胜，精确并列保留最早 checkpoint，LogLoss 绑定该点。五 seed 配对先逐 seed 运算，缺失不通过门。S2 安全选择、读出优先、trade-off 路由按设计 §8.1 的补充。S、M/P、C12 六对照和 Cstar 的相对门保持。条件派发的前置核验见问题 4。 |
| N、HyperGate、Gaussian、AutoIFS 入口 | 存在 RunSpec、构造、step 和处理队列的实有入口。N 含 D/T/DT 和整个三行域 embedding，排除普通共享 embedding。HyperGate 一次原生 objective 捕获 logits，保留 native auxiliary components 与 Adam 默认 flags。独立固定 ML 256 行检查中 HG_B0 loss、model、native Adam 状态与直接发表源精确相等，HG_B1/HG_H0 真实取得 2,048 对。Gaussian G_B0 loss 与固定原 GA31-R3 helper 精确相等，G_B1/G_H0 同一 clean 图取得 PAIR。AutoIFS 搜索入口成功做一次合成短批双优化器更新，源 rewind/ticket/frozen gate 代码有接入。S0 已有全 20 个发表端点，当前无需 baseline repair。 |
| RAM、真实恢复及资源准入 | 同一个前台 Supervisor 持有内存库，worker 通过内存 IPC，put 后实际解码 get 并检查完整字节 hash、owner 和身份。CPU 合同通过了重复 selected 回读、日志回退、loader/RNG/EMA/native Adam 私有状态的中断恢复、按卡阈值和归档停止作用域。生产实现按当前每卡显存/功耗加主机 RAM 授权，启动握手后重测，实际 OOM 后追加实测完整进程驻留量和 CUDA 物理可用容量。两工作区的当前停止文件进入采样及 epoch 边界。 |
| 全局冻结及一次授权 Test | 产物绑定所有阶段决定、五 seed、两个读出、固定名单、输入哈希和实际 RAM selected 的 model/packet hash。现有 `TEST_POLICY.json` 记录用户一次正式授权。前台 evaluate 在读取合格状态后使用不可变 `TEST_ACCESS.json` 去重，冻结后拒绝新增训练或选参。选点检查与实际 Test 启封之间存在可复现的停令竞态，见问题 1。 |

## 3. 阻断问题

1. **CRITICAL — Test 在 selected 预检期间收到新停令仍会 claim 并进入加载。** `cli.py:489–521` 的 `stop_status` 只在大量 selected RAM 核验前执行，随后直接写一次性访问 claim 并调用数据桥。隔离测试让 stop 在 `_selected` 时变为 `USER_STOPPED`，只观察到一次返回无停令的查询，已创建 claim 并抵达替代加载函数。测试没有读取 holdout。这会耗用授权并允许当前作用域停令生效后打开数据。最小修复是所有 selected/GPU 预检后紧邻一次性 claim 复核，实际启封入口再次核对，补同场景的拒绝断言。
2. **MAJOR — 条件候选零点与 no-PAIR 旁路没有采用父适配器的精确序列化身份。** `protocol.py:140–148` 返回父 arm 名，而 `objectives.py:63–70` 仍将原候选 RunSpec 传给 `HistoryUpdate`，`optimizers.py:211–213,244–253` 将其作为持久 config。实际 256 行对照中 HG_Cstar→HG_H0、G_Cstar→G_H0、Cstar_N→H0_N、Gaussian λ=0→G_B0 的 loss、模型和 EMA 精确相等，四对 optimizer 哈希均不等。设计 §6、05 §10 和 07 §7.1 要求 canonical 父优化器状态精确一致。最小修复为构造规范父 optimizer RunSpec，保留外层科学队列/标签及实际背景目标。
3. **MAJOR — 非零 raw PAIR 与 native None 的矛盾支持被静默跳过。** `optimizers.py:272–278` 对合法可选作用域的 `p.grad is None` 无条件 `continue`。05 §9 要求检测到 raw 非零且 native None 时报告图错误。应在跳过前区分真实 unused 或零 raw 与矛盾非零 raw，同时保持前两种情况不应用 WD、不推进参数时钟。
4. **MAJOR — 条件模块的触发后源验收缺可执行的前置检查。** `kernel_checks.py:151–198` 只覆盖主 H0 和两种候选，`cli.py:580–589,602–615` 没有条件检查或凭证依赖。尤其 06 §5.4 明确要求 Gaussian 两数据首尾批目标、自然行支持与噪声来源核对后再派发。现有 S3d 可在条件 ready 后直接启动完整训练。应先按触发执行最小源 oracle 检查并将通过收据绑定到受影响队列。N/HG/AutoIFS 的验收也应按各自实际触发记录。
5. **MAJOR — Gaussian 来源缺失会暂停整项 S0。** `sources.py:118–172` 把条件 Gaussian 目录混入必需读取闭包，错误直接抛出，成功时又将 `gaussian_source_verified` 固定为真。`cli.py:232–282` 因此无法表达 clean 来源齐备而 Gaussian 源未恢复的状态。当前文件齐备。07 §7.2 仍要求缺失时保留 clean 结论，暂停受影响模块。最小修复是区分清洁主干所需闭包与条件 Gaussian 来源的核验收据，并按既有条件状态值传递失败。

## 4. 本轮更正

重新运行 S0 audit，刷新了变更后 `diagnostics.py` 和 `observers.py` 的闭包记录，随后逐文件重新核对 SHA256。生产代码未修改，以上问题未修复。

## 5. 剩余风险

CUDA 同设备连续与中断重放、实有跨阶段前台 selected 保留、真实并发及实际 OOM 修复仍需在获准时验证。CPU 小合同与入口对照只验所用批次和条件，无法替代两数据的条件首尾批检查及 S1 的四十个锚点。正式队列、正式训练和新 Test 本轮均未启动。

## 6. 运行权限

当前审核结果不批准正式 RUN 或 Test。修复后重新生成 S0 来源清单，绑定新的独立审查结论，再按固定顺序推进。
