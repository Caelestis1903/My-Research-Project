# GATS02 恢复设计附录 R2

状态：DESIGN 完成，待协调者批准后交 coder；不包含算法实现或新的研究结论。本文仅修复已触发的历史数值基线接口故障及恢复执行所需的计时/身份衔接；原 [DESIGN SPECIFICATION](DESIGN_SPECIFICATION.md) 的数据、算法、seed、模型、优化器、50 epoch、batch40960、FP32、shadow、选点、门槛和候选预算全部保持。

## 1. 已发生的故障与真实接口

`attempt_20261102_0005` 在20条基础唯一科学轨迹完成后、匹配控制与统一测试之前，在 `historical_baseline_verification` 读取 `data[d]['encoder_hash']` 抛出 KeyError，协调进程退出。历史及本轮实际 `data_manifest.json` 中，这两个值均位于 `data[d]['features']['encoder_hash']` / `['scaler_hash']`；原始 CSV/数组/行指纹等仍位于 dataset 顶层。协议 `p['datasets'][d]` 的 encoder/scaler 则位于 dataset 顶层。不能把缺失键当相等，不能删除该身份检查。

已只读核对两侧真实文件：

- 历史/current resolved_config 的 `training`、`model`、`optimizer`、`rng` 四字典实际完全一致。
- 历史 frozen 的 `configurations[d]['A0'/'A1']['config']`、`kind`、`shadow_rates[kind]` 及现有逐端点查找所需列均存在。
- 从本轮已完成的 state_summary 提取 A0/A1 × online/fixed × selected/E10/E50 的48个端点，以现有映射查历史 validation/development 记录，48/48 的模型 hash 与 epoch 均相同。此为已落盘身份核对，不恢复任何权重、不增加训练或打开测试。

退出后RAM模型不可恢复。现有20条科学轨迹的36个完整物理pass与开发数值仍有效保留，但不能据此声称已统一测试或已完成全队列。

## 2. coder 的最小接口修复

1. 把历史 manifest 的 encoder/scaler 读取改为真实 `features` 路径。保留数组、行号等顶层比较；同时核对合同已有的 raw SHA、split row fingerprints、train domain counts，不以 raw_path 文本相同代替数据身份。历史原始路径仅为来源记录，不要求历史路径现在存在。
2. 以历史 `resolved_config.json` 中的 `training/model/optimizer/rng` 与本轮协议对应字典核对固定训练/模型/RNG身份；保留现有逐臂 lr/wd/betas/eps/lambda、A1 noise_scale、所选固定rho核对。不要比较 study_id/source_manifest/执行commit等预期不同的实验标识，也不要将A0未生效的noise_scale差异误判为训练差异。
3. 历史身份缺失或不匹配应按原合同返回该数据集 `verified=false` 并写明确原因，使后续走新增配对基线预算路径。不得静默接受缺失字段；不得把历史不匹配升级成无限重试或改变算法。真实schema解析错误必须能在下面的前置核验中显露，不能再次留到20条训练后才首次执行。
4. 将历史逐端点核对抽为可接收**扁平数值端点记录**的共享核心；当前RAM结果的wrapper仍由原 `checkpoint_rows` 生成记录后调用它。共享核心不要求model/optimizer tensors。现有历史映射保持：A0 online取A0、固定EMA取A0_EMA；A1固定取A1_EMA；缺少开发A1 online时取development S2 G1(ML)/G2(KR)，A0可回退S1 C。kind由 frozen 配置映射。每端点必须唯一命中并且epoch/model_hash一致。
5. 提供一个可复现的CPU调用入口/命令，从旧attempt的完整state_summary提取这48个实际数值端点调用同一核心，并落盘 preflight receipt。不得伪造RAM状态、mock掉实际核心或以手写“预期PASS”替代执行。全新恢复attempt正式训练前必须实际完成此预检。预检不能读取 test_metrics.csv；历史test只保留到最终统一测试阶段导入。
6. 正常队列完成基础训练后，仍使用新attempt的**实际RAM端点**重做历史身份核验；不能直接沿用预检结论。这同时覆盖同一核心的两种输入适配。

以上改动只允许在新实验包内相关接口、必要回归测试和文档。不要修复未触发的其他方向，不改变 objectives/shadows 的研究数学。

## 3. 原始时间预算必须延续

原始首条正式训练 launch 时间为 `1793549242.1426222`，即 `2026-11-01 16:07:22.142622 UTC`。恢复必须接受/记录这个绝对任务起点（建议 CLI `--task-start-unix`），并同时用于全部派发判断：

- 10h 停止低优先级新派发：`2026-11-02 02:07:22.142622 UTC`。
- 12h 停止所有新任务：`2026-11-02 04:07:22.142622 UTC`；完成在跑任务、冻结与汇总。

当前 `all_stages` 的相对elapsed比较及 `schedule` 的绝对 `dispatch_deadline` 必须指向同一个原始起点。可以由原始UNIX起点换算 perf_counter 的虚拟起点，不得在重启时把10/12h重新计满。此为恢复同一任务，不新增一轮授权预算。receipt同时记录原始起点、恢复起点、已用时间与deadline。

吞吐/超预估判定不能忘掉已消耗的计算。恢复运行计入前一attempt的已完成worker秒；使用原始吞吐预测作为预算参考，并明确旧成本与本轮重新执行成本分列。该计算只控制预登记的取消附加trial/第二确认等顺序，不能改变主比较门槛。若重算耗时逼近绝对deadline，按原合同停止派发并准确报告完成范围，不扩展时间。

## 4. fresh attempt 的恢复顺序

1. 保留旧attempt原样：失败 completion、全部primary/replay日志、原始指标、mask概率/能量、投影/rho/行序hash和外层错误日志。协调层故障另记；空worker failure ledger不能被写成整轮无失败。首启wrapper错误若原始字节已被覆盖，只能记录可证实的摘要及缺失事实，不重建成声称原始的traceback。
2. DESIGN附录批准 → coder最小修复/实现报告 → **不同代理**独立review PASS（或blocking已修正的PASS WITH FIXES）。先实际运行CPU真实历史预检与必要回归测试。
3. 冻结新的不可变执行commit/source manifest，另存恢复执行收据；保留旧ac2ec5c执行收据，不覆盖其身份。新源码hash与旧Stage0不相同，须重新进行两数据集真实CUDA Stage0，取得与新执行身份一致的PASS收据。
4. 使用全新空run-dir重算原20条基础科学轨迹，重建全部selected/E10/E50 RAM状态。无磁盘快照、无数值cache冒充模型、无直接跳过“失败”候选、无按旧最佳epoch提前终止。
5. 检查每个恢复后的基础轨迹与旧同(dataset,arm,seed,有效config)的初始化、逐step online模型hash、行序/view hash、最终optimizer hash、独立selected epoch和端点指标是否一致。保留跨attempt身份收据。只有这些成立，才将它归为相同科学轨迹的重算。若不一致，不挑较好结果、不合并成新seed，停止受影响晋级并交协调者/独立review判定。
6. 基础队列之后继续原注册控制筛选、有限trial、C1/C2确认裁决、必要配对基线、全清单冻结、统一测试与释放RAM。预检/旧验证结果均不得使候选门槛放宽。最终统一测试仅使用恢复attempt实际在RAM中的模型。

## 5. 唯一科学轨迹与物理成本

科学去重键为 dataset + arm/有效完整配置 + seed + 已核验的数据/模型/随机协议；本次接口/预算修复不改变上述研究算法。旧20条与恢复20条在第4节一致性核验成立时仍是20条唯一科学轨迹，不能称40个独立实验，更不能提高n。run_id/source_hash/执行commit不同应完整保留为两个执行attempt，建立显式映射。

原20条含A0四个单pass、其余16条各primary+replay，共36个完整物理pass。恢复重算至少再增加36 pass；匹配控制/确认另按真实执行增加。所有旧/新训练曝光、forward、梯度遍历、更新、验证、GPU event、worker秒和峰值分别保留并在总成本相加；10/12h墙钟采用原始起点。最终科学唯一轨迹仍≤40，附加≤8，新增确认含新基线≤12；科学预算不因故障重置。

旧mask-only能量记录缺口仍按 [诊断复审](DIAGNOSTIC_ACCOUNTING_REVIEW.md) 在展示中标N/A，不在本次恢复中顺便更改研究或测量通道。

## 6. 实现与独立审查验收

- 真实历史预检：两dataset、48个端点全部核验，receipt给出逐端点匹配、schema字段来源、固定配置比较及 `historical_test_accessed_for_decision=false`；依据当前实际来源，应得到两dataset verified=true。
- 故障回归：针对真实嵌套features路径验证；至少人为去掉一个必需identity键或改一个模型hash，结果必须拒绝该数据集复用并给原因，不抛出训练后才发现的KeyError，也不能把缺失值同时None判为相同。
- 对原始起点的10h/12h边界进行纯数值检查，证明恢复起点变化不延长deadline；确认新增基线计入12和总40的既有回归仍通过。
- 对被改的共享核心与紧接其后的 frozen config/端点列读取，使用实际来源执行，不以伪造schema作为唯一测试。reviewer审实际代码、真实预检receipt与coder报告；不需要重新设计或审查无变化的模型算法。
- 新source身份CUDA Stage0 PASS后才启动fresh完整训练；runner仅原样执行和记录，不解释研究效用。
- 最终报告明确这次协调器故障、RAM丢失、完整重算成本、跨attempt一致性、旧attempt没有统一test，以及完成/预算停止的实际状态。不能保留“本轮全程零失败”的旧措辞。
