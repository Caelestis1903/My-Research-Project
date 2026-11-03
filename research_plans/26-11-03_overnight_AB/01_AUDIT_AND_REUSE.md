# 01 审计结论、必须修正项与复用判定

## 1. 审计证据及边界

本规划真实读取了原方案入口和main三份权威上下文，并审查以下源代码/设计。只读取GitHub可见文件；服务器内存、GPU进程和未推送结果必须由执行时inventory查询，不能在本文件虚构。

| 固定来源 | 内容 |
|---|---|
| main `92a592418d3f11a588f1e628fbe9364ceab71e18` | 最新强制规则、总台账、`research_plans/26-11-03_v2/DESIGN_SPECIFICATION.md`及`PHASE2_PUBLISHED_MODEL_PROTOCOL.md` |
| 同main | `experiments/mdmt_history_information_20261103/{protocol,optimizers,objectives,run}.py`、其嵌套vendor数据/评价/Gaussian目标 |
| 同main | `experiments/conditional_information_momentum_20261102_v3/{optimizers,objectives,boundary_sampler,guardian}.py` |
| 同main | `experiments/mdmt_five_seed_completion_20261103/README.md`、补评CODER_IMPLEMENTATION_REPORT.md和报告目录清单 |
| baseline `d2fdd272a2cdf7da626e29b97f61555d25271f89` | `experiments/mdmt_sota_baselines_2526/{config,default_config,models,training,hypergate,autoifs}`；FINAL_REPORT.md；analysis/VALIDATION_STAGE.csv |
| 原方案 `759e9abb790bab8a599ae9b303dc575488161952` | 本目录00的旧版；本次只完善该方案，不重写历史实验报告 |

固定算法包完整source hash：history=`7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`；pair=`5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`。新的融合包使用自己的source hash，不冒用原包hash；旧包只读。

当前GitHub补评报告目录只有实现报告、CPU测试、source manifest和审查回执，未发布完整新五seed结果。实现报告声明其写作时尚未启动正式训练；这不是关于服务器当前状态的判断。执行者先读本地最新记录再派发。

## 2. 原执行清单必须改动的地方

| 原清单不足 | 本次确定修正 |
|---|---|
| 将A主要描述为旧LL轨迹重选 | 区分“历史重选附录”和“与当前候选同人口的正式主表”；旧30条主表均不等价 |
| 尚未确定新训练数量与预算 | 主表M3复用10；HG/AI预计新20；继承已制定50最终阶段epoch协议，无今晚超参搜索 |
| 未读取正在补评的精确矩阵 | 接入已有40逻辑轨迹/30新增的补评runner，C/P成为外部依赖，不启动第二份同身份训练 |
| “C_MIX+P01”未规定梯度/状态顺序 | 03给出唯一融合递推，避免双step、双L2和PAIR梯度重复计入一阶矩 |
| C与Gaussian共用kappa易串参数 | 配置分为c_mix.kappa与gaussian.kappa；源函数调用前分别构建原生配置 |
| 先跑A真实训练再整组B可能长时间饥饿 | 单卡按epoch量子A/B交替，双卡优先各一条；无另一条线性能门 |
| “保存checkpoint、恢复”没有解释RAM限制 | 继承RAM guardian；逐seed标量持久化；主机/guardian丢失时仅重放缺项 |
| 旧guardian直接用于新调度会出错 | 旧register最多2次、complete硬编码50；新guardian区分quantum次数与失败次数，并支持AutoIFS60训练epoch |
| 逐seed自动评价与全局等待未落实 | 整个配置/矩阵先全局冻结；每seed选点封存后自动评价，调度器不读取性能值；旧补评仍保持其原全局test冻结 |
| 不能确定Codex改哪些代码 | 新包逐文件/函数契约见02–04；旧包不修改，历史报告不改写 |

## 3. A：逐轨迹的确定性结论

### 3.1 旧30条不能用于本次统一主表

| 项目 | 10/29旧基线 | 本次完整自然人口 |
|---|---|---|
| ML train d0/d1/d2 | 134116 / 273995 / 280032 | 155949 / 318599 / 325619 |
| KR train d0/d1/d2 | 148591 / 1499752 / 76928 | 172782 / 1743898 / 89452 |
| 编码/缩放拟合 | T0 | 完整train |
| M3 | lr=.001，24epoch，选点12–24 | lr=.01，50epoch，选点1–50 |
| HyperGate停止 | LL patience10，max1000 | 固定50epoch，无早停 |
| AutoIFS最终阶段 | reuse LL patience1，max50 | 完整reuse50，无早停 |
| 主选点/指标 | V macro LL；raw-logit AUC | V macro AUC；统一概率评价函数 |

对旧三模型×两数据×2022–2026全部30条，在`A_REUSE_AUDIT.csv`登记`eligible_main=false, reason=TRAIN_POPULATION_AND_PREPROCESS_MISMATCH`。旧权重是否存在并不改变此结论。不能从旧最后checkpoint继续训练并冒充从头运行的新完整自然协议；那会变成新的预训练方案。

A主表中M3使用近期同源A0 online轨迹。其EMA shadow不反馈在线参数，因此允许同一轨迹的online端点作为M3 fixed1适配记录，fixedEMA端点作为内部参照；二者不是两个独立seed。历史采集成本包含EMA开销时如实标记`co_collected_shadow`，不虚构纯online独立计时。

### 3.2 旧轨迹是否只需重新选点：机械分类，不重训历史附录

旧`training.py`保存`selected.pt`和`fixed_final.pt`，AutoIFS另有`rewind_epoch9.pt`；并不保存每epoch模型。`epoch_metrics.jsonl`或run.json的epochs可含完整V曲线；`analysis/VALIDATION_STAGE.csv`只有selected/final两点，不能一般性地从这两点恢复全曲线argmax。

逐旧run执行：

1. 读取完整最终训练阶段V曲线，候选为该历史实际完成阶段的全部epoch；AutoIFS排除search，M3历史附录可检查1–24。记录`window=OBSERVED_LEGACY_TRAJECTORY`及旧LL早停，绝不命名为本次50epoch主协议。
2. 若全部V分数可得，严格最大且最早的epoch及状态能由保存模型、同epoch预测或已签名同epoch test端点恢复，标记`RESELECT_ONLY_LEGACY`。已有准确标量端点即可，不要求为复用再次前向。
3. 仅有V曲线无对应模型/预测/完整test指标，标记`MISSING_SELECTED_STATE_LEGACY`；只存两点而未覆盖候选窗口，标记`INCOMPLETE_V_CURVE_LEGACY`。**本晚不为补历史附录重训任何旧协议轨迹**。
4. 仅完整训练身份一致但新选点模型缺失的“近期协议”记录，才排入主表必要重放；不得拿旧LL端点或最近epoch顶替。

可优先机械核对的七条旧AutoIFS轨迹：ML seeds2022/2024/2025/2026和KR seeds2024/2025/2026。旧表显示其reuse只完成1–2epoch，因此两点覆盖该截短历史轨迹。ML四条及KR2024的V AUC最大为epoch2；KR2025/2026为epoch1。仍须校验对应test端点/六格/模型身份。此处没有声称其可用于完整reuse50主表，也未声称所有旧模型状态现在仍存在。

### 3.3 新主表的训练动作

| 模型×数据 | seeds | 默认动作 | 新训练数 |
|---|---|---|---:|
| M3OE_FIXED1×ML/KR | 2022–2026 | 接入近期A0 online合法selected端点 | 0（10条复用） |
| HYPERGATE×ML/KR | 2022–2026 | 本次完整自然人口重新训练 | 10 |
| AUTOIFS×ML/KR | 2022–2026 | 本次完整自然人口search10＋reuse50 | 10 |

若发现此前已完成的完全同身份新版HG/AI，则转为REUSE_DONE；只是同架构、同seed或同文件名不够。无论如何不得重复启动被其他协调器认领的同身份job。

## 4. B：历史补评与正收益组件接入

主查询路径（HISTORY_ROOT为运行中原仓库根，不是新工作树）：

- `artifacts/mdmt_five_seed_completion_20261103/attempt1/{PREPARED.json,progress.json,launch_ledger.jsonl}`及其中指向的family jobs/runs/guardian。
- `research_reports/2026-11-03_mdmt_five_seed_completion/{endpoints.csv,validation_epochs.csv,RUN_MANIFEST.json,DECISIONS.json}`，存在则读取；不存在不填造。
- history旧条目：`artifacts/mdmt_history_information_20261103/attempt1/runs/*/result.json`、`final_evaluation.json`、attempt*/epoch_*.json及data_manifest.json。
- P01旧条目：`artifacts/conditional_information_momentum_20261102_v3/supplement_20261103/`，及对应报告`supplement/endpoints.csv`。
- A0/Gaussian：`artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720/`下resolved_config.json、test_metrics.csv、validation_metrics.csv、validation_epoch_metrics.csv、state_summary.json及run配置。

| 方法 | 每数据已知历史seed | 每数据外部补评seed | 本轮处理 |
|---|---|---|---|
| C_MIX | 2022 | 2023/2024/2025/2026 | 全部导入/等待，不另派发 |
| P01，lambda_pair=.05 | 2022/2024 | 2023/2025/2026 | 全部导入/等待，不混用lambda=.2 |
| A0 online / fixedEMA | 2022–2026 | 无 | 同轨迹不同估计器，复用 |
| Gaussian A1 online / fixedEMA | 2022–2026 | 无 | 替代2×2所需时复用 |
| C_MIX_P01_SERIAL_V1 | 无 | 无 | 新10条；无需等待C/P五seed性能判定才训练 |

已有桥接审计见`research_plans/26-11-03_v2/DESIGN_REUSE_AUDIT.json`：数据raw/数组/行号/编码/scaler一致，所查24个控制端点模型hash完全相同。因此不重新设计控制组验证实验。执行时只是核对待导入文件checksum和该桥接身份，防止拿错文件。

A1既有五seed提供选择简单替代的依据：ML相对A0 online的总ΔAUC为+.004166933，KR为+.006092690；KR d0t0平均差−.001271286未满足逐格保护。相对A0 fixedEMA的增量分别仅+.000318088/+.001476291。保留总收益及代价，不能宣传其已达到组件增量+.003，更不据此预设组合一定有效。

## 5. 复用身份与外部任务认领

定义三种身份而非仅依赖旧job文件名：

- `trajectory_id`：dataset、seed、原数据/六数组/row_ids/encoder/scaler hash、模型与初始化、完整训练目标/超参/预算/采样/RNG规则、算法源码hash。输出目录、device、attempt、调度量子不是科学身份。
- `selection_id`：trajectory_id＋估计器online/fixedEMA＋rho/初始化/buffer规则＋合法选点阶段/窗口/严格比较规则＋评价函数身份。
- `endpoint_id`：selection_id＋选中epoch/阶段/model_hash＋split/指标语义。

原manifest与本协议字段不同但已存在等价桥接时存`identity_alias`及证据，不伪造原source hash。正式基线之间模型/优化器可不同；同一2×2控制必须除组件开关外完全匹配。

`inventory.py`对每条候选只给以下动作：

- `REUSE_DONE`：训练与选点身份匹配，完整V曲线或可信冻结选点记录、所选V/Test全指标及model_hash齐备；源文件checksum已记录。
- `REUSE_EVAL_ONLY`：所选RAM/磁盘历史模型仍存在，训练完整，缺同语义测试；只预约评价，不重复训练。不向别的任务guardian发送release。
- `WAIT_EXTERNAL`：原任务已声明/准备/正在运行，或原协调器仍控制对应GPU。登记外部owner/PID/boot_id/原jobkey和当前状态，轮询，不启动第二份。
- `REPLAY_MISSING`：确认无活跃owner、无同身份已完成端点/可恢复状态，并有具体缺失原因。已有补评队列中尚未启动的条目仍属其owner，不能擅自抢走；只有该owner退出且放弃/失败记录明确，才能原子认领补齐同一个seed。
- `INCOMPATIBLE`：字段不同，保留独立协议组，按本轮既定配置新训；不把不兼容数据合并进五seed。

执行时输出`A_REUSE_AUDIT.csv`（至少旧30条＋新30条身份）及`B_REUSE_AUDIT.csv`；字段为dataset,method,seed,source_commit,source_path,source_sha256,trajectory_id,selection_id,selected_epoch,model_hash,population_match,preprocess_match,budget_match,metric_match,state_available,external_owner,action,reason。源不可访问明确标记，不杜撰模型存在性。

## 6. 工程代码来源准备

main基准已包含history、pair及补评包；以原hash核对。baseline适配源码从固定commit按文件读取，在新包`vendor/sota2526/`保存`hypergate.py`,`autoifs.py`,`sources.py`和其确实依赖的同目录模块，并保存路径/blob/SHA256清单。不得把旧training/config/run作为新入口直接执行。若来源文件新增依赖，按其import确定性补齐，不寻找新模型替代。

`models.py`顶层会引入旧`mdmt_basp_v1.prefix`，因此新factory直接导入两adapter并复制原build_model相应映射；M3使用当前history.protocol.make_model，避免不必要的旧BASP依赖。旧训练文件仅作为02规定的逐段移植依据。上游源码根缺失时，按原sources.py记录的仓库与固定commit获取；下载失败只阻塞该发表模型，B不受牵连。禁止拿另一版本的上游代码冒充固定版本。
