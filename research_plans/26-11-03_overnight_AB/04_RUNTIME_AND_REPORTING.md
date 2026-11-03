# 04 自动执行队列、停止恢复、验收与报告契约

## 1. 必须新增的工程模块

根包：`experiments/overnight_ab_20261103/`，不增加Python依赖，使用既有phase0-3-env。

| 文件 | 核心接口 |
|---|---|
| `run.py` | argparse子命令prepare/smoke/run/run-job/status/pause/resume/retry/report；守护进程内部子命令_guardian |
| `inventory.py` | `scan(history_root,plan)->Inventory`；01定义的身份映射、旧轨迹审计与外部认领 |
| `config.py` | 读取protocol.json，拒绝未知arm/seed、错误rho/阈值/epoch、参数冲突；生成科学配置hash |
| `queue.py` | `claim_next`, `lease_device`, `dispatch_quantum`, `reconcile`, `poll_external`；只读状态/成本，不读test性能 |
| `guardian.py` | 独立RAM owner，authenticated AF_UNIX IPC；新协议命令见下 |
| `state.py` | versioned Session快照、cpu_copy/state_digest、RNG/阶段恢复，禁止权重落盘 |
| `worker_b.py`、`published_training.py` | 每次最多一个完整epoch的可恢复worker；A/B分别按02/03 |
| `selection.py`、`evaluation.py` | 配置冻结、每seed模型选择冻结、评价与签名标量原子提交 |
| `reporting.py` | 原始端点、五seed配对、2×2交互、成本、决策、Markdown报告和台账增量 |
| `tests/` | 本文有限工程验收；不新增科研筛选阶段 |

接口约定：`JobSpec`含job_id/trajectory_id/dataset/seed/arm/lane/config/source/data身份/needs；`EpochRecord`含stage/epoch/commit_index/validation/model_hash/selection_hash/cost；`SessionState`含可精确恢复的全部张量和随机状态。配置与矩阵字段从protocol.json及plan.py输出读取，不在多个worker散落第二套默认值。

## 2. 固定矩阵与计数

| 集合 | 逻辑训练身份 | 主端点槽位 | 预计新训 |
|---|---:|---:|---:|
| A：M3(alias A0)、HG、AI×两数据×5seed | 30 | 30个online selected | 20 |
| B主2×2：A0/C/P/CP×两数据×5seed | 40 | 40个fixedEMA selected | 10 |
| A＋B去重 | 60 | 70个不同用途端点槽位 | **30** |
| 预定义替代B：A0/C/G/CG×两数据×5seed | 40，与核心重合20 | 40个fixedEMA selected | 激活后10 |

若主组合和替代都被事先授权且执行，联合所有训练身份最多80条，正常新增40条。未激活的G/CG行仍在规划矩阵中，状态`CONDITIONAL_NOT_ENABLED`，不属于核心缺失。缺失控制的必要重放必须单列原因和额外成本，不把预估30误写为不论证据如何都必须重训30。

`plan.py materialize`输出80行去重训练矩阵、30行A发表模型端点映射、80行两套B因子槽位、30条核心预期新训练的种子级交错顺序、配置sha256。输出目录是计划编译产物，不是训练结果；不将其job-id当成完整科学hash。

默认主集合：A0、HYPERGATE、AUTOIFS、C_MIX、P01、C_MIX_P01_SERIAL_V1。替代集合增加GAUSSIAN_A1、C_MIX_GAUSSIAN_V1。补评的H_MIX/M可读取作历史台账背景，不为本次组合增加训练。

## 3. prepare与冻结

`prepare --plan PATH --history-root PATH --attempt PATH`必须幂等：

1. 保存原仓库及工作树路径、main/方案/源commit、三份main上下文checksum、环境信息。旧包hash不符时定位差异，不覆盖正在运行的代码，也不把新hash伪装成旧hash。
2. 运行计划编译与inventory，加载最新本地补评结果。已有合法DONE直接接入；正在运行/已被外部队列认领的条目WAIT_EXTERNAL；旧30条写入独立A审计，不能污染主表。
3. 输出`PLAN_EXPANDED.json`、`JOBS.csv`、`A_REUSE_AUDIT.csv`、`B_REUSE_AUDIT.csv`、`IDENTITY_ALIASES.json`、`SOURCE_LOCK.json`、`DATA_MANIFEST.json`、`ENVIRONMENT.json`。每个身份绑定train/val/test数组和预处理hash、实现hash、原配置及别名证据。
4. 在新训练开始前原子写`CONFIGURATION_FROZEN.json`，含全部主矩阵、替代矩阵及替代启用规则、模型/参数/预算/选点/指标/阈值、配置hash、source hash。不得根据新增test值修改任何一项。性能字段不进入queue决策模块。
5. 第二次prepare发现科学身份不一致则明确拒绝续同attempt，保留旧产物；工程代码修补是否改变数值路径由实现审查判断。改变科学身份须新attempt和协议组，不能把前后结果凑五seed。

数据缓存只允许只读数据数组/manifest（若既有协议允许），不是模型状态；缓存命中必须核对原数据hash。不得调用旧baseline.prepare_owned生成T0∪F。

## 4. GPU与外部补评协调

### 4.1 不抢正在补评的GPU

启动时读取同用户进程的cmdline、cwd、PID start time及boot_id，识别原补评coordinator/worker及其devices；查询其progress、launch ledger和guardian状态。**旧调度器未使用新锁时，不能仅见GPU当下空闲便抢占它即将派发的设备。** 活跃旧coordinator声明的设备保持外部保留，直至它结束或有明确交接记录；不杀、不暂停、不改其源码。未知外部GPU进程也视为设备占用。不能访问必要进程状态时标记WAIT_RESOURCE并报告原因。

两个协调器均使用新锁时，共用`$HOME/.cache/mdmt_job_locks/`，路径按host＋GPU UUID及trajectory_id区分，用flock原子认领。锁记录owner PID、process_start、boot_id、attempt、source hash。只有验证owner确实不存在/属于旧boot，才回收过期lease；不能按超时直接偷走仍活跃任务。

轮询60秒；允许在其他未被保留的空闲卡上执行。全部设备被原补评占有时，本任务完成CPU实现、审计和待运行队列后等待，不重复执行原补评。无法在物理上占用的GPU上保证同时训练，不虚报已给A/B GPU机会。

### 4.2 资源门及派发

每卡默认至多一个本队列训练worker。持有设备lease后再次查询nvidia-smi：显存占比<85%、功耗占比<85%、无未授权计算PID且可用显存覆盖已测峰值的1.2倍。首次未知峰值只允许工程smoke占卡测量；不同时启动多个尚未完成首次分配的worker。功耗读数不可用时记录unknown，使用独占设备＋显存门，不因此永久等待。

RAM门按本轮实际Session快照字节数计量：仅启动资源可容纳当前两个lane驻留快照、拟运行worker及提交时新旧两份快照临时共存的job，并至少保留max(6GiB,机器总RAM的10%)。不能只检查固定6GiB却无限堆积多个大模型的状态。OOM保留batch/precision/config，只释放本轮可释放状态后等待或恢复；不要改batch40960/1024/4096来“修复”。

### 4.3 公平队列：A和B都有实际机会

新训练的种子级优先级按seed2022→2026逐轮展开；每轮A顺序KR-HG、ML-HG、KR-AI、ML-AI；B顺序KR-CP、ML-CP。该优先级不是种子性能筛选，所有五seed都在冻结矩阵中。

- 两张可用卡时优先一张A、一张B。一个lane无READY任务则另一lane可临时使用；lane恢复可用后在最近epoch边界归还，不等待整条50epoch训练结束。
- 单张可用卡时，一个完整epoch为量子，交替A/B；第一次A，第二次B。A/B lane内部优先完成已开始的当前seed，然后下一身份，避免无限增加驻RAM模型数。
- worker完成epoch→RAM原子提交→退出并释放GPU；轮到同job时恢复下一epoch。允许在不影响另一lane公平性的条件下连续grant相邻epoch减少进程开销，但必须遵守另一laneREADY时最多一个epoch的等待边界。
- `--quantum-epochs 1`只控制让出点，不是每个seed只训练1epoch。全程50或Auto search10＋reuse50，checkpoint始终从完整合法窗口选择。
- B不等C/P的五seed性能判断才训练CP；只在汇总2×2时等待缺失控制。A不等所有B控制完成才开始HG/AI。
- 单个A模型工程阻塞，其他A模型及B继续；CP工程阻塞按03启动已定义CG替代。无论哪种情况都不以早期AUC数值取消剩余seed。

## 5. RAM guardian与恢复（必须重新适配，不能原封调用旧serve）

借用源pair.guardian的BytesIO/AF_UNIX通信、cpu_copy/state_digest，但新实现自有namespace、socket和authkey；不复用/关闭历史guardian。guardian元信息文件权限0600，不提交authkey或socket地址到GitHub。

命令契约：

- `register(key,job,pid)`：身份一致、无活跃worker才允许；返回当前Session及launch_index。**launch_index不限2次**，每个正常epoch量子都可能重新register；与failure_count分开。
- `allocated(key,peak_memory_bytes)`：首批资源确认。
- `commit(key,state,epoch_record_sha)`：commit_index必须恰好前值+1；检查key/source/config/data一致；在RAM替换完整快照后写非权重ACK标量，再应答。CPU序列化仅BytesIO IPC，不落盘pickle/pt。
- `yield(key)`：已完成最新commit且worker退出后标记PAUSED_RAM，不增加failure_count。
- `failed(key,failure)`：记录error、失败阶段和已完成物理成本；基础设施错误最多2次自动重试；程序/数值错误BLOCKED_ENGINEERING，隔离该job但队列继续。
- `train_complete(key,result)`：校验所需search/final epochs和全部选点完整；M3/HG/B最终50，AutoIFS10＋50。不可硬编码某个旧epoch字段等于50。
- `get(key)`：仅本attempt evaluator/worker访问RAM状态。
- `release(key)`：仅在该job所有授权端点、成本、hash和DONE.json已经落盘后释放；不得释放外部job。
- `status/shutdown`：输出标量状态；shutdown要求无活跃worker且所有必须保留状态已处理，不能为了退出丢失未评价模型。

B快照包含online、EMA、融合optimizer完整m/v/h/clocks、ordinary Adam、全RNG、loader状态、epoch/global_step、best/selected/E10、buffer、参数manifest、物理成本。A快照包含02中全部阶段状态、Auto selector/rewind/ticket/temp/gate冻结和各optimizer。selected/E10/final若模型hash相同，内部可去重存储，不重复占RAM。

### 断线与失败语义

nohup/tmux使SSH断线不终止coordinator；guardian独立start_new_session。worker进程死而guardian仍在时从最近完整epoch恢复；未提交epoch重放且费用记录，不能把失败样本作为新seed。resume时恢复RNG和sampler后才执行任何会消耗随机数的训练操作；重新构建模型所消耗的随机数必须在restore_rng之前完成。

coordinator死而worker仍活跃时，新coordinator只接管监控，不重复启动worker。worker完成但未evaluate时从RAM只重试evaluate。评价文件部分写入时删除/忽略临时文件，校验后补齐；不因此重训。

**主机重启或guardian死亡导致RAM丢失时，没有磁盘模型可续。** 完整已封存端点仍可复用；仅缺失job标记STATE_LOST_REPLAY并按同一seed从头重放。不得通过pickle/torch.save、memmap权重或其他绕过方式偷偷持久化参数。没有“支持断电从上一epoch恢复”的承诺。

### 状态机与停止

`PLANNED -> REUSE_DONE | WAIT_EXTERNAL | READY -> CLAIMED -> RUNNING -> PAUSED_RAM -> ... -> TRAIN_COMPLETE -> SELECTION_FROZEN -> EVAL_RUNNING -> DONE`。

旁路为WAIT_RESOURCE、RETRY_PENDING、BLOCKED_ENGINEERING、STATE_LOST_REPLAY、CONDITIONAL_NOT_ENABLED。没有`REJECTED_BY_SINGLE_SEED`状态。

`pause`写入STOP_AFTER_EPOCH：不启动新worker，当前worker在完整epoch提交后停止；guardian继续保存状态。SIGTERM只设置同类优雅停止标志，不能承诺SIGKILL安全。`resume`明确移除STOP_AFTER_EPOCH并按状态对账。普通status/report不启动训练。

```sh
$PY -m experiments.overnight_ab_20261103.run status --attempt "$ATTEMPT"
$PY -m experiments.overnight_ab_20261103.run pause --attempt "$ATTEMPT"
$PY -m experiments.overnight_ab_20261103.run resume --attempt "$ATTEMPT" --devices 0,1
$PY -m experiments.overnight_ab_20261103.run retry --attempt "$ATTEMPT" --job-id kuairand.C_MIX_P01_SERIAL_V1.s2023
```

retry只能在具体故障已修复、身份一致或审查明确确认数值路径不变后执行；额外人工重试写原因及成本，不能改变seed。正常量子数、恢复次数、online/EMA/E10/E50都不增加独立n。

## 6. 逐seed冻结/评价：避免全队列RAM积压

本规划有意细化旧phase2的“全部训练完才test”：**全体配置和方法矩阵在新训练前冻结；每条完整轨迹自己的选点在test前冻结**，随后自动评价并释放该seed RAM。这样不需要在RAM中同时保留20个发表模型和10个组合的全部状态。

这是执行粒度调整，不是让test参与选择：

1. coordinator和worker只读取配置、运行状态、资源与成本；其代码不能根据性能表更改队列/组合/参数。
2. 每条轨迹完成固定预算后写`selection_frozen.json`，含每估计器selected阶段/epoch/model_hash、V宏AUC/六格、完整V曲线hash、选点函数hash、CONFIGURATION_FROZEN hash。选点必须可由曲线严格argmax重建。
3. evaluator核验同一个RAM模型hash，对selected及预定义E10/E50评价，写未缩放六格指标；结果原子写临时文件→fsync→rename。
4. 最后写`DONE.json`，包含所有必需文件checksum和端点schema版本，再释放RAM。DONE之前崩溃不能算已完成。
5. 自动生成描述性进度；不足五seed的性能汇总标INCOMPLETE，不形成否决。完整五seed后输出最终裁定，不影响其余冻结矩阵。
6. 历史补评runner继续使用原先“全部30新训练完成后冻结test”的协议；本任务只等待/读取其合法产物，不替它提前解锁test。

已有历史test不是新盲测；报告如实标注historical_reuse。这不阻止本轮配对比较，但不能宣称首次未见确认集。

## 7. 保存路径与报告格式

运行目录：`HISTORY_ROOT/artifacts/mdmt_overnight_ab_20261103/attempt1/`；报告：执行工作树的`research_reports/2026-11-03_overnight_AB/`。原始逐seed记录不覆盖；修复用attempt编号和来源链。

每条job：`jobs/<job_id>.json`，`runs/<job_id>/launch_<n>/epoch_*.json`、`batch_trace.jsonl`（标量）、`selection_frozen.json`、`endpoints.json`、`cost.json`、`DONE.json`。状态/失败：`queue_state.json`、`events.jsonl`、`failures.jsonl`、`resource_usage.csv`、`progress.json`、`EXTERNAL_DEPENDENCIES.json`。JSON/CSV不允许NaN字符串，未定义值用null/空值及原因。

### 必须输出的报告表

| 文件 | 最小内容 |
|---|---|
| `ENDPOINTS.csv` | 每个dataset×arm×seed×estimator×endpoint×split一行，六格及全指标、epoch/hash/来源 |
| `ABSOLUTE_SUMMARY.csv` | 同协议每模型/arm五seed均值、样本SD、n；论文模型与内部参考分栏 |
| `PAIRED_SEED.csv` | candidate/reference原值及差；总收益、组件增量、对发表模型分别comparison_id |
| `PAIRED_SUMMARY.csv` | 每比较五seed均差/SD/t95%、宏达标/六格保护/LL独立标志 |
| `FACTORIAL_SEED.csv`、`FACTORIAL_SUMMARY.csv` | CP或CG四格及逐seed交互I；含六格和LL，不额外加合格门 |
| `VALIDATION_CURVES.csv` | 完整合法epoch宏/六格曲线；Auto search日志明确分阶段 |
| `COSTS.csv` | 原训练/本轮新增/重复恢复成本、steps、样本/标签暴露、GPU/worker秒、峰值显存、RAM、阶段、EMA附带开销 |
| `A_REUSE_AUDIT.csv`、`B_REUSE_AUDIT.csv` | 01规定全部来源、可重选/需重训/外部等待判定 |
| `DECISIONS.json` | 每dataset×comparison各自n、阈值、状态和理由；不得跨协议作结论 |
| `PROGRESS.md`、`FINAL_REPORT.md` | A/B实际机会及完成数、缺失/故障、科学结论、适用边界、下一步决定 |
| `LEDGER_APPEND.md` | main台账同格式新增协议分组，保留旧组、真实来源与n |

ENDPOINTS字段至少：protocol_id,data_protocol,config_hash,source_hash,trajectory_id,selection_id,endpoint_id,dataset,method,role,seed,estimator,checkpoint,stage,epoch,model_hash,split,source_path,source_sha256,reused,independent_run_id,d0t0_auc,d0t0_logloss,...,d2t1_auc,d2t1_logloss,macro_auc,macro_logloss,sample_weighted_cell_auc,sample_weighted_cell_logloss,pooled_task_mean_auc,pooled_logloss；同时保留各格count/positive/negative和domain/task指标。

role枚举：PUBLISHED_BASELINE、INTERNAL_REFERENCE、COMPONENT、COMBINED_METHOD。同一个A0轨迹可提供两个估计器端点，但不重复计成本和seed。正式模型M3显示M3OE_FIXED1_ADAPTED；A0标签仅用于内部因子表。

### 计算与裁定

逐seed先算candidate-reference；以2022–2026五个完整匹配seed算均值，不能用不匹配的两个五seed均值代替配对表。未四舍五入原值参与判定；SD为ddof=1；描述性95%配对t区间采用df=4系数2.7764451051977987乘SD/sqrt(5)，CI不作门。

`auc_macro_pass = mean(delta_macro_auc)>=.003`；每格独立`cell_pass[c]=mean(delta_cell_auc[c])>=-.001`；`auc_guard_pass=all(cell_pass)`；`ll_improved=mean(delta_macro_logloss)<0`。宏达标但逐格越线必须单列“宏达标、逐格保护不满足”以及具体格/幅度；不一概隐藏总收益。LL与AUC可以一升一降，均原样报告。

2×2 I按每seed计算`11-10-01+00`再汇总。对A0 online的总收益、A0 fixedEMA的组件增量、CP对C/P的增量及CP分别对三个发表模型的比较，各有独立comparison_id。0.003是用户总体收益标准，不意味着每个串联小组件都必须额外贡献0.003才获准保留；这些对比仍报告数值与相同阈值标记，但不能替代完整方案裁决。

结果报告采用“当前结论→核心数值证据→适用边界→研究决策”。不因机制未完全解释重新否定可靠五seed现象；不因为未测版本自动判所有方向失败。

## 8. 有限工程验收及完成收据

以下是实现验收，不是额外科研实验，不能依据其AUC提前淘汰算法：

1. 计划编译：60核心去重身份、30预期新增；A30槽位、主B40槽位；两数据×五seed覆盖；替代单独启用。
2. 选点：合成曲线最大值/严格并列最早；LL不介入；Auto search不可选；E0不可选；六格不完整禁止nanmean。
3. 评价：同model hash的V/Test、概率只sigmoid一次、统一未缩放LL；原始列名和macro/traffic/pooled不混称。
4. 算法：03的有限数值等价、单次更新、m/v/h/clock、参数分组及kappa隔离测试；保留已有模型机制测试，不重做理论证明。
5. 恢复：短合成或小fixture轨迹连续两epoch与1epoch→RAM→恢复1epoch的模型、optimizer、EMA、RNG、选点、数据顺序hash一致；Auto覆盖search9→search10→reuse1边界；模拟协调器重启不重复claim；正常量子不消耗失败预算。
6. 安全运行：禁torch.save/pickle落盘权重；STOP只在epoch边界；相同trajectory仅一owner；外部GPU保留；失败只阻塞具体job；不读取test值调度。
7. 真实batch smoke：每数据新B组合一完整原batch，HG/AI各模型每数据一batch；AI阶段转换的机械测试可用已有fixture。记录有限梯度及显存/RAM，丢弃smoke状态，正式轨迹重置完整seed。smoke不是单seed性能实验，不计算performance gate。
8. 汇总测试：恰好+.003/−.001等号通过；存在负seed但平均达标通过；LL改善独立；缺seed为INCOMPLETE；重复seed/不同协议拒绝合并；CI跨零不追加否决。

输出`IMPLEMENTATION_REPORT.md`和按A-HG/A-AI/B-CP/B-CG分别的`ENGINEERING_ACCEPTANCE.json`，绑定新源码hash、测试命令/结果、未解决阻塞。只执行PASS或无blocking的PASS_WITH_FIXES模块；允许另一个模块CPU开发时已通过模块进入GPU队列。没有独立reviewer时不得伪造独立审查身份；执行者必须完成上述实际验收并如实记录review方式，既有独立审查不冒充新融合代码已获审查。

## 9. 阶段完成与发布

每次job DONE自动刷新进度和已完整比较表。外部结果到达自动接入并更新2×2，不重训。所有核心授权job完成且A/B各五seed配对齐备时写FINAL_COMPLETE；若工程阻塞或外部未完成，写PARTIAL_ENGINEERING/WAIT_EXTERNAL并精确列出缺项，禁止写性能失败。

提交执行分支时只提交代码、规划、审计及标量报告，不提交数据、模型权重、RAM序列化、authkey、巨量逐batch日志。main台账更新采用新增协议分组与逐行来源链，不覆盖正在补评写入的记录。原夜间规划分支保留这套冻结规格及其提交版本；不要把仍未运行的矩阵写成已完成实验。
