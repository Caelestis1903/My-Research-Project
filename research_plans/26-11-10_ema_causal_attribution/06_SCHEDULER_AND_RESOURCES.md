# 06｜长程调度、资源预算与故障恢复

> 状态：**DESIGN_ONLY**。以下为未来自动执行规范；本次只提交规划。逻辑上按 Stage 0→4推进，物理上必须在参数仍驻RAM时完成依赖它们的归因，不能等所有训练结束并释放状态后再补做干预。

## 1. 运行结构：五seed条件块与实时归因

固定四个条件块的顺序：

1. MovieLens Clean：2022–2026；
2. KuaiRand Clean：2022–2026；
3. MovieLens Gaussian D0：2022–2026；
4. KuaiRand Gaussian D0：2022–2026。

每块训练五条原轨迹，同步采集规定的O/F、稀疏R、分数EMA与pulse。每条训练完成后释放其optimizer、活动pulse和已经消费完的历史缓存；保留该块anchor packet。五seed完整后，调度器执行 `finalize_block`：汇总粗组结果，依冻结规则运行必要细分、组×时间、功能干预与可选来源读出，再冻结该块机制结果。不能根据第一个seed决定取消后四个seed。

块内最终化完成后释放该块全部固定anchor与selected历史分量，只保留每seed的 **O独立selected、F独立selected、F选点同刻O** 三份FP32模型端点，直到全局分析冻结和Test。S-only/N-only at F-selected可以从后两份参数精确重构，不额外保存模型。已经封存的功能/时间结论由标量与有限预测复算，不需要保留全部旧参数。

这是Stage 1采集与Stage 2/3活状态操作的物理交错。`analyze --stage 1|2|3|4`只聚合已提交数值；需要活参数的动作由 `run/resume` 的块最终化执行。不得把 `analyze` 实现成重新训练、默默恢复已释放模型或根据新Test结果扩展实验。

## 2. 各阶段输入、输出和完成条件

| 逻辑阶段 / 物理动作 | 输入 | Codex动作与输出 | 完成条件 |
|---|---|---|---|
| Stage 0：来源与理论审计 | 固定源码commit、01既有证据、当前机器数据/cache | 读取源码/协议/释放记录；生成 `SOURCE_AUDIT.json`、复用清单、参数布局/support和数据身份 | 源loss/rho/clock/分组可落实；缺失状态明确记需新采集 |
| Stage 0：prepare | 00–09全部规范、可用环境与缓存 | 生成20-job协议、稀疏观察事件表、固定面板、读出预算、source/data/plan hash、资源预估 | 不包含Test标签；五seed/50epoch完整；预计RAM和观察字节受控 |
| 工程smoke | 审查通过的新包、两数据集原batch/尾批 | 最少必要的源语义、纯度、状态重构、pair计数、函数cut/导数、RAM恢复及吞吐测量 | 工程问题修复；身份与预算收据一致；不是性能筛选 |
| Stage 1：条件块采集 | prepare、对应数据与五seed jobs | 原轨迹50epoch；O/F每epoch完整V；R事件；每step固定面板score EMA；固定pulse；anchor packets | 五seed全部50epoch，源在线身份与观察事务完整 |
| Stage 2：块内参数/时间定位 | 五seed粗组/窗口完整V、活anchor | 运行最小充分S/N与时间对照；按规则最多12个细分状态/anchor；有限组×时间 | 五seed完整配对；无用组合不展开；未展开理由可追溯 |
| Stage 3：块内函数与来源解释 | Stage 2已定位对比、q、面板、活anchor | 全量/面板排序迁移、历史预测组合、完整图割、必做融合五项删除/最多一项来源链删除、有限路径积分与来源读出 | 参数→函数→排序证据闭合；残差和支持边界明确 |
| 块释放 | 该块所有活状态任务完成 | 写 `BLOCK_COMPLETE.json`；只保留三份FP32最终端点 | 不再存在引用被释放packet的待执行操作 |
| Stage 4：全局整合 | 四块冻结数表、历史adaptive证据 | 配对统计、H1/H2/H3、Q1–Q5与A/B/C/D方向；写 `MECHANISM_FREEZE.json` | 不查看新Test；不再修改分组、假设、指标或展开规则 |
| 冻结与Test | 全20轨迹/Stage4完整、端点hash | 五种预注册读出：Osel、Fsel、OatF、S-only atF、N-only atF；同hash去重/合法归档复用 | 各读出五seed齐全；Osel/Fsel为独立选点性能，后三者为同刻机制确认 |
| 报告与release | 已冻结Test与机制数值 | 生成完整数表、研究报告、执行成本、完成收据；释放本任务guardian | Q1–Q5均有答案或精确缺口；没有无决策价值的追加任务 |

正式Test不加入按时间窗口或函数图割挑选的赢家；时间和功能机制的主要识别来自完整V及预先冻结面板。Test不能用于修改机制分组或研究方向判据。

## 3. 任务身份、命令与依赖

未来新CLI统一为 `experiments.ema_causal_20261110.run`。只有09的实现与review完成后这些命令才存在。示例路径使用项目环境变量，避免依赖Conda base：

~~~bash
MRP_PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
MRP_PLAN=research_plans/26-11-10_ema_causal_attribution
MRP_OUT=artifacts/ema_causal_20261110/attempt1

"$MRP_PY" -m experiments.ema_causal_20261110.run audit \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT"

"$MRP_PY" -m experiments.ema_causal_20261110.run prepare \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT" \
  --ml-csv /home_new/xzf/datasets/ml-1m/ml-1m.csv \
  --kr-csv /home_new/xzf/datasets/kuairand/processed/kuai_rand.csv

"$MRP_PY" -m experiments.ema_causal_20261110.run smoke \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT" --dataset movielens
"$MRP_PY" -m experiments.ema_causal_20261110.run smoke \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT" --dataset kuairand

"$MRP_PY" -m experiments.ema_causal_20261110.run run \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT" --devices 0 1 --max-workers 2

"$MRP_PY" -m experiments.ema_causal_20261110.run status \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT"

"$MRP_PY" -m experiments.ema_causal_20261110.run resume \
  --plan-dir "$MRP_PLAN" --out "$MRP_OUT" --devices 0 1 --max-workers 2
~~~

`--devices 0 1`只是候选设备，非设备预约；实际只对可见且满足准入的卡派发。显式 `--cache-root`仅在已确认匹配的数据缓存可用时传入prepare；否则由原构建器在本任务cache目录重建同身份缓存。缓存成本独立记账，不能覆写冲突旧cache。

后续顺序为 `analyze --stage 1`、`2`、`3`、`4`，随后 `freeze`、`evaluate`、`report`、`release`。正常 `run`应自动完成块内采集及活状态归因；调度器可以在冻结依赖满足后自动接续聚合和后续阶段。`status`只读，`resume`只能继续同一冻结attempt。代码或科学配置改变时，新建身份清晰的attempt，不覆盖已经提交的观察。

## 4. 正常训练与观察预算

### 4.1 不可缩减的训练量

| 条件 | 轨迹 | 每轨迹epoch | step/epoch | 总Adam step |
|---|---:|---:|---:|---:|
| ML Clean | 5 | 50 | 20 | 5,000 |
| KR Clean | 5 | 50 | 49 | 12,250 |
| ML Gaussian D0 | 5 | 50 | 20 | 5,000 |
| KR Gaussian D0 | 5 | 50 | 49 | 12,250 |
| **合计** | **20** | — | — | **34,500** |

共1,000epoch。没有每项参数干预各训一条模型，也没有新增算法HPO。原始训练样本曝光为1,403,149,500行；Gaussian沿原replacement只增加规定view计算，不改变原loss分母。

### 4.2 可以预先精确列账的观察

| 项目 | 正常数量 / 上限 | 说明 |
|---|---:|---|
| O/F完整V常驻读出 | 2,000 | 20×50×2；各自选点 |
| 固定anchor | 100 | 20×5；selectedF最多再20个，重复去重 |
| S-only/N-only完整V新增状态 | ≤240 | 120anchor×2 |
| 四个时间窗口recent-only/old-only新增状态 | ≤960 | 120×4×2；初期h_eff、零旧分量、重复hash去重 |
| B细分新增状态 | ≤960 | 最多12状态×最多4指定细分anchor×20；与父状态相同者去重 |
| 融合内容/路由/交互/域/任务功能分解读出 | ≤240 | E35/F-selected两锚点×最多6状态×20；现场消费激活 |
| H2同H/同scope/范数匹配控制 | ≤160新增状态 | 只在H2将指导选B时；E35/E50，最多2scope、每scope两真实读出 |
| H pulse | 200 | 每轨迹10个，two shadow；无真实Adam增量 |
| H桥接完整rank梯度 | ≤200次 | 每pulse终点F上一次；08§4全部256×256 pairs风险，现场归约all/S/N标量，不保留参数梯度 |
| 逐step固定面板在线前向 | 34,500次，加初始化20次 | 每次去重后行数≤21,504 |
| H菜单面板状态 | 200×13逻辑状态上限 | 两组7动作共用F；F/O已有预测复用 |
| 正式Test读出 | ≤100逻辑端点 | 每job五读出；同state和归档复用减少实际量 |

其他有界部分——重要组×时间、两级完整图割、实际路径节点、JVP8→16→32、有限source assay——由04的注册descriptor精确展开。`prepare`必须写出这些类别各自的状态/前向/反向/参数快照**完整上限**，不得留“视情况无限追加”。细分只在E10/E35/E50及F-selected执行，最多两个主分支；S与N均重要时各三类的单独/删除读出合计12种。固定1路由support成立时，D/T/DT族在对应cell的读出即相应leaf读出，可按source支持关系拼接其他不变cell，无需再跑leaf全组合。

每次新物理读出登记 `logical_descriptor_id, state_or_operator_hash, reused_from, full_population_or_panel, forward_rows, backward_calls`。相关hash只用于计算复用，不删除逻辑比较。

### 4.3 全量排序表的CPU成本

完整AUC是每个规定完整V读出的必做项。全量3×3迁移表按03/04注册范围执行，不能用32768面板pair替代“精确全量”。

使用编译或向量化的O(N log N)计数内核；纯Python逐pair或Npos×Nneg矩阵不是正式实现。工程smoke用真实全量N测一次计数吞吐和峰值RSS，不产生科学性能判断。若当前环境仅有慢实现，先优化计数核或降低并行CPU任务数；不以抽样替代已注册精确计数。CPU只读分析可与GPU训练并行，但不能让其RAM峰值挤占guardian恢复空间。

## 5. RAM、GPU与观察磁盘预算

### 5.1 预算来源

[全局start.md](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/start.md)明确禁模型/检查点落盘，并规定单GPU显存/功耗85%、整机RAM90%准入。**4GiB guardian payload与2GiB观察输出是本方案主动继承的保守任务上限**，来源是[前次资源规划](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_plans/26-11-6/07_RESOURCES_AND_REUSE.md)，不是额外的用户确认流程。

模型/optimizer/参数历史用RAM；标量、有限预测、身份、代码与报告可落盘。禁止参数向量写入临时文件、压缩NPZ、memory-mapped checkpoint或tmpfs文件来绕过限制。普通进程内存、内存IPC与不可变共享RAM对象可用。

### 5.2 按实际dtype计算模型状态

| 状态 | 字节公式 | ML（P=773,470） | KR（P=1,614,254） |
|---|---:|---:|---:|
| 一份FP32参数 | 4P | 3,093,880 | 6,457,016 |
| 一份FP64 R | 8P | 6,187,760 | 12,914,032 |
| 单anchor：O32+R64+4old64 | 44P | 34,032,680 | 71,027,176 |
| 五seed×五固定anchor | 25×44P | 850,817,000 | 1,775,679,400 |
| 本块selectedF四old64的额外上限 | 5×32P | 123,755,200 | 258,280,640 |
| 一个活动worker稀疏R缓存安全上限 | ML19/KR27×8P | 117,567,440 | 348,678,864 |
| E35/E50匹配H的fast/slow两FP32向量×五seed | 20×4P | 61,877,600 | 129,140,320 |
| 可选两anchor source assay四FP32向量×五seed | 10×16P | 123,755,200 | 258,280,640 |

全20条最终端点的三份FP32模型合计最多 **286,526,880 bytes**。selectedO、OatF、Fsel若重复可去重；本预算不依赖去重节省。

以最重KR块、两个活动worker估算：固定anchor约1.776GB、selected旧分量0.258GB、全研究最终端点0.287GB、活动稀疏缓存0.697GB、活动online/Adam/F/理论R/pulse约0.103GB、匹配H控制保留向量0.129GB及两worker新增active向量0.026GB、原子提交额外余量约0.155GB、可选source assay0.258GB，合计约 **3.69GB** 参数类payload，再给有限score state和对象登记留空间。这个估算要求：

- completed job及时释放optimizer与活动历史；
- block完成后释放旧anchor和selected旧分量；
- immutable packet只提交一次，后续按引用使用；
- 原子提交仅双持active增量，不能复制整个五seed块。

真正准入以 `tensor_bytes` 递归精确计数为准：当前持有＋新提交＋尚未释放的旧commit＋必要重构临时副本都计入4GiB。Python进程RSS、IPC序列化、数据映射、CPU排序数组另记录并受整机RAM90%限制。如果两worker超预算，自动降到一worker；不能减seed、batch、epoch或科学观察窗口。

### 5.3 GPU

GPU只保留当前online、Adam、F、理论R、两个常规pulse、E35/E50有限窗口内额外两个匹配H pulse、scratch及当前batch计算。固定anchor与稀疏历史在CPU RAM。组读出/图割/JVP顺序消费，一次只建立一个必要计算图，不同时常驻所有候选模型。

前次实际allocated/reserved峰值约1.134/1.481GB，只是源任务参考，不作为新任务承诺。新任务图割与积分可能增加激活峰值，必须由原batch工程smoke测量。每卡分别检查显存与功耗；任一卡可用时可以使用，不要求所有卡同时空闲，不结束他人进程。

### 5.4 2GiB观察输出

采用“保存固定面板预测、压缩标量、即时消费大对象”的策略：

| 类别 | 计划额度 |
|---|---:|
| regular与anchor的固定面板O/F/q/有限干预预测 | ≤0.90GiB |
| H菜单与冻结未来预测需要的有限面板预测/效用 | ≤0.45GiB |
| 标量表、pair聚合、函数闭合与来源统计 | ≤0.35GiB |
| 面板身份、hash、日志、报告和图 | ≤0.20GiB |
| **预留余量** | **0.10GiB** |

按源panel上限，单状态P_A/P_B的p+z与V_mech的p+z共294,912 bytes；H状态的训练panel p+z、V_mech只p共196,608 bytes。可执行的默认保存清单如下，均不依赖压缩：

- O/F只在10个pulse终点与额外F-selected保存面板原预测，最多220个时刻×2状态×294,912 = 129,761,280 bytes；每epoch完整V标量仍全部保存。
- q_prob/q_logit仅五固定anchor与额外F-selected保存FP64数组，最多120×589,824 = 70,778,880 bytes。
- B/C核心最多1,200个额外状态保存原面板预测，1,200×294,912 = 353,894,400 bytes。
- B细分最多960状态保存原面板预测，960×294,912 = 283,115,520 bytes。
- 窗口起点只持久化p：最多120个固定/最终selected anchor×4窗口×147,456 = 70,778,880 bytes；窗口z/训练目标/rank-risk在RAM中现场归约，不持久化所有scheduled epoch×h的p/z。
- 上述合计908,328,960 bytes，约0.846GiB，低于0.90GiB；重复hash还能减少，但预算不依赖它。
- H菜单除已保存F外最多12×200个状态，2,400×196,608 = 471,859,200 bytes，约0.439GiB，低于0.45GiB。

组×时间、图割、融合五项、路径积分、H2匹配控制和source assay默认现场写完整标量、固定pair聚合及闭合误差，不再重复保存全部面板数组。必要的少量复核预测使用regular类别剩余约0.054GiB并在prepare预先列明。禁止把每step O预测加入上述持久清单。非固定anchor的临时selected数组随RAM packet替换，50epoch结束才写最终selected；历史选择曲线只保留标量。

这些是实现前的硬分区额度，不是对压缩率的承诺。`prepare`按04实际descriptor和每数组shape/dtype计算原始上界；若某类别可从已保存预测完全再生，优先不重复保存。大部分后续状态只需保存标量/有限pair聚合，在RAM算完即释放；面板预测的持久化范围必须预先列入 `PREDICTION_SCHEMA.json`，不能根据结果好坏删不利状态。

逐step q只保存当前RAM值；不持久化34,500步的全部面板预测。训练来源窗口只保留调度要求的端点预测和统计。全量V/Test预测只在该次评价RAM里临时使用。写入前预估、写入后更新实际字节；gzip标量从一开始写，不生成冗余巨大CSV。资源不足时先减少本任务并发与无信息重复存储，不改变科学矩阵。

## 6. 时间估算与长程运行预期

前次[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)和[COSTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/COSTS.csv)提供实测参照：

| 项目 | 前次实测 |
|---|---:|
| 正式训练 | 20轨迹、1,000epoch、34,500 Adam |
| 训练累计worker秒 | 13,226.080 |
| 4,500次完整V/regular观察累计秒 | 2,893.289 |
| 5,200个新增pulse状态累计秒 | 553.674 |
| 2,400次小面板梯度累计秒 | 301.436 |
| IPC累计秒 | 2,411.616 |
| 正式队列墙钟秒 | 9,980.239 |

这些worker计时包含共享设备等待，不能除以GPU数就宣称新任务的墙钟。新方案保留同样训练量，常驻完整V减少，但加入逐step预测EMA、完整R干预、精确排序迁移和函数解释。因此计划预留 **单设备8–16小时、两设备4–10小时** 的排程窗口，作为待smoke修正的量级，不承诺固定完成时刻。

按实际测量重算：

$$
T_{\rm work}=
\sum_c 5\{50T_{{\rm train},c}^{epoch}
+100T_{{\rm fullV},c}
+N_{{\rm step},c}T_{{\rm panel},c}\}
+N_B T_B+N_C T_C+N_D T_D+T_H+T_{\rm CPU}+T_{\rm IPC}.
$$

`T_train`必须区分源训练与观察；每一项来自对应数据集/目标的工程计时，读出数量来自冻结schedule。不能重复把观测包含在训练计时里再加一遍。原始面板前向上界21,504行×34,500步约7.42亿行；这是真实新增成本，不能写成“分数EMA只需几个标量运算所以几乎免费”。

未来 `RESOURCE_ESTIMATE.json` 输出下界/中位/保守上界、实际并发、GPU型号和CPU排序耗时。若资源允许，提升并发；若实际计时较慢，保留完整任务继续运行并更新预计完成时间，不临时删分析模块换取好看的耗时。

## 7. RAM生命周期与事务

### 7.1 不可变packet与增量IPC

E35/E50的匹配H控制须在source step `t-H`预先启动，恰好H步后保留fast/slow；即使触发判断要等五seed，也不能事后才发现没有该时间窗口状态。触发范围只按03/08决定，未触发向量直接释放。

每常规pulse终点的F完整rank梯度观察同样属于训练期采集依赖，必须先现场归约all/S/N摘要，才能释放该终点模型；block最终化只消费这些已保存标量，不要求保留10个pulse终点的全部参数。

每个 `AnchorPacket` 以 `job/step/type/hash` 为不可变ID。guardian登记引用和消费者。训练worker每epoch只发送新packet、active恢复状态与标量事务，已存在的anchor不重复序列化。分析worker读取时尽量按引用/共享RAM访问；若IPC必须复制，先预留该副本再准入。

一次提交的顺序：

1. 计算所有新对象字节，检查稳态与提交瞬时上限；
2. 登记未发布packet；
3. 保存新的active恢复状态，同时短暂保留上个已提交active状态；
4. 原子发布有限预测/标量和receipt；
5. 写epoch commit ledger，更新消费者；
6. 释放旧active状态、已消费稀疏历史和未再引用的旧selected状态。

完整训练状态不能写进epoch JSON。completed job不再保留Adam，以免20条optimizer一直累积。块finalize确认之后，才释放其研究anchor；最终endpoint直到Test完成才释放。

### 7.2 单owner和并行

一个job同时最多一个训练owner；一个不可变packet可有多个只读消费者。GPU同一时刻的读出任务必须拥有独立scratch，禁止多个线程轮流覆写同一个模型对象。CPU分析只能读已提交标量或immutable packet，不能抢读半写入状态。

调度器每次派发前检查完整依赖，不以文件存在代替完成收据。状态为 `PENDING/RUNNING/COMMITTED/BLOCK_FINALIZING/BLOCK_COMPLETE/RESOURCE_BLOCKED/FAILED`；已完成的科学任务不因重复执行命令再生成一次。

## 8. 故障恢复与重试上限

| 故障 | 恢复动作 | 科学身份处理 |
|---|---|---|
| 训练worker退出，guardian仍在 | 从最后已提交epoch恢复online/Adam/F/理论R/q/常规及匹配H pulse/稀疏缓存/loader/RNG/selected和观察cursor；同job最多一次自动重启 | 未完成epoch物理成本单列；不增加seed |
| 观察器失败，训练当前epoch未提交 | 保留旧commit；修复工程问题后同epoch恢复重做，拒绝混用半成品预测 | 原job身份不变；分析不读未commit记录 |
| 块finalize只读分析失败 | 保留block anchors，重试相同冻结descriptor | 不重训；不允许改有利结果对应规则 |
| guardian RAM丢失 | 对真正需要恢复的同seed从theta0确定性重放，逐epoch核对已有online/order/F hash；只补缺失状态与成本 | 每逻辑job至多一次额外物理重放；不新增科学seed |
| 已完成块仅最终endpoint丢失 | 可重放到各冻结端点所需最大epoch，重建O/F后校验hash；旧机制表不重选 | 不利用Test寻找更好恢复点 |
| OOM/资源超限 | 释放本任务临时图/无引用对象、降低并发，从已commit状态恢复 | 不减batch/seed/epoch；不落盘模型绕行 |
| source/data/row/hash冲突 | 停止受影响job，保存冲突元数据，定位正确来源 | 不能覆盖旧结果或把工程冲突计为算法失败 |
| 第二次同job完整重放仍失败 | 标记工程阻塞并给出缺失对象/原因 | 不无限重试、不用新seed补成“成功五seed” |

全研究发生最坏的一次完整重放上限为40条物理轨迹、2,000epoch、69,000真实Adam；这是故障上限，不是默认预算。通常只恢复失败epoch或所需冻结端点。报告同时给出 `logical_trajectories=20` 与真实物理成本，不能把故障重放扩充n。

## 9. 日志、自动报告与停止规则

`status`至少显示：当前条件块、完成seed/epoch、活状态阶段、逻辑/物理step、GPU/host/guardian内存、观察磁盘、最近commit、待消费packet数量、失败原因和由实测更新的剩余时间。进度日志只写必要状态变化及每epoch摘要，不记录大量逐参数文本。

科学表逻辑名以07为准，允许gzip等物理表示；`ARTIFACT_INDEX.json`保存逻辑名、物理路径、schema/hash/行数，所有阶段据此读取。自动报告必须从已冻结数表生成，不把预期机制写成发现。每个主要结论绑定：配对效应→真实状态或函数干预→排序计数→交互/残差→训练来源证据→下一步A/B/C/D方向。尚未识别的训练来源明确限定，不能降低已识别的参数/功能因果效应。

完成停止条件：

- 20条逻辑轨迹、所有正式五seed块完整；
- 全部注册核心干预与被触发的有界展开完成；
- 阶段完成、Test冻结、去重及源身份一致；
- Q1–Q5有明确研究判断，有限未解问题只保留会改变方向者；
- 观察文件与报告可复算，物理成本和故障记录完整；
- Test结束后释放本任务RAM，生成 `RELEASE_RECEIPT.json`。

不得为了增加“长程”时长继续跑新rho网格、PUB修复、自适应控制器或无关基线。长程预算用于一次轨迹产生更充分的因果与功能信息。


