# 12 — 报告模板与交付合同

## 1. 报告原则与目录

未来执行报告统一存放于 `research_reports/history_information_20261124/`，数表、权限与身份文件存放于 `artifacts/history_information_20261124/`。本文件给出**待执行模板**，所有结果初始状态为`UNRUN`；本规划中的历史重分析保持`HISTORICAL_REANALYSIS`，不能改写为新候选成绩。

每项实质结论按“当前结论 → 核心证据 → 适用边界 → 研究决策”展开。结论说明属于经验现象、局部因果、机制解释还是完整算法表现；边界只写会影响该判断或下一动作的具体条件。完整训练有收益即可进入预定控制，不要求穷尽所有机制；已完成负结果按对应定义关闭，不无限增补验证。

最终交付至少包含以下文件，格式与字段详细定义见07/09：

| 文件 | 内容与完成条件 |
| --- | --- |
| `FINAL_REPORT.md` | 本章第2节完整主报告；所有触发工作有结果或具体技术状态 |
| `MECHANISM_REPORT.md` | D1–D4、同状态作用、跨格信息与阶段行为；局部/完整层次分开 |
| `IMPLEMENTATION_REVIEW.md` | 独立审查人、源码锁、05/07关键合同结果、修复清单 |
| `NUMERICAL_APPENDIX.md` | 全部候选/控制/参照的五seed及六格、CI、选点、来源 |
| `SOURCE_MANIFEST.json` / `REUSE_REGISTER.csv` | 实际源码/数据闭包、历史复用与新训练身份 |
| `TEST_POLICY.json` / `FINAL_FREEZE.json` | 本轮权限范围、一次全局冻结、selected实际RAM读回 |
| `endpoints.csv` / `paired_summary.csv` | 全精度数值；宏指标与六格、读出、阶段均可追溯 |
| `selection_decisions.csv` / `decisions.json` | 参数、家族代表、各条件模块及最终研究标签的规则执行记录 |
| `trajectory_ledger.csv` / `costs.csv` | 科学轨迹、重建、attempt、真实更新/调用/计算成本分开 |
| `diagnostics/*.csv` | 04要求的来源、同状态、关系保留与EMA交互表 |

报告中的每个数字链接其表、比较ID和源身份；不能只有截图或四舍五入后的均值。图片可作为曲线/热图辅助，CSV与选择规则必须完整交付。

## 2. FINAL_REPORT.md 完整模板

以下标题全部保留，填入真实结果。未触发的条件模块保留标题、状态及原因，不以空白隐去。

### 报告标题与状态

```text
# 历史训练信息研究：来源二阶组织与历时预条件辅助历史
Study: history_information_20261124
Status: UNRUN
Evidence main: 358d3a465747eda6af1d55acc937759fa651a10b
Plan commit: 待实际提交填写
Execution source lock: UNRUN
Completion date (Asia/Shanghai): UNRUN
New training trajectories: 0 (规划初态)
New Test evaluations: 0 (规划初态)
Test identity: historically_exposed_holdout
```

运行后将数字替换为账本实际值；状态只能来自已完成、科学关闭、技术暂停或授权限定，不能从计划预算推定已完成。

### 2.1 当前研究结论

先用一段回答四件事：哪一种方法/组件在什么数据和读出上有效；超出了哪些匹配控制；是否通过最强发表模型和逐格保护；下一步采用什么方案或关闭什么定义。接一张结论表：

| 研究问题 | 当前结论 | 证据层级 | 核心比较/数表 | 适用边界 | 研究决策 |
| --- | --- | --- | --- | --- | --- |
| Q1 历史的有效信息与处理 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| Q2 EMA与在线历史关系 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| Q3 MDMT多来源与共享学习 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| Q4 方法贡献与正式性能 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

### 2.2 完成范围、配置与权限

| 模块 | 计划逻辑轨迹 | 实际新物理轨迹 | 数值/状态复用 | 技术重放 | 独立seed数 | 状态与理由 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0/B1/H0父参照 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| C1两非零配置 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| C2两非零配置 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| M/S/P | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| C12 / N / HyperGate / Gaussian | 分模块填 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| 发表基线身份恢复 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

说明自然CSV及split hash、train-only转换、PAIR支持、参数组、种子2022–2026、完整训练预算、各模型原生batch/优化器、选点规则和固定EMA。记录本轮Test权限来源、冻结时间、冻结后是否有定义变更。历史Test暴露状态直接写明；Validation限定任务以`TEST_PENDING_AUTHORIZATION`结束对应评价，并给出完整Validation结论。

### 2.3 两个候选实际实现

列出最终使用的05公式和参数：C1的ζ、C2的α、βP、WD、scope、PAIR λ和固定EMA ρ。说明C1的cross中含WD–PAIR项；C2保存的是`D_t h_t`而非完整已执行Adam更新；候选零点如何调用原H0。若最终选S短核代表，明确名称及βP=.9，不继续写为原.99候选。M/P覆盖后的去向按08报告，不自行创建新家族。

| arm/config | ζ | α | βP | scope | active state | source/config hash | 实现审查 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| H0/C1/C2及实际触发控制逐项填 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

### 2.4 默认完整训练与轻量搜索

两数据分别列出C1 `{0,.5,1}`与C2 `{0,.5,1}`，零点为复用H0。保留所有五seed的full50结果、selected online/EMA以及E10/E25/E50。先报告online主要选择，再报告同一配置EMA结果；另选的EMA依赖配置单列。

| 数据 | 家族/配置 | online V Macro AUC均值±SD | EMA V Macro AUC均值±SD | 对B0/B1/H0最差格均差 | online选择 | EMA依赖选择 | 规则/理由 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

所有家族代表写出readout、参数和选择比较；两数据参数可按冻结搜索规则不同，方法形式保持一致。不用Test修改代表，不挑单seed最优配置，不把HPO配置当独立重复。

### 2.5 机制证据及竞争解释

对03的每项假设，使用下表给出是否已足以接受、缩小或关闭解释。具体结果引用D1–D4和完整控制，不单独靠余弦/variance改变判断完整算法。

| 假设 | 预注册鉴别观测 | 实测结果/区间 | 与哪一解释区分 | 当前接受层次 | 对算法的实际决定 |
| --- | --- | --- | --- | --- | --- |
| H1–H8逐项填 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

必须报告：signed BCE–PAIR及WD–PAIR二阶cross；当前D与历时D累积方向差；PAIR条件覆盖和自然频率；各格纠正/破坏/净变化；D2同状态完整函数响应；C1/C2的全程曲线。噪声更小、方向更持久或排序更稳如果没有完整AUC增量，写为诊断现象，避免将它们直接改写成排序机制成功。

### 2.6 完整匹配控制与方法收敛

| 数据/读出 | 比较 | 配对ΔMacro AUC | SD | 95%CI | 正向/5 | 最差格均差 | 解释与动作 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| UNRUN | C−B0 / C−B1 / C−H0 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| UNRUN | C−M | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| UNRUN | C−S | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| UNRUN | C−P | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

M报告本轨迹shadow、两个组的拟提交步范数匹配残差与`UNMATCHED_ZERO`次数；实际FP32参数位移另计。S覆盖说明长辅助核不是该增量必要条件；P_C1保留.99辅助历史，因此其覆盖只能说明独立来源分母已足够，不能据此否定长核。P_C2覆盖时按其mixed预条件历史与β=.9的共同变化解释。

在1e−4分辨率带内执行简化规则，同时展示区间；“当前矩阵未显示可分辨优势”不写成严格统计等价。若控制覆盖，采用08规定的简化代表或组件结案，不再为保住原方法名追加复杂模块。

### 2.7 EMA交互与阶段关系

对固定同一算法配置、数据、seed，报告：

\[
\Delta_O=AUC(C,O)-AUC(H0,O),\quad
\Delta_F=AUC(C,F)-AUC(H0,F),\quad I=\Delta_F-\Delta_O.
\]

同样给出以B1为父参照的交互。各自selected的四项来自各自最优epoch，标题必须标“部署选点交互”；共同E10/E25/E50另表为“同阶段交互”。

| 数据/父参照 | 端点口径 | ΔO | ΔF | I及配对CI | EMA−online | 含义 |
| --- | --- | ---: | ---: | --- | --- | --- |
| UNRUN | selected / common E10/E25/E50分行 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

online与EMA是同一条轨迹两个读出；不新增seed数或算两倍训练。不同epoch或不同超参数的比较不能被写成同状态统一机制。

### 2.8 MDMT作用范围与条件迁移

N范围报告`Cstar_N−H0_N`、`Cstar_SD−H0_SD`及差中差，附参数数目、原生None/零梯度频率、cell可达支持；不按绝对分数断言共享结构的必要性。

HyperGate报告四臂、原生objective、真实batch/epoch/update和同readout差值。说明βP/ρ按每步直接迁移导致的样本跨度变化。Gaussian报告准确GA31-R3版本、D0覆盖、噪声身份、四臂与同增强对照。技术身份无法恢复的模块只标技术暂停，不能写为无收益。

| 模块 | 触发规则及证据 | 实际配置/读出 | 关键配对差 | 六格保护 | 结论边界 | 下一动作 |
| --- | --- | --- | --- | --- | --- | --- |
| C12 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| N | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| HyperGate | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| Gaussian | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

### 2.9 冻结后的正式性能

两数据分别给完整绝对性能，发表基线按各自冻结online读出；新算法明确online或fixedEMA。同表包含B0/B1/H0、匹配控制及适用Gaussian内部参照。

| 数据 | 模型/方法 | 读出 | V Macro AUC | Test Macro AUC±SD | Macro LL±SD | Traffic AUC±SD | Pooled AUC±SD | selected epoch(2022→2026) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNRUN | M3oE / HyperGate / AutoIFS / 各内部参照 / 最终候选逐行 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

| 数据 | 冻结比较 | ΔMacro AUC±SD | 95%CI | 正向/5 | 六格平均ΔAUC | +.003门 | −.001格保护 | 最终裁定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNRUN | 最终候选−每个发表模型 | UNRUN | UNRUN | UNRUN | 六项均列 | UNRUN | UNRUN | UNRUN |
| UNRUN | 最终候选−B0/B1/H0及匹配组件 | UNRUN | UNRUN | UNRUN | 六项均列 | 不用作机制门 | UNRUN | UNRUN |

强发表参照为冻结的匹配发表模型集合中该数据Test Macro AUC五seed均值最高者；先固定完整集合，再按规定计算门槛，不按候选结果挑较弱参照，也不用逐seed最大拼接虚构模型。全部发表模型比较均展示。+.003与六格保护用配对均值裁定，CI和正向数独立报告。LL、Traffic和Pooled结果有其独立结论，不能抵消未通过的Macro AUC门。

### 2.10 计算与工程成本

列出实际墙钟、累计worker/GPU时间、全局batch事务、optimizer调用、标签曝光、评价次数、D2虚拟步/来源autograd/面板前向、显存峰值、主机RAM峰值、原生/新增active状态和诊断shadow状态。原HyperGate小batch与AutoIFS search/reuse分开，不把所有模型当M3oE更新数。

| 模块/模型 | 科学轨迹 | 父轨迹重建 | 技术attempt | 独立seed | epoch/search epoch | 全局更新 | optimizer调用 | worker秒 | peak CUDA/RAM |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

资源等待、OOM、状态丢失、被停止和重放原因逐项记录；重放成本不计为新增独立证据。模型/optimizer/shadow保持RAM_ONLY，报告只交付数字、hash、代码和日志。

### 2.11 最终研究决策

| 家族/最终代表 | 决策标签（08） | 已足够接受的结论 | 必须保留的具体边界 | 本轮关闭项 | 下一步动作 |
| --- | --- | --- | --- | --- | --- |
| C1 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| C2 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |
| 最终方法/组件 | UNRUN | UNRUN | UNRUN | UNRUN | UNRUN |

`METHOD_READY / CONDITIONAL_METHOD / EMA_DEPENDENT_COMPONENT / GENERAL_OPTIMIZER_COMPONENT / AUXILIARY_ONLY / NO_ACTIONABLE_INCREMENT`按08定义使用。一个数据上的完整积极结果按条件接受；另一个数据失败说明成立边界，不能使前者结论重新归零。若两个候选均无可行动增量，明确关闭本轮定义，保留既有EMA和A21事实。

后续机会只给一个最有价值的问题、它如何由结果导出、需要改变的算法对象和最小区别性比较。关系保留方向若成为下一机会，写成“同pair/同曝光下训练标签支持的历史正确关系保留”，不自动启动未定义的buffer/KD训练，也不把该未来方向写为本轮已验证贡献。

## 3. 独立实现审查模板

```text
review_id / reviewer_role / source_lock / review_date:
Scientific specification: 05 + 07 + manifest版本
Status: UNRUN
Blocking findings: UNRUN
Fixes and verified hashes: UNRUN

Checks:
- 原B0/B1/H0与候选零点的真实入口及单步状态一致性
- 同forward的PAIR/raw r/native grads与原gate detach
- WD只一次、native None、PAIR事件、私有时钟和bias correction
- C1 signed cross、C2更新v后构造D及U不被二次预条件
- own-trajectory M、S/P准确来源与局部支持
- shadow/诊断/RNG隔离、父状态RAM导出与实际恢复
- Validation选点、双读出、概率ties、six-cell全覆盖
- source/data/row/selection/permission hashes
- 当前资源门、预算、队列去重与技术重放账本
```

审查只补足会改变数学、数据、评价或恢复行为的问题；通过有限关键检查后进入规定训练，不增加镜像实现的无效测试。

## 4. 数值附录与缺失结果规则

每个配对差保留五个seed值、均值、样本SD、95%配对t区间、正/零/负方向数、六格均值差及最差格。n始终为配对seed数量；cell、epoch、batch、pair和重复attempt均不作为独立seed。完整五seed系数为df4的2.7764451052；未完成五seed的正式表标`INCOMPLETE`，不替换缺seed、不把四seed旧结果混入。

曲线至少包括online/EMA的Validation Macro AUC与六格、Macro LL及E10/E25/E50标线；主选点仍来自全部50epoch。探索性统计和预注册比较分开，按09的多重比较规则报告。均值表显示足够小数，全精度值以CSV为准。

缺失值用空数值+状态字段表示，显示为`UNRUN / NOT_TRIGGERED / TECHNICAL_PAUSED / TEST_PENDING_AUTHORIZATION / INCOMPLETE`；不得填0充当分数。每条缺失均注明模块依赖、已完成部分和唯一下一动作。

## 5. 上下文更新与有限结案

本次规划提交不向实验上下文写入新性能。未来真实实验完成后：

1. 在`01_AI_EXPERIMENT_INDEX.md`登记新实验编号、问题、状态与报告入口；编号先查最新索引，不预占已使用编号。
2. 在`02_AI_EXPERIMENT_DETAILS.md`写入实际算法、目标/参数范围、状态时钟、对照、预算、来源与阶段选择。
3. 在`03_EXPERIMENT_RESULTS.md`写入完整主要数值、六格边界、配对统计、Test暴露身份、成本和研究决定。
4. 人类阅读版由本轮`FINAL_REPORT.md`提供；不读取或覆盖未知的04人类文件。若未来另有明确维护任务，再按该任务权限处理。
5. 独立复核报告表格与原始CSV、链接和冻结清单；提交代码、文本产物及完整结果到执行任务授权的分支，不提交模型张量。

结案要求：核心S2全完成；所有实际触发模块均完成或有具体技术暂停结论；允许的Test已按冻结规则结束或明确权限限定；数值/来源/成本/报告可追溯；最终研究标签与下一动作明确。达到这些条件后关闭队列及该study，不启动新的开放式循环。
