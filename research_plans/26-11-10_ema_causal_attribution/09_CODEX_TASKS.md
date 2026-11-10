# Codex 实现与验收任务清单

状态：**PLANNED / NOT_IMPLEMENTED**。本轮只交付规划；本清单供下一次明确的实现与执行任务使用。唯一运行入口为 [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)。

## 1. 执行总原则

先在固定源码工作树新增新包，再执行工程检查，最后启动20条正式轨迹。任何步骤都不能通过修改旧报告、换数据、缩seed或改训练语义来“通过”。遇到真正的实现/资源阻塞应修复可逆工程问题并继续；若超出冻结恢复预算，保留具体阻塞及已完成证据，不能伪报科学阴性。

模型状态禁止持久checkpoint。实现者必须先解决活状态依赖，再排队运行；不能把所有科学计算推迟到没有参数的最后阶段。

## 2. 依赖图与可并行开发

| 任务 | 依赖 | 可独立开发部分 |
|---|---|---|
| T00 来源/数据/协议冻结 | 本目录、固定source | 与T01纯数学、T05计数核并行 |
| T01 状态布局与数学 | T00实际layout | 独立于训练调度开发 |
| T02 只读训练观察器 | T00/T01 | 与T05/T06纯预测分析并行 |
| T03 RAM事务/调度 | T01/T02接口 | 与功能模块并行，最后联合验收 |
| T04 参数/时间干预 | T01/T02 | 不依赖完整正式训练 |
| T05 pair与统计 | 固定schema | 用小合成预测和真实人口维度验收 |
| T06 功能接口与融合 | T01/T04 | 必须用真实source图，不能另写近似模型 |
| T07 H桥接及来源诊断 | T02/T04/T05、08固定列 | 纯分析部分可并行；动作必须真实前向 |
| T08 CLI与工程验收 | T00–T07 | 串行关闭实际剩余问题 |
| T09 正式采集与block归因 | T08通过 | 最多2worker，按06资源准入 |
| T10 冻结/Test/报告/释放 | T09四块完成 | Test前所有科学定义与解释冻结 |

并行实现不得由多个worker同时修改同一在线模型或同一guardian记录。所有行为用不可变state_id/packet_id连接，不能依赖临时全局变量传递实验身份。

## 3. T00：锁定来源与数据

交付新 `protocol.py` 和CLI `audit/prepare`。

必须完成：

- 检查source SHA `ffa3de067fcda5a1293af83a465d741e069493f1`、新实现code hash、本目录plan hash、真实依赖版本。
- 只读读取main的三份总上下文及01/10的固定来源；旧停止门不得覆盖本轮最新规则。
- 解析04的20 jobs、真实rho/step/WD/loss/目标域，明确KR旧nested augmented_domain不生效。
- 复用源cache；不存在时仅在raw SHA与源预处理一致时重建。核对行、split、编码、scaler和数组hash。
- 重建或复用原P_A/P_B/V_mech及row-hash V_A/B；32768诊断pairs按dataset×seed无放回冻结，同seed目标间复用。
- 编译所有epoch末尾、anchor−h、H2 matched pulse开始/结束和来源诊断的观察事件表，含引用计数与状态保留截止。
- 在训练bundle中不加载Test数组。Test专用加载器由 `evaluate` 调用，`freeze`成功之前拒绝执行。

验收产物：SOURCE_RECEIPT、DATA_MANIFEST、PANEL_MANIFEST、GROUP_SUPPORT、RESOLVED_PROTOCOL、OBSERVATION_SCHEDULE、EXPECTED_OUTPUTS、PREPARED。所有计划未运行项标planned；不能用旧报告结果填入本次观测表。

## 4. T01：状态数学与分组

交付 `history.py`、`interventions.py`的纯状态核心。

必须支持：

1. 原FP32 O/F逐坐标复制，S/N与六个细分族无遗漏/重叠；domain embedding按行拆；gate/tower索引 `3*t+d`。
2. 实际 `R64=F32.double()-O32.double()`，四窗口old/recent，h_eff去重、R0零项；每步另维护一份理想FP64 R审计舍入，不改源F。
3. 已知常量/交替更新短序列的符号、指数、初始化与窗口闭合；这些是数学工程检查，不是训练收益实验。
4. source实际支持审计与KR补充：cell(d,t)只支持S/Dd/Tt/DTdt；普通图与完整函数导数的结构支持均正确。
5. 不把参数差浮点重构误差当成真实干预。所有状态有parent/background/mask/window/state_hash。

验收：布局counts与源一致，S/N端点精确引用原状态，数学残差按07通过。错误支持关系必须在正式采集前修复；不能靠参数名称猜测。

## 5. T02：采集一次训练轨迹的所有必要信息

交付 `trajectory.py` 与只读观察器。

原模型、loss、Gaussian noise key、loader、单次Adam、EMA顺序直接继承05；不重写等价但浮点顺序不同的目标。每步观察后，在线参数、optimizer、RNG、loader游标、buffer保持原值。

必须采集：

- O/F每epoch完整Validation与各自selected端点，F选中时同时保留同刻O；
- 每step固定面板的真实在线p/z，流式同rho历史预测EMA；
- 每个scheduled `t−h` 的R64和当时在线面板分数，不只留R；
- 固定anchor所需R/old、q预测、核心B/C状态；
- 每5epoch一epochpulse及H桥接面板实际预测；
- 每个pulse结束时，在仍可读取该F参数的现场计算一次P_A完整函数rank梯度，按08§4归约all/S/N供两套菜单共享；只保留摘要，不保存梯度，也不能等block结束后补算已释放的E15/E25等状态；
- E35/E50 H步matched pulse两个向量，起点为F_(t−H)，首更新从t−H+1；
- Clean E20/E35最后一个真实step的有限来源包：只常驻pretheta、gdata、mhat、vhat四份FP32向量和步计数标量；gWD和实际更新从这些与锚点O派生，不另存向量，以便五seed后按决策选择是否运行虚拟对照。

对C3不保存每步参数；对逐步预测也不无界落盘。源步前后score差、pair损害及窗口参考分数可流式/按事件计算；窗口端点只持久化最终anchor所需p，按06计入约70.8MB上界，不持久化全部scheduled epoch×h的p/z。固定对齐数组和统计足以支持分析后，释放冗余副本。

验收：在工程smoke上观察开/关的真实在线轨迹一致；q_t与同样时间权重的显式短序列展开相同；没有遗漏step或跨epoch误用时间单位。

## 6. T03：RAM guardian与长程调度

交付 `state.py`、调度器和恢复流程。

必须实现：

- 不可变anchor packet、按需只读借用、增量IPC，不在每个epoch深拷贝所有历史anchors。
- guardian只保存当前条件块和最终三份FP32端点；块内五seed齐后运行B/C/D/H最终化，然后释放old/anchor/moment。全局解释冻结后才加载Test。
- 准入计入FP64向量、当前worker、待提交副本、可选H2/source包；4GiB是参数payload预算，还需实际RSS满足host90%规则。
- writer按2GiB总观察预算记账，原子写预测/数表收据；参数、梯度、m/v、R、可恢复运行状态不写磁盘。
- worker失败从guardian最近完整事务恢复；guardian丢失只能同seed从初始化重放；每job重启/重放上限与物理成本记录清楚。
- `run`不能因为先跑seed结果弱而停止其他seed；条件分支只在该条件五seed核心完成后决定。

验收：一次真实小规模worker故障注入可恢复到相同epoch状态和后续结果；guardian丢失模拟只需验证计划重放路径与去重逻辑，不为工程测试运行多条完整50epoch训练。内存峰值使用实测RSS与tensor_bytes，不只参数理论大小。

## 7. T04：B/C干预与预算选择

实现03的S/N四态、四个recent/old四态；每个状态都是真实forward。先完成全部五seed core，再用V_A确定最多两个需细化大类，V_B单列确认。

B细化最多12额外状态/anchor；只覆盖 E10/E35/E50/tF。C按最多两个重要组、两个背景、主半衰期窗口执行，不做完整2^12组合。结构支持允许的叶组预测重组必须在审计后启用；不成立时记录范围并用实际有限替换，不偷换机制。

验收：每个state都能从packet+descriptor唯一重建；O/F共享端点被去重但逻辑比较保留；独立作用/删除作用/交互按07闭合；统计先seed内汇总；已有所有五seed不因CI/正向seed个数被丢弃。

## 8. T05：准确AUC、pair转移和统计

优先复用源AUC/LL评价；新模块负责精确错/平/对二维转移，整数计数用int64。禁止建立完整P×N矩阵。

实现07的Fenwick/二维支配核，优先利用现有可用编译/JIT条件；若只有Python核，也必须用真实完整人口规模实测CPU成本并纳入06预算，不能用采样AUC冒充准确人口结果。分数完全相同、真实ties、全正确/全错误、不同margin但相同rank等小算例必须与显式全pair计数一致。

同时实现无放回固定panel pairs、margin分层、纠正/破坏重叠、five-seed配对均值/SD/t区间、分层与LogLoss独立报告。用户+.003算法门不得放到机制干预gate。

验收：全人口转移计数=PN、加权净量=源ΔAUC；core同刻与own-selected分表；没有按pair/epoch扩seed数。提供 `ARTIFACT_INDEX.json` 将07逻辑表映射到实际压缩路径，报告脚本只通过该索引加载，避免同义文件名遗漏。

## 9. T06：真正的功能性解释

### T06a 完整接口

实现两处cut：STAR+skip的h必须同时馈入全部gate和experts；fused_dt必须按真实tower/routing送入下游。四态00/11必须闭合到实际原状态，两个混合态真实前向。

### T06b 专家融合成分及X连接

在E35/tF对O→F捕获真实g/e/domain/task，计算content/routing/cross/domain/task五项。以真实F融合输入为背景逐项删除、真实tower前向。按五seed V_A选一个top成分；在主要组/时间contrast `F−X→F` 重新计算同一成分并删除，检查X纠正的同pair是否丢失。全程每anchor最多6额外状态。

融合五项的FP64代数恒等式与源FP32融合舍入分开审计：五项之和对FP64重算差闭合；实际fused差与该和的rounding residual单列。实际干预背景始终是真实源u_F。

### T06c 完整函数积分

仅scratch启用完整gate函数导数，训练detach保持不变。用JVP而不是逐pair全梯度存储；如果某算子不支持forward AD，在有界小诊断集用reverse AD替代并记录成本。不得因此改主训练图。8/16/32节点有界加密，实际端点/有限差分/积分残差全部记录。

验收不是“表示距离有变化”，而是同一个X、同一组pair、同一contrast的参数替换→函数项→margin跨零→净AUC证据可连接。若积分不闭合但真实接口因果链成立，保留已识别机制并标积分边界，不无限追加积分点。

## 10. T07：自适应失败桥接与有限来源对照

严格按08§4实现两种ridge特征和无拟合direct carry：每菜单六非F动作池化标量回归，训练72行但只有12时序单元；P_A未来效用为训练target，P_B/V只评分，ridge惩罚定义固定。不得重用旧48维向量后默默删除字段，也不得让实现者临时选择新模型。

在pulse结束时采集一次F的P_A完整rank梯度并归约到all/S/N，梯度不落盘。两套七动作共用F，参数状态用03构造，所有future效用来自真实读出。

H2若决定方向B，执行已预留H步、共同scope、相同半径的最多4状态/anchor判别；机制动作必须相对F本身有价值。零公共半径/函数不可辨认记无信息，不当阴性。

若近期来源证据会改变方向C，使用已预留Clean来源包做两种虚拟Adam方向对照；与实际组位移范数匹配，其他组保留实际更新。候选只在scratch上前向，真实Adam新增0，不能伪装完整训练算法成绩。

验收：区分oracle、跨行选择、train可见选择、相对静态动作；不读Test；没有未来target进入origin输入；H1/H2/H3的支持结论绑定具体动作类和条件。

## 11. T08：CLI、必要工程检查与准入

公共命令按00/05实现，全部接 `--plan-dir/--out`；prepare接数据/cache路径；run/resume接设备与max-workers；smoke接dataset；analyze接stage。`--help`能说明未来运行行为与恢复状态。

工程检查只覆盖真实风险：源轨迹观察开/关、tail batch、组支持、FP64数学、ties转移、完整cut端点、完整导数、融合重构、原子事务和一次恢复。使用seed2022、每数据Clean/Gaussian的小预算只是正确性检查，报告 `ENGINEERING_ONLY`，不得作性能筛选。

在此阶段用真实完整Validation人口计时一个核心状态和转移核，用固定面板计时每step读出/一个JVP/一次fusion删除。按06公式估算全矩阵；不能只按空模型或训练batch时长估归因成本。

通过后冻结code hash、环境、表schema、资源策略。source仍不修改；若正式期间必须修科学计算bug，保留旧attempt、标影响范围并使用新attempt，不拼接不一致证据。

## 12. T09：一次连续队列完成训练与归因

执行00的长程命令顺序。条件块每五seed齐全后自动运行Stage2/3依赖，用户不需要半夜作科学选择；所有触发由预注册规则和真实五seed表决定。

`EXPECTED_OUTPUTS`逐项显示核心、条件触发、未触发、资源阻塞状态。定期输出进度、剩余任务、CPU/GPU/RAM、物理步数和估计完成时间。每块完成前检查数值闭合和Q1/Q2最小证据链，防止把漏采关键函数证据拖到释放后才发现。

核心完成条件：20×50epoch、34500正式Adam、四条件每项五seed完整；selected O/F与同刻anchor身份可追溯；B/C/D已在状态活着时完成；观察输出与模型payload在预算内。没有任何性能淘汰提前停止。

## 13. T10：冻结、Test、报告与释放

按08形成MECHANISM_FINDINGS和主方向，写MECHANISM_FREEZE，锁定分组/窗口/feature/菜单/报告模板；再由独立evaluator读取Test预定五态。

报告自动填真实表值、来源ID、五seed覆盖、交互与边界。正式性能AUC/LL独立裁定；同刻机制不按+.003门淘汰；未触发不记阴性。最后确认FINAL_REPORT、关键表、FREEZE/Test收据、数值闭合和COSTS均完整，才释放guardian并写RELEASE_RECEIPT。

未来只提交新实验代码、报告与可审阅小数表到独立执行分支；原报告只读，模型不推Git。不自动把后续结果写进main长期上下文，除非另有整合指令。

## 14. 最终验收不是“所有脚本退出0”

逐项检查：

- 能从Q1追到实际参数/时间干预及对应五seed，而不是只有幅度排名。
- 能从Q2追到同一X的完整函数变化、同pair的margin和净纠正；融合项若未连接成功，报告的解释范围相应收窄。
- Q3使用实际历史窗口/目标/排序或同状态更新证据，不预设噪声/过拟合/负迁移。
- Q4能对照具体历史版本，区分指标、预测器、动作与可利用空间。
- Q5给一个主要下一阶段方向及证据条件；允许明确选择D，不为求新算法继续追加。
- 五seed、Test隔离、source训练纯度、无checkpoint落盘及旧结论不覆盖均通过。

若只有“哪种AUC更高”“哪个组影响更大”，即使训练和表格齐全，也应在仍有活状态时完成T06而不能宣布研究完成。若已有主要因果功能解释且剩余细节不改变研究决定，则按计划结束，不再无限补实验。

