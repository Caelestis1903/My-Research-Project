# GATS02 CODER IMPLEMENTATION REPORT

实现依据为已批准的 [DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md)。新增代码仅在 `experiments/gradient_aug_time_select_20261102/`；历史包、原始数据、旧报告、旧计划均未改动。来源/数据核验复用协调者已有收据。此阶段没有运行真实CUDA Stage0或正式训练，也没有分析研究性能。

## 文件与实现

| 文件 | 实际责任 |
|---|---|
| protocol.py | 固定数据/模型协议、有限arm配置、原batch/50epoch/FP32/Adam、ML精确H2 rho、日程、源码/数据/配置/控制身份、RAM snapshot/restore、selected/E10/E50身份行。 |
| objectives.py | A0/A1直接复用源单backward标量；B1 detached average-tie rank概率及独立keyed字段mask；R1每step/field精确数量和能量匹配；B2两个task分别按原B归一化的clean/delta、E/P整组FP64投影、两个task均值；R2训练冻结组缩放，U零差分实测断言；gradnorm直接读取wd前实际optimizer输入梯度。 |
| shadows.py | fixed、θ5初始化的delayed5、E31末起epoch均值SWA、100step CV adaptive、独立replay状态；不反馈online，默认最多3个shadow。 |
| run.py | 两数据集原样加载及身份核验；20条基础开发；每条A1/B及控制的独立rho完整重算；每step参数hash、最终Adam、行序和view完全匹配；逐卡85%资源门与RAM IPC、故障账本和最多2次可恢复重试；条件控制、有限trial、最多2候选确认；严格历史baseline身份核验；全模型冻结后统一test、数表、RAM释放。 |
| analyze.py | seed原值配对、均值/sample SD、n2/n3/n5 t95%区间、注册validation门、LL/最差/最大域保护、数表生成；不输出机制解释。 |
| checks.py / tests/test_semantics.py | CPU公式、rank、RNG、数量/能量、任务/B后均值、空域、预算、shadow与RAM恢复检查；独立review后的真实batch CUDA Stage0入口。 |
| README.md | 检查/Stage0/完整队列命令、历史数值入口、RAM生命周期、条件设计交接和原始日志解释。 |

复用接口包括 GA31-R3 `load_dataset`、fixed1 `make_model`、Gaussian `view_key/draw_view`、真实metrics；AIGM参数组和evaluation计时；GAM的deterministic setup、CPU copy、state hash、RNG/loader恢复、shadow递推和BytesIO/Pipe内存传输；调度沿现有逐卡资源模式实现本轮worker，不更改历史调度器。

## 训练、控制与时间机制

A0/A1保持源loss/backward运算次序；κ分别为ML .05、KR .1，MLrho精确使用exp(log(.5)/40)。B2单任务loss除原B，四个task clean/delta梯度加full-real共5遍，最后两个task均值一次得到原B×2归一化。权重衰减留在标准Adam内，投影参考和delta不含wd；U任何非零target delta立即阻断，不能静默丢弃。投影开关关闭直接走源A1一次backward。

B1概率只决定增强字段位置。R1按每step/field固定数量随机选择，并对当前Gaussian扰动作detached单标量能量匹配，不读取难度。源B1每目标样本p和row_id/global_step保存为逐epoch压缩数值NPZ；mask key、seed/hash、概率分位数、字段数量和能量保存在CSV，可按记录重建。NPZ中没有模型、optimizer或RNG状态。

首遍A1/B维护fixed/adaptive；完整rho序列冻结后，seed1729打乱整个逐step值数组，重算相同online路径，维护fixed/adaptive/replay三个独立模型。每step参数hash、最终Adam hash、epoch行序和view hash必须一致；重算pass的曝光、前反向、更新、验证和时间全计成本。首遍及replay有效端点在RAM合并，绝不把同一个shadow状态改名后比较。A0维护fixed/delayed/SWA；SWA E10明确N/A。

磁盘只写数值/配置/hash/日志。RAM状态包含model/Adam/shadow、adaptive deque/rho/SWA count、RNG、loader、epoch/step、selected/best和cost。selected/E10/E50保留到全体端点冻结后才统一test；test不进入开发/控制/候选规则，之后无训练派发。完成数表后释放RAM。

## 有限排程与条件路径

基础为两数据×两开发seed×五arm=20条，每条50epoch。每数据先单worker完成一条A0实测完整吞吐，之后按显存/功耗和RAM余量并发。控制+trial最多8条，优先R2/R1/R3，按dataset两seed一对；未有主增量/探索门或稳定注册LL/worst-cell权衡者不消耗控制。未匹配不晋级，不声称与控制相当。

实用default+control已过门时直接优先确认，不为填预算加trial。仍只过探索门的B2/B3、已超过匹配控制且剩余至少4条预算时，可执行η=.5或delay10及其自己两seed控制；η对照重新从η训练统计冻结，delay10采用seed1729冻结十个随机关闭epoch、余40epoch λ=.625。新配置需超过实用门和自己的控制，且相对default不落入.0005近似平局才替换。最多两个候选配置，不扩网格。成本超过calibrated开发预估1.5倍取消trial，2倍取消第二确认；10h停止低优先级派发，12h停止所有新派发、完成在跑项并对已完成轨迹冻结/test/汇总。

τ2公式仍未定义，未预实现。只有探索分支确有判别价值且还剩足够同配置控制预算时，写 `conditional_design_request.json` 并在RAM中等待协调者的reviewed resolution。`STOP`分支明确记录其理由；新增公式需要另行设计、实现、review和来源冻结后交接，当前代码不会自行猜测。该路径不是已完成τ2实验。

C1最多一个B，C2固定锚定A1 adaptive；B臂adaptive仅为诊断。确认seed2023/2025/2026，最多两候选、总唯一轨迹≤40。历史数值默认从本报告 `historical_aigm_baseline/` 读；先核验数据和冻结配置，并对当前开发A0/A1 online/fixed的selected/E10/E50参数hash和epoch逐一匹配。缺身份或不一致就新跑配对baseline，占总预算，不足则取消相应确认。历史测试数值只在统一test阶段导入，不增加seed数，也不恢复旧权重。

## 实际验证

- `python -m compileall -q experiments/gradient_aug_time_select_20261102`：通过。
- `python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`：**1 passed，1.54s**。单一assert检查集合覆盖FP32/FP64投影、半投影、无冲突/零参考、rank tie/singleton、R1数量/能量、独立mask RNG、空目标域无view、任务/B后均值归一化、三shadow独立/不反馈、adaptive rho/随机排列、delayed/SWA有效期、完整RAM恢复下一步、lambda总预算及门。
- 20基础job构造及唯一(dataset,arm,seed)、batch/epoch检查：通过，见 `basic_queue_checks.json`。
- `python -m experiments.gradient_aug_time_select_20261102.run --help`：CLI可导入并列出audit/check/all及外部code-commit、Stage0 receipt、history-dir参数。

公式数值收据见 `formula_checks.json`。唯一warning来自冻结源setup使用PyTorch旧TF32设置API，不改本轮FP32规则。

下一步由不同reviewer独立检查设计、实际实现和本报告。PASS后runner使用真实不可变代码commit执行 `--mode check`，在两数据集原batch验证A0/A1标准Adam精确相同、B1/R1能量/RNG、B2两个task/U/投影、所有开关退化、三shadow及完整RAM恢复下一步。该receipt须与`all`的manifest和commit相同。CUDA资源、真实U语义、整队列吞吐和研究结果仍由这些后续运行给出。

## 独立审查修订 R1

审查发现确认排程仅检查全局40条上限，历史baseline不匹配时可能新增12条A0/A1配对基线和6条候选，超过确认阶段12条上限。已修正为所有确认期新增基线与候选共同计入12条预算，并按dataset完整配对组分配；组无法完整容纳时取消该dataset确认，在decision中记录阶段/全局/时间上限、已有确认数及所需基线/候选数。确认派发前同时断言阶段≤12、全局≤40。

新增一个小型assert检查：两数据均不能复用历史baseline、一个B候选时，各需9条完整配对；只接受第一数据集9条，第二数据集按确认阶段12条上限停止。另核对阶段12条可容纳时仍受全局40条约束。算法、数据和其余实验范围不变。

修订后验证：`python -m pytest -q experiments/gradient_aug_time_select_20261102/tests`，**2 passed，1.40s**；未运行CUDA或正式训练。
