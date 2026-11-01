# 06｜Codex完整执行提示词（唯一执行入口）

你负责把本目录的研究合同转化为可复现的分阶段实验实现与执行记录。必须先完整阅读本目录00–05文件，再读取仓库源码和原始报告。不得自行改变研究问题、算法公式、数据身份、选择规则或预算。不得直接开始正式训练。

## A. 仓库与来源核验

1. 仓库：Caelestis1903/My-Research-Project；目标分支main；本规划目录research_plans/26-11-1_v2/。
2. 核验main和26-11-01_v2当前远端HEAD，记录真实commit SHA、时间和读取路径。历史快照SHA只作对照，不能冒充当前HEAD。
3. 阅读GAM01的REPORT、DESIGN_SPECIFICATION、IMPLEMENTATION_REPORT、REVIEW_REPORT、全部指标CSV、diagnostics_summary、decision、phase1_validation_decision，以及gradient_aug_momentum_20261101完整源码和resolved_config。
4. 阅读GA31-R3实际依赖、数据manifest、source receipt、历史Polyak相关原始记录、AGENTS.md及目录内规范。对关键结论复算seed级配对差；缺失文件明确记录，不得推测。
5. 核验research_plans/26-11-1_v2目录是否与main上本规划一致。保留历史报告和旧代码，不覆盖或重写历史研究结论。

## B. 第一阶段：证据与协议冻结（不训练）

生成执行报告，明确：当前已知结果、数据和代码身份、指标语义、历史超参搜索证据、可复用缓存、需要重跑的原因、配置与成本ledger结构。修正报告中已指出的MovieLens LogLoss改善seed计数勘误时，只在新审查记录注明，不修改历史报告。

核验默认设置：自然80/10/10行序划分、fixed1模型、3域×2任务、原Sample权重预算、batch40960、50epoch、FP32、seed2022–2026、loss_scale=1。ML/KR历史coupled wd分别1e-4/1e-5。不得改变人口、标签、任务/域权重或网络结构。

## C. 第二阶段：基础优化器与增强配置有限搜索

严格按04、05的低预算默认合同执行。先实现trial ledger与配置hash，再生成有限配置列表；不得临时扩网格。开发seed固定2022和2024，每trial完整50epoch。Adam lr/wd、Gaussian κ/λ、SAD βδ/λ按04给定候选范围。βδ=.90是A1代数等价控制；beta_delta=0是当前实现的普通混合Adam分支哨兵，不可解释为差分通道零记忆。

对每个trial记录数据/源码/配置hash、seed、完整epoch和optimizer step数、forward、增强视图数、autograd遍历、GPU时间、峰值显存、validation开销、失败/重跑及缓存复用原因。OOM只按原运行协议降低并发重试；不能减少batch、epoch或seed。

## D. 第三阶段：算法实现

主候选AIG-M按03的公式和退化规则实现；第一版只动态调节λ_t，βδ固定，以避免时变β的偏差修正错误。创新度按预定义参数组计算，阈值由训练统计预先确定，不读取validation/test。weight decay进入真实梯度通道，差分q不含weight decay；混合梯度的二阶矩按既定公式更新。若当前SAD基线的实际语义与合同冲突，暂停并报告差异，不擅自选择一个版本。

AIG-SWA是备选，不与AIG-M默认并行实施。只有AIG-M因明确机制/实验原因停止，且协调者批准后才实现AIG-SWA。AIG-SWA只改变shadow时间尺度，不反馈online训练。

所有组件必须有独立开关、配置验证、可复现状态、必要的退化测试。沿用现有数据/模型/指标/RNG接口；不改历史实验包，不新增无关依赖，不全局monkey patch。权重与checkpoint按GAM01 RAM-only约束保存；磁盘只写数值、配置、哈希、日志和报告。

## E. 第四阶段：独立审查与Stage0

提交实现后由独立reviewer按03第6节逐项审查公式、状态、退化关系、随机流、恢复语义和成本记账。review未PASS不得训练。

Stage0只做真实模型短检查：数据身份、clean/aug梯度、Adam/SAD一步、AIG-M公式、shadow更新、RAM snapshot/restore下一步一致性、显存和随机流。Stage0不评价test、不作为性能结论。通过后才进入开发seed有限搜索。

## F. 第五阶段：开发搜索、候选选择与正式确认

按05阶段顺序执行。任何方法只有在开发seed的validation均值达到预设增益且无保护项时才晋级。完整50epoch训练，不按E10自动剪枝。选配置使用开发seed均值，不允许每seed单独挑超参。AIG-M筛选失败则停止，不自动转入更大搜索；AIG-SWA只在明确条件下作为备选。

正式五seed阶段必须覆盖A0、A0+EMA、A1+EMA、AIG-M+EMA；若候选未晋级，省略该臂并记录。固定配置机制比较与各自调参后的性能比较分开报告。新方法不能只超过原始在线Adam，必须超过最强普通增强+EMA参考并通过保护项。

Test只能在搜索、公式、超参、checkpoint选择规则、代码与配置身份冻结后统一评价。不得边跑边看test。最终报告主端点为validation选点后的test六格macro AUC，并同时报告LogLoss、pooled、traffic、worst cell、域/任务分解、共同epoch和seed配对区间。

## G. 研究去留与交付

按05中的GO、TRADEOFF、NO_PROMOTION、STOP规则裁决。区分调参收益、增强收益、EMA收益和新更新规则独立增量。机制未完全解释不否定可靠经验效应；固定配置失败不等于整个方法家族无效；有限搜索失败后也不得无限扩展。

最终新增本轮独立报告目录，不覆盖历史文件。交付至少包含：
- 证据审查与seed级复算；
- 实际读取的commit、数据和源码hash；
- 完整trial ledger与预算账本；
- 实现规格、代码开关、公式和必要测试；
- Stage0、开发搜索和正式实验各自的报告；
- 所有validation/test原值、配对效应、保护项与最终研究决策；
- 可复现运行入口和完整配置。

每个阶段结束先汇报结果与是否满足下一阶段门槛；不得将规划中的候选、尚未运行的实验或条件性收益写成已完成成果。默认允许将本轮新增报告和代码提交GitHub，但必须遵守仓库现有分支和归档规范。