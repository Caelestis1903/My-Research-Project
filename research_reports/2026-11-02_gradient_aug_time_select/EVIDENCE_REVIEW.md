# 本轮证据入口与协议裁决

执行入口为本地 `start.md`（GB18030）及 `research_plans/26-11-2/06_Codex完整执行提示词.md`。本轮采用 ponytail full 与 stop-that-shit change。范围固定为 A0/A1、B1/B2/B3、注册时间平均和条件控制；不新增候选或参数网格。

GitHub main 实际读取 HEAD 为 `0bf3e970ca47da44b129e67bb4c84a653ba6a20b`；最新证据分支 `26-11-01_v3` 实际 HEAD 为 `6f5742c2085c1e49d1767b0a7079bb9f28160eda`。本地为恢复文件树，没有 `.git`。计划从已连接 GitHub 恢复，来源和代码冻结分别记账。后续发布在最新 main 上增量创建新分支，保留远端已有提交。

已读取本轮 00–06/YAML、前置 26-11-1_v2 00–06、根 AGENT/PROJECT_ARCHITECTURE，以及 AIGM01 报告、搜索报告、设计、实现、证据审查、独立审查、运行报告、decision 和原始数表/源码。45 个计划及 AIGM 顶层源码/报告与远端 Git blob 一致；另外 45 个执行依赖与 AIGM 执行源码清单 SHA256 一致，根两份文档与 main 字节一致。详见本目录三份 SOURCE/ROOT_DOC verification。六数组、自然行号、encoder/scaler、原始 CSV 和域人口均已通过现有加载器严格核验，见 DATA_VERIFICATION.json。

历史 AIGM01 的模型/optimizer/shadow RAM 已释放，本轮重新训练规定的 A0/A1。历史数值可用于已核验配置相同的确认配对，不将历史结果合并成额外 seed。历史 test 已参与方法判断，本轮开发仍只使用 train/validation。旧 MASK 的高层文档和当前 Gaussian 代码可查；旧 MASK 完整实现和 Polyak 原始日志没有恢复，B1按本轮公式重新定义。

历史经验起点：ML EMA 五 seed 排序增益稳定但有 LogLoss 代价；KR EMA 总体排序改善，而普通增强存在最大域损害及任务差异；SAD/AIG-M 没有达到注册增量门。新候选须以当前训练评价，门非零、投影同向或难度相关本身不能作为收益证据。域规模和难度分别观察，不以小域代替难域。

合同裁决：06、最新03和YAML统一 A0仅固定/延迟EMA/SWA，A1/B维护固定/adaptive/replay；04中A0 adaptive描述为旧条款。20条基础开发、最多8条附加（匹配控制优先占用）、最多12条确认，总数不得超过40。匹配控制按数据集成对并按B2>B1>B3优先级派发；主增量或注册权衡仍需判别时才消耗预算。预算未覆盖的匹配比较直接标明未测，该候选不得晋级或宣称等同控制。确认需要真实同seed配对，新增基线也计总预算。

GPU0 有其他用户训练，GPU1初始空闲；不干预外部任务。使用指定 phase0-3-env Python 3.13.5 / PyTorch 2.9.0+cu126，两张3090实际可见。正式训练前完成独立review和两数据集真实CUDA Stage0。模型、optimizer、shadow及RNG快照仅RAM，磁盘保留数值、配置、日志、代码与报告。
