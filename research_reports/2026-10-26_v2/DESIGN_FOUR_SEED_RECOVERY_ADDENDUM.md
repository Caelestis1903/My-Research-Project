# DESIGN SPECIFICATION — 四训练种子与中断恢复修订

日期：2026-10-26。状态：DESIGN_COMPLETE，待独立实现与审查。修订来源：用户最新明确要求“无论哪个实验都必须至少进行四种子训练的数据证据才允许进行进入结论分析和报告撰写”。此要求优先于原方案的两开发种子预算。本文是执行设计，不包含实验效果分析。

## 1. 四种子硬门槛

RCF S0/S1/S2统一开发种子固定为 **5201、5202、5203、5204**，两个数据集均完整执行。新增5203/5204由本次用户要求授权，在查看其结果前登记；不更换原5201/5202，不按结果挑种子。S4原独立锁定种子 **6201、6202、6203、6204** 原样保留，只有四开发种子的S2/S3登记门槛成立才运行。

每一个进入科研结论或正式结果报告的训练条件，在每个数据集内至少有四个不同训练种子的完成证据。比较双方必须覆盖同一四种子。native、pilot及共享prefix属于对应种子的真实训练；不能以一个prefix重复训练、折数、task、epoch、bootstrap、随机reader重启或重复推理替代种子。不得把重放计成第五个独立样本。

现有5201/5202的S1/S2表、S2_DECISION以及由其产生的文字判断均为 **SUPERSEDED_TWO_SEED_PROVISIONAL**。保留为执行历史，不可转为最终结论。收集/审查原始日志、实现说明及进度收据不构成科研效果分析，不受“先四种子”的禁止；不能借此在技术报告中偷写两种子效果判断。正式四种子分析前不再解释两种子指标。

原实用阈值全部不变：每一开发种子ΔMacroLL<0；四种子等权平均ΔMacroLL≤−0.0005、平均ΔMacroAUC≥−0.0005、同一cell四种子平均ΔLL最大值≤0.003。实现中把“both_seeds_negative”语义改为“all_seeds_negative”，不能仅检查前两个。群体交叉要求四种子对应P_form对齐后的同向支持，不能继续使用两种子交集。S2/S3决策器缺一个种子或必需方法时返回BLOCKED_INCOMPLETE，禁止NO_START/晋级/最终freeze。

此前已经实际开始KuaiRand的K1_WIDE和SOURCE_INIT_RANDOM，现将这两个条件登记为 **四种子证据补全的必跑子集**，即5201–5204均执行。四种子S2若另触发更多已登记S3方法，执行两者并集；没有新触发则不扩其他对照。这样不会用最新门槛选择性抹去已经开始的条件。MovieLens尚未开始上述S3条件，仍只按四种子S2触发。S3每个实际用于分析的方法须齐四种子后才能判断；未触发方法写NO_START，不写效果结论。

SCALE Dconst01_NO_AUG已在每数据集2022–2025完成四种子，保留所有八条完成证据。此前2022/2023的资源扩展判断只作为历史执行来源，不再当两种子科学结论；正式SCALE比较使用2022–2025四种子且核历史对照同四种子身份。若某历史对照不足四种子，只记录缺失，不对该条件下效果作结论。数值恒等式、shape/梯度/导出一致性等实现验收不是训练性能实验，无需把一次解析核验扩成四次训练。

## 2. 中断事实与状态恢复

父编排者已由dmesg确认2026-10-26 19:25:04发生global OOM，KuaiRand seed5202的自有worker PID2002650被杀。其模型/Adam/RNG均只在RAM，已丢失，不能把已存在数据日志冒充可续训状态。其余三个worker仍存活。

优先使用正常、可审查的现有RPC保存允许的数据并保护RAM。然而当前已加载controller没有升级Campaign方法/配置的正常RPC，现有reader reload仅能更新worker，不能让controller合法接纳5203/5204。禁止gdb/ptrace、任意代码注入、伪装数据为模型归档或未经审查的运行时替换。因此允许记录成本后**有序关闭仅本次任务的service及三个自有worker，在新四种子artifact根受控重放**。不得终止或修改外部VLLM/其他用户进程。

在新目录 `artifacts/mdmt_rcf_20261026/v2_four_seed` 运行，旧v1完全保留为历史证据。四种子有效配置保存到本实验新代码/配置目录，不改取得的09计划原件，不覆盖旧config hash。新配置记录父配置hash `c3ba96191f76e638c734c91ddd6b28f723a5a2bdc4412ea4fd08185024270b43`、修订文件hash及明确的seed/storage/root变化；数学公式、角色划分、数据比例、所有学习率/epoch/batch/选点/λ/预算保持不变。

重放5201/5202的native selected、native epoch24、pilot state hash与旧v1逐项核对；总体source/config hash因修订而变化，不能强称完全相同身份。基准预测/选择在可得原始数据上也记录相等性。必要的物理重放时间/标签访问如实记账，统计单位仍是相应唯一seed。Hash不符先停止对应路径查明原因，不静默接受为等价复现。

## 3. 只外移允许的实验数据，解决已证实的RAM问题

禁止任何磁盘模型、optimizer、RNG或训练恢复包，包括临时、改后缀、pickle/NPZ及额外swap文件。允许将逐行输入、h/z/φ、群体、预测、功能输出和标签访问数据缓存为普通NPY/mmap，文件以row/role/schema/hash绑定；这些数据不能恢复模型权重/Adam。

实施最小的明确白名单数据缓存机制：

1. native每角色h/z产生后立即写只读数据mmap，再进入pilot。不要同时保留原全量h/z的第二份常驻内存。
2. PilotAccess原始隐藏激活 `_cache` 在产生时保存为mmap并返回同dtype/shape/order逐值相同的数据。标准化数学公式、FP64统计及float32返回不改。可按原模型确定性重算后清理纯输入派生cache，但不得清理后续导出依赖的模型。
3. LabelLedger.unique是每个(method,role,task,purpose)对全部数据行的bool访问记录。以文件backed同shape bool数组或经独立证明等义的压缩集合替代常驻np.zeros；新key才分配，避免Python setdefault每次调用无条件制造巨型零数组。保留entries的原顺序、count、new_unique_rows、row_hash、passes与summary逐字相同。不得把unique_rows降成count近似。
4. 形成FeatureBank.values、每方法的probabilities/terminal_probabilities、single-function诊断预测等逐行大数据可在阶段边界外移mmap。必须更新所有实际持有它们的同一对象/字典，防止隐式副本继续占RAM；读者系数、normalizer小数组、函数module、模型/optimizer/RNG对象不进入通用序列化。固定内容FeatureBank现已有的完成后释放策略保持。
5. 数据mmap以分块写入/校验避免构造同尺寸副本；写完flush后释放原数组。允许对仅属于数据映射的页使用安全的回收提示，禁止操作模型内存。每阶段GC/torch.empty_cache只释放未引用缓存，不释放仍需模型。

模型与优化器仅驻RAM；为降低GPU闲置保留成本，可将已冻结且闲置的module留CPU并在原设备计算前恢复，前提是tensor值逐位不变且不会改变原Adam设备/执行或引入训练状态序列化。不是必须实施项，优先用数据缓存足够完成。

并发继续按单卡显存和功耗均<85%启动；新增主机内存准入是为防已发生OOM，不改变batch或算法。启动额外worker前须有可用RAM足以容纳其尚未mmap的峰值工作集并保留至少8GiB系统余量；先测实际缓存后的单worker峰值，若无法满足则等待/分波，不能硬开八个常驻大数据副本。代码应避免在prefix全部建立完之前就被巨大未外移cache耗尽内存。

若全部八个context在mmap后仍不能安全驻留，允许**两遍、确定性批次重放**：第一遍逐批完成S1/S2（以及已登记必跑KR两control），只保留完整未评分外层预测/功能数据、所需决策Q摘要与模型身份hash；四种子齐备后统一决策。第二遍仅为第一遍释放模型的context按已冻结决策重放，核hash后执行触发的方法和全部raw-X/推理/导出验收。外层数据在全局freeze前仅允许产生未评分预测文件，禁止读取外层y作指标或调参；第一遍释放依赖之前必须明确登记第二遍重放义务，不能把缺失outer/功能导出合理化。优先实现mmap足够的单遍方案，不预先强制两遍。

## 4. 实施分工和最小验收

Coder A拥有 `protocol.py`、`prefix.py` 及新增data-cache helper：数据mmap和ledger等义外移。Coder B拥有 `campaign.py`、`cli.py`、四种子有效配置和恢复/运行接口：完整种子计数门槛、旧必跑control并集、OOM失联清晰诊断、safe memory admission、按方法幂等避免已完成训练重复；必要的数据缓存阶段调用与A协调。二者不能修改对方模块，均不得分析效果。独立reviewer读本设计、实际diff及各coder report；全部blocking修正并PASS才开始新正式训练。

最低检查只覆盖可能改变结论的路径：

- 同一小真实角色输入/原函数与mmap前后逐值相等、dtype/shape/行顺序/array hash一致；同seed重放native/pilot hash核对；不修改公式或随机计划。
- ledger原实现与新实现在包含重复row、多key、多pass、空访问的同一访问序列中每条rec及summary完全一致。
- 四种子decision缺一返回BLOCKED；全四种子方向规则及原阈值明确生效；旧二种子决策不能授权freeze。KR两个必跑control需要四种子完整。
- 每context实际方法清单、各method训练完成收据与四seed清单一致；避免把共享baseline推理重复算作训练。
- 运行首个prefix/阶段记录RSS/可用内存及数据cache清单；在持续推进时没有再次出现已知的per-key常驻巨型bool增长。不得为资源问题静默降batch/epoch或变采样。
- 全部正式导出仍执行原raw-X≤1e-6、指标复算、源/功能/成本导出要求；训练模型文件始终0。所有外层评估仅全局冻结后执行。

最终由主编排者在相应方法每数据集四训练种子完成后才做科学解释与正式结果报告。四seed样本单位、均值/SD/seed配对区间及六格/逐域/逐任务均按原规范；缺失条件标NOT_RUN/BLOCKED，不将缺失解释为算法失败。完成所有触发阶段、验收、分析、清理与原分支交付的目标保持不变。
