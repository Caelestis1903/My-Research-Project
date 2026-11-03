# 06 Codex 独立执行提示词

> 这是供用户随后交给Codex的执行入口。本次规划会话没有执行正式GPU训练；不要把规划文件中的候选、预算或CPU检查当作训练结果。

## 你的任务

在 `Caelestis1903/My-Research-Project` 中实现并执行 `research_plans/26-11-03/` 的冻结方案。研究轴仅限MDMT针对性信息增强、梯度与真实历史更新，不恢复差异化共享/原型主线，不继续修补v3的动态选择＋ECCM，不自行增加候选。

先读本目录01、03、04、05及experiment_queue.yaml；02提供候选自批判，SOURCE_INDEX提供固定来源。数学以03、工程以04、阶段门以05为准。完整理解后再编码，不能让runner自行选择算法定义。

## 必须继承的事实

固定无增强非反馈参数EMA已经具有重复selected AUC收益，是本轮A0强基线，不是等待发现的新问题。loss尺度、ANTI2、SAD、AIG-M、GATS、EMA02V2的正负结论按01继承。小域不是困难域，不安排复验，不把规模当难度代理。

v3分支固定 `a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288` 的11月3日补充已经完成P01/P10新增8条训练。KR动态选择与ECCM在当前配置中有重复负交互，单组件均未稳定超过A0；PCV/PCOLL及S3仍无正式性能结果。不要沿用原S1时期的“P01/P10未执行”，也不能把未运行控制描述成失败。

## 实现范围

优先H：真实与域任务条件增强probe共同检验并修复一份旧共享moment。备选T：同一当前样本/同一增强索引在当前与前一步完整参数处求梯度，运输旧moment。M/U_MIX为可直接胜出的普通Adam增强方案。F只作设计备选，不实现、不自动触发。

必须实现03精确定义的donor优先级、虚拟嵌入位置、自然anchor权重、H投影度量与活跃集求解、T质量因子与全S裁剪，以及H_REAL/C_MIX/H_U/H_NORM/T0/T_U/T_NORM/SCALE控制。不能换成梯度差分EMA、原版PCGrad、普通增强Adam或每任务多套moment后继续沿用H/T命名。

新包 `experiments/mdmt_history_information_20261103/`；报告 `research_reports/2026-11-03_mdmt_history_information/`。从固定v3提交提取实际依赖，保留vendor原字节和manifest，不合并整条实验分支，不修改历史报告。

## 执行顺序

1. **读取与实现审查。** 检查现有代码、源manifest、数据路径/身份及规划文件一致性；建立新包与04要求的CLI。输出设计到实现映射和coder说明。由另一个未参与编码的reviewer检查关键数学和状态语义；不能伪造独立审查。发现错误只做针对修复，不借机重新设计候选。
2. **CPU与S0。** 运行本目录CPU代数参考以及04规定的真实模型旁路、梯度、状态恢复和完整batch短smoke。源A0/A1与新旁路要匹配。仅检查当前接线，不重跑已接受历史科学结论。模型、optimizer和checkpoint只驻RAM。
3. **D0。** 两数据seed2022执行A0/A1/M/T0/TM/H_REAL/H_MIX/C_MIX/SCALE完整中间矩阵；可证明身份的旧A0/A1数值必须优先复用。所有正式轨迹50epoch，不早停、不改batch。
4. **D1与补2024。** 按05确定每数据最多一个家族；三组各四个固定配置，同预算调参；每组只给胜者补2024。未选家族是预算暂缓，不是阴性。
5. **D2。** 仅当05开发门触发时执行该家族至多六种匹配控制、两seed；T交互必须用同λ/τ/κ配置；简单方案同等或更好时选简单方案，不堆组件。
6. **CONFIRM。** 冻结最多四arm，再补2023/2025/2026。全部选择与选点冻结后统一test，不提前读test，不从test追加配置。分别报告两开发seed、三个本轮未开发seed、五seed合计。
7. **报告与提交。** 按下方交付清单完成分析，推送执行分支 `26-11-03_history_information`。若该分支已经存在且属于本计划，按已完成身份续跑，禁止覆盖或重复训练；若属于其他任务则停止写入并报告分支冲突，不强推。不要把原始结果擅自合入main。

## 不可更改的协议与预算

MovieLens/KuaiRand自然数据、原行序80/10/10、train-only编码与scaler、fixed1 M3oE、3域2任务、embedding16、4共享专家、FCN[512,256,256,64]。batch40960、50epoch、FP32、AMP/TF32关闭。源Adam默认lr=.01、β(.9,.999)、eps1e-8，coupled wd ML1e-4/KR1e-5；A0有限搜索按05允许明确lr/wd变化。固定EMA rho ML .9828205985452511、KR .995，不搜索rho。

主端点fixed_ema selected；同时保留online及E10/E50。六格AUC/LL、macro、traffic、pooled、worst必须齐全。选点只用validation，最早严格最大macro AUC；E0不参与。test全部冻结后统一访问。

正式请求槽位上限96，先按科学身份去重。S0不算正式训练；失败重放单列成本且每身份最多一次有原因的基础设施重放。不得新增seed、延长epoch、扩大搜索或因结果接近门槛自动追加实验。禁止以小域难度、EMA有效性、一般缩放规律为新主问题。

若某数据某阶段不达门，只停止该数据对应后续阶段，其他数据继续。未执行记NOT_TRIGGERED，预算未选记DEFERRED_BY_BUDGET，成本阻塞单列。不得因为一个配置失败否定整个方向。

## 完整性与计算资源

Python优先 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`；原CSV路径见04。检查数据hash和已核验源manifest，缺失或不匹配是实际阻塞，不能随意重建另一个人口继续。

按每张GPU的实际空闲与S0峰值安排并发，不杀其他任务、不缩batch。日志包括全部forward/autograd/probe/旧点计算、optimizer/EMA计数、同步训练步耗时、worker耗时与峰值显存。禁止把多worker时长总和称独占GPU时间。

checkpoint/参数/优化器不落盘；原指标CSV可以引用，但不能在没有原参数时声称本次前向复核。RAM丢失如实记录，不能伪造恢复或隐藏重放。源A1保留EMA02V2随机key，不因新study名改变噪声。

## 分析与交付

必须交付04列出的manifest、实现映射、coder/reviewer记录、CPU/S0、run/decision/freeze记录、原始逐seed六格与全部端点、配对统计、成本表、REPORT.md及QUESTION_DISPOSITION.md。

报告开头直接回答：当前接受哪个工作方案；是否胜A0f和有界调参A0b；增强是否必要、作用在当前梯度还是旧历史、方向性是否超越同范数缩放、另一任务条件是否有增量；是否有LL/traffic/具体cell代价；简单方案胜出时删掉哪些组件。区分经验结果、机制解释与适用范围，不将非零历史/低方差等同泛化收益。

阶段门是预先冻结的研究决策规则，不是绝对统计证明。给出逐seed差、均值、样本SD、95%配对t区间和方向数；不把任务/epoch当额外样本，不因区间宽无限补实验。沿用已接受历史结论，只对可能改变当前选择的不确定性作本轮授权的控制。

你可以因明确实现错误修复代码，但不得自行决定关键算法、添新候选或变更门槛。遇无法解决的真实阻塞，保留已完成结果并准确说明；不得编造完成状态。
