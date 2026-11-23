# 01 — 研究上下文与实验事实

## 1. 第一阶段报告身份

第二阶段提示词指定的第一阶段文件：
research_reports/26-11-23_multi_mechanism_discovery/00_RESEARCH_REPORT.md
及其三个支持文件，在当前 main 检索中未找到。

当前实际存在的第一阶段交付：
research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md

本方案以实际存在的报告为主要研究依据，并在 10_SOURCE_AND_REUSE_REGISTER.md 保留路径差异。不得伪称读取缺失文件。

## 2. 第一阶段研究发现

第一阶段报告列出 M1–M8 候选机制：
- M1 共享更新诱导的专有梯度一致性；
- M2 样本×任务条件影响；
- M3 阶段性与组合式训练影响归因；
- M4 域任务条件有害特征依赖；
- M5 语义合法多标签保持增强；
- M6 输出响应/敏感度条件化控制；
- M7 跨 seed 分布影响归因；
- M8 推荐任务冲突控制器。

第一阶段报告给出了候选排序，但本轮执行策略不继承其主次等级。八个家族全部纳入，进一步拆解为多个可执行变体。报告中的“候选排序”仅作为来源记录，不是本轮资源准入或实验删减规则。

## 3. 最新 main 上下文

权威文件：experiments/reports/EXP_AI_CONTEXT_V2.md。

关键事实：
- Equal 与 Sample 是不同有效训练目标。
- Sample 下普遍 residual negative transfer / Safe-MTL 主线未达到稳定实用门槛。
- Equal 下 KuaiRand tab1 存在条件性 target rescue。
- S5 将 Equal 下 d1 rescue 定位到 persistent Adam history 整体载体；S6 切换 Sample 后两个已测最大域 target 均未通过 +0.003 practical gate，HT/DH Phase B 未运行。
- raw-gradient norm matching 与 applied-step norm matching 均未复现 Equal 下 rescue。
- 小域不是困难域，域大小不能独立解释全部现象。
- 固定 EMA 的收益已被接受，不重新证明。

这些事实作为本轮控制条件与结果解释边界，不用于预先排除 M1–M8。

## 4. D2 最新分支事实

分支：26-11-23_v2。相对 main@f66bf17cba2980618e6f2edd1c7381b73c35cac5 ahead 1 / behind 0。

D2 S1：
- 2 datasets × 5 seeds；
- 10 条 baseline trajectories；
- 30 anchors（E10/E25/E40）；
- 570 forks；
- Test evaluations=0；
- SCOPE_DECISION=COMPLETE_NO_GO_S1；
- V_A 仅 KuaiRand P_AXIS 有初筛资格，V_B 独立确认失败；
- confirmed_datasets=[]；
- V_B P→U mean Macro AUC：MovieLens +1.13e−5，KuaiRand +5.27e−5；J_ACC 分别 −1.15e−6、−8.20e−5。

正确解释：D2 固定 private scope / 固定更新顺序不晋级。不能将其扩大成共享—专有反馈不存在。M1 的 P consistency objective、M8 的 shared-gradient aggregation、M2 的样本条件影响和其他家族均需独立测量。

## 5. 本轮执行策略变化

此前版本将 M1/M2 与 M3–M8 区别对待。本版取消该策略：
- M1–M8 全部执行；
- M3 不依赖 M2 阳性结果；
- M4/M5 不止做审计，真实可行的变体进入干预与训练；
- M8 纳入多种冲突控制变体；
- S2 对所有候选执行 V_B；
- S3 对所有可训练变体执行短程轨迹；
- S4 对所有通过机制与短程门的变体执行完整训练，不设 Top-K。

## 6. 固定实验规则

- 数据：MovieLens、KuaiRand。
- 模型：M3oE fixed1。
- seeds：2022–2026。此集合由第二阶段提示词明确指定，并与 D2 S1 一致；优先于 main 上下文“未另行预注册时”的默认 2022–2025。
- baseline 50 epochs；batch 40960；自然尾批保留。
- Validation 六格 Macro AUC 选点；Test 只在冻结后评价。
- 正式性能门槛：五 seed paired Macro AUC ≥ +0.003；每格平均下降 ≥ −0.001；LogLoss 独立评价。
- 局部机制诊断不强制 +0.003，但必须与真实排序响应分开报告。
- 六格、anchor、fork、sample panel、pair 不是独立 seed。
- 正式发表基线 M3oE、HyperGate、AutoIFS；新方法不得当作发表基线。
