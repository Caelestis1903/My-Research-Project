# 2026-10-16 Stage 2+ 执行报告

状态：**PARTIAL — 零训练工作已完成可访问范围，后续正式训练受科学前置条件阻塞**。不能将整份规划标为全部实验完成。新增正式训练0次；新增有限pool数值case 8/8；复用已有B1重建8次训练，不重复计数。

## 来源、指令与环境

从Caelestis1903/My-Research-Project的`codex/execute-2026-10-15@b17cb2707fc2448f09f21fd12279b2b3856c7756`拉取完整10月16日目录（实际只有FINAL_STAGE2_PLUS_RESEARCH_PLAN.md）。AGENT.md来自`main@357fa4098a69889e844cb9debe07bf43bf7d0092`；该文件不在规划分支。来源/原始blob/SHA见source_receipt.json。

本地start.md为GB18030；保留原字节，以正确编码完整阅读。其10月15日输出模板与本次明确10月16日目录冲突，使用本目录。start.md§实验验证原则11禁止非机械运行子代理，按最新用户明确要求覆盖早先AGENTS的多代理分工：主代理设计→实现→SELF_REVIEW→运行→分析，没有派发子代理，**没有独立review PASS**。原规划的科学/独立门不伪造通过。

原工作区Git无commit且全部旧文件未跟踪；没有初始化提交整个工作区、覆盖旧文件或改原分支。独立发布worktree为`/home/xzf/Code/phase0-3-20261016-git`，基于规划提交建立`codex/execute-2026-10-16`。

复用phase0-3-env：Python3.13.5、Torch2.9.0+cu126、CUDA build12.6、cuDNN91002、NumPy2.1.3、pandas2.2.3；驱动570.144，两张RTX3090。起始/home可用246GB；数据根目录/home/xzf/data，两数据集hash与历史一致。完整环境见environment.json。所有新数值/取证运行CPU，CUDA_VISIBLE_DEVICES为空；GPU未被占用或停止其他作业。后续训练若通过科学门，按每卡显存≤90%独立准入，不要求两卡同时空闲。

## 执行清单与依赖判定

| 顺序/节点 | 实际完成内容 | 状态及未完成项 |
|---|---|---|
| WP1 / A0 | 1457项工件索引；353项旧结果/receipt hash全部匹配；重新检查372个仍缺显式历史路径；恢复5个AO04内嵌state | PARTIAL：S4/S5/S6/S6-R2/S8原始链条不齐；372是路径数不是实验数；不能由外置checkpoint缺失推断所有内嵌state也缺失 |
| WP2 / A1/A2 | 复用旧预测重算/test bootstrap；3个KR run完整validation预测重放绑定；补6cell validation bootstrap；B1 dense表/双锚timing复核 | PARTIAL：原S6 exact逐epoch预测/完整state仍缺；AO04仅一个完整B1配对seed，ML验证预测缺失 |
| WP3 / 2A | 15篇原始来源的14项协议字段、同构机制novelty matrix、Q1–Q4和理论taxonomy | COMPLETE_TARGETED_REVIEW；未核实项显式标记，不是全球创新认证 |
| 2B | 复用旧C2.1全坐标trace/期望oracle；新增ESS、权重/曝光、B16与40960、所选坐标covariance；8case通过 | COMPLETE_FINITE_POOL_ONLY；不代表full-data/训练阶段/正常曝光协议 |
| 2C | 8case×4096条独立replay×64步，SGD/Adam/AdamW；m/v/bias correction/delta；56hash与32新旧covariance核验 | COMPLETE_FIXED_THETA_PROJECTED_ONLY；没有新的正常theta训练轨迹或persistent-history因果 |
| 2D | 检查2A–2C/WP1–2/审查条件 | NOT_RUN_UNMET_PREREQUISITES；WP1/2未闭环、无独立review，正式四臂协议/guards没有满足GO。不是观测到无效，也不要求先有2D收益才能做2D |
| 3 | B1既有8run、4800cell指标，新增epoch/cell val-test rank agreement、轨迹图、选点及timing分解 | COMPLETE_RECONSTRUCTION_FORENSIC；旧S6身份未替代，R0方法增益无稳定实用支持 |
| 3B | 原S6-R2 exact/grid及已有CSV重算可复用 | INCOMPLETE：无原预测，不能新做原S6 paired bootstrap或exact重选 |
| 4 | 检查S5 A0/A1门 | NOT_RUN：原S5来源/完整state未恢复，不能启动16run C×A矩阵 |
| 4B | 检查同theta shadow histories条件 | NOT_RUN：无有效同轨迹shadow state；没有拼接异轨迹m/v |
| 5 | AO04 B1/2022 selected与last_observed，192row×task局部敏感度、rho/专家差/gate诊断 | PARTIAL：只一个完整B1 seed、两个state；无全epoch/全seed覆盖；B3/B4保持暂停 |
| 6 | 依次判定domain/ratio/task/backbone前置门 | NOT_TRIGGERED：局部数值相互作用不足以确认正常训练selected机制；无独立setting |
| 7/E1 | 核查强Sample、4seed、独立setting/holdout、因果与查新门 | NOT_ALLOWED：未建立全部条件；未开发方法或新增基线网格 |
| Git | 归档代码、三报告、关键数值表图、来源/状态索引 | 实际提交/远端凭据见publication_status.json |

## 实现与最小验证

先有design_specification.md及AO04补充设计，后实现，复核文档为implementation_report.md、implementation_review.md、ao04_recovery_review.md。新代码都在本目录，旧实验源码/配置/数据/权重未修改。

数值脚本复用原StageS1M3OEAdapter，不是V1/V2。现有baseline含四learned fusion scalar及main/arch两个Adam；这些是实际旧底座，本轮未恢复或新增。loss为固定alpha的task-mean BCE、q域采样及alpha/q权重；无DBO、gradient balancing、augmentation或clip。新oracle只在detached坐标/固定theta推进状态，不调用训练fit。main与A的两种lr并不等于重放原epoch-level架构更新日程。

先做语法检查和64步torch公式oracle，最大绝对残差6.19e-15，再跑8个case。实际case均exit0、总producer用时238.91秒；数据/初始参数/梯度原状态不变。不同case沿CPU持久化队列执行，启动命令、PID、source SHA、状态见numeric_queue*.json；其运行时源码以base commit+内容SHA绑定，发布commit随后归档这些确切字节，不伪称启动时已有新代码commit。

## 失败、修订与恢复

AO04 attempt1：两个B0 validation全量预测通过；B1/2022 local analytic导数通过，但last-state h=.01有限差分失败，未计成功。attempt2使用三固定步长、保持原容差，B1检查通过；随后B1/2023缺config终止。attempt3按原state/NPZ hash复用3个完整预测匹配，缺config仅记录state结构并继续ML独立项；最终5state索引、3完整匹配、192probe完成。原partial输出和日志保留，不覆盖。

h=.01最大局部差5.06e-5，至少一个较小固定步长的最大残差4.67e-6，解析chain最大残差2.18e-8；与有限邻域非线性/float32影响相容，未宣称定位了具体ReLU分支根因。ML的validation `.downloading`文件为0bytes，未作为有效预测。初次GitHub REST tree请求403 rate limit后，使用已可用git fetch读取完整源树；没有依赖该失败请求补造内容。

## 恢复入口

当前无活动训练。先读取execution_status.json和gate_decisions.md。若需复算新数值，使用已保存gradient_pool.npz运行纯replay/聚合；原case目录拒绝覆盖，独立新attempt须保留原结果。命令、脚本和数据依赖见reproduce.md。要推进2D/4，先补齐明确列出的历史工件/审查条件，不修改旧hold或把已有test当新holdout。
