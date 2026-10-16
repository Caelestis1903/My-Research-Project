# DESIGN SPECIFICATION — Stage 2+ execution

2026-10-16，主代理。源规划 `../FINAL_STAGE2_PLUS_RESEARCH_PLAN.md`，来源提交 b17cb2707fc2448f09f21fd12279b2b3856c7756；AGENT.md 来源 main@357fa409。设计先于实现。

## 指令冲突与范围

最新用户要求严格遵守本地 start.md。其§实验验证原则11禁止非机械运行子代理，覆盖早先AGENTS的多代理职责安排。本轮采用主代理DESIGN→IMPLEMENT→SELF_REVIEW→RUN→ANALYZE，**没有独立review**。最终规划中的独立review/科学gate不能因此标为通过；不启动正式训练。10月15日输出路径是旧模板，以本轮规划10月16日目录为准。保留start原GB18030字节，不覆盖旧产物。

原工作区.git无提交；复用历史发布仓库的独立worktree，以源规划提交为基底，新分支codex/execute-2026-10-16。工作区源码/数据只读，新增执行目录和隔离数值脚本可写。WP1/WP2无法恢复的旧工件明确INCOMPLETE；新B1不替代旧S6。

## WP1/WP2与Stage 3

1. 逐项扫描S4/S5/S6/S6ML/S6-R2/S8/R2及B1重建；按实际文件路径记录config、manifest、metrics、prediction、checkpoint、optimizer、parameter manifest、selection的存在性、大小。原始大权重仅索引，关键小工件及旧审计receipt验hash；不重复全工程审计。
2. 复用旧AO04 test预测/指标重算及bootstrap，核对输入和结果hash；缺validation row IDs不能声称same-row identity通过。
3. 从B1已验证4800行cell指标复算selection、same-epoch/selected、timing双锚点恒等式、每seed/arm/epoch的val-test Spearman及六cell排名一致性。仅新增原规划要求的rank agreement；不改变R0–R4，不用test选checkpoint。产出独立CSV、轨迹及排名图；不新增训练。
4. 旧S6 exact selected/grid与bootstrap缺口保留，不能用B1补齐。

## WP3 / Stage 2A

用原始论文/作者代码核验M3oE、AdaTask、PCGrad、CAGrad、FAMO、Adam、AdamW、importance sampling、MDMT adaptive fusion/weighting、SAF、checkpoint selection。每篇记录规划14字段；未知字段写NOT_REPORTED/NOT_VERIFIED，不猜。输出机器可读novelty matrix及Q1–Q4回答；检索同构机制，尤其sampling×optimizer既有工作。没有检索到不等于全球创新。

## Stage 2B/2C有限pool零训练补充

旧C2.1具有两数据集×4 seeds、初始化θ、每域16真实row、全坐标trace和所选坐标covariance。先核验receipt，复用不重跑。新补充只填其缺失的ESS、曝光/权重分布及**多步state propagation**，不改变训练模型或加载旧checkpoint，不估计selected AUC。

在同一个旧pool和相同init上重建每row两个任务loss均值的梯度。逐一验证init hash/pool IDs/选中坐标和旧case相符。保存48×K所选坐标梯度、theta、标签/域/row IDs和source hashes，以便纯numpy复算。K沿旧case坐标（C/P各至多32，A全部4）；不把所选坐标结论推广全模型。

alpha∈{1/3,p_train}，q∈{p_train,1/3}，w=alpha/q。B∈{16,40960}，paired streams使用相同multinomial RNG种子（common random numbers、不同q产生不同count），4096个独立replay×64步。loss/gradient固定θ；每步计数、真实期望/协方差、梯度均值SE、ESS=(Σw)^2/Σw²、域曝光与缺域率保存。fixed-p与batch-cell Equal由旧oracle及解析式复核。

给SGD、Adam相同梯度流，fresh m=v=0、step=0，β=(.9,.999)、eps=1e-8、C/P lr=.01、A lr=.001。coupled decay=1e-5，以固定θ加入g；无clipping。m/v按递推，bias-corrected moments、一步delta、累计delta逐步记录。累计delta是**固定θ replay的积分，不是沿新θ重新求梯度的训练**。AdamW仅作为decay数值对照，decoupled decay亦取固定θ；不做另一训练臂。对相同确定性gradient stream两q结果必须相同；one-step/64-step递推与真实torch.optim在detached坐标副本、WD=0时核验至1e-12。WD非零时用每步重置parameter到固定θ的torch oracle核验；明确不是正常优化trajectory。

关键反证：SGD样本路径也会因噪声而分离。不能用Adam路径差异>0、SGD路径差异>0推翻/证明机制；比较**ensemble mean**及其SE。SGD的期望累计delta相同；Adam的E[vhat]与gradient variance相关、E[delta]是否不同由数据决定。报告四model seeds及replay sampling uncertainty，不把4096 replay当训练seeds。

最小门：精确期望相对残差≤1e-10、MC均值偏差≤8SE、torch公式残差≤1e-12、model hash和原grad/tau不变。任一失败先停数值case。跨seed/阶段不稳定只能停止推断，不增加救援网格。所有结果仅有限pool/初始化/所选坐标；不得给2D正式训练GO。

## 后续条件

2D：2A–2C、WP1/WP2和独立review都完成才可进入；本轮不以自审替代独立门。Stage4/4B依赖有效S5 lineage和同θ shadow histories；Stage5有checkpoint/配对身份才做敏感度；Stage6/E1按原条件，未触发标NOT_RUN，不伪造NO。最终三报告、status JSON、来源/结果hash、关键数据/图表提交并推送，远端hash确认。
