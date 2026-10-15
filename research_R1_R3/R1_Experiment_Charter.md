# R1 Experiment Charter — AO-04 非均匀他域融合

日期：2026-10-11（Asia/Shanghai）。版本：v1.1，独立 Reviewer 最终判决 **PASS（设计合同）**；执行准备度 **NOT READY**。

## 1. 范围、当前状态与授权边界

本次只交付 R1：代码与 baseline 审计、假设审查、可证伪实验预注册。没有实现新模型、没有运行训练、没有执行推理扫描、没有产生 R2/R3 结果。下文实验臂是未来任务的设计约束，不是本次执行授权。不得自动进入 R2/R3。

研究方向沿用 R0 第一候选 AO-04。AO-09 没有明确稀疏部署目标，排除在本 Charter 之外。在线亲和矩阵、EMA 亲和估计、选择性删除、curriculum、momentum 干预和额外正则项均不纳入。

执行准备度 **NOT READY**。必须分别核实设计审核、代码与数据身份、checkpoint 可用性、baseline 复现；通过设计审核不等于完成 baseline 复现。现有可见 R0 输入为六份正式报告，其引用的三份 work 报告未找到，九份阅读要求未全部满足。该缺口如实登记，不以重建材料冒充原报告。审计16个run所声明的112个checkpoint文件均缺失，selected可加载0/16；本次没有baseline前向复现。独立审查只批准可证伪合同，不宣布资源就绪。

## 2. 研究问题、假设与证据边界

核心问题：在当前强 Sample-M3OE、相同数据/初始化/优化/选模协议下，保持总体他域融合系数后，非均匀源域相对权重能否带来可重复、有实际意义且没有预先禁止退化的 AUC 收益？

- H0：所限定的静态非均匀权重族没有满足预注册实用门槛的净收益，或收益可被简单目标域权重、无意义固定权重、共享门重参数化对照解释。
- H1：训练期学习相对权重在固定配对协议下带来超出所测简单替代的实用AUC信号。两seed不能充分区分随机波动；本设计不穷尽新增容量差异，因此不能据此宣称超出所有容量解释的算法收益。
- Hpost：冻结模型后的质量守恒比例重分配能改善独立测试性能。Hpost 与 H1 是不同假设，不能互相替代。
- Htask：任务条件化的权重相对仅目标域条件化权重有增量收益。B1−B2 才是对应对照。

H0 是有限模型族与有限预算下的实用假设，不是“任何非均匀共享都无效”的普遍定理。当前 AO-04 性能证据为 **WEAK**：代码可证明等权结构，但没有本次干预性能结果。Sample 历史无稳定实用负迁移不能证明 AUC 已无提升空间；Equal 的局部 rescue 不能支持 Sample 下的新方法。

废止 R0 原置零阳性/阴性推论：删除通道造成损失只证明当前模型依赖该通道，不能证明等权不是最优；删除影响小也不能证明重新训练的非均匀权重无效。三个对象必须分开登记：置零为描述性依赖测量；冻结重分配为后处理效果；从头重新训练为算法效果。

## 3. 代码定义与质量守恒约束

以 Stage S1 适配器的真实计算图为基准，D=3，T=2，N=4，输出索引 i=tD+d。令 E_d 为域专家、S_n 为共享专家、Q_t 为任务专家融合；四个全局标量为 aD/aT/bD/bT，均由原始 raw 参数的 sigmoid 得到。

原始域项：R_d=bD·E_d+(1−bD)/2·(E_j+E_k)，其中 j<k 且 j,k≠d。逐格表示 F_dt=Σ_n g_dt,n·S_n+aD·R_d+aT·Q_t。共享 gate 只作用于 S_n，不直接独立控制 E_j/E_k。所有原主干、门、塔和四个全局 raw 参数保留。

新域项：R_dt=bD·E_d+(1−bD)·[ρ_dt·E_j+(1−ρ_dt)·E_k]。ρ=0.5 时恢复原融合；own 系数 aD·bD、总体 other 系数 aD·(1−bD) 的代数含义不变。保的是系数总和，**不保证表示向量范数、分布或预测值不变**。训练中 bD/aD 仍按原协议学习，各臂终值可以不同，不能把“同一参数定义”误写为“最终数值相同”。不得补乘 D−1、重新归一化 own+other 或删除原计算路径。

源域顺序全局固定：d0 的(j,k)=(1,2)，d1=(0,2)，d2=(0,1)。所有 seeds 使用同一语义顺序，报告必须记录。

E_j是模块索引，不代表只由源域数据训练的信息通道；所有域专家对同一目标样本STAR表示dense计算，也经回收接受其他域梯度。学得ρ不均匀不能直接识别真实域因果亲和。共享gate只直接作用共享专家，但共享表示、任务专家、STAR与非线性tower可能间接补偿均匀域融合。系数守恒也不守梯度量；须记录专家输出尺度、α/β轨迹和门熵。

## 4. 数据、baseline 及训练协议冻结

主数据集只使用 KuaiRand-Pure natural；d0=tab0，d1=tab1，d2=tab4；t0=is_click，t1=long_view。不换数据集寻找正结果。MovieLens 只用于当前 baseline 审计，不进入本次最小性能设计。

B0 选定 Stage S1 的强 Sample joint 族，不能混用旧 examples 目标、SD reference、S6 的 T/HT/DH 干预状态或 S8 continuation。Sample 是对 batch 中全部样本与两个任务的直接平均 BCE；不能再乘 p_d，也不能改成均匀采样域。模型配置与具体源文件、数据 SHA256、run config 指纹以 Baseline Protocol 与审计清单为准。

固定协议：embedding=16，FCN=[512,256,256,64]，expert_num=4，batch_size=40960，num_workers=0，shuffle=True，drop_last=False，50 epochs，无 scheduler。主 Adam lr=0.01，架构 Adam lr=0.001，二者 weight_decay=1e−5、betas=(0.9,0.999)，其余默认项需在未来配置显式记录。原四个架构标量保留每 epoch 一批训练数据的原更新路径。**历史 patience=10 是 shadow 记录，不实际中止；所有臂都保留该行为，不擅自新增 early stopping。**

新增B1/B2的q及Bcap的θ统一进入主Adam，lr=0.01、wd=1e−5、betas=(0.9,0.999)、eps=1e−8，每普通train batch更新一次；不进入arch optimizer。原四scalar仍仅进入arch optimizer，原main参数归属不变；B3比例为不求梯度的buffer，optimizer外。每步清空全部梯度，两optimizer参数交集为空。此为新参数的预注册优化协议，不是历史已有事实；阴性只限此学习率/更新族，不作充分探索断言。不得试跑多种optimizer归属或调整lr择优。

未来全部臂固定当前runtime：Python3.13.5/Torch2.9.0+cu126/CUDA12.6/cuDNN91002、numpy2.1.3/pandas2.2.3/sklearn1.6.1，同硬件与版本清单。S1历史Python3.8.20/Torch2.0.0/cu117不同，历史数字不能拼入paired B0。当前优先方案是同runtime匹配重建B0与全部臂，不声称严格恢复S1 EDDAenv；若未来找回原权重欲复用，必须在新test揭示前修改合同并重新review，不自动削减预算。

每个 seed 内从同一原始 B0 初始化状态拷贝原参数；新增零参数不得提前消耗 backbone/数据 shuffle/dropout 的 RNG。主训练 batch 序列、架构更新 batch 序列、数据预处理和所有随机流逐臂配对，保存各自 hash。不能只凭同 seed 声称初始化和曝光已经相同。

选模仅用每 epoch 验证集六格 macro AUC；保留历史“严格更优才替换、同分取最早 epoch”规则。未来必须保存 selected 模型状态及完整 config/hash，训练完成并冻结 checkpoint 后才进行测试。final/common epochs 的分析为事先固定的次要描述，不得替换 selected 主结论。

匹配重建B0验收：源码/数据/配置/split/encoder/hash完整；50个epoch验证日志完整；selected确为earliest validation max；所选模型严格load state_dict并在相同validation replay，原预测max_abs≤1e−6，逐格AUC/LogLoss及macro差≤1e−6；不通过即INVALID，先review，禁止扫描。原历史test指标不是当前runtime重建的硬性成功标准。baseline和新臂全部选模manifest及所有扫描ρ必须在首次读取新test结果前冻结；对历史test曾被研究使用的事实如实披露，本次只能保证不按新test改方向，不能称全新外部holdout。

## 5. 实验臂与对照

| 臂 | 定义 | 新可训练参数数 | 主要用途 |
|---|---|---:|---|
| B0 | 原始 Sample-M3OE，ρ固定0.5 | 0 | 唯一主 baseline |
| B1 | 每(d,t)一个q_dt，ρ_dt=sigmoid(q_dt)，初始q=0 | 6 | 任务条件静态可学习他域权重 |
| B2 | 每目标域一个q_d，两任务共用ρ_d，初始q=0 | 3 | 更简单的域条件权重是否足够 |
| B3 | 每(d,t)固定无意义ρ；一个预置表跨seeds共用，无学习 | 0 | 任意非均匀性/固定先验控制 |
| Bcap | ρ固定0.5；每(d,t)共享gate增加一个温度参数，初始温度1 | 6 | 同新增参数数的活跃门控重参数化/优化对照 |

B1/B2 都是无输入的静态参数，不是样本条件门。T=2 时他任务只有一个，改变他任务“相对比例”没有对象；本设计不改任务专家融合。

B3 使用独立numpy.random.RandomState(10404)，按task-major六格抽一次Uniform(0.1,0.9)，两个训练seed复用同一表；不通过任何验证/测试指标挑表。具体浮点表在下文附录锁定。B3与B0不初始等价，其作用是所测一次无关系偏置，不是参数容量匹配；一张随机表阴性不能排除所有随机表。

Bcap固定τ_dt=exp(θ_dt)，θ初始0，只将共享gate的原pre-softmax logits z变为softmax(z/τ)，保留emb.detach、原linear和原域/任务融合；不作用最终概率。N=4时六个θ须真正有有限非零梯度。逐epoch记θ/τ/门熵；任意τ/预测/loss/参数/梯度nonfinite即停止run及队列并标INVALID。ρ<0.01或>0.99，τ<0.1或>10只作饱和描述，不触发clamp、改lr或重抽表。

Bcap的温度可吸收进已有gate Linear的W/τ与b/τ，因此6个新增活跃参数不是6个新增可辨识函数类自由度，也未扩大该共享门函数类。它是同新增参数数量的优化/路由尺度重参数化对照，**不是容量匹配证明**。B1胜过Bcap不能排除域融合新增表达能力或其它容量解释。B1−B2检验任务粒度；B1−B3/Bcap仅检验所测固定偏置与重参数化替代。仅B1−B0阳性不足以证明亲和必要或新颖性。

## 6. 冻结比例扫描：独立后处理对象

只有真实可加载且 baseline 复现合格的 selected B0 checkpoint 才能扫描。读取历史 predictions 不能重建专家输出，不能用其冒充可扫描 checkpoint。

冻结全部模型参数与推理模式；每目标域两任务共用一个ρ_d。候选集合 G={0,0.1,0.2,…,1}，包含原始0.5。每域每档用该域两任务 validation AUC 平均评分。为防选中单任务改善同时损害另一任务，仅允许该域两格 validation ΔAUC均≥−0.003的候选。分数差≤1e−6时优先0.5，其次距离0.5最近，再其次ρ较小。若无可行非均匀项，选择0.5。每 seed 单独在验证集选一个三域配置并冻结。

这是三个独立11档选择，有效组合空间11³=1331；33个边际分数并不意味着只有33种选择风险。只能在全部ρ/config/checkpoint锁定后对独立 test 评估原始0.5及唯一选中配置，不能在test上再次扫描或改选。保存全部验证曲线、选中ρ、baseline验证分数、test配对差及验证−测试差异。

ρ=0或1仍保持other总系数，只将其中一个源的系数置零；它与直接删除一个未重分配通道不同。推理期置零仅可作为另行标明的描述性诊断，**不列为性能闸门，也不扩展本预算**。

扫描阳性只支持“冻结模型的验证选择后处理在所测test有收益”；扫描阴性不否定B1/B2训练。扫描不得用来选择B1/B2初始化、学习率、额外模块或重新分配调参预算。

## 7. 主指标、次指标、guardrails

每seed最终主端点：selected checkpoint独立test六格macro AUC，M_test=(1/6)Σ_dt AUC_test,dt；ΔM_test为同seed候选−B0差。另定义M_val与ΔM_val为已选checkpoint的validation六格macro，用于未来资源筛查。两者不得混用：方向/投入筛查只用validation；test是全部方法、selected、ρ和预算已冻结后的最终泛化检查。macro是逐格AUC均值，不是pooled AUC。

训练和模型选择只用 validation；test 只做已锁定协议的最终评估。学习权重各臂主指标并行报告，B1−B0为主比较，其余为解释与证伪对照，不能按test挑一个赢家再宣称预设主法有效。

次指标固定：六格validation/test AUC及LogLoss；逐域两任务均值、逐任务三域均值；worst-cell、最大退化格、改善格数量；使用训练域占比 p_d 的 W=(1/2)Σ_d p_d Σ_t AUC_dt；逐任务实际pooling的AUC若报告，必须从跨域原始预测计算、单独命名；LogLoss固定clip epsilon=1e−7；selected epoch、final/common预设epoch、各臂aD/aT/bD/bT/ρ、新增参数/梯度/训练时间/显存与推理成本。

报告样本数量、train/val/test域比例与每格label prior；域难度用同协议B0的逐格AUC/LogLoss以及可匹配的SD reference描述，不能用样本量定义难度。记录全部数据集、模型、训练阶段与checkpoint身份。不独立操纵size，因此只能讨论size相关性；小域改善/大域退化须逐格展示，不预设一定存在。

实用validation筛查线：ΔM_val均值≥+0.003（AUC绝对值，不是相对百分比）。validation逐任务聚合均值退化不得超过0.001；validation W均值退化不得超过0.001；六格任一validation均值退化不得超过0.003，任seed单格退化不得超过0.005。validation逐格LogLoss均值恶化不得超过0.01，macro恶化不得超过0.005。独立test最终评价用同一组数值线，但只决定当前固定计划的证据是否泛化，不用于改臂、扩展seed/数据/预算或再调参。以上为操作性实用门槛，0.003沿用项目历史，其余保护线是预先固定的保守选择（价值依据WEAK），不是普适显著性或业务标准，不得按结果放宽。

两seed报告全部差值、符号、均值、样本标准差ddof=1及范围；域、格、epoch和ρ不是独立重复。样本bootstrap只估评估样本不确定性，不能替代训练seed证据。

## 8. 预算、验收与停止条件

本次R1训练/预测扫描预算：**0**。下列仅为未来另行授权的最小设计预算。

- 固定配对 seeds：2022、2023；两 seeds只初筛，不作统计显著或稳定算法有效断言。2024、2025属于历史四seed审计，不在当前拟议pilot运行预算中。
- 最多5臂×2 seeds=10个完整训练attempt（包括失败attempt），每attempt按baseline最多50epochs，总上限500 run-epochs；2个匹配B0重训已计入，不增补其它方法、seed或超参。所有五臂和两seed固定完成后才科学筛查，不按某个seed/某臂表现中止其余对照。
- 最多一次每seed B0冻结扫描：2×3×11=66个validation域级候选评分（可缓存表示以节约计算）；锁定后每seed原始与所选配置各一次完整test评估。禁止test grid搜索。
- 每臂最多一次、固定seed2022、每次≤3 batches的数值smoke检查（共≤15 batches），不构成绩效训练；未来IMPLEMENT前先完成DESIGN→Coder→独立实现Review，Review PASS或全部blocking fixes复核通过后才允许Runner。基础设施故障仅允许同配置、完整已保存model/两optimizer/RNG/data位置的精确续跑，每run至多1次、全队列至多2次；不额外启动完整训练attempt，无法精确恢复就FAILED/INCOMPLETE。禁止通过重启择优或改协议“修复”阴性。
- 未来任务开始前按单batch测得成本登记wall-time估计并冻结；未测时不虚构GPU小时或训练时长。达到run/epoch/重试上限即停止，未完成标INCOMPLETE，不纳入稳定结论。

未来实现的阻塞验收项：B1/B2/Bcap初始eval-mode与B0数值等价（float32 max_abs≤1e−6、相对误差≤1e−5；原参数必须逐tensor相等）；ρ=0.5融合系数测试；六格输出维度/路由；全部新增参数梯度有限且实际非零；旧四scalar梯度/优化器归属不变；coeff total invariant；sample BCE与旧baseline等价；曝光/hash配对；test不能被提前访问；selected保存可严格重载；学习参数数量/缓冲区完整报告。任何失败先返回Coder修正，不运行正式训练。

先导资源判决（尚无结果）：

1. 若baseline、数据、初始化、梯度或test封存失败，**INVALID/INCOMPLETE**，停止，不作科学阴性判定。
2. 只有B1−B0两seed的ΔM_val都>0、均值≥0.003且validation guardrail全通过，才有“有限validation信号”；最多WEAK，不授权本次或未来自动扩展。全部十attempt及扫描选择冻结后才揭示新test；最终支持须ΔM_test也两seed>0、均值≥0.003且test guardrail全通过，仍只是两seed有限信号。
3. 非平凡性validation筛查：B1−B2、B1−B3、B1−Bcap各两seed均>0且均值≥0.001，才认为所测简单解释尚未完全覆盖信号。B2本身过线但B1−B2不达线，记录简单域级权重信号、停止复杂化；B1未过线不能把B2事后升级主比较。test对应差值完整报告，不据此挑新的赢家或追加实验。
4. 若validation主比较未过线、方向不一致或guardrail失败，按有限预算停止扩大。test未复现validation信号时只说明当前固定计划泛化证据不足，不能追test改方向。可按已冻结判据标CONDITIONAL/STOP投入，但不授权任何补跑，不断言理论无效；不得无限增seed、换dataset、改指标、延长训练或添加模块。
5. 只有冻结扫描阳性：仅记录Hpost信号；只有训练阳性：不要求冻结扫描必须阳性。二者皆阴性：有限模型族与预算未找到实用空间，不是非均匀共享的全局不存在证明。
6. 两seed乃至4/4同向都不是p<0.05的充分证据（4seed单侧精确符号检验最小p=0.0625）。不把0.003当显著性，不把grid最大值当无偏估计。

## 9. 竞争解释与后续最有判别力的实验

优先证伪顺序：匹配B0是否可靠→守恒是否真实→所测任意偏置B3是否足够→简单B2是否足够→共享门重参数化Bcap是否足够→任务条件B1是否存在额外收益。B0/B1/B2/B3/Bcap的配对矩阵比直接开发在线亲和模块更能区分这些具体竞争解释；容量是否全面排除仍未识别。

持续记录：aD/bD训练漂移、gate补偿、表示范数/校准变化、训练进度/选模epoch、temporal shift、验证选择偏差、有限seed波动。当前没有size独立因果证据，也没有curriculum或momentum有效性前提。若仅少数小域提升、主导域损失，必须按guardrail判定trade-off，不用macro正值遮掩。

即使所有对照均无法解释有限阳性，也只是在这些对照族里留下研究空间。静态标量足够时优先保留简单方法，不因其“太简单”而加在线算法；阳性不自动成立亲和机制或论文新颖性。最近邻工作核验属于未来另行任务，本次没有作全球新颖性断言。

## 10. 审核与执行门禁

审核者：独立 GPT-6.1 Sol · xHigh Reviewer；代码审计：GPT-6.1 Sol · Medium；Experiment Designer按AGENTS先完成DESIGN SPECIFICATION；主Agent综合并签署Charter。审查记录见 work/R1_Independent_Design_Review.md；规范见 work/R1_Design_Specification.md。主Agent未实现算法或运行训练。

v0.1初审PASS WITH FIXES的C1–C5按本v1.1修正：val/test角色分离；新参数optimizer/温度/随机表固定；B3统一六格；当前runtime重建及reload replay门槛；来源语义/间接补偿/尺度与梯度边界。额外关闭C6：Bcap“新增参数=新增函数类自由度”的误称。独立Reviewer最终判决 **PASS**，全部C1–C6与跨文档差异 **CLOSED**；实际复核快照hash与闭环记录见[Review第14节](work/R1_Independent_Design_Review.md)。主Agent接受该设计合同，仅更新审核状态元数据，科学规则保持复核版本一致。

执行准备度仍须满足九份R0输入核对、真实数据/split指纹、匹配baseline重建或可核实权重以及独立实现审核。缺项即NOT READY；本次无论设计审核结果都停止于R1。不存在CODER IMPLEMENTATION REPORT或Runner结果，因为尚未进入IMPLEMENT/RUN。

## 11. 固定B3配置附录

由Designer仅生成配置常数，无模型执行。NumPy2.1.3，RandomState(MT19937) seed10404，Uniform(0.1,0.9)，task-major数组：

```
[0.7502780865025745,0.8537642404691582,0.44953240522281324,0.2525338669196546,0.23434256306102627,0.765217885741144]
```

对应(t0d0,t0d1,t0d2,t1d0,t1d1,t1d2)。无空格UTF-8 JSON数组SHA256：182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957。未来直接保存该表为buffer，不在训练流重新生成或挑选。
