# R1 Experiment Charter v1.2 — AO-04 Limited Revision

日期：2026-10-11（Asia/Shanghai）。状态：独立Reviewer最终 **PASS（设计合同）**，P1/P2/P3全部 **CLOSED**，见 [最终复核](R1_to_R2_Independent_Review.md) §8；执行准备度 **NOT READY**。本次仅补审核状态元数据，科学规则保持Reviewer复核版本。当前授权仅R1 Limited Revision：读报告/必要源码、修订设计、实施计划和独立审核。当前训练、smoke、预测评估和checkpoint搜索预算均为0；不得自动进入R2/R3。

本合同完整替代未来使用的v1.1实验计划，但不覆盖原文件。输入为 `promt/r1_xiuding.md`、已审核v1.1 Charter、Architecture Audit和Baseline Protocol。只改变指定对照、条件预算、结论层级与准备安排；不重新审计全项目、不重排方向、不重新做文献分析。历史报告没有新的性能结果。三份R0 work报告缺失是文档完整性问题，不伪称已读，不作为匹配重建的独立运行硬阻塞。

## 1. Research question与假设

KuaiRand-Pure natural强Sample-M3OE中，保持全局own/other总系数定义，学习他域专家内部相对比例是否产生方向重复、达到实用量级且没有禁止退化的AUC信号？该信号是否相对简单域级权重、所测固定偏置和逐格own/other调节仍有余量？允许阴性和停止投入；不预设curriculum/momentum有效、小域等于难域或M3OE必然在难域失败。

- H0：限定参数化/优化/预算未找到达到门槛的收益，或所测简单对照解释了信号；不是任意非均匀共享普遍无效的命题。
- H1：B1学习他域内部比例，相对B0有符合固定线的有限性能信号；相对所测对照仍有净余量。唯一主比较始终B1−B0。
- Htask：B1相对B2显示额外任务条件化信号；不达线表示未显示必要性，不证明B2与B1等效或B2充分。
- Hown：逐格own/other调节B4是否可以解释B1所观察的信号。
- Hpost：冻结selected B0后，验证选择的守恒比例重分配是否在固定test评价中有收益。独立可选，非B1前置条件，与重训假设互不替代。

置零损失仅可说明模型依赖通道；不推出均匀非最优。置零无损不否定重训可学习权重。描述性权重、冻结后处理和重新训练收益分别登记。

## 2. 实际计算图和固定基线

B0为StageS1M3OEAdapter的设备安全forward、正确分离两optimizer、Sample joint；不能用含legacy分组缺陷的examples trainer或Equal、SD、S6 T/HT/DH、S8 continuation替代。D=3、T=2、N=4；4共享+3域+2任务专家；6个共享gate和6个tower。STAR、专家、门、塔、detach、dtype和设备路径不改。KuaiRand原参数1,614,258。

设i=t*3+d，E_d为域专家、H_n为共享专家、U_t为任务专家：

`F_dt=sum_n g_dt,n(detach(emb))*H_n + aD*R_dt + aT*Q_t`。

`Q_t=bT*U_t+(1-bT)*U_(1-t)`；T=2没有多他任务比例对象。aD/aT/bD/bT均是跨格global sigmoid，原四raw scalar初值1，sigmoid(1)≈0.7310586。

源模块顺序固定d0:(j,k)=(1,2)，d1:(0,2)，d2:(0,1)。所有E_j对同一目标样本STAR表示dense计算，且接受回收梯度；模块索引不是只由该域训练的隔离信息源。共享gate直接作用共享专家，STAR/任务分支/塔可间接补偿域融合。学得比例不识别真实跨域因果亲和。

## 3. 实验臂、数学定义与优化协议

| 臂 | 域项R_dt | 新可训练标量 | 初始化 |
|---|---|---:|---|
| B0 | bD*E_d+(1-bD)/2*(E_j+E_k) | 0 | 原样 |
| B1 | bD*E_d+(1-bD)*(rho_dt*E_j+(1-rho_dt)*E_k) | 6 | rho_dt=sigmoid(q_dt)，q=0 |
| B2 | 同B1但rho_d两任务共享 | 3 | q_d=0 |
| B3 | 同B1但每格rho是固定无意义表 | 0 | 固定buffer，不初始等价 |
| B4 | b_dt*E_d+(1-b_dt)/2*(E_j+E_k) | 6 | b_dt=sigmoid(raw_bD+delta_dt)，delta=0 |

B1/B2只改变other内部比例，own系数aD*bD、other总系数aD*(1-bD)的代数定义不变。保持系数总和不保证向量范数、预测分布或梯度量相同，且各臂aD/bD终值可不同。B4是竞争机制，主动允许own/other总比例因格改变；other内部仍均匀，不应声称B4维持B1的other质量约束。

B4保留原raw_bD和其余三raw scalar全部原路径，恰新增6个delta。raw_bD取 `_weight_bal_d.deep_weights` 的未sigmoid值，不能把已sigmoid的bD再加delta作raw造成双sigmoid。初始化delta=0时与B0前向等价；task-major六格raw_bD初始梯度可比B0，学后raw梯度变化属于预定干预而非违反更新协议。`raw_bD+c, delta_dt-c`给出同一b_dt，因此新增6个trainable不是6个新增可辨识函数类自由度；域融合b向量函数维度为6，较B0单global b最多增加5个可辨识方向。delta均值可用主Adam快速调整global own/other，中心化delta可反映格差异：B4是两种作用混合的对照，不是纯异质性机制。不得加sum-zero约束、冻结raw_bD、改变arch频率或新增正则来消除冗余。B1胜B4只排除本次所测own/other调节参数化/优化解释，不证明真实亲和、不排除所有容量/优化解释。无需因此加新臂。

原main参数仍主Adam：lr=0.01、wd=1e-5、betas=(0.9,0.999)、eps=1e-8，每普通train batch一次；B1/B2 q和B4 delta加入同一主Adam，规则完全相同，不加入arch optimizer。四原raw scalar只走arch Adam：lr=0.001、wd=1e-5、betas=(0.9,0.999)、eps=1e-8，每epoch一个train_loader batch一次，不用validation。B3的6ratio是buffer在optimizer外。B4 new delta不用 `_weight_`/architecture易误归前缀，参数归属按实际ID及原四raw显式集合断言，不能仅猜名字。每步清空全部梯度，两optimizer交集为空且覆盖全部trainable。原总数B0/B3=1,614,258，B1/B4=1,614,264，B2=1,614,261；B3 buffer6不计trainable。新参数归属是预注册新协议；阴性只限此优化方案，不作充分探索断言。不调lr、不比较多种参数归属。

Bcap保留为以后另行设计可考虑的共享门重参数化对照，本正式计划、smoke和训练预算均不含Bcap；不得在本预算内临时加回来。

## 4. 数据、训练、runtime和选模不变

只使用KuaiRand-Pure natural，d0=tab0、d1=tab1、d2=tab4；t0=is_click、t1=long_view。CSV字节SHA256=`a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`、416370411bytes。按既有processed date/time顺序全局80/10/10 split-first/filter-second，不split前shuffle；标签用stored值；encoder只base train拟合，OOV=0。执行前重新核对row/label/feature/split/encoder指纹，历史字节hash不代替这些验收。

历史期望train/val/test计数：d0=172782/2692/2613，d1=1743898/246232/246284，d2=89452/1843/1870；合计2006132/250767/250767。计数不匹配须排查，不临时更换split。训练域份额取实际验明的原train计数固定计算，不能取validation/test份额替代。train→eval域mixture和prior时间shift必须报告。

Sample objective：`L=(1/(B*T))*sum_i,t BCE(p_it,y_it)`，torch BCE reduction=mean；不再乘p_d，不改六格等权均值、不balanced sampling。batch平均与全epoch样本加权日志分别报告。

embedding=16，FCN=[512,256,256,64]，expert_num=4，batch40960，workers0，shuffle=True，drop_last=False；完整50epochs，无scheduler；patience10只shadow记录，不实际early stop。四原raw初值1。种子仅2022/2023，random/numpy/torch/cuda固定，cudnn deterministic=True、benchmark=False；不声称跨runtime位级一致。

全部新run采用v1.1锁定当前runtime：Python3.13.5、Torch2.9.0+cu126、CUDA12.6、cuDNN91002、NumPy2.1.3、pandas2.2.3、sklearn1.6.1，硬件/依赖manifest完整且逐臂一致。历史S1 Python3.8.20/Torch2.0.0/cu117不能与当前paired B0拼接。默认匹配重训两份B0，不声称严格恢复历史环境。

每seed在B0任何训练前保存原始初始化tensor、raw RNG与main/arch sampler初始state，B1/B2/B3/B4从这个未训练state复制原参数，绝不从B0已训练selected/final初始化；新增zero/buffer不消耗原模型/数据RNG，batch及arch batch序列逐臂hash配对。每epoch完整validation六格macro AUC，严格更优替换，tie earliest；50epoch完成，selected主端点，final/common9/19/29/39/49仅描述且不能据test择优。

## 5. 两子阶段与validation-only追加门

R2-A固定B0/B1/B2×seeds2022/2023=6完整训练attempt。先完成两份B0及reload replay，再执行B1/B2；所有6份有效、完整50epoch、paired证据齐备后才允许计算资源门。FAILED/INCOMPLETE/INVALID不能通过追加门，不把缺失当科学阴性。

令D_s=M_val(B1,s)-M_val(B0,s)，T_s=M_val(B1,s)-M_val(B2,s)，均在各臂earliest validation max selected上计算。所有D/T/means和guards使用原始未四舍五入的存储指标，展示表的rounded值不得入门。阶段阈值按严格>0与指定>=/<=直接判，不增tie tolerance；Hpost的1e-6并列规则仅用于其比例选择。

1. **实用筛查信号**：D_2022>0且D_2023>0、mean(D)>=0.003，并且第6节针对B1−B0的全部validation guards通过。
2. **探索性筛查信号**：两D均>0、0.001<=mean(D)<0.003，并且同一组全部validation guards通过；不放宽guards，不称实用通过。
3. **没有达到追加信号线**：其余情况，包括仅一个seed改善、mean(D)<0.001或guard失败。微小同向改善可以如实描述，不升级。
4. **唯一R2-A→R2-B门**：6份有效完整配对，且(1或2成立)，且T_2022>0、T_2023>0、mean(T)>=0.001。条件满足就按固定计划执行B3/B4两seed；不留主观择臂空间。不满足就冻结不进入B的终止决定，未消耗预算不挪给新方法/seed。

B2逐格/两seed指标和异常完整报告，其数值非有限按统一validity停队列；B2普通性能guard失败不自动使B1的追加门失效。B2自身如标性能信号，必须对B2−B0单独用同样门槛与guards。B2有信号而B1没有时仍是预注册次比较，不升级主方法。

B1−B2门不通过仅表示当前低功效预算未显示额外任务粒度必要性。两者都改善且接近时节约资源、停止复杂化，保留静态非均匀性能信号；B3/B4未执行意味着固定偏置和own/other解释未排除，不能判整体AO-04无效或B2已证充分。

R2-B仅B3/B4×2022/2023，最多4追加完整attempt。无新的调参/初始化搜索，无依A阶段数值调整B4或随机表。五臂公式与实现审查在A前冻结；B1/B2/B0不再训练。

## 6. 指标、原guardrails和结论层级

每seed主端点为selected独立test六格macro：`M_test=(1/6)sum_dt AUC_test,dt`；ΔM_test是候选−paired B0。M_val与ΔM_val同定义作用validation，只用于筛查。逐格mean不是pooled AUC。

报告六格val/test AUC和LogLoss；域内两任务均值、任务内三域均值；worst-cell、最大退化格、改善格数量；`W=(1/2)sum_d p_train_d sum_t AUC_dt`称训练域份额加权逐格AUC，不称混合总体AUC。额外pooled task AUC须实际拼原预测/标签计算、单列；LogLoss clip epsilon=1e-7。

守护线均为候选−B0的paired差：每任务宏AUC差两seed均值>=-0.001；W差均值>=-0.001；六格各AUC差均值>=-0.003、任seed单格>=-0.005；六格各LogLoss差均值<=+0.010、macro LogLoss差均值<=+0.005。追加门明确只检查B1−B0；其它臂同样报告各guard结果，不把所有臂必须改善误作validity条件。test最终评价沿用同样数值线，但不决定追加。

全部两seed原始差值、符号、均值、样本标准差ddof=1与范围公开；格/epoch/比例不算独立seed。两个训练seed证据最多 **WEAK**，0.003是操作性实用线不是显著性。失败不证普遍无效。

结论分四层：

- A描述性：rho/delta/b_dt、专家输出尺度、融合分布、门熵、训练轨迹；不直接推性能或因果机制。
- B初步性能：两个seed同向的有限信号，mean在[0.001,0.003)且guards通过时称探索性；存在选模乐观性，不等实用通过。
- C实用通过：B1−B0两seed>0、mean>=0.003、同split全部guards通过；validation筛查和最终test是否通过分别写，不混合。
- D剩余研究空间：必须执行有效完整B阶段，并且B1相对B2/B3/B4的validation差分别两seed>0且均值>=0.001，最终锁定test对应差也逐项报告并满足同一余量才有独立验证支持。仍只排除所测替代，最多WEAK，不证亲和/容量穷尽/论文创新。只有探索信号时称“探索性剩余线索”，不能借对照通过升级为实用创新机会。

没有B阶段时D未检验；test未重复信号时说明泛化证据不足，不能重启。GO/CONDITIONAL/STOP仅描述本预算证据与投入建议，不授权任何后续阶段。数值/协议问题是INVALID，未完成是INCOMPLETE，与科学阴性分开。

## 7. test隔离及可选Hpost

A阶段完全封存新的test预测/指标。追加门只读validation，自动生成decision manifest列出所有D/T、guards、资格与YES/NO。若YES，等B3/B4全部完成且selection/config锁定；若NO，先锁定终止决定。若因validity失败停止，先锁INCOMPLETE/INVALID状态；不为了凑test提前跳过审核。只有已有效完成run的已冻结配置可作限定最终报告。

所有已计划run训练完成、全部selected/final/common身份及配置和阶段终止状态锁定后，统一按原协议评test；B0/B1/B2有效6run，若追加再有B3/B4四run。test不重开B，不改臂/seed/dataset/阈值、不把B2改称主法。旧test在历史研究中已使用，只保证不按本次新test调整，不称全新未接触holdout。

Hpost默认disabled。未来授权执行前可登记enabled=true并审核，不能依据A/test结果择是否开启；它不增加训练，不要求在B1之前扫描。只有真实可load且B0 validation replay验收通过的selected可用。每seed每域rho grid={0,0.1,...,1}两task共用；先要求域内两格val ΔAUC各>=-0.003，再最大两task宏AUC；差<=1e-6优先0.5/最近0.5/较小rho；无可行nonuniform取0.5。选checkpoint先于扫描、冻结参数/eval，不按rho重选epoch。

有效选择空间11^3=1331，最多2×3×11=66域级validation分数；保存全曲线。所选三rho向量在同一test解封前锁定；test只评各seed唯一所选向量及原0.5（原值可复用B0最终评价），不test grid，不改变训练方案。权重缺失时没有扫描结果；Hpost阴性不否定重训，Hpost阳性不证明在线算法。

## 8. 有界baseline恢复、hard gates和实施位置

当前16核查run的声明112checkpoint缺失、selected可load0/16；本次没有前向复现。S1源码可读且审计hash匹配，Inventory字段不一致时以absolute file_inventory/hash及必要源文件核验为准，不能以missing_code单字段宣称源码不存在。baseline复用状态、设计PASS与资源ready分开。

未来最多一次搜索会话，只枚举下列逻辑根：①Inventory中S1 KuaiRand natural seed2022/2023 joint的两个原manifest绝对目录；②`D:/CODE/phase0-3/experiments/history_exp/data_archive/experiments/stage_S1_raw_results/kuairand/natural/m3oe`；③`D:/CODE/phase0-3/m3oe/examples`。会话总墙时<=60秒、总枚举文件/目录条目<=20000、最多32个候选二进制、候选hash/load累计读取<=512MiB；任一上限到立即停止。扩展名只.pth/.pt/.ckpt/.safetensors，允许匹配seed目录的config/run_manifest身份旁证。物理junction解析后仅允许留在已核验的这些物理根子树，不追未知reparse点；拒绝访问/不存在/超限如实记录，不扩到全盘或无限找。搜索只是未来计划，当前不实际进行。

找到候选只验证load和code/data/config/split/encoder/runtime/selected身份，不读取test挑权重。旧环境候选不能自动替代当前paired B0；v1.1当前runtime匹配重训仍是本合同默认，两个B0 run已计在A的6attempt。任何严格历史复用提案须在新test前另修设计并review，不自动削预算/改变baseline。本合同不依赖找到权重才能重建。

正式B0前hard gates：另行执行授权；设计PASS或全部blocking fixes闭环；冻结五臂实现与独立Implementation Review及Coder Report；canonical source/data字节与split/encoder/标签指纹、依赖/硬件manifest匹配；初始化与Sample目标测试、六格尺寸/路由、参数归属与梯度、固定配对RNG与曝光smoke合格；无泄漏。

正式B1/B2前追加hard gates：两份匹配B0完整50epoch、selected是earliest val max、完整checkpoint保存；strict load selected在同validation replay，预测max_abs<=1e-6、六格及macro AUC/LogLoss差<=1e-6；不通过就INVALID返回Reviewer，不据旧test数字判重建是否成功。没有baseline前向replay、split/encoder身份或实现审核才是真硬阻塞；三缺R0 work报告、非主dataset历史状态是文档限制，不单独阻断已验证重建。

未来隔离建议新增 `research_R1_R3/src/ao04_v12_model.py`、`ao04_v12_protocol.py`、`ao04_v12_runner.py`、`ao04_v12_decision.py`、可选`ao04_v12_hpost.py`与固定config；本次不创建代码。行为参考 `experiments/stage_S1_models.py`（fusion/adapter）、`stage_S1_run.py`（Sample/main/arch/selection/save）、`stage_S1_data.py`（split/encoder）、`stage_S1_metrics.py`、`stage_S1_protocol.py`，原模型`m3oe/MDMTRec/models/multi_domain/model.py`；全部旧文件只读。

原S1 `fit()`在50epoch末调用 `_evaluate_deferred_checkpoints`（约790行），`write_outputs()`（约879行）依赖test数组且绑定保存模型。未来隔离runner必须拆开train/validation+state writer和最终统一test evaluator，禁止直接裸调用原fit/write_outputs当A入口。实现Review用静态检查及synthetic spy/mock/test-loader动态tripwire计数证明A decision冻结前test prediction/metric调用=0；B启动后直到全局selection/config和test解封状态锁定前也为0。test labels只允许preflight计数、完整性/hash检查，不作performance评价/选择；无需正式数据性能实验来验证此门。恢复候选也只能先validation replay，不能因baseline查验提前做新test。详细步骤见新Execution Plan。

## 9. 实现验收、日志与预算

B1/B2/B4 zero初始化eval前向与同state B0比较FP32 max_abs<=1e-6、相对误差<=1e-5，原tensor逐个相等；B4 b_dt初值全等bD，域内own+other系数和=1、乘aD后和=aD，other两系数相等；B1/B2 other质量保持global bD定义，B4故意改为aD*(1-b_dt)，不能机械要求B4保持global other质量。全部新参数在非退化all-domain合成输入得到有限非零梯度并主step实际更新，arch step不写q/delta，main step不写四raw；每个optimizer独占归属。B4保留raw_bD路径及delta六个参数，不加约束改变算法。

逐epoch记录train Sample BCE、validation六格/宏AUC与LogLoss、global raw/sigmoid、rho或B4 delta全表/均值/中心化delta/raw_bD/有效b_dt、own/other完整系数、专家输出尺度、共享门熵、梯度/update norm与实际step数、曝光hash、shadow首次触发/选模epoch/边界censoring。新ratio或有效b_dt<0.01或>0.99仅标饱和，不clamp/改lr。任何参数/预测/loss/梯度nonfinite停止run和队列INVALID。保存异常与成功状态，失败不隐去。

保存模型selected/final/common及其epoch/config/hash；selected载荷可包含该epoch两optimizer/RNG快照，但不可冒充最终resume。每epochlast单独保存完整精确续跑state、main/arch optimizer、random/numpy/torch/cuda RNG、loader/arch iterator和epoch/batch位置，恢复只用一致完整last_resume而不是selected。保存seed/config/source/data/split/encoder/runtime与checkpoint hash、完整日志与selection/decision manifest。逐域样本数/label prior、domain/task指标、配对差异、参数/活跃量/训练时间/显存/推理成本全部报告。不虚构GPU小时，未来单batch测成本后冻结估计。

总预算：当前0。未来A固定6、B条件追加最多4完整attempt，失败也占名额，每attempt最多50epochs，总最多10attempt/500run-epochs。失败不从头增加完整attempt；只有同配置且完整model/两optimizer/RNG/数据位置state可精确resume，每run<=1次、全队列<=2次，不额外attempt、总epochs上限仍有效。不能以重启择优。

未来每启用臂最多1次、seed2022、<=3batch数值smoke；A+B共<=15batch，非绩效训练，单列日志；当前不执行。smoke失败返回Coder/Reviewer且不正式训练，禁止用smoke AUC调设计。正式attempt配对不足不进入追加门；上限触发就INCOMPLETE、停止，无补seed/新dataset。

## 10. Falsification、解释范围及审核停止点

有效A中B1无重复>=0.001信号或违反guards，停止追加；不是普遍无效。B1实用达线而B1−B2不达线，任务增量未显示，保留性能信号但竞争解释未测。B阶段B3/B4与B1差未达两seed>0/mean>=0.001，所测简单解释未排除；不能以强机制说法继续复杂化。test不重复已冻结val信号则泛化证据不足，不再调整。任何泄漏/目标/配对错误是INVALID而非反证。

域大小不是域难度，不操纵size就不能认定它是主因；natural三域不能用三点相关得因果结论。每格损益公开检查小域改善/大域退化的trade-off、B0强度、task、model、epoch/selected差异与seed一致性。专家共同适配、aD/bD漂移、B4 global+格混合调节、门/塔补偿、评估小域选择噪声、时间shift和不同优化几何均保留为竞争解释。

独立Reviewer须给PASS/PASS WITH FIXES/FAIL及阻塞闭环；本合同和Execution Plan需要同时检查。不管设计判决如何，本次完成后停止，等待正式R2执行授权。不存在本次Implementation Report、Runner结果或性能GO。

## 11. DESIGN SPECIFICATION结构索引

| 必需项 | 本合同的具体定义 |
|---|---|
| 1 Research question | §1静态非均匀融合机会及竞争解释 |
| 2 Hypothesis being tested | §1 H0/H1/Htask/Hown/Hpost，§10证伪边界 |
| 3 Variables changed | §3 B4替Bcap；§5条件追加；§7独立可选Hpost |
| 4 Variables held constant | §2主干/global scalar；§4数据/Sample/优化/50epoch/seed/选模 |
| 5 Mathematical/algorithmic definition | §2–3融合、ratio/delta和optimizer；§5纯validation门 |
| 6 Exact affected modules/files | §8只读S1行为来源与拟隔离v12文件；本次零源码修改 |
| 7 Implementation requirements | §8拆test入口/hard gates，§9数值/gradient/参数/工件验收 |
| 8 Ablations required | §3 B0/B1/B2/B3/B4；A固定三臂，B按唯一gate追加两臂，不加臂 |
| 9 Expected logging requirements | §6全指标/guard；§9参数、delta均值/中心化、有效b、梯度/RNG/state与decision日志 |
| 10 Acceptance criteria for reviewer | §8设计/实现审核、tripwire、source/data/forward gates；§9预算和§5追加规则 |
| 11 Conditions falsifying hypothesis | §10有限预算阴性、对照解释、test不泛化；INVALID不当科学阴性 |

## 附录：B3固定表

NumPy2.1.3，独立RandomState(MT19937)10404，一次Uniform(0.1,0.9)，task-major(t0d0,t0d1,t0d2,t1d0,t1d1,t1d2)，直接读常数不在训练流重抽：

`[0.7502780865025745,0.8537642404691582,0.44953240522281324,0.2525338669196546,0.23434256306102627,0.765217885741144]`

无空格UTF-8 JSON数组SHA256=`182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957`。一表不穷尽随机解释，禁止val/test挑表。
