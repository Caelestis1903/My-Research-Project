# R2 Execution Plan — v1.2（预注册计划，尚未执行）

日期2026-10-11（Asia/Shanghai）。依据 `promt/r1_xiuding.md` 与新 `R1_Experiment_Charter_v1.2.md`。本次只R1 Limited Revision，未搜索权重、未加载模型前向、未写算法代码、未做smoke或训练/预测评价。独立Reviewer最终 **PASS（设计合同）**，P1/P2/P3全部 **CLOSED**，见 [最终复核](R1_to_R2_Independent_Review.md) §8；执行准备度 **NOT READY**。本次仅补审核状态元数据，科学规则保持复核版本。读本文件不得视为执行授权。

## 1. 目标、假设、固定比较与停止边界

只检验KuaiRand natural强Sample-M3OE静态他域内部比例机会。B1−B0唯一主比较；B2是更简目标域比例；B3是固定无意义六格ratio；B4是逐格own/other比例、other内部仍均匀。B4替Bcap，正式计划不含Bcap、在线亲和、EMA、删除专家、curriculum、额外momentum干预或新数据集。Hpost独立可选默认disabled，不是训练前置。

本次预算0。未来必须人类另行明确授权，并完成DESIGN→IMPLEMENT→独立REVIEW→RUN→ANALYZE；未达到hard gates不得正式训练。Runner只记录原始结果，主Agent解释证据；Reviewer失败返回Coder，不越过审核。

## 2. 固定公式、优化、数据、指标

D3/T2/N4，i=t*3+d，d0/tab0,d1/tab1,d2/tab4；t0/click,t1/long_view。j<k固定other源模块顺序d0:(1,2),d1:(0,2),d2:(0,1)。`F_dt=sum_n gate_dt,n(detach(emb))*shared_n+aD*R_dt+aT*[bT*U_t+(1-bT)*U_(1-t)]`。global四raw初值1。B0 `R=bD E_d+(1-bD)(E_j+E_k)/2`；B1 `rho=sigmoid(q_dt)`、6q=0，B2为3q_d两task共用；B3同B1但固定buffer；B4 `b_dt=sigmoid(raw_bD+delta_dt)`、6delta=0，`R=b_dt E_d+(1-b_dt)(E_j+E_k)/2`。

B4 raw必须是 `_weight_bal_d.deep_weights` 未sigmoid值，实际 `sigmoid(raw+delta)`，禁止将sigmoid(bD)再当raw形成双sigmoid。保留raw_bD原arch路径；new delta与B1/B2 q走main Adam .01，每普通train batch一次。原四raw仅arch Adam .001，每epoch一train batch一次；两个optimizer wd1e-5、betas(.9,.999)、eps1e-8，交集为空、每步清全部梯度，无scheduler。new delta名称不采用 `_weight_`/architecture前缀避免误归，按实际parameter IDs与原四raw显式集合核验归属，不仅凭名称。B3不入optimizer。不修冗余、不改变raw_bD更新；记录raw、delta均值、中心化delta、有效b。raw+c/delta-c的共同shift不改变b，6trainable不是6新增可辨识函数方向，B4同时混合global快速比例与格差异竞争解释。参数总数B0/B3=1,614,258，B1/B4=1,614,264，B2=1,614,261，B3额外buffer6另列。

样本×task mean BCE Sample不重域加权；embedding16、FCN[512,256,256,64]、batch40960、workers0/shuffleTrue/drop_lastFalse，完整50epochs、shadow patience10不break。仅seed2022/2023，从同原tensor初始化复制、RNG和main/arch batch曝光hash配对。每epochval六格macro AUC严格>更新selected、tie最早；final和common9/19/29/39/49固定次指标。

数据SHA256=a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f；原全局时序80/10/10后filter，encoder仅base train、OOV0、stored labels；期望train[172782,1743898,89452]，val[2692,246232,1843]，test[2613,246284,1870]，每域/label/row/split/encoder指纹必须验明。只KuaiRand natural，不用MovieLens扩搜正结果。

runtime锁Python3.13.5/Torch2.9.0+cu126/CUDA12.6/cuDNN91002/NumPy2.1.3/pandas2.2.3/sklearn1.6.1；同硬件，random/numpy/torch/cuda seed和cudnn deterministicTrue/benchmarkFalse。当前runtime匹配重建B0，历史EDDAenv数值只上下文不作paired baseline。

指标macro=(1/6)sum六格AUC；task宏三域、domain宏两task；W=(1/2)sum_d p_train_d sum_t AUC_dt，p_train固定原train份额；LogLoss clip1e-7。AUC guard：task/W paired差均值>=-.001、每格均值>=-.003且任seed格>=-.005；LL guard：每格差均值<=+.01，macro<=+.005。追加门针对B1−B0，B2另行报告/标信号；test以同线最终验证但不追加。

## 3. 一次有界历史checkpoint搜索（未来才执行）

不重新做全R1。一次会话先从现有Inventory/manifest读取S1 KuaiRand seed2022/2023 joint的两个原绝对目录，验证逻辑路径和physical junction根；再按固定顺序检查 `D:/CODE/phase0-3/experiments/history_exp/data_archive/experiments/stage_S1_raw_results/kuairand/natural/m3oe`、`D:/CODE/phase0-3/m3oe/examples`。总墙时60秒、枚举文件/目录条目20000、候选binary32、候选hash/load累计512MiB，任一触顶停止。只.pth/.pt/.ckpt/.safetensors；匹配seed的run_manifest/config可作身份旁证。只跟到已核验物理根内部，不追未知junction；拒绝访问/不存在/超限写search receipt，不绕权限/扩全盘/继续找新根。

找到候选核验checkpoint/config/source/data/split/encoder/runtime/selected epoch/state身份，不能用新test分数选候选；预测npz不能恢复专家权重。当前历史checkpoint审计不可load，不是假定全磁盘绝无权重。S1源码可读/hash匹配，Inventory的missing_code等冲突字段不代替absolute file_inventory/hash和必要source可读核验。不要改旧Inventory或重审计全树。

搜索没找回合格权重、触限、拒绝访问或旧环境不匹配，都进入已计划两份B0当前runtime匹配重建，不新增attempt。即使找到历史候选，本合同仍默认重建；想改严格历史复用必须新test前修合同独立review，不自动复用/缩减/转移预算。记录物理根、限制、访问错误、候选身份与结论，当前不执行会话。

## 4. 未来实施位置与固定实现范围

只在新隔离目录建议实现 `research_R1_R3/src/ao04_v12_model.py`（B1/B2/B3/B4最小融合）、`ao04_v12_protocol.py`（固定臂/seed/参数/guards/预算）、`ao04_v12_runner.py`（S1合同与保存）、`ao04_v12_decision.py`（val-only追加门/锁selection）、可选`ao04_v12_hpost.py`；固定configs与results/attempt目录。当前不创建这些源码。

只读行为来源 `experiments/stage_S1_models.py` StageS1M3OEAdapter、`stage_S1_run.py` Sample loss/分组/更新/选模/save、`stage_S1_data.py` split/encoder、`stage_S1_metrics.py`指标、`stage_S1_protocol.py`协议；原构造器`m3oe/MDMTRec/models/multi_domain/model.py`。不写旧源文件、不覆盖S0–S8，不直接用examples trainer。

**必须拆解原runner的test耦合**：原S1 `fit()`约740起训练，约790调用 `_evaluate_deferred_checkpoints`；`write_outputs()`约879起依赖test数组并同时保存checkpoint。禁止直接裸调用原fit/write_outputs作为R2-A入口。新隔离runner复用原数值协议，独立train_val_loop、validation_selection、state_writer与最终统一test_evaluator；train状态和checkpoint保存不得要求test数组存在。静态代码审计与synthetic spy/mock/test-loader动态tripwire/计数必须证明A decision冻结前test prediction/metric调用=0，B启动后直到全局selection/config与test解封状态锁定前也=0；test labels仅用于preflight计数、完整性/hash，不performance评价/选择。这属于未来实现Review基础检查，无需正式数据性能测试。恢复baseline候选也先validation-only replay。

Coder一次实现全部五臂与条件门，出Implementation Report说明数学、逐tensor参数分组、diff和基础检查。即使B后续未启动，也须在A结果之前冻结B3/B4定义和实现审核，防止结果影响对照。Independent Implementation Reviewer审规范/代码/Coder Report，只有PASS或所有blocking fixes验收后的PASS WITH FIXES才允许Runner。设计Reviewer PASS不代替实现Review。

## 5. 启动hard gates与baseline验收

### G0：正式B0训练前

- 另行人类授权，v1.2设计最终审核通过，隔离实现审查通过。
- source/config/data字节、row/label/feature/split/encoder指纹和实际schema/计数验明；runtime与硬件manifest相符；test访问封存策略生效。
- baseline构造与forward/sample loss基础检查通过；B1/B2/B4零初始化eval-mode同原B0 tensor，FP32 max_abs<=1e-6、rtol<=1e-5；维度[B,2]、all6格路由、j/k顺序与own/other系数检查。
- 新q/delta梯度有限且非退化输入非零，main step只更新原main+new参数、arch step只四raw；B4六delta与raw路径实际存在，delta0时raw_bD梯度可比B0，学后梯度变化是预定干预；buffer/优化器分组用ID断言唯一、交集空、全覆盖。B4域内own+other总和1（乘aD后aD），两other相等；不把B1守global other质量的断言错加给B4。
- 原初始化tensor和main/arch batch序列hash配对；有限数值smoke每启用臂一次seed2022<=3batch，当前不执行、不用AUC调设计。隔离train/state writer在test数组缺省时能保存模型和完整resume；未来Implementation Reviewer必须检查test动态tripwire计数，Adecision前与B启动后到全局test解封前预测/metric调用=0，testlabels仅preflight完整性/hash。

### G1：正式B1/B2前

先在B0任何训练前保存每seed初始化tensor、raw RNG和main/arch sampler初始state，B1/B2随后从这些未训练state重建配对，不能从B0已训练selected/final初始化。再完成B0seed2022/2023两run的50epoch，完整validation logs和earliest val max selected；保存模型、两个optimizer/RNG与初始化/batch记录。strict load selected在相同validation重放：预测max_abs<=1e-6、六格及macro AUC/LogLoss差<=1e-6；失败停止队列INVALID返回reviewer。历史test数值不是新runtime重建的硬成功线，validation reload验收不是历史环境严格复现。

hard阻塞包括缺上述source/data/指纹/依赖/forward/load/梯度/配对/实现审核或授权；三缺R0 work报告只是文档问题，如实标未读，不用其单独否决已验明重建。历史checkpoint缺失阻止直接复用/立即Hpost，不阻止通过G0/G1的匹配重训。

## 6. R2-A固定队列与阶段锁

固定次序：

| 顺序 | 子阶段 | 臂 | seed | 最大epochs |
|---:|---|---|---:|---:|
| 1 | A | B0 | 2022 | 50 |
| 2 | A | B0 | 2023 | 50 |
| 3 | A | B1 | 2022 | 50 |
| 4 | A | B1 | 2023 | 50 |
| 5 | A | B2 | 2022 | 50 |
| 6 | A | B2 | 2023 | 50 |

每attempt状态PLANNED/RUNNING/COMPLETE/FAILED/INVALID/INCOMPLETE留痕。不能看seed2022性能停止2023或改变臂。数值/协议错误暂停队列排查，不能凑完整数据继续门。基础设施能精确恢复则按第9节resume，否则缺失A不进入B、冻结INCOMPLETE，不追加replacement attempt。

只有6有效完整配对后，用selected validation计算D_s=B1−B0和T_s=B1−B2；guard完全对B1−B0差计算。所有D/T/均值与guards只用原始未四舍五入metrics，不用展示rounded数；阶段阈值无额外tie tolerance，直接执行>0和>=/<=判式。两D>0、meanD>=.003是实用；两D>0且.001<=meanD<.003为探索性；两类都要求所有旧guard通过。追加YES当且仅当有效性通过、任一类信号成立、两T>0且meanT>=.001。否则NO并写不可变decision manifest（6run/选模hash、全部D/T、每guard、理由、预算余额）。无test字段、无主观例外。Hpost≤1e-6tie仅属grid选比例，不迁入阶段gate。

B2有信号但B1不成立只记录次比较，不升级主法。B1−B2不达线是本预算未显示task增量，不证明等效/充分；B1/B2都改善且相近时保留性能信号、预算停止复杂化，B3/B4机制未排除。B2普通性能guard失败不是全A invalid；非有限与泄漏等validity异常才按统一停队列。

## 7. 条件R2-B队列

仅decision YES执行：顺序7 B3seed2022、8 B3seed2023、9 B4seed2022、10 B4seed2023，各最多50epoch。再次确认原已冻结实现/config没有按A结果改动；配对原B0初始化/main/arch曝光。若NO，这四项标NOT SCHEDULED、不消耗四attempt且不用余额替换实验。

B3直接用固定task-major表 `[0.7502780865025745,0.8537642404691582,0.44953240522281324,0.2525338669196546,0.23434256306102627,0.765217885741144]`，NumPy2.1.3独立RandomState10404来源，无空格UTF8 JSON hash182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957；buffer不重抽、不梯度、不选方向。

B4原四raw全部保留（raw_bD及另外三个），raw_bD原arch更新不变，delta主Adam6paraminit0；记录delta mean/centered/raw_bD/有效b与own/other系数。不另加第五个raw、不加约束/正则、不按A改初始化。B4优劣只解释这一混合竞争参数化，不证真实亲和。

## 8. 可选Hpost与最终test解封

Hpost default disabled，只有正式执行前预先登记并review enabled=true时实施；不能看A/test决定开关，不能增加训练。可以在B1以后/全部train结束再扫描，不强制B1之前。B0selected必须G1可load验收。

冻结eval，每域两个task共用rho，grid0:.1:1；先域两格val ΔAUC各>=-.003筛可行，再两task macro max，≤1e-6tie优先.5/closest.5/smaller，无可行nonuniform回.5。三个rho有效组合11^3，最多66validation域级分数，两selected先于grid选。全曲线与唯一三rho配置锁进manifest，不能选epoch/选B1lr。

YES等B已计划训练全部完成并锁所有checkpoint/config；NO先锁不进B终止决定，再锁A所有checkpoint/config。validity导致中断先锁异常状态，只有有效完成已审核配置可作限定test描述，不能据此重开训练。所有selection、final/common身份、Hpost开关及rho（若启用）和decision状态先锁，才统一最终test。旧test历史已用，不叫全新holdout。

每完成臂selected主端点与final/common描述按原协议评，全部原差值/均值/SD/guard公开，不按test择赢家/epoch。Hpost仅原0.5与唯一选中vector各一次（原B0可复用），无test grid。test不改变YES/NO、不重开B、不改阈值/seed/dataset/主法。当前无任何实际评价。

## 9. 工件、失败恢复、预算和停止

每epochlast_resume checkpoint需完整model/main optimizer/arch optimizer/random-numpy-torch-cuda RNG/loader及arch iterator状态/data位置/epoch-step计数/config和hash，支持同配置精确续跑；selected/final/common另外保存epoch/config/hash且selected可strict重载。selected可含所选epoch两optimizer/RNG快照，但不是最终last_resume，禁止用selected位置接续final训练。保存B0未训练初始化、raw RNG/初始sampler、逐批曝光hash、每epoch指标与shadow记录、selection与Adecision immutable manifest、异常、完整source/data/encoder/runtime版本、预测及参数成本。B4额外delta全表/均值/centered/有效b日志，原a/b轨迹、专家尺度/gate entropy全部臂保留。

总完整attempt上限10（A6+B4，失败占名额），总run-epochs<=500；未进入B时A上限6/300epochs。每run最多1次、全队列最多2次基础设施精确resume，不新增完整attempt，不从头重训/改lr/择优重启；缺完整state则FAILED/INCOMPLETE停止，不用未启用B名额替换A。阶段性interruption由Reviewer修复代码需重启完整训练时，不能自动追加，现有预算/设计标未完成等待另行决定。

数值nonfinite（参数/预测/loss/梯度）即run与队列INVALID停止；rho/b_dt<.01或>.99只描述，不clamp。达到attempt/epoch/resume上限停止；失败和缺配对不算科学阴性。成本当前未测，不虚构GPU小时；未来单batch成本只估资源，不据performance调方案。

## 10. 最终分析与Reviewer验收

主Agent分析A描述、B初步、C实用、D剩余机会四层，两seed最多WEAK。B1−B0 practical两seed>0/mean>=.003+guards；exploratory .001到.003单列。B1相对B2/B3/B4各两seed>0且mean>=.001为所测替代未解释的有限余量，test对应差只作固定最终验证；B阶段没运行则这些机制问题未检验，不能推创新空间。B1胜B4也不识别亲和、B4六param不等六函数类自由度。

检查per-domain/task AUC/loss、seed重复、domain size与B0难度/训练stage，公开小域改善/大域退化，不从三域size相关推主因；保留共同适配、β漂移、门塔补偿、B4快速global变化/格差异混合、时间shift、小域validation噪声和不同优化几何解释。test失败不重启，A低信号不证普遍无效。

Reviewer验收：五臂公式/初始等价/optimizer定义明确；A六valid与val-only门纯函数固定；B条件队列和预算受限；test封存/终止锁/Hpost独立；一次搜索边界及matchedB0/reload/指纹hard gates明确；缺报告与缺前向/数据区分；旧文件只读、当前无执行。设计审核通过后仍需人类授权与独立实现Review，本次完成后停止。
