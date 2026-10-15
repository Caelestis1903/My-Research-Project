# R1 有限修订 v1.2 独立审核

日期：2026-10-11。Reviewer：独立研究设计审核。范围仅为 `promt/r1_xiuding.md` 指定的R1有限修订；本轮未实现算法、未训练、未进行AO-04预测或性能评估、未重新开展文献/方向排名。旧R1三报告不修改。

当前状态：**最终独立设计verdict：PASS（第8节）；execution NOT READY。** 已实际复核三份修订文档及初审修订闭环。PASS只批准有限设计合同，不是实施审核通过、性能判决或R2/R3执行授权。第2–5节保留预审攻击及建议；实际未来规范以本次审核的三份v1.2文档为准。

## 1. 继承与变化范围

继承v1.1的B0/数据/特征/Sample objective/50完整epochs/shadow patience/原两optimizer/验证选模/2022与2023/状态和随机性记录。新任务只允许设计文档变化：B4替换本预算的Bcap，R2-A六attempt与validation-only条件R2-B最多四attempt，以及有界baseline恢复/匹配重建计划。Bcap不进入正式预算；不新增研究臂、seed、dataset、超参或在线亲和模块。

原九份R0实际输入缺三work报告仍为真实缺口，但新有限任务可把它列为**文档完整性缺项**，不单独阻断已有可审代码和已知数据上的实施。不得称九份已读。真正的执行hard gates应来自数据/源码身份、实现与梯度/初始化审核、测试隔离、runtime和baseline可复核性，而不是缺失报告的数量本身。

## 2. B4 的辨别力与不可识别部分

B4保留raw_bD于原arch Adam，每格新增δ于与B1 q相同的主Adam，初始化δ=0。有效 `b_dt=sigmoid(raw_bD+δ_dt)`，域项 `b_dt*E_d+(1-b_dt)*(E_j+E_k)/2`。B1改变两个他域之间的相对分配；B4改变own-vs-other总体比例并保持他域内部等权。因此二者干预的系数轴不同，作为竞争机制有直接辨别价值，且δ=0在代数上恢复B0。

但是B4不是“同有效表达自由度”证明。六新增δ与原global raw共7个数只进入6个logit：对任意c，`raw_bD→raw_bD+c, δ_dt→δ_dt-c`不改变有效b。相对于B0一个global b，格级b系数族最多新增5个独立异质维度，存储/可训练参数仍恰好增加6个；整个网络的可辨识性还受专家和塔共同适配限制。

另一个实际竞争解释是更新速度：δ的共同均值通过每batch主Adam更新，可以快速改变global own/other强度；原raw_bD则仍每epoch arch更新。B4效应混合了共同幅度重调与格级异质性。保持原raw更新规则并不保持所有臂有效b的更新速度相同。Adam与weight decay选择了不同的参数路径，不能从δ或b的终值直接识别真实亲和。

最小修正是不增臂、不改变用户指定B4：披露上述混合estimand，记录raw_bD、全部δ和b_dt、δ均值及δ−均值、主/arch步数和参数梯度/更新。B1优于B4仅排除这一具体own/other适配替代；B4不如B1不能排除所有总质量/容量机制。B4可作竞争对照，不需因这种已披露冗余扩大实验。

## 3. 条件追加gate的独立攻击

建议用明确布尔表达式，而非“值得继续/可信/严重”主观文字：B1−B0两seed均>0、均值≥.001且所有旧guardrails通过，并且B1−B2两seed均>0、均值≥.001，才追加两臂×两seed的完整R2-B。其中B1−B0均值≥.003为practical signal，.001≤均值<.003为exploratory signal；这两个标签不能混成实用性能通过。

B1−B2门槛是一项固定**资源分配规则**，不是必要的科学真理，也不是B2等效性检验。门槛失败时只能说“本预算未显示额外任务粒度必要性，故不追加”，不能称已经证明B2足够、B1与B2等价，或AO-04整体无价值。尤其B1/B2都比B0改善而相近时，省掉B3/B4会留下简单非均匀收益的竞争机制未排除；应保留这个未识别部分，不冒充层级D算法机会结论。

gate应指定selected validation端点、差值方向/聚合式、exact≥与>、保护臂范围、缺失/失败的处理、冻结文件与触发后的固定四attempt。只因B1超过B2 .001但B1不超过B0 .001不能启动；B2强而B1失败不得将B2改称预注册主方法。禁止R2-A每seed先看test再决定做哪个臂，或只追加更有利的一seed。

## 4. 测试封存与既有runner的风险

必须将train/validation/state保存与final test evaluator隔离。主Agent只读检查发现历史S1 runner的fit末尾直接进入deferred test，旧write_outputs同时依赖test数组并保存checkpoint；因此裸复用fit/write_outputs会在阶段A结束、条件gate前碰test。未来Coder须在独立新入口保留B0训练行为，同时独立实现训练状态writer和最终test步骤；这属于未来必要实现工作，本轮不执行。

实施review应有test访问tripwire/计数审查：Adecision manifest冻结前test预测/metric调用=0，新增B阶段启动决定仅由validation；若启动B则所有10attempt与配置/selected/扫描向量冻结后才test，否则先冻结终止决定与A的6状态，再按已锁定协议test。test后不可重开B、增臂/seed、改阈值或换数据集。

Hpost冻结扫描与B1训练分开：checkpoint缺失不伪造，也不把扫描阳/阴作为B1前置gate。旧test已被历史研究使用，仍需披露，不能称全球未触碰holdout。

## 5. Baseline恢复与预算

一次有界权重搜索须限定根目录/具体候选族、扩展名、时间/遍历范围、最多一次，以及拒绝访问的报告方法；不无限搜磁盘/网络/备份。恢复weights只能证明某模型可load，不能自动证明它与当前runtime新训练候选具有相同初始化、曝光、optimizer和选择协议。建议将恢复的历史state用于验证核验/Hpost，当前runtime的两B0仍计入R2-A六attempt；不得把节省尝试挪给新增方法。

失败也计attempt。A最多6、B最多4、总最多10/500run-epochs；若某臂失败缺失，不能假装阴性，也不能补第11次完整训练。精确resume仅限完整model/两optimizer/RNG/data位置与相同源码/config/hash，不能将无完整状态的从头重训称resume。预先明确恢复次数与队列停止规则。

baseline preflight与实现审核均需在正式B1/B2前通过：源码和data/split/encoder指纹、当前runtime、B0无改目标/训练规则、初始化同tensor、六格路由/输出、参数optimizer唯一归属、B4δ为0的初始等价、有效梯度、训练状态保存/重载、test隔离。B0训练后的selected验证replay是其自身训练产物完整性验收，不应拿旧test绝对数字作启动筛查。

## 6. 结论层级与证据强度

| 层级 | 可以说什么 | 不能说什么 |
|---|---|---|
| A 描述性 | 权重/系数/门熵/梯度和训练轨迹发生何种变化 | 学得非均匀等于真实亲和或AUC可提高 |
| B 初步signal | 两seed方向一致、达到固定exploratory线 | 统计显著/稳定算法有效/层级D |
| C practical通过 | 预定宏AUC .003及全部guardrails满足 | .003是普遍显著性线/自动扩展授权 |
| D 剩余研究空间 | 所测B2/B3/B4替代不能覆盖固定信号 | 全部容量/优化/亲和机制已识别、论文新颖性成立 |

两seed新算法效益证据最多WEAK。逻辑边界/B4系数冗余推导为STRONG；实际哪条竞争机制主导仍未测。失败pilot不能证明非均匀共享普遍无效。域大小/难度、AUC/LogLoss、task/domain、baseline和selected/common训练阶段仍必须分别报告，不预设小域等于困难域或M3OE必然失败。

## 7. 三份v1.2初稿的正式初审

实际已完整读取 `R1_Experiment_Charter_v1.2.md`、`R1_Charter_Change_Log.md`、`R2_Execution_Plan.md`。**初审verdict：PASS WITH FIXES。** 设计只需以下具体修订；在blocking闭环之前不能判最终PASS。当前不存在实施PASS或执行许可。

| ID | 优先级 | 问题与具体证据 | 最小修订 |
|---|---|---|---|
| P1 | BLOCKING | Plan宣称test封存，却未把已知旧runner自动test风险转为实施要求。Reviewer只读核实 `experiments/stage_S1_run.py:794` 在fit尾部调用`_evaluate_deferred_checkpoints()`，第809–812行使用test_loader预测；`write_outputs()`第885–888行无test数组就报错，第896–903行才写checkpoint。裸复用这些接口将使B0/A各run在Adecision前触碰test，或使val-only训练无法保存必要state | 新隔离runner明确独立train/val、无test依赖的state writer、统一finaltest evaluator；禁止裸调用旧fit/write_outputs；在G0及独立实现Review加入静态调用链检查与动态tripwire/计数，decision冻结前test预测/metric调用=0 |
| P2 | CORRECTION | Plan第7节“raw_bD和其他四raw”容易误写成共5个旧scalar；原图总共4个，其中余3 | 改为“原四raw含raw_bD，另三保持”，计量为B4新增6、原4不增不删 |
| P3 | CLARIFICATION | gate以.001/.003及guards精确边界判定，不能让展示用四舍五入数决定YES/NO | 纯函数gate使用原始未舍入的selected validation metrics、保存full precision D/T与所有guard值；展示表不作输入 |

已通过的要点：B4精确定义和原raw/新增delta两optimizer分工；初始化δ0、六新增参数和task-major；共同shift与快速global/格混合解释明确；A全部6有效完整且selected差固定；practical/exploratory和B1−B2资源门；门失败不证等效；旧guards仅B1候选决定追加；B0当前runtime匹配重建及G1 reload；缺R0报告分类；10attempt/500epochs失败计数与有限精确resume；Bcap排除；Hpost默认disabled与前置开关；层级A/B/C/D和两seedWEAK；旧test历史曝光及test不可重开。

无需为B4冗余增加臂或改变定义。P1是防止已知实现入口违背设计的必要hard gate，并非要求本轮编写代码或运行检查。P2/P3可以文档修订闭环，本轮仍只做R1。

## 8. 修订后最终复核

**最终verdict：PASS。P1/P2/P3全部CLOSED；没有未关闭的设计blocking issue。执行准备度仍为NOT READY。** 独立Reviewer实际重读了三个文档的修订条款和完整Change Log，复核B4/参数/optimizer、原baseline协议、条件追加、test隔离、恢复和预算；必要源码只读检查限于已知S1 runner的test/save调用链。未重新开展完整R1或文献排名。

| 初审问题 | 修订后的实际复核 |
|---|---|
| P1旧runner自动test | Charter§8与Plan§4明列旧fit自动deferred test、write_outputs依test再写state的危险；禁止裸调用，拆train_val/validation_selection/state_writer/finaltest。Plan G0把test数组缺省时仍能保存完整state、静态审查及synthetic tripwire解封前零预测/metric列为实施Reviewer hard gate。Adecision前及B启动后到全局解封前均=0；baseline查验也先validation-only replay。CLOSED（设计要求；尚未实现/运行验证） |
| P2原raw数量 | B4保留原四raw（raw_bD+另三），恰新增6delta；全表参数账B0/B3=1,614,258、B1/B4=1,614,264、B2=1,614,261；B3 buffer6另列。按parameter IDs和原四raw显式集合审核分组，避免名称误归。CLOSED |
| P3舍入与门槛 | Charter§5/Plan§6明确D/T/mean/guard均用原始未舍入metrics；stage直接执行>0及>=/<=，无新容差；Hpost1e−6仅用于grid并列。CLOSED |

其他重要边界同步通过：B4取真实未sigmoid `_weight_bal_d.deep_weights`，不双sigmoid；δ0与B0数值等价要求明确；不把B1 other质量不变验收错套B4；raw始终arch、δ与q同main，每步只由指定optimizer写参数。共同shift冗余、最多5个额外b向量方向、δ均值快速global调节和格异质性的混合estimand已完整披露，无新增臂或约束。B1优于B4不识别真实亲和、不穷尽容量/优化机制。

### 8.1 追加gate足够明确，但结论有严格边界

最终追加YES当且仅当：A六份有效完整配对；两个selected validation D_s=B1−B0均>0且mean(D)>=.001；原B1−B0全部guards通过；两个T_s=B1−B2均>0且mean(T)>=.001。mean(D)>=.003为实用筛查、.001<=mean(D)<.003为探索。两个级别都不放宽guards，不用test，不从显示表舍入值决策。

该gate是一项自洽、可复现的有限资源规则，足以决定是否按预注册追加B3/B4，**不足以证明B2等效/充分或AO-04无效**。文件已明确NO分支保留B1/B2静态非均匀信号及未测竞争解释；B2普通性能guard失败不自动成为实现INVALID，其自身信号必须另按同guards判断。没有B阶段就没有层级D检验。两个seed的性能/剩余空间证据最多WEAK，不把.001/.003当统计显著性。

### 8.2 预算、恢复、配对与封存

未来仅在另行正式授权与独立实现Review通过后：A固定6完整attempt/300run-epochs；B仅YES条件追加固定4；总最多10完整attempt/500run-epochs，失败占名额，不用B剩余额度补失败A，不增加seed/方法/调参。每run至多1次、全队列至多2次完整state的基础设施精确resume；无state不能从头重训后称resume。科学失败、INVALID与INCOMPLETE分别记录。

所有候选从B0训练前保存的未训练初始化tensor/RNG/sampler复制，**不能从已训练selected/final迁移初始化**。last_resume保存一致完整model/两optimizer/RNG/loader位置，与selected检查用途分开。两份当前runtime B0先完整50epochs并通过validation reload≤1e−6，才正式B1/B2；这不是历史环境/旧test数字的严格重现。

一次历史权重搜索边界为固定三个逻辑根、60秒/20000枚举条目/32候选/512MiB上限、指定扩展名、junction物理根限制及访问错误receipt；当前未实际搜索。找到历史权重不自动替代当前paired B0，重建两份B0已计A预算；历史权重缺失不强制阻断可审核的匹配重建。三份R0缺work报告仅保留文档完整性缺项，不伪称读过；source/data/split/encoder/runtime、实现Review与前向/reload/配对/test隔离才是真hard gates。

Hpost默认disabled，只有正式执行前预登记并review开关，不能看A/test结果启用；不是B1训练前置。启用时最多66域级validation评分，1331组合选择风险公开，唯一rho向量在统一test解封前锁定；不test grid。YES分支等全部B结束锁定，NO分支先冻结终止决定再最终test；test后不可重开B或改变计划。旧holdout历史曝光披露不变。

### 8.3 证据与当前停止点

当前没有新算法性能结果，不能作性能GO/STOP。数学结构与逻辑边界可标STRONG；两seed未来效益结论最多WEAK。A/B/C/D层次、逐domain/task AUC/LogLoss、seed差、size与baseline难度、训练阶段和trade-off的分析要求均保留。B4混合调节、专家共同适配、global系数漂移、间接门塔补偿、小域验证噪声、时间shift与优化几何仍是竞争解释。

本轮只新增四份设计/审核文件；旧R1三报告SHA256与本轮开始时完全相同。没有算法实现、训练、smoke、新预测评估或实际checkpoint搜索。本轮到此停止，等待R2正式执行授权；本PASS不省略以后的DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE阶段。

### 8.4 实际复核快照

以下为Reviewer闭环时科学内容快照；主Agent随后只补签署/审核状态元数据时hash会改变，不改变本PASS范围。

| 文档 | SHA256 |
|---|---|
| R1_Experiment_Charter_v1.2.md | C231C764669516226782613A083CD895D702DF07739016E482E8F266C1C2996E |
| R1_Charter_Change_Log.md | E47662E090E4D258DDCF4887066D07456777B13FB84DEFFC57CFE204C5B423E4 |
| R2_Execution_Plan.md | 8AD2F12D8EC22E63B7D589EA8BE66EB7B0B3D34A031A1983164B5CF1F6039574 |

原R1三报告保留hash：Architecture=`A3D65E64C0BF461AF2F17196078CCC12F1D9D1F4F168BA6A69AC8C141A2F83A1`；Baseline=`C4113F2B0EB94D503E269CADC6E8AE0B50B3334D5DE87113149D667C3B7818A7`；v1.1 Charter=`CC23BACF553B206D8CCB4F57F7849D001F1838563883AC5C59A10CE284AB24F8`。
