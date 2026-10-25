> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`4ee562d370b2a26e105f48a067a22df29ec38e597186679524e772f8f316f9e3`。

# 04 Codex执行启动提示词


## 实验验证原则

本项目的目标是**验证算法/机制本身的实验效果与研究结论**，而不是证明整个代码工程不存在任何潜在问题。

执行实验时遵循「**最小必要验证**」原则：

1. **优先完成核心实验，不要主动扩大测试范围。**
2. 发现问题时，首先判断：  
   **这个问题是否可能实际改变训练过程、模型输出、指标比较或最终研究结论？**
3. 只有存在明确的「问题 → 实验结果变化 → 结论变化」路径时，问题才属于**必须处理的问题**。
4. 对不会影响当前实验结论的问题（如代码风格、非关键边界情况、未使用模块、一般性工程风险、理论上可能但无证据的异常），**无需额外验证或修复**。
5. 不要为了提高“代码可靠性”而进行与当前研究问题无关的大量防御性测试、重构或额外 ablation。
6. 如果已有实验结果能够回答当前研究问题，**不要因为追求更高程度的确定性而无限追加实验**。
7. 对发现的问题分为：
   - **Critical**：可能改变实验结论 → 必须修复并重新实验。
   - **Relevant**：可能影响结果但尚未证明 → 只做最小验证。
   - **Irrelevant**：不影响当前结论 → 记录即可，不阻塞实验。
8. **不要预设研究假设或为了证明某个结论成立而设计额外测试。** 实验应由研究问题和已有证据驱动。
9. 在无法证明某个问题会影响结论时，默认采用**最小成本验证**，而不是扩大测试矩阵。
10. 最终报告应重点回答：**算法是否有效、现象是否稳定、结论是否被实验支持，以及是否存在足以推翻结论的关键问题。**
11.严禁在非代码训练阶段派发任何子agent，除了机械的运行实验进程外的所有任务应当只由主agent进行.当实验进入到运行训练阶段，派发子agent：luna_worker进行执行。
12.每个GPU设备应当尽可能地多并发训练进程。当且仅当单张显卡同时满足显存占用不超过80%和功耗不到上限的85%，不要求两张显卡同时满足要求
13.整个实验必须完成所有规定的流程，除非是满足设定的提前最终停止条件，否则严禁在实验未完全进行完毕前停止。

版本：**MDMT-NEXT-v2.0-design**；日期：**2026-10-25（Asia/Tokyo）**。  
用途：这是交给下一次Codex实施会话的完整任务文本。下面的实现与训练指令在用户将本文件交给Codex要求执行后生效；本次方案编写没有执行这些训练，也没有修改远程仓库。

---

## 你的任务与完成目标

你是本项目的研究实现者。请在 `Caelestis1903/My-Research-Project` 的实际代码和服务器环境中，完成下一阶段MDMT研究的独立实验包、有限分阶段训练、分析与报告。

本任务关注两个问题：

1. 真实共享更新收益能否被可靠利用；简单历史反馈是否已经足够，状态ridge/NN是否有独立增量。
2. 来源内部多个知识单元能否服务同一目标域内部的不同任务/样本需求；根据收益选择内容是否超过域级规则、相似度与同信息直接BCE门控。

首先实现 **RHD-C8** 和 **LPS-RO** 两条独立主线。不要从一开始联合C8更新控制、多原型读取、动态RW、事件刷新和预算采样。不得预设动态、多原型或NN必然有效。

你需要实际完成代码与已登记的有限实验，不能只生成计划。按照阶段合同自动推进，不因每个模块、单次训练或负结果中途索要确认。允许创建新的独立实验包、必要的隔离工作树、配置、恢复文件、分析脚本和本地报告；保留历史源码和结果，不强制覆盖工作区、不重写历史、不擅自推送/合并/发布。设备和文件访问仍服从现场已明确的权限；缺失条件不得假装已满足。

## 一、先完整阅读，建立可核实上下文

把用户提供的以下三份新规范保存到一个明确的design目录并完整阅读：

1. `01_实验反思与算法重新设计.md`
2. `02_可靠收益决策与多原型共享算法规范.md`
3. `03_实验矩阵与研究决策规范.md`

本提示词给出启动必需上下文；公式与细粒度接口以02为准，矩阵/预算/评价以03为准，历史事实及论文边界以01为准。若冲突，写 `SPEC_RESOLUTION.md`，按保持数据角色、受控对象、主目标和对照一致的最小修正解决；改变这些研究语义则升级design版本，不能静默修改。

必须完整读取仓库 `10-24` 分支下：

~~~text
research_reports/2026-10-24/FINAL_REPORT.md
research_reports/2026-10-24/S1_REPORT.md
research_reports/2026-10-24/S2_REPORT.md
research_reports/2026-10-24/REPLAY_REPORT.md
research_reports/2026-10-24/README.md
~~~

报告固定提交：  
`b64b29222fcd36ca8ecbb323ccb127f7da1ccd57`。

这次提交只有报告与统计图。报告中的源码摘要  
`777ad7131d913268e5a51b708b012263059d557aaccde10d8589d85ab59f974d`  
不是已经核实的Git commit；服务器代码、原始配置、日志、checkpoint和结果字节必须现场核查。找不到旧checkpoint可以从头训练MVP，不能因此停止整个研究，也不能把它写成已经存在。

继续参考已核实的旧源码：

- DG/10月21日固定提交：`c5618e281785a4b7063ff9bf2a61cf38d4faf1be`。
- 自然数据复核：`f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e`。
- CM/10月19日：`eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c`。

重点路径：

~~~text
experiments/reproduction_20261021/DESIGN_STAGE2.md
experiments/reproduction_20261021/data.py
experiments/reproduction_20261021/audited_data.py
experiments/reproduction_20261021/protocol.py
experiments/data_generate/MDMTRec/models/multi_domain/fixed.py
experiments/natural_ml_reassessment/controlled.py
experiments/natural_ml_reassessment/mechanism.py
~~~

存在上一轮三份 `研究方案修订与扩展/算法与Codex执行规范/实验矩阵与研究决策` 文件时也作历史对照，不能机械继承它们已被v2明确修正的设计。

必须保留的历史判断：

- I1_PERM对固定RW的selected macro AUC：ML −.000334，KR −.000513，未确认原ranking优势。
- KR final LL −.008375只是旧协议次要信号；旧last epoch不同，不能当已证实的同阶段收益。
- ML有selected时尚未决策的seed，KR已有较多决策仍未取得主优势，启动过晚不能解释全部失败。
- 旧C_STATIC在ML选中Equal，但旧动态C7没有Equal，需统一动作再比较。
- S1非零效用不等于可预测，也不等于完整训练收益；B_UNIT/B_ADAPT的joint空间仍未充分测试。
- 旧test和Q_confirm已经被研究者分析，新seed不能把它们重新变成未见确认数据。

## 二、先创建的独立代码包

新包使用 `experiments/mdmt_next_v2_20261025/`；不要覆盖历史脚本。至少建立：

~~~text
run.py
data/roles.py
data/plans.py
models/fixed1_adapter.py
models/ownership.py
models/anchor.py
models/source_bank.py
actions/c8.py
actions/a51.py
utility/state.py
utility/update.py
utility/forward.py
policies/history.py
policies/state_residual.py
policies/local_tree.py
policies/direct.py
train.py
evaluate.py
costs.py
schedule.py
analyze.py
baselines/
configs/
designs/
~~~

共同runner处理角色、RNG、恢复、账本、选择和评价。算法模块只处理本体，不能绕过共同访问边界。run.py须提供prepare/smoke/train/probe/replay/analyze/freeze/evaluate子命令；**没有训练完自动评价test的默认分支**。

真实旧接口是：

- `load_data(dataset, cache_root)`，并没有你设想的任意row-role API。
- `MDMTRec.forward(x, test_flag=False)` 输入[F,B]，返回[B,2]概率，没有logits/features/source_mask。
- 旧protocol的lr=.01和六格mean相加，不是本次sample/task mean。
- 旧risk_clone依赖特定替换，不能作为新图的通用梯度接口。

新增Fixed1Adapter由真实前向机械提取：

~~~python
Batch:
    x       # [B,F], float32，类别列为可精确表示的整数编码
    y       # [B,2], float32
    domain  # [B], int64
    row_id  # [B], int64
    role

NativeFeatures:
    logits     # [B,2]，sigmoid前，不反解clamped probability
    embedding  # [B,d_in]，已审schema下ML113、KR544
    experts    # [B,4,64]，保留固定shared expert顺序
    domain     # [B]
~~~

适配器内转置一次，保留STAR、skip、gate.detach、LayerNorm和原生分支语义。bank旁路应数值等价原模型。原生共有专家不是纯source域专家；所有结论明确控制范围。

## 三、共同数据、训练、评价合同

### 数据身份与角色

原始路径和SHA：

- ML：`/home/xzf/data/ml-1m/ml-1m.csv`，  
  `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`。
- KR：`/home/xzf/data/kuairand/processed/kuai_rand.csv`，  
  `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。

ML年龄域{1,18}/{25}/{35,45,50,56}，任务rating>3与>4；KR tab0/1/4，读取已有is_click/long_view，不猜long_view阈值。原始自然行序外层80/10/10不是已经证明的用户互斥或时间外划分。没有真实user ID就不报告GAUC。

优先核对旧roles manifest；复建时使用03的整数切点与PCG64合同。original train逐域升序row_id、seed20261024+1009*d，切点(8*N)//10、(9*N)//10得到fit/probe/query；probe偏移20000按60/20/20分P_label/P_cal/P_update，query偏移40000对半。

核对总行数：

| 数据 | fit | P_label | P_cal | P_update | Q_explore | 旧Q_confirm |
|---|---:|---:|---:|---:|---:|---:|
| ML | 640133 | 48010 | 16003 | 16004 | 40008 | 40009 |
| KR | 1604904 | 120367 | 40123 | 40123 | 100307 | 100308 |

P_label按03新seed再诚实分为P_structure/P_value；P_value、P_cal各自固定8块，所有方法共享行身份。编码器、scaler、OOV及prototype只拟合fit。所有panel/support以纯 `at(anchor_id)` 索引访问，具体seed见03，不让probe消耗主loader位置。

角色职责：

- fit：主训练、来源知识、原型、真实更新support。
- P_update：RHD更新效用学习。
- P_label：LPS收益/门控元训练；P_structure学结构、P_value估响应。
- P_cal：回退/尺度/γ/深度选择，属于元训练/元选择。
- Q_explore：开发机制与晋级；不能进主模型或效用拟合。
- validation：预定checkpoint及有限配置选择。
- 旧Q_confirm、旧test：全矩阵锁定完成后的已暴露基准复评。
- 真正新工件：只有核实此前未用于任何训练、开发、选择或分析，才可登记UNSEEN_CONFIRM。

没有新数据时标 `UNSEEN_CONFIRM_PENDING_DATA`，继续MVP与锁定复评；不能重切旧CSV或换seed造“新确认”。

### 默认训练与选择

- Adam .001，betas(.9,.999)，eps1e−8，**耦合**weight_decay1e−5。
- 有效batch40960，natural shuffle，drop_last=False，workers0。
- FP32，AMP/TF32关闭，无scheduler/clipping。
- 主损失是所有自然行×两个任务的BCE mean。
- pilot固定24epoch，正式固定50epoch，无质量早停。
- warmup到10，主选点eligible从12开始；同时保存AUC-best/LL-best与共同epoch24/50。
- RHD预定新主张risk：validation macro LL选点；LPS主张ranking：validation macro AUC选点。不能用更好看的选点切换主结论。
- S2晋级得分在Q_explore上评价被选checkpoint；validation并列报告。
- 正式主要基准为benchmark_test_exposed，旧Q_confirm人口迁移另表，新工件若有独立并列。
- 开发seed3101、3102；正式4101、4102、4103、4104。禁止不显著就无条件追加seed。

合法微批必须保持同一逻辑batch、同动作、完整分母、一次Adam step；OOM不能静默缩support。不同模型结构不能伪称共享相同warmup。

## 四、第一条必须实现算法：RHD-C8

### 受控对象与八个模板

只改变原生共享参数组的fit梯度：embedding、共享STAR/skip、四shared experts及共享task experts等按真实图列PARAM_OWNERSHIP。域私有参数始终走自然目标梯度。不能仅根据参数名猜归属。

设当前完整逻辑batch域比例为p，各域两任务mean loss为L_d：

- BASE：α=p。
- SHARED_EQUAL：α=(1/3,1/3,1/3)。
- 六个i→j：α=p−δe_i+δe_j，δ=.25 min(p_i,p_j)。

同一forward计算共享梯度∇Σα_dL_d、私有梯度∇Σp_dL_d，组装后只做一次真实Adam。模板相同不代表每batch数值α相同。缺域时按02合法mask去重，非法静态回退映射BASE。

### 真实效用与历史优先

从同完整状态S及同fit support出发，对C8执行真实H1；评价P_update与独立P_cal每域1024行，标签是相对BASE更新后风险下降的六维向量。H5另ID、另矩阵，不混成H1的低噪声标签。

实现以下策略：

1. BASE。
2. STATIC8_DEV：3101的epoch10共同快照，八固定模板各续训5epoch，固定epoch15 validation macro LL选a★；两数据各一次，所有动态方法获得同a★。
3. LAST_FEEDBACK：最新合法收益，使用其明确回退合同。
4. HIST_SHRINK：最近32锚点、半衰期8、动作/格相关的有限面板误差收缩；根据过去prequential误差范围与MC误差形成经验margin，不能超出margin则回退a★。
5. RIDGE_RESIDUAL：24维可见state，action×state残差，带截距、lambda1、float64；不能把可分离state项写成在argmax中抵消的伪状态控制。
6. NN_RESIDUAL：24→16→42，ReLU、末层零初始化；same history与label预算，按02每锚点50小步、完整锚点采样。
7. FREEZE/mean/EMA作为共同trace的低成本机制对照；有明确机会才从reserve升完整训练。

保留三层比较：static、history、history+state。状态残差混合γ∈{0,.5,1}仅用过去P_cal选择；γ0严格退为HIST。所有γ的raw提议即使未部署仍记录，校准raw state相对HIST raw，不能造成γ永远无法开启的自锁；最终动作仍须通过相对保守HIST的margin gate。没有概率覆盖或长期安全保证。

### 时序

- warmup epoch5–10，每epoch half/end探测，共12锚点。
- 前4只初始化尺度，后8严格先预测后揭当前标签。
- 首次a★尚未产生时封存事件；静态校准后按时间重放，不能一次拟合全部反馈后补写历史预测。
- FREEZE_WARMUP使用12锚点选择，只能从13以后评估；不能在其选择数据上称前瞻。
- epoch11起每个主batch可用当前可见loss/moment摘要选动作；每half/end反馈。
- 24epoch共40锚点/320逻辑分支更新；50epoch92/736。
- BASE/STATIC不承担部署不需要的持续probe；研究shadow成本单列，不能故意抬高静态成本。

同一主batch先清grad、一次forward、detach状态特征、选动作、两路求导、一次Adam。当前fit标签可用于更新状态摘要，当前P_update/P_cal反馈只能在提交预测之后揭示。controller RNG与主模型隔离。

## 五、第二条必须实现算法：LPS-RO

### 内容与原型

native四专家分别做无仿射LayerNorm，拼接为256维，不先平均。三个来源各有K_content=4个rank4单元：

\[
e_{sk}(x)=U_{sk}\operatorname{GELU}(D_{sk}h(x)),\quad
D:4\times256,\ U:64\times4.
\]

每source共5120参数；K1内容对照用一个rank16同为5120。reader为目标域×任务×来源的64维向量，同来源四单元共享reader。

epoch5复制native embedding为冻结anchor，通过固定随机32维投影与归一化形成a(x)。每来源只用最多20000 fit输入拟合K_key=4中心；K-means++、n_init1、最多50迭代，温度只用fit输入将平均责任熵调到.5log4。中心/温度/anchor随后冻结，主native继续训练。

**K_key与K_content分开：** K1也保留4个key和相同42维需求输入，但只有一个content、内容责任为1。几何距离不等于目标收益。原型退化标PROTOTYPE_DEGENERATE；单独K1 fallback有新ID与预算，不能替代失败K4 seed。

### 前向与权限

own logit = native + 本域reader×责任加权owner单元。主BCE正常训练native、owner与reader。

外域RO读取同时detach source输入h和source输出，target reader仍可训练；source只由其owner路径更新。RO不冻结source全部学习，也不清零其Adam历史。

epoch6起默认aux=.1：以detach(native logit)+owner reader×unit(detach h)计算责任加权局部BCE，除以自然行×任务数，不对原型各自mean后再等权。aux只更新owner unit/reader。aux0变体从epoch5或从头分叉，在6–24全程aux0。

### 51动作与需求

每目标两个外来源、8个单元；每任务可单独选择：

- OFF：1。
- 单元×强度.5/1：16。
- 任意两个不同单元各.5，可同源/跨源：28。
- 任一来源aggregate×.5/1：4。
- 两来源aggregate各.5或各1：2。

总计51。aggregate用**来源自己的责任**，裸单元不再乘几何ρ。两个任务执行单元按并集去重；aggregate可能执行8个单元，不预先宣称稀疏加速。

K1去重后7动作。其coverage从相同A51虚拟slot抽样再映射/合并系数，不能把重复slot当更多内容。策略输入为anchor32+own logits2+两外来源key责任8，共42维；无当前真实标签。

### 收益策略与强对照

当前snapshot缓存逐行z0、单元残差、ρ，标量组合得到51个logit，计算精确配对BCE收益；不使用旧二次近似。收益最大化和同动作BCE最小化代数等价，不声称额外标签信息。

LPS51每(t,q)一棵诚实树：

- P_structure建最大深度2的轴对齐树，节点20/40/60/80分位候选，子节点至少256。
- P_value估根/叶收益，叶值按n/(n+128)向根收缩；n<64回根。
- 固定比较动作身份做8块配对jackknife；相对coarse的margin超过1个经验SE才用局部动作，否则回粗策略。
- P_cal只比较六树共同深度上限0/1/2的三个策略，以1SE规则选最浅。
- 每2epoch按当前snapshot重建，不把过期标签直接混入。
- 稀有群体/相似原型可以退到coarse或K1；不要求来源排序或符号反转。

必须实现K4_OWN_OFF、GLOBAL51、DOMAIN_GLOBAL51、SIM51、DIRECT51、K1_DIRECT7、LPS51及缓存上的CAL_PROB_SELECT/CAL_OFF。

DIRECT是42→16→51或7，ReLU、softmax温度1，最小化**动作BCE的期望**，不是混合logit的BCE；Adam.001、wd0、batch512，首训10epoch、刷新2epoch，继承被选权重和Adam，P_cal选择包括刷新前状态。CAL用42维logit残差逻辑校准、固定ridge1、float64 L-BFGS、三尺度{0,.5,1}，求解/两折/P_cal规则严格按02。

K1/K4只匹配bank，不自动匹配actor参数：六头4842 vs9330。要声称纯容量无关内容增量，执行02的K1_hidden31敏感性或明确未排除限制。CAL竞争力强时必须进入后续完整闭环与正式比较，不能隐藏。

### 日程与部署

- epoch1–5：owner、均匀内容责任、aux0、外域OFF。
- epoch6–10：固定原型、默认aux.1、RO覆盖；OFF/single/pair/aggregate类型各1/4。
- epoch10末拟合策略。
- epoch11后：选择器90%策略+10%同coverage；OWN_OFF/CAL_OFF始终OFF，免除外域coverage和不必要A51缓存。
- 每偶数epoch末刷新，包含预定终点；然后validation与保存完整predictor+actor。pilot8次/正式21次策略拟合事件。
- actor输入/输出在主fit均detach；P_label/P_cal只训练策略，不把其梯度传入主模型。
- 推理只用输入、当前预测、固定anchor、actor和选中知识单元，不可访问真实标签。实际全算unit时照实记成本。

## 六、完整状态与最小验收

必须区分 `READ_FROZEN`、`UPDATE_H1/H5`、`TRAINING_POLICY`，各自单独存estimand。

完整snapshot含模型、buffer、parameter groups、Adam m/v/逐参数step、Tensor/None梯度、模式、requires_grad、CPU/CUDA/NumPy/Python RNG、主/探测batch计划与位置、bank/anchor/reader、actor/critic/normalizer/replay、预序列预测、权限背景、counter及best checkpoint引用。

回滚原位恢复对象绑定，不恢复后继续使用旧controller或旧optimizer引用。纯anchor计划在事务外只提交一次；探测期间禁递归验证、早停、探测或策略更新。外部实际成本及标签访问账**不可回滚**。

H1可以缓存三个域共享梯度与自然私有梯度，但必须恢复forward造成的buffer/RNG，再执行各候选真实Adam；一个锚点对完整重算验收。H5逐步重算，不能复用初始梯度代替五步训练。

只做影响结论的验收：native旁路、loss/aux分母、C8分组梯度、RO/RW/grad=None、全状态恢复、无泄漏预序列、A51/7及BCE等价、checkpoint/成本/角色gate。通过即推进；不搭建无关庞大测试平台。

## 七、有限实验顺序与自动推进

### P0：身份、接口、权限与测速

先核raw/roles/source，创建参数/动作/暴露manifest，完成必要验收。核实当前允许GPU：10/24报告的最新政策是所选卡memory≤85%、power<85%、无固定进程数；不要自行恢复旧GPU0,1或使用所有卡。只有更早明确GPU0限制可核实时保持该限制，不处理其他用户进程。

用真实batch与首个完整warmup epoch量出各数据集native epoch时间t_d及probe/cache/actor组件，不用旧共租耗时猜新GPU小时。预留mandatory aux及阶段预算。

### S1：最小机制

seed3101，两数据集×四家族各10epoch：

1. native BASE+C8 shadow。
2. concat K4/rank4。
3. concat K1/rank16。
4. mean K1/rank16桥接。

共8条warmup；C8 H1为2×12×8=192逻辑更新，epoch10 H5为2×8×5=80步。利用相同风险预测切片比较256/1024行，不新增未登记support分支。保存epoch5与10完整状态。

检查动作机会、重要gap、真实预序列决策，以及域间/域内/可部署三层需求；同source aggregate上比较旧二次代理与精确BCE，不把组合动作差异归为代理误差。

S1无阳性不是停止joint MVP的条件。实现正确就继续S2。

### S2：必须完成的完整开发矩阵

开发seed3101、3102，两数据各24epoch：

| 家族 | 六个核心arm | 轨迹数 |
|---|---|---:|
| RHD | BASE、STATIC8_DEV、LAST_FEEDBACK、HIST_SHRINK、RIDGE_RESIDUAL、NN_RESIDUAL | 24 |
| LPS | K1_DIRECT7、K4_OWN_OFF、GLOBAL51、SIM51、DIRECT51、LPS51 | 24 |

共48条逻辑完整轨迹。STATIC校准另为两数据×8模板×5epoch，共16短分支；先确定a★再完成合法bootstrap。可以复用相同完整warmup，不伪称不同carrier共享相同bank状态。

RHD共同BASE trace共40锚点，前26开发、后14按冻结规则在线前瞻评分；数值标签一致的共同轨迹和各策略改变训练后的闭环分开。实际gated regret与raw proposal gain分开。

reserve最多288个实测native等价epoch，默认每数据144，包含反馈/cache/actor，不按逻辑epoch藏成本。mandatory先跑K1/K4×aux0/.1析因：已有aux.1，新增4条aux0×24；正确从epoch5分叉。超限则暂停可选包；必要包仍不足时标PENDING_BUDGET并限制归因，不无限扩大预算。

其他reserve仅针对明确竞争解释选择最小完整包：actor容量、FLAT/IDENTITY、CAL、reader适配、H5、权限/组合/稳定性等；开放问题数不限，新增运行数有限。

晋级按03明确规则：Q_explore selected配对、正确简单参照、实际状态使用、同轨迹可部署增量与成本；不以MSE单独晋级NN，不因旧test某格好看改变主目标。

### S4适配开发与S3锁定复评

S2之后、首个正式训练之前，默认完成AutoIFS、STEM_DOMAIN_TASK、STGU域组Adam适配的忠实性审核和有限pilot。适配额度最多3方法×2数据×3101×24epoch等价量，**独立于288 reserve**；多阶段搜索/重训全计费。材料不足可在锁前改用CDC或HoME/SMES，按契合与可获取性决定，不能看正式test挑对手。

STGU须保留proximal affinity中task-specific变化、分组与逐组顺序更新，未激活private参数保持None；仅EMA动作选择另名inspired。PRP部分作者代码有核心缺段，M2R只取得摘要，不能脑补后声称原版复现。

正式seed4101–4104、固定50epoch、eligible12。最多：

- F-R五臂×两数据×四seed=40。
- F-L七臂×两数据×四seed=56，含native参考。
- 近期三槽×两数据×四seed=24。
- 总上限120，不要求耗满；近期24已经包含，不再额外加。

仅RHD晋级也必须安排8条native参考。状态未胜历史则不强跑状态正式包。强CAL成立必须纳入：先占未晋级空余，满额时预先替换SIM正式槽，保留全部S2 SIM结果。不能替换已经跑完的不利结果。

首个正式train之前冻结MATRIX_LOCK，包括算法/预算/数据/seed/比较/选点/外部版本与失败处理。全部入包train完成或预定失败后统一外层评价；不得边看test边决定训练谁。新确认缺失不阻塞已暴露基准复评，结论准确命名。

## 八、建议CLI合同与工件

以下是**须新实现的命令合同**，不是宣称旧仓库已经支持。先实现并检查help，再用真实resolved配置运行。目录可按实际工作树调整并写manifest。

~~~bash
python -m experiments.mdmt_next_v2_20261025.run prepare --design-dir research_plans/mdmt_next_v2_20261025 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run smoke --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run train --stage S1 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run probe --stage S1 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run analyze --stage S1 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run train --stage STATIC_CAL --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run replay --stage BOOTSTRAP --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run train --stage S2 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run analyze --stage S2 --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run train --stage BASELINE_PILOT --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run freeze --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run train --stage FORMAL --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run evaluate --role Q_legacy_confirm_exposed --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run evaluate --role benchmark_test_exposed --campaign-dir artifacts/mdmt_next_v2/v2_0
python -m experiments.mdmt_next_v2_20261025.run analyze --stage FINAL --campaign-dir artifacts/mdmt_next_v2/v2_0
~~~

run train S1的warmup内C8探测属于其回调；run probe S1必须识别已有事件ID，只执行未完成的H5/缓存诊断，不能重复收费或再生成另一份“相同”反馈。S1分析中依赖a★的结论暂标待bootstrap，在STATIC_CAL/受控replay后补全再封版。

S2调度包含核心矩阵和必需aux包。analyze发现满足预定触发的reserve比较时，生成有预算的`S2_RESERVE`子矩阵，执行后再封版S2报告，不让待测消融被误当已完成。若已核实UNSEEN_DATA_LOCK，正式完成后还须执行`evaluate --role unseen_confirm`；没有新工件时记录PENDING并跳过，不用旧Q替代。各外层role都在同一冻结矩阵下评价，不相互选优。

命令必须支持幂等调度、attempt、已有完整checkpoint精确恢复及失败收据。不能靠修改文字summary掩盖退出码未知。纯日志错误可修，计算受影响则重跑对应比较。

每run保存：

- resolved YAML、source/data/action/parameter/design hash。
- events、epoch/cell指标、label_access、cost、utility/prequential表。
- epoch5/10、primary/AUC/LL-best、epoch24/50、last、budget_best完整checkpoint；不适用项NA。
- 实际raw预测/必要充分统计、选点依据、完整状态manifest。
- attempt失败、退出码或UNKNOWN、中断与恢复原因。

campaign根目录的`ARTIFACT_INDEX.json`统一映射02语义名和03真实路径/hash，例如DATA_MANIFEST与manifests/data_roles.json是同一工件；不得要求重复保存或误报缺失。

阶段报告至少有 `IMPLEMENTATION_REVIEW.md`、`S1_REPORT.md`、`S2_REPORT.md`、`FORMAL_REPORT.md`、`BASELINE_REPORT.md`、`FINAL_REPORT.md` 和 `OPEN_QUESTIONS.jsonl`。主报告每个判断提供真实表格/工件路径，不只贴曲线。

预算同时给研究执行、独立部署、开发搜索三张账。同epoch与同总成本各自评价；预先用STATIC或DIRECT的50epoch实测外推设预算前缀。阶段时间由03组件公式加25%执行余量冻结；到上限记BUDGET_EXHAUSTED/PENDING_BUDGET，不缩动作或面板冒充同一算法。

## 九、允许自主修复与必须升级版本的边界

可自主修复并记录：路径/API适配、依赖兼容、dtype/shape、grad=None、状态恢复、清晰的计算图错误、日志和数值稳定性，以及不改变逻辑batch的合法微批。只重跑受影响对照，不建大规模无关测试。

以下改变必须新design/method ID与完整受影响比较：

- 数据人口/角色/标签/编码范围。
- 共享对象、梯度权限、动作空间/幅度、原型训练目标。
- 主评价目标、主要对照、选点起点/训练终点。
- 反馈label角色、状态可见信息、H1与H5互换。
- 对已看过外层结果的算法改动或超既定预算的新搜索。

升级版本不等于每次向用户询问；在本任务有界开发与已有授权内记录后自主推进。超出资源/权限或既定campaign预算的实验作为下一campaign待办，不自行扩大。公开发布、远程合并等不属于本任务。

结果不佳时先分类：没有有利动作、标签噪声过大、当前表示不能承载、局部差异不可预测、预测有用但闭环失效、校准/容量足以解释、或成本抵消。只运行能区分这些解释的有限下一比较。

若历史已经足够，就采用HIST/LAST；若DIRECT已足够，就保留有证据的内容机制；若K1/coarse已足够，就收缩主张。不要为了维持方法名字强行使用NN或多原型。RO未升也不逻辑否决RW共同创造新知识，后者须独立joint权限研究；读收益、写收益、长期收益不得混用。

## 十、最后提交什么

完成以下可审计交付后结束本campaign：

1. 可运行独立包和明确入口，源码身份/变更说明。
2. 所有已执行配置、训练状态、数值、成本、失败与恢复收据。
3. 两条MVP的完整比较，以及有明确理由的晋级/不晋级记录。
4. 锁定多seed/强基线结果，或每项具体的未完成原因；不得把PENDING写成已验证。
5. 每数据集四seed原始配对差值、均值/SD/描述区间、六格损害、selected与同epoch、同预算表。
6. 明确决策：哪一个实际版本值得保留；哪些复杂模块无独立增量；哪些新机制值得另立下一版本。
7. 开放问题记录其证据、竞争解释、最小判别、角色、预算与状态；不限制合理问题数量，但不无上限运行。

结果表必须说明：本轮旧基准已经研究暴露；新确认是否真实存在；近期方法哪些是忠实复现、哪些是适配、哪些仍未核实。停止某版本并不等于否定整个方向，保留已有负结果和研究资产。

**现在从材料、真实接口与P0开始，完成已定义的有限阶段工作；不要停在复述计划或询问是否继续。**

