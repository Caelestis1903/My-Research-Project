# 05｜Codex 实施合同与交接提示词

## 0. 任务边界

本文件供后续 Codex 实现和执行 GA31-R3；当前这次任务只写规划，没有训练。算法与预算以04及 `phase1_protocol.json` 为准。本轮允许 A（含预注册A-online连续步只读探针）、A-probe、B-M、两个数据集必做的G-IID2、达到方差门的数据集额外执行的G-ANTI2；**C3–C7、自适应 MASK、VUG、DBO、参数冻结、共享控制、SAGU/SAOS 旧版不在授权范围。**

当前 main 不必已包含实验分支源码。必须取得固定提交 `ee6e81fdfd37284e6447e8149cb93d04736dba02` 的 `experiments/gradient_aug_20261031/` 和 `research_reports/2026-10-31/`。可 fetch `origin/26-10-31` 后按固定 SHA 读取；不能用 main 上不存在的文件、旧540171版本或新分支后来移动的HEAD替代。无需为了获取来源把整个实验分支并入main。

新代码放 `experiments/gradient_aug_20261031_v2/`；新报告放 `research_reports/2026-10-31_v2/`。旧计划、原代码和历史报告只读。

## 1. 来源与代码位置

| 来源文件 | 已核查职责 | 新实现处理 |
|---|---|---|
| `model.py`，blob `6f80172ba8e2cc829abb1ccea77cabdfa2953246` | fixed1模型、字段slice、Embedding forward hook、RMS噪声 | 保留主干；增加单字段零向量视图及显式noise输入 |
| `objectives.py`，blob `e3fb4f73cb7fc747979747a24a8ec238b739a1ef` | 原样本预算的真实/增强损失 | 扩为K=1/2视图；不改BT分母 |
| `protocol.py`，blob `8d10e865d7d8648254eb0536ae96c33f8633262f` | Adam、RAM snapshot、RNG恢复 | 复用语义；新parser不能沿用40run/P0–P3硬断言 |
| `probe.py`，blob `ecce254a7963a3bc4ce535b2097aa3c262d2aacb` | 同状态联合deepcopy与单步比较 | 改为同θ/B多视图；追加受控连续训练步探针，不能误用单次噪声当方差估计 |
| `run.py`，blob `8e97d0afe4b79447ac4035fc4a9e8fa89a0dda87` | train_one、RAM coordinator、最终评估 | 新coordinator支持A（含在线只读探针）→冻结基线→离线probe/ANTI2门→必做M1与IID2→可选ANTI2→统一test |
| `data.py`、`audited_data.py`、`basic/`、`reference.py`、`metrics.py` | 数据、基本层和评价 | 从同SHA固定来源；不混入其它版本同名包 |
| `resolved_protocol.json` | 五seed、数据路径与完整身份哈希 | 数据/模型/初始化协议继承源对象，新实验差异单独覆盖 |

可以把源包原样归档到新包的 `vendor_ga31/` 并记录blob/SHA256，或使用明确命名空间导入。相对路径适配属于工程移植，必须验证不改变特征次序、初始化顺序、state_dict键、参数数量和计算图。ML参数773470、KR1614254是源代码的断言；不加入额外网络。

## 2. 冻结的形式定义

### 2.1 数据与目标

直接继承源resolved_protocol中datasets的 raw/array/split/encoder/scaler哈希；只允许实际文件路径重定位。不要从10-21报告中的文本抄raw哈希，优先使用GA31实际配置和数据审计。固定自然人口，不调用旧MovieLens人工角色采样。

所有原样本两任务均可用。按真实B大小计算
\[
L_K=(BT)^{-1}\sum_{i,t}[(1-.5a_i)\ell^r_{it}+(.5a_i/K)\sum_v\ell^{a,v}_{it}].
\]
基线只算L_r；目标为空跳过增强forward，仍为L_r。损失c只乘数据项，L2仅在Adam内加一次。评价永远使用未缩放真实BCE及真实输入。

### 2.2 字段与视图

沿源模型 `field_slices` 的实际 sparse-then-dense 映射。ML eligible=`user_id,movie_id,gender,occupation,zip,cate_id`；KR eligible直接继承源配置的33项并记录完整名单。路由字段、domain Embedding和dense全部保持；不得猜“最后16维就是domain”。

M1每个目标样本独立均匀抽一个eligible字段，16维整段置0，不改原ID，不用padding索引，不反缩放其它字段。

G-IID2使用两个独立的标准高斯张量；G-ANTI2使用同一张量及其负值。两视图都使用 clean Embedding 计算的stopgrad RMS、κ=.1，不从已扰动视图再算RMS。不能先平均Embedding，因为ANTI平均输入会回到原始输入，错误地消灭原定非线性视图风险。

### 2.3 RNG

模型初始化和loader generator完全继承GA31：seed_all(seed)，train=seed+11000，validation=seed+33000，test=seed+44000。每epoch原始row_id顺序哈希必须在同seed全部方法臂一致。

新视图RNG与正式loader/模型全局RNG隔离。定义
`derive_seed(text)=int.from_bytes(SHA256(text.encode('utf-8')).digest()[:8],'little') & ((1<<63)-1)`。

训练key为`GA31-R3|dataset|seed|train|global_step|family|view`，family为`gaussian`或`fieldmask`，view从1开始。G-IID2/G-ANTI2共用gaussian/view1，ANTI第二视图直接取负，不额外抽样。probe key加入`probe|anchor_step|batch_index|family|view`，与train绝不重复。记录keys和hash，不使用Python进程随机hash。

## 3. Coordinator和RAM生命周期

默认沿用RAM-only状态策略：不写`.pt/.pth/.pkl`，不把权重存为其他扩展名。各worker可通过内存序列化传回model+Adam+RNG+loader状态，持有E1/E10/E50/selected，重复状态去重；过时非固定selected快照释放。

**正式运行必须在同一coordinator生命周期完成以下步骤：**

1. A阶段15run完成，每run真实训练过程同步采集04§3.1的少量连续步只读统计；按完整validation冻结ML工作基线（优先L2的实用替代规则见04§3），KR固定B-KR。
2. 在所选普通基线RAM状态完成离线多视图probe，冻结各数据集的**G-ANTI2单独触发状态**；G-IID2无论门槛结果均为必做。全部门槛与源代码在首个run前冻结，test此时不可访问。
3. 完成M1十run和G-IID2十run，另仅对通过方差门的数据集执行G-ANTI2各五run；所有B臂从同seed原始初始模型重新训练。
4. 冻结所有selected epoch、共同E10/E50、有效run键和决策，写入selection_frozen.json。
5. 一次统一test评估，生成主表、配对表、保护裁决，然后释放全部状态并记录释放。

不得把stage-A CLI结束后再调用stage-B假装能恢复已释放状态。异常退出且没有磁盘状态时，只能标记中断并从头重跑相关run，不能声称任意step恢复。降低存储开销不是理由去省略实际Adam state。

## 4. 只读探针实现

每个固定锚点用完整model+optimizer联合deepcopy保留参数映射、buffers、m/v、每参数step。真实full梯度、真实target梯度和各view target梯度分开计算；target项都除同一BT。δ_v=g_target_aug−g_target_real，g_a^B=g_r+δ_v。

None只在统计向量拼接时视为零；给optimizer提交反事实梯度时根据实际计算图连接性保留None/非None，不能把所有参数强行赋零。c在提交时乘，wd由Adam内部处理。不同反事实各用新的同源联合副本，不能在正式状态上step后近似回滚。

计算S/V的独立样本组和ANTI相关组明确分开；S估计负值原样记录。训练probe每batch1024；两验证panel每域n_d=min(1024,floor(N_val,d/2))，由固定域内permutation分为不交叠两段。分别计算macro及validation人口加权BCE。零范数cosine、零方差分母写null和原因，不伪造0或1。

G-ANTI2方差门的numerical floor定义为：`V_iid <= 1e-24 * max(1, ||g_r||²)`则该锚点-batch为NA。每seed至少有一个有效E10或selected阶段；不足五个有效seed则该数据集不触发ANTI2，记录测量不足，不追加种子。每seed先对有效batch/去重阶段分别平均V_anti与V_iid，**再取R_s=mean(V_anti)/mean(V_iid)**；跨五seed计算R均值和R<1的数量。计算过程中float64累加；方差负到−1e-12以下视为实现错误，微小负舍入截0且记录。**这些规则只控制G-ANTI2，绝不跳过必做G-IID2。**

A-probe离线最多60锚点-batch、840次梯度计算、300次实际反事实Adam step。另有A-online最多120个**真实训练轨迹上观测状态**、360次额外autograd.grad、0次新增optimizer.step；两项分别出账，实际少于上限据实计数。额外诊断不能自行扩成多步rollout或每step全模型grad dump。

### 4.1 在线时间探针的实施约束

在15个阶段A run中预定窗口：ML的E10/E20、KR的E10/E40，每窗口原训练batch1–4，访问每个真实Adam.step之前的参数；无需额外训练步，且**不得以已经改变的训练梯度代替只读诊断**。固定1024训练行参考B_ref（314159独立采样，跨同seed、臂和窗口相同），同B_ref计算目标真实梯度与两份独立Gaussian目标增强梯度，按|B_ref|T归一化。随机键固定为`GA31-R3|dataset|seed|temporal|epoch|original_batch_index|gaussian|view`（view=1/2），不得调用或推进训练/loader/global RNG；这些诊断梯度是固定参考batch上的条件量，不冒称原始40960行全batch梯度。记录每个视图的差分范数、S_t=<δ_t1,δ_t2>、跨步C_t=<barδ_t,barδ_(t+1)>、逐步均值与波动、原步数/epoch/row_id哈希。S/C可为负，不能截断并假装严格余弦或持续获益；窗口末释放全部暂存梯度向量。

使用torch.autograd.grad独立计算图，不改变正式模型参数、`.grad`、buffers、Adam、model.training标志或RNG；每次只读执行前后核对这些状态。在工程开发模式下，以临时epoch1的4步短窗测试启用/关闭探针的同seed训练轨迹匹配（在约定CPU/设备容差内），正式窗口仍仅按预注册E10/E20/E40执行。在线探针的**目的仅是判断下一轮是否值得研究C4差分时间滤波**，不作为本轮A/B算法训练门、不训练选择器。

## 5. 必要验收，不扩成防御性实验矩阵

**底座身份：** 在目标环境对同初始化、同batch验证原fixed1与新基线的预测、Sample loss、梯度、一次真实Adam更新；核对数据哈希和参数数量。关闭增强不执行无用视图或改变RNG。

**视图/目标：** 同输入K视图、λ=0、Gaussianκ=0、MASK显式关闭均恢复真实loss/梯度；缺目标/缺域/batch1有效；每原样本系数和=1；路由/dense/标签不变；反向配对噪声严格互为负。MASK无逆保留率放大。

**状态/配对：** 无增强B-ML与完整匹配Adam控制在相应m/c、v/c²状态映射后，单步参数更新一致；state counters及None一致。检查探针前后正式参数、optimizer、loader和Python/NumPy/Torch CPU/CUDA RNG不变。

**选模/协议：** 50epoch不早停，无test参与训练或门选择；同seed初始化/原样本顺序一致。明确检验G-IID2必做、ANTI2按统计门单独触发、ML工作基线仅由validation实用替代和AUC规则选择；静态展开应得35必要/45最大run及56,750/74,000个真实Adam更新。原始新代码检查CPU FP32 atol1e-6/rtol1e-5起步，超界先查原因，不要求长FP32训练逐位相同，不随意放宽一次step验收。

工程开发只用seed2022、最多每数据集1个epoch的短检查，不按开发test调λ、κ、字段数或训练预算。不存在预先已训练成功的候选数据。

## 6. 建议文件职责和接口

新包建议包含`source_bridge.py、views.py、objectives.py、protocol.py、run.py、probe.py、analyze.py、tests/`。继承数据/metrics接口，避免顶层`data`、`model`同名导入污染。

待实现入口：
```
python -m experiments.gradient_aug_20261031_v2.run \
  --mode all --protocol research_plans/26-10-31_v2/phase1_protocol.json \
  --run-dir <new-attempt> --devices <available-devices>
```
这是后续应实现的接口，当前目录没有可声称已存在的训练器。`--mode audit|check|all`分别执行静态来源/数据核验、工程检查或整个RAM生命周期。`all`严格按25到45run的预定义分支运行，不把任一候选默认为成功。

可切换配置：view_family=none/fieldmask/gaussian，num_views=1/2，pairing=iid/antithetic，loss_scale，weight_decay，eps，probe_enabled。source_momentum、cosine_gate、shared_control、adaptive_mask、vug、dbo、consistency均固定unsupported；设置true应报错，不以空实现伪装完成。

## 7. 工件和完成标准

保存source_manifest、resolved_config、data_manifest、init/order/noise/state摘要哈希、run_ledger、epoch_metrics、per_cell_metrics、runtime_stats、view_stats、increment_probe、**temporal_probe、temporal_summary及只读轨迹一致性检查**、counterfactual_update、baseline_frozen、candidate_frozen、selection_frozen、paired_effects、DECISIONS、REPORT。保留失败/资源中断attempt，不当算法阴性；不得悄悄缩batch、删seed或换split。

REPORT必须区分实际完成/未触发/阻塞，报告选点、共同E10、E50、全部六格与成本。若B-G未触发，应写未运行而非阴性。整体通过实用门但保护指标恶化，应写条件性TRADEOFF而非全面失败或全面成功。

不上传原始数据/大权重，不覆盖旧文件；只提交新代码、冻结配置、轻量可复核数据和报告。完成规定矩阵即停止，新的联合方法需要新合同。

## 8. 可直接交给Codex的任务文本

请读取本目录00–06及phase1_protocol.json，按05实现GA31-R3，按04执行有限矩阵。以固定实验提交ee6e81fdfd37284e6447e8149cb93d04736dba02为来源，保持自然数据、五seed、Sample原样本权重预算、fixed1和validation选模。先完成必要验收，然后在一个RAM coordinator内完成A阶段15run（含有界在线连续步只读探针）、规定离线probe、必做M1十run和G-IID2十run、仅在方差门通过时额外执行G-ANTI2，最后统一test、分析并提交新结果。ML基线实用替代判断只读validation，G-IID2不受ANTI2门限制，不能以跨优化器旧单视图结果解释新的视图数因果效应。不要恢复旧SAGU排序，不要求MASK胜出，不开发共享控制或未经冻结的来源动量。发生真实阻塞时交付完成内容与具体原因，不虚构训练或反复扩大实验。
