# Experimental Stable Findings


> **2026-11-02 最新结论：** 历史条目保留原日期和当时证据强度；新结论见下方追加条目及 [Global Summary](Experimental_Global_Summary.md)。不把旧日期的“最新决定”覆盖新的正式实验，也不重复登记相同seed或未触发阶段。

> **2026-10-26增量：** 原SF001–SF020保留在各自历史估计对象和数据人口内。新增SF021–SF028，区分事实、局部反例和方法独有性；[跨阶段总报告](Experimental_Branch_Synthesis_20261026.md)为当前统一结论。此处“稳定”表示可在限定协议内复述，不暗示新用户/时间窗口的普适性。



> **2026-10-18 补充：** SF001–SF012 是 10-09 历史协议内的稳定事实，后续不同 lineage 的阴性不能直接将其全局抹除。新证据与适用边界见 SF013–SF020；当前总体解释以[多分支综合报告](Experimental_Branch_Synthesis_20261018.md)为准。


以下是当前可安全复述的保守结论。每条都限定数据集、objective、模型、训练阶段或 checkpoint；已知例外列在同一条中。强度按阶段报告证据与后续反证综合判断。

## SF001 — 已测 Sample + SD reference 下未发现普遍 practical residual NT

- **Finding:** S1 的 Sample joint training 在所测 MovieLens/KuaiRand 与 M3OE/MMoE/PLE 条件下整体优于 single-domain reference；72 个汇总单元中 0 个达到 stable practical NT 门槛。
- **Evidence IDs:** E009、E018。
- **Scope:** Sample objective、已测自然数据比例、四 seeds、S1 reference 与 checkpoint 规则。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 单元级仍有小负差；不覆盖 Equal、极端未测 ratio 或所有 target 域。

## SF002 — Equal 与 Sample 的 selected 域级表现不同

- **Finding:** Equal 在 S2 的特定 Large-role / target cells 上出现可重复 selected loss；Sample 在同一报告预设门槛下未出现 practical selected loss。
- **Evidence IDs:** E004、E010。
- **Scope:** S2 的 KuaiRand natural/MovieLens d2_large、M3OE、SD reference 和 selected six-cell macro 规则。
- **Evidence Strength:** STRONG。
- **Known exceptions:** S2 MovieLens late/common views可能减小或反转差异；这不证明 Sample 在任何 objective 上普遍更好。

## SF003 — KuaiRand Equal/tab1 的 shared-access 干预有条件性 target 效果

- **Finding:** 在 S3 KuaiRand natural/M3OE/Equal/tab1 selected 对照中，TARGET_ONLY_SHARED 相对 FULL 有约 +0.039 AUC target effect，且四 seeds 同向。
- **Evidence IDs:** E011。
- **Scope:** 指定模型、dataset、objective、target 和 checkpoint。
- **Evidence Strength:** STRONG。
- **Known exceptions:** Sample 同一干预近零；MovieLens 与 later training 有不同结构；该因果效果不证明 pure gradient-conflict geometry。

## SF004 — S4 两类 magnitude controls 都不能解释该 target rescue

- **Finding:** raw-gradient-norm match 与 applied-step-norm match 均未复制 T−F rescue。
- **Evidence IDs:** E013。
- **Scope:** KuaiRand natural/M3OE/Equal、S4 A2 的 N/U/F/T arms。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 这排除两种充分解释，不单独确认其他机制。

## SF005 — Equal KuaiRand d1 rescue 在 S5 跟随完整 persistent history carrier

- **Finding:** S5 中 HT 保留了约 96–99% 的 T−F target rescue，而 DH 对 F 的 selected target 差约 0.001；这是完整 persistent Adam history carrier 的配置内事实。
- **Evidence IDs:** E015。
- **Scope:** KuaiRand-Pure/natural/M3OE/Equal/tab1、4 seeds。
- **Evidence Strength:** STRONG。
- **Known exceptions:** moment gate 为 15/16；不能拆成 m、v 或 bias-correction 单独机制，也不能外推到 Sample。

## SF006 — Target rescue/history structure 随 objective、dataset 与 checkpoint 改变

- **Finding:** Equal 下 selected rescue 在 S5/S5-ML 可观察，但 Sample 下 S6 Phase A/B selected target contrasts 均未达到 +0.003 practical gate；Sample MovieLens 仍有 late DH trajectory effect。
- **Evidence IDs:** E017–E021。
- **Scope:** 两数据集、M3OE、Equal/Sample 阶段对照；Sample Phase B KuaiRand 保留 M1 限定。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** “selected effect below threshold”不等于严格零；MovieLens late fixed checkpoints 结果不能代替 selected gate。

## SF007 — S8 未证明正常 selected model 的 practical optimizer-policy gain

- **Finding:** Selected-anchor continuation 的 T/DH 对 F best-AUC 差距接近零且低于 practical threshold；所有 MovieLens continuation best 仍低于该 seed 的冻结 F original best。
- **Evidence IDs:** E026、E027。
- **Scope:** S8 的完整 state、四 seeds、两数据集、selected/middle/late anchors 与 h=2-epoch primary horizon。
- **Evidence Strength:** STRONG。
  - **Known exceptions:** MovieLens late-anchor macro final-step 可有 positive rescue，但 seed-skewed、target gain小、未形成更高 best。S6-R2 另见 KuaiRand R0-selected test-side gain（E028）；其 validation 方向相反，且不改变 S8 的 MovieLens endpoint 结论。

## SF008 — MovieLens late gradient utility 与 overfit decline 同步，但不是充分性能机制

  - **Finding:** S7-A MovieLens late U_k 与 target validation deterioration 同步；KuaiRand 负 U_k snapshot 没有相同性能后果。S7-B 同 θ 改变 late utility 后，短程 validation 未形成稳定改善；anchor39 step5 target AUC 均值为 −0.001226，但仅 3/4 seeds 为负，LogLoss 均值为 +0.011130、3/4 变差。结论应限于“未见稳定收益”，不能称为逐 seed 一致恶化。
- **Evidence IDs:** E023–E025。
- **Scope:** Sample/M3OE、KuaiRand 与 MovieLens 的所测 anchors/checkpoints。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** S7-B 使用短窗口与 fresh-Adam private-state 近似；S8 full-state continuation仍不支持 selected-best gain。

## SF009 — 早期实验未单独识别 domain size 的独立因果效应

- **Finding:** 早期large-role关联存在，但后续对照未单独操纵domain size；目标权重、域身份、样本多样性、曝光与checkpoint等因素共同变化。
- **Evidence IDs:** E002–E004、E010、E014、E020。
- **Scope:** 跨P0–S6的历史实验记录；仅描述旧实验的可识别边界。
- **Evidence Strength:** STRONG（关于旧设计未单独识别size-only效应）；不对域难度作推断。
- **Research boundary:** 小域不是困难域。本条不用于证明或否定该固定判断，也不构成继续研究域大小与困难度关系的开放问题。

## SF010 — 当前阶段必须同时报告 target/non-target、AUC/LogLoss 与 checkpoint

- **Finding:** S4–S8 中多个 aggregate gain 与 subgroup/target metric、或 selected 与 final/common checkpoint 出现不同方向；任何单一 overall metric 都可能掩盖 trade-off。
- **Evidence IDs:** E014、E016–E020、E022、E027–E032。
- **Scope:** 已报告的 AUC/LogLoss、macro/worst、per-cell、validation/test layer、parameter-swap 和 training-stage 视图。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 这是报告口径结论，不等于所有指标都同等重要或存在一个统一部署目标。

## SF011 — KuaiRand S6-R2 的 DH test selected non-target gain 不出现在 validation

- **Finding:** Sample/M3OE 下，canonical R0-selected KuaiRand checkpoints 的 DH−F test ΔN 为 +0.0068（4/4 正，3/4 ≥ +0.005），test ΔM 为 +0.0044（4/4 正），而 test ΔT 均值为 −0.0004。相同 checkpoint 对的 validation ΔN 为 −0.0039（仅 1/4 正）。
- **Evidence IDs:** E028–E029。
- **Scope:** KuaiRand-Pure natural；F vs DH；四 seeds；R0 validation-macro selection。R1/R3/R4 的 test robustness 仅见于 A-grid sensitivity。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** seed2023 target test delta 为 −0.0047；seed2025 的 test non-target delta 近零；alternate-rule exact test values 不完整；此 finding 不识别 test-side gain 的机制。

## SF012 — S6-R2 的 KuaiRand cross-partition swaps 未保留完整 DH 解的表现

- **Finding:** 四 seeds 中，C/D hybrids 的 test macro 均显著低于完整 F/DH parent solutions；E_hybrid 近似 C_hybrid，F_hybrid 近似 F_full。此为所测参数交换下的非组合结果，不是对 DH 增益来源的定位。
- **Evidence IDs:** E032。
- **Scope:** KuaiRand-Pure natural；Sample/M3OE；四 seeds；parameter_manifest 的 shared/architecture/private 分区。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** seed2025 的 DH−F non-target denominator 近零，retention ratio 不可解释；结果不能推广到其他 partition definition 或模型。


---

## 10-15 至 10-18 跨分支追加稳定事实（编号续接，不重写历史）

### SF013 — B1 的“未复现”与旧 S6-R2 阳性必须同时保留

- **Finding:** KR Sample F/DH 新B1完整8run、四seed R0 test non-target AUC +0.000579（2/4正）未复现旧 S6-R2 +0.006757（4/4正），但原初始化和状态谱系不相同；R1/R4仍有test条件阳性和target LL代价。
- **Evidence IDs:** E028–E030、E033。
- **Scope:** 两个独立谱系与validation-only选点，不能当旧实验的同身份反证。
- **Evidence Strength:** STRONG（数值及身份），MEDIUM（新B1主选点无稳定实用收益）。

### SF014 — 同经验目标的自然/平衡域采样会产生协议条件性的总体训练差异

- **Finding:** 同α/不同q的固定θ均值梯度可相同但covariance不同；Stage2D KR Sample/Balanced验证选点宏ΔAUC −0.010406，四seed负。该q整体干预同时改变抽样高阶矩、重复曝光与轨迹。
- **Evidence IDs:** E034–E035。
- **Scope:** Stage2D KR/M3OE、learned fusion/dual Adam、特定50epoch；并非所有fixed fusion/optimizer LR/数据集。
- **Evidence Strength:** STRONG（所测条件），WEAK（归因单一中介）。

### SF015 — Adam 不是同目标采样效应的通用必要条件

- **Finding:** v2 的无动量SGD在fixed fusion协议下产生 SB−SN selected负效应（−0.002104，4/4负）；v3也有固定SGD的早期负向读数。由此足以反驳“只要改q造成差异，就必须有Adam历史”的**通用量词命题**。
- **Evidence IDs:** E036–E037。
- **Scope:** 否定通用必要性，不否定 Equal/S5 history bundle的历史局部结果，不宣称SGD与Adam机制等价。
- **Evidence Strength:** STRONG（存在反例）；长期 m/v 单独必要性仍未识别。

### SF016 — 最新N17 KR选点宏结果统计主门未通过，不应按种子多数宣布普适损伤

- **Finding:** 8seed配对下Adam SB−SN −0.005203、t95[−0.012006,+0.001600]、Holm p=.171875；SGD +0.000837、t95[−0.004309,+0.005984]、Holm p=.6953125。早期 KR Adam AULC1..20 −0.007480 是单独次级阶段现象。
- **Evidence IDs:** E039–E040。
- **Scope:** N17固定模型/冻结选点test；test有历史使用，不是pristine holdout。
- **Evidence Strength:** STRONG（数值与检验状态），WEAK/INCONCLUSIVE（普遍selected效应）。

### SF017 — 训练预算与选轮规则可以出现同组不同方向，不能以单一端点替代其他端点

- **Finding:** ML固定subset独立八seed own-LR q-gap selected −0.010875，而E100 +0.021145；两臂LR不同。v3固定/校准 optimizer 与early/selected 也有反号。
- **Evidence IDs:** E037、E040。
- **Scope:** 不同估计对象，E100正向并不是同LR固定q策略的已证明可部署收益。
- **Evidence Strength:** STRONG（数值方向），MEDIUM（训练阶段/预算影响），WEAK（独立归因）。

### SF018 — 评价加权方式可以对同一模型同一预测给出相反AUC差值

- **Finding:** 历史calAdam E50 macro Δ−0.009074、train-p-cell +0.001016、test-p-cell +0.003108；ML own在新轮 macro −0.010875、train-p-cell +0.000744、test-p-cell −0.013586。AUC cell加权不等于pooled AUC。
- **Evidence IDs:** E040。
- **Scope:** 同预测不同评价权重的精确算术；不识别域内分布变化的因果原因。
- **Evidence Strength:** STRONG。

### SF019 — 观察到的固定点协方差差异不能主要归于域计数波动

- **Finding:** N17 N2 16锚的q trace比在IID及quota约2倍以上，计数波动仅占模型预测trace的0.001297%–0.105936%；已排除**这组固定点**“随机域count主导”的解释。
- **Evidence IDs:** E041。
- **Scope:** 给定锚/数据/估计器；quota还改变批内依赖，未识别长期trace、m/v、domain size独立因果。
- **Evidence Strength:** STRONG（局部反证）。

### SF020 — 共享C仅有小规模条件局部作用，长期唯一必要性没有完成检验

- **Finding:** v3 C局部AUC筛查曾阳性但loss/J联合门失败；N17 embedding/E10独立确认两个路径均4/4方向为正、绝对约3e−5，实用/方向门未过，N4长期拓扑按预设不触发。参数支持已核对C/E共享、R私有、P混合。
- **Evidence IDs:** E038、E042。
- **Scope:** 同θ/局部干预；既不是“C全无效”，也不是“共享C必要中介已证实”。
- **Evidence Strength:** MEDIUM（局部及门槛），WEAK（唯一长期中介）。

**截至2026-10-18的历史总体阶段决定：** 条件效应存在，部分历史统一解释被反例削弱；当前没有已测候选同时达到预定收益与非劣保护，方法决定为**本轮 NO_GO**。这不是未来所有MDMT方法的否定，尤其不能把“小域=困难域”当作曾经成立的前提去反复证伪（E043）。 


## SF021–SF028：2026-10-26更新后的限定稳定事实

### SF021 — 人工与自然MovieLens是两种不同训练人口
- **Finding:** 多项历史ML使用人工187500行/域34500:3000:150000；自然基础训练800167行/155949:318599:325619。比较还改变N、支持和训练时钟；自然补充整体尚处IN_PROGRESS。
- **Evidence IDs:** E048–E049；[自然人口审计](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)。
- **Strength:** STRONG人口事实，WEAK比例单因果。原人工实验保留条件效度。

### SF022 — N18/N19实质完成，方法增量仍需强对照
- **Finding:** N18完成六方向有界诊断/冻结头；N19完成328正式网络训练与冻结测试，未建立稳定足够强的独立新方法贡献。DATA-BLOCKED/NO_GO仅限各自候选。
- **Evidence IDs:** E044–E045；[N18](../history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)、[N19](../history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)。
- **Strength:** STRONG执行，MEDIUM条件反例，WEAK通用方法化。

### SF023 — 共享乘性路径数值中性不等于所有模型共享无效
- **Finding:** CM19 KR fixed M3oE某输入乘性路径selected 24/24 cell中和预测不变；双factor正则/局部AdamW干预恢复响应，STAR等独立架构仍活跃，但selected整体泛化收益不稳定。
- **Evidence IDs:** E046–E047；[CM19](../history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)。
- **Strength:** STRONG特定路径数值事实，WEAK普遍性能解释。

### SF024 — D套件有效但动态样本区分的独立增量未证明
- **Finding:** 10-21 ML D−B test宏AUC +0.006303/LL −0.015962、Dconst01−B AUC +0.005676，均4/4有利。D权重趋向0.1下界；Dconst01对D没有通过正式非劣判定，仍保留复制/噪声与总尺度。
- **Evidence IDs:** E050–E051；[D/G报告](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。
- **Strength:** STRONG ML套件效果，WEAK动态权重单独因果；legacy_group_sum不可混作Sample/Equal。

### SF025 — 当前匹配梯度计算路径的τ调幅没有终点宏AUC优势
- **Finding:** G−Gtau0的ML/KR test宏AUC分别−0.003466/−0.006484，均4/4负，但KR置信区间跨0；KR早期validation更快、部分域或pooled方向可不同。
- **Evidence IDs:** E052；[梯度对照](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。
- **Strength:** STRONG ML匹配终点，MEDIUM KR方向，WEAK跨模型推论；G−D非纯τ对照。

### SF026 — 历史收益控制的末轮减损不同于最佳selected改善
- **Finding:** 10-25 v3 KR RHD HB−BASE正式test LL在selected +0.00074075而e50 −0.00875428；各方法自身e50仍差于合法selected。10-24 I1和10-25 v2没有通过各自selected主优势/晋级门。
- **Evidence IDs:** E053–E055；[v3](../history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)。
- **Strength:** MEDIUM阶段性实证，不是可部署最佳性能证明。

### SF027 — 来源可读、需求可预测与特定来源函数值得形成是不同命题
- **Finding:** v3 LPS有部分foreign读益但L4形成未稳定超过简单对照；DFD ML P1需求选择优于常量，KR P1失败，ML完整方法弱于H_DIRECT/ETA_DIRECT。
- **Evidence IDs:** E056–E057；[v3](../history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)、[v4](../history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md)。
- **Strength:** MEDIUM限定观察，WEAK独有形成机制。非零责任不足以证明原型分工。

### SF028 — 版本性停止、漏导出与尚未训练不能被误记为阴性实验
- **Finding:** NEXT-v2四seed正式50epoch NO_START；DFD P3/P4 NO_START、184预测/12计时完成但外层refitη/μ分组MISSING/MAJOR；最新RCF-v1仅为研究设计。
- **Evidence IDs:** E054、E057；[DFD-v4](../history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md)、[RCF方案](../research_plans/2026-10-26-v2/00_README.md)。
- **Strength:** STRONG执行/资料状态；不能据此否定全部局部/多原型共享思路。

**最新总体判断：** 优化响应、样本重加权和来源内容的条件收益值得研究，已测动态方案暂未建立普适强增量。研究者从未将“小域=困难域”作为成立先验；后期研究应先区分具体机制和强简单基线，而非反复防御性否定同一命题。

## 2026-11-02 新增有边界的稳定事实 SF029–SF037

### SF029 — 所测来源关系与BASP-A原型政策尚未胜强目标直接适配
- **Finding:** 10-27 Relation-Basis v2低秩/全关系在两自然数据四seed的冻结Q决策下没有胜TARGET_ONLY，低秩在历史E宏AUC各4/4负；10-28 BASP-A的CG相对TARGET_MATCH不稳定，留出收益μ与NULL读取记录限制当前政策解释。
- **Evidence IDs:** E058–E059；**Strength:** MEDIUM具体版本停线，STRONG所测机制状态；**Boundary:** 保留历史来源读取与局部群体收益问题，不否定整个共享方向。

### SF030 — HyperGate与AutoIFS已在当前仓库正式五seed比较
- **Finding:** 10-29两自然数据的HyperGate/AutoIFS相对M3OE_FIXED1宏AUC四组配对均5/5改善；KR AutoIFS macro LL改善而natural LL五seed变差。原论文参数完整核实及同训练预算的架构纯比较并未完成。
- **Evidence IDs:** E060；**Strength:** STRONG本协议实测，不能宣称严格论文复现。

### SF031 — 统一损失缩放在ML可提升selected AUC，简单L2解释必须优先考虑
- **Finding:** GA31无增强ML c=.1相对c=1 selected AUC+0.003473（4/5）、KR相同缩放方向负；ML普通L2对缩放基线的selected验证差较小。耦合Adam同时改变相对wd和eps，早/末训练时钟与selected结果不等价。
- **Evidence IDs:** E061–E062；**Strength:** STRONG ML条件性效应，MEDIUM控制比较；**Boundary:** 不当成旧D动态权重的独立收益。

### SF032 — 固定参数EMA在当前协议具有可重复排序收益
- **Finding:** ML两组五seed的EMA−online selected AUC分别+0.004399和+0.003849，均5/5正；KR+0.004616、4/5正。ML LogLoss在两组协议中多数seed恶化。EMA是非反馈shadow，不是对在线Adam动量递推的修改。
- **Evidence IDs:** E064；**Strength:** STRONG ML AUC，MEDIUM KR AUC；**Boundary:** 不是LL一致收益、不是独立新时间窗口。

### SF033 — 当前增强的局部收益不能代替总体/流量与LL保护
- **Finding:** NR-Gauss在GA31未过主门；KR ANTI2对普通基线有条件test增量但不胜同K IID2；KR Gaussian+EMA的LL五seed改善伴随d1两任务AUC多数seed下降、pooled/traffic整体下降。必须保留六格及主指标口径。
- **Evidence IDs:** E062–E063、E066；**Strength:** STRONG某些域损害，MEDIUM独有性停线。

### SF034 — 非零差分梯度/动态门不自动增加实用排序
- **Finding:** SAD βδ=.99实际修改Adam更新但两集selected宏AUC未胜普通增强A1；βδ=.9与普通Adam实数代数等价仍可能出现FP32长期轨迹差。AIG-M动态注入不为0而只停于开发门。
- **Evidence IDs:** E065–E066；**Strength:** STRONG实现与代数身份，MEDIUM已测性能停线。

### SF035 — GATS任务投影B2开发收益真实存在，却未形成稳定强基线增量
- **Finding:** KR B2开发两seed在验证对A1/R2正向，保留seed确认均值大幅缩小；统一test五seed对A1宏AUC+0.000005156（3/5），对A0固定EMA只有2/5正。已完成当前阶段且明确不晋级。
- **Evidence IDs:** E067；**Strength:** MEDIUM当前版本NO_PROMOTION；**Boundary:** 保留任务级/验证期条件性观察。

### SF036 — GATS难度覆盖、延迟增强与CV自适应EMA当前版本未晋级
- **Finding:** B1/B3/C2各在匹配控制或预定门下未建立独有macro增量，确认门未触发的臂不能声称正式阴性。EMA早期滞后/后期提升的阶段性事实并不证明CV→rho的新机制。
- **Evidence IDs:** E068；**Strength:** MEDIUM当前规则停止，WEAK普遍外推。

### SF037 — EMA02V2任务梯度视图确实产生不同方向，但不是有益融合证据
- **Finding:** 两自然数据各2开发seed、36训练全部完成；ML F1 test对A0+EMA +0.000359（2/2）但验证均值负、幅度不足晋级；KR TG/AG对A0+EMA test宏AUC约−0.0172/−0.0175并伴随LL恶化，两seed一致。梯度探针与实际扰动正常执行。
- **Evidence IDs:** E069；**Strength:** MEDIUM当前KR退化，WEAK ML F1微增量；**Boundary:** 不是新数据确认、也没有改在线动量公式。

## 2026-11-08 新增稳定事实（SF038–SF048）

### SF038 — 完整候选组合的总收益不等于新增组件的独立收益
- **Finding:** 11/03五seed补全中，C/H/P01+固定EMA相对普通M3oE online的Macro AUC增量：ML +0.003253/+0.003059/+0.003225，KR +0.003720/+0.005579/+0.008702，六格保护均满足。相对固定EMA的组件增量则不同：ML三候选均未超过EMA；KR P01虽有+0.004086宏增量，但d1t0/d1t1分别−0.001334/−0.001047，未满足逐格保护。
- **Source:** 11/04 Global Summary；[五seed补全报告](../research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)。
- **Strength:** STRONG（完整配置总收益）；MEDIUM（组件增量与逐格取舍）。

### SF039 — 统一AUC选点后的正式发表模型排序已明确
- **Finding:** 在统一自然人口、validation六格Macro AUC最早严格最大值选点的协议下，ML M3oE/HyperGate/AutoIFS Macro AUC为0.775190/0.771412/0.772425，M3oE最高；KR为0.671907/0.682113/0.678597，HyperGate最高。该排序替代旧V-LogLoss选点结果作为当前同协议参照；不同训练协议的绝对值不跨表排名。
- **Source:** [Overnight A+B报告](../research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)。
- **Strength:** STRONG（当前协议内排序）；适用范围限于已运行适配版本和本轮协议。

### SF040 — Gaussian覆盖的收益具有数据集与指标条件性
- **Finding:** ML Gaussian D0 Macro AUC为0.779357，相对M3oE +0.004167且六格保护通过，Macro LogLoss差+0.000931。KR G-D0/G-D2/G-D02 Macro AUC为0.679180/0.677999/0.674604，均未超过HyperGate 0.682113；KR不同覆盖的AUC与LL排序不一致。
- **Source:** [Gaussian覆盖报告](../research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)。
- **Strength:** STRONG（所测配置和协议）；不外推为普遍增强规律。

### SF041 — CP组合未形成相对P01的加性排序优势
- **Finding:** CP fixedEMA的ML/KR Macro AUC为0.777525/0.678168，相对M3oE总增量+0.002335/+0.006261；但相对单独P01分别−0.000890/−0.002441。KR对M3oE的总收益保留，但当前CP组合不替代P01排序方案。
- **Source:** [Overnight A+B报告](../research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)。
- **Strength:** MEDIUM（当前2×2组合和协议）。

### SF042 — 当前方向响应EMA没有超过固定EMA
- **Finding:** AUG_HISTORY中B方向响应确实执行，但D−F selected Test Macro AUC仅ML +0.000000104、KR −0.000004946；没有实用增量。
- **Source:** [AUG_HISTORY报告](../research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（当前方向定义、模型、候选网格和数据协议）。

### SF043 — 当前路径/分组时间尺度版本未获得selected收益
- **Finding:** AUG_HISTORY的GROUP/GLOBAL时间尺度相对F，ML Macro AUC分别−0.000750/−0.000728，KR分别−0.000175/−0.000213；部分早期验证优势没有转化为selected Test增益。
- **Source:** [AUG_HISTORY报告](../research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（当前实现和候选范围；不否定其他适应性信号）。

### SF044 — 当前梯度漂移协方差增强没有超过固定EMA
- **Finding:** MovieLens C_MERGED−F selected Test Macro AUC为−0.000667190、LogLoss差+0.000357811、AUC正向seed为0/5；C3_WEAK与旋转控制的均值也未超过F。当前数据不支持真实方向信息具有独立必要性；KuaiRand C为未运行，不记阴性。
- **Source:** [AUG_HISTORY报告](../research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（MovieLens当前候选）；KuaiRand C为NOT_RUN。

### SF045 — 固定EMA的selected排序收益超过年龄匹配online对照
- **Finding:** EMA_MECHANISM四组F−online selected Test Macro AUC为ML clean +0.003848845、ML Gaussian +0.003856802、KR clean +0.004616399、KR Gaussian +0.007766927；正向seed分别5/5、5/5、4/5、5/5。晚期Validation年龄匹配online对照仍有+0.002993/+0.003153/+0.004503/+0.003746，四项95%区间下界均大于0。早期F存在明显学习滞后，晚期排序变化率下降。
- **Source:** [EMA_MECHANISM最终报告](../research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** STRONG（四个已测条件的同轨迹配对和年龄匹配对照）。

### SF046 — 常驻时间尺度交叉收益目前只在KuaiRand Gaussian条件下成立
- **Finding:** 逐格常驻时间尺度交叉评价：ML clean −0.000047、ML Gaussian −0.000082、KR clean −0.000152、KR Gaussian +0.000375±0.000331（5/5）。这是条件性局部证据，不支持普遍的分组rho策略。
- **Source:** [EMA_MECHANISM最终报告](../research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（单条件局部交叉收益）。

### SF047 — 未来收益预测准确度与动作增益不是同一结论
- **Finding:** C_design在四组未来收益数值预测MAE均优于M2/M3，但整体单组动作收益较小。KuaiRand Gaussian d0t1共享组S的M3_direct局部净益+0.000346；M3相对C_simple的五seed增量+0.000048355、SD 0.000199264，复杂指标独立必要性证据弱。
- **Source:** [EMA_MECHANISM最终报告](../research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（C_design预测MAE）；WEAK（复杂指标独立动作价值）。

### SF048 — Gaussian×EMA交互随阶段和选点改变，不支持统一协同规则
- **Finding:** selected Test交互AUC为ML +0.000007957、KR +0.003150528，均仅3/5 seed正向；Validation selected和Test E50交互方向不同。ML E40分别校准的诊断将原F−online LL差+0.004593/+0.003986变为−0.000255/−0.000451，但这只解释E40锚点，不能据此消除selected Test LL代价。
- **Source:** [EMA_MECHANISM最终报告](../research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（当前四条件和阶段）；不外推为通用Gaussian×EMA协同。

**研究结论采用：** 固定EMA的排序收益作为已接受的强内部参照；当前B/A/C自适应候选不晋级。完整配置总收益与新增组件增量分开；未来自适应设计必须提供跨seed的未来收益预测证据，并实际改变参数时间递推。AUC与LogLoss分别报告，所有结论限定于对应数据、模型、seed和选点协议。


