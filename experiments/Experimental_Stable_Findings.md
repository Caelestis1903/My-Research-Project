# Experimental Stable Findings


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

## SF009 — 现有数据没有识别 domain size 的独立因果效应

- **Finding:** 早期 large-role 关联存在，但后续对照未单独操纵 domain size；目标权重、域身份、difficulty、样本多样性、曝光与 checkpoint 均可能共同变化。
- **Evidence IDs:** E002–E004、E010、E014、E020。
- **Scope:** 跨 P0–S6 的实验矩阵。
- **Evidence Strength:** STRONG（关于“尚未识别”的结论）；对 size 本身效应强弱不作判断。
- **Known exceptions:** 角色/样本规模与部分结果相关；相关不等于 size-only causality。

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

**总体阶段决定：** 条件效应存在，部分历史统一解释被反例削弱；当前没有已测候选同时达到预定收益与非劣保护，方法决定为**本轮 NO_GO**。这不是未来所有MDMT方法的否定，尤其不能把“小域=困难域”当作曾经成立的前提去反复证伪（E043）。 
