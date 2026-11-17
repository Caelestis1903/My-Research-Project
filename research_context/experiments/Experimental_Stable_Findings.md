# Experimental Stable Findings

## 2026-11-17 CWA-C04 选中模型可用性更新

11:21:33的40模型/1000ACK直接验收为有效历史证据。12:02:24独立审查发现owner2476891已退出，父级复查确认前台会话返回143；当前选中模型不可用，退出触发原因未确定。完整数值、源码和MIXED不变，不能以历史收据证明当前存活，不能据此续训或开放Test。后续C05使用新批准的源初始化。[可用性处置收据](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/SELECTED_RAM_AVAILABILITY_UPDATE.json)。

## 2026-11-17 / SF-CWA-C04 — 稳定条件现象及其边界

**SF-CWA-01 / MEDIUM：同BCE总质量下的task1损失收益。** W−S selected fixed EMA task1 LL在ML/KR分别−.002988932/−.005124581，五seed全部降低、描述性区间全负。KR Macro LL也五seed降低。只适用于本模型、自然切分、λ=.25、固定协议；class/cell/task一起变化，非class-only或平衡机制识别。W本身未获注册跨数据集AUC资格。

**SF-CWA-02 / MEDIUM：排序项作用随阶段/读出改变。** ML N−W online selected AUC五seed下降、CI全负；EMA selected AUC混合但LL五seed增加、CI全正。N−W E1 AUC五seed增加，E50 AUC/LL五seed均不利、CI不跨0。不能概括成排序总是有害，也不能把阶段反转直接当作curriculum或EMA中介已识别。

**SF-CWA-03 / MEDIUM：KR域间与指标间不一致是真实条件现象。** W−C最小d2 AUC五seed改善，另一个较小d0 LL五seed变差，最大d1 LL五seed改善。Macro AUC正均值与traffic-cell微负/pooled近零并存。规模主因和统一小域/大域交换仍WEAK；ML最小域基线最高且本轮三个域AUC均值微负，反对将规模当成困难度定义。

**未建立的命题：** W稳定提高selected matched-EMA AUC（ML均值负、KR区间跨0且seed偏斜）；移除R仍稳定保留N的ML五seed选中退化（W未满足该条件）；S仅等价于LR变化；局部gradient/update cosine或deltaEMA代数式已解释全轨迹中介比例。精确C/N复用和C3相同prefix不是新独立重复。

来源：[完整报告](../../research_reports/completion_weight_attribution_20261117/REPORT.md)、[全部五seed及区间](../../research_reports/completion_weight_attribution_20261117/ALL_PAIRED_SUMMARY.csv)、[数据可用性](../../research_reports/completion_weight_attribution_20261117/DATA_AVAILABILITY.md)。C4完整训练/状态已验收，证据审查与发布采用实际收据；之后仅一次完整C05，交付后停止。以下保留历史发现与修正。

## SF-NQR-C03 — ML排序成本未随后40epoch关闭而消失

EARLY10/LATE10完整20新增+20精确数值控制（CLEAN/constant-NQR各10），自然ML/KR、M3oE fixed1、seeds2022–2026、50epochs/40960/FP32，1000epoch/34500新更新；20物理attempt全完成、0失败。注册EARLY selected EMA−CLEAN Macro AUC：ML−.000340045（0/5正，CI[−.000459987,−.000220103]），KR−.000068613（3/5正，CI[−.008122287,+.007985061]）；NO_CONTINUATION_SIGNAL，无champion、Test未访问。ML/KR ΔLL均值−.000928551/−.000928905，但仅3/5和2/5降低且CI跨0；ML cell保护通过，KR最小cell均值−.006049390未通过。

ML EARLY−constant selected AUC+.000003472、CI跨0，LL−.003069812且5/5降低、CI全负（MEDIUM）：停止后40epoch干预改变损失，未恢复选中排序。LATE ML五seed选中状态/epoch等于其干预前CLEAN最优，属于已注册选择下界，不能作为时序有效证据。E50 EARLY−LATE AUC均值ML online/EMA−.000139590/−.000750982，KR+.001469902/−.000934445，四个CI全跨0（时序机制WEAK）；KR LATE−CLEAN选中EMA+.000738140，2正3零、CI跨0，为控制的WEAK条件信号，不升级候选。EARLY E10与C2常数轨迹完全相同，不算独立重复。

ML task0 AUC−.000408706、4/5负、CI略低于0，延续rank敏感性（MEDIUM），但LL均值已反向改善、CI跨0；不能声称重复了完整rank/loss联合成本。KR d0/d1/d2 ΔAUC−.003499683/−.001043873/+.004337717，全部CI跨0；两个较小域反向，无统一小域收益/大域下降，规模因果WEAK。KR pooled LL均值+.000520874与Macro LL负值相反，均不确定。先验/机会权重、ranking梯度、早期Adam/参数路径与EMA历史仍竞争。

[完整报告](../../research_reports/nqr_temporal_exposure_20261116/REPORT.md)、[所有seed/CI](../../research_reports/nqr_temporal_exposure_20261116/ALL_PAIRED_SUMMARY.csv)、[原数表/12分片](../../research_reports/nqr_temporal_exposure_20261116/DATA_AVAILABILITY.md)。单模型/两个自然数据集/固定协议/五seed验证；不外推为所有时序策略、EMA或M3oE都无效。

## SF-NQR-C02 — 当前排序设置的ML成本与KR大域损失响应

在自然ML/KR、固定M3oE/Adam/EMA、50epochs/40960/FP32五seed下，NQR selected EMA未形成CLEAN之外的主增量。ML Macro AUC与task0 AUC均5/5下降且CI低于0，task0 LL五seed上升且CI高于0；task0成本与AXIS-EVEN重复（MEDIUM，保留协议范围）。ML E10 EMA AUC却5/5提高，属于阶段反例。

KR最大d1域LL五seed降低且CI低于0（MEDIUM），d0/d2两个较小域AUC均值反向、最大域AUC近零；不支持统一小域收益/大域下降。KR主AUC与四个NQR机制AUC CI跨0（WEAK），Macro和Traffic/Pooled方向仍不同。注册非正主均值条件为直接数值事实STRONG，无champion/Test。证据：[E-NQR-C02](Experimental_Evidence_Ledger.md)与[完整报告](../../research_reports/natural_quadratic_ranking_20261116/REPORT.md)。


## SF-AXISEVEN-C01 — 当前even-view设置的局部退化与域间反向

完整五seed、M3oE fixed1、自然ML/KR、50epochs/40960/FP32中，AXIS-EVEN未形成selected EMA Macro AUC增量。ML task0及最小d0均5/5负且各自描述性CI不跨零（MEDIUM）；KR d0负/d2正的域均值与最大d1近零并存，d2域LL五seed均降低。该模式不支持统一的小域改善/大域下降。KR IID2均值正但仅2/5正、CI跨零（WEAK）。证据与边界见[E-AXISEVEN-C01](Experimental_Evidence_Ledger.md)及[完整报告](../../research_reports/task_axis_even_views_20261116/REPORT.md)。

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
