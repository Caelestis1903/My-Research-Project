# Experimental Stable Findings

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
