# Experimental Dead Ends / Closed Threads


> **2026-10-18 补充：** “closed/dead end”在此只否定**特定过强命题或当前可用方法候选**，不自动否定整个MDMT采样、共享、梯度与优化研究方向。D001/D012 是历史待检验解释，**不代表用户曾认定“小域=困难域”或“小域天然低SNR”**。保留 D001–D012 的原历史证据，续增D013–D018；参见[多分支终局](Experimental_Branch_Synthesis_20261018.md)。


“Dead end”指按原主张继续推进缺少当前证据基础；不否定历史结果。若更窄的设置仍有证据或未解决问题，会在末列明确保留。

## D001 — 以“小域低 SNR”为依据发展 per-domain momentum

> **来源澄清（2026-10-18）：** 该条是历史对竞争解释的否定，并非用户最初认同过“小域=困难域”或低SNR等式；不要因此设计重复证伪的实验。

- **Why Initially Considered:** 早期假设小域梯度噪声更高，per-domain EMA 可补偿。
- **Evidence Against:** E001 显示 SNR 排序随 batch size 消失/反转；E008 未显示 comp1 相对 uniform EMA 的独特优势。
- **Current Status:** CLOSED / FALSIFIED in measured settings。
- **Narrower Version Remaining Open:** 其他数据或统计口径可能有域间噪声差异，但不能用本项目已测数据支持该前提。

## D002 — M3oE-specific largest-domain sacrifice

- **Why Initially Considered:** 早期 M3oE MovieLens 结果显示 L-role negative transfer。
- **Evidence Against:** E003 的 P0-3 中 MMoE 同向；E004 后显示 objective dependence。
- **Current Status:** SUPERSEDED。
- **Narrower Version Remaining Open:** 某些 Equal/旧配置下的 L-role loss 仍成立；不再称 M3oE 特有。

## D003 — comp1 是解决 objective mismatch 的必要方法，且增益随 ratio 单调增长

- **Why Initially Considered:** P5/P6 中 comp1 高 ratio AUC 增益显著。
- **Evidence Against:** E002 的 ratio response 非单调且高 ratio 与 N 混杂；E004 显示更改 objective 会减轻 Large-cell loss；E008 显示 comp1 未胜过 uniform EMA。
- **Current Status:** CLOSED as a general method premise。
- **Narrower Version Remaining Open:** 旧协议中 comp1 的历史 performance difference 可作为限定观察保留；其独特机制优势未建立。

## D004 — Sample objective 下存在普遍、稳定、实用的 negative transfer，因此需 Safe-MTL 主线

- **Why Initially Considered:** 个别单元曾出现负差。
- **Evidence Against:** E009 的 S1 0/72 practical NT；S2 Sample 0/24；S6 Phase A 两个已测 target gate STOP（E018）。
- **Current Status:** CLOSED within tested Sample settings。
- **Narrower Version Remaining Open:** OQ001 的更极端 ratio / 未测 target 仍没有结论；当前没有已知 selected performance gap。

## D005 — Curriculum 或已测时间 schedule 稳定胜过 static Sample

- **Why Initially Considered:** P6 与 E-series 希望利用域难度顺序或时间变化。
- **Evidence Against:** E006；P1-2 curriculum 量级低于门槛，E5 未通过 GO，E7 没有运行。
- **Current Status:** CLOSED for the tested implementations and gates。
- **Narrower Version Remaining Open:** 仅未测试的不同配置仍未被排除；现有证据不授权把旧方向重新当成有效前提。

## D006 — E6 adaptive calibration/EMA 是已验证的主方法

- **Why Initially Considered:** adaptive coefficient calibration 可减 mismatch，EMA 可能平滑估计。
- **Evidence Against:** E006 中未超过 Sample 的预注册 performance/safety gate；E7 0 runs。
- **Current Status:** CLOSED / STOP。
- **Narrower Version Remaining Open:** 无已冻结 adaptive method；secondary metric 不改变裁决。

## D007 — 仅匹配 gradient norm 或 applied-step norm 就能复制 target rescue

- **Why Initially Considered:** S3/S4 的 T rescue 可能只是幅度缩放。
- **Evidence Against:** E013 的 N/U 对照均未复制 T 的 d1 效应。
- **Current Status:** FALSIFIED as a sufficient explanation。
- **Narrower Version Remaining Open:** 方向、history 和轨迹因素仍有条件性问题；幅度本身仍是测量量，但不能作为充分机制。

## D008 — Equal 下的 HT/DH 结果可直接当作 Sample 下的方法依据

- **Why Initially Considered:** S5/S5-ML 在 Equal 下发现 target rescue 与 history/current structure。
- **Evidence Against:** E018/E019 selected Sample target contrasts below practical threshold; E020 shows a MovieLens late-stage residual only; E021 confirms objective/stage dependence。
- **Current Status:** CLOSED as an objective-independent inference。
- **Narrower Version Remaining Open:** Equal 下 S5 的 history-dominant KuaiRand d1 fact remains valid within its scope；Sample 下状态机制不可写成 exact zero。

## D009 — Target-only/HT 是无 non-target cost 的安全部署方法

- **Why Initially Considered:** S3 的 Equal target rescue 看起来显著。
- **Evidence Against:** E014/E016/S5ML show non-target losses and checkpoint/dataset trade-offs；E019 no Sample practical rescue；这些 arms 是诊断干预。
- **Current Status:** CLOSED as a validated safe method claim。
- **Narrower Version Remaining Open:** 某些 target cells 可通过 intervention 有所改善；是否存在满足指定多指标目标的设置尚未建立。

## D010 — Universal gradient-conflict correction / U_k correction 研究线

- **Why Initially Considered:** S7-A 在 MovieLens late overfit checkpoints 观察到 harmful gradient structure 与 AUC decline 同步。
- **Evidence Against:** E024 的 KuaiRand negative-U counterexample；E025 same-θ utility change 未改善 validation；E026/E027 full-state S8 无 selected/best gain。
- **Current Status:** CLOSED as a method-development premise under the tested evidence。
- **Narrower Version Remaining Open:** S7 的 Domain×Task gradient structures 作为 MovieLens late-stage observation 保留；跨配置普遍性在 OQ006，而性能根因/方法价值未获支持。

## D011 — 从 MovieLens late DH checkpoint gain 推断通用 selected-performance gain

- **Why Initially Considered:** S6/S6 trajectory reanalysis 中 MovieLens late common checkpoints 的 macro/non-target gains。
- **Evidence Against:** E026/E027 的 S8 MovieLens primary best endpoint；selected continuation null，且无 continuation best 超过 original F best。S6-R2 的 E028–E030 显示结果依赖 dataset 和 metric layer：KuaiRand 有 test-side selected non-target gain，validation 不占优，MovieLens selected contrast 仍近零。
- **Current Status:** CLOSED as a general inference from MovieLens late checkpoints to selected performance; not a universal null claim。
- **Narrower Version Remaining Open:** MovieLens already-overfit late anchor 有 partial rescue/less deterioration；KuaiRand S6-R2 有 test-side selected gain，但其 validation/test reversal 与原因仍未解决（OQ009）。

## D012 — 把“域大小”当作当前解释或实验结论

> **来源澄清（2026-10-18）：** 该条是历史对竞争解释的否定，并非用户最初认同过“小域=困难域”或低SNR等式；不要因此设计重复证伪的实验。

- **Why Initially Considered:** P0 role findings与旧的 size-based 叙述。
- **Evidence Against:** E003/E004/E010/E014/E020；域身份、objective、样本分布、target 与 checkpoint 同时变化；S6 full analysis 纠正了 MovieLens d0/d1 大小标签。
- **Current Status:** CLOSED as a stand-alone causal conclusion。
- **Narrower Version Remaining Open:** Domain size 是否是独立调节因素仍是 OQ002；没有预设小域更困难或大域更易受损。


---

## 10-15—10-18 新增关闭的过强命题 / 当前方法化路线

### D013 — “保持 Sample 期望目标，直接Balanced + p/q就能稳定提高宏AUC并无害”

- **Evidence Against:** Stage2D 16run validation SB−SN宏−0.010406、4/4负，收益/非劣门失败；v3多LR/阶段与N17八seed不同主结果亦未出现通过完整七项保护的候选（E035、E037、E039–E043）。
- **Current Status:** **CLOSED AS A READY-TO-DEPLOY METHOD UNDER TESTED SETTINGS**。
- **Narrower Version Remaining Open:** 特定阶段下 q×LR 或共享局部反事实能否在**新的**独立场景、冻结指标上满足绝对收益与全cell保护；不能仅靠差距gap改善代替健康提升。

### D014 — “所有同目标采样效应都必须经 Adam 历史状态产生”

- **Evidence Against:** fixed fusion 下无动量SGD负向实例；固定θ预期场与局部协方差变化并不自动需要Adam；v2 SGD 4/4负（E034、E036–E037）。
- **Current Status:** **FALSIFIED FOR UNIVERSAL NECESSITY**。
- **Narrower Version Remaining Open:** 特定 Equal/S5历史carrier效应、v3特定q×LR interaction、同θm/v/ratio及长期必要性均尚可研究，不能据此写“Adam状态普遍无效”。

### D015 — “所有差异都只由 validation 选择或晚期过拟合造成”

- **Evidence Against:** B1某seed同epoch仍val/test反号；v3和N17早期固定时钟轨迹已有差，N17 KR Adam AULC1..20−0.007480、8/8负（E033、E037、E040）。
- **Current Status:** **FALSIFIED AS AN EXCLUSIVE EXPLANATION**。
- **Narrower Version Remaining Open:** 选择、有限正例、LR/速度及普通过拟合仍是有作用的**修饰项**，ML selected/E100反号未独立分解。

### D016 — “q间固定点方差差异主要由随机域计数波动造成”

- **Evidence Against:** N17 quota固定域计数后约2倍trace仍在；count项只占0.001297%–0.105936%（E041）。
- **Current Status:** **CLOSED AS DOMINANT EXPLANATION IN THOSE 16 ANCHORS ONLY**。
- **Narrower Version Remaining Open:** 域内真实梯度分布、重复/coverage/时序依赖及优化器非线性相互作用；未否定所有其他θ、batch模型中的count作用。

### D017 — “共享 C 已被局部AUC屏幕证实为长期必要机制，立即方法化”

- **Evidence Against:** v3局部C AUC有筛查阳性但loss/J门失败；N17独立局部仅约+3e−5且预设实际/方向门失败，N4未触发（E038、E042）。
- **Current Status:** **CLOSED AS A CLAIM OF ALREADY-DEMONSTRATED LONG-TERM NECESSITY OR READINESS**。
- **Narrower Version Remaining Open:** 局部C有效路径和容量匹配的长期必要性仍未识别；N4没运行不意味着C没有作用。

### D018 — “在历史S6 init/optimizer谱系缺失时，重建B1即可验证或推翻原S6-R2”

- **Evidence Against:** 新B1与旧S6-R2的init/lineage不同，数值虽有显著结论差却不属于同run精确复现；新R1/R4亦不可据test事后替换R0（E028–E030、E033）。
- **Current Status:** **CLOSED AS AN INFERENCE METHOD**。
- **Narrower Version Remaining Open:** 原selected validation/test反转的具体机制、可用历史状态恢复及独立协议下的多seed适用范围（OQ009/OQ010）仍保留。

**不能新增虚假死路：** “小域=困难域”不是起始结论，不需要按轮次反复验证“推翻”；“momentum/课程/gradient天然无价值”并未得到普遍证明；旧 S5 Equal history 不是被新 SGD 反例整体推翻。所有**未运行、未触发、检验功效不足、缺模型状态或暂时NO_GO**必须归入其对应的开放/条件状态，而非偷换成科学失败。
