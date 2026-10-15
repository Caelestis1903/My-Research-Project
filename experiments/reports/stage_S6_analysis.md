# Stage S6 — 完整 F/T/HT/DH optimizer 2×2 分析

Main Agent，2026-10-08。范围：M3OE；Sample objective；KuaiRand/natural、MovieLens/d2_large；seeds 2022–2025；50 epochs。

**矩阵执行与数值分析已完成。两个数据集各 16 个 canonical run，共 32 个；本次新增 HT/DH 共 16 个。原 Phase A 的 selected T−F gate 仍为双 STOP。追加四臂诊断显示 MovieLens 的晚期 DH 分量效应仍存在，故不能继续概括为“Sample 消除了所有 optimizer-history/current 效应”。扩展诊断裁决为 B. CONDITIONAL RESIDUAL，限定为 dataset/stage/metric-dependent 的 optimizer-policy 效应；这不表示 Phase A GO，也不支持直接发展新方法。**

**有效性限制：执行/实现一致性 PASS；MovieLens M1 PASS；KuaiRand M1 FAIL（早期 m 4/8、v 8/8）。完整设计识别要求的无条件通过声明为 FAIL。** 本报告在独立 reviewer 允许的范围内分析正确执行的策略轨迹；不声称 KuaiRand 的跨臂实际 Adam history 已满足预注册 proximity 检查。

## 1. 数据、执行与阶段记录

- [执行报告](stage_S6_phaseB_run_report.md)：本次两数据集各四个 HT/DH seed wave；每批两个并发训练 Python 进程；两个数据集顺序执行；16/16 新 run 成功，无重试、无失败。
- [独立审计](stage_S6_phaseB_audit_review.md)：检查 32 个 run、44,800 条梯度记录及文件哈希、路由、公式、计数器、RNG、配对和配置；未发现实现错误。该审查是 post-run 审查，不能追溯补齐缺失的历史 S6 coder/pre-run reviewer 文档。
- [追加范围](stage_S6_phaseB_extension_scope.md)：用户明确要求在 Phase A 双 STOP 后补全矩阵，直接进入 runner，不再请求批准。Phase B 是知道 Phase A 结果后的诊断扩展，不能改写为原 gate 触发的确认性实验。
- [设计规格](../stage_S6_design.md)与原始 F/T 保持不变；Phase A 输出保存于 `stage_S6_outputs_pre_phaseB_20261007` 和 `stage_S6ML_outputs_pre_phaseB_20261007`。
- 原 2026-09-29 总分析保留为 [Phase A 历史分析](stage_S6_phaseA_final_analysis_20260929.md)。本文件取代其“Phase B 未运行”的当前状态描述。

每数据集的合并 raw 表为 672 行 = 16 runs × 7 checkpoint labels × 6 cells；paired 表 1008 行 = 6 原有 arm contrasts × 4 seeds × 7 labels × 6 cells。五个 common epochs 为 9/19/29/39/49，epoch 编号从 0 开始；final 与 common49 完全相同，只算一个观察点。

主代理从原始 AUC/LogLoss 独立重算全部 contrasts：四个阶段 S5/S5ML/S6/S6ML 的原 paired 表均在 1e-12 以内复现；合并前后 F/T 指标完全相等；同一 final/common49 指标完全相等。Equal/Sample 的 1,440 项同 seed/arm manifest 锁项检查全部相等，排除明确允许变化的 stage/objective 字段。它验证记录的受控条件可比，不表示两个阶段的 runner 文件字节完全相同。

## 2. 配置与域大小

| 项 | KuaiRand | MovieLens |
|---|---|---|
| target | d1/tab1 | d2/ages_35_45_50_56 |
| task0 / task1 | is_click / long_view | rating>3 / rating>4 |
| train counts d0/d1/d2 | 172,782 / 1,743,898 / 89,452 | 34,500 / 3,000 / 150,000 |
| train shares d0/d1/d2 | 8.6127% / 86.9284% / 4.4589% | 18.4% / 1.6% / 80.0% |
| size order | d1 最大、d0 中等、d2 最小 | d2 最大、d0 中等、d1 最小 |
| optimizer | Adam，lr=.01，betas=(.9,.999)，weight_decay=1e-5 | 同左 |
| architecture optimizer | Adam，lr=.001，每 epoch 一步 | 同左 |
| main / architecture steps | 2450 / 50 | 250 / 50 |
| batch size / epochs | 40960 / 50 | 同左 |
| initialization / split / exposure | 与对应 Equal anchor 锁定；同 seed 四臂配对 | 同左 |
| checkpoint selection | validation-only unweighted six-cell macro-AUC 最大，并列最早 | 同左 |

Sample 使用普通串联打乱的 DataLoader，w_d=1，effective coefficient a_d=p_d；没有额外域均衡。Equal anchor 使用 w_d=1/(3p_d)，a_d=1/3。两种 objective 对应不同的域聚合目标；若研究目标是域宏平均，不能仅因 Sample 在最大域较好就把 Equal 定义为普遍错误的目标。

**更正旧报告：MovieLens d0 不是最小域。** 旧 Phase A 文档把 T 对 d0 的损伤写成“最小域损伤”，该大小解释不成立；最小域 d1 在多项对比中反而受益。两个数据集的 target 都是最大域，这些实验没有独立操纵 domain size，也没有将 size 与 difficulty、identity 或数据结构分离。

## 3. 真正的 2×2 定义与分析口径

| Persistent Adam history / current gradient source | FULL current | TARGET current |
|---|---|---|
| FULL history | F | DH |
| TARGET history | HT | T |

History 包括指定分区的 persistent m/v/step，未做 m/v 拆解。Current 是计算当次反事实 Adam delta 的梯度来源，delta 仍依赖本臂 pre-step state；不是直接应用一个裸梯度。HT/DH 只覆写指定 shared/architecture 参数，private 参数始终用 FULL loss 更新。每臂的梯度在自己的参数轨迹上计算；HT 的 history 不是从独立 T run 移植的实际 tensor。

对每个 seed、cell、checkpoint、metric，计算：

| 数量 | 公式 |
|---|---|
| 原有 contrasts | T−F、HT−F、DH−F、HT−T、DH−T、HT−DH |
| history effect，FULL current | HT−F |
| history effect，TARGET current | T−DH |
| current effect，FULL history | DH−F |
| current effect，TARGET history | T−HT |
| 平均 history contrast H | (HT+T−F−DH)/2 |
| 平均 current contrast D | (DH+T−F−HT)/2 |
| interaction I | T−HT−DH+F |

I 是两种 current 条件下 history contrast 的差，也是两种 history 条件下 current contrast 的差；不等于“救回的比例”，其正负没有独立优劣含义。所有数量描述 optimizer source-policy 的完整轨迹干预，不能自动识别单独的 m、v 或纯状态中介效应。

报告使用 4-seed mean、样本 SD、逐 seed 符号及阈值计数，不用“未显著”等同于零。AUC 为正通常有利，LogLoss 为负通常有利。AUC practical threshold .003、strong magnitude .005、至少 3/4 同方向不变；这两个 magnitude 标签不等于证据强度等级。selected 使用各臂不同的 validation-selected epoch；common 对比固定训练阶段。

## 4. Selected：完整矩阵与目标域 contrasts

### 4.1 四臂绝对 target AUC（4-seed mean）

| Dataset/cell | F (FULL/FULL) | DH (FULL/TARGET) | HT (TARGET/FULL) | T (TARGET/TARGET) |
|---|---:|---:|---:|---:|
| KuaiRand d1t0 | .697599 | .697442 | .696716 | .696942 |
| KuaiRand d1t1 | .726063 | .725388 | .724594 | .725273 |
| MovieLens d2t0 | .721797 | .721454 | .719491 | .723696 |
| MovieLens d2t1 | .739511 | .739993 | .738860 | .742066 |

### 4.2 Selected ΔAUC（mean，括号为正向 seeds/4）

| Dataset/cell | T−F | HT−F | DH−F | HT−T | H | D | I |
|---|---:|---:|---:|---:|---:|---:|---:|
| KuaiRand d1t0 | −.000657 (1) | −.000883 (3) | −.000157 (2) | −.000226 (2) | −.000692 (2) | +.000034 (2) | +.000384 (3) |
| KuaiRand d1t1 | −.000790 (1) | −.001470 (3) | −.000675 (2) | −.000680 (2) | −.000792 (2) | +.000002 (2) | +.001354 (3) |
| MovieLens d2t0 | +.001899 (2) | −.002306 (1) | −.000343 (2) | −.004205 (0) | −.000032 (2) | +.001931 (4) | +.004548 (3) |
| MovieLens d2t1 | +.002555 (2) | −.000651 (1) | +.000482 (3) | −.003206 (1) | +.000711 (2) | +.001844 (3) | +.002724 (3) |

没有一个 selected target 的 T−F、HT−F、DH−F mean 达到 +.003。Phase A 的两数据集 STOP 均保留。MovieLens 的 T−HT 为 +.004205/+ .003206（4/4、3/4 正向），说明改变 current 来源在 TARGET history 条件下有条件收益；FULL history 下 DH−F 却接近零。平均 H/D 接近零不能推出所有条件效应接近零。t0 的 selected I 达 practical magnitude，但固定 epoch 的 I 会变号，不能称为稳定交互机制。

KuaiRand HT−F selected：t0 SD=.006779，t1 SD=.006286。2024 为 −.010962/−.010740，另三 seeds 为小正，因此 3/4 正向与负均值同时成立。2024 HT selected=16，而 F=41；不能只报告正向计数或删除该 seed。即使诊断性地考察另三 seeds 的均值，t0≈+.002476、t1≈+.001620，仍低于 .003；这不是重定义 seed 集合。固定 common epochs 下 HT−F 均值也没有 positive practical rescue。

### 4.3 Selected target ΔLogLoss（mean，括号为负向 seeds/4）

| Dataset/cell | T−F | HT−F | DH−F | HT−T |
|---|---:|---:|---:|---:|
| KuaiRand d1t0 | +.002415 (2) | −.004416 (4) | −.001774 (3) | −.006831 (3) |
| KuaiRand d1t1 | +.000539 (2) | −.004624 (4) | −.001508 (2) | −.005164 (3) |
| MovieLens d2t0 | −.000639 (1) | +.016391 (0) | +.002860 (2) | +.017030 (1) |
| MovieLens d2t1 | −.002934 (2) | +.007405 (1) | +.003543 (2) | +.010339 (0) |

KuaiRand HT 的 selected LogLoss 双 task 均 4/4 改善，AUC 却没有 practical mean rescue。这是可记录的 metric-specific 效应，不能说所有指标无变化，也不能以 LogLoss 替代预注册 AUC gate；校准/置信度和 checkpoint 差异是竞争解释。MovieLens HT 对 target 的 AUC 和 LogLoss 均不支持 selected history rescue。MovieLens T−F t0 的 mean LogLoss 微改善由少数 seed 拉动，只有 1/4 seeds 为负，不是 seed-stable 的一致收益。

## 5. Training stage：晚期 DH 是关键新增发现

### 5.1 MovieLens target ΔAUC（均值；每格为 t0 / t1）

| Common epoch | T−F | HT−F | DH−F | I |
|---|---:|---:|---:|---:|
| 9 | +.001974 / −.000923 | +.001720 / +.000509 | +.000151 / −.001822 | +.000103 / +.000390 |
| 19 | −.002992 / −.000023 | −.002748 / +.000056 | −.003563 / −.002168 | +.003319 / +.002089 |
| 29 | −.005190 / −.001603 | −.002940 / +.000021 | +.001252 / +.002365 | −.003502 / −.003990 |
| 39 | −.002309 / +.002070 | −.001987 / +.000714 | +.003138 / +.008317 | −.003461 / −.006961 |
| 49 (=final) | +.000166 / −.002069 | +.000993 / −.005235 | +.010500 / +.008107 | −.011328 / −.004941 |

DH−F 在 c39 的正向 seed 为 3/4、4/4，c49 为 4/4、3/4；c49 t0 四个 seeds 分别 +.010910、+.012488、+.008597、+.010006，SD=.001632；t1 为 +.017840、−.000307、+.011520、+.003376，SD=.008156。t0 非单 seed 驱动；t1 证据更不均匀，但 c39/c49 两个相邻晚期观察点的均值都达到 strong magnitude。

final DH−F target LogLoss 为 −.073890/−.045756，两个 tasks 均 4/4 改善。它是 ranking 与 loss 同方向的晚期分量效应，不能只因 T−F≈0 而忽略。但 c19 t0 的 DH−F 为 −.003563（4/4 负向），selected DH−F 接近零，所以不能外推为全程收益。

final 的 H=−.004671/−.007706（两 tasks 均 4/4 负向），D=+.004836/+.005637（3/4、4/4 正向）。在 TARGET current 条件下，TARGET history 相对 FULL history 的 T−DH 为 −.010334/−.010176；与 FULL current 下 HT−F 的关系不同，t0 I=−.011328 且 4/4 负向。t1 I 的 SD=.013719、只有 2/4 负向，不能宣称双 task 稳定同一交互结构。c19 I 两 tasks 均 4/4 为正，c29 均 4/4 为负：阶段反转本身比一个 selected interaction 更值得关注。

**F reference 恶化是必须检验的竞争解释。** MovieLens F target AUC 从 selected .721797/.739511 降到 final .693747/.710896，LogLoss 从 .615022/.458290 升到 .829853/.592813。DH final AUC .704247/.719003、LogLoss .755964/.547056，仍差于早期 F selected。晚期 DH 优势符合“减轻持续训练的过拟合/轨迹损伤”，没有证明比 validation-selected F 更好，也不能直接归因于某个 optimizer state 子分量。

### 5.2 KuaiRand

T−F 在所有五个 common epochs、两个 target tasks 均 0/4 正向，均值约 −.00127 至 −.00242。HT−F、DH−F 的 common mean 也都为负；没有 MovieLens 那样的晚期 target rescue。final target ΔAUC：

| Cell | T−F | HT−F | DH−F | H | D | I |
|---|---:|---:|---:|---:|---:|---:|
| d1t0 | −.001579 | −.000788 | −.000591 | −.000888 | −.000691 | −.000199 |
| d1t1 | −.001878 | −.001337 | −.001264 | −.000976 | −.000903 | +.000723 |

这些均值多数在 practical threshold 以内，不等于参数/状态完全相等，也不排除弱负作用。KuaiRand 的 selected HT LogLoss 改善到 final 不再跨 seed/metric稳定。

### 5.3 Selected epoch

| Dataset/arm | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| KuaiRand F | 30 | 47 | 41 | 24 |
| KuaiRand T | 27 | 47 | 32 | 49 |
| KuaiRand HT | 42 | 47 | 16 | 49 |
| KuaiRand DH | 40 | 30 | 48 | 27 |
| MovieLens F | 7 | 8 | 8 | 12 |
| MovieLens T | 8 | 12 | 10 | 6 |
| MovieLens HT | 10 | 10 | 16 | 19 |
| MovieLens DH | 8 | 13 | 8 | 7 |

MovieLens HT mean selected epoch=13.75，明显晚于 F=8.75、T=9、DH=9；但不构成 Equal 下“HT 跟 T、DH 跟 F”的整齐复现。KuaiRand 无简单一致的二组分离。common-stage 分析表明 Sample 的机制变化不只来自 selected epoch 的移动。

## 6. Non-target、global、worst outcomes 与 trade-off

### 6.1 Selected ΔAUC（mean）

| Dataset/cell | T−F | HT−F | DH−F | HT−T |
|---|---:|---:|---:|---:|
| KuaiRand d0t0 | −.018075 | −.017659 | +.009319 | +.000417 |
| KuaiRand d0t1 | −.030619 | −.045341 | +.013688 | −.014722 |
| KuaiRand d2t0 | −.016332 | −.018320 | +.004391 | −.001988 |
| KuaiRand d2t1 | −.013905 | −.018162 | −.000371 | −.004257 |
| MovieLens d0t0 | −.023987 | −.001457 | −.001161 | +.022530 |
| MovieLens d0t1 | −.031600 | −.006604 | −.003341 | +.024996 |
| MovieLens d1t0 | +.005919 | +.010213 | −.001602 | +.004294 |
| MovieLens d1t1 | +.017185 | +.020876 | +.006096 | +.003691 |

KuaiRand DH 的 d0 双 task selected 改善为 4/4，macro +.004366（4/4）、non-target mean +.006757（4/4），macro LogLoss −.002331（3/4）。这与“DH 没有作用”的说法矛盾；但收益不稳定贯穿所有 common stages，macro c9 为 −.003756、c29 +.005269、final +.001927；d0t0 final为−.002338。不能把 selected 的 d0 优势概括为全阶段方法收益。

KuaiRand HT 未复现 Equal 中“保留 rescue 同时降低 T damage”：target 无 practical AUC rescue，selected HT−T macro −.003576，d0t1 −.014722；相对 F 的 macro −.016973（4/4 负向）。

MovieLens HT−T 的 d0 双 task 为 +.022530/+.024996（均 4/4），d1 双 task +.004294/+.003691（4/4、3/4），macro +.008017（4/4）。不过 target HT−T 是 −.004205/−.003206，target LogLoss也更差；所以是 non-target recovery 与 target cost 的诊断 trade-off，不能称为完整保留 target rescue。HT−F 在 d1 的 +.010213/+.020876 均 4/4，但 d0t1 的 −.006604 均 4/4 负向，six-cell macro 改善不等于每域改善。

### 6.2 Macro / worst ΔAUC

| Dataset/checkpoint | outcome | T−F | HT−F | DH−F |
|---|---|---:|---:|---:|
| KuaiRand selected | macro | −.013397 | −.016973 | +.004366 |
| KuaiRand selected | worst-cell | −.018000 | −.025310 | +.009395 |
| KuaiRand selected | worst-domain | 见完整表 | 见完整表 | +.010057 |
| KuaiRand final | macro | −.016770 | −.017179 | +.001927 |
| KuaiRand final | worst-cell | −.034088 | −.037126 | −.002338 |
| MovieLens selected | macro | −.004671 | +.003345 | +.000022 |
| MovieLens selected | worst-cell | −.004049 | +.002244 | −.003035 |
| MovieLens final | macro | +.020479 | +.025284 | +.012765 |
| MovieLens final | worst-cell | +.035714 | +.036541 | +.012000 |

worst 指标先在 seed/arm 内取极值，再对四 seeds 平均；极值 cell/domain 身份可以变化，不能等同于一个固定域的 causal effect。所有 worst-domain 和对应 LogLoss 的完整数值列于附表及 CSV。

MovieLens final DH 的 non-target d0t0/d0t1/d1t0 均 4/4 相对 F 改善；d1t1 mean −.007149 却有 3/4 正向，说明单 seed 大损伤能使均值反向。final macro 与 LogLoss 的大改善同时伴随 F reference 全局恶化，不是证实稳健部署收益。

**size trade-off 判断：局部存在，无法建立单调或普遍的 size 规律。** KuaiRand selected DH 提升较小 d0，最大 target 基本持平且微负；未证明“稳定 practical large-domain decline”。MovieLens selected HT 提升真正最小的 d1、损伤中等 d0、最大 d2 相对 F 的 AUC小负而 LogLoss更差，相对 T 则有 practical target cost。最小域并非普遍受伤或普遍受益。Domain size 可能调节有效贡献，但不是本矩阵可独立识别的主解释变量。

## 7. Equal vs Sample：baseline、magnitude 与 recovery structure

### 7.1 Selected target ΔAUC anchors

| Dataset/cell | Equal T−F | Sample T−F | Equal HT−F | Sample HT−F | Equal DH−F | Sample DH−F |
|---|---:|---:|---:|---:|---:|---:|
| KuaiRand d1t0 | +.041426 | −.000657 | +.041085 | −.000883 | +.000875 | −.000157 |
| KuaiRand d1t1 | +.042254 | −.000790 | +.040528 | −.001470 | +.000821 | −.000675 |
| MovieLens d2t0 | +.019408 | +.001899 | +.018114 | −.002306 | +.005859 | −.000343 |
| MovieLens d2t1 | +.014270 | +.002555 | +.014149 | −.000651 | +.003583 | +.000482 |

MovieLens selected T−F 缩小约 90.2%/82.1%；KuaiRand 从 strong positive 到微负。完整四臂进一步显示 selected HT rescue 也消失，不能只称为“同一 recovery pattern 等比变小”。

| Target cell | Equal F/T AUC | Sample F/T AUC |
|---|---:|---:|
| KuaiRand d1t0 | .6531 / .6946 | .6976 / .6969 |
| KuaiRand d1t1 | .6815 / .7238 | .7261 / .7253 |
| MovieLens d2t0 | .7042 / .7237 | .7218 / .7237 |
| MovieLens d2t1 | .7280 / .7423 | .7395 / .7421 |

rescue 变小主要伴随 FULL baseline 追上原 T 的绝对表现；T 的绝对 target AUC变化很小。这支持 objective aggregation 对最大域 F baseline 的影响，但不证明 history 操作等价于固定 reweighting，也不证明 objective 的普遍正确性。

MovieLens late DH−F 在 Equal final 为 +.009772/+.010257，Sample final 为 +.010500/+.008107。**晚期 current-source 分量效应在 Sample 下没有整体消失**；它与 selected history-dominant rescue 的消失是两个不同事实。KuaiRand 不复现该晚期 target结构。

### 7.2 HR / CR（数值计算，禁止解释为稳定 recovery）

| Dataset/cell | selected HR | selected CR | final HR | final CR |
|---|---:|---:|---:|---:|
| KuaiRand d1t0 | 1.344 | .240 | .499 | .374 |
| KuaiRand d1t1 | 1.860 | .854 | .712 | .673 |
| MovieLens d2t0 | −1.214 | −.181 | 5.999 | 63.413 |
| MovieLens d2t1 | −.255 | .189 | 2.531 | −3.919 |

所有这些 Sample target 分母 mean(T−F) 都未达到 positive practical magnitude，且 selected 不满足 seed稳定。MovieLens final d2t0 的 CR≈63 只是 denominator≈+.000166 的数学结果；KuaiRand selected HR>1 是负数相除，不是“超过100%救回”。不能与 Equal 的有效 positive-rescue HR/CR 直接作比例机制外推。应以绝对条件 contrasts、符号和阶段判断。

## 8. M1 未通过的科研含义

KuaiRand 8 个 early seed/epoch m conjunction 只有 4 个通过：2022 ep0/1、2023 ep1、2024 ep0 失败，共 5/16 个单边 inequality失败。v 为 8/8；MovieLens m/v 均 8/8。独立读取 64 个 early state dumps 重算 cosines 与原表完全相同；不是舍入噪声。

per-step installed source、applied delta、private delta、状态步数及 RNG 都通过。这支持“指定 optimizer policies 执行正确”，不支持“所有 M1 识别条件已满足”。因为梯度和 coupled weight decay 都在各臂不同 θ 上计算，same nominal history source 不要求 cross-run realized m tensors匹配；signed EMA、轨迹分歧和不同每 epoch 步数是 plausible explanations，尚未被独立证实为这几个失败的完整原因。

不豁免 gate，不改 threshold，不删 seed，不为强迫 M1 通过而重跑正确代码。KuaiRand 结论限定为策略轨迹的结果差异；对纯 persistent-state mediator 的机制证据为 WEAK。MovieLens 的 M1 通过同样不能把政策干预完全简化成 m/v单子状态的因果识别。本阶段没有新 optimizer、curriculum、ratio sweep 或 m/v factorization。

## 9. 稳定现象、反例、竞争解释与证据强度

| 判断 | 证据等级 | 理由 / 限制 |
|---|---|---|
| 请求的四臂矩阵已真实执行并记录，routing/paired-policy fidelity正确 | STRONG | 32 runs、8 个新增并发波次、独立哈希/全记录审计；科研 M1失败单列 |
| 两个 setting 下 selected T−F 无 practical、seed-stable target rescue | MEDIUM | 两数据集双task共4seeds、common-stage支撑；不能证明数学零或全域普遍零 |
| MovieLens 存在晚期 DH−F target优势 | MEDIUM | c39/c49、双指标；t0稳定4/4，t1变异较大；单数据集、过拟合窗口，观察点相关 |
| Equal selected history-dominant结构没有按原样延伸到 Sample | MEDIUM | HT−F 消失；MovieLens HT−T selected负，HR/CR失去rescue解释；KuaiRand M1限制 |
| KuaiRand selected DH non-target收益、HT损伤；MovieLens HT non-target recovery伴target cost | MEDIUM | 多cell/seed；固定stage和metric未全部一致，不是通用安全方法 |
| Domain size 是主要因果解释 | WEAK | 两target均最大、没有ratio/identity干预；MovieLens最小d1与中等d0表现非单调 |
| 所有 optimizer-state效应被 Sample 消除，或可形成可部署新方法 | WEAK | 晚期DH和LogLoss反例直接限制前者；selected/global trade-offs、M1限制后者 |

重复的主要现象：selected 目标 rescue 下降；KuaiRand T/HT的 non-target AUC损伤；MovieLens早晚阶段DH条件效应不同。主要反例：MovieLens late DH 的 strong-magnitude AUC/loss优势；KuaiRand selected HT 的4/4 target LogLoss改善；KuaiRand selected DH 的4/4 macro改善。它们都反驳“没有任何 optimizer分量差异”，但都不构成跨数据集、全阶段、全指标的稳健 target rescue。

竞争解释：

1. **Objective aggregation 改善原 FULL target baseline。** 与 F/T绝对数值、两个target均最大域一致；没有隔离“size本身”与target身份，也没有证明 history操作等同reweighting。
2. **持续训练/过拟合及 checkpoint效应。** MovieLens late F的 AUC下降和loss上升、DH早期无优势而晚期有优势支持此解释；common-stage排除了只比较不同selected epochs，却不能排除轨迹正则化。
3. **history×current条件交互或抵消。** T−F小而DH−F大、I早晚反转与此一致；证据是策略factorial contrasts，尚不是纯state中介识别。
4. **Metric/selection/seed异质性。** KuaiRand HT selected loss与AUC不同、2024 HT early selection、MovieLens final d1t1一个seed拉动均值，限制仅按均值或正向数裁决。
5. **Dataset、任务难度与训练时钟。** KuaiRand49 main steps/epoch，MovieLens5；label分布、稀疏性、域身份和任务不同。F的baseline给出这些任务在本协议的表现，不能单凭AUC跨dataset比较难度，也不能把小域等同困难域。

## 10. S6 原18个问题的完整回答

| 问题 | 本次答案 |
|---|---|
| 1–2 KuaiRand/MovieLens Sample T−F仍存在？ | selected 无stable practical rescue；不是证明零 |
| 3 magnitude比Equal减多少？ | KuaiRand≈+.041/+.042到微负；MovieLens selected缩小90.2%/82.1% |
| 4 .003 threshold？ | selected T/HT/DH相对F的target mean均未达到；MovieLens late DH达到 |
| 5 seed-stable？ | selected T−F否；late MovieLens DH t0 4/4、t1 3/4，任务1异质性更大 |
| 6 task-stable？ | 不全局成立；晚期DH loss双task一致，selected/full交互不同 |
| 7 checkpoint-stable？ | 否；强烈stage依赖；final/common49只算一次 |
| 8 AUC/LogLoss一致？ | 不普遍；late ML DH一致，selected KR HT不一致 |
| 9 non-target damage？ | 有，且不能用“最小域都受损”概括 |
| 10 Sample消除了Equal history effect？ | selected target history-rescue消失；不能声称所有history/current效应消失 |
| 11 HT≈T？ | 无stable recovery可据；ML selected HT有target cost，KR小mean伴较大seed变异 |
| 12 DH contribution？ | selected target近零；ML c39/c49显著practical magnitude且loss一致；HR/CR分母不可解释 |
| 13 两dataset DH discrepancy还在？ | 晚期仍在：ML late target收益，KR无；selected区别减弱 |
| 14 rescue/damage可分离？ | 未复现“保target rescue又消damage”的完整结构；ML有non-target恢复与target成本 |
| 15 residual值得继续？ | 可做有针对性的诊断证伪；不支持直接方法化 |
| 16 ratio/stage boundary？ | stage最有信息；没有独立size证据优先支持ratio sweep |
| 17 m/v factorization？ | 目前不优先：需先分离状态作用与轨迹/过拟合解释，KR M1失败也需解释 |
| 18 新方法必要性？ | 未建立；selected没有稳健target/global全面改善 |

## 11. 裁决与下一项最有判别力的实验

**原预注册 Phase A：双 STOP，保持不变。扩展四臂诊断：B. CONDITIONAL RESIDUAL，限定为 MovieLens晚期DH、以及部分非目标/LogLoss结果的条件性效应。不是 C. ROBUST RESIDUAL，也不是对方法发展的 GO。** 因此“完全停止一切机制边界问题”与“立刻发展新的optimizer”都超出当前证据；现有结果支持收缩到可被否定的机制问题。

最有判别力的下一项是**固定相同模型参数和训练时钟的 early/late state×current 交叉短程 continuation**：研究目的应是区分“persistent state自身携带作用”与“长程参数轨迹/过拟合导致late DH优势”。未来 designer 应预注册共同参数锚点、state来源、同序数据/RNG、固定观察时刻、validation-only规则和解释边界，而非实现一个新方法。

如果同 θ 下 state选择仍稳定改变更新和短程held-out指标，才增强独立状态作用解释；如果差异主要在已经分歧、过拟合的长程轨迹中出现，匹配参数后不复现，则削弱该解释并支持trajectory/regularization解释。应同时报告early与late、target与non-target、AUC与LogLoss及4seeds，保留否定机制假说的可能。本次没有启动该新实验、ratio或m/v拆解。

## 12. 完整产物与复现

- [完整绝对/contrast/逐seed target表](stage_S6_complete_tables.md)：所有 target/non-target cells 的 selected/final/common-stage AUC与LogLoss、macro、worst-domain、worst-cell、Equal/Sample anchors。
- [逐seed绝对结果](../stage_S6_analysis_tables/absolute_seed_outcomes.csv)、[absolute mean/SD](../stage_S6_analysis_tables/absolute_summary.csv)。
- [逐seed factorial contrasts](../stage_S6_analysis_tables/factorial_seed_contrasts.csv)、[factorial mean/SD/方向计数](../stage_S6_analysis_tables/factorial_summary.csv)。
- [HR/CR与分母有效性](../stage_S6_analysis_tables/recovery_ratios.csv)。
- [同seed Equal/Sample contrasts](../stage_S6_analysis_tables/objective_seed_comparison.csv)、[objective contrast summaries](../stage_S6_analysis_tables/objective_comparison_summary.csv)。
- [1440项anchor锁检查](../stage_S6_analysis_tables/equal_sample_manifest_locks.csv)、[F/T不变与final/common49检查](../stage_S6_analysis_tables/data_integrity.csv)。
- [目标四臂轨迹图](../stage_S6_analysis_tables/target_auc_trajectories.png)、[目标paired contrasts图](../stage_S6_analysis_tables/target_auc_contrasts.png)。图中阴影为seed SD，不是confidence interval。
- [原始KuaiRand表](../stage_S6_outputs/stage_S6_raw_results.csv)、[原始MovieLens表](../stage_S6ML_outputs/stage_S6ML_raw_results.csv)、对应 run、gradient、probe、moment审计及checkpoints保留。
- 分析复现：`venv/Scripts/python.exe experiments/stage_S6_factorial_analysis.py`，再执行 `experiments/stage_S6_factorial_tables.py`。两文件只做结果统计/渲染，不训练或修改算法；未引入统计显著性、bootstrap或事后threshold。

完整四臂结果已补齐并完成诊断分析；KuaiRand M1失败是报告中的负结果与识别限制，不是被隐去或“修复成通过”的结果。
