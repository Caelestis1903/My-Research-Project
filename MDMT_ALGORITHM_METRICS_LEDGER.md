# MDMT 算法性能总台账（AUC / LogLoss）

> 更新：2026-11-03。此文件位于 **main 主目录**，是后续研究论文与算法比较的**持续维护台账**。本轮已回填10月29日发表模型比较，以及11月1日至3日的八组已执行实验/补充数据，共计 **110 条“数据集×算法配置×评价协议”的结果记录**。不代表所有更早历史算法都已回填；未找到的数字一律不填造。
>
> **正式Baseline仅可使用已发表论文中的模型**。自研M、H、C_MIX、PAIR、ECCM、SAD、AIG、EMA改造、调参变体等只列为**研究方法/组件/内部消融控制**，不可自行升级为“论文基线”。正式比较须在一致的数据与训练/选模协议下重新评估；不同来源表格的绝对指标**不得直接排名**。

## 读取与更新准则

- **每一格格式均为 `AUC / LogLoss`（越高 / 越低越好）**，包括d0t0、d0t1、d1t0、d1t1、d2t0、d2t1，以及总体 `Macro`、`Traffic`、`Pooled`。所有值是**test/E上每个独立seed对应selected checkpoint指标，再跨seed算术平均**，不允许跨epoch挑test最优点。
- **Macro AUC与Macro LogLoss**是六个域任务格的等权平均；`Traffic` 是按测试人口/标签请求数加权的各格指标，**不是跨域pooled AUC**；`Pooled` 在原始数表具有对应总体指标时展示，否则为“—”。不同协议中“总AUC”可能指macro、traffic或pooled，因此**必须保留三者名称，不得混称**。
- `n` 是**真正独立训练seed数量**：n=1或n=2记录仅供比较现象、选择后续工作，**不得性能否决**。最新规范要求至少5seed方可做性能否决；五seed配对平均Δmacro AUC **≥+0.003**，并满足每个cell平均ΔAUC **≥−0.001** 才算AUC总体合格且满足单格保护；LogLoss以五seed配对平均下降为准，无需所有seed同向。**这三条只有在基线和候选处于同一实验协议、相同seed且基线身份合法时才应用**。
- 这里记录**绝对值**，不将本实验的自研方法当作下一实验的正式基线；内部X与A0＋EMA、M与TM之间的数值比较可以标注为“增量消融”，不能替代**与已发表模型的正式基线比较**。**固定EMA不是一篇独立新发表的论文模型**；在此处属于发表架构的内部训练配置/后处理变体。
- **审计口径**：尽量来自原始CSV中的每个seed指标；只存有各格均值的旧报告使用其已报告均值。不得以重用同源轨迹扩大n；假如存在不同试验协议（模型选择指标、训练阶段、数据人口、batch、优化器、学习率、loss权重、模型结构或评价split），建立独立分组，不合并为一行。
- 后续每完成一个正式实验，Codex/Pro须从冻结的逐seed结果更新本表的相应协议分组（新增或更新行），保留来源路径、n、所用模型参数平均方式与选点规则。没有原数值的历史方法按“待回填”处理，不能只填AUC缺少LogLoss却暗示全指标已知。

## 发表模型资格（已核验）

| 模型 | 对应论文与发表记录 | 备注 |
|---|---|---|
| M3oE | [SIGIR 2024, DOI:10.1145/3626772.3657686](https://doi.org/10.1145/3626772.3657686) | 仓库采用M3OE_FIXED1固定融合配置，属于发表模型的适配而非完全等同论文全部AutoML设置 |
| HyperGate | [WWW 2025 Companion, DOI:10.1145/3701716.3715236](https://doi.org/10.1145/3701716.3715236) | 10/29配置中部分超参未核验为作者原始配置 |
| AutoIFS | [WSDM 2026, DOI:10.1145/3773966.3777992](https://doi.org/10.1145/3773966.3777992) | 适配后的模型，不能直接声称复现了原论文的严格指标 |

## 已回填性能矩阵

### 已发表论文模型（MDMT-SOTA-2526，V按LogLoss选择）

基线比较仅限本节已经正式发表的模型；该组以 V 的 LogLoss 选择checkpoint，再在 E 测试。与下列固定EMA宏AUC选点实验**不是同一协议**，不得跨表用绝对值宣布优胜。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)。

**MovieLens**（3条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| AUTOIFS | 5 | 0.760862 / 0.587375 | 0.773400 / 0.407804 | 0.765855 / 0.578958 | 0.788988 / 0.388491 | 0.733951 / 0.593736 | 0.754921 / 0.439475 | **0.762996 / 0.499307** | 0.764383 / 0.497721 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |
| HYPERGATE | 5 | 0.763724 / 0.588721 | 0.778595 / 0.407529 | 0.770253 / 0.575598 | 0.790877 / 0.387861 | 0.741796 / 0.586407 | 0.755702 / 0.440394 | **0.766825 / 0.497752** | 0.768132 / 0.496053 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |
| M3OE_FIXED1 | 5 | 0.757174 / 0.607154 | 0.769158 / 0.423018 | 0.763044 / 0.588347 | 0.782624 / 0.397782 | 0.735875 / 0.604985 | 0.748798 / 0.451543 | **0.759446 / 0.512138** | 0.760736 / 0.510035 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |

**KuaiRand**（3条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| AUTOIFS | 5 | 0.632149 / 0.287863 | 0.657698 / 0.179818 | 0.696664 / 0.504585 | 0.724213 / 0.345310 | 0.661486 / 0.638915 | 0.683280 / 0.636802 | **0.675915 / 0.432216** | 0.709472 / 0.424544 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |
| HYPERGATE | 5 | 0.664229 / 0.284771 | 0.684231 / 0.178128 | 0.692170 / 0.506218 | 0.721500 / 0.347905 | 0.656408 / 0.643756 | 0.679572 / 0.638912 | **0.683018 / 0.433282** | 0.706206 / 0.426621 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |
| M3OE_FIXED1 | 5 | 0.641343 / 0.289298 | 0.652788 / 0.183433 | 0.688998 / 0.497666 | 0.715233 / 0.338312 | 0.646106 / 0.658003 | 0.667172 / 0.652983 | **0.668607 / 0.436616** | 0.701203 / 0.417867 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)¹ |

¹ 对应算法行存在来源报告中的精度/适配说明；以原始结果及原分支完整报告为准。

### 11/01 GA31-R3：MASK / Gaussian / 反向配对（普通Adam，无EMA）

普通Adam，无固定EMA；5seed；含 B-ML / L2-ML / B-KR。此组属于自研机制实验，不能将其算法变成新的正式论文基线。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv)。

**MovieLens**（5条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| B-ML · online | 5 | 0.770386 / 0.583069 | 0.785008 / 0.405187 | 0.777947 / 0.569970 | 0.797240 / 0.387010 | 0.752056 / 0.579750 | 0.767316 / 0.435245 | **0.774992 / 0.493372** | 0.776233 / 0.491780 | 0.776334 / 0.491780 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| G-ANTI2 · online | 5 | 0.769827 / 0.586416 | 0.785957 / 0.406435 | 0.777773 / 0.572416 | 0.796498 / 0.387463 | 0.750398 / 0.582389 | 0.766566 / 0.437241 | **0.774503 / 0.495393** | 0.775725 / 0.493733 | 0.775739 / 0.493733 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| G-IID2 · online | 5 | 0.770410 / 0.582885 | 0.785173 / 0.401211 | 0.778472 / 0.570645 | 0.797144 / 0.380556 | 0.751807 / 0.579265 | 0.766893 / 0.432579 | **0.774983 / 0.491190** | 0.776244 / 0.489521 | 0.775828 / 0.489521 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| L2-ML · online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | **0.775190 / 0.494891** | 0.776435 / 0.493229 | 0.776760 / 0.493229 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| M1 · online | 5 | 0.765511 / 0.588061 | 0.781541 / 0.410757 | 0.778836 / 0.571450 | 0.797075 / 0.384098 | 0.751150 / 0.582170 | 0.767263 / 0.436005 | **0.773563 / 0.495424** | 0.775080 / 0.493437 | 0.775272 / 0.493437 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |

**KuaiRand**（4条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| B-KR · online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | **0.671907 / 0.437172** | 0.711621 / 0.426397 | 0.716990 / 0.426397 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| G-ANTI2 · online | 5 | 0.634289 / 0.290672 | 0.644021 / 0.182163 | 0.699951 / 0.503112 | 0.727348 / 0.343877 | 0.656848 / 0.647031 | 0.676195 / 0.645921 | **0.673109 / 0.435463** | 0.712522 / 0.423208 | 0.717987 / 0.423208 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| G-IID2 · online | 5 | 0.637858 / 0.288326 | 0.656254 / 0.181178 | 0.698007 / 0.504154 | 0.725478 / 0.344828 | 0.651820 / 0.652064 | 0.676659 / 0.647145 | **0.674346 / 0.436282** | 0.710714 / 0.424193 | 0.716155 / 0.424193 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |
| M1 · online | 5 | 0.629936 / 0.291354 | 0.637275 / 0.184115 | 0.700186 / 0.499775 | 0.728270 / 0.340586 | 0.662714 / 0.644119 | 0.680441 / 0.644287 | **0.673137 / 0.434039** | 0.713070 / 0.419950 | 0.718590 / 0.419950 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv) |

### 11/01 GAM01：固定EMA与差分动量（在线 / Shadow分开）

A0、A1、A2、A3分别给出在线和shadow估计器；5seed。同一轨迹的在线/EMA不能计作两个科学重复。ML本组EMAρ=.995，与AIGM调参后的ML时间常数不同。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv)。

**MovieLens**（8条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | **0.775190 / 0.494891** | 0.776435 / 0.493229 | 0.776760 / 0.493229 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A0 · shadow | 5 | 0.775888 / 0.589058 | 0.790252 / 0.413316 | 0.781800 / 0.583237 | 0.800939 / 0.398475 | 0.757156 / 0.587798 | 0.771497 / 0.444260 | **0.779589 / 0.502691** | 0.780718 / 0.501492 | 0.781067 / 0.501492 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A1 · online | 5 | 0.769232 / 0.585462 | 0.784200 / 0.402234 | 0.778997 / 0.573600 | 0.797354 / 0.384972 | 0.753607 / 0.579324 | 0.767384 / 0.436268 | **0.775129 / 0.493643** | 0.776452 / 0.492125 | 0.776369 / 0.492125 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A1 · shadow | 5 | 0.776213 / 0.586291 | 0.790747 / 0.408542 | 0.782100 / 0.581334 | 0.800825 / 0.394932 | 0.756801 / 0.585881 | 0.771386 / 0.441917 | **0.779679 / 0.499816** | 0.780798 / 0.498663 | 0.781120 / 0.498663 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A2 · online | 5 | 0.771138 / 0.581731 | 0.784897 / 0.402380 | 0.777228 / 0.574467 | 0.797006 / 0.383816 | 0.752521 / 0.578976 | 0.766774 / 0.432909 | **0.774927 / 0.492380** | 0.776112 / 0.490996 | 0.775512 / 0.490996 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A2 · shadow | 5 | 0.776091 / 0.586754 | 0.790227 / 0.411072 | 0.781843 / 0.582930 | 0.800753 / 0.397759 | 0.757117 / 0.585760 | 0.771369 / 0.443555 | **0.779567 / 0.501305** | 0.780687 / 0.500228 | 0.780951 / 0.500228 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A3 · online | 5 | 0.768998 / 0.588135 | 0.785183 / 0.407790 | 0.778703 / 0.571086 | 0.796736 / 0.384744 | 0.751720 / 0.582272 | 0.767420 / 0.438540 | **0.774793 / 0.495428** | 0.776081 / 0.493502 | 0.776077 / 0.493502 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A3 · shadow | 5 | 0.776236 / 0.587117 | 0.790879 / 0.409841 | 0.781199 / 0.581833 | 0.800667 / 0.396001 | 0.757096 / 0.583262 | 0.770994 / 0.441273 | **0.779512 / 0.499888** | 0.780585 / 0.498778 | 0.780733 / 0.498778 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |

**KuaiRand**（8条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | **0.671907 / 0.437172** | 0.711621 / 0.426397 | 0.716990 / 0.426397 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A0 · shadow | 5 | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | **0.676523 / 0.435947** | 0.718165 / 0.423121 | 0.723348 / 0.423121 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A1 · online | 5 | 0.634508 / 0.290164 | 0.652883 / 0.181583 | 0.696363 / 0.495562 | 0.724239 / 0.337269 | 0.657782 / 0.643723 | 0.681168 / 0.637526 | **0.674491 / 0.430971** | 0.709303 / 0.416206 | 0.714759 / 0.416206 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A1 · shadow | 5 | 0.634346 / 0.289394 | 0.661911 / 0.180681 | 0.700652 / 0.504592 | 0.728460 / 0.345690 | 0.658848 / 0.645452 | 0.683779 / 0.639891 | **0.677999 / 0.434283** | 0.713542 / 0.424782 | 0.718855 / 0.424782 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A2 · online | 5 | 0.637577 / 0.289402 | 0.655814 / 0.181601 | 0.698522 / 0.506022 | 0.726876 / 0.347682 | 0.653469 / 0.649313 | 0.670616 / 0.651440 | **0.673812 / 0.437577** | 0.711633 / 0.426525 | 0.716950 / 0.426525 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A2 · shadow | 5 | 0.635604 / 0.289392 | 0.656748 / 0.181731 | 0.703312 / 0.501127 | 0.731538 / 0.342793 | 0.658087 / 0.647596 | 0.677752 / 0.645479 | **0.677174 / 0.434686** | 0.716314 / 0.421692 | 0.721568 / 0.421692 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A3 · online | 5 | 0.636437 / 0.289713 | 0.652827 / 0.181717 | 0.698516 / 0.501618 | 0.725908 / 0.339387 | 0.651213 / 0.649630 | 0.672721 / 0.646832 | **0.672937 / 0.434816** | 0.711133 / 0.420275 | 0.716581 / 0.420275 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |
| A3 · shadow | 5 | 0.637261 / 0.289453 | 0.651364 / 0.182635 | 0.703379 / 0.503424 | 0.731347 / 0.344707 | 0.653140 / 0.651985 | 0.677374 / 0.646345 | **0.675644 / 0.436425** | 0.716213 / 0.423785 | 0.721490 / 0.423785 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv) |

### 11/01 AIGM01：基线、固定EMA、Gaussian＋EMA

5seed；此组的单格均值根据报告中的A0绝对值及逐格EMA/增强差值复原，存在末位舍入；总宏指标为六格算术平均。与GA31在共享来源的A0轨迹不能重复计数。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)。

**MovieLens**（3条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 + fixed EMA | 5 | 0.774393 / 0.585267 | 0.789598 / 0.407691 | 0.782021 / 0.574434 | 0.800922 / 0.390560 | 0.756328 / 0.580900 | 0.770970 / 0.437985 | **0.779039 / 0.496140** | 0.780253 / 0.494692 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |
| A0 online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | **0.775190 / 0.494891** | 0.776435 / 0.493229 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |
| Gaussian + fixed EMA | 5 | 0.774415 / 0.585146 | 0.789658 / 0.407082 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437730 | **0.779357 / 0.495822** | 0.780600 / 0.494372 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |

**KuaiRand**（3条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 + fixed EMA | 5 | 0.636991 / 0.289644 | 0.649946 / 0.182405 | 0.705306 / 0.502457 | 0.733389 / 0.344333 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | **0.676523 / 0.435947** | 0.718165 / 0.423121 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |
| A0 online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | **0.671906 / 0.437172** | 0.711621 / 0.426397 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |
| Gaussian + fixed EMA | 5 | 0.634346 / 0.289393 | 0.661910 / 0.180680 | 0.700652 / 0.504591 | 0.728460 / 0.345690 | 0.658849 / 0.645452 | 0.683779 / 0.639891 | **0.677999 / 0.434283** | 0.713541 / 0.424782 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v3/research_reports/2026-11-01_aig_m/REPORT.md)¹ |

¹ 对应算法行存在来源报告中的精度/适配说明；以原始结果及原分支完整报告为准。

### 11/02 GATS02：增强覆盖、任务投影及时间平均

记录仓库已经运行的fixed_ema和必要adaptive/replay对照；B2/KuaiRand具备5seed，其余多数为2seed；不可把两seedNO_GO视为已完成五seed否决。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv)。

**MovieLens**（12条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 2（未满5） | 0.774743 / 0.584929 | 0.789599 / 0.405250 | 0.782404 / 0.573633 | 0.800817 / 0.388432 | 0.757187 / 0.580154 | 0.771098 / 0.437580 | **0.779308 / 0.494996** | 0.780512 / 0.493523 | 0.780946 / 0.493523 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · adaptive_ema | 2（未满5） | 0.774822 / 0.584333 | 0.788578 / 0.405411 | 0.782601 / 0.572285 | 0.800948 / 0.387546 | 0.757724 / 0.578703 | 0.771887 / 0.435247 | **0.779427 / 0.493921** | 0.780654 / 0.492415 | 0.780962 / 0.492415 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · fixed_ema | 2（未满5） | 0.775055 / 0.584389 | 0.789241 / 0.405461 | 0.782578 / 0.572967 | 0.801066 / 0.388140 | 0.757513 / 0.579286 | 0.771952 / 0.436048 | **0.779568 / 0.494382** | 0.780775 / 0.492908 | 0.781138 / 0.492908 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · replay_rho | 2（未满5） | 0.774058 / 0.584316 | 0.788562 / 0.404428 | 0.782427 / 0.572577 | 0.801104 / 0.387031 | 0.757277 / 0.578321 | 0.771574 / 0.435196 | **0.779167 / 0.493645** | 0.780425 / 0.492164 | 0.780746 / 0.492164 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B1 · fixed_ema | 2（未满5） | 0.773409 / 0.585778 | 0.788685 / 0.404648 | 0.781709 / 0.574003 | 0.800431 / 0.387552 | 0.756732 / 0.579680 | 0.770960 / 0.436273 | **0.778654 / 0.494656** | 0.779885 / 0.493176 | 0.780253 / 0.493176 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · adaptive_ema | 2（未满5） | 0.773779 / 0.585579 | 0.787583 / 0.406097 | 0.781509 / 0.574057 | 0.800646 / 0.387981 | 0.756187 / 0.580516 | 0.770743 / 0.436473 | **0.778408 / 0.495117** | 0.779669 / 0.493609 | 0.780008 / 0.493609 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · fixed_ema | 2（未满5） | 0.773710 / 0.587096 | 0.789010 / 0.406295 | 0.782117 / 0.575617 | 0.801210 / 0.389529 | 0.756631 / 0.581819 | 0.771044 / 0.438588 | **0.778954 / 0.496491** | 0.780215 / 0.495018 | 0.780654 / 0.495018 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · replay_rho | 2（未满5） | 0.773643 / 0.586385 | 0.788187 / 0.406297 | 0.781581 / 0.574841 | 0.801078 / 0.388168 | 0.756374 / 0.580635 | 0.770780 / 0.436948 | **0.778607 / 0.495546** | 0.779873 / 0.494041 | 0.780197 / 0.494041 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B3 · fixed_ema | 2（未满5） | 0.774328 / 0.584344 | 0.789353 / 0.403867 | 0.782298 / 0.573147 | 0.800078 / 0.387546 | 0.756062 / 0.580595 | 0.770687 / 0.436663 | **0.778801 / 0.494360** | 0.780008 / 0.492893 | 0.780433 / 0.492893 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| R1 · fixed_ema | 2（未满5） | 0.774418 / 0.585504 | 0.789404 / 0.405329 | 0.782157 / 0.575109 | 0.800900 / 0.388850 | 0.756793 / 0.580980 | 0.771734 / 0.436861 | **0.779234 / 0.495439** | 0.780443 / 0.494022 | 0.780845 / 0.494022 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| R2 · fixed_ema | 2（未满5） | 0.775111 / 0.584039 | 0.789842 / 0.403823 | 0.782481 / 0.573069 | 0.801260 / 0.386249 | 0.757397 / 0.579495 | 0.772208 / 0.435539 | **0.779716 / 0.493702** | 0.780908 / 0.492216 | 0.781320 / 0.492216 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| R3 · fixed_ema | 2（未满5） | 0.774272 / 0.586191 | 0.788901 / 0.406331 | 0.782457 / 0.574441 | 0.801166 / 0.389050 | 0.757858 / 0.580287 | 0.771857 / 0.437743 | **0.779418 / 0.495674** | 0.780654 / 0.494188 | 0.781083 / 0.494188 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |

**KuaiRand**（10条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | **0.676523 / 0.435947** | 0.718165 / 0.423121 | 0.723348 / 0.423121 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · adaptive_ema | 2（未满5） | 0.633575 / 0.289361 | 0.664454 / 0.180433 | 0.702103 / 0.500657 | 0.729946 / 0.341953 | 0.660704 / 0.644448 | 0.684114 / 0.638896 | **0.679149 / 0.432624** | 0.715001 / 0.421006 | 0.720264 / 0.421006 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · fixed_ema | 5 | 0.634346 / 0.289394 | 0.661911 / 0.180681 | 0.700652 / 0.504592 | 0.728460 / 0.345690 | 0.658848 / 0.645452 | 0.683779 / 0.639891 | **0.677999 / 0.434283** | 0.713542 / 0.424782 | 0.718855 / 0.424782 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| A1 · replay_rho | 2（未满5） | 0.634199 / 0.289288 | 0.664990 / 0.180384 | 0.702054 / 0.500759 | 0.729904 / 0.342066 | 0.660686 / 0.644516 | 0.684142 / 0.638928 | **0.679329 / 0.432657** | 0.714963 / 0.421111 | 0.720220 / 0.421111 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B1 · fixed_ema | 2（未满5） | 0.644504 / 0.287996 | 0.642713 / 0.183317 | 0.706243 / 0.501387 | 0.734465 / 0.342567 | 0.656572 / 0.650258 | 0.673650 / 0.649343 | **0.676358 / 0.435811** | 0.719142 / 0.421735 | 0.724348 / 0.421735 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · adaptive_ema | 5 | 0.636538 / 0.289140 | 0.658841 / 0.181472 | 0.704583 / 0.499656 | 0.732594 / 0.341272 | 0.657110 / 0.648184 | 0.677482 / 0.644765 | **0.677858 / 0.434081** | 0.717467 / 0.420220 | 0.722735 / 0.420220 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · fixed_ema | 5 | 0.638383 / 0.288697 | 0.660533 / 0.181179 | 0.703877 / 0.500059 | 0.732003 / 0.341803 | 0.655743 / 0.648685 | 0.677488 / 0.644594 | **0.678004 / 0.434169** | 0.716844 / 0.420676 | 0.722117 / 0.420676 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B2 · replay_rho | 5 | 0.636544 / 0.289177 | 0.658117 / 0.181588 | 0.704636 / 0.499465 | 0.732652 / 0.341126 | 0.656812 / 0.648277 | 0.677269 / 0.644770 | **0.677672 / 0.434067** | 0.717516 / 0.420056 | 0.722789 / 0.420056 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| B3 · fixed_ema | 2（未满5） | 0.644576 / 0.287943 | 0.661297 / 0.180771 | 0.702375 / 0.500818 | 0.730773 / 0.341806 | 0.654148 / 0.649496 | 0.675912 / 0.645948 | **0.678180 / 0.434464** | 0.715527 / 0.421052 | 0.720771 / 0.421052 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |
| R2 · fixed_ema | 2（未满5） | 0.648146 / 0.287689 | 0.667267 / 0.180761 | 0.702837 / 0.499749 | 0.730543 / 0.341903 | 0.664442 / 0.643670 | 0.683015 / 0.642527 | **0.682709 / 0.432717** | 0.715755 / 0.420539 | 0.720956 / 0.420539 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv) |

### 11/02 EMA02V2：联合标签与教师约束视图

仅2开发seed，确认阶段未触发；数值是固定EMA selected测试均值，不能据此作新规则下的性能否决。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv)。

**MovieLens**（9条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed EMA | 2（未满5） | 0.774743 / 0.584929 | 0.789599 / 0.405250 | 0.782404 / 0.573633 | 0.800817 / 0.388432 | 0.757187 / 0.580154 | 0.771098 / 0.437580 | **0.779308 / 0.494996** | 0.780512 / 0.493523 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| A1 · fixed EMA | 2（未满5） | 0.774282 / 0.585081 | 0.789491 / 0.404770 | 0.781682 / 0.574895 | 0.800313 / 0.389027 | 0.755929 / 0.580649 | 0.771174 / 0.436992 | **0.778812 / 0.495236** | 0.779996 / 0.493848 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| AG · fixed EMA | 2（未满5） | 0.775044 / 0.572843 | 0.787886 / 0.397591 | 0.782208 / 0.571244 | 0.799810 / 0.385937 | 0.757428 / 0.581687 | 0.770037 / 0.437853 | **0.778736 / 0.491193** | 0.779941 / 0.490057 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| F0 · fixed EMA | 2（未满5） | 0.773758 / 0.579919 | 0.788880 / 0.402074 | 0.782533 / 0.572209 | 0.800450 / 0.386429 | 0.756153 / 0.580527 | 0.770407 / 0.437330 | **0.778697 / 0.493081** | 0.779960 / 0.491699 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| F1 · fixed EMA | 2（未满5） | 0.774816 / 0.585085 | 0.789705 / 0.404975 | 0.782496 / 0.573341 | 0.800145 / 0.389052 | 0.758267 / 0.580145 | 0.772570 / 0.436772 | **0.779667 / 0.494895** | 0.780811 / 0.493448 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| J0 · fixed EMA | 2（未满5） | 0.773415 / 0.586885 | 0.788766 / 0.406400 | 0.782070 / 0.575182 | 0.800684 / 0.389636 | 0.756567 / 0.581143 | 0.771308 / 0.437320 | **0.778801 / 0.496094** | 0.780052 / 0.494635 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| J1 · fixed EMA | 2（未满5） | 0.774368 / 0.583966 | 0.789500 / 0.404497 | 0.782059 / 0.574253 | 0.800946 / 0.387637 | 0.756556 / 0.579925 | 0.771099 / 0.436883 | **0.779088 / 0.494527** | 0.780304 / 0.493106 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| RG · fixed EMA | 2（未满5） | 0.773603 / 0.584034 | 0.789264 / 0.403366 | 0.781806 / 0.572953 | 0.800041 / 0.387197 | 0.755735 / 0.580100 | 0.770519 / 0.436486 | **0.778495 / 0.494023** | 0.779709 / 0.492564 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| TG · fixed EMA | 2（未满5） | 0.770950 / 0.577288 | 0.784967 / 0.397351 | 0.781286 / 0.569384 | 0.799238 / 0.382720 | 0.755772 / 0.578910 | 0.769126 / 0.435405 | **0.776890 / 0.490176** | 0.778264 / 0.488781 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |

**KuaiRand**（9条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed EMA | 2（未满5） | 0.640627 / 0.288398 | 0.660977 / 0.179542 | 0.704018 / 0.504487 | 0.732405 / 0.345373 | 0.656823 / 0.649734 | 0.680181 / 0.644583 | **0.679172 / 0.435353** | 0.717138 / 0.424597 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| A1 · fixed EMA | 2（未满5） | 0.640779 / 0.288380 | 0.664019 / 0.180364 | 0.703169 / 0.504502 | 0.731325 / 0.345913 | 0.655107 / 0.649921 | 0.679267 / 0.644662 | **0.678944 / 0.435624** | 0.716198 / 0.424875 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| AG · fixed EMA | 2（未满5） | 0.638291 / 0.287117 | 0.646950 / 0.181285 | 0.695671 / 0.503664 | 0.723881 / 0.344831 | 0.618049 / 0.679287 | 0.646982 / 0.690123 | **0.661637 / 0.447718** | 0.708500 / 0.424210 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| F0 · fixed EMA | 2（未满5） | 0.636368 / 0.288136 | 0.664972 / 0.179567 | 0.699710 / 0.503549 | 0.728348 / 0.343970 | 0.651450 / 0.644611 | 0.675713 / 0.643338 | **0.676093 / 0.433862** | 0.712993 / 0.423423 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| F1 · fixed EMA | 2（未满5） | 0.631675 / 0.289815 | 0.651721 / 0.182507 | 0.705228 / 0.501402 | 0.733302 / 0.342993 | 0.652410 / 0.643097 | 0.676237 / 0.640866 | **0.675096 / 0.433447** | 0.718047 / 0.421898 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| J0 · fixed EMA | 2（未满5） | 0.637617 / 0.289870 | 0.664649 / 0.180453 | 0.706226 / 0.502278 | 0.734125 / 0.343786 | 0.652078 / 0.654240 | 0.676095 / 0.647209 | **0.678465 / 0.436306** | 0.719038 / 0.422772 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| J1 · fixed EMA | 2（未满5） | 0.643576 / 0.288154 | 0.653012 / 0.181726 | 0.697787 / 0.508150 | 0.725110 / 0.349449 | 0.656971 / 0.644668 | 0.677678 / 0.642702 | **0.675689 / 0.435808** | 0.710462 / 0.428382 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| RG · fixed EMA | 2（未满5） | 0.628284 / 0.292068 | 0.648797 / 0.183358 | 0.703904 / 0.502105 | 0.731700 / 0.343571 | 0.661448 / 0.644672 | 0.683418 / 0.640617 | **0.676259 / 0.434398** | 0.716638 / 0.422548 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |
| TG · fixed EMA | 2（未满5） | 0.633163 / 0.287844 | 0.652382 / 0.180534 | 0.698180 / 0.504299 | 0.726714 / 0.345233 | 0.617663 / 0.679185 | 0.643886 / 0.690131 | **0.661998 / 0.447871** | 0.711112 / 0.424718 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv) |

### 11/02–03 PAIR / ECCM：静态或动态选择与历史状态

原P00/P11含两个λ档；11月3日补充P01/P10只在λ=.05、2022/2024两seed上训练。所有数值均为固定EMA的selected测试。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md)。

**MovieLens**（8条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 | 2（未满5） | 0.774743 / 0.584929 | 0.789599 / 0.405250 | 0.782404 / 0.573633 | 0.800817 / 0.388432 | 0.757187 / 0.580154 | 0.771098 / 0.437580 | **0.779308 / 0.494996** | 0.780512 / 0.493522 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| A1 | 2（未满5） | 0.774282 / 0.585081 | 0.789491 / 0.404770 | 0.781682 / 0.574896 | 0.800313 / 0.389027 | 0.755929 / 0.580649 | 0.771174 / 0.436992 | **0.778812 / 0.495236** | 0.779996 / 0.493848 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P00 λ=0.05 | 2（未满5） | 0.773091 / 0.582147 | 0.788258 / 0.403681 | 0.780503 / 0.571618 | 0.800007 / 0.385403 | 0.755834 / 0.578729 | 0.771013 / 0.434155 | **0.778118 / 0.492622** | 0.779314 / 0.491126 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P00 λ=0.2 | 2（未满5） | 0.772172 / 0.580767 | 0.788904 / 0.400170 | 0.779526 / 0.569588 | 0.798271 / 0.381801 | 0.755383 / 0.576988 | 0.769547 / 0.432813 | **0.777300 / 0.490355** | 0.778418 / 0.488807 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P01 λ=0.05 | 2（未满5） | 0.773543 / 0.582574 | 0.788900 / 0.402964 | 0.780676 / 0.572134 | 0.799604 / 0.385732 | 0.756284 / 0.578538 | 0.770391 / 0.434970 | **0.778233 / 0.492818** | 0.779391 / 0.491358 | 0.779697 / 0.491358 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/endpoints.csv) |
| P10 λ=0.05 | 2（未满5） | 0.773978 / 0.583208 | 0.789044 / 0.404625 | 0.781104 / 0.572063 | 0.800007 / 0.386099 | 0.756758 / 0.578448 | 0.771100 / 0.435080 | **0.778665 / 0.493254** | 0.779826 / 0.491740 | 0.780084 / 0.491740 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/endpoints.csv) |
| P11 λ=0.05 | 2（未满5） | 0.773874 / 0.583332 | 0.788862 / 0.404429 | 0.780571 / 0.572296 | 0.799740 / 0.385962 | 0.756392 / 0.578505 | 0.770939 / 0.435245 | **0.778396 / 0.493295** | 0.779539 / 0.491784 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P11 λ=0.2 | 2（未满5） | 0.771303 / 0.585267 | 0.787000 / 0.403681 | 0.781209 / 0.569735 | 0.797810 / 0.384941 | 0.755681 / 0.577360 | 0.771716 / 0.432205 | **0.777453 / 0.492198** | 0.778668 / 0.490560 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |

**KuaiRand**（8条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0 | 2（未满5） | 0.640627 / 0.288397 | 0.660977 / 0.179542 | 0.704018 / 0.504487 | 0.732405 / 0.345373 | 0.656823 / 0.649734 | 0.680181 / 0.644583 | **0.679172 / 0.435353** | 0.717138 / 0.424597 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| A1 | 2（未满5） | 0.640779 / 0.288380 | 0.664019 / 0.180364 | 0.703169 / 0.504502 | 0.731325 / 0.345913 | 0.655107 / 0.649921 | 0.679267 / 0.644662 | **0.678944 / 0.435624** | 0.716198 / 0.424875 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P00 λ=0.05 | 2（未满5） | 0.636373 / 0.288168 | 0.647191 / 0.181588 | 0.701219 / 0.504405 | 0.730157 / 0.344925 | 0.652470 / 0.650670 | 0.673175 / 0.648910 | **0.673431 / 0.436444** | 0.714524 / 0.424366 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P00 λ=0.2 | 2（未满5） | 0.640750 / 0.287244 | 0.645519 / 0.181404 | 0.697916 / 0.507579 | 0.725892 / 0.348562 | 0.651671 / 0.648558 | 0.676420 / 0.643439 | **0.673028 / 0.436131** | 0.710830 / 0.427677 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P01 λ=0.05 | 2（未满5） | 0.662175 / 0.285823 | 0.671977 / 0.181569 | 0.702289 / 0.504199 | 0.730707 / 0.345147 | 0.659675 / 0.649471 | 0.686355 / 0.640693 | **0.685530 / 0.434484** | 0.715659 / 0.424327 | 0.720708 / 0.424327 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/endpoints.csv) |
| P10 λ=0.05 | 2（未满5） | 0.650441 / 0.285875 | 0.656309 / 0.180689 | 0.701206 / 0.502620 | 0.729277 / 0.344155 | 0.664093 / 0.641798 | 0.683613 / 0.639309 | **0.680823 / 0.432408** | 0.714288 / 0.423026 | 0.719567 / 0.423026 | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/supplement/endpoints.csv) |
| P11 λ=0.05 | 2（未满5） | 0.637414 / 0.288954 | 0.677992 / 0.178649 | 0.701208 / 0.506605 | 0.729657 / 0.346748 | 0.657483 / 0.647035 | 0.679699 / 0.642571 | **0.680575 / 0.435093** | 0.714482 / 0.426293 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |
| P11 λ=0.2 | 2（未满5） | 0.637406 / 0.289449 | 0.647209 / 0.182723 | 0.700664 / 0.512715 | 0.728980 / 0.351265 | 0.659096 / 0.648558 | 0.680177 / 0.645429 | **0.675589 / 0.438357** | 0.713730 / 0.431552 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v3/research_reports/2026-11-02_mdmt_information_momentum_v3/NUMERICAL_APPENDIX.md) |

### 11/03 条件信息增强、当前梯度修复与历史改造

新算法仅seed2022（n=1）；A0f/A1引入历史2022/2024两seed（n=2）。原D0停止归属于旧规则，不等于新五seed性能否决。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv)。

**MovieLens**（9条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0f | 2（未满5） | 0.774743 / 0.584929 | 0.789599 / 0.405250 | 0.782404 / 0.573633 | 0.800817 / 0.388432 | 0.757187 / 0.580154 | 0.771098 / 0.437580 | **0.779308 / 0.494996** | 0.780512 / 0.493523 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| A1 | 2（未满5） | 0.774282 / 0.585081 | 0.789491 / 0.404770 | 0.781682 / 0.574895 | 0.800313 / 0.389027 | 0.755929 / 0.580649 | 0.771174 / 0.436992 | **0.778812 / 0.495236** | 0.779996 / 0.493848 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| C_MIX | 1（未满5） | 0.773297 / 0.584777 | 0.788565 / 0.404090 | 0.780790 / 0.573698 | 0.800367 / 0.386854 | 0.755877 / 0.582347 | 0.770759 / 0.436746 | **0.778276 / 0.494752** | 0.779486 / 0.493238 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| H_MIX | 1（未满5） | 0.772708 / 0.584215 | 0.789118 / 0.402217 | 0.779671 / 0.571797 | 0.797801 / 0.385109 | 0.756456 / 0.578839 | 0.770019 / 0.434355 | **0.777629 / 0.492755** | 0.778693 / 0.491234 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| H_REAL | 1（未满5） | 0.772742 / 0.587459 | 0.789121 / 0.405632 | 0.782012 / 0.574174 | 0.800531 / 0.386669 | 0.755993 / 0.581506 | 0.770511 / 0.436999 | **0.778485 / 0.495407** | 0.779753 / 0.493792 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| M | 1（未满5） | 0.773896 / 0.584619 | 0.788051 / 0.404702 | 0.780890 / 0.571746 | 0.799037 / 0.387611 | 0.757590 / 0.577653 | 0.772186 / 0.434048 | **0.778609 / 0.493396** | 0.779728 / 0.491911 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| SCALE | 1（未满5） | 0.774714 / 0.586879 | 0.789181 / 0.405451 | 0.782361 / 0.575671 | 0.800836 / 0.389597 | 0.756530 / 0.581114 | 0.771172 / 0.437268 | **0.779132 / 0.495997** | 0.780355 / 0.494584 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| T0 | 1（未满5） | 0.772973 / 0.577454 | 0.788943 / 0.397398 | 0.778699 / 0.568166 | 0.796449 / 0.383371 | 0.755166 / 0.572110 | 0.769643 / 0.429037 | **0.776979 / 0.487922** | 0.777967 / 0.486658 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| TM | 1（未满5） | 0.771855 / 0.581456 | 0.788690 / 0.399201 | 0.779223 / 0.568598 | 0.797169 / 0.383716 | 0.758072 / 0.572209 | 0.771432 / 0.429643 | **0.777740 / 0.489137** | 0.778760 / 0.487732 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |

**KuaiRand**（8条配置记录）

| 模型/算法配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | **Macro AUC / LL** | Traffic AUC / LL | Pooled AUC / LL | 来源 |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| A0f | 2（未满5） | 0.640627 / 0.288398 | 0.660977 / 0.179542 | 0.704018 / 0.504487 | 0.732405 / 0.345373 | 0.656823 / 0.649734 | 0.680181 / 0.644583 | **0.679172 / 0.435353** | 0.717138 / 0.424597 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| A1 | 2（未满5） | 0.640779 / 0.288380 | 0.664019 / 0.180364 | 0.703169 / 0.504502 | 0.731325 / 0.345913 | 0.655107 / 0.649921 | 0.679267 / 0.644662 | **0.678944 / 0.435624** | 0.716198 / 0.424875 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| C_MIX | 1（未满5） | 0.635571 / 0.288790 | 0.674926 / 0.179286 | 0.704530 / 0.496243 | 0.732945 / 0.338373 | 0.664849 / 0.643618 | 0.685194 / 0.639103 | **0.683003 / 0.430902** | 0.717750 / 0.417069 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| H_MIX | 1（未满5） | 0.660467 / 0.284199 | 0.664962 / 0.179545 | 0.702522 / 0.498887 | 0.730354 / 0.340762 | 0.669609 / 0.637860 | 0.689807 / 0.635053 | **0.686287 / 0.429384** | 0.715604 / 0.419482 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| H_REAL | 1（未满5） | 0.622651 / 0.292937 | 0.642770 / 0.184002 | 0.706366 / 0.499457 | 0.734908 / 0.340385 | 0.657070 / 0.651854 | 0.676083 / 0.647858 | **0.673308 / 0.436082** | 0.719318 / 0.419745 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| M | 1（未满5） | 0.644841 / 0.286412 | 0.676090 / 0.178187 | 0.699211 / 0.514655 | 0.726807 / 0.336333 | 0.666005 / 0.645971 | 0.683487 / 0.638040 | **0.682740 / 0.433266** | 0.712176 / 0.425095 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| SCALE | 1（未满5） | 0.630424 / 0.290246 | 0.645741 / 0.182942 | 0.704718 / 0.498018 | 0.733427 / 0.339630 | 0.650735 / 0.649505 | 0.676253 / 0.643968 | **0.673550 / 0.434052** | 0.717814 / 0.418625 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |
| T0 | 1（未满5） | 0.641124 / 0.286915 | 0.654282 / 0.180887 | 0.701814 / 0.497140 | 0.730248 / 0.338318 | 0.648677 / 0.655267 | 0.669287 / 0.651416 | **0.674239 / 0.434991** | 0.714894 / 0.417571 | — | [指标](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv) |

## 待回填历史与使用限制

- 2026-10-31 GA31-v2及更早10月大量自研算法、共享/原型类实验，现已有跨阶段总结，但**尚未按同一规则逐个回填六格绝对AUC与LogLoss**；不能根据主摘要中的增量补造缺失的完整绝对值。后续应逐源CSV补齐，并保留每个协议的独立分组。
- 更早实验可能只存validation或只存某一指标，这些不满足本表完整测试记录标准；不得以其他实验的基线/seed填充。
- 10/29论文模型组使用V按LogLoss选点、E历史评价集，其他近期多组使用validation按Macro AUC选点、test。**当前表不是可以直接复制成论文跨组排名的最终Table**。必须先把正式发表模型与新算法放在同协议下比较。
- 本表的自研方法行是科研记录，不得因其数值较强就改名为正式baseline；允许用其作为组件消融或内部机制比较的reference，但在论文的baseline方法列表中只能使用已发表模型。
