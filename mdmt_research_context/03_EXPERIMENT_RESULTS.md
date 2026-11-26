# MDMTRec 数据对照与性能

所有配对差值均为候选减对照，AUC 正差和 LogLoss 负差分别表示各自指标改善。对比时核对数据划分、selected/common epoch、online/EMA、Macro/Traffic/Pooled、seed、参照臂和六格保护；历史实验沿用各自的权重、人口、读出和指标。缺失与未运行分别登记原状态。

## EXP080 · 条件历史共享更新的冻结 S5 数表

每行五 seed。下表是 selected Test 绝对均值，六格列按 d0t0、d0t1、d1t0、d1t1、d2t0、d2t1 排列。A00=clean+EMA，A10=匹配 PAIR Adam+EMA，A21=冻结候选；发表模型使用 online。V、SD、支持、配对统计与成本以原 CSV 为准。

| 数据 | 模型 | Macro AUC | Macro LL | 六格 AUC | 六格 LL |
| --- | --- | ---: | ---: | --- | --- |
| movielens | A00 | 0.779039 | 0.496140 | 0.774393 / 0.789598 / 0.782021 / 0.800921 / 0.756329 / 0.770970 | 0.585267 / 0.407690 / 0.574434 / 0.390560 / 0.580900 / 0.437986 |
| movielens | A10 | 0.778557 | 0.495570 | 0.773518 / 0.788648 / 0.781602 / 0.800831 / 0.756001 / 0.770739 | 0.584906 / 0.406734 / 0.573739 / 0.389644 / 0.580881 / 0.437518 |
| movielens | A20 | 0.778415 | 0.495552 | 0.773584 / 0.788645 / 0.781355 / 0.800109 / 0.756125 / 0.770672 | 0.585651 / 0.406545 / 0.574572 / 0.389436 / 0.580189 / 0.436916 |
| movielens | A30 | 0.778833 | 0.495200 | 0.774147 / 0.788934 / 0.781733 / 0.800697 / 0.756583 / 0.770902 | 0.585028 / 0.406352 / 0.573870 / 0.388840 / 0.579862 / 0.437248 |
| movielens | M3OE_FIXED1_ADAPTED | 0.775190 | 0.494891 | 0.770298 / 0.785173 / 0.778130 / 0.797449 / 0.752263 / 0.767825 | 0.584358 / 0.407449 / 0.572689 / 0.386143 / 0.581804 / 0.436902 |
| movielens | HYPERGATE | 0.771412 | 0.498554 | 0.766607 / 0.782252 / 0.774387 / 0.793831 / 0.748979 / 0.762414 | 0.596720 / 0.408779 / 0.576402 / 0.384556 / 0.587279 / 0.437587 |
| movielens | AUTOIFS | 0.772425 | 0.496094 | 0.768463 / 0.784873 / 0.773603 / 0.791790 / 0.751286 / 0.764532 | 0.587138 / 0.404390 / 0.575564 / 0.388102 / 0.584169 / 0.437202 |
| movielens | A21 | 0.778474 | 0.494491 | 0.773699 / 0.788558 / 0.781593 / 0.800763 / 0.755830 / 0.770402 | 0.584192 / 0.405088 / 0.573382 / 0.387821 / 0.579889 / 0.436573 |
| movielens | A22 | 0.778956 | 0.489623 | 0.774538 / 0.789702 / 0.782388 / 0.800462 / 0.756236 / 0.770408 | 0.579041 / 0.400519 / 0.566219 / 0.382107 / 0.576882 / 0.432968 |
| kuairand | A00 | 0.676523 | 0.435947 | 0.636991 / 0.649947 / 0.705306 / 0.733389 / 0.655677 / 0.677829 | 0.289644 / 0.182406 / 0.502458 / 0.344332 / 0.650714 / 0.646126 |
| kuairand | A10 | 0.676703 | 0.435269 | 0.638070 / 0.657511 / 0.702304 / 0.731157 / 0.653821 / 0.677358 | 0.287834 / 0.180630 / 0.503505 / 0.344375 / 0.650048 / 0.645224 |
| kuairand | A20 | 0.680609 | 0.435469 | 0.646211 / 0.658275 / 0.703972 / 0.732342 / 0.658672 / 0.684181 | 0.288999 / 0.183339 / 0.503536 / 0.344493 / 0.649815 / 0.642628 |
| kuairand | A30 | 0.679669 | 0.433447 | 0.642611 / 0.661030 / 0.701757 / 0.729962 / 0.660495 / 0.682161 | 0.287367 / 0.180512 / 0.502894 / 0.344257 / 0.644647 / 0.641007 |
| kuairand | M3OE_FIXED1_ADAPTED | 0.671907 | 0.437172 | 0.635617 / 0.645909 / 0.698833 / 0.726663 / 0.653260 / 0.671157 | 0.289725 / 0.181561 / 0.506479 / 0.346981 / 0.649400 / 0.648886 |
| kuairand | HYPERGATE | 0.682113 | 0.440590 | 0.659514 / 0.670749 / 0.699529 / 0.727857 / 0.656476 / 0.678551 | 0.289436 / 0.182902 / 0.514035 / 0.354503 / 0.654395 / 0.648268 |
| kuairand | AUTOIFS | 0.678597 | 0.432992 | 0.641729 / 0.657481 / 0.700239 / 0.727575 / 0.661152 / 0.683406 | 0.286408 / 0.179697 / 0.507256 / 0.346357 / 0.640774 / 0.637460 |
| kuairand | A21 | 0.680766 | 0.435339 | 0.645021 / 0.671626 / 0.701577 / 0.730044 / 0.657447 / 0.678879 | 0.287768 / 0.179556 / 0.505748 / 0.347183 / 0.647896 / 0.643884 |
| kuairand | A22 | 0.680906 | 0.438724 | 0.651229 / 0.664616 / 0.703714 / 0.731383 / 0.655626 / 0.678866 | 0.287007 / 0.181512 / 0.512434 / 0.352443 / 0.651899 / 0.647052 |

[main_table.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/main_table.csv) · [endpoints.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/endpoints.csv) · [paired_summary.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/paired_summary.csv) · [paired_seed.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/paired_seed.csv) · [S5_REPORT.json](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/S5_REPORT.json) · [costs.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5/costs.csv)

## EXP072 · HAT 原始五 seed 先导

自然 ML/KR，2×6×5 条 50 epoch；selected Validation fixed-EMA 差相对 CLEAN。95% t 区间按种子配对；Test、HPO 与条件消融均未触发。

| 数据 | 方法 | ΔMacro AUC | 配对 SD | 95% CI | ΔMacro LL | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ML | DT-PGNA | −.001436 | .000402 | [−.001935,−.000937] | +.004009 | −.000623 | +.000338 | −.002876 | −.001174 | −.000779 | −.003505 |
| ML | TGA-Pair | −.001353 | .000445 | [−.001906,−.000801] | +.003422 | −.000062 | −.000200 | −.002715 | −.001542 | −.000700 | −.002900 |
| ML | TGA-Example | −.001643 | .000522 | [−.002291,−.000995] | −.004180 | −.001622 | −.000690 | −.003026 | −.001289 | −.000537 | −.002695 |
| KR | DT-PGNA | −.022282 | .004103 | [−.027377,−.017188] | +.079627 | −.024104 | −.025496 | −.031235 | −.031870 | −.010357 | −.010632 |
| KR | TGA-Pair | −.021557 | .003091 | [−.025395,−.017719] | +.080400 | −.021051 | −.023004 | −.029885 | −.030579 | −.010304 | −.014519 |
| KR | TGA-Example | −.009188 | .006410 | [−.017147,−.001229] | +.002748 | −.009148 | −.007066 | −.017193 | −.017861 | −.000264 | −.003596 |

六行的 AUC 正向种子均为 0/5。CLEAN 的 ML/KR Macro AUC 为 .759768/.691098，Macro LL 为 .524690/.428640。全部六臂、逐种子、六格双指标、Traffic、Pooled、online、EMA 与阶段表在下方数据目录中。

## EXP081 · RCF-v1 与独立 SCALE

主 RCF 对 H_DIRECT/POOLED/LOCAL 的 Q 结果为 5201–5204 训练 seed。冻结历史 Test 按原协议展示，S4 未触发。SCALE 为另外四个训练 seed 2022–2025；其人口与主 RCF 按各自表解释。

| 设计与端点 | 比较 | ML ΔAUC / ΔLL | KR ΔAUC / ΔLL |
| --- | --- | --- | --- |
| RCF Q | RCF − H_DIRECT | −.00051304 / +.00024518 | +.01119579 / −.00304660 |
| RCF Q | RCF − POOLED_TARGET | −.00000381 / −.00002949 | +.00111838 / −.00033626 |
| RCF Q | RCF − LOCAL_ACTIVE | +.00005739 / −.00010966 | +.00097558 / −.00026485 |
| 冻结历史 Test | RCF − H_DIRECT | +.00010003 / −.00186226 | +.00676605 / −.00383557 |
| 冻结历史 Test | RCF − NATIVE | +.00271030 / −.00900884 | −.00750254 / +.00377258 |
| 独立 SCALE 暴露 Test | Dconst01_NO_AUG − B | +.00404890 / −.00924025 | +.00758835 / −.00655322 |

KR RCF−H_DIRECT 冻结 Test 的自然样本 LL +.00195334（4/4 恶化），Macro LL −.00383557（4/4 改善）；SCALE KR 的同一对照在 Traffic cell AUC / Pooled AUC 为 −.00451560/−.00445033。细分数表包含全部六格、配对区间、角色、预算、推理与训练成本。

## 原始数据与对照表目录

逐种子、六格、阶段、曝光与成本的高维表通过冻结路径索引。数值归属以各臂的 run/receipt 与选点字段为准。归档分支的结果访问固定提交；同一历史 CSV 或同轨迹读出只索引一次。

| 实验或记录 | 数据与性能来源 |
| --- | --- |
| EXP001–032、S 系列 | [archive_manifest.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/data/history_exp/data_archive/archive_manifest.csv) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/data/history_exp/data_archive) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/results) |
| EXP048 · 10-24 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/10-24) |
| EXP048–050 · 2026-10-25_15-56-04 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04) |
| EXP048–049 · 26-10-25 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-25) |
| EXP051 · 26-10-26 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26) |
| EXP045 · codex/conditional-paths-2026-10-19 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19) |
| EXP033–034 · codex/execute-2026-10-15 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-15) |
| EXP033–035 · codex/execute-2026-10-16 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-16) |
| EXP033–036 · codex/execute-2026-10-16-v2 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-16-v2) |
| EXP037 · codex/execute-2026-10-16-v3 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-16-v3) |
| EXP037–042 · codex/execute-2026-10-17-run | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-17-run) |
| EXP043 · codex/execute-2026-10-18 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18) |
| EXP043–044 · codex/execute-n19-2026-10-18 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18) |
| EXP047 · codex/m3oe-reproduction-2026-10-21 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21) |
| EXP046（已完成部分；自然 transfer 未闭环） · codex/natural-movielens-reassessment-2026-10-20 | [归档证据与数据目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20) |
| EXP048 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/b64b29222fcd36ca8ecbb323ccb127f7da1ccd57/research_reports/2026-10-24/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/b64b29222fcd36ca8ecbb323ccb127f7da1ccd57/research_reports/2026-10-24) |
| EXP048–049 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/619371d9c4d57f6f0b3e8a71b686498f9e5f36d4/research_reports/2026-10-25/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/619371d9c4d57f6f0b3e8a71b686498f9e5f36d4/research_reports/2026-10-25) |
| EXP048–050 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0/research_reports/2026-10-25_v3/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0/research_reports/2026-10-25_v3) |
| EXP052–053 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28) |
| EXP055 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31) |
| EXP056 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2) |
| EXP057–058 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m) |
| EXP057 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum) |
| EXP061、EXP068–071 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive) |
| EXP062 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt) |
| EXP063–064 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-10_ema_causal_attribution) |
| EXP063–064 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-11_ema_next/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/fb120946c50519ea3993bbd6a2532a2c457731fc/research_reports/2026-11-11_ema_next) |
| EXP052 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2) |
| EXP052–054 | [原报告与源表说明](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526) |
| EXP065 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3) |
| EXP066 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72/research_reports/2026-11-03_mdmt_history_information/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72/research_reports/2026-11-03_mdmt_history_information) |
| EXP067 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion) |
| EXP068 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB) |
| EXP069 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage) |
| EXP070 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history) |
| EXP071 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism) |
| EXP073 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/auxiliary_objective_attribution_20261115/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/auxiliary_objective_attribution_20261115) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/auxiliary_objective_attribution_20261115/DATA_AVAILABILITY.md) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/auxiliary_objective_attribution_20261115/NUMERICAL_APPENDIX.md) |
| EXP074 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/DATA_AVAILABILITY.md) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/NUMERICAL_APPENDIX.md) |
| EXP075 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_axis_even_views_20261116/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_axis_even_views_20261116) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_axis_even_views_20261116/DATA_AVAILABILITY.md) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_axis_even_views_20261116/NUMERICAL_APPENDIX.md) |
| EXP076 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116/DATA_AVAILABILITY.md) · [ALL_PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116/ALL_PAIRED_SUMMARY.csv) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116/NUMERICAL_APPENDIX.md) |
| EXP077 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/DATA_AVAILABILITY.md) · [ALL_PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/ALL_PAIRED_SUMMARY.csv) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/NUMERICAL_APPENDIX.md) |
| EXP078 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/DATA_AVAILABILITY.md) · [ALL_PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/ALL_PAIRED_SUMMARY.csv) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/NUMERICAL_APPENDIX.md) |
| EXP079 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117) · [分片/原始表映射](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/DATA_AVAILABILITY.md) · [SELECTED_PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/SELECTED_PAIRED_SUMMARY.csv) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/NUMERICAL_APPENDIX.md) |
| EXP080 | [报告及其证据链接](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/conditional_history_shared_update_20261122/S5) |
| EXP072 | [RESULT_TABLES.md](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/research_reports/2026-11-15_heterogeneity_aware/RESULT_TABLES.md) · [NUMERICAL_APPENDIX.md](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/research_reports/2026-11-15_heterogeneity_aware/NUMERICAL_APPENDIX.md) · [PRIMARY_PAIRED_COMPARISONS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/research_reports/2026-11-15_heterogeneity_aware/tables/PRIMARY_PAIRED_COMPARISONS.csv) · [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/artifacts/heterogeneity_aware_20261115/attempt1/tables/ENDPOINTS.csv) · [PAIRED_SUMMARY.csv](https://github.com/Caelestis1903/My-Research-Project/blob/52b48c9aab066d56fa91a060864c7c9abf12e50c/artifacts/heterogeneity_aware_20261115/attempt1/tables/PAIRED_SUMMARY.csv) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/52b48c9aab066d56fa91a060864c7c9abf12e50c/artifacts/heterogeneity_aware_20261115/attempt1/tables) |
| EXP081 | [ARTIFACT_INDEX.json](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/ARTIFACT_INDEX.json) · [S1_Q_four_seed_six_cells.csv](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables/S1_Q_four_seed_six_cells.csv) · [S2_Q_four_seed_paired_summary.csv](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables/S2_Q_four_seed_paired_summary.csv) · [S3_Q_four_seed_paired_summary.csv](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables/S3_Q_four_seed_paired_summary.csv) · [SCALE_four_seed_metrics_long.csv](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables/SCALE_four_seed_metrics_long.csv) · [SCALE_four_seed_paired_summary.csv](https://github.com/Caelestis1903/My-Research-Project/blob/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables/SCALE_four_seed_paired_summary.csv) · [原始目录](https://github.com/Caelestis1903/My-Research-Project/tree/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8/research_reports/2026-10-26_v2/tables) |

## 历次性能数表

以下保留旧台账的精度、顺序与归档层。旧运行状态按对应研究报告的最后收据解释。

## 2026-11-17 CWA-C04 选中模型可用性更新

11:21:33 通过 40 模型/1000 ACK 验收。12:02:24 独立审查发现 owner 2476891 已退出，父级确认前台会话返回 143，退出触发原因未确定。C04 选中模型不可用，完整数值、源码和 MIXED 裁定仍有效。C04 的选中状态无法用于续训或开放 Test，C05 使用新批准的源初始化。[可用性处置收据](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04/SELECTED_RAM_AVAILABILITY_UPDATE.json)。

## 2026-11-17 CWA-C04：完成权重/总质量归因完整五seed

W/S20新增+20精确C/N控制，自然ML/KR、M3oE fixed1、seeds2022–2026、50epochs/40960/FP32，1000epochs/34500有效更新。以下为各stream独立最早严格验证六格Macro AUC选中状态的五seed均值，LL同状态；全部内部匹配控制，Test未访问。

### movielens / online

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPLETION-BCE-025 | 0.756896524 | 0.525793127 | 0.761303952 | 0.584972161 | 0.767994466 | 0.445647150 | 0.755394673 | 0.575734272 | 0.754094365 | 0.489452397 | 0.751711630 | 0.578731893 | 0.750880056 | 0.480220889 | 0.756140130 | 0.526743453 | 0.756319384 | 0.526743453 |
| MASS-SCALE-025 | 0.756926043 | 0.526070054 | 0.761853057 | 0.585205243 | 0.767492332 | 0.444874144 | 0.755416065 | 0.579082606 | 0.754236232 | 0.487612138 | 0.751589335 | 0.579661223 | 0.750969239 | 0.479984966 | 0.756169298 | 0.527067675 | 0.756046215 | 0.527067675 |
| NQR-025 | 0.756484035 | 0.529409578 | 0.761175839 | 0.589657020 | 0.768368519 | 0.446261577 | 0.754691848 | 0.581141889 | 0.752467772 | 0.491045910 | 0.751620570 | 0.586872539 | 0.750579661 | 0.481478534 | 0.755686647 | 0.530461392 | 0.755313762 | 0.530461392 |
| CLEAN | 0.756771775 | 0.526718006 | 0.761064275 | 0.588792050 | 0.767772820 | 0.446321633 | 0.756266215 | 0.575765850 | 0.753313300 | 0.483251115 | 0.751474435 | 0.583447068 | 0.750739604 | 0.482730321 | 0.756023095 | 0.527607206 | 0.756036853 | 0.527607206 |

### movielens / fixed_ema

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPLETION-BCE-025 | 0.759613603 | 0.524384719 | 0.764851256 | 0.585710647 | 0.770802427 | 0.440903324 | 0.757738739 | 0.575584741 | 0.755584092 | 0.481711356 | 0.755210120 | 0.586043734 | 0.753494982 | 0.476354513 | 0.758824805 | 0.525443376 | 0.759059849 | 0.525443376 |
| MASS-SCALE-025 | 0.759590273 | 0.525693332 | 0.764588985 | 0.584684136 | 0.771347053 | 0.443791937 | 0.757701022 | 0.576115288 | 0.755544173 | 0.482380724 | 0.754865059 | 0.585424578 | 0.753495346 | 0.481763329 | 0.758784862 | 0.526803986 | 0.758960629 | 0.526803986 |
| NQR-025 | 0.759424707 | 0.526831296 | 0.764408741 | 0.586923571 | 0.770869529 | 0.443327423 | 0.757722884 | 0.577659014 | 0.755261962 | 0.485054280 | 0.754766606 | 0.588574650 | 0.753518520 | 0.479448840 | 0.758635416 | 0.527949597 | 0.758758348 | 0.527949597 |
| CLEAN | 0.759768224 | 0.524690036 | 0.764868194 | 0.583579787 | 0.771048488 | 0.443893090 | 0.758079599 | 0.575001886 | 0.755676531 | 0.481984797 | 0.755200181 | 0.582950996 | 0.753736352 | 0.480729658 | 0.758980646 | 0.525744512 | 0.759305864 | 0.525744512 |

### kuairand / online

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPLETION-BCE-025 | 0.691806753 | 0.425075676 | 0.678507460 | 0.291387022 | 0.697823769 | 0.152002161 | 0.697513538 | 0.492112055 | 0.724279958 | 0.332911076 | 0.674499361 | 0.639975322 | 0.678216430 | 0.642066424 | 0.710398886 | 0.412142551 | 0.715739394 | 0.412142551 |
| MASS-SCALE-025 | 0.684866073 | 0.428263923 | 0.665960318 | 0.291211433 | 0.682736510 | 0.152955269 | 0.698267805 | 0.492087094 | 0.724879705 | 0.336626364 | 0.666073522 | 0.646007178 | 0.671278578 | 0.650696199 | 0.710858864 | 0.414012397 | 0.716180164 | 0.414012397 |
| NQR-025 | 0.688602225 | 0.427971383 | 0.674939243 | 0.291064666 | 0.698335824 | 0.151743231 | 0.696301713 | 0.495421881 | 0.722889958 | 0.333051818 | 0.666446680 | 0.646424892 | 0.672699933 | 0.650121811 | 0.709055233 | 0.413886816 | 0.714331095 | 0.413886816 |
| CLEAN | 0.689454214 | 0.428829030 | 0.681188461 | 0.288003149 | 0.693289414 | 0.152070322 | 0.697674196 | 0.497960326 | 0.724213660 | 0.341292122 | 0.666760981 | 0.645865515 | 0.673598571 | 0.647782743 | 0.710389859 | 0.419153398 | 0.715620036 | 0.419153398 |

### kuairand / fixed_ema

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPLETION-BCE-025 | 0.693653127 | 0.424697238 | 0.675079022 | 0.291985423 | 0.695392658 | 0.151935592 | 0.702263318 | 0.492343673 | 0.729274116 | 0.332479797 | 0.677054659 | 0.639543692 | 0.682854988 | 0.639895249 | 0.715177732 | 0.412037816 | 0.720393932 | 0.412037816 |
| MASS-SCALE-025 | 0.687954420 | 0.428318275 | 0.664025754 | 0.291311084 | 0.685093488 | 0.152651928 | 0.703315407 | 0.496471424 | 0.730155354 | 0.340348363 | 0.670960031 | 0.642442760 | 0.674176487 | 0.646684090 | 0.715958017 | 0.417963326 | 0.721118047 | 0.417963326 |
| NQR-025 | 0.690769307 | 0.427617542 | 0.670693978 | 0.293086454 | 0.689627461 | 0.153141739 | 0.702738591 | 0.492992157 | 0.729578452 | 0.333308235 | 0.673022891 | 0.645825033 | 0.678954469 | 0.647351631 | 0.715476857 | 0.412825789 | 0.720667578 | 0.412825789 |
| CLEAN | 0.691098041 | 0.428639572 | 0.678220395 | 0.288468642 | 0.694874489 | 0.151441503 | 0.702408201 | 0.497338739 | 0.729577994 | 0.341355598 | 0.668583110 | 0.645034147 | 0.672924057 | 0.648198802 | 0.715344511 | 0.418876986 | 0.720389492 | 0.418876986 |

主判定MIXED，无champion：W EMA−C EMA AUC ML−.000154621（SD.000454841，2/5正，CI[−.000719381,+.000410138]），KR+.002555086（SD.004445801，3/5正，CI[−.002965105,+.008075276]）。两者均未到+.003；最小cell均值ML−.000340860通过、KRd0t0−.003141373失败−.001保护。

对应Macro LL：ML−.000305316（3/5降低，CI跨0）；KR−.003942334（5/5降低，CI[−.005845996,−.002038672]）。同m的W−S在KR Macro LL−.003621037、两数据集task1 LL−.002988932/−.005124581，均5/5降低且CI全负（MEDIUM）。W 同时改变 class、cell 和 task 权重，W−S 衡量同总质量下三者联合分配的效果。

ML N−W online selected AUC−.000412489、5/5负、CI全负，EMA selected排序不稳定但LL+.002446577、5/5增加、CI全正。ML N−W E1 AUC+.019277577到E50−.001496855，E50 LL+.007416875，相关方向均5/5且CI不跨0；各差值按对应阶段和读出解释，排名必要性与 Adam/EMA 子中介仍未解决。KR E50 W−C AUC+.003242111、5/5正、CI全正；主裁定采用 selected 端点。

KR d0/d1/d2 AUC−.001311602/−.000224380/+.009201240，d2五seed正、CI全正；d0 LL+.002005435、最大d1 LL−.006935433均5/5且CI不跨0。两较小域相反，规模主因及统一小域收益/大域代价仍WEAK。traffic/pooled排序与Macro分开；旧TVγ.20、C1-IID2、C3-LATE弱条件反例和复用非独立性保留。

[完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/REPORT.md)、[逐seed及区间](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/ALL_PAIRED_SUMMARY.csv)、[原数与89分片](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/DATA_AVAILABILITY.md)。source48 d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f；parent已验收40RAM选中状态191017920B。旧attempt1不计新完成信用，2次NVML暂停、140返回重放/未提交calls及2缺失terminal单列。C04真实审查/发布后，执行唯一追加C05并最终交付停止，不启动C06。

## 2026-11-16 Temporal NQR / NQR-C03：完整五seed验证

M3oE fixed1，ML/KR自然人口，50epochs/40960/FP32，seeds2022–2026；EARLY10/LATE10共20条新增、CLEAN/constant-NQR各10条精确数值复用，1000epoch/34500新Adam/EMA更新，20物理attempt全完成、0失败。以下全部是内部匹配控制，selected为各stream独立最早严格验证Macro AUC最大值；非新正式模型baseline，Test封存。

### movielens / online

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQR-EARLY10 | 0.756477946 | 0.524736714 | 0.759568064 | 0.586440265 | 0.768351131 | 0.444166555 | 0.755092481 | 0.575568193 | 0.753117893 | 0.482230360 | 0.751893294 | 0.579910520 | 0.750844812 | 0.480104387 | 0.755753512 | 0.525628302 | 0.755681020 | 0.525628302 |
| NQR-LATE10 | 0.756771775 | 0.526718006 | 0.761064275 | 0.588792050 | 0.767772820 | 0.446321633 | 0.756266215 | 0.575765850 | 0.753313300 | 0.483251115 | 0.751474435 | 0.583447068 | 0.750739604 | 0.482730321 | 0.756023095 | 0.527607206 | 0.756036853 | 0.527607206 |
| NQR-025 | 0.756484035 | 0.529409578 | 0.761175839 | 0.589657020 | 0.768368519 | 0.446261577 | 0.754691848 | 0.581141889 | 0.752467772 | 0.491045910 | 0.751620570 | 0.586872539 | 0.750579661 | 0.481478534 | 0.755686647 | 0.530461392 | 0.755313762 | 0.530461392 |
| CLEAN | 0.756771775 | 0.526718006 | 0.761064275 | 0.588792050 | 0.767772820 | 0.446321633 | 0.756266215 | 0.575765850 | 0.753313300 | 0.483251115 | 0.751474435 | 0.583447068 | 0.750739604 | 0.482730321 | 0.756023095 | 0.527607206 | 0.756036853 | 0.527607206 |

### movielens / fixed_ema

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQR-EARLY10 | 0.759428179 | 0.523761485 | 0.764484868 | 0.582808363 | 0.770902553 | 0.442713587 | 0.757771955 | 0.574188518 | 0.755296450 | 0.481943389 | 0.754665033 | 0.581575958 | 0.753448214 | 0.479339092 | 0.758632924 | 0.524811441 | 0.758858921 | 0.524811441 |
| NQR-LATE10 | 0.759768224 | 0.524690036 | 0.764868194 | 0.583579787 | 0.771048488 | 0.443893090 | 0.758079599 | 0.575001886 | 0.755676531 | 0.481984797 | 0.755200181 | 0.582950996 | 0.753736352 | 0.480729658 | 0.758980646 | 0.525744512 | 0.759305864 | 0.525744512 |
| NQR-025 | 0.759424707 | 0.526831296 | 0.764408741 | 0.586923571 | 0.770869529 | 0.443327423 | 0.757722884 | 0.577659014 | 0.755261962 | 0.485054280 | 0.754766606 | 0.588574650 | 0.753518520 | 0.479448840 | 0.758635416 | 0.527949597 | 0.758758348 | 0.527949597 |
| CLEAN | 0.759768224 | 0.524690036 | 0.764868194 | 0.583579787 | 0.771048488 | 0.443893090 | 0.758079599 | 0.575001886 | 0.755676531 | 0.481984797 | 0.755200181 | 0.582950996 | 0.753736352 | 0.480729658 | 0.758980646 | 0.525744512 | 0.759305864 | 0.525744512 |

### kuairand / online

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQR-EARLY10 | 0.689002280 | 0.429915443 | 0.668960743 | 0.290731400 | 0.698832211 | 0.152070385 | 0.696770828 | 0.506009629 | 0.723597900 | 0.345496813 | 0.671931399 | 0.639950344 | 0.673920600 | 0.645234090 | 0.709628333 | 0.425153133 | 0.714725471 | 0.425153133 |
| NQR-LATE10 | 0.690090873 | 0.429169737 | 0.679812182 | 0.288343557 | 0.696847809 | 0.151702044 | 0.697620286 | 0.499588052 | 0.724357457 | 0.339967984 | 0.667538655 | 0.646523984 | 0.674368848 | 0.648892800 | 0.710451391 | 0.419308796 | 0.715718371 | 0.419308796 |
| NQR-025 | 0.688602225 | 0.427971383 | 0.674939243 | 0.291064666 | 0.698335824 | 0.151743231 | 0.696301713 | 0.495421881 | 0.722889958 | 0.333051818 | 0.666446680 | 0.646424892 | 0.672699933 | 0.650121811 | 0.709055233 | 0.413886816 | 0.714331095 | 0.413886816 |
| CLEAN | 0.689454214 | 0.428829030 | 0.681188461 | 0.288003149 | 0.693289414 | 0.152070322 | 0.697674196 | 0.497960326 | 0.724213660 | 0.341292122 | 0.666760981 | 0.645865515 | 0.673598571 | 0.647782743 | 0.710389859 | 0.419153398 | 0.715620036 | 0.419153398 |

### kuairand / fixed_ema

| Arm | macro_auc | macro_logloss | d0t0_auc | d0t0_logloss | d0t1_auc | d0t1_logloss | d1t0_auc | d1t0_logloss | d1t1_auc | d1t1_logloss | d2t0_auc | d2t0_logloss | d2t1_auc | d2t1_logloss | sample_weighted_cell_auc | sample_weighted_cell_logloss | pooled_task_mean_auc | pooled_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQR-EARLY10 | 0.691029428 | 0.427710667 | 0.672171005 | 0.289025345 | 0.693924514 | 0.151544310 | 0.701455711 | 0.498241654 | 0.728442737 | 0.341561357 | 0.674576670 | 0.640176491 | 0.675605932 | 0.645714844 | 0.714313826 | 0.419397860 | 0.719432201 | 0.419397860 |
| NQR-LATE10 | 0.691836182 | 0.428448574 | 0.678749990 | 0.289177989 | 0.696369382 | 0.151529582 | 0.702454277 | 0.496493768 | 0.729613454 | 0.339307483 | 0.669784596 | 0.645667754 | 0.674045391 | 0.648514871 | 0.715403944 | 0.417464373 | 0.720454060 | 0.417464373 |
| NQR-025 | 0.690769307 | 0.427617542 | 0.670693978 | 0.293086454 | 0.689627461 | 0.153141739 | 0.702738591 | 0.492992157 | 0.729578452 | 0.333308235 | 0.673022891 | 0.645825033 | 0.678954469 | 0.647351631 | 0.715476857 | 0.412825789 | 0.720667578 | 0.412825789 |
| CLEAN | 0.691098041 | 0.428639572 | 0.678220395 | 0.288468642 | 0.694874489 | 0.151441503 | 0.702408201 | 0.497338739 | 0.729577994 | 0.341355598 | 0.668583110 | 0.645034147 | 0.672924057 | 0.648198802 | 0.715344511 | 0.418876986 | 0.720389492 | 0.418876986 |

主增量EARLY selected EMA−CLEAN Macro AUC：ML−.000340045（0/5正，CI全负），KR−.000068613（3/5正，CI跨0）；NO_CONTINUATION_SIGNAL，无champion。ML EARLY−constant selected LL−.003069812、5/5降低、CI全负，但AUC几乎不变。LATE ML选中状态全部等于干预前CLEAN最优，不能解释为时序收益；E50 EARLY−LATE的四项AUC CI均跨0。完整逐seed/SD/CI/阶段/六格：[REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/REPORT.md)、[PAIRED](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/ALL_PAIRED_SEEDS.csv)、[数值来源](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/DATA_AVAILABILITY.md)。

## 2026-11-16 Natural quadratic ranking / cycle02：完整五seed验证

M3oE fixed1，自然数据，50epochs/batch40960/FP32；30条新增、10条精确CLEAN历史对照。Online/EMA独立选点；候选NQR-025，两个控制为BRIER-OPPORTUNITY-025与MEAN-GAP-025。完整报告和所有seed/state/阶段见research_reports/natural_quadratic_ranking_20261116。

### movielens / online

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
|---|---|---|---|---|---|---|---|---|
| NQR-025 | 5 | 0.761175839 / 0.589657020 | 0.768368519 / 0.446261577 | 0.754691848 / 0.581141889 | 0.752467772 / 0.491045910 | 0.751620570 / 0.586872539 | 0.750579661 / 0.481478534 | 0.756484035 / 0.529409578 |
| BRIER-OPPORTUNITY-025 | 5 | 0.762243455 / 0.586416235 | 0.768343696 / 0.446006151 | 0.755589295 / 0.578815192 | 0.753464597 / 0.487068493 | 0.751286475 / 0.583239993 | 0.750889188 / 0.480063719 | 0.756969451 / 0.526934964 |
| MEAN-GAP-025 | 5 | 0.761753937 / 0.589554816 | 0.768140454 / 0.447441443 | 0.755815829 / 0.578476998 | 0.753021165 / 0.493394083 | 0.751154535 / 0.584806313 | 0.749735727 / 0.483647461 | 0.756603608 / 0.529553519 |
| CLEAN | 5 | 0.761064275 / 0.588792050 | 0.767772820 / 0.446321633 | 0.756266215 / 0.575765850 | 0.753313300 / 0.483251115 | 0.751474435 / 0.583447068 | 0.750739604 / 0.482730321 | 0.756771775 / 0.526718006 |

### movielens / fixed_ema

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
|---|---|---|---|---|---|---|---|---|
| NQR-025 | 5 | 0.764408741 / 0.586923571 | 0.770869529 / 0.443327423 | 0.757722884 / 0.577659014 | 0.755261962 / 0.485054280 | 0.754766606 / 0.588574650 | 0.753518520 / 0.479448840 | 0.759424707 / 0.526831296 |
| BRIER-OPPORTUNITY-025 | 5 | 0.764773099 / 0.586974495 | 0.770982647 / 0.445952728 | 0.757860314 / 0.576448445 | 0.755577164 / 0.484691687 | 0.754785510 / 0.588014339 | 0.753425390 / 0.482873532 | 0.759567354 / 0.527492538 |
| MEAN-GAP-025 | 5 | 0.764577736 / 0.589355246 | 0.770670461 / 0.444758084 | 0.757837697 / 0.579697605 | 0.755435292 / 0.486181759 | 0.754684363 / 0.591551687 | 0.753310161 / 0.481644976 | 0.759419285 / 0.528864893 |
| CLEAN | 5 | 0.764868194 / 0.583579787 | 0.771048488 / 0.443893090 | 0.758079599 / 0.575001886 | 0.755676531 / 0.481984797 | 0.755200181 / 0.582950996 | 0.753736352 / 0.480729658 | 0.759768224 / 0.524690036 |

### kuairand / online

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
|---|---|---|---|---|---|---|---|---|
| NQR-025 | 5 | 0.674939243 / 0.291064666 | 0.698335824 / 0.151743231 | 0.696301713 / 0.495421881 | 0.722889958 / 0.333051818 | 0.666446680 / 0.646424892 | 0.672699933 / 0.650121811 | 0.688602225 / 0.427971383 |
| BRIER-OPPORTUNITY-025 | 5 | 0.669517527 / 0.290629246 | 0.695756446 / 0.152578676 | 0.698156245 / 0.499893876 | 0.724110881 / 0.342464311 | 0.669864438 / 0.642521723 | 0.670743725 / 0.648168866 | 0.688024877 / 0.429376116 |
| MEAN-GAP-025 | 5 | 0.663347873 / 0.295726336 | 0.698892587 / 0.152522167 | 0.694986985 / 0.501560554 | 0.722097601 / 0.339015258 | 0.669978050 / 0.646268736 | 0.674582504 / 0.649779428 | 0.687314267 / 0.430812080 |
| CLEAN | 5 | 0.681188461 / 0.288003149 | 0.693289414 / 0.152070322 | 0.697674196 / 0.497960326 | 0.724213660 / 0.341292122 | 0.666760981 / 0.645865515 | 0.673598571 / 0.647782743 | 0.689454214 / 0.428829030 |

### kuairand / fixed_ema

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
|---|---|---|---|---|---|---|---|---|
| NQR-025 | 5 | 0.670693978 / 0.293086454 | 0.689627461 / 0.153141739 | 0.702738591 / 0.492992157 | 0.729578452 / 0.333308235 | 0.673022891 / 0.645825033 | 0.678954469 / 0.647351631 | 0.690769307 / 0.427617542 |
| BRIER-OPPORTUNITY-025 | 5 | 0.665144659 / 0.292451904 | 0.696769855 / 0.152278076 | 0.703968632 / 0.497677196 | 0.730285814 / 0.340369527 | 0.672498854 / 0.643516010 | 0.672057183 / 0.649618706 | 0.690120833 / 0.429318570 |
| MEAN-GAP-025 | 5 | 0.661825476 / 0.295035118 | 0.690181168 / 0.152968536 | 0.703202673 / 0.494150464 | 0.730222067 / 0.334215777 | 0.673657283 / 0.646891580 | 0.676588675 / 0.649998535 | 0.689279557 / 0.428876668 |
| CLEAN | 5 | 0.678220395 / 0.288468642 | 0.694874489 / 0.151441503 | 0.702408201 / 0.497338739 | 0.729577994 / 0.341355598 | 0.668583110 / 0.645034147 | 0.672924057 / 0.648198802 | 0.691098041 / 0.428639572 |

NQR selected EMA−CLEAN：ML ΔAUC−0.000343517（0/5正，CI[−0.000547122,−0.000139912]）、ΔLL+0.002141261（1/5降低）；KR ΔAUC−0.000328734（2/5正，CI[−0.006717219,+0.006059751]）、ΔLL−0.001022030（3/5降低，CI跨零）。两主均值非正，NO_CONTINUATION_SIGNAL，无champion/Test。两个机制AUC对照在两数据集的四个区间均跨零。ML E10 EMA AUC+0.001151311且5/5正，selected反向；KR最大d1 LL−0.006196973且5/5降低，不能用聚合/阶段替换主指标。


## 2026-11-16 Task-axis even views：完整五seed验证记录

M3oE fixed1，自然人口，50epochs/batch40960/FP32；30条新增、20条CLEAN/NDP-A历史对照。Online/EMA独立选点；注册候选仅AXIS-EVEN。完整数字见research_reports/task_axis_even_views_20261116。

### movielens / online

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AXIS-IID2 | 5 | 0.762245258 / 0.585249712 | 0.769139093 / 0.443994070 | 0.754788926 / 0.577636163 | 0.752758837 / 0.483842345 | 0.752107365 / 0.583115054 | 0.750919499 / 0.481321172 | 0.756993163 / 0.525859753 |
| AXIS-EVEN | 5 | 0.760623144 / 0.587753199 | 0.769386287 / 0.443361710 | 0.755205614 / 0.576529819 | 0.753640053 / 0.482931746 | 0.750748838 / 0.584542044 | 0.750665055 / 0.481834389 | 0.756711498 / 0.526158818 |
| ISOTROPIC-EVEN | 5 | 0.761133111 / 0.584817412 | 0.768753941 / 0.443554003 | 0.754925314 / 0.575384030 | 0.753175542 / 0.483222410 | 0.751289757 / 0.580342448 | 0.751019547 / 0.480234969 | 0.756716202 / 0.524592545 |
| CLEAN | 5 | 0.761064275 / 0.588792050 | 0.767772820 / 0.446321633 | 0.756266215 / 0.575765850 | 0.753313300 / 0.483251115 | 0.751474435 / 0.583447068 | 0.750739604 / 0.482730321 | 0.756771775 / 0.526718006 |
| NDP-A | 5 | 0.760958491 / 0.586166423 | 0.769227487 / 0.441847485 | 0.755228586 / 0.575058484 | 0.753408160 / 0.480596962 | 0.752609796 / 0.578694749 | 0.751355898 / 0.478369645 | 0.757131403 / 0.523455625 |

### movielens / fixed_ema

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AXIS-IID2 | 5 | 0.764318846 / 0.583891002 | 0.771466264 / 0.443153843 | 0.757768461 / 0.574960309 | 0.755727721 / 0.481515512 | 0.754993051 / 0.584090865 | 0.753770732 / 0.480091711 | 0.759674179 / 0.524617207 |
| AXIS-EVEN | 5 | 0.764060021 / 0.586236727 | 0.771035420 / 0.444311736 | 0.757624830 / 0.576021124 | 0.755717313 / 0.481588316 | 0.754751991 / 0.586221492 | 0.753994651 / 0.481349930 | 0.759530704 / 0.525954887 |
| ISOTROPIC-EVEN | 5 | 0.764480384 / 0.584627208 | 0.771286133 / 0.444165444 | 0.757835140 / 0.575685812 | 0.755467509 / 0.482628274 | 0.754847718 / 0.584721191 | 0.753607322 / 0.481260681 | 0.759587367 / 0.525514768 |
| CLEAN | 5 | 0.764868194 / 0.583579787 | 0.771048488 / 0.443893090 | 0.758079599 / 0.575001886 | 0.755676531 / 0.481984797 | 0.755200181 / 0.582950996 | 0.753736352 / 0.480729658 | 0.759768224 / 0.524690036 |
| NDP-A | 5 | 0.764628547 / 0.585320339 | 0.771341473 / 0.443229271 | 0.757830548 / 0.575656435 | 0.755595073 / 0.481924857 | 0.755270300 / 0.584454106 | 0.753335867 / 0.480870575 | 0.759666968 / 0.525242597 |

### kuairand / online

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISOTROPIC-EVEN | 5 | 0.660665371 / 0.293071021 | 0.672580257 / 0.155128440 | 0.699351855 / 0.503990413 | 0.725765722 / 0.347702535 | 0.668720120 / 0.647763041 | 0.675457230 / 0.648458257 | 0.683756759 / 0.432685618 |
| AXIS-EVEN | 5 | 0.662285514 / 0.292077067 | 0.695043391 / 0.151956022 | 0.697968819 / 0.500756704 | 0.724580807 / 0.343539794 | 0.667018598 / 0.644405145 | 0.671861966 / 0.646456074 | 0.686459849 / 0.429865134 |
| AXIS-IID2 | 5 | 0.672079598 / 0.290326255 | 0.695569202 / 0.151841556 | 0.696034281 / 0.504507175 | 0.722310668 / 0.347308331 | 0.675583134 / 0.638432729 | 0.676865261 / 0.643744068 | 0.689740357 / 0.429360019 |
| CLEAN | 5 | 0.681188461 / 0.288003149 | 0.693289414 / 0.152070322 | 0.697674196 / 0.497960326 | 0.724213660 / 0.341292122 | 0.666760981 / 0.645865515 | 0.673598571 / 0.647782743 | 0.689454214 / 0.428829030 |
| NDP-A | 5 | 0.662259827 / 0.292787160 | 0.682109052 / 0.153502304 | 0.696290078 / 0.499010788 | 0.723355201 / 0.342837116 | 0.671518682 / 0.640627796 | 0.672964028 / 0.645884888 | 0.684749478 / 0.429108342 |

### kuairand / fixed_ema

| arm | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC / LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISOTROPIC-EVEN | 5 | 0.660271219 / 0.292577787 | 0.670086476 / 0.154235542 | 0.702376574 / 0.499615770 | 0.728984810 / 0.343565899 | 0.672039342 / 0.642894440 | 0.675978282 / 0.646075304 | 0.684956117 / 0.429827457 |
| AXIS-EVEN | 5 | 0.663599410 / 0.291076574 | 0.679810922 / 0.152980787 | 0.702280939 / 0.498636190 | 0.729202398 / 0.342664586 | 0.673106592 / 0.640554680 | 0.677343726 / 0.643608191 | 0.687557331 / 0.428253502 |
| AXIS-IID2 | 5 | 0.675823457 / 0.288892330 | 0.694371224 / 0.151794852 | 0.702594627 / 0.499023736 | 0.729389976 / 0.343158011 | 0.675108908 / 0.640622959 | 0.676917367 / 0.645098731 | 0.692367593 / 0.428098436 |
| CLEAN | 5 | 0.678220395 / 0.288468642 | 0.694874489 / 0.151441503 | 0.702408201 / 0.497338739 | 0.729577994 / 0.341355598 | 0.668583110 / 0.645034147 | 0.672924057 / 0.648198802 | 0.691098041 / 0.428639572 |
| NDP-A | 5 | 0.660261545 / 0.291632526 | 0.673449985 / 0.153504605 | 0.701097007 / 0.497669879 | 0.727885629 / 0.341990378 | 0.673666450 / 0.640595481 | 0.675575716 / 0.645169453 | 0.685322722 / 0.428427054 |

## 2026-11-16 TV-GRM：完整五seed验证记录

M3oE fixed1，自然人口，50epochs/batch40960/FP32，seeds2022–2026。50条新增+20条历史CLEAN/NDP-A复用；所有行是validation、独立online/EMA选点。本组为内部方法/匹配控制，不作跨协议论文baseline排名；Test未开启。[完整分析](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md)，[逐seed/六格/总体原值](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/artifacts/26-11-16_augmentation_gradient_momentum/attempt1/tables/ENDPOINTS.csv)，[五seed均值](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/SELECTED_CELL_MEANS.csv)。

### movielens / online

| arm | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC | Pooled AUC | Worst-cell AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TV-GRM-SHUFFLE-010 | 5 | 0.760320231 / 0.585605109 | 0.767948331 / 0.444023999 | 0.754776364 / 0.575979468 | 0.753125301 / 0.483315304 | 0.752436626 / 0.581918152 | 0.751125946 / 0.478814733 | 0.756622133 / 0.524942794 | 0.755900130 | 0.755785338 | 0.750974396 |
| TV-GRM-005 | 5 | 0.760093515 / 0.586591927 | 0.768754153 / 0.443048812 | 0.755460906 / 0.575711989 | 0.753725920 / 0.485188048 | 0.752278416 / 0.581531185 | 0.752129293 / 0.478265703 | 0.757073701 / 0.525056277 | 0.756364782 | 0.755506159 | 0.751754913 |
| TV-GRM-020 | 5 | 0.762476563 / 0.582672438 | 0.768941389 / 0.441536855 | 0.755323171 / 0.576542546 | 0.753612709 / 0.484127326 | 0.752642768 / 0.577109855 | 0.752138399 / 0.477198753 | 0.757522500 / 0.523197962 | 0.756738359 | 0.756136954 | 0.751851215 |
| TV-GRM-010 | 5 | 0.761799592 / 0.586270554 | 0.769237979 / 0.443176284 | 0.755474282 / 0.578506123 | 0.752821316 / 0.485368962 | 0.751744425 / 0.579891744 | 0.751371484 / 0.480696084 | 0.757074846 / 0.525651625 | 0.756261878 | 0.756248673 | 0.750657344 |
| TV-GRM-INSTANT-010 | 5 | 0.762483199 / 0.588215545 | 0.769315091 / 0.445681127 | 0.755390112 / 0.576594767 | 0.753623560 / 0.484797045 | 0.751837034 / 0.584677278 | 0.751092968 / 0.480138494 | 0.757290327 / 0.526684043 | 0.756457744 | 0.755898028 | 0.750839317 |
| CLEAN | 5 | 0.761064275 / 0.588792050 | 0.767772820 / 0.446321633 | 0.756266215 / 0.575765850 | 0.753313300 / 0.483251115 | 0.751474435 / 0.583447068 | 0.750739604 / 0.482730321 | 0.756771775 / 0.526718006 | 0.756023095 | 0.756036853 | 0.750739604 |
| NDP-A | 5 | 0.760958491 / 0.586166423 | 0.769227487 / 0.441847485 | 0.755228586 / 0.575058484 | 0.753408160 / 0.480596962 | 0.752609796 / 0.578694749 | 0.751355898 / 0.478369645 | 0.757131403 / 0.523455625 | 0.756365866 | 0.756172800 | 0.751233212 |

### movielens / fixed_ema

| arm | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC | Pooled AUC | Worst-cell AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TV-GRM-SHUFFLE-010 | 5 | 0.764091606 / 0.585475912 | 0.771044499 / 0.444145037 | 0.757997506 / 0.575315465 | 0.755529837 / 0.482305882 | 0.754949719 / 0.585138102 | 0.753393941 / 0.481920692 | 0.759501185 / 0.525716848 | 0.758723424 | 0.758891192 | 0.753393941 |
| TV-GRM-005 | 5 | 0.763862849 / 0.585665845 | 0.771003291 / 0.443829684 | 0.757863250 / 0.575277611 | 0.755637027 / 0.481538879 | 0.754796153 / 0.584973051 | 0.753473886 / 0.481022186 | 0.759439409 / 0.525384543 | 0.758668271 | 0.758914096 | 0.753473886 |
| TV-GRM-020 | 5 | 0.765309238 / 0.582721742 | 0.771545655 / 0.441971123 | 0.757868086 / 0.574097320 | 0.755377937 / 0.481845720 | 0.755712213 / 0.579509066 | 0.753720167 / 0.478411133 | 0.759922216 / 0.523092684 | 0.759109886 | 0.759341395 | 0.753720167 |
| TV-GRM-010 | 5 | 0.764961374 / 0.584513010 | 0.771765326 / 0.442838518 | 0.757698390 / 0.575177814 | 0.755620203 / 0.481622676 | 0.754812192 / 0.583704276 | 0.753378228 / 0.480967713 | 0.759705952 / 0.524804001 | 0.758873282 | 0.759100054 | 0.753378228 |
| TV-GRM-INSTANT-010 | 5 | 0.765333477 / 0.584576490 | 0.771363297 / 0.444987389 | 0.758020022 / 0.576389536 | 0.755508102 / 0.483812005 | 0.754978516 / 0.584316175 | 0.753538029 / 0.481967643 | 0.759790241 / 0.526008206 | 0.758967386 | 0.759248279 | 0.753538029 |
| CLEAN | 5 | 0.764868194 / 0.583579787 | 0.771048488 / 0.443893090 | 0.758079599 / 0.575001886 | 0.755676531 / 0.481984797 | 0.755200181 / 0.582950996 | 0.753736352 / 0.480729658 | 0.759768224 / 0.524690036 | 0.758980646 | 0.759305864 | 0.753736352 |
| NDP-A | 5 | 0.764628547 / 0.585320339 | 0.771341473 / 0.443229271 | 0.757830548 / 0.575656435 | 0.755595073 / 0.481924857 | 0.755270300 / 0.584454106 | 0.753335867 / 0.480870575 | 0.759666968 / 0.525242597 | 0.758867439 | 0.759015434 | 0.753335867 |

### kuairand / online

| arm | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC | Pooled AUC | Worst-cell AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TV-GRM-005 | 5 | 0.672036229 / 0.289695643 | 0.701703536 / 0.151100168 | 0.697350979 / 0.505788072 | 0.724111744 / 0.348462460 | 0.669232458 / 0.643973994 | 0.671886948 / 0.647440466 | 0.689386982 / 0.431076801 | 0.710179968 | 0.715296820 | 0.664622108 |
| TV-GRM-020 | 5 | 0.673566967 / 0.290848794 | 0.695803830 / 0.151939825 | 0.696531297 / 0.499834965 | 0.723155476 / 0.341633807 | 0.667661915 / 0.643515946 | 0.672205409 / 0.646491580 | 0.688154149 / 0.429044153 | 0.709279999 | 0.714527095 | 0.667250277 |
| TV-GRM-SHUFFLE-010 | 5 | 0.659932779 / 0.292057344 | 0.686747729 / 0.152247891 | 0.696318932 / 0.497771370 | 0.723138889 / 0.342527362 | 0.670752587 / 0.642526676 | 0.674312817 / 0.646834769 | 0.685200622 / 0.428994235 | 0.709064904 | 0.714391702 | 0.656907018 |
| TV-GRM-010 | 5 | 0.666327615 / 0.291163644 | 0.692293201 / 0.152162940 | 0.699126287 / 0.494341923 | 0.725420626 / 0.337901807 | 0.672715344 / 0.642115682 | 0.678114598 / 0.644491039 | 0.688999612 / 0.427029506 | 0.711648706 | 0.716826940 | 0.662185601 |
| TV-GRM-INSTANT-010 | 5 | 0.667612320 / 0.291328883 | 0.697788231 / 0.151606676 | 0.697556033 / 0.499939855 | 0.724048886 / 0.344141628 | 0.666659310 / 0.648114215 | 0.672394439 / 0.647563445 | 0.687676536 / 0.430449117 | 0.710197427 | 0.715472168 | 0.662066402 |
| CLEAN | 5 | 0.681188461 / 0.288003149 | 0.693289414 / 0.152070322 | 0.697674196 / 0.497960326 | 0.724213660 / 0.341292122 | 0.666760981 / 0.645865515 | 0.673598571 / 0.647782743 | 0.689454214 / 0.428829030 | 0.710389859 | 0.715620036 | 0.663910354 |
| NDP-A | 5 | 0.662259827 / 0.292787160 | 0.682109052 / 0.153502304 | 0.696290078 / 0.499010788 | 0.723355201 / 0.342837116 | 0.671518682 / 0.640627796 | 0.672964028 / 0.645884888 | 0.684749478 / 0.429108342 | 0.709142389 | 0.714467422 | 0.661737673 |

### kuairand / fixed_ema

| arm | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC | Pooled AUC | Worst-cell AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TV-GRM-005 | 5 | 0.669332210 / 0.290368580 | 0.693657787 / 0.151596041 | 0.703791730 / 0.496088811 | 0.730407104 / 0.340107229 | 0.673139639 / 0.641683510 | 0.674898704 / 0.647130727 | 0.690871196 / 0.427829150 | 0.716400585 | 0.721474189 | 0.664741259 |
| TV-GRM-020 | 5 | 0.672164666 / 0.290479162 | 0.697281145 / 0.151718101 | 0.703138167 / 0.499083654 | 0.730113879 / 0.342504338 | 0.666155896 / 0.646326051 | 0.669871022 / 0.650636751 | 0.689787462 / 0.430124676 | 0.715926265 | 0.720931174 | 0.666155896 |
| TV-GRM-SHUFFLE-010 | 5 | 0.662094860 / 0.291773584 | 0.688642330 / 0.152113555 | 0.702162698 / 0.498998077 | 0.729386544 / 0.342898263 | 0.666596232 / 0.646421212 | 0.672097035 / 0.649219601 | 0.686829950 / 0.430237382 | 0.714999639 | 0.720146141 | 0.658174566 |
| TV-GRM-010 | 5 | 0.669034803 / 0.290622459 | 0.693321513 / 0.152223443 | 0.704096339 / 0.497298767 | 0.730793597 / 0.341448604 | 0.672166582 / 0.642087370 | 0.676176711 / 0.645939452 | 0.690931591 / 0.428270016 | 0.716737606 | 0.721792863 | 0.664727313 |
| TV-GRM-INSTANT-010 | 5 | 0.666958792 / 0.290750566 | 0.694652854 / 0.151439596 | 0.702252977 / 0.496628116 | 0.728735202 / 0.341160542 | 0.668878356 / 0.644171729 | 0.673817460 / 0.647459112 | 0.689215940 / 0.428601610 | 0.714797259 | 0.719889821 | 0.664374980 |
| CLEAN | 5 | 0.678220395 / 0.288468642 | 0.694874489 / 0.151441503 | 0.702408201 / 0.497338739 | 0.729577994 / 0.341355598 | 0.668583110 / 0.645034147 | 0.672924057 / 0.648198802 | 0.691098041 / 0.428639572 | 0.715344511 | 0.720389492 | 0.665555177 |
| NDP-A | 5 | 0.660261545 / 0.291632526 | 0.673449985 / 0.153504605 | 0.701097007 / 0.497669879 | 0.727885629 / 0.341990378 | 0.673666450 / 0.640595481 | 0.675575716 / 0.645169453 | 0.685322722 / 0.428427054 | 0.713686923 | 0.718951928 | 0.658791862 |

---

> 2026-11-15 更新包括 11/09 PUB 迁移、11/10 固定 EMA 因果归因及 11/11 两态历史读出的 selected Test 宏指标。11/09 A0 列在诊断条目下。

## 读取与更新准则

- 性能表的合并单元格采用 `AUC / LogLoss`，包括六个域任务格及 Macro、Traffic、Pooled。selected Test/E 先根据验证集规则为每个 seed 冻结 checkpoint，再对对应测试端点取 seed 算术均值。E10、E50、common epoch 和 Validation 按各表标记读取。
- Macro 是六格等权平均，Traffic 是按评价人口或标签请求数加权的格指标，Pooled 从合并人口的预测计算，原表缺少 Pooled 时记“—”。引用时保留原聚合名称。
- `n` 按独立训练 seed 计数。同轨迹的 online、EMA、重复读出、切片和复用均共享原训练 seed。少于五个 seed 的结果用于诊断和候选筛选。正式性能否决至少需要五个有效配对训练 seed。
- 正式比较采用已发表模型，对齐数据人口、划分、选点口径和 seed，使用可比训练协议并记录各模型的配置。五 seed 配对均值须同时满足 ΔMacro AUC ≥ +0.003 和各格 ΔAUC ≥ −0.001。LogLoss 以五 seed 配对均值下降裁定，方向数用于描述。自研组件与固定 EMA 等配置用于内部参照或增量消融。跨协议绝对值分组报告，正式排名使用同协议实测。
- 完成正式实验后，用冻结的逐 seed 结果更新相应协议分组，注明来源、`n`、参数平均方式和选点规则。仅存格均值的旧报告沿用原值；缺失指标标为待回填，并从原始结果补齐。

## 发表模型资格（已核验）

| 模型 | 对应论文与发表记录 | 备注 |
|---|---|---|
| M3oE | [SIGIR 2024, DOI:10.1145/3626772.3657686](https://doi.org/10.1145/3626772.3657686) | 仓库采用M3OE_FIXED1固定融合配置，属于发表模型的适配而非完全等同论文全部AutoML设置 |
| HyperGate | [WWW 2025 Companion, DOI:10.1145/3701716.3715236](https://doi.org/10.1145/3701716.3715236) | 10/29配置中部分超参未核验为作者原始配置 |
| AutoIFS | [WSDM 2026, DOI:10.1145/3773966.3777992](https://doi.org/10.1145/3773966.3777992) | 适配后的模型，不能直接声称复现了原论文的严格指标 |

## 已回填性能矩阵

### 11/09 EMA_ADAPTIVE A0：诊断性读出（不进入Test性能矩阵）

**协议身份：** A0复用11/07既有20条轨迹，五seed池2022–2026；新增正式训练0条、Adam更新0次、Test访问0次。以下记录训练面板与验证侧的预测和动作诊断。C_persist的P_B ΔAUC为−0.000421092±0.000109264，验证侧为−0.000103584±0.000169881；C_carry的P_B为+0.000445201±0.000224756、验证侧为+0.000101319±0.000244342，但相对统一Fast动作的验证增量仅−0.000000478。A0裁定NO_ACTIONABLE_SIGNAL；A1/B/C未触发。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

### 11/09 PUB-MDMT：原始尺度更新空间迁移与真实同α梯度控制（五seed）

**协议：** FULLTRAIN_NATURAL80_10_10，ML/KR自然人口、train-only预处理、M3oE fixed1，50epoch、batch40960、FP32、coupled Adam lr.01；种子2022–2026，n=5。Validation六格Macro AUC最早严格最大值选点。以下全部为selected Test，格内AUC / 未缩放LogLoss；Traffic为测试域人数加权cell指标，Pooled为各任务内跨域合并后平均。A/F属于同一物理轨迹，A/F保持既有同seed身份。

C0/P0两任务S/D平衡，C1/P1六格S全局系数投射，C2/P2局部S/D/T平衡；G0/G1/G2重放对应C的真实逐stepα与回退掩码，并用自身梯度/历史做普通Adam。F为固定EMA内部控制；本节PUB为优化方法候选。

**完成：** 70条正式轨迹、3500epoch、120750更新；18组solver数值校准、40条2epoch配置比较。冻结niter100/tol1e-4/ridge1e-8、频率10。PUB 18975次求解无失败、276000个平衡组-step无回退。两数据PUB+EMA五seedValidation门未触发，额外组合训练0。

| dataset | kind | D0T0 | D0T1 | D1T0 | D1T1 | D2T0 | D2T1 | Macro | Traffic | Pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | A | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | 0.775190 / 0.494891 | 0.776435 / 0.493229 | 0.776760 / 0.493229 |
| movielens | F | 0.774393 / 0.585267 | 0.789598 / 0.407690 | 0.782021 / 0.574434 | 0.800921 / 0.390560 | 0.756329 / 0.580900 | 0.770970 / 0.437986 | 0.779039 / 0.496140 | 0.780253 / 0.494692 | 0.780596 / 0.494692 |
| movielens | C0 | 0.537938 / 0.695674 | 0.544920 / 0.479248 | 0.571544 / 0.695524 | 0.581379 / 0.466541 | 0.553629 / 0.678762 | 0.557474 / 0.510357 | 0.557814 / 0.587684 | 0.560304 / 0.586991 | 0.539715 / 0.586991 |
| movielens | G0 | 0.509095 / 0.696926 | 0.509817 / 0.478134 | 0.527699 / 0.693411 | 0.529892 / 0.465079 | 0.549295 / 0.679266 | 0.539436 / 0.509817 | 0.527539 / 0.587105 | 0.528252 / 0.586264 | 0.531676 / 0.586264 |
| movielens | C1 | 0.602903 / 0.689887 | 0.602729 / 0.477151 | 0.611872 / 0.686323 | 0.618099 / 0.459426 | 0.595704 / 0.672840 | 0.585205 / 0.507866 | 0.602752 / 0.582249 | 0.604038 / 0.581221 | 0.595047 / 0.581221 |
| movielens | G1 | 0.518794 / 0.695675 | 0.521572 / 0.479205 | 0.530002 / 0.693121 | 0.545866 / 0.465127 | 0.530392 / 0.679448 | 0.505523 / 0.508488 | 0.525358 / 0.586844 | 0.526849 / 0.586012 | 0.522456 / 0.586012 |
| movielens | C2 | 0.637044 / 0.695213 | 0.638867 / 0.475798 | 0.644116 / 0.689340 | 0.661801 / 0.459514 | 0.623470 / 0.672871 | 0.621599 / 0.507530 | 0.637816 / 0.583377 | 0.639406 / 0.582366 | 0.632877 / 0.582366 |
| movielens | G2 | 0.511402 / 0.696296 | 0.507050 / 0.480058 | 0.524109 / 0.692762 | 0.539286 / 0.465529 | 0.511046 / 0.678775 | 0.525557 / 0.509946 | 0.519742 / 0.587228 | 0.521339 / 0.586346 | 0.528087 / 0.586346 |
| kuairand | A | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | 0.671907 / 0.437172 | 0.711621 / 0.426397 | 0.716990 / 0.426397 |
| kuairand | F | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | 0.676523 / 0.435947 | 0.718165 / 0.423121 | 0.723348 / 0.423121 |
| kuairand | C0 | 0.531027 / 0.296352 | 0.525341 / 0.187227 | 0.606253 / 0.546103 | 0.639977 / 0.380624 | 0.554925 / 0.675696 | 0.579067 / 0.682528 | 0.572765 / 0.461422 | 0.621707 / 0.462663 | 0.631399 / 0.462663 |
| kuairand | G0 | 0.515619 / 0.296329 | 0.506316 / 0.186612 | 0.542993 / 0.580285 | 0.543413 / 0.415729 | 0.534274 / 0.679966 | 0.497594 / 0.690550 | 0.523368 / 0.474912 | 0.542664 / 0.496730 | 0.555145 / 0.496730 |
| kuairand | C1 | 0.606938 / 0.292320 | 0.619091 / 0.183975 | 0.621666 / 0.553389 | 0.653480 / 0.385567 | 0.614903 / 0.664155 | 0.626697 / 0.672380 | 0.623796 / 0.458631 | 0.637192 / 0.468550 | 0.645052 / 0.468550 |
| kuairand | G1 | 0.516917 / 0.296246 | 0.526391 / 0.187019 | 0.544915 / 0.579596 | 0.530346 / 0.415475 | 0.486375 / 0.679218 | 0.507072 / 0.691013 | 0.518669 / 0.474761 | 0.537159 / 0.496268 | 0.549851 / 0.496268 |
| kuairand | C2 | 0.603759 / 0.292666 | 0.610479 / 0.184111 | 0.619167 / 0.542135 | 0.649915 / 0.373617 | 0.612321 / 0.662561 | 0.623094 / 0.671270 | 0.619789 / 0.454393 | 0.634130 / 0.457148 | 0.642201 / 0.457148 |
| kuairand | G2 | 0.513276 / 0.296369 | 0.515078 / 0.186775 | 0.535392 / 0.581172 | 0.541277 / 0.413879 | 0.538668 / 0.678877 | 0.485879 / 0.689538 | 0.521595 / 0.474435 | 0.537889 / 0.496251 | 0.550482 / 0.496251 |

**配对结果与裁定。** C0/C1/C2相对A的ΔMacro AUC，ML=−.217376/−.172438/−.137374，KR=−.099142/−.048111/−.052118；每项均0/5正向、全部六格均值下降，LL分别平均升高ML+.092793/+.087358/+.088487、KR+.024250/+.021459/+.017221。相对F也未合格。C−G对应AUC为ML+.030275/+.077394/+.118074、KR+.049397/+.105126/+.098194，均5/5正向且逐格通过；ML C0的LL略高于G0，其余C−G平均LL下降。

C1−C0在两数据均有重复收益。C2−C1的ML宏AUC+.035064、逐格通过，但LL+.001129；KR宏AUC−.004007且逐格未通过，LL−.004238。LL与AUC按各自平均规则独立裁定。

**机制与决定。** 当前三种coupled-WD迁移不晋级，固定F保留。正的数据效用与实际WD修正后的方向有显著差异；原始α半径、共享更新变小和正则/数据比变化均须保留为解释。ML/KR同步utility/BCE诊断为50/5窗每seed，seed为独立单位。域人口固定、相对A全格下降，未呈现小域改善/大域下降的均值补偿。

[完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)；[Validation/Test selected/E10/E50与全部状态hash](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/ENDPOINTS.csv)；[全部配对原值](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/SEED_PAIRS.csv)；[均值/SD/95%CI](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/PAIRED.csv)；[逐seed六格附表](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/NUMERICAL_TABLES.md)；[机制与覆盖](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/MECHANISM_PAIRS.csv)。

---

### 11/10 EMA_CAUSAL_ATTRIBUTION：固定EMA机制复核与因果归因（五seed）

**协议身份：** 4条件（ML/KR × Clean/Gaussian D0）、每条件seed2022–2026、20条正式轨迹、1000 epoch、34,500次Adam更新。下表为本轮selected Test的F−O复核；沿用 11/06 的数据人口、seed 池与协议。

| 条件 | n | F−O Macro AUC | F−O Macro LogLoss |
|---|---:|---:|---:|
| ML Clean | 5 | +0.003848845 | +0.001248644 |
| KR Clean | 5 | +0.004616399 | −0.001225342 |
| ML Gaussian D0 | 5 | +0.003856802 | +0.005334282 |
| KR Gaussian D0 | 5 | +0.007766927 | +0.000327687 |

正LogLoss差表示F的LogLoss更高。参数/时间消融是机制归因，不另建算法绝对性能行：删除共享S的晚期验证AUC损失为+0.0037572至+0.0043627；删除N约−0.0001632至+0.0001580；删除EMBED +0.0018738至+0.0030311；删除STAR_SKIP +0.0005041至+0.0016471；删除EXPERT接近0。删除recent损失+0.0045636至+0.0049918，删除old损失+0.0007898至+0.0025160；EMBED recent×old交互四条件均为正。

固定验证面板上的q_logit/q_probability组合相对F提高约+0.0006661至+0.0026413 AUC、同时降低LogLoss；该组合的结果范围为固定验证面板。current-preconditioned单步方向局部增益约+0.0003183至+0.0004076，momentum_isotropic替代约−0.0111386至−0.0127663；这些差值衡量同状态下的单步方向效果。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

### 11/11 EMA_NEXT：两态历史读出的selected Test宏指标（五seed）

**协议身份：** 五seed池2022–2026；95个job、1415 epoch、46,280次Adam更新。发表模型端点复用既有记录，不新增基线训练。以下为selected Test Macro AUC / LogLoss；六格逐 seed 端点见来源分支压缩端点表。

| 数据集/条件 | 冻结候选 | n | 候选Test Macro AUC / LL | F Test Macro AUC / LL | O Test Macro AUC / LL |
|---|---|---:|---:|---:|---:|
| ML Clean | B_H40_all_logit | 5 | 0.77906964 / 0.49238200 | 0.77903851 / 0.49613956 | 0.77518967 / 0.49489092 |
| ML Gaussian D0 | B_H40_all_logit | 5 | 0.77945326 / 0.49311043 | 0.77935660 / 0.49582227 | 0.77549980 / 0.49048798 |
| KR Clean | B_SHARED | 5 | 0.67638413 / 0.43569164 | 0.67652300 / 0.43594666 | 0.67190660 / 0.43717200 |
| KR Gaussian D0 | B_SHARED | 5 | 0.67748978 / 0.43478968 | 0.67918018 / 0.43551785 | 0.67141325 / 0.43519017 |

| 候选−同条件F | Test ΔMacro AUC [95% CI] | 正向seed | Test ΔMacro LogLoss [95% CI] |
|---|---:|---:|---:|
| ML Clean 两态B | +0.00003113 [−0.00077027,+0.00083254] | 3/5 | −0.00375756 [−0.00627633,−0.00123880] |
| ML Gaussian 两态B | +0.00009666 [−0.00045591,+0.00064922] | 4/5 | −0.00271183 [−0.00517072,−0.00025294] |
| KR Clean B_SHARED | −0.00013886 [−0.00268706,+0.00240934] | 3/5 | −0.00025502 [−0.00356439,+0.00305435] |
| KR Gaussian B_SHARED | −0.00169040 [−0.00447632,+0.00109551] | 1/5 | −0.00072818 [−0.00246452,+0.00100816] |
| KR Clean 两态B_H277（诊断） | +0.00069911 [−0.00000416,+0.00140238] | 5/5 | −0.00074549 [−0.00169601,+0.00020504] |

**正式发表模型对照ΔMacro AUC：**

| 数据集/条件 | 候选 | 对M3oE | 对HyperGate | 对AutoIFS | 裁定 |
|---|---|---:|---:|---:|---|
| ML Clean | B_H40_all_logit | +0.00387998 | +0.00765789 | +0.00664506 | 三个对照均通过 |
| ML Gaussian D0 | B_H40_all_logit | +0.00426359 | +0.00804150 | +0.00702867 | 三个对照均通过 |
| KR Clean | B_SHARED | +0.00447754 | −0.00572842 | −0.00221287 | 仅M3oE通过 |
| KR Gaussian D0 | B_SHARED | +0.00558318 | −0.00462278 | −0.00110723 | 仅M3oE通过 |

MovieLens两态B相对F新增AUC接近0，LogLoss稳定下降；KuaiRand B_SHARED未胜F，B_H277诊断读出只保留小幅AUC候选。B 的增量以 F 为参照，整套配置的总收益以发表模型为参照。校准需用对应的校准诊断裁定。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

### 11/06 EMA_MECHANISM：固定 EMA 收益机制与未来动作价值（五seed）

**协议身份：** EMA_MECHANISM_20261106；M3oE fixed1；MovieLens/KuaiRand × clean/Gaussian D0；每个条件seed2022–2026、n=5，共20条轨迹、1000 epoch、34,500次正式Adam。所有读出独立按Validation六格等权Macro AUC最早严格最大值选点，以下绝对值均为selected Test。online为当前模型参数读出，F为固定EMA影子参数读出；二者来自**同一条训练轨迹**，不能计为独立seed或额外训练。

[完整研究报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) · [端点绝对值原表](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-07/research_reports/2026-11-06_ema_mechanism/ENDPOINT_ABSOLUTE_SUMMARY.csv) · [配对逐seed表](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-07/research_reports/2026-11-06_ema_mechanism/PAIRED_SEED.csv)

**selected Test绝对值。** 每格与总体均为AUC / LogLoss；Macro为六格等权，Traffic为人口加权cell指标，Pooled为报告中的pooled总体指标。

| dataset | condition | readout | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro | Traffic | Pooled |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|
| movielens | clean | online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | 0.775190 / 0.494891 | 0.776435 / 0.493229 | 0.776760 / 0.493229 |
| movielens | clean | F | 5 | 0.774393 / 0.585267 | 0.789598 / 0.407690 | 0.782021 / 0.574434 | 0.800921 / 0.390560 | 0.756329 / 0.580900 | 0.770970 / 0.437986 | 0.779039 / 0.496140 | 0.780253 / 0.494692 | 0.780596 / 0.494692 |
| movielens | gaussian_d0 | online | 5 | 0.771805 / 0.581339 | 0.785984 / 0.401669 | 0.777753 / 0.571200 | 0.796278 / 0.379724 | 0.752856 / 0.577586 | 0.768322 / 0.431410 | 0.775500 / 0.490488 | 0.776604 / 0.488873 | 0.776639 / 0.488873 |
| movielens | gaussian_d0 | F | 5 | 0.774415 / 0.585145 | 0.789658 / 0.407081 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437731 | 0.779357 / 0.495822 | 0.780600 / 0.494372 | 0.780950 / 0.494372 |
| kuairand | clean | online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | 0.671907 / 0.437172 | 0.711621 / 0.426397 | 0.716990 / 0.426397 |
| kuairand | clean | F | 5 | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | 0.676523 / 0.435947 | 0.718165 / 0.423121 | 0.723348 / 0.423121 |
| kuairand | gaussian_d0 | online | 5 | 0.631667 / 0.289512 | 0.644247 / 0.182590 | 0.696595 / 0.503958 | 0.724781 / 0.344476 | 0.652655 / 0.647854 | 0.678535 / 0.642751 | 0.671413 / 0.435190 | 0.709594 / 0.423905 | 0.715083 / 0.423905 |
| kuairand | gaussian_d0 | F | 5 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 | 0.679180 / 0.435518 | 0.715563 / 0.424828 | 0.720768 / 0.424828 |

**同轨迹配对增量（F−online）。** LogLoss差为正表示F的损失更高。

| dataset | condition | ΔMacro AUC | 正向seed | ΔMacro LogLoss |
|---|---|---:|---:|---:|
| MovieLens | clean | +0.003848845 | 5/5 | +0.001248644 |
| MovieLens | Gaussian D0 | +0.003856802 | 5/5 | +0.005334282 |
| KuaiRand | clean | +0.004616399 | 4/5 | −0.001225342 |
| KuaiRand | Gaussian D0 | +0.007766927 | 5/5 | +0.000327687 |

晚期epoch31–50的Validation年龄匹配online对照中，F−age-matched-online AUC依次为MovieLens clean +0.002993±0.000422、MovieLens Gaussian +0.003153±0.000325、KuaiRand clean +0.004503±0.001214、KuaiRand Gaussian +0.003746±0.000840，四项95%配对区间下界均大于0。早期epoch1–10的F−online AUC约为MovieLens −0.0030、KuaiRand −0.0203/−0.0206；晚期固定pair排序时间变化率则明显下降。EMA 的早期代价是学习滞后，晚期排序收益由年龄匹配对照确认。排序变化率记录稳定性，配对 AUC 衡量排序收益。

逐格常驻时间尺度交叉评价仅KuaiRand Gaussian为正（+0.000375±0.000331，5/5）；MovieLens clean/Gaussian与KuaiRand clean分别为−0.000047、−0.000082、−0.000152。C_design在四组的未来收益数值预测MAE均优于M2/M3，但单组动作整体增益较小；KuaiRand Gaussian d0t1共享组S的M3_direct局部净益+0.000346，M3相对C_simple的五seed增量仅+0.000048355±0.000199264，复杂指标独立必要性仍弱。Gaussian×EMA 交互方向随阶段和选点变化。ML E40 校准诊断解释了该锚点的大部分 LL 差异，selected Test 的 LL 代价按对应端点报告。

---

### 11/05 AUG_HISTORY：定向EMA、路径分组EMA及历史漂移条件增强（五seed）

**协议：** FULLTRAIN_NATURAL80_10_10 / validation六格macro AUC最早严格最大值；MovieLens/KuaiRand自然人口、train-only预处理，候选模型均为M3oE fixed1，50epoch、batch40960、FP32、普通coupled Adam；seed2022–2026、n=5。以下各表全部为test，单元格为AUC / 未缩放LogLoss；Traffic为人口加权cell指标，Pooled为跨域池化任务指标。正式模型参照沿用11/04已冻结M3oE/HyperGate/AutoIFS端点及各自预算，本节候选均为内部研究方法。

**计数：** B/A各10条物理训练，分别500epoch/17250 Adam/8000 V；C为ML六配置×五seed共30条、1500epoch/30000 Adam/3000 V。总计50条物理训练、2500epoch、64500 Adam、19000 V；B/A相同seed的online/order hash逐epoch相同，是同一父路径的精确重放，不增加独立seed。一个seed中的影子、online/EMA、selected/E10/E50也不增加n。C中的F/父online沿用B冻结端点，未新增训练。

[中文总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md) · [B分析](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/research_reports/2026-11-05_aug_history/B_STAGE_ANALYSIS.md) · [A分析](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/research_reports/2026-11-05_aug_history/A_STAGE_ANALYSIS.md) · [C分析](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/research_reports/2026-11-05_aug_history/C_STAGE_ANALYSIS.md)。全部有限HPO候选只在validation比较；下列test配置为冻结家族赢家、必要匹配控制及online，未进入test的搜索臂只见HPO/全V曲线。

**B 原始数表：** [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/artifacts/aug_history_20261105/attempt1/B/ENDPOINTS.csv)。

**B / kuairand / selected**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K20 | 5 | 0.643899 / 0.287665 | 0.662802 / 0.181028 | 0.702451 / 0.504633 | 0.730722 / 0.345708 | 0.655602 / 0.649662 | 0.679575 / 0.644434 | 0.679175 / 0.435522 | 0.715562 / 0.424837 | 0.720767 / 0.424837 |
| F | 5 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 | 0.679180 / 0.435518 | 0.715563 / 0.424828 | 0.720768 / 0.424828 |
| M_H05_K20 | 5 | 0.643919 / 0.287658 | 0.662840 / 0.181032 | 0.702451 / 0.504629 | 0.730722 / 0.345703 | 0.655600 / 0.649660 | 0.679577 / 0.644435 | 0.679185 / 0.435520 | 0.715563 / 0.424832 | 0.720767 / 0.424832 |
| M_H20_K10 | 5 | 0.643931 / 0.287654 | 0.662787 / 0.181041 | 0.702451 / 0.504622 | 0.730723 / 0.345701 | 0.655611 / 0.649658 | 0.679569 / 0.644437 | 0.679179 / 0.435519 | 0.715563 / 0.424828 | 0.720768 / 0.424828 |
| S_H05_K20 | 5 | 0.643938 / 0.287652 | 0.662799 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655606 / 0.649660 | 0.679568 / 0.644437 | 0.679181 / 0.435518 | 0.715563 / 0.424827 | 0.720768 / 0.424827 |
| U_FAST | 5 | 0.643428 / 0.287669 | 0.663443 / 0.180790 | 0.702004 / 0.503337 | 0.730281 / 0.344587 | 0.655951 / 0.648812 | 0.679780 / 0.643753 | 0.679148 / 0.434825 | 0.715130 / 0.423643 | 0.720389 / 0.423643 |
| online | 5 | 0.631667 / 0.289512 | 0.644247 / 0.182590 | 0.696595 / 0.503958 | 0.724781 / 0.344476 | 0.652655 / 0.647854 | 0.678535 / 0.642751 | 0.671413 / 0.435190 | 0.709594 / 0.423905 | 0.715083 / 0.423905 |

**B / kuairand / E10**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K20 | 5 | 0.631044 / 0.291240 | 0.622982 / 0.183682 | 0.655159 / 0.530822 | 0.683854 / 0.372060 | 0.632805 / 0.655502 | 0.657113 / 0.657492 | 0.647160 / 0.448466 | 0.668881 / 0.450741 | 0.675742 / 0.450741 |
| F | 5 | 0.630983 / 0.291259 | 0.622769 / 0.183701 | 0.655158 / 0.530831 | 0.683853 / 0.372067 | 0.632794 / 0.655464 | 0.657114 / 0.657468 | 0.647112 / 0.448465 | 0.668878 / 0.450748 | 0.675739 / 0.450748 |
| M_H05_K20 | 5 | 0.631048 / 0.291248 | 0.622824 / 0.183694 | 0.655160 / 0.530821 | 0.683855 / 0.372055 | 0.632797 / 0.655471 | 0.657112 / 0.657474 | 0.647133 / 0.448461 | 0.668881 / 0.450737 | 0.675742 / 0.450737 |
| M_H20_K10 | 5 | 0.630975 / 0.291266 | 0.622764 / 0.183704 | 0.655156 / 0.530837 | 0.683852 / 0.372073 | 0.632798 / 0.655463 | 0.657111 / 0.657468 | 0.647109 / 0.448468 | 0.668877 / 0.450754 | 0.675738 / 0.450754 |
| S_H05_K20 | 5 | 0.630976 / 0.291259 | 0.622774 / 0.183700 | 0.655157 / 0.530830 | 0.683853 / 0.372067 | 0.632796 / 0.655465 | 0.657112 / 0.657468 | 0.647111 / 0.448465 | 0.668878 / 0.450748 | 0.675739 / 0.450748 |
| U_FAST | 5 | 0.636751 / 0.288953 | 0.642016 / 0.182090 | 0.666230 / 0.523381 | 0.694819 / 0.362766 | 0.636998 / 0.653531 | 0.663671 / 0.651827 | 0.656747 / 0.443758 | 0.679871 / 0.442474 | 0.686268 / 0.442474 |
| online | 5 | 0.627341 / 0.290082 | 0.637083 / 0.182375 | 0.672664 / 0.522345 | 0.701480 / 0.359344 | 0.637727 / 0.655158 | 0.663372 / 0.651299 | 0.656611 / 0.443434 | 0.686228 / 0.440296 | 0.692446 / 0.440296 |

**B / kuairand / E50**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K20 | 5 | 0.643749 / 0.288784 | 0.663614 / 0.181775 | 0.704651 / 0.497771 | 0.732867 / 0.340140 | 0.653248 / 0.653553 | 0.675411 / 0.647733 | 0.678924 / 0.434959 | 0.717675 / 0.418770 | 0.722812 / 0.418770 |
| F | 5 | 0.643820 / 0.288782 | 0.663607 / 0.181793 | 0.704650 / 0.497767 | 0.732866 / 0.340134 | 0.653254 / 0.653551 | 0.675421 / 0.647738 | 0.678936 / 0.434961 | 0.717675 / 0.418765 | 0.722810 / 0.418765 |
| M_H05_K20 | 5 | 0.643795 / 0.288779 | 0.663598 / 0.181773 | 0.704651 / 0.497780 | 0.732868 / 0.340143 | 0.653249 / 0.653551 | 0.675409 / 0.647737 | 0.678928 / 0.434961 | 0.717675 / 0.418775 | 0.722812 / 0.418775 |
| M_H20_K10 | 5 | 0.643822 / 0.288794 | 0.663587 / 0.181804 | 0.704650 / 0.497766 | 0.732866 / 0.340134 | 0.653255 / 0.653553 | 0.675425 / 0.647740 | 0.678934 / 0.434965 | 0.717674 / 0.418764 | 0.722810 / 0.418764 |
| S_H05_K20 | 5 | 0.643834 / 0.288781 | 0.663609 / 0.181792 | 0.704651 / 0.497767 | 0.732866 / 0.340134 | 0.653249 / 0.653553 | 0.675418 / 0.647739 | 0.678938 / 0.434961 | 0.717675 / 0.418764 | 0.722810 / 0.418764 |
| U_FAST | 5 | 0.644231 / 0.288787 | 0.662686 / 0.181853 | 0.704677 / 0.497615 | 0.732912 / 0.339924 | 0.652940 / 0.652369 | 0.675111 / 0.647005 | 0.678760 / 0.434592 | 0.717705 / 0.418580 | 0.722904 / 0.418580 |
| online | 5 | 0.639495 / 0.289919 | 0.657674 / 0.182063 | 0.699995 / 0.499537 | 0.727932 / 0.340156 | 0.649876 / 0.652356 | 0.672181 / 0.648228 | 0.674526 / 0.435377 | 0.712888 / 0.419649 | 0.718238 / 0.419649 |

**B / movielens / selected**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K10 | 5 | 0.774412 / 0.585126 | 0.789659 / 0.407050 | 0.782567 / 0.573868 | 0.801392 / 0.390403 | 0.756629 / 0.580687 | 0.771482 / 0.437725 | 0.779357 / 0.495810 | 0.780600 / 0.494361 | 0.780951 / 0.494361 |
| F | 5 | 0.774415 / 0.585145 | 0.789658 / 0.407081 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437731 | 0.779357 / 0.495822 | 0.780600 / 0.494372 | 0.780950 / 0.494372 |
| M_H05_K10 | 5 | 0.774413 / 0.585138 | 0.789656 / 0.407061 | 0.782567 / 0.573872 | 0.801390 / 0.390406 | 0.756629 / 0.580687 | 0.771482 / 0.437726 | 0.779356 / 0.495815 | 0.780599 / 0.494366 | 0.780950 / 0.494366 |
| M_H05_K20 | 5 | 0.774413 / 0.585136 | 0.789659 / 0.407066 | 0.782567 / 0.573872 | 0.801391 / 0.390408 | 0.756630 / 0.580687 | 0.771482 / 0.437727 | 0.779357 / 0.495816 | 0.780600 / 0.494366 | 0.780951 / 0.494366 |
| S_H05_K10 | 5 | 0.774414 / 0.585147 | 0.789657 / 0.407082 | 0.782566 / 0.573876 | 0.801390 / 0.390413 | 0.756630 / 0.580689 | 0.771482 / 0.437730 | 0.779357 / 0.495823 | 0.780600 / 0.494373 | 0.780950 / 0.494373 |
| S_H20_K20 | 5 | 0.774413 / 0.585146 | 0.789657 / 0.407080 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580690 | 0.771482 / 0.437731 | 0.779356 / 0.495822 | 0.780599 / 0.494372 | 0.780950 / 0.494372 |
| online | 5 | 0.771805 / 0.581339 | 0.785984 / 0.401669 | 0.777753 / 0.571200 | 0.796278 / 0.379724 | 0.752856 / 0.577586 | 0.768322 / 0.431410 | 0.775500 / 0.490488 | 0.776604 / 0.488873 | 0.776639 / 0.488873 |

**B / movielens / E10**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K10 | 5 | 0.766809 / 0.578753 | 0.778384 / 0.398354 | 0.772229 / 0.571992 | 0.793732 / 0.380394 | 0.739095 / 0.583767 | 0.758684 / 0.434277 | 0.768155 / 0.491256 | 0.769574 / 0.489757 | 0.769822 / 0.489757 |
| F | 5 | 0.766809 / 0.578750 | 0.778382 / 0.398336 | 0.772229 / 0.571996 | 0.793734 / 0.380388 | 0.739093 / 0.583767 | 0.758681 / 0.434277 | 0.768155 / 0.491252 | 0.769573 / 0.489753 | 0.769822 / 0.489753 |
| M_H05_K10 | 5 | 0.766811 / 0.578734 | 0.778384 / 0.398333 | 0.772229 / 0.571989 | 0.793733 / 0.380389 | 0.739093 / 0.583766 | 0.758681 / 0.434277 | 0.768155 / 0.491248 | 0.769574 / 0.489749 | 0.769823 / 0.489749 |
| M_H05_K20 | 5 | 0.766816 / 0.578713 | 0.778390 / 0.398292 | 0.772229 / 0.571984 | 0.793735 / 0.380374 | 0.739094 / 0.583764 | 0.758681 / 0.434275 | 0.768158 / 0.491234 | 0.769576 / 0.489736 | 0.769827 / 0.489736 |
| S_H05_K10 | 5 | 0.766808 / 0.578752 | 0.778380 / 0.398339 | 0.772229 / 0.571996 | 0.793734 / 0.380388 | 0.739093 / 0.583767 | 0.758680 / 0.434277 | 0.768154 / 0.491253 | 0.769573 / 0.489754 | 0.769822 / 0.489754 |
| S_H20_K20 | 5 | 0.766810 / 0.578749 | 0.778384 / 0.398334 | 0.772229 / 0.571996 | 0.793734 / 0.380388 | 0.739093 / 0.583767 | 0.758681 / 0.434277 | 0.768155 / 0.491252 | 0.769574 / 0.489753 | 0.769823 / 0.489753 |
| online | 5 | 0.764489 / 0.583439 | 0.778598 / 0.404617 | 0.769166 / 0.577057 | 0.790967 / 0.384678 | 0.740378 / 0.586804 | 0.758764 / 0.436412 | 0.767060 / 0.495501 | 0.768286 / 0.494007 | 0.768381 / 0.494007 |

**B / movielens / E50**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| D_H05_K10 | 5 | 0.773294 / 0.600498 | 0.785620 / 0.417069 | 0.779462 / 0.583676 | 0.798515 / 0.393991 | 0.754792 / 0.594261 | 0.767833 / 0.447675 | 0.776586 / 0.506195 | 0.777800 / 0.504283 | 0.778221 / 0.504283 |
| F | 5 | 0.773296 / 0.600489 | 0.785623 / 0.417053 | 0.779461 / 0.583675 | 0.798514 / 0.393987 | 0.754792 / 0.594262 | 0.767832 / 0.447675 | 0.776587 / 0.506190 | 0.777800 / 0.504279 | 0.778222 / 0.504279 |
| M_H05_K10 | 5 | 0.773297 / 0.600499 | 0.785623 / 0.417068 | 0.779461 / 0.583679 | 0.798515 / 0.393993 | 0.754792 / 0.594262 | 0.767833 / 0.447677 | 0.776587 / 0.506196 | 0.777801 / 0.504285 | 0.778221 / 0.504285 |
| M_H05_K20 | 5 | 0.773296 / 0.600491 | 0.785623 / 0.417067 | 0.779461 / 0.583677 | 0.798515 / 0.393992 | 0.754792 / 0.594262 | 0.767833 / 0.447677 | 0.776587 / 0.506195 | 0.777800 / 0.504283 | 0.778222 / 0.504283 |
| S_H05_K10 | 5 | 0.773297 / 0.600488 | 0.785623 / 0.417052 | 0.779461 / 0.583675 | 0.798515 / 0.393987 | 0.754792 / 0.594262 | 0.767833 / 0.447675 | 0.776587 / 0.506190 | 0.777800 / 0.504279 | 0.778222 / 0.504279 |
| S_H20_K20 | 5 | 0.773297 / 0.600489 | 0.785624 / 0.417052 | 0.779460 / 0.583675 | 0.798514 / 0.393987 | 0.754791 / 0.594262 | 0.767832 / 0.447675 | 0.776587 / 0.506190 | 0.777800 / 0.504279 | 0.778222 / 0.504279 |
| online | 5 | 0.768145 / 0.593259 | 0.781741 / 0.413327 | 0.774592 / 0.578008 | 0.794514 / 0.386082 | 0.749165 / 0.592251 | 0.763679 / 0.443984 | 0.771973 / 0.501152 | 0.773202 / 0.499071 | 0.773820 / 0.499071 |

**A 原始数表：** [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/artifacts/aug_history_20261105/attempt_A1/A/ENDPOINTS.csv)。

**A / kuairand / selected**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_S_FAST | 5 | 0.645008 / 0.287148 | 0.663616 / 0.180541 | 0.702640 / 0.506049 | 0.730530 / 0.345785 | 0.653779 / 0.650253 | 0.678204 / 0.645200 | 0.678963 / 0.435829 | 0.715559 / 0.425569 | 0.720732 / 0.425569 |
| A_GLOBAL_005_K40 | 5 | 0.643324 / 0.287848 | 0.661716 / 0.181178 | 0.702272 / 0.504323 | 0.730547 / 0.345452 | 0.656247 / 0.649049 | 0.679721 / 0.644464 | 0.678971 / 0.435386 | 0.715383 / 0.424558 | 0.720632 / 0.424558 |
| A_GLOBAL_020_K20 | 5 | 0.643432 / 0.287678 | 0.662643 / 0.180903 | 0.701904 / 0.504030 | 0.730205 / 0.345109 | 0.655888 / 0.648896 | 0.679731 / 0.644029 | 0.678967 / 0.435107 | 0.715038 / 0.424241 | 0.720296 / 0.424241 |
| A_GROUP_005_K40 | 5 | 0.643380 / 0.287707 | 0.662923 / 0.180864 | 0.701906 / 0.504977 | 0.730161 / 0.345746 | 0.655901 / 0.648894 | 0.679761 / 0.644333 | 0.679005 / 0.435420 | 0.715019 / 0.425021 | 0.720262 / 0.425021 |
| F | 5 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 | 0.679180 / 0.435518 | 0.715563 / 0.424828 | 0.720768 / 0.424828 |
| U_FAST | 5 | 0.643428 / 0.287669 | 0.663443 / 0.180790 | 0.702004 / 0.503337 | 0.730281 / 0.344587 | 0.655951 / 0.648812 | 0.679780 / 0.643753 | 0.679148 / 0.434825 | 0.715130 / 0.423643 | 0.720389 / 0.423643 |
| online | 5 | 0.631667 / 0.289512 | 0.644247 / 0.182590 | 0.696595 / 0.503958 | 0.724781 / 0.344476 | 0.652655 / 0.647854 | 0.678535 / 0.642751 | 0.671413 / 0.435190 | 0.709594 / 0.423905 | 0.715083 / 0.423905 |

**A / kuairand / E10**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_S_FAST | 5 | 0.630747 / 0.293092 | 0.626637 / 0.184600 | 0.663129 / 0.530301 | 0.692025 / 0.375077 | 0.634694 / 0.656978 | 0.660432 / 0.657118 | 0.651277 / 0.449528 | 0.676844 / 0.451984 | 0.683438 / 0.451984 |
| A_GLOBAL_005_K40 | 5 | 0.634480 / 0.290297 | 0.632763 / 0.183064 | 0.659964 / 0.527250 | 0.688551 / 0.368224 | 0.635184 / 0.654447 | 0.660456 / 0.655104 | 0.651900 / 0.446398 | 0.673637 / 0.447082 | 0.680299 / 0.447082 |
| A_GLOBAL_020_K20 | 5 | 0.636701 / 0.288943 | 0.642306 / 0.182064 | 0.666432 / 0.521673 | 0.694999 / 0.361439 | 0.636943 / 0.653159 | 0.663721 / 0.651354 | 0.656850 / 0.443105 | 0.680059 / 0.440980 | 0.686458 / 0.440980 |
| A_GROUP_005_K40 | 5 | 0.633304 / 0.290912 | 0.629384 / 0.183349 | 0.659825 / 0.527823 | 0.688523 / 0.368907 | 0.634637 / 0.654756 | 0.659862 / 0.655471 | 0.650923 / 0.446870 | 0.673527 / 0.447705 | 0.680206 / 0.447705 |
| F | 5 | 0.630983 / 0.291259 | 0.622769 / 0.183701 | 0.655158 / 0.530831 | 0.683853 / 0.372067 | 0.632794 / 0.655464 | 0.657114 / 0.657468 | 0.647112 / 0.448465 | 0.668878 / 0.450748 | 0.675739 / 0.450748 |
| U_FAST | 5 | 0.636751 / 0.288953 | 0.642016 / 0.182090 | 0.666230 / 0.523381 | 0.694819 / 0.362766 | 0.636998 / 0.653531 | 0.663671 / 0.651827 | 0.656747 / 0.443758 | 0.679871 / 0.442474 | 0.686268 / 0.442474 |
| online | 5 | 0.627341 / 0.290082 | 0.637083 / 0.182375 | 0.672664 / 0.522345 | 0.701480 / 0.359344 | 0.637727 / 0.655158 | 0.663372 / 0.651299 | 0.656611 / 0.443434 | 0.686228 / 0.440296 | 0.692446 / 0.440296 |

**A / kuairand / E50**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_S_FAST | 5 | 0.643703 / 0.288491 | 0.663401 / 0.181303 | 0.705023 / 0.498666 | 0.733005 / 0.340143 | 0.652114 / 0.653582 | 0.674556 / 0.648281 | 0.678634 / 0.435078 | 0.717917 / 0.419208 | 0.723065 / 0.419208 |
| A_GLOBAL_005_K40 | 5 | 0.643752 / 0.288880 | 0.662960 / 0.181893 | 0.704663 / 0.497460 | 0.732870 / 0.339834 | 0.653003 / 0.652849 | 0.675078 / 0.647349 | 0.678721 / 0.434711 | 0.717677 / 0.418464 | 0.722849 / 0.418464 |
| A_GLOBAL_020_K20 | 5 | 0.643811 / 0.288901 | 0.663107 / 0.181928 | 0.704675 / 0.497386 | 0.732884 / 0.339787 | 0.652772 / 0.653717 | 0.674745 / 0.647976 | 0.678666 / 0.434949 | 0.717689 / 0.418410 | 0.722839 / 0.418410 |
| A_GROUP_005_K40 | 5 | 0.643630 / 0.288885 | 0.663291 / 0.181806 | 0.704756 / 0.497285 | 0.732925 / 0.339695 | 0.652848 / 0.653647 | 0.675040 / 0.647929 | 0.678748 / 0.434874 | 0.717750 / 0.418314 | 0.722900 / 0.418314 |
| F | 5 | 0.643820 / 0.288782 | 0.663607 / 0.181793 | 0.704650 / 0.497767 | 0.732866 / 0.340134 | 0.653254 / 0.653551 | 0.675421 / 0.647738 | 0.678936 / 0.434961 | 0.717675 / 0.418765 | 0.722810 / 0.418765 |
| U_FAST | 5 | 0.644231 / 0.288787 | 0.662686 / 0.181853 | 0.704677 / 0.497615 | 0.732912 / 0.339924 | 0.652940 / 0.652369 | 0.675111 / 0.647005 | 0.678760 / 0.434592 | 0.717705 / 0.418580 | 0.722904 / 0.418580 |
| online | 5 | 0.639495 / 0.289919 | 0.657674 / 0.182063 | 0.699995 / 0.499537 | 0.727932 / 0.340156 | 0.649876 / 0.652356 | 0.672181 / 0.648228 | 0.674526 / 0.435377 | 0.712888 / 0.419649 | 0.718238 / 0.419649 |

**A / movielens / selected**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_T_FAST | 5 | 0.774622 / 0.584841 | 0.789789 / 0.407673 | 0.782435 / 0.574993 | 0.801233 / 0.392779 | 0.756445 / 0.581508 | 0.771469 / 0.438511 | 0.779332 / 0.496718 | 0.780556 / 0.495381 | 0.780807 / 0.495381 |
| A_GLOBAL_020_K20 | 5 | 0.773213 / 0.584797 | 0.788791 / 0.405507 | 0.782012 / 0.572698 | 0.801042 / 0.387837 | 0.755982 / 0.579214 | 0.770732 / 0.436049 | 0.778629 / 0.494350 | 0.779911 / 0.492841 | 0.780200 / 0.492841 |
| A_GROUP_020_K20 | 5 | 0.773358 / 0.584578 | 0.789057 / 0.406528 | 0.781852 / 0.572744 | 0.800982 / 0.389597 | 0.755740 / 0.579377 | 0.770651 / 0.437232 | 0.778607 / 0.495009 | 0.779872 / 0.493534 | 0.780128 / 0.493534 |
| F | 5 | 0.774415 / 0.585145 | 0.789658 / 0.407081 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437731 | 0.779357 / 0.495822 | 0.780600 / 0.494372 | 0.780950 / 0.494372 |
| online | 5 | 0.771805 / 0.581339 | 0.785984 / 0.401669 | 0.777753 / 0.571200 | 0.796278 / 0.379724 | 0.752856 / 0.577586 | 0.768322 / 0.431410 | 0.775500 / 0.490488 | 0.776604 / 0.488873 | 0.776639 / 0.488873 |

**A / movielens / E10**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_T_FAST | 5 | 0.766911 / 0.577860 | 0.778596 / 0.396564 | 0.772415 / 0.572985 | 0.793850 / 0.379755 | 0.739090 / 0.584006 | 0.758668 / 0.434254 | 0.768255 / 0.490904 | 0.769677 / 0.489492 | 0.769943 / 0.489492 |
| A_GLOBAL_020_K20 | 5 | 0.766974 / 0.581440 | 0.779659 / 0.402570 | 0.773691 / 0.572272 | 0.794188 / 0.384253 | 0.741675 / 0.584244 | 0.760616 / 0.435469 | 0.769467 / 0.493375 | 0.770868 / 0.491827 | 0.771150 / 0.491827 |
| A_GROUP_020_K20 | 5 | 0.766816 / 0.582470 | 0.778894 / 0.406565 | 0.773214 / 0.575096 | 0.793838 / 0.389233 | 0.741045 / 0.585541 | 0.760216 / 0.438216 | 0.769004 / 0.496187 | 0.770409 / 0.494764 | 0.770709 / 0.494764 |
| F | 5 | 0.766809 / 0.578750 | 0.778382 / 0.398336 | 0.772229 / 0.571996 | 0.793734 / 0.380388 | 0.739093 / 0.583767 | 0.758681 / 0.434277 | 0.768155 / 0.491252 | 0.769573 / 0.489753 | 0.769822 / 0.489753 |
| online | 5 | 0.764489 / 0.583439 | 0.778598 / 0.404617 | 0.769166 / 0.577057 | 0.790967 / 0.384678 | 0.740378 / 0.586804 | 0.758764 / 0.436412 | 0.767060 / 0.495501 | 0.768286 / 0.494007 | 0.768381 / 0.494007 |

**A / movielens / E50**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| A_FIXED_T_FAST | 5 | 0.773257 / 0.600337 | 0.785489 / 0.417528 | 0.779672 / 0.582912 | 0.798630 / 0.395011 | 0.754853 / 0.594756 | 0.767795 / 0.448231 | 0.776616 / 0.506462 | 0.777847 / 0.504540 | 0.778265 / 0.504540 |
| A_GLOBAL_020_K20 | 5 | 0.773989 / 0.600378 | 0.786369 / 0.417238 | 0.780354 / 0.583649 | 0.799097 / 0.394899 | 0.755608 / 0.593885 | 0.768496 / 0.447869 | 0.777319 / 0.506320 | 0.778534 / 0.504444 | 0.778952 / 0.504444 |
| A_GROUP_020_K20 | 5 | 0.773888 / 0.600763 | 0.786130 / 0.417669 | 0.780051 / 0.584020 | 0.798877 / 0.394896 | 0.755429 / 0.594187 | 0.768432 / 0.447676 | 0.777135 / 0.506535 | 0.778341 / 0.504651 | 0.778727 / 0.504651 |
| F | 5 | 0.773296 / 0.600489 | 0.785623 / 0.417053 | 0.779461 / 0.583675 | 0.798514 / 0.393987 | 0.754792 / 0.594262 | 0.767832 / 0.447675 | 0.776587 / 0.506190 | 0.777800 / 0.504279 | 0.778222 / 0.504279 |
| online | 5 | 0.768145 / 0.593259 | 0.781741 / 0.413327 | 0.774592 / 0.578008 | 0.794514 / 0.386082 | 0.749165 / 0.592251 | 0.763679 / 0.443984 | 0.771973 / 0.501152 | 0.773202 / 0.499071 | 0.773820 / 0.499071 |

**C 原始数表：** [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-06/artifacts/aug_history_20261105/attempt_C1/C/ENDPOINTS.csv)。

**C / movielens / selected**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| C3_WEAK | 5 | 0.774010 / 0.586269 | 0.789794 / 0.407358 | 0.782274 / 0.574449 | 0.801256 / 0.390664 | 0.756106 / 0.581467 | 0.771083 / 0.438298 | 0.779087 / 0.496418 | 0.780332 / 0.494945 | 0.780683 / 0.494945 |
| C3_WEAK_online | 5 | 0.768996 / 0.592227 | 0.784584 / 0.407125 | 0.778494 / 0.574885 | 0.796270 / 0.388556 | 0.752152 / 0.583657 | 0.767159 / 0.438301 | 0.774609 / 0.497459 | 0.775884 / 0.495730 | 0.775610 / 0.495730 |
| C_MERGED | 5 | 0.773642 / 0.586447 | 0.789303 / 0.406808 | 0.781817 / 0.574993 | 0.800711 / 0.390573 | 0.755772 / 0.581322 | 0.770891 / 0.436937 | 0.778689 / 0.496180 | 0.779924 / 0.494755 | 0.780213 / 0.494755 |
| C_MERGED_online | 5 | 0.768591 / 0.589422 | 0.783748 / 0.407423 | 0.778182 / 0.570877 | 0.797335 / 0.386023 | 0.751000 / 0.582058 | 0.766510 / 0.435058 | 0.774228 / 0.495143 | 0.775590 / 0.493280 | 0.775382 / 0.493280 |
| C_ROTATED | 5 | 0.773835 / 0.586282 | 0.789464 / 0.407450 | 0.782286 / 0.574723 | 0.801341 / 0.390579 | 0.756229 / 0.581234 | 0.771254 / 0.437628 | 0.779068 / 0.496316 | 0.780328 / 0.494859 | 0.780628 / 0.494859 |
| C_ROTATED_online | 5 | 0.770989 / 0.581492 | 0.784743 / 0.401480 | 0.778126 / 0.569661 | 0.797719 / 0.380229 | 0.751836 / 0.580585 | 0.766743 / 0.434972 | 0.775026 / 0.491403 | 0.776293 / 0.489667 | 0.776511 / 0.489667 |

**C / movielens / E10**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| C3_WEAK | 5 | 0.766686 / 0.578819 | 0.778312 / 0.398645 | 0.772220 / 0.571905 | 0.793637 / 0.380814 | 0.739171 / 0.583628 | 0.758763 / 0.434172 | 0.768131 / 0.491330 | 0.769549 / 0.489837 | 0.769742 / 0.489837 |
| C3_WEAK_online | 5 | 0.764374 / 0.584286 | 0.778558 / 0.405213 | 0.769096 / 0.577422 | 0.790810 / 0.387344 | 0.740287 / 0.587562 | 0.759009 / 0.436679 | 0.767022 / 0.496418 | 0.768241 / 0.494993 | 0.768229 / 0.494993 |
| C_MERGED | 5 | 0.766762 / 0.578833 | 0.778409 / 0.398655 | 0.772343 / 0.571752 | 0.793656 / 0.380670 | 0.739291 / 0.583717 | 0.758825 / 0.434389 | 0.768214 / 0.491336 | 0.769631 / 0.489826 | 0.769931 / 0.489826 |
| C_MERGED_online | 5 | 0.764286 / 0.582337 | 0.779019 / 0.403783 | 0.768801 / 0.577087 | 0.790747 / 0.385897 | 0.740712 / 0.587229 | 0.759163 / 0.436134 | 0.767121 / 0.495411 | 0.768308 / 0.494021 | 0.768302 / 0.494021 |
| C_ROTATED | 5 | 0.766745 / 0.578465 | 0.778423 / 0.397792 | 0.772187 / 0.571967 | 0.793628 / 0.380051 | 0.739157 / 0.583718 | 0.758687 / 0.434224 | 0.768138 / 0.491036 | 0.769551 / 0.489547 | 0.769812 / 0.489547 |
| C_ROTATED_online | 5 | 0.764494 / 0.589168 | 0.779011 / 0.408815 | 0.768981 / 0.580226 | 0.790437 / 0.388820 | 0.740136 / 0.589730 | 0.758964 / 0.438937 | 0.767004 / 0.499283 | 0.768190 / 0.497738 | 0.768119 / 0.497738 |

**C / movielens / E50**

| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| C3_WEAK | 5 | 0.772004 / 0.602516 | 0.785606 / 0.416618 | 0.778835 / 0.584210 | 0.797332 / 0.394627 | 0.754747 / 0.594645 | 0.767115 / 0.448417 | 0.775940 / 0.506839 | 0.777127 / 0.504916 | 0.777508 / 0.504916 |
| C3_WEAK_online | 5 | 0.766448 / 0.596813 | 0.780648 / 0.413795 | 0.773997 / 0.582328 | 0.793432 / 0.389476 | 0.749190 / 0.594043 | 0.762219 / 0.447091 | 0.770989 / 0.503924 | 0.772247 / 0.501982 | 0.772889 / 0.501982 |
| C_MERGED | 5 | 0.772638 / 0.601033 | 0.785482 / 0.416123 | 0.779747 / 0.582964 | 0.797993 / 0.393881 | 0.754173 / 0.595080 | 0.768063 / 0.447480 | 0.776349 / 0.506094 | 0.777581 / 0.504153 | 0.777966 / 0.504153 |
| C_MERGED_online | 5 | 0.767782 / 0.593775 | 0.781015 / 0.413076 | 0.775640 / 0.578814 | 0.794279 / 0.388285 | 0.749060 / 0.593067 | 0.763482 / 0.444603 | 0.771876 / 0.501937 | 0.773173 / 0.499952 | 0.773515 / 0.499952 |
| C_ROTATED | 5 | 0.773038 / 0.599116 | 0.785224 / 0.416019 | 0.779707 / 0.582126 | 0.797162 / 0.394078 | 0.755065 / 0.593107 | 0.769036 / 0.445297 | 0.776538 / 0.504957 | 0.777708 / 0.503098 | 0.778142 / 0.503098 |
| C_ROTATED_online | 5 | 0.768559 / 0.594356 | 0.781739 / 0.412744 | 0.774929 / 0.579459 | 0.792592 / 0.390380 | 0.749589 / 0.593014 | 0.764564 / 0.444831 | 0.771995 / 0.502464 | 0.773133 / 0.500581 | 0.773624 / 0.500581 |

本节共96条配置×数据集×端点记录；含B/A重放的F/online及U_FAST，不将重复展示计为新增科学轨迹。

### 11/04 正式发表模型统一AUC选点及C_MIX＋P01组合（五seed）

**协议与地位：** FULLTRAIN_NATURAL80_10_10 / AUC_SELECTED_V1；原行序自然train/validation/test 80/10/10，train-only编码/标准化；ML train=800167，KR train=2006132；各模型按完整验证曲线六格等权Macro AUC**最早严格最大值**选中checkpoint，在test记录绝对AUC与未经缩放LogLoss。均为seed2022–2026、n=5。M3 fixed1、HyperGate、AutoIFS为**已发表论文模型的工程适配（正式基线）**；其中M3为fixed1 online，HyperGate保留原生门控和辅助损失（部分超参UNVERIFIED_PROPOSAL），AutoIFS有原生search10/rewind/ticket/reuse，E10/E50专指reuse阶段；各模型保留各自batch/optimizer及预算差异，并非同一超参训练或原论文严格复现。CP为自研内部候选、固定EMA/online均是**同一CP轨迹**，不属于正式Baseline。

[统一A+B完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) · [逐seed六格及模型hash](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv) · [主比较与保护裁定](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/DECISIONS.json)。A共30个发表模型端点；B共40个2×2槽位；HG、AutoIFS、CP各新增两数据×五seed共30条训练，既有M3/A0/C/P/Gaussian从冻结来源复用。

**去重声明。** 下列M3 fixed1 / online与下方11/03五seed补全的A0 · online是相同五seed端点，重复显示只为论文模型同表比较；CP的fixedEMA/online同源训练，E10/E50也是同源阶段视图。下方已记录的A0 EMA、C_MIX、P01、Gaussian原值不复制。不同协议的10/29发表模型组仍单列，不能跨组对绝对AUC排序。

**movielens / selected**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | 0.775190 / 0.494891 | 0.776435 / 0.493229 | 0.776760 / 0.493229 |
| HyperGate / online | 5 | 0.766607 / 0.596720 | 0.782252 / 0.408779 | 0.774387 / 0.576402 | 0.793831 / 0.384556 | 0.748979 / 0.587279 | 0.762414 / 0.437587 | 0.771412 / 0.498554 | 0.772652 / 0.496516 | 0.772352 / 0.496516 |
| AutoIFS / online | 5 | 0.768463 / 0.587138 | 0.784873 / 0.404390 | 0.773603 / 0.575564 | 0.791790 / 0.388102 | 0.751286 / 0.584169 | 0.764532 / 0.437202 | 0.772425 / 0.496094 | 0.773370 / 0.494603 | 0.773466 / 0.494603 |
| CP / fixed_ema | 5 | 0.772428 / 0.584291 | 0.788066 / 0.405320 | 0.780659 / 0.572490 | 0.799565 / 0.386839 | 0.755160 / 0.579633 | 0.769271 / 0.436698 | 0.777525 / 0.494212 | 0.778763 / 0.492661 | 0.779014 / 0.492661 |
| CP / online | 5 | 0.766501 / 0.587252 | 0.782767 / 0.407019 | 0.774714 / 0.576457 | 0.794790 / 0.385938 | 0.750046 / 0.581527 | 0.764578 / 0.435819 | 0.772233 / 0.495669 | 0.773474 / 0.494097 | 0.773391 / 0.494097 |

**kuairand / selected**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | 0.671907 / 0.437172 | 0.711621 / 0.426397 | 0.716990 / 0.426397 |
| HyperGate / online | 5 | 0.659514 / 0.289436 | 0.670749 / 0.182902 | 0.699529 / 0.514035 | 0.727857 / 0.354503 | 0.656476 / 0.654395 | 0.678551 / 0.648268 | 0.682113 / 0.440590 | 0.712842 / 0.433824 | 0.718065 / 0.433824 |
| AutoIFS / online | 5 | 0.641729 / 0.286408 | 0.657481 / 0.179697 | 0.700239 / 0.507256 | 0.727575 / 0.346357 | 0.661152 / 0.640774 | 0.683406 / 0.637460 | 0.678597 / 0.432992 | 0.712927 / 0.426371 | 0.718171 / 0.426371 |
| CP / fixed_ema | 5 | 0.636471 / 0.289868 | 0.662738 / 0.181925 | 0.702990 / 0.500556 | 0.731046 / 0.342186 | 0.656792 / 0.648082 | 0.678971 / 0.644412 | 0.678168 / 0.434505 | 0.715949 / 0.421115 | 0.721222 / 0.421115 |
| CP / online | 5 | 0.628329 / 0.290796 | 0.652468 / 0.182604 | 0.697804 / 0.508376 | 0.725817 / 0.347886 | 0.652751 / 0.650548 | 0.674432 / 0.648487 | 0.671933 / 0.438116 | 0.710707 / 0.427787 | 0.716152 / 0.427787 |

**movielens / E10**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.763859 / 0.589397 | 0.778840 / 0.405701 | 0.769285 / 0.581406 | 0.790380 / 0.387048 | 0.739605 / 0.590423 | 0.758746 / 0.438293 | 0.766786 / 0.498711 | 0.768013 / 0.497223 | 0.768142 / 0.497223 |
| HyperGate / online | 5 | 0.768128 / 0.602180 | 0.780439 / 0.413276 | 0.773557 / 0.579909 | 0.789842 / 0.389132 | 0.748991 / 0.597640 | 0.759091 / 0.446620 | 0.770008 / 0.504793 | 0.771102 / 0.502564 | 0.770652 / 0.502564 |
| AutoIFS / online | 5 | 0.767685 / 0.584796 | 0.783572 / 0.403829 | 0.772303 / 0.575816 | 0.791250 / 0.387377 | 0.748716 / 0.584472 | 0.763021 / 0.436785 | 0.771091 / 0.495513 | 0.772071 / 0.494086 | 0.772128 / 0.494086 |
| CP / fixed_ema | 5 | 0.765728 / 0.579586 | 0.778805 / 0.401182 | 0.771881 / 0.572036 | 0.794172 / 0.383128 | 0.740168 / 0.583689 | 0.759313 / 0.434826 | 0.768344 / 0.492408 | 0.769765 / 0.490911 | 0.769760 / 0.490911 |
| CP / online | 5 | 0.763073 / 0.590022 | 0.778523 / 0.408698 | 0.768519 / 0.581407 | 0.789400 / 0.392059 | 0.740350 / 0.588192 | 0.757969 / 0.439010 | 0.766306 / 0.499898 | 0.767494 / 0.498529 | 0.767373 / 0.498529 |

**kuairand / E10**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.625661 / 0.289602 | 0.633640 / 0.182626 | 0.673181 / 0.513764 | 0.701915 / 0.351586 | 0.636584 / 0.653748 | 0.666763 / 0.646654 | 0.656291 / 0.439664 | 0.686677 / 0.432249 | 0.692923 / 0.432249 |
| HyperGate / online | 5 | 0.650345 / 0.288240 | 0.665651 / 0.180415 | 0.693030 / 0.513922 | 0.721699 / 0.352311 | 0.656489 / 0.648025 | 0.677879 / 0.641850 | 0.677515 / 0.437460 | 0.706550 / 0.432625 | 0.712091 / 0.432625 |
| AutoIFS / online | 5 | 0.640984 / 0.286813 | 0.656374 / 0.180166 | 0.701811 / 0.510205 | 0.729433 / 0.348985 | 0.660705 / 0.640937 | 0.684164 / 0.637112 | 0.678912 / 0.434036 | 0.714602 / 0.429113 | 0.719844 / 0.429113 |
| CP / fixed_ema | 5 | 0.625846 / 0.291220 | 0.632317 / 0.183305 | 0.656214 / 0.527324 | 0.684848 / 0.369648 | 0.631519 / 0.654252 | 0.656723 / 0.655506 | 0.647911 / 0.446876 | 0.669902 / 0.447824 | 0.676647 / 0.447824 |
| CP / online | 5 | 0.631778 / 0.288835 | 0.652008 / 0.182008 | 0.672844 / 0.513159 | 0.701180 / 0.352474 | 0.636670 / 0.652949 | 0.663117 / 0.648241 | 0.659599 / 0.439611 | 0.686265 / 0.432384 | 0.692523 / 0.432384 |

**movielens / E50**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.767515 / 0.596020 | 0.780156 / 0.415624 | 0.774560 / 0.579424 | 0.792513 / 0.389752 | 0.750014 / 0.590097 | 0.763942 / 0.442993 | 0.771450 / 0.502318 | 0.772646 / 0.500338 | 0.773042 / 0.500338 |
| HyperGate / online | 5 | 0.739839 / 0.820723 | 0.753159 / 0.542413 | 0.743546 / 0.743094 | 0.757933 / 0.468014 | 0.719488 / 0.881030 | 0.721622 / 0.613754 | 0.739264 / 0.678171 | 0.740241 / 0.670414 | 0.739852 / 0.670414 |
| AutoIFS / online | 5 | 0.768494 / 0.593688 | 0.784469 / 0.411098 | 0.774883 / 0.577971 | 0.790372 / 0.394653 | 0.753541 / 0.590429 | 0.766018 / 0.443418 | 0.772963 / 0.501876 | 0.773868 / 0.500221 | 0.774129 / 0.500221 |
| CP / fixed_ema | 5 | 0.772068 / 0.599820 | 0.784944 / 0.415701 | 0.779166 / 0.580003 | 0.796954 / 0.391770 | 0.756094 / 0.589358 | 0.768670 / 0.443713 | 0.776316 / 0.503394 | 0.777482 / 0.501410 | 0.777745 / 0.501410 |
| CP / online | 5 | 0.766259 / 0.593730 | 0.779423 / 0.410196 | 0.772870 / 0.577504 | 0.792802 / 0.386195 | 0.750571 / 0.587800 | 0.764768 / 0.443045 | 0.771116 / 0.499745 | 0.772295 / 0.497789 | 0.772644 / 0.497789 |

**kuairand / E50**

| 模型/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| M3 fixed1 / online | 5 | 0.632881 / 0.290156 | 0.646121 / 0.182558 | 0.700766 / 0.512846 | 0.728977 / 0.355560 | 0.650036 / 0.653260 | 0.670354 / 0.657882 | 0.671523 / 0.442044 | 0.713679 / 0.433792 | 0.718986 / 0.433792 |
| HyperGate / online | 5 | 0.647063 / 0.294013 | 0.658735 / 0.185548 | 0.700213 / 0.516010 | 0.728622 / 0.357424 | 0.653286 / 0.659716 | 0.675059 / 0.655941 | 0.677163 / 0.444775 | 0.713402 / 0.436314 | 0.718688 / 0.436314 |
| AutoIFS / online | 5 | 0.635761 / 0.288722 | 0.651978 / 0.180016 | 0.696047 / 0.530191 | 0.722579 / 0.362734 | 0.653157 / 0.650909 | 0.677869 / 0.645001 | 0.672899 / 0.442929 | 0.708304 / 0.445755 | 0.713418 / 0.445755 |
| CP / fixed_ema | 5 | 0.634109 / 0.291130 | 0.656554 / 0.183724 | 0.704559 / 0.501611 | 0.732591 / 0.343496 | 0.657480 / 0.651065 | 0.677858 / 0.646677 | 0.677192 / 0.436284 | 0.717432 / 0.422312 | 0.722668 / 0.422312 |
| CP / online | 5 | 0.631549 / 0.293440 | 0.645483 / 0.185158 | 0.698968 / 0.509530 | 0.726730 / 0.348772 | 0.654366 / 0.650286 | 0.673095 / 0.648352 | 0.671698 / 0.439257 | 0.711708 / 0.428815 | 0.717068 / 0.428815 |

**selected研究结论。** MovieLens M3/HG/AutoIFS宏AUC分别0.775190/0.771412/0.772425（最强M3）；KuaiRand分别0.671907/0.682113/0.678597（最强HG）。CP fixedEMA宏AUC/LL分别ML 0.777525/0.494212、KR 0.678168/0.434505。对已发表M3的总AUC为ML+0.002335、KR+0.006261；与P01相比为ML−0.000890、KR−0.002441；与内部无组件固定EMA相比为ML−0.001514、KR+0.001645。ML对最强M3未达到严格+.003门，KR对最强HG为−0.003945；CP不替代P01作为排序优先组合，但其他参照下的收益依原口径保留。**正式基线主表采用本节的 AUC 选点协议。** 10/29 的 V-LogLoss 结果见其历史分组。

### 11/04 Gaussian增强领域覆盖：KuaiRand G-D0/G-D02五seed和跨数据集来源别名

**KuaiRand协议：** FULLTRAIN_NATURAL80_10_10 / AUC_SELECTED_V1，fixed1 M3oE，batch40960、50epoch、Adam lr=.01/coupled wd=1e−5、Gaussian κ=.1/λ=.5、非反馈EMA ρ=.995；test使用validation六格macro AUC严格最大且最早选点的fixedEMA。增强集合G-D0=[0]、G-D02=[0,2]各新增5seed；G-D2=[2]为旧Gaussian五seed已收录，本表沿用 G-D2 的原训练轨迹和 seed。

[完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md) · [双数据集六格](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/G_D0_TWO_DATASETS.md) · [逐seed与来源](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv)

**kuairand / fixed_ema / selected** (n=5)

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---:|---|---|---|---|---|---|---|---|---|
| G-D0 | 5 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 | 0.679180 / 0.435518 | 0.715563 / 0.424828 | 0.720768 / 0.424828 |
| G-D02 | 5 | 0.628972 / 0.290649 | 0.648625 / 0.182832 | 0.704167 / 0.503186 | 0.732477 / 0.344576 | 0.656041 / 0.650127 | 0.677342 / 0.645664 | 0.674604 / 0.436172 | 0.717108 / 0.423602 | 0.722413 / 0.423602 |
| G-D2 | 5 | 0.634346 / 0.289394 | 0.661911 / 0.180681 | 0.700652 / 0.504592 | 0.728460 / 0.345690 | 0.658848 / 0.645452 | 0.683779 / 0.639891 | 0.677999 / 0.434283 | 0.713542 / 0.424782 | 0.718855 / 0.424782 |

**MovieLens G-D0已在11/03五seed补全段收录。** 当时的`Gaussian · fixed_ema`实际上增强d0，对应本次G-D0别名，历史Gaussian五seed与M3/HG/AI各五seed本次全复用、额外训练0。ML κ=.05/λ=.5、wd=1e−4、EMAρ=.9828205985452511，六格AUC/LL依次为0.774415/0.585145、0.789658/0.407081、0.782567/0.573874、0.801390/0.390412、0.756629/0.580689、0.771482/0.437731；Macro 0.779357/0.495822，Traffic 0.780600/0.494372、Pooled 0.780950/0.494372。[ML补充与来源审计](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md)。KR旧的`Gaussian · fixed_ema`对应G-D2，也是同一旧训练轨迹。

**比较裁定：** KR D0−D2宏AUC+0.001181/LL+0.001235，D02−D0宏AUC−0.004576/LL+0.000655且d0t0平均AUC−0.014949。KR三臂均未超过最强发表模型HyperGate（Macro AUC 0.682113）；KR G-D0相对已发表M3oE的**总**AUC+0.007274、LL−0.001654且六格保护满足，而对内部固定EMA的**组件**AUC+0.002657伴随d1两格平均下降。ML G-D0相对本轮最强发表M3oE AUC+0.004167且六格保护满足，LL平均+0.000931。KR按宏AUC优先单域G-D0、按宏LL优先单域G-D2；双域G-D02不作为当前排序优先方法。

### 11/03 五 seed 补全：固定 M3oE 内部候选与控制

协议：原行序自然80/10/10、train-only预处理、fixed1 M3oE、batch40960、50epoch；Adam lr=.01、coupled wd ML1e−4/KR1e−5；EMA ρ ML=.9828205985452511/KR=.995。selected按validation六格macro AUC最早严格最大值。每行n=5，种子2022–2026。本组为内部候选评价；与10/29发表模型的不同人口、选点结果分组保留。

[报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md) · [全部逐seed端点](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv) · [选点和来源](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/RUN_MANIFEST.json)

**movielens / selected**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.774393 / 0.585267 | 0.789598 / 0.407690 | 0.782021 / 0.574434 | 0.800921 / 0.390560 | 0.756329 / 0.580900 | 0.770970 / 0.437986 | 0.779039 / 0.496140 | 0.780253 / 0.494692 | 0.780596 / 0.494692 |
| A0 · online | 5 | 0.770298 / 0.584358 | 0.785173 / 0.407449 | 0.778130 / 0.572689 | 0.797449 / 0.386143 | 0.752263 / 0.581804 | 0.767825 / 0.436902 | 0.775190 / 0.494891 | 0.776435 / 0.493229 | 0.776760 / 0.493229 |
| C_MIX · fixed_ema | 5 | 0.773723 / 0.583627 | 0.788728 / 0.405790 | 0.781541 / 0.572540 | 0.800819 / 0.388243 | 0.755579 / 0.580856 | 0.770266 / 0.437209 | 0.778443 / 0.494711 | 0.779695 / 0.493203 | 0.780026 / 0.493203 |
| C_MIX · online | 5 | 0.769473 / 0.586194 | 0.784920 / 0.407299 | 0.777789 / 0.573105 | 0.797408 / 0.382782 | 0.752293 / 0.579651 | 0.767255 / 0.435134 | 0.774857 / 0.494027 | 0.776123 / 0.492246 | 0.775992 / 0.492246 |
| Gaussian · fixed_ema | 5 | 0.774415 / 0.585145 | 0.789658 / 0.407081 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437731 | 0.779357 / 0.495822 | 0.780600 / 0.494372 | 0.780950 / 0.494372 |
| H_MIX · fixed_ema | 5 | 0.773460 / 0.584293 | 0.788845 / 0.406451 | 0.781280 / 0.571527 | 0.799377 / 0.388182 | 0.755990 / 0.580066 | 0.770543 / 0.436598 | 0.778249 / 0.494519 | 0.779428 / 0.492947 | 0.779688 / 0.492947 |
| H_MIX · online | 5 | 0.768945 / 0.585771 | 0.783536 / 0.402983 | 0.777175 / 0.570391 | 0.794533 / 0.381557 | 0.752259 / 0.579331 | 0.767828 / 0.434001 | 0.774046 / 0.492339 | 0.775219 / 0.490550 | 0.775212 / 0.490550 |
| M · fixed_ema | 5 | 0.772902 / 0.585193 | 0.788080 / 0.404613 | 0.781051 / 0.571321 | 0.799790 / 0.385944 | 0.756354 / 0.577966 | 0.770700 / 0.434357 | 0.778146 / 0.493233 | 0.779363 / 0.491641 | 0.779651 / 0.491641 |
| M · online | 5 | 0.769164 / 0.585484 | 0.783408 / 0.402466 | 0.775208 / 0.572758 | 0.795935 / 0.382348 | 0.750018 / 0.581100 | 0.765154 / 0.435035 | 0.773148 / 0.493199 | 0.774355 / 0.491526 | 0.774216 / 0.491526 |
| P01 · fixed_ema | 5 | 0.773584 / 0.585651 | 0.788645 / 0.406545 | 0.781355 / 0.574572 | 0.800109 / 0.389436 | 0.756125 / 0.580189 | 0.770672 / 0.436916 | 0.778415 / 0.495552 | 0.779625 / 0.494107 | 0.779910 / 0.494107 |
| P01 · online | 5 | 0.767114 / 0.589069 | 0.783612 / 0.405837 | 0.775183 / 0.579573 | 0.794963 / 0.386390 | 0.751085 / 0.583374 | 0.766414 / 0.437802 | 0.773062 / 0.497007 | 0.774253 / 0.495516 | 0.774012 / 0.495516 |

**movielens / E10**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.766325 / 0.579081 | 0.778064 / 0.398841 | 0.771986 / 0.571901 | 0.793544 / 0.381158 | 0.738974 / 0.583859 | 0.758696 / 0.434304 | 0.767932 / 0.491524 | 0.769357 / 0.490027 | 0.769554 / 0.490027 |
| A0 · online | 5 | 0.763859 / 0.589397 | 0.778840 / 0.405701 | 0.769285 / 0.581406 | 0.790380 / 0.387048 | 0.739605 / 0.590423 | 0.758746 / 0.438293 | 0.766786 / 0.498711 | 0.768013 / 0.497223 | 0.768142 / 0.497223 |
| C_MIX · fixed_ema | 5 | 0.766072 / 0.579712 | 0.777289 / 0.400470 | 0.771336 / 0.573107 | 0.793193 / 0.382779 | 0.738795 / 0.584670 | 0.758310 / 0.435061 | 0.767499 / 0.492633 | 0.768920 / 0.491168 | 0.769065 / 0.491168 |
| C_MIX · online | 5 | 0.763576 / 0.588819 | 0.777223 / 0.411836 | 0.769408 / 0.576579 | 0.790558 / 0.387069 | 0.738479 / 0.589997 | 0.758282 / 0.437657 | 0.766254 / 0.498660 | 0.767567 / 0.496833 | 0.767462 / 0.496833 |
| Gaussian · fixed_ema | 5 | 0.766809 / 0.578750 | 0.778382 / 0.398336 | 0.772229 / 0.571996 | 0.793734 / 0.380388 | 0.739093 / 0.583767 | 0.758681 / 0.434277 | 0.768155 / 0.491252 | 0.769573 / 0.489753 | 0.769822 / 0.489753 |
| H_MIX · fixed_ema | 5 | 0.765991 / 0.578211 | 0.777728 / 0.398223 | 0.771361 / 0.570231 | 0.793058 / 0.379773 | 0.738956 / 0.584637 | 0.758801 / 0.435681 | 0.767649 / 0.491126 | 0.769047 / 0.489521 | 0.769503 / 0.489521 |
| H_MIX · online | 5 | 0.763939 / 0.587982 | 0.778002 / 0.406132 | 0.769021 / 0.579170 | 0.789034 / 0.388816 | 0.738332 / 0.589999 | 0.757383 / 0.438974 | 0.765952 / 0.498512 | 0.767168 / 0.497030 | 0.767045 / 0.497030 |
| M · fixed_ema | 5 | 0.764366 / 0.580445 | 0.775429 / 0.400935 | 0.770156 / 0.573213 | 0.792039 / 0.381421 | 0.737001 / 0.585630 | 0.757629 / 0.435544 | 0.766103 / 0.492865 | 0.767561 / 0.491297 | 0.768028 / 0.491297 |
| M · online | 5 | 0.760991 / 0.589431 | 0.773528 / 0.413330 | 0.766405 / 0.579902 | 0.785843 / 0.389055 | 0.734892 / 0.593143 | 0.753940 / 0.441825 | 0.762600 / 0.501114 | 0.763875 / 0.499354 | 0.763305 / 0.499354 |
| P01 · fixed_ema | 5 | 0.766181 / 0.579202 | 0.779599 / 0.399946 | 0.772221 / 0.571308 | 0.794271 / 0.382189 | 0.740579 / 0.582749 | 0.759803 / 0.434380 | 0.768776 / 0.491629 | 0.770167 / 0.490128 | 0.770358 / 0.490128 |
| P01 · online | 5 | 0.763008 / 0.582803 | 0.779193 / 0.401921 | 0.769658 / 0.573752 | 0.790125 / 0.383033 | 0.740449 / 0.587252 | 0.757538 / 0.436662 | 0.766662 / 0.494237 | 0.767913 / 0.492629 | 0.767700 / 0.492629 |

**movielens / E50**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.772554 / 0.600601 | 0.784934 / 0.416505 | 0.779380 / 0.583030 | 0.797092 / 0.394504 | 0.754866 / 0.593202 | 0.768598 / 0.445874 | 0.776237 / 0.505619 | 0.777421 / 0.503750 | 0.777789 / 0.503750 |
| A0 · online | 5 | 0.767515 / 0.596020 | 0.780156 / 0.415624 | 0.774560 / 0.579424 | 0.792513 / 0.389752 | 0.750014 / 0.590097 | 0.763942 / 0.442993 | 0.771450 / 0.502318 | 0.772646 / 0.500338 | 0.773042 / 0.500338 |
| C_MIX · fixed_ema | 5 | 0.772738 / 0.598710 | 0.785179 / 0.415432 | 0.778499 / 0.582298 | 0.795707 / 0.393568 | 0.755094 / 0.592823 | 0.767847 / 0.446228 | 0.775844 / 0.504843 | 0.776930 / 0.502991 | 0.777242 / 0.502991 |
| C_MIX · online | 5 | 0.767260 / 0.595775 | 0.780794 / 0.413661 | 0.774410 / 0.577798 | 0.792738 / 0.388228 | 0.749488 / 0.591886 | 0.762883 / 0.445214 | 0.771262 / 0.502094 | 0.772470 / 0.500000 | 0.772835 / 0.500000 |
| Gaussian · fixed_ema | 5 | 0.773296 / 0.600489 | 0.785623 / 0.417053 | 0.779461 / 0.583675 | 0.798514 / 0.393987 | 0.754792 / 0.594262 | 0.767832 / 0.447675 | 0.776587 / 0.506190 | 0.777800 / 0.504279 | 0.778222 / 0.504279 |
| H_MIX · fixed_ema | 5 | 0.772331 / 0.597196 | 0.784230 / 0.415499 | 0.779714 / 0.579510 | 0.796086 / 0.391719 | 0.756090 / 0.590554 | 0.767585 / 0.445698 | 0.776006 / 0.503363 | 0.777186 / 0.501398 | 0.777508 / 0.501398 |
| H_MIX · online | 5 | 0.767641 / 0.593671 | 0.779991 / 0.409692 | 0.775664 / 0.574701 | 0.792511 / 0.384111 | 0.750519 / 0.589872 | 0.762476 / 0.442570 | 0.771467 / 0.499103 | 0.772721 / 0.496946 | 0.773067 / 0.496946 |
| M · fixed_ema | 5 | 0.773216 / 0.592442 | 0.786996 / 0.411036 | 0.779949 / 0.577090 | 0.797791 / 0.390709 | 0.757055 / 0.584733 | 0.769365 / 0.440716 | 0.777395 / 0.499454 | 0.778517 / 0.497743 | 0.778867 / 0.497743 |
| M · online | 5 | 0.768142 / 0.590313 | 0.781694 / 0.407176 | 0.774393 / 0.576221 | 0.791731 / 0.388305 | 0.750721 / 0.587560 | 0.765118 / 0.441180 | 0.771966 / 0.498459 | 0.773040 / 0.496744 | 0.773454 / 0.496744 |
| P01 · fixed_ema | 5 | 0.772373 / 0.600665 | 0.784718 / 0.416730 | 0.780575 / 0.581953 | 0.798531 / 0.392733 | 0.754478 / 0.594598 | 0.767849 / 0.446475 | 0.776421 / 0.505526 | 0.777735 / 0.503509 | 0.778069 / 0.503509 |
| P01 · online | 5 | 0.766659 / 0.594157 | 0.779100 / 0.411516 | 0.774566 / 0.578470 | 0.794413 / 0.387027 | 0.747318 / 0.593531 | 0.762201 / 0.447057 | 0.770709 / 0.501960 | 0.772091 / 0.499908 | 0.772590 / 0.499908 |

**kuairand / selected**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.636991 / 0.289644 | 0.649947 / 0.182406 | 0.705306 / 0.502458 | 0.733389 / 0.344332 | 0.655677 / 0.650714 | 0.677829 / 0.646126 | 0.676523 / 0.435947 | 0.718165 / 0.423121 | 0.723348 / 0.423121 |
| A0 · online | 5 | 0.635617 / 0.289725 | 0.645909 / 0.181561 | 0.698833 / 0.506479 | 0.726663 / 0.346981 | 0.653260 / 0.649400 | 0.671157 / 0.648886 | 0.671907 / 0.437172 | 0.711621 / 0.426397 | 0.716990 / 0.426397 |
| C_MIX · fixed_ema | 5 | 0.638380 / 0.287745 | 0.649777 / 0.181241 | 0.700634 / 0.505497 | 0.729372 / 0.346173 | 0.657113 / 0.647834 | 0.678482 / 0.643876 | 0.675626 / 0.435394 | 0.713912 / 0.425482 | 0.719246 / 0.425482 |
| C_MIX · online | 5 | 0.632326 / 0.288750 | 0.642159 / 0.182279 | 0.696407 / 0.506768 | 0.724623 / 0.346683 | 0.652772 / 0.650924 | 0.673916 / 0.646491 | 0.670367 / 0.436982 | 0.709399 / 0.426388 | 0.714869 / 0.426388 |
| Gaussian · fixed_ema | 5 | 0.634346 / 0.289394 | 0.661911 / 0.180681 | 0.700652 / 0.504592 | 0.728460 / 0.345690 | 0.658848 / 0.645452 | 0.683779 / 0.639891 | 0.677999 / 0.434283 | 0.713542 / 0.424782 | 0.718855 / 0.424782 |
| H_MIX · fixed_ema | 5 | 0.642059 / 0.287698 | 0.651314 / 0.182537 | 0.703755 / 0.501354 | 0.731662 / 0.343360 | 0.657554 / 0.648428 | 0.678570 / 0.644588 | 0.677486 / 0.434661 | 0.716598 / 0.422078 | 0.721836 / 0.422078 |
| H_MIX · online | 5 | 0.642549 / 0.288415 | 0.660061 / 0.180890 | 0.699177 / 0.503812 | 0.726902 / 0.344719 | 0.653735 / 0.649797 | 0.672725 / 0.646977 | 0.675858 / 0.435768 | 0.712025 / 0.423961 | 0.717317 / 0.423961 |
| M · fixed_ema | 5 | 0.632850 / 0.288568 | 0.651030 / 0.181106 | 0.700122 / 0.512202 | 0.728264 / 0.337881 | 0.657034 / 0.649010 | 0.677345 / 0.641828 | 0.674441 / 0.435099 | 0.713090 / 0.424703 | 0.718444 / 0.424703 |
| M · online | 5 | 0.632350 / 0.288588 | 0.644297 / 0.181654 | 0.694508 / 0.508872 | 0.722276 / 0.338430 | 0.652528 / 0.649607 | 0.671587 / 0.647264 | 0.669591 / 0.435736 | 0.707316 / 0.423362 | 0.712787 / 0.423362 |
| P01 · fixed_ema | 5 | 0.646211 / 0.288999 | 0.658275 / 0.183339 | 0.703972 / 0.503536 | 0.732342 / 0.344493 | 0.658672 / 0.649815 | 0.684181 / 0.642628 | 0.680609 / 0.435469 | 0.717122 / 0.423715 | 0.722291 / 0.423715 |
| P01 · online | 5 | 0.640211 / 0.290633 | 0.649641 / 0.183992 | 0.697327 / 0.508289 | 0.725284 / 0.347381 | 0.653471 / 0.649492 | 0.680713 / 0.644132 | 0.674441 / 0.437320 | 0.710284 / 0.427482 | 0.715639 / 0.427482 |

**kuairand / E10**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.634157 / 0.291473 | 0.620435 / 0.183852 | 0.654081 / 0.543733 | 0.683087 / 0.378828 | 0.632829 / 0.659007 | 0.658395 / 0.660995 | 0.647164 / 0.452981 | 0.667982 / 0.460432 | 0.674844 / 0.460432 |
| A0 · online | 5 | 0.625661 / 0.289602 | 0.633640 / 0.182626 | 0.673181 / 0.513764 | 0.701915 / 0.351586 | 0.636584 / 0.653748 | 0.666763 / 0.646654 | 0.656291 / 0.439664 | 0.686677 / 0.432249 | 0.692923 / 0.432249 |
| C_MIX · fixed_ema | 5 | 0.630807 / 0.291488 | 0.622459 / 0.183691 | 0.655514 / 0.539707 | 0.684374 / 0.381354 | 0.634728 / 0.656285 | 0.660075 / 0.658706 | 0.647993 / 0.451872 | 0.669324 / 0.459676 | 0.676109 / 0.459676 |
| C_MIX · online | 5 | 0.625381 / 0.290693 | 0.638439 / 0.182493 | 0.672277 / 0.517299 | 0.700311 / 0.357315 | 0.637596 / 0.652721 | 0.665370 / 0.647687 | 0.656562 / 0.441368 | 0.685468 / 0.436803 | 0.691692 / 0.436803 |
| Gaussian · fixed_ema | 5 | 0.624739 / 0.291742 | 0.617940 / 0.183863 | 0.655503 / 0.523490 | 0.684354 / 0.365341 | 0.631633 / 0.655339 | 0.655238 / 0.658093 | 0.644901 / 0.446311 | 0.669225 / 0.443846 | 0.676162 / 0.443846 |
| H_MIX · fixed_ema | 5 | 0.629577 / 0.291526 | 0.620652 / 0.183774 | 0.654491 / 0.530868 | 0.683596 / 0.371121 | 0.632128 / 0.656143 | 0.655930 / 0.657545 | 0.646062 / 0.448496 | 0.668399 / 0.450307 | 0.675262 / 0.450307 |
| H_MIX · online | 5 | 0.631408 / 0.289199 | 0.641118 / 0.182255 | 0.672528 / 0.518762 | 0.700709 / 0.358170 | 0.638393 / 0.651449 | 0.664189 / 0.649757 | 0.658058 / 0.441599 | 0.685831 / 0.437936 | 0.692040 / 0.437936 |
| M · fixed_ema | 5 | 0.614461 / 0.292700 | 0.606166 / 0.184239 | 0.651257 / 0.506670 | 0.676631 / 0.404501 | 0.625315 / 0.655319 | 0.646088 / 0.677938 | 0.636653 / 0.453561 | 0.663174 / 0.454897 | 0.670200 / 0.454897 |
| M · online | 5 | 0.625859 / 0.290610 | 0.640185 / 0.182392 | 0.662335 / 0.525955 | 0.691828 / 0.351411 | 0.628117 / 0.660159 | 0.654128 / 0.654692 | 0.650408 / 0.444203 | 0.676354 / 0.438207 | 0.682964 / 0.438207 |
| P01 · fixed_ema | 5 | 0.633358 / 0.291094 | 0.628854 / 0.183102 | 0.654768 / 0.537774 | 0.684091 / 0.378085 | 0.632872 / 0.655660 | 0.657945 / 0.656858 | 0.648648 / 0.450429 | 0.668851 / 0.457108 | 0.675593 / 0.457108 |
| P01 · online | 5 | 0.635334 / 0.288396 | 0.644635 / 0.181843 | 0.672800 / 0.512113 | 0.701416 / 0.351501 | 0.637728 / 0.651635 | 0.664618 / 0.646644 | 0.659422 / 0.438689 | 0.686349 / 0.431378 | 0.692543 / 0.431378 |

**kuairand / E50**

| 配置 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 · fixed_ema | 5 | 0.638061 / 0.289841 | 0.654246 / 0.182291 | 0.706191 / 0.501596 | 0.734353 / 0.343379 | 0.655168 / 0.652106 | 0.675257 / 0.648780 | 0.677213 / 0.436332 | 0.719089 / 0.422245 | 0.724251 / 0.422245 |
| A0 · online | 5 | 0.632881 / 0.290156 | 0.646121 / 0.182558 | 0.700766 / 0.512846 | 0.728977 / 0.355560 | 0.650036 / 0.653260 | 0.670354 / 0.657882 | 0.671523 / 0.442044 | 0.713679 / 0.433792 | 0.718986 / 0.433792 |
| C_MIX · fixed_ema | 5 | 0.631585 / 0.290053 | 0.638600 / 0.183470 | 0.704860 / 0.501196 | 0.733265 / 0.343175 | 0.655289 / 0.651458 | 0.676223 / 0.647023 | 0.673304 / 0.436063 | 0.717790 / 0.421947 | 0.723086 / 0.421947 |
| C_MIX · online | 5 | 0.629280 / 0.290562 | 0.624765 / 0.184821 | 0.700365 / 0.503571 | 0.728545 / 0.343510 | 0.651583 / 0.653360 | 0.671946 / 0.651393 | 0.667747 / 0.437869 | 0.713151 / 0.423310 | 0.718727 / 0.423310 |
| Gaussian · fixed_ema | 5 | 0.630040 / 0.291828 | 0.660350 / 0.182036 | 0.705018 / 0.501301 | 0.733077 / 0.342510 | 0.660249 / 0.646869 | 0.684140 / 0.640407 | 0.678812 / 0.434159 | 0.717929 / 0.421632 | 0.723148 / 0.421632 |
| H_MIX · fixed_ema | 5 | 0.641493 / 0.288515 | 0.649720 / 0.183547 | 0.705434 / 0.499498 | 0.733446 / 0.341637 | 0.657301 / 0.649570 | 0.677293 / 0.646126 | 0.677448 / 0.434816 | 0.718282 / 0.420339 | 0.723484 / 0.420339 |
| H_MIX · online | 5 | 0.642858 / 0.289103 | 0.649068 / 0.182506 | 0.700434 / 0.513014 | 0.728396 / 0.350927 | 0.651541 / 0.654884 | 0.672616 / 0.655497 | 0.674152 / 0.440989 | 0.713311 / 0.431591 | 0.718708 / 0.431591 |
| M · fixed_ema | 5 | 0.633886 / 0.288498 | 0.648370 / 0.181436 | 0.700661 / 0.511527 | 0.728786 / 0.337630 | 0.656728 / 0.649296 | 0.677120 / 0.642198 | 0.674258 / 0.435098 | 0.713600 / 0.424252 | 0.718947 / 0.424252 |
| M · online | 5 | 0.628350 / 0.289433 | 0.639769 / 0.182445 | 0.694943 / 0.505916 | 0.722761 / 0.339434 | 0.650284 / 0.651692 | 0.671990 / 0.647069 | 0.668016 / 0.435998 | 0.707717 / 0.422420 | 0.713302 / 0.422420 |
| P01 · fixed_ema | 5 | 0.646020 / 0.289654 | 0.655992 / 0.184884 | 0.705194 / 0.501995 | 0.733447 / 0.343529 | 0.658450 / 0.650371 | 0.684161 / 0.642923 | 0.680544 / 0.435559 | 0.718250 / 0.422499 | 0.723404 / 0.422499 |
| P01 · online | 5 | 0.640450 / 0.289913 | 0.651935 / 0.183387 | 0.699666 / 0.509085 | 0.727532 / 0.349241 | 0.653504 / 0.651902 | 0.678152 / 0.646780 | 0.675206 / 0.438385 | 0.712540 / 0.428799 | 0.717878 / 0.428799 |

总收益相对同seed普通A0 online，组件增量相对A0固定EMA。C_MIX/H_MIX/P01两数据总AUC均达标且满足六格约束；M总ΔAUC为ML+.002956435/KR+.002534206，均未达+.003。KR P01组件ΔAUC+.004085725，但d1t0/t1为−.001333993/−.001047387，标记“宏AUC达标、逐格约束未满足”；总ΔAUC+.008702125且六格均升。LL独立裁定，见[完整裁定](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/DECISIONS.json)。

历史KR GATS B2＋EMA按新规则对普通训练的总ΔAUC+.006097847、ΔLL−.003002719，六格均值均提升；对固定EMA增量+.001481447/−.001777377，对Gaussian＋EMA+.000005156/−.000113784。其既有五seed记录保留在下方GATS分组，新增[逐seed配对复算](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/GATS_B2_HISTORICAL_PAIRED.csv)。

### 已发表论文模型（MDMT-SOTA-2526，V按LogLoss选择）

本组比较已发表模型，以 V 的 LogLoss 选点并在 E 评价。绝对值用于本组协议下的模型比较。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/baseline/mdmt-sota-2526/research_reports/mdmt_baselines_2526/analysis/CELL_COMPARISONS.csv)。

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

普通Adam，无固定EMA；5seed；含 B-ML / L2-ML / B-KR。该组配置用于内部机制比较。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01/research_reports/2026-10-31_v2/analysis/TEST_METRICS.csv)。

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

A0、A1、A2、A3 各用五个训练 seed，online 与 shadow 共享对应训练轨迹。ML本组EMAρ=.995，与AIGM调参后的ML时间常数不同。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-01_v2/research_reports/2026-11-01_momentum/test_metrics.csv)。

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

记录仓库已经运行的fixed_ema和必要adaptive/replay对照；B2/KuaiRand 完成五 seed，其余多数为两 seed 开发记录；NO_GO 表示当时开发门的裁定。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02/research_reports/2026-11-02_gradient_aug_time_select/tables/test_metrics.csv)。

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

本表为两个开发 seed 的 fixed EMA selected 测试均值，确认阶段未触发。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-02_v2/research_reports/2026-11-02_ema_view_design_v2/tables/cell_effects.csv)。

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

新算法仅seed2022（n=1）；A0f/A1引入历史2022/2024两seed（n=2）。原 D0 停止记录的是旧协议的开发门。  [本组主要原始记录](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-03_history_information/research_reports/2026-11-03_mdmt_history_information/six_cell.csv)。

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

- 2026-10-31 GA31-v2 及更早的自研算法、共享和原型实验已记录在跨阶段总结中。待从对应的原始 CSV 按各自协议补齐六格绝对 AUC 和 LogLoss。
- 只存 Validation 或单项指标的早期实验按实有字段登记。待回填值使用对应实验和 seed 的原始结果。



## C05 CELL-MASS-BCE-025 — 2026-11-18 完整预算实测

新增U十轨迹/500epochs；C/S/W三十轨迹为数值复用。下表是各stream独立selected AUC状态的五seed均值，LL属于同一状态。source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d；Testfalse，NO_CONTINUATION_SIGNAL，不开C06。

|dataset|arm|stream|cell|AUC mean|LL mean|n|数值复用|
|---|---|---|---|---:|---:|---:|---|
|movielens|CLEAN|online|d0t0|0.7610642751721525|0.5887920501968277|5|True|
|movielens|CLEAN|online|d0t1|0.767772820302342|0.4463216328993734|5|True|
|movielens|CLEAN|online|d1t0|0.7562662154860076|0.5757658503027605|5|True|
|movielens|CLEAN|online|d1t1|0.7533132995965143|0.48325111548090677|5|True|
|movielens|CLEAN|online|d2t0|0.7514744346188452|0.5834470681392702|5|True|
|movielens|CLEAN|online|d2t1|0.7507396042365948|0.4827303205956095|5|True|
|movielens|CLEAN|fixed_ema|d0t0|0.7648681944327926|0.5835797867184486|5|True|
|movielens|CLEAN|fixed_ema|d0t1|0.7710484878077977|0.44389309003470384|5|True|
|movielens|CLEAN|fixed_ema|d1t0|0.758079599329402|0.5750018860174813|5|True|
|movielens|CLEAN|fixed_ema|d1t1|0.7556765310616431|0.48198479698828844|5|True|
|movielens|CLEAN|fixed_ema|d2t0|0.7552001806316581|0.5829509955711275|5|True|
|movielens|CLEAN|fixed_ema|d2t1|0.7537363515516236|0.4807296581953573|5|True|
|movielens|MASS-SCALE-025|online|d0t0|0.7618530568570894|0.58520524324182|5|True|
|movielens|MASS-SCALE-025|online|d0t1|0.7674923324569153|0.4448741444931568|5|True|
|movielens|MASS-SCALE-025|online|d1t0|0.7554160645159157|0.5790826059127561|5|True|
|movielens|MASS-SCALE-025|online|d1t1|0.7542362316398888|0.48761213827081434|5|True|
|movielens|MASS-SCALE-025|online|d2t0|0.7515893352515823|0.5796612234036914|5|True|
|movielens|MASS-SCALE-025|online|d2t1|0.7509692394547924|0.47998496626934617|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d0t0|0.7645889850472334|0.5846841356568446|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d0t1|0.7713470530432031|0.44379193689750246|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d1t0|0.7577010218964768|0.576115287600851|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d1t1|0.7555441727702688|0.4823807241209292|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d2t0|0.7548650593908913|0.5854245778353785|5|True|
|movielens|MASS-SCALE-025|fixed_ema|d2t1|0.7534953460723681|0.48176332882963663|5|True|
|movielens|COMPLETION-BCE-025|online|d0t0|0.761303951929744|0.5849721608110726|5|True|
|movielens|COMPLETION-BCE-025|online|d0t1|0.7679944655986645|0.44564715022161927|5|True|
|movielens|COMPLETION-BCE-025|online|d1t0|0.7553946733024113|0.575734271990956|5|True|
|movielens|COMPLETION-BCE-025|online|d1t1|0.754094364695953|0.4894523970810441|5|True|
|movielens|COMPLETION-BCE-025|online|d2t0|0.7517116299617308|0.5787318930692661|5|True|
|movielens|COMPLETION-BCE-025|online|d2t1|0.7508800557583738|0.48022088890832987|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d0t0|0.7648512557899483|0.5857106471862196|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d0t1|0.7708024265597415|0.440903323945945|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d1t0|0.7577387388295359|0.575584740742003|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d1t1|0.7555840923614264|0.48171135556127237|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d2t0|0.7552101200897012|0.5860437340506752|5|True|
|movielens|COMPLETION-BCE-025|fixed_ema|d2t1|0.753494982498767|0.47635451319860145|5|True|
|movielens|CELL-MASS-BCE-025|online|d0t0|0.7607958873930328|0.5878661062341065|5|False|
|movielens|CELL-MASS-BCE-025|online|d0t1|0.7691603189814703|0.4449825788469233|5|False|
|movielens|CELL-MASS-BCE-025|online|d1t0|0.7550493670442219|0.5766602358240588|5|False|
|movielens|CELL-MASS-BCE-025|online|d1t1|0.7529159904015592|0.48529251187728467|5|False|
|movielens|CELL-MASS-BCE-025|online|d2t0|0.7516914357988755|0.5829384450883272|5|False|
|movielens|CELL-MASS-BCE-025|online|d2t1|0.75075550633798|0.48072199772676855|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d0t0|0.764059084514331|0.5843797597088277|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d0t1|0.7711754958468877|0.44365943040506045|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d1t0|0.7577911109189898|0.5748255298001321|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d1t1|0.7555608911276769|0.48114998369748246|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d2t0|0.7548126899102534|0.5851209356923021|5|False|
|movielens|CELL-MASS-BCE-025|fixed_ema|d2t1|0.7538831133057745|0.4809248722149434|5|False|
|kuairand|CLEAN|online|d0t0|0.6811884606723768|0.2880031486492482|5|True|
|kuairand|CLEAN|online|d0t1|0.6932894138506864|0.15207032222015998|5|True|
|kuairand|CLEAN|online|d1t0|0.6976741960151004|0.49796032622127673|5|True|
|kuairand|CLEAN|online|d1t1|0.7242136600218428|0.341292122440089|5|True|
|kuairand|CLEAN|online|d2t0|0.6667609812576887|0.6458655146890366|5|True|
|kuairand|CLEAN|online|d2t1|0.6735985705550519|0.6477827431854434|5|True|
|kuairand|CLEAN|fixed_ema|d0t0|0.6782203948191454|0.2884686415123903|5|True|
|kuairand|CLEAN|fixed_ema|d0t1|0.6948744893786947|0.15144150316062915|5|True|
|kuairand|CLEAN|fixed_ema|d1t0|0.7024082010652769|0.49733873914616167|5|True|
|kuairand|CLEAN|fixed_ema|d1t1|0.729577994388435|0.3413555982208853|5|True|
|kuairand|CLEAN|fixed_ema|d2t0|0.6685831102105796|0.6450341468032389|5|True|
|kuairand|CLEAN|fixed_ema|d2t1|0.672924057168281|0.648198802395421|5|True|
|kuairand|MASS-SCALE-025|online|d0t0|0.665960317923655|0.2912114327370828|5|True|
|kuairand|MASS-SCALE-025|online|d0t1|0.6827365098532239|0.15295526868301412|5|True|
|kuairand|MASS-SCALE-025|online|d1t0|0.6982678048847607|0.49208709400264244|5|True|
|kuairand|MASS-SCALE-025|online|d1t1|0.7248797054828549|0.33662636435094473|5|True|
|kuairand|MASS-SCALE-025|online|d2t0|0.6660735219625153|0.6460071782943861|5|True|
|kuairand|MASS-SCALE-025|online|d2t1|0.6712785775975904|0.6506961987093026|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d0t0|0.6640257541512723|0.2913110844309861|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d0t1|0.6850934881214867|0.15265192769299185|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d1t0|0.7033154066393911|0.4964714235735124|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d1t1|0.7301553539481965|0.3403483625791492|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d2t0|0.6709600308753648|0.6424427601297427|5|True|
|kuairand|MASS-SCALE-025|fixed_ema|d2t1|0.6741764869629767|0.646684089933057|5|True|
|kuairand|COMPLETION-BCE-025|online|d0t0|0.6785074601969926|0.2913870218583086|5|True|
|kuairand|COMPLETION-BCE-025|online|d0t1|0.6978237692545789|0.1520021605084265|5|True|
|kuairand|COMPLETION-BCE-025|online|d1t0|0.6975135383005608|0.4921120549608724|5|True|
|kuairand|COMPLETION-BCE-025|online|d1t1|0.7242799582630925|0.332911075767893|5|True|
|kuairand|COMPLETION-BCE-025|online|d2t0|0.6744993607834624|0.6399753219506225|5|True|
|kuairand|COMPLETION-BCE-025|online|d2t1|0.6782164295758061|0.6420664236740388|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d0t0|0.6750790222096195|0.291985422817149|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d0t1|0.6953926577528459|0.15193559165652862|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d1t0|0.7022633183955513|0.49234367331713863|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d1t1|0.7292741160788254|0.3324797971577889|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d2t0|0.677054659044311|0.6395436915082684|5|True|
|kuairand|COMPLETION-BCE-025|fixed_ema|d2t1|0.68285498794917|0.6398952488275506|5|True|
|kuairand|CELL-MASS-BCE-025|online|d0t0|0.6733653036204265|0.29081362599908334|5|False|
|kuairand|CELL-MASS-BCE-025|online|d0t1|0.7004352461490843|0.15179400545561286|5|False|
|kuairand|CELL-MASS-BCE-025|online|d1t0|0.6977986330472821|0.4981863520284918|5|False|
|kuairand|CELL-MASS-BCE-025|online|d1t1|0.7247625289124601|0.3417203063451008|5|False|
|kuairand|CELL-MASS-BCE-025|online|d2t0|0.6676158429215814|0.6429500365923635|5|False|
|kuairand|CELL-MASS-BCE-025|online|d2t1|0.6727724998394016|0.6472513499713985|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d0t0|0.670694144433417|0.2900628703106657|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d0t1|0.6987668662659396|0.15145713502237246|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d1t0|0.7021528655376537|0.5004614224890886|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d1t1|0.7294164125700698|0.34373965885888486|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d2t0|0.6707728489760475|0.642948429384234|5|False|
|kuairand|CELL-MASS-BCE-025|fixed_ema|d2t1|0.6733708776763431|0.64662643376011|5|False|

报告research_reports/cell_mass_bce_20261117/REPORT.md；95,508,960B新U选中模型已父get验收；C04模型当前不可用，仅数值有效。所有逐seed/CI/stage详见完整附录，不能用40轨迹充当n40。


## 2026-11-23 · 条件历史与共享更新

A21（static PAIR，λ=.05，S_DENSE，fixed EMA），seeds 2022–2026。selected Test Macro AUC / LogLoss：MovieLens .778474 / .494491，KuaiRand .680766 / .435339。相对已冻结最强发表对照的 ΔAUC / ΔLogLoss：ML +.003284 / −.000400，KR −.001347 / −.005250；匹配同 PAIR Adam 的 ΔAUC：ML −.000083，KR +.004062。 [完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md)

## EXP082 · 来源耦合与历时预条件辅助历史

| 对照，均为五 seed | MovieLens ΔMacro AUC / ΔLogLoss | KuaiRand ΔMacro AUC / ΔLogLoss |
| --- | --- | --- |
| Validation selected fixed EMA，C2_A100−H0 | +.000255 / −.000845 | +.001711 / −.000944 |
| Test selected fixed EMA，C1_Z050−B1 | −.000382 / +.000497 | +.003081 / −.000182 |
| Test selected fixed EMA，C2_A100−B1 | −.000190 / −.001581 | +.000863 / +.000029 |
| Test selected fixed EMA，C2_A100−H0 | −.000107497 / −.000502 | −.003199 / −.000041 |

MovieLens C2_A100 EMA 相对 M3oE online +.003177，95% CI `[+.002013,+.004340]`，5/5 正，最差格 +.002504，发表数值门通过。同读出相对 B0/B1/H0 全负。KR 相对 HyperGate −.004546，与 H0 比较时 d0 两格 −.01016/−.02038，d1、d2 四格为正。KR H0 相对 B1 的固定 EMA Test +.004062 重现旧人口结果，同一流程 Validation −.001989。80 条当前冻结轨迹的 EMA−online Test Macro AUC 有 77 条正。Test 已具历史暴露身份，训练与选型在新评价前冻结。

100 条轨迹全部首次完成，共 5,000 epoch、172,500 更新。C1、C2 均为 `NO_ACTIONABLE_INCREMENT`，Cstar 为空。 [逐 seed 与六格](../research_reports/history_information_20261124/NUMERICAL_APPENDIX.md) · [配对汇总](../artifacts/history_information_20261124/test/summary.csv) · [研究结论与成本](../research_reports/history_information_20261124/FINAL_REPORT.md)

