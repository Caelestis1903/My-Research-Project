# N17 IMPLEMENTATION REPORT

2026-10-20。仅 IMPLEMENT；设计为 `N17_N18_DESIGN_ADDENDUM.md` 的 D-N17。未启动正式训练，未分析研究结果。已运行必要语法与同batch工程身份检查。

## 本批交付范围

新增独立文件：`n17_bootstrap.py`、`n17_train.py`、`n17_manifest.py`、`n17_diagnostics.py`、`n17_readout.py`、`n17_basiccheck.py`，全部位于 `experiments/natural_ml_extensions/`。原V2运行依赖的四个wrapper、V2归档及其inventory所选条目完全未改。末次机械比对V2 sources与已冻结manifest一致。

历史source按Git对象归档到source_archive/n17，inventory逐文件记录原path/commit/SHA。本批补归档原geometry/local、validation split、test readout和performance定义；不编辑历史source。N17 source集合固定六个新增模块及其实际归档、core人口依赖，不纳入以后AO04/N18代码。

## 原算法与唯一预算注入

- 调用统一core `load_population`，自然和matched-N人口均固定seed2022。不得向loader传optimizer seed。原schema/held-out数组共用。
- 原 `v3_audit.frozen_split` 原样调用：domain×joint-label分层、固定row-ID SHA排序、半数selection，另一半diagnostic。split归档在N17/history；config另外记录两部分有序row-ID SHA。
- 训练类继承归档 `n17_ml.Runner`，其fixed_model、Sampler、`step`、`objective`、完整native state、capture save继承原实现。单Adam，B40960，betas(.9,.999)、eps1e-8、WD1e-5，无scheduler/clipping/early stopping；model原routing detach及sigmoid(1)fusion保持。
- `fit`为归档n17_train原loop逐句转录，唯一预算变量n可取预注册187500；主自然完整800167曝光/epoch仍20步，100epoch2000main步。匹配曝光187500为4×40960+23660尾批，共500步；matched-N完整187500同500步。每epoch J始终按该人口完整训练支持计算；匹配曝光的J没有切小支持。
- 原quota系统随机舍入Sampler直接复用；每实际batch含尾批重算 `B_t*p_d/n_d` 并逐batch记录counts/weights。
- capture保留每epoch模型，完整state保存selected/E10/E40/E100/last及初始化，原E20/E50模型保存保持。另写all_model_index与native_state_index。每epoch首两次实际update只读保存全部坐标gradient/native optimizer before+after/实际update；不改backward/step。

## 72基础配置及条件分支

`n17_manifest.base_configurations()` 断言72：16完整100epoch pilots（LR .0003/.001/.003/.01 ×SN/SB×2030/2031）；16自然正式四格；16完整自然支持匹配曝光四格；16自然matched-N四格；8共同LR冻结后quota。正式四格均为LR .0003/.001 ×SN/SB×2022–25。quota在base_manifest明确列为待共同LR的required模板，不能遗漏；全pilots结束后post_pilot_manifest给出8条可执行真实配置。

`freeze_winners`必须读取全部16条完整100epoch receipt与有限trajectory，完整响应面写pilot_response.csv；使用每run 1..100 selection宏AUC最大值均值选共同LR及各q独立LR，精确平局较小LR。共同/独立winner uses单独列账，.003/.01实际所需正式配置去重追加，原四格保留。2030/2031仅调参不当正式seed。

`plateau_gate`只读取共同LR全部八父轨迹的训练J，要求完整有限；rho按J81:90和J91:100原式，任一q mean rho>.005且至少3/4正时两个q全四seed延长至200。H100 source目录不变；H200执行前复制完整native state/history到独立H200目录后resume。H100 selected/final留在父目录；H200重新在1..200严格选轮，单独列账。显式记载历史8seed/6正到本次4seed/3正的迁移。gate完全不读test。

## 固定点诊断

- 共同LR SN/SB×E10/E40×四seed共16锚，checkpoint验证arm/seed/LR/epoch/population。
- N2调用原geometry实现，每锚SN-IID/SB-IID/SN-quota/SB-quota各512batch。ALL及C/P/R/E与C子块覆盖完整坐标，原exact finite-pool domain mean、原covariance trace/within/count项、energy/domain MCSE、原Adam当前分母和滞后一拍分母映射均保留。额外用已保存ALL坐标m2精确计算滞后共同分母affine-map trace及无动量SGD covariance/mean，不重采样或改变估计器。明确训练图导数字段与functional Jacobian不同。
- N3调用原local实现：同theta/原生m/v/step、SN/SB配对draw；ALL native、C替换、C norm×direction四格、五个预定C子块及P参考全部保留。原逐cell AUC/LL/J和applied block norm均写盘；wrapper仅做hybrid−SB、hybrid−SN、相对anchor的原始减法及|gap|≥1e-4时辅助attenuation，不作科学结论。
- 每锚两个独立32draw bank：bank0为draw0..31，bank1为同IID流后续独立draw32..63，q间配对，bank/seed层次显式列账。
- **RESOLVED_BEFORE_FORMAL_RESULTS**：原规范没有给“必要时增加至64draw”的精度数值门槛。初版曾禁用该adaptive precision加量分支；当前版本已实现冻结门槛，由主代理在新结果前补全精度门槛，见 N17_PRECISION_DESIGN_AMENDMENT.md；基础32draw双bank和72训练/条件LR/H200分支均可独立review。

## 冻结与原始读数

训练完全不读test。validation readout可从每epoch已存预测机械重算selection/diagnostic/full-val逐cell表。primary提供全部required/triggered stage关闭及训练/选轮定义冻结receipt后，`n17_readout.freeze`冻结全部模型SHA；test只构建model，无optimizer、无update，验证model前后hash一致。全部1..100（或200）test预测/逐cell指标保存，可取selected/final/E10/20/50/100。AULC严格沿原 `n17_performance.py` 定义：完成epoch指标的算术平均，保存1..20/1..100；未改为梯形面积。保持historical-test-exposed exploratory标签。

## 工程检查与证据

`/home_new/xzf_natural_movielens_reassessment/N17/engineering_implementation_v2/receipt.json` 为PASS_BASIC_IDENTITY_CHECKS：同batch原mean-row mean-task BCE×p/q恒等、与直接原式Adam update后逐参数bitwise相同、单optimizer/fixed fusion、完整保存恢复后sampler下一21927行精确一致、quota的40960/21927/23660三种batch实际权重逐域积分等于p、selection/diagnostic互斥且并集完整。仅一个工程update，不计正式实验或科研效果。其后仅补离线readout定义/diagnostic trace说明；六模块py_compile通过，无正式训练。

## 机械命令

统一Python：`/home/xzf/miniconda3/envs/phase0-3-env/bin/python`，工作目录新worktree。

1. `python -m experiments.natural_ml_extensions.n17_manifest prepare`：只写72基础清单及64个立即可执行config（16pilot+48四格）；quota待共同LR解析。
2. `python -m experiments.natural_ml_extensions.n17_train --config <N17/configs/ID.json> --device cuda:0`：完整预算；中断attempt保留后同seed全预算重跑；H200仅按预注册continuation恢复。
3. 全pilots后：`python -m experiments.natural_ml_extensions.n17_manifest freeze-winners`，生成额外LR及8quota真实config。
4. 共同LR八父轨迹完整后：`python -m experiments.natural_ml_extensions.n17_manifest plateau`；若触发，运行H200 config，首次命令自动复制父工件并resume，后续中断按同seed完整预算重新开始该阶段。
5. `python -m experiments.natural_ml_extensions.n17_diagnostics --manifest`：仅在winner冻结后生成16 N2+32 N3 bank配置。执行：同module `--config <ID.json> --device cuda:0`。
6. `python -m experiments.natural_ml_extensions.n17_readout validation --run-dir <RUN>`：原始validation读数。所有required/triggered stage关闭后：`... n17_readout freeze --config-list <JSON字符串路径数组> --closed-receipt <primary关闭receipt>`；随后 `... n17_readout test --run-dir <RUN> --device cuda:0`。

本报告仅交IMPLEMENT；按照用户随后明确补充的 start.md 第11条，由主代理完成自审并交 luna_worker。AO04/N18不在本批放行范围，尚未实现，不应冒报完成。

主代理补充：当前 independent_review=False，review gate 在运行前绑定全部N17 source hashes；N3 --precision-gate 按配对 development C 三比较 MCSE>2.5e-5或未定义触发双bank各额外32draw，流64..95/96..127不与原bank重合，最多64/bank，不按正号扩量。
