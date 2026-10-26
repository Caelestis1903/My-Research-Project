# DESIGN SPECIFICATION — 自然 MovieLens 全研究链复评

2026-10-20，独立 designer。已完整读 `restart.md`（gb18030），只读审计当前目录及6个执行worktree。本文与三个独立阶段规范共同构成完整设计：

- [CORE_DESIGN_SPECIFICATION.md](CORE_DESIGN_SPECIFICATION.md)：数据、A/B/C、目标/采样/历史状态及必要人口控制。
- [N17_N18_DESIGN_ADDENDUM.md](N17_N18_DESIGN_ADDENDUM.md)：v2、N17、N18、AO04。
- [N19_CM19_DESIGN_ADDENDUM.md](N19_CM19_DESIGN_ADDENDUM.md)：全部N19/CM19正式ML组、机制与匹配控制。

状态：**DESIGN COMPLETE；实现/独立审查/正式运行/分析尚不能由此文推定完成。** 新研究最新状态以新RUN_LEDGER为准；本文件不是结果报告。每阶段独立先设计后实现、review PASS后RUN，子阶段可按依赖并行。设计者没有实现算法或运行实验。

## 1. Research question

2026-10-15以来，人工ML(d2_large，部分AO04为d0_large)上的哪些科学结论受到训练人口的实质限制？自然ML能否复现其方向、量级、task/cell和阶段关系？当总独立样本、采样概率和更新/曝光改变时，旧机制能否经匹配控制保留？哪些结果仍只属于自然KR或受控模拟器？

## 2. Hypotheses being tested

对A–F逐项建立可拒绝的历史外推，不将任何旧正/阴性当先验真理。区分：相同目标梯度identity；有限估计器/优化轨迹差；共享/Adam历史因果；checkpoint与指标选择；条件模拟机制；数值路径活跃；可部署selected收益。发现数学identity成立不等于泛化收益成立，路径恢复不等于方法优于强基线。

## 3. Variables changed

人口独立配置：完整自然ML、人工固定2022 ML、自然比例等187500 base-train支持。原人工d0/d1/d2角色干预保留其受控身份，不把Natural=q=p当作自然人口。每原研究自己的alpha/q、优化器、模型/方法/variant和预算独立记录。必要组交叉匹配总支持N与更新/曝光，其它组只作人口迁移，不声称已唯一识别比例因果。

## 4. Variables held constant

基础raw CSV、80/10/10全局有序split、三年龄物理域、native两个rating阈值任务、输入schema、训练期预处理、固定人口及评估行。默认formal seeds2022/23/24/25。历史N17原8seed，新4seed明确披露；pilot2030/31属于设计明定的调参例外。所有旧方法保留实际算法、优化器/WD、fusion、detach、arch update、选轮和完整epoch预算。禁止用F0 Q容量/统一简化网络替代其它研究。

## 5. Mathematical / algorithmic definition

数据基础唯一入口 `experiments/stage_F0_data.py:load_movielens_natural`。正式dataset_config=`movielens_natural`，内部dataset=movielens保证旧dispatch，setting=natural。全base-train不做role抽样，p、q、alpha/q及meta后p从实数重算。

预期审计数：base-train800167，逐域155949/318599/325619；N19/CM19按5%每域floor meta后main148152/302670/309339，总760161。自然比例等N187500基于最大余数预计36543/74656/76301；5%每域floor后178126，历史人工main178125。因此base N是严格匹配，meta后差1行必须记录；不能伪报主训练人口完全等N。所有正式数量仍以实际source/hash计数为准。

精确算法、full预算和受控matching见三个规范。历史原始组逐配置索引672行，包括N19所有14addenda及CM19完整manifest；schema动态派生字段不得照抄原人工数。所有implementation要在原namespace以外独立保存，完整源码快照绑定原提交及新改动hash。

## 6. Exact affected modules/files

core：`experiments/natural_ml_reassessment/`；N17/N18等：`experiments/natural_ml_extensions/`；N19/CM19可各用独立新目录。研究材料均在本目录。原ML/人工缓存和历史worktree只读；重型工件根 `/home_new/xzf_natural_movielens_reassessment/`。参考源pins见HISTORICAL_EXPERIMENT_AUDIT，不引用“当前main包含全部研究”这一错误前提。

## 7. Implementation requirements

1. 数据审计记录raw CSV大小/SHA、基础split和全部derived subset/meta/selection/dev的row-ID及encoded fingerprint、每cell数量/正负/joint-label、OOV/encoder/scaler来源。优化seed不得改这些。
2. 逐历史组配置映射，保留原loss scale/estimator/网络/更新；任何结论相关修复先形成差异spec，不能冒称原协议迁移。
3. source/config提交冻结→独立reviewer读设计+实际实现+coder report→PASS或修完blocking的PASS WITH FIXES→luna-runner机械运行。runner不做科学分支选择、不分析结论。
4. 所有原预算完整训练，非有限或工程失败如实保留并修复后原seed重试，失败工件不覆盖。不得因早期阴性/代理失活取消后续种子。仅做科学正确性必要检查。
5. 持久manifest/attempt ledger，独立phase可并行，但active源码不可漂移。每GPU按实际显存与外部进程准入，≤5本研究进程只是上限而非占满目标；不干预他人任务。首次实际profile后给资源估算，设计不虚构吞吐。
6. selected只用原validation/selection规则，test在该阶段配置/选择/development处置冻结后零更新评价。已有历史test重复使用，所有新结果为探索性复现。

## 8. Ablations required

必须保留所有影响ML外推的正式候选、强对照、阴性方法；明确受控理论、KR-only和原未触发扩展不机械新增。不能只留baseline或表现最好的方法。核心alpha×q、S5四臂、S6阴性、无动量SGD；N17LR四格/对称pilot/quota/阶段；AO04B0/B1/B2；N19全部236；CM19全部436及匹配控制。原未触发B3/B4/N4不报失败。别名只按完整科学身份与实际轨迹验证，非配置名字相同。

## 9. Expected logging and analysis requirements

原始必需：配置、source/init/data/row/batch/checkpoint/prediction hashes；各任务域selected/final/common-stage AUC、LogLoss；宏、train-p及eval-p加权cell指标；实际epoch/main与arch steps、样本/独立支持/重复率、学习率与有效正则；完整state/optimizer/RNG/sampler和原始预测索引。weighted cell AUC不是pooled AUC。

科学分析由orchestrator执行，逐原问题并排：历史数值/条件、自然新四seed及绝对baseline、cell/task和训练阶段、方向/量级/符号、主/次指标冲突、匹配N/更新结果、保留/缩限/修改/反证/未识别。分开三个条件：自然KR、人工ML、自然ML；受控模拟器另有标签。不得以四个种子/六cell/512draw假造独立总体n。

证据强度STRONG/MEDIUM/WEAK均注明对象：算术identity、固定state机制、固定人口优化重复、或跨人口泛化。不把非拒绝当等效；效应实际界沿原研究，跨研究未定义者统一探索性AUC .003并明确此阈值来源，不能事后挑有利。保存全部配对seed差、均值/SD、描述性t区间/精确符号敏感性；n4双侧符号最小p=.125。已有用户ID可做same-user配对cluster bootstrap；否则row条件CI限制声明，不能代替训练seed不确定性。

每问题必须回应：稳定重复现象、矛盾、证据强度、竞争解释、大小是否可能主因、小域增/大域降是否存在、最能区分解释的下一实验。baseline AUC/LL与标签熵辅助评估经验难度，不以N命名困难域。所有开放问题不设数量上限。

## 10. Reviewer acceptance criteria

覆盖矩阵每项有原source/数据/预算及新run对应，三个规范计数能与执行manifest相符；所有未完成/条件未触发/不适用/严格alias分开；训练前不存在读取新test作决策；原population及源文件不覆盖；算法忠实，Adam数值/参数组/元数据更新可复核。每个阶段独立review通过才放行；全任务只有运行、评价、逐问题分析与GitHub交付均完成才可COMPLETE。

## 11. Conditions falsifying the hypotheses

自然迁移的方向逆转、seed/cell不稳定或only-final效果均反驳相应普遍外推；匹配N/steps消除现象反驳单纯比例或固有模型解释；独立STAR活跃反驳所有乘性结构必死；局部WD恢复但selected AUC/LL不改善反驳通用救活价值；同信息强控制解释候选效果反驳其独立机制；模拟世界的好结果不能确认真实未知桥接/MNAR/标注节约。阴性也按上述严格范围解释，不能反证未运行的更广问题。

## 预算、依赖与真实阻塞

基础名义配置1044：core120、V2/N17/N18/AO04100、N19316、CM19508，其中16为N17pilots，1028为正式名义配置。人工历史24可经身份核对复用、跨研究/数学等价可能进一步alias；可触发新增赢家/H200另列，不混入基础数。此计数是研究覆盖计划，不是已运行数，也不是强制制造重复run。

历史原ML成功receipt总进程墙钟：N19236条10.7094h、CM19436条17.3295h。自然main约4.267倍，但验证/保存成本不随train行同比，不能直接据此承诺总时间。仅这672条按曝光粗略外推尺度约120进程小时；加核心/控制可能达到约150–220进程小时，实际受模型/并发/磁盘影响。两3090最多十worker的进程累计小时不能称独占GPU小时；先以实际run更新ETA，不以估时大为由缩实验。

当前可用原始ML与历史源码/原始表；主代理已发现GPU/Python/大盘并建立独立branch。旧 `/home_new/xzf_cm19_artifacts`目前不存在，旧checkpoint/预测不得假称可加载；历史数值从报告/可得CSV重算的层级要如实记录。原缺失不阻塞独立初始化自然训练，但阻塞依赖旧state的精确诊断。必要状态由新自然训练完整留存。最终报告若仍缺历史bytes，明确证据边界，不能填造。
