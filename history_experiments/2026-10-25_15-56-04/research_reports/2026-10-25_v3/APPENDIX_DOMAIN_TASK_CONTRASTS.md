# 全部逐域与逐任务比较

本附录仅转录完成包的机器聚合数值，不作算法效果判断。质量数值固定显示10位小数，缺失或不可定义值显式为 `NaN`。`selected` 是各预测器按预登记验证指标、eligible≥12且平局取最早epoch的独立选点；`endpoint` 是同一预登记训练终点（开发24epoch，若请求正式manifest则为其50epoch）。Q_explore为已暴露开发评价，不是新未见确认。

统计单位为训练seed。配对差统一为候选减参照；AUC正差、LL负差对应有利方向。`favorable_gain`已统一为越大越有利。seed级Student t95%区间在n=2仅作开发描述；n=1无区间。仅actual formal阶段元数据对应的统计才属于锁定复评，实际n逐表列出；formal与开发、P2归因的seed分别分组，不能合成n=6。域、任务、cell与epoch不增加独立重复数；不报告p值。

domain均值为同域两任务等权；task均值为同任务三域等权。它们不替代预登记六格macro。

## domain 均值配对

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6645659682 | -0.0020868992 | -0.0020868992 | 138225 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7500343864 | -0.0028795518 | -0.0028795518 | 1395118 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.7001892017 | -0.0112283968 | -0.0112283968 | 71561 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2389242940 | 0.0018548136 | -0.0018548136 | 138225 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5296673863 | 0.0039764484 | -0.0039764484 | 1395118 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6173321411 | 0.0212632462 | -0.0212632462 | 71561 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6497578270 | 0.0127212420 | 0.0127212420 | 138225 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7485535602 | -0.0013987256 | -0.0013987256 | 1395118 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6910496598 | -0.0020888549 | -0.0020888549 | 71561 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2390103581 | 0.0017687496 | -0.0017687496 | 138225 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5305688130 | 0.0030750217 | -0.0030750217 | 1395118 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6235248907 | 0.0150704966 | -0.0150704966 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6683732979 | -0.0044090162 | -0.0044090162 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7493155245 | -0.0024155102 | -0.0024155102 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6918116432 | 0.0062995006 | 0.0062995006 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2398015285 | -0.0010515548 | 0.0010515548 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5313461049 | 0.0036013545 | -0.0036013545 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.6323537174 | -0.0132643581 | 0.0132643581 | 71561 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 18 | 23 | 0.6575247855 | 0.6647063230 | -0.0071815375 | -0.0071815375 | 138225 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 18 | 23 | 0.7455524817 | 0.7492090373 | -0.0036565556 | -0.0036565556 | 1395118 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 2 | 18 | 23 | 0.6953233059 | 0.6947975876 | 0.0005257183 | 0.0005257183 | 71561 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 18 | 23 | 0.2375284961 | 0.2380011829 | -0.0004726867 | 0.0004726867 | 138225 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 18 | 23 | 0.5377300475 | 0.5312351735 | 0.0064948740 | -0.0064948740 | 1395118 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 2 | 18 | 23 | 0.6230591551 | 0.6217881016 | 0.0012710535 | -0.0012710535 | 71561 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 18 | 24 | 0.6575247855 | 0.6497578270 | 0.0077669585 | 0.0077669585 | 138225 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 18 | 24 | 0.7455524817 | 0.7485535602 | -0.0030010785 | -0.0030010785 | 1395118 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 2 | 18 | 24 | 0.6953233059 | 0.6910496598 | 0.0042736461 | 0.0042736461 | 71561 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 18 | 24 | 0.2375284961 | 0.2390103581 | -0.0014818620 | 0.0014818620 | 138225 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 18 | 24 | 0.5377300475 | 0.5305688130 | 0.0071612346 | -0.0071612346 | 1395118 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 2 | 18 | 24 | 0.6230591551 | 0.6235248907 | -0.0004657357 | 0.0004657357 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6663312403 | -0.0023669586 | -0.0023669586 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7405755006 | 0.0063245137 | 0.0063245137 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.6866692519 | 0.0114418918 | 0.0114418918 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2371903589 | 0.0015596149 | -0.0015596149 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5355251073 | -0.0005776478 | 0.0005776478 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6263887154 | -0.0072993560 | 0.0072993560 | 71561 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8138380283 | 0.0031166845 | 0.0031166845 | 124759 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8174403943 | 0.0028372136 | 0.0028372136 | 254879 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8054502594 | 0.0030404727 | 0.0030404727 | 260495 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4742158328 | 0.0122473732 | -0.0122473732 | 124759 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4646630427 | 0.0066064632 | -0.0066064632 | 254879 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4767459578 | 0.0072029098 | -0.0072029098 | 260495 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8092835858 | 0.0076711270 | 0.0076711270 | 124759 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8138174744 | 0.0064601336 | 0.0064601336 | 254879 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8046122230 | 0.0038785091 | 0.0038785091 | 260495 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4841306021 | 0.0023326039 | -0.0023326039 | 124759 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4727628221 | -0.0014933162 | 0.0014933162 | 254879 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4793958231 | 0.0045530445 | -0.0045530445 | 260495 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8159609094 | -0.0004600539 | -0.0004600539 | 124759 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8194370982 | 0.0003312298 | 0.0003312298 | 254879 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8087318772 | -0.0029750109 | -0.0029750109 | 260495 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4794379447 | 0.0108088967 | -0.0108088967 | 124759 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4719695364 | -0.0024100895 | 0.0024100895 | 254879 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4804241746 | 0.0081633653 | -0.0081633653 | 260495 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 23 | 0.8076898769 | 0.8129224664 | -0.0052325895 | -0.0052325895 | 124759 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 23 | 0.8127024521 | 0.8168770034 | -0.0041745513 | -0.0041745513 | 254879 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 2 | 12 | 23 | 0.8022116233 | 0.8065080863 | -0.0042964630 | -0.0042964630 | 260495 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 23 | 0.4799775951 | 0.4758821564 | 0.0040954386 | -0.0040954386 | 124759 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 23 | 0.4694188145 | 0.4645134938 | 0.0049053207 | -0.0049053207 | 254879 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 2 | 12 | 23 | 0.4795692555 | 0.4753015282 | 0.0042677273 | -0.0042677273 | 260495 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 15 | 0.8076898769 | 0.8047347849 | 0.0029550920 | 0.0029550920 | 124759 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 15 | 0.8127024521 | 0.8116796432 | 0.0010228089 | 0.0010228089 | 254879 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 2 | 12 | 15 | 0.8022116233 | 0.8012245076 | 0.0009871157 | 0.0009871157 | 260495 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 15 | 0.4799775951 | 0.4840198339 | -0.0040422388 | 0.0040422388 | 124759 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 15 | 0.4694188145 | 0.4701914119 | -0.0007725974 | 0.0007725974 | 254879 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 2 | 12 | 15 | 0.4795692555 | 0.4823116834 | -0.0027424279 | 0.0027424279 | 260495 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 14 | 0.8076843096 | 0.8091408322 | -0.0014565226 | -0.0014565226 | 124759 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 14 | 0.8128438387 | 0.8136721279 | -0.0008282892 | -0.0008282892 | 254879 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 2 | 12 | 14 | 0.8030650002 | 0.8042607702 | -0.0011957700 | -0.0011957700 | 260495 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 14 | 0.4819283778 | 0.4796445642 | 0.0022838136 | -0.0022838136 | 124759 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 14 | 0.4691391859 | 0.4691950489 | -0.0000558631 | 0.0000558631 | 254879 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 2 | 12 | 14 | 0.4792607265 | 0.4790475562 | 0.0002131703 | -0.0002131703 | 260495 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6681512539 | 0.6619830103 | 0.0061682436 | 0.0061682436 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7472240891 | 0.7483570422 | -0.0011329531 | -0.0011329531 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6871004701 | 0.6862068017 | 0.0008936684 | 0.0008936684 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6544984805 | 0.6500084301 | 0.0044900505 | 0.0044900505 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7476622851 | 0.7477356529 | -0.0000733678 | -0.0000733678 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6853403005 | 0.6836438631 | 0.0016964373 | 0.0016964373 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2376876076 | 0.2378843447 | -0.0001967371 | 0.0001967371 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5329532239 | 0.5326278488 | 0.0003253750 | -0.0003253750 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6367978126 | 0.6385977380 | -0.0017999254 | 0.0017999254 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2427304218 | 0.2424731924 | 0.0002572294 | -0.0002572294 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5323617283 | 0.5318597214 | 0.0005020069 | -0.0005020069 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6390073098 | 0.6388239321 | 0.0001833777 | -0.0001833777 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6681512539 | 0.6587649502 | 0.0093863037 | 0.0093863037 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7472240891 | 0.7473430878 | -0.0001189987 | -0.0001189987 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6871004701 | 0.6908530260 | -0.0037525558 | -0.0037525558 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6544984805 | 0.6586742074 | -0.0041757268 | -0.0041757268 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7476622851 | 0.7479563574 | -0.0002940723 | -0.0002940723 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6853403005 | 0.6873646912 | -0.0020243907 | -0.0020243907 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2376876076 | 0.2390291689 | -0.0013415613 | 0.0013415613 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5329532239 | 0.5362402777 | -0.0032870538 | 0.0032870538 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6367978126 | 0.6317362322 | 0.0050615804 | -0.0050615804 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2427304218 | 0.2400644390 | 0.0026659828 | -0.0026659828 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5323617283 | 0.5318704590 | 0.0004912693 | -0.0004912693 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6390073098 | 0.6336653902 | 0.0053419195 | -0.0053419195 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6534086120 | 0.0090704570 | 0.0090704570 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7485716727 | -0.0014168381 | -0.0014168381 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6848729108 | 0.0040878941 | 0.0040878941 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6568117425 | 0.6455602423 | 0.0112515002 | 0.0112515002 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480133035 | 0.7476224974 | 0.0003908061 | 0.0003908061 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6880975689 | 0.6808556667 | 0.0072419022 | 0.0072419022 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2379685185 | 0.0028105891 | -0.0028105891 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5302800680 | 0.0033637667 | -0.0033637667 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6291214898 | 0.0094738975 | -0.0094738975 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2410265432 | 0.2396299483 | 0.0013965949 | -0.0013965949 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5313998687 | 0.5307141962 | 0.0006856725 | -0.0006856725 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6322366834 | 0.6297481940 | 0.0024884894 | -0.0024884894 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6681512539 | -0.0056721850 | -0.0056721850 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7472240891 | -0.0000692545 | -0.0000692545 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6871004701 | 0.0018603348 | 0.0018603348 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6568117425 | 0.6544984805 | 0.0023132620 | 0.0023132620 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480133035 | 0.7476622851 | 0.0003510185 | 0.0003510185 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6880975689 | 0.6853403005 | 0.0027572684 | 0.0027572684 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2376876076 | 0.0030915000 | -0.0030915000 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5329532239 | 0.0006906108 | -0.0006906108 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6367978126 | 0.0017975747 | -0.0017975747 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2410265432 | 0.2427304218 | -0.0017038786 | 0.0017038786 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5313998687 | 0.5323617283 | -0.0009618596 | 0.0009618596 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6322366834 | 0.6390073098 | -0.0067706264 | 0.0067706264 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6637000469 | -0.0012209779 | -0.0012209779 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7471336971 | 0.0000211375 | 0.0000211375 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6883596263 | 0.0006011786 | 0.0006011786 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6568117425 | 0.6603324625 | -0.0035207200 | -0.0035207200 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480133035 | 0.7475119825 | 0.0005013210 | 0.0005013210 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6880975689 | 0.6871762555 | 0.0009213134 | 0.0009213134 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2385045474 | 0.0022745602 | -0.0022745602 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5328458041 | 0.0007980306 | -0.0007980306 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6365876540 | 0.0020077333 | -0.0020077333 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2410265432 | 0.2399594634 | 0.0010670798 | -0.0010670798 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5313998687 | 0.5323445760 | -0.0009447072 | 0.0009447072 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6322366834 | 0.6333327819 | -0.0010960985 | 0.0010960985 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6638474837 | -0.0013684147 | -0.0013684147 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7472365764 | -0.0000817418 | -0.0000817418 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6901391215 | -0.0011783166 | -0.0011783166 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6568117425 | 0.6530374392 | 0.0037743033 | 0.0037743033 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480133035 | 0.7473183968 | 0.0006949067 | 0.0006949067 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6880975689 | 0.6864678986 | 0.0016296703 | 0.0016296703 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2387241933 | 0.0020549143 | -0.0020549143 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5328415327 | 0.0008023020 | -0.0008023020 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6337576523 | 0.0048377350 | -0.0048377350 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2410265432 | 0.2416334866 | -0.0006069434 | 0.0006069434 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5313998687 | 0.5327380318 | -0.0013381630 | 0.0013381630 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6322366834 | 0.6341392152 | -0.0019025318 | 0.0019025318 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6673860229 | 0.6615151516 | 0.0058708713 | 0.0058708713 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7479524825 | 0.7478778598 | 0.0000746227 | 0.0000746227 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6929011179 | 0.6954869982 | -0.0025858803 | -0.0025858803 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2445754669 | 0.2406946962 | 0.0038807707 | -0.0038807707 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5335826552 | 0.5329115842 | 0.0006710711 | -0.0006710711 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6308950992 | 0.6278229474 | 0.0030721518 | -0.0030721518 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6673860229 | 0.6615151516 | 0.0058708713 | 0.0058708713 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7479524825 | 0.7478778598 | 0.0000746227 | 0.0000746227 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6929011179 | 0.6954869982 | -0.0025858803 | -0.0025858803 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2445754669 | 0.2406946962 | 0.0038807707 | -0.0038807707 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5335826552 | 0.5329115842 | 0.0006710711 | -0.0006710711 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6308950992 | 0.6278229474 | 0.0030721518 | -0.0030721518 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6679158618 | 0.6615151516 | 0.0064007102 | 0.0064007102 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480694896 | 0.7478778598 | 0.0001916298 | 0.0001916298 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6933413207 | 0.6954869982 | -0.0021456775 | -0.0021456775 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2439968160 | 0.2406946962 | 0.0033021198 | -0.0033021198 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5344762313 | 0.5329115842 | 0.0015646472 | -0.0015646472 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6293489623 | 0.6278229474 | 0.0015260149 | -0.0015260149 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6679158618 | 0.6615151516 | 0.0064007102 | 0.0064007102 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7480694896 | 0.7478778598 | 0.0001916298 | 0.0001916298 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6933413207 | 0.6954869982 | -0.0021456775 | -0.0021456775 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2439968160 | 0.2406946962 | 0.0033021198 | -0.0033021198 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5344762313 | 0.5329115842 | 0.0015646472 | -0.0015646472 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6293489623 | 0.6278229474 | 0.0015260149 | -0.0015260149 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6661088936 | 0.6615151516 | 0.0045937420 | 0.0045937420 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7476339024 | 0.7478778598 | -0.0002439575 | -0.0002439575 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6934472842 | 0.6954869982 | -0.0020397141 | -0.0020397141 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2424171481 | 0.2406946962 | 0.0017224519 | -0.0017224519 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5327953279 | 0.5329115842 | -0.0001162562 | 0.0001162562 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6325329329 | 0.6278229474 | 0.0047099855 | -0.0047099855 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6548592547 | 0.6641384095 | -0.0092791548 | -0.0092791548 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7293164027 | 0.7485016556 | -0.0191852529 | -0.0191852529 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6809040803 | 0.6940831902 | -0.0131791099 | -0.0131791099 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6661088936 | 0.6615151516 | 0.0045937420 | 0.0045937420 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7476339024 | 0.7478778598 | -0.0002439575 | -0.0002439575 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6934472842 | 0.6954869982 | -0.0020397141 | -0.0020397141 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2567834392 | 0.2403722983 | 0.0164111409 | -0.0164111409 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5500220307 | 0.5332623454 | 0.0167596854 | -0.0167596854 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.7122714123 | 0.6346381276 | 0.0776332847 | -0.0776332847 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2424171481 | 0.2406946962 | 0.0017224519 | -0.0017224519 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5327953279 | 0.5329115842 | -0.0001162562 | 0.0001162562 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6325329329 | 0.6278229474 | 0.0047099855 | -0.0047099855 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6641384095 | -0.0001741277 | -0.0001741277 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7485016556 | -0.0016016414 | -0.0016016414 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6940831902 | 0.0040279535 | 0.0040279535 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6615151516 | -0.0017008426 | -0.0017008426 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7478778598 | -0.0001329249 | -0.0001329249 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6954869982 | 0.0000771394 | 0.0000771394 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2403722983 | -0.0016223245 | 0.0016223245 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5332623454 | 0.0016851141 | -0.0016851141 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.6346381276 | -0.0155487683 | 0.0155487683 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2406946962 | -0.0001989531 | 0.0001989531 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5329115842 | 0.0000127266 | -0.0000127266 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6278229474 | 0.0021872422 | -0.0021872422 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6641384095 | -0.0001741277 | -0.0001741277 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7485016556 | -0.0016016414 | -0.0016016414 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6940831902 | 0.0040279535 | 0.0040279535 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6615151516 | -0.0017008426 | -0.0017008426 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7478778598 | -0.0001329249 | -0.0001329249 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6954869982 | 0.0000771394 | 0.0000771394 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2403722983 | -0.0016223245 | 0.0016223245 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5332623454 | 0.0016851141 | -0.0016851141 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.6346381276 | -0.0155487683 | 0.0155487683 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2406946962 | -0.0001989531 | 0.0001989531 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5329115842 | 0.0000127266 | -0.0000127266 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6278229474 | 0.0021872422 | -0.0021872422 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6548592547 | 0.0091050270 | 0.0091050270 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7293164027 | 0.0175836116 | 0.0175836116 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6809040803 | 0.0172070634 | 0.0172070634 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6673860229 | -0.0075717138 | -0.0075717138 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7479524825 | -0.0002075475 | -0.0002075475 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6929011179 | 0.0026630197 | 0.0026630197 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2567834392 | -0.0180334654 | 0.0180334654 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5500220307 | -0.0150745713 | 0.0150745713 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.7122714123 | -0.0931820530 | 0.0931820530 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2445754669 | -0.0040797238 | 0.0040797238 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5335826552 | -0.0006583444 | 0.0006583444 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6308950992 | -0.0008849096 | 0.0008849096 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6548592547 | 0.0091050270 | 0.0091050270 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7293164027 | 0.0175836116 | 0.0175836116 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6809040803 | 0.0172070634 | 0.0172070634 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6679158618 | -0.0081015528 | -0.0081015528 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7480694896 | -0.0003245546 | -0.0003245546 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6933413207 | 0.0022228169 | 0.0022228169 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2567834392 | -0.0180334654 | 0.0180334654 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5500220307 | -0.0150745713 | 0.0150745713 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.7122714123 | -0.0931820530 | 0.0931820530 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2439968160 | -0.0035010729 | 0.0035010729 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5344762313 | -0.0015519205 | 0.0015519205 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6293489623 | 0.0006612273 | -0.0006612273 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6548592547 | 0.0091050270 | 0.0091050270 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7293164027 | 0.0175836116 | 0.0175836116 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6809040803 | 0.0172070634 | 0.0172070634 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6661088936 | -0.0062945846 | -0.0062945846 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7476339024 | 0.0001110326 | 0.0001110326 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6934472842 | 0.0021168535 | 0.0021168535 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2567834392 | -0.0180334654 | 0.0180334654 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5500220307 | -0.0150745713 | 0.0150745713 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.7122714123 | -0.0931820530 | 0.0931820530 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2424171481 | -0.0019214050 | 0.0019214050 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5327953279 | 0.0001289829 | -0.0001289829 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6325329329 | -0.0025227432 | 0.0025227432 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6648091131 | -0.0008448314 | -0.0008448314 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7483620699 | -0.0014620556 | -0.0014620556 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6946848756 | 0.0034262682 | 0.0034262682 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6598143091 | 0.6573853745 | 0.0024289346 | 0.0024289346 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7477449349 | 0.7476617318 | 0.0000832032 | 0.0000832032 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.6955641377 | 0.6940391123 | 0.0015250253 | 0.0015250253 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2431565527 | -0.0044065789 | 0.0044065789 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5318311177 | 0.0031163418 | -0.0031163418 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.6303881034 | -0.0112987441 | 0.0112987441 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.2404957431 | 0.2415490935 | -0.0010533504 | 0.0010533504 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.5329243108 | 0.5331050021 | -0.0001806913 | 0.0001806913 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.6300101897 | 0.6287905757 | 0.0012196140 | -0.0012196140 | 71561 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 15 | 15 | 0.6535030879 | 0.6469668186 | 0.0065362693 | 0.0065362693 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 15 | 15 | 0.7440904840 | 0.7451334056 | -0.0010429216 | -0.0010429216 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 2 | 15 | 15 | 0.6939194679 | 0.6901784974 | 0.0037409705 | 0.0037409705 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 23 | 23 | 0.6611817195 | 0.6557335770 | 0.0054481425 | 0.0054481425 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 23 | 23 | 0.7471912518 | 0.7469970891 | 0.0001941627 | 0.0001941627 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 2 | 23 | 23 | 0.6899357058 | 0.6900258991 | -0.0000901933 | -0.0000901933 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 15 | 15 | 0.2385315795 | 0.2384663458 | 0.0000652337 | -0.0000652337 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 15 | 15 | 0.5338562723 | 0.5340409471 | -0.0001846748 | 0.0001846748 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 2 | 15 | 15 | 0.6239056996 | 0.6265722887 | -0.0026665892 | 0.0026665892 | 71561 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 23 | 23 | 0.2396283877 | 0.2423352508 | -0.0027068630 | 0.0027068630 | 138225 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 23 | 23 | 0.5320541946 | 0.5320901851 | -0.0000359905 | 0.0000359905 | 1395118 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 2 | 23 | 23 | 0.6343278321 | 0.6336881793 | 0.0006396527 | -0.0006396527 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 15 | 15 | 0.6535030879 | 0.6515968693 | 0.0019062186 | 0.0019062186 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 15 | 15 | 0.7440904840 | 0.7438211697 | 0.0002693143 | 0.0002693143 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 2 | 15 | 15 | 0.6939194679 | 0.6928091924 | 0.0011102755 | 0.0011102755 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 23 | 22 | 0.6611817195 | 0.6624793305 | -0.0012976110 | -0.0012976110 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 23 | 22 | 0.7471912518 | 0.7472569837 | -0.0000657319 | -0.0000657319 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 2 | 23 | 22 | 0.6899357058 | 0.6905334724 | -0.0005977666 | -0.0005977666 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 15 | 15 | 0.2385315795 | 0.2381713385 | 0.0003602410 | -0.0003602410 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 15 | 15 | 0.5338562723 | 0.5349634562 | -0.0011071839 | 0.0011071839 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 2 | 15 | 15 | 0.6239056996 | 0.6249221666 | -0.0010164670 | 0.0010164670 | 71561 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 23 | 22 | 0.2396283877 | 0.2414810496 | -0.0018526619 | 0.0018526619 | 138225 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 23 | 22 | 0.5320541946 | 0.5338030102 | -0.0017488156 | 0.0017488156 | 1395118 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 2 | 23 | 22 | 0.6343278321 | 0.6277354233 | 0.0065924087 | -0.0065924087 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 18 | 14 | 0.6575247855 | 0.6420249250 | 0.0154998605 | 0.0154998605 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 18 | 14 | 0.7455524817 | 0.7440237852 | 0.0015286965 | 0.0015286965 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 2 | 18 | 14 | 0.6953233059 | 0.6850354343 | 0.0102878716 | 0.0102878716 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 21 | 19 | 0.6606528861 | 0.6512374426 | 0.0094154436 | 0.0094154436 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 21 | 19 | 0.7469696096 | 0.7454487075 | 0.0015209021 | 0.0015209021 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 2 | 21 | 19 | 0.6899710712 | 0.6840714412 | 0.0058996300 | 0.0058996300 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 18 | 14 | 0.2375284961 | 0.2399499496 | -0.0024214535 | 0.0024214535 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 18 | 14 | 0.5377300475 | 0.5332267238 | 0.0045033237 | -0.0045033237 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 2 | 18 | 14 | 0.6230591551 | 0.6267307719 | -0.0036716168 | 0.0036716168 | 71561 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 21 | 19 | 0.2391742904 | 0.2385106025 | 0.0006636879 | -0.0006636879 | 138225 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 21 | 19 | 0.5320451333 | 0.5330867348 | -0.0010416015 | 0.0010416015 | 1395118 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 2 | 21 | 19 | 0.6348465709 | 0.6288543954 | 0.0059921755 | -0.0059921755 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 18 | 15 | 0.6575247855 | 0.6535030879 | 0.0040216976 | 0.0040216976 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 18 | 15 | 0.7455524817 | 0.7440904840 | 0.0014619977 | 0.0014619977 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 2 | 18 | 15 | 0.6953233059 | 0.6939194679 | 0.0014038380 | 0.0014038380 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 21 | 23 | 0.6606528861 | 0.6611817195 | -0.0005288333 | -0.0005288333 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 21 | 23 | 0.7469696096 | 0.7471912518 | -0.0002216422 | -0.0002216422 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 2 | 21 | 23 | 0.6899710712 | 0.6899357058 | 0.0000353654 | 0.0000353654 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 18 | 15 | 0.2375284961 | 0.2385315795 | -0.0010030834 | 0.0010030834 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 18 | 15 | 0.5377300475 | 0.5338562723 | 0.0038737752 | -0.0038737752 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 2 | 18 | 15 | 0.6230591551 | 0.6239056996 | -0.0008465445 | 0.0008465445 | 71561 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 21 | 23 | 0.2391742904 | 0.2396283877 | -0.0004540974 | 0.0004540974 | 138225 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 21 | 23 | 0.5320451333 | 0.5320541946 | -0.0000090613 | 0.0000090613 | 1395118 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 2 | 21 | 23 | 0.6348465709 | 0.6343278321 | 0.0005187389 | -0.0005187389 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 18 | 18 | 0.6575247855 | 0.6533896112 | 0.0041351743 | 0.0041351743 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 18 | 18 | 0.7455524817 | 0.7457766199 | -0.0002241382 | -0.0002241382 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 2 | 18 | 18 | 0.6953233059 | 0.6980620646 | -0.0027387587 | -0.0027387587 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 21 | 20 | 0.6606528861 | 0.6611993485 | -0.0005464624 | -0.0005464624 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 21 | 20 | 0.7469696096 | 0.7464374587 | 0.0005321509 | 0.0005321509 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 2 | 21 | 20 | 0.6899710712 | 0.6890215443 | 0.0009495268 | 0.0009495268 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 18 | 18 | 0.2375284961 | 0.2386498126 | -0.0011213165 | 0.0011213165 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 18 | 18 | 0.5377300475 | 0.5336549242 | 0.0040751233 | -0.0040751233 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 2 | 18 | 18 | 0.6230591551 | 0.6214392590 | 0.0016198961 | -0.0016198961 | 71561 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 21 | 20 | 0.2391742904 | 0.2426773553 | -0.0035030650 | 0.0035030650 | 138225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 21 | 20 | 0.5320451333 | 0.5327904183 | -0.0007452850 | 0.0007452850 | 1395118 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 2 | 21 | 20 | 0.6348465709 | 0.6269532723 | 0.0078932986 | -0.0078932986 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 18 | 15 | 0.6575247855 | 0.6540135979 | 0.0035111877 | 0.0035111877 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 18 | 15 | 0.7455524817 | 0.7436155830 | 0.0019368987 | 0.0019368987 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 2 | 18 | 15 | 0.6953233059 | 0.6935320634 | 0.0017912425 | 0.0017912425 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 21 | 23 | 0.6606528861 | 0.6602483012 | 0.0004045849 | 0.0004045849 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 21 | 23 | 0.7469696096 | 0.7470800474 | -0.0001104378 | -0.0001104378 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 2 | 21 | 23 | 0.6899710712 | 0.6903799531 | -0.0004088819 | -0.0004088819 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 18 | 15 | 0.2375284961 | 0.2377844175 | -0.0002559214 | 0.0002559214 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 18 | 15 | 0.5377300475 | 0.5346780431 | 0.0030520044 | -0.0030520044 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 2 | 18 | 15 | 0.6230591551 | 0.6255125169 | -0.0024533618 | 0.0024533618 | 71561 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 21 | 23 | 0.2391742904 | 0.2397087926 | -0.0005345022 | 0.0005345022 | 138225 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 21 | 23 | 0.5320451333 | 0.5316057847 | 0.0004393486 | -0.0004393486 | 1395118 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 2 | 21 | 23 | 0.6348465709 | 0.6323855374 | 0.0024610335 | -0.0024610335 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6776940192 | 0.6637569756 | 0.0139370436 | 0.0139370436 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7328811619 | 0.7443955390 | -0.0115143771 | -0.0115143771 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 14 | 14 | 0.7007931685 | 0.6889728145 | 0.0118203540 | 0.0118203540 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6587048480 | 0.6564449846 | 0.0022598634 | 0.0022598634 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.7426418621 | 0.7453116287 | -0.0026697666 | -0.0026697666 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 14 | 17 | 0.6855266380 | 0.6888841023 | -0.0033574643 | -0.0033574643 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.2365469287 | 0.2383582355 | -0.0018113068 | 0.0018113068 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.5424356708 | 0.5364267723 | 0.0060088985 | -0.0060088985 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 14 | 14 | 0.6193373020 | 0.6247381442 | -0.0054008422 | 0.0054008422 | 71561 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.2376443652 | 0.2398807644 | -0.0022363992 | 0.0022363992 | 138225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.5383212871 | 0.5357549829 | 0.0025663042 | -0.0025663042 | 1395118 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 14 | 17 | 0.6341200937 | 0.6256270028 | 0.0084930908 | -0.0084930908 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6637569756 | 0.0002073061 | 0.0002073061 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7443955390 | 0.0025044752 | 0.0025044752 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.6889728145 | 0.0091383292 | 0.0091383292 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6647265134 | 0.6564449846 | 0.0082815288 | 0.0082815288 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7450196836 | 0.7453116287 | -0.0002919451 | -0.0002919451 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 2 | 17 | 17 | 0.6924523621 | 0.6888841023 | 0.0035682599 | 0.0035682599 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2383582355 | 0.0003917382 | -0.0003917382 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5364267723 | -0.0014793129 | 0.0014793129 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6247381442 | -0.0056487848 | 0.0056487848 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.2367260798 | 0.2398807644 | -0.0031546846 | 0.0031546846 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.5340711652 | 0.5357549829 | -0.0016838177 | 0.0016838177 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 17 | 17 | 0.6249945203 | 0.6256270028 | -0.0006324826 | 0.0006324826 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6637569756 | 0.0002073061 | 0.0002073061 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7443955390 | 0.0025044752 | 0.0025044752 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.6889728145 | 0.0091383292 | 0.0091383292 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6647265134 | 0.6564449846 | 0.0082815288 | 0.0082815288 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7450196836 | 0.7453116287 | -0.0002919451 | -0.0002919451 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 17 | 17 | 0.6924523621 | 0.6888841023 | 0.0035682599 | 0.0035682599 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2383582355 | 0.0003917382 | -0.0003917382 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5364267723 | -0.0014793129 | 0.0014793129 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6247381442 | -0.0056487848 | 0.0056487848 | 71561 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.2367260798 | 0.2398807644 | -0.0031546846 | 0.0031546846 | 138225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.5340711652 | 0.5357549829 | -0.0016838177 | 0.0016838177 | 1395118 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 17 | 17 | 0.6249945203 | 0.6256270028 | -0.0006324826 | 0.0006324826 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6776940192 | -0.0137297375 | -0.0137297375 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7328811619 | 0.0140188524 | 0.0140188524 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.7007931685 | -0.0026820247 | -0.0026820247 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6647265134 | 0.6587048480 | 0.0060216654 | 0.0060216654 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7450196836 | 0.7426418621 | 0.0023778215 | 0.0023778215 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 2 | 17 | 14 | 0.6924523621 | 0.6855266380 | 0.0069257242 | 0.0069257242 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2365469287 | 0.0022030451 | -0.0022030451 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5424356708 | -0.0074882113 | 0.0074882113 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6193373020 | -0.0002479426 | 0.0002479426 | 71561 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.2367260798 | 0.2376443652 | -0.0009182854 | 0.0009182854 | 138225 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.5340711652 | 0.5383212871 | -0.0042501219 | 0.0042501219 | 1395118 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 2 | 17 | 14 | 0.6249945203 | 0.6341200937 | -0.0091255734 | 0.0091255734 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6776940192 | -0.0137297375 | -0.0137297375 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7328811619 | 0.0140188524 | 0.0140188524 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.7007931685 | -0.0026820247 | -0.0026820247 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6647265134 | 0.6587048480 | 0.0060216654 | 0.0060216654 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7450196836 | 0.7426418621 | 0.0023778215 | 0.0023778215 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 2 | 17 | 14 | 0.6924523621 | 0.6855266380 | 0.0069257242 | 0.0069257242 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2365469287 | 0.0022030451 | -0.0022030451 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5424356708 | -0.0074882113 | 0.0074882113 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6193373020 | -0.0002479426 | 0.0002479426 | 71561 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.2367260798 | 0.2376443652 | -0.0009182854 | 0.0009182854 | 138225 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.5340711652 | 0.5383212871 | -0.0042501219 | 0.0042501219 | 1395118 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 2 | 17 | 14 | 0.6249945203 | 0.6341200937 | -0.0091255734 | 0.0091255734 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.6639642817 | 0.6776940192 | -0.0137297375 | -0.0137297375 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7469000143 | 0.7328811619 | 0.0140188524 | 0.0140188524 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 2 | 24 | 14 | 0.6981111437 | 0.7007931685 | -0.0026820247 | -0.0026820247 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6647265134 | 0.6587048480 | 0.0060216654 | 0.0060216654 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7450196836 | 0.7426418621 | 0.0023778215 | 0.0023778215 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 2 | 17 | 14 | 0.6924523621 | 0.6855266380 | 0.0069257242 | 0.0069257242 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.2387499738 | 0.2365469287 | 0.0022030451 | -0.0022030451 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.5349474595 | 0.5424356708 | -0.0074882113 | 0.0074882113 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 2 | 24 | 14 | 0.6190893594 | 0.6193373020 | -0.0002479426 | 0.0002479426 | 71561 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.2367260798 | 0.2376443652 | -0.0009182854 | 0.0009182854 | 138225 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.5340711652 | 0.5383212871 | -0.0042501219 | 0.0042501219 | 1395118 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 2 | 17 | 14 | 0.6249945203 | 0.6341200937 | -0.0091255734 | 0.0091255734 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 13 | 0.6639642817 | 0.6623964692 | 0.0015678126 | 0.0015678126 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 13 | 0.7469000143 | 0.7422436662 | 0.0046563481 | 0.0046563481 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 2 | 24 | 13 | 0.6981111437 | 0.6900191738 | 0.0080919699 | 0.0080919699 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6647265134 | 0.6575551689 | 0.0071713445 | 0.0071713445 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7450196836 | 0.7452780667 | -0.0002583830 | -0.0002583830 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 2 | 17 | 17 | 0.6924523621 | 0.6895693338 | 0.0028830283 | 0.0028830283 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 13 | 0.2387499738 | 0.2371806185 | 0.0015693553 | -0.0015693553 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 13 | 0.5349474595 | 0.5360346082 | -0.0010871487 | 0.0010871487 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 2 | 24 | 13 | 0.6190893594 | 0.6256326841 | -0.0065433248 | 0.0065433248 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.2367260798 | 0.2405308051 | -0.0038047253 | 0.0038047253 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.5340711652 | 0.5378377889 | -0.0037666237 | 0.0037666237 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 2 | 17 | 17 | 0.6249945203 | 0.6248269879 | 0.0001675324 | -0.0001675324 | 71561 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8170257948 | 0.8177807024 | -0.0007549076 | -0.0007549076 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8200716442 | 0.8197943044 | 0.0002773398 | 0.0002773398 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8076538713 | 0.8073895312 | 0.0002643401 | 0.0002643401 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8131076696 | 0.8131653031 | -0.0000576335 | -0.0000576335 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8179564189 | 0.8192114825 | -0.0012550636 | -0.0012550636 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8090116577 | 0.8073075638 | 0.0017040939 | 0.0017040939 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4834952438 | 0.4811632691 | 0.0023319748 | -0.0023319748 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4707361018 | 0.4720132527 | -0.0012771509 | 0.0012771509 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4833930617 | 0.4813430583 | 0.0020500034 | -0.0020500034 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4822691637 | 0.4823951269 | -0.0001259632 | 0.0001259632 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4745457152 | 0.4769970556 | -0.0024513404 | 0.0024513404 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4809243470 | 0.4838943103 | -0.0029699633 | 0.0029699633 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8170257948 | 0.8170390218 | -0.0000132270 | -0.0000132270 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8200716442 | 0.8202541341 | -0.0001824899 | -0.0001824899 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8076538713 | 0.8078092718 | -0.0001554005 | -0.0001554005 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8131076696 | 0.8131614123 | -0.0000537428 | -0.0000537428 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8179564189 | 0.8177824084 | 0.0001740105 | 0.0001740105 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8090116577 | 0.8090329365 | -0.0000212788 | -0.0000212788 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4834952438 | 0.4856526400 | -0.0021573962 | 0.0021573962 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4707361018 | 0.4706804413 | 0.0000556604 | -0.0000556604 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4833930617 | 0.4838224398 | -0.0004293781 | 0.0004293781 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4822691637 | 0.4835105867 | -0.0012414231 | 0.0012414231 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4745457152 | 0.4774255607 | -0.0028798455 | 0.0028798455 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4809243470 | 0.4784183324 | 0.0025060146 | -0.0025060146 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8153539858 | 0.0016007270 | 0.0016007270 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8208114875 | -0.0005338795 | -0.0005338795 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8093328338 | -0.0008421018 | -0.0008421018 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8119638566 | 0.8137735217 | -0.0018096650 | -0.0018096650 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8177872569 | 0.8208059256 | -0.0030186686 | -0.0030186686 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8089936537 | 0.8085805014 | 0.0004131523 | 0.0004131523 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4727478311 | 0.0137153749 | -0.0137153749 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4609771930 | 0.0102923129 | -0.0102923129 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4735172945 | 0.0104315731 | -0.0104315731 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4833026404 | 0.4732379341 | 0.0100647063 | -0.0100647063 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4719154195 | 0.4606246129 | 0.0112908066 | -0.0112908066 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4771369064 | 0.4739495769 | 0.0031873295 | -0.0031873295 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8170257948 | -0.0000710820 | -0.0000710820 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8200716442 | 0.0002059638 | 0.0002059638 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8076538713 | 0.0008368607 | 0.0008368607 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8119638566 | 0.8131076696 | -0.0011438129 | -0.0011438129 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8177872569 | 0.8179564189 | -0.0001691620 | -0.0001691620 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8089936537 | 0.8090116577 | -0.0000180040 | -0.0000180040 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4834952438 | 0.0029679621 | -0.0029679621 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4707361018 | 0.0005334041 | -0.0005334041 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4833930617 | 0.0005558059 | -0.0005558059 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4833026404 | 0.4822691637 | 0.0010334768 | -0.0010334768 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4719154195 | 0.4745457152 | -0.0026302957 | 0.0026302957 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4771369064 | 0.4809243470 | -0.0037874407 | 0.0037874407 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8168175110 | 0.0001372018 | 0.0001372018 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8208595016 | -0.0005818936 | -0.0005818936 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8085182801 | -0.0000275481 | -0.0000275481 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8119638566 | 0.8115141507 | 0.0004497059 | 0.0004497059 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8177872569 | 0.8176556490 | 0.0001316079 | 0.0001316079 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8089936537 | 0.8082369107 | 0.0007567430 | 0.0007567430 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4861913553 | 0.0002718507 | -0.0002718507 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4694223722 | 0.0018471337 | -0.0018471337 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4833983044 | 0.0005505632 | -0.0005505632 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4833026404 | 0.4869099563 | -0.0036073159 | 0.0036073159 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4719154195 | 0.4738877375 | -0.0019723180 | 0.0019723180 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4771369064 | 0.4791734649 | -0.0020365586 | 0.0020365586 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8157153385 | 0.0012393743 | 0.0012393743 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8204980879 | -0.0002204800 | -0.0002204800 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8083768467 | 0.0001138853 | 0.0001138853 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8119638566 | 0.8121139360 | -0.0001500794 | -0.0001500794 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8177872569 | 0.8178113155 | -0.0000240585 | -0.0000240585 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8089936537 | 0.8082342845 | 0.0007593692 | 0.0007593692 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4792222809 | 0.0072409251 | -0.0072409251 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4681724606 | 0.0030970453 | -0.0030970453 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4817224370 | 0.0022264306 | -0.0022264306 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4833026404 | 0.4836407554 | -0.0003381150 | 0.0003381150 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4719154195 | 0.4699175843 | 0.0019978352 | -0.0019978352 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4771369064 | 0.4782032818 | -0.0010663754 | 0.0010663754 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8161989547 | 0.8169518526 | -0.0007528978 | -0.0007528978 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8211123994 | 0.8199429470 | 0.0011694524 | 0.0011694524 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8067504343 | 0.8058019515 | 0.0009484827 | 0.0009484827 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8135403776 | 0.8166826061 | -0.0031422285 | -0.0031422285 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8221816480 | 0.8236594704 | -0.0014778224 | -0.0014778224 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8103866801 | 0.8110342194 | -0.0006475393 | -0.0006475393 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4878495221 | 0.4882332719 | -0.0003837498 | 0.0003837498 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4657831297 | 0.4688921157 | -0.0031089859 | 0.0031089859 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4828761001 | 0.4840470442 | -0.0011709441 | 0.0011709441 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4924265710 | 0.4797953061 | 0.0126312649 | -0.0126312649 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4660850961 | 0.4627571827 | 0.0033279135 | -0.0033279135 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4779122296 | 0.4872376203 | -0.0093253907 | 0.0093253907 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8161989547 | 0.8159967515 | 0.0002022032 | 0.0002022032 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8211123994 | 0.8199440496 | 0.0011683498 | 0.0011683498 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8067504343 | 0.8037791338 | 0.0029713005 | 0.0029713005 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8135403776 | 0.8156969532 | -0.0021565756 | -0.0021565756 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8221816480 | 0.8218570490 | 0.0003245990 | 0.0003245990 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8103866801 | 0.8072954484 | 0.0030912317 | 0.0030912317 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4878495221 | 0.4886596352 | -0.0008101131 | 0.0008101131 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4657831297 | 0.4712694316 | -0.0054863019 | 0.0054863019 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4828761001 | 0.4874752267 | -0.0045991266 | 0.0045991266 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4924265710 | 0.4831290945 | 0.0092974765 | -0.0092974765 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4660850961 | 0.4642604121 | 0.0018246840 | -0.0018246840 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4779122296 | 0.4846953582 | -0.0067831286 | 0.0067831286 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8152951136 | 0.8169518526 | -0.0016567389 | -0.0016567389 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8191442581 | 0.8199429470 | -0.0007986890 | -0.0007986890 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8077188195 | 0.8058019515 | 0.0019168679 | 0.0019168679 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8140954069 | 0.8166826061 | -0.0025871993 | -0.0025871993 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8219089508 | 0.8236594704 | -0.0017505196 | -0.0017505196 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8095857036 | 0.8110342194 | -0.0014485158 | -0.0014485158 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4905450823 | 0.4882332719 | 0.0023118104 | -0.0023118104 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4680997795 | 0.4688921157 | -0.0007923361 | 0.0007923361 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4795064199 | 0.4840470442 | -0.0045406244 | 0.0045406244 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4889899095 | 0.4797953061 | 0.0091946034 | -0.0091946034 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4663388857 | 0.4627571827 | 0.0035817030 | -0.0035817030 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4793924104 | 0.4872376203 | -0.0078452100 | 0.0078452100 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8152951136 | 0.8159967515 | -0.0007016379 | -0.0007016379 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8191442581 | 0.8199440496 | -0.0007997915 | -0.0007997915 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8077188195 | 0.8037791338 | 0.0039396857 | 0.0039396857 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8140954069 | 0.8156969532 | -0.0016015463 | -0.0016015463 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8219089508 | 0.8218570490 | 0.0000519018 | 0.0000519018 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8095857036 | 0.8072954484 | 0.0022902552 | 0.0022902552 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4905450823 | 0.4886596352 | 0.0018854471 | -0.0018854471 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4680997795 | 0.4712694316 | -0.0031696521 | 0.0031696521 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4795064199 | 0.4874752267 | -0.0079688069 | 0.0079688069 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4889899095 | 0.4831290945 | 0.0058608150 | -0.0058608150 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4663388857 | 0.4642604121 | 0.0020784736 | -0.0020784736 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4793924104 | 0.4846953582 | -0.0053029478 | 0.0053029478 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8177389167 | 0.8169518526 | 0.0007870641 | 0.0007870641 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8203116848 | 0.8199429470 | 0.0003687378 | 0.0003687378 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8071090446 | 0.8058019515 | 0.0013070930 | 0.0013070930 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8134941977 | 0.8166826061 | -0.0031884084 | -0.0031884084 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8236529485 | 0.8236594704 | -0.0000065220 | -0.0000065220 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8116776722 | 0.8110342194 | 0.0006434529 | 0.0006434529 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4818701167 | 0.4882332719 | -0.0063631552 | 0.0063631552 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4662832355 | 0.4688921157 | -0.0026088801 | 0.0026088801 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4814056903 | 0.4840470442 | -0.0026413539 | 0.0026413539 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4847532978 | 0.4797953061 | 0.0049579917 | -0.0049579917 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4635236389 | 0.4627571827 | 0.0007664562 | -0.0007664562 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4752432451 | 0.4872376203 | -0.0119943752 | 0.0119943752 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8177389167 | 0.8159967515 | 0.0017421652 | 0.0017421652 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8203116848 | 0.8199440496 | 0.0003676352 | 0.0003676352 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8071090446 | 0.8037791338 | 0.0033299108 | 0.0033299108 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8134941977 | 0.8156969532 | -0.0022027555 | -0.0022027555 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8236529485 | 0.8218570490 | 0.0017958995 | 0.0017958995 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8116776722 | 0.8072954484 | 0.0043822238 | 0.0043822238 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4818701167 | 0.4886596352 | -0.0067895185 | 0.0067895185 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4662832355 | 0.4712694316 | -0.0049861961 | 0.0049861961 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4814056903 | 0.4874752267 | -0.0060695364 | 0.0060695364 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4847532978 | 0.4831290945 | 0.0016242033 | -0.0016242033 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4635236389 | 0.4642604121 | -0.0007367732 | 0.0007367732 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4752432451 | 0.4846953582 | -0.0094521131 | 0.0094521131 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8169518526 | -0.0014509971 | -0.0014509971 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8199429470 | -0.0001746191 | -0.0001746191 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8058019515 | -0.0000450853 | -0.0000450853 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8166826061 | -0.0006846833 | -0.0006846833 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8236594704 | -0.0018029269 | -0.0018029269 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8110342194 | -0.0003884383 | -0.0003884383 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4882332719 | 0.0020135695 | -0.0020135695 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4688921157 | 0.0006673312 | -0.0006673312 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4840470442 | 0.0045404958 | -0.0045404958 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4797953061 | 0.0015562184 | -0.0015562184 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4627571827 | 0.0014707490 | -0.0014707490 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4872376203 | -0.0097904814 | 0.0097904814 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8159967515 | -0.0004958961 | -0.0004958961 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8199440496 | -0.0001757217 | -0.0001757217 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8037791338 | 0.0019777325 | 0.0019777325 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8156969532 | 0.0003009696 | 0.0003009696 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8218570490 | -0.0000005055 | -0.0000005055 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8072954484 | 0.0033503327 | 0.0033503327 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4886596352 | 0.0015872062 | -0.0015872062 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4712694316 | -0.0017099847 | 0.0017099847 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4874752267 | 0.0011123132 | -0.0011123132 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4831290945 | -0.0017775700 | 0.0017775700 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4642604121 | -0.0000324804 | 0.0000324804 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4846953582 | -0.0072482192 | 0.0072482192 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8161989547 | -0.0006980993 | -0.0006980993 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8211123994 | -0.0013440714 | -0.0013440714 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8067504343 | -0.0009935680 | -0.0009935680 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8135403776 | 0.0024575452 | 0.0024575452 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8221816480 | -0.0003251045 | -0.0003251045 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8103866801 | 0.0002591010 | 0.0002591010 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4878495221 | 0.0023973192 | -0.0023973192 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4657831297 | 0.0037763171 | -0.0037763171 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4828761001 | 0.0057114399 | -0.0057114399 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4924265710 | -0.0110750465 | 0.0110750465 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4660850961 | -0.0018571644 | 0.0018571644 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4779122296 | -0.0004650907 | 0.0004650907 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8152951136 | 0.0002057419 | 0.0002057419 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8191442581 | 0.0006240699 | 0.0006240699 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8077188195 | -0.0019619532 | -0.0019619532 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8140954069 | 0.0019025159 | 0.0019025159 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8219089508 | -0.0000524073 | -0.0000524073 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8095857036 | 0.0010600775 | 0.0010600775 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4905450823 | -0.0002982410 | 0.0002982410 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4680997795 | 0.0014596673 | -0.0014596673 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4795064199 | 0.0090811201 | -0.0090811201 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4889899095 | -0.0076383850 | 0.0076383850 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4663388857 | -0.0021109540 | 0.0021109540 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4793924104 | -0.0019452714 | 0.0019452714 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8177389167 | -0.0022380612 | -0.0022380612 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8203116848 | -0.0005433568 | -0.0005433568 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8071090446 | -0.0013521783 | -0.0013521783 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8134941977 | 0.0025037251 | 0.0025037251 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8236529485 | -0.0017964049 | -0.0017964049 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8116776722 | -0.0010318911 | -0.0010318911 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4818701167 | 0.0083767246 | -0.0083767246 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4662832355 | 0.0032762113 | -0.0032762113 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4814056903 | 0.0071818496 | -0.0071818496 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4847532978 | -0.0034017733 | 0.0034017733 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4635236389 | 0.0007042928 | -0.0007042928 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4752432451 | 0.0022038938 | -0.0022038938 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8155008555 | 0.8168792413 | -0.0013783858 | -0.0013783858 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8197683279 | 0.8212291695 | -0.0014608415 | -0.0014608415 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.8057568663 | 0.8049316663 | 0.0008252000 | 0.0008252000 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8159979228 | 0.8134143624 | 0.0025835604 | 0.0025835604 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8218565435 | 0.8214204984 | 0.0004360451 | 0.0004360451 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 2 | 24 | 24 | 0.8106457811 | 0.8088564871 | 0.0017892940 | 0.0017892940 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4902468414 | 0.4862169029 | 0.0040299384 | -0.0040299384 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4695594469 | 0.4710707656 | -0.0015113187 | 0.0015113187 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.4885875400 | 0.4868481840 | 0.0017393559 | -0.0017393559 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4813515245 | 0.4856909957 | -0.0043394713 | 0.0043394713 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4642279317 | 0.4660238431 | -0.0017959114 | 0.0017959114 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 2 | 24 | 24 | 0.4774471389 | 0.4825046834 | -0.0050575445 | 0.0050575445 | 260495 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076278321 | 0.8080575046 | -0.0004296725 | -0.0004296725 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8127107218 | 0.8130435988 | -0.0003328770 | -0.0003328770 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8022200228 | 0.8028429701 | -0.0006229474 | -0.0006229474 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 14 | 12 | 0.8102749114 | 0.8069816087 | 0.0032933028 | 0.0032933028 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 14 | 12 | 0.8149923816 | 0.8134716989 | 0.0015206827 | 0.0015206827 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 2 | 14 | 12 | 0.8036498709 | 0.8032697360 | 0.0003801349 | 0.0003801349 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4800711811 | 0.4803852413 | -0.0003140602 | 0.0003140602 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4694728731 | 0.4677868810 | 0.0016859921 | -0.0016859921 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4795800887 | 0.4790283351 | 0.0005517536 | -0.0005517536 | 260495 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 14 | 12 | 0.4778460774 | 0.4834862848 | -0.0056402075 | 0.0056402075 | 124759 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 14 | 12 | 0.4666628767 | 0.4691650948 | -0.0025022181 | 0.0025022181 | 254879 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 2 | 14 | 12 | 0.4786904493 | 0.4786732854 | 0.0000171639 | -0.0000171639 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076278321 | 0.8076834591 | -0.0000556270 | -0.0000556270 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8127107218 | 0.8126794672 | 0.0000312546 | 0.0000312546 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8022200228 | 0.8022171072 | 0.0000029155 | 0.0000029155 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 14 | 12 | 0.8102749114 | 0.8071921845 | 0.0030827270 | 0.0030827270 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 14 | 12 | 0.8149923816 | 0.8137127938 | 0.0012795878 | 0.0012795878 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 2 | 14 | 12 | 0.8036498709 | 0.8033166847 | 0.0003331862 | 0.0003331862 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4800711811 | 0.4799816567 | 0.0000895244 | -0.0000895244 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4694728731 | 0.4693833734 | 0.0000894998 | -0.0000894998 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4795800887 | 0.4796046869 | -0.0000245982 | 0.0000245982 | 260495 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 14 | 12 | 0.4778460774 | 0.4822994144 | -0.0044533370 | 0.0044533370 | 124759 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 14 | 12 | 0.4666628767 | 0.4680985331 | -0.0014356564 | 0.0014356564 | 254879 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 2 | 14 | 12 | 0.4786904493 | 0.4789632076 | -0.0002727582 | 0.0002727582 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076898769 | 0.8074790323 | 0.0002108446 | 0.0002108446 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8127024521 | 0.8133730216 | -0.0006705695 | -0.0006705695 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8022116233 | 0.8038515089 | -0.0016398857 | -0.0016398857 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 16 | 16 | 0.8132298480 | 0.8118865873 | 0.0013432607 | 0.0013432607 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 16 | 16 | 0.8167210616 | 0.8170428459 | -0.0003217842 | -0.0003217842 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 2 | 16 | 16 | 0.8049386165 | 0.8065194238 | -0.0015808073 | -0.0015808073 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4799775951 | 0.4789990316 | 0.0009785635 | -0.0009785635 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4694188145 | 0.4667839531 | 0.0026348615 | -0.0026348615 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4795692555 | 0.4774954353 | 0.0020738202 | -0.0020738202 | 260495 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 16 | 16 | 0.4755953722 | 0.4750360318 | 0.0005593404 | -0.0005593404 | 124759 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 16 | 16 | 0.4647717612 | 0.4633626912 | 0.0014090700 | -0.0014090700 | 254879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 2 | 16 | 16 | 0.4787169173 | 0.4749545909 | 0.0037623263 | -0.0037623263 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076898769 | 0.8076278321 | 0.0000620448 | 0.0000620448 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8127024521 | 0.8127107218 | -0.0000082697 | -0.0000082697 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8022116233 | 0.8022200228 | -0.0000083995 | -0.0000083995 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 16 | 14 | 0.8132298480 | 0.8102749114 | 0.0029549365 | 0.0029549365 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 16 | 14 | 0.8167210616 | 0.8149923816 | 0.0017286800 | 0.0017286800 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 2 | 16 | 14 | 0.8049386165 | 0.8036498709 | 0.0012887456 | 0.0012887456 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4799775951 | 0.4800711811 | -0.0000935860 | 0.0000935860 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4694188145 | 0.4694728731 | -0.0000540586 | 0.0000540586 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4795692555 | 0.4795800887 | -0.0000108332 | 0.0000108332 | 260495 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 16 | 14 | 0.4755953722 | 0.4778460774 | -0.0022507052 | 0.0022507052 | 124759 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 16 | 14 | 0.4647717612 | 0.4666628767 | -0.0018911154 | 0.0018911154 | 254879 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 2 | 16 | 14 | 0.4787169173 | 0.4786904493 | 0.0000264679 | -0.0000264679 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076898769 | 0.8077091100 | -0.0000192331 | -0.0000192331 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8127024521 | 0.8127057773 | -0.0000033252 | -0.0000033252 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8022116233 | 0.8022274483 | -0.0000158250 | -0.0000158250 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 16 | 14 | 0.8132298480 | 0.8101843008 | 0.0030455472 | 0.0030455472 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 16 | 14 | 0.8167210616 | 0.8149601864 | 0.0017608752 | 0.0017608752 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 2 | 16 | 14 | 0.8049386165 | 0.8036856831 | 0.0012529335 | 0.0012529335 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4799775951 | 0.4799442941 | 0.0000333010 | -0.0000333010 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4694188145 | 0.4693744518 | 0.0000443627 | -0.0000443627 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4795692555 | 0.4795950543 | -0.0000257989 | 0.0000257989 | 260495 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 16 | 14 | 0.4755953722 | 0.4780078584 | -0.0024124861 | 0.0024124861 | 124759 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 16 | 14 | 0.4647717612 | 0.4667627029 | -0.0019909417 | 0.0019909417 | 254879 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 2 | 16 | 14 | 0.4787169173 | 0.4786644569 | 0.0000524603 | -0.0000524603 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 12 | 14 | 0.8076898769 | 0.8097489249 | -0.0020590480 | -0.0020590480 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 12 | 14 | 0.8127024521 | 0.8133884253 | -0.0006859732 | -0.0006859732 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 2 | 12 | 14 | 0.8022116233 | 0.8038668740 | -0.0016552507 | -0.0016552507 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 16 | 16 | 0.8132298480 | 0.8129960516 | 0.0002337964 | 0.0002337964 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 16 | 16 | 0.8167210616 | 0.8172046272 | -0.0004835655 | -0.0004835655 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 2 | 16 | 16 | 0.8049386165 | 0.8052220787 | -0.0002834622 | -0.0002834622 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 12 | 14 | 0.4799775951 | 0.4795810042 | 0.0003965909 | -0.0003965909 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 12 | 14 | 0.4694188145 | 0.4676728123 | 0.0017460022 | -0.0017460022 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 2 | 12 | 14 | 0.4795692555 | 0.4785193402 | 0.0010499153 | -0.0010499153 | 260495 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 16 | 16 | 0.4755953722 | 0.4752886552 | 0.0003067170 | -0.0003067170 | 124759 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 16 | 16 | 0.4647717612 | 0.4636185361 | 0.0011532251 | -0.0011532251 | 254879 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 2 | 16 | 16 | 0.4787169173 | 0.4772819150 | 0.0014350022 | -0.0014350022 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8092381955 | 0.8089319545 | 0.0003062410 | 0.0003062410 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8136288798 | 0.8137468907 | -0.0001180109 | -0.0001180109 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 2 | 13 | 13 | 0.8037921387 | 0.8037593661 | 0.0000327726 | 0.0000327726 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8091783262 | 0.8087583161 | 0.0004200101 | 0.0004200101 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8132946297 | 0.8137326371 | -0.0004380075 | -0.0004380075 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8032316077 | 0.8037315185 | -0.0004999107 | -0.0004999107 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.4783979861 | 0.4785776701 | -0.0001796840 | 0.0001796840 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4680930483 | 0.4677432916 | 0.0003497567 | -0.0003497567 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 13 | 13 | 0.4792103127 | 0.4791874178 | 0.0000228949 | -0.0000228949 | 260495 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4773974194 | 0.4780667926 | -0.0006693732 | 0.0006693732 | 124759 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675312232 | 0.4674790470 | 0.0000521762 | -0.0000521762 | 254879 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4785421301 | 0.4794191402 | -0.0008770101 | 0.0008770101 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8092381955 | 0.8076412387 | 0.0015969568 | 0.0015969568 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8136288798 | 0.8130042502 | 0.0006246295 | 0.0006246295 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 13 | 12 | 0.8037921387 | 0.8027624927 | 0.0010296460 | 0.0010296460 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8091783262 | 0.8085345017 | 0.0006438245 | 0.0006438245 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8132946297 | 0.8135011547 | -0.0002065250 | -0.0002065250 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8032316077 | 0.8036041888 | -0.0003725810 | -0.0003725810 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.4783979861 | 0.4819178358 | -0.0035198497 | 0.0035198497 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4680930483 | 0.4690009195 | -0.0009078712 | 0.0009078712 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 13 | 12 | 0.4792103127 | 0.4793474403 | -0.0001371276 | 0.0001371276 | 260495 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4773974194 | 0.4782130755 | -0.0008156561 | 0.0008156561 | 124759 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675312232 | 0.4679317397 | -0.0004005165 | 0.0004005165 | 254879 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4785421301 | 0.4782598259 | 0.0002823042 | -0.0002823042 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8092088416 | 0.8089319545 | 0.0002768872 | 0.0002768872 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8136932394 | 0.8137468907 | -0.0000536513 | -0.0000536513 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 2 | 13 | 13 | 0.8038328347 | 0.8037593661 | 0.0000734687 | 0.0000734687 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8089461520 | 0.8087583161 | 0.0001878359 | 0.0001878359 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8135801712 | 0.8137326371 | -0.0001524659 | -0.0001524659 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8036677306 | 0.8037315185 | -0.0000637879 | -0.0000637879 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.4780451806 | 0.4785776701 | -0.0005324895 | 0.0005324895 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4684387562 | 0.4677432916 | 0.0006954646 | -0.0006954646 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 13 | 13 | 0.4787881584 | 0.4791874178 | -0.0003992594 | 0.0003992594 | 260495 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4777845868 | 0.4780667926 | -0.0002822058 | 0.0002822058 | 124759 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4673562907 | 0.4674790470 | -0.0001227562 | 0.0001227562 | 254879 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4789986992 | 0.4794191402 | -0.0004204410 | 0.0004204410 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8092088416 | 0.8076412387 | 0.0015676029 | 0.0015676029 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8136932394 | 0.8130042502 | 0.0006889892 | 0.0006889892 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 13 | 12 | 0.8038328347 | 0.8027624927 | 0.0010703421 | 0.0010703421 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8089461520 | 0.8085345017 | 0.0004116503 | 0.0004116503 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8135801712 | 0.8135011547 | 0.0000790165 | 0.0000790165 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8036677306 | 0.8036041888 | 0.0000635418 | 0.0000635418 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.4780451806 | 0.4819178358 | -0.0038726552 | 0.0038726552 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4684387562 | 0.4690009195 | -0.0005621633 | 0.0005621633 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 13 | 12 | 0.4787881584 | 0.4793474403 | -0.0005592819 | 0.0005592819 | 260495 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4777845868 | 0.4782130755 | -0.0004284887 | 0.0004284887 | 124759 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4673562907 | 0.4679317397 | -0.0005754490 | 0.0005754490 | 254879 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4789986992 | 0.4782598259 | 0.0007388734 | -0.0007388734 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8092338171 | 0.8089319545 | 0.0003018626 | 0.0003018626 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8137544007 | 0.8137468907 | 0.0000075100 | 0.0000075100 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 2 | 13 | 13 | 0.8037451074 | 0.8037593661 | -0.0000142587 | -0.0000142587 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8089461520 | 0.8087583161 | 0.0001878359 | 0.0001878359 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8135801712 | 0.8137326371 | -0.0001524659 | -0.0001524659 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8036677306 | 0.8037315185 | -0.0000637879 | -0.0000637879 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.4780640122 | 0.4785776701 | -0.0005136579 | 0.0005136579 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4678969959 | 0.4677432916 | 0.0001537042 | -0.0001537042 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 13 | 13 | 0.4792375862 | 0.4791874178 | 0.0000501684 | -0.0000501684 | 260495 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4777845868 | 0.4780667926 | -0.0002822058 | 0.0002822058 | 124759 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4673562907 | 0.4674790470 | -0.0001227562 | 0.0001227562 | 254879 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4789986992 | 0.4794191402 | -0.0004204410 | 0.0004204410 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8092338171 | 0.8076412387 | 0.0015925784 | 0.0015925784 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8137544007 | 0.8130042502 | 0.0007501504 | 0.0007501504 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 13 | 12 | 0.8037451074 | 0.8027624927 | 0.0009826147 | 0.0009826147 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8089461520 | 0.8085345017 | 0.0004116503 | 0.0004116503 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8135801712 | 0.8135011547 | 0.0000790165 | 0.0000790165 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8036677306 | 0.8036041888 | 0.0000635418 | 0.0000635418 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.4780640122 | 0.4819178358 | -0.0038538236 | 0.0038538236 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4678969959 | 0.4690009195 | -0.0011039236 | 0.0011039236 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 13 | 12 | 0.4792375862 | 0.4793474403 | -0.0001098541 | 0.0001098541 | 260495 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4777845868 | 0.4782130755 | -0.0004284887 | 0.0004284887 | 124759 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4673562907 | 0.4679317397 | -0.0005754490 | 0.0005754490 | 254879 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4789986992 | 0.4782598259 | 0.0007388734 | -0.0007388734 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8076843096 | 0.8089319545 | -0.0012476448 | -0.0012476448 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8128438387 | 0.8137468907 | -0.0009030519 | -0.0009030519 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 2 | 12 | 13 | 0.8030650002 | 0.8037593661 | -0.0006943658 | -0.0006943658 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8087583161 | -0.0002330809 | -0.0002330809 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8137326371 | -0.0006420197 | -0.0006420197 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8037315185 | -0.0003936508 | -0.0003936508 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.4819283778 | 0.4785776701 | 0.0033507077 | -0.0033507077 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4691391859 | 0.4677432916 | 0.0013958942 | -0.0013958942 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 2 | 12 | 13 | 0.4792607265 | 0.4791874178 | 0.0000733087 | -0.0000733087 | 260495 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4780667926 | 0.0002959363 | -0.0002959363 | 124759 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4674790470 | 0.0000446107 | -0.0000446107 | 254879 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4794191402 | -0.0007528638 | 0.0007528638 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8076843096 | 0.8076412387 | 0.0000430709 | 0.0000430709 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8128438387 | 0.8130042502 | -0.0001604115 | -0.0001604115 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 2 | 12 | 12 | 0.8030650002 | 0.8027624927 | 0.0003025075 | 0.0003025075 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8085345017 | -0.0000092665 | -0.0000092665 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8135011547 | -0.0004105373 | -0.0004105373 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8036041888 | -0.0002663211 | -0.0002663211 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.4819283778 | 0.4819178358 | 0.0000105419 | -0.0000105419 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4691391859 | 0.4690009195 | 0.0001382664 | -0.0001382664 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 2 | 12 | 12 | 0.4792607265 | 0.4793474403 | -0.0000867138 | 0.0000867138 | 260495 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4782130755 | 0.0001496534 | -0.0001496534 | 124759 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4679317397 | -0.0004080820 | 0.0004080820 | 254879 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4782598259 | 0.0004064506 | -0.0004064506 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8076843096 | 0.8092381955 | -0.0015538858 | -0.0015538858 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8128438387 | 0.8136288798 | -0.0007850410 | -0.0007850410 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 2 | 12 | 13 | 0.8030650002 | 0.8037921387 | -0.0007271385 | -0.0007271385 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8091783262 | -0.0006530910 | -0.0006530910 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8132946297 | -0.0002040123 | -0.0002040123 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8032316077 | 0.0001062599 | 0.0001062599 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.4819283778 | 0.4783979861 | 0.0035303917 | -0.0035303917 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4691391859 | 0.4680930483 | 0.0010461376 | -0.0010461376 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 2 | 12 | 13 | 0.4792607265 | 0.4792103127 | 0.0000504138 | -0.0000504138 | 260495 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4773974194 | 0.0009653095 | -0.0009653095 | 124759 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4675312232 | -0.0000075655 | 0.0000075655 | 254879 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4785421301 | 0.0001241464 | -0.0001241464 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8076843096 | 0.8092088416 | -0.0015245320 | -0.0015245320 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8128438387 | 0.8136932394 | -0.0008494006 | -0.0008494006 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 2 | 12 | 13 | 0.8030650002 | 0.8038328347 | -0.0007678345 | -0.0007678345 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8089461520 | -0.0004209168 | -0.0004209168 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8135801712 | -0.0004895538 | -0.0004895538 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8036677306 | -0.0003298629 | -0.0003298629 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.4819283778 | 0.4780451806 | 0.0038831972 | -0.0038831972 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4691391859 | 0.4684387562 | 0.0007004296 | -0.0007004296 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 2 | 12 | 13 | 0.4792607265 | 0.4787881584 | 0.0004725681 | -0.0004725681 | 260495 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4777845868 | 0.0005781421 | -0.0005781421 | 124759 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4673562907 | 0.0001673669 | -0.0001673669 | 254879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4789986992 | -0.0003324228 | 0.0003324228 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8076843096 | 0.8092338171 | -0.0015495074 | -0.0015495074 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8128438387 | 0.8137544007 | -0.0009105619 | -0.0009105619 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 2 | 12 | 13 | 0.8030650002 | 0.8037451074 | -0.0006801072 | -0.0006801072 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8089461520 | -0.0004209168 | -0.0004209168 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8135801712 | -0.0004895538 | -0.0004895538 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8036677306 | -0.0003298629 | -0.0003298629 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.4819283778 | 0.4780640122 | 0.0038643656 | -0.0038643656 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4691391859 | 0.4678969959 | 0.0012421900 | -0.0012421900 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 2 | 12 | 13 | 0.4792607265 | 0.4792375862 | 0.0000231403 | -0.0000231403 | 260495 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4777845868 | 0.0005781421 | -0.0005781421 | 124759 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4673562907 | 0.0001673669 | -0.0001673669 | 254879 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4789986992 | -0.0003324228 | 0.0003324228 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8076843096 | 0.8091973714 | -0.0015130618 | -0.0015130618 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8128438387 | 0.8138096269 | -0.0009657882 | -0.0009657882 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 2 | 12 | 13 | 0.8030650002 | 0.8038294120 | -0.0007644117 | -0.0007644117 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8085252352 | 0.8087929176 | -0.0002676823 | -0.0002676823 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8130906174 | 0.8132833625 | -0.0001927451 | -0.0001927451 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 2 | 13 | 13 | 0.8033378677 | 0.8035940179 | -0.0002561502 | -0.0002561502 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.4819283778 | 0.4783698606 | 0.0035585171 | -0.0035585171 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4691391859 | 0.4680343915 | 0.0011047944 | -0.0011047944 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 2 | 12 | 13 | 0.4792607265 | 0.4785959205 | 0.0006648060 | -0.0006648060 | 260495 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.4783627289 | 0.4779969400 | 0.0003657889 | -0.0003657889 | 124759 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4675236577 | 0.4676669133 | -0.0001432557 | 0.0001432557 | 254879 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 2 | 13 | 13 | 0.4786662765 | 0.4783935590 | 0.0002727175 | -0.0002727175 | 260495 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6429959839 | 0.6350242384 | 0.0079717455 | 0.0079717455 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.7435618676 | 0.7421328100 | 0.0014290576 | 0.0014290576 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 2 | 50 | 50 | 0.6734755689 | 0.6663606730 | 0.0071148959 | 0.0071148959 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6409528791 | 0.6515740371 | -0.0106211580 | -0.0106211580 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.7452831050 | 0.7430549563 | 0.0022281487 | 0.0022281487 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 2 | 50 | 50 | 0.6707322781 | 0.6768231732 | -0.0060908951 | -0.0060908951 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6280590974 | 0.6348380416 | -0.0067789442 | -0.0067789442 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.7419443069 | 0.7433521151 | -0.0014078082 | -0.0014078082 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 2 | 50 | 50 | 0.6818377672 | 0.6809913511 | 0.0008464160 | 0.0008464160 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6280817482 | 0.6360019972 | -0.0079202490 | -0.0079202490 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.7462873663 | 0.7442168263 | 0.0020705399 | 0.0020705399 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 2 | 50 | 50 | 0.6848449744 | 0.6801618749 | 0.0046830994 | 0.0046830994 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.2739057981 | 0.2823798831 | -0.0084740850 | 0.0084740850 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.5388373199 | 0.5453958633 | -0.0065585434 | 0.0065585434 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 2 | 50 | 50 | 0.6952606586 | 0.7167682434 | -0.0215075848 | 0.0215075848 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.2705825776 | 0.2691224609 | 0.0014601166 | -0.0014601166 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.5362619686 | 0.5425935684 | -0.0063315998 | 0.0063315998 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 2 | 50 | 50 | 0.7198828117 | 0.7121165839 | 0.0077662278 | -0.0077662278 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.2727610861 | 0.2756614625 | -0.0029003764 | 0.0029003764 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.5413468273 | 0.5414265162 | -0.0000796888 | 0.0000796888 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 2 | 50 | 50 | 0.6501456296 | 0.6963685189 | -0.0462228893 | 0.0462228893 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.2756670474 | 0.2758420942 | -0.0001750468 | 0.0001750468 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.5387922651 | 0.5395315704 | -0.0007393053 | 0.0007393053 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 2 | 50 | 50 | 0.6833202917 | 0.6879693863 | -0.0046490947 | 0.0046490947 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6429959839 | 0.6380037760 | 0.0049922079 | 0.0049922079 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.7435618676 | 0.7432058396 | 0.0003560280 | 0.0003560280 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 2 | 50 | 50 | 0.6734755689 | 0.6595708573 | 0.0139047116 | 0.0139047116 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6409528791 | 0.6444976744 | -0.0035447953 | -0.0035447953 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.7452831050 | 0.7436620171 | 0.0016210879 | 0.0016210879 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 2 | 50 | 50 | 0.6707322781 | 0.6712406919 | -0.0005084138 | -0.0005084138 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6280590974 | 0.6474386069 | -0.0193795096 | -0.0193795096 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.7419443069 | 0.7440699407 | -0.0021256338 | -0.0021256338 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 2 | 50 | 50 | 0.6818377672 | 0.6846315202 | -0.0027937531 | -0.0027937531 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6280817482 | 0.6256639934 | 0.0024177548 | 0.0024177548 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.7462873663 | 0.7444664176 | 0.0018209487 | 0.0018209487 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 2 | 50 | 50 | 0.6848449744 | 0.6811452740 | 0.0036997004 | 0.0036997004 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.2739057981 | 0.2861735605 | -0.0122677623 | 0.0122677623 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.5388373199 | 0.5427295539 | -0.0038922339 | 0.0038922339 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 2 | 50 | 50 | 0.6952606586 | 0.7192367101 | -0.0239760515 | 0.0239760515 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.2705825776 | 0.2807877755 | -0.0102051979 | 0.0102051979 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.5362619686 | 0.5394311143 | -0.0031691457 | 0.0031691457 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 2 | 50 | 50 | 0.7198828117 | 0.7182934935 | 0.0015893182 | -0.0015893182 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.2727610861 | 0.2574521550 | 0.0153089312 | -0.0153089312 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.5413468273 | 0.5438262981 | -0.0024794707 | 0.0024794707 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 2 | 50 | 50 | 0.6501456296 | 0.6903265155 | -0.0401808859 | 0.0401808859 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.2756670474 | 0.2805842912 | -0.0049172438 | 0.0049172438 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.5387922651 | 0.5400903815 | -0.0012981165 | 0.0012981165 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 2 | 50 | 50 | 0.6833202917 | 0.6912804248 | -0.0079601332 | 0.0079601332 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6429959839 | 0.6546533271 | -0.0116573432 | -0.0116573432 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.7435618676 | 0.7384260224 | 0.0051358453 | 0.0051358453 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 2 | 50 | 50 | 0.6734755689 | 0.6693569242 | 0.0041186447 | 0.0041186447 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6409528791 | 0.6351828294 | 0.0057700497 | 0.0057700497 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.7452831050 | 0.7401067028 | 0.0051764022 | 0.0051764022 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 2 | 50 | 50 | 0.6707322781 | 0.6687244285 | 0.0020078496 | 0.0020078496 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6280590974 | 0.6333745166 | -0.0053154192 | -0.0053154192 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.7419443069 | 0.7380696902 | 0.0038746167 | 0.0038746167 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 2 | 50 | 50 | 0.6818377672 | 0.6636659457 | 0.0181718215 | 0.0181718215 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6280817482 | 0.6421740494 | -0.0140923012 | -0.0140923012 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.7462873663 | 0.7456851950 | 0.0006021713 | 0.0006021713 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 2 | 50 | 50 | 0.6848449744 | 0.6702855861 | 0.0145593883 | 0.0145593883 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.2739057981 | 0.2756637999 | -0.0017580018 | 0.0017580018 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.5388373199 | 0.5414594256 | -0.0026221057 | 0.0026221057 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 2 | 50 | 50 | 0.6952606586 | 0.7220098289 | -0.0267491703 | 0.0267491703 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.2705825776 | 0.3016448070 | -0.0310622294 | 0.0310622294 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.5362619686 | 0.5421941901 | -0.0059322215 | 0.0059322215 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 2 | 50 | 50 | 0.7198828117 | 0.7492169522 | -0.0293341405 | 0.0293341405 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.2727610861 | 0.2877540192 | -0.0149929331 | 0.0149929331 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.5413468273 | 0.5410188144 | 0.0003280129 | -0.0003280129 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 2 | 50 | 50 | 0.6501456296 | 0.7898308769 | -0.1396852473 | 0.1396852473 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.2756670474 | 0.2815417634 | -0.0058747160 | 0.0058747160 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.5387922651 | 0.5367488256 | 0.0020434394 | -0.0020434394 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 2 | 50 | 50 | 0.6833202917 | 0.7534246385 | -0.0701043468 | 0.0701043468 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 0 | 50 | 50 | 0.6429959839 | 0.6553063558 | -0.0123103719 | -0.0123103719 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 1 | 50 | 50 | 0.7435618676 | 0.7436570139 | -0.0000951462 | -0.0000951462 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 2 | 50 | 50 | 0.6734755689 | 0.6668780003 | 0.0065975686 | 0.0065975686 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 0 | 50 | 50 | 0.6409528791 | 0.6391951261 | 0.0017577529 | 0.0017577529 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 1 | 50 | 50 | 0.7452831050 | 0.7427847541 | 0.0024983509 | 0.0024983509 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 2 | 50 | 50 | 0.6707322781 | 0.6670939682 | 0.0036383099 | 0.0036383099 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 0 | 50 | 50 | 0.6280590974 | 0.6520301255 | -0.0239710281 | -0.0239710281 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 1 | 50 | 50 | 0.7419443069 | 0.7429237767 | -0.0009794698 | -0.0009794698 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 2 | 50 | 50 | 0.6818377672 | 0.6786780229 | 0.0031597443 | 0.0031597443 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 0 | 50 | 50 | 0.6280817482 | 0.6518031921 | -0.0237214439 | -0.0237214439 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 1 | 50 | 50 | 0.7462873663 | 0.7437291456 | 0.0025582207 | 0.0025582207 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 2 | 50 | 50 | 0.6848449744 | 0.6750640156 | 0.0097809588 | 0.0097809588 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 0 | 50 | 50 | 0.2739057981 | 0.2875119939 | -0.0136061958 | 0.0136061958 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 1 | 50 | 50 | 0.5388373199 | 0.5409621717 | -0.0021248518 | 0.0021248518 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 2 | 50 | 50 | 0.6952606586 | 0.7306448029 | -0.0353841443 | 0.0353841443 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 0 | 50 | 50 | 0.2705825776 | 0.2801260620 | -0.0095434844 | 0.0095434844 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 1 | 50 | 50 | 0.5362619686 | 0.5430458521 | -0.0067838835 | 0.0067838835 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 2 | 50 | 50 | 0.7198828117 | 0.7039879508 | 0.0158948609 | -0.0158948609 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 0 | 50 | 50 | 0.2727610861 | 0.2582544037 | 0.0145066824 | -0.0145066824 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 1 | 50 | 50 | 0.5413468273 | 0.5380717582 | 0.0032750691 | -0.0032750691 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 2 | 50 | 50 | 0.6501456296 | 0.6940349093 | -0.0438892797 | 0.0438892797 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 0 | 50 | 50 | 0.2756670474 | 0.2749937780 | 0.0006732694 | -0.0006732694 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 1 | 50 | 50 | 0.5387922651 | 0.5424649286 | -0.0036726635 | 0.0036726635 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 2 | 50 | 50 | 0.6833202917 | 0.7108430383 | -0.0275227466 | 0.0275227466 | 71561 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 0 | 13 | 15 | 0.6497108256 | 0.6523059562 | -0.0025951306 | -0.0025951306 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 1 | 13 | 15 | 0.7411886842 | 0.7440866051 | -0.0028979210 | -0.0028979210 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 2 | 13 | 15 | 0.6877253878 | 0.6900427823 | -0.0023173944 | -0.0023173944 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 0 | 18 | 15 | 0.6661571873 | 0.6603178765 | 0.0058393108 | 0.0058393108 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 1 | 18 | 15 | 0.7444625749 | 0.7437615030 | 0.0007010719 | 0.0007010719 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 2 | 18 | 15 | 0.6893065723 | 0.6858147449 | 0.0034918274 | 0.0034918274 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 0 | 12 | 14 | 0.6423527051 | 0.6584915612 | -0.0161388561 | -0.0161388561 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 1 | 12 | 14 | 0.7402213409 | 0.7431475776 | -0.0029262367 | -0.0029262367 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 2 | 12 | 14 | 0.6862452650 | 0.6935207147 | -0.0072754497 | -0.0072754497 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6556963486 | 0.6511418685 | 0.0045544801 | 0.0045544801 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.7464524408 | 0.7460268913 | 0.0004255494 | 0.0004255494 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 2 | 21 | 21 | 0.6893023895 | 0.6865417506 | 0.0027606389 | 0.0027606389 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 0 | 13 | 15 | 0.2394645304 | 0.2398194970 | -0.0003549667 | 0.0003549667 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 1 | 13 | 15 | 0.5381320249 | 0.5340305217 | 0.0041015032 | -0.0041015032 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 2 | 13 | 15 | 0.6309849139 | 0.6268591172 | 0.0041257967 | -0.0041257967 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 0 | 18 | 15 | 0.2391724521 | 0.2431130663 | -0.0039406142 | 0.0039406142 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 1 | 18 | 15 | 0.5363768473 | 0.5372647491 | -0.0008879018 | 0.0008879018 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 2 | 18 | 15 | 0.6403368634 | 0.6264773516 | 0.0138595118 | -0.0138595118 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 0 | 12 | 14 | 0.2407774987 | 0.2410739796 | -0.0002964809 | 0.0002964809 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 1 | 12 | 14 | 0.5365840576 | 0.5353037630 | 0.0012802946 | -0.0012802946 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 2 | 12 | 14 | 0.6263450628 | 0.6205542480 | 0.0057908148 | -0.0057908148 | 71561 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.2398820831 | 0.2408840617 | -0.0010019786 | 0.0010019786 | 138225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.5341819829 | 0.5331798736 | 0.0010021093 | -0.0010021093 | 1395118 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 2 | 21 | 21 | 0.6258556807 | 0.6268238933 | -0.0009682126 | 0.0009682126 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 0 | 13 | 13 | 0.6497108256 | 0.6567846181 | -0.0070737926 | -0.0070737926 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 1 | 13 | 13 | 0.7411886842 | 0.7368015172 | 0.0043871670 | 0.0043871670 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 2 | 13 | 13 | 0.6877253878 | 0.6941797202 | -0.0064543323 | -0.0064543323 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 0 | 18 | 17 | 0.6661571873 | 0.6572147554 | 0.0089424319 | 0.0089424319 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 1 | 18 | 17 | 0.7444625749 | 0.7425598929 | 0.0019026820 | 0.0019026820 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 2 | 18 | 17 | 0.6893065723 | 0.6851848821 | 0.0041216902 | 0.0041216902 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 0 | 12 | 14 | 0.6423527051 | 0.6699182868 | -0.0275655817 | -0.0275655817 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 1 | 12 | 14 | 0.7402213409 | 0.7368593224 | 0.0033620186 | 0.0033620186 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 2 | 12 | 14 | 0.6862452650 | 0.7025507270 | -0.0163054619 | -0.0163054619 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6556963486 | 0.6534039271 | 0.0022924215 | 0.0022924215 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.7464524408 | 0.7465104067 | -0.0000579659 | -0.0000579659 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 2 | 21 | 21 | 0.6893023895 | 0.6883159257 | 0.0009864637 | 0.0009864637 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 0 | 13 | 13 | 0.2394645304 | 0.2378794261 | 0.0015851043 | -0.0015851043 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 1 | 13 | 13 | 0.5381320249 | 0.5389427732 | -0.0008107483 | 0.0008107483 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 2 | 13 | 13 | 0.6309849139 | 0.6202071059 | 0.0107778080 | -0.0107778080 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 0 | 18 | 17 | 0.2391724521 | 0.2393493354 | -0.0001768833 | 0.0001768833 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 1 | 18 | 17 | 0.5363768473 | 0.5347602050 | 0.0016166423 | -0.0016166423 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 2 | 18 | 17 | 0.6403368634 | 0.6268436377 | 0.0134932257 | -0.0134932257 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 0 | 12 | 14 | 0.2407774987 | 0.2394421660 | 0.0013353327 | -0.0013353327 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 1 | 12 | 14 | 0.5365840576 | 0.5385415460 | -0.0019574884 | 0.0019574884 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 2 | 12 | 14 | 0.6263450628 | 0.6146177468 | 0.0117273160 | -0.0117273160 | 71561 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.2398820831 | 0.2425550777 | -0.0026729946 | 0.0026729946 | 138225 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.5341819829 | 0.5328553641 | 0.0013266188 | -0.0013266188 | 1395118 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 2 | 21 | 21 | 0.6258556807 | 0.6253936334 | 0.0004620474 | -0.0004620474 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 0 | 13 | 12 | 0.6497108256 | 0.6633729982 | -0.0136621727 | -0.0136621727 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 1 | 13 | 12 | 0.7411886842 | 0.7335957233 | 0.0075929608 | 0.0075929608 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 2 | 13 | 12 | 0.6877253878 | 0.6942291150 | -0.0065037272 | -0.0065037272 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 0 | 18 | 12 | 0.6661571873 | 0.6662625400 | -0.0001053527 | -0.0001053527 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 1 | 18 | 12 | 0.7444625749 | 0.7316735624 | 0.0127890125 | 0.0127890125 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 2 | 18 | 12 | 0.6893065723 | 0.6922121491 | -0.0029055768 | -0.0029055768 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 0 | 12 | 12 | 0.6423527051 | 0.6422289778 | 0.0001237273 | 0.0001237273 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 1 | 12 | 12 | 0.7402213409 | 0.7402553872 | -0.0000340463 | -0.0000340463 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 2 | 12 | 12 | 0.6862452650 | 0.6882330982 | -0.0019878332 | -0.0019878332 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6556963486 | 0.6518995628 | 0.0037967858 | 0.0037967858 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.7464524408 | 0.7458543670 | 0.0005980738 | 0.0005980738 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 2 | 21 | 21 | 0.6893023895 | 0.6880356952 | 0.0012666942 | 0.0012666942 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 0 | 13 | 12 | 0.2394645304 | 0.2407739941 | -0.0013094638 | 0.0013094638 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 1 | 13 | 12 | 0.5381320249 | 0.5406097608 | -0.0024777359 | 0.0024777359 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 2 | 13 | 12 | 0.6309849139 | 0.6215037482 | 0.0094811657 | -0.0094811657 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 0 | 18 | 12 | 0.2391724521 | 0.2391882541 | -0.0000158019 | 0.0000158019 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 1 | 18 | 12 | 0.5363768473 | 0.5424844110 | -0.0061075637 | 0.0061075637 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 2 | 18 | 12 | 0.6403368634 | 0.6277287756 | 0.0126080879 | -0.0126080879 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 0 | 12 | 12 | 0.2407774987 | 0.2405518929 | 0.0002256058 | -0.0002256058 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 1 | 12 | 12 | 0.5365840576 | 0.5366604126 | -0.0000763550 | 0.0000763550 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 2 | 12 | 12 | 0.6263450628 | 0.6244521212 | 0.0018929416 | -0.0018929416 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.2398820831 | 0.2404017470 | -0.0005196640 | 0.0005196640 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.5341819829 | 0.5340755521 | 0.0001064308 | -0.0001064308 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 2 | 21 | 21 | 0.6258556807 | 0.6267257009 | -0.0008700202 | 0.0008700202 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 0 | 13 | 16 | 0.6497108256 | 0.6690450169 | -0.0193341913 | -0.0193341913 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 1 | 13 | 16 | 0.7411886842 | 0.7421460186 | -0.0009573345 | -0.0009573345 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 2 | 13 | 16 | 0.6877253878 | 0.6928975628 | -0.0051721750 | -0.0051721750 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 0 | 18 | 12 | 0.6661571873 | 0.6492006189 | 0.0169565684 | 0.0169565684 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 1 | 18 | 12 | 0.7444625749 | 0.7398582904 | 0.0046042845 | 0.0046042845 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 2 | 18 | 12 | 0.6893065723 | 0.6841918249 | 0.0051147474 | 0.0051147474 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 0 | 12 | 15 | 0.6423527051 | 0.6715917692 | -0.0292390641 | -0.0292390641 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 1 | 12 | 15 | 0.7402213409 | 0.7415177936 | -0.0012964527 | -0.0012964527 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 2 | 12 | 15 | 0.6862452650 | 0.6931800609 | -0.0069347959 | -0.0069347959 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 0 | 21 | 15 | 0.6556963486 | 0.6633798077 | -0.0076834591 | -0.0076834591 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 1 | 21 | 15 | 0.7464524408 | 0.7441453373 | 0.0023071035 | 0.0023071035 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 2 | 21 | 15 | 0.6893023895 | 0.6818101458 | 0.0074922437 | 0.0074922437 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 0 | 13 | 16 | 0.2394645304 | 0.2382327635 | 0.0012317668 | -0.0012317668 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 1 | 13 | 16 | 0.5381320249 | 0.5361678994 | 0.0019641255 | -0.0019641255 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 2 | 13 | 16 | 0.6309849139 | 0.6245042831 | 0.0064806308 | -0.0064806308 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 0 | 18 | 12 | 0.2391724521 | 0.2390747906 | 0.0000976615 | -0.0000976615 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 1 | 18 | 12 | 0.5363768473 | 0.5368461624 | -0.0004693151 | 0.0004693151 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 2 | 18 | 12 | 0.6403368634 | 0.6299682507 | 0.0103686127 | -0.0103686127 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 0 | 12 | 15 | 0.2407774987 | 0.2367175195 | 0.0040599792 | -0.0040599792 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 1 | 12 | 15 | 0.5365840576 | 0.5481480412 | -0.0115639836 | 0.0115639836 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 2 | 12 | 15 | 0.6263450628 | 0.6220192542 | 0.0043258086 | -0.0043258086 | 71561 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 0 | 21 | 15 | 0.2398820831 | 0.2385839562 | 0.0012981269 | -0.0012981269 | 138225 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 1 | 21 | 15 | 0.5341819829 | 0.5377965295 | -0.0036145466 | 0.0036145466 | 1395118 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 2 | 21 | 15 | 0.6258556807 | 0.6423860678 | -0.0165303871 | 0.0165303871 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.6551828195 | 0.6749712445 | -0.0197884251 | -0.0197884251 | 138225 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.7480647186 | 0.7485998679 | -0.0005351493 | -0.0005351493 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 2 | 24 | 24 | 0.6941862010 | 0.6966744599 | -0.0024882589 | -0.0024882589 | 71561 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.2409710963 | 0.2373442640 | 0.0036268323 | -0.0036268323 | 138225 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.5333860775 | 0.5334431922 | -0.0000571147 | 0.0000571147 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 2 | 24 | 24 | 0.6299871293 | 0.6276844398 | 0.0023026895 | -0.0023026895 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 15 | 12 | 0.6513582118 | 0.6622998421 | -0.0109416303 | -0.0109416303 | 138225 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 15 | 12 | 0.7462944854 | 0.7413257344 | 0.0049687510 | 0.0049687510 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 2 | 15 | 12 | 0.6907790942 | 0.6880853244 | 0.0026937697 | 0.0026937697 | 71561 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 15 | 12 | 0.2405541696 | 0.2376144167 | 0.0029397528 | -0.0029397528 | 138225 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 15 | 12 | 0.5317140132 | 0.5351682130 | -0.0034541998 | 0.0034541998 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 2 | 15 | 12 | 0.6265135281 | 0.6282131808 | -0.0016996527 | 0.0016996527 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.8200174174 | 0.8186768907 | 0.0013405266 | 0.0013405266 | 124759 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.8188784861 | 0.8191068067 | -0.0002283207 | -0.0002283207 | 254879 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 2 | 24 | 24 | 0.8084500669 | 0.8064956970 | 0.0019543699 | 0.0019543699 | 260495 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4872081244 | 0.4822480762 | 0.0049600481 | -0.0049600481 | 124759 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4718555068 | 0.4725779879 | -0.0007224812 | 0.0007224812 | 254879 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 2 | 24 | 24 | 0.4804721054 | 0.4877066140 | -0.0072345086 | 0.0072345086 | 260495 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 14 | 14 | 0.8121642324 | 0.8134437922 | -0.0012795598 | -0.0012795598 | 124759 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 14 | 14 | 0.8142576069 | 0.8160612770 | -0.0018036701 | -0.0018036701 | 254879 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 2 | 14 | 14 | 0.8040997314 | 0.8072446610 | -0.0031449295 | -0.0031449295 | 260495 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 14 | 14 | 0.4765257471 | 0.4781937923 | -0.0016680452 | 0.0016680452 | 124759 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 14 | 14 | 0.4691430895 | 0.4681030542 | 0.0010400353 | -0.0010400353 | 254879 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 2 | 14 | 14 | 0.4792064078 | 0.4768862691 | 0.0023201387 | -0.0023201387 | 260495 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.6551828195 | 0.6598582976 | -0.0046754781 | -0.0046754781 | 138225 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.7480647186 | 0.7485498381 | -0.0004851195 | -0.0004851195 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 2 | 24 | 24 | 0.6941862010 | 0.6936658622 | 0.0005203388 | 0.0005203388 | 71561 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.2409710963 | 0.2392860231 | 0.0016850732 | -0.0016850732 | 138225 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.5333860775 | 0.5325476687 | 0.0008384088 | -0.0008384088 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 24 | 24 | 0.6299871293 | 0.6307211451 | -0.0007340158 | 0.0007340158 | 71561 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.6749712445 | 0.6752762534 | -0.0003050089 | -0.0003050089 | 138225 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.7485998679 | 0.7495539162 | -0.0009540483 | -0.0009540483 | 1395118 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 2 | 24 | 24 | 0.6966744599 | 0.6987674109 | -0.0020929510 | -0.0020929510 | 71561 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.2373442640 | 0.2380926445 | -0.0007483806 | 0.0007483806 | 138225 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.5334431922 | 0.5325117496 | 0.0009314425 | -0.0009314425 | 1395118 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 24 | 24 | 0.6276844398 | 0.6241821792 | 0.0035022606 | -0.0035022606 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 15 | 15 | 0.6513582118 | 0.6548434845 | -0.0034852727 | -0.0034852727 | 138225 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 15 | 15 | 0.7462944854 | 0.7458414935 | 0.0004529919 | 0.0004529919 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 2 | 15 | 15 | 0.6907790942 | 0.6916181601 | -0.0008390660 | -0.0008390660 | 71561 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 15 | 15 | 0.2405541696 | 0.2392837579 | 0.0012704116 | -0.0012704116 | 138225 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 15 | 15 | 0.5317140132 | 0.5326473021 | -0.0009332889 | 0.0009332889 | 1395118 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 15 | 15 | 0.6265135281 | 0.6259570558 | 0.0005564723 | -0.0005564723 | 71561 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 12 | 18 | 0.6622998421 | 0.6758106079 | -0.0135107658 | -0.0135107658 | 138225 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 12 | 18 | 0.7413257344 | 0.7474569975 | -0.0061312631 | -0.0061312631 | 1395118 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 2 | 12 | 18 | 0.6880853244 | 0.6963510062 | -0.0082656818 | -0.0082656818 | 71561 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 12 | 18 | 0.2376144167 | 0.2353336246 | 0.0022807922 | -0.0022807922 | 138225 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 12 | 18 | 0.5351682130 | 0.5319433060 | 0.0032249070 | -0.0032249070 | 1395118 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 12 | 18 | 0.6282131808 | 0.6206681205 | 0.0075450603 | -0.0075450603 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.8200174174 | 0.8212354125 | -0.0012179951 | -0.0012179951 | 124759 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.8188784861 | 0.8186952563 | 0.0001832297 | 0.0001832297 | 254879 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 2 | 24 | 24 | 0.8084500669 | 0.8080174744 | 0.0004325925 | 0.0004325925 | 260495 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4872081244 | 0.4798917222 | 0.0073164021 | -0.0073164021 | 124759 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4718555068 | 0.4722586249 | -0.0004031181 | 0.0004031181 | 254879 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 24 | 24 | 0.4804721054 | 0.4831052910 | -0.0026331856 | 0.0026331856 | 260495 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.8186768907 | 0.8179510450 | 0.0007258457 | 0.0007258457 | 124759 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.8191068067 | 0.8194088100 | -0.0003020033 | -0.0003020033 | 254879 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 2 | 24 | 24 | 0.8064956970 | 0.8088936966 | -0.0023979996 | -0.0023979996 | 260495 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4822480762 | 0.4896745469 | -0.0074264707 | 0.0074264707 | 124759 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4725779879 | 0.4792056089 | -0.0066276209 | 0.0066276209 | 254879 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 24 | 24 | 0.4877066140 | 0.4857285342 | 0.0019780797 | -0.0019780797 | 260495 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 14 | 17 | 0.8121642324 | 0.8167032021 | -0.0045389697 | -0.0045389697 | 124759 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 14 | 17 | 0.8142576069 | 0.8165663024 | -0.0023086955 | -0.0023086955 | 254879 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 2 | 14 | 17 | 0.8040997314 | 0.8058608382 | -0.0017611067 | -0.0017611067 | 260495 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 14 | 17 | 0.4765257471 | 0.4742368695 | 0.0022888775 | -0.0022888775 | 124759 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 14 | 17 | 0.4691430895 | 0.4662567842 | 0.0028863053 | -0.0028863053 | 254879 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 14 | 17 | 0.4792064078 | 0.4795774666 | -0.0003710588 | 0.0003710588 | 260495 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 14 | 14 | 0.8134437922 | 0.8134789927 | -0.0000352006 | -0.0000352006 | 124759 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 14 | 14 | 0.8160612770 | 0.8158048293 | 0.0002564476 | 0.0002564476 | 254879 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 2 | 14 | 14 | 0.8072446610 | 0.8072543866 | -0.0000097256 | -0.0000097256 | 260495 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 14 | 14 | 0.4781937923 | 0.4789607052 | -0.0007669129 | 0.0007669129 | 124759 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 14 | 14 | 0.4681030542 | 0.4676980238 | 0.0004050304 | -0.0004050304 | 254879 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 2 | 14 | 14 | 0.4768862691 | 0.4757869090 | 0.0010993601 | -0.0010993601 | 260495 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.6598582976 | 0.6752762534 | -0.0154179558 | -0.0154179558 | 138225 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.7485498381 | 0.7495539162 | -0.0010040781 | -0.0010040781 | 1395118 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 2 | 24 | 24 | 0.6936658622 | 0.6987674109 | -0.0051015487 | -0.0051015487 | 71561 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.2392860231 | 0.2380926445 | 0.0011933785 | -0.0011933785 | 138225 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.5325476687 | 0.5325117496 | 0.0000359191 | -0.0000359191 | 1395118 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 2 | 24 | 24 | 0.6307211451 | 0.6241821792 | 0.0065389659 | -0.0065389659 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 15 | 18 | 0.6548434845 | 0.6758106079 | -0.0209671234 | -0.0209671234 | 138225 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 15 | 18 | 0.7458414935 | 0.7474569975 | -0.0016155039 | -0.0016155039 | 1395118 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 2 | 15 | 18 | 0.6916181601 | 0.6963510062 | -0.0047328461 | -0.0047328461 | 71561 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 15 | 18 | 0.2392837579 | 0.2353336246 | 0.0039501334 | -0.0039501334 | 138225 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 15 | 18 | 0.5326473021 | 0.5319433060 | 0.0007039961 | -0.0007039961 | 1395118 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 2 | 15 | 18 | 0.6259570558 | 0.6206681205 | 0.0052889352 | -0.0052889352 | 71561 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.8212354125 | 0.8179510450 | 0.0032843675 | 0.0032843675 | 124759 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.8186952563 | 0.8194088100 | -0.0007135537 | -0.0007135537 | 254879 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 2 | 24 | 24 | 0.8080174744 | 0.8088936966 | -0.0008762222 | -0.0008762222 | 260495 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4798917222 | 0.4896745469 | -0.0097828247 | 0.0097828247 | 124759 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4722586249 | 0.4792056089 | -0.0069469840 | 0.0069469840 | 254879 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 2 | 24 | 24 | 0.4831052910 | 0.4857285342 | -0.0026232433 | 0.0026232433 | 260495 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 17 | 14 | 0.8167032021 | 0.8134789927 | 0.0032242093 | 0.0032242093 | 124759 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 17 | 14 | 0.8165663024 | 0.8158048293 | 0.0007614730 | 0.0007614730 | 254879 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 2 | 17 | 14 | 0.8058608382 | 0.8072543866 | -0.0013935484 | -0.0013935484 | 260495 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 17 | 14 | 0.4742368695 | 0.4789607052 | -0.0047238356 | 0.0047238356 | 124759 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 17 | 14 | 0.4662567842 | 0.4676980238 | -0.0014412396 | 0.0014412396 | 254879 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 2 | 17 | 14 | 0.4795774666 | 0.4757869090 | 0.0037905576 | -0.0037905576 | 260495 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6617678302 | 0.0007112388 | 0.0007112388 | 138225 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7468666115 | 0.0002882231 | 0.0002882231 | 1395118 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6894924375 | -0.0005316326 | -0.0005316326 | 71561 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2403650418 | 0.0004140659 | -0.0004140659 | 138225 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5329823958 | 0.0006614389 | -0.0006614389 | 1395118 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6318440778 | 0.0067513095 | -0.0067513095 | 71561 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6617754639 | 0.0007036051 | 0.0007036051 | 138225 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7470323698 | 0.0001224648 | 0.0001224648 | 1395118 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6887079102 | 0.0002528947 | 0.0002528947 | 71561 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2388562473 | 0.0019228604 | -0.0019228604 | 138225 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5347415741 | -0.0010977394 | 0.0010977394 | 1395118 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6327255814 | 0.0058698059 | -0.0058698059 | 71561 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6624790690 | 0.6633624365 | -0.0008833675 | -0.0008833675 | 138225 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7471548346 | 0.7473232897 | -0.0001684551 | -0.0001684551 | 1395118 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.6889608049 | 0.6867933337 | 0.0021674712 | 0.0021674712 | 71561 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.2407791077 | 0.2384376625 | 0.0023414452 | -0.0023414452 | 138225 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.5336438347 | 0.5329030629 | 0.0007407718 | -0.0007407718 | 1395118 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.6385953873 | 0.6355256504 | 0.0030697369 | -0.0030697369 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6639642817 | 0.6651234516 | -0.0011591699 | -0.0011591699 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7469000143 | 0.7461780071 | 0.0007220072 | 0.0007220072 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 2 | 24 | 24 | 0.6981111437 | 0.6960947070 | 0.0020164368 | 0.0020164368 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.2387499738 | 0.2420589520 | -0.0033089782 | 0.0033089782 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.5349474595 | 0.5334344999 | 0.0015129595 | -0.0015129595 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 2 | 24 | 24 | 0.6190893594 | 0.6503729137 | -0.0312835544 | 0.0312835544 | 71561 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 18 | 18 | 0.6575247855 | 0.6573758916 | 0.0001488939 | 0.0001488939 | 138225 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 18 | 18 | 0.7455524817 | 0.7454991107 | 0.0000533709 | 0.0000533709 | 1395118 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 2 | 18 | 18 | 0.6953233059 | 0.6958401891 | -0.0005168833 | -0.0005168833 | 71561 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 18 | 18 | 0.2375284961 | 0.2376960609 | -0.0001675647 | 0.0001675647 | 138225 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 18 | 18 | 0.5377300475 | 0.5353429835 | 0.0023870641 | -0.0023870641 | 1395118 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 2 | 18 | 18 | 0.6230591551 | 0.6244400869 | -0.0013809318 | 0.0013809318 | 71561 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 18 | 19 | 0.6575247855 | 0.6613280485 | -0.0038032630 | -0.0038032630 | 138225 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 18 | 19 | 0.7455524817 | 0.7441147137 | 0.0014377680 | 0.0014377680 | 1395118 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 2 | 18 | 19 | 0.6953233059 | 0.6918005805 | 0.0035227254 | 0.0035227254 | 71561 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 18 | 19 | 0.2375284961 | 0.2372361906 | 0.0002923055 | -0.0002923055 | 138225 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 18 | 19 | 0.5377300475 | 0.5350412904 | 0.0026887571 | -0.0026887571 | 1395118 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 2 | 18 | 19 | 0.6230591551 | 0.6280488176 | -0.0049896625 | 0.0049896625 | 71561 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 18 | 18 | 0.6575247855 | 0.6571248584 | 0.0003999271 | 0.0003999271 | 138225 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 18 | 18 | 0.7455524817 | 0.7455040997 | 0.0000483819 | 0.0000483819 | 1395118 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 2 | 18 | 18 | 0.6953233059 | 0.6960341328 | -0.0007108269 | -0.0007108269 | 71561 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 18 | 18 | 0.2375284961 | 0.2374086126 | 0.0001198835 | -0.0001198835 | 138225 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 18 | 18 | 0.5377300475 | 0.5346114191 | 0.0031186284 | -0.0031186284 | 1395118 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 2 | 18 | 18 | 0.6230591551 | 0.6218912602 | 0.0011678949 | -0.0011678949 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 0 | 24 | 12 | 0.6639642817 | 0.6707548658 | -0.0067905841 | -0.0067905841 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 1 | 24 | 12 | 0.7469000143 | 0.7338602152 | 0.0130397991 | 0.0130397991 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 2 | 24 | 12 | 0.6981111437 | 0.6961281674 | 0.0019829764 | 0.0019829764 | 71561 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 0 | 24 | 12 | 0.2387499738 | 0.2363854659 | 0.0023645079 | -0.0023645079 | 138225 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 1 | 24 | 12 | 0.5349474595 | 0.5436999993 | -0.0087525398 | 0.0087525398 | 1395118 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 2 | 24 | 12 | 0.6190893594 | 0.6240245407 | -0.0049351813 | 0.0049351813 | 71561 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8166612567 | 0.0002934561 | 0.0002934561 | 124759 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8212106042 | -0.0009329962 | -0.0009329962 | 254879 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8074120150 | 0.0010787170 | 0.0010787170 | 260495 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4804881557 | 0.0059750503 | -0.0059750503 | 124759 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4721449063 | -0.0008754004 | 0.0008754004 | 254879 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4839659860 | -0.0000171184 | 0.0000171184 | 260495 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8163004074 | 0.0006543054 | 0.0006543054 | 124759 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8202984919 | -0.0000208840 | -0.0000208840 | 254879 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8073857906 | 0.0011049414 | 0.0011049414 | 260495 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4849298563 | 0.0015333496 | -0.0015333496 | 124759 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4719192963 | -0.0006497904 | 0.0006497904 | 254879 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4826223603 | 0.0013265073 | -0.0013265073 | 260495 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8169547128 | 0.8167027979 | 0.0002519149 | 0.0002519149 | 124759 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8202776080 | 0.8204551592 | -0.0001775512 | -0.0001775512 | 254879 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 2 | 24 | 24 | 0.8084907320 | 0.8082936100 | 0.0001971221 | 0.0001971221 | 260495 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.4864632060 | 0.4797396390 | 0.0067235670 | -0.0067235670 | 124759 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4712695059 | 0.4693611561 | 0.0019083498 | -0.0019083498 | 254879 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 2 | 24 | 24 | 0.4839488676 | 0.4843904825 | -0.0004416148 | 0.0004416148 | 260495 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | domain | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain | fit domain n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 12 | 12 | 0.8076898769 | 0.8076767817 | 0.0000130952 | 0.0000130952 | 124759 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 12 | 12 | 0.8127024521 | 0.8126094574 | 0.0000929947 | 0.0000929947 | 254879 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 2 | 12 | 12 | 0.8022116233 | 0.8023145971 | -0.0001029738 | -0.0001029738 | 260495 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 12 | 12 | 0.4799775951 | 0.4801844778 | -0.0002068827 | 0.0002068827 | 124759 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 12 | 12 | 0.4694188145 | 0.4687609072 | 0.0006579073 | -0.0006579073 | 254879 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 2 | 12 | 12 | 0.4795692555 | 0.4797352188 | -0.0001659634 | 0.0001659634 | 260495 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 12 | 14 | 0.8076898769 | 0.8096159674 | -0.0019260905 | -0.0019260905 | 124759 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 12 | 14 | 0.8127024521 | 0.8134237814 | -0.0007213293 | -0.0007213293 | 254879 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 2 | 12 | 14 | 0.8022116233 | 0.8039974375 | -0.0017858142 | -0.0017858142 | 260495 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 12 | 14 | 0.4799775951 | 0.4824792089 | -0.0025016138 | 0.0025016138 | 124759 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 12 | 14 | 0.4694188145 | 0.4685118834 | 0.0009069311 | -0.0009069311 | 254879 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 2 | 12 | 14 | 0.4795692555 | 0.4780124501 | 0.0015568054 | -0.0015568054 | 260495 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 12 | 12 | 0.8076898769 | 0.8076235583 | 0.0000663186 | 0.0000663186 | 124759 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 12 | 12 | 0.8127024521 | 0.8127167352 | -0.0000142831 | -0.0000142831 | 254879 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 2 | 12 | 12 | 0.8022116233 | 0.8022019051 | 0.0000097181 | 0.0000097181 | 260495 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 12 | 12 | 0.4799775951 | 0.4800940088 | -0.0001164137 | 0.0001164137 | 124759 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 12 | 12 | 0.4694188145 | 0.4694602814 | -0.0000414669 | 0.0000414669 | 254879 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 2 | 12 | 12 | 0.4795692555 | 0.4796059401 | -0.0000366846 | 0.0000366846 | 260495 |


## domain seed级统计

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [-0.0020868992] | -0.0020868992 | NaN | -0.0020868992 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0028795518] | -0.0028795518 | NaN | -0.0028795518 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 2 | 1 | [3101] | [-0.0112283968] | -0.0112283968 | NaN | -0.0112283968 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0018548136] | 0.0018548136 | NaN | 0.0018548136 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0039764484] | 0.0039764484 | NaN | 0.0039764484 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 2 | 1 | [3101] | [0.0212632462] | 0.0212632462 | NaN | 0.0212632462 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0127212420] | 0.0127212420 | NaN | 0.0127212420 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [-0.0013987256] | -0.0013987256 | NaN | -0.0013987256 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 2 | 1 | [3101] | [-0.0020888549] | -0.0020888549 | NaN | -0.0020888549 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [0.0017687496] | 0.0017687496 | NaN | 0.0017687496 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [0.0030750217] | 0.0030750217 | NaN | 0.0030750217 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 2 | 1 | [3101] | [0.0150704966] | 0.0150704966 | NaN | 0.0150704966 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0044090162] | -0.0044090162 | NaN | -0.0044090162 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [-0.0024155102] | -0.0024155102 | NaN | -0.0024155102 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 1 | [3101] | [0.0062995006] | 0.0062995006 | NaN | 0.0062995006 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [-0.0010515548] | -0.0010515548 | NaN | -0.0010515548 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [0.0036013545] | 0.0036013545 | NaN | 0.0036013545 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 1 | [3101] | [-0.0132643581] | -0.0132643581 | NaN | -0.0132643581 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [-0.0071815375] | -0.0071815375 | NaN | -0.0071815375 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0036565556] | -0.0036565556 | NaN | -0.0036565556 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 2 | 1 | [3101] | [0.0005257183] | 0.0005257183 | NaN | 0.0005257183 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [-0.0004726867] | -0.0004726867 | NaN | -0.0004726867 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0064948740] | 0.0064948740 | NaN | 0.0064948740 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 2 | 1 | [3101] | [0.0012710535] | 0.0012710535 | NaN | 0.0012710535 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0077669585] | 0.0077669585 | NaN | 0.0077669585 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [-0.0030010785] | -0.0030010785 | NaN | -0.0030010785 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 2 | 1 | [3101] | [0.0042736461] | 0.0042736461 | NaN | 0.0042736461 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [-0.0014818620] | -0.0014818620 | NaN | -0.0014818620 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [0.0071612346] | 0.0071612346 | NaN | 0.0071612346 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 2 | 1 | [3101] | [-0.0004657357] | -0.0004657357 | NaN | -0.0004657357 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0023669586] | -0.0023669586 | NaN | -0.0023669586 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [0.0063245137] | 0.0063245137 | NaN | 0.0063245137 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 1 | [3101] | [0.0114418918] | 0.0114418918 | NaN | 0.0114418918 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [0.0015596149] | 0.0015596149 | NaN | 0.0015596149 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [-0.0005776478] | -0.0005776478 | NaN | -0.0005776478 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 1 | [3101] | [-0.0072993560] | -0.0072993560 | NaN | -0.0072993560 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [0.0031166845] | 0.0031166845 | NaN | 0.0031166845 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [0.0028372136] | 0.0028372136 | NaN | 0.0028372136 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 2 | 1 | [3101] | [0.0030404727] | 0.0030404727 | NaN | 0.0030404727 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0122473732] | 0.0122473732 | NaN | 0.0122473732 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0066064632] | 0.0066064632 | NaN | 0.0066064632 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 2 | 1 | [3101] | [0.0072029098] | 0.0072029098 | NaN | 0.0072029098 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0076711270] | 0.0076711270 | NaN | 0.0076711270 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [0.0064601336] | 0.0064601336 | NaN | 0.0064601336 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 2 | 1 | [3101] | [0.0038785091] | 0.0038785091 | NaN | 0.0038785091 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [0.0023326039] | 0.0023326039 | NaN | 0.0023326039 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [-0.0014933162] | -0.0014933162 | NaN | -0.0014933162 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 2 | 1 | [3101] | [0.0045530445] | 0.0045530445 | NaN | 0.0045530445 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0004600539] | -0.0004600539 | NaN | -0.0004600539 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [0.0003312298] | 0.0003312298 | NaN | 0.0003312298 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 1 | [3101] | [-0.0029750109] | -0.0029750109 | NaN | -0.0029750109 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [0.0108088967] | 0.0108088967 | NaN | 0.0108088967 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [-0.0024100895] | -0.0024100895 | NaN | -0.0024100895 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 1 | [3101] | [0.0081633653] | 0.0081633653 | NaN | 0.0081633653 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [-0.0052325895] | -0.0052325895 | NaN | -0.0052325895 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0041745513] | -0.0041745513 | NaN | -0.0041745513 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 2 | 1 | [3101] | [-0.0042964630] | -0.0042964630 | NaN | -0.0042964630 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0040954386] | 0.0040954386 | NaN | 0.0040954386 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0049053207] | 0.0049053207 | NaN | 0.0049053207 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 2 | 1 | [3101] | [0.0042677273] | 0.0042677273 | NaN | 0.0042677273 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0029550920] | 0.0029550920 | NaN | 0.0029550920 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [0.0010228089] | 0.0010228089 | NaN | 0.0010228089 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 2 | 1 | [3101] | [0.0009871157] | 0.0009871157 | NaN | 0.0009871157 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [-0.0040422388] | -0.0040422388 | NaN | -0.0040422388 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [-0.0007725974] | -0.0007725974 | NaN | -0.0007725974 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 2 | 1 | [3101] | [-0.0027424279] | -0.0027424279 | NaN | -0.0027424279 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0014565226] | -0.0014565226 | NaN | -0.0014565226 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [-0.0008282892] | -0.0008282892 | NaN | -0.0008282892 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 1 | [3101] | [-0.0011957700] | -0.0011957700 | NaN | -0.0011957700 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [0.0022838136] | 0.0022838136 | NaN | 0.0022838136 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [-0.0000558631] | -0.0000558631 | NaN | -0.0000558631 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 1 | [3101] | [0.0002131703] | 0.0002131703 | NaN | 0.0002131703 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [0.0061682436, 0.0044900505] | 0.0053291470 | 0.0011866618 | 0.0053291470 | 2 | 0 | 0 | -0.0053325860 | 0.0159908801 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [-0.0011329531, -0.0000733678] | -0.0006031605 | 0.0007492399 | -0.0006031605 | 0 | 2 | 0 | -0.0073348139 | 0.0061284930 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 2 | 2 | [3101, 3102] | [0.0008936684, 0.0016964373] | 0.0012950529 | 0.0005676433 | 0.0012950529 | 2 | 0 | 0 | -0.0038050200 | 0.0063951258 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [-0.0001967371, 0.0002572294] | 0.0000302461 | 0.0003210028 | 0.0000302461 | 1 | 1 | 0 | -0.0028538492 | 0.0029143414 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [0.0003253750, 0.0005020069] | 0.0004136910 | 0.0001248976 | 0.0004136910 | 0 | 2 | 0 | -0.0007084695 | 0.0015358514 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 2 | 2 | [3101, 3102] | [-0.0017999254, 0.0001833777] | -0.0008082739 | 0.0014024070 | -0.0008082739 | 1 | 1 | 0 | -0.0134084012 | 0.0117918535 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0093863037, -0.0041757268] | 0.0026052885 | 0.0095898038 | 0.0026052885 | 1 | 1 | 0 | -0.0835556800 | 0.0887662570 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [-0.0001189987, -0.0002940723] | -0.0002065355 | 0.0001237957 | -0.0002065355 | 0 | 2 | 0 | -0.0013187961 | 0.0009057251 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 2 | 2 | [3101, 3102] | [-0.0037525558, -0.0020243907] | -0.0028884733 | 0.0012219973 | -0.0028884733 | 0 | 2 | 0 | -0.0138676832 | 0.0080907367 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [-0.0013415613, 0.0026659828] | 0.0006622108 | 0.0028337616 | 0.0006622108 | 1 | 1 | 0 | -0.0247981272 | 0.0261225487 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [-0.0032870538, 0.0004912693] | -0.0013978923 | 0.0026716779 | -0.0013978923 | 1 | 1 | 0 | -0.0254019657 | 0.0226061811 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 2 | 2 | [3101, 3102] | [0.0050615804, 0.0053419195] | 0.0052017500 | 0.0001982297 | 0.0052017500 | 0 | 2 | 0 | 0.0034207268 | 0.0069827731 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0090704570, 0.0112515002] | 0.0101609786 | 0.0015422305 | 0.0101609786 | 2 | 0 | 0 | -0.0036954126 | 0.0240173698 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [-0.0014168381, 0.0003908061] | -0.0005130160 | 0.0012781974 | -0.0005130160 | 1 | 1 | 0 | -0.0119971644 | 0.0109711324 |
| L4_BLOCK_META-L0_FEATURE | auc | 2 | 2 | [3101, 3102] | [0.0040878941, 0.0072419022] | 0.0056648981 | 0.0022302206 | 0.0056648981 | 2 | 0 | 0 | -0.0143728386 | 0.0257026349 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0028105891, 0.0013965949] | 0.0021035920 | 0.0009998449 | 0.0021035920 | 0 | 2 | 0 | -0.0068796581 | 0.0110868422 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0033637667, 0.0006856725] | 0.0020247196 | 0.0018936986 | 0.0020247196 | 0 | 2 | 0 | -0.0149894868 | 0.0190389261 |
| L4_BLOCK_META-L0_FEATURE | logloss | 2 | 2 | [3101, 3102] | [0.0094738975, 0.0024884894] | 0.0059811935 | 0.0049394295 | 0.0059811935 | 0 | 2 | 0 | -0.0383978194 | 0.0503602064 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [-0.0056721850, 0.0023132620] | -0.0016794615 | 0.0056465637 | -0.0016794615 | 1 | 1 | 0 | -0.0524118233 | 0.0490529003 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [-0.0000692545, 0.0003510185] | 0.0001408820 | 0.0002971779 | 0.0001408820 | 1 | 1 | 0 | -0.0025291551 | 0.0028109191 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 2 | 2 | [3101, 3102] | [0.0018603348, 0.0027572684] | 0.0023088016 | 0.0006342279 | 0.0023088016 | 2 | 0 | 0 | -0.0033895098 | 0.0080071130 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [0.0030915000, -0.0017038786] | 0.0006938107 | 0.0033908447 | 0.0006938107 | 1 | 1 | 0 | -0.0297717206 | 0.0311593420 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0006906108, -0.0009618596] | -0.0001356244 | 0.0011684730 | -0.0001356244 | 1 | 1 | 0 | -0.0106339378 | 0.0103626891 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 2 | 2 | [3101, 3102] | [0.0017975747, -0.0067706264] | -0.0024865259 | 0.0060586331 | -0.0024865259 | 1 | 1 | 0 | -0.0569211843 | 0.0519481325 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [-0.0012209779, -0.0035207200] | -0.0023708490 | 0.0016261633 | -0.0023708490 | 0 | 2 | 0 | -0.0169813461 | 0.0122396482 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [0.0000211375, 0.0005013210] | 0.0002612293 | 0.0003395410 | 0.0002612293 | 2 | 0 | 0 | -0.0027894257 | 0.0033118842 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 2 | 2 | [3101, 3102] | [0.0006011786, 0.0009213134] | 0.0007612460 | 0.0002263695 | 0.0007612460 | 2 | 0 | 0 | -0.0012726029 | 0.0027950949 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0022745602, 0.0010670798] | 0.0016708200 | 0.0008538176 | 0.0016708200 | 0 | 2 | 0 | -0.0060004271 | 0.0093420671 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0007980306, -0.0009447072] | -0.0000733383 | 0.0012323017 | -0.0000733383 | 1 | 1 | 0 | -0.0111451301 | 0.0109984534 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 2 | 2 | [3101, 3102] | [0.0020077333, -0.0010960985] | 0.0004558174 | 0.0021947405 | 0.0004558174 | 1 | 1 | 0 | -0.0192631441 | 0.0201747789 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [-0.0013684147, 0.0037743033] | 0.0012029443 | 0.0036364507 | 0.0012029443 | 1 | 1 | 0 | -0.0314692693 | 0.0338751579 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [-0.0000817418, 0.0006949067] | 0.0003065825 | 0.0005491734 | 0.0003065825 | 1 | 1 | 0 | -0.0046275445 | 0.0052407095 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 2 | 2 | [3101, 3102] | [-0.0011783166, 0.0016296703] | 0.0002256768 | 0.0019855465 | 0.0002256768 | 1 | 1 | 0 | -0.0176137510 | 0.0180651046 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0020549143, -0.0006069434] | 0.0007239855 | 0.0018822177 | 0.0007239855 | 1 | 1 | 0 | -0.0161870695 | 0.0176350404 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0008023020, -0.0013381630] | -0.0002679305 | 0.0015135374 | -0.0002679305 | 1 | 1 | 0 | -0.0138665242 | 0.0133306632 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 2 | 2 | [3101, 3102] | [0.0048377350, -0.0019025318] | 0.0014676016 | 0.0047660884 | 0.0014676016 | 1 | 1 | 0 | -0.0413540037 | 0.0442892069 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0058708713] | -0.0017041417 | 0.0107126861 | -0.0017041417 | 1 | 1 | 0 | -0.0979538079 | 0.0945455244 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, 0.0000746227] | -0.0095553151 | 0.0136187886 | -0.0095553151 | 1 | 1 | 0 | -0.1319152762 | 0.1128046459 |
| R2_FREEZE-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0025858803] | -0.0078824951 | 0.0074905445 | -0.0078824951 | 0 | 2 | 0 | -0.0751823672 | 0.0594173770 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0038807707] | 0.0101459558 | 0.0088603097 | 0.0101459558 | 0 | 2 | 0 | -0.0694607686 | 0.0897526802 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, 0.0006710711] | 0.0087153782 | 0.0113763683 | 0.0087153782 | 0 | 2 | 0 | -0.0934972356 | 0.1109279920 |
| R2_FREEZE-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0030721518] | 0.0403527182 | 0.0527226827 | 0.0403527182 | 0 | 2 | 0 | -0.4333417915 | 0.5140472280 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0058708713] | -0.0017041417 | 0.0107126861 | -0.0017041417 | 1 | 1 | 0 | -0.0979538079 | 0.0945455244 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, 0.0000746227] | -0.0095553151 | 0.0136187886 | -0.0095553151 | 1 | 1 | 0 | -0.1319152762 | 0.1128046459 |
| R2_FREEZE-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0025858803] | -0.0078824951 | 0.0074905445 | -0.0078824951 | 0 | 2 | 0 | -0.0751823672 | 0.0594173770 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0038807707] | 0.0101459558 | 0.0088603097 | 0.0101459558 | 0 | 2 | 0 | -0.0694607686 | 0.0897526802 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, 0.0006710711] | 0.0087153782 | 0.0113763683 | 0.0087153782 | 0 | 2 | 0 | -0.0934972356 | 0.1109279920 |
| R2_FREEZE-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0030721518] | 0.0403527182 | 0.0527226827 | 0.0403527182 | 0 | 2 | 0 | -0.4333417915 | 0.5140472280 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0064007102] | -0.0014392223 | 0.0110873388 | -0.0014392223 | 1 | 1 | 0 | -0.1010550093 | 0.0981765648 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, 0.0001916298] | -0.0094968116 | 0.0137015251 | -0.0094968116 | 1 | 1 | 0 | -0.1326001308 | 0.1136065076 |
| R3_CUM-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0021456775] | -0.0076623937 | 0.0078018149 | -0.0076623937 | 0 | 2 | 0 | -0.0777589193 | 0.0624341318 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0033021198] | 0.0098566303 | 0.0092694777 | 0.0098566303 | 0 | 2 | 0 | -0.0734263229 | 0.0931395835 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, 0.0015646472] | 0.0091621663 | 0.0107445146 | 0.0091621663 | 0 | 2 | 0 | -0.0873734671 | 0.1056977997 |
| R3_CUM-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0015260149] | 0.0395796498 | 0.0538159665 | 0.0395796498 | 0 | 2 | 0 | -0.4439376260 | 0.5230969255 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0064007102] | -0.0014392223 | 0.0110873388 | -0.0014392223 | 1 | 1 | 0 | -0.1010550093 | 0.0981765648 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, 0.0001916298] | -0.0094968116 | 0.0137015251 | -0.0094968116 | 1 | 1 | 0 | -0.1326001308 | 0.1136065076 |
| R3_CUM-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0021456775] | -0.0076623937 | 0.0078018149 | -0.0076623937 | 0 | 2 | 0 | -0.0777589193 | 0.0624341318 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0033021198] | 0.0098566303 | 0.0092694777 | 0.0098566303 | 0 | 2 | 0 | -0.0734263229 | 0.0931395835 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, 0.0015646472] | 0.0091621663 | 0.0107445146 | 0.0091621663 | 0 | 2 | 0 | -0.0873734671 | 0.1056977997 |
| R3_CUM-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0015260149] | 0.0395796498 | 0.0538159665 | 0.0395796498 | 0 | 2 | 0 | -0.4439376260 | 0.5230969255 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0045937420] | -0.0023427064 | 0.0098096194 | -0.0023427064 | 1 | 1 | 0 | -0.0904786396 | 0.0857932269 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, -0.0002439575] | -0.0097146052 | 0.0133935185 | -0.0097146052 | 0 | 2 | 0 | -0.1300505942 | 0.1106213839 |
| R4_EMA-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0020397141] | -0.0076094120 | 0.0078767423 | -0.0076094120 | 0 | 2 | 0 | -0.0783791342 | 0.0631603102 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0017224519] | 0.0090667964 | 0.0103864716 | 0.0090667964 | 0 | 2 | 0 | -0.0842519486 | 0.1023855414 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, -0.0001162562] | 0.0083217146 | 0.0119330928 | 0.0083217146 | 1 | 1 | 0 | -0.0988928702 | 0.1155362994 |
| R4_EMA-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0047099855] | 0.0411716351 | 0.0515645594 | 0.0411716351 | 0 | 2 | 0 | -0.4221175498 | 0.5044608199 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0092791548, 0.0045937420] | -0.0023427064 | 0.0098096194 | -0.0023427064 | 1 | 1 | 0 | -0.0904786396 | 0.0857932269 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0191852529, -0.0002439575] | -0.0097146052 | 0.0133935185 | -0.0097146052 | 0 | 2 | 0 | -0.1300505942 | 0.1106213839 |
| R4_EMA-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [-0.0131791099, -0.0020397141] | -0.0076094120 | 0.0078767423 | -0.0076094120 | 0 | 2 | 0 | -0.0783791342 | 0.0631603102 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0164111409, 0.0017224519] | 0.0090667964 | 0.0103864716 | 0.0090667964 | 0 | 2 | 0 | -0.0842519486 | 0.1023855414 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0167596854, -0.0001162562] | 0.0083217146 | 0.0119330928 | 0.0083217146 | 1 | 1 | 0 | -0.0988928702 | 0.1155362994 |
| R4_EMA-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [0.0776332847, 0.0047099855] | 0.0411716351 | 0.0515645594 | 0.0411716351 | 0 | 2 | 0 | -0.4221175498 | 0.5044608199 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0001741277, -0.0017008426] | -0.0009374852 | 0.0010795504 | -0.0009374852 | 0 | 2 | 0 | -0.0106368609 | 0.0087618906 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0016016414, -0.0001329249] | -0.0008672831 | 0.0010385394 | -0.0008672831 | 0 | 2 | 0 | -0.0101981893 | 0.0084636230 |
| R6_PB_HB-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0040279535, 0.0000771394] | 0.0020525465 | 0.0027936474 | 0.0020525465 | 2 | 0 | 0 | -0.0230473797 | 0.0271524726 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0016223245, -0.0001989531] | -0.0009106388 | 0.0010064756 | -0.0009106388 | 2 | 0 | 0 | -0.0099534629 | 0.0081321853 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0016851141, 0.0000127266] | 0.0008489204 | 0.0011825565 | 0.0008489204 | 0 | 2 | 0 | -0.0097759285 | 0.0114737693 |
| R6_PB_HB-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0155487683, 0.0021872422] | -0.0066807630 | 0.0125412533 | -0.0066807630 | 1 | 1 | 0 | -0.1193594535 | 0.1059979274 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0001741277, -0.0017008426] | -0.0009374852 | 0.0010795504 | -0.0009374852 | 0 | 2 | 0 | -0.0106368609 | 0.0087618906 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0016016414, -0.0001329249] | -0.0008672831 | 0.0010385394 | -0.0008672831 | 0 | 2 | 0 | -0.0101981893 | 0.0084636230 |
| R6_PB_HB-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0040279535, 0.0000771394] | 0.0020525465 | 0.0027936474 | 0.0020525465 | 2 | 0 | 0 | -0.0230473797 | 0.0271524726 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0016223245, -0.0001989531] | -0.0009106388 | 0.0010064756 | -0.0009106388 | 2 | 0 | 0 | -0.0099534629 | 0.0081321853 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0016851141, 0.0000127266] | 0.0008489204 | 0.0011825565 | 0.0008489204 | 0 | 2 | 0 | -0.0097759285 | 0.0114737693 |
| R6_PB_HB-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0155487683, 0.0021872422] | -0.0066807630 | 0.0125412533 | -0.0066807630 | 1 | 1 | 0 | -0.1193594535 | 0.1059979274 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [0.0091050270, -0.0075717138] | 0.0007666566 | 0.0117922366 | 0.0007666566 | 1 | 1 | 0 | -0.1051823853 | 0.1067156985 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [0.0175836116, -0.0002075475] | 0.0086880320 | 0.0125802492 | 0.0086880320 | 1 | 1 | 0 | -0.1043410229 | 0.1217170869 |
| R6_PB_HB-R2_FREEZE | auc | 2 | 2 | [3101, 3102] | [0.0172070634, 0.0026630197] | 0.0099350416 | 0.0102841919 | 0.0099350416 | 2 | 0 | 0 | -0.0824647567 | 0.1023348398 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [-0.0180334654, -0.0040797238] | -0.0110565946 | 0.0098667853 | -0.0110565946 | 2 | 0 | 0 | -0.0997061432 | 0.0775929539 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [-0.0150745713, -0.0006583444] | -0.0078664579 | 0.0101938118 | -0.0078664579 | 2 | 0 | 0 | -0.0994542228 | 0.0837213070 |
| R6_PB_HB-R2_FREEZE | logloss | 2 | 2 | [3101, 3102] | [-0.0931820530, -0.0008849096] | -0.0470334813 | 0.0652639360 | -0.0470334813 | 2 | 0 | 0 | -0.6334066815 | 0.5393397189 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [0.0091050270, -0.0081015528] | 0.0005017371 | 0.0121668892 | 0.0005017371 | 1 | 1 | 0 | -0.1088134257 | 0.1098168999 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [0.0175836116, -0.0003245546] | 0.0086295285 | 0.0126629858 | 0.0086295285 | 1 | 1 | 0 | -0.1051428846 | 0.1224019415 |
| R6_PB_HB-R3_CUM | auc | 2 | 2 | [3101, 3102] | [0.0172070634, 0.0022228169] | 0.0097149402 | 0.0105954623 | 0.0097149402 | 2 | 0 | 0 | -0.0854815115 | 0.1049113919 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [-0.0180334654, -0.0035010729] | -0.0107672691 | 0.0102759533 | -0.0107672691 | 2 | 0 | 0 | -0.1030930464 | 0.0815585082 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [-0.0150745713, -0.0015519205] | -0.0083132459 | 0.0095619580 | -0.0083132459 | 2 | 0 | 0 | -0.0942240304 | 0.0775975386 |
| R6_PB_HB-R3_CUM | logloss | 2 | 2 | [3101, 3102] | [-0.0931820530, 0.0006612273] | -0.0462604128 | 0.0663572199 | -0.0462604128 | 1 | 1 | 0 | -0.6424563790 | 0.5499355534 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [0.0091050270, -0.0062945846] | 0.0014052212 | 0.0108891698 | 0.0014052212 | 1 | 1 | 0 | -0.0964300878 | 0.0992405303 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [0.0175836116, 0.0001110326] | 0.0088473221 | 0.0123549791 | 0.0088473221 | 2 | 0 | 0 | -0.1021577608 | 0.1198524050 |
| R6_PB_HB-R4_EMA | auc | 2 | 2 | [3101, 3102] | [0.0172070634, 0.0021168535] | 0.0096619585 | 0.0106703898 | 0.0096619585 | 2 | 0 | 0 | -0.0862076899 | 0.1055316068 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [-0.0180334654, -0.0019214050] | -0.0099774352 | 0.0113929472 | -0.0099774352 | 2 | 0 | 0 | -0.1123390043 | 0.0923841339 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [-0.0150745713, 0.0001289829] | -0.0074727942 | 0.0107505362 | -0.0074727942 | 1 | 1 | 0 | -0.1040625301 | 0.0891169417 |
| R6_PB_HB-R4_EMA | logloss | 2 | 2 | [3101, 3102] | [-0.0931820530, -0.0025227432] | -0.0478523981 | 0.0641058127 | -0.0478523981 | 2 | 0 | 0 | -0.6238202734 | 0.5281154772 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [-0.0008448314, 0.0024289346] | 0.0007920516 | 0.0023149021 | 0.0007920516 | 1 | 1 | 0 | -0.0200065185 | 0.0215906217 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0014620556, 0.0000832032] | -0.0006894262 | 0.0010926630 | -0.0006894262 | 1 | 1 | 0 | -0.0105066135 | 0.0091277611 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 2 | [3101, 3102] | [0.0034262682, 0.0015250253] | 0.0024756467 | 0.0013443817 | 0.0024756467 | 2 | 0 | 0 | -0.0096031436 | 0.0145544370 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [-0.0044065789, -0.0010533504] | -0.0027299647 | 0.0023710906 | -0.0027299647 | 2 | 0 | 0 | -0.0240333685 | 0.0185734392 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [0.0031163418, -0.0001806913] | 0.0014678252 | 0.0023313544 | 0.0014678252 | 1 | 1 | 0 | -0.0194785633 | 0.0224142138 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 2 | [3101, 3102] | [-0.0112987441, 0.0012196140] | -0.0050395651 | 0.0088518158 | -0.0050395651 | 1 | 1 | 0 | -0.0845699750 | 0.0744908449 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [0.0065362693, 0.0054481425] | 0.0059922059 | 0.0007694219 | 0.0059922059 | 2 | 0 | 0 | -0.0009207752 | 0.0129051870 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [-0.0010429216, 0.0001941627] | -0.0004243795 | 0.0008747506 | -0.0004243795 | 1 | 1 | 0 | -0.0082837022 | 0.0074349433 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 2 | 2 | [3101, 3102] | [0.0037409705, -0.0000901933] | 0.0018253886 | 0.0027090420 | 0.0018253886 | 1 | 1 | 0 | -0.0225143877 | 0.0261651649 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [0.0000652337, -0.0027068630] | -0.0013208147 | 0.0019601684 | -0.0013208147 | 1 | 1 | 0 | -0.0189322288 | 0.0162905995 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [-0.0001846748, -0.0000359905] | -0.0001103327 | 0.0001051356 | -0.0001103327 | 2 | 0 | 0 | -0.0010549387 | 0.0008342734 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 2 | 2 | [3101, 3102] | [-0.0026665892, 0.0006396527] | -0.0010134682 | 0.0023378661 | -0.0010134682 | 1 | 1 | 0 | -0.0220183616 | 0.0199914252 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0019062186, -0.0012976110] | 0.0003043038 | 0.0022654496 | 0.0003043038 | 1 | 1 | 0 | -0.0200499537 | 0.0206585614 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [0.0002693143, -0.0000657319] | 0.0001017912 | 0.0002369135 | 0.0001017912 | 1 | 1 | 0 | -0.0020267918 | 0.0022303742 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 2 | 2 | [3101, 3102] | [0.0011102755, -0.0005977666] | 0.0002562544 | 0.0012077682 | 0.0002562544 | 1 | 1 | 0 | -0.0105951122 | 0.0111076210 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [0.0003602410, -0.0018526619] | -0.0007462104 | 0.0015647586 | -0.0007462104 | 1 | 1 | 0 | -0.0148050091 | 0.0133125882 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [-0.0011071839, -0.0017488156] | -0.0014279998 | 0.0004537021 | -0.0014279998 | 2 | 0 | 0 | -0.0055043517 | 0.0026483521 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 2 | 2 | [3101, 3102] | [-0.0010164670, 0.0065924087] | 0.0027879709 | 0.0053802876 | 0.0027879709 | 1 | 1 | 0 | -0.0455519955 | 0.0511279373 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0154998605, 0.0094154436] | 0.0124576520 | 0.0043023325 | 0.0124576520 | 2 | 0 | 0 | -0.0261972715 | 0.0511125756 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [0.0015286965, 0.0015209021] | 0.0015247993 | 0.0000055115 | 0.0015247993 | 2 | 0 | 0 | 0.0014752807 | 0.0015743178 |
| L4_BLOCK_META-L0_FEATURE | auc | 2 | 2 | [3101, 3102] | [0.0102878716, 0.0058996300] | 0.0080937508 | 0.0031029554 | 0.0080937508 | 2 | 0 | 0 | -0.0197851974 | 0.0359726991 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [-0.0024214535, 0.0006636879] | -0.0008788828 | 0.0021815244 | -0.0008788828 | 1 | 1 | 0 | -0.0204791018 | 0.0187213362 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0045033237, -0.0010416015] | 0.0017308611 | 0.0039208542 | 0.0017308611 | 1 | 1 | 0 | -0.0334966166 | 0.0369583388 |
| L4_BLOCK_META-L0_FEATURE | logloss | 2 | 2 | [3101, 3102] | [-0.0036716168, 0.0059921755] | 0.0011602794 | 0.0068333331 | 0.0011602794 | 1 | 1 | 0 | -0.0602347823 | 0.0625553411 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [0.0040216976, -0.0005288333] | 0.0017464321 | 0.0032177113 | 0.0017464321 | 1 | 1 | 0 | -0.0271635568 | 0.0306564210 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [0.0014619977, -0.0002216422] | 0.0006201777 | 0.0011905132 | 0.0006201777 | 1 | 1 | 0 | -0.0100761588 | 0.0113165142 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 2 | 2 | [3101, 3102] | [0.0014038380, 0.0000353654] | 0.0007196017 | 0.0009676562 | 0.0007196017 | 2 | 0 | 0 | -0.0079744448 | 0.0094136481 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [-0.0010030834, -0.0004540974] | -0.0007285904 | 0.0003881917 | -0.0007285904 | 2 | 0 | 0 | -0.0042163547 | 0.0027591739 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0038737752, -0.0000090613] | 0.0019323570 | 0.0027455800 | 0.0019323570 | 1 | 1 | 0 | -0.0227357008 | 0.0266004148 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 2 | 2 | [3101, 3102] | [-0.0008465445, 0.0005187389] | -0.0001639028 | 0.0009654011 | -0.0001639028 | 1 | 1 | 0 | -0.0088376875 | 0.0085098819 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [0.0041351743, -0.0005464624] | 0.0017943559 | 0.0033104170 | 0.0017943559 | 1 | 1 | 0 | -0.0279485611 | 0.0315372730 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0002241382, 0.0005321509] | 0.0001540063 | 0.0005347772 | 0.0001540063 | 1 | 1 | 0 | -0.0046507757 | 0.0049587884 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 2 | 2 | [3101, 3102] | [-0.0027387587, 0.0009495268] | -0.0008946159 | 0.0026080117 | -0.0008946159 | 1 | 1 | 0 | -0.0243266715 | 0.0225374397 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [-0.0011213165, -0.0035030650] | -0.0023121907 | 0.0016841505 | -0.0023121907 | 2 | 0 | 0 | -0.0174436826 | 0.0128193012 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0040751233, -0.0007452850] | 0.0016649192 | 0.0034085434 | 0.0016649192 | 1 | 1 | 0 | -0.0289596284 | 0.0322894667 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 2 | 2 | [3101, 3102] | [0.0016198961, 0.0078932986] | 0.0047565973 | 0.0044359655 | 0.0047565973 | 0 | 2 | 0 | -0.0350989711 | 0.0446121658 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [0.0035111877, 0.0004045849] | 0.0019578863 | 0.0021966999 | 0.0019578863 | 2 | 0 | 0 | -0.0177786791 | 0.0216944516 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [0.0019368987, -0.0001104378] | 0.0009132304 | 0.0014476855 | 0.0009132304 | 1 | 1 | 0 | -0.0120937077 | 0.0139201686 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 2 | 2 | [3101, 3102] | [0.0017912425, -0.0004088819] | 0.0006911803 | 0.0015557229 | 0.0006911803 | 1 | 1 | 0 | -0.0132864354 | 0.0146687959 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [-0.0002559214, -0.0005345022] | -0.0003952118 | 0.0001969864 | -0.0003952118 | 2 | 0 | 0 | -0.0021650643 | 0.0013746407 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0030520044, 0.0004393486] | 0.0017456765 | 0.0018474267 | 0.0017456765 | 0 | 2 | 0 | -0.0148527937 | 0.0183441467 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 2 | 2 | [3101, 3102] | [-0.0024533618, 0.0024610335] | 0.0000038358 | 0.0034750022 | 0.0000038358 | 1 | 1 | 0 | -0.0312178204 | 0.0312254920 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R2_FREEZE-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R2_FREEZE-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R2_FREEZE-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R2_FREEZE-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R3_CUM-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R3_CUM-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R3_CUM-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R3_CUM-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R4_EMA-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R4_EMA-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0139370436, 0.0022598634] | 0.0080984535 | 0.0082570133 | 0.0080984535 | 2 | 0 | 0 | -0.0660878672 | 0.0822847742 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0115143771, -0.0026697666] | -0.0070920719 | 0.0062540841 | -0.0070920719 | 0 | 2 | 0 | -0.0632827880 | 0.0490986443 |
| R4_EMA-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0118203540, -0.0033574643] | 0.0042314448 | 0.0107323382 | 0.0042314448 | 1 | 1 | 0 | -0.0921947883 | 0.1006576780 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0018113068, -0.0022363992] | -0.0020238530 | 0.0003005857 | -0.0020238530 | 2 | 0 | 0 | -0.0047245084 | 0.0006768023 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0060088985, 0.0025663042] | 0.0042876013 | 0.0024342818 | 0.0042876013 | 0 | 2 | 0 | -0.0175835527 | 0.0261587554 |
| R4_EMA-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0054008422, 0.0084930908] | 0.0015461243 | 0.0098244943 | 0.0015461243 | 1 | 1 | 0 | -0.0867234545 | 0.0898157031 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0002073061, 0.0082815288] | 0.0042444175 | 0.0057093376 | 0.0042444175 | 2 | 0 | 0 | -0.0470519458 | 0.0555407808 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0025044752, -0.0002919451] | 0.0011062651 | 0.0019773678 | 0.0011062651 | 1 | 1 | 0 | -0.0166596796 | 0.0188722097 |
| R6_PB_HB-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0091383292, 0.0035682599] | 0.0063532946 | 0.0039386338 | 0.0063532946 | 2 | 0 | 0 | -0.0290339263 | 0.0417405154 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0003917382, -0.0031546846] | -0.0013814732 | 0.0025076997 | -0.0013814732 | 1 | 1 | 0 | -0.0239122608 | 0.0211493144 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0014793129, -0.0016838177] | -0.0015815653 | 0.0001446068 | -0.0015815653 | 2 | 0 | 0 | -0.0028808057 | -0.0002823249 |
| R6_PB_HB-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0056487848, -0.0006324826] | -0.0031406337 | 0.0035470613 | -0.0031406337 | 2 | 0 | 0 | -0.0350097154 | 0.0287284480 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0002073061, 0.0082815288] | 0.0042444175 | 0.0057093376 | 0.0042444175 | 2 | 0 | 0 | -0.0470519458 | 0.0555407808 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0025044752, -0.0002919451] | 0.0011062651 | 0.0019773678 | 0.0011062651 | 1 | 1 | 0 | -0.0166596796 | 0.0188722097 |
| R6_PB_HB-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0091383292, 0.0035682599] | 0.0063532946 | 0.0039386338 | 0.0063532946 | 2 | 0 | 0 | -0.0290339263 | 0.0417405154 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0003917382, -0.0031546846] | -0.0013814732 | 0.0025076997 | -0.0013814732 | 1 | 1 | 0 | -0.0239122608 | 0.0211493144 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0014793129, -0.0016838177] | -0.0015815653 | 0.0001446068 | -0.0015815653 | 2 | 0 | 0 | -0.0028808057 | -0.0002823249 |
| R6_PB_HB-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0056487848, -0.0006324826] | -0.0031406337 | 0.0035470613 | -0.0031406337 | 2 | 0 | 0 | -0.0350097154 | 0.0287284480 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [-0.0137297375, 0.0060216654] | -0.0038540360 | 0.0139663509 | -0.0038540360 | 1 | 1 | 0 | -0.1293367201 | 0.1216286480 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [0.0140188524, 0.0023778215] | 0.0081983369 | 0.0082314519 | 0.0081983369 | 2 | 0 | 0 | -0.0657583239 | 0.0821549977 |
| R6_PB_HB-R2_FREEZE | auc | 2 | 2 | [3101, 3102] | [-0.0026820247, 0.0069257242] | 0.0021218497 | 0.0067937044 | 0.0021218497 | 1 | 1 | 0 | -0.0589171626 | 0.0631608620 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [0.0022030451, -0.0009182854] | 0.0006423798 | 0.0022071140 | 0.0006423798 | 1 | 1 | 0 | -0.0191877525 | 0.0204725121 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [-0.0074882113, -0.0042501219] | -0.0058691666 | 0.0022896750 | -0.0058691666 | 2 | 0 | 0 | -0.0264410802 | 0.0147027470 |
| R6_PB_HB-R2_FREEZE | logloss | 2 | 2 | [3101, 3102] | [-0.0002479426, -0.0091255734] | -0.0046867580 | 0.0062774329 | -0.0046867580 | 2 | 0 | 0 | -0.0610872551 | 0.0517137391 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [-0.0137297375, 0.0060216654] | -0.0038540360 | 0.0139663509 | -0.0038540360 | 1 | 1 | 0 | -0.1293367201 | 0.1216286480 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [0.0140188524, 0.0023778215] | 0.0081983369 | 0.0082314519 | 0.0081983369 | 2 | 0 | 0 | -0.0657583239 | 0.0821549977 |
| R6_PB_HB-R3_CUM | auc | 2 | 2 | [3101, 3102] | [-0.0026820247, 0.0069257242] | 0.0021218497 | 0.0067937044 | 0.0021218497 | 1 | 1 | 0 | -0.0589171626 | 0.0631608620 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [0.0022030451, -0.0009182854] | 0.0006423798 | 0.0022071140 | 0.0006423798 | 1 | 1 | 0 | -0.0191877525 | 0.0204725121 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [-0.0074882113, -0.0042501219] | -0.0058691666 | 0.0022896750 | -0.0058691666 | 2 | 0 | 0 | -0.0264410802 | 0.0147027470 |
| R6_PB_HB-R3_CUM | logloss | 2 | 2 | [3101, 3102] | [-0.0002479426, -0.0091255734] | -0.0046867580 | 0.0062774329 | -0.0046867580 | 2 | 0 | 0 | -0.0610872551 | 0.0517137391 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [-0.0137297375, 0.0060216654] | -0.0038540360 | 0.0139663509 | -0.0038540360 | 1 | 1 | 0 | -0.1293367201 | 0.1216286480 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [0.0140188524, 0.0023778215] | 0.0081983369 | 0.0082314519 | 0.0081983369 | 2 | 0 | 0 | -0.0657583239 | 0.0821549977 |
| R6_PB_HB-R4_EMA | auc | 2 | 2 | [3101, 3102] | [-0.0026820247, 0.0069257242] | 0.0021218497 | 0.0067937044 | 0.0021218497 | 1 | 1 | 0 | -0.0589171626 | 0.0631608620 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [0.0022030451, -0.0009182854] | 0.0006423798 | 0.0022071140 | 0.0006423798 | 1 | 1 | 0 | -0.0191877525 | 0.0204725121 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [-0.0074882113, -0.0042501219] | -0.0058691666 | 0.0022896750 | -0.0058691666 | 2 | 0 | 0 | -0.0264410802 | 0.0147027470 |
| R6_PB_HB-R4_EMA | logloss | 2 | 2 | [3101, 3102] | [-0.0002479426, -0.0091255734] | -0.0046867580 | 0.0062774329 | -0.0046867580 | 2 | 0 | 0 | -0.0610872551 | 0.0517137391 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [0.0015678126, 0.0071713445] | 0.0043695785 | 0.0039622954 | 0.0043695785 | 2 | 0 | 0 | -0.0312302334 | 0.0399693905 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [0.0046563481, -0.0002583830] | 0.0021989825 | 0.0034752397 | 0.0021989825 | 1 | 1 | 0 | -0.0290248072 | 0.0334227722 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 2 | [3101, 3102] | [0.0080919699, 0.0028830283] | 0.0054874991 | 0.0036832780 | 0.0054874991 | 2 | 0 | 0 | -0.0276054403 | 0.0385804386 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [0.0015693553, -0.0038047253] | -0.0011176850 | 0.0038000488 | -0.0011176850 | 1 | 1 | 0 | -0.0352597692 | 0.0330243991 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [-0.0010871487, -0.0037666237] | -0.0024268862 | 0.0018946749 | -0.0024268862 | 2 | 0 | 0 | -0.0194498650 | 0.0145960926 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 2 | [3101, 3102] | [-0.0065433248, 0.0001675324] | -0.0031878962 | 0.0047452926 | -0.0031878962 | 1 | 1 | 0 | -0.0458226589 | 0.0394468665 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [-0.0007549076, -0.0000576335] | -0.0004062706 | 0.0004930473 | -0.0004062706 | 0 | 2 | 0 | -0.0048361245 | 0.0040235834 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [0.0002773398, -0.0012550636] | -0.0004888619 | 0.0010835728 | -0.0004888619 | 1 | 1 | 0 | -0.0102243775 | 0.0092466537 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 2 | 2 | [3101, 3102] | [0.0002643401, 0.0017040939] | 0.0009842170 | 0.0010180597 | 0.0009842170 | 2 | 0 | 0 | -0.0081626868 | 0.0101311208 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [0.0023319748, -0.0001259632] | 0.0011030058 | 0.0017380246 | 0.0011030058 | 1 | 1 | 0 | -0.0145125258 | 0.0167185373 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [-0.0012771509, -0.0024513404] | -0.0018642457 | 0.0008302773 | -0.0018642457 | 2 | 0 | 0 | -0.0093239913 | 0.0055955000 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 2 | 2 | [3101, 3102] | [0.0020500034, -0.0029699633] | -0.0004599799 | 0.0035496525 | -0.0004599799 | 1 | 1 | 0 | -0.0323523424 | 0.0314323826 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [-0.0000132270, -0.0000537428] | -0.0000334849 | 0.0000286490 | -0.0000334849 | 0 | 2 | 0 | -0.0002908856 | 0.0002239158 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [-0.0001824899, 0.0001740105] | -0.0000042397 | 0.0002520839 | -0.0000042397 | 1 | 1 | 0 | -0.0022691233 | 0.0022606439 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 2 | 2 | [3101, 3102] | [-0.0001554005, -0.0000212788] | -0.0000883396 | 0.0000948384 | -0.0000883396 | 0 | 2 | 0 | -0.0009404286 | 0.0007637493 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [-0.0021573962, -0.0012414231] | -0.0016994096 | 0.0006476908 | -0.0016994096 | 2 | 0 | 0 | -0.0075186805 | 0.0041198612 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [0.0000556604, -0.0028798455] | -0.0014120925 | 0.0020757162 | -0.0014120925 | 1 | 1 | 0 | -0.0200616622 | 0.0172374772 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 2 | 2 | [3101, 3102] | [-0.0004293781, 0.0025060146] | 0.0010383183 | 0.0020756361 | 0.0010383183 | 1 | 1 | 0 | -0.0176105320 | 0.0196871686 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0016007270, -0.0018096650] | -0.0001044690 | 0.0024115113 | -0.0001044690 | 1 | 1 | 0 | -0.0217710387 | 0.0215621007 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [-0.0005338795, -0.0030186686] | -0.0017762741 | 0.0017570112 | -0.0017762741 | 0 | 2 | 0 | -0.0175623934 | 0.0140098452 |
| L4_BLOCK_META-L0_FEATURE | auc | 2 | 2 | [3101, 3102] | [-0.0008421018, 0.0004131523] | -0.0002144747 | 0.0008875987 | -0.0002144747 | 1 | 1 | 0 | -0.0081892325 | 0.0077602830 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0137153749, 0.0100647063] | 0.0118900406 | 0.0025814125 | 0.0118900406 | 0 | 2 | 0 | -0.0113030307 | 0.0350831119 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0102923129, 0.0112908066] | 0.0107915598 | 0.0007060417 | 0.0107915598 | 0 | 2 | 0 | 0.0044480269 | 0.0171350927 |
| L4_BLOCK_META-L0_FEATURE | logloss | 2 | 2 | [3101, 3102] | [0.0104315731, 0.0031873295] | 0.0068094513 | 0.0051224538 | 0.0068094513 | 0 | 2 | 0 | -0.0392139704 | 0.0528328730 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [-0.0000710820, -0.0011438129] | -0.0006074475 | 0.0007585353 | -0.0006074475 | 0 | 2 | 0 | -0.0074226168 | 0.0062077219 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [0.0002059638, -0.0001691620] | 0.0000184009 | 0.0002652540 | 0.0000184009 | 1 | 1 | 0 | -0.0023648114 | 0.0024016133 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 2 | 2 | [3101, 3102] | [0.0008368607, -0.0000180040] | 0.0004094284 | 0.0006044806 | 0.0004094284 | 1 | 1 | 0 | -0.0050216148 | 0.0058404715 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [0.0029679621, 0.0010334768] | 0.0020007195 | 0.0013678877 | 0.0020007195 | 0 | 2 | 0 | -0.0102892639 | 0.0142907029 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0005334041, -0.0026302957] | -0.0010484458 | 0.0022370736 | -0.0010484458 | 1 | 1 | 0 | -0.0211477543 | 0.0190508628 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 2 | 2 | [3101, 3102] | [0.0005558059, -0.0037874407] | -0.0016158174 | 0.0030711391 | -0.0016158174 | 1 | 1 | 0 | -0.0292089077 | 0.0259772729 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [0.0001372018, 0.0004497059] | 0.0002934539 | 0.0002209738 | 0.0002934539 | 2 | 0 | 0 | -0.0016919167 | 0.0022788245 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0005818936, 0.0001316079] | -0.0002251428 | 0.0005045217 | -0.0002251428 | 1 | 1 | 0 | -0.0047580909 | 0.0043078052 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 2 | 2 | [3101, 3102] | [-0.0000275481, 0.0007567430] | 0.0003645975 | 0.0005545775 | 0.0003645975 | 1 | 1 | 0 | -0.0046180838 | 0.0053472787 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0002718507, -0.0036073159] | -0.0016677326 | 0.0027429850 | -0.0016677326 | 1 | 1 | 0 | -0.0263124747 | 0.0229770095 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0018471337, -0.0019723180] | -0.0000625921 | 0.0027007602 | -0.0000625921 | 1 | 1 | 0 | -0.0243279595 | 0.0242027752 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 2 | 2 | [3101, 3102] | [0.0005505632, -0.0020365586] | -0.0007429977 | 0.0018293713 | -0.0007429977 | 1 | 1 | 0 | -0.0171792469 | 0.0156932515 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [0.0012393743, -0.0001500794] | 0.0005446475 | 0.0009824921 | 0.0005446475 | 1 | 1 | 0 | -0.0082826940 | 0.0093719889 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [-0.0002204800, -0.0000240585] | -0.0001222693 | 0.0001388909 | -0.0001222693 | 0 | 2 | 0 | -0.0013701546 | 0.0011256161 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 2 | 2 | [3101, 3102] | [0.0001138853, 0.0007593692] | 0.0004366273 | 0.0004564260 | 0.0004366273 | 2 | 0 | 0 | -0.0036641981 | 0.0045374526 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0072409251, -0.0003381150] | 0.0034514051 | 0.0053591906 | 0.0034514051 | 1 | 1 | 0 | -0.0446990122 | 0.0516018224 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0030970453, 0.0019978352] | 0.0025474403 | 0.0007772589 | 0.0025474403 | 0 | 2 | 0 | -0.0044359537 | 0.0095308342 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 2 | 2 | [3101, 3102] | [0.0022264306, -0.0010663754] | 0.0005800276 | 0.0023283655 | 0.0005800276 | 1 | 1 | 0 | -0.0203395060 | 0.0214995612 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0007528978, -0.0031422285] | -0.0019475632 | 0.0016895119 | -0.0019475632 | 0 | 2 | 0 | -0.0171272255 | 0.0132320991 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0011694524, -0.0014778224] | -0.0001541850 | 0.0018719060 | -0.0001541850 | 1 | 1 | 0 | -0.0169725927 | 0.0166642227 |
| R2_FREEZE-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0009484827, -0.0006475393] | 0.0001504717 | 0.0011285580 | 0.0001504717 | 1 | 1 | 0 | -0.0099892193 | 0.0102901628 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0003837498, 0.0126312649] | 0.0061237576 | 0.0092030051 | 0.0061237576 | 1 | 1 | 0 | -0.0765619631 | 0.0888094782 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0031089859, 0.0033279135] | 0.0001094638 | 0.0045515752 | 0.0001094638 | 1 | 1 | 0 | -0.0407848170 | 0.0410037445 |
| R2_FREEZE-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0011709441, -0.0093253907] | -0.0052481674 | 0.0057660645 | -0.0052481674 | 2 | 0 | 0 | -0.0570542015 | 0.0465578667 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0002022032, -0.0021565756] | -0.0009771862 | 0.0016679085 | -0.0009771862 | 1 | 1 | 0 | -0.0159627491 | 0.0140083768 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0011683498, 0.0003245990] | 0.0007464744 | 0.0005966219 | 0.0007464744 | 2 | 0 | 0 | -0.0046139604 | 0.0061069093 |
| R2_FREEZE-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0029713005, 0.0030912317] | 0.0030312661 | 0.0000848042 | 0.0030312661 | 2 | 0 | 0 | 0.0022693306 | 0.0037932016 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0008101131, 0.0092974765] | 0.0042436817 | 0.0071471451 | 0.0042436817 | 1 | 1 | 0 | -0.0599708695 | 0.0684582329 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0054863019, 0.0018246840] | -0.0018308089 | 0.0051696477 | -0.0018308089 | 1 | 1 | 0 | -0.0482782506 | 0.0446166327 |
| R2_FREEZE-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0045991266, -0.0067831286] | -0.0056911276 | 0.0015443226 | -0.0056911276 | 2 | 0 | 0 | -0.0195663156 | 0.0081840604 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0016567389, -0.0025871993] | -0.0021219691 | 0.0006579348 | -0.0021219691 | 0 | 2 | 0 | -0.0080332788 | 0.0037893406 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0007986890, -0.0017505196] | -0.0012746043 | 0.0006730459 | -0.0012746043 | 0 | 2 | 0 | -0.0073216820 | 0.0047724735 |
| R3_CUM-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0019168679, -0.0014485158] | 0.0002341761 | 0.0023796857 | 0.0002341761 | 1 | 1 | 0 | -0.0211464513 | 0.0216148035 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0023118104, 0.0091946034] | 0.0057532069 | 0.0048668696 | 0.0057532069 | 0 | 2 | 0 | -0.0379738813 | 0.0494802952 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0007923361, 0.0035817030] | 0.0013946835 | 0.0030929127 | 0.0013946835 | 1 | 1 | 0 | -0.0263940350 | 0.0291834019 |
| R3_CUM-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0045406244, -0.0078452100] | -0.0061929172 | 0.0023366949 | -0.0061929172 | 2 | 0 | 0 | -0.0271872880 | 0.0148014536 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0007016379, -0.0016015463] | -0.0011515921 | 0.0006363313 | -0.0011515921 | 0 | 2 | 0 | -0.0068688025 | 0.0045656182 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0007997915, 0.0000519018] | -0.0003739449 | 0.0006022382 | -0.0003739449 | 1 | 1 | 0 | -0.0057848400 | 0.0050369503 |
| R3_CUM-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0039396857, 0.0022902552] | 0.0031149704 | 0.0011663235 | 0.0031149704 | 2 | 0 | 0 | -0.0073640304 | 0.0135939713 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0018854471, 0.0058608150] | 0.0038731311 | 0.0028110096 | 0.0038731311 | 0 | 2 | 0 | -0.0213827878 | 0.0291290499 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0031696521, 0.0020784736] | -0.0005455892 | 0.0037109852 | -0.0005455892 | 1 | 1 | 0 | -0.0338874686 | 0.0327962901 |
| R3_CUM-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0079688069, -0.0053029478] | -0.0066358773 | 0.0018850470 | -0.0066358773 | 2 | 0 | 0 | -0.0235723527 | 0.0103005980 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0007870641, -0.0031884084] | -0.0012006721 | 0.0028110836 | -0.0012006721 | 1 | 1 | 0 | -0.0264572562 | 0.0240559119 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0003687378, -0.0000065220] | 0.0001811079 | 0.0002653487 | 0.0001811079 | 1 | 1 | 0 | -0.0022029556 | 0.0025651714 |
| R4_EMA-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0013070930, 0.0006434529] | 0.0009752729 | 0.0004692645 | 0.0009752729 | 2 | 0 | 0 | -0.0032409010 | 0.0051914469 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0063631552, 0.0049579917] | -0.0007025817 | 0.0080052597 | -0.0007025817 | 1 | 1 | 0 | -0.0726269866 | 0.0712218232 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0026088801, 0.0007664562] | -0.0009212120 | 0.0023867232 | -0.0009212120 | 1 | 1 | 0 | -0.0223650692 | 0.0205226453 |
| R4_EMA-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0026413539, -0.0119943752] | -0.0073178645 | 0.0066135848 | -0.0073178645 | 2 | 0 | 0 | -0.0667385666 | 0.0521028375 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0017421652, -0.0022027555] | -0.0002302952 | 0.0027894801 | -0.0002302952 | 1 | 1 | 0 | -0.0252927799 | 0.0248321896 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0003676352, 0.0017958995] | 0.0010817673 | 0.0010099354 | 0.0010817673 | 2 | 0 | 0 | -0.0079921420 | 0.0101556767 |
| R4_EMA-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0033299108, 0.0043822238] | 0.0038560673 | 0.0007440977 | 0.0038560673 | 2 | 0 | 0 | -0.0028293853 | 0.0105415199 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0067895185, 0.0016242033] | -0.0025826576 | 0.0059493997 | -0.0025826576 | 1 | 1 | 0 | -0.0560358931 | 0.0508705779 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0049861961, -0.0007367732] | -0.0028614847 | 0.0030047957 | -0.0028614847 | 2 | 0 | 0 | -0.0298585028 | 0.0241355335 |
| R4_EMA-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0060695364, -0.0094521131] | -0.0077608247 | 0.0023918429 | -0.0077608247 | 2 | 0 | 0 | -0.0292506806 | 0.0137290312 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0014509971, -0.0006846833] | -0.0010678402 | 0.0005418656 | -0.0010678402 | 0 | 2 | 0 | -0.0059363099 | 0.0038006295 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0001746191, -0.0018029269] | -0.0009887730 | 0.0011513875 | -0.0009887730 | 0 | 2 | 0 | -0.0113335794 | 0.0093560334 |
| R6_PB_HB-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0000450853, -0.0003884383] | -0.0002167618 | 0.0002427872 | -0.0002167618 | 0 | 2 | 0 | -0.0023981185 | 0.0019645950 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0020135695, 0.0015562184] | 0.0017848939 | 0.0003233961 | 0.0017848939 | 0 | 2 | 0 | -0.0011207044 | 0.0046904922 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0006673312, 0.0014707490] | 0.0010690401 | 0.0005681022 | 0.0010690401 | 0 | 2 | 0 | -0.0040351555 | 0.0061732357 |
| R6_PB_HB-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0045404958, -0.0097904814] | -0.0026249928 | 0.0101335311 | -0.0026249928 | 1 | 1 | 0 | -0.0936711577 | 0.0884211721 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0004958961, 0.0003009696] | -0.0000974632 | 0.0005634691 | -0.0000974632 | 1 | 1 | 0 | -0.0051600323 | 0.0049651058 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0001757217, -0.0000005055] | -0.0000881136 | 0.0001238966 | -0.0000881136 | 0 | 2 | 0 | -0.0012012800 | 0.0010250529 |
| R6_PB_HB-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0019777325, 0.0033503327] | 0.0026640326 | 0.0009705749 | 0.0026640326 | 2 | 0 | 0 | -0.0060562371 | 0.0113843023 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0015872062, -0.0017775700] | -0.0000951819 | 0.0023792561 | -0.0000951819 | 1 | 1 | 0 | -0.0214719496 | 0.0212815857 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0017099847, -0.0000324804] | -0.0008712326 | 0.0011861747 | -0.0008712326 | 2 | 0 | 0 | -0.0115285891 | 0.0097861239 |
| R6_PB_HB-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [0.0011123132, -0.0072482192] | -0.0030679530 | 0.0059117892 | -0.0030679530 | 1 | 1 | 0 | -0.0561832718 | 0.0500473658 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [-0.0006980993, 0.0024575452] | 0.0008797230 | 0.0022313776 | 0.0008797230 | 1 | 1 | 0 | -0.0191684091 | 0.0209278550 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [-0.0013440714, -0.0003251045] | -0.0008345880 | 0.0007205184 | -0.0008345880 | 0 | 2 | 0 | -0.0073081893 | 0.0056390133 |
| R6_PB_HB-R2_FREEZE | auc | 2 | 2 | [3101, 3102] | [-0.0009935680, 0.0002591010] | -0.0003672335 | 0.0008857707 | -0.0003672335 | 1 | 1 | 0 | -0.0083255678 | 0.0075911008 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [0.0023973192, -0.0110750465] | -0.0043388637 | 0.0095264012 | -0.0043388637 | 1 | 1 | 0 | -0.0899301826 | 0.0812524552 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [0.0037763171, -0.0018571644] | 0.0009595764 | 0.0039834730 | 0.0009595764 | 1 | 1 | 0 | -0.0348305088 | 0.0367496615 |
| R6_PB_HB-R2_FREEZE | logloss | 2 | 2 | [3101, 3102] | [0.0057114399, -0.0004650907] | 0.0026231746 | 0.0043674666 | 0.0026231746 | 1 | 1 | 0 | -0.0366169562 | 0.0418633054 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [0.0002057419, 0.0019025159] | 0.0010541289 | 0.0011998005 | 0.0010541289 | 2 | 0 | 0 | -0.0097256505 | 0.0118339083 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [0.0006240699, -0.0000524073] | 0.0002858313 | 0.0004783416 | 0.0002858313 | 1 | 1 | 0 | -0.0040118974 | 0.0045835600 |
| R6_PB_HB-R3_CUM | auc | 2 | 2 | [3101, 3102] | [-0.0019619532, 0.0010600775] | -0.0004509378 | 0.0021368984 | -0.0004509378 | 1 | 1 | 0 | -0.0196502085 | 0.0187483328 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [-0.0002982410, -0.0076383850] | -0.0039683130 | 0.0051902656 | -0.0039683130 | 2 | 0 | 0 | -0.0506009995 | 0.0426643735 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [0.0014596673, -0.0021109540] | -0.0003256433 | 0.0025248106 | -0.0003256433 | 1 | 1 | 0 | -0.0230101662 | 0.0223588795 |
| R6_PB_HB-R3_CUM | logloss | 2 | 2 | [3101, 3102] | [0.0090811201, -0.0019452714] | 0.0035679243 | 0.0077968362 | 0.0035679243 | 1 | 1 | 0 | -0.0664838697 | 0.0736197184 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [-0.0022380612, 0.0025037251] | 0.0001328319 | 0.0033529492 | 0.0001328319 | 1 | 1 | 0 | -0.0299922218 | 0.0302578857 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [-0.0005433568, -0.0017964049] | -0.0011698809 | 0.0008860388 | -0.0011698809 | 0 | 2 | 0 | -0.0091306238 | 0.0067908620 |
| R6_PB_HB-R4_EMA | auc | 2 | 2 | [3101, 3102] | [-0.0013521783, -0.0010318911] | -0.0011920347 | 0.0002264772 | -0.0011920347 | 0 | 2 | 0 | -0.0032268519 | 0.0008427825 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [0.0083767246, -0.0034017733] | 0.0024874756 | 0.0083286558 | 0.0024874756 | 1 | 1 | 0 | -0.0723425275 | 0.0773174788 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [0.0032762113, 0.0007042928] | 0.0019902521 | 0.0018186210 | 0.0019902521 | 0 | 2 | 0 | -0.0143494095 | 0.0183299137 |
| R6_PB_HB-R4_EMA | logloss | 2 | 2 | [3101, 3102] | [0.0071818496, 0.0022038938] | 0.0046928717 | 0.0035199463 | 0.0046928717 | 0 | 2 | 0 | -0.0269325911 | 0.0363183346 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [-0.0013783858, 0.0025835604] | 0.0006025873 | 0.0028015190 | 0.0006025873 | 1 | 1 | 0 | -0.0245680624 | 0.0257732370 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0014608415, 0.0004360451] | -0.0005123982 | 0.0013413014 | -0.0005123982 | 1 | 1 | 0 | -0.0125635128 | 0.0115387164 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 2 | [3101, 3102] | [0.0008252000, 0.0017892940] | 0.0013072470 | 0.0006817174 | 0.0013072470 | 2 | 0 | 0 | -0.0048177410 | 0.0074322350 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [0.0040299384, -0.0043394713] | -0.0001547664 | 0.0059180664 | -0.0001547664 | 1 | 1 | 0 | -0.0533264832 | 0.0530169504 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [-0.0015113187, -0.0017959114] | -0.0016536150 | 0.0002012374 | -0.0016536150 | 2 | 0 | 0 | -0.0034616618 | 0.0001544317 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 2 | [3101, 3102] | [0.0017393559, -0.0050575445] | -0.0016590943 | 0.0048061344 | -0.0016590943 | 1 | 1 | 0 | -0.0448404986 | 0.0415223100 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [-0.0004296725, 0.0032933028] | 0.0014318152 | 0.0026325411 | 0.0014318152 | 1 | 1 | 0 | -0.0222206278 | 0.0250842581 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [-0.0003328770, 0.0015206827] | 0.0005939029 | 0.0013106646 | 0.0005939029 | 1 | 1 | 0 | -0.0111819517 | 0.0123697575 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 2 | 2 | [3101, 3102] | [-0.0006229474, 0.0003801349] | -0.0001214062 | 0.0007092863 | -0.0001214062 | 1 | 1 | 0 | -0.0064940908 | 0.0062512784 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [-0.0003140602, -0.0056402075] | -0.0029771338 | 0.0037661549 | -0.0029771338 | 2 | 0 | 0 | -0.0368146927 | 0.0308604251 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [0.0016859921, -0.0025022181] | -0.0004081130 | 0.0029615119 | -0.0004081130 | 1 | 1 | 0 | -0.0270162416 | 0.0262000156 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 2 | 2 | [3101, 3102] | [0.0005517536, 0.0000171639] | 0.0002844587 | 0.0003780120 | 0.0002844587 | 0 | 2 | 0 | -0.0031118440 | 0.0036807615 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [-0.0000556270, 0.0030827270] | 0.0015135500 | 0.0022191514 | 0.0015135500 | 1 | 1 | 0 | -0.0184247339 | 0.0214518339 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [0.0000312546, 0.0012795878] | 0.0006554212 | 0.0008827049 | 0.0006554212 | 2 | 0 | 0 | -0.0072753672 | 0.0085862096 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 2 | 2 | [3101, 3102] | [0.0000029155, 0.0003331862] | 0.0001680509 | 0.0002335367 | 0.0001680509 | 2 | 0 | 0 | -0.0019301928 | 0.0022662945 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [0.0000895244, -0.0044533370] | -0.0021819063 | 0.0032122881 | -0.0021819063 | 1 | 1 | 0 | -0.0310431699 | 0.0266793573 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [0.0000894998, -0.0014356564] | -0.0006730783 | 0.0010784483 | -0.0006730783 | 1 | 1 | 0 | -0.0103625515 | 0.0090163949 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 2 | 2 | [3101, 3102] | [-0.0000245982, -0.0002727582] | -0.0001486782 | 0.0001754756 | -0.0001486782 | 2 | 0 | 0 | -0.0017252641 | 0.0014279077 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0002108446, 0.0013432607] | 0.0007770526 | 0.0008007391 | 0.0007770526 | 2 | 0 | 0 | -0.0064173028 | 0.0079714081 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [-0.0006705695, -0.0003217842] | -0.0004961769 | 0.0002466284 | -0.0004961769 | 0 | 2 | 0 | -0.0027120456 | 0.0017196918 |
| L4_BLOCK_META-L0_FEATURE | auc | 2 | 2 | [3101, 3102] | [-0.0016398857, -0.0015808073] | -0.0016103465 | 0.0000417747 | -0.0016103465 | 0 | 2 | 0 | -0.0019856775 | -0.0012350154 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0009785635, 0.0005593404] | 0.0007689519 | 0.0002964355 | 0.0007689519 | 0 | 2 | 0 | -0.0018944154 | 0.0034323193 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0026348615, 0.0014090700] | 0.0020219657 | 0.0008667654 | 0.0020219657 | 0 | 2 | 0 | -0.0057656126 | 0.0098095441 |
| L4_BLOCK_META-L0_FEATURE | logloss | 2 | 2 | [3101, 3102] | [0.0020738202, 0.0037623263] | 0.0029180733 | 0.0011939541 | 0.0029180733 | 0 | 2 | 0 | -0.0078091788 | 0.0136453254 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [0.0000620448, 0.0029549365] | 0.0015084907 | 0.0020455834 | 0.0015084907 | 2 | 0 | 0 | -0.0168703468 | 0.0198873281 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [-0.0000082697, 0.0017286800] | 0.0008602051 | 0.0012282089 | 0.0008602051 | 1 | 1 | 0 | -0.0101748142 | 0.0118952245 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 2 | 2 | [3101, 3102] | [-0.0000083995, 0.0012887456] | 0.0006401731 | 0.0009172201 | 0.0006401731 | 1 | 1 | 0 | -0.0076007225 | 0.0088810686 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [-0.0000935860, -0.0022507052] | -0.0011721456 | 0.0015253136 | -0.0011721456 | 2 | 0 | 0 | -0.0148765444 | 0.0125322532 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [-0.0000540586, -0.0018911154] | -0.0009725870 | 0.0012989953 | -0.0009725870 | 2 | 0 | 0 | -0.0126435971 | 0.0106984230 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 2 | 2 | [3101, 3102] | [-0.0000108332, 0.0000264679] | 0.0000078174 | 0.0000263759 | 0.0000078174 | 1 | 1 | 0 | -0.0002291604 | 0.0002447951 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [-0.0000192331, 0.0030455472] | 0.0015131570 | 0.0021671269 | 0.0015131570 | 1 | 1 | 0 | -0.0179577060 | 0.0209840201 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0000033252, 0.0017608752] | 0.0008787750 | 0.0012474781 | 0.0008787750 | 1 | 1 | 0 | -0.0103293708 | 0.0120869208 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 2 | 2 | [3101, 3102] | [-0.0000158250, 0.0012529335] | 0.0006185543 | 0.0008971477 | 0.0006185543 | 1 | 1 | 0 | -0.0074419982 | 0.0086791067 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0000333010, -0.0024124861] | -0.0011895926 | 0.0017294327 | -0.0011895926 | 1 | 1 | 0 | -0.0167279287 | 0.0143487436 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0000443627, -0.0019909417] | -0.0009732895 | 0.0014391775 | -0.0009732895 | 1 | 1 | 0 | -0.0139037866 | 0.0119572076 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 2 | 2 | [3101, 3102] | [-0.0000257989, 0.0000524603] | 0.0000133307 | 0.0000553376 | 0.0000133307 | 1 | 1 | 0 | -0.0004838581 | 0.0005105196 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [-0.0020590480, 0.0002337964] | -0.0009126258 | 0.0016212858 | -0.0009126258 | 1 | 1 | 0 | -0.0154793010 | 0.0136540494 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [-0.0006859732, -0.0004835655] | -0.0005847694 | 0.0001431239 | -0.0005847694 | 0 | 2 | 0 | -0.0018706862 | 0.0007011474 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 2 | 2 | [3101, 3102] | [-0.0016552507, -0.0002834622] | -0.0009693565 | 0.0009700010 | -0.0009693565 | 0 | 2 | 0 | -0.0096844695 | 0.0077457565 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0003965909, 0.0003067170] | 0.0003516540 | 0.0000635505 | 0.0003516540 | 0 | 2 | 0 | -0.0002193244 | 0.0009226323 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0017460022, 0.0011532251] | 0.0014496136 | 0.0004191567 | 0.0014496136 | 0 | 2 | 0 | -0.0023163598 | 0.0052155871 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 2 | 2 | [3101, 3102] | [0.0010499153, 0.0014350022] | 0.0012424588 | 0.0002722976 | 0.0012424588 | 0 | 2 | 0 | -0.0012040377 | 0.0036889552 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0003062410, 0.0004200101] | 0.0003631255 | 0.0000804469 | 0.0003631255 | 2 | 0 | 0 | -0.0003596610 | 0.0010859121 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0001180109, -0.0004380075] | -0.0002780092 | 0.0002262717 | -0.0002780092 | 0 | 2 | 0 | -0.0023109801 | 0.0017549618 |
| R2_FREEZE-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0000327726, -0.0004999107] | -0.0002335690 | 0.0003766640 | -0.0002335690 | 1 | 1 | 0 | -0.0036177611 | 0.0031506230 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0001796840, -0.0006693732] | -0.0004245286 | 0.0003462625 | -0.0004245286 | 2 | 0 | 0 | -0.0035355739 | 0.0026865167 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0003497567, 0.0000521762] | 0.0002009664 | 0.0002104212 | 0.0002009664 | 0 | 2 | 0 | -0.0016895927 | 0.0020915256 |
| R2_FREEZE-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0000228949, -0.0008770101] | -0.0004270576 | 0.0006363289 | -0.0004270576 | 1 | 1 | 0 | -0.0061442463 | 0.0052901310 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0015969568, 0.0006438245] | 0.0011203906 | 0.0006739663 | 0.0011203906 | 2 | 0 | 0 | -0.0049349559 | 0.0071757372 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0006246295, -0.0002065250] | 0.0002090522 | 0.0005877150 | 0.0002090522 | 1 | 1 | 0 | -0.0050713577 | 0.0054894622 |
| R2_FREEZE-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0010296460, -0.0003725810] | 0.0003285325 | 0.0009915243 | 0.0003285325 | 1 | 1 | 0 | -0.0085799596 | 0.0092370246 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0035198497, -0.0008156561] | -0.0021677529 | 0.0019121537 | -0.0021677529 | 2 | 0 | 0 | -0.0193477721 | 0.0150122663 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0009078712, -0.0004005165] | -0.0006541938 | 0.0003587539 | -0.0006541938 | 2 | 0 | 0 | -0.0038774698 | 0.0025690821 |
| R2_FREEZE-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0001371276, 0.0002823042] | 0.0000725883 | 0.0002965831 | 0.0000725883 | 1 | 1 | 0 | -0.0025921052 | 0.0027372818 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0002768872, 0.0001878359] | 0.0002323615 | 0.0000629688 | 0.0002323615 | 2 | 0 | 0 | -0.0003333905 | 0.0007981135 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0000536513, -0.0001524659] | -0.0001030586 | 0.0000698725 | -0.0001030586 | 0 | 2 | 0 | -0.0007308381 | 0.0005247209 |
| R3_CUM-R0_BASE | auc | 2 | 2 | [3101, 3102] | [0.0000734687, -0.0000637879] | 0.0000048404 | 0.0000970551 | 0.0000048404 | 1 | 1 | 0 | -0.0008671647 | 0.0008768454 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0005324895, -0.0002822058] | -0.0004073477 | 0.0001769773 | -0.0004073477 | 2 | 0 | 0 | -0.0019974258 | 0.0011827305 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0006954646, -0.0001227562] | 0.0002863542 | 0.0005785695 | 0.0002863542 | 1 | 1 | 0 | -0.0049118864 | 0.0054845948 |
| R3_CUM-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [-0.0003992594, -0.0004204410] | -0.0004098502 | 0.0000149777 | -0.0004098502 | 2 | 0 | 0 | -0.0005444191 | -0.0002752813 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0015676029, 0.0004116503] | 0.0009896266 | 0.0008173819 | 0.0009896266 | 2 | 0 | 0 | -0.0063542585 | 0.0083335118 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0006889892, 0.0000790165] | 0.0003840028 | 0.0004313158 | 0.0003840028 | 2 | 0 | 0 | -0.0034912157 | 0.0042592214 |
| R3_CUM-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0010703421, 0.0000635418] | 0.0005669419 | 0.0007119153 | 0.0005669419 | 2 | 0 | 0 | -0.0058293632 | 0.0069632471 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0038726552, -0.0004284887] | -0.0021505720 | 0.0024353935 | -0.0021505720 | 2 | 0 | 0 | -0.0240317146 | 0.0197305707 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0005621633, -0.0005754490] | -0.0005688061 | 0.0000093944 | -0.0005688061 | 2 | 0 | 0 | -0.0006532116 | -0.0004844006 |
| R3_CUM-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0005592819, 0.0007388734] | 0.0000897957 | 0.0009179344 | 0.0000897957 | 1 | 1 | 0 | -0.0081575175 | 0.0083371090 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0003018626, 0.0001878359] | 0.0002448492 | 0.0000806291 | 0.0002448492 | 2 | 0 | 0 | -0.0004795743 | 0.0009692728 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0000075100, -0.0001524659] | -0.0000724779 | 0.0001131200 | -0.0000724779 | 1 | 1 | 0 | -0.0010888213 | 0.0009438654 |
| R4_EMA-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0000142587, -0.0000637879] | -0.0000390233 | 0.0000350224 | -0.0000390233 | 0 | 2 | 0 | -0.0003536874 | 0.0002756408 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0005136579, -0.0002822058] | -0.0003979319 | 0.0001636614 | -0.0003979319 | 2 | 0 | 0 | -0.0018683710 | 0.0010725073 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0001537042, -0.0001227562] | 0.0000154740 | 0.0001954871 | 0.0000154740 | 1 | 1 | 0 | -0.0017409077 | 0.0017718557 |
| R4_EMA-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0000501684, -0.0004204410] | -0.0001851363 | 0.0003327711 | -0.0001851363 | 1 | 1 | 0 | -0.0031749661 | 0.0028046935 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0015925784, 0.0004116503] | 0.0010021143 | 0.0008350422 | 0.0010021143 | 2 | 0 | 0 | -0.0065004424 | 0.0085046711 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0007501504, 0.0000790165] | 0.0004145835 | 0.0004745633 | 0.0004145835 | 2 | 0 | 0 | -0.0038491989 | 0.0046783658 |
| R4_EMA-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0009826147, 0.0000635418] | 0.0005230782 | 0.0006498827 | 0.0005230782 | 2 | 0 | 0 | -0.0053158860 | 0.0063620425 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0038538236, -0.0004284887] | -0.0021411562 | 0.0024220776 | -0.0021411562 | 2 | 0 | 0 | -0.0239026598 | 0.0196203474 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0011039236, -0.0005754490] | -0.0008396863 | 0.0003736880 | -0.0008396863 | 2 | 0 | 0 | -0.0041971396 | 0.0025177671 |
| R4_EMA-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0001098541, 0.0007388734] | 0.0003145096 | 0.0006001409 | 0.0003145096 | 1 | 1 | 0 | -0.0050775427 | 0.0057065620 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0012476448, -0.0002330809] | -0.0007403629 | 0.0007174050 | -0.0007403629 | 0 | 2 | 0 | -0.0071859913 | 0.0057052655 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0009030519, -0.0006420197] | -0.0007725358 | 0.0001845776 | -0.0007725358 | 0 | 2 | 0 | -0.0024308999 | 0.0008858283 |
| R6_PB_HB-R0_BASE | auc | 2 | 2 | [3101, 3102] | [-0.0006943658, -0.0003936508] | -0.0005440083 | 0.0002126377 | -0.0005440083 | 0 | 2 | 0 | -0.0024544818 | 0.0013664652 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0033507077, 0.0002959363] | 0.0018233220 | 0.0021600495 | 0.0018233220 | 0 | 2 | 0 | -0.0175839531 | 0.0212305971 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0013958942, 0.0000446107] | 0.0007202525 | 0.0009555017 | 0.0007202525 | 0 | 2 | 0 | -0.0078645900 | 0.0093050950 |
| R6_PB_HB-R0_BASE | logloss | 2 | 2 | [3101, 3102] | [0.0000733087, -0.0007528638] | -0.0003397775 | 0.0005841922 | -0.0003397775 | 1 | 1 | 0 | -0.0055885358 | 0.0049089808 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0000430709, -0.0000092665] | 0.0000169022 | 0.0000370081 | 0.0000169022 | 1 | 1 | 0 | -0.0003156025 | 0.0003494070 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0001604115, -0.0004105373] | -0.0002854744 | 0.0001768657 | -0.0002854744 | 0 | 2 | 0 | -0.0018745493 | 0.0013036005 |
| R6_PB_HB-R1_STATIC | auc | 2 | 2 | [3101, 3102] | [0.0003025075, -0.0002663211] | 0.0000180932 | 0.0004022226 | 0.0000180932 | 1 | 1 | 0 | -0.0035957334 | 0.0036319198 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0000105419, 0.0001496534] | 0.0000800977 | 0.0000983667 | 0.0000800977 | 0 | 2 | 0 | -0.0008036917 | 0.0009638870 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0001382664, -0.0004080820] | -0.0001349078 | 0.0003863267 | -0.0001349078 | 1 | 1 | 0 | -0.0036059152 | 0.0033360996 |
| R6_PB_HB-R1_STATIC | logloss | 2 | 2 | [3101, 3102] | [-0.0000867138, 0.0004064506] | 0.0001598684 | 0.0003487199 | 0.0001598684 | 1 | 1 | 0 | -0.0029732555 | 0.0032929923 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [-0.0015538858, -0.0006530910] | -0.0011034884 | 0.0006369581 | -0.0011034884 | 0 | 2 | 0 | -0.0068263302 | 0.0046193534 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [-0.0007850410, -0.0002040123] | -0.0004945266 | 0.0004108494 | -0.0004945266 | 0 | 2 | 0 | -0.0041858617 | 0.0031968084 |
| R6_PB_HB-R2_FREEZE | auc | 2 | 2 | [3101, 3102] | [-0.0007271385, 0.0001062599] | -0.0003104393 | 0.0005893017 | -0.0003104393 | 1 | 1 | 0 | -0.0056051048 | 0.0049842263 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [0.0035303917, 0.0009653095] | 0.0022478506 | 0.0018137870 | 0.0022478506 | 0 | 2 | 0 | -0.0140483792 | 0.0185440804 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [0.0010461376, -0.0000075655] | 0.0005192860 | 0.0007450806 | 0.0005192860 | 1 | 1 | 0 | -0.0061749974 | 0.0072135694 |
| R6_PB_HB-R2_FREEZE | logloss | 2 | 2 | [3101, 3102] | [0.0000504138, 0.0001241464] | 0.0000872801 | 0.0000521368 | 0.0000872801 | 0 | 2 | 0 | -0.0003811503 | 0.0005557104 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [-0.0015245320, -0.0004209168] | -0.0009727244 | 0.0007803738 | -0.0009727244 | 0 | 2 | 0 | -0.0079841048 | 0.0060386560 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [-0.0008494006, -0.0004895538] | -0.0006694772 | 0.0002544501 | -0.0006694772 | 0 | 2 | 0 | -0.0029556209 | 0.0016166664 |
| R6_PB_HB-R3_CUM | auc | 2 | 2 | [3101, 3102] | [-0.0007678345, -0.0003298629] | -0.0005488487 | 0.0003096927 | -0.0005488487 | 0 | 2 | 0 | -0.0033313273 | 0.0022336299 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [0.0038831972, 0.0005781421] | 0.0022306696 | 0.0023370269 | 0.0022306696 | 0 | 2 | 0 | -0.0187666836 | 0.0232280229 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [0.0007004296, 0.0001673669] | 0.0004338983 | 0.0003769323 | 0.0004338983 | 0 | 2 | 0 | -0.0029527036 | 0.0038205002 |
| R6_PB_HB-R3_CUM | logloss | 2 | 2 | [3101, 3102] | [0.0004725681, -0.0003324228] | 0.0000700727 | 0.0005692145 | 0.0000700727 | 1 | 1 | 0 | -0.0050441167 | 0.0051842621 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [-0.0015495074, -0.0004209168] | -0.0009852121 | 0.0007980341 | -0.0009852121 | 0 | 2 | 0 | -0.0081552641 | 0.0061848399 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [-0.0009105619, -0.0004895538] | -0.0007000579 | 0.0002976977 | -0.0007000579 | 0 | 2 | 0 | -0.0033747653 | 0.0019746496 |
| R6_PB_HB-R4_EMA | auc | 2 | 2 | [3101, 3102] | [-0.0006801072, -0.0003298629] | -0.0005049850 | 0.0002476601 | -0.0005049850 | 0 | 2 | 0 | -0.0027301227 | 0.0017201526 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [0.0038643656, 0.0005781421] | 0.0022212538 | 0.0023237109 | 0.0022212538 | 0 | 2 | 0 | -0.0186564604 | 0.0230989681 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [0.0012421900, 0.0001673669] | 0.0007047785 | 0.0007600147 | 0.0007047785 | 0 | 2 | 0 | -0.0061236823 | 0.0075332392 |
| R6_PB_HB-R4_EMA | logloss | 2 | 2 | [3101, 3102] | [0.0000231403, -0.0003324228] | -0.0001546412 | 0.0002514210 | -0.0001546412 | 1 | 1 | 0 | -0.0024135697 | 0.0021042872 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [-0.0015130618, -0.0002676823] | -0.0008903721 | 0.0008806163 | -0.0008903721 | 0 | 2 | 0 | -0.0088023953 | 0.0070216511 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0009657882, -0.0001927451] | -0.0005792667 | 0.0005466240 | -0.0005792667 | 0 | 2 | 0 | -0.0054904883 | 0.0043319549 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 2 | [3101, 3102] | [-0.0007644117, -0.0002561502] | -0.0005102810 | 0.0003593952 | -0.0005102810 | 0 | 2 | 0 | -0.0037393184 | 0.0027187565 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [0.0035585171, 0.0003657889] | 0.0019621530 | 0.0022575997 | 0.0019621530 | 0 | 2 | 0 | -0.0183215759 | 0.0222458819 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [0.0011047944, -0.0001432557] | 0.0004807694 | 0.0008825046 | 0.0004807694 | 1 | 1 | 0 | -0.0074482201 | 0.0084097588 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 2 | [3101, 3102] | [0.0006648060, 0.0002727175] | 0.0004687618 | 0.0002772485 | 0.0004687618 | 0 | 2 | 0 | -0.0020222172 | 0.0029597407 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0079717455, -0.0106211580, -0.0067789442, -0.0079202490] | -0.0043371514 | 0.0083625878 | -0.0073495966 | 1 | 3 | 0 | -0.0176438948 | 0.0089695919 |
| R6_PB_HB-R0_BASE | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0014290576, 0.0022281487, -0.0014078082, 0.0020705399] | 0.0010799845 | 0.0016941513 | 0.0017497988 | 3 | 1 | 0 | -0.0016157882 | 0.0037757572 |
| R6_PB_HB-R0_BASE | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0071148959, -0.0060908951, 0.0008464160, 0.0046830994] | 0.0016383791 | 0.0057628535 | 0.0027647577 | 3 | 1 | 0 | -0.0075316069 | 0.0108083650 |
| R6_PB_HB-R0_BASE | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0084740850, 0.0014601166, -0.0029003764, -0.0001750468] | -0.0025223479 | 0.0043564483 | -0.0015377116 | 3 | 1 | 0 | -0.0094544294 | 0.0044097336 |
| R6_PB_HB-R0_BASE | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0065585434, -0.0063315998, -0.0000796888, -0.0007393053] | -0.0034272843 | 0.0034962579 | -0.0035354526 | 4 | 0 | 0 | -0.0089906109 | 0.0021360422 |
| R6_PB_HB-R0_BASE | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0215075848, 0.0077662278, -0.0462228893, -0.0046490947] | -0.0161533352 | 0.0233619285 | -0.0130783397 | 3 | 1 | 0 | -0.0533273768 | 0.0210207063 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0049922079, -0.0035447953, -0.0193795096, 0.0024177548] | -0.0038785855 | 0.0109350345 | -0.0005635203 | 2 | 2 | 0 | -0.0212786656 | 0.0135214945 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0003560280, 0.0016210879, -0.0021256338, 0.0018209487] | 0.0004181077 | 0.0018156357 | 0.0009885580 | 3 | 1 | 0 | -0.0024709739 | 0.0033071893 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0139047116, -0.0005084138, -0.0027937531, 0.0036997004] | 0.0035755613 | 0.0073926489 | 0.0015956433 | 2 | 2 | 0 | -0.0081877929 | 0.0153389154 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0122677623, -0.0102051979, 0.0153089312, -0.0049172438] | -0.0030203182 | 0.0126055203 | -0.0075612209 | 3 | 1 | 0 | -0.0230785140 | 0.0170378776 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0038922339, -0.0031691457, -0.0024794707, -0.0012981165] | -0.0027097417 | 0.0011037890 | -0.0028243082 | 4 | 0 | 0 | -0.0044661163 | -0.0009533671 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0239760515, 0.0015893182, -0.0401808859, -0.0079601332] | -0.0176319381 | 0.0183639441 | -0.0159680923 | 3 | 1 | 0 | -0.0468530711 | 0.0115891949 |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0116573432, 0.0057700497, -0.0053154192, -0.0140923012] | -0.0063237535 | 0.0088708197 | -0.0084863812 | 1 | 3 | 0 | -0.0204392072 | 0.0077917002 |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0051358453, 0.0051764022, 0.0038746167, 0.0006021713] | 0.0036972589 | 0.0021500712 | 0.0045052310 | 4 | 0 | 0 | 0.0002760158 | 0.0071185020 |
| R6_PB_HB-R7_PB_HISTORY | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0041186447, 0.0020078496, 0.0181718215, 0.0145593883] | 0.0097144260 | 0.0078677673 | 0.0093390165 | 4 | 0 | 0 | -0.0028049475 | 0.0222337995 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0017580018, -0.0310622294, -0.0149929331, -0.0058747160] | -0.0134219701 | 0.0129955874 | -0.0104338246 | 4 | 0 | 0 | -0.0341008496 | 0.0072569094 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0026221057, -0.0059322215, 0.0003280129, 0.0020434394] | -0.0015457187 | 0.0035020443 | -0.0011470464 | 2 | 2 | 0 | -0.0071182527 | 0.0040268152 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0267491703, -0.0293341405, -0.1396852473, -0.0701043468] | -0.0664682262 | 0.0526956585 | -0.0497192437 | 4 | 0 | 0 | -0.1503187780 | 0.0173823256 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0123103719, 0.0017577529, -0.0239710281, -0.0237214439] | -0.0145612727 | 0.0121631874 | -0.0180159079 | 1 | 3 | 0 | -0.0339156182 | 0.0047930727 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0000951462, 0.0024983509, -0.0009794698, 0.0025582207] | 0.0009954889 | 0.0018065318 | 0.0012016023 | 2 | 2 | 0 | -0.0018791064 | 0.0038700842 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0065975686, 0.0036383099, 0.0031597443, 0.0097809588] | 0.0057941454 | 0.0030620182 | 0.0051179392 | 4 | 0 | 0 | 0.0009217912 | 0.0106664996 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0136061958, -0.0095434844, 0.0145066824, 0.0006732694] | -0.0019924321 | 0.0125329518 | -0.0044351075 | 2 | 2 | 0 | -0.0219351551 | 0.0179502910 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0021248518, -0.0067838835, 0.0032750691, -0.0036726635] | -0.0023265824 | 0.0042070838 | -0.0028987577 | 3 | 1 | 0 | -0.0090209916 | 0.0043678268 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0353841443, 0.0158948609, -0.0438892797, -0.0275227466] | -0.0227253274 | 0.0266000795 | -0.0314534454 | 3 | 1 | 0 | -0.0650519898 | 0.0196013350 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0025951306, 0.0058393108, -0.0161388561, 0.0045544801] | -0.0020850489 | 0.0100771772 | 0.0009796747 | 2 | 2 | 0 | -0.0181200866 | 0.0139499887 |
| R6_PB_HB-R0_BASE | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0028979210, 0.0007010719, -0.0029262367, 0.0004255494] | -0.0011743841 | 0.0020097006 | -0.0012361858 | 2 | 2 | 0 | -0.0043722662 | 0.0020234980 |
| R6_PB_HB-R0_BASE | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0023173944, 0.0034918274, -0.0072754497, 0.0027606389] | -0.0008350945 | 0.0050108861 | 0.0002216222 | 2 | 2 | 0 | -0.0088085325 | 0.0071383436 |
| R6_PB_HB-R0_BASE | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0003549667, -0.0039406142, -0.0002964809, -0.0010019786] | -0.0013985101 | 0.0017246238 | -0.0006784726 | 4 | 0 | 0 | -0.0041427714 | 0.0013457512 |
| R6_PB_HB-R0_BASE | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0041015032, -0.0008879018, 0.0012802946, 0.0010021093] | 0.0013740013 | 0.0020577137 | 0.0011412020 | 1 | 3 | 0 | -0.0019002804 | 0.0046482830 |
| R6_PB_HB-R0_BASE | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0041257967, 0.0138595118, 0.0057908148, -0.0009682126] | 0.0057019777 | 0.0061516675 | 0.0049583058 | 1 | 3 | 0 | -0.0040866981 | 0.0154906535 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0070737926, 0.0089424319, -0.0275655817, 0.0022924215] | -0.0058511302 | 0.0158973711 | -0.0023906855 | 2 | 2 | 0 | -0.0311473951 | 0.0194451347 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0043871670, 0.0019026820, 0.0033620186, -0.0000579659] | 0.0023984754 | 0.0019290088 | 0.0026323503 | 3 | 1 | 0 | -0.0006710080 | 0.0054679588 |
| R6_PB_HB-R5_PB_H1 | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0064543323, 0.0041216902, -0.0163054619, 0.0009864637] | -0.0044129101 | 0.0090846543 | -0.0027339343 | 2 | 2 | 0 | -0.0188686224 | 0.0100428022 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0015851043, -0.0001768833, 0.0013353327, -0.0026729946] | 0.0000176398 | 0.0019553868 | 0.0005792247 | 2 | 2 | 0 | -0.0030938169 | 0.0031290965 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0008107483, 0.0016166423, -0.0019574884, 0.0013266188] | 0.0000437561 | 0.0017180284 | 0.0002579352 | 2 | 2 | 0 | -0.0026900106 | 0.0027775227 |
| R6_PB_HB-R5_PB_H1 | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0107778080, 0.0134932257, 0.0117273160, 0.0004620474] | 0.0091150993 | 0.0058774023 | 0.0112525620 | 0 | 4 | 0 | -0.0002371594 | 0.0184673579 |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0136621727, -0.0001053527, 0.0001237273, 0.0037967858] | -0.0024617531 | 0.0076780214 | 0.0000091873 | 2 | 2 | 0 | -0.0146791985 | 0.0097556924 |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0075929608, 0.0127890125, -0.0000340463, 0.0005980738] | 0.0052365002 | 0.0061070181 | 0.0040955173 | 3 | 1 | 0 | -0.0044811283 | 0.0149541288 |
| R6_PB_HB-R7_PB_HISTORY | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0065037272, -0.0029055768, -0.0019878332, 0.0012666942] | -0.0025326107 | 0.0031958499 | -0.0024467050 | 1 | 3 | 0 | -0.0076179211 | 0.0025526997 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0013094638, -0.0000158019, 0.0002256058, -0.0005196640] | -0.0004048310 | 0.0006783168 | -0.0002677329 | 3 | 1 | 0 | -0.0014841844 | 0.0006745225 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0024777359, -0.0061075637, -0.0000763550, 0.0001064308] | -0.0021388060 | 0.0028960147 | -0.0012770455 | 3 | 1 | 0 | -0.0067470116 | 0.0024693997 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0094811657, 0.0126080879, 0.0018929416, -0.0008700202] | 0.0057780437 | 0.0063154147 | 0.0056870536 | 1 | 3 | 0 | -0.0042711903 | 0.0158272778 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0193341913, 0.0169565684, -0.0292390641, -0.0076834591] | -0.0098250366 | 0.0199095388 | -0.0135088252 | 1 | 3 | 0 | -0.0415055557 | 0.0218554826 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0009573345, 0.0046042845, -0.0012964527, 0.0023071035] | 0.0011644002 | 0.0028104644 | 0.0006748845 | 2 | 2 | 0 | -0.0033076758 | 0.0056364762 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 2 | 4 | [4101, 4102, 4103, 4104] | [-0.0051721750, 0.0051147474, -0.0069347959, 0.0074922437] | 0.0001250051 | 0.0072358975 | -0.0000287138 | 2 | 2 | 0 | -0.0113889226 | 0.0116389328 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0012317668, 0.0000976615, 0.0040599792, 0.0012981269] | 0.0016718836 | 0.0016846931 | 0.0012649469 | 0 | 4 | 0 | -0.0010088391 | 0.0043526063 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0019641255, -0.0004693151, -0.0115639836, -0.0036145466] | -0.0034209300 | 0.0058894726 | -0.0020419309 | 3 | 1 | 0 | -0.0127923951 | 0.0059505352 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 2 | 4 | [4101, 4102, 4103, 4104] | [0.0064806308, 0.0103686127, 0.0043258086, -0.0165303871] | 0.0011611663 | 0.0120565307 | 0.0054032197 | 1 | 3 | 0 | -0.0180234645 | 0.0203457970 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0197884251] | -0.0197884251 | NaN | -0.0197884251 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0005351493] | -0.0005351493 | NaN | -0.0005351493 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 2 | 1 | [3101] | [-0.0024882589] | -0.0024882589 | NaN | -0.0024882589 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [0.0036268323] | 0.0036268323 | NaN | 0.0036268323 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [-0.0000571147] | -0.0000571147 | NaN | -0.0000571147 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 2 | 1 | [3101] | [0.0023026895] | 0.0023026895 | NaN | 0.0023026895 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0109416303] | -0.0109416303 | NaN | -0.0109416303 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [0.0049687510] | 0.0049687510 | NaN | 0.0049687510 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 2 | 1 | [3101] | [0.0026937697] | 0.0026937697 | NaN | 0.0026937697 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [0.0029397528] | 0.0029397528 | NaN | 0.0029397528 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [-0.0034541998] | -0.0034541998 | NaN | -0.0034541998 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 2 | 1 | [3101] | [-0.0016996527] | -0.0016996527 | NaN | -0.0016996527 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [0.0013405266] | 0.0013405266 | NaN | 0.0013405266 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0002283207] | -0.0002283207 | NaN | -0.0002283207 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 2 | 1 | [3101] | [0.0019543699] | 0.0019543699 | NaN | 0.0019543699 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [0.0049600481] | 0.0049600481 | NaN | 0.0049600481 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [-0.0007224812] | -0.0007224812 | NaN | -0.0007224812 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 2 | 1 | [3101] | [-0.0072345086] | -0.0072345086 | NaN | -0.0072345086 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0012795598] | -0.0012795598 | NaN | -0.0012795598 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0018036701] | -0.0018036701 | NaN | -0.0018036701 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 2 | 1 | [3101] | [-0.0031449295] | -0.0031449295 | NaN | -0.0031449295 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [-0.0016680452] | -0.0016680452 | NaN | -0.0016680452 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [0.0010400353] | 0.0010400353 | NaN | 0.0010400353 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 2 | 1 | [3101] | [0.0023201387] | 0.0023201387 | NaN | 0.0023201387 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0046754781] | -0.0046754781 | NaN | -0.0046754781 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0004851195] | -0.0004851195 | NaN | -0.0004851195 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 2 | 1 | [3101] | [0.0005203388] | 0.0005203388 | NaN | 0.0005203388 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0016850732] | 0.0016850732 | NaN | 0.0016850732 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0008384088] | 0.0008384088 | NaN | 0.0008384088 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 2 | 1 | [3101] | [-0.0007340158] | -0.0007340158 | NaN | -0.0007340158 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0003050089] | -0.0003050089 | NaN | -0.0003050089 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0009540483] | -0.0009540483 | NaN | -0.0009540483 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0020929510] | -0.0020929510 | NaN | -0.0020929510 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0007483806] | -0.0007483806 | NaN | -0.0007483806 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0009314425] | 0.0009314425 | NaN | 0.0009314425 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0035022606] | 0.0035022606 | NaN | 0.0035022606 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0034852727] | -0.0034852727 | NaN | -0.0034852727 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [0.0004529919] | 0.0004529919 | NaN | 0.0004529919 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 2 | 1 | [3101] | [-0.0008390660] | -0.0008390660 | NaN | -0.0008390660 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0012704116] | 0.0012704116 | NaN | 0.0012704116 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [-0.0009332889] | -0.0009332889 | NaN | -0.0009332889 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 2 | 1 | [3101] | [0.0005564723] | 0.0005564723 | NaN | 0.0005564723 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0135107658] | -0.0135107658 | NaN | -0.0135107658 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0061312631] | -0.0061312631 | NaN | -0.0061312631 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0082656818] | -0.0082656818 | NaN | -0.0082656818 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0022807922] | 0.0022807922 | NaN | 0.0022807922 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0032249070] | 0.0032249070 | NaN | 0.0032249070 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0075450603] | 0.0075450603 | NaN | 0.0075450603 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0012179951] | -0.0012179951 | NaN | -0.0012179951 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [0.0001832297] | 0.0001832297 | NaN | 0.0001832297 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 2 | 1 | [3101] | [0.0004325925] | 0.0004325925 | NaN | 0.0004325925 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0073164021] | 0.0073164021 | NaN | 0.0073164021 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [-0.0004031181] | -0.0004031181 | NaN | -0.0004031181 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 2 | 1 | [3101] | [-0.0026331856] | -0.0026331856 | NaN | -0.0026331856 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0007258457] | 0.0007258457 | NaN | 0.0007258457 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0003020033] | -0.0003020033 | NaN | -0.0003020033 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0023979996] | -0.0023979996 | NaN | -0.0023979996 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0074264707] | -0.0074264707 | NaN | -0.0074264707 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [-0.0066276209] | -0.0066276209 | NaN | -0.0066276209 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0019780797] | 0.0019780797 | NaN | 0.0019780797 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0045389697] | -0.0045389697 | NaN | -0.0045389697 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0023086955] | -0.0023086955 | NaN | -0.0023086955 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 2 | 1 | [3101] | [-0.0017611067] | -0.0017611067 | NaN | -0.0017611067 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0022888775] | 0.0022888775 | NaN | 0.0022888775 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0028863053] | 0.0028863053 | NaN | 0.0028863053 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 2 | 1 | [3101] | [-0.0003710588] | -0.0003710588 | NaN | -0.0003710588 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0000352006] | -0.0000352006 | NaN | -0.0000352006 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [0.0002564476] | 0.0002564476 | NaN | 0.0002564476 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0000097256] | -0.0000097256 | NaN | -0.0000097256 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0007669129] | -0.0007669129 | NaN | -0.0007669129 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0004050304] | 0.0004050304 | NaN | 0.0004050304 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0010993601] | 0.0010993601 | NaN | 0.0010993601 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0154179558] | -0.0154179558 | NaN | -0.0154179558 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0010040781] | -0.0010040781 | NaN | -0.0010040781 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0051015487] | -0.0051015487 | NaN | -0.0051015487 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0011933785] | 0.0011933785 | NaN | 0.0011933785 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0000359191] | 0.0000359191 | NaN | 0.0000359191 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0065389659] | 0.0065389659 | NaN | 0.0065389659 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0209671234] | -0.0209671234 | NaN | -0.0209671234 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0016155039] | -0.0016155039 | NaN | -0.0016155039 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0047328461] | -0.0047328461 | NaN | -0.0047328461 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0039501334] | 0.0039501334 | NaN | 0.0039501334 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0007039961] | 0.0007039961 | NaN | 0.0007039961 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0052889352] | 0.0052889352 | NaN | 0.0052889352 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0032843675] | 0.0032843675 | NaN | 0.0032843675 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0007135537] | -0.0007135537 | NaN | -0.0007135537 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0008762222] | -0.0008762222 | NaN | -0.0008762222 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0097828247] | -0.0097828247 | NaN | -0.0097828247 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [-0.0069469840] | -0.0069469840 | NaN | -0.0069469840 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [-0.0026232433] | -0.0026232433 | NaN | -0.0026232433 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0032242093] | 0.0032242093 | NaN | 0.0032242093 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [0.0007614730] | 0.0007614730 | NaN | 0.0007614730 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 2 | 1 | [3101] | [-0.0013935484] | -0.0013935484 | NaN | -0.0013935484 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0047238356] | -0.0047238356 | NaN | -0.0047238356 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [-0.0014412396] | -0.0014412396 | NaN | -0.0014412396 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 2 | 1 | [3101] | [0.0037905576] | 0.0037905576 | NaN | 0.0037905576 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0007112388] | 0.0007112388 | NaN | 0.0007112388 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [0.0002882231] | 0.0002882231 | NaN | 0.0002882231 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 2 | 1 | [3101] | [-0.0005316326] | -0.0005316326 | NaN | -0.0005316326 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0004140659] | 0.0004140659 | NaN | 0.0004140659 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0006614389] | 0.0006614389 | NaN | 0.0006614389 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 2 | 1 | [3101] | [0.0067513095] | 0.0067513095 | NaN | 0.0067513095 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [0.0007036051] | 0.0007036051 | NaN | 0.0007036051 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [0.0001224648] | 0.0001224648 | NaN | 0.0001224648 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 2 | 1 | [3101] | [0.0002528947] | 0.0002528947 | NaN | 0.0002528947 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0019228604] | 0.0019228604 | NaN | 0.0019228604 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [-0.0010977394] | -0.0010977394 | NaN | -0.0010977394 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 2 | 1 | [3101] | [0.0058698059] | 0.0058698059 | NaN | 0.0058698059 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [-0.0008833675] | -0.0008833675 | NaN | -0.0008833675 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [-0.0001684551] | -0.0001684551 | NaN | -0.0001684551 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 2 | 1 | [3101] | [0.0021674712] | 0.0021674712 | NaN | 0.0021674712 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0023414452] | 0.0023414452 | NaN | 0.0023414452 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0007407718] | 0.0007407718 | NaN | 0.0007407718 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 2 | 1 | [3101] | [0.0030697369] | 0.0030697369 | NaN | 0.0030697369 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 1 | [3101] | [-0.0011591699] | -0.0011591699 | NaN | -0.0011591699 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 1 | [3101] | [0.0007220072] | 0.0007220072 | NaN | 0.0007220072 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 2 | 1 | [3101] | [0.0020164368] | 0.0020164368 | NaN | 0.0020164368 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 1 | [3101] | [-0.0033089782] | -0.0033089782 | NaN | -0.0033089782 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 1 | [3101] | [0.0015129595] | 0.0015129595 | NaN | 0.0015129595 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 2 | 1 | [3101] | [-0.0312835544] | -0.0312835544 | NaN | -0.0312835544 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0001488939] | 0.0001488939 | NaN | 0.0001488939 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [0.0000533709] | 0.0000533709 | NaN | 0.0000533709 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 2 | 1 | [3101] | [-0.0005168833] | -0.0005168833 | NaN | -0.0005168833 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [-0.0001675647] | -0.0001675647 | NaN | -0.0001675647 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0023870641] | 0.0023870641 | NaN | 0.0023870641 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 2 | 1 | [3101] | [-0.0013809318] | -0.0013809318 | NaN | -0.0013809318 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [-0.0038032630] | -0.0038032630 | NaN | -0.0038032630 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [0.0014377680] | 0.0014377680 | NaN | 0.0014377680 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 2 | 1 | [3101] | [0.0035227254] | 0.0035227254 | NaN | 0.0035227254 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0002923055] | 0.0002923055 | NaN | 0.0002923055 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [0.0026887571] | 0.0026887571 | NaN | 0.0026887571 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 2 | 1 | [3101] | [-0.0049896625] | -0.0049896625 | NaN | -0.0049896625 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [0.0003999271] | 0.0003999271 | NaN | 0.0003999271 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [0.0000483819] | 0.0000483819 | NaN | 0.0000483819 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 2 | 1 | [3101] | [-0.0007108269] | -0.0007108269 | NaN | -0.0007108269 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0001198835] | 0.0001198835 | NaN | 0.0001198835 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0031186284] | 0.0031186284 | NaN | 0.0031186284 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 2 | 1 | [3101] | [0.0011678949] | 0.0011678949 | NaN | 0.0011678949 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 1 | [3101] | [-0.0067905841] | -0.0067905841 | NaN | -0.0067905841 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 1 | [3101] | [0.0130397991] | 0.0130397991 | NaN | 0.0130397991 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 2 | 1 | [3101] | [0.0019829764] | 0.0019829764 | NaN | 0.0019829764 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 1 | [3101] | [0.0023645079] | 0.0023645079 | NaN | 0.0023645079 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 1 | [3101] | [-0.0087525398] | -0.0087525398 | NaN | -0.0087525398 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 2 | 1 | [3101] | [-0.0049351813] | -0.0049351813 | NaN | -0.0049351813 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=endpoint

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0002934561] | 0.0002934561 | NaN | 0.0002934561 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [-0.0009329962] | -0.0009329962 | NaN | -0.0009329962 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 2 | 1 | [3101] | [0.0010787170] | 0.0010787170 | NaN | 0.0010787170 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0059750503] | 0.0059750503 | NaN | 0.0059750503 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [-0.0008754004] | -0.0008754004 | NaN | -0.0008754004 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 2 | 1 | [3101] | [-0.0000171184] | -0.0000171184 | NaN | -0.0000171184 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [0.0006543054] | 0.0006543054 | NaN | 0.0006543054 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [-0.0000208840] | -0.0000208840 | NaN | -0.0000208840 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 2 | 1 | [3101] | [0.0011049414] | 0.0011049414 | NaN | 0.0011049414 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0015333496] | 0.0015333496 | NaN | 0.0015333496 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [-0.0006497904] | -0.0006497904 | NaN | -0.0006497904 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 2 | 1 | [3101] | [0.0013265073] | 0.0013265073 | NaN | 0.0013265073 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [0.0002519149] | 0.0002519149 | NaN | 0.0002519149 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [-0.0001775512] | -0.0001775512 | NaN | -0.0001775512 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 2 | 1 | [3101] | [0.0001971221] | 0.0001971221 | NaN | 0.0001971221 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0067235670] | 0.0067235670 | NaN | 0.0067235670 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0019083498] | 0.0019083498 | NaN | 0.0019083498 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 2 | 1 | [3101] | [-0.0004416148] | -0.0004416148 | NaN | -0.0004416148 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=selected

| candidate - reference | metric | domain | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0000130952] | 0.0000130952 | NaN | 0.0000130952 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [0.0000929947] | 0.0000929947 | NaN | 0.0000929947 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 2 | 1 | [3101] | [-0.0001029738] | -0.0001029738 | NaN | -0.0001029738 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [-0.0002068827] | -0.0002068827 | NaN | -0.0002068827 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0006579073] | 0.0006579073 | NaN | 0.0006579073 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 2 | 1 | [3101] | [-0.0001659634] | -0.0001659634 | NaN | -0.0001659634 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [-0.0019260905] | -0.0019260905 | NaN | -0.0019260905 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [-0.0007213293] | -0.0007213293 | NaN | -0.0007213293 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 2 | 1 | [3101] | [-0.0017858142] | -0.0017858142 | NaN | -0.0017858142 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [-0.0025016138] | -0.0025016138 | NaN | -0.0025016138 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [0.0009069311] | 0.0009069311 | NaN | 0.0009069311 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 2 | 1 | [3101] | [0.0015568054] | 0.0015568054 | NaN | 0.0015568054 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [0.0000663186] | 0.0000663186 | NaN | 0.0000663186 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [-0.0000142831] | -0.0000142831 | NaN | -0.0000142831 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 2 | 1 | [3101] | [0.0000097181] | 0.0000097181 | NaN | 0.0000097181 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [-0.0001164137] | -0.0001164137 | NaN | -0.0001164137 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [-0.0000414669] | -0.0000414669 | NaN | -0.0000414669 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 2 | 1 | [3101] | [-0.0000366846] | -0.0000366846 | NaN | -0.0000366846 | 1 | 0 | 0 | NaN | NaN |

## task 均值配对

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6987795425 | -0.0006681707 | -0.0006681707 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7110801617 | -0.0101283945 | -0.0101283945 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4936081757 | 0.0090073303 | -0.0090073303 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4303410386 | 0.0090556752 | -0.0090556752 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6934194609 | 0.0046919108 | 0.0046919108 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.6994879037 | 0.0014638635 | 0.0014638635 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4952434226 | 0.0073720834 | -0.0073720834 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4334926186 | 0.0059040952 | -0.0059040952 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.7026200112 | -0.0017905921 | -0.0017905921 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.7037136325 | 0.0014405749 | 0.0014405749 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.4972096916 | -0.0022835728 | 0.0022835728 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4384578756 | -0.0048594660 | 0.0048594660 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 18 | 23 | 0.6995823739 | 0.6959252673 | 0.0036571066 | 0.0036571066 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 18 | 23 | 0.6993513415 | 0.7098833646 | -0.0105320232 | -0.0105320232 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 18 | 23 | 0.4970472474 | 0.4949770093 | 0.0020702381 | -0.0020702381 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 18 | 23 | 0.4351645517 | 0.4323726293 | 0.0027919224 | -0.0027919224 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 18 | 24 | 0.6995823739 | 0.6934194609 | 0.0061629130 | 0.0061629130 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 18 | 24 | 0.6993513415 | 0.6994879037 | -0.0001365623 | -0.0001365623 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 18 | 24 | 0.4970472474 | 0.4952434226 | 0.0018038248 | -0.0018038248 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 18 | 24 | 0.4351645517 | 0.4334926186 | 0.0016719331 | -0.0016719331 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6960970424 | 0.0047323767 | 0.0047323767 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.6996202861 | 0.0055339213 | 0.0055339213 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4970940516 | -0.0021679328 | 0.0021679328 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4356420694 | -0.0020436599 | 0.0020436599 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8062927216 | 0.0042386289 | 0.0042386289 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8181930664 | 0.0017576183 | 0.0017576183 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5228272646 | 0.0098516227 | -0.0098516227 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4209226243 | 0.0075195414 | -0.0075195414 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8056855740 | 0.0048457765 | 0.0048457765 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8127899481 | 0.0071607366 | 0.0071607366 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5259965917 | 0.0066822957 | -0.0066822957 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4315295732 | -0.0030874075 | 0.0030874075 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8083266778 | -0.0005321036 | -0.0005321036 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8210932454 | -0.0015371198 | -0.0015371198 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5306573169 | 0.0072956245 | -0.0072956245 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4238971203 | 0.0037458239 | -0.0037458239 |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 23 | 0.8020107835 | 0.8064851418 | -0.0044743583 | -0.0044743583 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 23 | 0.8130585180 | 0.8177198956 | -0.0046613775 | -0.0046613775 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 23 | 0.5283598453 | 0.5230855444 | 0.0052743009 | -0.0052743009 |
| L4_BLOCK_META-AUTOIFS | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 23 | 0.4242839314 | 0.4207125746 | 0.0035713569 | -0.0035713569 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 15 | 0.8020107835 | 0.7985770145 | 0.0034337689 | 0.0034337689 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 15 | 0.8130585180 | 0.8131822759 | -0.0001237579 | -0.0001237579 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 15 | 0.5283598453 | 0.5321313222 | -0.0037714769 | 0.0037714769 |
| L4_BLOCK_META-STEM_DOMAIN_TASK | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 15 | 0.4242839314 | 0.4255506306 | -0.0012666991 | 0.0012666991 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 0 | 12 | 14 | 0.8019029122 | 0.8029987725 | -0.0010958603 | -0.0010958603 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | auc | 1 | 12 | 14 | 0.8138258535 | 0.8150503810 | -0.0012245275 | -0.0012245275 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 0 | 12 | 14 | 0.5297834136 | 0.5286881824 | 0.0010952312 | -0.0010952312 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 3101 | logloss | 1 | 12 | 14 | 0.4237687798 | 0.4232365971 | 0.0005321827 | -0.0005321827 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7002976503 | 0.6973921196 | 0.0029055306 | 0.0029055306 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7013528919 | 0.7003057831 | 0.0010471087 | 0.0010471087 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6927940748 | 0.6913469494 | 0.0014471254 | 0.0014471254 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.6988733025 | 0.6962450147 | 0.0026282878 | 0.0026282878 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4997723262 | 0.5001697134 | -0.0003973871 | 0.0003973871 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4385201032 | 0.4392369077 | -0.0007168045 | 0.0007168045 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5039274243 | 0.5032154263 | 0.0007119980 | -0.0007119980 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4388055489 | 0.4388891376 | -0.0000835887 | 0.0000835887 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7002976503 | 0.6985177721 | 0.0017798781 | 0.0017798781 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7013528919 | 0.6994562705 | 0.0018966213 | 0.0018966213 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6927940748 | 0.6950114119 | -0.0022173371 | -0.0022173371 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.6988733025 | 0.7009854253 | -0.0021121228 | -0.0021121228 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4997723262 | 0.4994595479 | 0.0003127783 | -0.0003127783 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4385201032 | 0.4385442379 | -0.0000241348 | 0.0000241348 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5039274243 | 0.5007586669 | 0.0031687574 | -0.0031687574 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4388055489 | 0.4363081919 | 0.0024973570 | -0.0024973570 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6943493643 | 0.0037620075 | 0.0037620075 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.6968860994 | 0.0040656678 | 0.0040656678 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6951933559 | 0.6893864814 | 0.0058068745 | 0.0058068745 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7000883873 | 0.6933057895 | 0.0067825978 | 0.0067825978 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4960131392 | 0.0066023668 | -0.0066023668 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4355669117 | 0.0038298021 | -0.0038298021 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4998803631 | 0.4974389901 | 0.0024413730 | -0.0024413730 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4365617004 | 0.4359559022 | 0.0006057982 | -0.0006057982 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.7002976503 | -0.0021862785 | -0.0021862785 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7013528919 | -0.0004011246 | -0.0004011246 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6951933559 | 0.6927940748 | 0.0023992811 | 0.0023992811 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7000883873 | 0.6988733025 | 0.0012150848 | 0.0012150848 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4997723262 | 0.0028431797 | -0.0028431797 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4385201032 | 0.0008766106 | -0.0008766106 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4998803631 | 0.5039274243 | -0.0040470612 | 0.0040470612 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4365617004 | 0.4388055489 | -0.0022438485 | 0.0022438485 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6983645572 | -0.0002531854 | -0.0002531854 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7010976897 | -0.0001459225 | -0.0001459225 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6951933559 | 0.6952590787 | -0.0000657228 | -0.0000657228 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7000883873 | 0.7014213883 | -0.0013330010 | -0.0013330010 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.5009221256 | 0.0016933803 | -0.0016933803 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4377032114 | 0.0016935024 | -0.0016935024 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4998803631 | 0.5003941073 | -0.0005137442 | 0.0005137442 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4365617004 | 0.4366971069 | -0.0001354065 | 0.0001354065 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6992095417 | -0.0010981699 | -0.0010981699 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7016059126 | -0.0006541454 | -0.0006541454 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6951933559 | 0.6932644826 | 0.0019288733 | 0.0019288733 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7000883873 | 0.6979513405 | 0.0021370468 | 0.0021370468 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.5000445590 | 0.0025709469 | -0.0025709469 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4368376931 | 0.0025590207 | -0.0025590207 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4998803631 | 0.5013816046 | -0.0015012415 | 0.0015012415 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4365617004 | 0.4376255511 | -0.0010638507 | 0.0010638507 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6993300627 | 0.6984202904 | 0.0009097722 | 0.0009097722 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7061630195 | 0.7048330493 | 0.0013299702 | 0.0013299702 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5006082297 | 0.4986609311 | 0.0019472986 | -0.0019472986 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4387605846 | 0.4356252208 | 0.0031353638 | -0.0031353638 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6993300627 | 0.6984202904 | 0.0009097722 | 0.0009097722 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7061630195 | 0.7048330493 | 0.0013299702 | 0.0013299702 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5006082297 | 0.4986609311 | 0.0019472986 | -0.0019472986 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4387605846 | 0.4356252208 | 0.0031353638 | -0.0031353638 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6996461820 | 0.6984202904 | 0.0012258915 | 0.0012258915 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7065715994 | 0.7048330493 | 0.0017385501 | 0.0017385501 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5005233761 | 0.4986609311 | 0.0018624450 | -0.0018624450 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4380246304 | 0.4356252208 | 0.0023994096 | -0.0023994096 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6996461820 | 0.6984202904 | 0.0012258915 | 0.0012258915 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7065715994 | 0.7048330493 | 0.0017385501 | 0.0017385501 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5005233761 | 0.4986609311 | 0.0018624450 | -0.0018624450 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4380246304 | 0.4356252208 | 0.0023994096 | -0.0023994096 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6993455653 | 0.6984202904 | 0.0009252748 | 0.0009252748 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7054478215 | 0.7048330493 | 0.0006147722 | 0.0006147722 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5007615804 | 0.4986609311 | 0.0021006492 | -0.0021006492 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4377353589 | 0.4356252208 | 0.0021101381 | -0.0021101381 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.6861367314 | 0.6999462672 | -0.0138095357 | -0.0138095357 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.6905830937 | 0.7045359031 | -0.0139528093 | -0.0139528093 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6993455653 | 0.6984202904 | 0.0009252748 | 0.0009252748 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7054478215 | 0.7048330493 | 0.0006147722 | 0.0006147722 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5383608533 | 0.5013343846 | 0.0370264687 | -0.0370264687 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4743570682 | 0.4375141296 | 0.0368429386 | -0.0368429386 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5007615804 | 0.4986609311 | 0.0021006492 | -0.0021006492 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4377353589 | 0.4356252208 | 0.0021101381 | -0.0021101381 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6999462672 | 0.0008831519 | 0.0008831519 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.7045359031 | 0.0006183043 | 0.0006183043 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6984202904 | 0.0000565116 | 0.0000565116 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7048330493 | -0.0012275969 | -0.0012275969 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5013343846 | -0.0064082657 | 0.0064082657 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4375141296 | -0.0039157201 | 0.0039157201 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.4986609311 | -0.0000044797 | 0.0000044797 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4356252208 | 0.0013384902 | -0.0013384902 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6999462672 | 0.0008831519 | 0.0008831519 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.7045359031 | 0.0006183043 | 0.0006183043 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6984202904 | 0.0000565116 | 0.0000565116 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7048330493 | -0.0012275969 | -0.0012275969 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5013343846 | -0.0064082657 | 0.0064082657 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4375141296 | -0.0039157201 | 0.0039157201 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.4986609311 | -0.0000044797 | 0.0000044797 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4356252208 | 0.0013384902 | -0.0013384902 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6861367314 | 0.0146926877 | 0.0146926877 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.6905830937 | 0.0145711137 | 0.0145711137 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6993300627 | -0.0008532607 | -0.0008532607 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7061630195 | -0.0025575671 | -0.0025575671 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5383608533 | -0.0434347345 | 0.0434347345 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4743570682 | -0.0407586586 | 0.0407586586 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.5006082297 | -0.0019517783 | 0.0019517783 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4387605846 | -0.0017968736 | 0.0017968736 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6861367314 | 0.0146926877 | 0.0146926877 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.6905830937 | 0.0145711137 | 0.0145711137 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6996461820 | -0.0011693800 | -0.0011693800 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7065715994 | -0.0029661470 | -0.0029661470 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5383608533 | -0.0434347345 | 0.0434347345 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4743570682 | -0.0407586586 | 0.0407586586 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.5005233761 | -0.0018669247 | 0.0018669247 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4380246304 | -0.0010609194 | 0.0010609194 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6861367314 | 0.0146926877 | 0.0146926877 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.6905830937 | 0.0145711137 | 0.0145711137 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6993455653 | -0.0008687633 | -0.0008687633 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7054478215 | -0.0018423691 | -0.0018423691 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5383608533 | -0.0434347345 | 0.0434347345 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4743570682 | -0.0407586586 | 0.0407586586 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.5007615804 | -0.0021051290 | 0.0021051290 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4377353589 | -0.0007716479 | 0.0007716479 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6997035230 | 0.0011258961 | 0.0011258961 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.7055338494 | -0.0003796420 | -0.0003796420 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.6984768020 | 0.6968610668 | 0.0016157352 | 0.0016157352 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.7036054524 | 0.7025297456 | 0.0010757068 | 0.0010757068 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.4997341214 | -0.0048080026 | 0.0048080026 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4371830611 | -0.0035846515 | 0.0035846515 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.4986564514 | 0.4993198625 | -0.0006634111 | 0.0006634111 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4369637110 | 0.4363099183 | 0.0006537927 | -0.0006537927 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 15 | 15 | 0.6954072464 | 0.6935893442 | 0.0018179022 | 0.0018179022 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 15 | 15 | 0.6989347801 | 0.6945964702 | 0.0043383100 | 0.0043383100 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 23 | 23 | 0.6958214435 | 0.6946182807 | 0.0012031628 | 0.0012031628 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 23 | 23 | 0.7030510079 | 0.7005527628 | 0.0024982451 | 0.0024982451 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 15 | 15 | 0.4970856756 | 0.4976730481 | -0.0005873725 | 0.0005873725 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 15 | 15 | 0.4337766920 | 0.4350466730 | -0.0012699810 | 0.0012699810 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 23 | 23 | 0.5000982329 | 0.5006412202 | -0.0005429873 | 0.0005429873 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 23 | 23 | 0.4372420433 | 0.4381011899 | -0.0008591466 | 0.0008591466 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 15 | 15 | 0.6954072464 | 0.6949032918 | 0.0005039547 | 0.0005039547 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 15 | 15 | 0.6989347801 | 0.6972481958 | 0.0016865843 | 0.0016865843 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 23 | 22 | 0.6958214435 | 0.6966664109 | -0.0008449674 | -0.0008449674 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 23 | 22 | 0.7030510079 | 0.7035134468 | -0.0004624390 | -0.0004624390 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 15 | 15 | 0.4970856756 | 0.4974599537 | -0.0003742781 | 0.0003742781 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 15 | 15 | 0.4337766920 | 0.4345780205 | -0.0008013285 | 0.0008013285 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 23 | 22 | 0.5000982329 | 0.4994109262 | 0.0006873066 | -0.0006873066 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 23 | 22 | 0.4372420433 | 0.4359353959 | 0.0013066475 | -0.0013066475 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 18 | 14 | 0.6995823739 | 0.6901484982 | 0.0094338757 | 0.0094338757 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 18 | 14 | 0.6993513415 | 0.6905742648 | 0.0087770767 | 0.0087770767 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 21 | 19 | 0.6965144847 | 0.6906281310 | 0.0058863538 | 0.0058863538 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 21 | 19 | 0.7018812265 | 0.6965435965 | 0.0053376300 | 0.0053376300 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 18 | 14 | 0.4970472474 | 0.4973044747 | -0.0002572272 | 0.0002572272 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 18 | 14 | 0.4351645517 | 0.4359671555 | -0.0008026038 | 0.0008026038 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 21 | 19 | 0.4992309513 | 0.4978915277 | 0.0013394236 | -0.0013394236 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 21 | 19 | 0.4381463784 | 0.4357429607 | 0.0024034177 | -0.0024034177 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 18 | 15 | 0.6995823739 | 0.6954072464 | 0.0041751275 | 0.0041751275 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 18 | 15 | 0.6993513415 | 0.6989347801 | 0.0004165613 | 0.0004165613 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 21 | 23 | 0.6965144847 | 0.6958214435 | 0.0006930412 | 0.0006930412 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 21 | 23 | 0.7018812265 | 0.7030510079 | -0.0011697813 | -0.0011697813 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 18 | 15 | 0.4970472474 | 0.4970856756 | -0.0000384281 | 0.0000384281 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 18 | 15 | 0.4351645517 | 0.4337766920 | 0.0013878597 | -0.0013878597 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 21 | 23 | 0.4992309513 | 0.5000982329 | -0.0008672816 | 0.0008672816 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 21 | 23 | 0.4381463784 | 0.4372420433 | 0.0009043350 | -0.0009043350 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 18 | 18 | 0.6995823739 | 0.6986949522 | 0.0008874218 | 0.0008874218 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 18 | 18 | 0.6993513415 | 0.6994572450 | -0.0001059035 | -0.0001059035 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 21 | 20 | 0.6965144847 | 0.6954675723 | 0.0010469125 | 0.0010469125 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 21 | 20 | 0.7018812265 | 0.7023046621 | -0.0004234356 | -0.0004234356 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 18 | 18 | 0.4970472474 | 0.4951077483 | 0.0019394991 | -0.0019394991 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 18 | 18 | 0.4351645517 | 0.4340549156 | 0.0011096361 | -0.0011096361 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 21 | 20 | 0.4992309513 | 0.4991226657 | 0.0001082857 | -0.0001082857 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 21 | 20 | 0.4381463784 | 0.4358246983 | 0.0023216801 | -0.0023216801 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 18 | 15 | 0.6995823739 | 0.6963570688 | 0.0032253051 | 0.0032253051 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 18 | 15 | 0.6993513415 | 0.6977504274 | 0.0016009141 | 0.0016009141 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 21 | 23 | 0.6965144847 | 0.6967754137 | -0.0002609289 | -0.0002609289 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 21 | 23 | 0.7018812265 | 0.7016967875 | 0.0001844390 | 0.0001844390 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 18 | 15 | 0.4970472474 | 0.4975409036 | -0.0004936562 | 0.0004936562 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 18 | 15 | 0.4351645517 | 0.4344424147 | 0.0007221370 | -0.0007221370 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 21 | 23 | 0.4992309513 | 0.4987262780 | 0.0005046733 | -0.0005046733 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 21 | 23 | 0.4381463784 | 0.4370737985 | 0.0010725799 | -0.0010725799 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 14 | 14 | 0.6992107672 | 0.6965231966 | 0.0026875705 | 0.0026875705 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 14 | 14 | 0.7083681326 | 0.7015603561 | 0.0068077764 | 0.0068077764 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 14 | 17 | 0.6925077206 | 0.6947787432 | -0.0022710225 | -0.0022710225 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 14 | 17 | 0.6987411782 | 0.6989817339 | -0.0002405557 | -0.0002405557 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 14 | 14 | 0.4983556735 | 0.4978385634 | 0.0005171101 | -0.0005171101 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 14 | 14 | 0.4338575942 | 0.4351768713 | -0.0013192771 | 0.0013192771 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 14 | 17 | 0.5021990939 | 0.4982742284 | 0.0039248656 | -0.0039248656 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 14 | 17 | 0.4378580701 | 0.4359009384 | 0.0019571316 | -0.0019571316 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6965231966 | 0.0043062225 | 0.0043062225 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.7015603561 | 0.0035938512 | 0.0035938512 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6975137357 | 0.6947787432 | 0.0027349926 | 0.0027349926 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7039519704 | 0.6989817339 | 0.0049702365 | 0.0049702365 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4978385634 | -0.0029124445 | 0.0029124445 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4351768713 | -0.0015784618 | 0.0015784618 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.4965942486 | 0.4982742284 | -0.0016799798 | 0.0016799798 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.4339335949 | 0.4359009384 | -0.0019673436 | 0.0019673436 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6965231966 | 0.0043062225 | 0.0043062225 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.7015603561 | 0.0035938512 | 0.0035938512 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6975137357 | 0.6947787432 | 0.0027349926 | 0.0027349926 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7039519704 | 0.6989817339 | 0.0049702365 | 0.0049702365 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4978385634 | -0.0029124445 | 0.0029124445 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4351768713 | -0.0015784618 | 0.0015784618 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.4965942486 | 0.4982742284 | -0.0016799798 | 0.0016799798 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.4339335949 | 0.4359009384 | -0.0019673436 | 0.0019673436 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6992107672 | 0.0016186519 | 0.0016186519 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.7083681326 | -0.0032139252 | -0.0032139252 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6975137357 | 0.6925077206 | 0.0050060151 | 0.0050060151 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7039519704 | 0.6987411782 | 0.0052107922 | 0.0052107922 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4983556735 | -0.0034295546 | 0.0034295546 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4338575942 | -0.0002591846 | 0.0002591846 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.4965942486 | 0.5021990939 | -0.0056048453 | 0.0056048453 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.4339335949 | 0.4378580701 | -0.0039244752 | 0.0039244752 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6992107672 | 0.0016186519 | 0.0016186519 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.7083681326 | -0.0032139252 | -0.0032139252 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6975137357 | 0.6925077206 | 0.0050060151 | 0.0050060151 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7039519704 | 0.6987411782 | 0.0052107922 | 0.0052107922 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4983556735 | -0.0034295546 | 0.0034295546 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4338575942 | -0.0002591846 | 0.0002591846 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.4965942486 | 0.5021990939 | -0.0056048453 | 0.0056048453 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.4339335949 | 0.4378580701 | -0.0039244752 | 0.0039244752 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 14 | 0.7008294191 | 0.6992107672 | 0.0016186519 | 0.0016186519 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 14 | 0.7051542074 | 0.7083681326 | -0.0032139252 | -0.0032139252 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 17 | 14 | 0.6975137357 | 0.6925077206 | 0.0050060151 | 0.0050060151 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 17 | 14 | 0.7039519704 | 0.6987411782 | 0.0052107922 | 0.0052107922 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 14 | 0.4949261188 | 0.4983556735 | -0.0034295546 | 0.0034295546 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 14 | 0.4335984096 | 0.4338575942 | -0.0002591846 | 0.0002591846 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 17 | 14 | 0.4965942486 | 0.5021990939 | -0.0056048453 | 0.0056048453 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 17 | 14 | 0.4339335949 | 0.4378580701 | -0.0039244752 | 0.0039244752 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 13 | 0.7008294191 | 0.6948098894 | 0.0060195297 | 0.0060195297 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 13 | 0.7051542074 | 0.7016296501 | 0.0035245573 | 0.0035245573 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 17 | 17 | 0.6975137357 | 0.6948803810 | 0.0026333547 | 0.0026333547 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 17 | 17 | 0.7039519704 | 0.7000546653 | 0.0038973051 | 0.0038973051 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 13 | 0.4949261188 | 0.4972631450 | -0.0023370262 | 0.0023370262 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 13 | 0.4335984096 | 0.4353021289 | -0.0017037193 | 0.0017037193 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 17 | 17 | 0.4965942486 | 0.4988719544 | -0.0022777058 | 0.0022777058 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 17 | 17 | 0.4339335949 | 0.4365917668 | -0.0026581720 | 0.0026581720 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8101434707 | 0.8102922025 | -0.0001487318 | -0.0001487318 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8196907361 | 0.8196841562 | 0.0000065799 | 0.0000065799 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8074667818 | 0.8071474534 | 0.0003193284 | 0.0003193284 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8192503823 | 0.8193087795 | -0.0000583972 | -0.0000583972 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5306416759 | 0.5285108931 | 0.0021307828 | -0.0021307828 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4277745957 | 0.4278354936 | -0.0000608979 | 0.0000608979 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5333393732 | 0.5358603056 | -0.0025209324 | 0.0025209324 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4251534441 | 0.4263306896 | -0.0011772455 | 0.0011772455 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8101434707 | 0.8100664765 | 0.0000769942 | 0.0000769942 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8196907361 | 0.8200018086 | -0.0003110725 | -0.0003110725 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8074667818 | 0.8075717466 | -0.0001049648 | -0.0001049648 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8192503823 | 0.8190794249 | 0.0001709574 | 0.0001709574 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5306416759 | 0.5322278595 | -0.0015861836 | 0.0015861836 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4277745957 | 0.4278758213 | -0.0001012256 | 0.0001012256 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5333393732 | 0.5335677760 | -0.0002284028 | 0.0002284028 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4251534441 | 0.4260018772 | -0.0008484331 | 0.0008484331 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8104004967 | 0.0001308538 | 0.0001308538 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8199317080 | 0.0000189767 | 0.0000189767 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8070675194 | 0.8087585857 | -0.0016910663 | -0.0016910663 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8187623255 | 0.8200147134 | -0.0012523879 | -0.0012523879 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5202555332 | 0.0124233541 | -0.0124233541 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4179060125 | 0.0105361532 | -0.0105361532 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5315283197 | 0.5213518667 | 0.0101764530 | -0.0101764530 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4233749912 | 0.4171895492 | 0.0061854419 | -0.0061854419 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8101434707 | 0.0003878797 | 0.0003878797 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8196907361 | 0.0002599486 | 0.0002599486 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8070675194 | 0.8074667818 | -0.0003992624 | -0.0003992624 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8187623255 | 0.8192503823 | -0.0004880569 | -0.0004880569 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5306416759 | 0.0020372115 | -0.0020372115 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4277745957 | 0.0006675700 | -0.0006675700 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5315283197 | 0.5333393732 | -0.0018110535 | 0.0018110535 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4233749912 | 0.4251534441 | -0.0017784529 | 0.0017784529 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8105199430 | 0.0000114075 | 0.0000114075 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8202769187 | -0.0003262340 | -0.0003262340 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8070675194 | 0.8062486489 | 0.0008188704 | 0.0008188704 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8187623255 | 0.8186891580 | 0.0000731675 | 0.0000731675 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5322730795 | 0.0004058079 | -0.0004058079 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4270682752 | 0.0013738905 | -0.0013738905 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5315283197 | 0.5343554083 | -0.0028270886 | 0.0028270886 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4233749912 | 0.4256253642 | -0.0022503730 | 0.0022503730 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8102717479 | 0.0002596026 | 0.0002596026 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8194551009 | 0.0004955839 | 0.0004955839 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8070675194 | 0.8065099461 | 0.0005575732 | 0.0005575732 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8187623255 | 0.8189297445 | -0.0001674191 | -0.0001674191 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5270596746 | 0.0056192128 | -0.0056192128 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4256851111 | 0.0027570545 | -0.0027570545 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5315283197 | 0.5320550136 | -0.0005266939 | 0.0005266939 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4233749912 | 0.4224527340 | 0.0009222572 | -0.0009222572 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8087938276 | 0.8081358474 | 0.0006579803 | 0.0006579803 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8205806980 | 0.8203286534 | 0.0002520446 | 0.0002520446 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8091497775 | 0.8111321481 | -0.0019823706 | -0.0019823706 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8215893597 | 0.8231187158 | -0.0015293562 | -0.0015293562 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5317004904 | 0.5335523493 | -0.0018518589 | 0.0018518589 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4259720109 | 0.4272292719 | -0.0012572610 | 0.0012572610 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5319182018 | 0.5317161665 | 0.0002020352 | -0.0002020352 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4256977294 | 0.4214772395 | 0.0042204899 | -0.0042204899 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8087938276 | 0.8073605537 | 0.0014332739 | 0.0014332739 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8205806980 | 0.8191194029 | 0.0014612951 | 0.0014612951 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8091497775 | 0.8095399390 | -0.0003901614 | -0.0003901614 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8215893597 | 0.8203596947 | 0.0012296649 | 0.0012296649 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5317004904 | 0.5339645981 | -0.0022641077 | 0.0022641077 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4259720109 | 0.4309715976 | -0.0049995867 | 0.0049995867 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5319182018 | 0.5300505162 | 0.0018676856 | -0.0018676856 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4256977294 | 0.4246727270 | 0.0010250023 | -0.0010250023 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8079184130 | 0.8081358474 | -0.0002174343 | -0.0002174343 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8201870478 | 0.8203286534 | -0.0001416056 | -0.0001416056 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8090222366 | 0.8111321481 | -0.0021099116 | -0.0021099116 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8213711376 | 0.8231187158 | -0.0017475782 | -0.0017475782 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5323681200 | 0.5335523493 | -0.0011842293 | 0.0011842293 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4263994012 | 0.4272292719 | -0.0008298708 | 0.0008298708 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5310687269 | 0.5317161665 | -0.0006474396 | 0.0006474396 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4254120768 | 0.4214772395 | 0.0039348373 | -0.0039348373 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8079184130 | 0.8073605537 | 0.0005578593 | 0.0005578593 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8201870478 | 0.8191194029 | 0.0010676449 | 0.0010676449 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8090222366 | 0.8095399390 | -0.0005177024 | -0.0005177024 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8213711376 | 0.8203596947 | 0.0010114429 | 0.0010114429 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5323681200 | 0.5339645981 | -0.0015964781 | 0.0015964781 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4263994012 | 0.4309715976 | -0.0045721965 | 0.0045721965 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5310687269 | 0.5300505162 | 0.0010182107 | -0.0010182107 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4254120768 | 0.4246727270 | 0.0007393498 | -0.0007393498 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8095765302 | 0.8081358474 | 0.0014406828 | 0.0014406828 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8205299005 | 0.8203286534 | 0.0002012471 | 0.0002012471 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8096029586 | 0.8111321481 | -0.0015291895 | -0.0015291895 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8229469203 | 0.8231187158 | -0.0001717955 | -0.0001717955 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5296070278 | 0.5335523493 | -0.0039453214 | 0.0039453214 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4234323339 | 0.4272292719 | -0.0037969380 | 0.0037969380 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5278170453 | 0.5317161665 | -0.0038991213 | 0.0038991213 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4211964093 | 0.4214772395 | -0.0002808303 | 0.0002808303 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8095765302 | 0.8073605537 | 0.0022159765 | 0.0022159765 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8205299005 | 0.8191194029 | 0.0014104976 | 0.0014104976 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8096029586 | 0.8095399390 | 0.0000630196 | 0.0000630196 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8229469203 | 0.8203596947 | 0.0025872256 | 0.0025872256 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5296070278 | 0.5339645981 | -0.0043575702 | 0.0043575702 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4234323339 | 0.4309715976 | -0.0075392637 | 0.0075392637 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5278170453 | 0.5300505162 | -0.0022334709 | 0.0022334709 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4211964093 | 0.4246727270 | -0.0034763178 | 0.0034763178 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8081358474 | -0.0003412731 | -0.0003412731 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8203286534 | -0.0007725278 | -0.0007725278 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8111321481 | -0.0004048111 | -0.0004048111 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8231187158 | -0.0015125545 | -0.0015125545 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5335523493 | 0.0044005920 | -0.0044005920 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4272292719 | 0.0004136723 | -0.0004136723 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5317161665 | -0.0059768113 | 0.0059768113 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4214772395 | 0.0014678020 | -0.0014678020 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8073605537 | 0.0004340205 | 0.0004340205 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8191194029 | 0.0004367227 | 0.0004367227 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8095399390 | 0.0011873980 | 0.0011873980 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8203596947 | 0.0012464666 | 0.0012464666 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5339645981 | 0.0039883432 | -0.0039883432 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4309715976 | -0.0033286534 | 0.0033286534 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5300505162 | -0.0043111609 | 0.0043111609 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4246727270 | -0.0017276855 | 0.0017276855 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8087938276 | -0.0009992534 | -0.0009992534 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8205806980 | -0.0010245724 | -0.0010245724 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8091497775 | 0.0015775595 | 0.0015775595 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8215893597 | 0.0000168016 | 0.0000168016 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5317004904 | 0.0062524509 | -0.0062524509 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4259720109 | 0.0016709332 | -0.0016709332 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5319182018 | -0.0061788465 | 0.0061788465 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4256977294 | -0.0027526879 | 0.0027526879 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8079184130 | -0.0001238388 | -0.0001238388 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8201870478 | -0.0006309222 | -0.0006309222 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8090222366 | 0.0017051004 | 0.0017051004 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8213711376 | 0.0002350237 | 0.0002350237 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5323681200 | 0.0055848213 | -0.0055848213 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4263994012 | 0.0012435430 | -0.0012435430 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5310687269 | -0.0053293717 | 0.0053293717 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4254120768 | -0.0024670353 | 0.0024670353 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8095765302 | -0.0017819559 | -0.0017819559 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8205299005 | -0.0009737750 | -0.0009737750 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8096029586 | 0.0011243784 | 0.0011243784 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8229469203 | -0.0013407590 | -0.0013407590 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5296070278 | 0.0083459135 | -0.0083459135 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4234323339 | 0.0042106102 | -0.0042106102 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5278170453 | -0.0020776900 | 0.0020776900 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4211964093 | 0.0017486322 | -0.0017486322 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.8077945742 | 0.8088393963 | -0.0010448220 | -0.0010448220 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.8195561256 | 0.8198539884 | -0.0002978629 | -0.0002978629 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 24 | 24 | 0.8107273370 | 0.8086724939 | 0.0020548431 | 0.0020548431 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 24 | 24 | 0.8216061613 | 0.8204550715 | 0.0011510898 | 0.0011510898 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.5379529413 | 0.5320071490 | 0.0059457923 | -0.0059457923 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4276429442 | 0.4307500860 | -0.0031071418 | 0.0031071418 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 24 | 24 | 0.5257393552 | 0.5309326217 | -0.0051932664 | 0.0051932664 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 24 | 24 | 0.4229450415 | 0.4252137265 | -0.0022686850 | 0.0022686850 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8019805246 | 0.8026221691 | -0.0006416445 | -0.0006416445 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8130585265 | 0.8133405466 | -0.0002820201 | -0.0002820201 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 0 | 14 | 12 | 0.8043209440 | 0.8019118278 | 0.0024091162 | 0.0024091162 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | auc | 1 | 14 | 12 | 0.8149571654 | 0.8139035346 | 0.0010536308 | 0.0010536308 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5284063972 | 0.5271352242 | 0.0012711731 | -0.0012711731 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4243430313 | 0.4243317474 | 0.0000112839 | -0.0000112839 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 0 | 14 | 12 | 0.5260792232 | 0.5294701363 | -0.0033909131 | 0.0033909131 |
| L2_BLOCK_UNIFORM-L0_OWN | internal_mechanism | 3102 | logloss | 1 | 14 | 12 | 0.4227203790 | 0.4247463071 | -0.0020259281 | 0.0020259281 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8019805246 | 0.8019523535 | 0.0000281711 | 0.0000281711 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8130585265 | 0.8131010022 | -0.0000424757 | -0.0000424757 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 0 | 14 | 12 | 0.8043209440 | 0.8020382836 | 0.0022826603 | 0.0022826603 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | auc | 1 | 14 | 12 | 0.8149571654 | 0.8141094917 | 0.0008476737 | 0.0008476737 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5284063972 | 0.5284037380 | 0.0000026592 | -0.0000026592 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4243430313 | 0.4242427400 | 0.0001002914 | -0.0001002914 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 0 | 14 | 12 | 0.5260792232 | 0.5293244705 | -0.0032452473 | 0.0032452473 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | internal_mechanism | 3102 | logloss | 1 | 14 | 12 | 0.4227203790 | 0.4235829661 | -0.0008625871 | 0.0008625871 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8020107835 | 0.8024338934 | -0.0004231099 | -0.0004231099 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8130585180 | 0.8140351485 | -0.0009766305 | -0.0009766305 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 0 | 16 | 16 | 0.8065536562 | 0.8069962175 | -0.0004425613 | -0.0004425613 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | auc | 1 | 16 | 16 | 0.8167060279 | 0.8166363538 | 0.0000696741 | 0.0000696741 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5283598453 | 0.5265741672 | 0.0017856780 | -0.0017856780 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4242839314 | 0.4222781127 | 0.0020058187 | -0.0020058187 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 0 | 16 | 16 | 0.5246424261 | 0.5225102744 | 0.0021321517 | -0.0021321517 |
| L4_BLOCK_META-L0_FEATURE | internal_mechanism | 3102 | logloss | 1 | 16 | 16 | 0.4214136077 | 0.4197252682 | 0.0016883395 | -0.0016883395 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8020107835 | 0.8019805246 | 0.0000302589 | 0.0000302589 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8130585180 | 0.8130585265 | -0.0000000085 | -0.0000000085 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 0 | 16 | 14 | 0.8065536562 | 0.8043209440 | 0.0022327123 | 0.0022327123 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | auc | 1 | 16 | 14 | 0.8167060279 | 0.8149571654 | 0.0017488625 | 0.0017488625 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5283598453 | 0.5284063972 | -0.0000465520 | 0.0000465520 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4242839314 | 0.4243430313 | -0.0000590999 | 0.0000590999 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 0 | 16 | 14 | 0.5246424261 | 0.5260792232 | -0.0014367972 | 0.0014367972 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | internal_mechanism | 3102 | logloss | 1 | 16 | 14 | 0.4214136077 | 0.4227203790 | -0.0013067713 | 0.0013067713 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8020107835 | 0.8020187280 | -0.0000079446 | -0.0000079446 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8130585180 | 0.8130761624 | -0.0000176443 | -0.0000176443 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 0 | 16 | 14 | 0.8065536562 | 0.8042813333 | 0.0022723230 | 0.0022723230 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | auc | 1 | 16 | 14 | 0.8167060279 | 0.8149387803 | 0.0017672476 | 0.0017672476 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5283598453 | 0.5283499415 | 0.0000099038 | -0.0000099038 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4242839314 | 0.4242592587 | 0.0000246727 | -0.0000246727 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 0 | 16 | 14 | 0.5246424261 | 0.5262055205 | -0.0015630945 | 0.0015630945 |
| L4_BLOCK_META-L3_SCALAR_META | internal_mechanism | 3102 | logloss | 1 | 16 | 14 | 0.4214136077 | 0.4227511583 | -0.0013375505 | 0.0013375505 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 0 | 12 | 14 | 0.8020107835 | 0.8035006101 | -0.0014898266 | -0.0014898266 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | auc | 1 | 12 | 14 | 0.8130585180 | 0.8145022060 | -0.0014436880 | -0.0014436880 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 0 | 16 | 16 | 0.8065536562 | 0.8069515364 | -0.0003978802 | -0.0003978802 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | auc | 1 | 16 | 16 | 0.8167060279 | 0.8166636352 | 0.0000423927 | 0.0000423927 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 0 | 12 | 14 | 0.5283598453 | 0.5268012577 | 0.0015585876 | -0.0015585876 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3101 | logloss | 1 | 12 | 14 | 0.4242839314 | 0.4237141801 | 0.0005697514 | -0.0005697514 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 0 | 16 | 16 | 0.5246424261 | 0.5235138032 | 0.0011286228 | -0.0011286228 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | internal_mechanism | 3102 | logloss | 1 | 16 | 16 | 0.4214136077 | 0.4206122677 | 0.0008013400 | -0.0008013400 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8026607152 | 0.8024861151 | 0.0001746001 | 0.0001746001 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8151120941 | 0.8151393590 | -0.0000272649 | -0.0000272649 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027299753 | 0.8028304177 | -0.0001004424 | -0.0001004424 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8144064004 | 0.8146512301 | -0.0002448297 | -0.0002448297 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.5280376946 | 0.5281331818 | -0.0000954872 | 0.0000954872 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4224298701 | 0.4222057379 | 0.0002241322 | -0.0002241322 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5263338690 | 0.5268671266 | -0.0005332575 | 0.0005332575 |
| R2_FREEZE-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4226466461 | 0.4231095266 | -0.0004628805 | 0.0004628805 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8026607152 | 0.8018040832 | 0.0008566320 | 0.0008566320 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8151120941 | 0.8138012379 | 0.0013108562 | 0.0013108562 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027299753 | 0.8025093424 | 0.0002206329 | 0.0002206329 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8144064004 | 0.8145838877 | -0.0001774873 | -0.0001774873 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.5280376946 | 0.5296639875 | -0.0016262928 | 0.0016262928 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4224298701 | 0.4238468096 | -0.0014169395 | 0.0014169395 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5263338690 | 0.5271378780 | -0.0008040089 | 0.0008040089 |
| R2_FREEZE-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4226466461 | 0.4224652161 | 0.0001814300 | -0.0001814300 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8026951860 | 0.8024861151 | 0.0002090709 | 0.0002090709 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8151280912 | 0.8151393590 | -0.0000112678 | -0.0000112678 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027815278 | 0.8028304177 | -0.0000488899 | -0.0000488899 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8146811747 | 0.8146512301 | 0.0000299446 | 0.0000299446 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.5277905544 | 0.5281331818 | -0.0003426275 | 0.0003426275 |
| R3_CUM-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4223908425 | 0.4222057379 | 0.0001851046 | -0.0001851046 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268994225 | 0.5268671266 | 0.0000322959 | -0.0000322959 |
| R3_CUM-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4225269620 | 0.4231095266 | -0.0005825646 | 0.0005825646 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8026951860 | 0.8018040832 | 0.0008911028 | 0.0008911028 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8151280912 | 0.8138012379 | 0.0013268533 | 0.0013268533 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027815278 | 0.8025093424 | 0.0002721854 | 0.0002721854 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8146811747 | 0.8145838877 | 0.0000972870 | 0.0000972870 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.5277905544 | 0.5296639875 | -0.0018734331 | 0.0018734331 |
| R3_CUM-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4223908425 | 0.4238468096 | -0.0014559672 | 0.0014559672 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268994225 | 0.5271378780 | -0.0002384554 | 0.0002384554 |
| R3_CUM-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4225269620 | 0.4224652161 | 0.0000617459 | -0.0000617459 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 0 | 13 | 13 | 0.8027015495 | 0.8024861151 | 0.0002154344 | 0.0002154344 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | auc | 1 | 13 | 13 | 0.8151206673 | 0.8151393590 | -0.0000186918 | -0.0000186918 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027815278 | 0.8028304177 | -0.0000488899 | -0.0000488899 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8146811747 | 0.8146512301 | 0.0000299446 | 0.0000299446 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 13 | 13 | 0.5277593367 | 0.5281331818 | -0.0003738452 | 0.0003738452 |
| R4_EMA-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 13 | 13 | 0.4223730595 | 0.4222057379 | 0.0001673217 | -0.0001673217 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268994225 | 0.5268671266 | 0.0000322959 | -0.0000322959 |
| R4_EMA-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4225269620 | 0.4231095266 | -0.0005825646 | 0.0005825646 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 13 | 12 | 0.8027015495 | 0.8018040832 | 0.0008974663 | 0.0008974663 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 13 | 12 | 0.8151206673 | 0.8138012379 | 0.0013194294 | 0.0013194294 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8027815278 | 0.8025093424 | 0.0002721854 | 0.0002721854 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8146811747 | 0.8145838877 | 0.0000972870 | 0.0000972870 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 13 | 12 | 0.5277593367 | 0.5296639875 | -0.0019046508 | 0.0019046508 |
| R4_EMA-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 13 | 12 | 0.4223730595 | 0.4238468096 | -0.0014737501 | 0.0014737501 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268994225 | 0.5271378780 | -0.0002384554 | 0.0002384554 |
| R4_EMA-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4225269620 | 0.4224652161 | 0.0000617459 | -0.0000617459 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8019029122 | 0.8024861151 | -0.0005832029 | -0.0005832029 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8138258535 | 0.8151393590 | -0.0013135055 | -0.0013135055 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8028304177 | -0.0004313830 | -0.0004313830 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8146512301 | -0.0004144513 | -0.0004144513 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.5297834136 | 0.5281331818 | 0.0016502318 | -0.0016502318 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4237687798 | 0.4222057379 | 0.0015630419 | -0.0015630419 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5268671266 | -0.0000167439 | 0.0000167439 |
| R6_PB_HB-R0_BASE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4231095266 | -0.0002581339 | 0.0002581339 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 0 | 12 | 12 | 0.8019029122 | 0.8018040832 | 0.0000988290 | 0.0000988290 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | auc | 1 | 12 | 12 | 0.8138258535 | 0.8138012379 | 0.0000246156 | 0.0000246156 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8025093424 | -0.0001103077 | -0.0001103077 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8145838877 | -0.0003471089 | -0.0003471089 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 0 | 12 | 12 | 0.5297834136 | 0.5296639875 | 0.0001194262 | -0.0001194262 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3101 | logloss | 1 | 12 | 12 | 0.4237687798 | 0.4238468096 | -0.0000780298 | 0.0000780298 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5271378780 | -0.0002874953 | 0.0002874953 |
| R6_PB_HB-R1_STATIC | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4224652161 | 0.0003861766 | -0.0003861766 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8019029122 | 0.8026607152 | -0.0007578029 | -0.0007578029 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8138258535 | 0.8151120941 | -0.0012862406 | -0.0012862406 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8027299753 | -0.0003309407 | -0.0003309407 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8144064004 | -0.0001696216 | -0.0001696216 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.5297834136 | 0.5280376946 | 0.0017457190 | -0.0017457190 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4237687798 | 0.4224298701 | 0.0013389097 | -0.0013389097 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5263338690 | 0.0005165136 | -0.0005165136 |
| R6_PB_HB-R2_FREEZE | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4226466461 | 0.0002047466 | -0.0002047466 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8019029122 | 0.8026951860 | -0.0007922737 | -0.0007922737 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8138258535 | 0.8151280912 | -0.0013022377 | -0.0013022377 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8027815278 | -0.0003824931 | -0.0003824931 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8146811747 | -0.0004443959 | -0.0004443959 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.5297834136 | 0.5277905544 | 0.0019928592 | -0.0019928592 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4237687798 | 0.4223908425 | 0.0013779374 | -0.0013779374 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5268994225 | -0.0000490399 | 0.0000490399 |
| R6_PB_HB-R3_CUM | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4225269620 | 0.0003244307 | -0.0003244307 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8019029122 | 0.8027015495 | -0.0007986373 | -0.0007986373 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8138258535 | 0.8151206673 | -0.0012948137 | -0.0012948137 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8027815278 | -0.0003824931 | -0.0003824931 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8146811747 | -0.0004443959 | -0.0004443959 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.5297834136 | 0.5277593367 | 0.0020240770 | -0.0020240770 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4237687798 | 0.4223730595 | 0.0013957203 | -0.0013957203 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5268994225 | -0.0000490399 | 0.0000490399 |
| R6_PB_HB-R4_EMA | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4225269620 | 0.0003244307 | -0.0003244307 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 0 | 12 | 13 | 0.8019029122 | 0.8027767277 | -0.0008738154 | -0.0008738154 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | auc | 1 | 12 | 13 | 0.8138258535 | 0.8151142126 | -0.0012883590 | -0.0012883590 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 0 | 13 | 13 | 0.8023990346 | 0.8026571692 | -0.0002581346 | -0.0002581346 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | auc | 1 | 13 | 13 | 0.8142367789 | 0.8144563628 | -0.0002195839 | -0.0002195839 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 0 | 12 | 13 | 0.5297834136 | 0.5274328826 | 0.0023505310 | -0.0023505310 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3101 | logloss | 1 | 12 | 13 | 0.4237687798 | 0.4225672325 | 0.0012015473 | -0.0012015473 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 0 | 13 | 13 | 0.5268503826 | 0.5265533520 | 0.0002970306 | -0.0002970306 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 3102 | logloss | 1 | 13 | 13 | 0.4228513927 | 0.4228182561 | 0.0000331366 | -0.0000331366 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6837451373 | 0.6832890218 | 0.0004561155 | 0.0004561155 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.6896104763 | 0.6790561258 | 0.0105543505 | 0.0105543505 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6846141749 | 0.6841010189 | 0.0005131559 | 0.0005131559 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.6866979999 | 0.6968670922 | -0.0101690923 | -0.0101690923 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6826371991 | 0.6846229230 | -0.0019857239 | -0.0019857239 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.6852569152 | 0.6881647489 | -0.0029078337 | -0.0029078337 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6850957436 | 0.6863732645 | -0.0012775209 | -0.0012775209 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.6877136489 | 0.6872138678 | 0.0004997811 | 0.0004997811 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.5349049118 | 0.5456184072 | -0.0107134954 | 0.0107134954 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.4704309393 | 0.4840775860 | -0.0136466467 | 0.0136466467 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.5375929684 | 0.5358216391 | 0.0017713293 | -0.0017713293 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.4802252702 | 0.4800667698 | 0.0001585004 | -0.0001585004 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.5221287269 | 0.5332113309 | -0.0110826041 | 0.0110826041 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.4540403019 | 0.4757596675 | -0.0217193656 | 0.0217193656 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.5323420002 | 0.5320422224 | 0.0002997778 | -0.0002997778 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.4661777359 | 0.4701864782 | -0.0040087423 | 0.0040087423 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6837451373 | 0.6804761103 | 0.0032690269 | 0.0032690269 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.6896104763 | 0.6800442049 | 0.0095662714 | 0.0095662714 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6846141749 | 0.6835901722 | 0.0010240026 | 0.0010240026 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.6866979999 | 0.6893434167 | -0.0026454168 | -0.0026454168 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6826371991 | 0.6911391650 | -0.0085019659 | -0.0085019659 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.6852569152 | 0.6929542136 | -0.0076972984 | -0.0076972984 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6850957436 | 0.6812096427 | 0.0038861010 | 0.0038861010 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.6877136489 | 0.6863074806 | 0.0014061683 | 0.0014061683 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.5349049118 | 0.5466681635 | -0.0117632517 | 0.0117632517 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.4704309393 | 0.4854250528 | -0.0149941135 | 0.0149941135 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.5375929684 | 0.5469421079 | -0.0093491395 | 0.0093491395 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.4802252702 | 0.4787328143 | 0.0014924559 | -0.0014924559 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.5221287269 | 0.5281277489 | -0.0059990220 | 0.0059990220 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.4540403019 | 0.4662755635 | -0.0122352616 | 0.0122352616 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.5323420002 | 0.5379912591 | -0.0056492589 | 0.0056492589 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.4661777359 | 0.4699788059 | -0.0038010700 | 0.0038010700 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 0 | 50 | 50 | 0.6837451373 | 0.6863160118 | -0.0025708746 | -0.0025708746 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 1 | 50 | 50 | 0.6896104763 | 0.6886415039 | 0.0009689724 | 0.0009689724 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 0 | 50 | 50 | 0.6846141749 | 0.6785805385 | 0.0060336363 | 0.0060336363 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 1 | 50 | 50 | 0.6866979999 | 0.6840954353 | 0.0026025646 | 0.0026025646 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 0 | 50 | 50 | 0.6826371991 | 0.6736034973 | 0.0090337018 | 0.0090337018 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 1 | 50 | 50 | 0.6852569152 | 0.6831366044 | 0.0021203108 | 0.0021203108 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 0 | 50 | 50 | 0.6850957436 | 0.6812534936 | 0.0038422500 | 0.0038422500 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 1 | 50 | 50 | 0.6877136489 | 0.6908430600 | -0.0031294111 | -0.0031294111 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 0 | 50 | 50 | 0.5349049118 | 0.5451447317 | -0.0102398199 | 0.0102398199 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 1 | 50 | 50 | 0.4704309393 | 0.4809439713 | -0.0105130320 | 0.0105130320 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 0 | 50 | 50 | 0.5375929684 | 0.5652522886 | -0.0276593202 | 0.0276593202 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 1 | 50 | 50 | 0.4802252702 | 0.4967850109 | -0.0165597407 | 0.0165597407 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 0 | 50 | 50 | 0.5221287269 | 0.5649919508 | -0.0428632240 | 0.0428632240 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 1 | 50 | 50 | 0.4540403019 | 0.5140771895 | -0.0600368877 | 0.0600368877 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 0 | 50 | 50 | 0.5323420002 | 0.5560008603 | -0.0236588601 | 0.0236588601 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 1 | 50 | 50 | 0.4661777359 | 0.4918092914 | -0.0256315555 | 0.0256315555 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 0 | 50 | 50 | 0.6837451373 | 0.6842623456 | -0.0005172083 | -0.0005172083 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 1 | 50 | 50 | 0.6896104763 | 0.6929652343 | -0.0033547580 | -0.0033547580 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 0 | 50 | 50 | 0.6846141749 | 0.6827220495 | 0.0018921253 | 0.0018921253 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 1 | 50 | 50 | 0.6866979999 | 0.6833271827 | 0.0033708172 | 0.0033708172 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 0 | 50 | 50 | 0.6826371991 | 0.6905481433 | -0.0079109442 | -0.0079109442 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 1 | 50 | 50 | 0.6852569152 | 0.6918731401 | -0.0066162249 | -0.0066162249 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 0 | 50 | 50 | 0.6850957436 | 0.6861030600 | -0.0010073163 | -0.0010073163 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 1 | 50 | 50 | 0.6877136489 | 0.6942945088 | -0.0065808599 | -0.0065808599 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 0 | 50 | 50 | 0.5349049118 | 0.5521764495 | -0.0172715377 | 0.0172715377 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 1 | 50 | 50 | 0.4704309393 | 0.4872361962 | -0.0168052569 | 0.0168052569 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 0 | 50 | 50 | 0.5375929684 | 0.5391530369 | -0.0015600685 | 0.0015600685 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 1 | 50 | 50 | 0.4802252702 | 0.4789535397 | 0.0012717305 | -0.0012717305 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 0 | 50 | 50 | 0.5221287269 | 0.5280064104 | -0.0058776835 | 0.0058776835 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 1 | 50 | 50 | 0.4540403019 | 0.4655676371 | -0.0115273352 | 0.0115273352 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 0 | 50 | 50 | 0.5323420002 | 0.5405264220 | -0.0081844219 | 0.0081844219 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 1 | 50 | 50 | 0.4661777359 | 0.4783414078 | -0.0121636720 | 0.0121636720 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 0 | 13 | 15 | 0.6901869919 | 0.6927693136 | -0.0025823218 | -0.0025823218 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | auc | 1 | 13 | 15 | 0.6955629399 | 0.6981875821 | -0.0026246422 | -0.0026246422 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 0 | 18 | 15 | 0.6984129584 | 0.6946546066 | 0.0037583517 | 0.0037583517 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | auc | 1 | 18 | 15 | 0.7015379313 | 0.6986081429 | 0.0029297883 | 0.0029297883 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 0 | 12 | 14 | 0.6862482226 | 0.6955602407 | -0.0093120181 | -0.0093120181 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | auc | 1 | 12 | 14 | 0.6929646514 | 0.7012129950 | -0.0082483436 | -0.0082483436 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6943961028 | 0.6929294832 | 0.0014666196 | 0.0014666196 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.6999046831 | 0.6962108571 | 0.0036938260 | 0.0036938260 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 0 | 13 | 15 | 0.5015597181 | 0.4979149865 | 0.0036447316 | -0.0036447316 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4101 | logloss | 1 | 13 | 15 | 0.4374945946 | 0.4358911041 | 0.0016034906 | -0.0016034906 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 0 | 18 | 15 | 0.5010039497 | 0.5001554989 | 0.0008484509 | -0.0008484509 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4102 | logloss | 1 | 18 | 15 | 0.4429201588 | 0.4377479458 | 0.0051722130 | -0.0051722130 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 0 | 12 | 14 | 0.4998423804 | 0.4963498262 | 0.0034925543 | -0.0034925543 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4103 | logloss | 1 | 12 | 14 | 0.4359620322 | 0.4349381675 | 0.0010238647 | -0.0010238647 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.4980081086 | 0.4980636439 | -0.0000555353 | 0.0000555353 |
| R6_PB_HB-R0_BASE | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.4352717226 | 0.4358615752 | -0.0005898527 | 0.0005898527 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 0 | 13 | 13 | 0.6901869919 | 0.6922201656 | -0.0020331737 | -0.0020331737 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | auc | 1 | 13 | 13 | 0.6955629399 | 0.6996237381 | -0.0040607982 | -0.0040607982 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 0 | 18 | 17 | 0.6984129584 | 0.6942114185 | 0.0042015399 | 0.0042015399 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | auc | 1 | 18 | 17 | 0.7015379313 | 0.6957616018 | 0.0057763295 | 0.0057763295 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 0 | 12 | 14 | 0.6862482226 | 0.6978967162 | -0.0116484936 | -0.0116484936 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | auc | 1 | 12 | 14 | 0.6929646514 | 0.7083221745 | -0.0153575231 | -0.0153575231 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6943961028 | 0.6939088921 | 0.0004872107 | 0.0004872107 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.6999046831 | 0.6982446143 | 0.0016600688 | 0.0016600688 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 0 | 13 | 13 | 0.5015597181 | 0.4976645838 | 0.0038951343 | -0.0038951343 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4101 | logloss | 1 | 13 | 13 | 0.4374945946 | 0.4336882863 | 0.0038063083 | -0.0038063083 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 0 | 18 | 17 | 0.5010039497 | 0.4980664017 | 0.0029375481 | -0.0029375481 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4102 | logloss | 1 | 18 | 17 | 0.4429201588 | 0.4359023837 | 0.0070177751 | -0.0070177751 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 0 | 12 | 14 | 0.4998423804 | 0.4961996977 | 0.0036426828 | -0.0036426828 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4103 | logloss | 1 | 12 | 14 | 0.4359620322 | 0.4322012748 | 0.0037607574 | -0.0037607574 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.4980081086 | 0.4982320410 | -0.0002239324 | 0.0002239324 |
| R6_PB_HB-R5_PB_H1 | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.4352717226 | 0.4356373424 | -0.0003656199 | 0.0003656199 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 0 | 13 | 12 | 0.6901869919 | 0.6931022585 | -0.0029152667 | -0.0029152667 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | auc | 1 | 13 | 12 | 0.6955629399 | 0.7010296325 | -0.0054666927 | -0.0054666927 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 0 | 18 | 12 | 0.6984129584 | 0.6941177058 | 0.0042952526 | 0.0042952526 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | auc | 1 | 18 | 12 | 0.7015379313 | 0.6993144618 | 0.0022234695 | 0.0022234695 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 0 | 12 | 12 | 0.6862482226 | 0.6868055913 | -0.0005573687 | -0.0005573687 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | auc | 1 | 12 | 12 | 0.6929646514 | 0.6936727175 | -0.0007080661 | -0.0007080661 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 0 | 21 | 21 | 0.6943961028 | 0.6935386964 | 0.0008574064 | 0.0008574064 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | auc | 1 | 21 | 21 | 0.6999046831 | 0.6969877203 | 0.0029169628 | 0.0029169628 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 0 | 13 | 12 | 0.5015597181 | 0.5004346772 | 0.0011250409 | -0.0011250409 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4101 | logloss | 1 | 13 | 12 | 0.4374945946 | 0.4348236583 | 0.0026709364 | -0.0026709364 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 0 | 18 | 12 | 0.5010039497 | 0.5016035978 | -0.0005996480 | 0.0005996480 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4102 | logloss | 1 | 18 | 12 | 0.4429201588 | 0.4379973626 | 0.0049227962 | -0.0049227962 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 0 | 12 | 12 | 0.4998423804 | 0.4990219082 | 0.0008204722 | -0.0008204722 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4103 | logloss | 1 | 12 | 12 | 0.4359620322 | 0.4354210429 | 0.0005409894 | -0.0005409894 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 0 | 21 | 21 | 0.4980081086 | 0.4978259515 | 0.0001821571 | -0.0001821571 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 4104 | logloss | 1 | 21 | 21 | 0.4352717226 | 0.4363093819 | -0.0010376593 | 0.0010376593 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 0 | 13 | 16 | 0.6901869919 | 0.6982667703 | -0.0080797785 | -0.0080797785 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | auc | 1 | 13 | 16 | 0.6955629399 | 0.7044589619 | -0.0088960221 | -0.0088960221 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 0 | 18 | 12 | 0.6984129584 | 0.6889577461 | 0.0094552123 | 0.0094552123 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | auc | 1 | 18 | 12 | 0.7015379313 | 0.6932094100 | 0.0083285213 | 0.0083285213 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 0 | 12 | 15 | 0.6862482226 | 0.6981468155 | -0.0118985929 | -0.0118985929 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | auc | 1 | 12 | 15 | 0.6929646514 | 0.7060462670 | -0.0130816156 | -0.0130816156 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 0 | 21 | 15 | 0.6943961028 | 0.6919203893 | 0.0024757134 | 0.0024757134 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | auc | 1 | 21 | 15 | 0.6999046831 | 0.7009698045 | -0.0010651215 | -0.0010651215 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 0 | 13 | 16 | 0.5015597181 | 0.4971167940 | 0.0044429241 | -0.0044429241 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4101 | logloss | 1 | 13 | 16 | 0.4374945946 | 0.4354865033 | 0.0020080913 | -0.0020080913 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 0 | 18 | 12 | 0.5010039497 | 0.5006526015 | 0.0003513482 | -0.0003513482 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4102 | logloss | 1 | 18 | 12 | 0.4429201588 | 0.4366068676 | 0.0063132912 | -0.0063132912 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 0 | 12 | 15 | 0.4998423804 | 0.4997766196 | 0.0000657609 | -0.0000657609 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4103 | logloss | 1 | 12 | 15 | 0.4359620322 | 0.4381465903 | -0.0021845581 | 0.0021845581 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 0 | 21 | 15 | 0.4980081086 | 0.5043599657 | -0.0063518571 | 0.0063518571 |
| R6_PB_HB-STGU_DOMAIN_ADAM | external_adaptation_quality_reference | 4104 | logloss | 1 | 21 | 15 | 0.4352717226 | 0.4414844033 | -0.0062126808 | 0.0062126808 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.6957336367 | 0.7029782499 | -0.0072446133 | -0.0072446133 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.7025555227 | 0.7105187983 | -0.0079632755 | -0.0079632755 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4998443412 | 0.4976293317 | 0.0022150095 | -0.0022150095 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4363851943 | 0.4346852656 | 0.0016999286 | -0.0016999286 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 15 | 12 | 0.6921697115 | 0.6926610439 | -0.0004913323 | -0.0004913323 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 15 | 12 | 0.7001181494 | 0.7018128901 | -0.0016947407 | -0.0016947407 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 15 | 12 | 0.4976868460 | 0.4990598370 | -0.0013729910 | 0.0013729910 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 15 | 12 | 0.4348342946 | 0.4349373700 | -0.0001030754 | 0.0001030754 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.8108807747 | 0.8092105198 | 0.0016702549 | 0.0016702549 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.8206832055 | 0.8203090765 | 0.0003741290 | 0.0003741290 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.5316217019 | 0.5339859481 | -0.0023642462 | 0.0023642462 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4280687891 | 0.4277025040 | 0.0003662851 | -0.0003662851 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 0 | 14 | 14 | 0.8038614430 | 0.8061705956 | -0.0023091526 | -0.0023091526 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | auc | 1 | 14 | 14 | 0.8164862708 | 0.8183292244 | -0.0018429537 | -0.0018429537 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 0 | 14 | 14 | 0.5271508913 | 0.5271646053 | -0.0000137139 | 0.0000137139 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | logloss | 1 | 14 | 14 | 0.4227659383 | 0.4216241385 | 0.0011417998 | -0.0011417998 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.6957336367 | 0.6978371116 | -0.0021034749 | -0.0021034749 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.7025555227 | 0.7035455536 | -0.0009900309 | -0.0009900309 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4998443412 | 0.4995804220 | 0.0002639191 | -0.0002639191 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4363851943 | 0.4354561359 | 0.0009290584 | -0.0009290584 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.7029782499 | 0.7032341756 | -0.0002559257 | -0.0002559257 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.7105187983 | 0.7124975447 | -0.0019787465 | -0.0019787465 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4976293317 | 0.4964513992 | 0.0011779325 | -0.0011779325 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4346852656 | 0.4334063164 | 0.0012789492 | -0.0012789492 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 15 | 15 | 0.6921697115 | 0.6928948270 | -0.0007251155 | -0.0007251155 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 15 | 15 | 0.7001181494 | 0.7019739318 | -0.0018557824 | -0.0018557824 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 15 | 15 | 0.4976868460 | 0.4979331834 | -0.0002463374 | 0.0002463374 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 15 | 15 | 0.4348342946 | 0.4339922272 | 0.0008420674 | -0.0008420674 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 12 | 18 | 0.6926610439 | 0.7021663809 | -0.0095053370 | -0.0095053370 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 12 | 18 | 0.7018128901 | 0.7109126935 | -0.0090998034 | -0.0090998034 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 12 | 18 | 0.4990598370 | 0.4940902264 | 0.0049696106 | -0.0049696106 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 12 | 18 | 0.4349373700 | 0.4312064743 | 0.0037308957 | -0.0037308957 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.8108807747 | 0.8109370136 | -0.0000562389 | -0.0000562389 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.8206832055 | 0.8210284153 | -0.0003452097 | -0.0003452097 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.5316217019 | 0.5297929495 | 0.0018287524 | -0.0018287524 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4280687891 | 0.4270441426 | 0.0010246465 | -0.0010246465 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 24 | 24 | 0.8092105198 | 0.8104155866 | -0.0012050668 | -0.0012050668 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 24 | 24 | 0.8203090765 | 0.8204201145 | -0.0001110380 | -0.0001110380 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 24 | 24 | 0.5339859481 | 0.5399559817 | -0.0059700336 | 0.0059700336 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4277025040 | 0.4297831450 | -0.0020806410 | 0.0020806410 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 0 | 14 | 17 | 0.8038614430 | 0.8078055825 | -0.0039441395 | -0.0039441395 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | auc | 1 | 14 | 17 | 0.8164862708 | 0.8182813125 | -0.0017950418 | -0.0017950418 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 14 | 17 | 0.5271508913 | 0.5242663282 | 0.0028845632 | -0.0028845632 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 14 | 17 | 0.4227659383 | 0.4224477520 | 0.0003181862 | -0.0003181862 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 0 | 14 | 14 | 0.8061705956 | 0.8059953897 | 0.0001752059 | 0.0001752059 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | auc | 1 | 14 | 14 | 0.8183292244 | 0.8183634161 | -0.0000341916 | -0.0000341916 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 0 | 14 | 14 | 0.5271646053 | 0.5268118302 | 0.0003527751 | -0.0003527751 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | logloss | 1 | 14 | 14 | 0.4216241385 | 0.4214852618 | 0.0001388767 | -0.0001388767 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.6978371116 | 0.7032341756 | -0.0053970640 | -0.0053970640 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.7035455536 | 0.7124975447 | -0.0089519911 | -0.0089519911 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.4995804220 | 0.4964513992 | 0.0031290229 | -0.0031290229 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4354561359 | 0.4334063164 | 0.0020498195 | -0.0020498195 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 15 | 18 | 0.6928948270 | 0.7021663809 | -0.0092715539 | -0.0092715539 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 15 | 18 | 0.7019739318 | 0.7109126935 | -0.0089387617 | -0.0089387617 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 15 | 18 | 0.4979331834 | 0.4940902264 | 0.0038429570 | -0.0038429570 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 15 | 18 | 0.4339922272 | 0.4312064743 | 0.0027857529 | -0.0027857529 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 24 | 24 | 0.8109370136 | 0.8104155866 | 0.0005214270 | 0.0005214270 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 24 | 24 | 0.8210284153 | 0.8204201145 | 0.0006083007 | 0.0006083007 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 24 | 24 | 0.5297929495 | 0.5399559817 | -0.0101630322 | 0.0101630322 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 24 | 24 | 0.4270441426 | 0.4297831450 | -0.0027390025 | 0.0027390025 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 0 | 17 | 14 | 0.8078055825 | 0.8059953897 | 0.0018101928 | 0.0018101928 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | auc | 1 | 17 | 14 | 0.8182813125 | 0.8183634161 | -0.0000821035 | -0.0000821035 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 0 | 17 | 14 | 0.5242663282 | 0.5268118302 | -0.0025455020 | 0.0025455020 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | logloss | 1 | 17 | 14 | 0.4224477520 | 0.4214852618 | 0.0009624903 | -0.0009624903 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6983092106 | -0.0001978388 | -0.0001978388 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7004420422 | 0.0005097251 | 0.0005097251 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4990297767 | 0.0035857293 | -0.0035857293 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4377645669 | 0.0016321469 | -0.0016321469 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6981973789 | -0.0000860072 | -0.0000860072 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.7001464503 | 0.0008053169 | 0.0008053169 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4995650125 | 0.0030504935 | -0.0030504935 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4379839227 | 0.0014127911 | -0.0014127911 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.6981113718 | 0.6985229616 | -0.0004115898 | -0.0004115898 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.7009517672 | 0.6997964117 | 0.0011553555 | 0.0011553555 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5026155060 | 0.4993208537 | 0.0032946523 | -0.0032946523 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4393967138 | 0.4385900635 | 0.0008066503 | -0.0008066503 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 0 | 24 | 24 | 0.7008294191 | 0.6992126229 | 0.0016167962 | 0.0016167962 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 1 | 24 | 24 | 0.7051542074 | 0.7057181542 | -0.0005639468 | -0.0005639468 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 0 | 24 | 24 | 0.4949261188 | 0.5073560961 | -0.0124299773 | 0.0124299773 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 1 | 24 | 24 | 0.4335984096 | 0.4432214809 | -0.0096230714 | 0.0096230714 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 18 | 18 | 0.6995823739 | 0.6995152434 | 0.0000671305 | 0.0000671305 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 18 | 18 | 0.6993513415 | 0.6996282176 | -0.0002768761 | -0.0002768761 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 18 | 18 | 0.4970472474 | 0.4961678097 | 0.0008794377 | -0.0008794377 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 18 | 18 | 0.4351645517 | 0.4354849444 | -0.0003203927 | 0.0003203927 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 18 | 19 | 0.6995823739 | 0.6990026558 | 0.0005797181 | 0.0005797181 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 18 | 19 | 0.6993513415 | 0.6991595727 | 0.0001917688 | 0.0001917688 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 18 | 19 | 0.4970472474 | 0.4977518006 | -0.0007045532 | 0.0007045532 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 18 | 19 | 0.4351645517 | 0.4357990652 | -0.0006345135 | 0.0006345135 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 18 | 18 | 0.6995823739 | 0.6996062149 | -0.0000238410 | -0.0000238410 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 18 | 18 | 0.6993513415 | 0.6995025124 | -0.0001511710 | -0.0001511710 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 18 | 18 | 0.4970472474 | 0.4952106140 | 0.0018366334 | -0.0018366334 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 18 | 18 | 0.4351645517 | 0.4340635806 | 0.0011009711 | -0.0011009711 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 0 | 24 | 12 | 0.7008294191 | 0.6953403575 | 0.0054890616 | 0.0054890616 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | auc | 1 | 24 | 12 | 0.7051542074 | 0.7051551414 | -0.0000009340 | -0.0000009340 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 0 | 24 | 12 | 0.4949261188 | 0.5000965988 | -0.0051704800 | 0.0051704800 |
| R6_PB_HB-R7_PB_HISTORY | internal_mechanism | 3101 | logloss | 1 | 24 | 12 | 0.4335984096 | 0.4359767384 | -0.0023783289 | 0.0023783289 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8102126992 | 0.0003186512 | 0.0003186512 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8199765514 | -0.0000258666 | -0.0000258666 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5320680654 | 0.0006108219 | -0.0006108219 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4256646332 | 0.0027775324 | -0.0027775324 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8103483918 | 0.0001829587 | 0.0001829587 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8189747348 | 0.0009759499 | 0.0009759499 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5303389520 | 0.0023399354 | -0.0023399354 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4293087233 | -0.0008665577 | 0.0008665577 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 24 | 24 | 0.8105313505 | 0.8101782404 | 0.0003531100 | 0.0003531100 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 24 | 24 | 0.8199506847 | 0.8201228043 | -0.0001721195 | -0.0001721195 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 24 | 24 | 0.5326788873 | 0.5297428350 | 0.0029360523 | -0.0029360523 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 24 | 24 | 0.4284421656 | 0.4259180167 | 0.0025241490 | -0.0025241490 |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | task | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 0 | 12 | 12 | 0.8020107835 | 0.8018937399 | 0.0001170436 | 0.0001170436 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | auc | 1 | 12 | 12 | 0.8130585180 | 0.8131734842 | -0.0001149662 | -0.0001149662 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 0 | 12 | 12 | 0.5283598453 | 0.5283280267 | 0.0000318186 | -0.0000318186 |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | p2_formation_attribution | 3101 | logloss | 1 | 12 | 12 | 0.4242839314 | 0.4241257093 | 0.0001582222 | -0.0001582222 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 0 | 12 | 14 | 0.8020107835 | 0.8034672456 | -0.0014564621 | -0.0014564621 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | auc | 1 | 12 | 14 | 0.8130585180 | 0.8145575452 | -0.0014990272 | -0.0014990272 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 0 | 12 | 14 | 0.5283598453 | 0.5277572159 | 0.0006026294 | -0.0006026294 |
| L4_BLOCK_META-LPS_GEOM | p2_formation_attribution | 3101 | logloss | 1 | 12 | 14 | 0.4242839314 | 0.4249118124 | -0.0006278809 | 0.0006278809 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 0 | 12 | 12 | 0.8020107835 | 0.8019750859 | 0.0000356976 | 0.0000356976 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | auc | 1 | 12 | 12 | 0.8130585180 | 0.8130530465 | 0.0000054715 | 0.0000054715 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 0 | 12 | 12 | 0.5283598453 | 0.5284339909 | -0.0000741456 | 0.0000741456 |
| L4_BLOCK_META-LPS_OWNER_META | p2_formation_attribution | 3101 | logloss | 1 | 12 | 12 | 0.4242839314 | 0.4243394960 | -0.0000555645 | 0.0000555645 |


## task seed级统计

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [-0.0006681707] | -0.0006681707 | NaN | -0.0006681707 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0101283945] | -0.0101283945 | NaN | -0.0101283945 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0090073303] | 0.0090073303 | NaN | 0.0090073303 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0090556752] | 0.0090556752 | NaN | 0.0090556752 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0046919108] | 0.0046919108 | NaN | 0.0046919108 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [0.0014638635] | 0.0014638635 | NaN | 0.0014638635 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [0.0073720834] | 0.0073720834 | NaN | 0.0073720834 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [0.0059040952] | 0.0059040952 | NaN | 0.0059040952 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0017905921] | -0.0017905921 | NaN | -0.0017905921 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [0.0014405749] | 0.0014405749 | NaN | 0.0014405749 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [-0.0022835728] | -0.0022835728 | NaN | -0.0022835728 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [-0.0048594660] | -0.0048594660 | NaN | -0.0048594660 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [0.0036571066] | 0.0036571066 | NaN | 0.0036571066 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0105320232] | -0.0105320232 | NaN | -0.0105320232 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0020702381] | 0.0020702381 | NaN | 0.0020702381 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0027919224] | 0.0027919224 | NaN | 0.0027919224 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0061629130] | 0.0061629130 | NaN | 0.0061629130 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [-0.0001365623] | -0.0001365623 | NaN | -0.0001365623 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [0.0018038248] | 0.0018038248 | NaN | 0.0018038248 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [0.0016719331] | 0.0016719331 | NaN | 0.0016719331 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [0.0047323767] | 0.0047323767 | NaN | 0.0047323767 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [0.0055339213] | 0.0055339213 | NaN | 0.0055339213 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [-0.0021679328] | -0.0021679328 | NaN | -0.0021679328 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [-0.0020436599] | -0.0020436599 | NaN | -0.0020436599 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [0.0042386289] | 0.0042386289 | NaN | 0.0042386289 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [0.0017576183] | 0.0017576183 | NaN | 0.0017576183 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0098516227] | 0.0098516227 | NaN | 0.0098516227 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0075195414] | 0.0075195414 | NaN | 0.0075195414 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0048457765] | 0.0048457765 | NaN | 0.0048457765 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [0.0071607366] | 0.0071607366 | NaN | 0.0071607366 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [0.0066822957] | 0.0066822957 | NaN | 0.0066822957 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [-0.0030874075] | -0.0030874075 | NaN | -0.0030874075 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0005321036] | -0.0005321036 | NaN | -0.0005321036 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [-0.0015371198] | -0.0015371198 | NaN | -0.0015371198 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [0.0072956245] | 0.0072956245 | NaN | 0.0072956245 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [0.0037458239] | 0.0037458239 | NaN | 0.0037458239 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=development_external_reference; candidate_stage=p1; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-AUTOIFS | auc | 0 | 1 | [3101] | [-0.0044743583] | -0.0044743583 | NaN | -0.0044743583 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | auc | 1 | 1 | [3101] | [-0.0046613775] | -0.0046613775 | NaN | -0.0046613775 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 0 | 1 | [3101] | [0.0052743009] | 0.0052743009 | NaN | 0.0052743009 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-AUTOIFS | logloss | 1 | 1 | [3101] | [0.0035713569] | 0.0035713569 | NaN | 0.0035713569 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 0 | 1 | [3101] | [0.0034337689] | 0.0034337689 | NaN | 0.0034337689 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | auc | 1 | 1 | [3101] | [-0.0001237579] | -0.0001237579 | NaN | -0.0001237579 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 0 | 1 | [3101] | [-0.0037714769] | -0.0037714769 | NaN | -0.0037714769 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-STEM_DOMAIN_TASK | logloss | 1 | 1 | [3101] | [-0.0012666991] | -0.0012666991 | NaN | -0.0012666991 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 1 | [3101] | [-0.0010958603] | -0.0010958603 | NaN | -0.0010958603 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 1 | [3101] | [-0.0012245275] | -0.0012245275 | NaN | -0.0012245275 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 1 | [3101] | [0.0010952312] | 0.0010952312 | NaN | 0.0010952312 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 1 | [3101] | [0.0005321827] | 0.0005321827 | NaN | 0.0005321827 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [0.0029055306, 0.0014471254] | 0.0021763280 | 0.0010312482 | 0.0021763280 | 2 | 0 | 0 | -0.0070890694 | 0.0114417254 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [0.0010471087, 0.0026282878] | 0.0018376983 | 0.0011180625 | 0.0018376983 | 2 | 0 | 0 | -0.0082076946 | 0.0118830912 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [-0.0003973871, 0.0007119980] | 0.0001573054 | 0.0007844538 | 0.0001573054 | 1 | 1 | 0 | -0.0068907320 | 0.0072053429 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [-0.0007168045, -0.0000835887] | -0.0004001966 | 0.0004477512 | -0.0004001966 | 2 | 0 | 0 | -0.0044230812 | 0.0036226880 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0017798781, -0.0022173371] | -0.0002187295 | 0.0028264580 | -0.0002187295 | 1 | 1 | 0 | -0.0256134470 | 0.0251759880 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [0.0018966213, -0.0021121228] | -0.0001077507 | 0.0028346102 | -0.0001077507 | 1 | 1 | 0 | -0.0255757126 | 0.0253602112 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [0.0003127783, 0.0031687574] | 0.0017407678 | 0.0020194822 | 0.0017407678 | 0 | 2 | 0 | -0.0164035597 | 0.0198850954 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [-0.0000241348, 0.0024973570] | 0.0012366111 | 0.0017829639 | 0.0012366111 | 1 | 1 | 0 | -0.0147826843 | 0.0172559066 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0037620075, 0.0058068745] | 0.0047844410 | 0.0014459394 | 0.0047844410 | 2 | 0 | 0 | -0.0082068089 | 0.0177756909 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [0.0040656678, 0.0067825978] | 0.0054241328 | 0.0019211596 | 0.0054241328 | 2 | 0 | 0 | -0.0118368016 | 0.0226850672 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0066023668, 0.0024413730] | 0.0045218699 | 0.0029422669 | 0.0045218699 | 0 | 2 | 0 | -0.0219133494 | 0.0309570892 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0038298021, 0.0006057982] | 0.0022178001 | 0.0022797150 | 0.0022178001 | 0 | 2 | 0 | -0.0182646269 | 0.0227002272 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [-0.0021862785, 0.0023992811] | 0.0001065013 | 0.0032424803 | 0.0001065013 | 1 | 1 | 0 | -0.0290260283 | 0.0292390309 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [-0.0004011246, 0.0012150848] | 0.0004069801 | 0.0011428326 | 0.0004069801 | 1 | 1 | 0 | -0.0098609638 | 0.0106749240 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [0.0028431797, -0.0040470612] | -0.0006019407 | 0.0048721361 | -0.0006019407 | 1 | 1 | 0 | -0.0443763467 | 0.0431724652 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0008766106, -0.0022438485] | -0.0006836189 | 0.0022064978 | -0.0006836189 | 1 | 1 | 0 | -0.0205082151 | 0.0191409772 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [-0.0002531854, -0.0000657228] | -0.0001594541 | 0.0001325561 | -0.0001594541 | 0 | 2 | 0 | -0.0013504231 | 0.0010315149 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0001459225, -0.0013330010] | -0.0007394617 | 0.0008393913 | -0.0007394617 | 0 | 2 | 0 | -0.0082810929 | 0.0068021695 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0016933803, -0.0005137442] | 0.0005898181 | 0.0015606727 | 0.0005898181 | 1 | 1 | 0 | -0.0134322697 | 0.0146119058 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0016935024, -0.0001354065] | 0.0007790480 | 0.0012932339 | 0.0007790480 | 1 | 1 | 0 | -0.0108401978 | 0.0123982938 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [-0.0010981699, 0.0019288733] | 0.0004153517 | 0.0021404428 | 0.0004153517 | 1 | 1 | 0 | -0.0188157640 | 0.0196464674 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [-0.0006541454, 0.0021370468] | 0.0007414507 | 0.0019736709 | 0.0007414507 | 1 | 1 | 0 | -0.0169912792 | 0.0184741806 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0025709469, -0.0015012415] | 0.0005348527 | 0.0028794720 | 0.0005348527 | 1 | 1 | 0 | -0.0253361772 | 0.0264058826 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0025590207, -0.0010638507] | 0.0007475850 | 0.0025617569 | 0.0007475850 | 1 | 1 | 0 | -0.0222688878 | 0.0237640577 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0009097722] | -0.0064498817 | 0.0104081225 | -0.0064498817 | 1 | 1 | 0 | -0.0999631521 | 0.0870633886 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0013299702] | -0.0063114196 | 0.0108065570 | -0.0063114196 | 1 | 1 | 0 | -0.1034044821 | 0.0907816429 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0019472986] | 0.0194868837 | 0.0248047191 | 0.0194868837 | 0 | 2 | 0 | -0.2033746754 | 0.2423484427 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0031353638] | 0.0199891512 | 0.0238348547 | 0.0199891512 | 0 | 2 | 0 | -0.1941585217 | 0.2341368241 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0009097722] | -0.0064498817 | 0.0104081225 | -0.0064498817 | 1 | 1 | 0 | -0.0999631521 | 0.0870633886 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0013299702] | -0.0063114196 | 0.0108065570 | -0.0063114196 | 1 | 1 | 0 | -0.1034044821 | 0.0907816429 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0019472986] | 0.0194868837 | 0.0248047191 | 0.0194868837 | 0 | 2 | 0 | -0.2033746754 | 0.2423484427 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0031353638] | 0.0199891512 | 0.0238348547 | 0.0199891512 | 0 | 2 | 0 | -0.1941585217 | 0.2341368241 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0012258915] | -0.0062918221 | 0.0106316526 | -0.0062918221 | 1 | 1 | 0 | -0.1018134307 | 0.0892297865 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0017385501] | -0.0061071296 | 0.0110954666 | -0.0061071296 | 1 | 1 | 0 | -0.1057959422 | 0.0935816829 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0018624450] | 0.0194444569 | 0.0248647197 | 0.0194444569 | 0 | 2 | 0 | -0.2039561859 | 0.2428450996 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0023994096] | 0.0196211741 | 0.0243552529 | 0.0196211741 | 0 | 2 | 0 | -0.1992020914 | 0.2384444395 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0012258915] | -0.0062918221 | 0.0106316526 | -0.0062918221 | 1 | 1 | 0 | -0.1018134307 | 0.0892297865 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0017385501] | -0.0061071296 | 0.0110954666 | -0.0061071296 | 1 | 1 | 0 | -0.1057959422 | 0.0935816829 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0018624450] | 0.0194444569 | 0.0248647197 | 0.0194444569 | 0 | 2 | 0 | -0.2039561859 | 0.2428450996 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0023994096] | 0.0196211741 | 0.0243552529 | 0.0196211741 | 0 | 2 | 0 | -0.1992020914 | 0.2384444395 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0009252748] | -0.0064421305 | 0.0104190845 | -0.0064421305 | 1 | 1 | 0 | -0.1000538904 | 0.0871696295 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0006147722] | -0.0066690186 | 0.0103008357 | -0.0066690186 | 1 | 1 | 0 | -0.0992183550 | 0.0858803179 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0021006492] | 0.0195635590 | 0.0246962838 | 0.0195635590 | 0 | 2 | 0 | -0.2023237476 | 0.2414508655 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0021101381] | 0.0194765384 | 0.0245597987 | 0.0194765384 | 0 | 2 | 0 | -0.2011844982 | 0.2401375749 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [-0.0138095357, 0.0009252748] | -0.0064421305 | 0.0104190845 | -0.0064421305 | 1 | 1 | 0 | -0.1000538904 | 0.0871696295 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [-0.0139528093, 0.0006147722] | -0.0066690186 | 0.0103008357 | -0.0066690186 | 1 | 1 | 0 | -0.0992183550 | 0.0858803179 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0370264687, 0.0021006492] | 0.0195635590 | 0.0246962838 | 0.0195635590 | 0 | 2 | 0 | -0.2023237476 | 0.2414508655 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [0.0368429386, 0.0021101381] | 0.0194765384 | 0.0245597987 | 0.0194765384 | 0 | 2 | 0 | -0.2011844982 | 0.2401375749 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0008831519, 0.0000565116] | 0.0004698317 | 0.0005845230 | 0.0004698317 | 2 | 0 | 0 | -0.0047818992 | 0.0057215627 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0006183043, -0.0012275969] | -0.0003046463 | 0.0013052493 | -0.0003046463 | 1 | 1 | 0 | -0.0120318458 | 0.0114225533 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0064082657, -0.0000044797] | -0.0032063727 | 0.0045281605 | -0.0032063727 | 2 | 0 | 0 | -0.0438902808 | 0.0374775354 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0039157201, 0.0013384902] | -0.0012886149 | 0.0037152877 | -0.0012886149 | 1 | 1 | 0 | -0.0346691506 | 0.0320919207 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0008831519, 0.0000565116] | 0.0004698317 | 0.0005845230 | 0.0004698317 | 2 | 0 | 0 | -0.0047818992 | 0.0057215627 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0006183043, -0.0012275969] | -0.0003046463 | 0.0013052493 | -0.0003046463 | 1 | 1 | 0 | -0.0120318458 | 0.0114225533 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0064082657, -0.0000044797] | -0.0032063727 | 0.0045281605 | -0.0032063727 | 2 | 0 | 0 | -0.0438902808 | 0.0374775354 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0039157201, 0.0013384902] | -0.0012886149 | 0.0037152877 | -0.0012886149 | 1 | 1 | 0 | -0.0346691506 | 0.0320919207 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [0.0146926877, -0.0008532607] | 0.0069197135 | 0.0109926455 | 0.0069197135 | 1 | 1 | 0 | -0.0918452878 | 0.1056847148 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [0.0145711137, -0.0025575671] | 0.0060067733 | 0.0121118063 | 0.0060067733 | 1 | 1 | 0 | -0.1028134888 | 0.1148270354 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [-0.0434347345, -0.0019517783] | -0.0226932564 | 0.0293328796 | -0.0226932564 | 2 | 0 | 0 | -0.2862387236 | 0.2408522108 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [-0.0407586586, -0.0017968736] | -0.0212777661 | 0.0275501424 | -0.0212777661 | 2 | 0 | 0 | -0.2688059747 | 0.2262504424 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [0.0146926877, -0.0011693800] | 0.0067616538 | 0.0112161756 | 0.0067616538 | 1 | 1 | 0 | -0.0940116858 | 0.1075349935 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [0.0145711137, -0.0029661470] | 0.0058024833 | 0.0124007159 | 0.0058024833 | 1 | 1 | 0 | -0.1056135287 | 0.1172184954 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [-0.0434347345, -0.0018669247] | -0.0226508296 | 0.0293928802 | -0.0226508296 | 2 | 0 | 0 | -0.2867353805 | 0.2414337213 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [-0.0407586586, -0.0010609194] | -0.0209097890 | 0.0280705406 | -0.0209097890 | 2 | 0 | 0 | -0.2731135901 | 0.2312940121 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [0.0146926877, -0.0008687633] | 0.0069119622 | 0.0110036075 | 0.0069119622 | 1 | 1 | 0 | -0.0919515287 | 0.1057754531 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [0.0145711137, -0.0018423691] | 0.0063643723 | 0.0116060849 | 0.0063643723 | 1 | 1 | 0 | -0.0979121637 | 0.1106409083 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [-0.0434347345, -0.0021051290] | -0.0227699317 | 0.0292244443 | -0.0227699317 | 2 | 0 | 0 | -0.2853411464 | 0.2398012829 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [-0.0407586586, -0.0007716479] | -0.0207651533 | 0.0282750864 | -0.0207651533 | 2 | 0 | 0 | -0.2748067254 | 0.2332764189 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [0.0011258961, 0.0016157352] | 0.0013708157 | 0.0003463685 | 0.0013708157 | 2 | 0 | 0 | -0.0017411819 | 0.0044828133 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0003796420, 0.0010757068] | 0.0003480324 | 0.0010290871 | 0.0003480324 | 1 | 1 | 0 | -0.0088979480 | 0.0095940128 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [-0.0048080026, -0.0006634111] | -0.0027357069 | 0.0029306687 | -0.0027357069 | 2 | 0 | 0 | -0.0290667206 | 0.0235953069 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [-0.0035846515, 0.0006537927] | -0.0014654294 | 0.0029970326 | -0.0014654294 | 1 | 1 | 0 | -0.0283926992 | 0.0254618404 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [0.0018179022, 0.0012031628] | 0.0015105325 | 0.0004346864 | 0.0015105325 | 2 | 0 | 0 | -0.0023949699 | 0.0054160350 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [0.0043383100, 0.0024982451] | 0.0034182775 | 0.0013011224 | 0.0034182775 | 2 | 0 | 0 | -0.0082718431 | 0.0151083982 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [-0.0005873725, -0.0005429873] | -0.0005651799 | 0.0000313851 | -0.0005651799 | 2 | 0 | 0 | -0.0008471636 | -0.0002831962 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [-0.0012699810, -0.0008591466] | -0.0010645638 | 0.0002905038 | -0.0010645638 | 2 | 0 | 0 | -0.0036746370 | 0.0015455094 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0005039547, -0.0008449674] | -0.0001705064 | 0.0009538319 | -0.0001705064 | 1 | 1 | 0 | -0.0087403463 | 0.0083993336 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [0.0016865843, -0.0004624390] | 0.0006120727 | 0.0015195889 | 0.0006120727 | 1 | 1 | 0 | -0.0130408922 | 0.0142650375 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [-0.0003742781, 0.0006873066] | 0.0001565143 | 0.0007506538 | 0.0001565143 | 1 | 1 | 0 | -0.0065878422 | 0.0069008707 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [-0.0008013285, 0.0013066475] | 0.0002526595 | 0.0014905641 | 0.0002526595 | 1 | 1 | 0 | -0.0131395280 | 0.0136448470 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0094338757, 0.0058863538] | 0.0076601147 | 0.0025084768 | 0.0076601147 | 2 | 0 | 0 | -0.0148776553 | 0.0301978847 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [0.0087770767, 0.0053376300] | 0.0070573534 | 0.0024320561 | 0.0070573534 | 2 | 0 | 0 | -0.0147938036 | 0.0289085103 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [-0.0002572272, 0.0013394236] | 0.0005410982 | 0.0011290026 | 0.0005410982 | 1 | 1 | 0 | -0.0096025881 | 0.0106847844 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [-0.0008026038, 0.0024034177] | 0.0008004069 | 0.0022669995 | 0.0008004069 | 1 | 1 | 0 | -0.0195677755 | 0.0211685894 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [0.0041751275, 0.0006930412] | 0.0024340843 | 0.0024622068 | 0.0024340843 | 2 | 0 | 0 | -0.0196879664 | 0.0245561350 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [0.0004165613, -0.0011697813] | -0.0003766100 | 0.0011217136 | -0.0003766100 | 1 | 1 | 0 | -0.0104548072 | 0.0097015872 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [-0.0000384281, -0.0008672816] | -0.0004528549 | 0.0005860879 | -0.0004528549 | 2 | 0 | 0 | -0.0057186455 | 0.0048129358 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0013878597, 0.0009043350] | 0.0011460974 | 0.0003419036 | 0.0011460974 | 0 | 2 | 0 | -0.0019257846 | 0.0042179793 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [0.0008874218, 0.0010469125] | 0.0009671671 | 0.0001127770 | 0.0009671671 | 2 | 0 | 0 | -0.0000460937 | 0.0019804279 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0001059035, -0.0004234356] | -0.0002646695 | 0.0002245291 | -0.0002646695 | 0 | 2 | 0 | -0.0022819833 | 0.0017526443 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0019394991, 0.0001082857] | 0.0010238924 | 0.0012948635 | 0.0010238924 | 0 | 2 | 0 | -0.0106099943 | 0.0126577791 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0011096361, 0.0023216801] | 0.0017156581 | 0.0008570445 | 0.0017156581 | 0 | 2 | 0 | -0.0059845813 | 0.0094158975 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [0.0032253051, -0.0002609289] | 0.0014821881 | 0.0024651397 | 0.0014821881 | 1 | 1 | 0 | -0.0206662138 | 0.0236305899 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [0.0016009141, 0.0001844390] | 0.0008926766 | 0.0010015991 | 0.0008926766 | 2 | 0 | 0 | -0.0081063343 | 0.0098916875 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [-0.0004936562, 0.0005046733] | 0.0000055085 | 0.0007059256 | 0.0000055085 | 1 | 1 | 0 | -0.0063369810 | 0.0063479980 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0007221370, 0.0010725799] | 0.0008973585 | 0.0002478005 | 0.0008973585 | 0 | 2 | 0 | -0.0013290410 | 0.0031237580 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0026875705, -0.0022710225] | 0.0002082740 | 0.0035062548 | 0.0002082740 | 1 | 1 | 0 | -0.0312941755 | 0.0317107234 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0068077764, -0.0002405557] | 0.0032836103 | 0.0049839235 | 0.0032836103 | 1 | 1 | 0 | -0.0414951654 | 0.0480623861 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0005171101, 0.0039248656] | 0.0022209878 | 0.0024096470 | 0.0022209878 | 0 | 2 | 0 | -0.0194288315 | 0.0238708072 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0013192771, 0.0019571316] | 0.0003189273 | 0.0023167709 | 0.0003189273 | 1 | 1 | 0 | -0.0204964330 | 0.0211342875 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0043062225, 0.0027349926] | 0.0035206075 | 0.0011110273 | 0.0035206075 | 2 | 0 | 0 | -0.0064615768 | 0.0135027918 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0035938512, 0.0049702365] | 0.0042820439 | 0.0009732513 | 0.0042820439 | 2 | 0 | 0 | -0.0044622723 | 0.0130263600 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0029124445, -0.0016799798] | -0.0022962122 | 0.0008714842 | -0.0022962122 | 2 | 0 | 0 | -0.0101261871 | 0.0055337628 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0015784618, -0.0019673436] | -0.0017729027 | 0.0002749809 | -0.0017729027 | 2 | 0 | 0 | -0.0042435085 | 0.0006977032 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0043062225, 0.0027349926] | 0.0035206075 | 0.0011110273 | 0.0035206075 | 2 | 0 | 0 | -0.0064615768 | 0.0135027918 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0035938512, 0.0049702365] | 0.0042820439 | 0.0009732513 | 0.0042820439 | 2 | 0 | 0 | -0.0044622723 | 0.0130263600 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0029124445, -0.0016799798] | -0.0022962122 | 0.0008714842 | -0.0022962122 | 2 | 0 | 0 | -0.0101261871 | 0.0055337628 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0015784618, -0.0019673436] | -0.0017729027 | 0.0002749809 | -0.0017729027 | 2 | 0 | 0 | -0.0042435085 | 0.0006977032 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [0.0016186519, 0.0050060151] | 0.0033123335 | 0.0023952275 | 0.0033123335 | 2 | 0 | 0 | -0.0182079316 | 0.0248325987 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [-0.0032139252, 0.0052107922] | 0.0009984335 | 0.0059571748 | 0.0009984335 | 1 | 1 | 0 | -0.0525246583 | 0.0545215253 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [-0.0034295546, -0.0056048453] | -0.0045172000 | 0.0015381628 | -0.0045172000 | 2 | 0 | 0 | -0.0183370444 | 0.0093026444 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [-0.0002591846, -0.0039244752] | -0.0020918299 | 0.0025917518 | -0.0020918299 | 2 | 0 | 0 | -0.0253777960 | 0.0211941362 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [0.0016186519, 0.0050060151] | 0.0033123335 | 0.0023952275 | 0.0033123335 | 2 | 0 | 0 | -0.0182079316 | 0.0248325987 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [-0.0032139252, 0.0052107922] | 0.0009984335 | 0.0059571748 | 0.0009984335 | 1 | 1 | 0 | -0.0525246583 | 0.0545215253 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [-0.0034295546, -0.0056048453] | -0.0045172000 | 0.0015381628 | -0.0045172000 | 2 | 0 | 0 | -0.0183370444 | 0.0093026444 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [-0.0002591846, -0.0039244752] | -0.0020918299 | 0.0025917518 | -0.0020918299 | 2 | 0 | 0 | -0.0253777960 | 0.0211941362 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [0.0016186519, 0.0050060151] | 0.0033123335 | 0.0023952275 | 0.0033123335 | 2 | 0 | 0 | -0.0182079316 | 0.0248325987 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [-0.0032139252, 0.0052107922] | 0.0009984335 | 0.0059571748 | 0.0009984335 | 1 | 1 | 0 | -0.0525246583 | 0.0545215253 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [-0.0034295546, -0.0056048453] | -0.0045172000 | 0.0015381628 | -0.0045172000 | 2 | 0 | 0 | -0.0183370444 | 0.0093026444 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [-0.0002591846, -0.0039244752] | -0.0020918299 | 0.0025917518 | -0.0020918299 | 2 | 0 | 0 | -0.0253777960 | 0.0211941362 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [0.0060195297, 0.0026333547] | 0.0043264422 | 0.0023943873 | 0.0043264422 | 2 | 0 | 0 | -0.0171862740 | 0.0258391585 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [0.0035245573, 0.0038973051] | 0.0037109312 | 0.0002635725 | 0.0037109312 | 2 | 0 | 0 | 0.0013428264 | 0.0060790360 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [-0.0023370262, -0.0022777058] | -0.0023073660 | 0.0000419459 | -0.0023073660 | 2 | 0 | 0 | -0.0026842346 | -0.0019304974 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [-0.0017037193, -0.0026581720] | -0.0021809456 | 0.0006748999 | -0.0021809456 | 2 | 0 | 0 | -0.0082446811 | 0.0038827898 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [-0.0001487318, 0.0003193284] | 0.0000852983 | 0.0003309685 | 0.0000852983 | 1 | 1 | 0 | -0.0028883360 | 0.0030589326 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [0.0000065799, -0.0000583972] | -0.0000259086 | 0.0000459458 | -0.0000259086 | 1 | 1 | 0 | -0.0004387148 | 0.0003868976 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [0.0021307828, -0.0025209324] | -0.0001950748 | 0.0032892593 | -0.0001950748 | 1 | 1 | 0 | -0.0297478975 | 0.0293577478 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [-0.0000608979, -0.0011772455] | -0.0006190717 | 0.0007893769 | -0.0006190717 | 2 | 0 | 0 | -0.0077113422 | 0.0064731988 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0000769942, -0.0001049648] | -0.0000139853 | 0.0001286644 | -0.0000139853 | 1 | 1 | 0 | -0.0011699893 | 0.0011420187 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [-0.0003110725, 0.0001709574] | -0.0000700575 | 0.0003408466 | -0.0000700575 | 1 | 1 | 0 | -0.0031324428 | 0.0029923278 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [-0.0015861836, -0.0002284028] | -0.0009072932 | 0.0009600960 | -0.0009072932 | 2 | 0 | 0 | -0.0095334135 | 0.0077188271 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [-0.0001012256, -0.0008484331] | -0.0004748294 | 0.0005283555 | -0.0004748294 | 2 | 0 | 0 | -0.0052219153 | 0.0042722566 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [0.0001308538, -0.0016910663] | -0.0007801063 | 0.0012882920 | -0.0007801063 | 1 | 1 | 0 | -0.0123549509 | 0.0107947383 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [0.0000189767, -0.0012523879] | -0.0006167056 | 0.0008989905 | -0.0006167056 | 1 | 1 | 0 | -0.0086938152 | 0.0074604040 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0124233541, 0.0101764530] | 0.0112999036 | 0.0015887990 | 0.0112999036 | 0 | 2 | 0 | -0.0029748893 | 0.0255746965 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0105361532, 0.0061854419] | 0.0083607975 | 0.0030764174 | 0.0083607975 | 0 | 2 | 0 | -0.0192797163 | 0.0360013114 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [0.0003878797, -0.0003992624] | -0.0000056913 | 0.0005565936 | -0.0000056913 | 1 | 1 | 0 | -0.0050064861 | 0.0049951034 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [0.0002599486, -0.0004880569] | -0.0001140541 | 0.0005289197 | -0.0001140541 | 1 | 1 | 0 | -0.0048662093 | 0.0046381010 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [0.0020372115, -0.0018110535] | 0.0001130790 | 0.0027211342 | 0.0001130790 | 1 | 1 | 0 | -0.0243353420 | 0.0245615000 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [0.0006675700, -0.0017784529] | -0.0005554415 | 0.0017295994 | -0.0005554415 | 1 | 1 | 0 | -0.0160952753 | 0.0149843924 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [0.0000114075, 0.0008188704] | 0.0004151390 | 0.0005709625 | 0.0004151390 | 2 | 0 | 0 | -0.0047147559 | 0.0055450338 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0003262340, 0.0000731675] | -0.0001265333 | 0.0002824195 | -0.0001265333 | 1 | 1 | 0 | -0.0026639718 | 0.0024109052 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0004058079, -0.0028270886] | -0.0012106404 | 0.0022860030 | -0.0012106404 | 1 | 1 | 0 | -0.0217495625 | 0.0193282818 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0013738905, -0.0022503730] | -0.0004382413 | 0.0025627413 | -0.0004382413 | 1 | 1 | 0 | -0.0234635582 | 0.0225870757 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [0.0002596026, 0.0005575732] | 0.0004085879 | 0.0002106971 | 0.0004085879 | 2 | 0 | 0 | -0.0014844504 | 0.0023016262 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [0.0004955839, -0.0001674191] | 0.0001640824 | 0.0004688139 | 0.0001640824 | 1 | 1 | 0 | -0.0040480430 | 0.0043762078 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0056192128, -0.0005266939] | 0.0025462594 | 0.0043458123 | 0.0025462594 | 1 | 1 | 0 | -0.0364993149 | 0.0415918337 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0027570545, 0.0009222572] | 0.0018396558 | 0.0012973977 | 0.0018396558 | 0 | 2 | 0 | -0.0098169997 | 0.0134963114 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0006579803, -0.0019823706] | -0.0006621952 | 0.0018670100 | -0.0006621952 | 1 | 1 | 0 | -0.0174366146 | 0.0161122243 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0002520446, -0.0015293562] | -0.0006386558 | 0.0012596405 | -0.0006386558 | 1 | 1 | 0 | -0.0119560771 | 0.0106787654 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0018518589, 0.0002020352] | -0.0008249118 | 0.0014523225 | -0.0008249118 | 1 | 1 | 0 | -0.0138735116 | 0.0122236879 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0012572610, 0.0042204899] | 0.0014816144 | 0.0038733548 | 0.0014816144 | 1 | 1 | 0 | -0.0333190973 | 0.0362823262 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0014332739, -0.0003901614] | 0.0005215562 | 0.0012893635 | 0.0005215562 | 1 | 1 | 0 | -0.0110629154 | 0.0121060279 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0014612951, 0.0012296649] | 0.0013454800 | 0.0001637872 | 0.0013454800 | 2 | 0 | 0 | -0.0001260899 | 0.0028170499 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0022641077, 0.0018676856] | -0.0001982111 | 0.0029216191 | -0.0001982111 | 1 | 1 | 0 | -0.0264479169 | 0.0260514948 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0049995867, 0.0010250023] | -0.0019872922 | 0.0042600277 | -0.0019872922 | 1 | 1 | 0 | -0.0402621229 | 0.0362875385 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0002174343, -0.0021099116] | -0.0011636730 | 0.0013381835 | -0.0011636730 | 0 | 2 | 0 | -0.0131867746 | 0.0108594287 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0001416056, -0.0017475782] | -0.0009445919 | 0.0011355941 | -0.0009445919 | 0 | 2 | 0 | -0.0111475002 | 0.0092583163 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0011842293, -0.0006474396] | -0.0009158345 | 0.0003795676 | -0.0009158345 | 2 | 0 | 0 | -0.0043261138 | 0.0024944449 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0008298708, 0.0039348373] | 0.0015524833 | 0.0033691574 | 0.0015524833 | 1 | 1 | 0 | -0.0287181947 | 0.0318231612 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0005578593, -0.0005177024] | 0.0000200784 | 0.0007605370 | 0.0000200784 | 1 | 1 | 0 | -0.0068130754 | 0.0068532322 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0010676449, 0.0010114429] | 0.0010395439 | 0.0000397408 | 0.0010395439 | 2 | 0 | 0 | 0.0006824869 | 0.0013966008 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0015964781, 0.0010182107] | -0.0002891337 | 0.0018488642 | -0.0002891337 | 1 | 1 | 0 | -0.0169005191 | 0.0163222518 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0045721965, 0.0007393498] | -0.0019164234 | 0.0037558304 | -0.0019164234 | 1 | 1 | 0 | -0.0356612203 | 0.0318283736 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0014406828, -0.0015291895] | -0.0000442534 | 0.0021000169 | -0.0000442534 | 1 | 1 | 0 | -0.0189121562 | 0.0188236495 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [0.0002012471, -0.0001717955] | 0.0000147258 | 0.0002637810 | 0.0000147258 | 1 | 1 | 0 | -0.0023552524 | 0.0023847040 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0039453214, -0.0038991213] | -0.0039222214 | 0.0000326684 | -0.0039222214 | 2 | 0 | 0 | -0.0042157357 | -0.0036287070 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [-0.0037969380, -0.0002808303] | -0.0020388841 | 0.0024862636 | -0.0020388841 | 2 | 0 | 0 | -0.0243770765 | 0.0202993083 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0022159765, 0.0000630196] | 0.0011394980 | 0.0015223704 | 0.0011394980 | 2 | 0 | 0 | -0.0125384570 | 0.0148174531 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0014104976, 0.0025872256] | 0.0019988616 | 0.0008320723 | 0.0019988616 | 2 | 0 | 0 | -0.0054770115 | 0.0094747348 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0043575702, -0.0022334709] | -0.0032955206 | 0.0015019650 | -0.0032955206 | 2 | 0 | 0 | -0.0167901410 | 0.0101990999 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0075392637, -0.0034763178] | -0.0055077907 | 0.0028729366 | -0.0055077907 | 2 | 0 | 0 | -0.0313201021 | 0.0203045206 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0003412731, -0.0004048111] | -0.0003730421 | 0.0000449282 | -0.0003730421 | 0 | 2 | 0 | -0.0007767056 | 0.0000306214 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0007725278, -0.0015125545] | -0.0011425412 | 0.0005232779 | -0.0011425412 | 0 | 2 | 0 | -0.0058440067 | 0.0035589243 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0044005920, -0.0059768113] | -0.0007881096 | 0.0073379323 | -0.0007881096 | 1 | 1 | 0 | -0.0667168154 | 0.0651405962 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0004136723, 0.0014678020] | 0.0009407371 | 0.0007453823 | 0.0009407371 | 0 | 2 | 0 | -0.0057562570 | 0.0076377312 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0004340205, 0.0011873980] | 0.0008107093 | 0.0005327183 | 0.0008107093 | 2 | 0 | 0 | -0.0039755751 | 0.0055969936 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0004367227, 0.0012464666] | 0.0008415946 | 0.0005725754 | 0.0008415946 | 2 | 0 | 0 | -0.0043027912 | 0.0059859804 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0039883432, -0.0043111609] | -0.0001614089 | 0.0058686357 | -0.0001614089 | 1 | 1 | 0 | -0.0528890085 | 0.0525661908 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0033286534, -0.0017276855] | -0.0025281695 | 0.0011320553 | -0.0025281695 | 2 | 0 | 0 | -0.0126992826 | 0.0076429436 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [-0.0009992534, 0.0015775595] | 0.0002891530 | 0.0018220819 | 0.0002891530 | 1 | 1 | 0 | -0.0160816029 | 0.0166599090 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [-0.0010245724, 0.0000168016] | -0.0005038854 | 0.0007363626 | -0.0005038854 | 1 | 1 | 0 | -0.0071198411 | 0.0061120704 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [0.0062524509, -0.0061788465] | 0.0000368022 | 0.0087902548 | 0.0000368022 | 1 | 1 | 0 | -0.0789405034 | 0.0790141078 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [0.0016709332, -0.0027526879] | -0.0005408773 | 0.0031279725 | -0.0005408773 | 1 | 1 | 0 | -0.0286445950 | 0.0275628403 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [-0.0001238388, 0.0017051004] | 0.0007906308 | 0.0012932553 | 0.0007906308 | 1 | 1 | 0 | -0.0108288073 | 0.0124100690 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [-0.0006309222, 0.0002350237] | -0.0001979493 | 0.0006123162 | -0.0001979493 | 1 | 1 | 0 | -0.0056993920 | 0.0053034935 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [0.0055848213, -0.0053293717] | 0.0001277248 | 0.0077174999 | 0.0001277248 | 1 | 1 | 0 | -0.0692112603 | 0.0694667100 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [0.0012435430, -0.0024670353] | -0.0006117461 | 0.0026237751 | -0.0006117461 | 1 | 1 | 0 | -0.0241854300 | 0.0229619377 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [-0.0017819559, 0.0011243784] | -0.0003287888 | 0.0020550887 | -0.0003287888 | 1 | 1 | 0 | -0.0187930282 | 0.0181354506 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [-0.0009737750, -0.0013407590] | -0.0011572670 | 0.0002594969 | -0.0011572670 | 0 | 2 | 0 | -0.0034887543 | 0.0011742203 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [0.0083459135, -0.0020776900] | 0.0031341117 | 0.0073706007 | 0.0031341117 | 1 | 1 | 0 | -0.0630881084 | 0.0693563319 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [0.0042106102, 0.0017486322] | 0.0029796212 | 0.0017408813 | 0.0029796212 | 0 | 2 | 0 | -0.0126615770 | 0.0186208195 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [-0.0010448220, 0.0020548431] | 0.0005050105 | 0.0021917942 | 0.0005050105 | 1 | 1 | 0 | -0.0191874795 | 0.0201975006 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0002978629, 0.0011510898] | 0.0004266135 | 0.0010245643 | 0.0004266135 | 1 | 1 | 0 | -0.0087787314 | 0.0096319583 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [0.0059457923, -0.0051932664] | 0.0003762629 | 0.0078765040 | 0.0003762629 | 1 | 1 | 0 | -0.0703913176 | 0.0711438434 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [-0.0031071418, -0.0022686850] | -0.0026879134 | 0.0005928785 | -0.0026879134 | 2 | 0 | 0 | -0.0080147153 | 0.0026388885 |

## analysis_cohort=development_p1; candidate_stage=p1; reference_stage=p1; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 0 | 2 | [3101, 3102] | [-0.0006416445, 0.0024091162] | 0.0008837358 | 0.0021572136 | 0.0008837358 | 1 | 1 | 0 | -0.0184980593 | 0.0202655310 |
| L2_BLOCK_UNIFORM-L0_OWN | auc | 1 | 2 | [3101, 3102] | [-0.0002820201, 0.0010536308] | 0.0003858053 | 0.0009444477 | 0.0003858053 | 1 | 1 | 0 | -0.0080997209 | 0.0088713316 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 0 | 2 | [3101, 3102] | [0.0012711731, -0.0033909131] | -0.0010598700 | 0.0032965927 | -0.0010598700 | 1 | 1 | 0 | -0.0306785805 | 0.0285588406 |
| L2_BLOCK_UNIFORM-L0_OWN | logloss | 1 | 2 | [3101, 3102] | [0.0000112839, -0.0020259281] | -0.0010073221 | 0.0014405264 | -0.0010073221 | 1 | 1 | 0 | -0.0139499384 | 0.0119352943 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 0 | 2 | [3101, 3102] | [0.0000281711, 0.0022826603] | 0.0011554157 | 0.0015941646 | 0.0011554157 | 2 | 0 | 0 | -0.0131675850 | 0.0154784165 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | auc | 1 | 2 | [3101, 3102] | [-0.0000424757, 0.0008476737] | 0.0004025990 | 0.0006294306 | 0.0004025990 | 1 | 1 | 0 | -0.0052526109 | 0.0060578089 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 0 | 2 | [3101, 3102] | [0.0000026592, -0.0032452473] | -0.0016212940 | 0.0022966167 | -0.0016212940 | 1 | 1 | 0 | -0.0222555768 | 0.0190129887 |
| L2_BLOCK_UNIFORM-L1_FULL_COARSE | logloss | 1 | 2 | [3101, 3102] | [0.0001002914, -0.0008625871] | -0.0003811479 | 0.0006808579 | -0.0003811479 | 1 | 1 | 0 | -0.0064984135 | 0.0057361178 |
| L4_BLOCK_META-L0_FEATURE | auc | 0 | 2 | [3101, 3102] | [-0.0004231099, -0.0004425613] | -0.0004328356 | 0.0000137542 | -0.0004328356 | 0 | 2 | 0 | -0.0005564122 | -0.0003092591 |
| L4_BLOCK_META-L0_FEATURE | auc | 1 | 2 | [3101, 3102] | [-0.0009766305, 0.0000696741] | -0.0004534782 | 0.0007398491 | -0.0004534782 | 1 | 1 | 0 | -0.0071007582 | 0.0061938018 |
| L4_BLOCK_META-L0_FEATURE | logloss | 0 | 2 | [3101, 3102] | [0.0017856780, 0.0021321517] | 0.0019589149 | 0.0002449939 | 0.0019589149 | 0 | 2 | 0 | -0.0002422676 | 0.0041600973 |
| L4_BLOCK_META-L0_FEATURE | logloss | 1 | 2 | [3101, 3102] | [0.0020058187, 0.0016883395] | 0.0018470791 | 0.0002244917 | 0.0018470791 | 0 | 2 | 0 | -0.0001698991 | 0.0038640573 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 0 | 2 | [3101, 3102] | [0.0000302589, 0.0022327123] | 0.0011314856 | 0.0015573697 | 0.0011314856 | 2 | 0 | 0 | -0.0128609263 | 0.0151238974 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | auc | 1 | 2 | [3101, 3102] | [-0.0000000085, 0.0017488625] | 0.0008744270 | 0.0012366385 | 0.0008744270 | 1 | 1 | 0 | -0.0102363294 | 0.0119851834 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 0 | 2 | [3101, 3102] | [-0.0000465520, -0.0014367972] | -0.0007416746 | 0.0009830518 | -0.0007416746 | 2 | 0 | 0 | -0.0095740447 | 0.0080906956 |
| L4_BLOCK_META-L2_BLOCK_UNIFORM | logloss | 1 | 2 | [3101, 3102] | [-0.0000590999, -0.0013067713] | -0.0006829356 | 0.0008822369 | -0.0006829356 | 2 | 0 | 0 | -0.0086095195 | 0.0072436483 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 0 | 2 | [3101, 3102] | [-0.0000079446, 0.0022723230] | 0.0011321892 | 0.0016123926 | 0.0011321892 | 1 | 1 | 0 | -0.0133545838 | 0.0156189622 |
| L4_BLOCK_META-L3_SCALAR_META | auc | 1 | 2 | [3101, 3102] | [-0.0000176443, 0.0017672476] | 0.0008748016 | 0.0012621092 | 0.0008748016 | 1 | 1 | 0 | -0.0104647996 | 0.0122144029 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 0 | 2 | [3101, 3102] | [0.0000099038, -0.0015630945] | -0.0007765953 | 0.0011122777 | -0.0007765953 | 1 | 1 | 0 | -0.0107700143 | 0.0092168236 |
| L4_BLOCK_META-L3_SCALAR_META | logloss | 1 | 2 | [3101, 3102] | [0.0000246727, -0.0013375505] | -0.0006564389 | 0.0009632373 | -0.0006564389 | 1 | 1 | 0 | -0.0093107829 | 0.0079979051 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 0 | 2 | [3101, 3102] | [-0.0014898266, -0.0003978802] | -0.0009438534 | 0.0007721227 | -0.0009438534 | 0 | 2 | 0 | -0.0078811010 | 0.0059933942 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | auc | 1 | 2 | [3101, 3102] | [-0.0014436880, 0.0000423927] | -0.0007006477 | 0.0010508177 | -0.0007006477 | 1 | 1 | 0 | -0.0101418700 | 0.0087405747 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 0 | 2 | [3101, 3102] | [0.0015585876, 0.0011286228] | 0.0013436052 | 0.0003040310 | 0.0013436052 | 0 | 2 | 0 | -0.0013880048 | 0.0040752152 |
| L4_BLOCK_META-L5_BLOCK_CONSTANT | logloss | 1 | 2 | [3101, 3102] | [0.0005697514, 0.0008013400] | 0.0006855457 | 0.0001637579 | 0.0006855457 | 0 | 2 | 0 | -0.0007857608 | 0.0021568522 |
| R2_FREEZE-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0001746001, -0.0001004424] | 0.0000370789 | 0.0001944844 | 0.0000370789 | 1 | 1 | 0 | -0.0017102940 | 0.0017844518 |
| R2_FREEZE-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0000272649, -0.0002448297] | -0.0001360473 | 0.0001538415 | -0.0001360473 | 0 | 2 | 0 | -0.0015182587 | 0.0012461641 |
| R2_FREEZE-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0000954872, -0.0005332575] | -0.0003143724 | 0.0003095504 | -0.0003143724 | 2 | 0 | 0 | -0.0030955723 | 0.0024668275 |
| R2_FREEZE-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0002241322, -0.0004628805] | -0.0001193741 | 0.0004857914 | -0.0001193741 | 1 | 1 | 0 | -0.0044840363 | 0.0042452880 |
| R2_FREEZE-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0008566320, 0.0002206329] | 0.0005386325 | 0.0004497192 | 0.0005386325 | 2 | 0 | 0 | -0.0035019346 | 0.0045791995 |
| R2_FREEZE-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0013108562, -0.0001774873] | 0.0005666845 | 0.0010524178 | 0.0005666845 | 1 | 1 | 0 | -0.0088889143 | 0.0100222832 |
| R2_FREEZE-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0016262928, -0.0008040089] | -0.0012151509 | 0.0005814425 | -0.0012151509 | 2 | 0 | 0 | -0.0064392046 | 0.0040089029 |
| R2_FREEZE-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0014169395, 0.0001814300] | -0.0006177547 | 0.0011302179 | -0.0006177547 | 1 | 1 | 0 | -0.0107723601 | 0.0095368506 |
| R3_CUM-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0002090709, -0.0000488899] | 0.0000800905 | 0.0001824058 | 0.0000800905 | 1 | 1 | 0 | -0.0015587606 | 0.0017189416 |
| R3_CUM-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0000112678, 0.0000299446] | 0.0000093384 | 0.0000291416 | 0.0000093384 | 1 | 1 | 0 | -0.0002524884 | 0.0002711651 |
| R3_CUM-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0003426275, 0.0000322959] | -0.0001551658 | 0.0002651109 | -0.0001551658 | 1 | 1 | 0 | -0.0025370924 | 0.0022267609 |
| R3_CUM-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0001851046, -0.0005825646] | -0.0001987300 | 0.0005428241 | -0.0001987300 | 1 | 1 | 0 | -0.0050758109 | 0.0046783509 |
| R3_CUM-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0008911028, 0.0002721854] | 0.0005816441 | 0.0004376407 | 0.0005816441 | 2 | 0 | 0 | -0.0033504012 | 0.0045136894 |
| R3_CUM-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0013268533, 0.0000972870] | 0.0007120702 | 0.0008694347 | 0.0007120702 | 2 | 0 | 0 | -0.0070994905 | 0.0085236308 |
| R3_CUM-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0018734331, -0.0002384554] | -0.0010559443 | 0.0011561038 | -0.0010559443 | 2 | 0 | 0 | -0.0114431246 | 0.0093312361 |
| R3_CUM-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0014559672, 0.0000617459] | -0.0006971106 | 0.0010731852 | -0.0006971106 | 1 | 1 | 0 | -0.0103392972 | 0.0089450760 |
| R4_EMA-R0_BASE | auc | 0 | 2 | [3101, 3102] | [0.0002154344, -0.0000488899] | 0.0000832723 | 0.0001869055 | 0.0000832723 | 1 | 1 | 0 | -0.0015960070 | 0.0017625515 |
| R4_EMA-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0000186918, 0.0000299446] | 0.0000056264 | 0.0000343911 | 0.0000056264 | 1 | 1 | 0 | -0.0003033655 | 0.0003146183 |
| R4_EMA-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [-0.0003738452, 0.0000322959] | -0.0001707746 | 0.0002871851 | -0.0001707746 | 1 | 1 | 0 | -0.0027510306 | 0.0024094814 |
| R4_EMA-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0001673217, -0.0005825646] | -0.0002076215 | 0.0005302497 | -0.0002076215 | 1 | 1 | 0 | -0.0049717258 | 0.0045564828 |
| R4_EMA-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0008974663, 0.0002721854] | 0.0005848259 | 0.0004421403 | 0.0005848259 | 2 | 0 | 0 | -0.0033876476 | 0.0045572993 |
| R4_EMA-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0013194294, 0.0000972870] | 0.0007083582 | 0.0008641851 | 0.0007083582 | 2 | 0 | 0 | -0.0070560373 | 0.0084727537 |
| R4_EMA-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [-0.0019046508, -0.0002384554] | -0.0010715531 | 0.0011781780 | -0.0010715531 | 2 | 0 | 0 | -0.0116570628 | 0.0095139565 |
| R4_EMA-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0014737501, 0.0000617459] | -0.0007060021 | 0.0010857596 | -0.0007060021 | 1 | 1 | 0 | -0.0104611653 | 0.0090491611 |
| R6_PB_HB-R0_BASE | auc | 0 | 2 | [3101, 3102] | [-0.0005832029, -0.0004313830] | -0.0005072929 | 0.0001073528 | -0.0005072929 | 0 | 2 | 0 | -0.0014718198 | 0.0004572339 |
| R6_PB_HB-R0_BASE | auc | 1 | 2 | [3101, 3102] | [-0.0013135055, -0.0004144513] | -0.0008639784 | 0.0006357274 | -0.0008639784 | 0 | 2 | 0 | -0.0065757622 | 0.0048478054 |
| R6_PB_HB-R0_BASE | logloss | 0 | 2 | [3101, 3102] | [0.0016502318, -0.0000167439] | 0.0008167439 | 0.0011787298 | 0.0008167439 | 1 | 1 | 0 | -0.0097737234 | 0.0114072113 |
| R6_PB_HB-R0_BASE | logloss | 1 | 2 | [3101, 3102] | [0.0015630419, -0.0002581339] | 0.0006524540 | 0.0012877658 | 0.0006524540 | 1 | 1 | 0 | -0.0109176626 | 0.0122225706 |
| R6_PB_HB-R1_STATIC | auc | 0 | 2 | [3101, 3102] | [0.0000988290, -0.0001103077] | -0.0000057394 | 0.0001478820 | -0.0000057394 | 1 | 1 | 0 | -0.0013344066 | 0.0013229279 |
| R6_PB_HB-R1_STATIC | auc | 1 | 2 | [3101, 3102] | [0.0000246156, -0.0003471089] | -0.0001612466 | 0.0002628489 | -0.0001612466 | 1 | 1 | 0 | -0.0025228502 | 0.0022003570 |
| R6_PB_HB-R1_STATIC | logloss | 0 | 2 | [3101, 3102] | [0.0001194262, -0.0002874953] | -0.0000840346 | 0.0002877369 | -0.0000840346 | 1 | 1 | 0 | -0.0026692482 | 0.0025011791 |
| R6_PB_HB-R1_STATIC | logloss | 1 | 2 | [3101, 3102] | [-0.0000780298, 0.0003861766] | 0.0001540734 | 0.0003282435 | 0.0001540734 | 1 | 1 | 0 | -0.0027950775 | 0.0031032243 |
| R6_PB_HB-R2_FREEZE | auc | 0 | 2 | [3101, 3102] | [-0.0007578029, -0.0003309407] | -0.0005443718 | 0.0003018372 | -0.0005443718 | 0 | 2 | 0 | -0.0032562716 | 0.0021675280 |
| R6_PB_HB-R2_FREEZE | auc | 1 | 2 | [3101, 3102] | [-0.0012862406, -0.0001696216] | -0.0007279311 | 0.0007895689 | -0.0007279311 | 0 | 2 | 0 | -0.0078219263 | 0.0063660641 |
| R6_PB_HB-R2_FREEZE | logloss | 0 | 2 | [3101, 3102] | [0.0017457190, 0.0005165136] | 0.0011311163 | 0.0008691794 | 0.0011311163 | 0 | 2 | 0 | -0.0066781511 | 0.0089403837 |
| R6_PB_HB-R2_FREEZE | logloss | 1 | 2 | [3101, 3102] | [0.0013389097, 0.0002047466] | 0.0007718281 | 0.0008019744 | 0.0007718281 | 0 | 2 | 0 | -0.0064336263 | 0.0079772826 |
| R6_PB_HB-R3_CUM | auc | 0 | 2 | [3101, 3102] | [-0.0007922737, -0.0003824931] | -0.0005873834 | 0.0002897586 | -0.0005873834 | 0 | 2 | 0 | -0.0031907614 | 0.0020159946 |
| R6_PB_HB-R3_CUM | auc | 1 | 2 | [3101, 3102] | [-0.0013022377, -0.0004443959] | -0.0008733168 | 0.0006065858 | -0.0008733168 | 0 | 2 | 0 | -0.0063232739 | 0.0045766403 |
| R6_PB_HB-R3_CUM | logloss | 0 | 2 | [3101, 3102] | [0.0019928592, -0.0000490399] | 0.0009719097 | 0.0014438407 | 0.0009719097 | 1 | 1 | 0 | -0.0120004843 | 0.0139443037 |
| R6_PB_HB-R3_CUM | logloss | 1 | 2 | [3101, 3102] | [0.0013779374, 0.0003244307] | 0.0008511840 | 0.0007449417 | 0.0008511840 | 0 | 2 | 0 | -0.0058418517 | 0.0075442197 |
| R6_PB_HB-R4_EMA | auc | 0 | 2 | [3101, 3102] | [-0.0007986373, -0.0003824931] | -0.0005905652 | 0.0002942583 | -0.0005905652 | 0 | 2 | 0 | -0.0032343713 | 0.0020532409 |
| R6_PB_HB-R4_EMA | auc | 1 | 2 | [3101, 3102] | [-0.0012948137, -0.0004443959] | -0.0008696048 | 0.0006013363 | -0.0008696048 | 0 | 2 | 0 | -0.0062723967 | 0.0045331871 |
| R6_PB_HB-R4_EMA | logloss | 0 | 2 | [3101, 3102] | [0.0020240770, -0.0000490399] | 0.0009875186 | 0.0014659150 | 0.0009875186 | 1 | 1 | 0 | -0.0121832048 | 0.0141582419 |
| R6_PB_HB-R4_EMA | logloss | 1 | 2 | [3101, 3102] | [0.0013957203, 0.0003244307] | 0.0008600755 | 0.0007575161 | 0.0008600755 | 0 | 2 | 0 | -0.0059459368 | 0.0076660878 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 2 | [3101, 3102] | [-0.0008738154, -0.0002581346] | -0.0005659750 | 0.0004353521 | -0.0005659750 | 0 | 2 | 0 | -0.0044774585 | 0.0033455085 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 2 | [3101, 3102] | [-0.0012883590, -0.0002195839] | -0.0007539715 | 0.0007557382 | -0.0007539715 | 0 | 2 | 0 | -0.0075440094 | 0.0060360665 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 2 | [3101, 3102] | [0.0023505310, 0.0002970306] | 0.0013237808 | 0.0014520441 | 0.0013237808 | 0 | 2 | 0 | -0.0117223174 | 0.0143698791 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 2 | [3101, 3102] | [0.0012015473, 0.0000331366] | 0.0006173419 | 0.0008261912 | 0.0006173419 | 0 | 2 | 0 | -0.0068056913 | 0.0080403752 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0004561155, 0.0005131559, -0.0019857239, -0.0012775209] | -0.0005734933 | 0.0012557801 | -0.0004107027 | 2 | 2 | 0 | -0.0025717197 | 0.0014247330 |
| R6_PB_HB-R0_BASE | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0105543505, -0.0101690923, -0.0029078337, 0.0004997811] | -0.0005056986 | 0.0086117576 | -0.0012040263 | 2 | 2 | 0 | -0.0142089266 | 0.0131975295 |
| R6_PB_HB-R0_BASE | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0107134954, 0.0017713293, -0.0110826041, 0.0002997778] | -0.0049312481 | 0.0069176523 | -0.0052068588 | 2 | 2 | 0 | -0.0159387766 | 0.0060762804 |
| R6_PB_HB-R0_BASE | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0136466467, 0.0001585004, -0.0217193656, -0.0040087423] | -0.0098040636 | 0.0098247697 | -0.0088276945 | 3 | 1 | 0 | -0.0254374645 | 0.0058293374 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0032690269, 0.0010240026, -0.0085019659, 0.0038861010] | -0.0000807088 | 0.0057472970 | 0.0021465148 | 3 | 1 | 0 | -0.0092259408 | 0.0090645232 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0095662714, -0.0026454168, -0.0076972984, 0.0014061683] | 0.0001574311 | 0.0072947093 | -0.0006196242 | 2 | 2 | 0 | -0.0114500793 | 0.0117649415 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0117632517, -0.0093491395, -0.0059990220, -0.0056492589] | -0.0081901680 | 0.0029078905 | -0.0076740808 | 4 | 0 | 0 | -0.0128172707 | -0.0035630653 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0149941135, 0.0014924559, -0.0122352616, -0.0038010700] | -0.0073844973 | 0.0075955562 | -0.0080181658 | 3 | 1 | 0 | -0.0194707222 | 0.0047017275 |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0025708746, 0.0060336363, 0.0090337018, 0.0038422500] | 0.0040846784 | 0.0049209222 | 0.0049379432 | 3 | 1 | 0 | -0.0037456070 | 0.0119149638 |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0009689724, 0.0026025646, 0.0021203108, -0.0031294111] | 0.0006406092 | 0.0026051015 | 0.0015446416 | 3 | 1 | 0 | -0.0035046887 | 0.0047859070 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0102398199, -0.0276593202, -0.0428632240, -0.0236588601] | -0.0261053060 | 0.0134280989 | -0.0256590902 | 4 | 0 | 0 | -0.0474724080 | -0.0047382041 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0105130320, -0.0165597407, -0.0600368877, -0.0256315555] | -0.0281853040 | 0.0221247066 | -0.0210956481 | 4 | 0 | 0 | -0.0633906494 | 0.0070200415 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0005172083, 0.0018921253, -0.0079109442, -0.0010073163] | -0.0018858359 | 0.0042118824 | -0.0007622623 | 1 | 3 | 0 | -0.0085878806 | 0.0048162089 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0033547580, 0.0033708172, -0.0066162249, -0.0065808599] | -0.0032952564 | 0.0046997908 | -0.0049678090 | 1 | 3 | 0 | -0.0107736723 | 0.0041831595 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0172715377, -0.0015600685, -0.0058776835, -0.0081844219] | -0.0082234279 | 0.0066275357 | -0.0070310527 | 4 | 0 | 0 | -0.0187693161 | 0.0023224603 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0168052569, 0.0012717305, -0.0115273352, -0.0121636720] | -0.0098061334 | 0.0077508563 | -0.0118455036 | 3 | 1 | 0 | -0.0221394754 | 0.0025272086 |

## analysis_cohort=formal_locked_reevaluation; candidate_stage=formal; reference_stage=formal; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6_PB_HB-R0_BASE | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0025823218, 0.0037583517, -0.0093120181, 0.0014666196] | -0.0016673421 | 0.0057311467 | -0.0005578511 | 2 | 2 | 0 | -0.0107868755 | 0.0074521912 |
| R6_PB_HB-R0_BASE | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0026246422, 0.0029297883, -0.0082483436, 0.0036938260] | -0.0010623429 | 0.0055569068 | 0.0001525731 | 2 | 2 | 0 | -0.0099046216 | 0.0077799359 |
| R6_PB_HB-R0_BASE | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0036447316, 0.0008484509, 0.0034925543, -0.0000555353] | 0.0019825504 | 0.0018693076 | 0.0021705026 | 1 | 3 | 0 | -0.0009919352 | 0.0049570360 |
| R6_PB_HB-R0_BASE | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0016034906, 0.0051722130, 0.0010238647, -0.0005898527] | 0.0018024289 | 0.0024306511 | 0.0013136777 | 1 | 3 | 0 | -0.0020652794 | 0.0056701373 |
| R6_PB_HB-R5_PB_H1 | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0020331737, 0.0042015399, -0.0116484936, 0.0004872107] | -0.0022482292 | 0.0067698688 | -0.0007729815 | 2 | 2 | 0 | -0.0130206012 | 0.0085241428 |
| R6_PB_HB-R5_PB_H1 | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0040607982, 0.0057763295, -0.0153575231, 0.0016600688] | -0.0029954808 | 0.0091755797 | -0.0012003647 | 2 | 2 | 0 | -0.0175958756 | 0.0116049141 |
| R6_PB_HB-R5_PB_H1 | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0038951343, 0.0029375481, 0.0036426828, -0.0002239324] | 0.0025628582 | 0.0019015410 | 0.0032901154 | 1 | 3 | 0 | -0.0004629180 | 0.0055886343 |
| R6_PB_HB-R5_PB_H1 | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0038063083, 0.0070177751, 0.0037607574, -0.0003656199] | 0.0035548052 | 0.0030258643 | 0.0037835329 | 1 | 3 | 0 | -0.0012600200 | 0.0083696305 |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0029152667, 0.0042952526, -0.0005573687, 0.0008574064] | 0.0004200059 | 0.0030159664 | 0.0001500188 | 2 | 2 | 0 | -0.0043790697 | 0.0052190815 |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0054666927, 0.0022234695, -0.0007080661, 0.0029169628] | -0.0002585816 | 0.0038109980 | 0.0007577017 | 2 | 2 | 0 | -0.0063227299 | 0.0058055666 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0011250409, -0.0005996480, 0.0008204722, 0.0001821571] | 0.0003820056 | 0.0007633125 | 0.0005013146 | 1 | 3 | 0 | -0.0008325949 | 0.0015966060 |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0026709364, 0.0049227962, 0.0005409894, -0.0010376593] | 0.0017742656 | 0.0025913417 | 0.0016059629 | 1 | 3 | 0 | -0.0023491373 | 0.0058976685 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 0 | 4 | [4101, 4102, 4103, 4104] | [-0.0080797785, 0.0094552123, -0.0118985929, 0.0024757134] | -0.0020118614 | 0.0097672784 | -0.0028020325 | 2 | 2 | 0 | -0.0175537810 | 0.0135300582 |
| R6_PB_HB-STGU_DOMAIN_ADAM | auc | 1 | 4 | [4101, 4102, 4103, 4104] | [-0.0088960221, 0.0083285213, -0.0130816156, -0.0010651215] | -0.0036785595 | 0.0094276089 | -0.0049805718 | 1 | 3 | 0 | -0.0186799890 | 0.0113228701 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 0 | 4 | [4101, 4102, 4103, 4104] | [0.0044429241, 0.0003513482, 0.0000657609, -0.0063518571] | -0.0003729560 | 0.0044593366 | 0.0002085545 | 1 | 3 | 0 | -0.0074687556 | 0.0067228437 |
| R6_PB_HB-STGU_DOMAIN_ADAM | logloss | 1 | 4 | [4101, 4102, 4103, 4104] | [0.0020080913, 0.0063132912, -0.0021845581, -0.0062126808] | -0.0000189641 | 0.0053931541 | -0.0000882334 | 2 | 2 | 0 | -0.0086006758 | 0.0085627476 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0072446133] | -0.0072446133 | NaN | -0.0072446133 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0079632755] | -0.0079632755 | NaN | -0.0079632755 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [0.0022150095] | 0.0022150095 | NaN | 0.0022150095 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [0.0016999286] | 0.0016999286 | NaN | 0.0016999286 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0004913323] | -0.0004913323 | NaN | -0.0004913323 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0016947407] | -0.0016947407 | NaN | -0.0016947407 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [-0.0013729910] | -0.0013729910 | NaN | -0.0013729910 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [-0.0001030754] | -0.0001030754 | NaN | -0.0001030754 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [0.0016702549] | 0.0016702549 | NaN | 0.0016702549 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [0.0003741290] | 0.0003741290 | NaN | 0.0003741290 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [-0.0023642462] | -0.0023642462 | NaN | -0.0023642462 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [0.0003662851] | 0.0003662851 | NaN | 0.0003662851 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 0 | 1 | [3101] | [-0.0023091526] | -0.0023091526 | NaN | -0.0023091526 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | auc | 1 | 1 | [3101] | [-0.0018429537] | -0.0018429537 | NaN | -0.0018429537 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 0 | 1 | [3101] | [-0.0000137139] | -0.0000137139 | NaN | -0.0000137139 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | logloss | 1 | 1 | [3101] | [0.0011417998] | 0.0011417998 | NaN | 0.0011417998 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0021034749] | -0.0021034749 | NaN | -0.0021034749 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0009900309] | -0.0009900309 | NaN | -0.0009900309 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0002639191] | 0.0002639191 | NaN | 0.0002639191 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0009290584] | 0.0009290584 | NaN | 0.0009290584 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0002559257] | -0.0002559257 | NaN | -0.0002559257 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0019787465] | -0.0019787465 | NaN | -0.0019787465 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0011779325] | 0.0011779325 | NaN | 0.0011779325 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0012789492] | 0.0012789492 | NaN | 0.0012789492 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0007251155] | -0.0007251155 | NaN | -0.0007251155 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0018557824] | -0.0018557824 | NaN | -0.0018557824 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [-0.0002463374] | -0.0002463374 | NaN | -0.0002463374 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0008420674] | 0.0008420674 | NaN | 0.0008420674 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0095053370] | -0.0095053370 | NaN | -0.0095053370 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0090998034] | -0.0090998034 | NaN | -0.0090998034 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0049696106] | 0.0049696106 | NaN | 0.0049696106 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0037308957] | 0.0037308957 | NaN | 0.0037308957 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0000562389] | -0.0000562389 | NaN | -0.0000562389 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0003452097] | -0.0003452097 | NaN | -0.0003452097 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0018287524] | 0.0018287524 | NaN | 0.0018287524 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0010246465] | 0.0010246465 | NaN | 0.0010246465 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0012050668] | -0.0012050668 | NaN | -0.0012050668 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0001110380] | -0.0001110380 | NaN | -0.0001110380 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0059700336] | -0.0059700336 | NaN | -0.0059700336 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [-0.0020806410] | -0.0020806410 | NaN | -0.0020806410 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 0 | 1 | [3101] | [-0.0039441395] | -0.0039441395 | NaN | -0.0039441395 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | auc | 1 | 1 | [3101] | [-0.0017950418] | -0.0017950418 | NaN | -0.0017950418 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 0 | 1 | [3101] | [0.0028845632] | 0.0028845632 | NaN | 0.0028845632 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | logloss | 1 | 1 | [3101] | [0.0003181862] | 0.0003181862 | NaN | 0.0003181862 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0001752059] | 0.0001752059 | NaN | 0.0001752059 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0000341916] | -0.0000341916 | NaN | -0.0000341916 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0003527751] | 0.0003527751 | NaN | 0.0003527751 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0001388767] | 0.0001388767 | NaN | 0.0001388767 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0053970640] | -0.0053970640 | NaN | -0.0053970640 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0089519911] | -0.0089519911 | NaN | -0.0089519911 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0031290229] | 0.0031290229 | NaN | 0.0031290229 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0020498195] | 0.0020498195 | NaN | 0.0020498195 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [-0.0092715539] | -0.0092715539 | NaN | -0.0092715539 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0089387617] | -0.0089387617 | NaN | -0.0089387617 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [0.0038429570] | 0.0038429570 | NaN | 0.0038429570 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0027857529] | 0.0027857529 | NaN | 0.0027857529 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0005214270] | 0.0005214270 | NaN | 0.0005214270 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [0.0006083007] | 0.0006083007 | NaN | 0.0006083007 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0101630322] | -0.0101630322 | NaN | -0.0101630322 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [-0.0027390025] | -0.0027390025 | NaN | -0.0027390025 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 0 | 1 | [3101] | [0.0018101928] | 0.0018101928 | NaN | 0.0018101928 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | auc | 1 | 1 | [3101] | [-0.0000821035] | -0.0000821035 | NaN | -0.0000821035 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 0 | 1 | [3101] | [-0.0025455020] | -0.0025455020 | NaN | -0.0025455020 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | logloss | 1 | 1 | [3101] | [0.0009624903] | 0.0009624903 | NaN | 0.0009624903 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [-0.0001978388] | -0.0001978388 | NaN | -0.0001978388 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [0.0005097251] | 0.0005097251 | NaN | 0.0005097251 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0035857293] | 0.0035857293 | NaN | 0.0035857293 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0016321469] | 0.0016321469 | NaN | 0.0016321469 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [-0.0000860072] | -0.0000860072 | NaN | -0.0000860072 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [0.0008053169] | 0.0008053169 | NaN | 0.0008053169 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0030504935] | 0.0030504935 | NaN | 0.0030504935 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [0.0014127911] | 0.0014127911 | NaN | 0.0014127911 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [-0.0004115898] | -0.0004115898 | NaN | -0.0004115898 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [0.0011553555] | 0.0011553555 | NaN | 0.0011553555 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0032946523] | 0.0032946523 | NaN | 0.0032946523 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0008066503] | 0.0008066503 | NaN | 0.0008066503 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 1 | [3101] | [0.0016167962] | 0.0016167962 | NaN | 0.0016167962 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 1 | [3101] | [-0.0005639468] | -0.0005639468 | NaN | -0.0005639468 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 1 | [3101] | [-0.0124299773] | -0.0124299773 | NaN | -0.0124299773 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 1 | [3101] | [-0.0096230714] | -0.0096230714 | NaN | -0.0096230714 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=kuairand; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0000671305] | 0.0000671305 | NaN | 0.0000671305 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [-0.0002768761] | -0.0002768761 | NaN | -0.0002768761 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0008794377] | 0.0008794377 | NaN | 0.0008794377 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [-0.0003203927] | -0.0003203927 | NaN | -0.0003203927 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [0.0005797181] | 0.0005797181 | NaN | 0.0005797181 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [0.0001917688] | 0.0001917688 | NaN | 0.0001917688 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [-0.0007045532] | -0.0007045532 | NaN | -0.0007045532 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [-0.0006345135] | -0.0006345135 | NaN | -0.0006345135 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [-0.0000238410] | -0.0000238410 | NaN | -0.0000238410 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [-0.0001511710] | -0.0001511710 | NaN | -0.0001511710 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0018366334] | 0.0018366334 | NaN | 0.0018366334 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0011009711] | 0.0011009711 | NaN | 0.0011009711 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 0 | 1 | [3101] | [0.0054890616] | 0.0054890616 | NaN | 0.0054890616 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | auc | 1 | 1 | [3101] | [-0.0000009340] | -0.0000009340 | NaN | -0.0000009340 | 0 | 1 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 0 | 1 | [3101] | [-0.0051704800] | -0.0051704800 | NaN | -0.0051704800 | 1 | 0 | 0 | NaN | NaN |
| R6_PB_HB-R7_PB_HISTORY | logloss | 1 | 1 | [3101] | [-0.0023783289] | -0.0023783289 | NaN | -0.0023783289 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=endpoint

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0003186512] | 0.0003186512 | NaN | 0.0003186512 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [-0.0000258666] | -0.0000258666 | NaN | -0.0000258666 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0006108219] | 0.0006108219 | NaN | 0.0006108219 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0027775324] | 0.0027775324 | NaN | 0.0027775324 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [0.0001829587] | 0.0001829587 | NaN | 0.0001829587 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [0.0009759499] | 0.0009759499 | NaN | 0.0009759499 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0023399354] | 0.0023399354 | NaN | 0.0023399354 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [-0.0008665577] | -0.0008665577 | NaN | -0.0008665577 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [0.0003531100] | 0.0003531100 | NaN | 0.0003531100 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [-0.0001721195] | -0.0001721195 | NaN | -0.0001721195 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [0.0029360523] | 0.0029360523 | NaN | 0.0029360523 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [0.0025241490] | 0.0025241490 | NaN | 0.0025241490 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=p2_attribution; candidate_stage=p1; reference_stage=p2; dataset=movielens; view=selected

| candidate - reference | metric | task | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 0 | 1 | [3101] | [0.0001170436] | 0.0001170436 | NaN | 0.0001170436 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | auc | 1 | 1 | [3101] | [-0.0001149662] | -0.0001149662 | NaN | -0.0001149662 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 0 | 1 | [3101] | [0.0000318186] | 0.0000318186 | NaN | 0.0000318186 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_CAL_PROB_CLIP | logloss | 1 | 1 | [3101] | [0.0001582222] | 0.0001582222 | NaN | 0.0001582222 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 0 | 1 | [3101] | [-0.0014564621] | -0.0014564621 | NaN | -0.0014564621 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | auc | 1 | 1 | [3101] | [-0.0014990272] | -0.0014990272 | NaN | -0.0014990272 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 0 | 1 | [3101] | [0.0006026294] | 0.0006026294 | NaN | 0.0006026294 | 0 | 1 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_GEOM | logloss | 1 | 1 | [3101] | [-0.0006278809] | -0.0006278809 | NaN | -0.0006278809 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 0 | 1 | [3101] | [0.0000356976] | 0.0000356976 | NaN | 0.0000356976 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | auc | 1 | 1 | [3101] | [0.0000054715] | 0.0000054715 | NaN | 0.0000054715 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 0 | 1 | [3101] | [-0.0000741456] | -0.0000741456 | NaN | -0.0000741456 | 1 | 0 | 0 | NaN | NaN |
| L4_BLOCK_META-LPS_OWNER_META | logloss | 1 | 1 | [3101] | [-0.0000555645] | -0.0000555645 | NaN | -0.0000555645 | 1 | 0 | 0 | NaN | NaN |

