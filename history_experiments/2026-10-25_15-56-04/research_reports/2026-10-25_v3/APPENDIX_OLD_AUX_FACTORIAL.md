# 旧组织×aux二乘二析因的完整匹配数值

本附录仅转录完成包的机器聚合数值，不作算法效果判断。质量数值固定显示10位小数，缺失或不可定义值显式为 `NaN`。`selected` 是各预测器按预登记验证指标、eligible≥12且平局取最早epoch的独立选点；`endpoint` 是同一预登记训练终点（开发24epoch，若请求正式manifest则为其50epoch）。Q_explore为已暴露开发评价，不是新未见确认。

统计单位为训练seed。配对差统一为候选减参照；AUC正差、LL负差对应有利方向。`favorable_gain`已统一为越大越有利。seed级Student t95%区间在n=2仅作开发描述；n=1无区间。仅actual formal阶段元数据对应的统计才属于锁定复评，实际n逐表列出；formal与开发、P2归因的seed分别分组，不能合成n=6。域、任务、cell与epoch不增加独立重复数；不报告p值。

此处为旧离散K1/K4算法：aux=.1复用旧完整轨迹，aux=0为可核实断点续接到24。每数据仅seed3101；n=1无训练seed区间。它们不混入新v3 LPS六臂形成机制。历史源码身份完整保留。

## analysis_cohort=historical_bridge_reference; stage=aux-baselines; dataset=kuairand; view=endpoint

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0 | 3101 | main | 24 | 0.6991445797 | 0.4681147677 | 0.5125094037 | 0.7622648008 | 0.7713141656 | 0.6523504774 | 0.6387338539 |
| OLD_K1_DIRECT7_AUX0 | 3101 | main | 24 | 0.7067485241 | 0.4661572987 | 0.5121440053 | 0.7628462823 | 0.7722256686 | 0.6718441880 | 0.6366140470 |

## analysis_cohort=historical_bridge_reference; stage=aux-baselines; dataset=kuairand; view=selected

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0 | 3101 | main | 15 | 0.6961439305 | 0.4662605703 | 0.5108651028 | 0.7617359379 | 0.7703082961 | 0.6458693695 | 0.6363554916 |
| OLD_K1_DIRECT7_AUX0 | 3101 | main | 12 | 0.6972369670 | 0.4669986035 | 0.5136903789 | 0.7575827983 | 0.7663001936 | 0.6573623903 | 0.6364872131 |

## analysis_cohort=historical_bridge_reference; stage=aux-baselines; dataset=movielens; view=endpoint

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0 | 3101 | main | 24 | 0.8157819901 | 0.4798452455 | 0.4783539871 | 0.8092671360 | 0.8177535228 | 0.8046989085 | 0.5428564779 |
| OLD_K1_DIRECT7_AUX0 | 3101 | main | 24 | 0.8147597982 | 0.4808442261 | 0.4806190507 | 0.8085671686 | 0.8180070208 | 0.8031220990 | 0.5355692867 |

## analysis_cohort=historical_bridge_reference; stage=aux-baselines; dataset=movielens; view=selected

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0 | 3101 | main | 14 | 0.8101738569 | 0.4749584148 | 0.4746770694 | 0.8033889645 | 0.8160543967 | 0.7985271718 | 0.5292937168 |
| OLD_K1_DIRECT7_AUX0 | 3101 | main | 14 | 0.8122499100 | 0.4743943719 | 0.4736438712 | 0.8040110753 | 0.8179202941 | 0.8016730795 | 0.5287088508 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=endpoint

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 0.7006913326 | 0.4675182790 | 0.5116681947 | 0.7628320642 | 0.7718434869 | 0.6587015501 | 0.6381128869 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 0.7078658602 | 0.4649288578 | 0.5112425978 | 0.7637096168 | 0.7733703144 | 0.6701862775 | 0.6335827190 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=selected

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 0.6974343794 | 0.4659627053 | 0.5115421636 | 0.7612763873 | 0.7699818392 | 0.6490739330 | 0.6343827835 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 0.7065395372 | 0.4626483503 | 0.5103541353 | 0.7633182577 | 0.7719047312 | 0.6714932023 | 0.6301413848 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=endpoint

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 0.8159827144 | 0.4784185460 | 0.4781601904 | 0.8092114360 | 0.8167140660 | 0.8037548391 | 0.5348834267 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 0.8154178506 | 0.4848695634 | 0.4839003214 | 0.8100591417 | 0.8180846225 | 0.8056502784 | 0.5452297516 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=selected

| method | seed | predictor | epoch | macro AUC | macro LL | natural LL | task0 pooled AUC | task1 pooled AUC | worst-cell AUC | worst-cell LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 0.8130434475 | 0.4733570401 | 0.4732327579 | 0.8072909807 | 0.8170548918 | 0.8018234031 | 0.5252107507 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 0.8121794029 | 0.4741485460 | 0.4731846831 | 0.8048777687 | 0.8181055371 | 0.8017734769 | 0.5310280998 |


## 历史aux=.1全部六格

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=endpoint

| method | seed | predictor | epoch | cell | domain | task | n | positive | positive rate | AUC | LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 0 | 0 | 0 | 8639 | 817 | 0.0945711285 | 0.6587015501 | 0.3015044485 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 1 | 1 | 0 | 87195 | 32928 | 0.3776363432 | 0.7442502251 | 0.5739074143 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 2 | 2 | 0 | 4473 | 2758 | 0.6165884137 | 0.6905595596 | 0.6233294033 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 3 | 0 | 1 | 8639 | 385 | 0.0445653424 | 0.6610150450 | 0.1770675977 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 4 | 1 | 1 | 87195 | 22257 | 0.2552554607 | 0.7528494510 | 0.4911879231 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 5 | 2 | 1 | 4473 | 2149 | 0.4804381728 | 0.6967721648 | 0.6381128869 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 0 | 0 | 0 | 8639 | 817 | 0.0945711285 | 0.6701862775 | 0.3000122442 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 1 | 1 | 0 | 87195 | 32928 | 0.3776363432 | 0.7447434495 | 0.5745603138 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 2 | 2 | 0 | 4473 | 2758 | 0.6165884137 | 0.6947727998 | 0.6147816394 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 3 | 0 | 1 | 8639 | 385 | 0.0445653424 | 0.6803662294 | 0.1761730448 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 4 | 1 | 1 | 87195 | 22257 | 0.2552554607 | 0.7543643829 | 0.4904631855 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 5 | 2 | 1 | 4473 | 2149 | 0.4804381728 | 0.7027620220 | 0.6335827190 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=selected

| method | seed | predictor | epoch | cell | domain | task | n | positive | positive rate | AUC | LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 0 | 0 | 0 | 8639 | 817 | 0.0945711285 | 0.6490739330 | 0.3020074803 |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 1 | 1 | 0 | 87195 | 32928 | 0.3776363432 | 0.7415351480 | 0.5742607417 |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 2 | 2 | 0 | 4473 | 2758 | 0.6165884137 | 0.6880754001 | 0.6175313280 |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 3 | 0 | 1 | 8639 | 385 | 0.0445653424 | 0.6606130361 | 0.1765600356 |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 4 | 1 | 1 | 87195 | 22257 | 0.2552554607 | 0.7501478391 | 0.4910338625 |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | 5 | 2 | 1 | 4473 | 2149 | 0.4804381728 | 0.6951609202 | 0.6343827835 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 0 | 0 | 0 | 8639 | 817 | 0.0945711285 | 0.6714932023 | 0.2970497327 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 1 | 1 | 0 | 87195 | 32928 | 0.3776363432 | 0.7430986978 | 0.5740260902 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 2 | 2 | 0 | 4473 | 2758 | 0.6165884137 | 0.6919072425 | 0.6111948562 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 3 | 0 | 1 | 8639 | 385 | 0.0445653424 | 0.6801280135 | 0.1736175164 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 4 | 1 | 1 | 87195 | 22257 | 0.2552554607 | 0.7518152972 | 0.4898605218 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | 5 | 2 | 1 | 4473 | 2149 | 0.4804381728 | 0.7007947699 | 0.6301413848 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=endpoint

| method | seed | predictor | epoch | cell | domain | task | n | positive | positive rate | AUC | LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 0 | 0 | 0 | 7797 | 4434 | 0.5686802864 | 0.8150395177 | 0.5348834267 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 1 | 1 | 0 | 15930 | 8981 | 0.5637790561 | 0.8140166840 | 0.5273690109 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 2 | 2 | 0 | 16281 | 9868 | 0.6061052680 | 0.8037548391 | 0.5271264109 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 3 | 0 | 1 | 7797 | 1864 | 0.2390663028 | 0.8274313073 | 0.4249000178 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 4 | 1 | 1 | 15930 | 3565 | 0.2237915844 | 0.8233738287 | 0.4171482389 |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 5 | 2 | 1 | 16281 | 3897 | 0.2393587679 | 0.8122801098 | 0.4390841711 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 0 | 0 | 0 | 7797 | 4434 | 0.5686802864 | 0.8124390489 | 0.5452297516 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 1 | 1 | 0 | 15930 | 8981 | 0.5637790561 | 0.8131574325 | 0.5408337000 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 2 | 2 | 0 | 16281 | 9868 | 0.6061052680 | 0.8056502784 | 0.5338044933 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 3 | 0 | 1 | 7797 | 1864 | 0.2390663028 | 0.8234630412 | 0.4341193423 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 4 | 1 | 1 | 15930 | 3565 | 0.2237915844 | 0.8256601875 | 0.4175775177 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 5 | 2 | 1 | 16281 | 3897 | 0.2393587679 | 0.8121371148 | 0.4376525751 |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=selected

| method | seed | predictor | epoch | cell | domain | task | n | positive | positive rate | AUC | LL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 0 | 0 | 0 | 7797 | 4434 | 0.5686802864 | 0.8116206225 | 0.5229796093 |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 1 | 1 | 0 | 15930 | 8981 | 0.5637790561 | 0.8099727220 | 0.5252107507 |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 2 | 2 | 0 | 16281 | 9868 | 0.6061052680 | 0.8018234031 | 0.5246086245 |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 3 | 0 | 1 | 7797 | 1864 | 0.2390663028 | 0.8217857817 | 0.4254941297 |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 4 | 1 | 1 | 15930 | 3565 | 0.2237915844 | 0.8231598827 | 0.4073028176 |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | 5 | 2 | 1 | 16281 | 3897 | 0.2393587679 | 0.8098982732 | 0.4345463087 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 0 | 0 | 0 | 7797 | 4434 | 0.5686802864 | 0.8087845643 | 0.5310280998 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 1 | 1 | 0 | 15930 | 8981 | 0.5637790561 | 0.8074281278 | 0.5272658558 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 2 | 2 | 0 | 16281 | 9868 | 0.6061052680 | 0.8017734769 | 0.5221415349 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 3 | 0 | 1 | 7797 | 1864 | 0.2390663028 | 0.8181734212 | 0.4268933105 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 4 | 1 | 1 | 15930 | 3565 | 0.2237915844 | 0.8241815308 | 0.4081301917 |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | 5 | 2 | 1 | 16281 | 3897 | 0.2393587679 | 0.8127352962 | 0.4294322830 |


## 精确历史身份

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=endpoint

| method | seed | predictor | epoch | checkpoint SHA256 | source SHA256 | historical source SHA256 | server checkpoint path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 3a3d8342ce4acae80031da27dfa2a17ff5d45c7bf43239d33d95b09db3fc33bf | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_kuairand_DIRECT51_3101/epoch24.pt |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | ba7d9b5a45a3bd12e79dcbcce5030922cbdd27e0315c4916b2dce12867135302 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_kuairand_K1_DIRECT7_3101/epoch24.pt |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=kuairand; view=selected

| method | seed | predictor | epoch | checkpoint SHA256 | source SHA256 | historical source SHA256 | server checkpoint path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 15 | a88b7759f1f78a361e7ebabae004e7eda83020b0794e315a9bdaa90930d1e2e5 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_kuairand_DIRECT51_3101/primary_best.pt |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 18 | a6992748c7d7f38924c1df089cad684ba5a8c6b1623eeae761200a721a08bcc3 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_kuairand_K1_DIRECT7_3101/primary_best.pt |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=endpoint

| method | seed | predictor | epoch | checkpoint SHA256 | source SHA256 | historical source SHA256 | server checkpoint path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 24 | 3d9a8f064b4dd1c63a1c00d3b713095acbdcbd11e1df51fc7b8ff671d0c70990 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_movielens_DIRECT51_3101/epoch24.pt |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 24 | 8c2ab30dedaaa9f71cbcce52d3929ab55563354e16b49b7a431bc725a4136a26 | cbe015172169a261025614bf2c14bd77d13eb69d0855677df992926068b26af5 | cbe015172169a261025614bf2c14bd77d13eb69d0855677df992926068b26af5 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_movielens_K1_DIRECT7_3101/epoch24.pt |

## analysis_cohort=historical_bridge_reference; stage=old_complete_reference; dataset=movielens; view=selected

| method | seed | predictor | epoch | checkpoint SHA256 | source SHA256 | historical source SHA256 | server checkpoint path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01 | 3101 | main | 17 | d44cec9834c90cb1019c2b0a17982716577c0ad56b28b413cc09bb630ba06b6b | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | f07322dc861bae8150f39a767d44cbf640511ebcdac7867d11ab2b9e5fab4ed7 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_movielens_DIRECT51_3101/primary_best.pt |
| OLD_K1_DIRECT7_AUX01 | 3101 | main | 14 | d7d764cc07260fcd2598a091e2ff4d0a96b9f4fc88cf6fd3ba3149d1eccb57e6 | cbe015172169a261025614bf2c14bd77d13eb69d0855677df992926068b26af5 | cbe015172169a261025614bf2c14bd77d13eb69d0855677df992926068b26af5 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/checkpoints/S2_movielens_K1_DIRECT7_3101/primary_best.pt |


## 二乘二配对

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_auc | 24 | 24 | 0.6991445797 | 0.7067485241 | -0.0076039444 | -0.0076039444 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_logloss | 24 | 24 | 0.4681147677 | 0.4661572987 | 0.0019574691 | -0.0019574691 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | natural_logloss | 24 | 24 | 0.5125094037 | 0.5121440053 | 0.0003653984 | -0.0003653984 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.7622648008 | 0.7628462823 | -0.0005814814 | -0.0005814814 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.7713141656 | 0.7722256686 | -0.0009115031 | -0.0009115031 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_auc | 15 | 12 | 0.6961439305 | 0.6972369670 | -0.0010930365 | -0.0010930365 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_logloss | 15 | 12 | 0.4662605703 | 0.4669986035 | -0.0007380332 | 0.0007380332 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | natural_logloss | 15 | 12 | 0.5108651028 | 0.5136903789 | -0.0028252761 | 0.0028252761 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_0 | 15 | 12 | 0.7617359379 | 0.7575827983 | 0.0041531396 | 0.0041531396 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_1 | 15 | 12 | 0.7703082961 | 0.7663001936 | 0.0040081025 | 0.0040081025 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_auc | 24 | 24 | 0.8157819901 | 0.8147597982 | 0.0010221920 | 0.0010221920 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_logloss | 24 | 24 | 0.4798452455 | 0.4808442261 | -0.0009989805 | 0.0009989805 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | natural_logloss | 24 | 24 | 0.4783539871 | 0.4806190507 | -0.0022650636 | 0.0022650636 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.8092671360 | 0.8085671686 | 0.0006999673 | 0.0006999673 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.8177535228 | 0.8180070208 | -0.0002534980 | -0.0002534980 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_auc | 14 | 14 | 0.8101738569 | 0.8122499100 | -0.0020760531 | -0.0020760531 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | macro_logloss | 14 | 14 | 0.4749584148 | 0.4743943719 | 0.0005640429 | -0.0005640429 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | natural_logloss | 14 | 14 | 0.4746770694 | 0.4736438712 | 0.0010331982 | -0.0010331982 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_0 | 14 | 14 | 0.8033889645 | 0.8040110753 | -0.0006221107 | -0.0006221107 |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | old_organization_factorial | 3101 | task_pooled_auc_1 | 14 | 14 | 0.8160543967 | 0.8179202941 | -0.0018658974 | -0.0018658974 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_auc | 24 | 24 | 0.6991445797 | 0.7006913326 | -0.0015467529 | -0.0015467529 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_logloss | 24 | 24 | 0.4681147677 | 0.4675182790 | 0.0005964888 | -0.0005964888 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | natural_logloss | 24 | 24 | 0.5125094037 | 0.5116681947 | 0.0008412090 | -0.0008412090 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.7622648008 | 0.7628320642 | -0.0005672633 | -0.0005672633 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.7713141656 | 0.7718434869 | -0.0005293213 | -0.0005293213 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_auc | 24 | 24 | 0.7067485241 | 0.7078658602 | -0.0011173361 | -0.0011173361 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_logloss | 24 | 24 | 0.4661572987 | 0.4649288578 | 0.0012284409 | -0.0012284409 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | natural_logloss | 24 | 24 | 0.5121440053 | 0.5112425978 | 0.0009014075 | -0.0009014075 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.7628462823 | 0.7637096168 | -0.0008633346 | -0.0008633346 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.7722256686 | 0.7733703144 | -0.0011446458 | -0.0011446458 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_auc | 15 | 15 | 0.6961439305 | 0.6974343794 | -0.0012904489 | -0.0012904489 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_logloss | 15 | 15 | 0.4662605703 | 0.4659627053 | 0.0002978650 | -0.0002978650 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | natural_logloss | 15 | 15 | 0.5108651028 | 0.5115421636 | -0.0006770608 | 0.0006770608 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 15 | 15 | 0.7617359379 | 0.7612763873 | 0.0004595506 | 0.0004595506 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 15 | 15 | 0.7703082961 | 0.7699818392 | 0.0003264569 | 0.0003264569 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_auc | 12 | 18 | 0.6972369670 | 0.7065395372 | -0.0093025702 | -0.0093025702 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_logloss | 12 | 18 | 0.4669986035 | 0.4626483503 | 0.0043502532 | -0.0043502532 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | natural_logloss | 12 | 18 | 0.5136903789 | 0.5103541353 | 0.0033362436 | -0.0033362436 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 12 | 18 | 0.7575827983 | 0.7633182577 | -0.0057354594 | -0.0057354594 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 12 | 18 | 0.7663001936 | 0.7719047312 | -0.0056045376 | -0.0056045376 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_auc | 24 | 24 | 0.8157819901 | 0.8159827144 | -0.0002007243 | -0.0002007243 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_logloss | 24 | 24 | 0.4798452455 | 0.4784185460 | 0.0014266995 | -0.0014266995 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | natural_logloss | 24 | 24 | 0.4783539871 | 0.4781601904 | 0.0001937968 | -0.0001937968 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.8092671360 | 0.8092114360 | 0.0000556999 | 0.0000556999 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.8177535228 | 0.8167140660 | 0.0010394568 | 0.0010394568 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_auc | 24 | 24 | 0.8147597982 | 0.8154178506 | -0.0006580524 | -0.0006580524 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_logloss | 24 | 24 | 0.4808442261 | 0.4848695634 | -0.0040253373 | 0.0040253373 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | natural_logloss | 24 | 24 | 0.4806190507 | 0.4839003214 | -0.0032812707 | 0.0032812707 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.8085671686 | 0.8100591417 | -0.0014919730 | -0.0014919730 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.8180070208 | 0.8180846225 | -0.0000776018 | -0.0000776018 |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_auc | 14 | 17 | 0.8101738569 | 0.8130434475 | -0.0028695906 | -0.0028695906 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | macro_logloss | 14 | 17 | 0.4749584148 | 0.4733570401 | 0.0016013747 | -0.0016013747 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | natural_logloss | 14 | 17 | 0.4746770694 | 0.4732327579 | 0.0014443115 | -0.0014443115 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 14 | 17 | 0.8033889645 | 0.8072909807 | -0.0039020162 | -0.0039020162 |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 14 | 17 | 0.8160543967 | 0.8170548918 | -0.0010004951 | -0.0010004951 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_auc | 14 | 14 | 0.8122499100 | 0.8121794029 | 0.0000705072 | 0.0000705072 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | macro_logloss | 14 | 14 | 0.4743943719 | 0.4741485460 | 0.0002458259 | -0.0002458259 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | natural_logloss | 14 | 14 | 0.4736438712 | 0.4731846831 | 0.0004591881 | -0.0004591881 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_0 | 14 | 14 | 0.8040110753 | 0.8048777687 | -0.0008666934 | -0.0008666934 |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | old_aux_factorial | 3101 | task_pooled_auc_1 | 14 | 14 | 0.8179202941 | 0.8181055371 | -0.0001852430 | -0.0001852430 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_auc | 24 | 24 | 0.7006913326 | 0.7078658602 | -0.0071745276 | -0.0071745276 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_logloss | 24 | 24 | 0.4675182790 | 0.4649288578 | 0.0025894212 | -0.0025894212 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | natural_logloss | 24 | 24 | 0.5116681947 | 0.5112425978 | 0.0004255970 | -0.0004255970 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.7628320642 | 0.7637096168 | -0.0008775527 | -0.0008775527 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.7718434869 | 0.7733703144 | -0.0015268275 | -0.0015268275 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_auc | 15 | 18 | 0.6974343794 | 0.7065395372 | -0.0091051578 | -0.0091051578 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_logloss | 15 | 18 | 0.4659627053 | 0.4626483503 | 0.0033143549 | -0.0033143549 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | natural_logloss | 15 | 18 | 0.5115421636 | 0.5103541353 | 0.0011880283 | -0.0011880283 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_0 | 15 | 18 | 0.7612763873 | 0.7633182577 | -0.0020418704 | -0.0020418704 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_1 | 15 | 18 | 0.7699818392 | 0.7719047312 | -0.0019228919 | -0.0019228919 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_auc | 24 | 24 | 0.8159827144 | 0.8154178506 | 0.0005648639 | 0.0005648639 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_logloss | 24 | 24 | 0.4784185460 | 0.4848695634 | -0.0064510173 | 0.0064510173 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | natural_logloss | 24 | 24 | 0.4781601904 | 0.4839003214 | -0.0057401310 | 0.0057401310 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_0 | 24 | 24 | 0.8092114360 | 0.8100591417 | -0.0008477056 | -0.0008477056 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_1 | 24 | 24 | 0.8167140660 | 0.8180846225 | -0.0013705565 | -0.0013705565 |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | scope | seed | metric | candidate epoch | reference epoch | candidate value | reference value | paired delta | favorable gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_auc | 17 | 14 | 0.8130434475 | 0.8121794029 | 0.0008640447 | 0.0008640447 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | macro_logloss | 17 | 14 | 0.4733570401 | 0.4741485460 | -0.0007915059 | 0.0007915059 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | natural_logloss | 17 | 14 | 0.4732327579 | 0.4731846831 | 0.0000480748 | -0.0000480748 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_0 | 17 | 14 | 0.8072909807 | 0.8048777687 | 0.0024132120 | 0.0024132120 |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | old_organization_factorial | 3101 | task_pooled_auc_1 | 17 | 14 | 0.8170548918 | 0.8181055371 | -0.0010506453 | -0.0010506453 |


## 二乘二seed统计

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_auc | 1 | [3101] | [-0.0076039444] | -0.0076039444 | NaN | -0.0076039444 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_logloss | 1 | [3101] | [0.0019574691] | 0.0019574691 | NaN | 0.0019574691 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | natural_logloss | 1 | [3101] | [0.0003653984] | 0.0003653984 | NaN | 0.0003653984 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_0 | 1 | [3101] | [-0.0005814814] | -0.0005814814 | NaN | -0.0005814814 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_1 | 1 | [3101] | [-0.0009115031] | -0.0009115031 | NaN | -0.0009115031 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=kuairand; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_auc | 1 | [3101] | [-0.0010930365] | -0.0010930365 | NaN | -0.0010930365 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_logloss | 1 | [3101] | [-0.0007380332] | -0.0007380332 | NaN | -0.0007380332 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | natural_logloss | 1 | [3101] | [-0.0028252761] | -0.0028252761 | NaN | -0.0028252761 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_0 | 1 | [3101] | [0.0041531396] | 0.0041531396 | NaN | 0.0041531396 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_1 | 1 | [3101] | [0.0040081025] | 0.0040081025 | NaN | 0.0040081025 | 1 | 0 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_auc | 1 | [3101] | [0.0010221920] | 0.0010221920 | NaN | 0.0010221920 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_logloss | 1 | [3101] | [-0.0009989805] | -0.0009989805 | NaN | -0.0009989805 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | natural_logloss | 1 | [3101] | [-0.0022650636] | -0.0022650636 | NaN | -0.0022650636 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_0 | 1 | [3101] | [0.0006999673] | 0.0006999673 | NaN | 0.0006999673 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_1 | 1 | [3101] | [-0.0002534980] | -0.0002534980 | NaN | -0.0002534980 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=aux-baselines; dataset=movielens; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_auc | 1 | [3101] | [-0.0020760531] | -0.0020760531 | NaN | -0.0020760531 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | macro_logloss | 1 | [3101] | [0.0005640429] | 0.0005640429 | NaN | 0.0005640429 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | natural_logloss | 1 | [3101] | [0.0010331982] | 0.0010331982 | NaN | 0.0010331982 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_0 | 1 | [3101] | [-0.0006221107] | -0.0006221107 | NaN | -0.0006221107 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_K1_DIRECT7_AUX0 | task_pooled_auc_1 | 1 | [3101] | [-0.0018658974] | -0.0018658974 | NaN | -0.0018658974 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_auc | 1 | [3101] | [-0.0015467529] | -0.0015467529 | NaN | -0.0015467529 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_logloss | 1 | [3101] | [0.0005964888] | 0.0005964888 | NaN | 0.0005964888 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | natural_logloss | 1 | [3101] | [0.0008412090] | 0.0008412090 | NaN | 0.0008412090 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0005672633] | -0.0005672633 | NaN | -0.0005672633 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0005293213] | -0.0005293213 | NaN | -0.0005293213 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [-0.0011173361] | -0.0011173361 | NaN | -0.0011173361 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [0.0012284409] | 0.0012284409 | NaN | 0.0012284409 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0009014075] | 0.0009014075 | NaN | 0.0009014075 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0008633346] | -0.0008633346 | NaN | -0.0008633346 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0011446458] | -0.0011446458 | NaN | -0.0011446458 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_auc | 1 | [3101] | [-0.0012904489] | -0.0012904489 | NaN | -0.0012904489 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_logloss | 1 | [3101] | [0.0002978650] | 0.0002978650 | NaN | 0.0002978650 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | natural_logloss | 1 | [3101] | [-0.0006770608] | -0.0006770608 | NaN | -0.0006770608 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_0 | 1 | [3101] | [0.0004595506] | 0.0004595506 | NaN | 0.0004595506 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_1 | 1 | [3101] | [0.0003264569] | 0.0003264569 | NaN | 0.0003264569 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [-0.0093025702] | -0.0093025702 | NaN | -0.0093025702 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [0.0043502532] | 0.0043502532 | NaN | 0.0043502532 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0033362436] | 0.0033362436 | NaN | 0.0033362436 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0057354594] | -0.0057354594 | NaN | -0.0057354594 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0056045376] | -0.0056045376 | NaN | -0.0056045376 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_auc | 1 | [3101] | [-0.0002007243] | -0.0002007243 | NaN | -0.0002007243 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_logloss | 1 | [3101] | [0.0014266995] | 0.0014266995 | NaN | 0.0014266995 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | natural_logloss | 1 | [3101] | [0.0001937968] | 0.0001937968 | NaN | 0.0001937968 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_0 | 1 | [3101] | [0.0000556999] | 0.0000556999 | NaN | 0.0000556999 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_1 | 1 | [3101] | [0.0010394568] | 0.0010394568 | NaN | 0.0010394568 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [-0.0006580524] | -0.0006580524 | NaN | -0.0006580524 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [-0.0040253373] | -0.0040253373 | NaN | -0.0040253373 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [-0.0032812707] | -0.0032812707 | NaN | -0.0032812707 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0014919730] | -0.0014919730 | NaN | -0.0014919730 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0000776018] | -0.0000776018 | NaN | -0.0000776018 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=aux-baselines; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_auc | 1 | [3101] | [-0.0028695906] | -0.0028695906 | NaN | -0.0028695906 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | macro_logloss | 1 | [3101] | [0.0016013747] | 0.0016013747 | NaN | 0.0016013747 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | natural_logloss | 1 | [3101] | [0.0014443115] | 0.0014443115 | NaN | 0.0014443115 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0039020162] | -0.0039020162 | NaN | -0.0039020162 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX0-OLD_DIRECT51_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0010004951] | -0.0010004951 | NaN | -0.0010004951 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [0.0000705072] | 0.0000705072 | NaN | 0.0000705072 | 1 | 0 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [0.0002458259] | 0.0002458259 | NaN | 0.0002458259 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0004591881] | 0.0004591881 | NaN | 0.0004591881 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0008666934] | -0.0008666934 | NaN | -0.0008666934 | 0 | 1 | 0 | NaN | NaN |
| OLD_K1_DIRECT7_AUX0-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0001852430] | -0.0001852430 | NaN | -0.0001852430 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [-0.0071745276] | -0.0071745276 | NaN | -0.0071745276 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [0.0025894212] | 0.0025894212 | NaN | 0.0025894212 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0004255970] | 0.0004255970 | NaN | 0.0004255970 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0008775527] | -0.0008775527 | NaN | -0.0008775527 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0015268275] | -0.0015268275 | NaN | -0.0015268275 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=kuairand; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [-0.0091051578] | -0.0091051578 | NaN | -0.0091051578 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [0.0033143549] | 0.0033143549 | NaN | 0.0033143549 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0011880283] | 0.0011880283 | NaN | 0.0011880283 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0020418704] | -0.0020418704 | NaN | -0.0020418704 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0019228919] | -0.0019228919 | NaN | -0.0019228919 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=endpoint

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [0.0005648639] | 0.0005648639 | NaN | 0.0005648639 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [-0.0064510173] | -0.0064510173 | NaN | -0.0064510173 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [-0.0057401310] | -0.0057401310 | NaN | -0.0057401310 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [-0.0008477056] | -0.0008477056 | NaN | -0.0008477056 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0013705565] | -0.0013705565 | NaN | -0.0013705565 | 0 | 1 | 0 | NaN | NaN |

## analysis_cohort=historical_aux_factorial; candidate_stage=old_complete_reference; reference_stage=old_complete_reference; dataset=movielens; view=selected

| candidate - reference | metric | n seeds | seeds | paired deltas in seed order | mean delta | sample SD | median delta | favorable count | unfavorable count | tie count | paired t95% low | paired t95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_auc | 1 | [3101] | [0.0008640447] | 0.0008640447 | NaN | 0.0008640447 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | macro_logloss | 1 | [3101] | [-0.0007915059] | -0.0007915059 | NaN | -0.0007915059 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | natural_logloss | 1 | [3101] | [0.0000480748] | 0.0000480748 | NaN | 0.0000480748 | 0 | 1 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_0 | 1 | [3101] | [0.0024132120] | 0.0024132120 | NaN | 0.0024132120 | 1 | 0 | 0 | NaN | NaN |
| OLD_DIRECT51_AUX01-OLD_K1_DIRECT7_AUX01 | task_pooled_auc_1 | 1 | [3101] | [-0.0010506453] | -0.0010506453 | NaN | -0.0010506453 | 0 | 1 | 0 | NaN | NaN |

