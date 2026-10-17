# q、学习率与预算响应

状态：IN_PROGRESS。以下调参与diagnostic资源读数已完成；独立确认test、quota及固定ML人口结果尚待全部训练闭合后统一解封。本文件不作最终算法收益判断。

## 完整H100对称调参

KR Sample objective、原fixed_model、B40960、WD1e-5；Adam四点.0003/.001/.003/.01，SGD四点.03/.1/.3/1.0；每个LR×SN/SB×2030/2031，32条全部H100。无短期淘汰、无失败。每条实际4900更新、200613200样本暴露；49更新/epoch，尾40052。每条学习率的分数是完整selection轨迹1..100宏AUC的最早最大值。共同eta对两q×两调参seed平均；own赢家仅按各q×两seed平均；并列较小LR。

| optimizer | horizon | COMMON | SB | SN |
| --- | --- | --- | --- | --- |
| adam | 10 | 0.001 | 0.001 | 0.001 |
| adam | 20 | 0.01 | 0.001 | 0.01 |
| adam | 50 | 0.01 | 0.001 | 0.01 |
| adam | 100 | 0.01 | 0.01 | 0.01 |
| sgd | 10 | 0.3 | 0.3 | 0.3 |
| sgd | 20 | 0.3 | 0.3 | 0.3 |
| sgd | 50 | 1 | 1 | 1 |
| sgd | 100 | 1 | 1 | 1 |

H10/H20/H50仅为预定预算响应描述，不替换真正H100选参。共同与各自H100赢家均Adam .01、SGD1.0，均在预设网格上边界。不能称连续LR最优或更宽网格最优；本轮不据此延展网格。Adam SB的中期赢家仍.001，SN为.01；SGD共同赢家从H10/20的.3变为H50/100的1.0。**WEAK（外推）**：两调参seed上的赢家变化说明阶段/预算不能忽略，但不是所有人口/优化器的一般定律。

完整响应面包含每q/LR/horizon的AUC、LL、J、固定端点、选轮、AULC和真实steps/examples；不是只报赢家。下表为各LR在H100的两调参seed selection最优分数平均，原始seed差见raw_seed_response.csv；两q可以在不同epoch达到最优。不能将这些被选过的selection分数当独立test效应。

| optimizer | lr | horizon | SB | SN | SB_minus_SN |
| --- | --- | --- | --- | --- | --- |
| adam | 0.0003 | 100 | 0.67788429 | 0.69080076 | -0.012916471 |
| adam | 0.001 | 100 | 0.69027189 | 0.69067109 | -0.0003991935 |
| adam | 0.003 | 100 | 0.68066059 | 0.69489367 | -0.014233076 |
| adam | 0.01 | 100 | 0.69335002 | 0.69599746 | -0.0026474424 |
| sgd | 0.03 | 100 | 0.63064476 | 0.62687417 | 0.0037705905 |
| sgd | 0.1 | 100 | 0.63863368 | 0.63750546 | 0.0011282209 |
| sgd | 0.3 | 100 | 0.65130309 | 0.65160704 | -0.00030395059 |
| sgd | 1 | 100 | 0.65837187 | 0.66412861 | -0.0057567377 |

## 独立八优化种子与预算资源

N1正式确认2032–2039、两优化器×两q、32条全部完成且核验。common与own LR完全相同，因此对应两种估计对象复用这32条轨迹，不重复计独立seed。逐epoch纯model及原生E10/E40状态保留。

N1_DIAGNOSTIC_GATE_ANALYSIS.md给H100内selection选轮、diagnostic评价的配对结果：Adam宏SB−SN=−.01059792，t95[−.02285727,+.00166144]；SGD=+.00152046，t95[−.00219955,+.00524047]。宏均跨0；Adam部分cell负向一致，SGD由d0t1触发重要残差。这是预设资源选择，不能当test主结论。完整cell/seed/损失由正式读数保留。

训练J的rho四组均未超过平均>.005，因此H200两优化器均不触发。该结果不是证明已收敛、也不排除更长预算可能有效；不临时放宽近阈值SGD。N2 quota与N3局部研究按原门槛继续，N5-C也按已冻结资源解释执行。

## 待闭合的正式评价

test仍SEALED：最终保留E10/20/50/100、selection选择轮、AULC1..20/100、宏/加权cell/六cell/各task/各domain及LL/J。以八配对训练seed计算t95、exact sign randomization与20k bootstrap；两common-eta selected macro主比较做Holm，其余是次级pointwise读数。最终将加入quota对比、ML本环境共同/own赢家及实际samples/steps/训练阶段；不能用同epoch声称KR与ML同优化进度。

来源：N1_pilot_surface/*、N1_winners_frozen.json、N1_pilot_validation.json、N1_confirmation_validation.json、N1_diagnostic_gates.json。无新训练或额外test访问。
