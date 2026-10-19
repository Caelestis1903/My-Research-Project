# P6 complete development reading:32 full trainings plus reused full-label reference

Eight observation/augmentation policies x4seeds completed100epochs. Natural ML
fixed serves the identical-population full-label reference; this reuse is not new
training. Same main rows, objective, initialization, shuffle and marginal pi=.55.
No test read. Observation budget is per visit, not unique annotation acquisition.

|Method|Selected macro AUC|Selected macro LogLoss|Final macro AUC|Final macro LogLoss|
|---|---:|---:|---:|---:|
|Full labels|.730898|.550470|.674069|1.148884|
|Independent|.730816|.548229|.684200|1.029629|
|Positive coupling|.730635|.547632|.682056|1.026975|
|Antithetic|.730510|.548012|.680201|1.049926|
|Adaptive raw alignment|.730923|.547626|.680349|1.051115|
|Independent+CV|.730293|.549585|.683863|1.021542|
|Raw adaptive+CV|.730743|.548897|.682003|1.022997|
|Antithetic+CV|.729985|.549976|.681053|1.014861|
|Residual adaptive+CV|.730588|.549087|.682689|1.009198|

**Repeated phenomena.** Selected performance is close across policies, whereas
all masked variants reduce the full-label model's late overfitting. The latter is
compatible with ordinary stochastic regularization. It is not evidence uniquely
for adaptive joint design: independent masks have higher final macro AUC than
the adaptive variants. Evidence **MEDIUM** for a stage-dependent regularization
pattern on this fixed native population, **WEAK** for a unique new mechanism.

**Strong-control comparisons.** Raw adaptive minus independent selected AUC
+.000107(3/4),95% t interval[-.001983,.002197]. Versus fixed antithetic +.000413
but only1/4 positive signs, showing an outlier-sensitive mean. Raw adaptive+CV
minus independent+CV +.000450(3/4),interval[-.002382,.003282]. CV itself does not
give a stable selected improvement. A tiny positive mean cannot establish value.

**Actual mechanism improvement/falsification completed.** The initial augmented
policy measured ordinary alignment even though augmentation's stochastic field
is a residual. ADDENDUM_09 corrects the design target and adds the missing fixed
antithetic+CV control. Residual minus raw-adaptive+CV selected AUC-.000155 with
3/4 positive signs (one larger negative); versus antithetic+CV +.000603 in2/4,
interval[-.001623,.002829], loss-.000889 with mixed signs. Versus independent+CV
+.000295,interval[-.002004,.002593]. This is not a stable endpoint improvement.
Relative to full labels, selected train-weighted AUC is LOWER in4/4 by-.000868,
interval[-.001623,-.000113], despite a larger favorable final-checkpoint delta.

**What decisions actually changed.** Exact residual-policy high-endpoint row
fractions are.221%,2.279%,1.908%,.556%;97.7%–99.8% of rows use the antithetic
endpoint. Legacy raw copula observed joint rates .10069/.10950/.10834/.10274
also indicate concentration near .1, not exact identity to a fixed policy. Thus
much of the design is explained by a conventional endpoint, with brief adaptive
differences and no established added performance. Corrections preserve the mean
field, as exact four-mask enumeration confirms; this is not a changed objective.

**Cells, size, difficulty.** Residual-minus-antithetic+CV selected task0 AUC deltas
are+.000436(N32775),+.002077(N2850),+.000117(N142500); task1+.000155,+.000944,
-.000113. All cell intervals cross zero. Smallest-domain means look larger but
signs are mixed and baseline quality differs. There is no stable small-domain
improvement/large-domain decline law. Only one fixed three-domain dataset and
two nested native labels are tested; size is not identified as a main cause.

**Cost and scope.** Actual labels/row/visit are1.09996–1.10006 versus2 for full
labels. Residual policy operational time287.96s versus antithetic+CV166.68s and
independent+CV161.20s; roughly1171MiB peak reserved, under shared GPU contention.
These are recorded run costs, not isolated speed ratios. Dense forwards remain;
EMA/probes add work. Repeated native labels could be cached; under the declared
fixed marginal randomization, a label's probability of never being queried after
nine visits is .45^9≈.000757. This is an expectation, not a measured unique-label
counter. No45% real annotation saving is established.

**Competing explanations and stopping scope.** Large batches suppress observation
noise; the shared-representation inner product approximates full-parameter
covariance; early coupling changes and ordinary noise regularization can explain
late behavior. Residual targeting was the bounded discriminator actually run,
and it did not establish a new advantage. Copula/antithetic design and prediction
control variates also have direct prior coverage. Further blind batch/probability
grids do not resolve originality. A future distinct question would require a
meaningful persistent observation/execution cost and a calibrated full-gradient
metric; it cannot be claimed solved or disproved here. Current algorithmic
independent value remains **WEAK**, pending frozen test readout; the broader
sampling/observation direction is not universally rejected.

Raw evidence: all32run receipts, query counts, policy_counts on new arms, complete
trajectories and development_readout/{paired_comparisons,paired_cells,resources}.
All budgets complete; no proxy or single-seed cancellation. Test pending.

## 冻结测试与最终范围判定（全部预算完成后）

选中测试残差自适应相对反相关CV宏AUC+.000896、3/4，区间[-.001859,.003652]；相对独立CV AUC-.000412、3/4负，损失+.002754四种子差。相对完整标签宏AUC仅+.000013、2/4，加权+.000186、2/4。开发阶段相对完整标签加权AUC四种子负的现象没有在测试保持，不能写成普遍人口风险退化。

最终残差法相对完整标签损失-.136729四种子改善，区间[-.193791,-.079667]，加权AUC+.011911四种子正；但相对独立CV最终宏AUC-.003017、3/4负。普通缺标正则是必要竞争解释，自适应不具独立稳定收益。97.7%—99.8%访问选择静态反相关端点；实际约1.1标签/行/访问而非2，计算仍稠密且标签可缓存。MEDIUM：后期正则效应；WEAK：残差自适应独立价值。32预算完成，停止当前copula/CV变体；真实持久观测与全参数协方差问题留作不同条件的研究。

完整数值见 [测试表](test_readout/NUMERIC_TABLES.md)、[分任务/分cell表](test_readout/PER_TASK_CELL_TABLES.md) 和 [配对cell数据](test_readout/paired_cells.csv)。测试后没有修改方法或选模规则。
