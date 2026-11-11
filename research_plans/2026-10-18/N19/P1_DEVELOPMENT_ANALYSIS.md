# P1 complete primary development reading; extension in progress

All7methods x4seeds completed100epochs from scratch. Test remains deferred.
The data is explicitly synthetic in Z/task1 and uses shifted validation for
selection; these are not native business stability claims.

|Method|Selected macro AUC|Selected macro LogLoss|Final macro AUC|Final macro LogLoss|
|---|---:|---:|---:|---:|
|ERM fixed|.579645|.690772|.513464|1.816634|
|Learned fusion/Sample|.580443|.690772|.515023|1.963615|
|Global flip|.571970|1.007821|.562529|1.553518|
|Task0 V-REx|.578897|.811494|.522398|1.871074|
|Task conditional mask|.761993|.557888|.707656|1.155917|
|Task consistency|.707460|.644451|.680066|1.168021|
|Task worst-case DRO|.764532|.555418|.712865|1.036913|

**Stable phenomena — STRONG within this controlled population.** Compatible
masking and DRO preserve task1 while improving the unstable task0 compared with
ERM. DRO-minus-ERM selected macro AUC +.184887,4/4 signs, descriptive95% t
interval[.130113,.239662]; LogLoss -.135354,4/4 improvements. Global flip raises
task0 selected AUC to.698213 but degrades task1 to.445728, versus DRO
.730471/.798594. This demonstrates a cost of the specified incompatible rule,
not failure of all global representation learning or of SAF.

**Independent method gain — WEAK.** DRO-minus-known-mask selected macro
+ .002540,4/4 signs, interval[-.001972,.007051]. Train-weighted +.000210,2/4 signs,
interval[-.002987,.003406]; task1 -.000528,3/4 negative signs. Soft consistency
falls well short of masking. The large ERM gap is therefore mostly explained by
the known intervention rule; ordinary masking captures nearly all selected gain.
Ordinary counterfactual consistency/worst-case learning also have prior coverage.

**Size, difficulty and trade-off.** DRO-minus-mask task0 differences by domain
are +.005114(N32775),+.012235(N2850),-.000526(N142500). Smallest-domain gain has
3/4 positive seeds but a wide interval[-.007232,.031702]; largest is2/4. Task1
domain0 AUC declines in4/4 seeds(-.001141), even though it is not the largest
domain. Thus a mean small-task0 gain/large-task0 decline pattern exists, but its
seed stability is weak and cannot establish domain size as the main cause.
Representation reuse, class/shift structure and checkpoint timing compete.

**Stage contradiction.** All methods degrade substantially by epoch100. DRO's
final loss advantage over masking is larger(-.119004,4/4), while selected AUC
advantage is small and uncertain. Do not substitute final loss for selected AUC
or claim training-stage-independent cell protection. Selection uses epochs9–11
for mask/DRO, versus epochs1–2 for most ERM arms; full budgets still completed.

**Best next discriminator actually launched.** ADDENDUM_07_P1_BACKBONE runs the
same fixed/mask/DRO comparison on MMoE, four seeds/full100epochs. It tests whether
the small residual DRO gain survives a different shared representation, without
changing nuisance, data, objective or checkpoint rule. No penalty/shift grid.
Final direction judgment waits for this extension and frozen test readout.

Raw evidence: development_readout/{runs,method_means,paired_comparisons,paired_cells,resources}.csv,
all28run receipts/trajectories, and MATHEMATICAL_SCOPE/PROTOCOL_LIMITS. Concurrent
operational time varies by device load; masks and DRO both require two forwards.

## Completed MMoE falsification (12 additional full runs)

Selected macro AUC mask .762775, DRO .765055; delta+.002280 in4/4 seeds,
interval[-.000479,.005038]. Train-weighted delta+.002537 in4/4,
interval[.001041,.004032]. Task0 delta+.006423 in4/4,
interval[.001767,.011079], whereas task1 delta-.001864 in4/4. Selected LogLoss
is WORSE by+.010959 in3/4, interval[-.019395,.041312]. Final loss instead
improves-.113479 in4/4, matching the stage dependence seen with M3oE.

The small task0 effect survives a different shared representation, weakening a
STAR-only explanation. It still trades stable-task performance and calibration
against unstable-task discrimination. Evidence **MEDIUM** for this residual
controlled trade-off across two architectures, **WEAK** for a uniform method
advantage or independent contribution. Known-rule masking captures most of the
large robustness benefit; ordinary DRO's principle has prior coverage. No further
penalty/shift grid is justified for the current novelty claim. The40 full P1
trainings are complete; final scope and disposition wait for frozen test readout.

## 冻结测试与最终范围判定（全部预算完成后）

完整测试支持任务兼容目标相对ERM的受控效果：DRO选中宏AUC差+.188437，4/4，区间[.135042,.241832]；LogLoss-.130544，4/4改善。相对已知规则掩蔽，AUC+.004101，4/4，但区间[-.000682,.008885]，加权仅+.000341、2/4。任务0+.008015（4/4），任务1+.000188（3/4）；因此不能沿用开发集任务1全负的说法。MMoE相对掩蔽AUC+.001656仅2/4，任务0+.007319（4/4），任务1-.004008（3/4下降），选中损失+.015694四种子均差。最终epoch两骨干损失相对掩蔽改善，但不等于选中策略的统一优势。

任务0三个域相对掩蔽AUC+.008665/+.016761/-.001380；前两域4/4正，最大域2/4负，各区间仍含0。已知Z和任务规则、偏移验证集可用是关键条件。STRONG：受控兼容性相对ERM；MEDIUM：跨骨干任务/阶段权衡；WEAK：超越强掩蔽的独立算法价值。停止当前CF/DRO变体，真实未知干预规则仍开放。

完整数值见 [测试表](test_readout/NUMERIC_TABLES.md)、[分任务/分cell表](test_readout/PER_TASK_CELL_TABLES.md) 和 [配对cell数据](test_readout/paired_cells.csv)。测试后没有修改方法或选模规则。
