# P3 first primary development reading; alternatives launched

MovieLens uniform/protected each completed4seeds x100epochs. KuaiRand protected
is still running; partial means are not used for final conclusions. Test deferred.

MovieLens selected macro AUC: fixed .730898, uniform .729633, protected .729633.
Protected-minus-uniform AUC=-.0000001535, descriptive95% paired interval
[-.000022,.000022]; LogLoss=+.00002878, interval[-.000092,.000149]. Selected
checkpoints do not support an allocation-specific gain (WEAK evidence of benefit).
Final AUC difference +.002209 in3/4 seeds, interval[-.002344,.006763], and loss
-.014284 in3/4 are directionally different but uncertain. Training stage matters.

Protected ranks by seed are [6,16,26], [9,16,23], [6,18,24], [7,14,27]. Main
domain sizes are [32775,2850,142500]. Largest domain receives more capacity,
but the medium domain receives less than the smallest; this is not a monotone
size allocator. Scores, baseline risk and optimization are competing explanations.

Both preplanned alternatives now train on both datasets with all4seeds:
shared domain dictionary and task-conditioned shared dictionary. Slots are
generated BEFORE nonlinear STAR, distinct from a repair head; task conditioning
doubles STAR/expert paths. Storage matching is not runtime matching. This tests
shared factors and task codes after the first allocation mechanism showed no
stable selected benefit. Full KuaiRand, cell metrics, cost and test remain pending.

## Bounded checkpoint sensitivity measurement

Motivation: MovieLens selected epochs are11/13/11/10 for both rank arms, with
near-zero selected differences despite different allocations. To distinguish
weak current slot influence from successful allocation, n19_rank_diagnostics.py
sets all rank masks to zero at each completed selected/final checkpoint on a
fixed train-meta sample (up to512/domain). It reports per-cell loss and absolute
probability change, with no optimizer steps or test access. This is conditional
sensitivity, not retrained ablation or independent generalization evidence.
Primary self-review, not independent: the intervention changes only rank_masks,
retains skip/biases and uses identical rows for both arms; PASS for this bounded
measurement. No new training algorithm or cancelled formal comparison results.

## Complete KuaiRand primary comparison and active-capacity limitation

Both rank arms have now completed all4seeds x50epochs. Their selected AND final
cell metrics match exactly in every paired seed. Selected macro AUC .666786
versus full fixed .672900; selected loss .445601 versus .444001. This is strong
descriptive evidence of endpoint equivalence in this regime, not proof that
allocation is intrinsically irrelevant. Protected allocations are [16,16,16],
[14,19,15],[9,21,18],[13,19,16]; the effective capacity decision actually differs.

Finite-removal diagnostics found zero prediction changes at epochs5/10,selected
and final for all4seeds in both KuaiRand rank arms and the dense fixed baseline.
Double-precision norms avoid treating float32 norm underflow as exact parameter
zero. Across4seeds, fixed shared norm falls from1.599 at epoch1 to8.21e-5 at5,
2.74e-8 at10,4.28e-15 selected and1.03e-33 final. Uniform shared norm at10 is
3.65e-10 and generated slot norm4.96e-12. The slot path is already numerically
inactive before the epoch10 allocation. No claim of a neural capacity limit or
size-driven failure can be supported by this comparison.

MovieLens selected mean absolute probability sensitivity is .002591 for dense
fixed,.001093 uniform,.000974 protected; final uniform/protected .000429/.000333.
Selected slot removal slightly lowers train-meta loss on average, but this is a
conditional sample intervention, not a retrained ablation or held-out benefit.
Optimization regularization, skip-path dominance and training-update count
compete with the proposed allocation explanation.

ADDENDUM_11 therefore launches20 matched KuaiRand AdamW full trainings across
dense/uniform/protected/domain-dictionary/cell-dictionary. This is one bounded
optimizer discrimination with same decay coefficient, rates and budgets; it
changes all parameter updates and does not isolate slot weight decay causally.
The original Adam results remain fully archived. Final capacity conclusions
wait for this active-capacity check, dictionary results and frozen test.

A diagnostic-script container mismatch when adding the dense ParameterList was
corrected before any new diagnostic table was written; old completed training
and initial rank-only measurements were unaffected. The final script stacks
slot lists without mutating parameter dtypes.

## Complete MovieLens dictionary alternative reading

Both dictionaries completed4seeds x100epochs. Selected cell-dictionary macro
AUC .730472/loss .547607/weighted AUC .738662, domain-dictionary
.730211/.550588/.739164, versus uniform .729633/.551001/.739082 and dense
fixed .730898/.550470/.739844. Cell-minus-domain AUC+.000261 in2/4, interval
[-.003652,.004175], loss-.002981 in3/4, interval[-.006857,.000895].
Cell-minus-uniform AUC+.000840 in3/4, interval[-.000763,.002442], loss-.003393
in4/4, interval[-.007282,.000495]; weighted AUC instead-.000420 in3/4.
There is no stable task-coefficient ranking gain beyond the domain dictionary
or dense reference (WEAK evidence of independent value). The macro loss gain
is a trade-off, not uniform protection.

Final cell-minus-domain macro AUC+.002589 in3/4, but weighted AUC-.003491
in3/4 and loss mean-.001995 is driven by one larger improvement (3/4 worsen).
Selection/stage, task and population weights alter the apparent conclusion.
Finite slot-removal selected mean |probability change| .001405 cell versus
.001355 domain, with retained shared norms6.896/6.383; unlike KR Adam, these
branches are active but modest. Conditional removal slightly lowers train-meta
loss; that is not a retrained no-slot benefit or held-out evidence.

Deployed parameters629559(cell)/629418(domain) fit below uniform629902 versus
dense773470, about18.6% total storage reduction. Cell dictionaries evaluate two
STAR/expert paths. Native KR plus matched AdamW arms still run; no additional
MovieLens optimizer grid is triggered by this mixed, small dictionary effect.

Reporting coverage now explicitly includes dictionary-versus-dense and all
confidence/router strong controls, plus nested-versus-unconstrained P2 likelihood
controls in paired tables. This adds no model/data/selection changes and occurs
before test freeze; it makes already-designed comparisons directly reviewable.

## Complete KuaiRand Adam dictionaries and first matched-optimizer evidence

Both original-Adam dictionaries finished4seeds x50epochs. Selected cell/domain
macro AUC .671825/.671609, loss .446614/.443077, weighted AUC .699370/.699395.
Cell-minus-domain AUC+.000216 in2/4, interval[-.008056,.008488], loss+.003538
mixed. Final cell-minus-domain weighted AUC-.001794 in0/4 positive seeds.
Both slots are numerically inactive at selected/final under this optimizer, so
these means cannot establish a useful learned slot-sharing mechanism.

All4 dense and4uniform AdamW trainings now finish full50epochs. Dense selected
AUC .665257/loss .447922/weighted AUC .697624; uniform
.667099/.449850/.698438. Uniform-minus-dense AUC+.001842 in2/4, interval
[-.018972,.022655], loss+.001927 mixed. No stable practical win follows from
this first matched pair. Other AdamW candidates continue to their full budgets.

Crucially, full4seed finite-removal measurements show the dense AdamW path is
active: selected mean |probability change| .030597, shared norm49.655, versus
exactly zero selected sensitivity under original Adam. Uniform AdamW selected
sensitivity .037650, shared norm58.011. Dense final sensitivity rises to.087515.
The remedy changes parameter activity, but average selected AUC remains below
original Adam's .672900 and final loss rises to.636211 versus .492377.
Activating more capacity is not automatically better regularization/generalization.
The optimizer changes every parameter, so these are regime comparisons rather
than an isolated causal effect of slot decay.

Complete protected/dictionary matched comparisons and their four-seed activity
diagnostics remain pending; no universal allocation failure or optimizer cure is
asserted. Active-factor export and frozen test still follow after all decisions.

## Complete20-network KuaiRand AdamW capacity cohort

Selected cell/domain dictionary AUC .671507/.666411, loss .450340/.449772,
weighted AUC .699650/.697690. Cell-minus-domain selected AUC+.005095 in3/4
with wide interval[-.018263,.028454], loss+.000568 mixed. Final AUC+.020424
and weighted AUC+.006361 both improve4/4, but final loss is mixed2/2. Against
uniform rank final AUC+.008108 improves4/4 while loss+.026607 worsens4/4,
interval[.003907,.049306]. This is a meaningful late ranking/calibration
contradiction, not a uniformly better task dictionary. ADDENDUM_14 therefore
adds exactly8 ML AdamW dictionary transfer runs; no coefficient/rank/decay grid.

Protected-minus-uniform selected macro AUC+.003168 and loss-.005195 improve
4/4, intervals[-.002388,.008724] and[-.012335,.001945]. Weighted AUC instead
-.000969 in3/4 negative seeds. Final protected-minus-uniform AUC-.007372 in0/4
positive seeds; task1-.014192 in0/4, interval[-.022375,-.006009]. ADDENDUM_13
adds12 ML AdamW dense/uniform/protected runs to test transfer of this selected
macro trade-off. Both full matched cohorts remain separate from original Adam.

These two bounded transfers complete the native dataset x optimizer capacity
comparison with72 exclusive networks. Activity is necessary to interpret a
slot allocation, but neither activity nor a positive selected mean establishes
finite-risk protection or independent novelty. The final4seed ML reading, all
checkpoint sensitivities, compact exports and frozen test follow.

## Complete MovieLens AdamW rank-transfer falsifier (12 full networks)

Selected dense/uniform/protected macro AUC .730708/.730109/.730092 and loss
.550455/.551148/.551180. Protected-minus-uniform macro AUC-.00001705 in1/4
positive seeds, interval[-.0000752,.0000411], loss+.0000315 worse3/4; weighted
AUC-.0000154 in0/4 positive seeds. The KR selected allocation effect does not
transfer to this second native dataset/optimizer condition. Against dense both
rank arms have worse selected loss4/4; no independent protection claim follows.

Final protected-minus-uniform macro AUC+.001545 in3/4, interval[-.002168,
.005258], but loss+.004214 worse3/4; task1 AUC+.003410 in4/4. Both rank arms
lose final macro AUC4/4 against dense, while uniform improves final loss4/4
(-.054487, interval[-.095568,-.013406]). Stage/ranking/calibration distinctions
remain material. Evidence is MEDIUM for condition dependence, WEAK for a stable
allocation-specific improvement. Finish the8 dictionary transfers, then close
this complete bounded dataset/optimizer capacity matrix without a tuning grid.

## Complete MovieLens AdamW dictionary transfer and pre-test disposition

All8 additional100epoch networks complete. Selected cell/domain AUC
.730222/.730255, loss .549858/.551907, weighted AUC .738722/.739182.
Cell-minus-domain selected AUC-.000034 mixed2/2, interval[-.002680,.002613],
loss-.002049 mixed2/2 and weighted AUC-.000460 mixed2/2. Final AUC-.001470
and weighted AUC-.001109 are mixed2/2, loss+.002988 worsens1/4; task0 AUC
+.001664 improves4/4 while task1-.004603 is mixed. The repeated KR late macro
ranking effect does not transfer. Versus ML dense final macro AUC-.002143
in3/4 negative seeds while loss-.030068 improves4/4. This is a metric/stage
trade-off, not universal degradation or improvement.

Final finite-removal collection contains2400cell records=80 models x5saved
stages x6cells. Every dataset/optimizer/method group has4seeds. All KR Adam
selected/final paths remain prediction-invariant; AdamW selected sensitivity
cell/domain .035390/.037344, final .114821/.104247. ML AdamW selected
cell/domain sensitivity .002414/.002781 and shared norms14.47/14.46 retain
an active but modest modulation. Conditional removal can lower train-meta
loss; that is not a retrained no-slot generalization effect.

Pre-test disposition: complete72 exclusive networks plus native dense controls,
stop these tested rank/dictionary mechanisms without a further hyperparameter
grid. STRONG descriptive evidence of original-KR numerical path inactivity;
MEDIUM for stage/optimizer/dataset dependence; WEAK for independent allocation
or task-dictionary value. Practical storage compression remains measurable,
with exact active-factor exports after freeze. No causal size conclusion or
finite-risk protection, and no forced independent top-tier novelty claim.

## 冻结测试与最终范围判定（全部预算完成后）

原Adam KR保护/均匀秩的选中和最终测试仍逐种子精确相同；数值失活解释不能丢弃。AdamW下选中保护减均匀AUC+.001630仅2/4，损失-.005745四种子改善，加权AUC-.001345四种子下降。任务1最小域+.005419四种子正、最大域-.002299四种子负，但两个区间均含0，且最终阶段不保持这一取舍。ML AdamW选中差仅+.000055，原Adam仅+.000016，无实际稳定分配收益。

必须保留验证/测试反转：KR AdamW最终保护减均匀，开发AUC-.007372四种子负，而测试+.009646四种子正、区间[.000458,.018835]；测试加权+.002807四种子正。KR AdamW最终任务减域字典，开发+.020424四种子正，而测试-.002723四种子负；ML测试最终-.002376、3/4负。选中任务字典测试AUC KR+.003601、ML+.002174均3/4正，但损失/加权不统一。固定总体上的种子一致性不是跨总体泛化保证。

2400条无训练敏感度记录证实AdamW下路径活跃；64紧凑导出最大误差1.1921e-6。KR参数1614254降至约829千，ML773470降至约630千，分别约48.6%/18.6%存储缩减；任务字典双STAR计算，不能声称相同比例加速。STRONG：指定KR原设置失活；MEDIUM：条件/阶段/评价总体依赖及局部取舍；WEAK：风险保护或独立分配创新。72专属预算及所有迁移完成，停止这些机制的调参扩展。

完整数值见 [测试表](test_readout/NUMERIC_TABLES.md)、[分任务/分cell表](test_readout/PER_TASK_CELL_TABLES.md) 和 [配对cell数据](test_readout/paired_cells.csv)。测试后没有修改方法或选模规则。
