# P5 first complete MovieLens primary reading

All4seeds x100epochs completed for finite individual-source regret, ordinary
full routing and specified train-meta utility routing. KuaiRand continues; test
remains deferred. No claim to reproduce all validation-allocation methods.

|Method|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Fixed|.730898|.550470|.739844|
|Ordinary full routing|.730463|.549649|.738824|
|Train-meta utility|.730603|.550316|.739158|
|Finite regret|.730719|.551931|.739312|

Regret-minus-full selected AUC+.000256 in3/4, interval[-.000805,.001316],
loss+.002282 worse in3/4, interval[-.002796,.007360]. Versus utility AUC+.000116
in2/4, interval[-.000685,.000916], loss+.001615 with mixed signs. Versus fixed
AUC-.000179 in2/4, weighted AUC-.000532 in3/4 and loss+.001461 worse in3/4.
Evidence of an independent selected benefit is WEAK.

Final-epoch regret-minus-fixed AUC+.004801 in4/4, interval[-.002559,.012161],
but loss+.010232 worse in4/4, interval[-.005264,.025728]. Versus utility, final
AUC+.006165 in4/4 and loss-.040209 in4/4, intervals still include0. This stage
difference is consistent with optimization/regularization effects; it is not a
uniform generalization advantage or proof that finite credit is more faithful.

Preplanned alternative now launches8 full runs: leave-one-source-out conditional
mixture credit, both native datasets x4seeds. It compares finite removal costs
through the same tower rather than competing individual expert outputs. Ordinary
full routing/utility and the individual-source arm isolate this change. Primary
self-review in ALTERNATIVE_REVIEW passed its sign and actual prediction path;
all network weights train and all budgets complete regardless of early scores.

Source permutation covariance alone cannot establish an MDMT contribution;
finite counterfactual routing/regret have direct primary-source coverage.
Competing explanations include near-uniform regularization targets, router
capacity, auxiliary KL and training-stage effects. If a repeatable advantage
remains on KuaiRand/coalition with near-uniform weights, a fixed uniform-mixture
control is the bounded discriminator; it is not launched from this null selected
MovieLens result alone. Endpoint decision comparisons cannot replace training.
Per-cell size/difficulty, both datasets, routing behavior/cost and frozen test
will determine scope; the broad function-credit question is not rejected here.

For completed MovieLens checkpoints, regret routing source entropy is.805296
selected/.951829 final, versus ordinary full .762772/.745510 (maximum log3).
Mean source-weight distance from uniform decreases from.250203 to.172964.
This is consistent with auxiliary distillation encouraging broader mixtures,
not by itself proof of correct expert attribution. Selected regret operational
training time averages216.0s versus177.4s ordinary full and186.2s utility, with
1142.5MiB peak reserved versus916MiB. These concurrent costs are descriptive;
no isolated speed claim. KuaiRand/coalition receipts and final tests remain pending.

## Complete KuaiRand primary reading and matched fixed-target discriminator

All4seeds x50epochs completed. Selected regret AUC .676326/loss .443091/weighted
AUC .701129, versus full .675159/.444122/.702412 and utility
.668062/.441074/.697671. Regret-minus-full macro AUC+.001167 in4/4, interval
[-.001607,.003940], but weighted AUC-.001284 with2/4 positive signs; mean loss
-.001031 is driven by one seed (3/4 worsen). Versus utility AUC+.008264 in4/4,
weighted+.003457 in4/4, interval[.000918,.005997], but loss+.002017 worse3/4.
Versus fixed AUC+.003426 in3/4, weighted-.001167 in3/4.

Task1 accounts for the regret-minus-full macro advantage (+.003895,4/4),
while task0 decreases-.001562 in3/4. Final regret-minus-full AUC instead
-.005462 in0/4 positive seeds, interval[-.011192,.000269]. A selected macro
advantage is not a stable all-task/all-stage improvement.

Selected source entropy1.093691 is near log3=1.098612, and mean weight distance
from uniform is.018610; full routing distance.291492. The repeated macro sign
and near-uniform behavior trigger ADDENDUM_12:8 full trainings of entropy_router,
same architecture/init/auxiliary coefficient but fixed-uniform distillation
target. This is better matched than static forced-uniform weights and explicitly
conventional regularization. It tests whether data-dependent finite credit adds
value beyond uniformization, not whether another lambda can produce a win.
Leave-one-out and fixed-target arms both complete before final conclusions/test.

## Complete MovieLens coalition and fixed-target controls

Selected coalition AUC/loss/weighted AUC .730765/.550121/.739739; fixed-target
.730579/.549624/.739641. Coalition-minus-regret AUC+.000047 in2/4, loss-.001810
in3/4; coalition-minus-entropy AUC+.000186 in3/4, interval[-.000785,.001157],
loss+.000497 mixed. Regret-minus-entropy AUC+.000139 in2/4, interval
[-.001495,.001774], loss+.002307 worse3/4 and weighted AUC-.000329 mixed.
Coalition-minus-full AUC+.000302 in4/4, interval[-.000116,.000720], but
coalition-minus-fixed AUC-.000133 in2/4. Selected independent value remains WEAK.

Final coalition-minus-full loss-.031848 in4/4, interval[-.059817,-.003879],
but versus fixed loss+.014902 worse3/4 and AUC+.002358 mixed. Regret-minus-
entropy final AUC+.006421 in4/4, interval[-.001025,.013867], with loss-.003240
mixed. Thus uniform target explains much of selected behavior without explaining
every late ranking difference. KR final regret-minus-full AUC is negative4/4;
there is a stage/dataset contradiction, not a universal credit benefit. The
bounded same-checkpoint diagnostic in ADDENDUM_12 examines actual finite versus
local-coordinate choices on disjoint train-meta halves; it does not replace the
32 P5-dedicated full trainings plus8 shared full-router controls or identify a causal business policy.

The auxiliary KL averages6cells equally whereas the main risk averages rows.
Its relative strength can therefore differ by domain frequency independently of
data-dependent credit. KR regret-minus-full task1 gains concentrate in medium
domain0(+.013161,3/4), versus largest domain1(-.002295,2/4) and smallest
domain2(+.000820,3/4); all intervals cross0. Task0 falls in all domain means
(-.002973/-.001709/-.000005). A mean macro/weighted trade-off exists, but a
repeatable small-size law does not; smallest gains are not largest and task
identity/auxiliary geometry compete with size.

Operational ML times coalition148.6s, entropy125.3s, regret216.0s, full177.4s;
peak reserved1328.5/916/1142.5/916MiB. Cohorts differ in GPU contention, so no
claim that the extra coalition computations are intrinsically faster. Exact
finite towers/probe count and peak memory are more interpretable than those
wall ratios. KR coalition/entropy cohorts and frozen test remain pending.

## Conditional finite decisions really differ, without proving trained benefit

The first bounded probe covers384cell records (ML all5routing methods, KR the
3complete primary methods; remaining KR controls will be added after training).
Each group has4seeds x6correlated cells. For regret, finite/local coordinate
preferences agree12/24 ML selected and11/24 KR selected; finite-choice A-to-B
replication is8/24 and10/24. Early sampled finite-target entropies1.0893/1.0818
are near log3. These are uncertain small-cost differences, not24 independent
policy trials. They support a near-uniform explanation of early auxiliary targets.

At final regret states, finite versus coordinate-preferred SINGLE-SOURCE actions
on B differ in mean risk by-.04353 ML/-.07714 KR; disagreement is not merely
about expert names. But comparison with the CURRENT MIXTURE is essential:
A-chosen finite source minus mixture B risk is-.126877 ML (4/4seed improvements)
yet+.027892 KR (4/4worsen). Even the B-oracle individual source remains worse
than the KR current mixture on average. Being better than a local preference
among singleton actions does not mean replacing a nonlinear mixture is beneficial.
The local preference itself is not utility_router's complete small-gradient step.

Selected ML finite-minus-mixture mean-.004776 is favorable in only2/4seeds;
full-router selected mean-.001843 is favorable in only1/4. The large late signal
coexists with poor selected replication and whole-network overfitting. Similar
finite/local gaps already occur in ordinary full routing, so they are not unique
to training the proposed credit loss. The coalition arm already tests conditional
removal credit as the distinct mixture-aware alternative.

Evidence MEDIUM for conditional finite/local decision differences on these fixed
checkpoints, WEAK for a unique selected endpoint benefit. A new late model-selection
controller would be a distinct question, not implied effective by this oracle
probe; it would require same-information whole-training controls and originality
beyond ordinary validation selection. No additional ad-hoc controller or KL grid
is accepted from these mixed selected results. Complete KR alternatives, frozen
test and final80-checkpoint diagnostic still determine the scoped disposition.

## Complete KuaiRand mixture-aware leave-one-source-out alternative

Four seeds complete50epochs. Coalition selected AUC/loss/weighted AUC
.673052/.446467/.700971. Relative to fixed AUC+.000152 mixed2/2, loss
+.002466 worse4/4, weighted AUC-.001326 in3/4 negative seeds. Relative
to full router AUC-.002108 in3/4 negative seeds and loss+.002345 worse3/4.
Against train-meta utility AUC+.004990 and weighted AUC+.003299 improve4/4,
but loss+.005393 worsens4/4, interval[.001874,.008911]. Outperforming one
utility comparator on ranking is insufficient when fixed/full are stronger.

Final coalition-minus-fixed macro AUC+.000634 and weighted+.000321 are both
mixed2/2; final loss-.000187 mean is driven by one improvement(3/4 worsen).
Coalition-minus-full final AUC-.000309 in3/4 negative seeds. A mixture-aware
finite contribution does not resolve the general endpoint utility problem.
The remaining KR fixed-uniform KL control and full credit diagnostics complete
before freeze; no ad-hoc meta-controller or unlimited auxiliary-loss search is
justified by this selected/later ranking/calibration pattern.

## Complete KuaiRand uniform-target strong control and finite-action record

All4 fixed-uniform KL runs finish50epochs. Selected AUC/loss/weighted AUC
.675933/.444693/.701452. Regret-minus-uniform selected macro AUC+.000394
but only1/4 positive seed, interval[-.011451,.012238]; weighted AUC-.000323
in3/4 negative seeds. Loss mean-.001602 is driven by one improvement(3/4
worsen). Coalition-minus-uniform selected AUC-.002881 in3/4 negative seeds
and loss+.001773 worsens3/4. Thus the apparent regret-versus-full macro gain
is not a stable independent finite-credit gain beyond ordinary KL shrinkage.
KR final regret-minus-uniform AUC+.000950 mixed2/2; ML final difference is
+.006421 in4/4 but selected effects are tiny/mixed and fixed-model loss is
better4/4. No shared stage-invariant ranking/calibration advantage emerges.

The complete80-checkpoint/480cell two-half credit measurement preserves every
finite cost, choice, local derivative, same-tower mixture risk and sample hash.
At selected regret states finite/local preferences agree12/24 ML and11/24 KR;
finite A choice replicates on B only8/24 and10/24 respectively. These24cells
share fixed populations/seeds and are not24 independent trials. Target entropy
1.0893/1.0818 is near log(3), supporting the uniform-distillation explanation.

At final regret states A-chosen source beats the local-preferred SOURCE on B
by loss-.043529 ML/-.077144 KR. Yet against the CURRENT MIXTURE it changes
loss-.126877 ML (4/4 improve) and+.027892 KR (4/4 worsen); even the B-oracle
one-hot source loses to KR's mixture on average. The same large ML late gap is
already present under full_mix(-.152726), so it is not unique proposed-method
credit. Selected ML regret finite-versus-mixture mean-.004776 improves only2/4
seed means, whereas KR+.003935 improves1/4. A different discrete preference
is therefore real conditional evidence (MEDIUM), not proof of a better trained
controller or of superiority to the actual utility optimizer.

Pre-test disposition: complete32 exclusive+8 shared-control networks; stop the
tested finite/coalition/utility variants, retaining broader mixture-aware credit
as open. WEAK evidence of independent algorithm utility/originality. No ad-hoc
meta-controller or new loss grid is scientifically required by late one-hot
oracle gaps with contradictory selected/current-mixture evidence. Final test
can update consistency/scope, never select a replacement or trigger retuning.

## 冻结测试与最终范围判定（全部预算完成后）

选中测试遗憾减均匀KL：KR AUC-.002480、2/4，ML-.000197、2/4；对应损失+.000418/+.001291，均混合。ML遗憾减普通完整路由选中AUC-.000908四种子负。留一源相对完整路由KR选中AUC+.001371、2/4、损失+.001037；ML AUC-.001015、3/4负、损失+.001374。均不支持稳定选中优势。

最终遗憾减均匀KL测试AUC KR+.004386、4/4，ML+.007960、4/4（区间[.001677,.014243]）；但KR相对完整路由最终AUC仅+.000773、2/4，ML相对固定最终AUC+.007590四种子正却损失+.008073四种子差。不能把对弱后期参照的排名差等同统一校准/泛化收益。此类后期差异为MEDIUM条件性证据，独立算法价值WEAK。

480条同tower、两半集信用记录已完整。单源有限选择优于局部坐标所偏好的另一单源，不代表优于实际非线性混合：最终遗憾样本有限选择相对当前混合风险ML-.126877四种子改善、KR+.027892四种子恶化。ML同类晚期缺口在普通full_mix已存在。有限动作确有决策差异（MEDIUM），但不是对完整utility更新或独立训练单专家的比较。32专属+8共享对照全完成；停止当前遗憾/留一源变体，广义混合函数信用仍开放。

完整数值见 [测试表](test_readout/NUMERIC_TABLES.md)、[分任务/分cell表](test_readout/PER_TASK_CELL_TABLES.md) 和 [配对cell数据](test_readout/paired_cells.csv)。测试后没有修改方法或选模规则。
