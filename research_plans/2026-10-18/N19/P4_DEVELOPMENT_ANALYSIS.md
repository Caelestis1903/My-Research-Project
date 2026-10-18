# P4 first complete MovieLens primary reading

Four seeds x100epochs each completed for fixed/domain/full/interaction. KuaiRand
primary and axis-additive controls continue. Test remains deferred.

|Method|Selected AUC|Selected LogLoss|Train-weighted AUC|
|---|---:|---:|---:|
|Fixed|.730898|.550470|.739844|
|Domain-only|.730529|.549941|.738742|
|Full cell mixing|.730463|.549649|.738824|
|Centered interaction|.730879|.549460|.739332|

Interaction-minus-domain selected AUC+.000351 in4/4, interval[-.000006,.000707],
loss-.000481 in4/4, interval[-.000847,-.000115]. Yet interaction-minus-fixed
AUC-.000019 in2/4, loss-.001009 in3/4, both intervals include0. Full mixing
comparison has AUC+.000416 in3/4, interval[-.000345,.001178]. Evidence of an
independent interaction benefit is WEAK; outperforming one learned router does
not establish improvement beyond the strong fixed model. Against MMOE AUC
+.001796 in4/4, but loss+.000211 with wide uncertainty. PLE also has lower
selected AUC, yet its architecture alone cannot identify the interaction effect.

At the final epoch, interaction-minus-domain AUC=-.006473 in0/4 positive seeds,
interval[-.011788,-.001158]. Interaction-minus-fixed loss=+.040323 in4/4. The
selected/final reversal argues against a stage-invariant sharing mechanism.

Preplanned replicated-confidence alternative now enters8 full runs, both
datasets x4seeds. Its engineering update receipt already passed; all ordinary
weights train. The corresponding confidence-off arm is interaction, with full
cell and additive-axis controls. Sampling noise in two train-meta halves, generic
shrinkage and subspace capacity remain competing explanations. Effective-logit
centering is a constraint, not evidence of utility or causal domain structure.
KuaiRand, actual replicated signs/cost, per-cell size/difficulty and final frozen
test results remain pending; no broad direction closure is declared here.

Checkpoint routing diagnostics use the implemented router projection and actual
saved selected/final coefficients, reporting all18 source weights and effective
logit margins without any data/test forward. Primary self-review (not independent):
PASS; the same fixed prior is used as training, and logit constraints are kept
distinct from probability margins. Source entropy alone is not useful credit.

## Complete KuaiRand primary comparison

Now all4seeds x50epochs are complete. Selected interaction AUC .669914/loss
.441687/train-weighted AUC .698495; fixed .672900/.444001/.702296;
domain-only .674587/.444756/.702568; full .675159/.444122/.702412.
Interaction-minus-fixed AUC-.002986 in0/4, interval[-.006468,.000495], weighted
AUC-.003801 in0/4, interval[-.006411,-.001192]. Versus full AUC-.005246 in0/4
and weighted AUC-.003918 in0/4, interval[-.006905,-.000930]. Loss instead
improves-.002435 against full in4/4, interval[-.005260,.000391]. This is a
ranking/calibration trade-off, not a uniform interaction benefit. Final AUC
remains lower on average versus fixed/domain/full, with mixed seed signs.

Together with MovieLens, the current primary mechanism does not establish a
stable endpoint advantage. The preplanned confidence alternative still completes
all8runs, with axis-additive controls and actual cost/sign diagnostics pending.
Do not infer a size law or interaction impossibility from one centered subspace.

Within selected KR cells, interaction-minus-fixed AUC changes are
[-.005102,+.000494] for domain0(N164143),[-.004233,-.003785] for the largest
domain1(N1656704),[-.006608,+.001316] for the smallest domain2(N84980). The
large-domain task0 decrease repeats4/4 with interval[-.007654,-.000813], but
the small-domain task0 mean also declines. There is no stable across-task
small-domain improvement/large-domain decline law. Baseline cell AUCs differ
(.652800/.642530,.696916/.722716,.663296/.659142), so size is not difficulty.

Actual interaction logits' maximum destination margins are around1e-9 for KR
and5e-10 for ML at selected states. Mean absolute source-probability departures
from the fixed prior are only.004292(KR)/.001520(ML), versus full mixing
.027309/.003173. The constrained source law changes very little early; limited
movement, subspace restriction and calibration can explain near-baseline scores.
This is measured behavior, not proof that a larger interaction would help.

A pre-result mathematical interpretation check also shows positive confidence
does not change the feasible interaction-logit family: PCP is positive definite
on range(P) for c in[.25,1]. See MATHEMATICAL_SCOPE. Its scientific distinction
is data-dependent optimization geometry/regularization; it cannot be described
as certified hard selection of reliable interactions. If it improves endpoints,
a matched generic-shrink explanation must be examined before attributing gains
to replicated risk signs. No confidence result has been used for this identity.

## Complete additive-axis strong control (8 formal runs)

Selected axis-additive AUC/loss/weighted AUC: KR .677149/.439387/.700489;
ML .730829/.550432/.739454. Interaction-minus-axis KR macro AUC-.007235 in0/4,
interval[-.016975,.002505], loss+.002300 worse3/4. ML macro AUC+.000050 in4/4,
interval[-.000056,.000157], loss-.000971 with mixed signs. The tiny ML direction
does not offset the KR contradiction or establish practical magnitude.

Axis-minus-fixed KR macro AUC+.004249 but only2/4 signs, weighted AUC-.001807
in3/4 and loss mean-.004614 driven by one seed(3/4 worsen). ML macro AUC-.000069
in2/4. Ordinary additive mixing therefore explains another plausible sharing
family without a robust win. No larger ANOVA interaction should be presumed
necessary from a fixed-gradient tensor; actual trained strong controls adjudicate.
Replicated-confidence complete cohorts and frozen test still remain pending.

## Complete MovieLens replicated-confidence alternative

Selected confidence AUC .731063/loss .550442/weighted AUC .739665. Confidence-
minus-off AUC+.000183 in3/4, interval[-.000164,.000531], loss+.000981 worse3/4.
Versus fixed AUC+.000165 in2/4, interval[-.000658,.000987], weighted AUC-.000179
in2/4 and loss-.000028 mixed. Versus full AUC+.000600 in3/4 but loss+.000793
mixed. No stable practically material selected advantage is established (WEAK).

Final confidence-minus-off AUC+.003812 in4/4, interval[-.003025,.010650], but
loss-.011682 with2/4 improvements; versus fixed final loss+.028641 worse3/4.
Late ranking differences do not demonstrate uniform protection/calibration.

Each seed records all20 scheduled probe updates. Mean sign agreement per seed
.7444/.6389/.6528/.6611; mean median absolute slopes .001720/.001054/.000995/
.001095. Selected confidence mean .9500 (minimum .8575), final mean .7891
(minimum .5912). Coordinates/cells and repeated probes are correlated, so these
are not independent reliability trials or calibrated confidence probabilities.
Recorded probe costs93.4/93.7/74.5/159.5s, mean105.3s extra. Total operational
time256.7s versus confidence-off158.8s, same916MiB peak reserved, under differing
concurrent workloads. These are costs, not an isolated speed ratio.

No further ML shrinkage grid is justified by this mixed tiny selected effect.
If KR shows a substantial repeatable advantage, a matched global-confidence
control remains the falsifier; pending KR must complete regardless.

## Complete KuaiRand replicated-confidence alternative and disposition

All4seeds finish50epochs. Selected AUC/loss/weighted AUC .665568/.446510/
.700113. Confidence-minus-fixed macro AUC-.007332 in0/4 positive seeds,
interval[-.018185,.003521]; weighted AUC-.002183 in0/4; task1 AUC-.010206
in0/4, interval[-.018009,-.002402]. Versus full mixing weighted AUC-.002300
in0/4, interval[-.004570,-.000029]. Relative to confidence-off, selected macro
AUC-.004346 in1/4 and loss+.004823 worse3/4, though weighted AUC+.001618
in3/4. This does not establish a useful selected sharing improvement.

At final epoch confidence-minus-full macro AUC-.012379 in0/4, interval
[-.022556,-.002201], and task1 AUC-.013086 in0/4, interval[-.026061,-.000111].
Versus axis final AUC-.012912 in0/4, interval[-.025741,-.000083]. Training-stage
and task contradictions reinforce the limited scope. The fixed/global-shrink
falsifier is not triggered: the required substantive benefit beyond fixed/full
is absent. Adding a shrinkage grid to repair an unsuccessful heuristic would
change the research question rather than distinguish an observed success.

Evidence: MEDIUM against this tested replicated-confidence mechanism as a
general endpoint improvement; WEAK for independent algorithm value. Positive
PCP confidence preserves the feasible interaction family, and risk-sign
replication is not calibrated reliability. All40 exclusive networks complete;
stop these two tested families, leaving broader interaction-information sharing
open. Frozen test checks consistency and cannot select a replacement algorithm.

KR records10 probes/seed, mean sign agreement .6167/.6556/.6333/.5778 and
extra times82.6/77.9/79.2/53.1s (mean73.2s). Total operational training
time1059.1s versus confidence-off771.1s,1516MiB both. Differing contention
means the total wall-time difference is not attributable solely to probes.

## 冻结测试与最终范围判定（全部预算完成后）

选中测试KR交互减固定宏AUC-.002399（3/4负），训练量加权-.003517四种子负、区间[-.005669,-.001364]；置信门控减固定宏AUC-.000091、2/4，加权-.000694、2/4。ML门控减固定AUC+.000458、3/4，区间[-.000873,.001789]。门控的独立选中收益仍为WEAK；两半集符号一致不构成校准置信度，正PCP保留同一可表达交互族。

不能隐去正面反例：KR最终交互减固定测试AUC+.013137、4/4，区间[.008741,.017533]，加权+.003033、4/4，损失-.006218、4/4改善。相同对比在开发最终阶段均值方向不同，选中阶段也不受益。最终测试域0两任务+.017160/+.032611、最小域+.014696/+.013766（均4/4正），最大域+.000030/+.000556、3/4正。这是MEDIUM的条件性后期收益，不是小域增/大域降，也不能证明交互必要或独立创新。

40专属训练全部完成，冻结前没有实质门控收益触发全局收缩控制。测试新现象不触发在同一测试上的修复/选点；保留更广交互研究及独立新总体、预定部署阶段的复现问题，停止当前门控机制。

完整数值见 [测试表](test_readout/NUMERIC_TABLES.md)、[分任务/分cell表](test_readout/PER_TASK_CELL_TABLES.md) 和 [配对cell数据](test_readout/paired_cells.csv)。测试后没有修改方法或选模规则。
