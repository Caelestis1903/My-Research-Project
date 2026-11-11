# First completed baseline observations — DEVELOPMENT ONLY

Recorded during ongoing full training. These observations do not terminate any
candidate or authorize a final direction-level conclusion. Fixed four-seed arms
are complete on both native populations. Same-objective learned fusion and
literal original protocol are complete on ML; other controls still run.

Selected-checkpoint development macro AUC: fixed ML .730898, same-objective
learned fusion .730795, literal original protocol .711805. Corresponding mean
LogLoss .550470/.550578/.573554. The literal reference changes objective and uses
the source two-Adam overlap/extra architecture update, so its difference cannot
be attributed to learned fusion alone. MMoE ML .729084 AUC/.549250 LogLoss shows
that model ordering can differ by metric even before new mechanisms are tested.

Fixed KR baseline by domain/task, main-train counts after5% exclusion:

|Domain|N|Task0 AUC|Task1 AUC|Task0 LogLoss|Task1 LogLoss|
|---|---:|---:|---:|---:|---:|
|0|164143|.652800|.642530|.322853|.167844|
|1|1656704|.696916|.722716|.503152|.345810|
|2|84980|.663296|.659142|.660247|.664102|

The smallest native domain is not the lowest-AUC domain. LogLoss differences
also reflect label prevalence/entropy, so an absolute loss ordering should not
be equated with intrinsic difficulty. Raw cells now include prevalence, empirical
label entropy and loss skill versus an empirical constant, as descriptive context.
This empirical constant uses evaluation prevalence and is not a deployable trained
baseline or extra training input.

Evidence strength: **STRONG** for these repeated descriptive fixed-population
orderings, **WEAK** for causal explanations involving size versus domain identity,
label distribution or user/item coverage. Only three domains/dataset, with task
and cell dependence, cannot identify size as a primary cause. Candidate gains,
small/large trade-offs, contradictions and training-stage effects remain pending.
Raw source: development_readout/cells.csv and method_means.csv; no test read.

## MovieLens size/difficulty context (complete4-seed development)

|Domain|Main train N|Task0 AUC/loss|Task1 AUC/loss|Task0/1 loss-skill vs empirical constant|
|---|---:|---|---|---|
|0|32775|.727930/.617921|.735701/.461903|.105449/.105052|
|1|2850|.714597/.649270|.722935/.503031|.038755/.087848|
|2|142500|.743610/.585896|.740615/.484798|.134351/.114229|

The smallest ML domain has lower AUC and lower loss skill in this fixed population.
In KR, the smallest is not lowest AUC. These observations contradict a universal
size=difficulty equivalence. They do not identify the causal effect of sample
size: native domain identity, label prevalence, support/feature distribution, task
relationship and historical sampling differ together. Raw LogLoss also depends
on entropy, so KR domain2's higher absolute loss cannot alone mark it hardest.
A crossed within-domain random size intervention with a fixed evaluation population
would distinguish size from these explanations better than correlating three
correlated domain points. This is a future falsifier, not an unrun declared arm.
