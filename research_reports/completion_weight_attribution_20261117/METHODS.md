# Cycle04 — Methods and interpretation of the objective chain

The registered comparison is CLEAN (C) → MASS-SCALE-025 (S) → COMPLETION-BCE-025 (W) → NQR-025 (N). W is the candidate, S is its new control, and C/N are twenty exact reused numeric trajectories. The final experimental population consists of two natural datasets, two new arms and seeds2022–2026, each trained for50epochs. The operational restart uses attempt2 after attempt1 lost its RAM states. Attempt1 remains a separate record of incomplete work and physical costs; its275 acknowledged epochs receive no completion credit in attempt2.

## Objectives

For batch size B and cell c=(domain,task), let n be the cell row count, w=n/(2B), pi_a=n_a/n, and bbar_a the present class mean probability-space BCE. Define C as the unchanged source BCE mean over both tasks, H_c=Σ_a pi_a² bbar_a, and o_c=2pi_0pi_1. Lambda=.25 at every formal step. The implemented losses are

`W=.75C+.25Σ_c w_c H_c`, `r=.25Σ_c w_c o_c`, `m=1−r`, `S=mC`.

The diagnostic/reference objective is `N=W+R`, where `R=.25Σ_c w_c o_c Q_c` and `Q_c=.5[(1−mu_1+mu_0)²+var_1+var_0]`. Class score means and variances use population reductions. Production W is constructed directly in the registered reduction order. It is never obtained by subtracting ranking loss from N. Only the actual arm's scalar loss receives the live backward.

Both W and S have actual total BCE coefficient mass m, with .875≤m≤1; neither is normalized by m. S preserves the natural allocation between cells and tasks while applying the exact current batch's global scale. W changes class, cell and task allocation jointly. Its minority-class rows generally receive less weight than its majority-class rows. Consequently W−S measures heterogeneous completion weighting conditional on the shared batch mass. A class-only, domain-only or balancing interpretation would require another intervention.

S−C includes the effect of global data-gradient scale under the actual source optimizer. Adam has coupled weight decay, populated moment history and nonzero epsilon; this contrast cannot be replaced by a claim of an equivalent learning-rate change. N−W adds the ranking component conditional on W and includes subsequent optimizer/parameter/EMA dynamics. The four trained endpoint means telescope algebraically; their differences do not identify percentages of trajectory mediation. The missing S+R arm leaves the interaction between ranking and heterogeneous weighting unestimated.

An empty cell has zero coefficient mass and null class statistics. An absent class has null mean/variance/BCE/coefficient; an observed zero retains its numerical value. All-single-class and engineering lambda-zero branches delegate to the frozen source CLEAN step. In ordinary mixed support, a single-class cell in W retains its source entry coefficient, while S still uses the global m from other cells.

## Natural data, model and optimization

MovieLens has800167train/100021validation rows,20updates per epoch and tail21927. KuaiRand has2006132/250767rows,49updates per epoch and tail40052. The fixed batch size is40960, with natural row permutations and actual tails, FP32 and no altered split, resampling, generated data or additional model evaluations. Train-only preprocessing, features, source initialization and the seed+11000 private CPU loader are preserved. Each new run verifies its initialization and all50epoch order hashes against the exact controls.

All arms use the repository's matched fixed-fusion M3oE fixed1: three domains, two tasks, embedding dimension16, four experts and FCN[512,256,256,64]. Ordinary gate inputs retain the source detach. Adam uses lr.01, betas(.9,.999), eps1e-8, coupled WD1e-4 for ML and1e-5 for KR, and foreach/fused/amsgrad=false. Fixed EMA is initialized from online, with rho=.9828205985452511 for ML and.995 for KR, and is updated once after each actual Adam step. CLEAN fixed EMA is the matched internal source control; this matrix evaluates mechanism attribution within that model/protocol.

The complete new matrix requires1000whole epochs,34500ordinary forwards/backwards/live Adam/EMA steps,1403149500training rows and2806299000task entries. These are useful logical operations in the completed attempt. Engineering, abandoned, partial and replayed physical work is recorded separately.

## Common-state diagnostics

The first ordinary batch at epochs1/10/30/50 is diagnosed on each of the20new trajectories:80points. It uses the same ordinary forward graph, three additional retained-graph gradient sweeps and four sequential disposable source-Adam clones from the same pre-update state. Totals are240extra sweeps and320virtual Adam steps. The actual arm's virtual and live post-update hashes must match exactly. Diagnostics add no model forward, live EMA update or algorithm RNG draw. Clone state is independently copied, and forward/optimizer hooks on source-identity branches are removed before model cloning and in finally blocks.

Reported quantities include objective losses, actual masses, per-parameter activity (None, tensor-zero and nonzero), private Adam steps, data-gradient/update norms and W−S/N−W differences and cosines. Groups cover ALL and actual top-level parameter prefixes. Zero-norm cosines retain null reasons. Actual-alpha/diagnostic-on/off, restored suffix and RNG/loader comparisons are exact. FP64 algebra uses atol1e-12/rtol1e-10; FP32 algebra uses atol2e-6/rtol2e-5. These tolerances do not relax exact state identities.

A diagnostic compares counterfactual objectives at one host arm's current state. Across hosts/epochs, model state, optimizer history and batch composition differ. A local gradient or virtual-step contrast therefore informs a competing explanation at that state; it does not identify the full trained trajectory's optimizer or EMA mediation effect. Reused C/N trajectories lack these new common-state measurements and remain explicitly unavailable in that table.

## Selection and statistics

Online and fixed EMA independently select the earliest strict maximum of six-cell validation Macro AUC across epochs1..50. Selected LogLoss, cells and state hashes use the same AUC-selected state. Fixed endpoints are E1/E10/E30/E50, alongside selected. Macro is an equal mean over the six cells. Traffic-weighted cell metrics and source task-pooled metrics are retained separately.

Within each dataset the statistical unit is the paired seed, n=5. Every contrast retains all five raw deltas, mean, sample SD(ddof1), sign/favorable counts and descriptive95% interval `mean±2.7764451051977987*SD/sqrt(5)`. AUC favors positive differences and LogLoss favors negative differences. Shared validation populations, repeated adaptation and correlated historical comparisons constrain interpretation; a zero-crossing interval does not establish equivalence.

The primary is independently selected W fixed EMA minus independently selected CLEAN fixed EMA, separately for both datasets. The practical gate is mean Macro AUC≥+.003 and every six-cell mean AUC difference≥−.001 in each dataset. Loss improvement is a separate axis. Both primary AUC means≤0 gives NO_CONTINUATION_SIGNAL; exactly one positive gives MIXED; two positive means are evaluated against the practical/cell gates. Controls, another estimator, pooled metrics or a favorable fixed epoch do not replace this primary comparison. Qualification does not itself select a champion or open Test.

The fifteen contrast families cover W−C, S−C, W−S, N−W and N−C for both matched streams, W EMA−C online, and each arm's EMA−online contrast. Each uses all registered endpoints and available source metrics. Domain size, train/validation class proportions, baseline performance, tasks, model and stage are examined together. Evidence is labeled STRONG, MEDIUM or WEAK according to the specific claim. The final report identifies stable observations, contradictions, competing explanations, size plausibility and any small/large-domain trade-off. After this report is reviewed and published, a fresh Astra selects the discriminating question for the one additional authorized Cycle05.

## Storage, admission and final boundary

Model, EMA, optimizer, loader and RNG states stay in RAM/BytesIO. Whole-epoch commits have an actual owner ACK and immutable numeric transaction/record linkage. Finished runs retain online/EMA selected states only. Parent acceptance directly checks all40selected model hashes, selected epochs/validation,1000ACKs and191017920tensor bytes in the current live owner before report acceptance.

Actual availability changed after that successful direct acceptance at2026-11-17T11:21:33.883200+00:00. The final evidence reviewer observed owner2476891 absent at12:02:24, and the parent confirmed its foreground session returned143. The selected tensors are now unavailable; no disk reconstruction or replacement owner was used. The historical acceptance and numeric analysis remain valid, while live-state continuation and Test release are unavailable. The post-acceptance availability receipt is published outside the unchanged raw archive.

Each GPU independently admits work when current measured memory and power are both below85%, current host RAM is below90%, startup allocation is acknowledged, and the measured incoming full-process envelope fits within CUDA-usable100% capacity. There is no worker-count cap or both-card gate. The repaired startup latch binds the newly launched PID and expected owner registration attempt, and releases only after that running generation reports allocated=true and current telemetry is resampled. Other users' processes are left untouched.

All agents are fresh native children of the current conversation, with fork_context=false, explicit file handoffs and separate coder/reviewer identities. Completed contexts are closed permanently. No independent/background AI or retired autonomous driver is used. The latest user instruction authorizes completion of Cycle04, its report/review/publication, followed by exactly one additional complete Cycle05. USER_STOPPED follows the Cycle05 report/delivery; Cycle06 is not authorized. Test stays sealed throughout these research cycles. The current continuation scope receipt supersedes the earlier stop-after-Cycle04 instruction.
