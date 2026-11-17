# Algorithm research handoff — CWA-C04

Selected candidate: **COMPLETION-BCE-025** (W). New control: **MASS-SCALE-025** (S). Reused controls: **CLEAN** (C), **NQR-025** (N). The fresh designer must formalize these contracts in a separate DESIGN_SPECIFICATION after parent acceptance. No implementation is supplied here.

## Common notation and source algebra

A complete source batch contains B rows, including its natural tail, two binary task labels per row and a domain in {0,1,2}. Source model probabilities p have shape [B,2]; source inputs x have layout [F,B], with domain in x[-1]. Let c=(d,t), n_c its number of rows, n_ca its class-a count, pi_ca=n_ca/n_c if n_c>0, and w_c=n_c/(2B). Let ell_it be the exact source probability-space BCE, not BCEWithLogits or a new clipping rule.

`C = mean_{i,t} ell_it = sum_c w_c [pi_c0 bbar_c0 + pi_c1 bbar_c1]`.

Define `H_c = sum_{a present} pi_ca^2 bbar_ca`, `o_c=2 pi_c0 pi_c1`, and `Q_c=.5[(1-mu_c1+mu_c0)^2+var_c1+var_c0]` when both classes exist. Class variance is the population variance `(scores-mean)^2.mean()`, not a sample variance. The frozen C2 NQR objective is

`N=(1-lambda)C + lambda sum_c w_c [H_c + o_c Q_c]`, lambda=.25.

Empty cells contribute zero; one-class cells contribute H only, with no fabricated opposite-class statistic. This is the actual frozen algebra in `experiments/natural_quadratic_ranking_20261116/core.py`.

## Candidate W: remove all ranking gradient, preserve completion BCE

`W = (1-lambda)C + lambda sum_c w_c H_c`

`  = (1/(2B)) sum_{i,t} q_c,y_it ell_it`, where `q_ca=.75+.25 pi_ca`.

Compute using source BCE and the same cell/class reduction order as the completion part of C2. The row-weight expression is an independent algebraic check. Do not optimize a reconstructed value `N-R` that could introduce avoidable cancellation; compute W directly. Do not add residual momentum, auxiliary sampling, centering, a rank term, coefficient renormalization or history.

Define `r=lambda sum_c w_c o_c` and `m=1-r`. With lambda=.25, `.875 <= m <= 1`. The total actual BCE coefficient mass of W is **m**, not 1. The missing ranking mass r is deleted, not reassigned.

## Control S: match that batch's total BCE coefficient mass

`S = m C`, with m computed from the very same batch labels/domains as W. All rows/tasks receive multiplier m. It is not a dataset-constant .895/.899 approximation, an inverse-class weight, a gradient-norm match or a learning-rate change.

Its purpose is conditional identification: W-S changes the distribution of BCE coefficients while keeping their total equal. S-C measures this exact global scaling intervention under unchanged source Adam/EMA. The same actual batch order makes m identical between W and S for the same dataset/seed/step, even after their model trajectories diverge.

## Support and mass ledger

| Quantity | C | S | W | N |
|---|---|---|---|---|
| Natural support | all 2B entries | all 2B | all 2B | all 2B plus analytic opposite-class risk |
| Per-present-entry BCE coefficient | 1/(2B) | m/(2B) | (.75+.25 pi_a)/(2B) | same as W |
| BCE total mass | 1 | m | m | m |
| Ranking coefficient total | 0 | 0 | 0 | r |
| Total objective coefficient mass | 1 | m | m | m+r=1 |
| Task t BCE mass | .5 | m/2 | sum_d w_dt[1-lambda o_dt] | same as W |
| Cell c BCE mass | w_c | m w_c | w_c[1-lambda o_c] | same as W |

W's task masses need not equal m/2 or .5. Nonuniform class prevalence changes cell/task allocation, so W-S identifies the combined heterogeneous completion weighting, not class-only versus domain-only causality. Record actual cell/task masses. Preserve raw class counts and pi; rare-class rows are generally downweighted more than majority-class rows. Do not describe this as rare-class balancing or natural unit-mass training.

For n_c=0: w=0, n0=n1=0, pi/statistics/coefficient-per-absent-entry=null, cell mass=0. For n_ca=0 in a nonempty cell: pi_ca=0 is defined, that absent class's mean/variance/BCE and per-entry coefficient are null; o=0. Single-class cells have W coefficient1/(2B), but S still scales their rows by global m when other cells have both classes. If every nonempty cell is single-class, r=0,m=1 and both arms delegate to the exact source CLEAN step. Lambda=0 is an engineering identity branch only, also an exact source-CLEAN delegate. B=0 or nonbinary labels/invalid domain/shape/nonfinite predictions are technical errors, not silently skipped training steps.

Do not change the production arm or lambda in response to batch support. Ordinary single-class and absent-domain cases are covered without dropping rows or renormalizing supported cells.

## Objective contrasts and their limits

`R=lambda sum_c w_c o_c Q_c`, so N=W+R in real arithmetic. Check the loss and probability/parameter gradients against the frozen pure C2 operator within preregistered FP tolerances; floating reduction order need not give bitwise identical alternative expressions. Do not demand trajectory equality between W and N.

Registered same-estimator contrasts are W-C (primary when selected EMA), S-C (scale), W-S (nonuniform weighting), N-W (ranking addition conditional on completion), N-C (historical anchor). Also report W/EMA-C/online (total package), all four within-arm EMA-online contrasts and matched online components. Endpoints are selected, E1,E10,E30,E50. Selected is the independently chosen epoch for each side, not an equal-epoch comparison. Fixed-epoch contrasts compare like epochs.

This chain does not measure the missing interaction `(N-W)-[(S+R)-S]`. It also does not isolate a ranking direction from its norm, nor identify full-trajectory Adam/EMA mediation. Those would need different interventions. This explicit boundary is why only two new arms are necessary now.

## Frozen training and optimizer contracts

Both datasets use unchanged natural labels/populations and original 80/10/10 split with train-only preparation; Test stays unopened. Model is M3oE fixed1, 3 domains, 2 tasks, embedding16, 4 experts, FCN [512,256,256,64]. Ordinary source gating uses detached embeddings as gate input. Routing, towers, experts, fusion, sharing and all parameters remain unchanged; the source uses LayerNorm, no new dropout or feature augmentation. No architecture or group-specific update changes.

FP32, batch40960, 50 epochs, EXACT seeds [2022,2023,2024,2025,2026]. ML has20 complete batches/epoch with tail21927; KR49 with tail40052. Source Adam: lr=.01, betas=(.9,.999), eps=1e-8, coupled weight decay ML1e-4/KR1e-5, foreach=False, fused=False, amsgrad=False. Do not substitute AdamW, reset moments, rescale weight decay, retune epsilon or create a second live optimizer. Each complete batch has one ordinary model forward, one actual scalar backward, one live parameter update and one EMA update. Fixed rho ML .9828205985452511 / KR .995. EMA starts as a source model copy; update parameters by rho*old+(1-rho)*online and copy source buffers.

`zero_grad(set_to_none=True)` and source autograd determine activity. Tensor-zero gradients remain active: coupled decay and moments update, and private parameter step increments. `grad is None` skips that parameter's Adam step/state/decay. Do not replace None by zero or manually deactivate zero-gradient/absent-domain groups. Do not infer activity from domain support; actual shared/gated model graph decides. Named-parameter order and group membership remain source-defined.

For an active parameter with private step k, the mathematical source update is `g_tilde=g+wd*theta`, `m_k=.9*m_prev+.1*g_tilde`, `v_k=.999*v_prev+.001*g_tilde^2`, then `theta <- theta - [.01/(1-.9^k)]*m_k/[sqrt(v_k)/sqrt(1-.999^k)+1e-8]`. The installed source implementation, including its floating-point operation order and lazy state initialization, is authoritative. The W/S gradient changes the data term g; coupled decay is not scaled by m or q.

Use Python3.13.5/PyTorch2.9.0+cu126 from the exact project environment/lock, one CPU thread, TF32 off for CUDA matmul and cuDNN, source deterministic seed settings and cuDNN benchmark=False. Preserve actual source settings; do not add an unrelated deterministic-algorithms switch. Seed Python, NumPy, CPU torch and the selected CUDA device as source does. The private CPU loader generator uses seed+11000 and one randperm per epoch. W/S consume no algorithm RNG. Diagnostics and restore must leave Python/NumPy/CPU/CUDA/loader RNG unchanged. Match all 50 recorded epoch-order hashes and recorded initialization identities against available CLEAN numeric references. Establish any unrecorded initial EMA/Adam identity through fresh source-equivalence fixtures, without inventing historical fields.

## Minimal common-state diagnostics

At the first actual batch of epochs1,10,30,50 in each new trajectory (80 points total), use its ordinary forward and the same current theta, labels and live pre-update Adam state. Compute C,S,W,N from that probability graph. Obtain gradients of the three objectives other than the live arm using autograd without assigning live `.grad`, retaining the graph. The one registered live scalar backward supplies the fourth gradient. Thus a W-hosted point probes C,S,N; an S-hosted point probes C,W,N. Check g_S=m*g_C within the fixed FP tolerance, but use each objective's actual autograd gradient for its counterfactual Adam step; scaling an already reduced g_C is not substituted for S's floating-point backward. Complete all probes while live parameters/Adam are still at the pre-update state, then perform the single live optimizer/EMA update. No extra data forward, pair sampling, live optimizer step, live EMA update or loader consumption is allowed.

For each diagnostic point, sequentially apply each of C,S,W,N gradients to a disposable cloned model plus **the actual source Adam implementation**, starting from the same pre-update parameters and a deep copy of each parameter's private m,v,step state. Preserve parameter ordering, activity, coupled decay and lazy initialization. One clone at a time limits residency. Four virtual Adam steps per point; all cloned tensors stay in RAM and are discarded. A hand-coded approximation or CLEAN moments substituted for the current arm's moments is not acceptable. Counterfactual C at a W/S state is labelled as such, never as the historical CLEAN trajectory.

Record global and existing top-level parameter-prefix groups: data-gradient norms for C,S,W,N; norms of g_W-g_S and g_N-g_W; cosines W/S and N/W; each corresponding parameter-update norm and update-difference norm/cosine; active/None/tensor-zero counts and private-step ranges. For norms only, an absent vector component is mathematically zero; never materialize that zero as a live/clone `.grad`. Record any activity-mask difference explicitly. Compare actual arm update with its corresponding virtual update. At a fixed pre-state, EMA update-difference is (1-rho) times online update-difference; verify algebraically rather than running four extra EMA trajectories.

An undefined cosine has null value plus `zero_norm` reason. Unobserved historical diagnostics have null plus `not_recorded_in_source`; they are not zero. No per-cell gradient sweep is required. Cell/class means and population variances are diagnostic-only for W/S, using actual class-count denominators. Off-diagnostic values are null with `not_scheduled`, while support/mass counts exist every complete batch. Historical NQR diagnostics at different epochs are not interpolated. Report mean-of-batch statistics as such, never as a pooled population variance.

These diagnostics distinguish an immediate objective gradient from the source optimizer's immediate transformation at a given evolving state. They do not compare alternative full training histories while holding Adam fixed; causal statements must stay conditional.

## Interfaces for formal design

The following are contracts, not implemented APIs. Designer may retain existing source-compatible field names where semantics are unambiguous.

* `loss_components(probabilities, targets, domains, *, arm, mixture_lambda=.25, diagnose=False)` returns differentiable actual loss, clean C, completion-only W, detached m/r and a six-cell ledger with explicit actual mass semantics. Production arms are only W/S. Probability-space FP32 production and FP64 finite fixtures are supported.
* `train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` has source layout/activity semantics; returns finite detached measurements and actual operation counters. `context` may carry identifiers/counters, not persistent algorithm state. row_ids bind the loader/order audit.
* `protocol` owns new package/artifact/report roots and a fixed explicit dependency list. Reuse pure model/evaluator/optimizer/RNG/digest/CPU-copy helpers and frozen C2 loss_components for diagnostics. Never call an old root-bound prepare/run/source_manifest/export/admission/ensure_guardian routine or scan mutable old roots into a new source identity.
* `runner` owns all new jobs, numeric logs and recovery journals; `tables` owns extraction only. No new plugin/adapter/framework or old-package edit is needed. New source consists of minimal sibling core/protocol/runner/tables/checks/readme and necessary package/test files.

## Alternative formulas, not authorized implementations

The zero-mean view-noise and instantaneous projection mechanisms in RESEARCH_EXPLORATION are alternatives, not extra arms. Their conditional-expectation and projection formulas are distinct from W/S. Their additional covariance/direction controls are necessary if a later plan selects them. No alternative, lambda sweep, normalized W/m, inverse-frequency arm or S+R arm is silently included in this budget.
