# Cycle02 algorithm designs

Status: research specification for a separate fresh designer to formalize and check. Only A and its two controls are selected. No research code has been implemented. All formulas below define the proposed intervention; they are not empirical results or global-novelty claims.

## A. Natural quadratic ranking, NQR-025

### A1. Input, model, and masks

Each natural loader batch has actual size B>0 (including the tail). Input x has the source shape [F,B]; last feature x[-1] is the unchanged domain indicator in {0,1,2}. Binary target y has shape [B,2]. Ordinary `p=model(x)` has shape [B,2], FP32 probabilities. Features, routing, tower selection and labels are unchanged; no continuous interpolation, label smoothing, negative resampling, support cap, hard-example selection, or new model parameter is introduced.

For cell c=(d,t), d=0,1,2 and t=0,1, define masks I_c^a={i:d_i=d,y_it=a}, a in {0,1}. Counts n_a=|I_c^a|, n=n_0+n_1, pi_a=n_a/n and w_c=n/(2B). Here n is the domain count and is identical for both tasks, but pi_a is task-specific. Masks/counts are nondifferentiable labels/indices; all score statistics below remain differentiable. Valid labels/domain values and p shape are checked. Empty domain: contributes zero, no empty mean. Missing class: take the one-class branch below, not epsilon division. No row or task is discarded.

Let b_it=probability BCE(p_it,y_it), exactly the source `F.binary_cross_entropy(..., reduction='none')` semantics. CLEAN scalar C is the source `F.binary_cross_entropy(p,y,reduction='mean')`, not a sum of separately rounded cell means. Do not substitute BCEWithLogits, add probability clamps, change sigmoid/routing, normalize logits, or reconnect detached gates.

For each nonempty class:

    mu_a = mean_{i in I_c^a}(p_it)
    v_a  = mean_{i in I_c^a}((p_it - mu_a)^2)       # population variance, divisor n_a
    bbar_a = mean_{i in I_c^a}(b_it)
    S_c = pi_1^2*bbar_1 + pi_0^2*bbar_0

Use centered squared deviations in FP32, not `E[p^2]-mu^2` followed by a clipping fix. `mu_a` is not detached inside v_a; no running class mean, stop-gradient teacher, or unbiased n_a−1 correction. For a singleton v_a=0. A missing-class term in S is absent, not `0*NaN`.

### A2. Exact pair risk and O(B) reduction

Conceptually draw two indices independently and uniformly from the n natural rows of this domain, with self-pairs allowed. There are n^2 ordered pairs; these are a mathematical sum, not a loader, RNG stream, extra examples, or materialized tensor.

For equal labels, pair loss is (b_i+b_j)/2. For unequal labels, orient the positive first, and use

    q_R(p_positive,p_negative) = 0.5*(1-p_positive+p_negative)^2.

Both label orientations contribute, hence opposite-label probability 2*pi_1*pi_0. The exact opposite-label average is

    Q_R,c = 0.5*((1-mu_1+mu_0)^2 + v_1 + v_0)
    A_R,c = S_c + 2*pi_1*pi_0*Q_R,c
    A_R = sum_c w_c*A_R,c
    L_R = 0.75*C + 0.25*A_R.

No n_1*n_0 matrix is formed. Six cell loops and their class reductions cost O(B*T), T=2, memory O(B*T) beyond the ordinary network graph. The mathematical derivation follows by expanding the square under the independent within-class product measure. It is exact for this batch, including finite counts; there is no Monte Carlo pair estimate and no claim that random-batch ratios are an unbiased population objective.

The margin 1 is the full probability range; there is no margin/temperature sweep. Since p in [0,1], the gap cannot exceed1. The quadratic cannot reverse its pairwise gradient because an already-correct gap exceeds1, unlike an unbounded-score square loss. Sigmoid saturation, label noise, calibration distortion and compression of useful within-class variation remain possible. These facts do not prove AUC consistency or endpoint efficacy.

### A3. Matched controls

All three new arms use the same masks, S_c, w_c, opportunity factor 2*pi_1*pi_0, lambda=.25, model call and optimizer sequence. Only Q changes:

| Arm | Opposite-label component Q_c | Purpose |
|---|---|---|
| NQR-025 | .5*((1-mu_1+mu_0)^2+v_1+v_0) | Selected candidate |
| BRIER-OPPORTUNITY-025 | (1-mu_1)^2+mu_0^2+v_1+v_0 | Exact average of q_B=(1-p_positive)^2+p_negative^2 under the same pair opportunities; pointwise squared-error competitor |
| MEAN-GAP-025 | .5*(1-mu_1+mu_0)^2 | Remove the candidate's within-class variance penalty, holding the mean-gap term exactly fixed |

For any arm a, A_a=sum_c w_c*(S_c+2*pi_1*pi_0*Q_a,c) and L_a=.75C+.25A_a. MEAN-GAP is a batch-mean objective, not an average of the same per-pair loss. All arms compute the same sufficient statistics and all three detached diagnostic Q values, then backpropagate only their assigned scalar. An unused control scalar does not create a second training update.

Exact identities:

    Q_R - Q_M = .5*(v_1+v_0)
    q_B - q_R = .5*(1-p_positive-p_negative)^2
    Q_B - Q_R = .5*((1-mu_1-mu_0)^2+v_1+v_0).

At p_positive=p_negative=.5, q_B=q_R=.5 and both first derivatives are (−1,+1). This is a reference-point scale match, not an all-state norm match. B versus R changes the pair common-score penalty and half of the variance contribution, so it is **not a pure mean-offset ablation**. R versus M is the exact variance ablation. Their combination distinguishes more than an unmatched generic noise control without pretending to identify all training dynamics.

There is also an exact reference-point check against CLEAN: if every p_it=.5, all three new objectives satisfy `dL/dp_it=(1-2*y_it)/B`, the source mean-BCE derivative, for any natural class proportions. The same-label BCE part contributes `(.75+.25*pi_class)*(1-2*y)/B`; the opposite-label part contributes `.25*pi_other*(1-2*y)/B`. They sum to CLEAN. This motivates the fixed quadratic scaling without data-dependent calibration. It does not hold at general scores or imply identical losses/trajectories.

Example for algebra checks: mu_1=.8, mu_0=.2, v_1=v_0=0 gives Q_R=Q_M=.08 and Q_B=.08; shifting both means by .1 gives Q_R=Q_M=.08 and Q_B=.10. A common shift preserves a probability difference within valid bounds, but the complete objective, sigmoid Jacobian and parameter update are not offset-invariant.

### A4. Weight mass and natural support

Within a nonempty cell, pi_1^2+pi_0^2+2*pi_1*pi_0=1. Sum_c w_c=1 and each task's sum_d w_c=.5. Thus the pair-completed risk has total mixture-coefficient budget1; the final scalar mixture is .75+.25=1. Unlike Stage1, no extra .5 BCE mass is added.

This does **not** preserve every row's BCE coefficient or gradient magnitude. The explicit BCE coefficient for a row of class a is

    k_BCE(i,t) = [0.75+0.25*pi_a]/(2B).

Its remaining role belongs to the opposite-label term. Total cell BCE mass is w_c*[.75+.25*(pi_1^2+pi_0^2)]; opposite-label component mass is w_c*.25*2*pi_1*pi_0. At pi_1=pi_0=.5, these are .875*w_c and .125*w_c. At pi_1=0 or1, they are w_c and0. No inverse class-prior or inverse-domain-size compensation is applied.

In the conceptual uniform ordered-pair measure each row has the natural uniform marginal in each index role. This is not equal influence on gradients: an example's opposite-label partner frequency depends on the other-class proportion. Scores, noise and shared parameters add further heterogeneity. Record this rather than describing the objective as class-balanced or claiming equal effective regularization. The .5 in q_R and the unaveraged two squared errors in q_B are part of the fixed loss definitions; coefficient accounting is not a theorem that their scales equal BCE.

### A5. Zero, degenerate and error branches

If lambda=0 in the engineering identity check, call frozen `task_view_gradient_residual_momentum_20261116.core.source_clean_step` directly before any candidate forward/statistics. No candidate RNG or probe work occurs. Lambda=0 is not a new formal arm.

If every nonempty cell in a batch has only one class, A=C mathematically for all arms. Detect from labels before the forward and use the same direct source-clean branch, to retain exact source operation order. If only some cells are one-class, set A_a,c=mean BCE of that cell (for all arms), calculate the other cells normally, and keep every row. No synthetic positive/negative is supplied. B=0, nonbinary labels, invalid domains, nonfinite probabilities/loss/gradients are implementation/data errors causing technical PAUSE, not an efficacy verdict. No clipping/fallback that changes this registered objective.

In a direct source-clean branch the frozen function returns only scalar C. Record branch identity, counts, C=A=L and computable coefficient masses; use JSON null for unobserved class score/BCE statistics and undefined Q. Do not perform a second forward merely to fill those diagnostics. For partially one-class batches the ordinary shared forward already provides class statistics; Q for a missing-class cell remains null, while its cell risk is its observed mean BCE. Null means unavailable/inapplicable, never zero evidence.

### A6. Ordinary graph, optimizer and EMA

There is **no perturbation/probe graph**: probe forwards=0, embedding Jacobian calls=0, view forwards=0, residual channels=0. Use the source ordinary M3oE forward including detached gate inputs. Embeddings, STAR/skip, experts, gates, towers, fusion and sharing remain trainable exactly as in CLEAN. Do not use the full-gate-derivative hook in auxiliary `forward(...,probe=True)` or construct views.

Per nondegenerate batch: `zero_grad(set_to_none=True)`; one ordinary forward; FP32 BCE/moments/loss; one scalar backward; finite-gradient check; one source torch Adam step; one source EMA update. No optimizer update for statistics/control values. No gradient projection, clipping, residual injection, schedule, preconditioner override, accumulating microbatch, or new optimizer group. Coupled weight decay is applied once by Adam, not added twice to the data loss.

For each active parameter, with g=grad L and d=weight_decay:

    h = g + d*theta_old
    k = k_previous + 1
    m = .9*m_previous + .1*h
    v = .999*v_previous + .001*h^2
    theta_new = theta_old - .01*(m/(1-.9^k))/(sqrt(v/(1-.999^k))+1e-8).

Actual implementation must invoke source `torch.optim.Adam` to preserve operation order, not recode this mathematical summary. `grad=None` skips that parameter's decay, moments and counter; an explicit zero gradient remains active and receives ordinary decay/counter update. Active masks must not be broadened by synthetic zero-gradient assignments.

EMA starts as an exact copy of initialized online state, then `ema=rho*ema+(1-rho)*theta_new` once after each successful Adam step; copy buffers exactly, no bias correction. rho_ML=.9828205985452511, rho_KR=.995. If a step fails between mutations, recover the last acknowledged epoch transaction from RAM; never complete only half an Adam/EMA transaction and then continue.

### A7. Loader, RNG, tails and selected states

Use frozen setup(seed,device) once, including one CPU torch thread, source Python/NumPy/CPU/device RNG seeding, TF32 off and cuDNN benchmark off. Source model constructor and parameter order are unchanged. Create one private CPU torch.Generator seeded seed+11000; consume one `randperm(N,generator=loader)` each epoch, once, and advance normally. No algorithm RNG is needed. Statistics, logging and evaluation do not consume loader/global RNG. Diagnostic-on/off smoke must preserve state and RNG identity.

Natural batches are 40960 with no drop_last. ML: N=800167,20 batches/epoch,tail21927,1000 updates in50 epochs. KR: N=2006132,49 batches/epoch,tail40052,2450 updates. Use actual B for every coefficient. Epoch orders and initialization hashes must match each seed's frozen CLEAN index; do not reseed per epoch or arm-specific loader stream. Each formal arm uses all seeds2022–2026,50 epochs,FP32 without early stopping.

At every epoch evaluate unmodified natural validation metrics separately on online and fixed EMA. Select each estimator's earliest strict maximum of six-cell macro AUC; take LL and all cells from the same selected state. E10/E30/E50 are descriptive fixed-time reads, never replacements for the primary selected EMA endpoint. Retain selected online and EMA states in RAM; no disk model/optimizer/weights/checkpoints. Current continuation contains online, EMA, Adam state/counters, RNG states, loader state, epoch/global step, selected states/epochs/metrics, best scores, counters and accumulated raw costs. There is no NQR-specific persistent tensor state.

### A8. Minimal diagnostics and costs

Per batch log B, row-order hash, n_0/n_1 for each cell, C, A, L, S, Q_R/Q_B/Q_M, class mean/variance, and the two component masses above. Record observed BCE coefficients per class, total mass and each task mass. No per-example prediction dumps or pair lists. Epoch aggregation must distinguish row-weighted mean losses from sums of per-update mass; do not make 20 or49 unit-mass batches appear to have mass1 before averaging.

On first batch of epochs1,10,25,40,50, additionally log data-gradient L2 norm, actual theta-after minus theta-before norm (including Adam/decay), active/None counts, Adam counter range, parameter-group norms by source name prefix, and Q decompositions. Read gradients after ordinary backward; temporary pre-step parameter copies are RAM only and discarded after the update. No additional parameter-gradient probe, control update or trajectory is needed. Scalar reductions for logging may use float64 on detached tensors; training stays FP32. Do not call a data-gradient norm an update norm.

New work:3 arms*2 datasets*5seeds=30 trajectories,1500 epochs,51750 Adam and51750 EMA updates. Formal training model forwards/backwards each51750; validation and engineering smoke are separately accounted. Extra arithmetic is O(B); no additional persistent O(P) state, no n^2 pairs. Historical CLEAN cost for an equivalent 30-run matrix is 15*(407.738488+1155.568793)/3600=6.513780 worker-hours. This is a scale reference, not a measured NQR forecast/bound or wall-time claim. New full-process peak, throughput and RAM serialization peaks require reviewed smoke; hardware rules are in CODEX_EXECUTION_PLAN.

## B. Prediction-distribution smoothness — unselected

Define an input-embedding displacement r_adv approximately maximizing KL between detached clean Bernoulli predictions and perturbed predictions under ||r_i||<=epsilon_i, with a label-independent finite-difference direction. Optimize C+eta*mean KL; route the original domains and ordinary detached training gates. It tests local prediction stability rather than BCE ascent. A matched isotropic equal-radius KL control would be necessary. Radius/eta, multi-task reduction and zero-direction rules are unresolved, so this bundle authorizes no implementation of B. It would need its own complete design before any run. Unlike NQR, it changes local input response and requires extra evaluations. It is an established VAT-family idea, not a discovery from C1's local E/O diagnostics.

## C. Parameter-neighborhood risk — unselected

For ordinary clean data gradient g over the source active parameters, choose epsilon=rho*g/(||g||+epsilon_num), use epsilon=0 if g=0, evaluate clean loss at theta+stop(epsilon), restore theta, then take one Adam step using that perturbed gradient and one EMA update. Weight decay belongs to the restored-base Adam update, not the ascent direction. This is a SAM-family approximation to a parameter-neighborhood maximum, distinct from both A's output risk and B's input-distribution smoothness. A random equal-norm parameter displacement and CLEAN are required controls; rho, norm convention across embedding/towers, and exact restore checks remain future design decisions. No SAM run, sweep, or code is authorized here.
