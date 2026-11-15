# DESIGN SPECIFICATION — heterogeneity-aware augmentation

Design completed before algorithm implementation, 2026-11-15. Source audit: [SOURCE_MAP.md](SOURCE_MAP.md). Governing plan: addendum16 + manifest_v3. The user authorizes execution and overrides the development pool: **only seeds2022,2023,2024,2025,2026**. Development and confirmation reuse these seeds; Test remains sealed throughout development/HPO. No synthetic data or altered natural sampling.

## 1. Frozen platform

Use the current-main vendored fixed1 model and audited dataset loaders mapped in SOURCE_MAP. Two datasets, three domains, two tasks, natural80/10/10 split fixed at2022, train-only encoders/scalers. ML task0=`rating>3`, task1=`rating>4`; KR click/long_view. Batch40960, 50epochs, no early stopping. Preserve source CPU loader generator(seed+11000), all rows once/epoch, final short batch, FP32/no AMP/no TF32. Single Adam lr=.01, betas(.9,.999), eps1e-8; coupled WD ML1e-4/KR1e-5; no scheduler/clipping/extra optimizer. Fixed EMA from initial model, one after-Adam update/batch, rho ML.9828205985452511/KR.995. Select online and EMA independently by earliest strict maximum of validation six-cell Macro AUC over epochs1–50.

The clean objective is the **actual source sample-mean probability BCE**, `mean(BCE(model(x),y))` over `B*2` scalars. Retain source gate-input detach during parameter training. This corrects the obsolete Equal-style expression in earlier plan text without changing the accepted fixed1 protocol.

## 2. Full input sensitivity without changing training gradients

`e` is the full concatenated EmbeddingLayer output; no field/coordinate is excluded in this new method. Input IDs, domain selection, labels and architecture are unchanged. The whole embedding radius is used, rather than the historical Gaussian per-field radius.

Capture raw tower logits using the existing tower-hook pattern. For **sensitivity probe only**, reconnect each gate input to the differentiable sum of the captured outputs of `star_mlp` and `skip_conn`, using temporary gate forward-pre-hooks. The source gate receives exactly this value detached, so probability/logit values must match the ordinary forward. This recovers the derivative of the actual numerical function, including its gate dependence. Remove every hook in `finally`. Do not use the reconnected graph for clean or augmented parameter-training backward. A numerically identical isolated probe implementation is acceptable if this hook method cannot pass the tests.

Probe selected rows only, compute `autograd.grad(logits[:,t].sum(),e)` for t0/t1 with `create_graph=False`; do not accumulate into parameter `.grad`. Compute both gradients for a single union of sampled rows when practical. Detach generated delta before augmented training forward; keep the actual embedding lookup attached so auxiliary BCE updates the source model. No second-order differentiation.

## 3. Sampling, directions and cap

Every minibatch has six target cells `(d,t)`. For pair routes sample without replacement `k=min(128,n_positive,n_negative)` positive and negative row indices independently in the cell, zip in sampled order. No constraint on other-task label. Empty/single-label cells contribute no pairs. Use deterministic, independent keyed RNG from `(dataset,seed,epoch,global_step,stream)`; pair and normal/random streams must not advance the clean loader/global RNG.

For each pair member get target gradient `g`, other-task gradient `h`. A uses `q=g-dot(h,g)/dot(h,h)*h` when `dot(h,h)>1e-20`, otherwise `q=g`. Compute scalar dot products in float64; cast delta to embedding dtype. B uses `q=g`. Skip entire pair if either usable direction norm<=1e-10 or embedding RMS<=1e-12. A records normalized projection residual. Never replace invalid q with noise.

Raw direction: positive `-q/||q||`, negative `+q/||q||`; radius `alpha*sqrt(mean(e**2))`. Denominator epsilon1e-12. Let `m=z_pos-z_neg`, `dm=g_pos·delta_pos-g_neg·delta_neg`. Skip pair if `abs(m)<=1e-7` or `abs(dm)<=1e-12`. Otherwise multiply both members by `c=min(1,abs(m)/(abs(dm)+1e-12))`. No kappa, damping or margin floor.

A then measures actual non-target-logit drift at the two perturbed rows. Require `abs(z_other_aug-z_other_clean)<=1e-3` for both. If violated, halve both deltas, at most two times; skip the pair if still violated. Record before/after margin, predicted margin change, actual non-target drift, cap/backtrack/skip counts, q/g norms and delta norms. B has no non-target rejection/backtracking; report its drift directly.

C samples `2*k` distinct rows uniformly from each domain for each target task, where k is the available-pair budget above; it does not construct pairs, balance labels or apply margin caps. Perturb along normalized `grad_e BCE(z_t,y_t)=(sigmoid(z_t)-y_t)*g_t` at the same raw alpha*RMS radius. Skip only near-zero gradients/RMS. It uses no cross-task projection/backtracking.

## 4. Generic controls and objective

RANDOM and GAUSSIAN use the same selected pairs as A and its accepted per-row final delta norms. RANDOM uses independent Rademacher coordinates normalized to that norm; GAUSSIAN uses iid standard-normal coordinates normalized to that norm. Do not project, sign by labels, or reject controls based on non-target drift. Thus direction distributions differ while norm, target support and objective are matched. Log all actual scales; A/B/C may have different post-cap norms, so their ordinary comparison is at matched nominal alpha, not proof of an exact norm-controlled projection effect.

If A produces no accepted pairs while another route remains viable, generic controls instead use the uncapped C budget/radii for that batch and record the fallback. A numerical viability failure alone must not eliminate B or C.

For A/B/controls let P be number of accepted pairs over six cells:

`L_aug = sum_pairs(BCE_positive_target + BCE_negative_target) / P`.

Equivalently, twice the mean target BCE over 2P augmented rows. C uses twice its mean target BCE, matching the same auxiliary loss scale. All use `L=L_clean+0.25*L_aug`. When no valid rows remain or lambda_aug=0, loss is exactly clean. No pairwise softplus loss, other-task augmented BCE, or change to clean population. Probability BCE matches source numerics; raw logits supply sensitivities/diagnostics.

For each training batch: zero_grad once; create detached deltas with autograd.grad; ordinary clean forward and augmented forward(s); sum loss; backward once; Adam once; EMA once. Model eval at validation/Test is clean/unperturbed.

## 5. Required minimal checks before research runs

CPU checks, with compact real-model fixtures rather than a broad engineering suite:

1. Same-domain target-positive/negative sampler and C row budget, unconstrained other label, deterministic independent RNG.
2. Exact rank-one linear projection, zero h, zero q, cap and label preservation; no NaN/Inf.
3. Probe ordinary/reconnected forward values agree. Per-row Jacobians equal gradients of logit sums; adding unrelated rows leaves derivatives unchanged.
4. Full-function finite differences including gate dependence agree on a smooth real-model point: double precision step1e-5, rtol1e-3/atol1e-7; avoid ReLU kink rows rather than relax the function definition. FP32 smoke tolerance rtol1e-2/atol1e-4.
5. Augmented delta is detached, hooks removed after success/error, ordinary gate-detach semantics retained for training; lambda_aug=0 yields source loss and Adam/EMA state equality.
6. One optimizer/EMA update/batch; in-memory snapshot/resume with loader and keyed sampler reproduces continuation.
7. One real batch at batch40960 for each dataset on an actual GPU: all six route/control paths finite, projection/drift/backtracking diagnostics, time/peak allocated+reserved memory. Smoke uses a single allowed seed and supports engineering conclusions only.

If an assertion fails, correct the relevant implementation and review again. Do not broaden unrelated test coverage.

## 6. Stage matrix and selection gates

All scientific performance decisions require five **full 50epoch** runs per dataset/configuration. The plan does not freeze a shorter pilot budget; use the full inherited budget. No 1–4seed performance rejection. No Test access during stages0–3.

| Stage | Frozen work | Runs and advancement |
|---|---|---|
|0|Source/data/protocol audit, design, independent implementation review|No research run until reviewer PASS or blocking fixes resolved|
|1|CPU checks and real full-batch smoke|2 dataset smoke jobs; engineering only|
|2|alpha=.03, K128, lambda=.25; CLEAN, A, B, C, RANDOM, GAUSSIAN; online+EMA each|2 datasets×6 arms×5 seeds=60 full runs /3000epochs|
|3|At most two promising A/B/C routes; alpha∈{.01,.03,.05}; all other settings fixed|Reuse identity-matching .03 trajectories; at most40 new runs /2000epochs|
|4|Freeze route(s), alpha, code/data/selection; selected candidate plus decisive controls, five seeds, both datasets|Reuse identical frozen trajectories only if exact code/config/data/selection identity holds; otherwise rerun. Same seeds by user override; no claim of untouched-seed confirmation|
|5|Unperturbed selected Test evaluation, all metrics/decisions/report, independent final review|No HPO after Test|

Stage2 primary development score is selected **fixed-EMA validation Macro AUC**, compared within each dataset. Report online separately. A route supplies a continuation signal if its five-seed mean is strictly above clean fixedEMA and both generic controls on at least one dataset. Rank at most two by number of positive datasets, then mean of the two dataset mean paired increments over the strongest of those controls; ties prefer C, then B, then A. Keep all six cell differences and cell violations visible; a validation cell violation does not erase an observed mean increment. If none qualify, stop the family, mark HPO/formal/Test NOT_TRIGGERED, write the full development report. This is a validation-stage decision, not a Test rejection.

Stage3 select a **single common alpha per selected route** maximizing the mean of the two datasets' selected fixedEMA validation Macro AUC. Use all five seeds. Tie by smaller alpha. Eligibility for frozen advancement: positive five-seed mean increment over cleanEMA and both generic controls on at least one dataset; report the other dataset explicitly. No extra K/lambda search is triggered under this specification. If no HPO configuration meets that condition, later stages remain NOT_TRIGGERED.

The strongest C/B/A route is the candidate; decisive formal controls include CLEAN, C, B for A; CLEAN,C for B; CLEAN for C; and matched RANDOM/GAUSSIAN at the frozen norm policy. All listed comparisons need five full runs. Identical selected alpha=.03 control trajectories may be reused; otherwise run the frozen controls. An A superiority claim requires one matched-norm A/B contrast, obtained by giving B the same A final row radii (including A cap/backtracking) and same support, with no new B cap. Run only if A shows positive five-seed incremental validation over B and C; no broad ablation grid.

## 7. Storage, resource lifecycle and identity

No model, optimizer, gradient, embedding, selected-state or checkpoint files on disk. Keep active/selected states and full RNG/optimizer/EMA continuation in RAM; only scalar/array data, prediction outputs, hashes, code, logs and reports may persist. For a finished development run, discard selected weights after validation; if frozen Test later requires that state and it is no longer in RAM, deterministically replay the frozen trajectory and verify selection/hash rather than reconstruct weights from metrics. Prefer retaining required selected states in a RAM guardian when affordable. Do not silently treat lost state as resumable. Test array access for hashing/preprocessing identity is not Test metric access.

Dispatch new workers on each GPU independently when used memory and power are both below85% of its capacity/limit and host RAM below90%; do not interrupt unrelated jobs. Profile full-batch memory before increasing concurrency. No silent batch reduction on OOM; first avoid overlapping own workers/release probe graphs. If full batch is impossible, record ENGINEERING_BLOCKED and do not change the comparison protocol.

Before runs write immutable run_manifest.json/source_receipt.json with main/code hashes, dataset identities, seed pool override, source clean-loss correction, model/optimizer/EMA configuration, staged matrix and gate definitions. Unique run/config IDs; attempts append rather than overwrite. Record seeds, epoch/step, loader order hash, initial/selected state hash, selected epoch, validation curves, cost and actual perturbation diagnostics.

## 8. Evaluation, analysis and acceptance

Report every seed and all six cells' AUC/unscaled LogLoss; Macro, population-weighted cell (Traffic) and pooled-task metrics separately; task/domain means, worst cell, selected epoch plus common epochs10/50 validation, online/EMA, total gain over clean online and component increment over cleanEMA. Analyze training-stage behavior and consistency, natural domain counts, baseline performance and model identity. Domain size is not difficulty; do not design a size-based hypothesis test.

For seed-paired differences compute mean, sample SD, 95% t interval `mean±2.776445*SD/sqrt(5)` and positive seed count; seeds are the independent units. AUC formal gate: mean Test Macro gain>=.003 vs strongest applicable published model, every cell mean delta>=-.001; no all-seed-positivity or CI-excludes-zero gate. LogLoss decision is independently mean delta<0. Main-only historical published aggregates can support an aggregate comparison; absent raw seed endpoints cannot support invented paired CI. Existing ordinary M3oE/current clean trajectories provide paired internal contrasts. State source/config comparability explicitly where it changes a conclusion.

Final decision JSON distinguishes COMPLETED, development NO_CONTINUATION_SIGNAL, NOT_TRIGGERED, numerical NOT_INFORMATIVE and ENGINEERING_BLOCKED. Label evidence STRONG/MEDIUM/WEAK, describe repeated and contradictory findings, competing explanations and the most discriminating next experiment. Promote DT-PGNA only if its independent ranking gain survives B,C and norm controls. Prefer the simpler route when complexity adds no measured value. No revival of PAIR/ECCM/gradient balancing/dynamicEMA and no unbounded follow-up experiments.

## 9. Numerical smoke refinement after review R1

The actual-data FP32 finite-difference probe encountered ReLU crossing at step1e-3 and cancellation at step1e-4. For the small-batch numerical check, copy the same FP32-initialized coefficients into a float64 probe model and use the already specified step1e-5 and rtol1e-3/atol1e-7. Also compare the original FP32 full sensitivity against the same-coefficient float64 sensitivity at the original FP32 tolerance rtol1e-2/atol1e-4. The full-batch smoke, all training, and all evaluation remain FP32. This replaces only the numerical probe's precision; it does not change an algorithm, a performance gate or a training setting. The orchestrator approved this correction after the independent reviewer identified the concrete smoke blocker.
