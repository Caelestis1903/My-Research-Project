# Three distinct mathematical candidates
Only candidate A is selected. B and C are complete research alternatives, not an instruction to implement or run them in parallel.

## Common protocol and optimizer semantics
Datasets, natural rows, split/support, fixed1 M3oE architecture, gates, experts, fusion and sharing remain frozen. Three domains, two tasks, 40,960 batch including tail, FP32, 50 epochs, seeds exactly 2022–2026. Let B be actual batch size, z_it a logit, p_it its probability, and ell_it the existing probability-BCE implementation. Lc=(1/(2B)) sum_it ell_it. Training gate inputs remain detached as in the source; only the existing sensitivity probe uses full gate-input derivatives.

All candidates use one coupled Adam step per batch, lr=.01, betas=(.9,.999), eps=1e-8, ML weight decay=1e-4, KR=1e-5; foreach=false, fused=false, amsgrad=false, decoupled_weight_decay=false. For each parameter p with supplied data gradient g_p (or recurrence output d_p):
h_p=g_p+wd*theta_p;
n_p <- n_p+1;
m_p <- .9*m_p+.1*h_p;
v_p <- .999*v_p+.001*h_p^2;
theta_p <- theta_p-.01*[m_p/(1-.9^n_p)]/[sqrt(v_p/(1-.999^n_p))+1e-8].
Moments and counters start at zero and are lazily created. A parameter with grad=None skips decay, moments, parameter update and counter increment. A present zero tensor is active, including decay. Never replace None with a zero tensor merely to make a flattened vector. Actual source torch.optim.Adam performs the arithmetic; no approximate hand-written optimizer is substituted.

EMA starts as an exact model copy, then once after each optimizer step: thetaEMA <- rho*thetaEMA+(1-rho)*theta; copy buffers. rho ML=.9828205985452511, KR=.995. No EMA bias correction or adaptive schedule. Select online and EMA independently by earliest strict maximum validation six-cell Macro AUC; equal scores retain the earlier model. LL is separately reported at the AUC-selected state.

The true update vector is U=concat(theta_after-theta_before), including all effects actually applied by Adam; use zero displacement only for measurement on skipped parameters. Its global L2 norm is sqrt(sum_p ||U_p||_F^2), with detached FP64 reductions for scalar diagnostics. A data-gradient norm/cosine excludes weight decay and is never called an update norm/cosine. None masks and parameter order accompany measurements.

## A — Task-axis even views (selected)
Motivation: NDP-A's mass/support repair did not repair selected-EMA AUC. Its one-sided supervised ascent objective combines odd and even effects. Existing residual channels confound this objective change with Adam preconditioning/capping and history.

Information flow: the frozen sensitivity helper returns detached embedding e_i, clean logit z_it, and J_it=partial z_it/partial e_i using its full-derivative probe. Training still uses the ordinary detached-gate graph. Compute r_i=sqrt(mean_j e_ij^2), b_it=(sigmoid(z_it)-y_it)*J_it, n_it=||b_it||_2. With alpha=.03,
delta_it=stopgrad(alpha*r_i*b_it/(n_it+1e-12)) when n_it>1e-10 and r_i>1e-12; otherwise delta_it=0.
This is the exact NDP-A make_views construction; do not mask fields, change label support or renormalize its delta. It perturbs the same complete embedding output as NDP-A. The discrete domain indicator and routing remain original.

Define Lplus=(1/(2B))sum_it ell(p_t(e_i+delta_it),y_it), Lminus analogously. Each target task has its own full-batch forward; only that task's BCE contributes from that forward. Define O=(Lplus-Lminus)/2 and E=(Lplus+Lminus)/2-Lc. The new objective is
L_A=.75*Lc+.125*Lplus+.125*Lminus = Lc+.25*E.
NDP-A at the same theta and detached delta is L_N= .75*Lc+.25*Lplus = Lc+.25*(E+O). The exact removed component is .25*O. These equalities concern the objective at a common state, not a claim that two divergent training trajectories have identical E.

If a locally smooth scalar loss is expanded with delta held fixed, O begins with its first-order directional term, while E begins with one-half delta-transpose-Hessian-delta. The finite-difference identities above are exact; a Hessian expansion at ReLU boundaries or through a reattached probe is an explanatory approximation. It does not establish an exact Hessian-penalty optimizer.

Control A1 AXIS-IID2: independently sample s_it1,s_it2 in {-1,+1} with equal probabilities and use
L_I=.75*Lc+.125*L(s_1 delta)+.125*L(s_2 delta).
Conditional on theta, batch and detached delta, E_sign[L_I]=L_A. Two full target-task views per task give the same graph count as A. Sampling is independent per row, task and replica, not one shared batch sign. A sign pair may coincide; do not force distinct signs.

Control A2 ISOTROPIC-EVEN: draw independent Gaussian vectors q_it with a private generator; set u_it=||delta_it||_2*q_it/||q_it||_2, with zero u for zero delta. If q has exactly zero norm and delta is nonzero, use the first coordinate unit vector times ||delta||; no retry loop. Train .75*Lc+.125*L(+u)+.125*L(-u). This matches **actual input perturbation norm** per row-task, not data-gradient, curvature, or Adam-update norms. It also runs the same task probe to obtain delta's radius; it does not obtain a cheaper computation budget.

All three use the common Adam and fixed EMA recurrence on the complete combined objective. No residual q, gradient projection, clipping, cap, extra optimizer, EMA adaptation, or second derivative is added.

Assumption audit: an even pair discards the sign of the task-BCE ascent direction. Apart from validity thresholds and the small normalization epsilon, its axis is normalized logit Jacobian; it does not preserve label-ascent information. Supervised targets still enter BCE. Useful task-axis curvature, damaging odd components, and endpoint benefits of sign variance reduction are hypotheses, not measured facts. Five forwards change computation versus NDP-A even though steps and loss mass match. AXIS-IID2 and ISOTROPIC-EVEN control the five-forward cost.

Simpler/stronger alternatives: unchanged CLEAN fixed EMA is the simplest; NDP-A supplies the exact one-sided comparator. A generic smaller lambda could mimic a weak regularizer, but is not an arm or an HPO rescue. Isotropic even views test whether task geometry contributes beyond smoothing. Earlier Gaussian ANTI2 reduced conditional variance without beating IID2; the new IID2 control is therefore essential.

Falsifiers: A fails to outperform NDP-A on both datasets -> the broad odd-component account fails its endpoint prediction. A recovers NDP-A but not CLEAN -> removal of damage without added utility. A does not outperform isotropic even views -> no demonstrated value of task-axis information. A and IID2 have comparable endpoints -> no demonstrated endpoint advantage from eliminating sign-sampling noise. Positive LL with nonpositive AUC does not rescue the primary objective.

Exact minimal matrix: three new arms x {movielens,kuairand} x {2022,2023,2024,2025,2026}, 50 epochs each; 30 new / 1,500 epochs / 51,750 updates. Reuse exactly ten CLEAN and ten NDP-A only after the identity gate. See execution plan for full deterministic implementation. Cost estimate 19.55–32.58 worker-hours from current NDP-A timings; initial memory reservations ML4.5 GiB/KR5.5 GiB replaced by measured peaks. Per-worker CPU memory includes current/selected/rollback states and IPC copies; record real tensor bytes before concurrency admission.

## B — Cell-centered output-error routing
Motivation: observed LL/AUC and pooled/Macro disagreements suggest testing a ranking-relevant gradient decomposition, rather than assuming a cosine conflict means useful interference.

Let b_it=partial ell_it/partial z_it from the actual probability-BCE graph, before 1/(2B) normalization; do not replace its numerically implemented derivative with ideal p-y. For each present domain-task cell C_dt, bar_b_dt=mean_{i in C_dt} b_it. Let J_it,p=partial z_it/partial theta_p through the ordinary training graph.
G_p= sum_it b_it J_it,p/(2B).
For parameters other than the six existing final tower scalar biases, use
Gcontrast_p=sum_it (b_it-bar_b_{d_i,t}) J_it,p/(2B).
For the six final biases, use their original G_p. Pass these data gradients through the common coupled Adam/EMA recurrence once.

A deterministic surrogate expressing the nonbias vector is S(theta;stop(b))=sum_it stop(b_it-bar_b_dt)*z_it/(2B); it is a gradient-routing surrogate evaluated at the current model, not a new claim about minimizing Lc exactly. Evaluate/report the original BCE Lc. Obtain b by autograd with retain_graph=true, no create_graph; VJP the centered coefficients; preserve the source clean gradients only for final biases. Do not add calibration parameters, alter the tower, or center features/parameter tensors.

A present singleton cell has centered coefficient zero but retains its natural source bias gradient. Empty cells are omitted. Preserve original grad=None masks; a source-connected zero tensor remains active. No normalization by domain population across cells: each original row-task retains denominator 2B.

Control B0 SOURCE-PASSTHROUGH performs the source ordinary gradient in the same new package, unchanged input/Adam/EMA. Control B1 PERMUTED-GROUP uses a private CPU permutation of the batch domain-label vector, preserving all three group sizes; it uses these labels only to form bar_b, never for model input or routing. Labels and examples remain original. The random grouping applies to both tasks and uses the same key specification as A with stream "group". Restore the same true six bias gradients. TRUE-CENTER and PERMUTED-GROUP each use one forward and the required VJPs; no view construction.

Nonexistent-assumption audit: a constant logit shift preserves within-cell AUC, but suppressing a gradient component in a shared nonlinear model is **not** guaranteed to induce only such a shift. Better calibration may support ranking; removing its feature gradient may hurt both. Domain means are an output decomposition, not evidence that small domains are difficult. Existing Gradient Centralization centers parameter-coordinate gradients; this method centers samplewise output-error coefficients and preserves existing biases.

Simpler/stronger alternatives: source Adam fixed EMA; an explicit pairwise ranking loss would change the training objective and require a separately designed control, so it is not smuggled into this matrix. Generic loss reweighting, PCGrad, gradient whitening and norm balancing are different mechanisms.

Falsifier: TRUE-CENTER does not improve selected fixed-EMA Macro AUC over source control on both datasets, or behaves like arbitrary PERMUTED-GROUP. Any LL deterioration is reported even with positive AUC. A positive mean at only one epoch or pooled metric is insufficient.

Exact minimal prospective matrix: {SOURCE-PASSTHROUGH,TRUE-CENTER,PERMUTED-GROUP} x both datasets x five exact seeds, all50 =30 new /1,500 epochs/51,750 updates. Source passthrough is explicitly rerun because this unimplemented derivative-routing path has no identity receipt. No NDP-A reuse as a causal control. Current CLEAN costs give 6.514 worker-hours for three single-pass blocks and a roughly 10.856 worker-hour planning point if the two routed arms cost twice CLEAN. Additional storage is at most two detached P-element FP32 gradient vectors plus one retained clean graph and Bx2 coefficients; measure the model's P, use 1.5x current CLEAN reserved peaks (about1.47/1.84 GiB) as an initial planning estimate, and replace it with full-batch smoke. These are estimates, not measured bounds. B is not scheduled.

## C — Same-batch transported gradient memory
Motivation: TV's augmented-minus-clean history has weak mixed evidence. A different temporal question is whether tracking the change in the **clean gradient at two parameter points on the same batch** helps. This is a STORM-derived recurrence, not a novelty claim.

Objective remains Lc, no augmentation. At step k, with current theta_k and saved pre-update theta_{k-1}, compute g_k=grad Lc(theta_k;B_k) and h_k=grad Lc(theta_{k-1};B_k), both excluding weight decay and using identical row order and ordinary detached-gate graphs. For the first step, d_1=g_1. Thereafter, for a continuously active parameter,
d_k=g_k+.9*(d_{k-1}-h_k)
    =.1*g_k+.9*d_{k-1}+.9*(g_k-h_k).
Feed detached d_k as data gradient to the common Adam step. Apply coupled decay only there, once; do not include it in h or in recurrence memory. Save a detached full copy of theta_k and its buffers before updating for the next batch. The previous-model copy has no optimizer or EMA. No derivatives through h, d history or the old model.

Parameter semantics: if current g_p=None, set supplied grad=None and invalidate that parameter's transport history. On the next active occurrence reset d_p=g_p. If current g_p is active but old h_p=None, reset d_p=g_p. All present zero tensors remain active. Adam's own counter n_p increments only on active supplied gradients. There is no extra bias correction of d; it starts at the first gradient. Adam's usual m/v bias corrections still apply. Fixed EMA is unchanged.

Control C1 LAG uses d_1=g_1, then d_k=.9*d_{k-1}+.1*g_k with identical activity resets. It computes h on the same old model and discards it to match transport's forward/backward count. Control C0 SOURCE-ADAM uses ordinary clean g and its own standard optimizer state. There is no learned or tuned recurrence coefficient; .9 is frozen, not borrowed as an effective best value.

Information flow is same batch/two model states -> temporal derivative correction -> ordinary Adam -> fixed EMA. Unlike TV-GRM it contains no augmented view, residual cap or post-Adam injection. Unlike adaptive EMA it changes the incoming data-gradient estimator. Unlike AdaBelief it leaves Adam's second-moment equation unchanged. No inference-time change.

Assumption audit: STORM literature supplies this recurrence family, not an efficacy theorem for fixed1 M3oE with an additional Adam filter. Nested momentum can amplify stale or biased estimates. FP32 nonlinear old/current evaluations can change masks. Existing residual/INSTANT evidence does not measure this same-batch transport term. No assumption is made that momentum is effective.

Simpler/stronger alternatives: ordinary source Adam; LAG isolates an additional low-pass filter; increasing batch size is disallowed and would change data/compute. Falsifier: TRANSPORT fails to improve both source Adam and LAG on selected EMA AUC in both datasets. LL, online and fixed-epoch contrasts remain separate.

Exact minimal prospective matrix: {SOURCE-ADAM,LAG,TRANSPORT} x both datasets x five exact seeds x50 =30 new /1,500 epochs/51,750 updates. All are new source-matched runs; no universal historical optimizer baseline. The two double-evaluation arms plus one CLEAN arm give about10.856 worker-hours from measured CLEAN means (planning, not measured). Sequential graphs permit approximately CLEAN activation memory plus previous model and d (8P bytes FP32, plus masks/buffers); an initial estimate is1.5x CLEAN reserved peaks, replaced by measured smoke and explicit tensor accounting. Host snapshots also carry old parameters and d. C is not scheduled.

