# C05 candidate mathematics and complete experimental alternatives

Only candidate U is selected for implementation. V/G are fully specified research alternatives, not queued runs. The parent dispatches a fresh designer to turn the chosen research plan into the formal DESIGN SPECIFICATION. No alternative substitutes after observing C05 results.

## Common frozen protocol and Adam recurrence

Natural ML/KR train/validation; M3oE fixed1,3domains×2tasks, source architecture, routing, fusion and ordinary gate detach. Seeds exactly[2022,2023,2024,2025,2026], 50epochs, batch40960 with tails21927/40052, FP32, source TF32 settings, initialization and seed+11000 private CPU loader unchanged. No sampling/split change or synthesized training rows. ML20/KR49 updates/epoch. No Test access.

Let b index the natural minibatch of B rows, c=(d,t), n_c its rows, π_ca=n_ca/n_c, w_c=n_c/(2B), ell_it=source probability-space BCE(p_it,y_it), C_c=Σ_aπ_ca*mean(ell|c,a), C=Σ_cw_c C_c. A missing cell contributes zero, with statistics null. Class counts/coefficients have no model gradient. No score-dependent weighting or validation information enters the update.

For every active parameter p, input data gradient d_bp is followed by unchanged coupled source Adam:

    z_bp = d_bp + wd*theta_bp
    k_p <- k_p + 1
    m_p <- .9*m_p + .1*z_bp
    v_p <- .999*v_p + .001*z_bp^2
    theta_p <- theta_p - .01 * [m_p/(1-.9^k_p)] /
                               [sqrt(v_p/(1-.999^k_p)) + 1e-8]
    thetaEMA <- rho*thetaEMA + (1-rho)*theta

wd=1e-4 ML /1e-5 KR; rho=.9828205985452511 ML /.995 KR; EMA initialized from online. Source Adam foreach/fused/amsgrad=false. Equation is explanatory: production uses the frozen installed source operator and preserves its floating-point ordering. `grad=None` means no Adam update, no decay, no private-step increment for that parameter; a present zero tensor is an active gradient and preserves the source behavior. Exactly one live Adam and one live EMA update per natural batch. Online/EMA independently select earliest strict validation six-cell MacroAUC maximum over E1..50; all selected losses/cells refer to that same state. Common endpoints E1/10/30/50 supplement full curves.

## U — CELL-MASS-BCE-025, chosen

### Objective and information flow

Fix λ=.25, inherited from the completed attribution chain, with no search. Define:

    q_ca = 1-lambda + lambda*pi_ca
    kappa_c = sum_a pi_ca^2
    o_c = 2*pi_c0*pi_c1
    a_c = sum_a pi_ca*q_ca = 1-lambda*o_c
    m_b = sum_c w_c*a_c
    W = sum_c w_c sum_a pi_ca*q_ca*mean(ell|c,a)
    U = sum_c w_c*a_c*C_c
    S = m_b*C

This W is C04 COMPLETION-BCE-025; production historical W used `.75*C+.25*ΣwH` in its frozen reduction order. U is built directly from the ordinary unreduced probability BCE, multiplying **every row/task in cell c by a_c/(2B)**. Do not construct U by subtracting two large objectives, divide by m_b, renormalize tasks/cells, or reconstruct W in another reduction order when testing reference identity.

For all cells, `.875<=a_c<=1`; for the global mass `.875<=m_b<=1`. U preserves W's exact cell mass w_c*a_c, hence each task's summed mass and the global mass m_b. Class masses in U are w_c*a_c*pi_ca. They intentionally differ from W's w_c*pi_ca*q_ca. This is class homogenization, not equal-class balancing; natural label ratios and all2B task entries remain present. Total mass is not one. No new draw, forward, model tensor, momentum channel, routing change or hyperparameter is introduced.

The exact local decomposition is:

    W-U = lambda*sum_c w_c*Cov_{a~pi_c}(pi_ca, mean(ell|c,a))
    U-S = sum_c w_c*(a_c-m_b)*C_c
    grad(W)-grad(U) = lambda*sum_c w_c sum_a pi_ca*
                       (pi_ca-kappa_c)*grad(mean(ell|c,a))

The within-cell residual has zero coefficient mass, not necessarily zero loss or gradient. U−S changes cell/task mass at fixed global mass. A complete trained U−W contrast changes only this registered coefficient policy at initialization; later parameters, moments and selected epochs are consequences of that intervention. Endpoints telescope arithmetically but do not give mediation fractions or the missing S+R interaction.

For balanced two-class cells π0=π1=.5, W and U are mathematically identical locally. For one-class cells, a_c=1 and source coefficient is preserved. Empty cells have no rows and zero mass; absent-class means/variances remain null, never fabricated zeros. For a batch with all observed cells single-class, or engineering λ=0, delegate to the exact source CLEAN operator for exact model/Adam/EMA/RNG identity. In general mixed batches, compute U directly; arithmetic equivalence tolerances do not license a different production source path. Do not invoke model.forward again for counts or diagnostics.

Information flow: natural labels/domain IDs -> detached counts -> six a_c coefficients -> ordinary BCE -> one scalar backward -> source Adam -> source fixed EMA -> validation selection. Validation does not influence a_c, λ or training.

### Motivation, assumptions and rivals

C04 W−S task1 LL benefits on both datasets are MEDIUM evidence that a common global m alone is insufficient. KR d0t0 loss harm alongside d2 benefit leaves class prior versus cell/task mass as a concrete rival. U removes one component without guessing a difficulty ranking. The efficacy hypothesis remains WEAK until five-seed training.

If fixed population class weights q0/q1 were external constants and a separate cell predictor had unlimited capacity, weighted BCE would minimize at p*=q1*eta/[q1*eta+q0*(1-eta)], shifting posterior log-odds by log(q1/q0). A class-independent positive scalar leaves that ideal optimum unchanged. This is a derived motivation, **not** a theorem about the finite shared M3oE trajectory: batch priors include the observed labels; parameters are shared; coupled decay and finite training matter. Pure constant logit shifts would not change within-cell AUC, so observed ranking differences must involve more than that ideal calibration argument.

Simpler alternatives are C and S, already fully available. Globally normalizing W/m would change both scale/Adam response and class policy, defeating this test. Stronger alternatives are S+R factorial completion and W-fixed R timing; their distinct causal questions are compared in RESEARCH_EXPLORATION. A further task-only aggregation would split cell from task mass, but no current result requires that additional arm to answer the chosen class question.

### Falsifiers and outputs

F1 (candidate): U EMA−C EMA fails either dataset's practical MacroAUC/cell gates after all five seeds -> no qualified cross-dataset candidate. A negative or mixed result still completes C05.

F2 (cell/task sufficiency direction): predict U−S selected EMA task1 LL mean<0 in each dataset. If either mean>=0, that dataset fails the directional prediction. If W−U task1 LL remains negative while U−S is nonnegative, the W-associated benefit is consistent with needing its class component under this protocol. Report all five deltas/intervals; do not infer equality from a crossing interval or universal necessity from one failed U version.

F3 (class-skew harm direction): predict KR U−W selected EMA d0t0 AUC>0 and LL<0. Nonpositive AUC or nonnegative LL rejects the corresponding rescue prediction. Also report d2 and dominant d1, both tasks, ML analogues and all stage endpoints. Improvement of d0 coupled to loss elsewhere is a trade-off, not a free rescue.

F4 (strong class necessity explanation): U−S task1 LL<0 on both datasets, with retained W-relative rank/probability behavior, weakens class skew as a necessary source of the task benefit. No posthoc equivalence margin is invented. Coexisting smaller class effects remain possible.

F5 (technical identification): any mismatch in six cell masses W/U, natural support/order, source init, or uncontrolled live diagnostic mutation invalidates the intended contrast and requires correction before scientific interpretation.

Report full objective masses by batch/cell/task/class, ordinary C/U/W/S loss values where computed, selected and common AUC/LL, per-cell/per-task/per-domain, macro/traffic/pooled/worst, selection epochs/hashes, source identity, optimizer/EMA update counts and costs. On U's first ordinary batch of E1/10/30/50, same-state diagnostics evaluate gradients and disposable source-Adam steps for C/S/U/W: 40 points,120 extra autograd sweeps,160 virtual steps. U's virtual/live state hashes match exactly. No additional forward, live EMA or RNG draw. Record ALL and frozen existing parameter groups, data-gradient/update norms, U−W/U−S differences and cosines, activity counts/None distinctions, null reasons. These local diagnostics do not supply absent historical U-host data or identify trajectory mediation.

### Complete chosen matrix and cost

| Arm | ML seeds | KR seeds | Epochs each | Source | New trajectories |
|---|---|---|---:|---|---:|
| U CELL-MASS-BCE-025 |2022,2023,2024,2025,2026|2022,2023,2024,2025,2026|50|fresh approved source initialization|10|
| C CLEAN |same five|same five|50|exact numeric controls through C04/C03 index, original source identity preserved|0,10 reused|
| W COMPLETION-BCE-025 |same five|same five|50|C04 attempt2 numeric results/50 ACK epochs each|0,10 reused|
| S MASS-SCALE-025 |same five|same five|50|C04 attempt2 numeric results/50 ACK epochs each|0,10 reused|

New work: ML250epochs×20=5,000updates; KR250×49=12,250; total500epochs/17,250updates,701,574,750training rows,1,403,149,500task entries. 17,250 ordinary forwards/backwards/Adam/EMA each; 5,000 source validation chunks (ML1,500, KR3,500),175,394,000 evaluated validation rows across both streams. Twenty selected model states in ten pairs; source-size forecast95,508,960 tensor bytes, to be measured rather than used as a storage guarantee. Engineering/replay/partial work is additional physical cost and receives no formal completion credit.

Asymptotic overhead over C is O(BT+DT) scalar reductions/multiplies, with BT=2B and DT=6; no extra ordinary forward/backward. Temporary unreduced losses/weights cost O(BT); optimizer remains O(P). Sequential diagnostic clones need O(P) extra peak state, not four simultaneous clones. Same-state diagnostics use120 extra sweeps/160 virtual source Adam operations only at40 registered points. Full-process GPU/host/serialization envelopes must be measured in the new source's real full/tail/validation/RAM smoke, not inherited as current measurements.

## V — predictive-KL virtual view, considered and not selected

Motivation: label-BCE ascent in NDP-A and the odd/even axis variants did not establish matched-EMA gain; a direction built from prediction curvature tests a different information source. Contradictions are the weak local view results, known teacher-view failures and no evidence that local smoothness increases AUC in these domains. Evidence for usefulness here is WEAK. This is a VAT adaptation, with no global novelty claim.

Define the existing embedding boundary h_i, scalar r_i=.03*||h_i||2, and q_it=stopgrad(clip(p_it,1e-6,1-1e-6)). For each row sample a unit Gaussian vector z_i using a private generator, seed+22000, independent of loader/model RNG. Probe at h_i+xi_i*z_i, xi_i=1e-3*||h_i||2. Use the ordinary detached-gate computational derivative, unchanged on clean/probe/final passes. Let D_i(delta)=.5*Σ_t KL(Ber(q_it)||Ber(clip(p_t(h_i+delta),1e-6,1-1e-6))). Form v_i=grad_delta D_i at xi_i*z_i and u_i=v_i/||v_i|| if nonzero; otherwise u_i=z_i. If ||h_i||=0, delta_i=0. The perturbation is delta_i=stopgrad(r_i*u_i). Train:

    L_V = C + .25 * (1/B) * sum_i D_i(delta_i)
    d_b = grad_theta L_V; source Adam and fixed EMA as above.

The matched random-KL control uses delta_i=stopgrad(r_i*z_i), otherwise identical loss and support. No model-history teacher, label pairs, supervised view BCE, projection acceptance, domain-specific sampling or gradient-norm matching is used. At radius0 the KL vanishes and delegate exactly to C. Probe gradients are used only to construct detached delta; the clean q side is detached. Labels enter C but not view geometry. No parameter-gradient second derivative is taken. Only the ordinary clean pass advances source nonparameter buffers/model RNG: extra probe/final passes retain source training flags but use disposable copies of those buffers and the post-clean model RNG state, restoring the live values afterward. Private perturbation RNG advances explicitly and is separately retained. This prevents extra BatchNorm/dropout state changes from becoming an unregistered recurrence. Local KL curvature, not BCE label ascent or even-view averaging, is the material distinction from NDP-A/AXIS. This also differs from historical teacher-constrained acceptance: it has no teacher, accept/reject subset or added labeled auxiliary mean.

Assumptions: infinitesimal prediction smoothness might be useful, the one-iteration direction approximates a local KL-sensitive axis, source gate detach defines the relevant surrogate derivative. These are hypotheses, not established domain semantics. Simpler alternative random-KL is included; C has no view cost. Stronger exact eigenvector or multi-iteration VAT would add unmotivated compute and is excluded.

Falsifiers: VAT−random selected EMA AUC nonpositive on either dataset fails the directional information claim there; VAT−C fails the shared primary gates -> no qualified increment. A low KL or a nonzero perturbation alone is not success. Report six-cell costs/benefits, loss, full stages, actual radii/KL and RNG counts. No assumed small=hard relationship.

Complete alternative matrix: VAT-KL-025 and RANDOM-KL-025, each ML/KR×seeds2022–2026×50epochs (20new,1,000epochs,34,500Adam/EMA); C10 numeric controls. VAT has one clean, one probe and one final-view forward per batch, one probe-input sweep and one final scalar backward; random-KL has one clean and one view forward, one scalar backward. Across the complete alternative:86,250 model forwards,17,250 extra probe sweeps,34,500live backward/Adam/EMA,1,403,149,500natural training rows. Views repeat these rows internally rather than changing the training population. Sequential probe/final graphs cost O(B*dim(h)+P) beyond the clean activation peak. Private RNG state stays RAM; full actual memory is measured. New radius/xi decisions and known VAT overlap make this lower priority than U; these constants are fixed for comparison, not a search grid.

## G — signed two-step gradient filtering before source Adam, considered and not selected

Motivation: C04 shows nontrivial data-gradient versus Adam-update response, while repeated stage reversals leave optimization history open. TV's extra residual EMA and historical A_MATCH feedback did not establish a transferable incremental mechanism. A finite impulse filter tests lag/lead without adding a new optimizer, adaptive EMA or view residual. Evidence for beneficial one-step prediction is WEAK; batch variation can dominate temporal drift.

Use the exact W objective, compute its ordinary data gradient g_bp. For each parameter with a present gradient and a present gradient on the immediately preceding global update, store h_p=g_(b-1)p detached and form delta_bp=g_bp-h_p. On first use or after any absent-gradient step, delta_bp=0. If current gradient is None, pass None to Adam and mark history invalid. Tensor-zero is present. The two arms are:

    LEAD: d_bp = g_bp + .5*delta_bp = 1.5*g_bp-.5*h_p
    LAG:  d_bp = g_bp - .5*delta_bp = .5*g_bp+.5*h_p
    after constructing d: h_p <- detached g_bp

Then source Adam adds coupled decay and updates both m and v from d, followed by the unchanged fixed EMA. No parameter-space post-Adam injection, clean-v-only preconditioner, cap, residual EMA, per-domain optimizer or validation feedback exists. The coefficient sum is one in both filters; the local departures from g have equal norm and opposite sign. Their complete data-gradient norms need not match. This is a finite two-step gradient filter inspired by optimism, not the exact published Optimistic Adam update.

Assumptions: consecutive gradients contain useful low-frequency drift versus high-frequency noise; LEAD corrects lag, LAG attenuates it. For white independent gradient noise LEAD has variance factor2.5 and LAG.5 relative to the current-gradient variance, so noise amplification is a concrete competing explanation. Nonconvex/coupled Adam destroys simple guarantees. The source Adam betas and fixed EMA are unchanged, but input recurrence changes both moments.

Simpler alternative is historical W with its ordinary Adam. LAG is a stronger explicit smoothing competitor than an unstructured shuffled residual; no new auxiliary objective is introduced. A true second-moment optimizer replacement such as AdaBelief would violate the frozen source-Adam requirement and is not proposed for C05. Same-state gradients, updates and activity are measured; no geometrical norm observation substitutes for AUC.

Falsifiers: LEAD−LAG selected EMA AUC nonpositive on either dataset rejects the directional lag-correction prediction there. LEAD−W nonpositive and failure versus C rejects a useful fixed configuration even if a local response is faster. Strong LL/AUC divergence or worsening high-noise cells undermines a generic noise-removal explanation. A late endpoint rescue without selected increment does not qualify.

Complete alternative matrix: W-LEAD-050 and W-LAG-050, each ML/KR×seeds2022–2026×50epochs (20new,1,000epochs,34,500updates); exact W10 and C10 numeric controls. 34,500ordinary forwards/backwards/Adam/EMA,1,403,149,500training rows; O(P) arithmetic and one previous data-gradient tensor plus activity flags per parameter, approximately4P extra FP32 bytes. No extra ordinary pass or RNG. This alternative is not implemented or run in the chosen cycle.
