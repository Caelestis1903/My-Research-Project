# First-Round Design — Task-View Gradient Residual Momentum

Approved equations and edge cases: DESIGN_SPECIFICATION sections5.1–5.4. Lc is mean BCE over natural2B clean row-task entries. Lv=.5*(mean_B view0/task0 BCE + mean_B view1/task1 BCE); non-target view outputs do not enter Lv. Actual tail B is used.

a_it=(sigmoid(z_it)-y_it)*grad_e(z_it), R_i=RMS(e_i), delta_it=stop_gradient(.03*R_i*a_it/(||a_it||+1e-12)). If ||a_it||<=1e-10 or R_i<=1e-12, delta=0 and retain the entry. Probe temporarily reconnects gate input sensitivity without parameter .grad, then releases hooks/graphs. Training preserves source gate detach. Forward is embedding_theta(x)+delta.detach(); embedding parameters remain trainable. No second derivatives.

Compute g_c=grad Lc, g_v=grad Lv, r=g_v-g_c at identical theta BEFORE mutation, fixed parameter names/order, data gradients without WD. Active A={p: current g_c,p is not None}. Clean None skips parameter/WD/m/v/q/step and cap/random norms even with historical Adam state. Zero tensor is active. Active clean with view None uses g_v=0; view-only parameters are skipped and counted.

For active p, n_p is its own source Adam step AFTER this update, not a global batch counter:

    h_p=g_c,p+wd*theta_p
    m_p=.9*m_p+.1*h_p
    v_p=.999*v_p+.001*h_p^2
    q_p=.9*q_p+.1*r_p
    mhat=m/(1-.9^n_p); vhat=v/(1-.999^n_p); qhat=q/(1-.9^n_p)
    D=sqrt(vhat)+1e-8; u_c=mhat/D; u_r=qhat/D
    C=||u_c||; U=||u_r||
    s=0 if U==0 else min(1,gamma*C/U)
    b=s*u_r; theta_next=theta-lr*(u_c+b)

One global L2 norm over A uses detached FP64 accumulation; parameter/m/v/q stay FP32. C=0 or U=0 implies b=0. q has no WD. Reuse source Adam clean rounding/order then add residual in the same atomic transaction. EMA, observer and RAM commit follow the FULL update once. No second optimizer, microbatch step or .75/.25 gradient sent into TV-GRM Adam.

SHUFFLE-.10 computes real q and capped/preconditioned b on its OWN current theta/clean m/v. R=||b||, N=active coordinates. Independent keyed Rademacher xi(dataset,seed,epoch,batch,parameter_name,stream) gives b_shuffle=R*xi/sqrt(N), exact zero at R=0. Apply theta-lr*(u_c+b_shuffle). Match actual ||lr*b|| with FP32 relative error<=1e-5. Do not precondition, accumulate or cap the random vector again. Do not read TV-.10's state/amplitude. Real q supplies local amplitude only; cross-trajectory equality is not required.

INSTANT-.10 replaces qhat with current r: no residual history/bias correction. Clean Adam beta1=.9/beta2=.999, views, D and cap remain fixed. Report amplitude/cap differences; this comparison measures the whole residual-history channel.

Gamma0 takes canonical source clean with no views/probe/random and exact source model/Adam/EMA/RNG/loader equality over multiple steps and RAM resume. Compare source Adam separately from new q state.

Run gamma=.05/.10/.20, SHUFFLE-.10 and INSTANT-.10. Primary selected validation EMA Macro AUC vs CLEAN EMA; also online, total vs CLEAN online, direct historical NDP-A at matching readouts, .10 vs SHUFFLE/INSTANT. NDP-A changes both Adam moments and only identifies overall routing. Test stays sealed in every autonomous cycle.
