# DESIGN ADDENDUM — equivalent reader coordinates and frozen delivery measurements

Version: NUMERICAL_DELIVERY_V1, 2026-10-26. Status: DESIGN APPROVED CONDITIONALLY; implementation, actual-matrix verification and independent review remain required. This supplements DESIGN_SPECIFICATION.md and the published 04/05/08 specifications. It introduces no new research arm, seed, training pass, hyperparameter candidate, scientific objective, promotion rule or outer-data selection.

## 1. Observed failure and narrow scope

Four S0 prefixes are complete and remain in RAM in the existing campaign service (parent reports PID1974641). S1 MovieLens seed5201 failed in FULL_SOURCE, target0/task0/group0, lambda .001, n=2359. The actual failure receipt is `artifacts/mdmt_rcf_20261026/v1/failures/movielens_5201_1793006522.json`: L-BFGS-B reached its iteration limit after the registered300 and same-start600 numerical retry. No criterion may be relaxed to turn that event into PASS. This is a numerical fit failure, not a negative research result.

Existing `readers.fit_group` evaluates the intended original FP64 objective. Its unbounded FULL feature coordinates may be correlated or badly scaled; that hypothesis must be checked on the **actual failed P_read matrix**, not only a toy fixture. The approved repair is an invertible coordinate transformation of free FULL coefficients. Alpha/beta constraints, the actual composite-correction RMS penalty, and the original1e-6 coefficient ridge remain unchanged.

Before numerical implementation is deployed, capture b, X, y, lambda, original bounds, original start, config/row/feature/source hashes and exact method/target/task/group from the resident failed computation. This is authorized training-role experiment data, not a model/checkpoint. The fixture must contain no native/pilot/B/optimizer parameter state. Record n, feature count, finite checks, column scales, rank and eigen/condition diagnostics from that actual matrix. Do not access Q/test metrics to select a numerical method.

## 2. Approved first repair: free-w Hessian whitening

For one FULL group, the unchanged original objective is

\[
J(\alpha,\beta,w)=\operatorname{mean}\{\log(1+e^a)-ya\}
+\frac{\lambda}{2}\operatorname{mean}(Xw)^2
+\frac{r}{2}w^Tw,
\quad a=\alpha b+\beta+Xw,\quad r=10^{-6}.
\]

Alpha remains directly optimized in[.25,4] and beta in[−4,4]. Apply this transform only when **all** content-coordinate bounds are(None,None), as in FULL_SOURCE, FULL_SOURCE_ALL, FULL_TARGET and RANDOM_FEATURE. Ordinary bounded-unit readers retain their existing coordinates; no transformed box approximation is permitted.

Use the fixed original initialization theta0=(1,0,0), p0=sigmoid(b), v0=p0(1−p0). In FP64 form and symmetrize the free-w Hessian at that initialization:

\[
H_0=X^T\operatorname{diag}(v_0)X/n+\lambda X^TX/n+rI.
\]

Compute its symmetric eigendecomposition and P=H0^(−1/2). The ridge makes this positive definite in exact arithmetic. Check finite eigenvalues, symmetry and reconstruction/conditioning; do not silently alter an invalid Hessian. Numerical eigendecomposition roundoff handling, if any, must be disclosed and restricted to making this **preconditioner** invertible; it cannot change J, r or model bounds. A failure to construct a numerically invertible P is reported, not hidden by changing the ridge.

Optimize z=(alpha,beta,u) with **w=P u**. Original alpha/beta and their boxes are unchanged, and u is unbounded. The initial point remains(1,0,0). Evaluate J by mapping to original w, preferably through the existing `objective_gradient`, and transform only its gradient:

\[
\nabla_z J=(g_\alpha,g_\beta,P^Tg_w).
\]

In particular the original ridge becomes r*uᵀPᵀP*u/2; replacing it by r*||u||²/2 would change the research method and is forbidden. The actual correction is X P u, so composite RMS regularization also remains exactly the same. Do not center X, absorb terms into alpha/beta, or add a Schur block transform in this first repair. Return and store analytical reader coefficients in original coordinates (alpha,beta,w), with a preconditioner hash and numerical-revision receipt.

Keep the existing L-BFGS-B settings: maxiter300, same-start retry maxiter600, configured gtol1e-8, existing ftol1e-12 and maxls40. No extra λ/start, larger iteration allowance or looser tolerance is approved. Whitening is deterministic from the authorized training matrix and original start; it is not selected by outer performance. If this free-w-only repair still cannot satisfy the registered numerical contract, report its actual objective/condition/KKT and return to design before any different transformation or budget change.

## 3. Required actual-matrix evidence before deployment

The implementation report must contain the following small numerical receipt, explicitly separated from research performance:

1. Fixture identity equals the failed group and P_read rows; original matrix/bounds/config/source hashes are recorded. No outer labels or metrics are used.
2. At original start and several fixed nonzero vectors, verify mapped original logits, objective, composite penalty and original ridge agree with direct original-coordinate evaluation. Target numerical agreement is FP64 relative1e-10 with absolute1e-12 for scalar objective components; report actual errors and diagnose arithmetic-order deviations instead of changing the scientific objective.
3. Verify transformed analytical gradient equals the original gradient under the exact chain rule and a focused finite-difference directional check. Verify the inverse map reproduces original w within normal FP64 error. Alpha/beta feasibility and unbounded free-w coverage must be identical.
4. Fit the actual failed matrix using the registered300/600 limits and fixed original start. Report solver status/message, evaluations/iterations, J, original alpha/beta/w bounds, original-space free gradient and original-space projected KKT residual. Report transformed-space residual separately; do not substitute it for original-space KKT.
5. Preserve the existing solver acceptance thresholds. In particular, a failed SciPy termination with original projected KKT>1e-5 remains a failure. A SciPy success alone must not be described as satisfying original gtol1e-8 unless the measured original KKT actually does. No success flag or model metric may conceal a material original-space residual.
6. An independently reviewed focused integration check verifies that returned original coefficients reproduce the fitted logits and that the FULL method bound mask remains unbounded. Existing complete-six-head P_cal/OFF selection and all formation contracts are unchanged.

These checks justify the equivalent numerical implementation; they do not establish source usefulness. Do not require the failed original solver to reach a high-precision optimum to prove coordinate equivalence: equality of J/gradient/feasible set is the relevant evidence. Retain the actual original failed attempt receipt and revised numerical attempt, rather than overwriting it.

## 4. RAM-preserving repair and honest identity

Do not write model/checkpoint/optimizer files or kill/restart the resident workers to install this change. A narrow trusted-local maintenance helper may extract the failed training matrix, reload the reviewed numerical implementation and resume the failed S1 reader using existing RAM objects. Its actual IPC/injection mechanism and commands must be independently reviewed before use; this design approves the scope, not an unseen arbitrary-code channel.

The helper must target only the current campaign's explicitly recorded process IDs, verify current-user ownership and campaign identity, use an owner-only local socket/channel, and restrict operations to the reviewed fixture/numerical reload/remaining-S1/observational-export functions. No remote/untrusted code or pickle input, no model serialization, no changes to other-user or unrelated processes. Any temporary debugging suspension must be released on completion/failure. Record original/revised module SHA, worker IDs, exact operation list, timestamps, and retained native/pilot/B state hashes before/after; they must match. If this safe in-memory path is unavailable, same-identity deterministic replay is the last resort and its actual extra cost must be reported.

Resume at the failed FULL reader and then complete remaining S1 methods. Reuse completed B/direct/probability fits and their actual prior policy/results; do not repeat all S1 training or silently give extra target-label passes merely because a command restarts. A deterministic repetition needed solely for a verified lost fit is labeled replay, with the additional actual label/step/cost ledger. All subsequent FULL fits should use the same reviewed equivalent coordinate policy, independent of their positive/negative results. Other completed valid methods need no unrelated rerun.

Initial IMPLEMENTATION_IDENTITY and S0 receipts remain historical facts and are not rewritten to claim the new reader existed earlier. Add a supplemental implementation identity with prior identity, changed file hashes, equivalent-objective justification, actual reviewed scope and affected methods/stages. Stage receipts distinguish the native/prefix code actually executed from numerical reader code actually used. Review authorization must cover those revised files before affected operations continue.

## 5. Required observational frozen-predictor delivery

The original delivery requires measured full-forward cost and parameter accounting. Missing observations can be collected from frozen resident predictors without new training, labels, model selection or research arms. This is approved observational delivery work, subject to the same focused review and identity supplement.

Use one fixed raw-X batch of4096 natural rows per dataset, with recorded row IDs/hash and feature schema. Keep the same rows/input placement, inference mode, device, dtype, threading, transfer policy and synchronization for all compared methods. Do not pick an easier batch for a preferred method; no labels are needed. If batch construction requires multiple roles, choose by a predetermined row rule and record it. Do not use repeated/tiled rows to manufacture4096 when natural eligible rows exist.

Primary timing is raw-X→native H/Z→required base/pilot/delta components→final p, not cached H/Z→reader. Construct the deployment predictor once outside timed sections. Use3 warmups and10 measured repetitions, fixed in advance; synchronize the relevant CUDA device before/after each timed forward, including transfers if the chosen declared end-to-end scope includes them. Report each repetition, mean/median/spread, batch size, rows/s, device/environment and measured peak CUDA allocation. CPU-only components and GPU/CPU transfers actually required by the deployment count. Cached-head-only timings may be supplemental and must be named accordingly. No training or optimizer time is included in inference timing.

FULL controls must receive the fair algebraic implementation already allowed by05: fuse each group's learned linear feature coefficients with the appropriate frozen source/target/random pilot hidden-output readout; account for source-fit feature centering/scaling exactly. For standardized hidden phi=(hidden−mean)/std, use effective hidden weight w/std and intercept−sum(w*mean/std), preserving ReLU before the fused linear readout. Do not fuse through a nonlinear activation or drop native/B/pilot computation. Apply the same valid linear coefficient folding opportunity to formed functions where applicable. Group choice still uses the frozen native threshold.

Before timing, compare the fused full-forward p with the retained unfused frozen predictor/raw-X validation on the identical batch. Max absolute probability difference must meet the existing1e-6 raw-X acceptance. Report the measured difference; if it fails, repair/review the algebra or report timing as unavailable, rather than changing tolerance. Timing/fusion cannot refit any coefficient or change policy/OFF selection.

Parameter accounting must separate stand-alone deployed total (shared native counted once), trainable additions during this method, frozen added components, per-target active parameters, reader-only parameters and shared-prefix research overhead. For an OFF policy, count the components actually needed for exact B_refit deployment. FULL inference does not need frozen pilot output parameters if an exact fused hidden reader removes their use; training source-pilot costs still count in end-to-end training cost. State each counting formula; do not add shared-prefix cost once per arm and then claim it as total campaign cost. Include completed native/pilot/base/formation/reader time and any real numerical/replay cost as their actual scopes allow. A new benchmark implementation is labeled as such and cannot retroactively certify an unmeasured old predictor.

## 6. Acceptance and limits

Approved changes are limited to equivalent unbounded-FULL solver coordinates, reviewed RAM-preserving deployment/recovery, and missing observational cost/parameter/raw-forward exports. Actual-matrix equality and KKT receipts plus independent PASS/PASS WITH FIXES with blocking issues corrected are prerequisites to revised S1 execution. If free-w whitening fails, return the actual evidence for a further narrow design decision; do not increase solver budgets or try a family of transforms guided by Q.

All scientific definitions, lambda grid, coefficient bounds in original space, data roles, seeds, native/formation training budgets, source/RMS constraints, six-head selection, stages/gates and final analysis obligations remain those of the original design. No new performance claim is made in this addendum.

## 7. Execution resolution: externally lost resident service

Execution update, 2026-10-26: the parent orchestrator and R0 independently report that the prior service/worker PIDs are absent from `/proc`, and the campaign's CUDA allocations are gone after runner-turn session cleanup. Consequently no prior native/pilot/B/direct/probability-reader fitted object remains available in RAM. Existing experiment data, logs and receipts remain. This is external process/state loss, not a scientific failure or an algorithm-selection signal.

The requirement in section4 to reuse completed RAM S1 fits is now **UNAVAILABLE_DUE_TO_PROCESS_LOSS**. Do not describe missing objects as reused or restored. Archive and preserve the original attempt receipts and measured costs. Authorize one controlled deterministic replay of the SAME dataset/seed/code/config/data/row-plan identities to reconstruct the necessary prefixes and frozen S1 fits. It is not a new seed, independent training repeat or extra statistical observation. Restart native/pilot/head construction from their original initialization and original fixed schedule; do not append passes to an earlier trained state or average predictions across attempts. Record reconstruction time/label accesses/steps as real replay cost, separate from the one intended scientific trajectory.

For each recreated prefix, compare native selected/terminal and pilot hashes against the prior retained state-hash receipts wherever available. Report exact matches and explicit missing comparators; if a relevant hash differs, stop affected downstream reuse and diagnose the implementation/environment/row-plan mismatch before proceeding. Do not silently call a mismatch identical. B/direct/probability fits are reconstructed using their same fixed identity and schedule after prefix replay; prior RAM-fit reuse is not claimed.

Before replay, launch the campaign as a detached external process with `start_new_session=True`, outside tool-owned PTY/session lifetime. Record and verify actual service/worker PID, PPID, session identity and process liveness across the launching tool's return. This does not authorize managing unrelated processes, changing global process defaults or writing model/optimizer/checkpoint state. All training state remains RAM-only. The durable launch must be checked before committing to full replay so the same tool cleanup mechanism is not knowingly repeated.

Normal campaign RPC operations for failed-matrix capture, reviewed reader-module reload and observational cost collection may be added and independently reviewed before launch. The old pickle-based maintenance proposal/code was **NOT_EXECUTED**, is now **IRRELEVANT** and removed from this execution path; no historical record may imply it ran. Use explicit reviewed operations on the campaign's own owner-only local channel, with config/source identity and operation receipts. Actual failed P_read fixture equality, original-space KKT and independent review requirements remain in force before revised reader deployment.

This resolution changes only process durability and honest reconstruction accounting. No numerical definition, scientific budget, solver300/600 limit, coefficient ridge, original-space bound, data/seed identity, stage/gate or evaluation rule changes. Replay is justified solely by externally lost indispensable RAM state and must not be triggered selectively by research outcomes.
