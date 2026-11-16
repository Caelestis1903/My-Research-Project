# Codex Implementation Specification

Implement only the approved DESIGN_SPECIFICATION in experiments/task_view_gradient_residual_momentum_20261116. Coder1 owns core/protocol/runner/tables/test_core and other algorithm files. Coder2 owns autonomous.py/test_autonomous.py/AUTONOMOUS_MAIN_PROMPT.md, these plans, AUTONOMOUS_STATE and specified context integration. Preserve concurrent edits and historical/vendor/data files.

Follow 04 exactly: same-theta clean/view data gradients; detached delta with trainable embedding; clean coupled Adam m/v/WD once; per-parameter active n_p and current None semantics; separate q without WD; one global cap; SHUFFLE matches the actual preconditioned/capped residual parameter update on its own trajectory; INSTANT removes residual history only. One full parameter transaction then EMA/RAM. Gamma0 canonical source clean.

Minimal new CPU checks: natural loss/target/detach; two-parameter active/absent/re-entry Adam/q/WD/INSTANT; cap/SHUFFLE positive+zero norm and independent RNG; multistep gamma0 exact source state and one RAM resume equality. Reuse valid reviewed Jacobian/finite-difference evidence rather than rebuild it. Driver fixture checks actual event parsing, exact resume ID, CLI errors, sticky stop/both workspaces, final Test gate. No fake research cycles.

After independent static/CPU REVIEW, runner executes one real full-batch harness per dataset containing source clean, gamma0, TV-.10, SHUFFLE-.10 and INSTANT-.10 single updates (two harnesses/ten branches). Record finite/norm/update/EMA/source equality and measured resources. Reviewer closes any smoke blockers before formal run.

Runner module interface, using project Python:
- prepare --attempt PATH --review PATH --cache-dir PATH --reference-root PATH
- smoke --attempt PATH --device cuda:N --review PATH
- run --attempt PATH --devices 0,1 --review PATH
- tables|status --attempt PATH
- retain --attempt PATH --arm ARM
- release --attempt PATH [--key KEY] (no key releases all)

Read actual help/receipts before execution. run never accesses Test; manual stop returns MANUAL_STOPPED and retains selected RAM. Shared autonomous.stop_requested checks both execution and original control prelaunch/pre-epoch/postcommit. After main analysis, retain the champion arm's five selected seeds on both datasets, verify status/identities, then release displaced/rejected keys. No-key release clears an entire attempt. Current recovery/selected state remains in existing guardian only, never disk weights. Final Test has no current CLI; main must separately commission a reviewed evaluator after manual stop/freeze. No new dependencies/framework.
