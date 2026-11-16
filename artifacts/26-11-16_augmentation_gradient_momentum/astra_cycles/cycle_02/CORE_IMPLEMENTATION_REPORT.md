# Core implementation report — Cycle02 NQR-025

Status: **IMPLEMENTED; CORE CPU CHECKS PASS; RUNTIME INTEGRATION PENDING.**

The accepted DESIGN_SPECIFICATION.md, especially sections 5.3, 6 and 7, is the implementation contract. W/AGENTS.md, O/start.md decoded as GB18030, W/buchong.md, the runbook and all seven Cycle02 specification files were read. Applied ponytail FULL and stop-that-shit CHANGE. No specification ambiguity blocked this scope.

Changed paths, all within exclusive ownership:

- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/__init__.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/core.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/checks.py`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/CORE_IMPLEMENTATION_REPORT.md`
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/CORE_CPU_CHECKS.json`

The core implements the exact `loss_components(probabilities, targets, domains, *, arm, mixture_lambda=0.25)` and `train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` interfaces. The three risks use centered differentiable population variances, natural cell/opportunity weights and actual batch denominators. All specified ledger fields, unavailable-statistic nulls, row-ID digest conversion and diagnostic fields are present. Global coefficient mass is recorded analytically as BCE mass 1 minus opposite mass, retaining exact clean mass 1 in direct branches.

The branch order is lambda_zero, all_single_class, ordinary. Both direct branches delegate to the frozen TV source_clean_step before candidate forward/statistics, record C=A=L and leave unavailable cell/class statistics null. Ordinary training invokes model(x), one scalar backward, the caller's source Adam and one source EMA update. It checks gradients before mutation. Direct delegation checks surviving gradients/model/Adam/EMA for finite values on return. Exceptions produce no successful record; the runtime must discard the unacknowledged transaction and recover its previous RAM ACK.

Diagnostics use detached existing gradients and temporary pre-step parameter copies. They preserve None versus explicit zero gradients and include source prefix groups, named-tensor counts, coordinate counts, post-step active Adam counter ranges, data-gradient norms and actual parameter-update norms. No persistent candidate state or new-protocol import is introduced.

Final check command, working directory `/home/xzf/Code/phase0-3_tvg_rm_20261116`:

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.natural_quadratic_ranking_20261116.checks
```

Result: exit 0, five finite registered groups PASS.

- Independent literal ordered-pair R/B values and gradients; M by literal R minus variance; both decomposition derivative identities. Four fixed fixtures, both tasks, balanced/unequal support, absent domains, missing/singleton classes, FP64 1e-10 absolute/relative and FP32 atol1e-6/rtol1e-5.
- All-arm CLEAN probability-gradient identity at p=.5 with batch sizes 1,3,7,13,21927,40052; cell/task/total masses, explicit row BCE coefficients, balanced .875/.125 split and lambda0 clean identity.
- Frozen source-clean model/Adam/EMA/gradient-activity identity for lambda0 and all-single-class branches over two consecutive sizes; diagnose off/on bitwise model/Adam/EMA/activity/global-RNG/private-loader identity for every arm and all three branches. Ordinary data gradients and resulting parameters also agree with an independent literal objective within the registered FP32 tolerance.
- Independent grouped norms, coordinates and name order; production frozen model factory on synthetic features; ordinary detached gate inputs with trainable gates and embedding gradients; one forward/backward/Adam/EMA call and zero probes/views/candidate RNG.
- Source Adam None/explicit-zero activity, coupled weight decay once, parameter counters, one EMA update and buffer copying.

An intermediate extension of the checks failed because its B=3 fixture assertion assumed opposite-label mass despite each domain containing only one row. The assertion was corrected; the final full check run passed. Frozen setup emits its existing PyTorch TF32-setting deprecation warning.

The three assigned Python files also passed in-memory compile checks. The protected 79-file aggregate matched `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1` before and after. Exact verification commands, outcomes and the intermediate fixture failure are in CORE_CPU_CHECKS.json. No new source hash was frozen.

At final runtime inspection, protocol.py existed while runner.py/tables.py/README.md were absent. Registered runtime admission and cross-module integration checks remain for parent integration after the runtime coder finishes. Seven-file bundle/final source binding belongs to that integration; the core imports frozen helpers only.

GPU/full-batch model smoke, RAM-owner integration and scientific runs were not performed in this CPU coder task. Next required gate: integrate both reports and obtain distinct implementation REVIEW before RUN.

