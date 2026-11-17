# Cycle04 core implementation report

Status: **CORE COMPLETE; finite syntax/CPU checks PASS**. Runtime integration is pending arrival of runtime-owned `runner.py` and `tables.py`. Parent performs the combined source freeze and dispatches the distinct reviewer.

Implemented only `experiments/completion_weight_attribution_20261117/core.py`, `checks.py`, this report and `CORE_CPU_CHECKS.json`. The runtime coder owns the remaining package files. No source freeze or attempt preparation was performed.

Approved provenance:

- Seven-design bundle: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`.
- Six-research bundle: `a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a`.
- A4 design specification: `cb320632dcebdb8f75fe298b5b35f95e6dc07659ff0146fbe5f8ffd79ffed9db`; read all 459 lines, alongside DESIGN_ACCEPTANCE.json and ASTRA_HANDOFF.json.
- Published source base: `4c95dc877b85cc676bf7366958f8c7de0291871e`.
- Read W AGENTS.md, GB18030 start.md, buchong.md, current runbook correction, Ponytail FULL and Stop That Shit change instructions.

`loss_components` implements the ordered domain/task/class completion BCE reduction, actual-batch detached m/r, W and S, the six-cell coefficient ledger, support-aware null reasons, scheduled detached statistics and optional pure frozen N reference. Actual total mass remains m in both new arms. Production W is formed directly from C and A; S multiplies source C by the Python-float m. No objective is divided by m. Opposite-class mass applied by either new arm is zero. The frozen N helper is called only on requested diagnostics and finite checks.

`train_step` validates configuration, clock, support, row identity and real full/tail cardinality before training mutation. Ordinary calls run one model forward, one selected-loss backward, source Adam and one source EMA update. Every parameter's actual None/tensor-zero/nonzero activity and source private step are reported; private steps are checked against activity and the previous step map. All-single-class and lambda-zero branches delegate to frozen `source_clean_step` exactly.

Diagnostics use the same ordinary graph for C/S/W/N, with three retained-graph `autograd.grad` calls for the objectives other than the actual arm. Four sequential disposable model/source-Adam clones use independently copied pre-update source optimizer state. Their gradients preserve None. Each clone is discarded before the next model clone is created. Source Adam is never implemented numerically. Prestate immutability, live optimizer identity and RNG equality are checked around clone execution. Actual-arm virtual/live model and optimizer hashes must match exactly, with update difference norm zero.

For identity diagnostics, a local forward observer captures that one source output and removes itself immediately, before any model deepcopy; the local optimizer pre-hook observes the source backward and performs the same four virtual steps. Both handles are removed in finally. The freshly constructed clone optimizer has no observer hook. Frozen modules are not monkeypatched.

Diagnostic vectors use the full named-parameter coordinate space, FP64 squared norms/dots, mathematical zero for missing components, explicit activity counts, private-step summaries and null/zero_norm cosine semantics. EMA difference norms are logged algebraically as `(1-rho)` times the actual update-difference norm. Live EMA is updated once only.

The runtime integration interfaces are exactly `ARMS`, `OBJECTIVES`, `CELLS`, `loss_components(...)`, `train_step(...)` and `check_core()`. Numeric return fields and ledger names follow SPEC5.3. `diagnostics.parameter_groups` contains `ALL` and each top-level parameter prefix. Each group contains:

- `coordinates`, `parameter_names`.
- `objectives[C|S|W|N]`: `data_gradient_norm`, `parameter_update_norm`, active/none/tensor_zero/nonzero parameter and coordinate counts, `pre_private_steps` and `post_private_steps` (`min,max,uninitialized`).
- `pairs['W-S'|'N-W']`: `data_gradient` and `parameter_update` dictionaries (`norm,right_norm,difference_norm,cosine,null_reason`), activity-mask difference parameter/coordinate counts, and algebraic `ema_difference_norm`/relationship.

The diagnostic root includes host arm, actual objective, row-order hash, pre-model/optimizer/RNG/loader hashes, objective losses, m/r, actual/virtual post hashes, exact comparison flag, difference norm zero and the 3/4 operation counts. Parent was sent this explicit schema for the runtime tables handoff. Engineering CPU fixtures declare `context.fixture_rows`; real engineering/formal calls require source row IDs and actual full/tail sizes. Diagnosed callers provide the actual `pre_loader_hash`.

Read-only local imports are within SPEC6.2: auxiliary protocol's `cpu_copy,optimizer,rng_state,state_digest` plus the fixture-only `make_model,restore_rng,setup`; auxiliary stage2 `synchronize`; conditional protocol `update_ema`; TV core `source_clean_step`; NQR core pure `loss_components`. No additional local dependency outside the explicit closure is needed. Standard-library modules and existing NumPy/PyTorch are used. No historical root-bound routine is called.

Actual checks were run in W with the required explicit Python and `PYTHONDONTWRITEBYTECODE=1` prefix. Full command text and results are recorded in CORE_CPU_CHECKS.json:

1. Initial `compile(source_text,...)` syntax check of core.py: PASS.
2. `python -m experiments.completion_weight_attribution_20261117.checks --scope core`: PASS.
3. Final heredoc compiling core.py/checks.py, then `check_core()`, after adding independent parameter-chain derivatives and clone-model finiteness: PASS.

One final suite performs four FP64/FP32 arm algebra fixtures; 12 lazy/populated/branch/arm diagnostic-neutrality comparisons and 12 restored-suffix comparisons; eight direct source identity calls; and four exact comparisons using actual source M3oE with an explicit tiny feature schema. It rejects 15 invalid pre-mutation calls and four invalid probability inputs, and checks four actual-form ML/KR full/tail cardinalities. Independent row/pair formulas check probability and parameter derivatives. Fixed FP64/FP32 tolerances are unchanged; every replay/state/activity/hash comparison is exact.

Per final suite: 54 core calls, ten direct frozen source calls, 28 diagnosed core calls, 84 diagnostic extra gradient sweeps and 112 real cloned source Adam steps. Six explicit population-setup Adam/EMA steps prepare the required populated unused-parameter fixture. Loss-algebra fixtures use 56 additional `autograd.grad` calls. The suite's 64 ordinary forwards/scalar backwards and 182 total actual Adam steps (including six population steps and 112 clones) are CPU fixture accounting, separate from the formal and real engineering budgets.

Checks include source None versus tensor-zero coupled WD/private-step/moment behavior, actual-arm virtual equality, hook removal, no constructor invocation by deepcopy, exact live gradients/RNG/loader/modes/buffers/EMA, CPU BytesIO roundtrip, optimizer tensor data_ptr/storage separation, saved-prefix immutability and next-permutation identity. A disjoint-activity fixture checks full-coordinate norms/cosine; zero norms retain explicit null reasons; independent EMA difference arithmetic meets the fixed FP32 tolerance.

`checks --scope all` invokes runtime finite checks as modules become present and labels missing modules INTEGRATION_PENDING. It was not run in this core handoff because runner/tables were absent at completion. Runtime integration is the next combined gate; core completion does not depend on runtime report-file presence. The frozen source setup emitted the installed PyTorch TF32 API deprecation warning; its flags were preserved.

File hashes at handoff:

| File | SHA256 |
|---|---|
| E/core.py | `790ae97d01a3ab3d39e42712580e333aa017b96d5869f36aed682181b03de683` |
| E/checks.py | `7bbf10c3e4249d9a3af7e1ad80b7c93267a66f99ddc52e3050c709ba99605ae1` |
| A4/CORE_CPU_CHECKS.json | `146d279d66081711398dcfdd96f4f5d52e34ab5b99c83a8427afd4e22d16c68d` |

This report's own SHA is returned separately to parent to avoid self-reference. No GPU query, GPU execution, IPC, owner contact, dataset-cache/Test access, historical source edit, tensor file, Git/global setting, rootstate/context/history edit or descendant was used. All fixture tensor state stayed in CPU RAM/BytesIO. The core child returns control permanently after this file handoff; parent closes its context.
