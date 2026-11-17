# C05 numerical IMPLEMENT handoff

Implemented the accepted U objective and assigned CPU fixtures. Coder: **Laplace / 01a1e9d3-c739-7933-a66e-50368081af1e**; parent: 01a1dfb8-605b-7430-8462-5fd3da0b048f; gpt-6.1-sol, fork_context=false. No descendants. Permanent return after this handoff.

## Changed files and source identity

- `experiments/cell_mass_bce_20261117/__init__.py` — `5d0de7415a2fdb3c78d2fbe61cda3c211c7860cecd072ac073e7fbccd61ed666` (76 bytes)
- `experiments/cell_mass_bce_20261117/core.py` — `5beb32779c14e78756794a91d2a27a030688053cc82a675e4d2d83ae1ba270e3` (20996 bytes)
- `experiments/cell_mass_bce_20261117/checks.py` — `03ee6e5ce1c7aa7b59f28bf0732ccdb7451b89559659f6866d014a092217572e` (23922 bytes)

Accepted specification SHA256: `a1f61ae81535bb33873da72185d9400187c5b988e81c0d3a5019c5505e821af9`. Seven-file design bundle: `9a7e826794f4cdc3fcc7c437e1462149a1dc9372b5d5e3a38981f69fb3af299c`. Research bundle: `bfd6230c9cae8971a3a8d793027fc54ba6209d193429340107bf64bb22dc4caf`. All 48 frozen inputs and installed Adam matched their recorded hashes. Existing source files were unchanged. No source freeze or attempt preparation was performed.

## Numerical implementation

U uses the registered d-major/t-minor scalar loop over ordinary probability-space unreduced BCE: `(w*a)*cell.mean()`, with count ratios from frozen C04 support. C/H/W/S use frozen `loss_components(..., diagnose=False)`; scheduled detached class statistics use frozen `_statistics`. The NQR objective is never constructed. W/U cell mass is the exact shared `completion_bce_mass`; frozen `1-removed_opposite_mass` order supplies m. Ledger records natural/W/U/S cell/class coefficients and masses, present-label q values, a/kappa, task masses and null reasons.

Lambda-zero and all-single-class live calls delegate to frozen source CLEAN, with one-forward/pre-Adam diagnostic hooks removed on exit. Ordinary calls have one live U backward/source Adam/EMA. Diagnostics obtain C/S/W gradients from the retained forward and U gradients from live backward; sequential disposable real source-Adam steps use the same immutable prestate and post-forward buffers. Actual/virtual U hashes and zero update difference are exact. None gradients retain source skip semantics; tensor-zero gradients remain active.

## Exact interface and schema handoff

`loss_components(probabilities, targets, domains, *, arm='CELL-MASS-BCE-025', mixture_lambda=.25, diagnose=False)` and `train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False)` follow section5.1. Returned scalar fields: loss, clean_loss, same_class_loss, completion_loss, scaled_loss, cell_mass_loss. **Mixed-label engineering lambda0 returns same_class_loss=null with same_class_loss_null_reason=not_computed_delegate in both diagnostic modes.** All-single-class H=C. Ordinary formal lambda.25 H is observed.

Diagnostics use ordered objectives C/S/U/W and pairs U-W, U-S, S-C. `parameter_groups[group]` contains coordinates/objectives/pairs; **no repeated parameter_names list**. Groups are ALL then sorted prefixes. Parameter activity/private steps retain ordered name keys for the runtime dictionary join. FP64 full-coordinate norms/dots preserve missing gradients as comparison zeros without installing .grad tensors. Coordinate zero/nonzero counts count individual coordinates.

Point metadata includes point_id, context, optional job_id/physical_attempt_id, prestate/row/loader hashes, four objective values and virtual posts, actual post/exactness, masses/counts/timing and grouped metrics. Runtime joins actual job/registration identity and normalizes metadata/names on disk. Cell-ledger additions and every point field are enumerated in the JSON report's api section. Frozen mass/statistic field names remain available.

**Final change after the interim schema message:** added the specification-required `core.finite_checks()` entry point delegating to the single `checks.check_core()` fixture implementation. No other schema change. `checks.finite_checks()` and `python -B -m experiments.cell_mass_bce_20261117.checks check` are supported. Core/checks import no new runtime module.

Config validation pins cycle/objective/U arm, scientific constants, dataset-definition hash and selection hash. Formal context enforces exact sample cardinality, source row IDs, clock and diagnostic schedule. Engineering uses its separate clock; engineering_check requires explicit CPU fixture cardinality. Schedule/switch/off-update keys are rejected.

## Final basic checks

Three final source strings compiled in RAM. The final CLI path ran via runpy with `check` argv and captured stdout in RAM, using the exact project Python, -B and PYTHONDONTWRITEBYTECODE=1. Result: **CPU checks PASS** (5.801s). FP64 algebra atol1e-12/rtol1e-10; FP32 atol2e-6/rtol2e-5. State, dtype/shape/ordered structures, hashes, None masks, counts and actual/virtual zero difference compare exactly.

Finite evidence covers hand-enumerated U coefficients; W-U covariance and U-S identities/gradients; frozen C/H/W/S values; six exact W/U cell masses; natural entries and task/class/global masses; balanced, singleton, one-class, empty/absent, native 0/1 boundary BCE and lambda0 cases. Six stochastic-fixture diagnosed/off comparisons and RAM BytesIO suffix restores preserve model/EMA/Adam/modes/gradients/RNG/private loader, and source delegates match exactly in lazy/populated states. Eight independent objective source-Adam oracle steps reproduce all four virtual posts. The real source architecture verifies [F,B] to [B,2], finite backward, gate input detach, prefix dictionary and diagnosed/source-delegate identities. Tiny source validation metrics verify unscaled BCE, undefined-AUC nulls, independent earliest strict selection and LL bound to AUC-selected state. Fifteen invalid calls reject before mutation; four invalid probability fixtures and four full/tail cardinality checks pass.

Per successful full CPU pass: 27 core calls;14 diagnostic points;42 extra sweeps;56 virtual Adam steps;5 direct source CLEAN calls;8 independent oracle forwards/Adam steps;3 populated-fixture initializer Adam steps;1 separate real-architecture forward/backward. Two successful passes occurred: initial, then final after adding core.finite_checks. These are CPU fixture operations, zero scientific epochs. Full machine-readable final output is in CORE_IMPLEMENTATION_REPORT.json.

## Remaining integration and stage boundaries

No numerical-scope blocker. The runtime coder owns the protocol/transport/clock/pending, paired17-family/table arithmetic, telemetry/terminal/witness/admission/root/archive fixtures and normalized disk output; those checks were not run here. A distinct reviewer must assess both scopes before RUN. No cache unpickling, actual-data preparation, owner IPC, GPU training, Test access, disk training state, or research-result analysis occurred.
