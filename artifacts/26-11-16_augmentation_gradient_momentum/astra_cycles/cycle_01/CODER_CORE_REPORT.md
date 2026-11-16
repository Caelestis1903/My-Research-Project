CODER CORE IMPLEMENTATION REPORT — 2026-11-16

Core implementation is complete for AXIS-EVEN, AXIS-IID2 and ISOTROPIC-EVEN under the approved seven-file bundle `f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`. Independent implementation review remains pending; the design approval is not an implementation approval.

Written files, all relative to `/home/xzf/Code/phase0-3_tvg_rm_20261116`:

- `experiments/task_axis_even_views_20261116/__init__.py`
- `experiments/task_axis_even_views_20261116/core.py`
- `experiments/task_axis_even_views_20261116/test_core.py`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/CODER_CORE_REPORT.md`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/CODER_CORE_REPORT.json`

The API is `train_step(model, optimizer, ema, x, y, config, context)`. Required config keys are `arm`, `alpha=.03`, `mixture_lambda=.25`, and `rho`. Context supplies `dataset`, `seed`, one-based `epoch` and `batch`, optionally `row_order_hash` and `diagnose`. The runner owns diagnostic scheduling at first batch of epochs 1,10,25,40,50. There are no additional options or persistent core states.

The flat JSON-safe return fields are `loss`, `clean_loss`, `view_loss`, `replica_losses={R1,R2}`, `task_view_losses={task0:[a01,a02],task1:[a11,a12]}`, `zero_delta_canonical`, `ledger`, `diagnostics`, `row_order_hash`, optimizer/EMA update counts and probe/construction/training/total seconds. Canonical records have R1=R2=view=C and null task-view entries because no views were evaluated. An absent row-order hash stays null; the core never substitutes local row indices for original IDs.

The ledger records actual B and 2B denominator, every domain/task/label count, conceptual clean/replica coefficients and masses, and natural total/task masses. Canonical records additionally show the actual clean coefficient 1/(2B) and actual zero replica coefficients while preserving conceptual weights. Diagnostics contain unchanged source probe info, replica construction info, RNG version/stream names, and, when requested, actual input norm errors/radius summaries, IID2 duplicate-sign fractions, even-arm E/O, total data-gradient norm, None count, per-parameter counter range, and actual parameter displacement L2 with FP64 reductions. Exact keys and helper signatures are recorded in the JSON report.

The core calls unchanged NDP-A `make_views` and the audited embedding-addition `forward`. It evaluates clean then task0 replica1/task0 replica2/task1 replica1/task1 replica2; only each view's target BCE contributes. Arithmetic is exactly R1=.5*(a01+a11), R2=.5*(a02+a12), V=.5*(R1+R2), total=.75*C+.25*V. Deltas and view construction stay detached while ordinary embedding parameters remain trainable. The all-zero branch uses C directly and skips replica draws/forwards. One backward is followed by the caller's audited Adam step and source EMA update exactly once; the core never changes None gradients, optimizer state arithmetic or weight decay.

The RNG key is the exact UTF-8 pipe-joined `axis_even_v1|dataset|seed|epoch|batch|task|stream`, first eight SHA-256 bytes unsigned little-endian. Independent private device generators draw sign1/sign2 or isotropic streams. Isotropic pairs inherit each actual task-axis radius, with the registered first-coordinate fallback for an exactly zero Gaussian draw. No global or loader RNG is consumed.

Executed:

```text
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.task_axis_even_views_20261116.test_core
```

Result: **PASS: four CPU fixture groups**, exit 0. Coverage is the approved mass/support/target/zero/tail/symmetry/exact-sign-expectation group; keyed RNG/norm/fallback group; source Adam None/zero/counter/WD/EMA group; and exact canonical/source plus actual three-arm step group. The maximum measured isotropic norm relative error was **1.6109599698665988e-7**, below 1e-6. Canonical full/tail checks require exact gradients/None masks/model/Adam moments/counters/EMA equality with the unchanged source clean step. Each actual arm matches its explicit target-only objective and audited Adam/EMA state exactly. The script is also importable by existing pytest; no new test framework, broad suite, finite-difference matrix or dependency was added.

Built-in compilation of all three owned Python sources passed without bytecode output. Python was 3.13.5, torch 2.9.0+cu126. Frozen setup emitted torch's TF32 API deprecation warning; it was left unchanged. The frozen old source aggregate recomputed as **d443b9a7d5732e799e4c19859c719d9871afdea15bbf4f700a5070ed8db05de6**, unchanged. The JSON report records all 39 local imported source hashes for core/fixtures and the read-only old source-manifest audit, including the three new core files. Both stop controls were ACTIVE/IMPLEMENT, manual stop false, Test false, with absent sentinels.

Implementation blockers: **none found in this owned scope**. Protocol/runner/tables integration belongs to the other fresh coder and was not edited or validated here. The parent still needs a distinct fresh implementation review and runner-owned reviewed full-batch/tail/source, historical-reuse, storage and resource gates before formal execution. This report makes no research conclusion.

No GPU smoke/formal run, validation/Test use, model/optimizer/checkpoint file, storage/driver/guardian action, child/subsession dispatch, commit or push occurred. Only the five owned files were written; concurrent edits remain intact. Parent should close this context permanently after receiving the reports; do not resume or reuse it.
