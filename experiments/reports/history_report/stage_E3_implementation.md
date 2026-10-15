# CODER IMPLEMENTATION REPORT — Stage E3

## Scope and stage boundary

Implementation follows the approved `experiments/reports/stage_E3_design.md` only.
The coder did not read any `promt` file, did not read `group3.md`, did not run the
formal 60-run joint matrix or 36-run oracle matrix, and did not perform REVIEW,
RUN, or scientific analysis.

## Design-to-implementation mapping

| Design requirement | Implementation |
|---|---|
| Registered alpha grid and canonical directory keys | `stage_E3_weighting.py`: `ALPHA_GRID`, `ALPHA_KEYS`, `canonical_alpha`, `alpha_key` |
| Exact `D*p**alpha/sum(p**alpha)` and task-major replication | `weighting_coefficients`; float64 coefficients are used by the runner for the 1e-8 audit tolerance |
| One explicit six-cell weighted loss and normal backward/step | `weighted_loss_from_terms`; `stage_E3_run.py::_train_epoch` calls one `loss.backward()` and one main optimizer step |
| Shared-only observer with no `.grad` mutation or extra step | `collect_shared_cell_gradients` uses `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`; only shared partition vectors are logged |
| Raw/weighted norms, signed contribution, mismatch, update alignment and reconstruction | `make_shared_gradient_step_rows`, `SharedGradientEpochAccumulator`; outputs `shared_gradient_step.csv` and `shared_gradient_epoch.csv` |
| Fixed data/model/optimizer/sampling/checkpoint/test semantics | `stage_E3_run.py` mirrors E1 data/model path, role sampling, Adam/architecture update, validation macro-AUC checkpoint and RNG-preserved test evaluation |
| 60-run fixed queue and five-alpha block identity | `stage_E3_orchestrator.py`: `build_tasks`, `compare_alpha_block`, `done_set`, resume-safe status handling |
| Oracle audit and frozen fallback | `resolve_oracle_metrics` accepts only formal, complete explicit seed-level artifacts; `stage_E3_oracle_run.py` records the frozen two-task BCE-sum objective, formal protocol gate, selected-checkpoint cells, model checkpoint and split identities |
| Resume identity and support safety | Joint/oracle manifests and configs carry normalized dataset identity, device identity and ordered train/validation/test row fingerprints; `compare_alpha_block` accepts only complete common prefixes and explicitly censored early-stop tails |
| Non-finite checkpoint handling | `strict_validation_macro_auc` fails a run when any required validation AUC/logloss cell is missing/non-finite; selected joint/oracle summaries require complete finite checkpoint cells and report aggregation never drops cells silently |
| Summary field semantics | Per-run and stage summaries carry `domain_size`/`domain_share`; `magnitude_share` is domain-level in both step and epoch diagnostics; `common_support` is set only by a valid block comparison at the selected epoch |
| Schema-preserving report with missing paths | `stage_E3_report.py` enumerates all 60 expected paths, preserves seed-level rows, computes mechanical macro/minimum fields and oracle coverage only |
| Focused verification | `experiments/tests/test_stage_E3_weighting.py` and `test_stage_E3_matrix.py` |

## Files added/updated

- `experiments/stage_E3_weighting.py`
- `experiments/stage_E3_run.py`
- `experiments/stage_E3_oracle_run.py`
- `experiments/stage_E3_orchestrator.py`
- `experiments/stage_E3_report.py`
- `experiments/tests/test_stage_E3_weighting.py`
- `experiments/tests/test_stage_E3_matrix.py`
- this report

No E1 implementation, core model, data loader, trainer, prompt file, or existing
raw result was modified.

## REVIEW #1 blocking fixes

The implementation was revised only for the CRITICAL/MAJOR findings.  The
oracle main path now calls `oracle_main_loss`, which is the sum of the two
per-task BCE means; its architecture optimizer remains the separate frozen
unmasked path.  Oracle resolution now requires `formal_valid=true` in both
config and manifest, the exact 50-epoch/patience/hyperparameter protocol,
normalized dataset/device and split-row identities, a complete selected
checkpoint/model file, contiguous observed support and finite validation/test
cells.  Consequently a one-epoch smoke cannot be reused as an oracle.

Joint checkpoint selection is strict over all six cells.  Block comparison
checks ordered train/architecture support for every observed epoch and allows
only a suffix paired with an explicit `stopped_epoch`/`censored_after_epoch`;
it no longer rescues malformed traces with an arbitrary set intersection.

## REVIEW #2 final targeted fix

Oracle reuse now requires external identity from the current invocation at all
three layers: `audit_oracles`, `resolve_oracle_metrics`, and
`validate_oracle_artifact` pass the requested device and the expected
per-domain train/validation/test row fingerprints.  The expected rows are
computed once from the current MovieLens arrays using the current
`setting/seed` role reversal and the same physical-domain-to-zero view as the
oracle runner.  A formal artifact whose internal config/manifest agrees but
comes from another device or different rows is therefore rejected.

## Verification performed

1. Python syntax compilation passed for all five E3 modules and both E3 test files.
2. Focused E3 tests passed: **11 passed**.  The tests include the oracle
   task-sum regression, strict non-finite checkpoint rejection, and rejection of
   unrecorded trace tails, plus wrong-device and wrong-row oracle fixtures.  The
   only output warnings were existing
   pytest cache permission warnings.
3. Five real-data CUDA short smokes (`d0_large`, seed `2022`, one epoch; all
   registered alpha values) completed in an isolated temporary directory.  The
   resulting five-arm block comparison was `VALID`, with common epoch `[0]`
   and six complete trace entries (five train batches plus one architecture
   update).  Each arm verified:
   - role counts `150000/34500/3000`;
   - 30 shared step rows (five train batches × six cells) and 6 shared epoch rows;
   - `weighted_gradient_norm == loss_coefficient * raw_gradient_norm` maximum error `0`;
   - maximum shared-gradient reconstruction relative L2 error was `9.11e-7`;
   - normalized dataset/device and validation/test row fingerprints in config and manifest;
   - `formal_valid=false` for the intentional one-epoch smokes.
4. One real-data CUDA one-epoch single-domain oracle smoke (`d0_large`, physical
   domain 0, seed `2022`) completed with explicit `original_physical_domain=0`,
   `model.pth`, and `formal_valid=false`; resolving that short artifact produced
   zero covered oracle cells (`formal_hyperparameters`, `formal_valid` and
   `formal_stop_support` rejected), so it was not eligible for reference reuse.
5. Orchestrator dry-run loaded the current dataset, computed external
   remapped split identities, then reported exactly **36** oracle candidates
   and a fixed **60** joint matrix without launching a child process.
6. No additional GPU smoke, formal oracle matrix, formal joint matrix, or
   scientific analysis was run in this final targeted-fix turn.  Temporary
   dry-run directories were removed after verification.

## Raw-output and report behavior

Each joint run writes `config.json`, `run_manifest.json`, `stdout.log`,
`metrics_epoch.csv`, `metrics_summary.csv`, `shared_gradient_step.csv`,
`shared_gradient_epoch.csv`, `batch_trace.csv`, `nonfinite_inventory.json`, and
`model.pth`.  The report command writes `stage_E3_summary.csv`,
`stage_E3_summary_manifest.json`, and `reports/stage_E3_run_report.md`.

The one-epoch smoke is intentionally not eligible for formal resume completion:
`formal_valid` is false and `done_set` requires the formal protocol and a valid
five-alpha block.

## Known limitations / handoff

- The current workspace does not contain verified P0-2 r50 seed-level oracle
  artifacts.  The orchestrator therefore reports them as missing and, on a real
  run, queues the required 36 frozen oracle references before any joint arm.
- Aggregate Markdown or CSV summaries without a complete explicit oracle
  manifest are rejected and remain uncovered; transfer and negative-transfer
  fields stay unresolved rather than being imputed.
- Formal execution requires the repository's `venv` plus the torch-enabled
  `D:\CODE\.venv\Lib\site-packages`; the orchestrator propagates that path to
  child processes automatically.
- No `stage_E3_analysis.md` is generated here.  Main analysis must be performed
  after REVIEW and RUN, using the registered response-surface analysis plan.

## Formal runner command for the next stage

```powershell
& ".\venv\Scripts\python.exe" experiments/stage_E3_orchestrator.py `
  --dataset_path "D:/CODE/phase0-3/m3oe/examples/data/ml-1m" `
  --device cuda:0
```

This command first audits/reuses or generates the 36 oracle references, then
resumes/executes the fixed 60 joint tasks and writes raw status artifacts.  It is
intentionally not executed by the coder.
