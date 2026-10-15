# CODER IMPLEMENTATION REPORT — Stage E1

## Scope and status

Implementation is complete for the approved `DESIGN SPECIFICATION — Stage E1：Shared Optimization Contribution 机制诊断` in this directory. The design report was first made available at `experiments/reports/stage_E1_design.md` and was treated as the only implementation authority. No `group2.md` or `group3.md` file was read. No core model/trainer file was changed, and no formal 24-run matrix was started.

## Files changed

Added:

- `experiments/stage_E1_instrumentation.py`
- `experiments/stage_E1_run.py`
- `experiments/stage_E1_orchestrator.py`
- `experiments/stage_E1_report.py`
- `experiments/tests/test_stage_E1_instrumentation.py`
- `experiments/tests/test_stage_E1_equivalence.py`
- this report: `experiments/reports/stage_E1_implementation.md`

Read and retained without coder changes:

- `experiments/reports/stage_E1_design.md`

No unrelated files were modified.

## Mapping from specification to implementation

`stage_E1_instrumentation.py` provides:

- explicit, disjoint MDMTRec parameter groups: `shared`, `domain_specific`, `task_specific`, `cell_specific`, and `architecture`;
- parameter-name/count/shape/state hashes for the run manifest;
- exact task-major cell ordering (`idx = task * D + domain`);
- equal coefficients `1` and sample coefficients `D * p_d`, with six-cell sum `6`;
- raw per-cell gradients through `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`;
- raw norm/RMS, coefficient-scaled norm/RMS, signed projection, normalized contribution, magnitude share, overrepresentation, and mismatch;
- actual/equal-counterfactual/sample-counterfactual contribution views from one raw-gradient collection;
- full vector Welford statistics for `shared`, norm-dispersion statistics for private groups, and explicit `variance_scope` logging;
- actual combined-gradient reconstruction error and post-step update alignment/projection;
- auxiliary shared-gradient cosine values;
- read-only gate hooks with expert probabilities, entropy, effective expert count, and top-1 frequency;
- stable batch fingerprints, CPU/CUDA RNG preservation, exact masked BCE terms, metric helpers, and CSV/JSON writers.

`stage_E1_run.py` provides:

- MovieLens-1M role-reversal sampling copied from the existing P0-4 rule;
- the fixed r50 role sizes `150000,34500,3000` and the three role rotations;
- MDMTRec/M3OE with the existing dimensions and Adam hyperparameters;
- only `equal` and `sample` training arms; no curriculum, momentum, EMA, new router, new sampler, or scheduler;
- normal weighted `loss.backward(); optimizer.step()` training, with observer gradients excluded from `.grad`;
- the existing separate unmasked architecture-parameter update path and optimizer;
- epoch-level train/validation/test AUC and LogLoss per domain/task;
- gradient step/epoch, cosine, routing, batch-trace, config, manifest, model, and summary outputs;
- equal six-cell validation-AUC checkpoint selection, explicit post-stop censored metric rows, and read-only P0-2 ratio-50 oracle linkage when the artifact exists; otherwise `oracle_source=missing`;
- `--instrument 0` for observer-bypass equivalence checks and `--instrument 1` for the formal run.

`stage_E1_orchestrator.py` provides:

- the exact queue `2 weighting × 3 settings × 4 seeds = 24` in weighting → setting → seed order;
- resume-by-OK status, per-run `stdout.log`, master raw status CSV, and a PID lock;
- a `--dry_run` mode that emits commands without launching a process;
- no result aggregation or scientific interpretation.

`stage_E1_report.py` provides:

- schema-preserving collection of available per-run files into `stage_E1_summary.csv`;
- per-seed/per-epoch/per-cell rows with the required AUC, LogLoss, gradient, contribution, role, and oracle fields;
- an availability manifest for missing runs/files;
- no means, hypothesis tests, or research conclusions.

## Checks executed

All commands were run from `D:\CODE\phase0-3`.

1. Syntax compilation:

   `python -m py_compile experiments/stage_E1_instrumentation.py experiments/stage_E1_run.py experiments/stage_E1_orchestrator.py experiments/stage_E1_report.py experiments/tests/test_stage_E1_instrumentation.py experiments/tests/test_stage_E1_equivalence.py`

   Result: `PASS` (exit code 0).

2. Initial test invocation using the default/system interpreter:

   `python -m pytest -q experiments/tests/test_stage_E1_instrumentation.py experiments/tests/test_stage_E1_equivalence.py`

   Result: `FAIL during collection`, original error `ModuleNotFoundError: No module named 'torch'`. This was an environment dependency issue; no test body ran.

3. Unit and equivalence tests using the available PyTorch environment:

   `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -m pytest -q experiments/tests/test_stage_E1_instrumentation.py experiments/tests/test_stage_E1_equivalence.py`

   Result: `5 passed` (exit code 0; pytest emitted only pre-existing `.pytest_cache` permission warnings).

4. Orchestrator dry run:

   `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python experiments/stage_E1_orchestrator.py --dry_run --max_runs 2 --device cpu --python python`

   Result: `PASS`; output reported the fixed 24-run matrix and 24 pending runs, then printed two commands without starting a run.

5. Short real-model integration checks, one epoch, seed 2022, setting `d0_large`, batch size 40960, CUDA, fixed r50 role sizes:

   - equal/instrumented: `PASS`; all required per-run files were written and shared reconstruction error was below `4e-7` in the smoke output.
   - sample/instrumented: `PASS`; coefficients were `2.4, 0.552, 0.048` per domain/task and the same file set was written.
   - equal/observer bypass: `PASS`; no gradient-step or routing rows were produced, as requested for bypass mode.

6. Short real-model ON/OFF equivalence check, three epochs, same seed/setting/config:

   Result: `PASS`. Initial and final parameter hashes matched; train and architecture batch fingerprints matched; maximum validation-AUC and test-AUC absolute differences were `0`; the instrumented run had `450` gradient-step rows and the bypass run had `0`.

7. Empty-input report smoke check:

   `python experiments/stage_E1_report.py --raw_root D:/CODE/phase0-3/_tmp_stage_E1_empty --output D:/CODE/phase0-3/_tmp_stage_E1_summary.csv --manifest D:/CODE/phase0-3/_tmp_stage_E1_summary_manifest.json`

   Result: `PASS`; wrote the required summary header with `0` runs and `0` rows.

## Known limitations

- Full-vector Welford is enabled for `shared`; private groups use scalar norm dispersion to bound memory and are labeled `variance_scope=norm_scalar`. This is the explicitly documented low-memory branch in the design.
- Oracle fields remain `missing` unless the existing P0-2 ratio-50 oracle tree is present at `--oracle_root`; E1 never launches an oracle or fabricates seed-level values.
- The formal runner expects a working PyTorch environment. The orchestrator adds the workspace's available `D:\CODE\.venv\Lib\site-packages` to child `PYTHONPATH` when present; direct invocation may need the same environment setup.
- The legacy MDMTRec forward allocates its final output tensor on CPU. The E1 runner retains that model code and explicitly casts the returned prediction to the selected device at trainer/evaluation boundaries.
- The coder did not run the formal matrix, inspect scientific patterns, or make a GO/STOP decision. Those actions belong to reviewer, runner, and main analysis respectively.

## Formal runner command

From the repository root, after reviewer approval:

`$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -u experiments/stage_E1_orchestrator.py`

The command launches the fixed 24-run matrix with seeds `2022, 2023, 2024, 2025` and writes raw outputs below `experiments/stage_E1_raw_results/`. Use `--dry_run` for a non-launching command audit.

## REVIEW RETURN — blocking MAJOR fixes

The first independent review was `FAIL`. The following corrections are now
implemented, without changing the E1 factors, model, loss definitions, role
sizes, or optimizer design.

1. **P0-4 RNG/validation boundary (original: FAIL / baseline trajectory
   deviation).** `stage_E1_run.py` now evaluates validation in the ordinary
   RNG stream and wraps only the extra test evaluation in
   `preserve_rng_state()` (CPU plus all CUDA streams). This removes the old
   validation-plus-test rollback that caused epoch-1 divergence.

2. **Paired equal/sample validity (original: FAIL / no comparator).**
   `stage_E1_instrumentation.py:compare_pair_runs` compares the complete
   train/architecture batch sequence, including fingerprint, order, epoch,
   step, size, and count fields. A missing arm is `PENDING`; any present
   mismatch is `INVALID`. The orchestrator revalidates both arms, records
   `pair_status`/`pair_valid` in each manifest and master CSV, appends an
   invalidation correction for a prior `OK` row, and excludes invalid pairs on
   resume. Equal/sample objective losses are intentionally not compared by
   this pair comparator; `compare_losses=True` is available for ON/OFF loss
   equivalence checks.

3. **Split/routing summary (original: FAIL / no split or gate read).**
   `stage_E1_report.py` now enumerates all 24 expected paths, emits explicit
   `train`, `validation`, and `test` long-form rows, reads `gate_epoch.csv`,
   and joins cell routing fields plus domain task-average routing fields.
   Missing run directories and missing report files are listed in the report
   manifest.

4. **NaN/Inf inventory (original: FAIL / silent `NA`).** Each run now writes
   `nonfinite_inventory.json` before CSV conversion. Every non-finite value
   includes file, CSV row, row key, field, value, and a reason. The manifest
   stores inventory and unexplained counts; the orchestrator rejects an
   unexplained or inconsistent inventory. CSV `NA` remains a display encoding,
   while blanks remain distinct from non-finite values.

5. **Per-step loss (original: FAIL / unauditable <=1e-7 gate).**
   `batch_trace.csv` now records the exact weighted training scalar for every
   train step and the exact architecture-update scalar for each architecture
   row. `compare_pair_runs(..., compare_losses=True)` checks the maximum
   absolute loss difference against `1e-7` for short ON/OFF equivalence.

6. **Resume/failure/missing-run semantics (original: FAIL / stale OK and
   child failures).** `done_set` now takes the latest master status and
   independently validates manifest, config, required files, headers, batch
   rows, inventory, and pair state. A child failure or invalid pair causes
   `run_queue` to return exit code `1`. The report manifest always contains the
   fixed 24-run expectation and explicitly enumerates missing paths.

Safe related corrections also add the explicit signed domain contribution
field, a domain-level task-average routing row, and the actual linked
`oracle_source` in `metrics_summary.csv`.

## Post-fix checks

All commands were run from `D:\CODE\phase0-3`; no formal 24-run matrix was
started and no scientific conclusion was computed.

1. `python -m py_compile` over all six Stage E1 modules/tests: **PASS**, exit
   code `0`.

2. `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -m pytest -q experiments/tests/test_stage_E1_instrumentation.py experiments/tests/test_stage_E1_equivalence.py`:
   **PASS**, `9 passed` (only pre-existing pytest cache permission warnings).
   The added checks cover fingerprint invalidation, loss comparison,
   non-finite reasons, stale-OK revalidation, child failure status, 24-path
   missing enumeration, split rows, and gate routing joins.

3. Orchestrator `--dry_run --max_runs 2` with an isolated raw/master path:
   **PASS**; it reported the fixed `2 x 3 x 4 = 24` queue, 24 pending, and
   launched no child.

4. One-epoch real MDMTRec/CUDA smoke (`seed=2022`, `d0_large`, equal,
   instrumentation ON): **PASS**. All required artifacts were written;
   `batch_trace.csv` contains train and architecture losses;
   `gate_epoch.csv` contains cell and domain-average routing rows; the
   inventory contained 54 explained non-finite entries and `0` unexplained.

5. Three-epoch real same-config ON/OFF smoke: **raw invariants PASS; original
   generic comparator FAIL**. Initial/final parameter hashes matched, the full
   18-row train/architecture trace matched, and the maximum absolute loss
   difference was `0.0` (threshold `1e-7`).  The then-current generic
   comparator nevertheless returned `INVALID` because it compared the expected
   OFF/ON `is_probe=0/1` field difference.  The earlier wording that this
   comparator had returned PASS was inaccurate and is superseded by the
   second-round correction below.

6. Three-epoch equal/sample short smoke: **PASS** for the intended pair
   contract. The default equal/sample comparator reported matching fingerprints
   and `VALID` while reporting the expected objective-loss difference; enabling
   loss comparison correctly reports `INVALID`, demonstrating that objective
   losses are not incorrectly used as a pair-fingerprint criterion.

7. Report smoke with one populated run and an isolated raw root: **PASS**;
   it emitted 270 split-long rows, reported `available_run_count=1` and
   `missing_run_count=23`, and joined gate entropy/probability fields.

The reviewer should independently rerun the corrected real E1-OFF versus
P0-4 three-epoch baseline regression before approving RUN. The available
historical `experiments/history_exp/first/p4_run.py` entry point could not be
used for that check in this checkout because its startup path expects a
missing `history_exp/m3oe/examples` directory; this is recorded as a check
environment limitation, not treated as evidence about the experiment.

## Fix-stage known limitations

- Full-vector Welford remains enabled for `shared`; private groups use scalar
  norm dispersion and retain `variance_scope=norm_scalar`.
- Equal/sample pair loss values are intentionally different by design. The
  pair-integrity comparator checks fingerprints and cadence; the optional loss
   comparison is reserved for same-objective ON/OFF equivalence
   (`compare_on_off_runs`).
- `gate_entropy` is the entropy of the accumulated mean gate probability, as
  documented by the raw schema; it is not silently relabeled as a mean of
  per-sample entropies.

## SECOND-ROUND CORRECTION — reviewer blocking items

The preceding first-round sentence claiming a PASS from
`compare_pair_runs(..., compare_losses=True)` for the real ON/OFF pair was
incorrect.  The actual pre-second-round helper result was `INVALID` with 15
`is_probe mismatch` entries.  The raw/manual invariants were equal, but that
did not constitute a passing automated comparator.  This section supersedes
that claim.

Files changed in this round: `experiments/stage_E1_instrumentation.py`,
`experiments/stage_E1_run.py`, `experiments/stage_E1_orchestrator.py`,
`experiments/tests/test_stage_E1_instrumentation.py`,
`experiments/tests/test_stage_E1_equivalence.py`, and this report.  The design
specification and research factors were not changed.

1. **Common-support early-stop comparison.** `compare_pair_runs` now keys rows
   by `(kind, epoch, step)`, compares only the intersection, and permits an
   asymmetric tail only when the shorter arm records an observed stopped epoch
   and a legal `censored_after_epoch`.  The returned object records both trace
   lengths, common/arm-only row counts, observed epochs, stop epochs and
   censoring boundaries.  Any asymmetric row at or before the shorter arm's
   stop, or any unexplained length/support difference, remains `INVALID`.

2. **Pair identity and formal protocol.** Pair validity now checks complete
   controlled config equality (only weighting/loss-weighting, run path/hash,
   and explicitly requested bypass fields may differ), initial parameter hash,
   split counts/positive rates, data/domain/task identity, and partition
   groups/counts/shapes/hashes.  Formal pair checks additionally require the
   registered 50-epoch protocol and its manifest marker.  Resume validation
   independently checks those constants, normalized dataset path, contiguous
   observed/censored epochs, metric cardinality, identity hashes and partition
   metadata; a 3-epoch smoke output is not eligible for `done_set`.

3. **ON/OFF equivalence semantics.** `compare_on_off_runs` explicitly ignores
   only `is_probe`, while retaining batch key/fingerprint/order checks,
   per-step loss tolerance `1e-7`, all `metrics_epoch.csv` and
   `metrics_summary.csv` fields at `1e-8`, initial/final parameter hashes, and
   non-consuming RNG restoration evidence.  Missing or malformed evidence is
   invalid.

### Second-round verification

All commands below were run from `D:\CODE\phase0-3`; no formal 24-run matrix
was launched, no `group2.md`/`group3.md` was read, and no scientific result
was interpreted.

1. Syntax compilation of the six Stage E1 modules/tests: **PASS**, exit code
   `0`.

2. `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -m pytest -q
   experiments/tests/test_stage_E1_instrumentation.py
   experiments/tests/test_stage_E1_equivalence.py`: **PASS**, `9 passed`.
   The tests now cover legal unequal early-stop tails, common-support
   fingerprint invalidation, identity mismatch, OFF/ON `is_probe` differences,
   formal completion/resume checks, and stale 3-epoch rejection.

3. Real three-epoch same-config ON/OFF smoke (`seed=2022`, `d0_large`,
   `equal`, batch `40960`, CUDA) was run into isolated temporary directories
   `_tmp_stage_E1_second_off` and `_tmp_stage_E1_second_on`.  The corrected
   `compare_on_off_runs(off, on)` returned **`VALID`** with:
   `equal_trace_row_count=18`, `sample_trace_row_count=18`,
   `common_trace_row_count=18`, `max_abs_train_loss_diff=0.0`,
   `max_abs_architecture_loss_diff=0.0`, `metrics_max_abs_diff=0.0`,
   `parameter_hash_equal=true`, and `rng_evidence_equal=true`; the ignored
   field was exactly `is_probe`.

4. The same real smoke was passed to formal `validate_run_artifacts`; it
   returned **`False`** with expected reasons including
   `formal_config_epoch_mismatch`, `formal_config_instrument_mismatch`,
   `formal_valid_marker_missing`, and observed/metric cardinality mismatch.
   This is the explicit evidence that a 3-epoch output cannot be resumed as a
   formal run.  The temporary raw directories were removed after inspection.

5. The P0-4 versus E1-OFF three-epoch final-state regression was not rerun in
   this coder round; the immediately preceding independent reviewer recorded
   it as PASS (all final state tensors equal, max absolute difference `0`, and
   validation/test values agreeing within the historical CSV formatting
   tolerance).  No new coder claim is made beyond that evidence.

6. The formal runner command remains unchanged and is permitted only after a
   fresh reviewer PASS:

   `$env:PYTHONPATH='D:\CODE\.venv\Lib\site-packages'; python -u
   experiments/stage_E1_orchestrator.py`
