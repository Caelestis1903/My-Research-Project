# Stage S2 Coder Implementation Report

Status: IMPLEMENTED and repaired after the second independent Stage S2 review
verdict `FAIL`. This is an implementation hand-off only. No formal Equal run,
formal queue, scientific analysis, or runner-stage conclusion was performed.

## Scope and unchanged surfaces

The implementation is an isolated S2 layer. The following frozen surfaces
were read and imported but not edited:

- `experiments/stage_S1_data.py`
- `experiments/stage_S1_models.py`
- `experiments/stage_S1_metrics.py`
- `experiments/stage_S1_protocol.py`
- `experiments/stage_S1_run.py`
- `experiments/stage_S1_orchestrator.py`
- all `experiments/stage_S1_raw_results/**`
- `m3oe/MDMTRec/models/multi_domain/model.py`

The audited S1 implementation hashes remain unchanged. In particular,
`stage_S1_run.py` remains
`7ad50d6a19f4b5a84939aa5f01977c8d703817ff560c2f5d181e7541aeda3210`, and
the M3OE model file remains SHA-256
`a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`.

## Design-to-code mapping

| Design requirement | Implementation |
|---|---|
| Exact fresh matrix | `experiments/stage_S2_protocol.py`: 16 unique `equal|dataset|setting|m3oe|seed` joint keys, four settings and seeds 2022–2025; `validate_equal_matrix` fails closed on drift. |
| Canonical fresh output layout | `equal_run_dir` emits `stage_S2_raw_results/equal/<dataset>/<setting>/m3oe/seed<seed>/joint`; repaired dry-run manifest confirms the namespace. |
| Exact S1 reuse set | `build_reuse_matrix` enumerates 16 M3OE joint Sample keys plus 48 matching single-domain MTL reference keys. The generated `experiments/stage_S2_reuse_audit.csv/.json` contains 64/64 `REUSE_OK` records. |
| Equal objective | `experiments/stage_S2_objectives.py`: fixed full-training `p_d`, `w_d=1/(3p_d)`, `p_dw_d=1/3`, cell coefficient `1/6`, population mean weight one, and fixed-count/no-double-weight guards. |
| Objective equivalence | `objective_oracle` proves Sample is the empirical per-example/task mean and Equal is `(1/3) sum_d mean_t(cell BCE)` for a matching fixed-prediction fixture. `assert_rejects_doubled_domain_share` is a negative fixture. |
| Shared main/architecture objective | `stage_S2_run.S2Runner._objective` is the only runner dispatch and is called by both `_train_epoch` and `_architecture_update`. Both paths use the same coefficient-table hash. |
| S1-derived runner | `stage_S2_run.py` subclasses the frozen `S1Runner`, reusing S1 data loading, M3OE construction/partition, checkpoint selection, deferred metrics, and RNG-preserving architecture-batch behavior. S2 changes are limited to Equal dispatch, S2 audit fields, and output identity/path. |
| Required objective logging | Per-run `objective_coefficients.json`; `batch_trace.csv` adds raw fractions, weighted mass by domain, total weighted mass, objective name/formula, and coefficient-table hash. |
| Checkpoint/test contract | S1 fit/evaluation path is retained: 50 epochs, validation-only earliest-tie selection, selected/final/common checkpoints, and deferred test evaluation. The S2 validator checks final/common-epoch-49 metric identity. |
| Two-process scheduler | `stage_S2_orchestrator.py` freezes eight within-setting two-seed co-run pairs, records the plan/digest in the parent queue manifest, passes pair/peer/worker identity to both children, and never compresses pending tasks. Scheduler locks are under the sibling `.stage_S2_locks/` namespace, never inside a child canonical directory. A retained sibling with an incomplete peer stops fail-closed for reviewer/user decision. |
| Fresh reuse gate | Every approved `run_formal_queue` invocation rebuilds and validates the exact 64-key S1 matrix before any child launch, fails closed on any rejection, and binds the queue manifest to a deterministic audit digest and source root. |
| Reuse validator | `audit_reuse_tasks` delegates to the frozen S1 `validate_run` read-only and records artifact hashes, data/split/physical-view fingerprints, code/runtime hashes, initialization and batch hashes, counters, selected epoch, checkpoint hashes, and reasons. |
| Joint/reference pairing | `stage_S2_orchestrator.py` re-runs the frozen S1 block audit over each 3-reference + joint block; `compare_equal_sample_manifests` checks held-constant data, initialization, batch order, optimizer/exposure, checkpoint, metric, and runtime fields. Report row pairing uses physical-view/semantic identity for all contrasts; batch-order equality is required only for the Equal/Sample D=3 pair, not for joint-vs-D=1 reference. Invalid rows fail `collect` and are excluded from every aggregate. |
| Structural report | `stage_S2_report.py` validates raw prediction metrics with the shared S1 AUC/LogLoss implementation, maps `selected`/`final` to their actual checkpoint files, builds the exact raw and paired schemas, and emits mechanical four-seed, role, fixed-physical rotation, per-common-epoch four-seed means, 3/5 stage-support, validation-difficulty, and method-stage summary tables. |
| Batch audit integrity | `S2Runner` retains main and architecture fingerprints; `validate_batch_trace_contract` recomputes ordered hashes, row counts, exposure, optimizer/example counters, and coefficient identity from `batch_trace.csv`. |
| Formal code provenance | Both runner and orchestrator inventories include the actual `m3oe/MDMTRec/models/multi_domain/model.py` source hash. |
| Failure boundary | No resume path is used in the S2 runner. `archive_failed_attempt` is append-only; a failed key is the only key eligible for a later epoch-0 retry. OOM/resource failure is returned to the review/user gate. |

## Files added

- `experiments/stage_S2_protocol.py`
- `experiments/stage_S2_objectives.py`
- `experiments/stage_S2_run.py`
- `experiments/stage_S2_orchestrator.py`
- `experiments/stage_S2_report.py`
- `experiments/tests/test_stage_S2_objective.py`
- `experiments/tests/test_stage_S2_pairing.py`
- `experiments/tests/test_stage_S2_orchestrator.py`
- `experiments/tests/run_stage_S2_smoke.py`
- `experiments/stage_S2_reuse_audit.csv`
- `experiments/stage_S2_reuse_audit.json`

Repair-round code/report changes were confined to the five S2 Python modules
above, `experiments/tests/test_stage_S2_pairing.py`,
`experiments/tests/run_stage_S2_smoke.py`,
`experiments/tests/test_stage_S2_orchestrator.py`, and this implementation
report. The repair regenerated only the two nonformal dry-run queue fixtures
listed below; no S1 source, S1 raw artifact, M3OE source, design, or review
report was modified.

The dry-run-only queue fixture `experiments/stage_S2_queue_manifest_dryrun.json`
was regenerated after repair. It contains 16 tasks and `launch_count=0`; it is
not a formal result. `stage_S2_queue_manifest_dryrun_repair.json` is an
additional equivalent repair-check fixture.

## Review finding → repair → regression evidence

| Review finding | Repair | Reproducible evidence |
|---|---|---|
| C1 lock polluted canonical child directory | `lock_path_for_task` hashes the formal key under `<raw_root>/.stage_S2_locks`; `_run_one` asserts the lock parent is not `run_dir` and returns its path for provenance. | `test_scheduler_lock_is_outside_canonical_child_dir`; EDDA two-child subprocess smoke crosses the real runner boundary and passes. |
| C2 main batch fingerprint/hash audit was empty and under-validated | `_train_epoch` appends the returned fingerprint; validator helper recomputes ordered main/architecture hashes, exact counts, exposure, and optimizer/example counters. | `test_batch_trace_contract_recomputes_hashes_and_counters`; `run_tiny_runner_fixture` exercises both optimizer paths and reports 3 main + 1 architecture fingerprints. |
| C3 approved launch skipped a fresh 64-key S1 audit | `run_formal_queue` unconditionally rebuilds/audits all 64 keys before scanning or launching fresh outputs; queue state records audit digest/counts/source root. | `test_approved_queue_fresh_reuse_gate_fails_before_child`; read-only EDDA audit: 64/64 valid, 0 rejected. |
| C4 joint/reference batch invariant was invalid and invalid rows aggregated | Pairing now checks physical view/identity/task semantics; batch/init/split equality is limited to Equal/Sample. `collect` raises on any invalid row; aggregators filter invalid rows. | `test_joint_reference_pairing_does_not_require_batch_order_identity`; `test_invalid_pair_rows_never_enter_aggregate_or_rotation_summaries`; real S1 d0 block physical-view/checkpoint smoke passed. |
| M1 selected/final hashes were empty | Added explicit label-to-artifact mapping for `selected_checkpoint.pth` and `final_checkpoint.pth`. | `test_checkpoint_label_maps_selected_and_final_artifacts`; real S1 d0 block confirmed both hashes nonempty. |
| M2 smoke did not cover runner/optimizers/CUDA/subprocess isolation | Smoke invokes real tiny `S2Runner._train_epoch` and `_architecture_update`; two concurrent subprocesses run synthetic M3OE on `cuda:0` and write isolated temporary JSON outputs. | `run_stage_S2_smoke.py --allow-skip` passed in EDDAenv; both child records reported CUDA, 3 main and 1 architecture fingerprints. |
| M3 failed queue overstated launched count | Final/intermediate queue payload uses `launch_index` and records `completed_record_count`, not pending-task length. | `test_failed_pair_manifest_reports_actual_launch_count` returns `launched=2` after a synthetic first-pair failure in a 16-key queue. |
| M4 code inventory omitted M3OE source | Added `m3oe/MDMTRec/models/multi_domain/model.py` to runner and orchestrator inventories. | `test_formal_code_inventory_includes_actual_m3oe_forward_source`; source hash remains `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`. |
| M5 fresh path omitted `equal/` | Updated protocol path builder and regenerated dry-run manifests. | `test_fresh_paths_have_explicit_equal_namespace`; repaired dry-run first path contains `/stage_S2_raw_results/equal/` and launch count is zero. |
| M6 structural summaries were incomplete | Added MovieLens cross-setting fixed-physical summaries, a separately labeled KuaiRand natural fixed-physical view, and paired common-epoch summaries with expected 4-seed×5-epoch counts and negative/positive mechanical counts. | `test_invalid_pair_rows_never_enter_aggregate_or_rotation_summaries`; report exports both new CSVs. |

### Second-review finding → repair → regression evidence

| Finding | Repair | Evidence |
|---|---|---|
| C1 recovery compressed pending tasks and could cross settings or launch a singleton | Added deterministic `build_co_run_plan`: eight fixed `(2022,2023)`/`(2024,2025)` within-setting pairs. Recovery iterates this plan directly; one valid retained sibling plus an incomplete peer raises reviewer/user fail-closed, and no singleton path exists. | `test_co_run_plan_is_eight_stable_within_setting_pairs`; `test_retained_sibling_stops_instead_of_singleton_recovery`; `test_existing_cross_setting_or_odd_plan_is_rejected`. |
| M1 co-run provenance absent from child and parent artifacts | Added `co_run_pair_id`, `planned_peer_run_key`, and stable `worker_id` to task/plan, command-line args, child `config.json`, child `run_manifest.json`, parent queue records, and validator cross-checks. | `test_command_and_child_provenance_fields_are_explicit`; repaired dry-run manifest contains 8 plans and per-task peer/worker fields. |
| M2 training-stage support counted 20 seed-epoch signs instead of five four-seed means | Added `paired_training_epoch_mean_rows` and rewrote stage summary to compute one mean per group×common epoch, then count negative means and apply explicit `stage_support_3_of_5`. Seed-level values remain serialized separately. | `test_training_stage_gate_aggregates_four_seeds_before_counting_epochs`; `test_training_stage_gate_rejects_many_negative_seeds_when_epoch_means_are_positive`. |
| m1 smoke profile was under-representative | The nonformal CUDA subprocess smoke now uses a synthetic KuaiRand schema with 1,614,258 parameters and batch 40,960 for each one-step main/architecture run while writing only temporary outputs. | EDDAenv smoke passed; direct representative fixture reported `parameter_count=1614258`, one main and one architecture step. |
| m2 audit-only CLI ignored custom S1 root | Audit branch now builds the reuse matrix from `args.s1_raw_root`. | `test_audit_cli_honors_custom_s1_root`. |

## Verification performed

- EDDAenv Python syntax compilation passed for all five S2 modules and the
  smoke script.
- System-Python focused S2 tests passed: `24 passed, 2 skipped`. The two skips
  are Torch-dependent cases in the dependency-light interpreter; EDDAenv has
  no pytest installation.
- EDDAenv smoke passed:
  `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments/tests/run_stage_S2_smoke.py --allow-skip`.
  It exercised tiny Equal/Sample objective equivalence, double-weight
  rejection, coefficient/batch audit, real tiny S2 main/architecture optimizer
  updates, M3OE forward/loss/backward, artifact serialization, and a
  batch-contract/pairing-corruption check plus a two-process representative
  KuaiRand-scale CUDA/M3OE subprocess-isolation smoke (1,614,258 parameters,
  batch size 40,960, one main and one architecture step per child).
- The read-only frozen S1 validator was rerun over all 64 M3OE reuse keys:
  `expected_count=64`, `valid_count=64`, `rejected_count=0`,
  `all_reuse_ok=true`. No S1 file or artifact was written or modified.
- S2 dry-run matrix check passed with exactly 16 fresh tasks, eight frozen
  within-setting pairs, and zero launches.
- No formal training/testing process was started, no S2 raw Equal artifact was
  generated, and no scientific result was computed.

## Formal tasks intentionally not run (initial implementation hand-off)

All 16 Equal tasks remain pending for the next stage:

- MovieLens: `d0_large`, `d1_large`, `d2_large` × seeds 2022–2025;
- KuaiRand: `natural` × seeds 2022–2025;
- backbone: M3OE only; method: Equal joint objective.

The orchestrator refuses to enter this queue without the explicit approval
flag required after independent review and user approval, and it also requires
the exact matching pre-frozen dry-run queue manifest (including its pair-plan
digest). If a frozen pair has one valid retained sibling and one incomplete
peer, recovery stops fail-closed for reviewer/user disposition. No Sample or
single-domain reference retraining is queued. This statement describes the
initial coder hand-off before the subsequent approved RUN; the later formal
16-run execution is retained by the post-run review and is not repeated by
this reporting-only fix.

## Known limitations / reviewer hand-off

1. The formal S2 validator and report are implemented but cannot validate
   fresh Equal artifacts until `luna_runner` produces them.
2. The reuse audit is immutable and currently valid for the frozen 64-key S1
   subset. A future invalid key must be handled as an exact replacement and
   requires reviewer approval; the other 63 keys must remain reused.
3. The current implementation report contains no AUC/LogLoss interpretation,
   hypothesis decision, evidence-strength label, or GO/STOP recommendation.
4. The repaired two-run concurrent CUDA/M3OE smoke passed in EDDAenv, but it
   should be rerun immediately before any formal launch. A resource failure
   must stop at the review/user gate; silently reducing concurrency is not
   implemented.
5. The repository-wide test command cannot collect unrelated Torch-dependent
   suites under the system Python (Torch is unavailable there); the focused S2
   suite and EDDAenv smoke were used for this implementation hand-off.

## Post-run reporting fix (collector-only IMPLEMENT回合)

The later post-run review verdict was `PASS WITH FIXES`; the formal execution
itself remained valid and the only blocker was the structural Equal/Sample
pairing layer. This implementation adds only the independent adapter
`experiments/stage_S2_postrun_collect.py` and its dedicated regression suite
`experiments/tests/test_stage_S2_postrun_collect.py`.

The adapter temporarily wraps the existing strict comparator at runtime. It
deep-copies both manifests, validates the existing
`stage_S2_protocol.canonical_dataset` identity, and changes only the copied
top-level `dataset` value. Both canonical `dataset_key` values must already be
exact, both `dataset_label` values must be present and equal, and both labels
must canonicalize to the same dataset. The unchanged comparator then checks
all remaining fields; no non-dataset mismatch is bypassed. Raw manifests,
configs, predictions, checkpoints, logs, queue manifests, S1 artifacts, and
the hash-bound core S2 modules are not edited.

### Post-run finding → repair → regression evidence

| Finding | Repair | Evidence |
|---|---|---|
| Top-level `dataset` differed only as `MovieLens-1M`/`movielens` or `KuaiRand-Pure`/`kuairand` | Added copied in-memory normalization using the frozen canonicalizer; strict comparator remains authoritative for every other field. | Two legal-alias regressions and the real 16-block audit pass: 12 MovieLens + 4 KuaiRand mappings. |
| Unknown/missing/cross-dataset identity could be accepted | Require canonical `dataset_key`, equal present labels, and canonicalizable labels; failures are fail-closed. | Four negative identity regressions pass. |
| A reporting adapter could accidentally hide provenance drift or mutate sources | Preserve strict mismatch reasons, record zero non-dataset bypasses, deep-copy inputs, and hash every file in all 80 raw run directories before/after collection. | Seven held-constant-field mismatch regressions, deterministic/no-mutation regression, and actual collector dry-run with unchanged raw snapshot. |
| Collection contract needed re-validation after the adapter | Delegate existing reuse audit, fresh validators, prediction/metric recomputation, uniqueness checks, and structural writers; enforce 2,016 raw rows, 2,016 paired rows, 80 inventory rows, zero invalid pairs, 16 fresh OK, and 64 reused OK. | Actual `--dry_run` completed with `raw_row_count=2016`, `paired_row_count=2016`, `run_inventory_count=80`; adapter accepted exactly 16 dataset normalizations and zero bypasses. |

### Post-run verification and boundary

- Dedicated post-run regressions: `15 passed`.
- Combined dedicated plus existing S2 objective/pairing/orchestrator suite:
  `39 passed, 2 skipped` (the two skips are dependency-light Torch skips).
- EDDAenv syntax compilation passed for the new adapter and its tests were
  executed with the system Python.
- Collector dry-run command:
  `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments/stage_S2_postrun_collect.py --dry_run --s2_raw_root experiments/stage_S2_raw_results --s1_raw_root experiments/stage_S1_raw_results`.
  It wrote derived tables/audits only to a temporary directory, then removed
  that directory; no final structural output was released to the canonical
  `experiments` directory.
- The dry-run performed the read-only 64-key reuse audit, all 16 fresh
  validators, prediction/metric recomputation, cardinality/uniqueness gates,
  and pre/post full-file hashes over 16 fresh plus 64 reused raw run
  directories. The raw snapshot was unchanged.
- No retraining, test inference, checkpoint recreation, queue modification,
  or scientific analysis was performed or permitted. This fix requires no
  retraining; after independent review, only a bounded collector continuation
  may write the prescribed derived structural outputs.
