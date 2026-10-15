# REVIEW REPORT

## 1. Verdict: PASS

The third independent review finds all blocking findings from the first and second reviews corrected. The formal Stage S2 implementation is safe to enter `luna_runner` **only after the user gives explicit approval**. This verdict does not itself authorize a formal launch.

No implementation file was modified during review. No formal training, formal test evaluation, or scientific result analysis was performed.

## 2. Requirement-by-Requirement Verification

| # | Design acceptance requirement | Result | Independent verification |
|---:|---|---|---|
| 1 | Exactly 16 fresh Equal M3OE joint keys | PASS | The current dry-run contains 16 unique `equal|dataset|setting|m3oe|seed` keys, all four settings and seeds 2022–2025, eight fixed co-run pairs, explicit `equal/` output paths, and zero launches. |
| 2 | All 64 frozen S1 M3OE artifacts pass a complete read-only audit | PASS | A fresh reviewer rerun returned `expected_count=64`, `valid_count=64`, `rejected_count=0`, `all_reuse_ok=true`. The checked-in audit has 64 unique keys and populated artifact, checkpoint, initialization, batch-order, code, runtime, split, and physical-view evidence. |
| 3 | Sample is ordinary shuffled concatenated-loader BCE with effective `p_d` | PASS | Frozen `stage_S1_run.py` remains SHA-256 `7ad50d6a...3210`; its loader is ordinary `shuffle=True`, and both main and architecture paths use the direct unweighted per-example/task BCE mean. |
| 4 | Equal changes only to fixed `w_d=1/(3p_d)` under identical exposure | PASS | `stage_S2_objectives.py` derives shares from immutable full-training counts, applies the fixed coefficient once after task averaging, and enforces `p_d w_d=1/3`, cell coefficient `1/6`, and population mean weight 1. Tests reject double weighting; no sampler or per-batch renormalization exists. |
| 5 | Main and architecture paths use the same Equal objective | PASS | Both `S2Runner._train_epoch` and `_architecture_update` call the single `_objective` dispatch. The real two-process CUDA smoke executed one main and one architecture step per child with the same coefficient-table hash. |
| 6 | Equal/Sample pairing invariants are enforced | PASS | Manifest comparison enforces data/split/view, initialization, model, runtime, optimizer, exposure, batch-order, metric, and checkpoint invariants. Batch-trace hashes and counters are recomputed. Joint/reference pairing uses physical-view/semantic evidence without requiring invalid D=3/D=1 batch equality. |
| 7 | M3OE architecture/model and baselines remain unchanged | PASS | M3OE source remains SHA-256 `a7702a64...ce1`; the S1 audit validates the frozen S1 code inventory. The actual M3OE source is included in the S2 formal inventory. No added backbone, sampler, schedule, Momentum/EMA, curriculum, or tuning path was found. |
| 8 | Validation-only checkpoint selection and deferred test evaluation | PASS | S2 retains the frozen S1 selection/evaluation flow. The validator enforces validation-only earliest-tie selection, deferred test flags, selected/final/common coverage, artifact hashes, and numerical final/common-epoch-49 agreement. |
| 9 | Exact unique 2,016-row raw and paired schemas | PASS structurally | The collector requires 2,016 raw rows, indexes exact unique method-cell keys, generates exactly 2,016 paired rows, and refuses any invalid pairing before aggregation. Fresh Equal rows necessarily remain unavailable until the approved run. |
| 10 | Shared metrics, prediction recomputation, correct domain/task identity | PASS | The shared S1 tie-aware AUC/fixed-epsilon LogLoss module is imported. Deferred predictions are independently recomputed against every metric row; cell, physical-domain, physical-view, and task identities are checked. |
| 11 | Two-process scheduler isolates outputs, records co-runs, and fails safely | PASS | The queue freezes eight same-setting pairs: `(2022,2023)` and `(2024,2025)` for each setting. It launches complete pairs only, waits for both futures, stops after a failed pair, uses external locks, and records pair/peer/worker, PID, times, device, peak memory, exit code, and validator result. |
| 12 | Failed-key-only epoch-0 recovery is append-only and controlled | PASS | Invalid attempts are archived append-only and no mid-run resume exists. A pair with one valid retained member and one incomplete/invalid member now stops fail-closed before launch; it cannot silently overwrite the valid sibling, run a singleton, or re-pair with another setting. Any execution-plan decision after such a partial pair therefore returns to reviewer/user. |
| 13 | Objective, runner, artifact, corruption, and concurrent-resource smokes pass | PASS | Focused tests passed `24 passed, 2 skipped`; the skips are Torch-only cases in the dependency-light interpreter. EDDA/CUDA smoke passed, including two concurrent representative KuaiRand-scale children. |
| 14 | Changes remain localized and frozen artifacts are untouched | PASS | Implementation/report inventory is complete. Fresh S1 validation and frozen hashes pass. `experiments/stage_S2_raw_results` contains zero files and zero directories. |
| 15 | Reviewer gate and explicit user approval | PASS | This report supplies the independent PASS. The CLI still refuses formal execution without `--approved` and an exact matching pre-frozen queue manifest. The user's approval remains mandatory. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Equal implements `mean_i(w[d_i] * mean_t(BCE_i,t))`, with `w_d=1/(3p_d)` derived only from immutable full-training counts.
- Tensor guards require matching nonempty `[B,2]` predictions/targets and `[B]` valid domain IDs. Nonfinite predictions, targets, weights, objectives, gradients, and parameters fail immediately.
- No balanced sampling, domain cycling, oversampling, undersampling, per-batch normalization, or repeated `p_d` factor is present.
- The 16 fresh and 64 reused key matrices are exact; the report's raw/paired key construction and 2,016-row contracts are consistent.

### Gradient flow & Loss participation

- The same connected Equal objective participates in both main and architecture backward passes.
- The main optimizer owns non-architecture parameters; the architecture optimizer owns the M3OE architecture parameters. Both update paths were exercised in CUDA subprocesses.
- Main and architecture batch fingerprints are populated, and the validator recomputes their ordered hashes, exact row counts, exposure, and optimizer/example counters.
- The representative smoke returned, for each of two independent children: `parameter_count=1,614,258`, `main_optimizer_steps=1`, `architecture_optimizer_steps=1`, one main fingerprint, one architecture fingerprint, and two objective calls.

### Controlled variables & Baseline preservation

- A fresh 64-key reuse audit passed before this verdict; approved queue execution repeats it before any child launch.
- S1 data/model/metric/runner artifacts are reused read-only. Current S1 runner and M3OE hashes match the frozen evidence.
- Equal/Sample equality checks cover data, split/view identities, semantic initialization, parameter manifest/FLOPs, batch order, optimizer/exposure, runtime, metric, and checkpoint contract.
- Test evaluation remains deferred until validation-based selection is frozen.
- The scheduler's concurrency blocks are now deterministic, same-setting, and immutable across initial execution and state scans.

### Fail-fast enforcement

- A missing or changed pre-frozen plan, stale reuse artifact, malformed batch trace, mismatched child provenance, invalid paired row, incomplete artifact/hash inventory, test leakage, or nonfinite value stops execution/reporting.
- `co_run_pair_id`, `planned_peer_run_key`, and `worker_id` are required by the command builder and runner; they are written to child config/manifest and cross-checked by `validate_equal_run`.
- Partial-pair recovery fails before any child launch. There is no compressed pending list, cross-setting reassignment, or singleton tail.
- A first child/validator failure stops later pair scheduling while the already-running peer is allowed to finish and retain its evidence.

## 4. Over-Engineering & Code Bloat Audit

The implementation remains an isolated S2 layer and adds only experiment-critical objective, provenance, validation, reporting, and scheduling logic. No production framework, registry/factory layer, generic retry system, or speculative compatibility abstraction was introduced. The added guards directly prevent silent experimental drift and are proportionate to the research risk.

## 5. Bugs / Deviations Found

### Status of all previous blocking findings

| Finding | Third-review status | Evidence |
|---|---|---|
| Lock inside canonical child directory | FIXED | Lock paths are under `<raw_root>/.stage_S2_locks`; the concurrent smoke confirmed both canonical directories were lock-free. |
| Main batch fingerprints absent | FIXED | `_train_epoch` appends every fingerprint; validator recomputation and smoke counts pass. |
| Approved launch skipped fresh reuse audit | FIXED | Fresh audit is unconditional before state scan/launch; 64/64 passed independently. |
| Joint/reference false batch invariant and invalid-row aggregation | FIXED | Pairing scope is correct; complete collection fails on invalid rows and aggregators filter them. |
| Selected/final checkpoint hashes absent | FIXED | Label-to-artifact mapping resolves selected, final, and common checkpoints. |
| Inadequate CUDA/two-optimizer resource smoke | FIXED | Checked-in smoke now runs two representative KuaiRand-scale M3OE/CUDA subprocesses with both optimizer paths. |
| Incorrect launch count | FIXED | Parent queue records actual launch count and completed-record count. |
| M3OE source omitted from inventory | FIXED | Actual model source is included and hash-validated. |
| Missing `equal/` output namespace | FIXED | Protocol and both dry-run fixtures use the explicit namespace. |
| Missing structural summaries | FIXED | Four-seed, role, fixed-physical rotation, epoch-mean, stage-support, difficulty, and method-stage outputs are present. |
| Recovery re-paired pending tasks / singleton tail | FIXED | Eight frozen pairs are iterated directly; one-valid partial pairs stop fail-closed. |
| Missing co-run provenance | FIXED | Task, plan, command, child config/manifest, validator, and parent record fields agree. |
| 3/5 gate counted seed-epoch signs | FIXED | One four-seed mean is computed per common epoch before counting negative epochs. |

### Independent reproduction of the three second-review blockers

1. **Frozen co-runs and recovery:** the dry-run has eight unique pair IDs and reciprocal two-key assignments. Regression tests reject a changed cross-setting/odd plan and prove a retained sibling prevents any child launch. Scheduler inspection confirms no pending-list compaction exists.
2. **Provenance cross-consistency:** all 16 tasks agree with their pair records on pair ID, reciprocal peer key, and worker ID. Those exact values are passed on the child command line, required by the runner, persisted in config/manifest, copied into parent records, and rejected on mismatch by the formal validator.
3. **Epoch support:** an independent fixture produced epoch means `{9:-0.045, 19:-0.045, 29:-0.045, 39:0.005, 49:0.005}`. The implementation counted three negative epoch means and set `stage_support_3_of_5=1`. A separate regression with many negative individual seeds but positive epoch means correctly returns zero support.

### CRITICAL

None.

### MAJOR

None.

### MINOR

None requiring correction before the formal run.

## 6. Corrections Made

None. Per the review-stage instruction, the reviewer did not modify implementation code. Only this independent review report was rewritten.

## 7. Remaining Risks

- Fresh Equal artifacts do not yet exist, so full 16-run artifact validation, 2,016-row collection, and Equal/Sample equality can only be exercised after formal execution. The implemented validators are fail-closed for that boundary.
- If a formal pair ends partially valid, the queue intentionally stops instead of choosing a singleton or new peer. Resuming then requires a renewed reviewer/user execution-plan decision; this is an explicit safety boundary, not an automatic recovery path.
- A machine/resource change between review and launch could invalidate the capacity smoke. The representative two-process smoke should be rerun immediately before formal launch; any failure must stop without silently lowering concurrency.
- Scientific interpretation, hypothesis support, and evidence-strength labels remain reserved for main analysis after the runner completes and structural validation passes.

## 8. Whether Experiments Are Safe to Run

**YES, after and only after explicit user approval.** The approved `luna_runner` must use the exact frozen 16-key/eight-pair dry-run manifest, rerun the representative CUDA concurrency smoke, and allow the orchestrator's mandatory fresh 64-key reuse audit to pass before the first formal child starts. Any partial pair, validator failure, provenance mismatch, reuse rejection, or resource failure must return to the review/user gate.

Independent review evidence:

- focused S2 tests: `24 passed, 2 skipped` (the skips are dependency-light-interpreter Torch cases covered by EDDA smoke);
- EDDA syntax compilation: PASS;
- checked-in objective/M3OE/artifact/concurrency smoke: PASS;
- direct concurrent CUDA smoke: two distinct PIDs, both exit 0, both `cuda:0`, both 1,614,258 parameters, batch 40,960, one main and one architecture step, canonical locks isolated;
- fresh frozen S1 audit: 64/64 valid, 0 rejected;
- dry-run: 16 tasks, 8 same-setting reciprocal pairs, 0 launches, matching pair-plan digest;
- formal S2 output tree: 0 files, 0 directories.

---

## 9. Post-run Data Acceptance Review

### 9.1 Post-run verdict: PASS WITH FIXES — collector blocked, formal runs retained

The completed formal execution is **not invalidated**. All 16 fresh Equal children returned code 0 and passed the formal child validator; the queue is `COMPLETE`; all 64 reused S1 keys pass a new read-only audit; and there were no failures, retries, or partial pairs. The current blocker is confined to the structural pairing/collection layer.

The required raw/paired/run/summary tables are not yet accepted because the fail-closed collector correctly stopped before writing them. Main scientific analysis must remain blocked until the reporting fix is independently reviewed and the collector completes successfully.

### 9.2 Independent diagnosis

All 16 Equal/Sample joint blocks were compared field by field. The only reason returned by `compare_equal_sample_manifests` was:

`pairing_field_mismatch:dataset` — 16/16 blocks.

The two observed patterns were:

| Blocks | Fresh manifest `dataset` | Fresh `dataset_key` | Fresh `dataset_label` | Frozen S1 `dataset` | Frozen S1 `dataset_key` | Frozen S1 `dataset_label` |
|---:|---|---|---|---|---|---|
| 12 MovieLens | `MovieLens-1M` | `movielens` | `MovieLens-1M` | `movielens` | `movielens` | `MovieLens-1M` |
| 4 KuaiRand | `KuaiRand-Pure` | `kuairand` | `KuaiRand-Pure` | `kuairand` | `kuairand` | `KuaiRand-Pure` |

Both raw strings canonicalize to the same dataset key. Fresh child `config.json`, task keys, paths, formal keys, and all fresh `metrics_test.csv` rows retain the canonical dataset key. The fresh manifest's top-level `dataset` value was overwritten at serialization time by the human-readable `bundle_manifest["dataset"]`; this did not change the loaded dataset or any experimental input.

Independent comparison found **zero mismatches** across all 16 Equal/Sample blocks for:

- dataset file SHA-256;
- selected/base split fingerprints and row-ID hashes;
- physical-view train/validation/test fingerprints;
- preprocessing metadata, features, task definitions, physical-domain identities, train counts/shares, and cell statistics;
- semantic initialization hash, parameter manifest, and forward-FLOP estimate;
- runtime fingerprint, both optimizer definitions, batch counts/exposure/steps, and main/architecture batch-order hashes;
- metric/checkpoint contract, fixed final epoch, and common epochs;
- per-checkpoint metric-row physical-domain identity, task identity, and train/validation/test sample and positive counts.

All 48 joint/reference physical-domain links were also checked. Dataset hashes, common base-split fingerprints, task definitions, per-domain physical-view fingerprints, and emitted metric identities/counts agree. Joint and D=1 reference row-ID hashes differ as expected because their row sets have different structural scope; the frozen S1 block validator accepts all 16 blocks and is the authoritative semantic pairing check. A fresh full audit returned 64/64 reused keys valid and 0 rejected.

Therefore this is a **pure canonical-label mismatch**, not substantive provenance drift.

### 9.3 Whether coder may apply a minimal normalization

**YES, but only in a post-run reporting compatibility layer.** The normalization may canonicalize the two manifest `dataset` strings in memory solely for Equal/Sample pairing, then delegate every other comparison to the existing strict comparator.

The preferred and approved implementation scope is:

1. Add one small post-run collector adapter, for example `experiments/stage_S2_postrun_collect.py`.
2. Add focused regression tests, preferably `experiments/tests/test_stage_S2_postrun_collect.py`.
3. Add a post-run implementation addendum documenting the exact normalization and source hash.
4. After review approval, allow generation of only the prescribed derived structural outputs from the existing immutable artifacts.

The adapter must:

- canonicalize using the existing `stage_S2_protocol.canonical_dataset`; no new alias table may be invented;
- require both sides' `dataset_key` to equal the canonical value;
- require both sides' `dataset_label` to be present, equal, and canonicalizable to the same value;
- normalize only copied in-memory manifest dictionaries and never write a raw manifest;
- delegate all remaining held-constant fields to the existing comparator unchanged;
- run the existing fresh validators, 64-key reuse audit, prediction/metric recomputation, uniqueness checks, and 2,016-row cardinality gates;
- emit an explicit normalization audit containing the 16 formal block keys, original two values, canonical value, adapter source hash, and zero non-dataset bypasses;
- record pre/post hashes proving the 16 fresh and 64 S1 raw manifests/artifacts were not modified.

### 9.4 Explicitly forbidden changes

The coder must not modify:

- any fresh S2 or frozen S1 manifest, config, metric, prediction, checkpoint, batch trace, log, or artifact hash;
- `stage_S2_run.py`, objective/protocol/model/data/metric code, the queue manifest, or any formal run output;
- any scientific metric or row value;
- any comparator field other than the in-memory canonical treatment of top-level `dataset`;
- the existing run report's factual record of the collector failure.

The current fresh manifests bind the formal run to exact hashes of `stage_S2_orchestrator.py` and `stage_S2_report.py`. Directly editing either hashed module would make the existing formal validator report source-hash drift. Therefore an edit to those files is **not approved** as the minimal fix unless a new review first designs an explicit, fail-closed provenance treatment for post-run code versions. The small external adapter avoids conflating training-time source provenance with a reviewed reporting compatibility correction.

### 9.5 Mandatory regressions

The new tests must prove all of the following:

1. `MovieLens-1M` versus `movielens` is accepted only when both keys are `movielens` and both labels identify MovieLens.
2. `KuaiRand-Pure` versus `kuairand` is accepted only when both keys are `kuairand` and both labels identify KuaiRand.
3. Unknown labels, cross-dataset values, missing labels, or mismatched `dataset_key` fail closed.
4. Corrupting any other held-constant field—at minimum dataset hash, split row-ID hash, physical-view fingerprint, task/preprocessing identity, initialization hash, or batch-order hash—still fails.
5. The real 16-block audit has exactly 16 allowed dataset normalizations and zero other mismatches.
6. Complete collection yields exactly 2,016 unique raw rows, 2,016 unique paired rows, 80 run inventory rows, zero invalid pair rows, 16 valid fresh runs, and 64 valid reused runs.
7. The normalization audit is complete and all raw-artifact pre/post hashes are unchanged.

### 9.6 Retraining decision

**Retraining is neither required nor permitted for this issue.** The dataset bytes, split/view identities, task semantics, initialization, exposure, optimizers, batch order, checkpoints, predictions, and formal validators all agree. Repeating training would add stochastic/resource exposure without correcting the report-layer string comparison and would unnecessarily replace valid evidence.

### 9.7 Required re-review and collector-only rerun

After coder implementation, the reviewer must independently:

1. inspect the adapter and tests and confirm that only `dataset` is normalized in memory;
2. rerun the new negative regressions and the existing S2 objective/pairing/orchestrator suite;
3. confirm current formal source hashes and all 80 raw artifact inventories remain unchanged;
4. inspect the normalization audit and reproduce the 12 MovieLens plus 4 KuaiRand mappings;
5. authorize a bounded collector-only continuation;
6. after collection, verify exact output cardinalities, key uniqueness, zero invalid pairings, metric recomputation, summary provenance, and unchanged raw hashes before releasing the data to main analysis.

**Rerunning only the collector is compliant and is the required action after the reporting fix passes review.** It must read the existing artifacts and write only derived structural tables/audits. It must not launch `stage_S2_run.py`, recreate checkpoints/predictions, rerun test inference, or alter any formal/raw artifact. Scientific analysis remains forbidden until the post-collector acceptance check passes.

---

## 10. Post-run Reporting Fix Re-review

### 10.1 Verdict: PASS

The new post-run adapter faithfully implements the narrow reporting-only correction authorized in Section 9. No blocking issue remains for a bounded formal collector continuation. This PASS authorizes only `luna_runner` execution of the post-run collector against the existing immutable artifacts; it does not authorize retraining, checkpoint recreation, test inference, or scientific analysis.

No implementation or data file was modified during this re-review. Only this review report was appended.

### 10.2 Scope and provenance preservation

The implementation added only:

- `experiments/stage_S2_postrun_collect.py`;
- `experiments/tests/test_stage_S2_postrun_collect.py`;
- the post-run implementation-report addendum.

The adapter source SHA-256 independently observed during review is:

`a7ed02735ebb845190e6269d4bf3078e160422d9f265a698373d62801bfc7406`.

All 11 files in the formal S2 source inventory still match the hashes embedded in every one of the 16 fresh manifests. In particular:

- `stage_S2_run.py`: `d5f4e4345aa749656dca6575a057bf5dbeb173ca7f1f1c5ebd3af6c4933a6334`;
- `stage_S2_orchestrator.py`: `d76d0b80bface06de386194fa38fe35798fe18a5619f623c18ae590ed959a39e`;
- `stage_S2_report.py`: `fccd9ddb66ebb0290c39b3c6fb98ae0fd390b775e836308ac8c7e730d212e477`;
- frozen `stage_S1_run.py`: `7ad50d6a19f4b5a84939aa5f01977c8d703817ff560c2f5d181e7541aeda3210`;
- frozen M3OE source: `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`.

The formal queue manifest remained `COMPLETE`, with 16 launched records, 16 valid records, and zero invalid records. Its hash before and after the reviewer dry-run was unchanged:

`ad16ebb78bc5ceaada41bec77722bf1304e8340c59b7f512ee6efa2b8e1d8a2`.

The adapter's full pre/post snapshot covered every file in all 80 run directories—16 fresh and 64 reused—and returned exact equality. Representative fresh and S1 manifest hashes and timestamps remained unchanged, and no canonical structural CSV, summary manifest, or normalization audit was written during review.

### 10.3 Algorithm and fail-closed audit

`normalize_dataset_pair` deep-copies both manifests, validates the existing canonicalizer, and changes only the copied top-level `dataset` field. It requires:

- a present, recognized raw dataset value on both sides;
- an exact canonical `dataset_key` on both sides;
- present and exactly equal `dataset_label` values;
- label, key, and raw value to resolve to the same canonical dataset.

The adapter then calls the captured, unchanged strict comparator. It has no suppression or allow-list for other reasons: every returned non-dataset reason remains blocking and `non_dataset_bypass_count` is fixed and validated as zero. The temporary module-level comparator wrapper is restored in a `finally` block; independent inspection after the full dry-run confirmed identity with the original strict comparator.

No runner module is imported or launched. The adapter delegates the existing 64-key reuse audit, 16 fresh validators, prediction-to-metric recomputation, pairing audit, uniqueness/cardinality checks, and structural writers.

### 10.4 Independent regression reproduction

The combined focused suite passed:

`39 passed, 2 skipped`.

The two skips are the existing dependency-light-interpreter Torch cases, not post-run adapter tests. All 15 dedicated post-run cases passed.

The reviewer additionally injected the following independent corruptions into copied manifests and confirmed that each failed with the original `pairing_field_mismatch:<field>` reason and zero bypasses:

- `dataset_file_hash`;
- `split_row_id_hashes`;
- `initial_parameter_state_hash`;
- main `batch_order_hash`;
- `architecture_batch_order_hash`;
- main `optimizer`;
- `architecture_optimizer`;
- `fixed_final_epoch` and `common_epochs` checkpoint contract;
- `metric_implementation`.

Legal `MovieLens-1M`/`movielens` and `KuaiRand-Pure`/`kuairand` pairs passed without mutating inputs. Unknown dataset, cross-dataset identity, missing dataset, missing label, and mismatched canonical key all failed closed. Repeated inspection produced byte-equivalent audit records in stable formal-key order.

### 10.5 Audit completeness and determinism

The normalization audit records:

- adapter path and source hash;
- all 16 Equal and Sample formal keys;
- both original dataset values and the canonical value;
- both keys and labels;
- pre/post hashes of the original and normalized in-memory manifest representations;
- strict comparison reasons and explicit zero bypass count;
- full Equal/Sample source snapshots for each block;
- full pre/post snapshots for all 80 raw run directories, including per-file hashes and artifact counts;
- mapping counts, structural cardinalities, invalid-pair count, and the no-analysis scope marker.

Ordering is deterministic: raw snapshots are sorted by formal key, directory/file traversal is sorted, normalization records are sorted by Equal formal key, and JSON is written with sorted keys. Two repeated real 16-block inspections were identical. The full reviewer dry-run also proved that its independently computed pre/post raw snapshots were identical.

### 10.6 Real-artifact dry-run result

The reviewer ran the adapter against the real artifacts with derived output confined to an automatically removed temporary directory. The process exited 0 and returned:

| Contract | Observed |
|---|---:|
| Raw rows | 2,016 |
| Paired rows | 2,016 |
| Run inventory | 80 |
| Allowed normalization blocks | 16 |
| MovieLens mappings | 12 |
| KuaiRand mappings | 4 |
| Invalid pair rows | 0 |
| Non-dataset bypasses | 0 |
| Raw artifact snapshot unchanged | true |

No formal training process was started, and the canonical derived-output directory remained untouched.

### 10.7 Bugs / deviations

#### CRITICAL

None.

#### MAJOR

None.

#### MINOR

None requiring correction before the collector continuation.

### 10.8 Authorization boundary

**YES: `luna_runner` may execute only the formal post-run collector.** It must invoke `stage_S2_postrun_collect.py` against the existing canonical S2/S1 raw roots and write only the prescribed derived structural tables and normalization audit to the canonical experiments output directory.

The runner must not invoke `stage_S2_run.py`, modify the queue, retrain any key, regenerate predictions/checkpoints, or interpret metrics. After collector completion, the resulting canonical files must receive a final data-acceptance check for hashes, cardinalities, uniqueness, zero invalid pairs/bypasses, and normalization-audit linkage before main analysis begins.

---

## 11. Final Post-collector Data Acceptance

### 11.1 Verdict: PASS

The canonical Stage S2 structural outputs pass final data acceptance. The collector completed without changing the queue or any of the 80 raw run directories. Cardinality, uniqueness, method/comparison coverage, checkpoint coverage, normalization provenance, checkpoint hashes, prediction-derived metrics, paired deltas, and mechanical aggregate tables were independently checked.

No experiment file or derived output was modified during this acceptance review. Only this review report was appended. No training, test inference, or scientific interpretation was performed.

### 11.2 Queue and immutable-source acceptance

The formal queue remains:

- `resume_state=COMPLETE`;
- expected/launched/record count: 16/16/16;
- valid/invalid child records: 16/0;
- reuse expected/observed/valid: 64/64/64.

The queue SHA-256 remains unchanged from the pre-collector review:

`ad16ebb78bc5ceaada41bec77722bf1304e8340c59b7f512ee6efaf2b8e1d8a2`.

The reviewer recomputed the current full-file snapshot of all 80 raw run directories and compared it with the formal normalization audit's post-collector snapshot. The two mappings are exactly equal, including every per-file SHA-256 and artifact count. The audit's own pre/post snapshots are also exactly equal. Thus the collector did not alter any fresh manifest, S1 artifact, prediction, checkpoint, metric file, batch trace, config, or log.

### 11.3 Canonical output inventory

| Output | Rows / status | SHA-256 |
|---|---:|---|
| `stage_S2_raw_results.csv` | 2,016 | `7a566933bd36885ca67a67fcbaaeeae2811d2c5b2e6a637bfaa620da4c1b833f` |
| `stage_S2_paired_results.csv` | 2,016 | `afe07fc70c1727bde7816143e48715e0775d4ec87b8f0577daad2504ec3b4905` |
| `stage_S2_runs.csv` | 80 | `9790f56edcce33fdbb62a10ca52a7218181cc7d0e29565f95726f1f2799d11b3` |
| `stage_S2_summary_manifest.json` | accepted | `2287dc207d58ab47c3b063fa305422aa0e640b86daeda13ff27d3d2598809620` |
| `stage_S2_normalization_audit.json` | 16 blocks | `8ff1af89692a1140e8c5c60cbae797a7ba0b61b8f4a09b1bc3c8eb7c72c8cd4b` |
| `stage_S2_normalization_audit.csv` | 16 blocks | `0a60284562f3c170b42b7b8b7a282bcf902a861d0dbc860469c3a60c836fa482` |
| `stage_S2_run_report.md` | accepted structural report | `18317ed615684aa3fc978b8eeed5e4fe97420b19966d9af9320380521d28a126` |

The structural run report accurately states 16 expected fresh runs, 64 valid reused runs, 2,016/2,016 raw and paired rows, zero invalid pair rows, all seven checkpoint labels, and zero artifact-validation errors.

### 11.4 Raw table contract

The 2,016 raw rows have 2,016 unique keys over:

`dataset, setting, seed, physical_domain, task, checkpoint, method`.

Method coverage is exact:

- `equal`: 672 rows;
- `sample`: 672 rows;
- `single_domain_mtl_reference`: 672 rows.

Each method has 96 rows at each of the seven checkpoint labels. Across methods, every checkpoint has exactly 288 rows:

`selected`, `final`, `common_epoch_9`, `common_epoch_19`, `common_epoch_29`, `common_epoch_39`, `common_epoch_49`.

All rows have `status=OK`, a nonempty 64-character checkpoint hash, finite required numeric values, and valid method/provenance identity.

### 11.5 Paired table contract

The 2,016 paired rows have 2,016 unique keys over:

`comparison, dataset, setting, seed, physical_domain, task, checkpoint`.

Comparison coverage is exact:

- `equal_vs_single_domain`: 672 rows;
- `sample_vs_single_domain`: 672 rows;
- `equal_vs_sample`: 672 rows.

Each comparison has 96 rows per checkpoint, and every checkpoint has 288 rows across comparisons. All 2,016 rows have `pairing_valid=1`, empty pairing reasons, finite required numeric fields, and nonempty 64-character left/right checkpoint hashes.

The reviewer independently joined every paired row back to the raw table and recomputed the left/right metrics, AUC and LogLoss deltas, formal keys, and checkpoint hashes. All 2,016 rows agreed exactly within numeric tolerance; error count was zero.

### 11.6 Checkpoint hashes and metric recomputation

All 80 prediction artifacts were opened read-only. For every one of the 2,016 method/domain/task/checkpoint cells, the reviewer recomputed tie-aware AUC and fixed-epsilon LogLoss from the frozen labels and predictions and compared them with the canonical raw CSV.

Results:

- artifacts checked: 80;
- cell metrics recomputed: 2,016;
- AUC/LogLoss mismatches: 0;
- raw checkpoint-hash mapping mismatches: 0;
- formal-run-key mismatches: 0;
- final versus `common_epoch_49` prediction mismatches: 0/80.

The raw and paired selected/final/common checkpoint hashes are complete and resolve to the corresponding manifest checkpoint artifacts.

### 11.7 Normalization audit acceptance

The formal normalization audit satisfies all approved constraints:

- schema: `stage_S2_postrun_normalization_audit_v1`;
- adapter hash: `a7ed02735ebb845190e6269d4bf3078e160422d9f265a698373d62801bfc7406`;
- block count and unique Equal keys: 16/16;
- allowed normalizations: 16;
- mappings: 12 MovieLens and 4 KuaiRand;
- normalization status failures: 0;
- strict comparison reasons: 0;
- non-dataset mismatch reasons: 0;
- non-dataset bypasses: 0;
- raw snapshot pre/post counts: 80/80;
- raw snapshot unchanged: true.

The CSV and JSON audits both contain 16 block rows/records. The summary manifest links the audit path, schema, adapter hash, normalization count, zero bypass count, and unchanged snapshot flag.

### 11.8 Aggregate and training-stage table acceptance

All required aggregate keys are unique and their required fields are present and nonempty:

| Table | Rows | Structural contract |
|---|---:|---|
| Four-seed aggregates | 504 | four seeds and four valid pairs per group |
| Role summaries | 504 | four seeds per setting/role group |
| Fixed-physical rotation summaries | 252 | MovieLens 3 settings/12 seed-setting rows; KuaiRand 1 setting/4 rows |
| Paired common-epoch means | 360 | four seeds at epochs 9, 19, 29, 39, 49 |
| Paired training-stage summaries | 72 | five common epochs and 20 valid seed-epoch pairs per group |
| Reference difficulty summaries | 168 | four seeds per group |
| Method-stage summaries | 336 | six domain-task cells per method/seed/checkpoint group |

The reviewer independently recomputed every four-seed aggregate mean/SD/count, every fixed-physical rotation mean/count, every common-epoch four-seed mean, and every 3-of-5 stage-support count. All values matched the emitted tables; error count was zero. This was a mechanical integrity check only, not scientific interpretation.

### 11.9 Documentation retention note

The canonical structural report `experiments/stage_S2_run_report.md` exists and passes the checks above. The earlier interim execution narrative previously observed at `experiments/reports/stage_S2_run_report.md` is no longer present. Its execution facts remain independently preserved in the immutable queue, the implementation report, and Sections 9–11 of this review. This is a documentation-retention deviation, but it does not alter raw or derived data and is not a blocker for analysis.

### 11.10 Final authorization

**PASS: the Main Agent may enter the ANALYZE stage.**

Analysis must use the accepted canonical structural outputs and follow the preregistered Stage S2 analysis requirements. This authorization does not permit further training, artifact mutation, post-hoc threshold changes, or silent exclusion of any accepted row. Scientific conclusions and evidence-strength labels remain the Main Agent's responsibility.
