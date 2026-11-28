# Design specification | C0 offline audit

**Ready for implementation.** C0 uses the fixed-EMA snapshots retained by the existing A1.B0/P0 trajectories at audited main `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`. It requires zero new training and zero Test access. S0 found the old inputs unavailable. At this design handoff the current artifact directory has the S0 receipts and no A1 roster or completion receipts. Execution requires independent review and fresh verified readback of the A1 inputs. Both original workspaces and the study worktree had no applicable stop markers at design time. [S0 design](DESIGN_SPECIFICATION.md) · [A design](A_EXECUTION_DESIGN.md) · [Shared runner handoff](B_CODER_IMPLEMENTATION_REPORT.md)

## 1. Research question

Do Train relations correctly ordered at fixed EMA E5 or E10 reverse by the same trajectory’s Validation-selected fixed EMA? Do historical confidence and identity distinguish these relations from current Train-global macro6 supervision, and where do the reversals coincide with changes in six-cell Validation AUC?

## 2. Hypothesis

Historically correct relations show material, reproducible reversals before selection. E10 provides the decision-relevant teacher because prospective memory begins at E11. Historical correctness filtering or the teacher margin may distinguish the signal from current pair supervision. The audit tests these observable premises. The prospective C3−C2 and C3−C1 comparisons test the effect of training with memory. [Algorithm §3](../../research_plans/26-11-28_three_directions/02_ALGORITHM_DESIGN.md#3-c历史正确排序关系记忆) · [Matrix §§7–8](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#7-c0历史关系离线审计)

## 3. Variables changed

Compare E5, E10 and each trajectory’s original selected fixed EMA on identical candidate pairs. Compare A1.B0 (clean BCE with an exposure-matched no-gradient pair forward) and A1.P0 (clean BCE plus current pairwise loss at λ=.05) within dataset and seed. Evaluate each historical snapshot against the selected epoch of its own arm. Record unequal selected horizons in the arm comparison.

## 4. Variables held constant

Use MovieLens-1M and KuaiRand-Pure, seeds 2022–2026, exactly 20 A1 trajectories. Keep the registered Train split, Train-fitted features, Sample/natural M3oE fixed1 and FP32 fixed-EMA readout. Fix the active condition list, cell weights, pair candidate identities and order by dataset/seed across both arms and all three checkpoints. Use the same Train endpoints and clean inference path at all checkpoints. Use the original Validation six-cell Macro AUC selection, with the earliest exact tie and same-checkpoint LogLoss. [Master plan](../../research_plans/26-11-28_three_directions/00_MASTER_PLAN.md) · [Decision rules](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md)

## 5. Mathematical and algorithmic definition

### Candidate set and inference

Instantiate A’s `MacroPairSampler` with canonical Train domains, task labels, stable row IDs and the Train row fingerprint. For each active condition `i=(d,t,a,b)`, use its existing ordered positive and negative joint-label pools. A pair key consists of `i` and the ordered positive/negative Train IDs. Keep task and condition in the key when the same endpoints serve multiple tasks.

Let `U_i = len(pos_pool) * len(neg_pool)` and `n_i = min(U_i, 4096)`. In the existing v3 `boundary_sampler.rng` use `context={dataset, seed, epoch:0, global_step:0}` and stream `c0_macro6_candidates|{d}|{t}|{a}|{b}`. Draw `n_i` unique integer ranks uniformly without replacement from `[0, U_i)` with `Generator.choice`. Sort ranks for canonical output order. Quotient and remainder by the negative pool length give the two pool offsets. Store one candidate hash per condition using A’s `ordered_ids_hash` on the positive ID block followed by the negative ID block. Bind the hash to condition, dataset, seed and Train fingerprint. Exact enumeration follows when `n_i=U_i`. The 4,096 limit is the frozen *candidate* limit.

For each distinct verified state within a job, run clean `eval()` and `torch.inference_mode()` on the sorted unique union of candidate Train offsets in batches of up to 40,960. Gather from `train_x` and use the pinned v3 model factory and `boundary_sampler.forward_logits`. Rejoin task logits to candidate order. Compute `m_e = float64(z_e^+) − float64(z_e^−)` from the FP32 routed logits. Keep original margins for all audit metrics. Use the plan’s clip `[-5,5]` and `T=1` only for the E10 teacher-confidence summaries.

### Correctness, timing and aggregation

For each condition and each snapshot, report candidate counts with margin positive, zero and negative, strict correct rate, and tie-aware pair concordance `P(m>0)+0.5 P(m=0)`. For each eligible historical epoch `h < e*`, use `H_h={m_h>0}`. A strict reversal is `m_selected<0` among `H_h`. A tie loss is `m_selected=0` among `H_h`. Report both rates and their sum as loss of strict correctness. Show raw margins and selected-minus-historical margin changes for strict reversals, tie losses and retained pairs, with counts, fixed quantiles (0, .25, .5, .75, 1) and mean. Track `m_h≥.001` to `m_selected≤−.001` as a numerical sensitivity count. The `.001` in this diagnostic is in logit units and has no advancement role.

Use A’s `wi` and `qi`. Per-cell target rates normalize the `wi` within that cell. Macro rates use `wi` over the six equally weighted cells. For any conditional rate such as the flip rate, first aggregate its joint numerator and its conditioning denominator with those weights, then divide. Provide raw condition counts alongside the conditional rates. Encode undefined rates as null with their denominators. Do not pool candidate pairs or cells as independent replicates. Summaries use the trajectory seed as unit, report eligible seed counts and actual horizon lengths, and pair B0/P0 on their common eligible seeds. Report per-seed results, mean and sample SD, with a two-sided paired t interval for arm differences when at least two matched seeds are eligible.

Apply the following classification independently at E5 and E10:

| Actual relationship | Classification | Contribution to pre-selection flip evidence |
|---|---|---|
| `e* > h` | `PRE_SELECTED` | Measure reversal, tie loss, and overlap through `e*`. |
| `e* = h` | `SELECTED_AT_SNAPSHOT` | Require identical model hashes and margins. Mark the comparison ineligible, with a zero-length horizon. |
| `e* < h` | `POST_SELECTED_SNAPSHOT` | Report checkpoint and selected candidate statistics. Keep the pre-selection rate and overlap null. |

E10-to-selected is primary. E5-to-selected is corroboration and characterizes early formation. If E5 precedes selection and E10 coincides with or follows it, include the E5 findings without counting that seed as evidence for the E10 teacher. Record E5→E10 as descriptive formation only if E10 is at or before selected, with its own overlap window, and keep post-selected snapshots descriptive. E50 is outside the audit.

### Current sampler overlap and Validation alignment

Replay A’s `sample_macro6` once per dataset/seed over the exact logical step schedule through the later selected epoch of its two arms. Use completed-trajectory batch sizes and `S=ceil(n_train/40960)` with `epoch=1+(global_step−1)//S`, `batch=1+(global_step−1)%S`. Map each eligible teacher to its own window `(h*S, e* * S]`. Replay uses the original `macro6_condition` and `macro6_pair_indices` streams. Differences in recorded attempts or device do not change the stream. Verify replayed condition and ordered pair hash against committed trace entries at a minimum on the first and last logical steps of each eligible window for both arms. Resolve retries by committed epoch and logical step using the epoch receipt’s writer attempt. Require matching hashes when an earlier interrupted attempt contains the same logical step.

Per condition and window, report total current draws `D_i`, the number of distinct current pair keys, and exact repeated exposure counts. For all candidates, historical correct candidates, strict reversals and tie losses, report subset size `M`, distinct members observed, and observed draw hits including repeats. Use `observed_members/M` for coverage and `draw_hits/D_i` for share of current draws. Report `draw_hits/M` as actual exposures per candidate in that subset. Give the uniform-support reference `D_i*M/U_i` for expected hits and `1−(1−1/U_i)^D_i` for expected coverage conditional on the realized current condition schedule. Handle empty denominators explicitly. Within any active condition the current sampler supports every legal candidate, even when the realized coverage is sparse. Interpret overlap against this support and exposure reference.

For E10-correct candidates, report clipped historical margin bins `(0,.1]`, `(.1,1]`, `(1,2]` and `(2,5]`, and their counts, selected margin statistics and strict reversal rates within condition for eligible seeds. This displays whether teacher confidence carries an observable ordering signal within the single hard correctness label. Match this to six-cell and Macro Validation AUC and same-checkpoint LogLoss for E5, E10 and selected, from already-recorded A1 metrics. Provide per-seed/cell joins with baseline AUC, selected AUC, change, Train population and exposure. Examine cross-seed/cell alignment descriptively, accounting for horizon, current-pair arm and baseline AUC. Validation AUC is the selection endpoint and enters no candidate, overlap or margin computation.

## 6. Affected modules and files

| File | Handoff |
|---|---|
| `experiments/three_directions_20261128/c0_audit.py` | New foreground inspection entry, eligible-input checks, candidate/inference and replay logic, aggregate/raw outputs. |
| `experiments/three_directions_20261128/tests/test_c0_audit.py` | New focused implementation checks. |
| `research_reports/three_directions_20261128/C0_CODER_IMPLEMENTATION_REPORT.md` | Coder’s implementation and check receipt for independent review. |
| `artifacts/three_directions_20261128/C0/` | New scoped numeric outputs, manifests and runtime status. |

Read A1 runtime modules without editing them. The frozen A1 roster hashes `runner.py`, `protocol.py`, `ram_state.py`, `scheduler.py`, `a_step.py` and `macro_pair_sampler.py` among its inputs. Reuse `MacroPairSampler` and its read-only `_pools` in the isolated inspection module, `runner.freeze_roster`/`job_journal`, `scheduler.stop_status`, registered cache hash helpers, and v3 `setup`, `load_protocol`, `make_model`, `state_digest` and `boundary_sampler.forward_logits`. Record the C0 source hash separately. [Shared source and data contract](B_EXECUTION_DESIGN.md)

## 7. Implementation requirements and inspection entry

```bash
cd /home/xzf/Code/phase0-3_three_directions_20261128/m3oe/examples
export PYTHONPATH=/home/xzf/Code/phase0-3_three_directions_20261128
export PYTHONDONTWRITEBYTECODE=1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  -m experiments.three_directions_20261128.c0_audit \
  --artifacts /home/xzf/Code/phase0-3_three_directions_20261128/artifacts/three_directions_20261128
```

Use a single foreground CPU process in the project environment. Set and record the FP32 v3 backend and seed configuration. Check `scheduler.stop_status` before inspection and between complete job units. On a stop, publish the scoped status and preserve verified source artifacts and completed audit units. There are no training calls or Test entry points in this module.

1. Verify the existing A1 roster with `freeze_roster(artifacts, "A1", create=False)`. Select only A1.B0 and A1.P0 across the two five-seed grids. For each, require `COMPLETE_VERIFIED.json` under `A1/runs/<job.key>/` with matching job record, full `job_journal` binding, epoch 50 and zero Test access. Require the journal’s recorded completion to agree with this result and its `compact_models` expectations. Read completion metadata directly so inspection has no journal recovery or cleanup side effects. Use `Journal.read_model(key, expected=...)` for `fixed_ema-E05`, `fixed_ema-E10` and `fixed_ema-selected`. Verify each against the completion’s expected `kind`, `readout`, `epoch`, `model_hash` and the logical global step. Preserve writer, generation, byte SHA and decoded-state digest from the returned receipt. Fresh verified readback is the eligibility evidence. Verify and release one model packet at a time and reuse logits within a job when two verified keys have the same model hash.
2. Read only the audited Train cache `features.json`, `train_x.npy`, `train_y.npy`, `train_row_ids.npy` plus pinned identity metadata. Use immutable mmap with pickle disabled. Reuse the canonical registered `FeatureSchema`, `array_hash`, `row_ids_hash` and `split_row_fingerprint` helpers. Check Train shapes/dtypes, binary labels, unique IDs, encoder/scaler hashes, Train array hashes, row counts, ordered IDs and content fingerprint, Train domain/cell support and shared A1 `data_hash` binding against the audited and pinned metadata. The full Train/Validation `data_hash` comes from the verified A1 identity. This command rechecks its Train projection and has no need to open Validation arrays, raw CSV or Test files. Build the v3 model with the verified schema and pinned dataset `source` config. Verify parameter scope against the roster, load state strictly and check FP32 parameters and finite logits.
3. From the 50 committed `epoch_XX.json` records, verify the fixed-EMA selected epoch is the earliest strict maximum of finite six-cell Macro AUC. Join the three original fixed-EMA validation records to the snapshot receipts via job, epoch and logical step. Require the selected hash and metrics to agree with the verified completion endpoint and final selection record. Use E10 endpoint hash as an extra cross-check where published. Save the Validation values as recorded.
4. Keep A1 artifacts and named RAM states read-only. Record missing roster or incomplete jobs as input pending. Report missing compact weights after expected completion, corrupt packets and source/data mismatch as technical input failures, listing affected jobs. Keep available observations as partial receipts and gate scientific interpretation on complete input accounting. Technical failures do not supply zero flip rates. Scope writes to `C0/` and use atomic per-unit publication. Persist audit values and metadata only.

## 8. Required diagnostic comparisons

- E10→selected within each eligible trajectory, with E5→selected as timing corroboration and matched E5→E10 formation diagnostics.
- B0 vs P0 by dataset and seed on identical candidate IDs, retaining both arms’ selected horizons and matching the same history epoch. Each dataset gets its own summary.
- All candidate versus historically correct relations, reversals versus retained relations, and confidence-bin contrasts within historically correct relations. For each, include current-sampler overlap with realized and uniform-support exposure summaries. These are offline comparisons and consume no new trajectories.

## 9. Logging and outputs

- `MANIFEST.json` and `STATUS.json` record roster/source/config/data/scope and C0 file hashes, input inventory, cache and RAM verification receipts, selected epochs, eligibility, settings, cost environment, stop checks, completed jobs and `test_accesses=0`.
- `candidates/<dataset>_<seed>.npz` stores condition IDs and stable positive/negative Train row IDs in shared order with SHA receipt. `margins/<job.key>.npz` stores aligned E5/E10/selected margins and corresponding hashes. Omit raw user/item features and model tensors.
- `PAIR_METRICS.json`, `OVERLAP.json`, `VALIDATION.json` and `SUMMARY.json` hold checkpoint, condition/cell/Macro and aligned dataset/seed/arm rows. Include historical eligible counts per condition and cell, clip saturation, exact tie and near-zero counts, current draws, coverage and exposures, selected AUC and LogLoss, null reasons and paired arm summaries. Place E10 and E5 results next to their eligible seed counts and horizon lengths.
- `COST.json` records elapsed input verification, RAM bytes/read/decode time, separate inference/replay/write time, FP32 forwards and Train rows per model, unique endpoints, candidate counts, replayed steps and draw counts, process peak RSS with units and bytes of numeric outputs. Give wall seconds per scored candidate and eligible relation when defined. Report per-condition and per-cell eligible E10 counts and the observed current exposures per relation. For a later assigned size `b_i>0`, the prospective uniform-with-replacement memory exposure is `2048 × events_i/b_i` draws per entry over the same logical schedule. Log `events_i` and supported eligible counts so cost can be instantiated after the bank ruling. Avoid presenting that conditional exposure as a measured C1 cost.

## 10. Reviewer acceptance criteria

1. The reviewed command reads the existing roster and only the registered Train arrays, observes current stop controls, and writes only scoped audit artifacts. A fixture must show that missing inputs are reported separately from zero flips and a wrong snapshot or Train identity is rejected. Verify `Journal.read_model` is called with the frozen key, epoch and hash and the original A1 identities remain intact.
2. A small enumerated pool fixture proves exact unique candidate sampling, matching hashes across the two arms/checkpoints, label polarity, condition ownership, and correct cell mass. A fixture with unequal condition weights must reproduce weighted conditional rates from independent numerator/denominator calculations. Verify current replay identity against the A sampler, repeated-draw counting, coverage and null cases using known counts. Require the runtime trace boundary checks specified in §5.
3. On a small fixed-EMA-shaped fixture, route both tasks and at least two domains. Require probability/logit sigmoid parity in FP32 (`atol=1e-6`, `rtol=1e-5`), compare task margins with direct independent indexing/subtraction, and check positive, tie and negative transition counts. Cover selected before, equal to and after E5/E10, including an earlier exact Validation tie and a hash mismatch at the same epoch. All metrics use actual strict signs; the margin-band sensitivity is diagnostic.
4. Verify complete coverage of 20 trajectories or explicit input dispositions, matching paired fixtures, source/model hash receipts and only capacity-independent costs. The reviewer gates scientific RUN. Execute focused checks through the project Python with `-m pytest -q experiments/three_directions_20261128/tests/test_c0_audit.py`.

## 11. Falsification and C1 decision

Apply plan 03 §7 and 06 §6 by dataset. An adequately observed target window with absent or unstable strict reversals, or high current-sampler redundancy accompanied by no observable confidence signal, defeats the corresponding C0 premise. Require repetition in at least two eligible seeds within a dataset to carry a dataset-specific claim, and report any replication across both datasets. Show seed-level magnitude, margin strength, condition coverage, horizon and both arms. The frozen plan supplies no numerical cutoff for materiality or high redundancy. Interpret clear outcomes using these observations and report borderline outcomes explicitly. Use Validation alignment to assess relevance and the E5 diagnostics to characterize timing. Early selection can remove the E10 teacher’s pre-selection window, and missing verified inputs constitute technical unavailability. Neither case is a measured zero flip rate.

Exact identity overlap and teacher-margin summaries assess the plausibility of additional information. The exposure-matched C3−C2 and C3−C1 comparisons would test the training benefit, with C4 conditional under the frozen rules. If capacity-independent C0 evidence otherwise supports proceeding, report C1 as awaiting a governing capacity decision. Plan 02 §3.3 and 04 §5 permit up to 2,048 unique bank pairs per active condition, while plan 06 §5 specifies 512 per cell. A ruling must fix the unique-pair cap and its unit, plus the within-cell condition allocation if applicable, before capacity-specific feasibility or C1 dispatch. The common 2,048 draws per event are an exposure dose. Bank capacity, bank identity and the C1 control-reuse hash check remain later decisions. [Frozen conflict decision](DESIGN_SPECIFICATION.md#decisions-on-conflicting-clauses) · [Implementation plan §5](../../research_plans/26-11-28_three_directions/04_CODEX_IMPLEMENTATION_SPEC.md#5-c-memory-implementation) · [Decision rules](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md)
