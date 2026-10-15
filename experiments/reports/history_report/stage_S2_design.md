# Stage S2 DESIGN SPECIFICATION

## Scope, evidence basis, and hard boundary

This specification covers **Stage S2: training-objective and negative-transfer reproduction only**. It does not design or authorize any later mechanism-diagnosis stage. It introduces no new model, recommendation algorithm, optimizer, sampler, curriculum, Momentum/EMA/gamma variant, hyperparameter search, or mitigation method. Formal training remains blocked until implementation review is complete **and the user explicitly approves entry into `luna_runner`**.

The design was produced from a read-only audit of:

- `promt/negative_analyse.md`;
- `experiments/reports/stage_S0_{design,implementation,review,run_report,analysis}.md` and the S0 aggregate tables;
- `experiments/reports/stage_S1_{design,implementation,review,run_report,analysis}.md`;
- the S1 protocol, data, model, runner, orchestrator, metric and report modules;
- `experiments/stage_S1_*.csv`, `stage_S1_summary_manifest.json`, and the target M3OE raw run directories;
- `m3oe/MDMTRec/models/multi_domain/model.py`, the original trainer, `m3oe/README.md`, and `PROJECT_ARCHITECTURE.md`.

No standalone file named an “experiment charter” or `stage_S1_data_audit.md` exists. For this audit, `stage_S1_design.md` plus `stage_S1_protocol.py` are the available charter, while per-run `run_manifest.json`, `initialization_audit.json`, `parameter_manifest.json`, split fingerprints, artifact hashes, and the Stage S1 validator are the available data-audit trail. The repository root is not a Git checkout, so no commit SHA is available; provenance must use exact SHA-256 inventories. These absences must be recorded in the S2 reuse audit rather than filled by assumption.

---

## 1. Research question

### RQ1 — reproducibility of negative transfer under Equal

Under the exact S1 M3OE data, initialization, exposure, optimizer, checkpoint, and evaluation protocol, does an equal-domain joint objective produce practically meaningful and seed-stable degradation relative to the matching single-domain two-task reference?

### RQ2 — whether Sample changes the Equal outcome

Does the already validated sample-proportional joint objective improve the same physical-domain/task cells relative to Equal, and does it convert any Equal-versus-reference degradation into non-degradation? An Equal deficit relative to Sample is only an objective contrast; it is negative transfer only if Equal is also below the single-domain reference.

### RQ3 — size role versus physical identity/difficulty

Across the three MovieLens rotations, does the Equal-minus-Sample effect change systematically when the same physical domain occupies L, M, and S roles, after retaining physical domain, task, and independent-reference performance as separate fields? KuaiRand natural sizes provide a second dataset condition but are not a synthetic rotation.

### RQ4 — eligibility of a later diagnosis object

Does S2 identify a stable, practical, precisely localized negative-transfer condition that survives seeds and training-stage checks, or should the negative-transfer mechanism line stop?

### Stage-level acceptance of the scientific test

S2 is scientifically complete only if all 16 fresh Equal runs and all 64 reused S1 M3OE runs pass provenance/coverage validation; every domain-task-seed cell is present for Equal, Sample, and the matching single-domain reference; and all three preregistered pairwise contrasts are available at selected, fixed-final, and common epochs. Completion does not imply that the affirmative hypothesis is supported.

---

## 2. Hypotheses being tested

The primary affirmative hypothesis is:

> H-EQ-NT: Equal produces at least one domain-task condition whose four-seed mean `Equal − single-domain` AUC is at most `-0.003`, at least 3/4 seed deltas are negative, mean LogLoss is worse, and the negative direction is supported at at least 3/5 common epochs; Sample materially reduces that same deficit.

Competing hypotheses that must remain live are:

- **H0/no practical Equal NT:** Equal effects are positive, below practical magnitude, seed-unstable, or checkpoint-dependent.
- **H-objective-ranking only:** Sample outperforms Equal, but both remain above single-domain reference; this is not negative transfer.
- **H-shared-NT:** Equal and Sample both show stable practical degradation, so objective weighting alone is insufficient.
- **H-identity/task:** effects follow a physical domain or task rather than size role.
- **H-size:** within the same physical domain/task, the Equal-minus-Sample effect changes systematically with L/M/S role.
- **H-difficulty:** effects track validation-only single-domain baseline performance rather than size.
- **H-dataset/stage/noise:** effects occur in only one dataset, one selected checkpoint, or isolated seeds.

The design is intended to reject H-EQ-NT if the required magnitude, stability, metric concordance, and stage support do not coexist.

---

## 3. Variables changed

Only one experimental factor changes within a paired joint-training block:

| factor | values | status |
|---|---|---|
| training method/objective | `Equal Joint`, `Sample Joint`, `single_domain_mtl_reference` | Equal is fresh; Sample and reference are reused from S1 after audit |
| dataset/setting | MovieLens `d0_large`, `d1_large`, `d2_large`; KuaiRand `natural` | crossed fixed conditions |
| seed | `2022`, `2023`, `2024`, `2025` | repeat unit |
| checkpoint view | validation-selected, fixed final epoch 49, common epochs `9,19,29,39,49` | prespecified sensitivity, not a tunable factor |

The formal backbone is **M3OE only** in S2. MMoE/PLE are not added. No parameter sweep or secondary round is authorized by this design.

### Strict paired matrix

For each of 4 settings × 4 seeds, the block contains:

| method | scope | physical domains represented | number of runs per block | source |
|---|---|---:|---:|---|
| Equal | one `D=3,T=2` joint run | d0,d1,d2 | 1 | fresh S2 |
| Sample | one `D=3,T=2` joint run | d0,d1,d2 | 1 | frozen S1 |
| single-domain reference | three `D=1,T=2` runs | one physical domain each | 3 | frozen S1 |

Totals: **16 fresh Equal runs**, **16 reused Sample runs**, **48 reused single-domain runs**, and **80 run artifacts in the S2 evidence inventory**. At each checkpoint label this yields 96 cell rows per method and 288 raw method-cell rows total. With 7 checkpoint labels, the expected long-form raw table has 2,016 rows. The three pair types also yield 2,016 long-form paired rows (`3 contrasts × 96 seed-cells × 7 checkpoints`). `final` and `common_epoch_49` are separate protocol labels but must reference numerically identical model state/metrics.

---

## 4. Variables held constant

Within every `dataset × setting × seed` block, Equal and Sample must be identical in all of the following, verified by artifact values rather than configuration names:

- raw data file bytes, global row order, train/validation/test split IDs, preprocessing artifacts, OOV rule, task definitions, and physical-domain mapping;
- MovieLens deterministic subsample rule `RandomState(10*seed + physical_domain)` and role sizes L/M/S = `150000/34500/3000`;
- KuaiRand global-row-order 80/10/10 split-first/filter-second construction and train counts `172782/1743898/89452` for local d0/tab0, d1/tab1, d2/tab4;
- M3OE architecture (`D=3,T=2`, embedding 16, `fcn_dims=[512,256,256,64]`, 4 experts, `exp_d=exp_t=bal_d=bal_t=1`), parameter partition, and forward logic;
- semantic initial parameter state for the joint model;
- train batch order and architecture-update batch order;
- 50 epochs, batch size 40,960, `shuffle=True`, `drop_last=False`, `num_workers=0`, and identical per-domain example exposure;
- Adam main optimizer (`lr=0.01`, betas `0.9/0.999`, weight decay `1e-5`) and M3OE architecture optimizer (`lr=0.001`, same betas/weight decay), no scheduler;
- exactly one architecture-optimizer update after each epoch using the existing RNG-preserving S1 rule;
- validation-only checkpoint selection by unweighted joint six-cell macro-AUC, earliest-epoch tie break; fixed-final and common checkpoints; deferred test evaluation;
- the shared S1 tie-aware AUC and fixed-epsilon (`1e-12`) LogLoss implementation;
- device target `cuda:0`, deterministic seed setup, label indexing `cell=t*D+d`, and all instrumentation other than objective-specific weight fields.

The reused single-domain references retain S1's `D=1,T=2` architecture-matched definition. They isolate the addition of cross-domain joint training while retaining within-domain task sharing; they must not be described as per-task STL. Their own selected checkpoint is based on their two-task validation macro-AUC, exactly as in S1.

---

## 5. Mathematical and algorithmic definition

### 5.1 Actual S1 Sample objective from code

`stage_S1_run.py:511-514` constructs one shuffled DataLoader over the concatenated joint examples, without a weighted/balanced sampler. `direct_per_example_bce` at lines 114-120 applies BCE reduction `mean` over the `[B,T]` tensor, and `_train_epoch` uses it directly at lines 657-686. Thus the sampling probability for domain `d` is its empirical training share

\[
p_d=\frac{n_d}{N},
\]

the explicit per-example loss coefficient is `w_d=1`, and the objective for fixed model parameters is

\[
L_{Sample}=\frac{1}{T}\sum_{t=1}^T\sum_{d=1}^D p_d\,\bar\ell_{d,t}.
\]

The expected domain gradient coefficient is `a_d=p_d w_d=p_d`; the expected cell coefficient is `a_d/T`. No second factor of `p_d` is applied. `normalized_cell_bce` in S1 is only an equivalence oracle and confirms the same expression; it is not an additional training weight.

### 5.2 Equal objective required for S2

Sampling remains exactly the same concatenated shuffled DataLoader. Equal is implemented solely by fixed inverse-share example coefficients derived from the immutable full training counts:

\[
w_d^{Equal}=\frac{1}{D p_d}=\frac{N}{D n_d}.
\]

For a batch of size `B`, domain IDs `d_i`, two tasks, and unreduced BCE `\ell_{i,t}`, the training scalar is

\[
\widehat L_{Equal}=\frac{1}{BT}\sum_{i=1}^{B}\sum_{t=1}^{T}
w_{d_i}^{Equal}\ell_{i,t}.
\]

Therefore

\[
\mathbb E[g_{Equal}]
=\sum_d p_d w_d^{Equal}\frac{1}{T}\sum_t
\mathbb E[\nabla\bar\ell_{d,t}]
=\frac{1}{D}\sum_d\frac{1}{T}\sum_t
\mathbb E[\nabla\bar\ell_{d,t}].
\]

This is equal-domain, equal-task weighting with population-average coefficient `sum_d p_d w_d = 1`, preserving the Sample objective's global loss scale. The implementation must **not** also use equal/balanced domain sampling, must not multiply by `p_d` again, and must not renormalize weights by the domains present in each minibatch. Fixed full-dataset shares, not per-batch shares, are used. The same objective function must be used by both the main update and M3OE architecture update.

### 5.3 Single-domain reference objective

For each physical-domain reference, `D=1`, `p_0=1`, `w_0=1`:

\[
L_{SD(d)}=\frac{1}{T}\sum_t\bar\ell_{d,t}.
\]

### 5.4 Concrete coefficients and effective weights

MovieLens has `N=187500` in every rotation. Coefficients follow the role assigned to each physical domain:

| size role | n | p_d | Sample w_d | Sample a_d | Equal w_d | Equal a_d |
|---|---:|---:|---:|---:|---:|---:|
| L | 150000 | 0.800000 | 1 | 0.800000 | 0.4166666667 | 0.3333333333 |
| M | 34500 | 0.184000 | 1 | 0.184000 | 1.8115942029 | 0.3333333333 |
| S | 3000 | 0.016000 | 1 | 0.016000 | 20.8333333333 | 0.3333333333 |

KuaiRand natural has `N=2006132`:

| physical domain | n | p_d | Sample w_d | Sample a_d | Equal w_d | Equal a_d |
|---|---:|---:|---:|---:|---:|---:|
| d0/tab0 | 172782 | 0.0861269348 | 1 | 0.0861269348 | 3.8702565468 | 0.3333333333 |
| d1/tab1 | 1743898 | 0.8692837759 | 1 | 0.8692837759 | 0.3834574423 | 0.3333333333 |
| d2/tab4 | 89452 | 0.0445892892 | 1 | 0.0445892892 | 7.4756368406 | 0.3333333333 |

Because `T=2`, divide each `a_d` by 2 for an individual domain-task cell's expected coefficient. Per-batch realized weighted mass must be logged, but it does not redefine `w_d`.

### 5.5 Pairwise estimands

For each dataset `k`, setting `r`, seed `s`, physical domain `d`, task `t`, checkpoint `c`:

\[
\Delta^{AUC}_{E-SD}=AUC_E-AUC_{SD},\quad
\Delta^{AUC}_{S-SD}=AUC_S-AUC_{SD},\quad
\Delta^{AUC}_{E-S}=AUC_E-AUC_S.
\]

LogLoss gaps use the same left-minus-right convention. For LogLoss, a positive gap is worse. Negative transfer is assessed only for `E-SD` or `S-SD`, never from `E-S` alone.

Preregistered group-level candidate criteria retain S1:

- practical AUC degradation: four-seed mean delta `<= -0.003`;
- strong AUC degradation: four-seed mean delta `<= -0.005`;
- stability: at least 3/4 seed deltas are negative; separately list 4/4;
- concordance: mean LogLoss gap `>0` and at least 3/4 seed gaps are positive;
- stage support: mean matched-epoch AUC delta negative at at least 3/5 common epochs.

Thresholds are practical, not claims of statistical significance. Report mean, sample SD, range, worst seed, and worst group. Seeds are the repeat unit; cells and rotations must not be treated as independent replicates.

---

## 6. Exact affected modules/files

S1 raw artifacts and S1 source modules are frozen and read-only. The coder should add an isolated S2 layer:

- `experiments/stage_S2_protocol.py`: immutable 16-run Equal matrix, objective labels/coefficients, seeds/settings, paths, thresholds, and exact row-count contracts.
- `experiments/stage_S2_objectives.py`: the only new mathematical operation: unreduced BCE plus fixed `w_d=1/(3p_d)`; pure coefficient/equivalence helpers.
- `experiments/stage_S2_run.py`: S1-derived M3OE-only Equal runner. Reuse `stage_S1_data.py`, `stage_S1_models.py`, and `stage_S1_metrics.py` by import. Preserve S1 training/checkpoint/artifact behavior; apply Equal in main and architecture updates.
- `experiments/stage_S2_orchestrator.py`: 16 unique fresh Equal tasks, two-process scheduler, fail-fast/preservation behavior, and Equal artifact validation.
- `experiments/stage_S2_report.py`: read-only validation/import of the 64 frozen S1 M3OE artifacts, collection of 16 S2 Equal artifacts, and construction of raw/paired tables without scientific interpretation.
- `experiments/tests/test_stage_S2_objective.py`: exact objective/effective-weight/no-double-weight tests.
- `experiments/tests/test_stage_S2_pairing.py`: S1 reuse, initialization/data/batch/checkpoint pairing, matrix and report-cardinality tests.
- `experiments/tests/run_stage_S2_smoke.py`: nonformal tiny forward/loss/backward and artifact smoke only.
- `experiments/stage_S2_reuse_audit.csv` and `.json`: immutable enumeration of all 64 reused S1 keys and trust evidence.
- fresh raw outputs under `experiments/stage_S2_raw_results/equal/<dataset>/<setting>/m3oe/seed<seed>/joint/`.
- final structural outputs `experiments/stage_S2_raw_results.csv`, `experiments/stage_S2_paired_results.csv`, `experiments/stage_S2_runs.csv`, `experiments/stage_S2_summary_manifest.json`, and `experiments/reports/stage_S2_run_report.md`.

Do not edit `m3oe/MDMTRec/models/multi_domain/model.py`, any S1 raw artifact, or any S0/S1 source/report/table. If the coder mechanically derives S2 runner logic from S1, the implementation report must provide a diff inventory proving that semantic changes are restricted to stage identity/output paths, the M3OE-only matrix, objective dispatch, objective logging, and two-process orchestration.

---

## 7. Implementation requirements

### 7.1 S1 reuse audit and decision

The Designer audit found:

- `stage_S1_runs.csv` contains 192/192 total valid runs and exactly **64 M3OE target runs**: 16 joint and 48 single-domain;
- the target subset covers all 4 settings × 4 seeds, with three references per block;
- all 64 manifests report `status=OK`, `formal_valid=true`, objective `direct_per_example_task_mean_bce`, and 50 observed epochs;
- all 64 use S1 runner source hash `7ad50d6a19f4b5a84939aa5f01977c8d703817ff560c2f5d181e7541aeda3210`;
- the audited required raw files, including selected/final/common checkpoints, metrics, predictions, initialization/parameter audits, optimizer/RNG state, batch trace, and nonfinite inventory, are present in all 64 directories;
- raw dataset files remain accessible and match recorded hashes: MovieLens `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`, KuaiRand `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`;
- `stage_S1_summary_manifest.json` records 192 valid, zero missing, 288 paired seed-cell rows; the final S1 review gate is PASS.

Decision: the 16 M3OE Sample joint runs and 48 matching single-domain runs are **eligible for direct reuse**, so the formal fresh task count is 16 Equal runs. This eligibility is fail-closed: before reporting or launching Equal, S2 must rerun the frozen S1 validator on the 64 keys and write the reuse audit containing artifact hashes, dataset/split/physical-view fingerprints, code hashes, runtime fingerprint, initialization hash, batch-order hashes, optimizer/exposure counters, selected epoch, checkpoint hashes, and validation reasons. Any failed reused key adds only its exact necessary replacement run to the task list; it does not justify retraining the other valid keys. Replacement requires reviewer approval because it changes the planned 16-run matrix.

### 7.2 Exact pairing invariants

For each Equal/Sample pair in the same block, require equality of:

- dataset file hash; base split and selected row-ID fingerprints; physical-view fingerprints; feature/task/domain identities;
- joint model parameter manifest and initial parameter-state hash;
- train counts/shares; main batch fingerprints/order hash; architecture batch fingerprints/order hash;
- epoch count, batches/epoch, examples processed, main/architecture optimizer steps;
- optimizer definitions, parameter partition, model parameter count, and forward-FLOP estimate;
- checkpoint rule, metric module identity, common epochs, and deferred-test flag.

The only allowed unequal fields are stage/run key/output path/code hash, objective name/formula, `w_d`, realized weighted mass/objective values, learned state/checkpoints/metrics, wall time, and incidental resource measurements.

For each joint/reference pair, retain the existing S1 physical-view/reference-pairing fingerprints and semantic initialization mapping. Do not require equal parameter shapes or total optimizer steps between `D=3` joint and `D=1` reference; those are structural consequences already logged in S1.

### 7.3 Objective implementation and guards

1. Compute `p_d` only from immutable full training counts, in physical/local domain order; verify `sum p_d=1`.
2. Compute Equal `w_d=1/(3p_d)` in float64 metadata and apply the cast tensor on device to unreduced BCE.
3. Reduce in exactly this order: mean over the two task losses per example, multiply by that example's domain coefficient, then mean over examples. The algebraically equivalent full `[B,T]` weighted mean is acceptable.
4. No `WeightedRandomSampler`, domain cycling, oversampling, undersampling, per-batch domain-normalization, or extra domain coefficient is permitted.
5. Main and architecture update paths must call the same objective function and log the same coefficient table hash.
6. Fail before training on zero/missing domain counts, nonfinite weights/loss/gradients, unexpected domain IDs, or coefficient mismatch.
7. A fixed-prediction oracle must show: Sample direct mean equals `sum_d p_d mean_t(cell BCE)`; Equal weighted mean equals `(1/3)sum_d mean_t(cell BCE)`; `p_d*w_d=1/3`; population mean weight is one. A deliberately doubled `p_d` fixture must fail.

### 7.4 Checkpoint and training-stage contract

- Train every Equal run for epochs 0–49; shadow patience remains diagnostic and cannot stop training.
- Select Equal checkpoint using validation six-cell unweighted macro-AUC with earliest tie. Freeze selection before any test evaluation.
- Save selected state, final epoch 49, and common epochs 9/19/29/39/49. Require `final` and `common_epoch_49` state hashes and metrics to agree.
- Test is evaluated only after selection is frozen. No test value may affect retries, checkpoint, preprocessing, objective, or inclusion.
- Selected-to-selected is the primary S1-compatible comparison. Common-epoch and final comparisons are mandatory stage checks; they cannot be used to replace an unfavorable primary result.

### 7.5 Two-process formal execution plan

After reviewer PASS/PASS WITH FIXES, blocking fixes resolved, and explicit user approval:

- run exactly two worker processes, maximum concurrency 2, both consuming an immutable 16-task queue;
- pair seeds within setting, e.g. `(2022,2023)` then `(2024,2025)`, so every task is run under the same concurrency policy; worker assignment is logged and is not a scientific factor;
- each formal key has a unique output directory, stdout/stderr log, checkpoints, lock, and temporary files; no shared writable result file exists during training;
- the parent alone atomically updates the queue manifest after child exit and artifact validation;
- log co-run key, worker ID, PID, start/end timestamps, GPU identity, peak memory, exit code, and validator result;
- perform a reviewed nonformal concurrent memory smoke before formal launch. If two Equal runs cannot coexist safely, stop before formal training. Changing concurrency to one is a protocol deviation requiring renewed reviewer and user approval; it must not happen silently.

### 7.6 Failure and recovery boundary

- No mid-run checkpoint resume. A failed or invalid Equal attempt is archived append-only and rerun from epoch 0 with the same formal key/config.
- On the first child or validator failure, stop scheduling new tasks; allow an already-running sibling to finish and validate rather than kill it. Preserve both logs.
- Valid completed siblings are retained and never overwritten. A failed task cannot write into another task's directory.
- OOM or repeated concurrent-resource failure is not treated as seed evidence. Pause and return to reviewer/user for an execution-plan decision.
- Dataset/hash/protocol/pairing mismatch is blocking and cannot be bypassed by editing manifests or hashes.

### 7.7 Required raw-data schema

`stage_S2_raw_results.csv` must be long-form with one row per `method × dataset × setting × seed × physical_domain × task × checkpoint`. Required fields:

`stage, method, source_stage, provenance_type, formal_run_key, source_formal_run_key, dataset, setting, backbone, seed, physical_domain, physical_domain_identity, size_role, task, task_identity, checkpoint, evaluated_epoch, selected_epoch, auc, logloss, validation_auc_at_stage, validation_logloss_at_stage, validation_macro_auc_at_stage, train_sample_count, validation_sample_count, test_sample_count, domain_share_p, loss_coefficient_w, effective_domain_coefficient_a, effective_cell_coefficient, objective_name, objective_formula, initial_parameter_state_hash, batch_order_hash, architecture_batch_order_hash, split_row_id_hash, physical_view_fingerprint, dataset_file_hash, checkpoint_hash, code_inventory_hash, runtime_fingerprint, status`.

`stage_S2_paired_results.csv` must contain:

`comparison, lhs_method, rhs_method`, all cell identity/checkpoint keys, `lhs/rhs formal_run_key`, `lhs/rhs auc`, `delta_auc`, `lhs/rhs logloss`, `delta_logloss`, practical/strong/raw/concordant flags, both selected epochs, both checkpoint hashes, and a `pairing_valid` field. Positive `delta_logloss` means the left method is worse.

Per-run artifacts must retain S1 files and add objective evidence:

- `objective_coefficients.json` with counts, `p_d`, `w_d`, `p_dw_d`, cell coefficient, formula, and hash;
- batch trace columns for raw domain counts, raw domain fractions, weighted mass by domain, total weighted mass, and objective;
- config/run manifest with source-module hashes, S1-derived code provenance, worker/co-run fields, and test-leakage flags.

The structural report must also emit four-seed aggregates with mean, sample SD, range, negative counts, 4/4 indicator, worst seed, worst group, AUC/LogLoss sign agreement, per-role summaries, fixed-physical-domain rotation summaries, validation-reference difficulty summaries, macro AUC/LogLoss, worst-domain AUC, and worst-cell AUC. It must not contain scientific interpretation.

---

## 8. Ablations and comparisons required

No optional algorithmic ablation is authorized. The complete required discriminative comparison set is:

1. Equal versus single-domain reference, per cell and aggregate.
2. Sample versus the same single-domain reference, reconstructed from frozen S1 artifacts.
3. Equal versus Sample, per cell and aggregate.
4. Each comparison at validation-selected, fixed-final, and all five common epochs.
5. MovieLens fixed physical-domain/task contrasts across L/M/S roles.
6. Role summaries that keep physical identity and validation-only reference difficulty visible.
7. KuaiRand natural replication/contradiction check, without calling its natural size order a rotation.
8. AUC/LogLoss concordance and all counterexamples, including Equal worse than Sample but still above reference.

No Momentum, gamma, EMA, additional backbone, balanced sampler, alternate Equal normalization, extra seed, task-specific weighting, or threshold adjustment may be added after observing results.

---

## 9. Expected logging requirements

Every fresh Equal run must log:

- exact config and complete code SHA-256 inventory;
- dataset file hash, split/row/preprocessor fingerprints, task/domain identities, and per-domain counts/shares;
- objective coefficient table and hash, including `p`, `w`, `p*w`, and the no-double-weight invariant;
- semantic initialization audit, parameter manifest/counts, initial/final parameter hashes;
- every main and architecture batch fingerprint, domain count, weighted mass, objective, examples, and cumulative optimizer steps;
- validation cell AUC/LogLoss and macro metrics for every epoch; selected epoch and shadow stop diagnostic;
- selected/final/common checkpoints and hashes; deferred test predictions, labels, row IDs, per-cell AUC/LogLoss;
- main/architecture optimizer states, RNG state, nonfinite inventory, runtime/device information, wall time, peak memory;
- worker/co-run/process fields and all failure/archive/retry events.

The S1 reuse audit must log the same provenance fields available in frozen artifacts and an explicit reason for each `REUSE_OK` or `REUSE_REJECT`. Missing values cannot be silently replaced.

---

## 10. Acceptance criteria for reviewer

The reviewer must independently verify all items below before recommending formal execution:

1. The formal fresh matrix is exactly 16 unique M3OE Equal joint keys with seeds 2022–2025 and the four prescribed settings; no Sample/reference retraining is queued unless a named reuse key failed audit.
2. All 64 reused S1 M3OE artifacts pass the frozen validator and the reuse audit is complete, hash-addressed, and read-only.
3. Sample's actual code path is unweighted per-example/task mean BCE over an ordinary shuffled concatenated loader: `p_d=n_d/N`, `w_d=1`, effective domain coefficient `p_d`.
4. Equal uses the same loader/order/exposure and only `w_d=1/(3p_d)`; numerical tests prove effective coefficient 1/3 and reject double weighting or per-batch renormalization.
5. Main and architecture update paths use the same selected objective; no path silently retains Sample.
6. Equal and Sample pairing invariants (data, initialization, batch order, optimizer, exposure, metric/checkpoint contract) are enforced against the frozen S1 manifests.
7. M3OE model code and architecture are unchanged; no extra backbone, algorithm, optimizer, Momentum/EMA/gamma, schedule, sampler, or tuning path exists.
8. Checkpoint selection is validation-only, test is deferred, selected/final/common state coverage is complete, and epoch-49 duplicate labels agree.
9. Raw and paired schemas have exact key uniqueness and expected 2,016-row cardinality each after completion; all 96 seed-cells exist for every method/contrast at each checkpoint label.
10. Metric calculations use the shared S1 implementation and can be recomputed from raw predictions; cell index/physical-domain/task identity is correct.
11. The two-process scheduler isolates outputs, records co-runs, stops scheduling after first failure, preserves a running sibling, and never overwrites valid/failed evidence.
12. Failure recovery restarts only the failed key from epoch 0 and archive behavior is append-only.
13. Nonformal objective, tiny M3OE forward/backward, artifact, pairing-corruption, and concurrent-resource smokes pass. No formal result is generated during review.
14. Implementation report lists every changed/added file and demonstrates that S1 artifacts/modules remain unchanged.
15. Reviewer verdict is `PASS`, or `PASS WITH FIXES` with every blocking fix independently rechecked. Even then, the orchestrator must stop and request the user's explicit approval before invoking `luna_runner`.

---

## 11. Conditions that would falsify the hypothesis

H-EQ-NT is falsified or unsupported if any of the following holds:

- no Equal-versus-reference group has mean AUC `<= -0.003` with at least 3/4 negative seeds and concordant LogLoss;
- apparent deficits are only isolated seed events, are comparable to reference seed variability, or disappear/reverse at common/final epochs;
- Equal is worse than Sample but both remain above the single-domain reference;
- Sample fails to improve the same cells that are degraded under Equal, or Sample is itself similarly degraded;
- the pattern does not recur across relevant MovieLens rotations and has no compatible KuaiRand evidence, implying condition dependence rather than a general phenomenon;
- the pattern follows physical domain/task or validation baseline difficulty without a systematic within-domain L/M/S change, weakening the size explanation;
- results depend on post-hoc checkpoint choice, threshold revision, missing cells, invalid provenance, or failed pairing.

If Equal and Sample both lack stable practical negative transfer, the negative-transfer mechanism line must stop. If both exhibit it, objective weighting is not established as the main explanation. If only narrowly conditioned cells satisfy the gate, any later research question must be restricted to those conditions. S2 itself does not design that later stage.

---

## Design hand-off summary

The minimum valid implementation is an isolated, M3OE-only S2 extension that leaves S1 frozen, revalidates and reuses 64 S1 artifacts, launches only 16 Equal joint runs after review and user approval, and proves through logged `p_d`, `w_d`, and `p_dw_d` that the comparison is Sample effective weights `p_d` versus Equal effective weights `1/3` under identical sampling and exposure. All conclusions remain reserved for the main-analysis stage after the runner has produced accepted raw data.
