# Stage S3 DESIGN SPECIFICATION

## Amendment A1 — source-identity repair after the first formal pair (binding)

This amendment supersedes every earlier statement in this document that treats the eight historical S1 Sample FULL joint artifacts as legal source-identical S3 controls, and every count or queue rule derived from that assumption. It does **not** add a scientific factor, alter a hypothesis, or authorize Stage S4.

The newly discovered blocking fact is exact: the historical S1 `run_manifest.json/code_hashes` inventories do not contain `m3oe/MDMTRec/models/multi_domain/model.py`. The S2 and original S3 reuse audits copied that absence; they did not create contemporaneous evidence of the S1 forward source. Equality of dataset, parameter manifest, parameter count, forward-FLOP estimate, semantic initial state, optimizer, exposure, batch order, checkpoints, or learned outputs cannot prove source-file identity. The missing hash must not be imputed from the current checkout, from S2, or from checkpoint compatibility. Therefore all eight S1 Sample FULL joint keys are **revoked as controlled S3 FULL evidence** and must fail the strict source-identity validator with `model_source_hash_mismatch` (missing is not equal).

The minimum compliant repair is:

1. Run exactly eight fresh `sample_full_replacement` joint jobs: two frozen settings × seeds 2022–2025. They use the current independently reviewed M3OE source and the unchanged Sample objective, `w_d=1`, under ordinary FULL shared-gradient access.
2. Do **not** retrain Equal FULL. All eight accepted S2 Equal FULL manifests contemporaneously record SHA-256 `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` for `m3oe/MDMTRec/models/multi_domain/model.py`, equal to the audited current file and to both completed S3 wave-2 target-only manifests. Their data, initialization, batches, optimizer, exposure, metric, and checkpoint evidence also remains complete. Any later hash drift invalidates this decision and stops execution; it does not trigger an automatic Equal retrain.
3. Keep the 16 target-only keys. The completed KuaiRand/natural/seed-2022 Equal and Sample wave-2 runs may be retained only by amendment-specific, read-only revalidation described below. The remaining 14 target-only jobs retain their exact frozen scientific configurations.
4. Keep the 24 historical S1 D=1 artifacts only as `legacy_sd_context`. Their missing model-source hash means they are not members of the strict source-identical 2 × 2 factorial and cannot independently establish a new causal S3 comparison. They preserve continuity with the already accepted S1/S2 negative-transfer classification. No S3 mechanism conclusion may depend on assuming their unrecorded source hash.

The amended evidence inventory is thus 56 artifacts: **24 fresh joint runs** (16 target-only plus 8 Sample FULL replacements), **8 reused S2 Equal FULL joint runs**, and **24 reused S1 D=1 legacy contextual references**. The formal launch matrix is 24 keys, not 16. The scientific four-condition factorial is unchanged.

### A1.1 Strict four-way block

For every `setting × seed` block, the four joint conditions are:

| condition | objective | access | provenance after amendment |
|---|---|---|---|
| `sample_full` | Sample | FULL | fresh S3 replacement |
| `equal_full` | Equal | FULL | reused accepted S2 artifact with explicit model-source hash |
| `sample_target_only_shared` | Sample | TARGET_ONLY_SHARED | fresh S3 |
| `equal_target_only_shared` | Equal | TARGET_ONLY_SHARED | fresh S3 |

All four must have a nonempty, exactly equal M3OE forward-source hash and matching dataset/split/view, architecture/parameter manifest, initial state, exposure, optimizer, main and architecture batch order, metric, and checkpoint contracts within the allowances already defined for objective and access. No blank-hash equivalence is permitted. The Sample FULL replacement must match its corresponding Sample target-only run exactly on all held-constant fields; Equal target-only must match the accepted S2 Equal FULL control. The cross-objective comparisons then use the same source hash and the prespecified objective as the only objective difference.

### A1.2 Updated structural counts

The scientific row shapes do not change because a replacement changes provenance, not a condition:

- joint raw rows: `4 × 2 × 4 × 6 × 7 = 1,344`;
- D=1 legacy contextual raw rows: `2 × 4 × 3 × 2 × 7 = 336`;
- total raw rows: **1,680**;
- joint-to-joint strict contrasts: `4 contrast types × 336 = 1,344` rows;
- condition-to-D=1 legacy contextual contrasts: `4 × 336 = 1,344` rows;
- combined paired table: **2,688** rows, with an explicit `contrast_evidence_class` separating the two 1,344-row subsets;
- strict four-way interactions: **336** rows;
- run inventory: **56** rows.

`pairing_valid=true` for a strict joint contrast requires all source hashes to be present and equal. Legacy D=1 contrasts must instead carry `contrast_evidence_class=legacy_sd_context`, `strict_source_identity=false`, and must never be counted as strict causal pairs.

### A1.3 Updated execution and recovery plan

The amended executor contains 12 fixed two-process launch groups: the original eight same-setting/same-seed Equal/Sample target-only pairs, plus four resource-only Sample FULL replacement pairs (`KuaiRand seeds 2022+2023`, `KuaiRand 2024+2025`, `MovieLens 2022+2023`, `MovieLens 2024+2025`). The latter grouping is only a fixed resource schedule; scientific pairing remains by identical setting and seed. Maximum concurrency remains two and may not be reduced or increased silently.

The first target-only pair's wave-1 parent-abort archive and wave-2 full archive remain append-only. The two wave-2 children returned 0 and contain complete artifacts, but were rejected by the now-identified empty-inventory reader defect and subsequently exposed the Sample source gap. Neither event makes their metrics accepted evidence. They may be promoted without retraining only if an independent reviewer verifies all of the following against the immutable archive: exact queue/task identity; empty `{}` nonfinite inventory interpreted as no entries; all artifact hashes and prediction-level metrics; 50-epoch/checkpoint/batch/step completeness; no live or partial canonical result; the recorded forward hash equals the reviewed hash above; Equal pairs strictly with S2 Equal FULL; and Sample is marked `PENDING_SAMPLE_FULL_REPLACEMENT` until its fresh seed-2022 FULL replacement passes the full strict pairing validator. Promotion requires a new hash-bound recovery receipt and atomic two-member copy. If either archived child fails revalidation, execution stops for review; no silent third target-only attempt is authorized.

The remaining 14 target-only jobs run from epoch 0 in new immutable attempt directories. All eight Sample FULL replacements also run from epoch 0. A completed target-only run is not a valid factorial member until its same-block FULL control passes. Pair/launch recovery remains append-only, at most one retry of the affected new launch group from epoch 0, with no singleton repair or checkpoint resume.

Observed runtimes provide planning estimates, not acceptance thresholds. A KuaiRand process has taken about 5,186–5,343 seconds (the completed S3 pair about 5,286 seconds wall); a MovieLens process about 556–746 seconds. Under the fixed two-process schedule, the full amended 24-run plan is approximately 10 wall-clock hours and 20 GPU-process-hours. With the two completed wave-2 target-only runs retained, approximately 22 runs / 11 launch groups remain, about **8.5 wall-clock hours** and **17 GPU-process-hours**, plus validation/reporting overhead. Runtime variation cannot justify changing the matrix or concurrency.

All later sections are read subject to Amendment A1. Where a later statement conflicts, A1 controls.

## Scope, evidence basis, and hard boundary

This specification covers **Stage S3 only: a falsification-oriented diagnosis of objective weighting, cross-domain shared-update access, and training-stage dependence in the stable Large-domain negative-transfer conditions identified by Stage S2**.

The design does not introduce a new recommender, mitigation method, sampler, optimizer, scheduler, backbone, hyperparameter search, Momentum/EMA/gamma mechanism, curriculum, or capacity change. It does not design Stage S4 or authorize post-hoc expansion. Scientific interpretation remains reserved for the Main Agent after the runner has completed and the post-run data have passed independent review.

The design was produced from a read-only audit of:

- `promt/negative_analyse.md`;
- `experiments/reports/stage_S2_{design,implementation,review,analysis}.md`;
- all accepted S2 raw, paired, role, rotation, difficulty, checkpoint, and training-stage tables;
- the 16 fresh Equal and 64 reused S1 raw artifact directories accepted by the final S2 review;
- `experiments/stage_S1_models.py`, `stage_S1_run.py`, `stage_S2_objectives.py`, `stage_S2_run.py`, the S1/S2 protocol and reporting modules;
- `m3oe/MDMTRec/models/multi_domain/model.py` and the accepted per-run parameter manifests.

S2's formal decision was Situation D. Therefore S3 is restricted to the stable, conditional diagnosis object. The two KuaiRand `tab1` tasks are primary. MovieLens `d2_large`, physical domain `ages_35_45_50_56`, task `rating>3`, is a secondary replication/falsification target; its `rating>4` task is retained as a prespecified within-domain counterexample. The other MovieLens rotations are not part of S3.

The implementation and review gates remain mandatory. In accordance with the user's current instruction, a reviewer `PASS`, or `PASS WITH FIXES` after all blocking fixes are independently rechecked, authorizes automatic transition to `luna_runner`; there is no additional user-approval pause. This removal of the approval pause does not relax any technical or scientific review criterion.

---

## 1. Research question

### RQ1 — does the Equal deficit survive removal of non-target shared updates?

For the preregistered Large target domain in each setting, does Equal remain worse than the source-identical Sample joint control when non-target-domain examples are prevented from updating every parameter that can affect more than one domain, while batch order, example exposure, optimizer steps, initialization, forward architecture, objective coefficients, and checkpoint rules remain unchanged? Historical single-domain two-task results are retained only to locate the S3 cells within the already accepted S1/S2 negative-transfer context; their missing forward-source hash prevents a new source-controlled S3 causal claim against them.

### RQ2 — do cross-domain shared updates causally contribute to the deficit?

Within Equal and within Sample, does target-only access to cross-domain shared parameters improve the target Large-domain AUC relative to normal full shared-gradient access? Does the intervention preferentially improve Equal, thereby reducing the Equal-minus-Sample gap?

### RQ3 — is the result instead a checkpoint/convergence phenomenon?

Are the objective and access effects present at validation-selected checkpoints but absent, reversed, or practically negligible at matched common epochs and the fixed final epoch? S3 must distinguish a persistent target effect from different convergence rates or selected-checkpoint timing.

### RQ4 — how narrowly may any mechanism claim be made?

Does the same intervention pattern occur on both KuaiRand tab1 tasks and, secondarily, on MovieLens d2_large `rating>3`? The experiment may support a mechanism contribution within these conditions. It cannot establish that domain size is causal, that Equal is generally harmful, or that gradient conflict/capacity competition is a universal M3OE mechanism.

---

## 2. Hypotheses being tested

All hypotheses are competing explanations; none is assumed true.

### H-W — effective weighting/direct-optimization explanation

The Large-domain deficit primarily follows the effective target coefficient and its downstream optimization consequences, not non-target shared gradients. Prediction: the Equal-minus-Sample deficit persists under target-only shared access, and the access intervention has little Equal-specific effect.

### H-I — cross-domain shared-update interference contribution

Gradients contributed by non-target domains through parameters used by the target materially worsen target performance. Prediction: blocking those contributions improves Equal on the target cells across seeds and matched stages.

### H-IxW — weighting-by-shared-access interaction

Non-target shared updates explain specifically why Equal is worse than Sample. Prediction: target-only shared access attenuates the Equal-minus-Sample gap; its improvement is larger under Equal than under Sample. This is stronger than merely observing that both objectives benefit equally from the access control.

### H-C — convergence/checkpoint explanation

The apparent deficit is chiefly a difference in optimization timing or checkpoint selection. Prediction: selected-checkpoint effects fail to persist at at least three of five common epochs and are absent or reversed at epoch 49.

### H0 — no identified mechanism under this intervention

The access effect is below practical magnitude, seed-unstable, metric-discordant, or selected-checkpoint-only; or the objective gap persists without an access interaction. In that case S3 must not claim that shared interference has been identified.

S3 is explicitly designed to falsify H-I and H-IxW. A failure of the access intervention to rescue the target is an informative result, not a failed experiment.

---

## 3. Variables changed and the only allowed experimental matrix

### 3.1 Factors

| factor | levels | meaning |
|---|---|---|
| objective weighting `W` | `Sample`, `Equal` | existing S2 definitions; no new normalization |
| shared-gradient access `A` | `FULL`, `TARGET_ONLY_SHARED` | whether non-target domains may update cross-domain shared parameters |
| setting | KuaiRand `natural`; MovieLens `d2_large` | fixed diagnostic conditions |
| seed | 2022, 2023, 2024, 2025 | paired repeat unit |
| checkpoint view | `selected`, `final`, `common_epoch_9/19/29/39/49` | prespecified stage analysis |

### 3.2 Target-domain mapping

| setting | target local domain | physical identity | size role | primary tasks |
|---|---:|---|---|---|
| KuaiRand `natural` | 1 | `tab1` | Large | `click`, `long_view` |
| MovieLens `d2_large` | 2 | `ages_35_45_50_56` | Large | `rating>3` primary; `rating>4` counterexample |

### 3.3 Formal 2 × 2 matrix

For each setting and seed:

| condition | objective | shared access | source |
|---|---|---|---|
| `sample_full` | Sample | FULL | fresh S3 source-identity replacement |
| `equal_full` | Equal | FULL | reuse accepted S2 joint artifact |
| `sample_target_only_shared` | Sample | TARGET_ONLY_SHARED | fresh S3 run |
| `equal_target_only_shared` | Equal | TARGET_ONLY_SHARED | fresh S3 run |

This is the complete and unique S3 matrix. It contains:

- **24 fresh runs**: 16 `TARGET_ONLY_SHARED` plus 8 Sample FULL replacements;
- **8 reused full-access joint runs**: accepted S2 Equal only;
- **24 reused S1 single-domain two-task references**: legacy contextual evidence only;
- **56 total run artifacts** in the S3 evidence inventory.

All six domain-task cells from every joint run are retained. Mechanism gates are applied only to the preregistered Large target cells. Other cells are redistribution/safety outcomes and may not be substituted post hoc as primary targets.

No other setting, rotation, model, seed, target-domain choice, access boundary, objective coefficient, or follow-up round is authorized. A failed hypothesis may not trigger an unreviewed matrix expansion.

### 3.4 Expected structural cardinalities

Seven checkpoint labels are retained. Expected raw rows are:

- joint conditions: `4 conditions × 2 settings × 4 seeds × 6 cells × 7 checkpoints = 1,344`;
- single-domain references: `2 settings × 4 seeds × 3 domains × 2 tasks × 7 checkpoints = 336`;
- total `stage_S3_raw_results.csv`: **1,680 unique rows**.

The required eight direct contrasts remain, but their evidence class is explicit:

1. `equal_full - SD`;
2. `sample_full - SD`;
3. `equal_target_only_shared - SD`;
4. `sample_target_only_shared - SD`;
5. `equal_full - sample_full`;
6. `equal_target_only_shared - sample_target_only_shared`;
7. `equal_target_only_shared - equal_full`;
8. `sample_target_only_shared - sample_full`.

Across all cells/checkpoints these produce `8 × 336 = 2,688` unique rows in `stage_S3_paired_results.csv`. A separate interaction table contains `2 settings × 4 seeds × 6 cells × 7 checkpoints = 336` rows.

Contrasts 5–8 are the **1,344 strict source-identical factorial rows**. Contrasts 1–4 are **1,344 legacy SD-context rows** and may not be promoted to strict causal evidence. The interaction table is strict four-way evidence.

---

## 4. Variables held constant

Within each `dataset × setting × seed` block, the four joint conditions must hold constant:

- exact dataset bytes, global row order, splits, selected row IDs, preprocessing, feature vocabularies, OOV rules, task definitions, physical/local-domain mapping, and train/validation/test samples;
- M3OE forward architecture, dimensions, expert count, parameter count, parameter initialization, and all model source logic;
- ordinary concatenated shuffled loader, batch size 40,960, `drop_last=False`, `num_workers=0`, train batches and architecture-update batches;
- 50 epochs, per-example exposure, number of main optimizer steps, one architecture-optimizer step after every epoch, and no early stopping;
- Adam definitions: main `lr=0.01`, architecture `lr=0.001`, betas `0.9/0.999`, weight decay `1e-5`, no scheduler;
- objective coefficients for the chosen `W` level, including their global scale;
- validation-only six-cell unweighted macro-AUC checkpoint selection with earliest-epoch tie break; fixed epoch 49 and common epochs 9/19/29/39/49; deferred test evaluation;
- S1 tie-aware AUC and fixed-epsilon LogLoss, task-major cell indexing, device/determinism policy, and artifact schema;
- exactly one forward pass and one main optimizer step per train batch, and one forward pass and one architecture optimizer step per architecture batch;
- identical semantic initial state and identical main/architecture batch-order hashes between fresh target-only runs and their reused full-access counterpart of the same objective/setting/seed.

The only permissible algorithmic difference between `FULL` and `TARGET_ONLY_SHARED` is which decomposed loss supplies `.grad` to the preregistered cross-domain shared parameter set. Private parameters continue to receive the full objective. No parameter is frozen, duplicated, reinitialized, or removed; no forward activation is detached beyond the existing M3OE detach in the gate input.

---

## 5. Mathematical and algorithmic definition

### 5.1 Objective factor

For batch size `B`, two tasks, local domain `d_i`, prediction `p_it`, target `y_it`, define

\[
\ell_i=\frac{1}{2}\sum_{t=1}^{2}\operatorname{BCE}(p_{it},y_{it}).
\]

The unchanged full batch objective is

\[
L_W=\frac{1}{B}\sum_{i=1}^{B}w^{W}_{d_i}\ell_i.
\]

For Sample:

\[
w_d^{Sample}=1,
\qquad
\mathbb E[\text{domain coefficient}]=p_d.
\]

For Equal, with `D=3` and immutable full-training share `p_d=n_d/N`:

\[
w_d^{Equal}=\frac{1}{3p_d},
\qquad
\mathbb E[\text{domain coefficient}]=\frac13.
\]

The target component is

\[
L_W^{*}=\frac{1}{B}\sum_{i=1}^{B}
\mathbf 1[d_i=d^{*}]w^{W}_{d_i}\ell_i,
\qquad
L_W^{non}=L_W-L_W^{*}.
\]

Critically, `L_W*` keeps the denominator `B` and the original coefficient `w_d`. It must **not** be renormalized by the number of target examples, multiplied by `p_d` again, rescaled to match a gradient norm, or divided by its observed minibatch share. This preserves the objective-weighting factor while changing only gradient access.

Concrete target coefficients are:

| setting/target | `p_target` | Sample `w_target` | Equal `w_target` | Sample effective coefficient | Equal effective coefficient |
|---|---:|---:|---:|---:|---:|
| KuaiRand tab1 | 0.8692837759 | 1 | 0.3834574423 | 0.8692837759 | 1/3 |
| MovieLens d2 Large | 0.8000000000 | 1 | 0.4166666667 | 0.8000000000 | 1/3 |

### 5.2 Audited parameter dependency partition

The old S1/S2 `parameter_manifest.shared_names` field is a semantic D=3-to-D=1 initialization/reporting partition. It is **not** a valid S3 cross-domain dependency boundary. In particular, the real forward graph mixes every `domain_expert.*` into every domain through `weighted_domain`, and each `task_expert.*` is used by the same task across all domains.

The S3 intervention partition is therefore:

#### Cross-domain shared main parameters `C`

- `embedding.*`;
- `shared_weight`, `shared_bias`;
- `skip_conn.*`;
- `star_mlp.*`;
- `expert.*`;
- `domain_expert.*`;
- `task_expert.*`.

#### Domain/cell-private main parameters `P`

- `slot_weight.*`, `slot_bias.*`;
- `gate.*`;
- `tower.*`.

#### Cross-domain shared architecture parameters `A_arch`

- `_weight_exp_d.deep_weights`;
- `_weight_exp_t.deep_weights`;
- `_weight_bal_d.deep_weights`;
- `_weight_bal_t.deep_weights`.

The accepted parameter manifests imply the following audited counts under this partition:

| setting | cross-domain shared main | private main | shared architecture | total |
|---|---:|---:|---:|---:|
| KuaiRand | 80 tensors / 744,848 parameters | 54 / 869,406 | 4 / 4 | 1,614,258 |
| MovieLens | 53 tensors / 566,080 parameters | 54 / 207,390 | 4 / 4 | 773,474 |

Implementation must derive and hash the complete named partition, fail on any unclassified/duplicated name, and verify it against an autograd reachability audit. A per-domain two-task loss must reach every member of `C` and `A_arch` from at least two domains at the tensor level; private slot/gate/tower tensors may only be reached from their owning domain/cell. Sparse embedding rows need not all be shared, but the table is a common tensor; target-only gradients naturally update only target-reached rows.

### 5.3 Gradient-access factor

For a main training batch:

#### FULL

\[
g_c=\nabla_{\theta_c}L_W \quad(c\in C),
\qquad
g_p=\nabla_{\theta_p}L_W \quad(p\in P).
\]

#### TARGET_ONLY_SHARED

\[
g_c=\nabla_{\theta_c}L_W^{*} \quad(c\in C),
\qquad
g_p=\nabla_{\theta_p}L_W \quad(p\in P).
\]

For the one architecture update after each epoch:

\[
g_a^{FULL}=\nabla_{\alpha_a}L_W,
\qquad
g_a^{TARGET\_ONLY}=\nabla_{\alpha_a}L_W^{*}
\quad(a\in A_{arch}).
\]

The runner must compute both losses from the same model forward for a batch. Under target-only access, it may obtain target gradients with `autograd.grad`, obtain full gradients for the existing private path, then replace only the shared `.grad` tensors before the single optimizer step. It must not take separate optimizer steps, accumulate the target twice, or alter Adam state/weight decay semantics.

This intervention removes `∇L_non` only from cross-domain shared parameters. All domains still appear in every epoch; their private parameters still learn from the complete objective; their metrics remain observable. The control does not assert that the removed gradients are conflicting. It tests whether access to the target's shared parameters is causally important.

### 5.4 Factorial estimands

Let `Y_{W,A}` be AUC for one cell/seed/checkpoint. Define:

\[
D_W^{FULL}=Y_{Equal,FULL}-Y_{Sample,FULL},
\]

\[
D_W^{BLOCK}=Y_{Equal,TARGET}-Y_{Sample,TARGET},
\]

\[
R_E=Y_{Equal,TARGET}-Y_{Equal,FULL},
\qquad
R_S=Y_{Sample,TARGET}-Y_{Sample,FULL},
\]

\[
I=(Y_{Equal,TARGET}-Y_{Sample,TARGET})
 -(Y_{Equal,FULL}-Y_{Sample,FULL})
 =R_E-R_S.
\]

Positive `R_E` is an Equal rescue. Positive `I` means access blocking attenuates the Equal-minus-Sample disadvantage more under Equal than under Sample. Each condition is also compared with the same single-domain two-task reference:

\[
N_{W,A}=Y_{W,A}-Y_{SD}.
\]

For LogLoss the raw left-minus-right difference is retained, where negative is better. The report must additionally compute `Q=-LogLoss` and the same factorial estimands on `Q` so AUC and probability-quality directions can be compared without sign ambiguity.

---

## 6. Exact affected modules/files if identifiable

The coder should add an isolated S3 layer and leave all S1/S2 code and artifacts immutable:

- `experiments/stage_S3_protocol.py`: the immutable 24-key fresh matrix (16 target-only + 8 Sample FULL replacements), two settings, target mapping, seeds, factors, paths, thresholds, contrasts, cardinalities, and 12 frozen launch groups.
- `experiments/stage_S3_gradient_access.py`: exact name partition, partition/reachability audits, target/full loss decomposition, and gradient substitution helpers.
- `experiments/stage_S3_run.py`: M3OE-only runner derived from or importing the accepted S2/S1 behavior; supports the two fresh `TARGET_ONLY_SHARED` objectives and, after explicit formal-key gating, Sample-only `FULL_REPLACEMENT`. A separate thin replacement runner is also acceptable if it imports the same reviewed training/data/model path and produces the identical evidence schema.
- `experiments/stage_S3_orchestrator.py`: reuse audit, immutable two-process queue, attempt isolation, validation, and recovery policy.
- `experiments/stage_S3_report.py`: read-only collection, raw/direct-contrast/interaction tables, mechanical aggregates, and no scientific interpretation.
- `experiments/tests/test_stage_S3_gradient_access.py`: partition, reachability, decomposition, gradient replacement, and FULL-equivalence tests.
- `experiments/tests/test_stage_S3_pairing.py`: 56-run evidence inventory, hashes, pairing, row counts, and corruption tests.
- `experiments/tests/run_stage_S3_smoke.py`: nonformal real-M3OE main/architecture update and two-process CUDA resource smoke.
- `experiments/stage_S3_reuse_audit.{csv,json}`, `stage_S3_raw_results.csv`, `stage_S3_paired_results.csv`, `stage_S3_interactions.csv`, `stage_S3_runs.csv`, `stage_S3_summary_manifest.json`, and `experiments/reports/stage_S3_run_report.md`.
- fresh target-only artifacts under `experiments/stage_S3_raw_results/target_only_shared/<objective>/<dataset>/<setting>/m3oe/seed<seed>/joint/` and Sample FULL replacements under `experiments/stage_S3_raw_results/full_replacement/sample/<dataset>/<setting>/m3oe/seed<seed>/joint/`, with attempt subdirectories or append-only archives outside canonical accepted directories.

Do not edit `m3oe/MDMTRec/models/multi_domain/model.py`, any S1/S2 source module, manifest, table, checkpoint, prediction, or report. If the exact intervention cannot be implemented without changing the M3OE forward architecture, implementation must stop and return to design/review rather than approximate the control.

---

## 7. Implementation requirements

### 7.1 Legal reuse audit

The accepted S2 evidence already establishes that, for both target settings and all four seeds:

- Historical Equal and Sample artifacts have identical dataset/split/view fingerprints, initial parameter-state hashes, main batch-order hashes, architecture batch-order hashes, optimizer definitions, exposure, step counts, metric implementation, and checkpoint contract; this remains useful controlled-field evidence but does not repair the missing Sample forward-source hash;
- KuaiRand joint runs have 2,450 main and 50 architecture steps; MovieLens runs have 250 main and 50 architecture steps;
- the accepted artifacts contain selected, final, and five common checkpoints plus predictions/metrics;
- all S1/S2 raw artifacts passed prediction-level metric recomputation and immutable-hash review.

The old conclusion that all 40 keys are controlled reuse evidence is revoked. The amended audit has three explicit inventories:

- **8/8 controlled reused joint keys:** S2 Equal FULL, each with a nonempty forward-source hash equal to the reviewed S3 source;
- **8/8 rejected controlled joint keys:** S1 Sample FULL, preserved in the audit as `REJECTED_MISSING_MODEL_SOURCE_HASH` and excluded from S3 raw/paired/interaction scientific inputs;
- **24/24 legacy contextual keys:** S1 D=1 references, accepted only as `legacy_sd_context`, not as strict source-identical pairs.

The new reuse evidence file must retain all 40 audited historical keys so that the eight revocations are visible; its summary must report `controlled_reuse_valid=8`, `controlled_reuse_rejected=8`, and `legacy_context_valid=24`, rather than claiming `40/40 REUSE_OK`. Fresh replacement is fail-closed and does not modify any S1/S2 artifact.

### 7.2 Fresh-to-reused pairing invariants

For each fresh target-only run and its same-objective FULL counterpart—fresh Sample replacement or reused S2 Equal—require exact equality of:

- dataset bytes and all split, selected-row, physical-view, preprocessing, task, and domain identities;
- a present and equal `m3oe/MDMTRec/models/multi_domain/model.py` SHA-256, parameter manifest, parameter count, forward-FLOP estimate, and initial parameter-state hash;
- train counts/shares, objective coefficient table/hash, batch size, main/architecture batch fingerprints and ordered hashes;
- epochs, examples processed, optimizer steps, architecture steps, optimizer definitions, and runtime/determinism policy;
- metric/checkpoint contract, fixed/common epochs, validation-only selection, and deferred-test flags.

The only allowed unequal fields are stage/run/output identity, access label, gradient-partition/audit fields, learned state/checkpoints/metrics, target/non-target component logs, wall time, and incidental resource measurements. Missing source hashes never compare equal. A complete, hash-addressed relevant-source inventory must cover the M3OE forward model plus S1 data/model/metric code and the objective/access implementation actually used by that condition. Condition-specific wrapper hashes may differ only where they implement the prespecified objective/access factor; shared imported sources must match exactly.

### 7.3 Exact gradient-control invariants

1. `L_full = L_target + L_non` within numerical tolerance on every audited batch.
2. The target mask is the immutable local-domain ID from the protocol. The target must appear in every formal main/architecture batch; absence is a hard error, not a renormalization trigger.
3. Every main parameter is classified exactly once as `C` or `P`; every architecture parameter is classified exactly once as `A_arch`; no unclassified name is allowed.
4. In target-only main updates, applied shared gradients equal `grad(L_target)` and applied private gradients equal `grad(L_full)` within tolerance. In architecture updates, applied architecture gradients equal `grad(L_target)`.
5. Removed non-target shared gradients are never added, norm-matched, clipped separately, projected, or used in an extra update.
6. Exactly one main optimizer step occurs per train batch and one architecture optimizer step per epoch. Adam parameter groups and state layout remain unchanged.
7. The reviewed `FULL` compatibility path becomes a formal path **only for the eight Sample FULL replacements**. It must reproduce direct Sample `L_W`, gradients, one-step parameters, Adam state, batch fingerprint, and counters from the same initial state/batch. Equal FULL remains reused S2 evidence and no other fresh FULL condition is legal.
8. Nonfinite loss components, gradients, parameters, missing gradients, name drift, partition-hash drift, target-ID mismatch, or batch-order mismatch fail before an accepted artifact is written.

### 7.4 Checkpoint and test boundary

- Train all fresh runs through epochs 0–49; shadow patience is diagnostic only.
- Select each fresh condition independently using validation six-cell unweighted macro-AUC and earliest tie, before test evaluation.
- Save selected, final epoch 49, and common epochs 9/19/29/39/49. Final and common-49 state/prediction/metric values must be identical.
- Primary reporting is selected-to-selected for continuity with S2. Common-epoch and final comparisons are mandatory and may falsify a selected-checkpoint mechanism claim.
- Test data cannot choose the target, access boundary, checkpoint, retry, threshold, or inclusion.

### 7.5 Two-process formal plan

After reviewer acceptance:

- freeze exactly eight co-run pairs, each pairing `equal_target_only_shared` and `sample_target_only_shared` for the same setting and seed;
- freeze four additional two-process resource launch groups containing the eight Sample FULL replacements in the exact grouping in Amendment A1; these groups do not redefine scientific pairing;
- run exactly two child processes per pair, maximum concurrency two, with independent directories, locks, stdout/stderr, checkpoints, and temporary files;
- process seeds in deterministic order 2022, 2023, 2024, 2025 within each setting; worker assignment is logged and is not a scientific factor;
- the parent alone atomically updates the queue after both children exit and validate;
- record pair ID, reciprocal peer key, worker ID, PID, start/end times, GPU/device, peak memory, exit code, validator outcome, and attempt ID;
- immediately before launch, run reviewed two-process CUDA smoke coverage for target-only Equal/Sample and a formal Sample FULL replacement update at representative KuaiRand scale. Failure stops formal launch; concurrency may not be silently reduced.

### 7.6 Failure recovery

- No mid-run checkpoint resume.
- A data/hash/protocol/partition/pairing mismatch is blocking and is not retried by changing metadata.
- If either child in a co-run pair fails for a transient process/resource reason, allow the sibling to finish and preserve both attempts. Archive the entire pair append-only, then automatically rerun **both members once from epoch 0** under the same frozen pair/config so the accepted pair shares an attempt wave. The earlier valid sibling is evidence of execution, not the accepted scientific result.
- If the repeated pair fails, or if the failure is OOM/capacity or deterministic numerical failure, stop scheduling. The issue returns to coder/reviewer; there is no user-approval gate, but no matrix/concurrency change may bypass review.
- Valid completed earlier pairs are retained and never overwritten. Canonical promotion occurs only after both members of a pair pass validation.
- The archived first wave-2 pair follows the narrower A1.3 recovery rule. Its recovery is a validation/promotion operation, not a new run or an exception to source identity. The Sample member remains scientifically pending until its fresh Sample FULL control passes.

---

## 8. Required comparisons and ablations

The following are all required and no optional algorithmic ablation is permitted:

1. The complete 2 × 2 objective-by-access factorial comparison on both settings and all seeds.
2. Every condition versus the same physical-domain single-domain two-task reference as a clearly labeled legacy contextual comparison, not a new strict source-identical causal contrast.
3. Equal versus Sample under FULL and under TARGET_ONLY_SHARED.
4. TARGET_ONLY_SHARED versus FULL within Equal and within Sample.
5. The preregistered difference-in-differences interaction `I`.
6. All comparisons at selected, final, and five common epochs.
7. Independent analysis of each KuaiRand tab1 task; their average cannot conceal a failed task.
8. MovieLens d2_large `rating>3` as secondary replication and `rating>4` as the prespecified same-domain/task counterexample.
9. All non-target cells as redistribution/safety outcomes, including degradation caused by allowing only the target to update shared parameters.
10. AUC and LogLoss/negative-LogLoss concordance, plus validation trajectory and selected-epoch differences.

Prohibited additions include gradient projection/surgery, norm matching, freezing all shared parameters, domain-specific model copies, balanced sampling, altered batch composition, new objectives, alternate target domains, new backbones, extra seeds, learning-rate changes, architecture changes, and threshold changes after viewing results.

---

## 9. Expected logging and raw schema

### 9.1 Per-run evidence

Every fresh run, including Sample FULL replacement, must log:

- exact config, complete relevant-source SHA-256 inventory including `m3oe/MDMTRec/models/multi_domain/model.py`, runtime/device fingerprint, formal/attempt/launch-group identity, and provenance role (`fresh_target_only` or `fresh_sample_full_replacement`);
- all dataset/split/view/preprocessing/task/domain fingerprints and counts;
- objective counts, shares, coefficients, effective coefficients, formula, and coefficient-table hash;
- target local/physical identity, access mode, complete `C/P/A_arch` name lists, tensor/parameter counts, and partition hash;
- autograd reachability audit and explicit explanation that `domain_expert` and `task_expert` are cross-domain shared for S3;
- semantic initialization audit, initial/final parameter hashes, optimizer states, RNG state, and batch-order hashes;
- for every main and architecture batch: batch fingerprint, domain counts, `L_full`, `L_target`, `L_non`, equality residual, applied shared/private/architecture gradient norms, removed non-target shared-gradient norm, and gradient substitution checks;
- examples and optimizer-step counters; validation cell metrics for every epoch; selected epoch and shadow-stop diagnostic;
- selected/final/common checkpoint hashes, deferred test predictions/labels/row IDs, per-cell AUC/LogLoss, nonfinite inventory, timing, memory, and all failure/archive/retry events.

Gradient norms are implementation/intervention evidence. They must not be treated as proof of gradient conflict or beneficial alignment.

### 9.2 `stage_S3_raw_results.csv`

Required identity fields include:

`stage, condition, objective_weighting, shared_access, source_stage, provenance_type, formal_run_key, source_formal_run_key, dataset, setting, backbone, seed, target_local_domain, target_physical_domain, evaluated_local_domain, physical_domain, physical_domain_identity, size_role, task, task_identity, checkpoint, evaluated_epoch, selected_epoch, auc, logloss, validation_auc_at_stage, validation_logloss_at_stage, validation_macro_auc_at_stage, train_sample_count, validation_sample_count, test_sample_count, domain_share_p, loss_coefficient_w, effective_domain_coefficient_a, objective_name, objective_formula, gradient_partition_hash, access_formula, initial_parameter_state_hash, batch_order_hash, architecture_batch_order_hash, split_row_id_hash, physical_view_fingerprint, dataset_file_hash, checkpoint_hash, code_inventory_hash, runtime_fingerprint, status`.

Reference rows use `condition=single_domain_mtl_reference`, `shared_access=NA`, `provenance_type=legacy_sd_context`, `strict_source_identity=false`, and preserve their actual D=1 scope. Sample FULL rows use `source_stage=S3`, `provenance_type=fresh_sample_full_replacement`; S1 Sample FULL rows are never emitted.

### 9.3 Direct contrasts and interactions

`stage_S3_paired_results.csv` must contain the eight named contrasts, all identity/checkpoint keys, left/right run keys and metrics, `delta_auc`, `delta_logloss`, both selected epochs/checkpoint hashes, practical/strong/direction/concordance flags, `contrast_evidence_class`, `strict_source_identity`, and `pairing_valid`. Joint-to-joint rows fail closed unless every required source hash is present and equal. Legacy SD rows are never assigned strict identity.

`stage_S3_interactions.csv` must contain the four source condition keys, the two objective gaps, two access rescues, `interaction_auc`, `interaction_neg_logloss`, all checkpoint hashes, and a four-way pairing-valid flag. Every value must be independently reconstructible from the raw table.

The structural report must emit four-seed aggregates with mean, sample SD, range, direction counts, 4/4 indicators, worst seed, AUC/LogLoss concordance, per-stage persistence, target/non-target summaries, selected-epoch summaries, and reuse/gradient-access audits. It must not state scientific conclusions.

---

## 10. Prespecified scientific thresholds and interpretation rules

Seeds are the repeat unit. Thresholds are practical gates, not claims of statistical significance.

### 10.1 Negative-transfer eligibility

The original target-cell eligibility is inherited from accepted S2; S3 does not re-prove it from source-incomplete D=1 artifacts. For continuity, a condition is descriptively consistent with that stable practical target deficit if, for a preregistered target cell:

- four-seed mean `condition − SD` AUC `<= -0.003` (`<= -0.005` is strong);
- at least 3/4 seed deltas are negative; report 4/4 separately;
- mean LogLoss delta is positive and at least 3/4 seeds are worse;
- the four-seed mean AUC delta is negative at at least 3/5 common epochs.

### 10.2 Access rescue

An Equal access rescue requires:

- mean `R_E >= +0.003` (`>= +0.005` is strong);
- at least 3/4 seed values are positive;
- mean target-only-minus-full LogLoss is negative with at least 3/4 seeds improving;
- positive mean AUC rescue at at least 3/5 common epochs.

`equal_target_only_shared − SD > -0.003` may be reported as elimination relative to the historical S1/S2 reference only, with the legacy-source limitation attached. The strict S3 rescue estimand remains `R_E = equal_target_only_shared − equal_full`; if a claim depends on source-controlled evidence, `R_E` and the four-way interaction—not the D=1 comparison—must pass. If the historical deficit remains, report absolute and percentage attenuation; percentage alone cannot pass a gate.

### 10.3 Equal-specific interaction

Evidence that shared access explains the Equal-versus-Sample gap requires:

- mean interaction `I >= +0.003` and at least 3/4 positive seeds;
- positive interaction at at least 3/5 common epochs;
- concordant negative-LogLoss interaction;
- and either elimination of the blocked Equal-minus-Sample practical gap (`D_W_BLOCK > -0.003`) or a separately reported strong attenuation without claiming elimination.

If `R_E` and `R_S` are both positive but `I` is below `+0.003`, the result supports a general shared-access effect, not an Equal-specific explanation.

### 10.4 Persistent weighting effect

H-W gains support if `D_W_BLOCK <= -0.003`, at least 3/4 seeds favor Sample, LogLoss is concordant, and the blocked objective gap persists at at least 3/5 common epochs, while `I < +0.003`. This identifies persistence under the shared-access control; it does not by itself distinguish loss scale, Adam dynamics, private-path optimization, or calibration.

### 10.5 Convergence classification

An effect is **selected-only/stage-limited** if it passes its selected-checkpoint magnitude/direction gate but is supported at fewer than 3/5 common epochs and is below practical magnitude or reversed at epoch 49. Such an effect cannot support a persistent mechanism claim.

KuaiRand's two tasks must be reported independently. Passing both yields stronger within-domain task replication; passing only one is task-conditional evidence. MovieLens can strengthen or contradict transferability but cannot replace failure on the primary KuaiRand diagnosis object.

### 10.6 Evidence-strength boundary

- **STRONG** may be assigned only to a preregistered causal access effect that meets magnitude, seed, metric, and stage gates on both KuaiRand target tasks, with compatible MovieLens evidence or a clearly bounded MovieLens convergence exception.
- **MEDIUM** applies to one-task, one-setting, selected-stage, or partially discordant intervention evidence.
- **WEAK** applies to descriptive trends below gate, isolated seeds, only aggregate macro changes, or observational gradient norms.

No result may be described as proof of gradient conflict. The intervention can identify harmful cross-domain shared-update access, not the geometric reason those updates are harmful.

---

## 11. Acceptance criteria for reviewer

The reviewer must independently verify all of the following before formal execution:

1. The amended fresh matrix is exactly 24 unique M3OE joint keys: 16 target-only keys plus 8 Sample FULL replacements, covering only the two frozen settings and seeds 2022–2025. No new scientific condition is queued.
2. The historical 40-key audit explicitly yields 8 controlled S2 Equal FULL valid, 8 S1 Sample FULL rejected for missing model-source hash, and 24 S1 D=1 accepted only as legacy context. No audit may report 40/40 controlled reuse, and no S1/S2 file is modified.
3. Sample and Equal objective formulas and concrete coefficients exactly match accepted S1/S2 behavior; no sampler, repeated share factor, target-count normalization, or gradient-norm scaling exists.
4. The intervention partition is exhaustive and name-hash stable. `domain_expert.*`, `task_expert.*`, and all four architecture scalars are correctly treated as cross-domain shared; only slot/gate/tower parameters are private.
5. An independent autograd reachability test supports the partition on the real model for both schemas and rejects deliberately misclassified parameters.
6. Numerical tests prove `L_full=L_target+L_non`, shared gradients come only from `L_target`, private gradients come from `L_full`, architecture gradients come only from `L_target`, and removed gradients are not reapplied.
7. The formal Sample FULL replacement path reproduces the original Sample objective and direct FULL gradients, one-step parameters, optimizer state, batch fingerprint, and counters from the same initial state/batch; it is impossible to select Equal or any extra FULL condition.
8. Every strict four-condition block has a present, equal forward-model hash. Sample replacement/target-only and S2 Equal/target-only pairs match dataset, split/view, initialization, objective coefficients, batch orders, exposure, optimizer/step counts, metrics, and checkpoints; only prespecified objective/access fields may differ. Missing hashes fail closed.
9. The M3OE source/forward architecture and parameter counts remain unchanged; no new model, optimizer, schedule, sampler, momentum, clipping, projection, freezing, or hyperparameter path is present.
10. Validation-only selection, deferred test use, all seven checkpoint labels, and final/common-49 numerical identity are enforced.
11. The two-process queue has exactly 12 frozen launch groups: eight reciprocal same-setting/same-seed target-only pairs and four resource-only Sample FULL replacement groups. Outputs are isolated, parent updates atomic, maximum concurrency two, and resource grouping is never used as scientific pairing.
12. Failure recovery is append-only, has no mid-run resume, reruns a failed pair at most once from epoch 0, and cannot silently change concurrency or matrix.
13. Objective/access unit tests, corruption tests, tiny real-model forward/backward tests, artifact tests, and the representative two-process CUDA main-plus-architecture smoke pass without creating a formal result.
14. The implementation report lists every new/changed file, proves S1/S2 source and artifacts are unchanged, records the old and amended plan digests, and explains why Equal FULL was not retrained.
15. The structural collector enforces exactly 1,680 unique raw rows, 2,688 paired rows split into 1,344 strict joint and 1,344 legacy-context rows, 336 strict interaction rows, 56 run-inventory rows, all required checkpoint coverage, zero invalid strict pairs, and prediction-level AUC/LogLoss recomputation. No row may originate from an S1 Sample FULL joint artifact.
16. The reviewer verdict is `PASS`, or `PASS WITH FIXES` with all blocking fixes independently rechecked. On that outcome, the orchestrator may automatically invoke `luna_runner` without an additional user-approval stop.
17. The amendment recovery independently revalidates both archived wave-2 runs from immutable bytes, fixes only the empty `{}` inventory interpretation, proves their model-source hash, preserves the original archives, emits a hash-bound receipt, and either atomically promotes both target-only members or fails closed. Sample remains pending until its replacement FULL pair validates; no third target-only attempt is silently launched.
18. The amended runtime/recovery manifest reports 24 total formal fresh keys, 12 launch groups, the retained/revalidated count, remaining count, seeds, checkpoints, and append-only recovery states without counting parent-aborted or rejected attempts as scientific results.

Post-run, the reviewer must separately accept raw-artifact immutability, task completeness, metrics, hashes, direct deltas, interactions, cardinalities, and mechanical aggregates before Main analysis.

---

## 12. Conditions that would falsify or weaken each explanation

### H-I / shared-update interference is weakened or falsified if

- `R_E < +0.003`, is seed-unstable, worsens LogLoss, or is selected-only;
- Equal remains practically below the historical SD context by nearly the same magnitude under target-only access; this is descriptive unless corroborated by strict joint contrasts;
- target-only access harms Equal or improves Sample equally, leaving `I < +0.003`;
- apparent rescue arises only from a different checkpoint and is absent at matched epochs.

### H-IxW / Equal-specific shared interference is weakened or falsified if

- the interaction is below `+0.003`, negative, or fewer than 3/4 seeds are positive;
- the blocked Equal-minus-Sample gap remains stable and practical;
- rescue occurs under both objectives to the same degree;
- only aggregate macro-AUC improves while the target cells do not.

### H-W / persistent objective-weighting explanation is weakened if

- the Equal-minus-Sample gap disappears under target-only shared access;
- Equal is rescued relative to its strict FULL control, with a positive stable interaction; improvement relative to legacy SD is secondary context;
- the gap exists only at selected checkpoints and not at common/final epochs.

H-W gains only bounded support if the gap persists: the experiment does not separate coefficient scale from Adam conditioning, calibration, or private-path effects.

### H-C / convergence explanation is weakened if

- the deficit and intervention effect are directionally consistent at at least 3/5 common epochs and remain practical at epoch 49;
- selected and matched-epoch results give the same conclusion on both KuaiRand tasks.

H-C gains support if selected effects disappear/reverse at common/final epochs, especially in MovieLens, but that does not erase persistent KuaiRand evidence.

### Generalization is weakened if

- only one KuaiRand task passes;
- KuaiRand and MovieLens show opposite access interactions;
- MovieLens `rating>3` behaves like the preregistered `rating>4` counterexample;
- non-target cells suffer large redistribution costs, showing that the control is diagnostic rather than a viable mitigation.

Any result remains conditional to M3OE, the two audited settings, the fixed target domains, the existing optimizer and 50-epoch budget. Domain size remains a condition label, not an isolated causal variable.

---

## Design hand-off summary

S3 remains a fixed 2 × 2 factorial diagnosis on KuaiRand natural and MovieLens d2_large. The amended plan has 24 fresh runs (16 target-only plus 8 Sample FULL replacements), reuses 8 source-proven S2 Equal FULL artifacts, and retains 24 S1 D=1 artifacts only as legacy context. The eight source-unproven S1 Sample FULL joint artifacts are expressly excluded. The intervention keeps every example, batch, objective coefficient, parameter, optimizer step, and checkpoint rule, but routes target-component gradients to all parameters that can affect multiple domains while retaining the full objective for slot/gate/tower private parameters. The decisive evidence is the strict source-identical four-condition block: whether blocking stably rescues the preregistered Large cells and attenuates the Equal-minus-Sample gap across matched training stages.

The principal risks are interpretive: the intervention is target-specific, it changes shared-gradient composition and magnitude by design, and it cannot identify gradient geometry. Those limitations are controlled through the crossed objective factor, difference-in-differences estimand, unchanged objective coefficients, complete non-target reporting, and explicit prohibition on calling the result universal gradient conflict.
