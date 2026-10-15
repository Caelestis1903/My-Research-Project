# Stage S1 DESIGN SPECIFICATION

## Scope and decision inherited from Stage S0

This specification covers **Stage 1: Residual Negative Transfer Reproduction only**. Stage S0 formally concluded `STOP Momentum — uniform smoothing is sufficient`; therefore no Comp1, EMA, domain-conditioned momentum, Adam-memory sweep, controller, curriculum, schedule, or new Safe-MTL mechanism is permitted here. Equal-cell weighting is not a formal arm: prior Equal results may be cited as historical context, but all fresh Stage S1 runs use the target-aligned sample objective.

Stage S1 is a problem-existence test, not a mitigation experiment. Throughout the stage, a joint-model deficit is called a **residual degradation** or **residual negative-transfer candidate**. It must not be called sharing-induced negative transfer unless a later causal stage supports that interpretation.

---

## 1. Research question

After controlling the training objective to match the empirical sample distribution, does joint multi-domain/multi-task training still produce practically meaningful, seed-stable residual degradation relative to an independently trained, architecture-matched single-domain reference?

The question is deliberately decomposed into falsifiable subquestions:

1. Does any domain-task cell retain an AUC deficit of at least `0.003`, with concordant LogLoss degradation, in at least `3/4` seeds?
2. Does the result survive matched training-stage comparisons rather than appearing only because joint and reference models select different checkpoints?
3. Does a pattern recur in at least two of `M3OE`, `MMoE`, and `PLE`, rather than being M3OE-specific?
4. Does a pattern recur on both MovieLens and KuaiRand, rather than being a dataset-specific artifact?
5. Is the pattern better described by domain size, independently measured reference difficulty/baseline performance, physical domain, task, backbone, or training stage? Stage S1 reports these alternatives but does not causally choose among them.

### Competing explanations that Stage S1 must keep alive

- **H0: no research-scale residual NT.** Objective correction reduces remaining deficits to noise or sub-practical effects.
- **H-residual: reproducible residual candidate.** Practical, concordant deficits remain across seeds, backbones, and datasets.
- **H-model:** the effect exists only in one backbone, especially M3OE.
- **H-dataset/physical-domain:** the effect belongs to one dataset or one physical domain/task identity.
- **H-size:** the effect tracks training-set share or the L/M/S role.
- **H-difficulty:** the effect tracks independent-reference performance, not size. Small domain and difficult domain must never be treated as synonyms.
- **H-checkpoint/stage:** the effect is created by checkpoint flexibility, convergence speed, or late-training divergence.
- **H-reference-instability:** apparent degradation is comparable to the reference model's own seed variation or comes from mismatched data/preprocessing.
- **H-shift/capacity/optimization/sharing:** these remain possible mechanisms, but Stage S1 is not allowed to diagnose or mitigate them.

---

## 2. Hypothesis being tested

The affirmative hypothesis is:

> Under a sample-proportional objective and leakage-free matched evaluation, at least one interpretable domain-task pattern shows a concordant joint-versus-single-domain deficit of practical size on both datasets, in at least two backbones, with at least `3/4` seed consistency and support at matched training epochs.

The design is optimized to reject this hypothesis. A single negative cell, a single seed, one selected checkpoint, one backbone, or one dataset cannot establish it.

The null/stop hypothesis is:

> After objective alignment, remaining joint-versus-reference differences are mostly below `0.003` AUC, sign-unstable, metric-contradictory, checkpoint-dependent, backbone-specific, dataset-specific, or comparable to seed variability.

---

## 3. Variables changed

### 3.1 Formal factors

| Factor | Formal values |
|---|---|
| Dataset | `MovieLens-1M`, `KuaiRand-Pure` |
| Backbone | `M3OE`, `MMoE`, `PLE` |
| Training scope | `joint` (`D=3,T=2`), architecture-matched `single_domain` (`D=1,T=2`) |
| MovieLens size setting | `d0_large`, `d1_large`, `d2_large` |
| KuaiRand size setting | `natural` only |
| Seed | `2022`, `2023`, `2024`, `2025` |
| Evaluation stage | independently validation-selected checkpoint, fixed final epoch 49, common epochs `9,19,29,39,49` |

The single-domain model is the primary reference allowed by the repository Stage 1 definition (`STL_or_single-domain`). It retains the two tasks inside one physical domain and therefore isolates the cost/benefit of adding cross-domain joint training while keeping task sharing present. It must be named `single_domain_mtl_reference`, not `STL`, in code and reports. Stage S1 does not add per-cell STL runs; consequently it must not claim to isolate task-sharing effects.

### 3.2 MovieLens rotations

Physical domains remain the age-defined groups used by the repository:

- physical domain 0: ages `{1,18}`;
- physical domain 1: age `{25}`;
- physical domain 2: ages `{35,45,50,56}`.

Only training rows are subsampled. Validation and test rows stay fixed and untouched.

| setting | domain 0 | domain 1 | domain 2 |
|---|---:|---:|---:|
| `d0_large` | L = 150,000 | M = 34,500 | S = 3,000 |
| `d1_large` | S = 3,000 | L = 150,000 | M = 34,500 |
| `d2_large` | M = 34,500 | S = 3,000 | L = 150,000 |

For seed `s` and physical domain `d`, the selected training rows are the first role-size rows of the deterministic permutation generated from `RandomState(10*s+d)`, exactly matching the established S0 role-reversal rule. The joint and every corresponding single-domain reference must consume the same selected row IDs.

### 3.3 KuaiRand natural setting

Use physical domains `tab=0`, `tab=1`, and `tab=4`, mapped to `d0,d1,d2`, and tasks `is_click` and `long_view`. Preserve the established global row-order 80/10/10 split and split-first/filter-second reference construction. Expected train counts are:

- d0/tab0: `172,782` (`0.086127` share);
- d1/tab1: `1,743,898` (`0.869284` share);
- d2/tab4: `89,452` (`0.044589` share).

KuaiRand is not synthetically rotated in Stage S1. Calling its natural largest/smallest domains a rotation would confound physical identity with an artificial intervention. Its manifest setting is `natural`, and its L/M/S label is descriptive only (`d1=L`, `d0=M`, `d2=S`).

---

## 4. Variables held constant

Within each `dataset × setting × backbone × seed` block, joint and references must hold constant:

- raw dataset file identity and row order;
- global train/validation/test row IDs before filtering;
- feature schema, label definitions, encoder/scaler artifacts, and OOV rule;
- the exact domain-specific train/validation/test rows;
- task definitions and label arrays;
- embedding dimension `16`;
- hidden/tower dimensions specified below;
- batch size `40,960`, `num_workers=0`, `shuffle=True` for train, `drop_last=False`;
- fixed 50-epoch exposure (`0..49`), with no training termination by early stopping;
- Adam main optimizer `lr=0.01`, `betas=(0.9,0.999)`, `weight_decay=1e-5`, no LR scheduler;
- M3OE architecture optimizer `lr=0.001`, `betas=(0.9,0.999)`, `weight_decay=1e-5`, using the same established update timing in joint and reference runs;
- validation-only checkpoint rule and earliest-epoch tie break;
- test metric implementation and prediction clipping;
- seed setup for Python, NumPy, PyTorch, and CUDA, with deterministic cuDNN behavior;
- no EMA/Comp1/momentum manipulation, curriculum, balanced sampler, importance correction, adaptive weighting, temporal schedule, fallback, safety gate, or mitigation.

Exposure is matched per physical-domain example: every retained training row is presented for 50 epochs in both its joint run and its corresponding reference. Therefore domain `d` has exactly `50*n_d` example presentations in each scope (before counting the two task labels). Total optimizer-step counts are not forced equal between a three-domain joint dataset and a one-domain reference because doing so would change per-example exposure; the actual counts must be logged and cannot be interpreted as an experimentally manipulated factor.

Different backbone parameter counts are not forced equal in Stage S1. Every scientific transfer contrast is within backbone. Trainable parameter count, shared/private count where definable, and forward FLOPs estimate must be logged so cross-backbone differences are not misreported as controlled capacity comparisons.

---

## 5. Mathematical and algorithmic definition

### 5.1 Target-aligned objective

For example `i`, task `t`, prediction `p_it`, and binary label `y_it`, define binary cross-entropy `b_it`. Every formal joint run minimizes the direct per-example task mean

\[
L_{joint}=\frac{1}{BT}\sum_{i=1}^{B}\sum_{t=1}^{T} b_{it}.
\]

Equivalently over the full training distribution,

\[
L_{joint}=\frac{1}{T}\sum_t\sum_d p_d\,\bar b_{d,t},
\]

where `p_d=n_d/sum_d n_d`. This is the normalized target-aligned/sample-proportional objective. It is the same target utility as S0's coefficients `D*p_d`, but removes the irrelevant global factor `D` so joint and `D=1` reference losses have matched scale. The coder must unit-test full-epoch equivalence between the direct sample mean and the `p_d` cell-mean expression.

For the single-domain two-task reference,

\[
L_{SD}(d)=\frac{1}{T}\sum_t\bar b_{d,t}.
\]

No equal-cell objective is included in the formal matrix.

### 5.2 Backbones

1. **M3OE**: current `MDMTRec`, `fcn_dims=[512,256,256,64]`, `expert_num=4`, `exp_d=1`, `exp_t=1`, `bal_d=1`, `bal_t=1`. Joint uses `D=3,T=2`; its reference uses `D=1,T=2`.
2. **MMoE**: repository P7 FuxiCTR-compatible layer math, common embedding, `4` experts, expert hidden units `[256,64]`, gate hidden units `[64]`, and one `[64 → LayerNorm/ReLU → 1]` tower per `(d,t)` cell.
3. **PLE**: repository P7 FuxiCTR-compatible CGC layer, `4` shared experts, `1` task-specific expert per output cell, expert hidden units `[256,64]`, gate hidden units `[64]`, and the same tower definition.

The P7 ShareBottom adaptation is excluded because it is not a faithful implementation of the upstream single-bottom layer and has a known parameter-count/fidelity qualification. Adding it would weaken rather than strengthen this falsification test.

MMoE/PLE must not carry a trainable dummy architecture parameter. They have only the main Adam optimizer. M3OE's architecture parameters and second optimizer are part of its fixed backbone, not a Stage S1 method.

For M3OE, perform exactly one architecture-optimizer update after each main-training epoch, using the next deterministic batch from a separately tracked iterator over the same training loader, matching the established S0 timing. Record that batch fingerprint and restore/verify RNG state so the extra observation path cannot perturb the next epoch's main batch order. Joint and single-domain M3OE runs use the identical timing rule; MMoE/PLE record zero architecture steps.

### 5.3 Matched initialization

For each `dataset × setting × backbone × seed`, build one deterministic semantic initialization bundle before training:

- embeddings and shape-compatible shared/expert tensors are identical between joint and all three references;
- a reference's local task `t` gate/tower/task-specific tensors map from joint global cell index `t*D+d`;
- M3OE reference domain-specific tensors map from physical joint domain `d`; task-specific and shared tensors map by task/name;
- the physical-domain embedding row used by the joint model maps to the reference model's remapped local-domain row 0;
- tensors that have no shape/semantic counterpart use a deterministic name-derived sub-seed, recorded in the manifest.

The mapping table, common-tensor hashes, unmapped tensor list, and parameter counts are required artifacts. A silent same-integer-seed claim without this semantic mapping is not sufficient pairing because model construction order and output count differ.

### 5.4 Checkpoint protocol and leakage barrier

- Train all runs for exactly 50 epochs. Patience 10 may be computed as a `shadow_stop_epoch` diagnostic but never stops training.
- Joint checkpoint: maximize unweighted validation macro-AUC over all six cells; ties choose the earliest epoch.
- Single-domain checkpoint: maximize unweighted validation macro-AUC over its two tasks; ties choose the earliest epoch.
- Save the selected state, epoch 49 state, and common-epoch states `9,19,29,39,49`.
- Do not evaluate test during training. After all validation selections are frozen and written, reload only the selected/final/common states and compute test metrics. Test values must never select a checkpoint, hyperparameter, preprocessing rule, retry, or included run.

### 5.5 Data preprocessing and pairing

Both datasets must be split before any fit operation. Fit categorical vocabularies and numeric scaling on the global base training split only; reserve index 0 for validation/test OOV values. For MovieLens, fit once on the unsampled base training split so rotations differ only in selected training rows, not feature representation. Persist encoder hashes and split row-ID hashes.

MovieLens labels are `click = 1[rating>3]` and `like = 1[rating>4]`. KuaiRand labels are the stored binary `is_click` and `long_view`. No label-derived statistic from validation or test may enter preprocessing or training.

For every reference, first construct the global split/subsample, then filter physical domain `d`, copy the arrays, and remap only the routing domain ID to local 0. Keep `original_physical_domain=d` in all metadata. Joint-domain row fingerprints must exactly equal the corresponding reference fingerprints after accounting for this deterministic domain-ID remap.

### 5.6 Transfer metrics

For dataset `k`, setting `r`, backbone `m`, seed `s`, domain `d`, task `t`:

\[
\Delta^{AUC}_{k,r,m,s,d,t}=AUC^{joint}_{k,r,m,s,d,t}-AUC^{SD}_{k,r,m,s,d,t}.
\]

\[
G^{LL}_{k,r,m,s,d,t}=LogLoss^{joint}_{k,r,m,s,d,t}-LogLoss^{SD}_{k,r,m,s,d,t}.
\]

Thus negative `Delta_AUC` and positive `G_LL` both indicate joint degradation.

Indicators:

- raw AUC candidate: `Delta_AUC < 0`;
- raw LogLoss candidate: `G_LL > 0`;
- practical AUC candidate: `Delta_AUC <= -0.003`;
- strong AUC candidate: `Delta_AUC <= -0.005`;
- practical LogLoss degradation: `G_LL >= 0.005`;
- concordant practical candidate: `Delta_AUC <= -0.003 AND G_LL > 0`;
- concordant strong candidate: `Delta_AUC <= -0.005 AND G_LL >= 0.005`.

A cell is **seed-stable practical** only if its four-seed mean `Delta_AUC <= -0.003`, at least `3/4` seed deltas are negative, mean `G_LL > 0`, at least `3/4` seed LogLoss gaps are positive, and the mean matched-stage AUC delta is negative at at least `3/5` common epochs. These thresholds are preregistered and may not be lowered after results are seen.

### 5.7 Aggregate metrics

Report, without substituting them for cell-level transfer:

- unweighted cell macro AUC and macro LogLoss;
- target-weighted cell AUC and LogLoss using training shares `p_d`;
- per-task pooled AUC across domains and their macro average;
- pooled LogLoss across all examples, reported separately from macro LogLoss;
- worst-domain AUC (minimum two-task domain macro);
- worst-cell AUC and maximum cell LogLoss;
- raw, practical, strong, concordant-practical, and concordant-strong NT rates;
- assembled single-domain-reference ensemble versions of the same summaries.

Do not pool different task labels into one AUC. Cross-domain pooled AUC for the assembled independent-reference ensemble is calibration-sensitive and is descriptive only; it is not a GO/STOP gate.

Seeds are the repeat unit. MovieLens rotations are fixed crossed conditions, not extra independent replicates; domains and tasks are also not replicates. Report paired seed means, sample SD, range, `3/4` sign counts, and descriptive paired intervals. Do not manufacture significance by treating cells or rotations as independent `n`.

---

## 6. Exact affected modules/files if identifiable

Coder should isolate Stage S1 rather than mutate closed-stage runners:

- `experiments/stage_S1_protocol.py`: immutable factors, paths, matrix, thresholds, and validation contract.
- `experiments/stage_S1_data.py`: split-first preprocessing, MovieLens rotations, KuaiRand natural bundle, fingerprints, and reference filtering.
- `experiments/stage_S1_models.py`: audited constructors/adapters for M3OE, MMoE, PLE and semantic initialization mapping. Reuse the layer math from `m3oe/MDMTRec/models/multi_domain/model.py` and `experiments/history_exp/first/p7_models.py`; do not silently rewrite it.
- `experiments/stage_S1_run.py`: one joint or single-domain formal run, fixed-budget training, validation selection, deferred test evaluation, and raw artifacts.
- `experiments/stage_S1_orchestrator.py`: resume-safe 192-run queue, preflight, fail-fast behavior, and block validation.
- `experiments/stage_S1_report.py`: structural collection and raw paired tables only; no scientific conclusion.
- `experiments/tests/test_stage_S1_*.py`: matrix, objective, data/reference pairing, initialization, checkpoint leakage, metrics, and resume tests.
- outputs under `experiments/stage_S1_raw_results/`, `experiments/stage_S1_*.csv/json`, and runner report `experiments/reports/stage_S1_run_report.md`.

Closed S0 artifacts and all earlier E-stage raw results are read-only. Existing E3/E4 oracle artifacts are not formal Stage S1 references because they are M3OE-only and do not satisfy the new cross-backbone preprocessing, initialization, objective-scale, and deferred-test contract.

---

## 7. Implementation requirements

1. Formal matrix keys must be explicit and collision-free: joint key `joint|dataset|setting|backbone|seed`; reference key `single_domain|dataset|setting|backbone|physical_domain|seed`.
2. Expected formal tasks:
   - MovieLens joint: `3 settings × 3 backbones × 4 seeds = 36`;
   - MovieLens references: `3 settings × 3 backbones × 4 seeds × 3 domains = 108`;
   - KuaiRand joint: `1 × 3 × 4 = 12`;
   - KuaiRand references: `1 × 3 × 4 × 3 = 36`;
   - **total = 192 runs**, yielding `48 × 6 = 288` paired joint cell observations at each reported checkpoint.
3. The orchestrator must build/validate all references before launching the corresponding joint block, so no joint result is reported with missing reference coverage.
4. Resume only artifacts that pass the complete validator; never overwrite a valid run. Archive failed attempts append-only.
5. Formal device is `cuda:0`; runtime preflight must verify the requested Python environment, PyTorch/CUDA, both dataset files, writable output paths, and absence of another Stage S1 orchestrator/training process.
6. Every run must record exact examples processed, optimizer steps, architecture steps, batches/epoch, epoch count, RNG hashes, batch-order fingerprints, checkpoint hashes, wall time, device identity, and non-finite inventory.
7. Objective code must be identical between joint/reference and across backbones. M3OE may differ only in its inherent architecture optimizer path.
8. Validation/test metric code must be one shared implementation. Clip probabilities only for LogLoss with a fixed epsilon; AUC uses unclipped ranking-equivalent predictions.
9. Training and validation histories must remain available even if the selected epoch is early. No result can be accepted from summary-only output.
10. Report parameter counts and the semantic initialization audit for every model; do not interpret backbone ranking as a controlled capacity comparison.
11. Do not add gradient-cosine, capacity sweeps, shift-controlled splits, predictors, fallback, or mitigation. Those belong to later decisions and are outside Stage S1.

---

## 8. Ablations and comparisons required

There are no method arms. Required comparisons/sensitivities are:

1. **Primary paired contrast:** joint sample-objective model versus architecture-matched single-domain two-task reference at independently validation-selected checkpoints.
2. **Checkpoint sensitivity:** the same contrast at fixed final epoch 49.
3. **Training-stage sensitivity:** paired contrasts at common epochs `9,19,29,39,49`.
4. **Backbone replication:** M3OE, MMoE, PLE analyzed separately and by recurrent pattern; never average away a backbone reversal.
5. **MovieLens size-role rotation:** analyze by physical domain, task, and L/M/S role. This distinguishes size role from fixed physical identity.
6. **Dataset replication:** MovieLens versus KuaiRand; do not treat three MovieLens rotations as three datasets.
7. **Difficulty versus size audit:** plot/tabulate transfer against train count/share and separately against single-domain reference AUC/LogLoss. Include label prevalence/entropy as descriptive covariates. Do not infer that low AUC means optimization difficulty or that small means hard.
8. **Selection-flexibility audit:** report selected epochs and validation curves for joint and references. A candidate that disappears at common epochs is classified checkpoint/stage-dependent.
9. **Historical context only:** S0 and E4 sample/equal findings may be shown in a clearly separated table, never pooled into fresh Stage S1 estimates.

---

## 9. Expected logging requirements

### Per-run manifest

- stage, run type, formal key/status, dataset/file hash, setting, backbone, seed;
- physical-to-local domain mapping, physical identity, size role, task names;
- all split and per-domain row fingerprints, selected-row IDs hash, encoder/scaler/OOV hashes;
- train/validation/test counts, shares, positive counts/rates, label entropy;
- full model constructor arguments, parameter names/shapes/counts, shared/private counts if defined;
- semantic initialization source/mapping, common tensor hashes, unmapped tensor sub-seeds;
- exact loss formula and coefficients, optimizer groups/hyperparameters, scheduler=`None`;
- batch size, batch fingerprints, examples processed, main/architecture optimizer steps;
- selected epoch/score, shadow-stop epoch, final/common checkpoints and hashes;
- evidence that the test pass occurred only after selection was frozen;
- runtime environment, device, code/protocol fingerprints, wall time, peak memory, exception/non-finite inventory.

### Per epoch

- train objective and sample count;
- validation AUC and LogLoss for every domain-task cell;
- validation cell macro, target-weighted summary, worst domain/cell;
- learning rate, optimizer-step counts, selected-checkpoint marker after training;
- no test metric in the training-time epoch log.

### Deferred test output for selected/final/common states

For every `seed × dataset × setting × backbone × domain × task`:

- joint/reference AUC and LogLoss;
- `Delta_AUC`, `G_LL`, all NT indicators;
- joint/reference selected epochs and checkpoint type;
- train/validation/test cell counts, domain share/role, physical identity, task identity;
- reference AUC/LogLoss difficulty fields and baseline seed variation;
- model parameter count and training-stage identifier.

### Aggregate raw outputs

- complete run inventory and block validator;
- paired cell table;
- per-seed/per-setting/per-backbone summaries;
- per-physical-domain, per-task, per-role, per-difficulty summaries;
- checkpoint/common-epoch trajectory table;
- macro/target-weighted/pooled/worst-domain/worst-cell metrics;
- NT rates and stable-cell inventory;
- contradictions table listing sign flips across seed, backbone, dataset, rotation, metric, and stage.

The runner report records structure and raw aggregation only. Scientific interpretation and GO/STOP belong exclusively to the main agent.

---

## 10. Acceptance criteria for reviewer

Reviewer must reject RUN if any blocking criterion fails:

1. The matrix is exactly 192 unique tasks with no missing/duplicate key.
2. M3OE/MMoE/PLE constructors match the specified layer math; ShareBottom and prohibited mechanisms are absent.
3. Sample objective unit tests show direct per-example BCE equals the normalized `p_d` cell expression over a full dataset, and the same code path serves every backbone/run type.
4. MovieLens rotations exactly reproduce `150000/34500/3000` and the S0 deterministic selected-row rule.
5. KuaiRand counts, domain/task mapping, and split-first/filter-second behavior match the frozen contract.
6. Preprocessing fit uses training rows only; validation/test OOV handling is deterministic; no validation/test label statistic enters training.
7. Every joint domain view and its reference have matching row/label fingerprints after the declared local-domain remap.
8. Semantic initialization mappings are complete for every shape-compatible tensor and verified by hashes; unmapped tensors are explicit and deterministic.
9. Main optimizer, M3OE architecture optimizer, fixed 50 epochs, batching, and no-scheduler contract are exact. MMoE/PLE have no trained dummy parameter.
10. No early stop truncates training. Selected/final/common checkpoints are all present and finite.
11. Checkpoint selection uses validation macro-AUC only with earliest tie break; test evaluation is deferred until selection is frozen and cannot influence retries.
12. AUC/LogLoss/cell indexing are verified on synthetic fixtures; every evaluation cell contains both classes or preflight fails before formal RUN.
13. Reports distinguish macro, target-weighted, per-task pooled, and calibration-sensitive assembled-reference pooled AUC.
14. Seed is the repeat unit; report code does not treat rotations/cells as independent samples.
15. Instrumentation/logging has an ON/OFF equivalence smoke test where applicable and does not change parameters or metrics.
16. Resume validation cannot accept partial/corrupt artifacts and cannot overwrite valid runs.
17. The report language uses residual candidate/degradation and contains no causal sharing claim or Safe-MTL proposal.

A reviewer verdict may be `PASS` or `PASS WITH FIXES` only after all blocking issues are corrected. Syntax success or a smoke run alone is not scientific-protocol acceptance.

---

## 11. Falsification, GO/STOP/INCONCLUSIVE gates, and run-stop conditions

### 11.1 Stage 2 GO gate

Main Agent may issue `GO — residual NT is a primary research problem` only if all hold under the primary single-domain reference:

1. Both MovieLens and KuaiRand contain at least one seed-stable practical cell.
2. In each dataset, the qualifying pattern occurs in at least two of the three backbones.
3. Every qualifying pattern has mean `Delta_AUC <= -0.003`, at least `3/4` negative AUC seed deltas, mean `G_LL > 0`, and at least `3/4` positive LogLoss gaps.
4. The AUC deficit remains negative at at least `3/5` matched common epochs; it is not solely an independently selected-checkpoint artifact.
5. The cross-dataset evidence shares at least one prespecified non-model attribute (size tier, independently measured difficulty tier, or a genuinely comparable task property). Equal numeric domain IDs across datasets are not a shared attribute.
6. No stable opposite result in another backbone/dataset reduces the claimed pattern to model- or dataset-specific behavior. Contradictions must be reported, not averaged away.

Effects with `|Delta_AUC| >= 0.005` and practical LogLoss concordance are strong signals, but magnitude alone cannot waive replication requirements.

### 11.2 STOP gate

Issue `STOP Residual Negative Transfer as a primary research problem` if the integrity-complete matrix shows any of the following overall outcomes:

- no cell satisfies the seed-stable practical definition;
- most deficits are `0.000x–0.001x`, or practical candidates frequently flip sign across seeds;
- candidates are AUC-only and LogLoss consistently improves, or vice versa;
- candidates vanish in matched common-epoch comparisons;
- a pattern appears in only one backbone;
- a pattern disappears on the second dataset;
- apparent effects are of the same order as reference seed variation and lack `3/4` paired sign consistency;
- size-role conclusions reverse under physical-domain rotation without any stable physical-domain/task alternative.

Thresholds must not be relaxed to preserve the research direction.

### 11.3 INCONCLUSIVE gate

`INCONCLUSIVE` is allowed only for protocol/integrity failure: incomplete matrix, invalid/missing reference, data or initialization mismatch, test leakage, non-finite metrics, missing class in an evaluation cell, incompatible checkpoint support, or irreproducible runtime failure. Borderline scientific effects below the preregistered threshold are a STOP result, not an integrity-based INCONCLUSIVE escape.

### 11.4 Conditions that falsify the affirmative hypothesis

The hypothesis is falsified by sub-practical magnitude, seed instability, metric discordance, stage dependence, M3OE-only behavior, dataset-only behavior, or a result explained entirely by a fixed physical domain while failing its size-role rotations. A finding that small cells degrade does not support H-size if independently hard large cells show the same pattern or if the same physical cell changes sign when its size role changes.

### 11.5 Runner fail-fast conditions

Before launch, stop if CUDA/environment, dataset identity/counts, class coverage, preprocessing fingerprints, matrix cardinality, or output locking fails. During RUN, stop the queue on the first non-zero return code, non-finite parameter/optimizer state/metric, incomplete checkpoint set, fingerprint mismatch, test-before-selection evidence, or artifact validator failure. Preserve the failed attempt, do not overwrite prior valid artifacts, and resume only after an approved correction.

---

## Required main-analysis questions

The final Stage S1 analysis must answer, with `STRONG / MEDIUM / WEAK` evidence labels:

1. How much residual candidate remains after sample correction, in AUC and LogLoss?
2. Which physical domain/task cells are stable, and which are counterexamples?
3. Is any pattern backbone-independent, or M3OE-specific?
4. Is any pattern dataset-independent?
5. Does domain size plausibly explain more than independent-reference difficulty/baseline performance, or vice versa?
6. Is there a small-domain improvement / large-domain decline trade-off, its reverse, or neither?
7. Does the conclusion change across selected, final, and common training stages?
8. Is the evidence sufficient for the preregistered GO gate?

This specification intentionally ends at the Stage S1 decision gate. **It does not design Stage 2.** If and only if the main agent returns GO, a new designer call must design the next stage separately.
