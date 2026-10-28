# REVIEW REPORT — MDMT SOTA 2526

**Latest verdict: PASS — FIX2 final re-review (2026-10-28).** All previously identified F1–F9 implementation findings are closed in the reviewed state. The initial FAIL and FIX1 FAIL history below remain intact. This PASS covers the specified implementation and synthetic CPU acceptance checks; **real RUN still requires the user's explicit approval and available resources**. It does not validate HyperGate's unavailable paper configuration or establish benchmark performance.

## 1. Verdict

**FAIL — initial independent review, 2026-10-28.** Do not enter RUN. The existing synthetic suite passes, but it does not detect several material departures from the design and official training lifecycle. This initial verdict and findings must remain in the record when fixes are re-reviewed.

No research training, E evaluation, dependency installation or checksum calculation was performed by the reviewer. No implementation file was changed; this review owns only this report. Required fixes return to the respective coders.

## 2. Requirement verification

| Requirement | Independent finding |
|---|---|
| Fixed official sources and HyperGate gitlink | PASS: actual RecStudio HEAD is `f08cca607e998e8be4c5e915a393346baffa71f6`; AutoIFS HEAD is `65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e`; HyperGate parent gitlink matches RecStudio. |
| Preserve HyperGate gates, anchors, detach, BCE/auxiliary equations | Forward equations agree with the inspected official `hypergate.py`; three gates remain present. **FAIL lifecycle initialization and cross-process reproducibility: F1.** |
| Preserve AutoIFS paths, LoRA, search/rewind/final gate/reuse | Core stays in the official external class. Bias correction uses affine output widths. Search optimizers are disjoint and exhaustive; rewind is before the final temperature increase at start of zero-based epoch 9; final selector and final backbone embedding transplant into rewind, then gate parameters freeze and fresh base Adam is built. **FAIL pruning accounting: F4.** |
| Preserve fixed-fusion M3oE comparator | Architecture/init reuse is present and AutoML remains disabled. **FAIL objective scale: F2.** |
| Locked rows, T0-only fit, T0∪F optimizer permission | Current implementation and preparation receipts support the intended mapping and permissions. Reviewer inspected preparation/role code and root's complete direct-value/support receipts; did not repeat raw reconstruction. No historical data file was changed. |
| Masks, singleton/missing-domain/empty input, gradients | Existing model assertions and CPU suite pass. One sparse-vocabulary boundary is off by one (F9). |
| Checkpoint/RNG/Adam state | Checkpoint captures model, optimizer(s), RNG, stage, temperature/ticket and rewind/selector states. Existing generic Adam/dropout next-step replay passes. **FAIL binding evaluated checkpoint to global formal freeze: F3.** |
| V-only selection, Q development, global E freeze | Training loops call V only. **FAIL E gate validates matrix keys/status but not actual formal run/checkpoint identity: F3.** |
| Macro/natural/pooled metrics and seed SD | Existing metric formulas, explicit support counts and sample SD are consistent. **FAIL report identity/paper-table delivery: F5.** |
| Runtime, environment, failure/provenance records | Some training time/parameter/peak-memory fields exist. Required inference cost and complete failure/implementation/environment provenance are missing: F6. |
| Five working configurable CLI entries and clean reproduction | Smoke/train work on their stated inputs; audit has actual source inventory. Mandatory filtering/output options and clean source setup/development budget have gaps: F7/F8. |
| RUN approval and honest status | PASS: no real training was launched. HyperGate `UNVERIFIED_PROPOSAL`, external AutoIFS license absence and absence of research performance remain disclosed. |

## 3. Bugs and deviations

### F1 — MAJOR: HyperGate skips official initialization, and its seed is process-order dependent

`hypergate.py:57–64` constructs the core and calls `_init_model()` but never performs the official post-construction initialization. In the pinned `recstudio/model/basemodel/recommender.py`, `fit()` calls `_init_model(train_data)` and then `_init_parameter()` (around lines 126–128). `_init_parameter()` (356–372) applies the configured initializer to the model's children. The pinned base YAML declares `train.init_method=xavier_normal`; `recstudio/model/init.py:xavier_normal_initialization` initializes Linear/Embedding weights, zeros Linear biases and padding rows, and initializes LayerNorm. The shim only implements `_init_model`, so default PyTorch initial values remain. Independent inspection found **28 nonzero Linear bias tensors**, where the official initializer requires zero. The current parity test shares this uninitialized shim core and therefore cannot establish lifecycle parity.

The field input order is canonicalized only after official modules have already been built from a Python set. Independent fresh processes with the same Torch seed 2022 but `PYTHONHASHSEED=1` and `2` produced different logits, e.g. the first row was `[-0.02740890,0.01509777]` versus `[-0.01387648,0.02062963]`. This is not a numerical-tolerance effect.

**Minimum fix/verification:** load and apply the pinned official initializer at the corresponding lifecycle point; record initialization provenance. Make field/module initialization ordering deterministic, or establish and enforce a fixed Python process hash-seed launch contract. Test against an independently initialized official reference, not solely a core constructed by the same shim; check zero biases/padding and same-seed fresh-process predictions. Preserve the official initializer's rules rather than inventing a different initialization.

### F2 — MAJOR: M3OE_FIXED1 doubles the preserved natural BCE objective

`models.py:74–82` sums two per-task means. The reused native trainer `experiments/mdmt_basp_v1/prefix.py:73` uses `binary_cross_entropy_with_logits(logits,y)` with the default mean over all valid requests. For fully observed two-task labels the new objective is exactly twice the preserved comparator objective, changing its scale relative to Adam epsilon and weight decay. Independent synthetic evidence: actual `1.6020426750`, native mean `0.8010213971`, ratio `1.99999988`.

**Minimum fix/verification:** use sum of all valid per-label BCE divided by the total valid-label count for M3OE_FIXED1, including uneven task masks. Leave HyperGate/AutoIFS official task-sum objectives intact. Compare both loss and gradients to native BCE on fully observed input and a hand-computed unequal-mask example.

### F3 — MAJOR: E authorization is not bound to the actual frozen formal checkpoint

`data.py:268–280` checks 30 keys and status only. `cli.py:233–261` checks current run/config/data but does not require `phase=formal`, compare the current run path/checkpoint path with the frozen entry, or compare its run identity with `frozen.identity`. Thus the development seed-2022/2023 run with the same method/dataset can be evaluated on E using another formal run's complete freeze. An independent call to `Data.authorize_final()` accepted a matrix that contains no current checkpoint or phase information. `build_freeze()` also accepts completion JSONs without checking that their registered checkpoint files exist and match the relevant run metadata.

**Minimum fix/verification:** before E labels are read, bind the actual formal run, selected/final view, checkpoint file and embedded run/config/source/data identity to the exact frozen entry/global identity. Reject a development run, another run directory/checkpoint, or changed source/data/config. Freeze must register actual available checkpoints, not completion flags alone. Use direct metadata/state comparisons and existing Git identities; add no hashing, token service or security framework. Add a positive exact-formal fixture and negative development/mismatched-path/mismatched-identity fixtures.

### F4 — MAJOR: AutoIFS pruning fraction is inverted

`training.py:382–389` counts `gate_logit>0` as `active` and reports its complement as `pruned`. In the official `modules/autoifs.py:199–204`, a positive logit creates a mask that is **subtracted**, so it denotes a removed path. All resulting pruning summaries are reversed.

**Minimum fix/verification:** count positives as removed/pruned and nonpositives as retained; name both unambiguously and keep the denominator. A fixed all-positive selector must report 100% removed, all-negative 0%, and zeros must follow the strict `>0` rule.

### F5 — MAJOR: paper comparison and result provenance are incomplete

`report.py:200–205` writes blank AUC/LL for every row even when actual metric files were read. It never creates the promised usable numerical comparison. `experiment_manifest.json` receives status/path summaries only (`statuses` around lines 67–72, export 194–199), omitting the requested data/source/config/environment/checkpoint identities. Pairing at 168–182 uses method/dataset/seed/role/view/cell but does not verify common protocol, data/split, source or config; the report fixture even omits these identities and therefore cannot catch incompatible records being compared.

**Minimum fix/verification:** populate actual formal selected-E numerical rows and seed-mean/sample-SD comparison from the parsed metrics, keeping missing values blank and selected/final/roles separate. Preserve a pending custom-method row if required by the task. Include the requested run identity/config/environment/checkpoint references in the manifest, and reject or explicitly segregate incompatible records before paired deltas. Add a completed synthetic formal-result fixture that asserts actual Markdown numbers and provenance, plus one incompatible-pair rejection fixture.

### F6 — MAJOR: required runtime/failure/implementation traceability is not implemented

`train_one()` only writes `run.json` after every training/evaluation/save operation succeeds (`training.py:398–467`). A reachable missing-source/data/CUDA error or in-training exception leaves no machine-readable FAILED run record; partial epoch logs have no final failure status. No inference timing implementation exists in training/evaluation/report code, and `runtime_cost.csv` has no inference P50/P95 or measurement metadata. Per-run environment is absent, and source identity records the old reference commit plus official upstream commits, not the actual newly implemented package version/snapshot. These are explicit requested deliverables, not optional instrumentation.

**Minimum fix/verification:** record a minimal started/failure receipt and elapsed/error/stage information while preserving the raised exception and partial files. Capture actual runtime environment and the executed implementation Git version or direct source snapshot identity without hashes. Add bounded fixed-batch warmup/timed inference with CPU/GPU synchronization as appropriate, and record batch/device/repetition metadata with P50/P95. Ensure report/export retains these fields. Exercise a synthetic intentional failure and a synthetic timing record; no real-data training is needed.

### F7 — MAJOR: mandatory CLI selection/output parameters are incomplete

`make_parser()` gives audit/report no model/dataset/seed filters; evaluate infers identities from `--run-dir` but cannot choose an output path and offers no explicit identity assertions. This fails the task's stated five-entry configuration requirement, despite smoke/train supporting their flags.

**Minimum fix/verification:** add small meaningful audit/report filters and evaluate output configuration; model/dataset/seed may be inferred from run-dir with optional equality assertions or documented equivalent selection, but must never be ignored. Ensure a custom evaluation output is discoverable by reporting (a recorded output reference or explicit report input is sufficient). Add narrow parser/behavior checks; no generic CLI framework is required.

### F8 — MAJOR: clean reproduction commands disagree with the config and development budget

The guide creates RecStudio at `/tmp/mdmt_sota_sources_2526/HyperGate/RecStudio` (lines 10–13) but asserts that default config paths match; the default uses `/tmp/mdmt_sota_sources_2526/RecStudio`. A clean user cannot run the subsequent commands. Line 55 says to continue development for all three methods and repeats seed 2022, which both exceeds the specified **8 SOTA-only development trajectories** and hits overwrite rejection for completed trials.

**Minimum fix/verification:** align the checkout path or show the exact config override; document eight development trajectories with seed-2022 trials reused, then only missing seed-2023 SOTA trajectories, and thirty fresh formal trajectories. Keep M3OE_FIXED1 formal-only for this prescribed budget. Validate path construction and command/matrix consistency without launching training.

### F9 — MINOR: sparse vocabulary validators accept one unsupported ID

`models.py`, `autoifs.py:53` and `hypergate.py:100` accept `value<=vocab_size`, whereas the stored strict vocab size is the count including OOV and legal IDs stop at `vocab_size-1`. Adapters allocate an extra unused row, hiding that boundary error. Current real prepared data stays within the legal range, so this is not the reason for the FAIL verdict.

**Minimum fix/verification:** reject ID equal to the declared vocabulary size, preserve OOV 0, and clarify any padding-only extra allocation. Test max-valid and first-invalid IDs.

## 4. Independent checks performed

- Actual pinned Git HEADs and HyperGate gitlink read directly; no digest computed.
- Read design, coder reports, current model/source/data/training/evaluation/report/CLI code, and relevant official HyperGate base lifecycle/initializer and AutoIFS model/trainer.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_sota_baselines_2526/tests -q`: **10 passed in 1.59s**. This result is retained alongside the uncovered errors, not treated as proof of complete correctness.
- CPU synthetic probes confirmed M3OE loss ratio ≈2, weak freeze authorization and 28 nonzero HyperGate Linear biases. Separate synthetic processes confirmed same-Torch-seed initialization differs across Python set seeds. No empirical AUC/LogLoss was measured.

## 5. Corrections made

None to implementation. The parent requested coder-owned fixes and a follow-up independent re-review. Fixes must include targeted regression evidence for the above concrete failures, and reports should distinguish the initial FAIL from the final verdict.

## 6. Remaining risks and permission boundary

Even after these fixes, HyperGate configuration remains an explicitly unverified official-core adaptation, not a numerical as-published reproduction. AutoIFS uses the documented CSV/input/selection adaptations and an external source with no LICENSE file. Historical E is already exposed and remains historical reassessment. Actual training convergence, GPU execution, runtime and research performance remain unmeasured.

**Experiments are not safe to release into RUN yet:** blocking implementation/traceability issues above require correction and re-review, followed by the user's explicit approval. Do not use negative model performance as grounds to alter either algorithm.

## 7. FIX1 independent re-review

The reviewer independently ran the revised complete CPU suite: **19 passed in 4.06s**. Inspected actual fixes rather than treating that result as sufficient. No real training, E evaluation, hashing or dependency installation was performed.

F1 is corrected in the model: `_OrderedFieldSet` stabilizes construction, the shim loads the official initializer and reproduces the official per-child apply order, and the adapter calls it immediately after `_init_model`. The revised independent-reference and cross-process tests exercise the missed lifecycle. F2 is corrected to the natural valid-label mean and checked against loss and gradients, including unequal masks. F4 now counts strict-positive selector logits as removed. F7 has effective audit/report filters, evaluation identity assertions and indexed custom output. F8 source commands match configured paths; the guide explicitly distinguishes 8 SOTA-only development trajectories from 30 fresh formal trajectories and documents the existing environment/fresh-environment limitation. F9 now rejects the first invalid categorical ID. Fixed source inventory and guide sparse patterns agree; RecStudio's initializer is present in the separately cloned RecStudio tree.

F3/F5/F6 are substantially implemented but remain blocked by these concrete omissions:

1. **MAJOR — frozen same-method parameters and implementation are not actually common.** The revised `_common_run_identity` drops model settings. Independent synthetic evidence changed only `HYPERGATE/movielens/seed2023` `lambda_domain` to `.999`, synchronized that run and its two embedded checkpoint configs, and `build_freeze` still accepted the matrix (`MIXED_SAME_METHOD_HYPERPARAMETERS_ACCEPTED`). The rule must at least compare each method's effective `models[method]` configuration across its formal seeds/datasets; allowing different architectures across methods is appropriate, but allowing different settings within a frozen method is not. The newly recorded package snapshots are also not compared across formal runs, so the old fixed reference/upstream SHA strings can accompany different newly implemented training code. Compare the actual snapshot files directly across formal runs without hashes, and preserve the common source reference in the freeze. The report's pairing identity still omits source commits; aggregation also does not reject settings/source changes within the same method across seeds. These are the original F3/F5 frozen-source/config obligations, not extra hardening.
2. **MAJOR — implementation snapshot and execution environment are incomplete/inaccurate.** `_snapshot_implementation` uses `reference_data.glob('*.json')`, but the four required JSON files are in per-dataset child directories. The actual snapshot lists only `default_config.json`; fix the recursive inventory. `_environment_record` is called before `seed_all` forces IEEE math. The reviewer explicitly set matmul precision to `tf32`: the recorded value was `tf32`, then actual post-seed precision became `ieee`. Capture/refresh execution settings after seed setup so the stored environment describes the run that actually executes. These are the original F6 provenance requirements.

The E gate now requires formal phase, exact run/checkpoint absolute paths and direct metadata/config/data binding; missing checkpoints are rejected. Numerical Markdown, run provenance, failure records and bounded inference timing now exist. These improvements do not remove the above remaining obligations. The parent should return only these narrow fixes to the pipeline coder and request a final focused re-review; previously valid model evidence remains reusable.

## 8. FIX2 final independent re-review

**Verdict: PASS.** This focused review inspected the changed `report.py`, `data.py`, `evaluation.py` and `training.py`, the FIX2 implementation report and targeted regression cases. It reused the already valid model/source/CLI/documentation findings rather than reopening unrelated scope. The reviewer made no implementation corrections.

Independent command:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_sota_baselines_2526/tests -q
```

Result: **21 passed in 4.04s**. All inputs used by training/authorization regression cases were synthetic fixtures, including their simulated E role; no real-data optimizer step, real E evaluation, GPU experiment, installation or checksum calculation occurred.

| Previously open finding | Actual correction and independently checked evidence | Final status |
|---|---|---|
| F3: same-method parameter drift within formal freeze | `build_freeze` now records `model_config_by_method` and compares each method's effective model block across all formal seeds/datasets. The earlier `.999` HyperGate auxiliary coefficient mutation, with both checkpoint configs synchronized, now raises the intended method-specific drift error. Valid heterogeneous **methods** remain allowed. | CLOSED |
| F3/F6: actual implementation source consistency | Freeze reads all declared source-snapshot files directly, compares their byte contents across formal runs, and retains canonical bytes as base64, not a digest. The regression mutates `pipeline-source-v1` to equal-length `pipeline-source-v2` and is rejected. The exact valid 30-run/60-checkpoint synthetic fixture still freezes successfully. | CLOSED |
| F3: source identity at E access | `evaluate_role` clears previous authorization, rechecks the current source snapshot against frozen bytes, and only then authorizes the exact formal run/checkpoint and reads labels. A post-freeze equal-size source change raises PermissionError; subsequent direct label access remains locked. | CLOSED |
| F5: reporting source/config compatibility | Pair identity now includes pinned source commits. Aggregation enforces one effective model configuration per phase/method across its selected seeds/datasets. Synthetic mismatched-source pairs and same-method parameter drift are rejected, while actual numeric Markdown, sample SD and compatible pairing fixtures still pass. | CLOSED |
| F6: nested preprocessing metadata omitted from snapshot | `_snapshot_implementation` now recursively inventories reference JSONs. The regression confirms both PREPROCESSOR and DATA_MANIFEST JSONs for MovieLens and KuaiRand are present in the copied snapshot. | CLOSED |
| F6: recorded precision preceded actual execution setup | `seed_all` now precedes `_environment_record`. The failure fixture starts with both precision controls at `tf32`, then verifies that the captured receipt accurately records the executed `ieee` values. STARTED/FAILED propagation and original exception behavior remain intact. | CLOSED |

Together with FIX1, the final acceptance state is: F1 official HyperGate initialization/reproducible construction CLOSED; F2 native M3OE objective CLOSED; F3 exact formal freeze and source/config consistency CLOSED; F4 removed/retained accounting CLOSED; F5 numerical/provenance/compatible reporting CLOSED; F6 failure/environment/source/cost records CLOSED; F7 effective CLI selection/output CLOSED; F8 clean reproduction paths/matrix/environment documentation CLOSED; F9 categorical boundary CLOSED. No known blocking implementation finding remains from this review.

### Final scope and remaining limits

The reviewed package is suitable for the orchestrator to present a concrete RUN approval request. After explicit approval and a fresh resource check, the staged trial/development/five-seed procedure may proceed under the frozen specification. Reviewer PASS is not the user's RUN approval and does not authorize interference with existing GPU processes.

HyperGate remains an **official-core unified adaptation with UNVERIFIED_PROPOSAL parameter fields**, not verified as-published numerical reproduction. AutoIFS retains its disclosed input/selection/bias-shape adaptations and external pinned source; no redistribution permission is inferred from the absence of a license. Raw checksum strings are provenance only under the active hash restriction; direct-value reconstruction checks are the recorded data evidence. Historical E is reassessment, not an unseen confirmation set. Real-data convergence, GPU execution, measured benchmark AUC/LogLoss, full runtime/memory, and formal paper-table eligibility remain unmeasured until the approved experiments actually finish.
