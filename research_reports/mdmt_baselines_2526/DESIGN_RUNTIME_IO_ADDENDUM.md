# DESIGN SPECIFICATION ADDENDUM — runtime feature I/O revision R2

Date: 2026-10-29. Protocol remains `MDMT-SOTA-2526-v1`. This document authorizes only the implementation below, followed by independent review and a bounded replay; it is not a claim that the native crash root cause is known or fixed.

## 1. Question, hypothesis and actual incident evidence

Research algorithms and empirical hypotheses remain those of DESIGN_SPECIFICATION.md. The engineering question is whether replacing the feature-array memmap with an ordinary resident ndarray removes the observed failing access path without changing any model inputs, label permissions or training trajectory.

Evidence read for this amendment:

- `research_reports/mdmt_baselines_2526/RUN_AUTHORIZATION.json`: user explicitly approved RUN on 2026-10-29; 8 development and 30 formal trajectories, E only after complete freeze, reviewed delivery `25ea0edba908d8103981d892a28245e5277646ed`.
- `research_reports/mdmt_baselines_2526/RUN_INCIDENTS.json`: original M3OE_FIXED1/KuaiRand/2024 attempt terminated with signal11/exit139 after16 complete epochs; native fault in NumPy `_multiarray_umath`. Its one identical-source/config/seed full retry succeeded. No observed OOM evidence; cause unresolved.
- `research_reports/mdmt_baselines_2526/FORMAL_REPLAY_COMPARISON.json`: first16 V macro LL and AUC values match that retry exactly (maximum absolute differences both0).
- `artifacts/mdmt_baselines_2526/run_logs/formal_hypergate_kuairand_seed2022.stderr.log`: second native SIGSEGV, after5 complete epochs. Faulthandler stack is NumPy `_core/memmap.py:349 __getitem__` ← `evaluation.py:41 evaluate_role` feature indexing ← `training.py:_eval/_fit_standard`. This establishes the observed failing path, **not its underlying cause**.
- Orchestrator subsequently preserved that second failed output at `artifacts/mdmt_baselines_2526/attempts/formal_kuairand_HYPERGATE_seed2022_attempt01_SIGSEGV` and appended the incident receipt (5 completed epochs; recorded elapsed-epoch sum lower bound76.139612s). Original run/epoch/source files remain unchanged; canonical20 successful entries remain in place.
- `research_reports/mdmt_baselines_2526/FORMAL_RAW_RESULTS.json`:20 complete formal runs,1 partial,9 not started at inspection. E has not been read. Parent reports GPU1 approximately24GB free and host available56GiB; a KuaiRand x array is approximately330MB. Check current resources before restart; do not change anyone else's process.
- Actual source `data.py:Data.__init__` loads both x and y with `np.load(...,mmap_mode="r",allow_pickle=False)`; `evaluation.py` directly indexes x. Actual `report.py:build_freeze` rejects any mixed implementation snapshot bytes; `verify_frozen_source_snapshot` binds E evaluation to the saved source bytes. These checks must be preserved and amended explicitly, never bypassed.

H_IO: R1 and R2 produce identical input values/order/dtype and same permitted labels, model/optimizer/RNG state evolution and checkpoint decision for a fixed run, while R2 avoids the observed x-memmap indexing path. A successful replay provides bounded operational evidence; it does not prove why NumPy segfaulted or rule out faults in other remaining native paths.

## 2. Changed variable and invariants

Only the runtime representation of **features x** changes, from read-only disk-backed np.memmap to read-only ordinary in-memory np.ndarray.

Unchanged: original/prepared files and recorded data manifest, raw/provenance identities, row IDs and roles, vocabulary/scaler, features and labels, x dtype/shape/order, task masks, batch/epoch/optimizer schedules, all model equations and parameters, initialization, sampler, all RNG streams, checkpoint selection/stopping/metrics, 2022–2026 seeds,30 formal matrix, external source versions, test-release conditions. No dependency installation/version change, no dataset mutation, no synthetic duplication/drop_last, no repeated-until-success seed policy. `hash=deny` remains binding; all checks below are direct values/bytes.

## 3. Exact implementation change and ownership

Pipeline coder owns only:

1. `experiments/mdmt_sota_baselines_2526/data.py`: in **Data.__init__ only**, load x with `np.load(self.root / "x.npy", allow_pickle=False)` without mmap_mode. Set x read-only (`self.x.setflags(write=False)`) to retain previous read-only semantics. Do not call np.asarray on a memmap as the fix: it can retain mapped backing. An ordinary ndarray loaded directly without mmap avoids that ambiguity. No alternate flag/configuration or user-selectable storage backend is needed.
2. `.../report.py`: narrowly extend existing formal source-freeze handling to the exact approved R1/R2 pair described in section5. All current config, model-specific hyperparameter, dataset, run, checkpoint and source-inventory checks remain active.
3. `.../evaluation.py`: only the necessary selection of the current run's frozen R1/R2 source snapshot and verification of the approved runtime evaluator source before granting E. The arithmetic and batching loop stay unchanged.
4. Focused tests in existing `tests/test_pipeline.py` or one `tests/test_runtime_io.py`, and the implementation/review/replay provenance reports. Update the reproduction guide with the one explicit amended freeze invocation.

No algorithm/model/training/config/CLI module changes are required. If an explicit Python function argument is the smallest way to pass the amendment to build_freeze, use it and document a short project-Python command; do not add a new CLI/general compatibility framework. No other worker modifies these owned files concurrently.

**Labels y stay exactly as before**: `np.load(...,mmap_mode="r",allow_pickle=False)` with existing role-specific indexing after authorization. Do not load all y into RAM, call `np.array(y)`, enumerate its values, or scan all labels during this runtime repair. In particular constructor feature loading must not eagerly read E labels. Existing role/mask checks remain untouched except source-verification integration if strictly necessary; no change is currently needed to Data.authorize_final.

No edits to prepare_owned are required: the prepared files are already verified and fixed. Do not add mmap retries, exception wrappers that cannot catch native SIGSEGV, fallback read loops, allocation heuristics or OOM handling. If resident x cannot be allocated in the actual available memory, stop with that concrete error; do not silently switch back to memmap or change training.

## 4. Why the repair preserves mathematical inputs

For stored feature array X and row-ID batch I, R1 supplies `np.asarray(M[I])`, where M is the read-only memmap of X. R2 supplies `A[I]`, where A is an ordinary read-only ndarray loaded from the same X bytes. The required equality is elementwise and exact, including dtype and row order:

`A.shape == M.shape`, `A.dtype == M.dtype`, and `array_equal(A[I], M[I])` for every compared row.

The existing `tensor` already constructs `np.array(value)` before float32 conversion; it is unchanged. Loading x creates no Python/NumPy/Torch random draws. Existing advanced row indexing, tensor conversion, batch boundaries and forward/backward arithmetic therefore have no intended change. Setting writeable=False prevents accidental in-memory feature mutation. These are implementation invariants to test, not substitutes for replay evidence.

## 5. Exact R1/R2 source equivalence and final-freeze contract

### Preserve successful runs and their identities

Retain the20 successfully completed R1 formal trajectories, original `run.json`, both checkpoint files and implementation snapshots, byte-for-byte. Do not rewrite embedded run manifests, relabel their code as R2, copy R2 source into their snapshots, edit their runtime costs, or treat failures as extra independent repetitions. Preserve the two original failed attempts and logs with attempt IDs/actual status; move/archive only through the orchestrator's explicit owned-artifact bookkeeping, never over an existing artifact. The failed HyperGate/KuaiRand/2022 restarts from its original seed/initialization under R2; no partial state is available to assert an exact mid-epoch resume.

The previous unchanged-source M3OE/2024 successful replay remains the single valid R1 result for that matrix cell. The failed attempt stays in the failure/cost ledger. Keeping successful R1 results is justified only by the strictly delimited revision and evidence below; it does not authorize arbitrary future mixed sources.

### One finite, reviewed amendment record

Create one external report artifact, e.g. `research_reports/mdmt_baselines_2526/RUNTIME_IO_SOURCE_EQUIVALENCE.json`. It describes only this amendment ID `MDMT-SOTA-2526-RUNTIME-IO-R2`, and contains:

- R1 and R2 full implementation file inventories and complete file contents encoded as base64 (or exact references to immutable archived snapshots plus their complete contents in the final freeze); existing direct-source-snapshot format can be reused.
- Actual exact changed paths and before/after bytes/unified diff. Allowed executable paths are **data.py, report.py, evaluation.py only**. The only data.py computation change is Data.__init__ resident-read-only-x; the report/evaluation changes implement this exact revision binding. Default config and every model/loss/optimizer/training/sampler/helper/reference file must be identical by direct byte comparison. The same file inventory is required; no wildcard/path-substring exceptions or ignored source files.
- This design, independent review PASS receipt, CPU value/RNG/permission/source-gate test results, and subsequently the five-epoch real replay receipt; recorded paths are actual evidence, not an unchecked string assertion that tests passed.
- Exact protocol/matrix, prepared manifests and configuration remain as approved; revision R2 is explicitly a runtime-storage amendment, not a new algorithmic arm. Source snapshots label R1 and R2 separately. No invented Git SHA or newly calculated hash.

The record is completed/approved by the orchestrator after implementation review, with pending real replay explicitly marked before the first retry. It is accepted for **final freeze** only after required replay passes. An independent reviewer inspects the exact R1/R2 byte pair and the corresponding narrow diff; the orchestrator must not update either approved snapshot after this review. Any later source edit requires a new reviewed revision rather than silently regenerating this record. This is ordinary checked evidence for one actual repair, not a new signature/authorization service.

### Minimal existing-freeze extension

`build_freeze(run_root,output_path,*,runtime_io_amendment=...)` (equivalent explicit parameter name allowed):

- No argument: preserve the current single-source byte-equality behavior unchanged.
- Explicit approved amendment: validate its fixed kind/revision pair, inventory, changed-path constraint, review/test/replay completion and exact source pair. Directly compare the current installed package inventory/bytes with approved **R2**, and each run's entire snapshot map with either exact approved R1 or exact approved R2. A one-byte difference in any source file, including same-length mutations, must fail. Merely presenting a revision label, note or allowlist of filenames is insufficient.
- Keep all currently enforced30-run completion/status/path/config/source-commit/prepared-data/model-config/checkpoint-embedded-identity checks. Nothing is normalized out of an existing run identity. Both selected and fixed_final checkpoint associations remain exact.
- Freeze both complete snapshot maps, the finite amendment/evidence, and a `source_revision` R1/R2 binding on each run entry (additional metadata alongside its unchanged identity). Freeze the actual R2 implementation used for evaluation separately from the training source revision. A dictionary keyed by the two fixed revision IDs is sufficient; do not build generic migration rules.
- The amended freeze must never claim all30 trained on one common code revision. It may state one common algorithm/config/data protocol with a reviewed runtime-only source amendment and explicit per-run revisions.

`evaluate_role(...,role="E")` under amended freeze:

- Reset authorization first as now.
- Find the exact run key and its bound source_revision, retrieve that revision's full frozen source snapshot, then call existing strict byte verification against the run's actual implementation snapshot.
- Verify actual installed evaluator/package source equals the approved R2 evaluation snapshot, using direct bytes; do not rely only on the old run snapshot when executing changed evaluator code.
- Then execute the existing exact run/checkpoint/data/matrix authorization. A missing run/revision/map, modified source snapshot, changed current package, changed config or substituted checkpoint rejects before `data.labels(E,final)`.
- Old unamended single-source freezes retain existing behavior. No aliasing of R1 bytes into R2 and no “runtime changes are safe” Boolean bypass.

A full byte-pair comparison is deliberately selected over normalizing arbitrary source or trusting filenames: there are only two concrete revisions. If this cannot be implemented and reviewed with this bounded extension, stop and report the concrete gap; the alternative is rerun all30 under one fixed R2 source and charge/recode the cost, not disable the freeze checks. That expensive alternative is not currently necessary or authorized as the first choice.

## 6. CPU acceptance before resuming any GPU RUN

All checks execute with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; no hashes or new dependencies.

1. A synthetic prepared fixture loads x as exact ordinary np.ndarray, no memmap-backed base, nonwriteable, original float32/order/shape. y remains np.memmap. Record constructor label calls (none) and test E denial before freeze; S/legacy remain denied.
2. Direct full-feature comparison for both actual prepared datasets, chunked if desired: ordinary-load x versus the original file's read-only memmap view, including domain column; sample ordered training/V row-ID batches also match exactly. This reads **features only**, not y/E labels. A controlled read-only CPU equality check may exercise the old mapping once; failure stops the check, not repeated retries. No prepared file/manifest is modified.
3. CPU synthetic batch model prediction/loss and one optimizer step under R1-equivalent input versus R2 input: identical batch values, RNG states, logits, loss, per-parameter gradients (including None status), model parameters and optimizer state by direct comparison. Use an already-supported small model; no need to repeat all scientific model tests. Constructor feature load leaves Python/NumPy/Torch RNG states unchanged.
4. Existing source-freeze tests continue to pass. Add one small synthetic30-entry matrix with mixed exact R1/R2 sources accepted only with the explicit completed amendment; no amendment still rejects mixed sources. Verify per-run binding and both checkpoint views.
5. Negative checks: one-byte same-size mutation in a supposedly allowed data.py, an unapproved model/training file change, incomplete/failed replay evidence, altered source revision and current evaluator/package drift all reject before E labels. Tests should reuse the existing fixture, not create a general adversarial-source framework.
6. Existing train/label permissions, source-manifest/checkpoint equality and model-configuration-drift tests remain valid. Run affected pipeline tests and any required repository checks, not unrelated research experiments.

Independent reviewer then inspects this spec, actual diff and CODER IMPLEMENTATION REPORT and returns PASS or PASS WITH FIXES with every blocking issue corrected. This permits the bounded retry below, not final E release yet.

## 7. Bounded real replay and resumption rule

After review PASS, resume **only HyperGate/KuaiRand/seed2022**, from initialization, with original formal config/source external dependencies, original RNG setup and same device, plus R2 runtime loader. It is the replacement attempt for the existing failed matrix cell, not an added seed or algorithmic experiment. Faulthandler diagnostics stay enabled. Preserve failed original attempt and logs. Do not change max epochs/patience/batch or introduce an abbreviated real training configuration.

Compare its first5 completed epochs against the preserved failed attempt:

- V macro LogLoss and macro AUC maximum absolute difference exactly0, as established feasible by the previous16-epoch replay. Compare available per-cell V metrics and train loss/update/exposure counters too, retaining the exact fields actually present. Exclude only elapsed times/process/environment diagnostic fields from trajectory equality; do not hide changed scientific values.
- Input/row plans, data/config/source provenance and training-stage/epoch identities match; only approved runtime source revision differs. CPU equality and reviewed diff support equivalence, but aggregate V equality alone does not prove full state identity.
- Keep full new epoch log, comparison record and all observed differences. Do not relax equality after looking at results or claim unlogged model state from the failed process was compared.

The first retry must complete the original standard training/checkpoint protocol and normal output checks cleanly. Only after exact replay and successful completion may the remaining9 not-started formal entries run under the same frozen R2. If the first5 differ, stop launching further runs and investigate the specific discrepancy. If any further native crash occurs, preserve it and stop rather than cycling through retries or expanding to y/dependency changes automatically. The underlying fault remains unresolved even if this replay succeeds; report this bounded evidence honestly.

No historical E/test labels are read during CPU checks, retry validation or remaining formal training. Only after all30 valid entries, final amendment/replay evidence and normal complete freeze checks pass may existing E evaluation proceed. This restores the original user-approved workflow; no extra user permission is required for this reversible necessary runtime repair within the approved matrix.

## 8. Logging, interpretation and falsification

Keep R1/R2 source revision and storage semantics in the amendment and final per-run freeze. Existing run/checkpoint records are not edited; no new training config option or global protocol renaming. Record actual memory footprint/load seconds if trivially available, not a new benchmarking effort. Failed attempt time is retained as additional execution cost and never folded into a fictitious successful timing or statistical repetition.

Required reports: this design; CODER runtime implementation report; reviewer verdict; exact source diff/pair and direct-value/RNG tests; HyperGate5-epoch replay comparison; updated incident/cost ledger and reproduction guide. The final research report notes the native incidents, bounded replay evidence and mixed runtime revisions.

Falsification/stop conditions: x values/dtype/order change; RNG/optimizer evolution changes; forbidden labels are eagerly accessed; any algorithm/config/model/sampler source differs; a run cannot be bound to an exact reviewed snapshot; altered source/config/checkpoint passes the freeze; first5 scientific metrics differ; repeated native crash. None can be repaired by selecting favorable seeds/checkpoints or declaring runtime equivalence without new reviewed evidence.

Evidence strength now: **STRONG** for the location of the second observed crash and prior R1 replay equality; **WEAK** for a causal claim that memmap storage caused either crash. After a successful R2 full retry with exact first5 V replay, input/trajectory equivalence has direct bounded evidence, while general crash elimination remains unproven. AUC/LogLoss efficacy analysis awaits the original complete formal results; this repair introduces no new research-performance hypothesis or ablation.
