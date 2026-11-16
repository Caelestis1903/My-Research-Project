# Cycle02 execution plan — pending formal DESIGN and distinct REVIEW

Selected candidate NQR-025; controls BRIER-OPPORTUNITY-025 and MEAN-GAP-025. The original foreground parent alone controls DESIGN -> IMPLEMENT -> distinct REVIEW -> current-conversation RUN -> ANALYZE -> fresh native Astra -> next plan. This document authorizes no stage skipping. It is a research handoff, not a review PASS or a runnable implementation.

## 1. Ownership, paths and frozen evidence

Workspace W=`/home/xzf/Code/phase0-3_tvg_rm_20261116`; control workspace O=`/home/xzf/Code/phase0-3`. Cycle plan P=`research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02`; future raw root N=`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1`. Future report root=`research_reports/natural_quadratic_ranking_20261116`. The existing artifacts/cycle_02 dispatch/context directory belongs to the parent; this child has not written there.

Publication gate already satisfied: main/local HEAD `556b4ae354fa1465c0bd33f1744eec8d93c59dd2`, tree `6b235aab0f3b4e760ec17b611c0881efa8788edc`; C1 PUBLICATION_RECEIPT.json records remotely verified CAS publication. Current frozen worker79-file aggregate is `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`. This child independently recomputed it without importing experiment modules. Keep the original bootstrap source receipt and RESOURCE_1_SUPERSESSION as separate provenance.

Every new native agent gets explicit fork_context=false and file handoff only. Only the parent dispatches. Each finishes and is closed permanently; no resume/reuse or child-of-child. No independent/headless/background AI, `codex exec/resume/fork`, tmux, nohup, systemd, autonomous.py start/run/recover, replacement driver, or agent loop. Existing guardians and retired driver stay intact.

Before every dispatch and training admission, read `manual_stop_requested` in `artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json` and the `STOP_AUTONOMOUS` sentinel under that same root in **both W and O**. A stop in either is authoritative. Parent owns root-state writes; neither coder nor runner rewrites historical controls. Technical failures PAUSE with evidence. Only explicit USER_STOPPED normally terminates the parent loop. All max_cycles/max_astra_cycles/max_experiments are null.

## 2. Fresh DESIGN handoff

Parent gives the fresh designer all six files, the C1 completeness addendum, current AGENTS/start/buchong/runbook and source receipts. Designer produces a formal DESIGN SPECIFICATION/completeness decision before coder starts. It must independently check the NQR moment identity, coefficient budget, controls, historical PAIR non-equivalence, baseline identity, no-Test restriction and runtime/RAM contract. A missing implementation/profile is a pending later-stage gate, not an unspecified algorithm hyperparameter. If a genuine scientific ambiguity remains, return REVISE to the parent; no sub-session and no opportunistic candidate sweep.

Hash the final accepted design bundle using SHA256 of compact UTF-8 JSON, sorted cycle-relative `[path,SHA256(file_bytes)]` pairs, separators comma/colon, no appended newline. Include the formal designer's additional file(s) as well as these six. Put this hash in the later run/review receipt, not self-referentially inside its own input bundle. Preserve this original research bundle's provenance when the formal design adds detail.

## 3. IMPLEMENT: smallest isolated package

Future new package only: `experiments/natural_quadratic_ranking_20261116/`. Proposed entrypoints/files:

| New path / entrypoint | Responsibility |
|---|---|
| `__init__.py` | Minimal package marker |
| `core.py::loss_components`, `core.py::train_step` | Six-cell FP32 moment objective, three fixed arms, exact source-clean branch, one Adam/EMA update and small scalar ledger |
| `protocol.py::config`, `job`, `source_manifest`, `validate_attempt`, `stop_requested` | Cycle02-only roots, frozen data/model settings, job identity,10-control reference, source/selection/design binding |
| `checks.py` | Finite CPU algebra/gradient/source-clean identity checks; no efficacy screening |
| `runner.py` with `prepare`, `smoke`, `run`, `worker`, `handoff` | Foreground, bounded mechanical execution of this matrix; original parent owns the RAM process, native runner supervises workers; no agent dispatch or research-loop logic |
| `tables.py::export` | Exact new30+reused10 collection; raw curves/endpoints/costs/masses/diagnostics; no research verdict in runner |
| `README.md` | Short commands and receipt references only |

Do not edit/add anything within C1's frozen scan roots: `experiments/task_axis_even_views_20261116/`, `experiments/auxiliary_objective_attribution_20261115/` including stage2, `experiments/conditional_information_momentum_20261102_v3/` including vendor, or `experiments/task_view_gradient_residual_momentum_20261116/`. Do not change requirements lock, old orchestration overlay, raw data, old artifacts/reports or root files. The sibling package lies outside all these source-manifest trees. New checks also stay in this sibling; avoid a root test framework or dependency installation.

Reuse exact pure/source helpers from auxiliary `protocol.py`: `SEEDS, DATASETS, SELECTION, digest, immutable, sha_file, dataset_cache, optimizer, load_protocol, make_model, setup, cpu_copy, state_digest, rng_state, restore_rng, evaluate, write_json`. Source-clean identity uses TV `core.source_clean_step`. EMA uses conditional protocol `update_ema`. RAM uses frozen `guardian.request` and its existing BytesIO transport; server is frozen TV `runner.serve` at N only. If reusing C1 `gpu_status`/`ram_usage`, call only those read-only measurement functions; never its root-bound prepare/run/worker/recover/retain methods or projected-reservation admission. C1's b7 overlay is runtime evidence for the correct predicate, not a callable C2 scheduler.

Copy/adapt only necessary foreground worker/epoch/receipt plumbing into the new package. Do not port old RESOURCE_1 repair machinery, autonomous driver, generic plugin system or new storage service. Source manifest must enumerate all new training and execution files and all transitive frozen model/data/optimizer/evaluator/storage imports. Reviewer binds the actual aggregate; a novel source hash must not masquerade as the C1 worker identity. Recompute the old79-file aggregate before/after implementation to show preservation.

Run project commands only with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` (or explicit conda run in that environment), never base. Set PYTHONDONTWRITEBYTECODE=1 for checking/importing frozen packages so new pycache files are not introduced there. No implementation command in this plan has been executed by the research child.

## 4. Frozen protocol and baseline compatibility

M3oE fixed1:3 domains,2 tasks,embedding_dim16,fcn[512,256,256,64],4 experts,exp_d=exp_t=bal_d=bal_t=1. No architecture, fusion, routing, gate-detach, feature, label, split or population change. Ordinary training gates retain source detach; no full-derivative probe is needed. FP32, no AMP/TF32, no batch/precision cuts for concurrency. Source CPU threads1 and cuDNN benchmarkfalse.

Adam:lr.01,betas(.9,.999),eps1e-8,coupled WD ML1e-4/KR1e-5,foreach=false,fused=false,amsgrad=false; one step/batch, no schedule/clipping. EMA:ML rho.9828205985452511/KR.995, initialized exactcopy, one update after each successful Adam step, buffers copied. All three arms lambda=.25 from epoch1 to50, margin1 on probabilities; no warmup, curriculum, HPO, seed prefilter, or endpoint-dependent gate.

Reuse the immutable verified cache `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache` through the frozen source data helper. Raw CSVs are `/home_new/xzf/datasets/ml-1m/ml-1m.csv` and `/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`. The relocated path must agree with recorded raw hashes, arrays, row IDs, encoders/scalers and train-only preprocessing. No prepared mdmt_next artifact or `/home/xzf/data` fallback. Use train for learning and validation for evaluation; no Test inference, metrics, selection or exploratory inspection. Existing recorded Test fingerprints are provenance, not permission to evaluate Test.

Registered dataset identities, train/validation array fingerprints, feature metadata and raw digests are embedded/referenced in EXPERIMENT_MANIFEST. ML train/validation800167/100021, domain counts[155949,318599,325619]; KR2006132/250767, counts[172782,1743898,89452]. Same private CPU loader seed+11000 and retained tails21927/40052. Verify initialization and each of the500 dataset/seed/epoch order hashes against the C1 reused-reference index before accepting comparisons. No extra RNG stream is needed.

Reuse **10 CLEAN trajectories only** for primary/current tables. NDP/TV/C1 remain historical context, not extra observations or new baseline runs. Exact CLEAN entries, result hashes and index receipt are in the manifest. This child checked all20 historical control result hashes, including the10 CLEAN subset, against the published index; all CLEAN results record50 epochs and test_accessed=false. Run-stage reuse additionally verifies matching source model/data/optimizer/EMA/selection/environment, five complete seeds, initialization/loader,50 epochs and earliest-max selection. The overall training source hash necessarily differs because the objective differs; compare the shared dependency identities, not equality of the entire new source tree. If identity cannot be established, PAUSE for parent-directed correction; do not silently swap an old Test baseline, a different split, or rerun extra CLEAN trajectories outside the registered budget.

## 5. Required distinct REVIEW and finite engineering smoke

Coder reports only implemented specification, changed paths, hashes and basic/CPU checks. It does not analyze research results. Parent closes coder and spawns a distinct fresh reviewer to inspect formal design, actual implementation and CODER IMPLEMENTATION REPORT. Allowed training gate is PASS, or PASS WITH FIXES with every blocking issue corrected and evidenced. FAIL returns through a fresh coder and distinct fresh review; no hidden revision cap.

Required correctness checks, written only in the new package:

1. Tiny literal n^2 pair enumeration versus moment values/gradients for R and B; verify M variance subtraction and all-three-arm CLEAN score-gradient identity at p=.5 for arbitrary class proportions; fixed inputs cover balanced/imbalanced classes, absent domain, missing/singleton class and both tasks. FP64 fixture tolerance1e-10 absolute/relative; FP32 tolerance1e-6 absolute and1e-5 relative. This tests a distinct literal definition, not just duplicated reduction code.
2. Unit and per-task coefficient sums, explicit row-class BCE coefficients, all-zero-lambda exact CLEAN state identity, all-one-class source delegation, tail B normalization, source `grad=None` behavior. For exact branches require bitwise equality of model/Adam/EMA/RNG under identical device/settings. Diagnose-on/off preserves those identities. No need for a broad testing framework.
3. Source initialization/model routing, training gate detach, no perturbation calls, no candidate RNG, no extra optimizer/EMA updates. One source model forward and scalar backward per ordinary training batch. No model/tensor files written by any code path.
4. New-root RAM roundtrip/recovery contract, current85 admission and physical fit, no hard process cap or both-card gate, pending allocation ACK and remeasurement. Include concrete CPU admission fixtures where projected85 fails but actual physical fit passes, and where current85 passes but usable-capacity fit fails.

After code review, a fresh current-conversation luna-runner performs engineering smoke only. For each dataset/arm, use seed2022, true full batch40960 and true tail shape, ordinary model/optimizer/EMA allocations, the registered diagnostic operation, validation path and RAM transfer. Smoke has its own disposable new-root keys and never contributes to efficacy endpoints or seed counts. It may validate objective/state identity but may not select arms based on validation AUC. Reviewer signs the actual source-bound smoke/resource receipts before formal30.

Expected future command surface: `python -m experiments.natural_quadratic_ranking_20261116.checks`; `...runner prepare --attempt N --review <code-review>`; `...runner smoke --attempt N --review <code-review>`; `...runner run --attempt N --devices 0,1 --review <final-signoff>`. Commands are to be implemented by coder, not existing executable claims. No `ensure_guardian`, autonomous `start`, or detached launcher.

## 6. CURRENT utilization and actual physical capacity

Mandatory runtime evidence: C1 `PHYSICAL_CAPACITY_CORRECTION.md`, `PHYSICAL_CAPACITY_MEASUREMENTS.json`, `IMPLEMENTATION_REVIEW_PHYSICAL_CAPACITY.{md,json}`, and `orchestration/task_axis_even_views_current_utilization_20261116.py` hash `b7eeba76d21727b596897eec84e0e5766c01e8a41fba832c1036d38bdf7c6f3f`. Preserve the actual OOM receipt and all superseded policies; do not repeat their admission assumptions.

For each individual card j, independently, admit the next worker only when all are true:

    CURRENT_NVML_used_j / CURRENT_NVML_total_j < .85
    CURRENT_measured_board_power_j / CURRENT_power_limit_j < .85
    CURRENT_host_RAM_fraction < .90
    no unresolved startup/allocation ACK on this card
    CURRENT_used_j_bytes + incoming_measured_full_process_bytes <= CUDA_usable_total_j_bytes.

The first three are the user's current-utilization rule. The last is physical hardware capacity after an observed OOM, **not** projected85%. Do not add predicted worker power, the old195/206W budgets, a projected-memory85% constraint, a fixed number of workers or a requirement that both cards fit. A valid one-card opportunity is sufficient. Current RAM includes all existing owners; no new projected90 gate. Pending startup protection may temporarily serialize allocation on that card, while the other card remains independently eligible. Resample current GPU memory/power and host RAM after every real allocation ACK before another admission; update observed peaks as workers allocate/train/validate.

C1 CUDA usable capacity was25307578368B/card (24135.1875MiB), smaller than NVML25769803776B (24576MiB). Its actual full-process residency was ML4515168256B and KR5540675584B. Maximum torch reserved was4183818240/5209325568B; non-torch/context residency was331350016B=316MiB. Full-residency-minus-earlier-smoke corrections were352321536/348127232B; b7 used the larger measured correction. A torch-only estimate was demonstrably insufficient.

C2 must obtain its **own** real full-process smoke profile, not claim C1's multi-view peaks as C2 measurements. Record PID-specific NVML residency, torch allocated/reserved peaks and their difference across full/tail training, diagnostics, evaluation and RAM serialization, plus per-card CUDA usable total. For incoming jobs use the maximum measured full C2 residency for the dataset/arm and any subsequently observed larger residency; ensure the profile also covers measured torch peak plus observed non-torch context, using the C1 316MiB observation until a compatible C2 full-process measurement supersedes it. Do not double-count context already included in NVML full residency. During initial profiling, C1's measured full-process footprint and b7 correction are the available bootstrap capacity evidence; record that it is a reference, and replace it after actual C2 allocation ACK/profile. No formal admission from an unprofiled torch-only estimate. If insufficient physical room exists, wait under current supervision; do not cut algorithm/batch/precision.

Log contemporaneous measurements, proposed job/profile, admitted/rejected reason, ACK and post-allocation resample. Board power is observed, never summed with predicted worker power. Actual OOM causes technical PAUSE, preserves raw failure and last RAM ACK, and triggers correction/review of the physical profile without altering scientific code. It does not authorize restart of old guardians or a more restrictive projected85 policy.

## 7. RAM storage, recovery and handoff

After review, only the original parent may create the **new C2** mechanical owner in a persistent foreground exec session, calling unchanged TV `runner.serve` with absolute N. This is storage, not a new AI session or autonomous driver. Parent records and retains the live exec session/PID; current native runner supervises it and all mechanical workers. No shell backgrounding. The old C1 owner (user-reported session59365/PID1570727) and every old root/state remain untouched; this child has not contacted it. No client writes to an old endpoint, including new-key mirrors or status mutation.

New job key = hash of full dataset/arm/seed/config/source/design/cycle identity. All job/result/journal/recovery paths are under N, absolute when sent to serve. Import frozen `guardian.request` for N only. Do not invoke old entrypoint defaults. The server's output root is constructor-bound; an IPC path field cannot redirect a request sent to an old owner. The new exporter reads exactly its30 results plus the immutable10-entry CLEAN subset, never old broad globs/tables.export.

Use only RAM tensors/CPU clones and BytesIO IPC. No torch.save to paths, pickle model files, weights, optimizer dumps, checkpoints, temporary model files, or model files in archives. Data caches are preexisting read-only data artifacts, not new model serialization. Numeric JSON/CSV, digests, sources and reports may be persisted. Private connection credentials stay in the new root's local credential receipt and are never published/copied into research manifests.

At each epoch boundary after validation, send a complete continuation snapshot and selected states to RAM; write numeric epoch receipt and require matching epoch/hash ACK. Last acknowledged epoch is the only recovery boundary. Snapshot contains model/EMA/Adam/counters/RNG/loader/global step/selected states/best scores/common endpoints/raw cost counters. Preserve an explicit continuation hash, selected-state hashes and serialized byte counts. No C2-specific history exists. Cumulative costs include failed/paused physical attempts, not only last successful attempt.

On stop or lost supervision, finish the current supported atomic epoch/parameter-EMA transaction safely and commit/checkpoint when possible, then pause. On exception, preserve last valid ACK; do not continue a half-updated state. Recovery uses the same new-root owner only, after all relevant recorded workers are dead and both workspaces' stop controls permit continuation. Epoch0 reinitialization is allowed only when no acknowledged snapshot ever existed, with an explicit failed-attempt receipt. Loss of the live owner containing unpersisted states is a technical PAUSE, not permission to create a replacement and pretend continuity. Recoverable transport warnings are investigated against ACK/worker evidence, not automatically treated as scientific fatal errors.

After each complete trajectory retain selected online/EMA states and necessary numeric provenance; release obsolete continuation/optimizer state per the existing server contract. Retain all30 completed selected-state pairs until parent analysis/review and parent-owned storage acceptance. Parent decides candidate eligibility and any C2 discard after analysis; do not release old or eligible incumbent states. Completion handoff lists new root, live owner session/PID, job keys, selected epochs/hashes, retained bytes, all500-per-arm epoch receipts, source identity and Test=false. Parent accepts ownership before the runner child is closed. No shutdown while protected states or live workers remain. This is bounded retention per experiment; it is not an unlimited collection of discarded model histories.

## 8. RUN matrix, raw records and cost

For each arm NQR-025, BRIER-OPPORTUNITY-025, MEAN-GAP-025, run ML and KR, seeds2022,2023,2024,2025,2026,50 epochs each. Total30 trajectories,1500 epochs,51750 parameter and51750 EMA updates, natural populations, fixed40960 batch/FP32. CLEAN reuse adds10 trajectories to tables only, not new training. All three configurations finish full five-seed evaluation; no single-seed science gate.

Runner records raw results only: per-epoch validation online/EMA six-cell AUC/LL; macro/traffic/pooled/worst-cell aggregates; selected epochs/hashes and metrics; E10/E30/E50; every seed's pairs; data identity/order; coefficient/statistic ledger; Adam/EMA counts; raw timing/throughput; full process/torch/host memory; failures/pauses/admission history. Distinguish snapshot-selected LL from minimum LL. Four endpoints*two estimators*40 trajectories=320 endpoint rows, with metric columns/cell rows as needed. Record all selected and all common-time endpoints, not only favorable ones.

Algorithm loss diagnostics are not evaluation LL: evaluation uses unscaled BCE/logloss on actual validation populations via the frozen evaluator. Costs include model forward/backward, validations, diagnosis, transfer/serialization, attempts, worker seconds and wall seconds separately. CLEAN-derived6.513780 worker-hours is a reference scale only. Actual NQR cost, concurrency and full-process memory are unmeasured until smoke/run; report those limitations rather than an invented time guarantee.

## 9. Parent ANALYZE and registered kill gates

Primary for each dataset: paired seed difference NQR selected fixed-EMA validation six-cell equally weighted macro AUC minus CLEAN selected fixed EMA. Report all5 differences, mean, sample SD (ddof1), positive/negative/zero counts, descriptive95% t interval `mean +/- 2.7764451051977987*sd/sqrt(5)`. Seed is the independent unit. Never count cells/epochs/online/EMA as additional seeds. No CI-sign or seed-unanimity efficacy gate.

Separate registered comparisons:

- Component increment: NQR EMA−CLEAN EMA, primary per dataset.
- Total algorithm+EMA difference: NQR EMA−CLEAN online, each independently selected; cannot attribute the EMA portion to NQR.
- Online component difference: NQR online−CLEAN online, secondary.
- Mechanisms: NQR−BRIER and NQR−MEAN-GAP selected EMA, plus each control−CLEAN EMA for interpretation; controls are never posthoc substitute champions.
- Common E10/E30/E50 contrasts for both estimators; all six cells, domain means, task means, traffic-weighted cell AUC, pooled-task mean, worst cell, LL and selected epoch. Compare stage-specific evidence without replacing the registered endpoint.

After all required full seeds and identity gates:

| Gate | Registered label/action |
|---|---|
| NQR primary means<=0 on both datasets | NO_CONTINUATION_SIGNAL; no NQR champion; parent reports and proceeds to fresh research |
| Positive on exactly one dataset | MIXED_DATASET_SIGNAL; report both, no cross-dataset claim |
| Positive on both but practical target not met | POSITIVE_VALIDATION_SIGNAL_BELOW_TARGET; no claimed qualification |
| For a particular named comparison/dataset, mean AUC>=.003 and each six-cell mean>=−.001 | VALIDATION_SCREEN_AUC_TARGET_AND_CELL_PROTECTION_MET; still not final Test qualification |
| Macro target met but any cell mean<−.001 | MACRO_TARGET_MET_CELL_PROTECTION_NOT_MET; disclose exact benefit/trade-off |
| Mean LL difference<0 | COMPREHENSIVE_LOSS_DECREASE for that comparison; judge separately from AUC |
| NQR−BRIER mean<=0 | Ranking-specific superiority unsupported on that dataset |
| NQR−MEAN-GAP mean<=0 | Beneficial variance-compression explanation unsupported on that dataset |

AUC eligibility is not vetoed solely by LL cost; a joint AUC+LL claim needs both conditions. These are validation labels in this active loop, not final publication/model superiority claims. Both positive mechanisms without a CLEAN gain support neither a champion nor a rescue. CI and sign counts inform evidence strength STRONG/MEDIUM/WEAK, not extra hard gates. Repeated validation use limits certainty even if one interval excludeszero.

Analyze size, measured prevalence, CLEAN performance, task, model and training stage separately. Identify repeated phenomena, contradictions, evidence strength, competing explanations, whether size plausibly dominates, whether small-gain/large-harm is actually present, and the next best falsifier. No causal size inference from this fixed-population matrix. No assumption curriculum/momentum is effective or M3oE must fail on difficult domains.

## 10. Outer-loop continuation and Test seal

Every candidate outcome, including NO-GO, NO_CONTINUATION_SIGNAL, mixed signal and Astra STOP, returns through parent report/context/publication gate to a fresh research child. No alternative here runs automatically and no posthoc HPO is authorized. Unlimited parent work remains under the original current conversation. A technical obstacle is PAUSE, not USER_STOPPED.

Test never opens while ACTIVE/PAUSED or merely because a candidate is positive/negative. Future Test requires explicit user stop, final freeze, and the existing eligible retained full-five-seed selected-RAM verification gate. Historic CLEAN numeric reuse does not prove current Test-ready RAM ownership, and this child has not verified it. Parent must apply that existing gate when it becomes relevant, without restarting old owners or dumping states to disk.

The next concrete action is the parent's fresh designer dispatch with this completed six-file handoff. Then this research context is closed permanently.
