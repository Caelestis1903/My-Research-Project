# REVIEW REPORT

**Review pass:** 1/2  
**Scope:** Stage S1 implementation review only. No formal 192-run experiment was launched and no scientific result was interpreted.

## 1. Verdict

**FAIL**

The implementation is not safe to run. The exact 192-run matrix and most data/objective definitions are present, but several blocking defects can either prevent formal runs from executing or allow scientifically invalid/non-formal artifacts to be accepted. The most immediate execution blocker is that the unadapted M3OE returns a CPU prediction tensor from a CUDA forward pass, while the shared BCE target is on `cuda:0`. There are also blocking semantic-initialization, formal-protocol, retry/resume, validator, and reporting defects.

**Blocking issues:** C1-C6 and M1-M4 below.  
**Non-blocking issues:** m1-m3 below.

The implementation must return to `luna_coder`. Reviewer pass 2/2 may occur only after the coder fixes every blocking item and supplies executable focused tests. A formal Stage S1 run must not begin before pass 2.

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Independent verification |
|---|---|---|
| Exactly 192 unique tasks | PASS | `stage_S1_protocol.py:141-180,193-223`; constructed matrix contains 48 joint + 144 reference tasks and 192 unique keys. |
| MovieLens rotations and S0 sampling rule | PASS | `stage_S1_data.py:91-122,446-479`; actual `d0_large/seed2022` counts were independently loaded as `{0:150000,1:34500,2:3000}`. Rule is `RandomState(10*seed+domain)`. |
| KuaiRand natural split/domain/task contract | PASS WITH METADATA DEFECT | `stage_S1_data.py:483-607`; independent load produced `{0:172782,1:1743898,2:89452}` with both classes in every validation/test cell. Split-first/filter-second is implemented. However, KuaiRand physical d2 is mislabeled `tab2`, not `tab4`, in `manifest_fields` (`stage_S1_data.py:358-361`). |
| Train-only preprocessing / deterministic OOV | PASS | MovieLens vocab/scaler fits `train_idx` only (`stage_S1_data.py:407-438`); KuaiRand first pass is bounded to `train_end` (`stage_S1_data.py:519-532`); OOV is index 0. |
| Joint/reference row-label pairing | CODE PATH PASS; VALIDATION BLOCKER | `reference_view` correctly filters then remaps only the route and asserts equality (`stage_S1_data.py:266-345`). The formal validator/block audit does not independently enforce those pairing fingerprints, so corrupted or mismatched artifacts can still be accepted (C5). |
| MMoE fidelity | PASS | FuxiCTR-compatible `MMoE_Layer`, four experts `[256,64]`, gates `[64]`, and per-cell towers are wired at `stage_S1_models.py:75-96,102-116`. Tiny CUDA forward/backward passed. |
| PLE fidelity | PASS | FuxiCTR-compatible CGC with four shared experts, one cell-specific expert, gates and towers is wired at `stage_S1_models.py:86-96,104-116`. Tiny CUDA forward/backward passed. |
| M3OE fidelity and runnable CUDA path | FAIL | Raw `MDMTRec` is returned (`stage_S1_models.py:144-148`). Its legacy forward assembles outputs on CPU. Independent CUDA smoke showed `m3oe` output device `cpu`; `direct_per_example_bce` then raised a CPU/CUDA device mismatch against the CUDA target (C1). |
| D=1,T=2 single-domain MTL reference | PARTIAL | Constructors use D=1,T=2 and keep both tasks (`stage_S1_run.py:472-483`), but M3OE cell-gate semantic mapping is wrong for physical d1/d2 (C2). |
| Direct per-example BCE | PASS | Sole training path is `F.binary_cross_entropy(..., reduction="mean")` (`stage_S1_run.py:111-117,531-543,583-592`). Independent oracle check differed by `2.98e-08`. Test coverage is insufficient for the required full-epoch/all-backbone acceptance criterion (M3). |
| Semantic initialization pairing | FAIL | Domain embedding, towers, MMoE/PLE gates, and M3OE domain experts are mostly mapped, but M3OE `gate.*` is omitted from the cell-index remapper (`stage_S1_models.py:258-271`), and the audit does not store source-side hashes sufficient to verify equality (C2/M1). |
| Same data/split/batch/step/checkpoint protocol | PARTIAL | Fixed batch size, 50 epochs, no scheduler, common epochs and validation selection are present. Formal deviations can be silently accepted, retry paths are not resumable, and validators do not prove exact exposure/step/fingerprint contracts (C3-C5). |
| Deferred test / no leakage | STATIC PATH PASS; ACCEPTANCE TEST MISSING | Test evaluation is called only after 50 validation epochs and selection freeze (`stage_S1_run.py:598-665`). Epoch CSV excludes test fields. There is no focused leakage test, and validator relies substantially on self-declared booleans (M3/C5). |
| Selected/final/common states | PARTIAL | Model states are produced, but all 50 full model states are retained in RAM (`stage_S1_run.py:626`), and raw optimizer/RNG states are not saved (M2/m2). |
| AUC/LogLoss and cell indexing | CODE PASS; TEST BLOCKER | Shared `_cell_records` uses task-major indexing, unclipped AUC and fixed-epsilon LogLoss (`stage_S1_run.py:233-270`). No synthetic metric/index fixture exists (M3). |
| Cell-level raw/practical/concordant NT | PARTIAL | Selected/common pairing fields exist (`stage_S1_report.py:123-175`), but final paired contrast, practical LogLoss indicator, full stable inventory, and required summaries are incomplete (C6). |
| Domain size, independent-reference difficulty, baseline, model, stage fields | FAIL | Paired output has size and backbone, but omits domain share/physical identity/task identity/entropy and baseline seed variation; `reference_validation_difficulty` incorrectly uses reference macro validation AUC rather than complete per-cell AUC/LogLoss fields (`stage_S1_report.py:147-163`). |
| Formal protocol lock | FAIL | `--allow_protocol_deviation` bypasses runner checks and suppresses mismatch reporting; orchestrator does not lock all formal parameters (C3). |
| Resume-safe/fail-fast | FAIL | Failed retries are written under an undiscoverable nested attempt path; validator is structurally incomplete; corrupt prior manifests are silently replaced with `{}` in direct runner (`stage_S1_run.py:378-385`) (C4/C5). |
| Raw optimizer/model/RNG/log artifacts | FAIL | Model checkpoints exist, but optimizer state, architecture-optimizer state, raw RNG state, and required stdout validation are absent (M2). |
| Mechanical report only | SCOPE PASS; CONTENT FAIL | Report makes no GO/STOP decision, but it cannot produce several preregistered raw aggregates and accepts non-formal artifacts by default (C6). |
| Exclusion of Momentum/Curriculum/Safe-MTL/Stage 2 | PASS | No prohibited training method is present in Stage S1 implementation; ShareBottom is excluded. Documentation references to Stage 2 are gate/scope text only. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- The direct sample objective is mathematically correct, and the normalized `p_d` expression agrees on the independent synthetic check.
- MMoE and PLE produced finite CUDA outputs with the expected `[B,2]` shape and supported backward propagation.
- M3OE produced `[B,2]` on **CPU** after a CUDA forward. The current shared loss receives CUDA labels, so the M3OE third of the formal matrix cannot train.
- Actual MovieLens/KuaiRand counts and validation/test class coverage matched the frozen data contract in the independently loaded fixtures.
- KuaiRand's physical identity metadata is wrong for d2 (`tab2` instead of `tab4`), which would contaminate physical-domain comparisons.

### Gradient flow & Loss participation

- MMoE/PLE main parameters participate in the shared direct BCE path; no dummy architecture parameter exists.
- M3OE main and architecture parameter partitioning is explicit (`stage_S1_run.py:485-502`), and the architecture step calls the same objective.
- Training checks prediction, gradient, parameter, and optimizer-state finiteness. However, M3OE cannot reach the loss on CUDA in the present implementation.
- M3OE reference gate initialization is scientifically mismatched: for physical d2, independent inspection produced `gate.0.0.weight <- gate.0.0.weight` and `gate.1.0.weight <- gate.1.0.weight`; the required sources are global cell gates 2 and 5.

### Controlled variables & Baseline preservation

- Changes are localized to Stage S1 files and reuse existing backbone math. The workspace is not a Git worktree from this execution context, so a version-control diff could not independently establish that no pre-existing file was changed.
- Dataset splits, MovieLens rotations, objective, optimizer constants, fixed epoch budget, and reference-before-joint queue order are represented correctly.
- The formal lock is not authoritative: a non-formal `expert_num=99` configuration with `allow_protocol_deviation=1` was independently accepted by `assert_formal_launcher`.
- The report cannot provide all controlled comparisons because it omits final paired deltas and exact pooled/assembled-reference AUC.

### Fail-fast enforcement

**Not satisfied.** Runtime numeric checks are appropriately strict, but the following silent-acceptance paths remain:

1. `stage_S1_run.py:426-428` returns early from all argument checks under `allow_protocol_deviation`; `:733-734` then records no mismatches.
2. `stage_S1_orchestrator.py:308-321` does not lock expert count, FCN dimensions, shadow patience, or reject the deviation flag; `:247-256` validates the frozen declared protocol rather than the observed protocol.
3. `_check_finite_rows` checks a required field only if it exists (`stage_S1_orchestrator.py:107-112`), so missing numeric columns can pass.
4. Corrupt prior manifests are broadly caught and treated as empty (`stage_S1_run.py:378-385`) rather than failing immediately.
5. The report deliberately calls `collect(..., require_formal=False)` (`stage_S1_report.py:277-280`), so smoke/protocol-deviating artifacts can enter the formal raw tables.

## 4. Over-Engineering & Code Bloat Audit

- The implementation is generally stage-local and avoids enterprise abstractions.
- Keeping 50 complete model snapshots in memory (`stage_S1_run.py:626`) is unnecessary for a protocol needing only the current best, epoch 49, and five common epochs. On KuaiRand this adds avoidable memory pressure and can create a late OOM after otherwise valid training. Retain only the selected state plus required fixed/common states.
- Compatibility aliases in `stage_S1_models.py:425-439` and `stage_S1_protocol.py:184-186` are small, but they are not required by the approved design. They are non-blocking.

## 5. Bugs / Deviations Found

### CRITICAL

**C1 — M3OE formal CUDA runs are not executable.**  
Evidence: `stage_S1_models.py:144-148`, `stage_S1_run.py:529-534`; independent smoke: M3OE CUDA input -> CPU output; BCE raised `Expected all tensors to be on the same device`.  
Required minimal fix: add a Stage S1-local fidelity adapter (do not edit the legacy model) that returns the raw M3OE result on the input device while preserving the graph, then add CUDA forward/loss/backward tests for D=3 and D=1.

**C2 — M3OE reference cell gates are initialized from the wrong physical cells.**  
Evidence: `_semantic_source_name` remaps `tower.*`, `core.gate.*`, and PLE specific experts but not M3OE `gate.*` (`stage_S1_models.py:258-271`). Physical d1/d2 references therefore copy d0/task-adjacent gates by exact name.  
Required minimal fix: include M3OE `gate.*` in task-major cell mapping and test every backbone × physical domain, asserting source names and tensor hashes for all shape-compatible tensors.

**C3 — Formal protocol can be silently bypassed.**  
Evidence: `stage_S1_run.py:426-428,733-734`; `stage_S1_orchestrator.py:308-321,484`; independent check accepted `expert_num=99` with `allow_protocol_deviation=1`.  
Required minimal fix: remove/reject protocol-deviation mode from the formal orchestrator, validate every immutable field (including expert count, FCN dimensions, patience, workers, Python runtime), persist observed protocol, and require exact observed-vs-frozen equality in `validate_run`.

**C4 — Append-only retry output cannot become the canonical valid result.**  
Evidence: reruns are redirected from canonical `run_dir` to `run_dir/attempt_n` (`stage_S1_orchestrator.py:324-347`), while every later `build_matrix`, validator, and report looks only at canonical `run_dir`. A successful retry is therefore never resumed or collected.  
Required minimal fix: move/archive an invalid canonical attempt to an append-only failed-attempts path, then rerun at the canonical path, or implement and validate an immutable canonical success pointer consistently across queue and report.

**C5 — Validator/block audit can accept scientifically invalid artifacts.**  
Evidence: `stage_S1_orchestrator.py:107-112,141-257,413-455`. It does not enforce exact epoch-row counts/headers, exact batch/exposure/optimizer/architecture-step counts, RNG restoration, checkpoint-file hashes, selected epoch recomputation, semantic source-target hash equality, unmapped-subseed integrity, dataset/code hashes, or joint-reference row/label fingerprint equality. `audit_block` is not run by the completed queue/report and does not compare the pairing fingerprints.  
Required minimal fix: fail closed on exact schemas/cardinalities and all declared hashes/counters; validate every block after its joint run and again before report generation; add corruption tests for each critical artifact class.

**C6 — Required raw outputs cannot support the preregistered analysis.**  
Evidence: `stage_S1_report.py:248-250` writes pooled AUC as unavailable; `write_outputs` pairs only selected and common checkpoints (`:281-291`), not final; contradictions only detect within-cell seed flips (`:256-274`). Raw predictions needed for per-task pooled and assembled-reference pooled AUC are not persisted.  
Required minimal fix: persist row-aligned deferred-test labels/predictions (prefer compressed per-checkpoint artifacts), validate their hashes and row IDs, and mechanically emit selected/final/common paired tables, per-task pooled AUC, assembled-reference pooled AUC, all aggregate families, stable-cell inventory, baseline seed variation, and contradictions across seed/backbone/dataset/rotation/metric/stage.

### MAJOR

**M1 — Semantic audit is not hash-verifiable.** `common_tensor_hashes` contains only target hashes, not the mapped source tensor/slice hash; a mapping table can claim equality without proving it (`stage_S1_models.py:350-405`). Add source hash and source-slice hash and make validator recompute/compare them.

**M2 — Required raw audit artifacts are incomplete.** `stage_S1_orchestrator.py:34-41` and `stage_S1_run.py:706-814` do not save/require main optimizer state, M3OE architecture optimizer state, raw RNG state, or `stdout.log`. Add these artifacts and hashes; for MMoE/PLE explicitly record architecture optimizer as absent.

**M3 — Acceptance tests are materially incomplete.** Present tests cover only matrix basics, one small objective fixture, a weak sampling/fingerprint assertion, and missing-directory resume. There are no tests for initialization mapping/hashes, D=1/D=3 CUDA forward-loss-backward, cell indexing/AUC/LogLoss, deferred-test leakage, earliest tie selection, exact exposure and optimizer steps, instrumentation ON/OFF equivalence, validator corruption, block pairing, final/report aggregation, or retry recovery. The objective test is not a full-epoch oracle and does not prove a shared path across all backbones/scopes.

**M4 — Required analysis covariates are missing or wrong.** `METRICS_TEST_FIELDS`/paired outputs omit domain share, physical identity, dataset-specific task identity, label entropy, reference seed-variation baseline, and complete independent-reference difficulty fields (`stage_S1_run.py:84-93`; `stage_S1_report.py:147-163`). KuaiRand d2 identity is mislabeled at `stage_S1_data.py:358-361`.

### MINOR

**m1 — Preflight checks directories, not exact files or the requested child Python.** `runtime_preflight` imports Torch in the orchestrator interpreter and checks path existence only (`stage_S1_orchestrator.py:350-378`). Verify `args.python`, Torch/CUDA in that interpreter, exact CSV files/hashes, and actual device identity.

**m2 — Excess model-state retention.** All 50 model states are retained (`stage_S1_run.py:626`) although only selected/final/common states are required.

**m3 — Redundant compatibility aliases.** The extra aliases/wrappers are harmless but not required by the experimental specification.

## 6. Corrections Made

No implementation correction was made in this review pass. The defects span the model adapter, semantic initialization, formal lock, retry layout, artifact validator, raw prediction contract, report aggregation, and tests. Fixing them inside the reviewer would exceed a minimal localized correction and would effectively replace the coder stage. Per the required workflow, the implementation is returned to `luna_coder` with the exact blocking list above.

This review report itself is the only file created by the reviewer.

## 7. Remaining Risks

- The current EDDAenv contains PyTorch/CUDA but not `pytest`; the coder must provide a reproducible way to run the focused tests in the exact formal environment, or execute equivalent deterministic test scripts whose exit status is auditable.
- Full CUDA tiny training was not attempted because C1 already proves that the M3OE formal path fails at the first loss calculation.
- The current context did not expose a usable Git repository, so baseline/stable-path preservation must be rechecked by file hashes or a repository diff in the coder/reviewer-2 handoff.
- No formal Stage S1 artifact exists yet; the smoke outputs must remain clearly separated and must never be promoted into the formal collector.

## 8. Whether Experiments Are Safe to Run

**NO. Formal Stage S1 experiments are not safe to run.**

Required next action: return all blocking findings to `luna_coder`, implement and test the minimal fixes, then invoke the reviewer exactly once more (pass 2/2). Only a `PASS` or `PASS WITH FIXES` with every blocking issue closed may authorize `luna_runner`.

---

# REVIEW REPORT — FINAL REVIEW PASS

**Review pass:** 2/2 (final permitted reviewer pass)  
**Scope:** Stage S1 implementation and acceptance evidence only. No formal 192-run experiment was launched and no scientific conclusion was analyzed.  
**Required runtime used for CUDA checks:** `C:\Users\86199\.conda\envs\EDDAenv\python.exe`, PyTorch 2.0.0, CUDA 11.7, NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Verdict

**FAIL**

**Blocking issue count: 6. Non-blocking issue count: 2. RUN authorization: DENIED.**

The coder corrected the first review's model-device, M3OE gate-mapping, protocol-lock, canonical-retry, KuaiRand identity, and raw-state-artifact defects. The implementation now passes the requested CUDA model-path smoke and the 192-task dry-run. It nevertheless remains unsafe for formal RUN because the report uses a different LogLoss definition from the runner, corrupts task identity in pooled output, cannot emit two preregistered contradiction axes, omits required role/difficulty aggregate families, and the formal validator can still accept metric/provenance corruption. The required metric/index and logging-equivalence acceptance coverage is also absent.

Because this is review 2/2, there is no further reviewer retry available under the user's limit. Any remaining blocking issue therefore prohibits `luna_runner` from starting the formal Stage S1 matrix.

## 2. Requirement-by-Requirement Verification

| Requirement | Final status | Independent evidence |
|---|---|---|
| 192 unique formal tasks | PASS | `stage_S1_protocol.py:141-180,193-223`; EDDAenv dry-run reported exactly 192. Independent queue-manifest check found 192 tasks, 192 unique keys, 48 joint, 144 references, and zero launches. |
| MovieLens rotations | PASS | Actual `d0_large/seed2022` bundle had `{0:150000,1:34500,2:3000}`; reference construction retained paired rows. Frozen role maps and `RandomState(10*seed+d)` remain in `stage_S1_protocol.py:30-37` and `stage_S1_data.py`. |
| KuaiRand natural domains/tasks/counts | PASS | Actual EDDAenv load produced `{0:172782,1:1743898,2:89452}`, identities `tab0/tab1/tab4`, roles `M/L/S`, labels `stored_is_click/stored_long_view`, and mapping `0->tab0,1->tab1,2->tab4`. Code checks exact counts at `stage_S1_data.py:619-622`; corrected identities are emitted at `:371-373`. |
| M3OE/MMoE/PLE joint and reference forward/loss/backward | PASS | Synthetic EDDAenv CUDA smoke passed all three backbones for D=3,T=2 and D=1,T=2. Actual MovieLens feature/data smoke also passed joint/reference BCE and backward for all backbones. M3OE losses were finite on `cuda:0`. |
| Backbone fidelity | PASS | M3OE Stage S1 adapter is localized (`stage_S1_models.py:134-193`) and an independent identical-state comparison against legacy `MDMTRec` gave exact equality (`max_abs=0`) for D=3 and D=1. MMoE/PLE use the prescribed expert/gate/tower path. |
| Sample-target direct per-example BCE | PASS | One shared function is used (`stage_S1_run.py:115-121`), and the CUDA smoke verified equality with the normalized `p_d` oracle. No equal-cell training arm is present. |
| Semantic initialization pairing | PASS | M3OE `gate.*` is now included in task-major physical-cell mapping (`stage_S1_models.py:320-329`); smoke checked every backbone and all three physical domains. Source/source-slice/target hashes and equality flags are persisted (`:467-496`). |
| Data/split/batch/step/checkpoint pairing | PASS WITH VALIDATOR DEFECT | Runner/data paths implement the intended counts, fixed 50 epochs, common checkpoints and row pairing. However, validator trust gaps below mean corrupted formal artifacts can still be resumed/reported (B4-B5). |
| Deferred test/no leakage | PASS ON STATIC PATH | Selection is frozen before deferred prediction collection, epoch history excludes test fields, and validator recomputes earliest validation tie selection (`stage_S1_orchestrator.py:356-377`). No test-time call was found in the training epoch loop. |
| Shared AUC/LogLoss implementation | FAIL | Runner imports `safe_logloss` with epsilon `1e-12`; report independently clips at `1e-7` (`stage_S1_report.py:45-48`). A direct fixture returned runner `23.025850888570275` versus report `16.1180956512215` (B1). |
| Cell-level/raw/practical/concordant NT fields | PASS | Selected/final/common paired tables include AUC/LogLoss deltas and raw/practical/strong/concordant indicators (`stage_S1_report.py:180-228`). |
| Pooled/raw prediction artifacts | PARTIAL | `predictions_test.npz` is emitted and hashed (`stage_S1_run.py:845-856,965-970`), and pooled/assembled-reference metrics are computed. Task identity is wrong for task 1 and pooled LogLoss uses the wrong epsilon (B1-B2). |
| Domain size/difficulty/baseline/model/stage report fields | PARTIAL/FAIL | Cell rows contain most covariates and baseline seed variation exists, but required per-role and per-difficulty aggregate outputs are absent (`stage_S1_report.py:530-568`) (B3). |
| Contradiction inventory | FAIL | `write_outputs` supplies selected rows only (`stage_S1_report.py:534-540`), so stage flips cannot be emitted. Dataset grouping also includes `setting` (`:438-452`), but MovieLens and KuaiRand settings are disjoint, so dataset flips cannot be emitted (B3). |
| Formal protocol lock | PASS | Runner rejects both deviation flags; orchestrator independently locks frozen fields (`stage_S1_orchestrator.py:666-686`). Independent mutations of batch size, epoch, both LRs, weight decay, expert count, patience, device, workers, FCN dims, and both allow flags were rejected. |
| Resume/retry canonical recovery | PASS | Invalid canonical contents move to `failed_attempts/attempt_NNN`, while retry retains the canonical path (`stage_S1_orchestrator.py:689-734`). Focused retry test passed. |
| Formal artifact validator | FAIL | It does not recompute CSV AUC/LogLoss from raw predictions, does not require exact design counts, and treats `source_code_hash` as optional (`stage_S1_orchestrator.py:280-327,503-544,597-618`) (B4-B5). |
| Raw optimizer/model/RNG/log artifacts | PASS ON WRITER PATH | Writer saves main optimizer, M3OE architecture optimizer or explicit absence JSON, raw RNG state, stdout, all required model states, and hashes (`stage_S1_run.py:835-907`). Validator requires their files and checks checkpoint finiteness/hashes. |
| Code/protocol/runtime fingerprints | FAIL | Only `stage_S1_run.py` is hashed; model/data/protocol/report code and concrete runtime environment are not fingerprinted. Validator skips the source check entirely when the field is absent (`stage_S1_orchestrator.py:612-618`) (B5). |
| Required acceptance tests | FAIL | CUDA forward/loss/backward, objective, tie and semantic mapping smokes pass, but no synthetic AUC/LogLoss/cell-index fixture, instrumentation/logging ON/OFF equivalence, validator-corruption suite, or block-pairing negative test exists (B6). |
| Mechanical report scope / prohibited Stage 2 methods | PASS | Report does not interpret scientific results or issue GO/STOP. No Momentum, Curriculum, Safe-MTL, mitigation or Stage 2 algorithm was introduced. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- All six requested model configurations (three backbones × joint/reference) produce `[B,2]` CUDA tensors, finite direct BCE, and connected backward gradients.
- The M3OE adapter preserves legacy numerical output exactly while correcting only device-local tensor allocation. The cell gate map now resolves reference task/domain cells to the correct joint physical cell.
- The training objective is correct. The reporting objective is not: the report's `1e-7` clipping changes LogLoss relative to the frozen runner's `1e-12`. Thus cell-level CSV LogLoss and pooled/assembled LogLoss do not share one metric definition.
- Pooled rows derive both task identities from one arbitrary representative metric row per run (`stage_S1_report.py:286-315`). A fixture with distinct task labels produced `[(0,'rating_gt_3'), (1,'rating_gt_3')]`; task 1 should be `rating_gt_4`.

### Gradient flow & Loss participation

- Direct BCE participates in each backbone's forward graph. Synthetic and actual-feature CUDA backward checks observed gradients for trainable parameters.
- M3OE architecture parameters remain separated from the main optimizer, with exactly one architecture step intended per epoch. MMoE/PLE have no dummy architecture optimizer.
- Non-finite prediction/loss/gradient/parameter/optimizer checks remain fail-fast. No unintended `.detach()` was found on the main prediction-to-loss path; the legacy M3OE gate input detach is faithfully preserved.

### Controlled variables & Baseline preservation

- Formal launcher constants, seeds, 50-epoch exposure, no scheduler, checkpoint rule, split logic and reference-first queue order are controlled.
- Model, data and protocol implementation is Stage S1-local; however, this directory is not exposed as a Git worktree, so stable-legacy-path preservation could not be verified by diff.
- Raw reference baseline fields and seed variation are present, but the missing role/difficulty aggregate families and impossible dataset/stage contradiction axes mean the mechanical output cannot support the complete preregistered analysis without post-hoc code changes.

### Fail-fast enforcement

**Not satisfied.** The formal launcher is now strict, but acceptance remains fail-open in these material ways:

1. The validator checks raw prediction shapes/counts but never recomputes per-cell AUC/LogLoss and compares them with `metrics_test.csv`/`metrics_summary.csv` (`stage_S1_orchestrator.py:503-544`). Finite but wrong cell metrics can pass and are then used by paired NT tables.
2. `train_domain_counts` is only checked for cardinality and internal consistency (`:280-327`), not against the exact MovieLens role counts or KuaiRand natural counts. Coordinated count corruption can pass resume validation.
3. `source_code_hash` is conditional (`:612`), and when present fingerprints only `stage_S1_run.py`; missing or changed model/data/protocol/report provenance is not rejected.
4. Artifact declarations are checked against their own declared hash map, but the validator does not require that every `REQUIRED_FILES` member be included in that hash inventory. Presence alone is insufficient provenance for optimizer/RNG/log artifacts.

## 4. Over-Engineering & Code Bloat Audit

The implementation remains localized and appropriately research-oriented. No factories, registries, compatibility framework, broad retry suppression, or enterprise logging system was added. The checks that were added generally protect experimental validity.

Non-blocking bloat remains: all 50 full model states are retained in memory even though only best/final/common checkpoints are required. This is avoidable memory pressure but does not alter results if the run completes.

## 5. Bugs / Deviations Found

### CRITICAL (blocking)

**B1 — Runner and report use different LogLoss mathematics.**  
Evidence: `stage_E1_instrumentation.py:29,1506-1511` clips at `1e-12`; `stage_S1_report.py:45-48` clips at `1e-7`. Direct fixture difference was `-6.907755237348777`. This violates the single shared metric implementation requirement and makes pooled/assembled-reference LogLoss incomparable with cell rows.  
Minimal fix: import the same `safe_auc`/`safe_logloss` implementation into the report, expose one frozen epsilon, and add exact fixture comparisons.

### MAJOR (blocking)

**B2 — Pooled task identity is silently corrupted.**  
Evidence: one arbitrary row is retained per run at `stage_S1_report.py:286-288`; its identity is reused for every task at `:307-315`. Independent fixture reproduced task 1 receiving task 0's identity.  
Minimal fix: build a `(formal_run_key, task) -> task_identity` lookup from validated rows and fail if identities are absent or inconsistent.

**B3 — Required aggregate and contradiction outputs are incomplete.**  
Evidence: output list at `stage_S1_report.py:530-568` has no per-role or per-difficulty summary. Contradictions receive selected rows only at `:539`, so stage flips are impossible. Dataset grouping at `:441` includes `setting`, preventing MovieLens (`d*_large`) and KuaiRand (`natural`) from ever sharing a dataset-comparison group.  
Minimal fix: emit explicit role and independently defined reference-difficulty raw summaries; pass selected/final/common paired rows to contradiction enumeration; define the preregistered comparable dataset grouping without requiring equal dataset-specific setting names.

**B4 — Cell metric artifacts are not validated against raw predictions.**  
Evidence: validator checks shapes, finiteness, row/label hashes and counts only (`stage_S1_orchestrator.py:503-544`). It never recomputes task-major cell AUC/LogLoss for every checkpoint. Paired NT calculations later trust CSV metrics.  
Minimal fix: recompute all deferred cell metrics from `predictions_test.npz` using the single shared metric implementation and require equality within a fixed tolerance; add corruption tests for score, domain, task, and checkpoint indexing.

**B5 — Formal provenance/count validation remains incomplete.**  
Evidence: counts are self-consistency checked rather than compared with `ROLE_SIZES/ROLE_MAPS` and exact Kuai counts (`stage_S1_orchestrator.py:280-327`); code hash is optional and runner-only (`:612-618`, `stage_S1_run.py:920`); concrete runtime/GPU/library metadata is not persisted.  
Minimal fix: derive exact expected counts from dataset/setting/run type/physical domain; require hashes for runner, models, data, protocol and report; persist Python/PyTorch/CUDA/GPU identifiers; require every required artifact in the hash inventory.

**B6 — Mandatory acceptance coverage is still absent.**  
Evidence: `experiments/tests/run_stage_S1_smoke.py:59-99` covers CUDA forward/loss/backward, objective, tie and semantic mapping only. `test_stage_S1_acceptance.py:21-104` covers one confounded launcher rejection, retry, schema and pooled assembly. Search found no AUC/LogLoss/cell-index fixture, instrumentation/logging ON/OFF equivalence, validator-corruption suite, or block-pairing negative test. This fails design acceptance criteria 12, 15 and 16 (`stage_S1_design.md:353-357`).  
Minimal fix: add focused tests for shared metric/index behavior, logger ON/OFF parameter/metric equality, corrupted raw/CSV/hash artifacts, exact counts, and mismatched joint/reference pairing.

### MINOR (non-blocking)

**N1 — Misleading completion log ordering.** The runner prints a completion message before all output writing/validation has completed. Move the success message after manifest creation and validation.

**N2 — Excess checkpoint RAM retention.** Retain only selected/final/common states rather than every epoch state.

## 6. Corrections Made

No research implementation was changed in this reviewer pass. The reviewer appended this final report only. The remaining defects span shared metric semantics, report grouping, formal validation and acceptance coverage; changing them in review would exceed a minimal reviewer correction and would constitute a new coder iteration.

## 7. Remaining Risks

- There are no completed formal Stage S1 raw artifacts, so raw optimizer/RNG/log writing could be verified only by the actual writer/validator code, not by accepting a completed formal run. Launching even one formal run for this purpose was intentionally avoided.
- The actual KuaiRand loader is expensive but independently reached and verified exact counts, domain identities, role labels, task definitions and physical mapping. A follow-up diagnostic attempted an incorrect attribute name only after those checks had printed; it did not alter data or create a formal artifact.
- The project directory is not a Git worktree in this environment, preventing independent diff-based confirmation that legacy files remained unchanged.
- Smoke output locations and the queue dry-run were isolated from the formal raw-results root; no 192-run training was started.

## 8. Whether Experiments Are Safe to Run

**NO. Stage S1 formal experiments are not safe to run.**

Final gate status: **review 2/2 exhausted; blocking issues remain; `luna_runner` must not launch the formal 192-run matrix.** The implementation must return to a coder for correction, but under the user's two-review maximum it cannot be authorized by another reviewer unless the user explicitly revises that limit or provides another acceptance authority.

---

# REVIEW REPORT — PASS 3

**Review pass:** 3 (the user removed the former two-review limit)  
**Scope:** Stage S1 implementation acceptance only. No formal experiment was launched and no scientific result was analyzed.  
**CUDA runtime:** `C:\Users\86199\.conda\envs\EDDAenv\python.exe`, PyTorch 2.0.0, CUDA 11.7, NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Verdict

**FAIL**

**Blocking issues: 2. Non-blocking issues: 1. RUN authorization: DENIED.**

B1, B2, B4 and B5 are materially corrected. B3 and B6 are not fully corrected: the dataset contradiction implementation can misclassify a MovieLens rotation-only sign reversal as a cross-dataset contradiction, and the claimed logging ON/OFF acceptance test is invalid. The pytest version fails whenever Torch is installed; the CUDA smoke version logs on both branches and therefore has no actual OFF condition.

## 2. Requirement-by-Requirement Verification

| Requirement / prior blocker | Status | Independent verification |
|---|---|---|
| B1 — one LogLoss implementation and epsilon | PASS | `stage_S1_metrics.py:18,52-62` defines `LOGLOSS_EPS=1e-12`; runner imports it at `stage_S1_run.py:45`, validator imports `cell_metrics` at `stage_S1_orchestrator.py:33`, and report aliases the same function at `stage_S1_report.py:20-26`. Search found no second AUC/LogLoss implementation in Stage S1. EDDAenv metric smoke passed. |
| B2 — pooled task identity | PASS | `task_identity_lookup` indexes `(formal_run_key, task)` and rejects missing/inconsistent identities (`stage_S1_report.py:261-277`); pooled rows use this lookup at `:280-313`. Independent fixture now returned `[(0,rating_gt_3),(1,rating_gt_4)]`. |
| B3 — role/difficulty/stage summaries | PASS WITH DEFECT | Role, difficulty and training-stage CSVs are emitted (`stage_S1_report.py:541-637,640-690`). Selected/final/common rows are supplied to the inventories. However, the cross-dataset contradiction axis is not semantically sound (R3-B1). |
| B3 — contradiction axes | FAIL | Dataset grouping omits setting but also omits `size_role` (`stage_S1_report.py:436-468`). Multiple MovieLens rotations therefore occupy one dataset level; mixed signs within MovieLens can trigger a row labeled `comparison_axis=dataset` even when KuaiRand has no opposing sign. Independent three-row fixture reproduced this false dataset contradiction. No report-axis acceptance test exists. |
| B4 — raw prediction metric recomputation | PASS | `validate_raw_prediction_metrics` rebuilds every checkpoint × task × domain cell, checks task-major index, physical identity/share, task identity, counts, AUC and LogLoss, and compares selected summary metrics (`stage_S1_orchestrator.py:320-434`). Positive fixture and CSV/prediction corruption checks passed. |
| B5 — exact counts | PASS | Exact MovieLens role and KuaiRand natural counts are derived by `expected_train_domain_counts` (`stage_S1_orchestrator.py:166-184`) and enforced at `:520-549`; exact shares and cell train counts are also checked. |
| B5 — provenance/runtime/artifact hashes | PASS | Required artifact hashes are enforced at `stage_S1_orchestrator.py:216-246`; complete Stage S1 code hashes and runtime fingerprints are required at `:881-913`. Writer persists matching inventories and Python/PyTorch/CUDA/GPU metadata (`stage_S1_run.py:209-238,916-978`). Artifact corruption negative test passed. |
| B5 — block pairing | PASS | `audit_pairing_fingerprints` checks joint physical views, reference split fingerprints and explicit pairing evidence (`stage_S1_orchestrator.py:187-213`), and `audit_block` invokes it (`:1104-1154`). Negative fingerprint fixture passed. |
| B6 — metric/index corruption coverage | PASS | Tests cover shared metric identity, `1e-12` clipping, task-major indices, raw CSV corruption, raw prediction corruption, exact count contracts and artifact hash corruption (`test_stage_S1_acceptance.py:116-178,210-244`). |
| B6 — logging ON/OFF equivalence | FAIL | The pytest test appends only on the ON path but asserts ON and OFF logs are equal (`test_stage_S1_acceptance.py:181-207`); manual EDDAenv execution produced identical parameters but non-empty ON log, empty OFF log, and a false assertion. The CUDA smoke instead appends on both paths (`run_stage_S1_smoke.py:71-95`), so it does not exercise OFF. Neither test invokes Stage S1's actual batch/logging path. |
| 192 matrix/formal lock | PASS | EDDAenv dry-run generated 192 unique keys: 48 joint, 144 reference, launch count 0. No formal raw run was started. |
| CUDA models/objective/semantic mapping | PASS | EDDAenv smoke passed M3OE/MMoE/PLE, D=3 and D=1, CUDA forward/direct BCE/backward, objective equivalence, earliest tie and all physical semantic mappings. |
| Scope exclusions | PASS | No Momentum, Curriculum, Safe-MTL, mitigation or Stage 2 algorithm was introduced; report remains mechanical. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- The shared Stage S1 AUC and fixed-epsilon LogLoss functions are now used throughout runner, validator and report.
- Raw deferred predictions are recomputed at task-major cell granularity, preventing finite but incorrect CSV AUC/LogLoss from passing validation.
- Pooled task labels remain task-specific and no longer inherit an arbitrary representative row.
- CUDA forward/loss/backward and M3OE semantic pairing remain valid.

### Gradient flow & Loss participation

- All three backbones and both joint/reference domain cardinalities have connected direct-BCE backward paths on `cuda:0`.
- No new detach, dummy optimizer, fallback objective or disconnected training component was introduced by the B1-B6 changes.

### Controlled variables & Baseline preservation

- Formal counts, shares, runtime, code hashes, artifacts and joint/reference pairing now fail closed.
- The dataset contradiction table still confounds the dataset axis with multiple MovieLens rotations. A sign mixture entirely inside the MovieLens dataset is sufficient to emit a dataset contradiction once a same-sign KuaiRand row is present. This would give the main analysis a mechanically mislabeled contradiction.

### Fail-fast enforcement

Runtime and artifact validation is substantially improved and no silent metric fallback was found. The remaining failure is acceptance coverage: the logging equivalence test cannot pass in the actual Torch environment, while the separate smoke does not test the OFF branch. Therefore the design's acceptance criterion 15 is not demonstrated.

## 4. Over-Engineering & Code Bloat Audit

The shared NumPy metric module and focused validator helpers are proportionate to scientific reproducibility. No unnecessary framework, registry, broad exception suppression or production monitoring layer was added.

## 5. Bugs / Deviations Found

### MAJOR — BLOCKING

**R3-B1 — Dataset contradiction rows can be false positives caused only by MovieLens rotations.**  
Evidence: dataset groups use `(backbone, physical_domain, task, checkpoint, seed)` and exclude both setting and size role (`stage_S1_report.py:436-468`). Fixture: MovieLens `d0_large=-0.01`, MovieLens `d1_large=+0.01`, KuaiRand `natural=+0.02` produced `comparison_axis=dataset`, although no MovieLens-vs-KuaiRand opposing sign existed; the negative sign occurred only inside MovieLens.  
Minimal fix: compare cross-dataset signs at a prespecified matched stratum, at minimum adding `size_role` while retaining physical domain, or first aggregate a deterministic per-dataset value before comparing dataset levels. Add positive and false-positive negative fixtures for dataset, rotation and stage axes.

**R3-B2 — Logging ON/OFF acceptance evidence is invalid.**  
Evidence: `test_stage_S1_acceptance.py:196-207` leaves `off_log=[]` but asserts it equals the populated `on_log`; EDDAenv reproduction returned `params_equal=True`, three ON losses, empty OFF list, `test_assertion=False`. In `run_stage_S1_smoke.py:82-95`, both branches append losses, so the OFF path is not OFF.  
Minimal fix: run identical seeded Stage S1-relevant update steps with batch fingerprint/log collection enabled versus disabled; assert identical final parameters and post-step metrics, assert only the ON branch emitted logs, and execute this test in EDDAenv without requiring unavailable pytest.

### MINOR — NON-BLOCKING

**R3-N1 — Difficulty summary mixes validation AUC with test LogLoss.** `difficulty_summary_rows` documents independent reference validation AUC/LogLoss but emits `reference_validation_auc_*` and `reference_test_logloss_mean` (`stage_S1_report.py:575-610`). The raw paired table retains validation LogLoss, so no information is lost, but the summary naming/measure pair should be made consistent and covered by a fixture.

## 6. Corrections Made

No research implementation was changed in review pass 3. This report append is the only reviewer edit. The two blocking issues belong to report grouping and acceptance-test implementation and must return to the coder stage.

## 7. Remaining Risks

- The full writer/validator contract has not consumed a completed formal run, as required by the prohibition on formal experiments during review.
- The system-Python suite skips Torch-dependent tests; the EDDAenv has Torch/CUDA but no pytest. The standalone CUDA smoke covers models and raw metric corruption but currently masks the logging-test defect described above.
- This directory is not exposed as a Git worktree, so stable legacy-path preservation cannot be verified by diff.

## 8. Whether Experiments Are Safe to Run

**NO. Stage S1 formal experiments are not safe to run.**

Pass 3 gate: **blocking=2; return to `luna_coder`; do not dispatch `luna_runner`.** A later reviewer pass is permitted because the user removed the former two-review limit.

---

# REVIEW REPORT — PASS 4

**Review pass:** 4  
**Scope:** Final verification of R3-B1, R3-B2 and R3-N1 plus focused Stage S1 regression checks. No formal experiment was launched and no scientific result was analyzed.  
**CUDA runtime:** `C:\Users\86199\.conda\envs\EDDAenv\python.exe`, PyTorch 2.0.0, CUDA 11.7, NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Verdict

**PASS**

**Blocking issues: 0. RUN authorization: GRANTED.**

R3-B1, R3-B2 and R3-N1 are corrected. Dataset contradictions now require a matched physical-domain and frozen size-role stratum; the positive/false-positive dataset, rotation and training-stage fixtures all pass. The logging equivalence fixture executes the real `S1Runner._train_epoch` and `_record_batch` methods with emission enabled/disabled and proves equal parameters, optimizer state, cell metrics/objective, RNG states and batch trajectories while emitting rows only in the ON branch. Difficulty AUC and LogLoss now both come from validation data.

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Independent verification |
|---|---|---|
| R3-B1 matched dataset contradiction stratum | PASS | Dataset grouping includes `(backbone, physical_domain, task, checkpoint, seed, size_role)` at `stage_S1_report.py:438-463`. Missing roles are excluded from the dataset axis. A MovieLens rotation-only opposing-sign fixture no longer emits a dataset contradiction, while matched MovieLens/KuaiRand physical-domain + size-role opposition does. |
| False-positive and positive dataset fixtures | PASS | `test_stage_S1_acceptance.py:225-244` and standalone CUDA smoke cover both cases. Independent execution passed. |
| Rotation contradiction fixtures | PASS | Opposing MovieLens settings emit `comparison_axis=rotation`; equal signs do not (`test_stage_S1_acceptance.py:246-258`). |
| Training-stage contradiction fixtures | PASS | Selected/common opposing signs emit `comparison_axis=stage`; stable signs do not (`test_stage_S1_acceptance.py:259-270`). |
| R3-B2 same S1Runner training path | PASS | `run_instrumentation_equivalence_fixture` invokes the actual `S1Runner._train_epoch`, which invokes the actual `_record_batch`, direct BCE, optimizer step, batch fingerprint and `_cell_records` path (`stage_S1_run.py:639-694,1048-1178`). Only `instrumentation_enabled` changes. |
| Logging ON emits / OFF suppresses | PASS | Independent EDDAenv result: `on_log_rows=3`, `off_log_rows=0`. Formal runner initializes instrumentation enabled at `stage_S1_run.py:532-535`; the toggle is confined to the non-research acceptance fixture. |
| Numerical equivalence | PASS | Independent fixture returned `parameter_equal=True`, `optimizer_equal=True`, `metrics_equal=True`, and equal objective/exposure/optimizer-step counters. |
| RNG and batch equivalence | PASS | Fixture compares Torch CPU, every CUDA generator state, NumPy state, Python state and complete batch fingerprint trajectory (`stage_S1_run.py:1116-1178`); all were equal. |
| R3-N1 validation-only difficulty | PASS | Paired rows carry independent-reference validation macro-AUC and validation macro-LogLoss (`stage_S1_report.py:184-190`). Difficulty summaries use only those validation fields and declare `difficulty_evaluation_split=validation` (`:587-629`). A deliberately different test LogLoss fixture proved it cannot enter the difficulty summary. |
| Shared metrics/task identity/raw validation | PASS | Prior B1/B2/B4 fixes remain intact; focused regression tests passed. |
| Exact counts/provenance/artifact/block validation | PASS | Prior B5 fail-closed checks remain intact; focused corruption and pairing tests passed. |
| CUDA model paths | PASS | EDDAenv standalone smoke again passed M3OE/MMoE/PLE × D=3/D=1 forward, direct BCE, backward, objective equivalence, earliest tie and semantic mapping. |
| Formal matrix | PASS | Dry-run manifest contains 192 tasks and 192 unique keys: 48 joint and 144 single-domain references; `launch_count=0`. |
| Scope exclusions | PASS | No Momentum, Curriculum, Safe-MTL, mitigation, Stage 2 design or scientific GO/STOP conclusion was introduced. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Dataset contradiction rows now represent a sign reversal between two dataset levels inside the same backbone, physical domain, task, checkpoint, seed and size role. MovieLens rotations cannot independently manufacture a dataset contradiction.
- Validation difficulty fields use a single evaluation split. Test LogLoss remains available elsewhere as an outcome metric but is not mixed into the difficulty summary.
- Shared AUC/LogLoss and task-major cell indexing continue to pass raw-prediction recomputation fixtures.

### Gradient flow & Loss participation

- Logging ON/OFF executes identical real S1Runner update code. Both branches have identical model and optimizer states after three batches, demonstrating that fingerprint/log emission does not alter loss participation or parameter updates.
- CUDA smoke confirms all backbone and run-scope paths retain finite connected gradients.

### Controlled variables & Baseline preservation

- The equivalence fixture resets the same seed and initial state for each branch, uses one identical deterministic data loader, and changes only batch-row emission.
- Batch fingerprints, optimizer steps, examples processed, objective, per-cell metrics and all RNG families remain equal.
- Report grouping now separates dataset, rotation and stage axes with explicit positive and negative fixtures.

### Fail-fast enforcement

The reviewed paths fail closed:

- dataset comparisons lacking a size-role match are excluded rather than pooled;
- contradiction fixtures test both detection and non-detection;
- the instrumentation fixture raises immediately on any parameter, optimizer, metric, RNG, batch or emission mismatch;
- difficulty fields are explicitly labeled as validation split.

No silent fallback was found in the reviewed corrections.

## 4. Over-Engineering & Code Bloat Audit

The equivalence fixture is longer than an ordinary unit helper but is narrowly scoped, invokes existing runner methods, launches no research training, and directly satisfies a mandatory acceptance criterion. It does not affect the formal path because the formal runner always enables instrumentation. No enterprise abstraction or speculative algorithm was added.

## 5. Bugs / Deviations Found

No blocking or non-blocking implementation deviation remains from R3-B1, R3-B2 or R3-N1.

## 6. Corrections Made

No implementation changes were made by the reviewer in pass 4. This report append is the only reviewer edit.

## 7. Verification Evidence and Remaining Risks

- EDDAenv CUDA standalone smoke: **PASS**.
- Direct real-path instrumentation fixture: **PASS**, returning all equality flags true and ON/OFF rows `3/0`.
- Focused pytest suite: **16 passed, 2 skipped**. The skips are Torch-dependent under system Python and are covered by the successful EDDAenv standalone smoke.
- Syntax compilation of the modified Stage S1 modules/tests: **PASS**.
- 192-task dry-run: **PASS**, with zero launches.
- No completed formal artifact was available for an end-to-end validator acceptance run; producing one was deliberately prohibited during review. This is an execution-stage risk, not an implementation blocker, because the writer/validator contracts and their corruption fixtures have passed.
- The directory is not exposed as a Git worktree, so legacy-path preservation could not be independently verified by Git diff.

## 8. Whether Experiments Are Safe to Run

**YES. Stage S1 is safe to enter the RUN stage.**

Pass 4 gate: **PASS; blocking=0; `luna_runner` may launch the formal 192-run matrix under the approved Stage S1 protocol.** The runner must still stop immediately on the first process failure or artifact-validator failure, as required by the design.

---

# REVIEW REPORT — PASS 5

**Review pass:** 5  
**Scope:** Independent review of the first-formal-key failure and the canonical-identity, post-child hash-closure, retry/archive, and cache corrections. No formal training was continued and no scientific result was analyzed.  
**CUDA runtime:** `C:\Users\86199\.conda\envs\EDDAenv\python.exe`, PyTorch 2.0.0, CUDA 11.7, NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Verdict

**PASS WITH FIXES**

**Blocking issues after reviewer correction: 0. RUN authorization: GRANTED FOR RESUME.**

The submitted repair correctly resolves the two first-key failures: the manifest now preserves a canonical machine dataset identity while retaining a separate display label, and the parent process closes the artifact hash inventory only after the child has exited and the inherited stdout handle has closed. The failed first attempt remains untouched and is still rejected by the current validator.

This review found one additional integrity defect: the artifact validator could reuse a stale digest after a same-size mutation whose mtime was restored. The reviewer made a localized correction so artifact validation always evicts cached generations for the file and reads its bytes. A new negative regression test reproduces this exact case. All checks pass after that correction.

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Independent verification |
|---|---|---|
| First-key failure preserved and understood | PASS | `stage_S1_queue_manifest.json` records exactly one launched task and the original reasons `identity_mismatch:dataset` and `artifact_hash_mismatch:stdout.log`. The raw directory was inspected read-only. Its manifest/stdout SHA-256 values at review were `04ED8F...BBBA4` and `E915DC...5C17B`. No formal continuation was launched. |
| Canonical dataset identity | PASS | `S1Runner` canonicalizes the input at `stage_S1_run.py:467-475`, writes canonical `dataset` and `dataset_key` after bundle metadata merge, and keeps the human label only as `dataset_label` (`:1045-1053`). Validator canonicalizes queue/manifest values but independently requires exact canonical `dataset_key` (`stage_S1_orchestrator.py:526-543`). |
| Compatibility is not fuzzy or over-broad | PASS | `canonical_dataset` accepts an explicit finite alias list and otherwise raises (`stage_S1_protocol.py:92-103`). Positive `MovieLens-1M`/`movielens` and negative `MovieLens-1M-copy`/`movie` fixtures pass (`test_stage_S1_acceptance.py:331-340`). The preserved old artifact no longer reports dataset identity mismatch but still fails on stale hash/finalization/current-code evidence. |
| Child-exit/stdout ordering | PASS | Parent opens stdout, waits synchronously for `subprocess.run`, exits the file-handle context, and only then calls `finalize_artifact_hashes` (`stage_S1_orchestrator.py:1324-1347`). Thus the runner completion line cannot race the final stdout digest. |
| Atomic manifest closure | PASS | Finalization requires an OK manifest and every declared file, recomputes hashes, records method/stdout hash/stdout size, writes a PID-specific temporary file, then uses `os.replace` and cleans remnants (`stage_S1_orchestrator.py:259-315`). The regression fixture confirms no temporary finalization file remains. |
| Manifest self-reference excluded | PASS | Required hash names explicitly exclude `run_manifest.json`; the declaration still requires the manifest to exist, and an attempted manifest hash is rejected as `artifact_hash_self_reference` (`stage_S1_orchestrator.py:224-255`). This is the correct non-self-referential closure. |
| Post-finalization tampering | PASS WITH REVIEWER FIX | Ordinary append tampering was already rejected. Reviewer reproduced a stronger case: same-size byte replacement plus restored mtime changed the real SHA-256 while the submitted validator returned no reasons. `_file_hash(..., fresh=True)` now clears all path-local cache generations and rereads bytes at the artifact trust boundary (`stage_S1_orchestrator.py:93-113,231-251`). The exact attack now returns `artifact_hash_mismatch:stdout.log`. |
| Hash cache clearing on canonical retry | PASS | Finalization clears every prior path-local cache entry before recomputing each artifact (`stage_S1_orchestrator.py:282-295`); validation now also performs a fresh byte hash. This covers a retry that happens to reuse the same path, size, and timestamp granularity. |
| Append-only failed-attempt archive | PASS | `archive_failed_attempt` excludes the existing archive root, chooses the first unused `attempt_NNN`, creates it without overwrite, moves current canonical contents into it, and retains the canonical directory (`stage_S1_orchestrator.py:1103-1127`). The expanded test proves `attempt_001` remains unchanged while `attempt_002` receives the next failure (`test_stage_S1_acceptance.py:47-66`). |
| Canonical retry path is safely reusable | PASS | Classification records the archive path and returns the original canonical `run_dir` as pending (`stage_S1_orchestrator.py:1082-1100`). `S1Runner` refuses an existing/corrupt manifest and any unexpected canonical contents, while deliberately allowing only `failed_attempts` and the parent-owned new `stdout.log` (`stage_S1_run.py:487-504`). |
| Positive full validator path | PASS | In an isolated temporary copy only, the preserved real single-domain first-key artifacts were given current code provenance and post-child hash closure. EDDAenv full `validate_run(require_formal=True)` returned `ok=true`, `reason_count=0`. This was a validator replay, not a formal result, and the source raw directory was not changed. |
| Negative preserved-artifact path | PASS | Current EDDAenv validation of the untouched failed attempt returns only stale stdout hash, missing finalization, and expected old-code-hash reasons; it does not accept or retrofit the artifact. |
| 192-run formal matrix | PASS | EDDAenv dry-run returned `[stage_S1] dry-run matrix 192` with no launch. Matrix generation/lock remained unchanged. |
| Scope exclusions | PASS | No Momentum, Curriculum, Safe-MTL, Stage 2 logic, mitigation algorithm, result interpretation, or scientific conclusion was added. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- The repair does not alter the model, routing, task/domain indices, direct per-example BCE, checkpoint selection, or reporting mathematics.
- EDDAenv/CUDA smoke passed M3OE/MMoE/PLE for both joint-equivalent `D=3,T=2` and single-domain `D=1,T=2` forward, finite direct BCE, backward, device consistency, and M3OE physical-domain gate mapping.
- The standalone metric validator and real S1Runner logging ON/OFF equivalence checks also passed.

### Gradient flow & Loss participation

- All three models produced connected gradients in both domain cardinalities on `cuda:0`.
- The hash/identity/retry changes are outside the training graph and do not change optimizer participation or baseline behavior.
- A separate EDDAenv CUDA objective fixture returned direct/normalized losses `0.3872614503/0.3872614205`, absolute difference `2.98e-08`, and `equivalent=True`.

### Controlled variables & Baseline preservation

- Formal keys, dataset settings, four seeds, model set, 50 epochs, deferred test protocol, and reference-first ordering are unchanged.
- The original invalid run is not treated as evidence and will be archived before retry. The successful retry remains discoverable at the same formal canonical path.
- The directory is not exposed as a Git worktree, so legacy-path localization cannot be proven via Git diff. Direct inspection found changes confined to Stage S1 orchestrator/tests plus this review report.

### Fail-fast enforcement

- Unknown or near-match dataset names raise instead of silently mapping.
- Missing/invalid/self-referential artifact inventories fail.
- Hash finalization failure stops the queue before `validate_run`.
- Any invalid canonical artifact blocks unless `--rerun_failed` explicitly archives it.
- Same-size/restored-mtime content mutation can no longer evade artifact validation.

No broad exception suppression or silent fallback exists in the reviewed execution path.

## 4. Over-Engineering & Code Bloat Audit

The parent-owned finalizer, atomic replace, append-only attempt directory, and fresh artifact digest are small, local mechanisms directly required for trustworthy resume behavior. Fresh hashing at the validator trust boundary may add repeated I/O during block audits, but caching untrusted artifact digests by only size/mtime was scientifically unsafe. No production framework, registry, compatibility layer, or speculative abstraction was introduced.

## 5. Bugs / Deviations Found

### CRITICAL — corrected during review

**R5-C1 — Artifact tampering could evade validation through the process-local hash cache.**  
Submitted behavior: `_file_hash` keyed cached SHA-256 values only by absolute path, file size and `mtime_ns`. After finalization/validation populated the cache, a same-size content replacement with the old mtime restored produced a different real SHA-256 but `validate_artifact_hash_inventory` returned `[]`.  
Correction: add a fresh trust-boundary mode that evicts all cache generations for the artifact path and reads the file bytes; use it for every declared artifact validation. Add matching pytest and EDDAenv smoke regressions.

### MAJOR

None remaining.

### MINOR

None remaining.

## 6. Corrections Made

1. Updated `stage_S1_orchestrator.py` so artifact validation cannot accept a stale cached digest (`:93-113,250`).
2. Expanded the retry test to prove two append-only attempts preserve history and keep the canonical path (`test_stage_S1_acceptance.py:47-66`).
3. Added exact positive/negative canonical identity cases, atomic-temp cleanup assertion, and same-size/restored-mtime corruption regression (`test_stage_S1_acceptance.py:331-383`).
4. Strengthened the EDDAenv standalone hash-closure smoke with the same cache-evasion corruption case (`run_stage_S1_smoke.py:80-114`).

These changes do not redesign the experiment or modify any research algorithm.

## 7. Verification Evidence and Remaining Risks

- EDDAenv/CUDA standalone smoke: **PASS** (`PASS: Stage S1 CUDA forward/loss/backward, objective, tie, and semantic mapping smoke`).
- EDDAenv isolated real-artifact full validator replay: **PASS**, `ok=true`, zero reasons, CUDA available.
- EDDAenv untouched failed-artifact validation: **correctly FAILS CLOSED** with stale stdout/finalization/old-code evidence and no dataset identity mismatch.
- Focused non-Torch pytest suite: **17 passed, 2 skipped**. Skips are Torch-dependent in system Python and were covered by the EDDAenv CUDA smoke.
- EDDAenv manual CUDA objective check: **PASS**.
- EDDAenv syntax compilation: **PASS**.
- EDDAenv 192-task dry-run: **PASS**, zero formal launches.
- EDDAenv does not contain pytest (`No module named pytest`), so pytest was run with system Python and all Torch/CUDA-dependent paths were run directly with EDDAenv.
- A successful formal retry has not yet been executed; per workflow, that belongs to `luna_runner`. The isolated replay verifies the complete current validator contract but is not a formal research artifact.
- On resume, the runner must use `--rerun_failed` so the preserved first attempt is moved to `failed_attempts/attempt_001`; it must then stop again on the first child, finalization, per-run validator, or joint-block audit failure.

## 8. Whether Experiments Are Safe to Run

**YES. Stage S1 is safe to resume the RUN stage.**

Pass 5 gate: **PASS WITH FIXES; blocking=0; `luna_runner` may resume the formal Stage S1 queue using the approved EDDAenv/CUDA launcher and `--rerun_failed`.** This authorizes resume only; it does not authorize bypassing the fail-fast validator, deleting the preserved failed attempt, or analyzing results before the RUN stage completes.

---

# REVIEW REPORT — PASS 6

**Review pass:** 6  
**Scope:** Independent review of the Windows resume/archive failure at formal progress 63/192, the preserved `attempt_002`, canonical placeholder state, locking, child-handle lifetime, archive transaction, queue state, and concurrent/repeated resume behavior. No formal run was launched and no scientific result was analyzed.  
**Runtime:** Windows, `C:\Users\86199\.conda\envs\EDDAenv\python.exe`, PyTorch 2.0.0, CUDA 11.7, NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Verdict

**PASS WITH FIXES**

**Blocking issues after reviewer corrections: 0. RUN authorization: GRANTED FOR SAFE RESTART/RESUME.**

The submitted correction had the correct high-level ordering—classification/archive after the atomic stage lock, a post-`Popen` child marker, transactional staging, owner-token lock release, and atomic queue replacement—but independent Windows testing found three additional blocking defects:

1. a canonical probe sharing violation was swallowed and treated as empty/pending;
2. real `shutil.move` behavior on WinError32 left a partial staging file even though the synthetic monkeypatch test passed;
3. dry-run could overwrite a live formal queue manifest, and a max-run-limited queue could be mislabeled `COMPLETE` with pending tasks.

All three are corrected locally and covered by positive/negative tests. The actual incident directory and `attempt_002` remain unchanged.

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Independent verification |
|---|---|---|
| Original WinError32 incident | PASS | The preserved queue records `launched=64`, Windows return code `4294967295`, and failed key `joint|movielens|d1_large|m3oe|2025`. The canonical directory contains only a 476-byte launch-header `stdout.log`; it has no `[stage_S1] child_started` marker. `attempt_001` and `attempt_002` each retain the same 476-byte evidence with SHA-256 `d8ff1d...0f53a`. |
| Preserved canonical/attempt_002 | PASS | Read-only classification of the real incident returns `pending=1`, `blocked=0`; hashes before/after are identical, archive set remains exactly `attempt_001/attempt_002`, and no staging directory is created. |
| Classification/archive after atomic lock | PASS | Formal `main` acquires the stage lock before publishing `RUNNING` and before `classify_task_state` (`stage_S1_orchestrator.py:1495-1500`). A test reads the queue during classification and proves both `RUNNING` and the lock file are already present (`test_stage_S1_resume.py:286-311`). |
| Canonical inspection sharing violation | PASS WITH REVIEWER FIX | Submitted `_canonical_attempt_entries`/stdout probing converted `OSError` to “no material attempt,” permitting another launch. Probe errors now propagate and classification emits blocking `canonical_attempt_probe_busy[:winerror]` without creating an archive (`stage_S1_orchestrator.py:1103-1116,1144-1165`; test `:98-116`). |
| Empty/launch-only canonical behavior | PASS | Missing, empty, and stdout-only-without-child-marker directories remain pending and are not archived. A stdout marker is material only after successful `Popen`. Existing tests and EDDAenv smoke pass. |
| Successful `Popen` marker | PASS | `_run_child_process` writes and flushes `[stage_S1] child_started pid=...` only inside a successfully entered `Popen` context (`stage_S1_orchestrator.py:1441-1458`). The real child fixture contains both the marker and child output. |
| Process/stdout handle closure before finalization | PASS WITH REVIEWER HARDENING | Child wait and context exit happen inside `_run_child_process`; the Windows process HANDLE wrapper is explicitly closed after context exit, and the outer stdout context closes before return (`stage_S1_orchestrator.py:1441-1468`). Tests prove `entered → waited → exited → handle_closed`, then successfully archive the redirected stdout immediately (`test_stage_S1_resume.py:198-239`). Parent calls finalization only after this function returns. |
| WinError32 staging/rollback | PASS WITH REVIEWER FIX | The submitted `shutil.move` test passed, but a real held Windows stdout produced WinError32 and left `.staging_attempt_*` because `shutil.move` fell back to a partial copy. Entries now use same-volume atomic `os.replace`, rollback also uses non-overwriting atomic replace, and the completed staging tree alone is renamed to `attempt_NNN` (`stage_S1_orchestrator.py:1168-1237`). Real WinError32 now leaves both canonical files restored, `attempt_002` unchanged, and no `attempt_003` or staging residue. |
| Append-only attempt numbering | PASS | New attempts use `max(prior attempt number)+1`, so historical gaps are not backfilled and chronology is not falsified (`stage_S1_orchestrator.py:1188-1197`). Existing attempts are never overwritten. |
| Owner token, fsync, safe release | PASS WITH REVIEWER HARDENING | Lock creation uses exclusive `x`, UUID owner token, flush and `fsync`; failed write/fsync removes only a partial lock carrying the same token. Release removes only the currently owned matching token and leaves a replacement owner's lock untouched (`stage_S1_orchestrator.py:1283-1330`; tests `:142-196`). Concurrent acquisition has exactly one winner. |
| Atomic queue manifest | PASS WITH REVIEWER HARDENING | Queue JSON is written to a unique temp file, flushed/fsynced, and atomically replaced; failure preserves the prior complete manifest and removes the temp (`stage_S1_orchestrator.py:1399-1416`; test `:242-264`). |
| RUNNING/FAILED/COMPLETE accuracy | PASS WITH REVIEWER FIX | Classification observes `RUNNING` under lock. Child failure records `FAILED_CHILD`, return code and launch count. `COMPLETE` is emitted only at `pending_after=0`; a max-runs cutoff with pending work is `PARTIAL_MAX_RUNS` (`stage_S1_orchestrator.py:1497-1584`; tests `:286-336`). Finalization, validation, block-audit and unexpected failures retain distinct terminal states. |
| Dry-run concurrency | PASS WITH REVIEWER FIX | Dry-run now takes the same stage lock before publishing `DRY_RUN`, so it cannot overwrite a live `RUNNING/FAILED` manifest (`stage_S1_orchestrator.py:1478-1490`; test `:267-283`). |
| Repeated/concurrent resume | PASS | Repeated classification does not manufacture a second archive once canonical contents are gone. Concurrent archive has one winner; concurrent stage lock has one winner. Current-valid classification never archives or overwrites its sentinel (`test_stage_S1_resume.py:73-85,119-164,338-348`). |
| 192-task matrix | PASS | Final EDDAenv dry-run: state `DRY_RUN`, 192 tasks, 192 unique keys, 48 joint, 144 references, zero launches, and no residual stage lock. |
| Scope exclusions | PASS | No model, objective, routing, checkpoint, report, Momentum, Curriculum, Safe-MTL, Stage 2, or scientific analysis behavior was added. |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Resume changes do not alter M3OE/MMoE/PLE forward logic, task/domain indexing, semantic initialization, BCE, AUC/LogLoss, checkpoint selection, or deferred-test semantics.
- EDDAenv/CUDA smoke passed all three backbones in D=3 and D=1, finite forward/direct-BCE/backward, device consistency, semantic mappings, raw metric validation and logging equivalence.
- Direct objective oracle remains equivalent with absolute difference `2.98e-08`.

### Gradient flow & Loss participation

- No training-graph code changed in this pass.
- All model paths retain connected CUDA gradients in the standalone smoke.
- Resume/archive operations occur before child launch or after child/process/stdout closure and therefore cannot participate in or silently perturb optimization.

### Controlled variables & Baseline preservation

- Matrix, seeds, 50 epochs, datasets/settings, architectures, reference ordering, splits and checkpoint protocol remain frozen.
- Valid runs are skipped before any archive path. Invalid attempts are moved append-only and never overwritten.
- Because formal provenance requires the exact `stage_S1_orchestrator.py` hash, the necessary pass-6 repair intentionally invalidates the 63 artifacts produced under the prior orchestrator revision. A checked completed key now fails only with `code_hash_mismatch:stage_S1_orchestrator.py`. Safe resume must therefore archive and rerun those stale-code artifacts; mixing them with new-code artifacts or manually editing hashes is forbidden.

### Fail-fast enforcement

- Busy/unreadable canonical paths are BLOCKED, not pending.
- Real sharing violations restore canonical evidence and expose no completed attempt.
- A second orchestrator or dry-run cannot pass the exclusive stage lock.
- Lock/queue write failures preserve the prior state and do not silently continue.
- Child, finalization, validator, block-audit and unexpected orchestrator failures publish distinct terminal states when the queue path remains writable.

No silent fallback remains in the reviewed Windows resume path.

## 4. Over-Engineering & Code Bloat Audit

The additions are localized to the Stage S1 orchestrator and resume tests. Exclusive lock files, UUID owner tokens, same-volume rename transactions, one child marker and atomic queue replacement directly protect experiment identity and evidence. No registry, service layer, production monitoring system, or speculative abstraction was introduced.

## 5. Bugs / Deviations Found

### CRITICAL — corrected during review

**R6-C1 — Canonical sharing errors were misclassified as empty pending.**  
`os.listdir` or stdout-open errors were swallowed, allowing launch into a path potentially owned by another process. They now block with explicit probe evidence.

**R6-C2 — Real WinError32 left a partial staging artifact.**  
`shutil.move` fell back from failed rename to copy and left a partial destination. Same-volume `os.replace` eliminates copy fallback; real Windows handle testing now proves complete rollback and zero staging residue.

### MAJOR — corrected during review

**R6-M1 — Queue state could be logically overwritten or mislabeled.**  
Dry-run could overwrite live queue state without the stage lock, and max-runs cutoff emitted `COMPLETE` with pending work. Dry-run is now locked and partial completion is explicit.

**R6-M2 — Windows process HANDLE closure relied on CPython object destruction.**  
The handle wrapper is now explicitly closed after wait/context exit and before returning to finalization.

### MINOR

None remaining.

## 6. Corrections Made

1. Made canonical inspection errors fail closed as BLOCKED.
2. Replaced archive `shutil.move` with same-volume `os.replace`, added safe atomic rollback, and made attempt numbering monotonic append-only.
3. Hardened lock acquisition failure cleanup, token-safe release tests and lock `fsync` evidence.
4. Added queue-file `fsync`, accurate partial/complete state, and dry-run locking.
5. Explicitly closed the Windows child process HANDLE after context exit.
6. Added real/synthetic WinError32, probe-busy, owner-token, fsync-failure, process-handle, atomic queue, dry-run concurrency, queue-state, valid-preservation and repeated/concurrent-resume tests.

These corrections do not redesign the experiment.

## 7. Verification Evidence and Remaining Risks

- Actual incident read-only audit: **PASS**, `attempt_001/attempt_002` unchanged; canonical is launch-only pending; no staging/attempt duplication.
- Real Windows held-file WinError32: **PASS after fix**; canonical restored, `attempt_002` unchanged, no `attempt_003`, no staging residue.
- Focused system-Python suite: **33 passed, 2 skipped**. Skips are Torch-dependent and covered by EDDAenv.
- EDDAenv/CUDA standalone smoke: **PASS**.
- EDDAenv objective oracle: **PASS**.
- Syntax compilation: **PASS**.
- EDDAenv dry-run: **PASS**, 192/192 unique, 48 joint + 144 references, zero launches, state `DRY_RUN`, no residual lock.
- The repository is not exposed as a Git worktree, so localization was verified by direct file inspection rather than Git diff.
- Resume will not continue from artifact 64: exact provenance makes prior-code artifacts invalid after this orchestrator repair. They remain recoverable under append-only `failed_attempts`, but the formal matrix must be regenerated under one code revision.
- No formal child or scientific report was run during review pass 6.

## 8. Whether Experiments Are Safe to Run

**YES, with a provenance-driven full formal regeneration.**

Pass 6 gate: **PASS WITH FIXES; blocking=0.** `luna_runner` may invoke the approved EDDAenv/CUDA orchestrator with `--rerun_failed`. It must allow the validator to archive/rerun all stale-code canonical artifacts, preserve existing `attempt_001/attempt_002`, and stop immediately on `BLOCKED`, `FAILED_CHILD`, `FAILED_FINALIZATION`, `FAILED_VALIDATION`, `FAILED_BLOCK_AUDIT`, or `FAILED_ORCHESTRATOR`. It must not edit code hashes, reuse old-code results as current formal evidence, delete archives, or begin scientific analysis before the complete 192-run matrix validates.


---

# REVIEW REPORT — PASS 7

Date: 2026-09-19. Scope: coder's post-pass-6 revision — stale-lock (dead-PID) reclaim in `experiments/stage_S1_orchestrator.py` plus 7 new tests in `experiments/tests/test_stage_S1_resume.py`. Reviewer made no code changes (read-only pass; only this report was appended).

## 1. Verdict

**PASS**

**Blocking issues: 0. RUN authorization: UNCHANGED — the pass-6 grant for a safe `--rerun_failed` restart/resume stands.**

All coder claims were independently re-verified, not transcribed: the new reclaim code was read line-by-line, the full 5-file pytest suite was re-run (43 passed), the EDDAenv/CUDA standalone smoke re-run (PASS), the EDDAenv dry-run re-run (192, exit 0, no residual lock, queue `DRY_RUN`), and two reviewer-devised live Windows probes (self-PID refusal; real dead-PID reclaim after process-handle release) both behaved correctly.

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Independent verification |
|---|---|---|
| Pass-6 fixes still present (6/6) | PASS | os.replace archive + rollback (`stage_S1_orchestrator.py:1211,1216,1233`); owner-token release (`:1470-1488`); atomic fsync'd queue write (`:1557-1574`); dry-run under stage lock (`:1640-1646`); probe fail-closed `canonical_attempt_probe_busy` (`:1106-1116`); explicit child HANDLE close (`:1622-1625`). All sampled in current file; pass-6 tests still pass. |
| `_pid_is_deterministically_dead` correctness | PASS | Read at `:1284-1339`. Windows: non-zero `OpenProcess` handle → alive; `ACCESS_DENIED(5)` → alive; `INVALID_PARAMETER(87)` → dead; every other outcome (ctypes import failure, OSError, GetLastError failure, unknown error) → alive/refuse. POSIX: only `ProcessLookupError` → dead. Reviewer live probe: a freshly exited child PID with the parent's process handle still open is judged **alive** (conservative, correct — the PID object is not yet reusable); after closing the handle the same PID is judged dead and the lock is reclaimed with audit. |
| `_try_reclaim_stale_lock` fail-closed coverage | PASS | Read at `:1389-1427`. Refuses on: not-a-file, unreadable/invalid JSON, non-dict payload, `stage != "S1"`, missing/empty token, non-integer/`<=0` PID, self PID, dead-check False, `os.remove` failure. Each refusal leaves the lock untouched and surfaces the original fail-closed `RuntimeError` from `acquire_stage_lock` (`:1444-1445`). Reviewer live probe: self-PID lock refused, lock preserved, no reclaim log written. |
| Reclaim audit (`_record_lock_reclaim`) | PASS | Read at `:1342-1386`. Append-only JSON list, unique temp file + flush + fsync + `os.replace`; write failure is swallowed by design and never blocks reclaim. Real log at `experiments/stage_S1_raw_results/.stage_S1.lock_reclaims.json` contains the genuine pid=32028 incident reclaim plus coder's fake-lock smoke entry; both well-formed. Reviewer's own dry-run added **no** event (lock acquired cleanly — correct). |
| TOCTOU: death-check → delete → retry `open("x")` | PASS (with minor note) | The retried exclusive `open("x")` (`:1446-1453`) guarantees exactly one winner after a reclaim: a concurrent acquirer that legitimately recreated the lock between our delete and retry causes a fail-closed `RuntimeError`. Concurrent reclaimers race on a single `os.remove`; the loser gets `OSError` → refuse. See Minor R7-M2 for the residual micro-window. |
| PID-reuse safety | PASS | If a dead PID has been recycled by any *alive* process, `OpenProcess` succeeds (or ACCESS_DENIED) → dead-check False → reclaim refused. The unsafe direction (reclaiming a live owner's lock) requires the recorded PID to be dead at check time; a lock written by a live orchestrator always names a live PID. Audit log retains previous pid/token plus reclaimer pid/time for forensics. |
| Classification-in-lock invariant preserved | PASS | Reclaim executes entirely inside `acquire_stage_lock` before the lock is granted; `classify_task_state` and archiving still run only after acquisition (`:1653-1658`). A provably dead former owner cannot be inside classification, so reclaim cannot bypass the invariant. |
| Stale queue-manifest fields at formal RUN | PASS | `_write_queue_manifest` builds a fresh payload (`{"stage","expected_run_count","tasks",**extra}`, `:1557-1559`) — it replaces, not merges. The formal path immediately rewrites the manifest with `resume_state="RUNNING"`, fresh `lock_path`/`lock_pid` (`:1655-1656`); stale `dry_run`/`launch_count`/old `lock_pid=32028` keys are dropped. Current queue reads `DRY_RUN`, 192 tasks, launch_count 0 — consistent. |
| Formal RUN path unblocked | PASS | Chain read: `main` → `runtime_preflight` (`:1649`; lock-absence check `:1279-1281` — lock currently absent, verified on disk) → `acquire_stage_lock` (reclaim available if needed) → `RUNNING` write → `classify_task_state(rerun_failed)` → per-task launch/finalize/validate. `--rerun_failed` archiving is append-only `attempt_NNN` with monotonic numbering (`:1189-1197`). Reviewer dry-run exited 0 with no residual lock. |
| Coder-claimed test suite | PASS | `venv\Scripts\python.exe -m pytest` (5 files, `--basetemp=./.pytest_temp`): **43 passed** (reviewer run, not coder's number). EDDAenv `run_stage_S1_smoke.py`: **PASS**. EDDAenv `--dry_run`: `dry-run matrix 192`, exit 0, no `.stage_S1.lock` residue, queue `DRY_RUN`. `py_compile` of all 12 S1 modules + tests: OK. |
| Scope discipline | PASS | Only `stage_S1_orchestrator.py` (mtime 2026-09-19 11:20) and `test_stage_S1_resume.py` (11:21) carry this round's mtimes. Stronger evidence: SHA-256 of the current `stage_S1_run.py`, `stage_S1_models.py`, `stage_S1_data.py`, `stage_S1_protocol.py`, `stage_S1_report.py`, `stage_S1_metrics.py` each **byte-match** the `code_hashes` attested in an existing completed formal `run_manifest.json`; only the orchestrator hash differs, exactly as expected from pass-6+pass-7 changes. No model/loss/data/protocol/report algorithm code was touched. |

## 3. Issue List

### Blocking

None.

### Major (non-blocking for the authorized start)

**R7-M1 — `runtime_preflight` refuses an existing lock without attempting reclaim, so the new reclaim logic is unreachable on the formal RUN path when it matters most.**
Evidence: `stage_S1_orchestrator.py:1279-1281` raises `RuntimeError` on lock presence *before* `acquire_stage_lock` (`:1653`) is ever called. Today this is moot — the pid=32028 lock has already been reclaimed and verified absent — but after any future mid-matrix crash that strands a lock, an orchestrator restart will die in preflight, and the intended recovery is: run `--dry_run` once (reclaims the dead-PID lock), then restart the formal run. Failure direction is safe (refuses rather than double-runs) and a working recovery exists, so this does not block the runner. Minimal fix for coder: have `runtime_preflight` call `_try_reclaim_stale_lock` (or simply drop its lock check and let `acquire_stage_lock` be the single lock authority) before raising.

### Minor

**R7-M2 — Reclaim TOCTOU micro-window: token not re-verified before `os.remove`.**
Evidence: `:1420-1424` deletes the path based on a payload read moments earlier. A brand-new orchestrator could reclaim and rewrite the lock in between, and our delete would remove its fresh lock (its token-safe release later protects the *next* owner, but mutual exclusion is briefly void). Exploitation requires PID recycling plus two orchestrators racing inside a microsecond window on a single-user workstation; the retried `open("x")` bounds the damage to a fail-closed error for the loser in all realistic interleavings. Minimal fix: re-read the file immediately before `os.remove` and require the token to equal the checked payload's token.

**R7-M3 — `ctypes.GetLastError` used without `use_last_error=True`.**
Evidence: `:1309-1321` (`ctypes.windll.kernel32` + `ctypes.GetLastError()`). Without `use_last_error`, an intervening ctypes/FFI call could in principle clobber the thread's last-error; the only unsafe outcome would be a spurious 87 for a live process. Reviewer live probes on this machine produced correct verdicts in both directions, and the mock tests pin the intended mapping. Minimal fix: `kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)` and `ctypes.get_last_error()`.

## 4. Remaining Risks (accepted)

- PID 0 / idle-process PIDs are judged dead by the Windows API, but the orchestrator never writes PID 0 and reclaim requires `pid > 0` anyway — doubly guarded.
- POSIX branch is mock-verified only; the formal matrix runs on Windows/EDDAenv, so this is informational.
- The reclaim audit log is forensic, not authoritative; its external deletion never blocks recovery (intended design, re-confirmed acceptable).
- R7-M1/M2/M3 above are returned to the coder as improvement items; none gates the run.

## 5. Whether Experiments Are Safe to Run

**YES.** Pass 7 gate: **PASS, blocking=0.** The stale pid=32028 lock is gone (reclaim audited), the lock lifecycle was observed end-to-end (acquire → DRY_RUN publish → clean release, no residue), and the full launcher chain preflight → lock → classification → launch reads correctly with `--rerun_failed` append-only archiving intact.

`luna_runner` may invoke the approved EDDAenv/CUDA orchestrator with `--rerun_failed` under the pass-6 constraints (archive/rerun stale-code artifacts, preserve `attempt_001/attempt_002`, stop on any `BLOCKED`/`FAILED_*` state, never edit hashes or mix code revisions). Operational note for the runner: if a future crash strands `.stage_S1.lock`, the formal restart will refuse in preflight (R7-M1) — run one `--dry_run` first to reclaim the dead-PID lock, then resume.

**Ready for runner: YES**
