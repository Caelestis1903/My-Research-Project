# Stage S4 CODER IMPLEMENTATION REPORT

## IMPLEMENT of S4 Design Amendment A2 — output finalization and isolated recovery

The A2 changes repair the S4 finalization crash without changing the training calculation. `S3Runner._output_formal_protocol` defaults to the original `formal_protocol_for(self.fresh_role)` call. S4 overrides only that hook and supplies its own arm/seed/settings protocol, so inherited S3 output writing no longer asks the S3 role function to accept `target_only_shared`. S4 compares the expected protocol with observed arguments and optimizer betas before it marks its config and manifest formal. It writes `gradient_step.csv` and all 100 `probe.csv` rows, then hashes both into the S4 manifest. The per-run validator now rejects any S3 role/protocol leakage, unequal expected/observed S4 protocol, incomplete config identity, or config/source mismatch.

`stage_S4_orchestrator.py` adds an A2-only freeze and launch route selected with `--a2`. It checks the frozen A1 queue/state SHA-256, blocked zero-wave state, absent canonical and collector outputs, original abort receipt, and deterministic SHA-256 inventory of all 56 A1 attempt files. It compares all 16 normalized scientific task records and the exact eight ordered waves with A1, requires an empty A2 raw/output root before initial freeze, and freezes the complete 21-entry current producer inventory. `stage_S4_report.py` binds collection to a matched isolated A2 queue/raw/output tuple and writes its output only under `stage_S4_A2_outputs`. A1 collection remains available only for its own matched tuple; no cross-root collection is accepted. No A2 queue or raw/output root was created in IMPLEMENT.

Read-only A1 preflight found the design's queue SHA-256 `c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66`, state SHA-256 `540aeb37729a749585dd018adb3e242caf05c83a5d88dfb647a51675a5f2d554`, 16 tasks, eight waves, zero accepted waves, and 56 retained attempt files. Calling `freeze_a2_queue(allow_create=False)` correctly rejects because the independent reviewer has not approved or frozen the queue.

The full disposable completed-run regression copies an accepted S3 seed-2023 artifact set into a temporary directory, supplies synthetic S4 step/probe records, executes the actual inherited S3 writer and S4 writer (only the already-completed S2 base writer is stubbed), finalizes process-console hashes, and passes the real S4 `validate_run`. The fixture is a structural test only; its synthetic S4 gradient/probe values are never formal research data. The test then rejects a missing or corrupted probe, mismatched observed protocol, partial manifest, altered source hash, and leaked S3 fresh role. A separate A2 test rejects any A1 canonical path, altered A1 attempt bytes, or mixed A1/A2 collector paths. The completed fixture exposed an additional validator bug: the S3 `_read_json` helper rejects a valid empty JSON list, while `nonfinite_inventory.json` is specified as an empty list. S4 validation now loads that one artifact with `json.load` and requires it to equal `[]`; no scientific criterion was relaxed.

Basic checks: `py_compile` passed for the five changed runtime modules and `stage_S3_run.py`; 21 S4 control/orchestration tests passed in EDDAenv, including the complete disposable run validator and adversarial mutations. Formal training and scientific analysis were not run. Independent reviewer acceptance remains required before any A2 freeze or RUN.

Status: implementation complete for independent REVIEW. No formal S4 training or scientific analysis was run.

## Scope and design mapping

The only binding design is `stage_S4_design.md`. This implementation provides exactly four fresh Equal/KuaiRand/natural/M3OE arms `F`, `T`, `N`, `U` for seeds 2022–2025. It reuses the S1–S3 data, model, metric, objective, parameter partition, checkpoint, and artifact paths without editing their sources or historical outputs.

| Design requirement | Implementation |
| --- | --- |
| Frozen 16-key matrix, eight `(F,T)` then `(N,U)` waves | `stage_S4_protocol.py` |
| One-forward decomposition, `N` raw-gradient matching, side-effect-free same-state AdamDelta, `U` FULL moments with scaled shared delta, private FULL gradients | `stage_S4_gradient_controls.py`, `stage_S4_run.py` |
| Main `C` and architecture `A_arch` control; one ordinary optimizer step each; per-step norms, residuals, private update, Adam counters, 50 main and 50 architecture probes per run | `stage_S4_run.py` |
| Source/data/initialization/checkpoint/prediction/step and four-arm pairing gates | `stage_S4_orchestrator.py` |
| Two concurrent independent Python training processes, isolated append-only attempts, two-member validation and promotion, at most one whole-wave retry | `stage_S4_orchestrator.py` |
| Mechanical 16/672/1008 run/raw/paired tables plus steps, probes, aggregates, manifest, raw-only run report | `stage_S4_report.py` |

`AdamDelta` includes coupled weight decay, per-parameter state step, bias correction, installed/default epsilon, and the main or architecture learning rate. It reads the optimizer state using `.get` to avoid initializing an absent state. A disposable Adam clone gate covers named vector/scalar tensors, first and warmed steps, zero and nonzero gradients, both learning rates, and weight decay. The `U` arm checks its ordinary FULL Adam move before rescaling, then checks shared and private parameter deltas against same-state formulas. `N` checks the pre-Adam group gradient norm and FULL direction. Undefined source-zero/reference-positive controls raise and stop. A target-zero count is retained and is rejected if it exceeds 1% of main steps.

Formal queue creation/execution remains after REVIEW. `--reviewer-pass` is required by the launcher; no queue manifest was frozen during IMPLEMENT so reviewed source hashes can be frozen once review is complete.

## Historical context audit

The existing S3 A1 queue is `COMPLETE`; its summary records 24/24 fresh artifacts, 8/8 strict four-way blocks, and zero strict invalid pairs/interactions. The current M3OE forward file SHA-256 equals the S3 audited value `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`. The KuaiRand Equal S3 target-only runs for seeds 2023–2025 pass current per-run read-only validation. Seed 2022 is an earlier manifest schema: its `code_hashes` contains the same forward-file SHA, but the manifest lacks `fresh_role` and explicit `forward_model_source_sha256`; the present S3 per-run validator rejects those missing fields. This limitation is historical context only. No S3 artifact was backfilled or promoted into the formal S4 matrix. S4 freezes the accepted KuaiRand data-file SHA, Equal coefficient-table SHA, and per-seed semantic initialization SHA from the existing S3 manifests, then requires four fresh S4 arms per seed.

## Basic checks

- All five S4 modules and the control test module pass `py_compile` in `EDDAenv` (Python 3.8.20, PyTorch 2.0.0).
- Five local tests pass: disposable installed-Adam reference, one-forward gradient partition, `F/T/N/U` main update, `F/T/N/U` architecture update, and undefined/zero norm cases. The toy runner-step tests check once-only Adam step counters and private FULL deltas.
- No 50-epoch training, raw-result collection, or mechanism interpretation was performed.

## Review focus and boundaries

The reviewer should independently inspect the actual source files and this report, especially same-state AdamDelta on CUDA, `None` gradient handling, source and artifact hashes, per-step matching, private FULL state transitions, and whole-wave failure/recovery. A one-batch CUDA smoke is permitted before formal RUN. Formal RUN requires reviewer `PASS` or corrected `PASS WITH FIXES`. Any formal code correction requires a new source inventory before queue freeze.

## IMPLEMENT repair after independent REVIEW FAIL

The first independent review (`stage_S4_review.md`) found the core four-arm step locally consistent but blocked RUN on the launcher and probe validation. This repair addresses every listed blocking issue without changing the S4 design or starting formal training.

| Review issue | Repair and evidence |
| --- | --- |
| Partial canonical pair after a second promotion failure | `_promote_wave` prepares and validates both copies, records a promotion journal, and rolls the first canonical member back if the second fails. `_recover_promotion` resolves an interrupted journal before any new wave action. Injected failure after the first rename left zero canonical members, retained both source attempts, then allowed a complete pair promotion. A separate simulated crash with one canonical member recovered to `ROLLED_BACK`. |
| Failed child leaves peer running | `_monitor_children` polls both PIDs and terminates a live peer immediately on nonzero exit, then confirms both have exited. A real two-process local test used one immediate exit and one 30-second sleeper; the sleeper was terminated and the test finished under five seconds. |
| No enforced actual overlap | Each S4 Python child now writes its own `process_interval.json` with PID and nanosecond start/finish timestamps. The launcher requires a positive intersection of both intervals and records the exact overlap, process intervals, observed exit times, and at-most-two-process launch. A zero-overlap injected interval is rejected. |
| Canonical stdout incomplete | On successful exit, the launcher appends the full captured child console to the run's `stdout.log`, hashes that file and `process_interval.json` in the manifest, then validates and copies the run. Source attempt console and run remain retained. A regression checks retained bytes and hash. |
| Probe validation checked count only | `validate_probe_trace` requires exactly the first main step and architecture step for every epoch 0–49, unique support, matching step content and fingerprint, identity, finite directional values, defined/undefined cosine semantics, equal-norm gain identity, and directional Cauchy bounds. Missing, duplicated, corrupted-fingerprint, inconsistent-gain, and NaN probes are rejected by adversarial tests. |
| Retry classification too permissive | Only a child `OSError` with an explicitly recorded transient errno (`EINTR`, `EAGAIN`, `ETIMEDOUT`, or Windows 10060) permits the single whole-wave retry from epoch zero. All validation failures and unclassified exits stop for review. The classification and reason are persisted in queue state. Regression covers missing receipt, algorithm assertion, explicit transient error, and scientific/validation success. |
| CUDA Adam clone gate incomplete | The disposable installed-Adam parity gate now runs on the selected formal device. It covers first and warmed states, zero and nonzero gradients, named vector/scalar tensors, both main and architecture learning rates, and coupled weight decay. It passed on the installed CUDA device. |

Repair verification: `py_compile` passed for all S4 modules and both test modules; 12 local unit/fault-injection tests passed in `EDDAenv` (Python 3.8.20, PyTorch 2.0.0). The CUDA installed-Adam clone gate returned `True`. No queue was frozen, no formal S4 training was performed, and no scientific results were analyzed. The changed source inventory must be independently reviewed and frozen only after a new `PASS` or corrected `PASS WITH FIXES` verdict.

## IMPLEMENT numerical repair after first formal-wave fail-fast

The runner's first frozen `(F,T)` wave for seed 2022 stopped with `0/8` accepted waves and `0/16` canonical runs. F raised `S4 norm mismatch shared Adam formula: 1.2112876425028966e-06 != 0.0`; T was terminated by the reviewed whole-wave fail-fast. The frozen queue, queue state, both attempt directories, failure receipt, and console files were read only in this repair. No further formal training was launched and no scientific result exists from that wave.

### Diagnosis

`stage_S4_run.py` compared the Euclidean residual between an observed float32 Adam parameter move and the analytically predicted move by calling `assert_match(residual, 0.0)`. That call made the tolerance `max(1e-6, 1e-4 × 0) = 1e-6`, independent of the magnitude of the predicted update. The binding S4 design instead specifies `max(1e-6, 1e-4 × reference_norm)` for the *predicted update vector*. The observed residual exceeded the mistaken absolute gate by `2.112876425028966e-7`; the failure receipt did not record the predicted group norm, so this report does not infer its exact permitted threshold from the failed attempt.

The frozen S1 optimizer uses ordinary `torch.optim.Adam` with default `foreach=None` and `fused=None`, betas `(0.9, 0.999)`, coupled decay `1e-5`, main LR `0.01`, and architecture LR `0.001`. On the installed PyTorch 2.0.0 CUDA path, the default dispatcher selects foreach and does not select fused. The analytic `AdamDelta` retains the same decay, bias correction, epsilon and per-parameter state, while float32 foreach parameter arithmetic can differ slightly from a separately computed analytic delta. No optimizer setting, update rule, or design tolerance was changed.

### Repair

- Added `assert_vector_match` in `stage_S4_gradient_controls.py`. It computes the vector residual in float64, compares it to `max(1e-6, 1e-4 × ||predicted_delta||_2)`, validates exact `None`/no-update semantics, and reports both residual and predicted norm on failure.
- Replaced the three incorrect zero-reference checks in `stage_S4_run.py`: `U` pre-rescale FULL Adam move, applied shared move for all arms, and private FULL move. The same correction applies to the disposable installed-Adam clone gate. Norm matching for `N` and `U` remains at the original design threshold and is still enforced separately.
- The per-run validator now checks logged shared and private formula residuals against their reference move norms, preserving the same bound at collection time.
- Added regressions showing that a residual above `1e-6` but below the prescribed relative threshold is accepted, and a larger residual is rejected. The CUDA test uses real KuaiRand M3OE parameter shapes and both `C` and `A_arch` optimizers, with first/warmed moments, zero gradient and `None` gradient. It verifies PyTorch's installed default CUDA dispatcher chooses foreach, then compares the actual move to the same-state formula.

Repair checks: 14 S4 control/orchestration tests passed, including the real-device CUDA regression; all changed modules passed `py_compile`. The failed F attempt and frozen queue were not amended. These source edits intentionally cause source-inventory drift against `stage_S4_queue_manifest.json`. The reviewer must approve the numerical repair and an explicit S4 source/queue recovery amendment before any changed formal run. A valid route is a new reviewed S4 amendment queue and isolated raw root for all 16 fresh arms, retaining the original frozen queue and failed attempts as immutable abort evidence. The coder has not created that new queue or resumed RUN.

## IMPLEMENT of S4 Design Amendment A1

This section implements the designer-authored A1 at the top of `stage_S4_design.md` after the subsequent independent FAIL review. No original frozen queue, state, attempt, or accepted S1–S3 artifact was edited. No A1 queue was frozen and no formal training was launched in this IMPLEMENT round.

### Predicted N delta reference

`stage_S4_run.py` now logs `delta_N_predicted_norm` on every main and architecture step, directly from the same-state virtual `Δ_N` computed before the Adam update. `stage_S4_orchestrator.py` uses that logged **predicted** norm as the `N` arm's shared-formula residual reference. `F` continues to use predicted `Δ_F`, and `T/U` predicted `Δ_T` for the applicable formula gate. The threshold remains exactly `max(1e-6, 1e-4 × reference_norm)`; no update rule, optimizer setting, or matching threshold changed. Fixed probes also verify their `delta_N_norm` matches the new per-step predicted norm. A regression constructs an `N` row where a residual of `1.5e-6` would pass if the actual norm (`0.02`) were used but fails under the predicted norm (`0.01`).

### Isolated A1 queue and evidence boundary

`stage_S4_protocol.py` fixes the A1 queue/raw-root paths and the old queue/state byte hashes from the amendment. `stage_S4_orchestrator.py` provides an A1 freeze path, selected with `--a1`, that checks the old queue/state SHA-256, its stopped 0-wave/blocked state, all 16 absent old canonical outputs, absent old collector outputs, and a deterministic per-file hash inventory of the retained old `.attempts` tree. It requires the new A1 raw root to be empty before initial freeze. The new queue records that abort receipt under `aborted_predecessor` and freezes the full current producer inventory, including the audited M3OE forward SHA. It compares old and new 16-task records after removing **only `run_dir`**, and requires the eight-wave list to be exactly equal. A1 training cannot start before its separate queue exists. The A1 route does not read old attempts for promotion or counting, and the A1 retry allowance begins fresh at attempt 1.

`stage_S4_report.py` now accepts only the isolated A1 queue/raw root, validates that queue's source and predecessor binding, and records the A1 queue byte SHA/raw-root path plus explicit old-attempt exclusion in the summary and raw execution report. It does not search or merge the old raw root. The original source-bound queue remains unusable with changed code.

Read-only live preflight in this coder round: old queue SHA `693b46697ba40bf6e9cd03c6e0d4688a3ee7aa80a5d6ce2a442bed5c846fb221`, old state SHA `8ebff1bc35dd7fa5a182cfb3d56b88f7e336bedb66cd93fba588c5bc188c622e`, old matrix 16 keys/eight waves, old canonical count `0`, retained attempt files `4`, attempt inventory digest `4bae95e4e35bb27fceaf3f70fe2ab21f57d24110f9471a1951a701f3ff1a0e18`. The A1 queue path and raw root were absent.

Basic verification: all changed modules passed `py_compile`; 16 S4 tests passed in `EDDAenv`, including CUDA Adam parity, the predicted-N reference regression, and a temporary-directory A1 freeze test that detects old-attempt byte drift. Tests used only disposable fixtures. Reviewer must independently verify the final source inventory, old abort receipt, exact A1 matrix, and all remaining S4 gates before authorizing A1 freeze/RUN.

## IMPLEMENT repair after A1 collector lifecycle REVIEW FAIL

The A1 collector now has the fixed, isolated output root `experiments/stage_S4_A1_outputs`; both its command-line default and `collect` enforce that exact root. Its run report is `experiments/stage_S4_A1_outputs/reports/stage_S4_A1_run_report.md`. The original abort report at `experiments/reports/stage_S4_run_report.md` is never a collector destination. Root-level `experiments/stage_S4_*.csv/json` checks in the predecessor receipt remain exclusively checks for legacy aborted-queue products; A1 products are in the separate output root and cannot trip that check on repeated validation. The predecessor receipt also verifies the original abort report's frozen SHA-256, `52b6d5924fb60c6453e669d6318a11cc064cf8858c770b77440ff30e33a5d30e`, and binds it into the A1 queue receipt.

A disposable-fixture regression freezes an A1 queue, collects the synthetic 16-run matrix twice, revalidates the queue after each collection, confirms the original abort report hash is unchanged, rejects a legacy-root collector request, and rejects subsequent validation after deliberate old-report tampering. This does not train or interpret any model. The old live queue, state, attempts, and report were read-only; the A1 queue, raw root, and output root remain absent in the live workspace. The final changed source inventory requires a new independent REVIEW verdict before freeze or formal RUN.
