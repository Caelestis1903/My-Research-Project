# REVIEW REPORT — corrected S5 implementation

Reviewer: `/root/s5_review_repair`. Date: 2026-11-23. Scope: independent code review of the corrected S5 scalar freeze/evaluate/report module, its dependencies and actual Validation artifacts. This reviewer made no implementation changes.

## 1. Verdict: PASS

The corrected implementation satisfies the reviewed S5 contract. This accepts code only. It does not approve the scientific selection, the Gaussian control audit, FINAL_FREEZE, actual Test access or final study delivery. The former code receipt is preserved as `S5_REVIEW_RECEIPT_SUPERSEDED_2a297c1c.json`; its historical review remains `REVIEW_REPORT_S5.md`.

Inputs independently read: plan 00/05/09/10, the design specification, the final coder report, `S5_GUARDIAN_CORRECTION.md`, actual `final_evaluation.py` and focused tests, `results.py`, `state.py`, `protocol.py`, reuse and guardian interfaces, the pinned numerical evaluator, current accepted results, accepted A decision tables and Validation-only historical endpoints. Source SHA-256 is `f2a41c88f0c4d147925437afe0c18341c1d5b45be9492236a9d8b11d7f28f2c8`; coder report SHA-256 is `cc16366e9038b423989240521ae2e10514f014653531770efd8e16423fc9ca19`.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Correct current guardian lookup | For `<attempt>/runs/<key>/result.json`, `parents[2]` is `<attempt>`, where `guardian.json` is stored. `generation`, `get` and `status` consistently use that directory. The regression creates the actual directory depth and exercises production generation calculation. Live A10 and unaugmented A21 readbacks independently passed. |
| Accepted selected-state identity | `accepted_result` checks the accepted result/COMPLETE_VERIFIED pair, job/config/source identity, all 50 owner-generation Validation epochs, epoch/RAM scalar links and the earliest maximum with matched selected metrics. `selected_ram` checks current owner generation, accepted transaction, completed registry, job, data/group/attempt, selected epoch, metrics and tensor hash. The compact selected-only guardian shape is supported without pretending it contains the final optimizer. |
| Narrow optional G technical pause | Only G_FROZEN can pass `terminal_gates` as TECHNICAL_BLOCKED. It requires the specifically located, hash-bound source control audit and independent G_CONTROL PASS/PASS WITH FIXES receipt with no blocking issues. The audit must record a real prescribed-seed coverage gap on the two datasets, author, reason and the original trigger hash; reviewer must differ from audit author. The bound trigger must be Validation-only, genuinely TRIGGERED and identify an A method. No G results are accepted on this path, and every proposed final candidate must equal that unaugmented A method. This implements plan 00's instruction to pause the Gaussian supplement on control identity mismatch, without turning missing controls into a scientific negative. |
| Other branches and main matrix preserved | All other branches still require COMPLETE or NOT_TRIGGERED, reasons and bound evidence. A decision-chain triggers, completed B search's full 60-job coverage, AB/G completed-result coverage and B core/triggered pulse evidence remain checked. An existing TRIGGERED G receipt prevents declaring G NOT_TRIGGERED. An augmented candidate requires COMPLETE G_FROZEN and its G-D0 control; TECHNICAL_BLOCKED cannot admit it. |
| Unaugmented A21 final matrix | A read-only diagnostic proposal reflecting the existing A21/.05 decisions produced nine arms and eleven directed comparisons per dataset: A00/A10/A20/A30, all three published baselines, A21/A22; it includes A21−A10, A21−A00, A21−strongest published, A20−A21/A22 and A21−A22. No G arm/control is silently substituted. This diagnostic was held in memory and was not a selection handoff or freeze. |
| Actual endpoint pairing and baseline preservation | All 90 Validation rows joined: 58 archived and 32 accepted local endpoints. Every internal trajectory/model/epoch matched the accepted A decision tables, and all 28 reused core endpoints matched S0 identities and metrics. Published baselines retain selected online readout, internal arms selected fixed EMA. The immutable Git source and natural population checks are unchanged. No training or baseline source was edited. |
| Validation-only selection and authority | `authority` requires current source/report/design hashes and a distinct selection review bound to both selection and current code receipt. Freeze enforces the declared method parameters, matching lambda/eta controls, common form/scope, A decision identities and an independent positive Validation increment on an eligible target. A code receipt does not create scientific authority. |
| One-shot Test and restart semantics | FINAL_FREEZE remains immutable and binds methods, parameters, selected identities, comparisons, source evidence and reviewed code. `frozen` rechecks the evidence. Evaluation preflights pending RAM states, reloads/verifies each immediately before scoring, serializes access with a lock and deduplicates physical states through scalar receipts. A durable completed receipt is reused; scoring intent without a durable result blocks automatic rescoring. NO_CANDIDATE rejects evaluation. Neither correction bypasses any of these gates. |
| Mathematical formulas and final decisions | Five-seed paired deltas, sample SD and descriptive t4 radius `2.776445 * SD / sqrt(5)` remain correct. Macro, cell, domain/task, traffic and available pooled metrics remain distinct. Both six-cell guards use mean delta ≥−.001; total .002/.003 thresholds and independent component increments remain separate. No new confidence-interval or all-seeds-positive gate was introduced. |
| Tensor shapes, masks, routing, weights and gradients | S5 strictly loads and hashes the frozen model state, then uses the pinned evaluator under `torch.no_grad()`. S5 adds no optimizer step, training gradient, parameter routing, data-loss weighting or mask change. The correction changes only state location and scalar branch admission. |
| RAM-only and scope | New artifacts are scalar JSON/CSV and plots. No model/optimizer tensor was written to disk; guardian access in this review used only status/get. No training, source G audit, scientific selection, freeze, Test scoring or extra experiment was performed. |

Independent checks with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `pytest -q .../tests/test_final_evaluation.py .../tests/test_results.py --disable-warnings`: **9 passed**, 14 plotting deprecation warnings (six S5 checks and three results checks).
- Production Validation adapters: **90 endpoints**, 58 archived/32 local, all selected internal identities joined; **28 core/S0 matches**.
- Actual MovieLens seed 2023 A10: trajectory `6f458790c6be9480e837d93fc4fbdd4fc831e1aee4817375b6f23845ccd71cd1`, E20, selected tensor SHA-256 `b5ff208db80e1a33f4043621d9a78e847ac9e60d714e74ff3279a23f3a531170`.
- Actual MovieLens seed 2023 A21: trajectory `4d7ec2bae1fb7aa953fa9e9c9de03292efe5f779a109dd7d0e5d7fd70014b9ee`, E20, selected tensor SHA-256 `8e8034277dafec5003e1d43cb2b43bcb5ec3a7444a6545dbb766f1cd6f11fd52`.
- Both live readbacks resolved owner generation `f2a6427b61b812f225f910e6991ede53507fd72383c1a03eaac50db9ea01d353` and 107 state-dict entries. These are observations at review time, not a substitute for freeze/evaluation's fresh verification of every selected state.

## 3. Bugs/deviations found

**CRITICAL:** none found in the reviewed S5 component.

**MAJOR, corrected before this review:** the previous `parents[3]` guardian lookup addressed the directory above the actual attempt, blocking production selected-state readback. The coder's `parents[2]` correction is verified with both the production-depth regression and actual accepted RAM states. No unresolved MAJOR issue found.

**MINOR, carried forward:** `endpoints.csv` omits plan 07's constant `study_id`; the directory, freeze and lineage still identify the study. S5 costs are evaluation-only and do not constitute the plan-wide physical-attempt/cost appendix. The final delivery must supply that consolidated schema from existing evidence, preserving unavailable measurements as missing. The separate `final_evaluation` module remains the working S5 CLI entrypoint, as documented in the coder report.

The G technical-pause exception is a permitted runbook behavior, not a change to the experiment hypothesis or a waiver for the main matrix.

## 4. Corrections made

No code corrections. This reviewer wrote this report, preserved the former receipt byte-for-byte under a superseded filename and issued a new current source-bound S5 code receipt. The corrected source and test belong to the coder.

## 5. Remaining risks

The actual `SOURCE_G_CONTROL_AUDIT.json` and independent `G_CONTROL_REVIEW_RECEIPT.json` were absent during this review. Their factual coverage/identity conclusions are not accepted here; a separate independent control review must validate them before the technical-pause path is used. The live G trigger does identify unaugmented A21 and Validation-only selection.

No actual S5 selection was approved, no production `authority`/`terminal_gates`/`freeze` closure was executed, and only two live selected states were read back. The distinct scientific selection review must bind the current code receipt and all actual branch evidence; runtime must verify all selected states. Changed source/report hashes invalidate this acceptance. An indeterminate Test intent remains a technical blocker requiring recovery of its scalar outcome.

The full research report, mechanism/Validation evidence including unselected registered configurations, all-stage budget/recovery appendix, research context/ledger updates and final independent delivery review remain the foreground orchestrator's delivery obligations, as described in the preserved earlier report and plan 10.

## 6. Whether experiments are safe to run

**Yes for the corrected S5 code, once its separate actual control/selection/stage prerequisites pass.** This code receipt itself does not authorize Test access or additional training. The foreground orchestrator may subsequently prepare the Validation-only selection and obtain its independent review. The Gaussian supplement remains technically paused until its identity gap is resolved; the reviewed narrow path permits only the unchanged eligible unaugmented A candidate to proceed through the normal final-freeze gates.

No Test scoring or scientific selection was performed. Production closure and Gaussian control facts remain unverified in this code-only review.
