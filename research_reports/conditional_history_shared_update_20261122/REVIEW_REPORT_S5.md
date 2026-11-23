# REVIEW REPORT — S5 code

## 1. Verdict: PASS

This is an independent acceptance of the scalar freeze/evaluate/report implementation, bound to the source hashes in `S5_REVIEW_RECEIPT.json`. It is **not** acceptance of a scientific selection, a completed experiment matrix, current selected RAM states, Test access, or the final research delivery. No `S5_SELECTION_REVIEW.json` was created.

Inputs inspected: the actual `final_evaluation.py` and its tests; its local protocol, results, reuse and state dependencies; the core runner/receipt and B gate schemas; `DESIGN_SPECIFICATION.md`; `CODE_IMPLEMENTATION_REPORT_FINAL.md`; and the original 11-22 runbook and implementation/statistics/decision/report contracts. The sibling Git repository supplies historical blobs; this workspace itself has no `.git`.

## 2. Requirement-by-requirement verification

| Requirement | Verification |
|---|---|
| Completed prescribed matrix before freeze | `terminal_gates` checks accepted core audits/checks/reviews/run receipts and all registered jobs, A decision-chain identities/tables, 40 matching B anchors, 20 parent windows, triggered pulse slots and conditional result completeness. Unresolved or technically paused stages cannot close as scientific negatives. Bridge/Gaussian scientific validity remains part of the separate selection review. |
| Validation-only scientific selection | `authority` requires an independently reviewed Validation-only `S5_SELECTION.json`, source-bound code review, author/reviewer separation and resolved selection status. No Test performance is used by `freeze` to choose methods or parameters. |
| Frozen identities and comparisons | Immutable, content-addressed `FINAL_FREEZE.json` binds selection, parameters, comparisons, sources, controls and selected identities. Every production action rechecks bound evidence and numerical source closure. |
| Selected epoch and matched LL | New-state adapter requires all 50 accepted Validation epochs, chooses the earliest maximum over E1–E50, and compares all selected metrics with that same epoch. Historical selected identities are joined to the audited reuse inventory. |
| Current RAM verification | New endpoints require a current guardian generation, completed registry/result identity, selected epoch/Validation metrics and tensor hash. This supports the existing guardian's compact selected-state representation. Evaluation checks all pending states before scoring and reloads each new state immediately before its pass. Synthetic verification passes; no actual forthcoming selected state was accepted in this review. |
| Historical reuse and baseline preservation | Fixed Git blobs and population/preprocessing identities are checked. Published baselines preserve their selected online readout; internal methods preserve selected fixed EMA. Validation/Test joins require matching selected-state/config/source identity. Existing numeric endpoints incur zero new forwards. No baseline is retrained or training algorithm changed by this module. |
| Test isolation | NO_CANDIDATE rejects `evaluate`; candidate evaluation needs frozen, independently reviewed selection. Historical exposure and frozen reuse of the historical population are explicitly distinguished. The original 11-22 final-freeze authority applies; there is no added manual-stop gate. |
| Once-only and restart behavior | Evaluation lock, immutable per-state intent/outcome receipts, freeze hashes and complete receipt index prevent normal duplicate scoring. Completed receipts are reused. A crash before scoring intent is retryable; after intent without a durable outcome the slot is deliberately indeterminate and blocks automatic rescoring. Recovery must restore its scalar outcome or report a technical blocker. |
| Tensor shape, masks, gradients and routing | S5 strictly loads the frozen model state and delegates full-population scoring to the pinned source evaluator, which runs under `no_grad`. S5 introduces no optimizer update, research gradient path, loss weighting, source routing or training mask. Underlying training algorithms are outside this code-only review. |
| Statistics | Five seed units are enforced. Six-cell Macro, domain/task aggregates, traffic weighting and source-provided pooled quantities remain distinct; unavailable pooled metrics stay missing. Sample SD, mean difference, all five deltas, sign counts and descriptive t4 interval use the specified formulas. |
| Final comparisons and decisions | A includes same-dose Adam and simple history controls; B includes ADAPT/STATIC/UNIFORM; AB includes constituents and the corresponding factorial contrast; Gaussian includes its unaugmented parent and G-D0. Both internal and published six-cell guards are reported. Total thresholds .002/.003, independent positive increments and guard −.001 are implemented without a CI/all-seeds-positive requirement. Constituent/config matching must be independently accepted at selection review. |
| NO_CANDIDATE closure | `freeze` validates the completed evidence chain and requires a reason with no dataset selections. `report` writes zero new Test evaluations and separate already-exposed historical core context. The complete A/B Validation/mechanism report must still be assembled from its bound stage evidence. |
| RAM-only training states | This module writes scalar JSON/CSV and plots, not model/optimizer tensors. It reads selected tensors through the existing guardian. |

Independent checks performed with the project Python environment:

- `tests/test_final_evaluation.py`: **5 passed**. Covers immutable/ambiguous access, earliest ties, compact current-guardian readback and wrong tensor rejection, mixed-readout tables/plots, NO_CANDIDATE evaluation rejection and restart consumption of receipts.
- `tests/test_results.py`: **3 passed**. Covers seed-unit statistics, guards, A stage gates and identity freeze.
- Validation-only historical adapter: **68 unique selected rows** independently parsed and indexed: 28 core, 30 published, 10 G-D0. No historical Test adapter was invoked for this check.
- Final module SHA256: `2a297c1c1589dd9a166bbf4acc086a03581cf05abd410327006c8d74ea28d180`.
- Coder report SHA256: `471c0dc659014b897057d0cdf83cba615de434d2f5a44799fc83b9110a0c8b89`.

## 3. Bugs/deviations found

**CRITICAL:** none found within the reviewed component.

**MAJOR:** no blocking implementation defect found within the reviewed component.

**MINOR, nonblocking output-schema gap:** generated `endpoints.csv` omits the constant `study_id` specified in plan 07. Its study is still bound by its directory, freeze and source lineage. S5 `costs.csv` is an evaluation-only subset of the plan-wide cost schema; it does not provide all of `attempt`, `cuda_event_seconds`, `autograd_sweeps` and `process_residency`. Preserve missing measurements as missing and provide the consolidated schema/attempt lineage in the final appendix; do not invent measurements or rerun Test to populate them.

The original `cli freeze/evaluate/report` actions still reject. The separately documented `python -m experiments.conditional_history_shared_update_20261122.final_evaluation ...` commands are the usable S5 entrypoints. This is an explicit integration difference, not a mathematical defect.

## 4. Corrections made

No code corrections made. Only this independent review and its source-bound code receipt were written. The compact-guardian adaptation already present in the final coder source was independently inspected and tested.

## 5. Remaining risks and delivery obligations

The synthetic tests do not exercise a real completed production selection through `authority`/`terminal_gates`/`freeze`, and do not establish live future selected-state availability. The actual S5 selection needs a distinct review binding its source hash and this code receipt; reviewed completion of every triggered branch and current selected RAM readbacks are runtime prerequisites. A later S4 layout change requires a separately reviewed adapter. An indeterminate Test intent requires technical recovery, not automatic rescoring.

`S5_REPORT.json` is a scalar closure/performance artifact, **not the plan's complete FINAL_REPORT**. Before closing this study, the foreground orchestrator must still deliver:

1. Full A five-seed Validation attribution, history/PAIR/simple-control conclusions and historical Test context; full B mechanism and pulse/window evidence, including NO_ACTION and contradictory results; all registered HPO values and controls, including unselected values.
2. The five explicit research answers in plan 10, separating total, component and interaction effects; per-domain/task/seed/stage evidence; baseline-supported difficulty versus domain size; competing explanations, evidence strengths and specific remaining questions.
3. Complete source/selection/run identities and tables; all-stage physical attempt, valid trajectory/replay, virtual/pulse/forward, wall/worker, resource and failure/recovery accounting. S5 costs alone cannot serve as this appendix.
4. Updates to `EXP_AI_CONTEXT.md`, `research_context/experiments/Experimental_Global_Summary.md`, `Experimental_Stable_Findings.md`, `Experimental_Open_Questions.md`, `MDMT_ALGORITHM_METRICS_LEDGER.md` and the applicable experiment index, preserving provenance.
5. A final independent delivery review covering actual results, pairing, freeze, guards, budget and state/recovery evidence, followed by the authorized branch submission and remote commit readback.

These are remaining stage/delivery requirements, not permission to select a candidate from unavailable evidence or to label technical incompleteness NO_CANDIDATE.

## 6. Whether experiments are safe to run

**Code-only acceptance: yes**, under the existing reviewed stage/selection prerequisites. NO_CANDIDATE may use freeze/report once its completed scientific closure is independently reviewed; it opens no new Test. A final candidate may use freeze/evaluate/report only after the separately reviewed actual selection and required current RAM acceptance. This receipt alone authorizes neither new research training nor actual Test access.

No training, Test scoring, live guardian mutation, scientific selection approval or publication was performed. Live selection/state acceptance and full final delivery remain unverified.
