# CODER IMPLEMENTATION REPORT

Implementation: `experiments/ema_causal_20261110/`. Source checkout: `ffa3de067fcda5a1293af83a465d741e069493f1`. Source model/loss/Adam/data modules were not edited. This report covers implementation and engineering checks; formal research results are not produced by the coder.

| Task | Implementation and connected execution |
|---|---|
| T00 | `protocol.resolve/audit/prepare/schedule`: pinned source blob verification, five seeds/20 jobs, original cache hash verification, original panels and row splits, frozen configuration and events. Training bundles exclude Test. |
| T01 | `history.py`, `interventions.py`: exact group masks, original endpoint coordinate copies, FP64 actual/history decomposition, sparse capture and score-EMA arithmetic. `trajectory` captures original FP32 F and separately audits ideal FP64 R. |
| T02 | `trajectory.create_session/train_batch/advance_epoch`: source loss and source single Adam/F update; per-step panel q, all prospective selected-epoch history origins, O/F selected states, pulse gradients/menus, H-step pulses and pre-Adam source packets. |
| T03 | `state.Guardian`, `run.worker/train`: immutable anchor packets are committed once, current recovery payload is separate, last complete epoch survives worker failure, guardian-loss replay is bounded, four five-seed blocks release to three selected endpoints. GPU/RAM admission and 4 GiB payload/2 GiB output budgets are enforced. |
| T04 | `block._core/_selection/_fine/_time_groups`: all-anchor S/N and four-window states, five-seed V_A selection, finite family/leaf and group-time contrasts; real full Validation and independent V_B reporting. |
| T05 | `pairs.py/pairs_kernel.cpp`, `Anchor.record/panel_records`: exact int64 full-population transitions without P×N allocation, source AUC closure, fixed without-replacement panel pairs, margin strata, same-pair overlap/cases with original row IDs; compressed indexed logical tables. |
| T06 | `functional.py`, `block._function_contrast/_fusion`: actual complete h and fused cuts, two mixed forward states, FP64 fusion identity and source rounding, five deletions and the selected sixth X-linked deletion, full-function bounded JVP integration, conditional STAR cross. |
| T07 | `trajectory.pulse_observations/rank_gradient`, `bridge.py`, `block._conditional`: full 256×256-per-cell rank risk gradient, fixed all/S/N menus and exact feature definitions, LOSO ridge/direct/static/oracle, deterministic H2/Adam/STAR triggers and real matched/source readouts. |
| T08 | `run.py` public CLI and `run_all.sh`; `tests/` and two-dataset `smoke`. Smoke covers original full and tail batches, model/gradient/Adam/F/RNG/loader purity, RAM epoch transaction and process exit 37, subsequent real update equality, and actual-population costs. |
| T09 | `run.train` waits for all five seeds and calls `block.finalize_block` before releasing anchors; all four blocks and 34,500 logical Adam steps are required. |
| T10 | `analysis.analyze/freeze/evaluate/report`: committed scalar aggregation, explicit orchestrator Q1–Q5 scientific findings gate, input-hash freeze, five-state isolated Test, raw paired table and numerical report. Guardian remains alive until explicit release after report/Test. |

## Engineering checks

- `python -m pytest experiments/ema_causal_20261110/tests -q`: 12 passed on the final code (6.33 seconds). The final check log is in `artifacts/ema_causal_20261110/engineering/tests.log`.
- Six numerical/source-graph tests cover ties and brute-force agreement; group completeness/support in both actual source models; h/fused endpoint equality; fusion algebra; full-function JVP and finite differences; history/score EMA closure.
- Three bridge tests cover exact penalty/tie conventions, 72-row LOSO/five-seed statistics, and held-out/validation leakage and missing-feature treatment.
- Two block tests cover frozen coarse selection and X descriptors. One state test injects a worker exit after a RAM epoch transaction.
- Real-data smoke receipts: `SMOKE_MOVIELENS.json`, `SMOKE_KUAIRAND.json`, and `BLOCK_SMOKE_*` in the engineering directory. They contain measured costs and the code hash.
- Both raw CSVs and the existing cache were verified against source identities. No checkpoint, parameter, optimizer, R, or gradient file was written.

## Cost and numerical limits

Real complete Validation engineering calibration ran on 100,021 ML and 250,767 KR rows. Native exact-pair counting closes to source AUC at approximately machine precision. The block receipts measure full-V/core states, mixed cut, fusion deletion, and bounded JVP, rather than extrapolating from training alone. `RESOURCE_ESTIMATE.json` combines training/step observers, pulse costs and an upper finite causal-readout count; it records worker seconds, not an asserted wall-clock completion time.

In the first-step engineering contrast, the 32-node path integral did not close for all diagnostic pairs. The code records each actual endpoint, integral, residual and unresolved flag and stops at the registered node cap. Complete-cut and fusion finite-forward checks passed. Formal analysis must use the actual integral residuals alongside the finite interventions.

## Handoff

1. Independent reviewer inspects this report, the frozen design and actual code; correct blocking findings before formal launch.
2. Create a fresh formal attempt with current audit/prepare; smoke receipts must bind the current implementation hash. Place the independent review verdict in that attempt.
3. Run in tmux. `run` performs all live-state block operations; `analyze` reads committed scalar outputs only.
4. Orchestrator writes `MECHANISM_FINDINGS.json` (`author_role=orchestrator`, one `main_direction`, Q1–Q5 with claims/evidence). Only then `freeze`, `evaluate`, `report`, final scientific narrative, and `release`.

Formal 20-trajectory training, complete five-seed block execution, Test and scientific analysis remain for the authorized RUN/ANALYZE stages after independent review. No scientific result is claimed from engineering smoke.

Final engineering status: both full-batch/pulse/fault-recovery/block-calibration smoke receipts PASS and bind code hash `1a925e95bb5c0bc4d1434f620668af8e5b954499ddea1e6d233a31556ef83b6e`. Measured finite-queue upper estimate: 18741.4 worker seconds; GPU sharing and block serial work determine actual wall time.

## Independent-review corrections

The first review found outputs that would lose the parameter→function→same-pair chain after RAM release. The implementation now writes:

- `FUNCTIONAL_PAIR_DETAILS`: eight outcome-independent hash-ordered pairs per cell from the registered diagnostic pairset, with original positive/negative row IDs, four concrete state IDs, probability and logit margins at 00/10/01/11, representation/downstream/interaction, and official probability-based rank transitions. The complete diagnostic-pair population still contributes to `FUNCTIONAL_MARGIN` reductions and full population counts.
- `PATH_CLOSURE` and `PATH_INTEGRALS.pair_identities`: the same eight-per-cell pair IDs and row IDs used in the functional detail table, with actual delta, integral, group integrals, residual and unresolved flag.
- Discovery-only margin strata: P_A boundaries apply unchanged to P_B; V_A boundaries apply unchanged to V_B. Boundary values, discovery population and rule hash are retained.
- Fixed-anchor numerical readouts: the actual scratch forward of FP32-cast `O+R_ideal` is compared with real FP32 F on all three fixed panels, recording probability/logit error, exact rank transitions and AUC change. Sparse events retain maximum rounding error and temporal error upper bounds as scalars.

These are observation corrections. Source training, matrix, test states and scientific gates remain unchanged. A focused discovery/confirmation test and real-data assertions check the new pair identities, four-state closure, concrete state IDs and numerical-readout output. Updated test log: `tests_review_fixes.log` (13 passed). Updated two-dataset engineering smoke logs use the suffix `_review_fixes`.

Review-corrected final code hash: `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`. Both updated dataset smoke receipts PASS and bind this hash; same-pair/output assertions, ideal-R readout and observer-purity checks passed. Finite resource upper estimate refreshed in RESOURCE_ESTIMATE.json: 25784.1 worker seconds. No further checks are pending from the coder.
