# Independent static review — final exposed scoring launcher

**Verdict: PASS. No blocking issue found.**

Actual source inspected, including its reviewed capture/checkpoint-identity dependency and frozen base/conditional evaluate_exposed interfaces. No new quality records or arrays read, no launch/scoring/GPU test executed, no research outcome analyzed, no frozen experiment implementation changed.

## Requirement verification

| Requirement | Actual implementation | Result |
|---|---|---|
| Whole package before scoring | Exact non-warmup manifest scopes52P1+10bridge+7P2+20formal, correct stage and24/50 endpoints, exact config equality and COMPLETE status for every run. Unique89 IDs required before scheduler or worker proceeds. Formal measurement fragments at11 cannot pass. | PASS |
| Frozen source and method identity | Manifest source consistency requires common base P1/bridge and common conditional P2/formal. Final freeze enumerates exactly89 IDs and explicitly locks both source versions. Before any new inference, each worker independently rehashes actual composite/base source bytes. | PASS |
| Endpoint/selection identity | Reuses only existing primary_best and fixed endpoint aliases. Reviewed checkpoint_identity validates validation-only eligible>=12/earliest metric rule, epoch and checkpoint SHA against retained final identity. No test/Q metric chooses a model or changes a method. | PASS |
| Actual scoring and immediate capture | Same child invokes actual conditional evaluate_exposed dispatcher, which uses frozen base scoring for base methods and actual extension scoring for conditional methods. Fixed role order benchmark_test_exposed then Q_legacy_confirm_exposed. Capture occurs immediately after each evaluation and before the next role overwrites transient metadata. | PASS |
| Certified reuse | Backup content-address token, source/config, exact expected predictor/checkpoint coverage, run/role, prediction SHA, checkpoint SHA and validation-derived identity all verified. Skips do not reopen test row values or run a model. Finished live metadata can be captured honestly after interruption; no metadata invented from role NPZ alone. | PASS |
| GPU capacity | Worker child env sets CUDA_VISIBLE_DEVICES=0 before importing torch/experimental source. Only cuda:0 passed. Scheduler permits max1 or2 child processes, counts other GPU0 compute processes before launch and never terminates external jobs; occupied GPU1 is unavailable. | PASS |
| Original and new CSVs | Existing canonical test_metrics.csv backed up with SHA and restored through evaluation finally and outer failure paths; unexplained later edits refuse overwrite. Role-generated CSVs are retained separately. Full two-role merged CSV has a new filename and expected six-cell coverage per predictor/checkpoint-role. | PASS |
| Honest failure/completion | Role errors preserve failed receipt/traceback/byte offsets and STOP; scheduler stops further launches on known errors, drains existing children and records UTC/PID/exit/stdout SHA. Missing/failed child receipt cannot count COMPLETE. Final status independently rechecks all89runs/178roles/388predictor-checkpoint-role records. | PASS |
| Cost/access provenance | Before/after canonical journal byte offsets per role retained. Inclusive wrapper wall explicitly additiveFalse and separate from per-run component leaves; no fabricated training update, GPU-hour or unseen-confirmation claim. | PASS |

## Bugs, corrections and remaining risks

No result-affecting implementation error found in this release scope; reviewer made no code correction.

A hard process kill can bypass Python finally. On a later resume, a preexisting-original CSV that differs from its certified backup causes refusal rather than silently overwriting a possible external edit; verified backup and role files remain available for explicit recovery. GPU process counts are checked at launch and cannot prevent an unrelated external process from starting later. Retained partial outputs are not final completion; downstream publication must use the successful final execution and aggregation receipts.

The role wrapper's wall/journal byte-range evidence is retained under final_exposed. Final cost reporting should preserve that provenance separately from component sums, especially for old bridge scoring paths whose evaluation/import wall is not fully component-journaled.

## Safe execution

Safe for the authorized final exposed scoring only after all gate requirements pass. This receipt authorizes no training, checkpoint reselection, scientific interpretation, or extra engineering tests.

- Launcher SHA256: `4fb6379be1460341b82b47a6a1294134a1d7a6f0272111a0938590f8f54cf884`
- Capture/aggregation dependency SHA256: `a9dbc8ceb2141213c8868c9217a59135a09a0d5e5d8488fd930b714ec100b82d`
