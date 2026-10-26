# Independent static review — exposed benchmark aggregation

**Verdict: PASS. No blocking issue found.**

Inspected the actual helper and frozen base/conditional exposed-scoring filename and checkpoint conventions. No new quality data, arrays or research results were read, no helper aggregation was executed, no GPU test ran, and no experiment implementation changed.

## Requirement verification

| Requirement | Actual source behavior | Result |
|---|---|---|
| Role backup and overwrite defense | Captures transient exposed_evaluation.json immediately after each role; exact predictor/checkpoint coverage required. Backup identity includes config/source, CP epoch/SHA, prediction SHA, role, original metadata and identity-source hashes. Content-addressed immutable JSON refuses conflicting replacement. Earlier overwritten role cannot be reconstructed from a role-named NPZ alone: absent matching backup raises and requests reevaluation, not metadata invention. | PASS |
| Frozen admission | Capture checks FINAL_METHOD_FREEZE membership and exact COMPLETE endpoint. Full aggregation gates all requested non-warmup logical trajectories, and freeze membership, before numerical work. Existing Q development tables are separate from analysis_exposed role directories and APPENDIX_EXPOSED files. | PASS |
| Validation-only state selection | Reuses validation rows only, eligible epochs >=12 with earliest tie and correct declared LL/AUC metric; independent main/local predictor selection. Alias checkpoint SHA and expected epoch must match retained final Q identity metadata and exposed record. Q metric values never choose a checkpoint. | PASS |
| Two exposed roles | Only benchmark_test_exposed and Q_legacy_confirm_exposed supported. Role NPZ row IDs/order, labels and domains checked against saved role IDs/original prepared arrays. Role is present in output tables/statistical dimensions and each role has its own directory/appendices. Both remain labelled previously exposed, never unseen confirmation. | PASS |
| Cohort and stage separation | Inherits independently reviewed metadata-defined cohort and stage pairing. Formal/dev pairs excluded; formal n<=4, duplicate seed units rejected; development n2 and P2 single-seed results remain distinct. Local heads are separate predictor identities rather than independent seeds. | PASS |
| Numerical completeness | Recomputes macro AUC/LL, natural LL, pooled task AUC, worst-cell AUC/LL and all six task-domain cells. Recorded macro/natural values must match. Selected/endpoint CP identities remain separate. Whole predictor/checkpoint count and six cells per checkpoint required. | PASS |
| Domain/task and uncertainty | Reuses reviewed paired cell/domain/task averaging, worst/reference-worst and size flags with explicit fit sizes. Student t95 uses paired training-seed differences, sample SD and n-1 degrees of freedom; n1 has no interval. Task/cell/epoch counts do not enlarge n. Appendix columns and grouped headings preserve role/cohort/stage. | PASS |
| Actual path compatibility | Base L0 mono/feature aliases use lowercase internal prefixes for selected states and shared endpoint state; their role prediction filenames retain predictor prefix. Other base/conditional exposed-scoring paths agree with helper construction. Fresh STGU selection is LL via declared checkpoint_selection; old bridge family keeps its original selection. | PASS |

## Deviations and corrections

No new blocker or result-affecting deviation found. No corrections made by reviewer.

## Remaining risks and execution boundary

Each role must be captured before the next role overwrites transient exposed metadata. Raw NPZs/checkpoints are referenced and hashed, not copied: later replacement invalidates that backup and requires honest reevaluation/recapture. An interrupted two-role aggregation can leave the earlier role's tables, but no final COMPLETE receipt is emitted until all requested roles succeed; downstream publication must require that final receipt. Interval statements describe training-seed variation on already exposed data, not fresh-data confirmation. Historical adapted AUC retains its historical ranking convention.

Safe for parent-authorized mechanical execution after final freeze, completed requested runs and valid role-specific captures. This PASS does not authorize method selection from exposed outcomes or assert any scientific advantage.

## Source identities

- aggregate_exposed_results.py: `a9dbc8ceb2141213c8868c9217a59135a09a0d5e5d8488fd930b714ec100b82d`
- aggregate_results.py: `3fa23c52bfad91acd5832c5aed0a55b40c847e06fe3dddce2f35d1dc30de5d08`
- write_numerical_appendices.py: `1c63d231020a8b3b87e339a2f7e656486df9e1162ebbc9235c9628978634d77f`
