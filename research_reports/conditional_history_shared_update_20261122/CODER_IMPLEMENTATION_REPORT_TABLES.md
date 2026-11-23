# A result tables and selection interface

## Scalar record contract for implementation handoff

Supply one wide row per dataset, arm, seed, split, readout and checkpoint in CSV or as Python dictionaries. Publish only accepted complete 50-epoch attempts, after identity audit and required current-run RAM commit. Historical aliases must be normalized to the new arm IDs before this boundary. The results code rejects duplicate identities instead of selecting one attempt.

Required columns are dataset (movielens/kuairand), arm (matrix ID), seed, split (validation), readout (online/fixed_ema), checkpoint_rule (selected/E10/E50), epoch, model_hash, source_ref (path or locator), reuse_status (REUSE_NUMERIC or NEW_TRAIN for A science rows), trajectory_id, and source metric columns d0t0_auc/d0t0_logloss/d0t0_count through d2t1_auc/d2t1_logloss/d2t1_count. Optional task0_pooled_auc, task1_pooled_auc, pooled_task_mean_auc, pooled_logloss follow the source evaluator. Arbitrary additional scalar provenance fields are preserved in expanded endpoints. Missing pooled metrics remain blank.

The results module exposes write_stage(stage, rows, output, previous=None, localization=None). Stage IDs are A_CORE, A_HISTORY, A_DOSE and A_SCOPE. Pass the complete cumulative Validation rows and the preceding stage's returned decision for later stages. The CLI can invoke the module directly via python -m experiments.conditional_history_shared_update_20261122.results STAGE --input TABLE.csv [--input MORE.csv] --output DIR [--previous decisions.json] [--localization FILE.json]. The stage writer creates endpoints.csv, paired_summary.csv, paired_seed.csv, a paired effects plot and decisions.json in a stage-specific directory. No Test evaluation occurs in this module.

## Completed implementation

The stage writer preserves absolute AUC and LL for each seed and six cells at each supplied checkpoint. It derives domain, task, macro and support-weighted traffic summaries and retains the supplied pooled metrics. Each paired summary carries both absolute means and sample SDs, all five ordered differences, the paired SD and descriptive t4 limits. The primary plot displays all five raw differences and the mean interval. The decision records link to the SHA256 of each supporting CSV and the statistics script.

A_CORE requires complete selected fixed-EMA Validation identities across both datasets. A_HISTORY checks the six-cell guard, simple-method band, conditional/collapsed/long-history contrasts and the predeclared target tie break. A_DOSE compares reduced-dose history with its reduced-dose Adam control, freezes a per-dataset dose, and accepts a full same-trajectory localization panel. A_SCOPE compares restricted forms with full history, matched Adam and A00 on the same dataset and fixes the cross-dataset scope from the previously chosen target. The reporting code returns task names to the orchestrator. Missing primary identities return TECHNICAL_BLOCKED.

For localization provide a JSON array with dataset, arm (frozen dose arm ID), history_form, split=validation, readout=fixed_ema, action=whole_group_to_adam, group=STAR_SKIP/EXPERT, seed, anchor_epoch=5/20/35/50, scale=raw/norm_matched, source_ref and six d0t0_delta_auc … d2t1_delta_auc values. Deltas are the group replacement minus the actual update, evaluated on the same held-out Validation panel. A non-OK status explicitly marks an uninformative observation. Average all four anchors within each seed before assessing the five-seed mean. The group replacement must improve a damaged cell and the macro under both scales. The weaker-dose cost must be observed in current Validation or supplied trajectory evidence.

Module input accepts the stored REUSE_MANIFEST.json, verified individual result.json files with a sibling COMPLETE_VERIFIED.json receipt, or normalized wide CSV. The adapters audited_reuse and completed_rows attach source/config or per-attempt/data/group/generation identities. freeze_identities constructs a content-addressed selected-model payload for the final report, including method parameters, registered component/total comparisons, references, scripts and model hashes. The final release workflow owns verification of current selected RAM and task closure.

For equal-complexity dose choices the code takes the higher Validation macro and retains the original dose on an exact tie. For scope choices the protected stratum has priority, followed by higher macro and the full group on an exact tie with a restricted group. These deterministic ties resolve cases for which the original plan gives no additional complexity order.

## Basic checks

The project Conda pytest command passed all three synthetic tests in tests/test_results.py. They cover exact seed pairing and t4 arithmetic, missing seeds, duplicate and Test rejection, selected-only plot/table output, protection, simplification, target tie-breaking, reduced-dose matching, localization gating, scope choice and identity freeze. The plotting stack emitted upstream pyparsing deprecation warnings. A scalar compatibility check also accepted all 168 audited historical Validation checkpoint/readout rows from 28 unique trajectories. It did not compute a research decision.

Ready for distinct independent review of results.py and tests/test_results.py. No scientific training or new Test evaluation was run in this subtask.
