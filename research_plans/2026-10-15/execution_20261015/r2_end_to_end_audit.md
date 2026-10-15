# AO04 end-to-end audit status: PARTIAL

Numeric evidence: [experiment_results.md](experiment_results.md), `ao04_test_metric_recalc.csv`, `ao04_test_aggregates_recalc.csv`, `ao04_paired_bootstrap.csv`.

Run/config/source identity: `run_identity.json`, `ao04_pairing_claims.json`, `source_hash_checks.csv`, `selection_checks.csv`, `dataset_prediction_binding.json`, `ao04_validation_identity.json`, `g1_receipt_artifact_checks.csv`, `csv_newline_diagnostics.json`.

KR B0/B1 seed2022 manifests claim matching main/arch batch hashes and shared paired-init-manifest identity; full model hashes differ with architecture parameter count and are not a common-tensor proof. No checkpoint deserialization or common-tensor rehash was performed. Thus pairing is partially supported by stored claims, not independently reconstructed.

KR three completed runs' test predictions are source-row bound; validation lacks saved row IDs. Nine completed runs described in the plan are not all available as complete directories locally: only four COMPLETE manifests plus one INCOMPLETE manifest are present. Source-level numerical agreement does not close the missing-file/full-state chain.

End-to-end L4 replay and complete A1 acceptance remain INCOMPLETE. No silent source replacement, model load or B3/B4 extension occurred.
