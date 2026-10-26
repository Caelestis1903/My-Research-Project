# N19 — complete end-to-end MDMT study

Status: **COMPLETE. [PR #6](https://github.com/Caelestis1903/My-Research-Project/pull/6) is open against main; main is not merged.** All328 formal
networks finished their full budgets and frozen selected/final test evaluations.
The six scoped scientific dispositions are complete; negative and contradictory
results are retained. No post-test method/selection changes. Completion of this
matrix does not establish six successful new algorithms or reject six broad
scientific directions.

- [Final scientific report](FINAL_REPORT.md),[scoped dispositions](SCIENTIFIC_DISPOSITIONS.json)
- [Actual execution/delivery status](EXECUTION_STATUS.json),[run ledger](RUN_LEDGER.csv),[inventory](INVENTORY_STATUS.json)
- [Revised entry](source_start_utf8.md),[design](DESIGN_SPECIFICATION.md),[mathematical claims and counterclaims](MATHEMATICAL_SCOPE.md)
- [Primary-source novelty audit](LITERATURE_UPDATE.md),[protocol limits](PROTOCOL_LIMITS.md)
- [Pre-test final development decisions](DEVELOPMENT_DECISION_RECORD.md),[immutable test freeze](TEST_FREEZE.json),[readout/export verification](EXPORT_VERIFICATION.json)
- [Implementation report](IMPLEMENTATION_REPORT.md),[primary self-review](REVIEW.md),[alternative review](ALTERNATIVE_REVIEW.md)
- Direction records including frozen-test final judgments: [P1](P1_DEVELOPMENT_ANALYSIS.md),[P2](P2_DEVELOPMENT_ANALYSIS.md),[P3](P3_DEVELOPMENT_ANALYSIS.md),[P4](P4_DEVELOPMENT_ANALYSIS.md),[P5](P5_DEVELOPMENT_ANALYSIS.md),[P6](P6_DEVELOPMENT_ANALYSIS.md)
- [Test means/costs and paired effects](test_readout/NUMERIC_TABLES.md),[tasks/cells](test_readout/PER_TASK_CELL_TABLES.md),[raw paired cell data](test_readout/paired_cells.csv),[decision contrasts](test_readout/DECISION_CONTRASTS.csv)
- [Development evidence](development_readout/NUMERIC_TABLES.md),[new questions and future discriminators](OPEN_PROBLEMS.md)
- [Reproduction](REPRODUCE.md),[environment](ENVIRONMENT.json),[local predictions](LOCAL_PREDICTION_INDEX.csv),[local checkpoints](LOCAL_CHECKPOINT_INDEX.csv)

328 formal networks=40 shared native baselines+40 P1+72 P2+72 P3+40 P4+32 P5+
32 P6;82 four-seed groups. Full from-scratch training, all backbone parameters
trainable, KR50epochs/ML100epochs, batch40960, common5% train-meta reservation.
No early stopping or seed exclusion. P5 reuses8 full-router controls; P6 reuses4
native full-label controls, not extra training. P2 privileged clean oracle was
actually retrained. Nine engineering runs are excluded from research means.

14 numbered addenda record actual development triggers and bounded extensions,
including audit profiling/misspecification, MMoE transfer, residual observations,
uniform-KL control and both native datasets' matched AdamW capacity families.
All declared alternatives ran; no open-ended hyperparameter grid. Only mechanical
execution was delegated under revised start.md11; all reviews are explicitly
primary self-reviews, not independent reviews.

The sealed queue completed658/658 jobs with zero failures. Each of328 test
receipts records zero optimization steps and the immutable freeze hash. The64
physically compact rank exports have maximum prediction discrepancy1.1921e-6.
2400 capacity-cell,2592 routing-cell/source and480 finite-credit cell records
are supplemental correlated mechanism observations, not replacement training.

Source: experiments/n19_{data,models,objectives,train,queue,freeze,summarize,
dynamics,rank_diagnostics,routing_diagnostics,credit_diagnostics,channels,evaluate,
figures,tables,inventory}.py. Raw configs/source snapshots/epoch and cell metrics
and receipts are in runs/, commands/logs in logs/ and queue events. Large arrays
and full states remain local under paths recorded in the SHA256 indices;
GitHub contains key raw numeric evidence, source, logs, tables, PNG/PDF and reports.
Historical tests/populations were reused; the freeze is not fresh confirmation.
