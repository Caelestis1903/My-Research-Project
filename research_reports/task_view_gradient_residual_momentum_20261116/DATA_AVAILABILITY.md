# Data availability

The nine CSV tables under `artifacts/26-11-16_augmentation_gradient_momentum/attempt1/tables/` contain all exported validation endpoints, curves, paired seed values, summaries, natural-weight audits, diagnostics and costs. The full report, numerical appendix and reproducible analysis are published alongside the experiment code.

Individual run results, all 2,500 epoch JSON records, compact batch logs and frozen job configurations are stored losslessly in the numbered `RAW_RUNS.tar.gz.part-*` files in that attempt directory. Restore their original relative paths before reading individual raw records:

```bash
cd artifacts/26-11-16_augmentation_gradient_momentum/attempt1
cat RAW_RUNS.tar.gz.part-* > RAW_RUNS.tar.gz
tar -xzf RAW_RUNS.tar.gz
```

The original unarchived records remain in the server worktree. Source/data/review/smoke identities and launch/epoch ledgers are also published directly. No model weights, optimizer states, checkpoints or guardian authentication metadata are included. RAM model states remain owned by the current-conversation workflow.

Publication uses the connected GitHub application because this terminal has no HTTPS Git push credential. The archive representation changes only storage layout; scientific source identities, data, scores, selections and paired arithmetic are unchanged. The local full-file commit and the published archive-layout commit are recorded separately in the publication receipt.
