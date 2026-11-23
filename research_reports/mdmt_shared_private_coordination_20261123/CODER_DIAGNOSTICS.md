# Coder implementation report for S1 diagnostics

## Runtime contract

- `diagnostics.prepare(bundle, dataset, output_dir)` freezes `PANELS_<dataset>.json`. The returned object exposes `.manifest` and retains Validation arrays and pair indexes in memory.
- `diagnostics.evaluate(model, bundle, device)` returns `(metrics, counters)` with the source EMA02 metric keys and `macro_auc` for strict-max selection.
- `diagnostics.forks(model, opt, ema, batches, panels, metadata, device)` accepts two complete `(x, y)` training batches. `metadata` supplies `dataset`, `seed`, `epoch`, `source_hash`, `data_hash`, `config_hash`, `logical_id`, `attempt_id`, and the runtime's next-epoch batch, loader, ledger and anchor provenance. Its scalar receipt contains `metadata`, `panel_hash`, `endpoints`, `transitions`, `responses`, `costs`, and `purity`.
- `statistics.analyze(artifact_dir)` verifies completion and the registered anchor/fork matrix, writes `FORK_ENDPOINTS.csv`, `PAIR_TRANSITIONS.csv`, `CELL_RESPONSE.csv`, `PAIRED_SEEDS.csv`, and `SCOPE_DECISION.json`, and returns the decision. The runtime writes baseline endpoints, attempt costs and the anchor index.

## Implementation

- Panel preparation preserves canonical Validation row order and uses the historical label-blind split. The D2 namespace keys a private NumPy generator for each dataset, panel and cell. Each pair ID is its zero-based ordinal within that cell of the frozen manifest. Missing-class fold cells keep undefined AUC and link to the descriptive full-Validation-cell fallback.
- Fork diagnostics use the native block steps, the canonical tower projection hook and the canonical probability metrics. Each measured state produces one full Validation prediction pass for all panels. Pair grades use the source FP32 probabilities so ties follow the AUC input. Pair margins use the routed presigmoid logits. Each fork verifies its inherited state and the identity of the untouched diagnostic parent, including model modes, gradients, native Adam and Python, NumPy, Torch CPU and the assigned CUDA device RNG. The runtime owns the verified RAM identities, baseline ledger and loader preview receipts.
- The matrix contains every named intervention and mapping in the approved design and its count correction. This yields 19 forks per anchor and 570 across S1. Retained intermediate online reads support both shared-update and literal post-U recovery contrasts.
- Scalar reporting requires ten completed sealed trajectories, matching fork receipt digests and all registered endpoint arms, phases and readouts. Paired tables average AB and BA within each anchor, then anchors within seeds, and report five-seed means, sample SD, intervals, sign counts and leave-one-seed-out means. Equal-dataset values average matching seed deltas across the two datasets. The online V_A ranking freezes one scope before the V_B gate. If a proper subset passes S1, its result remains GO and the S2 handoff is technically blocked by the excluded group's extra full J_ACC clock.
- A missing fold cell prevents an independent six-cell estimate for that dataset. The conditional confirmation records unavailable V_B intervals explicitly and flags an adverse result when a defined upper 95% bound is below zero. An equal-dataset ranking requires defined V_A estimates on both datasets.

## Checks and remaining review

The project Python passed syntax checks and the six focused CPU tests in `tests/test_diagnostics.py`. Fixtures covered panel identity and disjointness, fallback support, all 19 native-Adam forks, RNG and parent state purity, actual read counts, online-only scope selection, both batch mappings, anchor replication, six-cell protection, exact scope ties, the proper-subset clock blocker, equal-dataset and seed sensitivity, and scalar artifact round-trip and incomplete-receipt rejection.

Independent implementation review and the real S0 CUDA and RAM checks remain workflow gates. Implementation checks used synthetic fixtures.
