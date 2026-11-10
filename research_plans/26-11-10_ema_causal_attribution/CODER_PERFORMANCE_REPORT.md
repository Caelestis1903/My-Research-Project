# CODER PERFORMANCE REPORT

Date: 2026-11-10. Scope: approved PERFORMANCE_AMENDMENT.md only. Status: IMPLEMENTED; independent reviewer verdict pending.

## Implementation and lineage

Changed `experiments/ema_causal_20261110/pairs.py` and added the single focused check `experiments/ema_causal_20261110/tests/test_pair_cache.py`. No other source, model, training, caller, preflight, scientific matrix, or receipt was changed by this amendment.

Current implementation hash: `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`.

`pairs.py` SHA-256: `befcff364e512e2b2f7b54de6e1359a4a8def6c1086ffd52d1f42ee2a6b53241`.

Focused test SHA-256: `178a5aac6f89632e3c7a7bfc9cdc8f11bf83921ac233efc4fa8765c17bd31668`.

Removing only the new cache wrapper/import/helper and test from the identity calculation reconstructs exactly the prior reviewed implementation hash `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`. Existing ML and KR smoke receipts both say PASS and retain that prior hash. The prior 13-test evidence belongs to that implementation. Existing engineering PREPARED.json has an older preparation hash and was not relabeled. Attempt2 preparation and post-review smoke execution remain the orchestrator/runner's responsibility under the unchanged preflight guards.

The new process-local standard-library LRU retains at most 128 entries. Admission requires at most 4,096 participating positive/negative rows and limit at most 32,768. Keys use the exact ordered positive and negative int64 row-index bytes plus seed and limit. Misses reconstruct those indices and execute the original Floyd loop, private `default_rng(seed)`, sorted IDs and row mapping without changes. Out-of-bound requests run the same uncached helper. Cached arrays are marked non-writable. No predictions, parameters, gradients, model states, global RNG, or disk writes enter the cache.

Maximum retained pair payload is 67,108,864 bytes (64 MiB); index-key payload is 4,194,304 bytes (4 MiB), plus Python/LRU overhead. This process host memory is visible in existing RSS accounting. The wrapper's temporary indices are not retained. No discovery-quantile cache was added.

## Caller inspection

All `sample_pairs` production calls are in `block.py`: diagnostic pair identity generation, discovery bounds, panel records, source overlap, fusion connection, source assay, engineering sampled inputs, and factorial score/margin records. `_function_contrast` also consumes `_diagnostic_pairs` output. Each reads, iterates, indexes, gathers, or passes sampled pairs to the read-only statistics/overlap consumers. No caller mutates sampled arrays; engineering constructs its own separate list of remapped pairs. Existing sampler tests also read only. No caller copy or modification was necessary.

## Focused verification

Executed through the required project interpreter:

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -s experiments/ema_causal_20261110/tests/test_pair_cache.py
1 passed in 14.10s
```

The reference is a literal copy of the original uncached sampler. Checks pass for the saved MovieLens seed2022 P_A domain0/task0 32,768-pair request, reordered labels, changed seed, changed limit, small all-pairs input, and empty-class input. Cold/warm arrays have exact byte, dtype, shape and order equality to the reference. Repeated requests return the same non-writable array and increment hits without additional misses. Changed inputs create distinct misses. Both admission bounds bypass the cache. Inserting 129 distinct tiny requests retains 128 entries and accessing the evicted first key causes a new miss.

Timing uses the actual saved panels at `artifacts/ema_causal_20261110/engineering/panels/movielens/seed2022`. The 60-call sequence follows `panel_records`: five populations × six cells × population/discovery requests, including the production SHA-256 seed construction. All 60 cold and warm outputs equal the reference outputs. Timings are observations, with no wall-clock pass threshold.

| Check | Elapsed seconds | Cache counters after check |
| --- | ---: | --- |
| One actual panel, cold | 0.114114297 | single sequence: 1 miss, 1 hit, 1 entry after warm |
| Same panel, warm | 0.000242995 | as above |
| Original uncached 60-call sequence | 8.256593109 | no cache |
| Cached 60-call sequence, initially empty | 4.371564096 | 30 misses, 30 hits, 30 entries |
| Same 60-call sequence, warm | 0.002765770 | 30 misses, 90 hits, 30 entries cumulatively |

No formal experiment or scientific analysis was run. Broad tests and full model smokes were not repeated by this coder; prior evidence is explicitly retained under its original identity. Independent review must precede attempt2 preparation/smokes and the full 20-trajectory rerun. Attempt1 partial trajectories/results must not be spliced into attempt2.
