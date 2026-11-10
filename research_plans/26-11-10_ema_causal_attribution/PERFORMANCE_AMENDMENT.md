# PERFORMANCE AMENDMENT — deterministic panel-pair memoization

Date: 2026-11-10. Status: DESIGN APPROVED FOR IMPLEMENTATION; implementation and independent review remain required.

## Measured problem and responsibility

Attempt1 completed the five ML-C training trajectories but spent over 100 minutes in block core attribution. Coder timing identifies deterministic `sample_pairs` as 84% of `panel_records`: about 60 calls and 1.96 million Floyd-loop iterations per state; the first seed core made 6,372 calls and 208.8 million iterations. Identical dataset/seed/population/cell label vectors repeatedly request identical indices. This amendment removes that repeated calculation without changing any scientific comparison, sampling distribution, pair identity, or training process.

## Smallest complete change

Ownership: coder changes `experiments/ema_causal_20261110/pairs.py` and one focused test/assertion in the existing test location. All existing callers, including discovery bounds, retain the `sample_pairs(y, seed, limit=32768)` interface. No changes to training, model, matrices, state descriptors, statistics, or table schema.

Use a bounded process-local in-memory LRU of the deterministic sampling result. The miss path retains the current Floyd algorithm, `default_rng(seed)`, ascending `upper` loop, sorted selected IDs, positive/negative row mapping, output dtype and output order exactly.

The cache key must identify the actual ordered positive and negative row-index vectors, integer seed, and limit. Immutable int64 index bytes are a direct collision-free key; reconstructing them with `frombuffer` in the miss helper is sufficient. A key consisting only of class counts, object identity, dataset or seed is incorrect: different row ordering would then reuse wrong pairs. No prediction, parameter, objective, epoch, or anchor belongs in this key because none affects the current function.

Bound the cache to 128 entries and admit only existing panel-sized requests: at most 4,096 participating positive-plus-negative indices and limit at most 32,768. Requests outside this bound use the unchanged uncached algorithm. Each cached pair array is at most 524,288 bytes; 128 arrays total at most 64 MiB, index keys at most 4 MiB, plus small Python/LRU overhead. Cache lifetime is the attribution process lifetime; eviction and process restart only recompute identical indices. Count this host-memory overhead in the existing RSS/resource accounting; it contains no model state and is never serialized.

Return cached arrays read-only to prevent a consumer from corrupting future hits. Current inspected callers index/gather/read the pairs and do not mutate them; coder must confirm the existing caller search before adopting shared read-only arrays. If an existing caller needs a writable array, copy only at that caller.

Do not cache discovery quantile bounds in this amendment. Those depend on the reference predictions, while the dominant repeated pair generation beneath `_discovery_bounds` is already removed by the shared function cache. A separate prediction-identity cache is unnecessary for the measured gap.

## One focused equality and performance check

Use one compact assertion-based check with the original uncached Floyd implementation as reference. Cover one actual 32,768-pair panel request, a reordered binary label vector, a changed seed, changed limit, and a small all-pairs/empty-class request. Assert exact `np.array_equal`, dtype and order equality for cold and warm calls; assert cached results are non-writable. An LRU hit on a repeated exact request must avoid rerunning the sampler. Verify the selected key distinguishes the changed inputs.

Record cold and warm timing on that same actual panel and a repeated-call sequence matching the measured 60-call pattern. Report elapsed time, cache hits/misses and bounded cache size; do not impose a flaky wall-clock test threshold. The expected improvement is elimination of repeated Floyd loops after the first request per distinct key. No model training or additional ablation is needed for this change's check.

The sampler's RNG is private `default_rng(seed)` on a miss, and cache hits do not consume any global or training RNG. Coder/reviewer must preserve this property by inspection; no new global RNG or mutable training state may enter the cache.

## Attempt and review handling

Independent reviewer checks the key, unchanged miss algorithm, all callers' read-only use, fixed memory bound, focused equality result and measured hit-path cost. Only PASS or corrected PASS WITH FIXES releases the new code.

Archive attempt1 logs and completed/discarded cost with its original code identity. Attempt2 reruns the frozen 20-trajectory matrix after this fix and uses its own code hash; do not splice partial attempt1 research tables into attempt2. The cache contains only deterministic diagnostic indices, so neither this amendment nor the rerun adds scientific seeds or authorizes new experiments.

No scientific result is inferred from runtime profiling. Required final outputs and the five-seed causal/functional evidence chain remain unchanged.
