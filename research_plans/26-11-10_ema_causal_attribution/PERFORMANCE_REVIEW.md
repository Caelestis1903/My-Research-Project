# Independent performance amendment review

Date: 2026-11-10. Verdict: **PASS**.

Reviewed implementation hash: `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`.

Scope: `PERFORMANCE_AMENDMENT.md`, `CODER_PERFORMANCE_REPORT.md`, actual `pairs.py`, focused `tests/test_pair_cache.py`, and all production sampler consumers. This is an independent review of the cache amendment, supplementing the previously closed `REVIEW_REPORT.md` findings.

No blocking findings or corrections are required.

- The key contains the actual ordered positive and negative int64 index bytes, seed, and limit. Reordered labels cannot incorrectly reuse the same mapping; prediction/state changes appropriately do not enter this deterministic index key.
- The miss helper retains the original Floyd loop, private `np.random.default_rng(seed)`, sorted IDs, pair order, label-to-row mapping and int64 output. Hits neither run the sampler nor consume global/training RNG. All current production seeds and limits are integers.
- The standard-library LRU has `maxsize=128`; the public wrapper admits only at most 4,096 participating indices and limit at most 32,768. Retained pair payload is bounded by 64 MiB and index-key payload by 4 MiB, plus object/LRU overhead. Out-of-bound requests use the unchanged uncached algorithm.
- Cached results have their writable flag disabled. Inspected calls in `block.py`, including `_diagnostic_pairs` consumers, only read/index/gather/iterate these arrays. Statistics and overlap helpers likewise only read them; engineering/path construction creates separate remapped lists. No writable-copy change is needed.
- The cache contains only deterministic indices and immutable key bytes. It adds no serialization, model state, prediction cache, training modification, scientific matrix modification, or discovery-bound cache. Existing process RSS accounting includes its host allocation.

Independently recomputed the current source identity. Removing only the cache import/wrapper/helper and the new focused test in memory reconstructs exactly `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`, proving that other identity-covered sources retain the prior reviewed bytes.

Independent focused verification:

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -s experiments/ema_causal_20261110/tests/test_pair_cache.py
1 passed in 10.56s
```

The check confirms exact reference equality/order/dtype for an actual 32,768-pair panel, reordered labels, changed seed/limit, all-pairs and empty-class cases; read-only shared hits; both admission bypasses; 128-entry retention and eviction. The production-shaped 60-call sequence also matches the uncached reference exactly. Independently observed timings: uncached 6.376111 seconds, initially empty cache 2.931825 seconds, warm cache 0.004197 seconds. Counters were 30 misses/30 hits/30 entries after the first sequence and 30 misses/90 hits/30 entries after the warm sequence. These are engineering observations without a timing pass threshold or scientific inference.

The amendment is released for fresh preparation and hash-bound engineering smokes. The formal attempt2 queue still requires those fresh smokes under the unchanged preflight checks; old smoke evidence remains bound to the old hash. Retain attempt1 raw logs and costs, and run the specified full 20 trajectories without splicing partial attempt1 research outputs into attempt2.

Unrelated implementation review, broad tests, model smokes and formal experiments were not repeated here. No research conclusions are drawn; fresh hash-bound smokes remain the next required execution gate.
