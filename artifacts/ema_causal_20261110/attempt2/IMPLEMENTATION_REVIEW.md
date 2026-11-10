# Independent implementation review

Verdict: PASS WITH FIXES

Re-reviewed implementation hash: `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`.

All three blocking findings below are corrected. The formal queue is released for this implementation. The original FAIL and its rationale remain below as revision history.

## Re-review and closure

1. `block._function_contrast` now registers both actual endpoints, records concrete four-state IDs, and writes `FUNCTIONAL_PAIR_DETAILS` with probability/logit margins, their three-term decomposition, official probability ranks, original row IDs and reproducible pair IDs. `PATH_CLOSURE` uses the identical bounded pair set and aligned cell/row identities. Complete diagnostic-pair and full-population aggregates remain present. The detailed subset is explicitly limited to eight outcome-independent hash-ordered pairs per cell; its values must not be represented as the full-population distribution.
2. `block._discovery_bounds` derives P_B boundaries only from reference P_A and V_B boundaries only from reference V_A, and `panel_records` passes those bounds into `panel_pair_statistics`. The discovery source, boundaries and rule hash are retained. The focused test changes confirmation predictions and confirms unchanged boundaries.
3. `trajectory.advance_epoch` calls `rounding_observation` at every fixed anchor. The scratch `O+R_ideal` forward is compared with actual F on all three panels/six cells; probability/logit errors and exact ranking transitions are committed in epoch records. The temporary ideal parameter vector is not persisted. Sparse historical error scalars provide the temporal rounding bound.

Verification inspected: `tests_review_fixes.log` reports 13 passed; both `SMOKE_MOVIELENS.json` and `SMOKE_KUAIRAND.json` are PASS and bind the exact re-reviewed hash above. Their actual-data block checks assert pair identity alignment, four-state closure and the 18-cell rounding output. The repair also keeps STAR trigger calculations restricted to logit-space interaction rows, avoiding double-counting the newly added probability rows.

No blocking findings remain within this review scope. Required formal coverage and scientific interpretation must still be assessed from the upcoming run outputs.

## Original review — FAIL

Runtime review addendum: the initial attribution parent inherited 90 Torch CPU threads, unlike the one-thread engineering environment. An execution-only restart with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` is approved without changing the reviewed code. Stop only the attribution parent, retain the independently launched guardian, and invoke `resume` against the same attempt. `ensure_guardian` reuses its live owner; COMPLETE jobs bypass training, and the unfinished block attribution is recomputed. Record discarded attribution wall time separately and verify unchanged guardian PID, completed job epochs/steps and launch counts. This action adds no training seed or Adam step.

Reviewed implementation hash: `1a925e95bb5c0bc4d1434f620668af8e5b954499ddea1e6d233a31556ef83b6e`.

The original review required the three corrections below before the formal queue. They concern evidence that must be captured while parameters remain live; they did not request additional experiments.

## Blocking findings

1. **The functional chain loses pair and endpoint identities.** `block.py:355–396` computes complete-cut four-state margins, but `FUNCTIONAL_MARGIN` retains only means/RMS and aggregate flips. `Anchor.factorial` records all four functional states as `actual_readout`; `PATH_INTEGRALS` stores arrays without their pair/row/cell identities. This does not deliver the same-pair `margin_00/10/01/11`, representation/downstream/interaction and ranks required by 03 D2 and 07 §9. Once the block packets are released, an identified X cannot be traced to the concrete four-state margin evidence. Preserve stable four-state identities, a bounded reproducible set of pair IDs with four-state terms (or row-keyed bounded predictions from which those terms are reconstructed), and aligned integral pair/row/cell IDs. Keep complete sampled-population aggregates and the 2 GiB observation cap.

2. **Confirmation strata are refitted on confirmation rows.** `block.py:165` calls `panel_pair_statistics` without bounds for every population; `pairs.py:87` therefore recomputes quartiles independently for P_A, P_B, V_A and V_B. Under 07 §3.3, discovery-reference boundaries must be applied unchanged to confirmation. The current bins compare different margin ranges and can alter the apparent cross-row concentration of corrections. Freeze P_A bounds for P_B and V_A bounds for V_B, recording their source and rule identity. Verify this with different discovery/confirmation distributions.

3. **The required rounding-function audit is not collected.** `trajectory.epoch_packet` records only parameter `rounding_rms`; `ideal_R` is subsequently removed by `Guardian.complete`. 05 §4.2 requires actual fixed-anchor panel reads of `O+R_ideal` against actual FP32 F, including probability/ranking effects. Parameter RMS alone cannot identify whether floating-point accumulation changes the ranking mechanism. Add this read-only scratch forward at fixed anchors and retain bounded scalar/pair summaries, then release its temporary vector. No extra training is needed.

## Inspected paths and existing evidence

- New package calls the pinned source loss, one Adam update and original FP32 fixed-shadow update. Original model/loss/data files remain unchanged. Natural full/tail batches, five seeds and 50-epoch budget are retained.
- Observations use scratch models, restore RNG, and retain the prospective selected-epoch history origins. Core S/N, four windows, finite refinements, complete h/fused cuts, fusion deletion, LOSO bridge and conditional assays are connected to `finalize_block` before RAM release.
- Test loading is isolated behind orchestrator findings, freeze receipts and live selected endpoints. Full-population pair counting uses the native exact kernel and checks AUC closure.
- The RAM guardian retains epoch transactions and bounded restart/replay bookkeeping; source cache and raw identity checks are connected. Resource admission checks each GPU and host RAM.
- Existing final engineering evidence: 12 tests passed; both datasets' source-purity, real-tail and RAM-recovery smoke passed. Cost receipts include complete Validation, cuts, fusion and bounded JVP. These checks do not replace the missing outputs above.
- The reported 32-node first-step integral residual is handled by the specified finite-node cap and unresolved flag; it does not block real finite-forward causal evidence.

## Re-review scope

Inspect the corrected capture/output call chain and the small tests exercising the three findings. Confirm current implementation identities and necessary smoke coverage after code changes. Formal research results and final scientific conclusions remain outside this implementation review.
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
