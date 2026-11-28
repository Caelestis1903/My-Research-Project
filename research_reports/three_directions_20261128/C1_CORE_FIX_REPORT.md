# CODER IMPLEMENTATION REPORT | C1 core fixes

## Changes

- Existing-roster validation checks every dataset/seed against the verified C0 manifest. It matches the complete candidate and P0 margin receipts, ordered snapshot labels and model hashes, and the E10 fixed-EMA teacher's original job key, epoch, global step, model SHA and RAM receipt SHA. The verifier authenticates the same manifest bytes it parses. The adapter repeats this check on the exact input record before preparing the bank at E10 or rebuilding it on recovery. [Protocol](../../experiments/three_directions_20261128/c1_protocol.py#L295) · [Adapter](../../experiments/three_directions_20261128/c_step.py#L51)
- Committed replay compares the reported event weights, uniqueness, miss reason and actual endpoint work with the reconstructed event. It accepts zero pair work, null weights and zero unique counts when the sampler yields no event. A valid event in an empty bank uses the current pair and reports full pair work. [Replay](../../experiments/three_directions_20261128/memory_bank.py#L203)

## Shared integration contract

Pass the exact seed record into `make_training_parts` and validate the record the adapter supplies to the callback.

```python
def verify_inputs(inputs):
    c1_protocol.verify_seed_inputs(
        artifacts, expected=roster["gates"], dataset=job.dataset, seed=job.seed, inputs=inputs,
    )

seed_inputs = roster["seed_inputs"][f"{job.dataset}_{job.seed}"]
optimizer, adapter = c_step.make_training_parts(model, job, bundle, seed_inputs, verify_inputs)
```

Replay takes chronological committed rows from E1 onward. Write the observed work to `counts` and the event details to `diagnostics`.

| Field | Warmup, bank or fallback event | No event |
| --- | --- | --- |
| `counts.pair_rows` | 4,096 | 0 |
| `counts.pair_draws` | 2,048 | 0 |
| `counts.pair_forwards`, `counts.pair_diagnostic_vjp_calls` | 1 each | 0 each |
| `diagnostics.w`, `diagnostics.q`, `diagnostics.w_over_q` | Stock sampler `w`, `q` and exact `w / q` | `null` each |
| `diagnostics.unique_pairs`, `diagnostics.unique_endpoints` | Exact applied draw counts | 0 each |
| `diagnostics.condition`, `diagnostics.pair_index_hash`, `diagnostics.current_pair_index_hash` | Sampled condition, applied pair hash, stock pair hash | `null` each |
| `diagnostics.memory_source` | `current`, `bank` or `current_fallback` | `none` |
| `diagnostics.memory_miss` | `empty_bank` for fallback, `null` otherwise | `no_event` |

Include `diagnostics.train_identity` and `memory_phase` on every row, `bank_hash` on every memory-phase row, and `draw_positions_hash` on bank events. The C adapter supplies memory-specific diagnostics. The shared transaction must report actual forwards and VJPs and run the clean BCE update on a no-event step. [Design](C1_EXECUTION_DESIGN.md) · [Core review](C1_CORE_IMPLEMENTATION_REVIEW.md)

## Checks and remaining work

- The pre-fix regression reproduced acceptance of a rehashed roster pointing to an otherwise valid all-zero E10 archive and acceptance of impossible replay work. The corrected tests reject each discrepancy independently and cover current, bank, fallback and no-event rows. [Regression tests](../../experiments/three_directions_20261128/tests/test_c1_memory.py)
- The five focused test modules passed **62 tests** with CUDA hidden. Twelve CUDA cases were skipped. The existing PyTorch TF32 deprecation warning appeared. All four changed Python files compiled. The read-only gate check validated the ten live C0 seed receipts and all ten scoped stop paths were clear.
- The independent core review and shared integration acceptance remain pending, including the historical source archive, shared transaction, real CUDA cases and formal runner entry point. All checks here were CPU implementation checks or read-only receipt checks. Test remains sealed.
