# REVIEW REPORT | C1 core fixes

## 1. Verdict

**PASS** for the C-only core. Both MAJOR findings from the initial implementation review are closed. Shared runtime integration and CUDA acceptance remain separate gates for C1 training. [Design](C1_EXECUTION_DESIGN.md) · [Initial review](C1_CORE_IMPLEMENTATION_REVIEW.md) · [Coder fix report](C1_CORE_FIX_REPORT.md)

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
| --- | --- |
| Authenticate the exact frozen input record | The existing-roster path checks each dataset and seed against the authenticated C0 manifest bytes. It matches the complete candidate and margin receipts, the ordered snapshot labels and model hashes, and the P0 fixed-EMA E10 job key, epoch, step, model SHA and RAM receipt SHA. A rehashed disposable roster with an otherwise valid all-zero E10 archive failed the roster check. A read-only check verified the ten original live seed records and matching archive SHA, target shape, finite FP64 values, candidate SHA and E10 model SHA. It also reproduced rejection of an all-zero replacement against the live seed gate. [Protocol](../../experiments/three_directions_20261128/c1_protocol.py#L295) · [Regression](../../experiments/three_directions_20261128/tests/test_c1_memory.py#L477) |
| Apply the same identity gate at E10 and recovery | Both fresh E10 installation and E10-or-later recovery call `_prepare`, which passes the adapter's actual input record to the verification callback before loading archive bytes or building a bank. The replacement failed at that callback in the CPU fixture. The loader checks the size and SHA of the exact bytes opened with `allow_pickle=False`. [Adapter](../../experiments/three_directions_20261128/c_step.py#L52) · [Loader](../../experiments/three_directions_20261128/memory_bank.py#L29) |
| Match committed exposure and actual work | Replay recomputes the current and applied identity hashes, selected condition, source and miss reason, pair work, exact `w`, `q` and `w / q`, and applied distinct counts. It validates the bank hash in memory phase and position hash on bank draws before aggregating exposure. The original combined impossible trace retained valid hashes and 2,048 draws but reported zero endpoint work and VJPs, zero importance weight and an empty-bank miss. Replay rejected it at logical step 11. Individual field tests rejected each discrepancy. Valid warmup, bank, fallback and no-event traces replayed successfully. [Replay](../../experiments/three_directions_20261128/memory_bank.py#L203) · [Regressions](../../experiments/three_directions_20261128/tests/test_c1_memory.py#L238) |
| Preserve warmup, sampling and gradient semantics | The focused C1 suite covered the exact P0 config, stock warmup and E10 rollback, bank sign and capacity, C2/C3 common positions, C3/C4 matched ordered endpoint blocks, and detached FP32 soft targets with reference gradients for soft and hard loss. C-only review work left the shared transaction bytes untouched. [C1 tests](../../experiments/three_directions_20261128/tests/test_c1_memory.py) |

All **34 focused C1 CPU tests passed** with CUDA hidden. All ten study stop sentinels were absent across the registered workspaces at the time of checking. The live checks used existing C0 Train receipts and archives. [Stop scopes](../../experiments/three_directions_20261128/scheduler.py#L24)

## 3. Findings

- **MAJOR, resolved.** The nested receipt substitution now fails at existing-roster validation and at the callback before bank construction.
- **MAJOR, resolved.** The impossible work and weight trace now fails before exposure aggregation.

## 4. Corrections reviewed

This review verified the coder's two corrections. The review report is the sole workspace edit.

## 5. Remaining integration checks

Use the exact `roster["seed_inputs"][f"{job.dataset}_{job.seed}"]` record in `make_training_parts`. The callback must accept the record supplied by the adapter and call `verify_seed_inputs(artifacts, expected=roster["gates"], dataset=job.dataset, seed=job.seed, inputs=record)`. The earlier coder handoff's gate-only callback is superseded by this contract. [Callback](../../experiments/three_directions_20261128/c1_protocol.py#L324) · [Adapter](../../experiments/three_directions_20261128/c_step.py#L12)

Pass chronological committed rows from E1 onward, using the committed writer for each epoch. Record actual `counts.pair_rows`, `pair_draws`, `pair_forwards` and `pair_diagnostic_vjp_calls` as 4,096, 2,048, 1 and 1 for warmup, bank and fallback events, and zero for no-event. In `diagnostics`, use exact applied uniqueness counts and stock `w`, `q` and `w_over_q` for events. No-event has zero uniqueness counts and null weights, condition and pair hashes. Log `train_identity`, `memory_phase`, `memory_source`, `memory_miss`, `condition`, `current_pair_index_hash` and applied `pair_index_hash` on every row, `bank_hash` on every memory-phase row and `draw_positions_hash` on bank events. Sources are `current`, `bank`, `current_fallback` and `none`. The associated miss reasons are null, null, `empty_bank` and `no_event`. The shared transaction still needs to demonstrate actual work and retain the clean BCE update when the pair event is absent. [Replay contract](../../experiments/three_directions_20261128/memory_bank.py#L212) · [Adapter diagnostics](../../experiments/three_directions_20261128/c_step.py#L118)

## 6. Run authorization

The corrected core is cleared for shared integration once the B1 terminal-state and supervision gates are met. Formal C1 training still requires the frozen A1/B1 source archive, reviewed C1 runner integration, roster verification and CUDA acceptance. Test stays sealed. [Acceptance gates](C1_EXECUTION_DESIGN.md#10-reviewer-acceptance)
