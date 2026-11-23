# D2 implementation review

## 1. Verdict

**FAIL.** D2-R1, D2-R2 and D2-R3 are closed. D2-R4 still drops completed diagnostic costs when a restarted supervisor encounters a prior journal without a worker receipt. The verdict binds implementation SHA-256 `87137de4a13274e6f79cb86d5d309a9c62278ce679bc7271f5187b587f86f861` and review-input SHA-256 `7f389ae8d9957fd2ae415331da6a870a8216642d3f10cc456ee641d094f197cb`. [Bound receipt](IMPLEMENTATION_REVIEW.json)

## 2. Requirement verification

| Requirement | Independent verification |
| --- | --- |
| Fixed1 source, data and baseline fidelity | Recomputed all reviewed source and evidence hashes. The source manifest binds the loaded 11-22 `state.py` to its pinned Git blob, and the review input binds the same bytes. A simulated mismatch was rejected. Train/Validation and frozen panel hashes match the initial review. The refreshed support CSV binds the new source manifest. The first and natural tail CPU receipts match donor probabilities, BCE, model, EMA and Adam for both datasets. |
| Shapes, routing and gradient support | The model, source adapter and native update module match the initial review hashes. The core fixtures verified the routed probabilities, disjoint parameter ownership, and native None and explicit-zero gradient semantics on rerun. The unchanged parameter manifest retains the `[F,B] → [B,2]` routing audit, six-cell and off-axis evidence, and detached gate inputs with trainable gate weights. |
| Native Adam and EMA | The code retains row-weighted accumulation at a common online state and recomputes the second forward in either sequential order. S1's two EMA tick states and canonical J_ORIG are preserved. The CPU fixtures reran equal and unequal concatenated-gradient comparisons, sequential poststates and masked native Adam with existing moments and coupled decay. |
| J_ORIG schedule, selection and RAM restore | The worker still completes 50 natural permutations with tails and selects each readout's earliest strict Validation maximum independently. The loader, selection and mocked hash-verified RAM replay fixtures passed. The resumed E10 fixture reached the E11 callback twice, published one fresh allocation acknowledgment and preserved the larger diagnostic residency sample. The existing scheduler predicate allowed further admission on the sampled card when the fresh measurements qualified. |
| S1 diagnostic matrix and panel evidence | The named arms, AB/BA mappings, after-U readouts, and pair metrics still follow the design and erratum. The fixture exercised all 19 native forks, verified parent state and RNG purity, and matched physical counts to the scientific counts on a complete anchor. Accepted scientific outputs still require the whole anchor. |
| S1 decision and controls | The statistics module is byte-identical to the initial review. The fixtures passed for V_A scope freezing, online V_B confirmation, the six-cell guard, anchor replication, seed sensitivity and the proper-subset S2 clock handoff. This review covers S0 and S1. |
| CUDA entry point and Test seal | Standalone `s0` exposes `prepare` and `cpu`. The test rejects the former CUDA action. The staged `cli smoke` checks stop state, the bound passing review, the runtime environment, and live resource admission before invoking CUDA fidelity. The reviewed data receipt marks Test sealed. |
| Retry accounting | Completed diagnostic clones, updates and readouts are now journaled and a worker failure records their counts. The supplied interrupted fixture preserves three Adam updates and four Validation readouts without accepting the anchor. Final publication counts accepted fork work once. Previously committed journals without a corresponding top-level worker receipt still disappear after supervisor restart. |

## 3. Bugs and deviations

- **MAJOR D2-R4.** `scheduler.publish()` reads journals only while iterating `attempts/*.json`. The missing-receipt fallback in `dispatch()` only runs when the current supervisor reaps a process in its in-memory `live` map. An interrupted supervisor or host can leave a committed journal and no attempt receipt. Restart can recover or replay the logical run, but the resulting `COSTS.csv` omits the completed diagnostic work. An independent CPU reproduction wrote a prior journal containing one completed Adam update to a temporary run directory, supplied a completed replacement result and called the real publisher. The prior attempt had no row in `COSTS.csv`. Include durable journals from prior attempts in final physical accounting even when the worker receipt is missing, keeping the existing fork-key rule for accepted work. [Publisher and current-process fallback](../../experiments/mdmt_shared_private_coordination_20261123/scheduler.py)
- **MINOR.** `CODER_REPAIR.md` adds a leading digit to the prose SHA-256 for the donor `state.py`. The actual file, the pinned Git blob and both machine-readable bindings agree on `93d10e84df94acbee5dae41c405c6b264b2b1337a6b27fe4cd6d3c604c8fcf74`. [Coder repair report](CODER_REPAIR.md)

## 4. Corrections made

This review updated the Markdown report and its bound JSON receipt. Implementation changes to D2-R4 remain with the coder. [Initial review](IMPLEMENTATION_REVIEW_INITIAL.md)

## 5. Remaining risks and evidence

All 19 D2 CPU tests passed in the project environment. PyTorch emitted the frozen donor TF32 setter deprecation warning. The canonical real-batch CPU comparisons were verified from the refreshed receipts. Genuine CUDA first/tail fidelity, live guardian replay and full CUDA process residency are exercised by the separate smoke. [Canonical CPU fidelity](../../artifacts/mdmt_shared_private_coordination_20261123/S0/MODEL_FIDELITY.json)

## 6. Execution clearance

**Return D2-R4 to the coder before CUDA/RAM smoke.** Formal S1 also requires separate smoke acceptance. Test remains sealed.
