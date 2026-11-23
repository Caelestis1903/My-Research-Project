# D2 implementation review

## 1. Verdict

**PASS WITH FIXES.** All four implementation findings are closed. The final D2-R4 repair preserves completed diagnostic costs across a supervisor restart and counts accepted work once. The review binds implementation SHA-256 `f45b68ef687179f756f1397314b47d46359fbd8ecbdd9312ef6da3704d819202` and review-input SHA-256 `ec0e196c5cebda442a307a86f4458a1d8f21a2542bce607ee743fda154339e2d`. [Bound receipt](IMPLEMENTATION_REVIEW.json)

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
| --- | --- |
| Source, data and baseline fidelity | Recomputed 21 bound input hashes, nine evidence hashes and all 72 donor source hashes. All S0 CPU evidence, including native first and natural-tail Adam fidelity for both datasets, matches the prior independent review. The design, erratum, data and frozen panels match that review. |
| Routing, gradients, Adam and EMA | The reviewed code and numerical evidence for fixed1 routing, detached gate inputs, parameter membership, masked and explicit-zero gradients, weighted accumulation, sequential recomputation and EMA ticks are byte-identical. The prior independent checks remain valid. |
| J_ORIG schedule and RAM restore | Scheduler edits are confined to final cost publication. The focused CPU run passed 50-epoch permutation coverage, strict independent Validation selection, mocked RAM replay and the resumed startup acknowledgment regression for D2-R1. |
| Executable source closure and CUDA entry point | D2-R2 continues to bind the donor `state.py` in both manifests. D2-R3 continues to route CUDA fidelity through gated `cli smoke`. The focused run passed the direct-entry rejection and stop-control checks. |
| S1 diagnostic matrix and decisions | The erratum registers 19 branches per anchor and 570 across the planned S1. The focused run passed complete-anchor purity and accounting, V_A scope freezing, independent online V_B confirmation and the proper-subset S2 clock gate. |
| Physical retry cost after restart | The publisher enumerates every `attempts/*/FORK_COSTS.json` for each completed logical run and uses the worker receipt status when available. The focused restart fixture confirms one completed orphaned optimizer step and its 0.125-second phase survive with `WORKER_RECEIPT_MISSING`. Unknown attempt wall time, host RSS and GPU process memory remain blank. A prior accepted fork and replacement accepted forks each publish once under their accepted records. The interrupted native diagnostic fixture retains all journaled retry phases for an attempt with a worker receipt. |

The prior scheduler hash was recovered by reversing the publisher edit in memory. The prior runtime-test hash was recovered by removing the CSV import and added restart test. The earlier S0 and S1 implementation review applies to all other bound inputs. [Prior independent review](IMPLEMENTATION_REVIEW_REPAIR1.md)

## 3. Bugs and deviations

- **MAJOR D2-R4, fixed.** A prior supervisor could leave an atomic diagnostic journal without a worker receipt. The publisher now finds that journal directly and includes its unaccepted phases in `COSTS.csv`. It still uses accepted fork keys to avoid double counting. [Publisher](../../experiments/mdmt_shared_private_coordination_20261123/scheduler.py#L137)
- **MINOR donor SHA-256 typo, resolved.** The coder report now agrees with the donor file and the machine-readable source and review bindings. [Coder repair](CODER_REPAIR.md)

## 4. Corrections made

The coder supplied the publisher fix and regression fixture. This review replaced the canonical Markdown and JSON review pair. The initial and first repair review pairs retain their original contents. [Restart regression](../../experiments/mdmt_shared_private_coordination_20261123/tests/test_runtime.py#L176)

## 5. Remaining risks and evidence

The 14 focused CPU tests passed. PyTorch emitted the existing frozen donor TF32 deprecation warning. The journal records completed phases that reached publication. A worker exit before receipt publication can leave aggregate runtime and resource measurements unavailable. Real CUDA first/tail fidelity, live guardian recovery and full CUDA process residency remain to be verified in smoke.

## 6. Execution clearance

**The implementation is cleared for gated S0 GPU/RAM engineering smoke.** Formal S1 requires separate acceptance of the real smoke evidence for this reviewed implementation and RAM owner. This receipt covers S0 and S1 implementation review. Test remains sealed.
