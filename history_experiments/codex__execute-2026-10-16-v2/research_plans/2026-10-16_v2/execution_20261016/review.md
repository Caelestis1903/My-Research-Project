# PRIMARY SELF REVIEW

DESIGN and actual worker inspected after implementation. This is not independent review; latest start.md forbids non-running subagents. PASS_PRIMARY_SELF_REVIEW for engineering run; formal queue requires CUDA profile as an additional gate.

Checked: old training loop not inherited (there is only one optimizer/no arch step); constants replace learned scalar modules while preserving other initialization; alpha/q and task mean match specification; no objective-gradient balancing; full train support; shared RNG pairs across optimizer/objective arms; held-out preprocessing unchanged; snapshots one-based; initial excluded from checkpoint selection; earliest strict validation argmax saved before tests; calibration has no test path; no source change allowed during active worker; attempt dirs require nonexistence. Optimizer telemetry reads detached values; first-two-step lag1 state resets every epoch. Actual routing gates are R, architecture scalars have no parameters.

Known limitations: four-seed bootstrap low precision; fixed-fusion result is a protocol boundary to historical learned fusion, not exact identity; SGD grid may underoptimize and cannot prove necessity by a null alone; one-step probes alone cannot establish a performance mediator; no verified partial-epoch resume CLI (complete states retained). Cyclic controls are not yet experimentally authorized until supplemental design. Queue initialization headroom is conservative and admits cards independently.

CUDA engineering found default-device memory telemetry (0 when model cuda:1). Fixed only reset/max/synchronize to explicit model device after both workers exited. Old statistic archived as invalid. Full49-step epoch succeeded; training semantics unchanged. New profile required before queue.

Profile attempt2 exposed PyTorch reset-before-init failure. Set actual CUDA device before resetting peak stats; no model/loss/sampler change. Preserve failed attempt2; profile attempt3 required.

Mechanical queue review: one persistent scheduler can append primary-authored prerequisite-satisfied tasks; existing ids/args/paths immutable. Supplemental worker paths require their own self-review SHA and matching committed source. Each task has explicit measured memory reservation; per-device cap includes other v2 queues. No scientific branch decisions are delegated to the queue.

Final engineering gate PASS: original full epoch49updates exit0; corrected device-specific profile attempt3 exit0,1192MiB peak,1942.4MiB reservation. Formal training can start after committing identical worker+queue. Probe source reviewed separately against supplement: immutable model loads, autograd.grad without optimizer steps, finite pool/projections labeled, PSD/mean checks mandatory; no scientific verdict delegated.
