# Scheduler and Resource Plan

## 1. Queue order

1. Static source audit and CPU unit tests.
2. One real-batch GPU smoke on each dataset, one seed, no scientific inference.
3. Five-seed pilot for all predeclared arms.
4. HPO only if pilot numerical and validation gates pass.
5. Five-seed confirmation after config freeze.
6. Post-freeze test and report.

A smoke failure is an engineering failure, not a scientific negative. An untriggered stage is NOT_TRIGGERED.

## 2. Cost controls

- Use K=128 pairs per domain-task cell initially; cap by available positive and negative examples.
- Deduplicate repeated samples within each batch where possible.
- Reuse clean model forward values and embedding tensors.
- Profile sensitivity-gradient cost separately from augmented forward cost.
- Do not keep full per-example Jacobians on GPU after perturbations are generated.
- Release intermediate graphs before optimizer step where safe.
- Record GPU allocated/reserved peaks, wall-clock per epoch, pair count, skip rate and augmentation forward count.

## 3. Memory lifecycle

Follow current project RAM lifecycle rules:
- keep model, optimizer, EMA, RNG and active checkpoint state in RAM only for the active run;
- do not serialize full model/optimizer checkpoints to disk unless established run protocol explicitly permits it;
- save scalar metrics, manifests, hashes, logs and compact diagnostics;
- on resume, restore complete allowed state through project's existing mechanism;
- ensure guardian/worker cleanup and no orphaned process remains.

## 4. Stop and recovery

- Every worker claims a unique immutable run ID.
- A failed attempt is recorded and not silently overwritten.
- Recovery verifies model/config/data hashes and optimizer/EMA/RNG state.
- Resource unavailability is WAIT_RESOURCE / NOT_DISPATCHED, not a method failure.
- No repeated run is counted as a new independent seed.
