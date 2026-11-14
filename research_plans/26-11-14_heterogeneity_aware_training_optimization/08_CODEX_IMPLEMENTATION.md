# Codex Implementation Contract

## 1. Source inspection first

Inspect and record exact current paths and signatures for:
- M3oE fixed1 model;
- embedding lookup/output;
- STAR and expert/tower forward path;
- pre-sigmoid task logits;
- current BCE objective and domain-task aggregation;
- optimizer and fixed EMA update;
- data batch domain/task labels;
- validation selection and checkpoint save/load.

Do not assume a file path based on a historical branch. Main AGENT.md is authoritative. Keep fixed fusion and single Adam; do not restore learned fusion or AutoML/DARTS.

## 2. Preferred integration

Implement an augmentation adapter that:
1. receives actual embedded batch tensor and domain IDs;
2. samples target-positive/negative pairs per domain-task;
3. computes per-example target-logit gradients and non-target-logit Jacobian rows;
4. constructs TGA or DT-PGNA perturbations;
5. injects perturbed embeddings through existing forward graph without changing model weights;
6. computes target-only BCE on perturbed examples;
7. returns augmentation loss and diagnostics.

The clean forward and clean loss must remain equivalent when augmentation weight is zero, within existing deterministic tolerance.

If the model cannot expose embedding injection cleanly, add a narrow optional argument or adapter at the embedding boundary. Do not refactor experts, gates, towers, or fusion.

## 3. Gradient isolation and order

- Compute sensitivity gradients with torch.autograd.grad, not by calling backward on the training graph and accumulating into model .grad.
- Detach generated perturbation before augmented training forward unless a deliberate second-order method is separately proposed. The first version is first-order only.
- The augmentation loss is differentiated through the perturbed forward into model parameters.
- Sum clean and augmented loss once; call backward once; call optimizer.step once; update EMA once.
- No extra optimizer, no custom Adam state, no momentum coefficient changes.
- Keep clean batch order and natural sampling unchanged.
- Use a separate deterministic RNG stream for pair selection and perturbation; persist/restore its state.

## 4. Numerical controls

- Use stable pre-sigmoid logits. If only probabilities are exposed, add a narrow raw-logit readout without changing default public output.
- Projection solve uses float32 or float64 stable small-matrix solve; test singular/near-collinear Jacobians.
- Record q norm, projection residual, target margin before/after, non-target logit drift, skip rate, and backtracking count.
- Clip perturbation radius by frozen pair-level cap.
- Do not compute per-sample gradient by summing across examples unless sample independence is verified.
- Avoid in-place mutation of clean embedding tensor.
- Do not leak target labels into model input.

## 5. Tests

- sampler tests: no cross-domain pairs; target labels correct; other-task labels unconstrained;
- projection tests: Jq residual, exact simple linear toy case, rank-deficient J;
- first-order finite-difference tests;
- nonlinear drift and backtracking;
- zero-gradient skip;
- lambda_aug=0 baseline equivalence;
- optimizer/EMA exactly once per batch;
- resume determinism;
- two-dataset GPU smoke with real batch sizes;
- no accidental test-set read before freeze.

## 6. Required engineering output

Codex must report source file paths, changed files, test results, measured runtime/memory, model hash, config hash, data hash, seed, and commit SHA. Keep implementation isolated in a new experiment package; do not edit historical experiment packages.
