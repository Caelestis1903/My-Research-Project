# Implementation Specification for Codex

## 1. Allowed new paths

Create a new isolated implementation only after Stage 0 audit:
- experiments/auxiliary_objective_attribution_20261115/
- research_reports/auxiliary_objective_attribution_20261115/
- artifacts/auxiliary_objective_attribution_20261115/

Do not edit experiments/heterogeneity_aware_20261115/ or any historical/vendor model, protocol, report, metric ledger, data or checkpoint. Reuse current main model/data/optimizer/EMA utilities by import, after confirming exact identities.

## 2. Required modules

Suggested files:
- protocol.py: immutable dataset/model/optimizer/EMA and run identity.
- sampler.py: keyed row selection and support/acceptance trace.
- objective.py: clean, zero-delta auxiliary and natural-mixture objectives.
- sensitivity.py: embedding capture, task-logit Jacobian and perturbation construction.
- runner.py: deterministic full-budget training, validation and resume.
- metrics.py: six-cell/Macro/Traffic/Pooled AUC and unscaled LogLoss.
- analysis.py: paired statistics, effective-weight ledger and decision JSON.
- tests/: sampler, objective equivalence, zero-delta, Jacobian, deterministic resume and clean-path tests.

## 3. Run identity and deterministic RNG

Every trajectory identity must include:
- git commit and hashes of every imported/new source file;
- raw CSV SHA256, processed array hashes, row-order hashes and preprocessing/split identity;
- dataset, seed, model config and fixed1 coefficients;
- optimizer/WD/EMA settings, batch size, epoch count, precision;
- sampler and objective versions, all hyperparameters;
- validation selection and metric implementation version.

Use keyed independent RNG streams derived from dataset, seed, epoch, global_step and stream name. At minimum separate:
- loader permutation;
- pair candidate selection;
- example candidate selection;
- task-view subset selection;
- random-direction control;
- Gaussian control.

Do not consume the global torch/numpy RNG from inside a perturbation branch. Repeated invocation with the same context must reproduce identical row/task indices and random values.

## 4. Stage 1 implementation

Add Z-PAIR and Z-EXAMPLE as explicit arms in the new code, not as a branch that silently changes the original pilot.

For each B/Z-PAIR step:
1. Generate candidate positive/negative pairs with the same keyed sampler and K=128 cap.
2. Compute the same direction validity, pair-margin validity and (where applicable) A drift/backtracking validity rules as the source B/A implementation.
3. Record candidate and accepted row IDs, pair IDs, target task IDs, labels and skip reasons.
4. For B use the generated delta; for Z-PAIR replace accepted delta with exact zeros.
5. Compute the same target-only auxiliary probability BCE reduction on the same accepted support.

For each C/Z-EXAMPLE step:
1. Generate candidate example rows with the same keyed sampler.
2. Compute the same near-zero gradient and RMS validity rules.
3. Record proposed/accepted indices, task IDs, labels and skip reasons.
4. For C use the generated delta; for Z-EXAMPLE replace accepted delta with exact zeros.
5. Compute the same target-only auxiliary BCE reduction.

Because model states diverge during training, accepted masks can differ after the first steps even with keyed candidate rows. The causal estimand is the perturbation operator under the same candidate sampler and adaptive acceptance rule. If accepted-support differences become large enough to change interpretation, one targeted intention-to-augment control may be triggered; do not silently claim exact row identity.

## 5. Effective-weight accounting

For every batch export a machine-readable ledger with:
- B, M and M per domain-task cell;
- selected label counts by domain-task;
- unique selected rows and repeated row IDs across task-specific cell selections;
- clean BCE numerator/denominator;
- auxiliary BCE numerator/denominator;
- effective scalar coefficient per selected entry and ratio to clean row-task coefficient;
- contribution to total objective by domain, task and label.

For the current legacy loss, assert the formula:
L_total = mean over all 2B clean BCE entries + 0.25 * (2 * mean over M auxiliary entries).

Do not infer effective weights from lambda alone. For NDP-A/B, assert that the empirical objective targets the original natural row-task population and that the weights sum to the declared total scale.

## 6. Numerical and semantic tests

Mandatory tests:
1. Z-PAIR/Z-EXAMPLE delta is bitwise zero on accepted rows.
2. Zero-delta augmented forward equals the corresponding clean forward subset.
3. Zero-delta auxiliary loss equals clean probability BCE on the same rows/tasks.
4. Total loss and parameter gradients equal the explicit clean-plus-auxiliary formula.
5. M=0 gives exact clean objective and one optimizer/EMA update.
6. Candidate sampler reproducibility and RNG-stream independence.
7. Row/task support, label preservation and per-cell counts.
8. Probe forward outputs match ordinary forward outputs exactly where required.
9. Row-wise Jacobian from logit-sum gradient equals stacked row gradients on the actual model; adding unrelated rows does not alter a row's gradient. Verify the source model's row independence; do not assume it if modules change.
10. Finite differences on smooth points, including the gate path if the model's numerical function requires it. Use float64 copy with same FP32-initialized coefficients and the audited step/tolerances from the prior review; avoid ReLU kink rows.
11. Delta is detached; training gate-detach semantics remain unchanged.
12. Exactly one optimizer and EMA update per batch; deterministic resume reproduces uninterrupted continuation and selected checkpoints.

## 7. Stage 2 implementation details

### NDP-A full-batch view mixture
- For every original row and task, calculate the task-specific BCE gradient with respect to the embedding.
- Construct a detached first-order adversarial delta with radius alpha * RMS(e), normalized by the task gradient norm.
- Use a fixed, documented sign convention equivalent to gradient ascent on target BCE; it creates a locally harder view, not a direct ranking improvement.
- Evaluate each task loss on its own perturbed view only.
- Combine clean and augmented per-entry BCE with convex mixture lambda, preserving the same 2B row-task entries and weights.
- Do not add a second independently normalized auxiliary BCE mean.

### NDP-B subset estimator
- Select row-task entries with known inclusion probability pi, independent of labels unless a label-stratified design is explicitly required.
- Use the Horvitz-Thompson correction for the augmented-minus-clean loss difference.
- Keep pi positive for every eligible row-task entry; record realized inclusion and weights.
- Compare against NDP-A only if cost makes it decision-relevant. Do not cap importance weights without quantifying bias.

## 8. Model/optimizer/EMA state discipline

For each minibatch:
- optimizer.zero_grad once;
- sensitivity extraction must use autograd.grad and must not populate parameter .grad;
- generate detached delta;
- compute clean and augmented views;
- one combined backward;
- one Adam step;
- one after-Adam EMA update.

Validation and Test are unperturbed clean inference. EMA state is initialized as a copy of initial model and updated at the inherited timing. Keep model, optimizer, EMA, loader RNG and all continuation state in RAM or a deterministic state guardian; do not save model checkpoints to historical paths.

## 9. Data outputs

Write immutable:
- run_manifest.json and source_receipt.json;
- per-trajectory result.json and selected validation endpoint;
- per-epoch validation curve;
- per-batch objective/effective-weight ledger;
- sampler/support diagnostics;
- perturbation norm and drift diagnostics;
- COSTS.csv, ENDPOINTS.csv, PAIRED_SEEDS.csv, PAIRED_SUMMARY.csv;
- decision.json and report.

Never overwrite a previous attempt. Use unique config IDs and append attemptN on retries. If a source/data/config hash differs, stop reuse and retrain matched arms.
