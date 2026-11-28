# Implementation review | Direction A

## 1. Verdict — PASS

The Direction A algorithm implements the approved 2,048 pair draws per optimizer step, auxiliary coefficient .05 and temperature 1. The A1 arms meet the matched-exposure and baseline identity requirements. The shared-noise adapter meets the conditional A2 algorithm contract. Scientific dispatch requires the separate runtime review and its RAM verification. [Execution design](A_EXECUTION_DESIGN.md) · [Coder report](A_CODER_IMPLEMENTATION_REPORT.md) · [Algorithm plan](../../research_plans/26-11-28_three_directions/02_ALGORITHM_DESIGN.md) · [Matrix](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md) · [Implementation plan](../../research_plans/26-11-28_three_directions/04_CODEX_IMPLEMENTATION_SPEC.md)

## 2. Requirement verification

| Requirement | Result |
|---|---|
| Canonical Train pools and frozen sampler state | The adapter uses the verified read-only Train feature, label and stable ID arrays. Pool offsets preserve canonical Train order and map to stable IDs for the trace. All six cells have support, with 12 active conditions on MovieLens and 20 on KuaiRand. The fixtures check empty-stratum omission, missing-cell rejection and restore identity. Independent checks confirmed the tested endpoint IDs belong to Train and exclude Validation. |
| Macro6 sampling and loss | The fixed condition order, FP64 target and proposal, right-sided condition draw, and separate replacement stream match the design. The enumerated risk equals the mean six-cell softplus pair risk, and finite margin gradients have the intended sign. The independent computation matched the pair-ID hashes and `w/q` on both datasets' first and tail batches in all four arms. |
| Canonical logits, routing and forward order | Raw `(n,1)` tower tensors route through `3*t+d` to two logits per row. Their sigmoid matches the original clean, D0-local and pair probabilities at FP32 epsilon. Hooks are released on both success and failure. All arms run the clean BCE forward followed by one 4,096-row pair forward. The Gaussian arms add a D0-local BCE forward on the same ordered rows. |
| Exact clean and Gaussian BCE | The opt-in callbacks preserve the vendored scalar reduction, original model probabilities and default caller behavior. The empty-D0 case follows the clean reduction. Both datasets target domain 0. All ten archived first-draw fixtures matched through actual G0 adapter updates, including ordered BCE and D0 row IDs, target counts, keys and noise digests. Field order and exclusion of the domain embedding matched the registered source. |
| Pair views and gradient flow | B0/G0 run the clean pair forward under `no_grad` and use the base loss as the training scalar. P0 uses clean pair logits. GP0 uses the independent `gaussian_pairwise` stream on D0 and clean logits elsewhere. Repeated endpoints retain distinct draws, and the BCE noise stream is identical for G0 and GP0. Pair gradients enter native Adam through the ordinary model graph. The direct VJP fixture verified all four reported norms and the one-backward/one-probe counts. |
| Native Adam, fixed EMA and deterministic exposure | The configured Adam options, dataset weight decay, EMA coefficients, seeded batch order and original tail sizes match the design. Eight first/tail control updates matched the archived source exactly in predictions, loss, gradients, Adam moments, online parameters, EMA parameters and buffers. Independent matched runs preserved CPU and CUDA RNG state. Each step reports one native Adam and one post-Adam EMA update. |
| Conditional A2 | The shared mode uses GP0's draw and copies the positive field-noise block to the paired negatives. Each endpoint retains its own RMS. The off-D0 path is clean, and BCE noise, endpoint IDs and forward counts match the independent mode. The A1 roster contains 40 jobs and excludes `GP-shared`. A2 registration and dispatch remain contingent on the per-dataset selected Validation gate. |
| Trace and shared-runner boundary | The adapter binds ordered BCE and pair hashes to the step context and Train fingerprint. It records forward and row exposure, condition weights, pair multiplicity, losses, margin summary, gradient norms, view keys and noise hashes, timing and allocated/reserved memory. Strict Validation selection, RAM readback/resume, and E5/E10 retention are shared-runtime acceptance items. |

[A adapter](../../experiments/three_directions_20261128/a_step.py) · [Macro6 sampler](../../experiments/three_directions_20261128/macro_pair_sampler.py) · [A0 fixtures](../../experiments/three_directions_20261128/tests/test_direction_a.py) · [Shared protocol](../../experiments/three_directions_20261128/protocol.py) · [Source audit](../../artifacts/three_directions_20261128/SOURCE_AUDIT.json)

Independently ran all 16 A0 fixtures on physical GPU 1, an RTX 3090 with driver 570.144 and pinned PyTorch 2.9.0+cu126. The maximum absolute deviation in the eight archived control checks was zero for each measured gradient, Adam moment, online parameter and EMA parameter. The independent sampler check generated these seed-2022 IDs from the Train arrays and matched all four A1 arms at each context.

| Dataset | Epoch 1 step | BCE rows | Condition `(d,t,a,b)` | Ordered pair SHA-256 |
|---|---:|---:|---|---|
| MovieLens | 1 | 40,960 | `(2,0,2,0)` | `47d3a663c95b252a6ed97d69df19b3b2c844e01f9fd4bc7f1bc2188d9359ce17` |
| MovieLens | 20 | 21,927 | `(2,1,3,0)` | `617e30b1f8e7713dd65ec8d1ff318fd310d56bb99e19c7cfe2001876b3377d55` |
| KuaiRand | 1 | 40,960 | `(0,1,3,0)` | `5049e3fd3c305f99baf73d40e0ceaf9caf8a36894ea1b5e2bac01ae0ca189c36` |
| KuaiRand | 49 | 40,052 | `(0,0,3,0)` | `4eded932e5f11d3ffa7fb94236d83c6978161436790c3557fbb77802d9c29beb` |

The source checks verified the GA31 objective before amendment (`c4b58d0e230c03bfe69f8822ddac818d332d5ed1`) and after amendment (`1ad26a3ba28aa4affeefe12b43eb8477747009e9`). The registered views and key-protocol blobs matched exactly. The diff against main `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc` preserves the natural A21 sampler and scopes the vendored change to two optional callbacks. [Amended objective](../../experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py) · [Archived source register](../../research_plans/26-11-24_history_information/11_SOURCE_AND_REUSE_REGISTER.md#L63)

## 3. Bugs and deviations

No critical, major or minor algorithm defects found. The approved execution design requires all six supported cells and a fixed 2,048 draws, with missing support rejected at setup.

## 4. Corrections made

Review report only. Source edits were unnecessary. `git diff --check` passed.

## 5. Remaining risks

The CUDA checks used the available RTX 3090 stack. All source comparisons used identical starting states for each checked batch. These checks establish step identity and exposure. Shared runtime readback and resumed-schedule equivalence require their own acceptance evidence. The active study and history stop markers were absent in both original workspaces and the current worktree. The archived cycle04 stop and pause controls retain their prior study scope.

## 6. Run safety

A1 passes algorithm review. Dispatch requires shared-runtime approval and verified RAM checkpoint/readback under the supervised runner. A2 requires the per-dataset A1 GP0−G0 selected Validation mean ΔAUC of at least +.001 and the six-cell guard. A2 dispatch and matched A1.GP0 reuse need a review when that trigger resolves. Test stayed sealed and engineering checks started zero scientific trajectories.
