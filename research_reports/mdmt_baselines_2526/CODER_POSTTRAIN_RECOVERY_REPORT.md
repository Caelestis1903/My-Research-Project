# Coder implementation report — one posttraining completion recovery

Status: **IMPLEMENTED; independent review pending.** The recovery command was not run. No checkpoint, prepared array, V logits/metrics artifact or GPU was loaded for this implementation pass.

## Delivered

- Added `runtime_posttrain_recovery.py`, restricted to the existing formal `AUTOIFS/kuairand/seed2022` trajectory. `selfcheck` is CPU-synthetic. The separate `recover` command refuses an existing recovery directory, preserves the original `run.json` bytes, and launches one worker only.
- The worker reads the existing review-approved R2 identity, audits all 17 epoch-log records and the reuse selection/patience rule, checks search/rewind/reuse counters and the pruning summary, and validates checkpoint stage/cursor/optimizer/RNG/model metadata and original embedded identities.
- The Crash4 prerequisite is consumed only when the single GDB receipt reports one normal, signal-free `NOT_REPRODUCED` completion and both complete V views show exact logged parity. Recovery directly checks the receipt, checkpoint paths, row-ID file and stage log. On the actual artifacts, the structural gate passed without loading checkpoints, arrays, logits or metric files.
- When invoked after independent review, the worker restores selected and fixed-final AutoIFS states strictly and their ticket/temperature metadata, recomputes V metrics from the existing Crash4 logits, then measures selected inference using the existing 5-warmup/25-repetition procedure on the first `min(256, N_V)` rows of `cuda:1`. It has no training, optimizer, backward, E-label, S-row or legacy-row call.
- Completion preserves the original checkpoint-embedded identity and writes `TRAIN_COMPLETE_NOT_EVALUATED`, `completion_mode=CHECKPOINT_POSTTRAIN_RECOVERY`, all original stage records, selected/final checkpoints and metrics, and the `posttraining_recovery` receipt path. Original `training_seconds`, `total_elapsed_seconds`, and `peak_cuda_bytes` remain JSON `null`; the epoch-duration lower bound and original wrapper wall time are separately recorded. The recovery and inference durations are separate. The original `LAUNCH_STARTED` receipt and exit-139 record are read-only inputs.
- Atomic replacement is the final worker operation after all checks and receipts are written. A failed attempt leaves the canonical run manifest as-is and reserves the one-use recovery directory; it cannot be retried through `recover`.

## Verification

Using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- AST parse: **PASS**.
- `runtime_posttrain_recovery.py selfcheck`: **PASS_CPU_SYNTHETIC_ONLY**. Negative cases cover changed run identity/source, invalid selection/patience/checkpoint metadata, missing or mismatched Crash4 evidence, missing stage boundaries, and a second execution. A synthetic completed manifest retains the checkpoint-embedded `TRAINING` identity while its three unavailable original cost fields remain null.
- Structural check of the completed Crash4 JSON receipts: **PASS**. It checked receipt identity, D1/cuda:1, both checkpoint references, the 250,767-row V count and both view summaries. It did not read V labels, checkpoint files, logits or metric arrays.

Crash4's recorded outcome is `NOT_REPRODUCED`: the GDB inferior exited normally without SIGSEGV; both views completed 62 batches and their logged V metrics were marked exact. This is prerequisite process evidence only and does not resolve the native cause. The actual recovery remains blocked on the separate independent review.

No real training, recovery inference, dependency installation, configuration/package/R2-evidence change, or new hash calculation was performed.

## Independent-review fix — RCV-1

The direct-script worker now inserts the repository root into `sys.path` immediately after establishing the D1 stack and before importing the experiment package. Added `importcheck`, which exercises the same direct-file invocation path and imports only the modules used by recovery; it does not load prepared data or checkpoints.

Verification from `/tmp`, using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `python /home/xzf/Code/phase0-3/research_reports/mdmt_baselines_2526/runtime_posttrain_recovery.py importcheck`: **PASS_CPU_DIRECT_SCRIPT_IMPORT_ONLY**; all six recovery modules imported, repository root was on `sys.path`, and D1 Torch threads were 1/1.
- No recovery directory, checkpoint, prepared array, V output, or GPU inference was used. Remaining independent review findings are pending.

## Independent-review fix — RCV-2

Checkpoint validation now checks the fixed run's serialized Adam groups against the search/reuse ownership logs and registered `lr`, `weight_decay`, `betas`, and `eps`. It requires the expected local update count for rewind (3798), selected reuse (844), and fixed-final reuse (1266), and checks each present Adam state has the expected step, float32 moments and parameter shape. The one zero-element `embedding.dense_vector` parameter is explicitly expected to have no Adam state; other group/state IDs must match.

RNG validation now rejects absent/malformed Python and MT19937 states and requires the recorded Torch CPU state (uint8[5056]) and both CUDA states (uint8[16]) to be nonempty and correctly typed. These snapshots are inspected only; they are not restored or consumed. No optimizer is constructed.

The CPU synthetic self-check passes the exact step/hyperparameter/state structure and zero-element parameter case, and rejects a reset Adam step, missing state, changed learning rate, and null/empty Python, NumPy, Torch CPU, or CUDA RNG records. It did not load any real checkpoint or initialize recovery inference. Remaining independent review findings are pending.

Re-run after RCV-2 using the direct-script command from `/tmp`: `runtime_posttrain_recovery.py selfcheck` returned **PASS_CPU_SYNTHETIC_ONLY**. No real checkpoint was loaded, no optimizer was created, and no RNG state was restored or consumed. The recovery command remains unrun pending review.
