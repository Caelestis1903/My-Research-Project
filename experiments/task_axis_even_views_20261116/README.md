# Task-axis even views — cycle 01

The approved seven-file design has SHA-256
`f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754`.
Only AXIS-EVEN, AXIS-IID2 and ISOTROPIC-EVEN run: two datasets, seeds
2022–2026, 50 epochs, batch 40960 including the natural tail, FP32,
51750 Adam/EMA updates each. Twenty exact historical CLEAN/NDP-A results
are read-only controls. AXIS-EVEN is the registered candidate; controls
cannot be promoted by the exporter. Test has no CLI/evaluation path.

All commands run from `/home/xzf/Code/phase0-3_tvg_rm_20261116` with
`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -u -m
experiments.task_axis_even_views_20261116.runner COMMAND`.
`--attempt` defaults to the sole authorized isolated root:
`artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1`.
An aliased root or output directory is rejected.

1. Finish both coder reports, then obtain the complete current source receipt
   using `protocol.source_manifest()`. A distinct fresh reviewer must inspect
   the seven-file design, all source and both reports. The review JSON requires
   `verdict` PASS or PASS WITH FIXES, `blocking_issues: []`,
   `source_hash`, `design_bundle_sha256`, and `cpu_checks_passed: true`.
   A review does not claim GPU smoke success.
2. `prepare --review /absolute/current/review.json` verifies the reviewed source,
   installed Python/torch versions, all twenty pinned controls and their source,
   initial hashes and 50 epoch orders, verified source caches and old inventories.
   It creates only cycle-local numerical receipts. Optional `--cache-dir` selects
   the existing verified cache; a missing cache is a pause, without reconstruction.
3. **The original foreground parent** starts `_guardian` through its own
   persistent foreground `exec` tool session and retains that session/PID.
   The wrapper calls only the unchanged frozen TV `serve(new_attempt)`; it refuses
   any existing `guardian.json`. No shell backgrounding or detached session.
   `run`, `smoke`, recovery and attachment never start storage.
4. `attach --owner-session TOOL_SESSION_HANDLE` records that parent-owned session
   and verifies the distinct root/live PID/socket. Socket credentials remain only
   in the private `guardian.json`; public receipts contain no authentication key.
   After a fresh-agent handoff, the same command checks the existing immutable
   identity and attaches to that live owner. Missing/incompatible/lost RAM pauses.
5. A fresh reviewed runner invokes `smoke --review REVIEW --device cuda:0` in a
   supervised foreground session. It performs one source-ordered 40960 FP32 step
   for each arm/dataset, measures board power and memory, and checks private RNG,
   support and BytesIO round-trip identity. Independent reinitializations check
   all three null/source identities and unchanged one-sided/source identity on
   the first full batch and the actual final tail rows (ML 21927, KR 40052).
   Gradients including None masks, every Adam counter/moment, model/EMA tensors
   and buffers, global RNG and loader state must match exactly. No validation
   tuning or smoke epochs. Failure records PAUSED; no automatic retry or budget
   reduction. A failed smoke receipt is preserved for the parent/reviewer.
6. `run --review REVIEW --devices 0,1` requires verified smoke and existing live
   storage. It owns the PIDs and stdin pipes of same-session mechanical workers,
   samples both cards independently, reserves pending allocations, and requires
   projected memory and measured power below 85%, host RAM below 90% including
   pending workers and transient CPU/IPC copies. No process-count cap and no
   requirement that both cards be admissible. The foreground tool caller must
   stay in the current conversation and supervise the returned tool session.
   There is no heartbeat/lease timer. Actual parent process loss, signal or
   scheduler pipe EOF stops admission and requests safe epoch/RAM-ACK pause.
7. `status` reads the cycle owner's key/status/epoch/count/retained-byte inventory.
   Stop controls in **both** workspaces are checked before launches and epochs;
   a stop during an epoch finishes batches, validation and acknowledged RAM
   commit where technically possible. Technical errors preserve the previous
   acknowledged state and write only the cycle's `PAUSE_TRAINING`/failure records.
8. `recover --key KEY --review REVIEW` is an explicit unchanged-source/identity
   RAM recovery after all recorded workers have died. It never resets epochs,
   auto-retries, changes batch/precision or reconstructs lost RAM from disk.
   Epoch 50 can finalize from the committed RAM snapshot without retraining.
9. `tables` requires all 30 new plus 20 pinned control results, all 50 validation
   epochs and matching loader orders. It exports online/EMA independent selected
   endpoints and E10/E30/E50, all cells/domain/task/Macro/traffic/pooled/worst
   metrics, loss/support/cost/diagnostic facts and the four registered paired
   contrasts. Seed is the unit, n=5, SD ddof=1, descriptive CI uses
   2.7764451051977987. Positive/negative/zero counts are separate. It makes no
   scientific decision or candidate ranking. `PHYSICAL_ATTEMPTS.csv` preserves
   completed, paused and failed attempt costs; the launch/exit ledgers also
   retain owned process lifetimes. A missing terminal cost is identified in the
   final report rather than presented as a measured zero.
10. `handoff --owner-session HANDLE` emits the live numeric owner/key/selected
    hash/retained-byte receipt. **Parent** runs `accept-handoff --owner-session
    HANDLE`, verifies the tool session remains live and accepts supervision
    **before** permanently closing the native runner. The same mechanical owner
    remains in the parent's foreground session across ANALYZE and fresh agents.
    If the host cannot preserve that lifecycle, pause and report the concrete
    limitation; no detached replacement is available.
11. After parent analysis, `release --key KEY [--key KEY ...]` releases only
    explicitly designated complete cycle-local states. `shutdown` requires zero
    retained states and zero live workers. No old-owner mutation is reachable.

`core.train_step(model, optimizer, ema, x, y, config, context)` uses flat loss,
ledger, diagnostic and timing fields. The runner supplies the exact original
row-ID digest and `diagnose=true` only on the first batch of epochs 1/10/25/40/50.
Batch logs preserve each actual-B denominator; epoch ledgers sum batch facts.
Snapshots retain current model/EMA/Adam, RNG/loader, selected states and cost in
RAM only. Absolute cycle-local epoch paths go to commit; RAM_COMMITTED must be
acknowledged before advancing. Completion uses frozen selected-state compaction.
No tensor/model/optimizer/pickle/checkpoint output or spill is written to disk.

Before and after runtime writes, whole old TV/NDP journal hashes and complete
sorted result path/hash inventories must remain equal. The old twenty-control
index also stays byte-identical: old TV export still sees 50+20 inputs, while
this exporter sees 30+20. Historical source, journals, tables, results, receipts,
guardian PID 1025015, retired driver and shared state are never mutated.

Implementation and CPU checks do not satisfy independent review, GPU smoke,
actual tool-session lifecycle verification or the full formal matrix. These
remain gates for the parent and fresh runner.
