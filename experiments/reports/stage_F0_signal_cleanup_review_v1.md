# F0 v1.1 signal cleanup independent review — first snapshot

Date: 2026-10-12 (Asia/Shanghai). Reviewer: independent REVIEW agent.

## 1. Verdict and execution boundary

**FAIL.** Three MAJOR blocking deviations remain in the submitted snapshot. Do not launch search, recovery, formal training, or test evaluation on this approval. Keep the pending gate and production incident hold. This report records the submitted hashes below; concurrent subsequent coder changes require a new report/inventory and targeted review.

## 2. Independently verified identities

Shell SHA-256 and standard-library Perl JSON/Digest checks, without executing project Python, confirmed:

| Artifact | SHA-256 |
|---|---|
| Design §12 | `6e021b6fc7d9ed62ec4fb02b9f200e723325d38df8faab6f1dc12203d1bab55c` |
| Coder implementation report | `647105ea3d7c389d516d0d3d8271db7579f27405f25089c85ea5a00a5141f755` |
| Pending execution gate | `b30f74c4726168f6eb7cdd8b3d724813650e55923a13a387cf5e36f3ff3b67cd` |
| Active production global hold | `de33c8228d0ba7f0988bf01cd3bf374cbef7290dd30654109286fbd80d4d603e` |
| Prior search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Control inventory, 14 actual files | `a605feb7d5fb63f203fe98a95e33ef33a5dbab09b42025c5a4f63c6dd1fcefba` |
| Runtime inventory, 28 actual files | `06d8662645b4a95df79b308cfaec90cc7a74a79f92adc5bef5064ec6139f52a0` |

All 14 file hashes matched the gate individually. The 28-file inventory was independently rebuilt from `stage_F0_queue.IMPLEMENTATION_FILES`; both compact sorted-JSON aggregates matched the gate/report. Frozen scientific canonical code identity remains `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`.

## 3. Requirement verification

- **§12.1 peer TERM: PASS.** `run_search_claim_worker.py:190–265,631–638` verifies host/worker/PID/PGID/startticks/nonce, opens pidfd, rechecks identity, and sends TERM through that descriptor to supervisor only. Unsupported syscall ABI/errors fail closed. Operator entry: FAIL (MAJOR-01).
- **§12.1 unique forwarder: PASS by source.** Supervisor `:641–647,799–845` preserves first signum/source, forwards once, and counts repeats outside its handler. Global hold enters the same latch. Initial forwarding sets the sole deadline; repeats do not reset it. The real fixture checks one forward and an unchanged deadline.
- **§12.2 duplicate child signals: PASS by source and fixture inspection.** `run_search_trajectory_child.py:188–195` first latches and raises once; all subsequent INT/TERM/HUP increment an in-memory counter without handler I/O. The real fixture synchronizes actual duplicate deliveries inside manifest cleanup. Receipt transaction lock boundary: FAIL (MAJOR-02).
- **§12.2 checkpoint/status/receipt identity: PASS for successful path, with lock-boundary limitation.** Runner `stage_F0_run.py:1777–1853` rolls projection back to the last committed resume checkpoint and validates identity/epoch; queue and manifest name the same checkpoint hash. Child `:130–184,290–326` verifies unique status, matching manifest/epoch/checkpoint and emits cleanup-failure receipt on failure. Supervisor `:516–574,1080–1142` validates interruption records, writes failure/hold, exits, and starts no further child. No automatic retry/recovery is introduced.
- **§12.3 escalation: PASS by source, test coverage qualified.** Supervisor `:850–897` persists failure/global hold, verified peer-stop outcome, phase and escalation events before its group KILL. Persistence failure uses child-only termination and retains supervisor for failure handling. Production default is 30 seconds; existing timeout fixture and explicit default assertion are present (`test_stage_F0_worker_control.py:792–828`). Timeout test uses a shorter fixture deadline and a peer-request stub; it does not itself exercise repeat requests during the timeout, while the duplicate fixture checks unchanged deadline without actually timing out.
- **§12.4 fixtures: PARTIAL/FAIL (MAJOR-03).** Actual supervisor, child, CPU checkpoint, pidfd peer TERM, repeated supervisor TERM plus active temporary hold, real child TERM/INT/HUP, checkpoint hash/epoch, and independent lock reacquisition checks exist. The mandatory receipt preparation/commit signal window is not exercised.
- **STOP/CONT, PDEATHSIG and reservations: PASS by source preservation.** Wrapper still uses negative PGID for STOP/CONT; child installs/rechecks parent-death guard before runner import and unblocks inherited signals only after handlers. Wrapper reserves two of five total slots before Python; adopted child copies close without unlocking the supervisor's shared lease. Existing group/PDEATHSIG fixtures remain. No runtime revalidation was performed.
- **§12.5 gate and hold: PASS (closed).** Gate remains `reviewer_verdict=PENDING`, launch/formal/test/recovery authorization false; amendment is PENDING. `stage_F0_worker_control.py:101–111` rejects pending verdict/launch, and `stage_F0_run.py:2103–2113` validates gate/hold before queue load. Production hold is active and its SHA unchanged. All prior search-gate held keys are retained in the 45-item frozen list. The two newest incident keys are separately queue-blocked; before any eventual release, reconcile the complete 47-key protection set rather than treating the old 45-item list as exhaustive.

## 4. Blocking deviations and exact fixes

**MAJOR-01 — operator PID reuse race (§12.1).** `run_search_worker.sh:31–59` reads the control identity and `/proc`, then sends INT/TERM/HUP using shell `kill "$pid"`. There is no pidfd binding; a replacement PID can receive the signal between validation and send. The design explicitly classifies unclosed PID reuse risk as blocking. Fix the normal operator stop path to use a verified, rechecked pidfd-bound supervisor send, with no unsafe PID fallback. Preserve group STOP/CONT and prelaunch Python accounting. Add identity-change/fail-closed coverage for the actual operator path.

**MAJOR-02 — interruption receipt committed after lock release (§12.2/12.5).** `stage_F0_run.py:1854–1858` releases GPU/canonical locks while unwinding `runner.run`; only afterward does outer child `run_search_trajectory_child.py:290–307` reconstruct/hash/write the interruption receipt. Status/manifest alone are committed at unlock, leaving a window with reusable locks but no durable receipt. Move successful interruption receipt generation/commit inside the protected runner transaction before both locks release, retaining explicit cleanup-failure behavior. This does not require numerical/research changes. Independently assert locks cannot be reacquired before receipt commit and can be acquired afterward.

**MAJOR-03 — required receipt-window real signal coverage absent (§12.4).** Fixture source `test_stage_F0_worker_control.py:337–355` constructs and writes the receipt before exposing its second barrier (`receipt_committed`). Real child repeat signals at `:597–608` are all injected while the manifest barrier is held. No repeated signal is injected at receipt preparation/commit, and no pre-receipt lock assertion exists. Add an explicit precommit receipt barrier using the production commit path; deliver actual duplicate TERM and INT/HUP there, prove receipt/status/manifest/checkpoint identity and first TERM exit survive, and check lock boundaries. Keep existing manifest-window coverage and slot accounting.

No CRITICAL issue identified in this scoped review. Source/mathematics/masks/data/selection/baselines were not redesigned or modified.

## 5. Corrections and evidence limits

Reviewer changed no source, tests, gate, hold, manifest, or experimental result. This new report is the sole reviewer write.

The coder reports reservation-wrapped project-env commands: dedicated `test_real_duplicate_signals_preserve_interrupted_checkpoint_and_cleanup` **1 passed in 2.44s**, then `test_stage_F0_worker_control.py` plus `test_stage_F0_run.py` **38 passed in 12.71s**, at most three Python processes, plus syntax checks. Command/test source was inspected; the reviewer did **not** rerun Python, send signals, train, or independently reproduce those outputs. Passing counts do not close the uncovered acceptance windows.

CPU fixtures do not prove CUDA/native cleanup within 30 seconds and do not explain the historical SIGSEGV, whose cause remains UNKNOWN. Prior full historical artifact byte invariance cannot be proven from this snapshot alone. Repair all three blockers, refresh inventories/report/gate, retain hold, and request targeted independent re-review before any run authorization.
