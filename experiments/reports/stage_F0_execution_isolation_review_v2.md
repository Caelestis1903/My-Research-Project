# REVIEW REPORT — F0 execution isolation v1, fixed snapshot v2

Date: 2026-10-12 13:35 UTC. Independent reviewer: `/root/f0_ml_reviewer`.

## 1. Verdict: PASS WITH FIXES

All eight MAJOR blockers from `stage_F0_execution_isolation_review.md` are closed in the inspected source, tests and pending gate. No new blocking process-control, resource-accounting, scientific-identity or automatic-recovery defect was found. Three nonblocking forensic limitations below require precise treatment of future incident evidence; this verdict does not approve recovery or release any execution gate.

This is a new review of the exact snapshot below, not an amendment of the earlier FAIL. The reviewer read actual changed source/tests and independently recomputed identities; no Python, pytest, training, CUDA action, signals, checkpoint loading or scientific result analysis was performed. Only this new report was written. Source, gates, holds and historical artifacts were not modified.

## 2. Exact inspected identities

| Identity | SHA256 |
|---|---|
| Approved design | `5dc5e0ed15f651e4a23cd9a3c762995092d88140cbe4edf8eb3dd714aac562b6` |
| Actual coder report | `cc927249045617dbf17e4430c8b5a8aa77bec4917f4b1d8e35f04a33a97674b8` |
| 14-file execution-control inventory | `3dc11d27af1a6388bf6ff9a70f993b331bbdc829afe6a3bcf44ef3988ba2fd99` |
| 28-file runtime inventory | `6ea31aa85f1c3889df5aaaf98107de6398b2983a0a6e79438981f6c27e39dd7b` |
| Preserved scientific/canonical code identity | `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19` |
| Preserved old search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Pending execution gate | `e9104c9aaf3adccb868123c45911ea2a0ba00f0fa13d3ce7b4f708eaf0c3b92f` |
| Active global hold | `4711885a0766b6e772eea82a6e41c5805e7fcbfdae70e77d638c0304dc6cbbaf` |

All 14 actual control file hashes match the gate's complete `control_hashes` map. The independently computed 28-file digest equals the runtime amendment field; the inventories are distinct. Old search-gate bytes match the recorded complete SHA. Design/report/gate/hold actual hashes match the handoff and coder report. These checks used standard shell and Perl SHA256/canonical JSON, not Python.

The 14 control inputs are the design, supervisor, worker wrapper, fresh child, task wrapper, standard-library control module, queue, runner, control documentation, pytest conftest, and launcher/control/queue/runner tests. The immutable pending gate identified above records every individual path/hash.

## 3. Requirement-by-requirement verification

| Requirement / earlier finding | Actual evidence and result |
|---|---|
| Fresh native lifecycle per trajectory | PASS. Supervisor remains standard-library-only, calls `Popen` for every invocation, and never calls runner in-process. Fresh child imports runner only after guard/slot/gate checks and calls it once. Distinct-PID fixture now reserves its child slot before spawning. |
| PID=PGID, wrapper exec, group STOP/CONT | PASS. Worker wrapper `exec`s supervisor; supervisor owns group; child retains that group; changed PGID is rejected. Wrapper checks process start ticks and sends signals to negative PGID; PID-only STOP is explicitly rejected/unsupported. Group fixture checks both process states. |
| PDEATHSIG installation and race defense | PASS. Unchanged inspected child installs Linux SIGKILL-on-parent-death before training imports and rechecks PPID, parent start ticks, parent PGID and own PGID. Both installed-guard death and installation-window fixtures use the actual child entry. No detached session or `preexec_fn` was added. |
| INT/TERM/HUP and no next child | PASS in source. Spawn masks those signals until child assignment; child installs handlers then unmasks; supervisor forwards its stop marker and prohibits later invocation. Before-start and handler-ready fixtures cover all three signals. Actual arrival during the masked spawn interval is not independently runtime-tested. |
| MAJOR-01: released mutable GPU owner | CLOSED. Runner `_locked_owner_snapshot` and `_write_completion_receipt` (248–326) capture exact locked GPU owner and canonical JSON hash. `F0Runner.run` writes completion receipt before lock release (1753–1756 versus 1784–1788). Supervisor validates frozen owner, never current mutable GPU owner. Receipt regression invokes actual writer, releases GPU flock, reclaims/rewrites it as peer, then validates the previous receipt; also rejects PID/checkpoint corruption. |
| MAJOR-02: timeout durable failure before self-kill | CLOSED. Supervisor 694–727 fsyncs logs, writes failure audit/global hold, requests verified peer group TERM, then records durable escalation before own-group SIGKILL. Persistence-error path kills only child and keeps supervisor alive for ordinary failure handling. Unresponsive-child fixture checks hold/audit/peer-stop request/events before group death. Its peer request is a fixture substitute; it does not demonstrate two live research workers stopping. |
| MAJOR-03: abnormal ordinary-claim bypass | CLOSED. Queue 278–330 rejects unreadable status; ordinary claim requires no prior status, attempt tree or canonical-lock history. All historical non-complete records need explicit identity-bound authorization; allowance remains one. FAILED/UNKNOWN/malformed/missing-status checkpoint regression covers old bypass. Production launcher supplies no recovery authorization. Completed and held records remain skipped. |
| MAJOR-04: spoofed empty receipt | CLOSED. Supervisor 407–446 validates child/parent/PGID/worker/nonce/output root, frozen queue hash and exact duplicate-free gate hold set. Both runner empty outcomes write that set. Regression rejects forged identities, nonce and extra held keys. |
| MAJOR-05: active owner race / corrupt-status bypass | CLOSED. Supervisor scan uses shared execution-state lock (232–282); claim, runner canonical/GPU owner rewrites and active/resource status rewrites use the same inode. No replace of the canonical/GPU flock inode is introduced. Invalid status fails before child start. Threaded fixture pauses with partial JSON under writer lock and checks reader waits; malformed-status fixture checks zero children plus durable hold. State reader does not acquire canonical/GPU locks, avoiding lock-order inversion with claim/runner. |
| MAJOR-06: inventory-domain mismatch | CLOSED. Actual gate runtime field contains the verified 28-file digest. `validate_runtime_control_amendment` compares this domain, separate from 14 control hashes. New unit fixture computes both actual inventories, proves distinctness, accepts runtime and rejects control digest. |
| MAJOR-07: old gate / hold drift | CLOSED. Control validator 114–126 checks entire prior gate SHA and old blocked-set coverage. `frozen_held_run_ids` requires exact duplicate-free caller/new-gate agreement. Supervisor and runner use this set. Old-blocked-list-only mutation regression fails even with unchanged scientific hash. |
| MAJOR-08: Python starts before reservation | CLOSED for prescribed task launch paths. New task shell wrapper reserves before `exec`; pytest conftest adopts inherited FD/inode/owner/role and refuses missing identity before collection. Every inspected fixture Python spawn has prior slot reservation. Four simulated worker leases plus counted pytest prevent marker for sixth Python from being created. Worker wrapper pre-reserves two slots; child closes only its duplicate, without parent-lease `LOCK_UN`; worker reserves both throughout lifetime. All task Python must use these documented launch paths. |
| Worker ≤2, GPU exclusivity, global hold serialization | PASS in source. IDs restricted to 0/1 with per-ID flock; original GPU/canonical lock paths remain. Hold writes and canonical claims share execution-state lock. Monitoring of hold stops running child; normal failure requests peer group termination and never retries. No detached training process or implicit multiprocessing/DataLoader worker was introduced. |
| Completion / scientific checkpoint semantics | PASS within static scope. Completion requires exit0 plus full identity/status/manifest and committed checkpoint file hashes. Prior checkpoint projection/role/state validation remains. Control patch adds receipt and locking, not training mathematics. Future recovery still needs full state/epoch/checkpoint audit. |
| Scientific formulas, shapes, routing, masking, gradients and baselines | PASS for preservation within this control review. Historical completed manifest's 28 source hashes were compared to current files: only queue, runner, conftest, queue tests and runner tests differ. Protocol/data/models/objectives and all reused algorithm dependencies retain original hashes. Reviewed runner additions concern locks, receipts, gate/lease interfaces and administrative events. No algorithm, optimizer update, RNG stream, loss/mask semantics, seed, epoch budget, canonical identity, selection or test release change was found. |
| Pending fail-closed launch and preserved history | PASS. Actual gate review and amendment remain PENDING, launch=false, training_started=false; principal incident release absent. Global hold remains active with unchanged SHA. All 45 original blocked keys and three named protected incident keys are present. Administrative scan found 46 validation-complete, 6 resource-complete and 45 recovery-blocked statuses; no active/unverified status was present. These counts are administrative verification, not scientific result analysis. |

## 4. Test evidence, bugs and corrections

Coder reports **53 passed**, with synthetic CPU optimizer/checkpoint steps and no research/CUDA training. The actual suite structure accounts for 53 focused cases: launcher 9, worker control 16, queue 9, runner 19. Reviewer independently read the fixtures/assertions and inspected source/hash relationships; the pass result itself was not rerun or independently observed under this task's no-Python constraint.

All eight original MAJORs are closed. No CRITICAL issue or new MAJOR blocker was identified. Reviewer made no code correction.

Nonblocking MINOR forensic limitations:

1. Timeout failure audit hashes phase/native logs before group kill; subsequent phase appends and possible child stderr writes mean these are capture-time hashes, not guaranteed final full-file hashes. Record byte-length/prefix and capture semantics or add a separate immutable postmortem digest after group-death verification. Do not reject intact incident evidence merely because later final-file hashes differ.
2. Failure audit records canonical status but does not itself freeze the latest completed resume checkpoint file/hash inventory. Preserve files as implemented, then record and verify the last complete transaction checkpoint SHA/state/epoch in the mandatory incident/recovery audit before approving any recovery. Temporary files and partial epochs remain inadmissible.
3. `_current_claim_for_pid` associates historic status by PID alone. A reused PID can produce incorrect preclaim attribution. Future audit binding should also require this invocation's claim-phase/nonce/start identity; until then, a preclaim failure must be adjudicated from phase/owner identity and retain canonical=null when no current claim is proved. No inference from PID-only historical matches should authorize recovery or alter completed results.

These limitations concern forensic attribution/digest interpretation. They do not authorize retry, affect the ≤5 reservation path, detach child processes, bypass locks, change training mathematics, or release a hold. Core presence is likewise limited to inspected candidate locations, not an OS-wide proof that no core exists.

## 5. Remaining risks and experiment boundary

The historical native abort root cause remains UNKNOWN; reuse was a WEAK competing hypothesis. Process isolation is a mitigation, not a root-cause finding. Fixtures do not prove live CUDA/native cleanup timing, nor every possible signal/OS/storage failure. Timeout group kill requires external confirmation that no group member remains. Full prior artifact byte-for-byte invariance cannot be independently proved without a pre-change complete artifact inventory; old gate/hold and scientific dependency identities were verified instead.

**RUN is not currently authorized.** Process-control implementation is acceptable for search after the orchestrator freezes this passing review and its SHA in the new gate, approves the runtime amendment, verifies current processes/resources, and explicitly records the appropriate principal incident/global hold release while retaining protected keys. Old scientific gate/canonical identities must remain unchanged.

This review does not release protected recovery `59474a37e9d12d956633` or either other incident key, does not approve extra recovery budget, and does not open formal/test gates. Every future abnormal outcome must stop/hold, preserve raw evidence and obtain fresh incident adjudication; forensic limits above must be resolved in that audit before any recovery.
