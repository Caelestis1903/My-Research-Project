# REVIEW REPORT — F0 execution isolation v1

Date: 2026-10-12 12:46 UTC. Independent reviewer: `/root/f0_ml_reviewer`.

## 1. Verdict: FAIL

The reviewed handoff has blocking implementation defects. Research execution must remain held. This verdict applies to the hashes below, before the coder's subsequent repairs; it is not a verdict on any later working-tree changes.

Scope: read the approved design, coder report, actual supervisor/wrapper/child/control/queue/runner changes, control documentation, and focused tests. No Python, tests, training, signals, checkpoint loading, model evaluation, or scientific result analysis was performed. Only this report was added; no source, gate, manifest, or historical result was changed.

## 2. Inspected identities

| Artifact | SHA256 |
|---|---|
| Design | `5dc5e0ed15f651e4a23cd9a3c762995092d88140cbe4edf8eb3dd714aac562b6` |
| Coder implementation report | `a06513274ddae8a9f1e56ab5565044d21ed8f8854464a0ca1cf6b6ed3574ec6e` |
| Supervisor | `29fb2fb2857dc0dd9f44c9e164d30c081e868f49aa05b0c629f83fb9953fbddb` |
| Shell wrapper | `ddf1642be31891e0636619cbded15c052a0ab62d2d6f5a80a952f5072cec9823` |
| Fresh child entry | `d673a246177696ddadae9b598975e2678a9b12518ac154768187cd4a7c7c077d` |
| Standard-library control module | `181ff47c458b0bcdbebe69b3a4a16af1ad4225c820e262ca6afc052350d9d5d4` |
| Queue module | `89ff06e95b3cdc421950ffcd854b415d85c3cc9e25d60180a05363954219bd02` |
| Runner module | `4a19afe4e5726ea0ce18f43abcb86699a9019204c783d38472e671f65c803fa5` |
| Control documentation | `062f75ee99ae7dc1ed7b7a269236232b6a2537df77b5a1e83dae07c68e3842d8` |
| Pending execution gate | `b6e6133af48663765615097a3ec215b656dfefb5aada6579a0809b472d55fda2` |
| Active global hold | `4711885a0766b6e772eea82a6e41c5805e7fcbfdae70e77d638c0304dc6cbbaf` |
| Preserved old search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |

Independently recomputed canonical JSON inventory digests using Perl SHA256/JSON, without Python:

- 12-file execution-control inventory: `fe8be0b6a9afab6bbc08c269bec9aa97590f3543c05883b59b12ac408aa033cf`.
- Current 28-file runtime `code_hash_inventory`: `f9cacb47fd7a6398e1e84209a92c78755c00ffac4b7f71117004091ff987afd1`.
- Frozen original scientific identity: `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`.

## 3. Requirement-by-requirement verification

| Designer requirement | Independent result and limit |
|---|---|
| Fresh child per trajectory; supervisor does not import training/native modules | PASS in source: each `_spawn_one` uses `Popen`; child calls runner once. Standard-library control module remains independent of torch. Fixture checks distinct PIDs. |
| Wrapper `exec`; supervisor PID=PGID; same child PGID; group STOP/CONT | PASS in source. Wrapper group command checks PID/PGID/start ticks; PID-only STOP is rejected/documented unsupported. Generic real-group fixture covers STOP/CONT; this is not a live training demonstration. |
| Parent death before training imports, including installation race | PASS in source: child installs `PR_SET_PDEATHSIG(SIGKILL)`, checks PPID, parent start ticks and both PGIDs before runner import. No `preexec_fn` or detached session. Actual child-entry fixtures exercise guarded death and installation window. |
| INT/TERM/HUP, spawn window, 30-second escalation | PARTIAL: handlers, signal masking around spawn, forwarding and waiting exist. Escalation violates durable failure requirements: MAJOR-02. Current tests cover before-start and handler-ready execution, not the actual masked spawn window or unresponsive timeout. |
| Pre-reserved supervisor+child slots; child must not unlock parent lease | Worker production path PASS in source: wrapper reserves two existing-namespace locks before Python; child receives only its FD; adopted lease closes duplicate without `LOCK_UN`; supervisor retains both reservations. Test/monitor paths remain incomplete: MAJOR-08. |
| Worker IDs ≤2; original canonical/GPU locks | PASS in source for production launcher: choices 0/1, worker-ID flock, existing GPU and canonical namespaces retained. No live contention test was run by reviewer. |
| Shared hold serialized with new claims; peer stops on failure | Writer/claim both use the same execution-state lock. Normal child failure writes durable hold and group-TERM request. Timeout path fails; active-owner validation can race concurrent owner rewrites: MAJOR-02/05. |
| No automatic held/stale/abnormal recovery | Known incomplete states and held IDs are skipped by default, but unrecognized existing states can be claimed/resumed: MAJOR-03. Protected interrupted key remains in initial active hold; no release is present. |
| Identity-bound completion and checkpoint transaction verification | Completion validates child/parent/PGID/nonce, frozen queue row, canonical owner, status/manifest and checkpoint hashes. GPU owner validation reads a released mutable owner: MAJOR-01. Empty receipt lacks equivalent binding: MAJOR-04. No native checkpoint was loaded in this review. |
| Unique native stderr, phase events, failure audit | PASS for ordinary outcomes: OS-level file redirection; nonce-specific files; fsync events and audit/hold. Escalation must write failure before killing supervisor. Core reporting is bounded to candidate paths and available limits; absence of a candidate file is not proof no core was produced. |
| New review-bound gate; fail closed; preserve old scientific identity | Pending gate currently blocks launch correctly. Approved-path inventory comparison is mismatched and full old gate hash is unused: MAJOR-06/07. No launch authorization/release exists. |
| Scientific formulas, shapes, masks, gradients and baselines preserved | No algorithm redesign found in control additions. Compared 28 source hashes recorded by historical completed trajectory `749e8617426c9912a040`: only queue, runner and queue-test hashes differ; protocol/data/models/objectives and all reused scientific dependencies match. Changed runner needs fixes above. Full historical artifact immutability cannot be proven without a previously frozen complete artifact inventory. |
| Focused evidence 29+18 | Coder reports 47 passed; independently read tests but did not rerun under the read-only/no-Python instruction. Missing regression coverage listed below means that number does not close the blockers. |

## 4. Bugs/deviations and exact corrections required

All eight issues below are MAJOR and blocking; no CRITICAL mathematical defect was identified.

1. **MAJOR-01 — completed GPU owner race.** Supervisor `_validate_completion_receipt`, lines 343–355, reads `gpuN.lock` after child exit. `F0Runner.run`, lines 1723–1727, already releases GPU lock before runner main writes receipt. A peer can legally reuse that GPU and overwrite the owner, falsely rejecting a successful trajectory and holding both workers. Freeze immutable identity/owner evidence while the lock is held, then validate that evidence; do not validate historical completion against current mutable GPU owner. Test peer reuse before validation.

2. **MAJOR-02 — group SIGKILL loses required hold/audit/peer-stop.** Supervisor lines 638–648 writes escalation events then kills its own PGID, including itself. Later failure audit and hold writer are unreachable. With an operator TERM and no existing hold, peer can keep executing. Before group kill, durably write identity-bound failure audit/global hold and request peer stop. Add an unresponsive-child timeout fixture proving those records exist and no peer continues to another claim.

3. **MAJOR-03 — abnormal-state recovery bypass.** Queue lines 301–315 authorization-gate only four named incomplete states. Existing `FAILED`, `UNKNOWN`, missing-state records, etc., fall through to ordinary claim, possibly returning an existing resume checkpoint without authorization. Restrict pristine new claims to absent status/history; every existing non-complete/non-authorized state must fail closed. Cover abnormal/unknown/malformed state and missing-status historical attempt/checkpoint cases.

4. **MAJOR-04 — empty receipt can spoof identity and holds.** Supervisor lines 369–391 checks schema/status/queue hash only, trusting receipt-provided `held_run_ids` to skip arbitrary rows. Bind PID/PPID/PGID/worker/nonce/output root to this invocation; accept the hold set only from frozen gates/caller and require agreement. Test wrong identity/nonce and extra held IDs.

5. **MAJOR-05 — active-owner read/write race and unreadable-status bypass.** Supervisor lines 199–258 reads owner/status without execution-state serialization. `QueueFileLock.write_owner`, lines 80–86, truncates and rewrites owner JSON; runner rewrites active canonical/GPU owners at startup (lines 1512–1535). Peer can read partial owner JSON and falsely trigger hold. Serialize both readers and all relevant owner/state writers using the shared state lock, or provide immutable/atomic owner records without replacing the actual flock inode. Unreadable queue status at supervisor lines 247–250 is silently skipped and must reject before child/data/GPU startup. Add real concurrent rewrite/read and corrupt status fixtures.

6. **MAJOR-06 — amendment compares different hash inventories.** Gate `runtime_control_code_hash` equals the 12-file control digest `fe8be0…`; runner lines 1965–1974 compares it to the 28-file runtime code digest `f9cacb…`. The approved path would still fail. Freeze the correct current 28-file digest for runtime amendment separately from the 12-file control inventory, refresh all reviewed hashes, and test distinct inventories through the actual code-check path.

7. **MAJOR-07 — recorded old gate hash is not enforced.** Control `validate_execution_gate` never checks `prior_search_gate_sha256`; validation compares only selected fields. Supervisor blocked IDs and runner claim holds are drawn from mutable old search gate, without enforcing new gate's preserved hold inventory. Validate the entire frozen old gate hash and hold coverage; use the frozen hold set. Keep release in new control artifacts and leave old gate unchanged. Test modifying only old blocked keys while retaining scientific code hash.

8. **MAJOR-08 — tests are not consistently reserved before Python startup.** `tests/conftest.py` obtains pytest's lease in `pytest_sessionstart`, after Python exists. Fresh-child test in `test_stage_F0_launcher.py` uses `/dev/null` rather than a reserved child slot before spawning Python. Current held/serial test setting happens to stay below five, but does not implement the design's startup ceiling for all task Python processes. Provide a startup reservation/adoption entry for tests/monitoring, account for every fixture child, and verify the sixth Python is rejected before creation under four worker processes plus one counted test process.

MINOR reporting deviation: coder report says no optimizer run occurred, while runner checkpoint/resume unit fixtures execute CPU optimizer steps. State the precise boundary: no research/CUDA training; synthetic CPU unit fixtures may execute optimizer steps. Also avoid interpreting `core_dumped: false` from limited candidate paths as a definitive OS observation.

## 5. Corrections made and remaining risks

No implementation correction was made by reviewer. Findings were sent to the orchestrator for a bounded coder repair and new independent review. Original abort root cause remains UNKNOWN; native process reuse remains a WEAK competing hypothesis. Passing fresh-process fixtures or future uninterrupted execution would not establish that root cause.

The initial pending gate and active global hold remain intact. No protected recovery, including `59474a37e9d12d956633`, is approved; its original committed checkpoint and three-epoch identity/record audit are separate prerequisites for an explicit principal recovery release. Formal/test gates remain closed.

## 6. Are experiments safe to run?

**No.** Correct all blocking items, supply focused regression evidence, independently re-review the repaired source/report/hash inventories, and freeze that review in the new gate. Only then may the principal explicitly authorize the appropriate incident/global hold release while preserving protected keys. This report does not authorize search, recovery, formal training, or test evaluation.
