# DESIGN SPECIFICATION: F0 v1.5 successor fresh-only execution control

Designer verdict: PASS (design complete). This document authorizes implementation and independent review only. The current failure hold remains active; it does not authorize a worker launch, release, retry, recovery, checkpoint load, formal run, or test-data access.

## 1. Research question

Can a separate, reviewed successor control protocol continue the untouched F0-B1-ML search keys after a pre-claim bootstrap failure while preserving all historical execution evidence and admitting each GPU independently under the user's strict memory threshold?

This is an execution-control qualification, not a new model hypothesis. Scientific experiments, configurations, queue order and analysis requirements remain those already specified for F0. The qualification must demonstrate that launch, supervisor, trajectory bootstrap and canonical claim use the same approval contract and cannot release a later hold through an earlier transaction.

## 2. Hypothesis being tested

The recorded first worker1 invocation failed before a canonical claim because v1.4's production gate validator omitted amendment approval metadata checked by the trajectory runner. Matching runtime/control digests alone were insufficient. A versioned successor with a unified contract and an incident-bound one-use release can correct the control inconsistency without revisiting a previously claimed key. Independently qualified GPU selection should permit GPU1 when GPU0 is busy or unreadable, and vice versa.

Competing explanations remain possible: other bootstrap contracts, a process-slot or GPU identity mismatch, or an incomplete historical protection graph could still block launch. Read-only qualification must reject those conditions; passing the metadata predicate alone is not sufficient evidence of launch readiness.

Frozen incident evidence verified during design:

| Artifact | SHA256 |
| --- | --- |
| Canonical v1.4 execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Current active failure hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Worker1 failure record | `1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4` |
| Worker1 stderr | `2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a` |
| Immutable v1.4 release | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| Committed v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Committed v1.4 H51 migration ledger | `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30` |
| v1.4 source snapshot file | `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` |
| Frozen search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Frozen search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |

Current hold ID: `cd2dbf50-3a50-4b81-b976-38f1b03991a3`. Failure file: `worker_logs/worker_1_failure_1791870436_8fc4a33fc7574a368312b4965e24a283.json`, nonce `8fc4a33fc7574a368312b4965e24a283`, exit 1, `canonical_claim:null`, absent receipt. Preserve the stdout, stderr and phase files and the receipt's absence. The historical v1.4 release is already COMMITTED and cannot be replayed against these new hold bytes. The old gate's `launch_authorized:true` does not make this hold released.

## 3. Variables changed

Only current execution-control validation, version dispatch, successor evidence construction, release transaction, target-device resource qualification and isolated contract tests. Add v1.5 artifacts and source module. Integrate the separately specified narrow amendment remediation and bind its actual implementation/review evidence. Its amendment status and verdict cannot be inserted without independent approval.

## 4. Variables held constant

Frozen scientific identity `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`; queue rows, canonical keys, data/splits, model/objective/optimizer/loss, widths, configurations, seeds, epochs, evaluation and selection rules. Use the existing F0 queue seeds, not repository default seeds to replace frozen identities. Search phase only. Initial partition is exactly disjoint C97/H51/F140, whose union is the 288 fixed search keys. Six resource-preflight runs remain outside that partition.

Set digests (canonical compact sorted-list hashing already used by v1.4): C97 `06cba20a42bfc3f8dfc20b4df68ac7dc774c66bb34972ab4158a7227db4d7fce`; H51 `e9077aefa1ed48a3237b769b20c0cd9e8ca3f9b0054ee537de1b7a40cd870a86`; F140 `78e529b19e6fb9cc55cdc32524f3243005c0bf88f5dbe4206c44062a645f585b`. Freeze actual sorted lists as well as counts/hashes. H51 statuses stay RECOVERY_BLOCKED; C97 status/run assets and historical incident trees remain byte-identical. No production status or run-tree changes occur during preparation or release.

Forbidden flags are exactly `resume:false`, `retry:false`, `new_attempt:false`, `checkpoint_load:false`, `formal:false`, `test:false`. Initial `attempt-00` creation for a genuinely never-claimed key remains permitted; `new_attempt:false` means no replacement attempt on a previously touched key. No attempt-01, checkpoint loading, key rehashing, deletion of lock/status history or automatic failed-worker restart.

## 5. Mathematical and control definition

### 5.1 Current approval and historical protection

New gate schema: `F0-EXECUTION-ISOLATION-v1.5`; source snapshot schema: `F0-SUCCESSOR-SOURCE-v1.5`; prelaunch schema: `F0-PRELAUNCH-AUDIT-v1.5`; immutable release schema: `F0-GLOBAL-HOLD-RELEASE-v1.5`; ledger event schema: `F0-SUCCESSOR-RELEASE-LEDGER-v1.5`. Scope must equal `F0-B1-ML search only; fresh UNCLAIMED canonical keys only`.

For amendment A, frozen scientific hash S, complete actual runtime hash R and actual control inventory digest K, require A to be a mapping, `status==APPROVED`, `reviewer_verdict` in `{PASS,PASS WITH FIXES}`, `frozen_scientific_code_hash==S`, `runtime_control_code_hash==R`, and `control_hash_inventory_sha256==K`. R remains the complete `stage_F0_queue.code_hash(require_tests=True)` digest domain; K is a separate control inventory digest. Compute and log both actual inventories. Production cannot silently omit missing inventory files with `require_tests=False`.

Current review JSON must contain independent reviewer identity, verdict, design/report paths and file hashes, exact changed-source hashes, runtime inventory/hash and control inventory/hash, and confirmation of zero unresolved blockers and scientific preservation. Approval record binds those review bytes and the current incident/source snapshot. Review/report files are excluded from their own inventory digest to avoid a hash cycle; bind their hashes explicitly in the gate instead. Retain old v1.4 inventory semantics; introduce a distinct v1.5 control inventory function with a deterministic unique file list covering new module, dispatch code, launchers, child/bootstrap code, current relevant tests and this design.

Historical v1.4 migration/release/snapshot/reviews/prelaunch artifacts are immutable historical evidence. Validate their frozen raw bytes and internal transaction/identity consistency, and compare protected C97/H51 assets with the approved old inventories. Do not call v1.4 current-release validation expecting its old control hash or old released hold to match new runtime. Do not amend its frozen review/report/source snapshot to force compatibility. In particular, the migration COMMITTED hold hash is historical, while v1.5 current source hold is the distinct failure hash above. Bind both with distinct field names. Check v1.4 release event continuity plus the appended incident failure event in the current hold. Preserve all old events exactly, not only a selected projection.

### 5.2 Fresh eligibility

For a queue key k, fresh eligibility is `k in frozen F140 AND no status path AND no canonical lock path AND no run/attempt tree path AND no earlier durable claim history`. Use `lexists` so symlinks/broken symlinks are history, not absence. Explicit UNCLAIMED status files also count as history. Verify full canonical JSON and full key SHA against frozen rows. Check history before creating a canonical lock; recheck after acquiring the per-key lock. An owner file left by a crash before status publication makes the key ineligible permanently under this authorization.

Before release all F140 must satisfy this predicate and no active claims/workers exist. After release allow legitimate monotonic progress only inside frozen F140; preserve original C97 and H51. Existing F140 statuses/attempts are never claimable again, even after a new child nonce or a subsequent metadata fix. Any failed/incomplete F140 claim activates the global hold and prevents new claims; this protocol does not release that later hold. Existing receipt validation and peer-stop behavior remain strict. A pre-claim failure with null claim adds no canonical identity to H51/F140 and grants no retry authority.

### 5.3 Per-device memory admission

For physical GPU i with UUID u, total bytes T and used bytes U, define `eligible(i) = readable(i) AND valid_identity(i,u) AND integer(T,U) AND T>0 AND 0<=U<=T AND 10*U<9*T`. Reject booleans masquerading as integers. Floating fractions are logged and checked against U/T, finite and consistent; the exact integer inequality decides the boundary. Neither GPU utilization nor other applications/process names enter admission. External GPU processes are observed, never killed or treated as exclusive use.

Query each GPU independently using an explicit physical index and UUID. Missing/unreadable/malformed telemetry on a non-target GPU is recorded with reason and `eligible:false`; it does not veto the other target. Duplicate/ambiguous identity for an intended target fails that target. Release requires nonempty target subset of `{0,1}`, every chosen target eligible, and unique physical UUIDs. It does not require two rows eligible or two worker processes. Initial target subset equals the eligible devices in the reviewed prelaunch observation; if both qualify, both workers may run subject to slots. A worker ID stays equal to its physical GPU index; CUDA-visible local remapping must not change this physical identity.

Gate and prelaunch bind exact target index/UUID pairs, readable telemetry, U/T and UTC observation time. Live observation maximum age is 60 seconds at release, supervisor pre-child check, child pre-claim check and claim; refresh via target-specific query under the existing serialization. Use live memory again immediately before durable claim, not only a saved audit. Release checks currently chosen targets: a newly ineligible device is deferred and cannot launch; if target subset must shrink, generate a newly reviewed immutable prelaunch/candidate before starting the release transaction. A non-target becoming eligible does not implicitly authorize its worker; prepare a separately reviewed admission extension or successor transaction before adding it.

After release supervisor/claim qualify only the assigned target. A transient target-memory NO-GO before any claim defers that device with an administrative event and no training-child spawn; it does not write a canonical failure or stop an otherwise eligible peer. Gate corruption, UUID mismatch, bootstrap exception, abnormal child exit or incomplete claimed trajectory retains fail-closed global hold behavior. Do not stop a running legitimate trajectory solely because training allocations later exceed 90%; the rule is admission for launching/claiming subsequent work.

### 5.4 Concurrency

One GPU training child per physical device with its existing exclusive GPU lock. Each persistent supervisor reserves two existing process slots (supervisor + child) before starting Python, so one eligible GPU uses at most two worker Python processes, both use at most four. The global task Python maximum remains five, including tests/audits/operator helpers and child processes. Reuse the shell slot reservation mechanism and inherited-FD ownership checks; do not bypass it to parallelize. Parallel independent qualification tests may use up to five slots. Fewer eligible GPUs never requires fake idle workers. No additional training multiprocessing subprocesses may escape this accounting.

## 6. Exact affected modules/files

Create `experiments/stage_F0_successor_v15.py` (standard library only) and `experiments/tests/test_stage_F0_successor_v15.py`. Add explicit v1.5 production dispatch in `experiments/stage_F0_worker_control.py` (`validate_execution_gate`, `_hold_is_released`, `assert_no_global_hold`), `experiments/stage_F0_queue.py` (production claim and inventory bindings), `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` (version-aware target pre-child check and resource deferral), and `experiments/stage_F0_outputs/runs/run_search_trajectory_child.py` if its current gate assertion otherwise binds v1.4. In `experiments/stage_F0_run.py`, only bootstrap/control amendment or GPU-admission glue may change; its scientific loops must stay byte-identical outside exact reviewed hunks. Review `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` for incompatible current-version dispatch; patch control dispatch only if necessary.

Existing shell launcher `run_search_worker.sh` accepts one worker ID already. Change it only if required to preserve per-target checks or expose audited deferral exit status; retain shell reservation and no hidden Python helper. Extend existing launcher/control/queue/run tests and WORKER_CONTROL.md as necessary. No model, objective, protocol, data/splits or scientific metric module changes. Do not change v1.4 release/migration semantics; the separate narrow metadata fix remains governed by its existing design.

## 7. Implementation requirements and transaction

### 7.1 Preparation under active hold

Coder must first freeze a raw inventory of current historical gate/release pointer/hold, all v1.4 artifacts, new incident logs plus receipt absence, C97/H51 statuses/trees and F140 absence. Retain exact bytes of mutable historical pointer files in content-addressed immutable ordinary files. Include file modes and path-kind/absence checks; symlinks cannot replace gate/hold/release source artifacts. Verify incident hashes above before mutation; mismatch is NO-GO. Preparation/report generation never overwrites the canonical gate, hold or old release pointer. Source edits are allowed only under DESIGN → IMPLEMENT → REVIEW.

Provide side-effect-free APIs for source/prelaunch/candidate construction and validation. Do not create absent F140 canonical lock files when probing lock availability; those paths prove freshness. Probe existing locks only. If a receipt path previously absent appears, fail. Freeze the current post-migration H51 status hashes and original C97 inventory, compare to approved v1.4 graph, and snapshot raw queue/search gate hashes. New paths use `postbootstrap_v15_source_snapshot.json`, `prelaunch_audit_v15_fresh_search_<nonce>.json`, `bootstrap_amendment_v15.json`, and `execution_isolation_gate_v15_pending_<nonce>.json`; all immutable O_EXCL, file fsync and parent-directory fsync. Distinct versioned no-go reports preserve unsuccessful observations; never edit a failed immutable audit into GO.

Independent reviewer inspects actual source diff, implementation report and isolated test evidence, then validates staged source/prelaunch/gate graph with current incident hold still active. Reviewer writes structured implementation PASS and staged-gate GO records. Candidate gate binds both actual files and hashes. Since the code inventory cannot include the final review bytes, review records bind source hashes and approved candidate core, then final gate binds review hashes. The reviewer rechecks this final graph. All bootstrap contracts must pass a non-training read-only readiness check using staged gate validation, with release authorization explicitly withheld. Do not patch/stub production validators for that check.

### 7.2 New release transaction

Only root/principal invokes release after all blockers fixed and staged GO. Persistent user continuation authority suffices; record that authority explicitly with latest per-device rule, `approved_by_role:principal`, named identity, intent `fresh-only F0-B1-ML search`, new nonce (>=16 characters), UTC approval, forbidden flags false, exact failure nonce/hold SHA, candidate/source/reviews/amendment/prelaunch hashes and chosen targets. Never reuse v1.4 identity or nonce.

Use new immutable release `execution_control/global_hold_release_v15.json`, immutable final authorized gate `execution_isolation_gate_v15_released_<nonce>.json`, and append-only ledger `execution_control/postbootstrap_v15_release.jsonl`. At most one logical release transaction may exist in this ledger. It is not resettable to handle another incident.

All transaction operations hold `execution_state.lock`; acquire free worker/GPU locks in the existing deterministic order, then existing protected-key locks if needed, never absent F140 locks. The function is synchronous and must release locks in reverse order. Revalidate live source/hold/incident/protected inventory/F140 absence/target telemetry/slot counts while locked. Any new worker, hold event or source drift rejects release. Exact serialization of payloads is deterministic UTF-8 JSON, sorted keys, no NaN, fixed newline; derive exact released hold bytes without deleting or altering old events. Append `V15_PRINCIPAL_FRESH_ONLY_RELEASE` with successor ID, source incident hold/failure SHA, historical v1.4 release SHA/ledger SHA, frozen set hashes and target pairs. Set active false and an explicit new release timestamp; preserve old release timestamp as historical metadata rather than falsely reusing it for v1.5.

Required ordered durable boundaries:

1. Archive exact mutable gate/release-pointer/active-hold preimages by content hash, then append and fsync PREPARED. It binds principal identity, all evidence, exact pending gate hash, exact intended release and released-hold hashes and target pairs.
2. Create/fsync immutable v1.5 release artifact; append RELEASE_ARTIFACT_DURABLE.
3. Publish/fsync canonical pending v1.5 gate with `launch_authorized:false`; append PENDING_GATE_DURABLE. Old gate bytes remain archived. Update/fsync current release pointer to exact v1.5 release; append RELEASE_POINTER_DURABLE. Hold remains active. This pointer replacement is explicitly permitted administrative change.
4. Revalidate the exact staged graph, active source hold and live resource checks, atomically replace/fsync hold with exact predetermined inactive projection; append HOLD_RELEASED only after bytes and parent are durable. The still-pending gate prevents claims.
5. Append/fsync COMMITTED once. Its fields bind release, released hold and pending gate hashes. The final gate's ledger hash may bind this now-final ledger; avoid embedding a gate hash back into the same hashed ledger (no cycle).
6. Create/fsync immutable authorized final gate with `launch_authorized:true`, exact release/hold hashes and COMMITTED ledger SHA; atomically publish/fsync it at canonical `execution_isolation_gate.json` as the last authorization action. Validate full release graph before returning success. No ledger append follows this final publication; a separate immutable publication receipt can record final gate SHA.

Claim/supervisor/child admission requires BOTH final authorized v1.5 gate + exact complete COMMITTED ledger AND inactive hold bytes equal released projection AND exact release pointer/artifact + current actual control/review graph. An inactive hold alone, old authorized gate alone, PREPARED/partial ledger or pending v1.5 gate cannot launch.

### 7.3 Crash, re-entry and later failure

Every boundary supports fault injection. Re-entry with precisely the PREPARED identity may finish missing administrative steps only when observed mutable paths are exact expected source/pending/released projections, evidence unchanged, all F140 untouched and no live workers. Record every observed expected phase; never blindly overwrite an unrecognized path. If hold was already written but HOLD_RELEASED append failed, compare exact predetermined released bytes before appending. If ledger committed but final gate absent, reconstruct exact final bytes from recorded identity; publish only after the same fail-closed checks. Before hold deactivation any newly stale target audit requires NO-GO/new reviewed evidence before a transaction is PREPARED; once PREPARED do not change its targets/evidence in place.

If final gate already published, same-identity call is a READ-ONLY idempotent acknowledgement of the exact final graph; allow legitimate F140 progress under that committed release, do not repeat pending requirements, rewrite pointers, append events, or reactivate/deactivate holds. A different nonce always rejects. Any later active hold, later failure event or unexpected inactive hold projection rejects every replay, including same identity after COMMITTED. No partial-transaction recovery may erase or release a new incident hold. Conflicting/truncated ledger fails closed; do not truncate an append-only ledger automatically. If torn trailing bytes occur, preserve entire raw file and report NO-GO for separately reviewed repair, not discard them.

Production dispatch reads schema once and explicitly chooses the matching current validator; unknown schema rejects, no fallback to legacy fixtures. v1.4 paths remain capable of failing on the current active hold; v1.5 becomes current only via the transaction above. Isolated tests may inject roots/query providers; production paths cannot use fixture bypass to avoid review or hold checks. Helpers do not spawn Python internally or import training/model modules.

## 8. Ablations and qualification cases

No scientific ablation or training is required for this control change. Required isolated discriminative cases are:

* Amendment hash-only legacy payload rejected consistently by gate/supervisor/runner; corrected metadata passes all; each metadata/hash mismatch and missing inventory file rejects.
* GPU0 >=90%, GPU1 <90% admits only worker1; mirror admits only worker0; both <90% admits both; both >=90% admits neither; exactly 90% rejects; other-card unreadable does not veto readable target; target unreadable/UUID drift rejects; utilization 100% with memory <90% still admits.
* Non-target becomes eligible after snapshot does not silently add its worker; live target becomes ineligible after prelaunch defers/rejects new admission without touching a key or stopping eligible peer. Busy process slots/duplicate worker/GPU lock reject additional launch.
* F140 with absent status but existing lock, attempt tree, broken symlink, hidden durable claim or explicit UNCLAIMED status is ineligible; concurrent duplicate claim can succeed at most once; failed claim never becomes eligible after process death or new child nonce.
* Old v1.4 release or new-incident hold replay, wrong source incident/nonce, historical artifact mutation, C97/H51 change, queue identity mutation or unauthorized flags reject.
* Isolated E2E readiness uses actual amendment validator, production dispatch and release/claim contracts with fake telemetry and stubbed training AFTER claim only. It must reproduce the former pre-claim rejection, then demonstrate approved v1.5 contract/claim success without running data/training or creating real production statuses.

## 9. Expected logging requirements

Implementation report: exact allowed diff, protected inventory before/after, runtime vs control digest domains, readiness/test commands and outputs, Python environment, unresolved issues. Use `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` or specified conda environment; reserve task slots for every Python check. Runner logs raw observations only: prelaunch all discovered device rows/reasons, eligible and selected target pairs, disk_free_bytes (>0 integer), process/slot counts, no-residual evidence, locks, active claim count, immutable artifact SHAs, transaction phase/time, admission/defer reason, claim identity/nonce and complete receipt/failure. No AUC/loss conclusion follows a control test.

## 10. Acceptance criteria for independent reviewer

PASS only after inspecting this specification, actual implementation and coder report and correcting every blocker. Verify historical bytes preserved, current failure/no-receipt graph bound, exact C97/H51/F140 evidence, all current entry points dispatch coherently, complete approval predicate, one-target GPU policy end-to-end, five-slot maximum, strict fresh claim semantics and separate release history. Review staged final graph separately from code PASS before principal release.

Require fault injection after every file rename/fsync and ledger append boundary, including inactive hold before its event, COMMITTED before final gate and final gate before return. Every interrupted state must reject launch until exact re-entry completes; same identity completes once, different identity rejects. New failure hold injected at each recovery opportunity must remain active and reject replay. Mutated pending/final pointer, release pointer, partial ledger, source inventory, target identity and asset absence cases all fail closed. Verify ledger events are not optimistic (HOLD_RELEASED follows durable hold), no circular digest, no production mutations from tests/readiness, and old historical validation does not compare new runtime to old inventory. Broaden to relevant existing control/launcher/queue/run/migration regression suites; scientific metrics need no rerun for this change.

## 11. Conditions that falsify the hypothesis

The proposed control correction fails qualification if any production layer still rejects the approved amendment despite matching evidence, allows hash-only metadata, requires both devices eligible, launches a non-target device, exceeds five Python processes, modifies scientific identities/assets, admits a key with history, launches through a partial release, replays v1.4 authority, or releases a later incident hold. Any such result is FAIL/NO-GO and returns to implementation with observed raw evidence. Passing qualification supports safe control consistency; it provides no evidence about AUC improvement, curriculum/momentum effectiveness, domain size/difficulty or M3OE behavior. Scientific analysis remains deferred until the authorized F0 experimental RUN stage is complete.
