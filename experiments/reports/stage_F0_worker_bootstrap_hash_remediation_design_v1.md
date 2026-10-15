# DESIGN SPECIFICATION: F0 worker bootstrap amendment remediation v1

Designer verdict: DESIGN COMPLETE; implementation/review may proceed under the active hold. RELEASE NO-GO. This specification does not authorize a second release or a training invocation.

## 1. Research question

This is execution-control remediation, not a scientific experiment. Why did the first worker1 child pass supervisor gate checks but fail before canonical claim, and what minimal correction makes that inconsistency fail closed before launch without changing scientific identities or granting recovery?

## 2. Hypothesis being tested and observed root cause

The failure is inconsistent amendment-schema validation, rather than a changed runtime digest. Read-only inspection established:

* `stage_F0_run.py:2301-2315` requires amendment `status == APPROVED`, `reviewer_verdict` in `PASS`/`PASS WITH FIXES`, the frozen scientific hash, and the actual complete 28-file runtime hash.
* The actual gate's amendment contains only `frozen_scientific_code_hash` and `runtime_control_code_hash`; both approval metadata fields are absent.
* `stage_F0_postabnormal_v14.py:validate_current_gate` checks the two hash values but not those approval metadata fields. Production `stage_F0_worker_control.py:validate_execution_gate` delegates to that validator and returns without the stricter legacy amendment checks.
* Read-only recomputation found actual runtime hash `4b48c8258765459f12b335f7254712b4b39ac0bd428e0b6437686a3439960d98`, exactly matching the gate. Actual 21-file control inventory digest is `fd4733a4eaa967398e50b21f728f1fcaa97cc00c199ee72826349b49f0f255c8`, also matching the gate. These are different digest domains and must remain distinct.
* The existing runner amendment test supplies the approval fields; the v1.4 end-to-end fixture omits them. The latter never established that the runner's stricter bootstrap accepted the reviewed gate.

The worker audit records supervisor PID/PGID `1826142`, child PID `1826436`, invocation `1`, nonce `8fc4a33fc7574a368312b4965e24a283`, exit `1`, `canonical_claim: null`, and no receipt. Stderr identifies the stricter runner check. A null claim supplies no canonical identity to retry; it must never be converted into a recovery authorization.

Frozen candidate evidence from this inspection (recheck before implementation):

| Artifact | SHA256 |
| --- | --- |
| Actual canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Actual active `global_hold.json` | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Worker1 failure record | `1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4` |
| Worker1 stderr | `2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a` |
| Frozen search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Existing v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Existing immutable v1.4 release | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |

The active hold ID is `cd2dbf50-3a50-4b81-b976-38f1b03991a3`. Its appended failure event, original release event, and administrative migration history must all remain preserved.

## 3. Variables changed

Only v1.4 amendment approval validation and focused regression fixtures/tests. New versioned design, implementation, review, validation and staged amendment/gate records may be created. No production pointer or hold transition is part of this implementation.

## 4. Variables held constant

Frozen scientific hash `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`; search queue identity `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23`; H51, C97 and F140 sets; every scientific model/objective/data/protocol/optimizer/seed/selection definition; all queue/status/run-attempt/data/checkpoint bytes; old reports, source snapshot, migration ledger, old immutable release and release ledger; existing failure/log/phase/receipt-absence evidence. `resume`, `retry`, `new_attempt`, `checkpoint_load`, `formal`, `test` remain false everywhere. No automatic restart is permitted.

## 5. Exact control definition

For amendment A, frozen scientific digest S and complete runtime digest R, require:

`A is a Mapping AND A.status == APPROVED AND A.reviewer_verdict in {PASS, PASS WITH FIXES} AND A.frozen_scientific_code_hash == S AND A.runtime_control_code_hash == R`.

The runtime digest is `stage_F0_queue.code_hash(require_tests=True)` over `IMPLEMENTATION_FILES`; it is never the control inventory digest. Retain the existing actual-control inventory comparison separately. The runner's stricter validation remains intact. This predicate is a necessary approval condition and never overrides any provenance, independent review, hold, migration or release condition.

## 6. Exact affected files

Permitted source edit: `experiments/stage_F0_postabnormal_v14.py`, only the amendment predicate in `validate_current_gate`, adding the missing status/verdict checks with a specific fail-closed error. Do not import the runner to perform gate validation; avoid changing scientific imports or introducing dependency cycles.

Permitted regression edits: `experiments/tests/test_stage_F0_postabnormal_v14.py` and `experiments/tests/test_stage_F0_run.py`. Correct existing valid v1.4 amendment fixtures to provide approval fields. Add the contract checks below. `stage_F0_worker_control.py` and `stage_F0_run.py` require no production-source change for this narrow fix; their existing production dispatch and strict runner check are preserved.

Create new report `experiments/reports/stage_F0_worker_bootstrap_hash_remediation_implementation_v1.md`; do not amend the old implementation report, structured review, frozen design or inventories merely to make old provenance pass. A fresh independent review report/JSON and staged records use distinct versioned paths.

## 7. Implementation and review/re-gating procedure

1. Preserve the actual active failure hold. Before source edits, record exact protected file inventories and process/lock observations. No live F0 supervisor/child may exist. Rehash the evidence above; drift means stop and report NO-GO, not overwrite the expected hashes.
2. Coder implements only the source predicate and specified tests. Write a new implementation report with actual changed-file hashes, complete runtime and control inventories, test commands/results, and confirmation that no production artifacts were modified. Tests operate in isolated temporary roots and do not execute training.
3. Independently review the design, actual diff and new implementation report. A PASS/PASS WITH FIXES with all blocking fixes complete is required before constructing any approved amendment record. Reviewer must establish that the scientific implementation is byte-identical and the exact digest changes come only from the approved control/test edits.
4. Under the still-active hold, create a new immutable approval record at `runs/execution_control/bootstrap_amendment_remediation_v1.json`, bound to this design, new coder report, independent review path/SHA, current failure/nonce/hold SHA, frozen scientific digest, actual post-fix runtime hash, complete control hashes/digest, and all forbidden authorizations false. No self-referential hash: reports bind the amended source/inventory and a reviewed amendment core; the approval record then binds their actual file SHAs. This record is administrative evidence, not a release.
5. A runner may prepare a separately named candidate `experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json`. Its amendment includes `status: APPROVED`, the actual independent reviewer verdict, frozen scientific digest, and post-fix runtime digest. It records the amendment approval record path/SHA and intended updated inventory. Set `launch_authorized: false`, `release_eligible: false`, and document that it is a diagnostic candidate awaiting a successor protocol. Do not replace the canonical gate, old review or prelaunch path. An isolated compatibility report must show the amendment predicate passes while current full v1.4 pending/release validation still rejects the reactivated-hold graph.
6. Complete the reviewer checks below and retain RELEASE NO-GO. Root can then commission a separately specified successor control protocol. This narrow correction does not make the current v1.4 graph safely releasable.

Why a second v1.4 pending gate cannot be produced safely: current `validate_current_gate` binds `active_hold_sha256` to the original migration COMMITTED hold SHA, and its pending branch also requires that SHA equal actual current hold bytes. The new failure hold differs. Changing gate hold SHA fails the migration condition; keeping the old value fails the pending condition. Source edits also make the old structured implementation review's exact current inventory stale; replacing that review in the gate conflicts with the migration ledger's frozen reviewer SHA. These are historical provenance constraints, not fields to rewrite.

Why a second `commit_principal_release` call cannot be used: the existing v1.4 release ledger is already COMMITTED. A different request nonce is rejected; idempotent use of the old identity requires the exact old active/released projection and rejects the new failure hold. Do not delete/truncate/archive-and-reinitialize that ledger, replay the old release, restore old hold bytes, mark the current hold inactive manually, or repurpose the old immutable release path.

Required next protocol if continuation is to occur: a separately designed, implemented and independently reviewed version (recommended `F0-EXECUTION-ISOLATION-v1.5`) must distinguish historical v1.4 migration/release evidence from the new current bootstrap incident, bind the new active hold/failure nonce and reviewed control amendment, preserve all old artifacts, use a new versioned release artifact/ledger, validate an explicit fresh-only principal release, and reject old-release replay. No production v1.5 code is authorized by this narrow spec; root must obtain its concrete DESIGN SPECIFICATION first.

Existing user continuation instructions provide intent for development and eventual fresh-only execution. They do not satisfy or bypass the old API's source-graph constraints. Any new principal release must separately record user authority, fresh-only intent, a new request nonce/time, the exact newly reviewed current hold/control/incident graph and forbidden flags false. Root may use persistent user authorization according to session instructions; this design does not require an additional user confirmation as a technical workaround. A reviewed successor protocol and separate release transaction are mandatory.

The successor protocol must also implement the newest user scheduling rule: at least one readable GPU with used/total strictly below 0.90 is sufficient; select only individually eligible GPUs and bind their UUIDs, with one worker per selected GPU and at most five task Python processes. Current v1.4 prelaunch code hard-requires two target GPUs; this narrow metadata patch must not silently loosen that unrelated predicate.

## 8. Required ablations and tests

No scientific ablation or seed run applies. Use discriminative control cases:

* Exact observed two-field amendment: both runner validation and v1.4 gate validation reject it, even when both digests match.
* Remove status alone; remove verdict alone; status PENDING/REJECTED; verdict PENDING/FAIL: reject before dataset loading, claim or artifact creation.
* Approval status with PASS and PASS WITH FIXES: both layers accept the amendment predicate, contingent on the rest of a valid isolated graph.
* Wrong frozen digest, wrong runtime digest, or control-inventory digest substituted for runtime digest: reject. The accepted fields do not bypass hold/release checks.
* Extend the isolated real v1.4 graph fixture to test amendment contract agreement between the real v1.4 validator and real runner helper. Do not merely assert fixture JSON shape or mock the helper under test. Use sentinel dataset/claim functions when probing the runner so no training or production claim occurs.
* After narrow correction, a production read-only check must still reject the active failure hold. In isolated clones, old v1.4 release identity replay and a different nonce must reject the reactivated hold without mutating it or adding release stages. Never exercise a write-capable release API against production for a negative test.

Run targeted amendment/v1.4 checks and applicable worker-control, queue, runner, launcher and migration regressions through `/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest`. Independent groups may run concurrently within the user's maximum five Python task processes, provided temporary scheduler/lock roots do not collide. No GPU training is needed.

## 9. Logging and hash changes

Record exact raw source/evidence SHAs, full post-fix runtime/control inventories, process count and absent live workers/claims, lock results, pre/post protected inventories, actual test results, reviewer identity/verdict, and explicit RELEASE NO-GO. Do not record fabricated scientific AUC/loss or training progress.

Expected changed source SHA: `stage_F0_postabnormal_v14.py`; expected changed test SHAs: the two named tests. The 21-file control inventory/digest changes. The 28-file runtime digest changes if `test_stage_F0_run.py` changes (it is in that inventory); v1.4 module/test alone are not in the 28-file inventory. Calculate actual values after final edits; never precompute or reuse old values. New report/review/approval/candidate hashes are additional identities. The actual canonical gate, active hold, original release artifact/ledger, migration, scientific queue/data/run assets must remain byte-identical during this narrow scope.

## 10. Acceptance criteria for reviewer

PASS only if the exact metadata gap is corrected before supervisor acceptance; runner checks remain equally strict; mismatched inventories and unauthorized metadata fail; all focused/applicable regressions pass; only permitted source/test files change; protected production artifacts retain their actual preimplementation hashes; no training/claim/receipt/run attempt is created; independent approval records are honest and bound to actual final code; current rehold and old-ledger replay remain rejected; and the staged candidate is explicitly nonauthorizing. Historical v1.4 review need not pass against modified code: its expected stale-inventory rejection is preserved and documented. Never change a validator to accept historical stale review evidence as current approval.

## 11. Falsification and stop conditions

The root-cause hypothesis is rejected if current digests do not match the observed gate or a correctly approved matching amendment still fails the runner metadata predicate; investigate fresh evidence without retries. Any accepted incomplete approval, stale hash acceptance, historical release replay, production hold mutation, queue/status/run-asset change, training child, scientific code change, or forbidden authorization falsifies this implementation's control claim. Keep the hold active and return to coder/reviewer. Even a successful narrow correction ends at RELEASE NO-GO until the separately reviewed successor protocol and principal release are complete.
