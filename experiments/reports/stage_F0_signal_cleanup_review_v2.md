# F0 execution-control v1.2 independent review

Date: 2026-10-12 (Asia/Shanghai). Scope: actual v1.2 source/tests, design §§12–13, implementation report and pending gate. No project Python, signals, training or source edits were performed. The sole reviewer write is this report.

## 1. Verdict

**FAIL.** Previous MAJOR-01–03 are closed in the submitted implementation. One remaining **MAJOR-04** violates design §13.4/13.5: full 47-key hold preservation is present in the current artifact but not enforced by the gate/claim/release control logic or mandatory rejection tests. Do not launch search, recovery, formal or test evaluation. Keep the active incident hold and pending authorization until this blocker is corrected and independently re-reviewed.

## 2. Frozen snapshot identities

Actual file SHA-256 checks matched the submitted identities:

| Artifact | SHA-256 |
|---|---|
| Design including v1.2 §13 | `85b4ff62e0128b8c22e521f0c6a377f35839e348e249be2603e47623cd0dfece` |
| Implementation report v1.2 | `b565ba9e5da0af12b9fe7d52683a2ad627c97035af970ffa86b5af77c75d6888` |
| Pending gate | `6ae64153b9c9103de99437fdeadb9d4c5eb24da3cefc0b43549583c70aa847d8` |
| Active global hold, unchanged | `de33c8228d0ba7f0988bf01cd3bf374cbef7290dd30654109286fbd80d4d603e` |
| Prior search gate, unchanged | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Actual 14-file control aggregate | `73bfe9a6a5bf06fb1b1f933b9c211dd24b37ec94ac5a65b44b0eef6195bc3c2c` |
| Actual 28-file runtime aggregate | `e53e566f3849d1c6ae8a6d16b7647ab26a89221f3ce057c8abc47978036e5edf` |
| Sorted compact JSON of actual 47 held IDs | `ccbf64945396b542ee2746f4bab37d7d2fcdaea49b00817f096facec45f7f8bf` |

The reviewer independently rebuilt both inventories with Perl JSON/Digest, hashing actual file bytes without importing project Python. All 14 individual control hashes and both aggregates matched. The actual 47-item gate list is unique and exactly equals the unchanged historical 45-item `prior_incident_release.preserved_held_run_ids` plus `b3a55740aadd0e3377a9` and `cfbf6cc0e541e76cfb92`; no missing or substituted item. The prior search gate independently contains the same 45 prior keys. Current artifact correctness does not close MAJOR-04's runtime validation gap.

## 3. Requirement verification

| Requirement | Result and actual evidence |
|---|---|
| Operator/peer INT/TERM/HUP pidfd identity binding, §13.1 | **PASS; closes MAJOR-01.** `run_search_worker.sh:29–34` executes a pre-reserved Python helper. `run_search_claim_worker.py:190–300` reads recorded identity, opens pidfd before live validation, checks pidfd `/proc/self/fdinfo` identity, verifies host/worker/PID=PGID/startticks/nonce and rechecks pinned target before pidfd send. There is no positive-PID or group TERM fallback. Unavailable/expired descriptor fails closed. `test_stage_F0_launcher.py:157–239` exercises actual wrapper stale identity and successful pidfd delivery; unavailable pidfd and target exit at final send are fault-injected around real subprocess/descriptors. |
| Canonical/GPU locks enclose interrupted status/manifest/receipt, §13.2 | **PASS; closes MAJOR-02.** `stage_F0_run.py:1543–1674` snapshots both held owners/process identities and validates canonical/attempt/process fields, then holds execution-state lock while projecting status/manifest and committing receipt. `_write_json:198–216` and `_write_invocation_receipt:239–256` flush/fsync temp file, replace atomically, fsync directory. Caller `:2004–2039` completes this call before `finally` releases GPU/canonical. Last committed checkpoint rollback remains the interruption boundary; no partial epoch or new recovery allowance is introduced. |
| Child outer exception cannot fabricate successful receipt | **PASS.** `run_search_trajectory_child.py:119–139,249–301` only reads/binds an already-committed receipt after runner invocation; failed runner cleanup propagates failure and cannot produce outer success/fallback receipt. Supervisor validates immutable interruption identity, manifest/checkpoint hash, and frozen owner snapshots rather than a subsequently reused owner file. |
| Real pre/post receipt barriers and repeats, §13.3 | **PASS by source/test inspection; closes MAJOR-03.** `test_stage_F0_worker_control.py:306–329,660–729` uses hooks inside the actual runner commit, actual writer and resource locks. At barrier A receipt is absent and both flocks exclude independent pytest; actual TERM/INT/HUP are delivered after manifest-window repeats. At barrier B receipt exists after writer fsync, both locks still exclude pytest, and status/manifest/receipt identity/epoch/hash/startticks/frozen owners agree. First TERM reason remains and six repeats are recorded. After unwind, actual canonical/GPU/worker/slot reacquisition checks occur (`:742–796`); one child, one production forward, unchanged deadline and no continuation are asserted. |
| Receipt commit failure, §13.3 | **PASS with evidence limit.** Writer-entry injected OSError fixture (`:332–342,815–885`) produces no receipt/commit/post-barrier marker, preserves checkpoint, verifies held supervisor failure and no next child/validated interrupt, and probes lock release on unwind. Production outer child also cannot fabricate success. The CPU fixture wraps runner directly rather than executing the complete production child `main`, so outer behavior is verified by source. This fixture injects before atomic writer entry, not separately at every fsync/rename syscall; those failure paths are source-inspected. |
| First-stop latch, unique forwarder, 30-second durable-before-group-KILL | **PASS by preserved source/tests.** Shared supervisor latch, child-forward marker, duplicate no-raise handler and deadline are preserved. Existing timeout fixture plus production default=30 assertion remain. No CUDA/native timeout guarantee is inferred. |
| Group STOP/CONT, parent-death defense, total Python≤5 and GPU≤2 | **PASS by source preservation.** STOP/CONT still negative-PGID; child guard/parent-race defenses precede runner import. Workers reserve two slots before Python, operator reserves one before Python, inherited child copies do not unlock supervisor lease. No new training worker or process-count bypass is added. |
| Full held-key gate/release rejection invariant and tests, §13.4 | **FAIL: MAJOR-04 below.** Current array is correct; validation and mandatory negative tests are incomplete. |
| Pending gate and hold, §13.5 | **PASS: closed.** Actual reviewer verdict is PENDING; launch, formal/test and recovery authorizations are false; amendment remains pending. Active production hold SHA unchanged. Gate rejection occurs before child/queue/dataset work; current state cannot run or recover. |

## 4. Remaining blocking deviation

**MAJOR-04 — complete 47-key hold/release invariant is not enforced (§13.4/13.5).**

Actual `stage_F0_worker_control.py:157–177` checks only list/string shape, duplicates, old search holds as subset and `caller == gate`. It does not independently require the two new incident keys or equality to the full frozen baseline-45-plus-two set. `validate_execution_gate:120–125` similarly checks only old search subset. `_hold_is_released:192–209` requires only the three historical `PRESERVED_INCIDENT_KEYS` defined at `:44–48`, rather than the complete current frozen set. Required missing-new-key, 47-item replacement, and old-45-only rejection fixtures are absent from the changed launcher/control tests.

Source-level reproduction (logical walkthrough, **not executed**, no production gate/hold mutation):

1. Start from submitted gate/search mappings. Replace new key `b3a55740aadd0e3377a9` in the gate list with any fresh unrelated string; supply the identical changed list as caller. It still has 47 unique strings and includes all 45 old search keys. Every check in `frozen_held_run_ids` passes, contrary to required refusal of 47-item substitution.
2. Alternatively set gate and caller to the 45-item prior list. Old-search subset, uniqueness and caller equality all pass. Both new incident keys disappear without refusal.
3. Correct gate=47 and caller missing a new key is rejected by caller equality; this alone is insufficient because cases 1/2 are accepted.
4. For a prospective reviewed inactive-hold release with matching hold ID/hash/principal metadata, a release `preserved_held_run_ids` containing only the old three constants satisfies `_hold_is_released`. It does not bind the full 47-item protection set. The present active hold and pending gate block execution; this is a required future-release validation defect, not evidence of a current launch bypass.

Required correction: enforce the independently frozen full baseline-plus-two identity set before accepting gate/caller, and enforce that same complete set in global-hold release consumption. Preserve historical 45-key release records as historical records, without treating them as authorization for the new hold. Add all §13.4 negative cases, correct positive 47-case, release omission/substitution/duplicate/drift coverage, and evidence that refusal precedes child start/claim. Refresh control/runtime inventories, report and pending gate, retain active hold, and return for independent re-review. Do not change research algorithms, checkpoint values, selection, or automatic recovery policy.

## 5. Corrections, test evidence and remaining risks

No source or control-state correction was made by reviewer. Previous review's three MAJOR blockers are closed; no CRITICAL issue identified in this targeted snapshot.

Coder reports project-env, reservation-wrapper commands: receipt failure **1 passed**; duplicate-signal/operator-pidfd/exit fixtures **3 passed**; affected worker-control, launcher and runner files **50 passed**, plus syntax checks. Reviewer inspected the actual tests and command topology but did not rerun Python or independently reproduce outputs. This is CPU/Linux control evidence, not CUDA/native fault validation. Reported passing counts omit the mandatory full-47 negative tests and therefore do not change FAIL.

The historical native crash cause remains UNKNOWN. Full previous experimental artifact byte invariance cannot be proven without a prior complete snapshot. Only this control repair is reviewed; no scientific conclusions or actual test-set evaluation were performed. Subsequent coder changes supersede this hash-specific snapshot and require a new verdict before authorization.
