# REVIEW REPORT: F0 v1.5 implementation v4

1. **Verdict: FAIL. RELEASE and RUN remain NO-GO.**

Independent reviewer `/root/f0_v15_implementation_review` inspected approved DESIGN v2 (`8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`), current source/tests, coder diagnostic implementation v4 (`2de80e4cb8cf0cf7038bd5d231d58d92e971a51f066b165476369c30e80670b9`) and v3 C1–C3. Coder explicitly calls implementation incomplete and does not claim qualification PASS. B6 disposition (`e87797f710eaa0aa75f4666f07349bd2e5050af9098b4cc28cf60366c66276b5`) remains OPEN/FAIL. This review does not authorize staged GO, principal release or experimental training.

2. **Requirement-by-requirement verification**

| Requirement / prior finding | Independent verification | Result |
|---|---|---|
| C1 actual worker/child identities, DESIGN §5.3 B2 | successor `1562–1787` now joins worker lock/control supervisor PID/PGID/start-ticks/worker nonce to CHILD_STARTED and live parent; canonical/status child identity remains distinct. GPU bootstrap schema accepted with later richer fields checked when present | Prior schema mismatch corrected by code and independent fixture; real changed-peer E2E still unqualified |
| C1 host binding | worker/control/GPU host bound to nonempty invocation host; optional status host must agree (`1652–1702`) | Missing-status-host positive and host mismatch negative fixture pass; v4 correctly says missing host was not established as prior rejection cause |
| C2 complete observation log, §7.3 B3 | valid stage, reason vs GO, integer/fraction/time consistency and one exact current graph verified (`1893–2007`) | Prior three counterexamples corrected; independent negatives/current-graph fixture pass |
| C3 actual changed-peer deferral and both boundaries, §§8/10 | harness now signals pre-spawn snapshot, starts later peer, records changed-member evidence and predicate trace | Attempt failed before peer claim; no auth predicate invocation. Cannot close C3 or full C1 qualification |
| C3 fault/long-reentry/current regressions | earlier matrix/stale-input source retained; v4 records compile, one focused test and failed peer diagnostic only | Full required qualification and current relevant regression evidence still incomplete |
| B6 preservation, §§7.1/10 | raw incident retained, hold active, frozen maps/current F140 absent; before/after diagnostic files identical | OPEN/FAIL; prospective COW does not reconstruct missing historical preimages |
| Scientific baseline/model/mathematics | control-only changes; frozen queue/search identity preserved; runner/model/objective/data/metric hashes unchanged from prior bound sources | No new formula/tensor/mask/weight/gradient algorithm; missing original runner diff remains B6 limitation |

Current hashes independently match coder v4: successor `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`; successor test `ea670cb5b3c21c38a8fa80fdd41877d61a4a61e30e1cce1102f5e5d919b27e0b`. All current control/runtime source inventories and changes relative to the v3 bound hashes are recorded in evidence JSON.

3. **Bugs/deviations and remaining blockers**

**D1 — MAJOR, C3 qualification remains blocking.** DESIGN §§8/10 require both spawned-child deferral races, actual supervisor orderly exit/no-respawn, independently authenticated progressing peer, controlled long PREPARED reentry, all specified fault/restart opportunities and current regressions. The single v4 attempt is reported as `1 failed in 332.42s` and current archived phase evidence supports no peer claim/no predicate execution, not a successful race. Trace contains only `{"event":"hook_installed","worker_id":"0"}`. Worker0 current nonce `0a434c47c3fe491398258d1b74d0177b` has only supervisor start/spawn, child guard and child_preclaim_entry; no serialized_preclaim_memory_check, canonical_claim_before or training_before. Worker1 current nonce `c66e05b87a1f4dfe865f380fdf95d5fb` has supervisor start, child spawn and cleanup signal, with no canonical claim. Most copied peer log/phase files are historical; their old training phases must not be attributed to this attempt. A corrected predicate plus a small fixture cannot substitute for required successful changed-member process qualification. Return to implementation/qualification without training or production release. Previous C3 controlled >60-second/fine syscall boundary/current regression limitations remain open.

**D2 — MAJOR evidence inconsistency: diagnostic lock attribution exceeds the archive.** Coder v4 says child PID 2192979 had `wchan=locks_lock_inode_wait` and calls the attempt a reproducible global-lock wait. Independently checked archived bytes instead contain:

| PID | Archived wchan |
|---|---|
| 2191116 (test process) | `hrtimer_nanosleep` |
| 2192463 (worker0 supervisor) | `0` |
| 2192978 (worker1 supervisor) | `0` |
| 2192979 (worker0 child) | `hrtimer_nanosleep` |

The archived `/proc/locks` row **does** establish a FLOCK owner: `2192463 08:01:54138857`. Supervisor fd targets name the clone `execution_state.lock`. This establishes ownership at that snapshot, not permanent blocking or a deadlock cycle. Archived syscall/stack files contain permission-error text; they provide no usable kernel stack/syscall proof. The actual test wrapper (`test_stage_F0_successor_v15.py:1772–1787`) waits with `time.sleep(0.02)` for peer readiness **before** calling real runner main. That wait is consistent with the child sleep and entry-only phases. It is not proof that this wait alone caused the overall failure.

Static control flow is separately identifiable: supervisor `_spawn_one` polls `assert_no_global_hold` while a child runs (`run_search_claim_worker.py:1044`), whose v1.5 path validates the terminal graph under execution-state serialization; source/inventory validation can execute while that lock is held. The claim path also uses that state lock. These are possible contention sites. The snapshot does not establish which site caused the failed orchestration, starvation, nested-lock deadlock or a permanent block. **Root cause remains undetermined.** Correct the report's exact wchan statement and keep lock ownership, wrapper waiting, and causal hypotheses distinct. Do not treat missing-status-host handling as this failure's demonstrated cause either.

**D3 — MAJOR, B6 continues to block release/RUN.** Incident disposition explicitly remains OPEN/FAIL: original protected pre-edit inventory/runner bytes unavailable, historical production append/flock incident retained. V4 identical post-incident before/after maps confirm bounded current containment; they cannot authenticate a missing earlier baseline. Preserve active hold, nine production appended CHILD_STARTED rows and current raw evidence. No code/staged-GO or test success closes B6 without its separate authorized disposition.

**M1 — MINOR archive manifest self-entry inconsistency.** Actual aggregate manifest SHA `75849b0b59e51ba72b97e76040e159a827d558f0136f8d49c79f409579dae144` agrees with report. Its 172 rows include itself with the empty-file digest `e3b0…b855`; that row does not match the completed manifest. Independent verification found **171 payload entries matching and only that self-entry mismatching**. Pre-kill JSON manifest has **38/38 matching payload rows** and hash `5ab2c0ce0468f17d3dcfb25c32dedf7b2cf047ce66933895fe7e6427564b5ffa`. Avoid self-hash cycles in future manifests and identify this exception explicitly; do not rewrite the existing archived evidence. This finding alone does not imply payload corruption.

4. **Corrections made**

No implementation, test or production artifact changes. The reviewer created an independent copy clone and external archive for bounded checks and adds this review/evidence only. Raw diagnostic files/manifests remain unchanged. No diagnostic retry, larger E2E or training was run.

5. **Independent checks and evidence limitations**

Bounded original evidence root: `/tmp/f0-v15-v4-authtrace-20261014-a`. Independent archive: `/tmp/f0-v15-v4-review-20261014-b/archive`. Its artifact audit records every aggregate/prekill manifest entry, current production map, raw wchan values and copied original manifest/before/after files outside pytest cleanup trees.

The clone contains ordinary copied source files and frozen queue/source identity inputs; no shared production lock/log inode is used. Focused command:

`setsid bash /home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v15-v4-review-clone /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q --basetemp=/tmp/f0-v15-v4-review-20261014-b/basetemp /tmp/f0-v15-v4-review-20261014-b/clone/experiments/tests/test_stage_F0_successor_v15.py -k 'authenticated_live_peer_claim_is_accepted_during_deferral or runtime_observation_full_schema or observation_history_binds'`

Independent raw output: **11 passed, 42 deselected in 0.10s**. Output retained externally as `archive/focused-test-output.txt`. This includes a controlled live sleeper child with canonical/GPU flocks, supervisor/control identities, optional-host positives/negatives, observation decision/graph negatives. At most two project Python processes; all fixture state stayed in the clone/unique basetemp. This is unit/focused validation, not full two-worker changed-peer E2E.

Diagnostic before/after raw JSON files both hash to `cde34a0fb684066ff74105b447d1153f059284631771b85fb7c5123d7a6c4025` and are byte-identical. Their eleven protected-file hashes all match current production. They report F140 count 140/state `{}`; current independent frozen identity check also finds no materialized F140 status/lock/run paths. Production worker_1.log was correctly excluded from the saved before/after map; its current hash and archived evidence copy both equal `b349368dd3330234032eba7456f8daeb83e73757b798a385ff59431f2c1651c4`, establishing present equality only. Production audited_worker0 remains `b27ac612…e3eb7c2`, worker1 `1725b85a…022f960`; hold/gate/release/queue/remediation hashes remain frozen. Hold is active and successor production release/observation ledgers absent.

The exact diagnostic pytest failure count/duration and coder py_compile result are report claims; no retained raw pytest stdout was supplied in the bounded archive. Phase/trace/process artifacts independently verify that this attempt never demonstrated predicate acceptance or training. Process/wchan snapshots are instantaneous; unsupported syscall/stack text cannot establish a causal wait cycle. No scientific AUC/loss/domain/curriculum/momentum conclusion follows these checks.

6. **Whether experiments are safe to run**

**NO. FAIL / NO-GO.** C1/C2 source and focused corrections are accepted as progress; D1–D3 remain blockers. Complete faithful qualification and report corrections, preserve B6 OPEN/FAIL and active hold, and do not invoke principal release or advance to RUN.
