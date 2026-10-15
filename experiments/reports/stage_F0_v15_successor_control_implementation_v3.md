# F0 v1.5 Successor Control — Implementation Report v3

Status: **IMPLEMENT complete for this revision; independent REVIEW required. Release and RUN remain NO-GO.** This report binds the approved v2 design and the current implementation/test bytes. It does not approve production migration, release, worker admission, or training. The previous independent implementation review v2 remains FAIL until a new independent review examines this report and the updated source/tests.

## Specification and immutable evidence

- Approved design v2 SHA-256: `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`.
- Independent design review v2 PASS SHA-256: `9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb`.
- Bootstrap-remediation implementation v2 SHA-256: `491bf69c0bdbc8708e2e9bbd050d3213270e643e460e66369d9a165c464140ea`.
- Independent bootstrap-remediation review PASS WITH FIXES SHA-256: `1a2196eda5a182901bfa646ab2964c8461af4aa0fe587874d70132a1aa3c0065`.
- Immutable amendment record SHA-256: `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138`.
- Immutable no-go candidate SHA-256: `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`.
- Previous independent implementation review v2 SHA-256: `78d5cebbf0bb74e6b71b8bfe2237d4aa0b4dc6fa9ad98bf8885e174aca8af23a` (FAIL; its findings are the implementation work addressed here).
- B6 incident disposition review SHA-256: `e87797f710eaa0aa75f4666f07349bd2e5050af9098b4cc28cf60366c66276b5` (B6 OPEN/FAIL).

## B1–B5 implementation and evidence

| Finding | Current implementation and focused evidence | Status / limits |
|---|---|---|
| B1 — locked live release checks | Release reentry now refreshes stale telemetry; checks the full release lock set including non-target worker/GPU resources; and requires untouched F140, current disk/slots, and resource consistency at the pre-hold boundary. A pre-hold memory NO-GO may leave only the design-authorized non-authorizing pending graph: active hold bytes remain exact, canonical `launch_authorized` stays false, and release ledger has no `HOLD_RELEASED` or `COMMITTED`. Tests `test_release_resource_lock_set_covers_non_target_worker_and_gpu`, `test_release_requires_untouched_f140_and_current_disk_slots`, and `test_release_refresh_no_go_and_pre_hold_resource_drift_stay_held` passed in the targeted run(s) below. | Focused tests pass after correcting one test-only schema assumption. Production API was not called. No actual production resource release was attempted. |
| B2 — first canonical mutation boundary | `stage_F0_queue.py` records `CANONICAL_MUTATION_STARTED` immediately before first canonical lock creation. Its exact lock-creation fault fixture verifies the marker is present at that boundary. | Focused fixture passed. |
| B3 — authenticated peer continuity and receipt | A completed peer must have a real receipt whose bytes/hash/identity/checkpoint manifest pass the existing validator; forged/missing receipts reject. Live peers are joined to canonical/status/worker identity, PID/start ticks, and live canonical, worker, and GPU locks. The new real `supervisor.run()` test holds peer worker 1 after actual canonical claim while worker 0 receives an authenticated serialized-memory deferral. | Forged and real-completion fixtures passed. The new concurrent real-process peer test passed; exact evidence and hashes are below. |
| B4 — GO/NO-GO observation graph | Runtime observations are durably appended with source/prelaunch/PREPARED transaction identity, selected target, UTC/monotonic times, integer memory values, outcome, and reason. Validation rejects inconsistent decision rows. Observation append does not alter the release-ledger COMMITTED digest. | Focused schema/digest tests passed. |
| B5 — failure/reentry falsification | Each of 14 release publication/durability boundaries starts from an independent PREPARED-only clone. Each case checks its immediate canonical/pending/hold/ledger phase, admission rejection before authorized final gate, later foreign-hold rejection without mutation, and same-identity recovery; the final-gate rename case checks the alternate rename-not-survived restart. Other focused cases cover source/target/hold drift, stale telemetry, memory deferral, F140 drift at pre-hold, truncation, and true supervisor lifecycle behavior. The real peer test below asserts orderly exit 75, authenticated receipt, no automatic respawn, no peer stop, and continuing peer heartbeat/progress. | Fault matrix and focused falsification fixtures passed. Controlled >60-second evidence is represented by the real process test duration, but this is not a dedicated clock-controlled stale-telemetry test; stale input/re-query has its own focused test. No production release was run. |

The targeted group was run once before correction and returned exactly `1 failed, 16 passed in 546.12s (0:09:06)`. The sole failure was test-only: the NO-GO fixture looked for a nonexistent `transaction_state` property on the canonical gate (`KeyError`); the actual canonical gate contract is `launch_authorized`. The implementation had already left that field false and the expected ledger phases intact. The test now checks the real schema: `launch_authorized == false` plus the exact ledger sequence and absence of `HOLD_RELEASED`/`COMMITTED`. Only that one test was rerun after the edit; it returned `1 passed in 108.48s (0:01:48)`. The 14-boundary matrix itself passed in the original targeted group. The combined 17 tests were not rerun as one suite after this test-only correction.

## Real supervisor peer-continuity E2E

Exact command:

```text
V15_TEST_PEER_CONTINUITY=1 V15_TEST_SINGLE_BOUNDARY=SERIALIZED_PRECLAIM_MEMORY_CHECK V15_TEST_FAULTHANDLER_SECONDS=600 V15_TEST_SUPERVISOR_TIMEOUT_SECONDS=600 experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-live-peer-continuity-single-serialized-600s-identity /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_successor_v15.py::test_real_pending_graph_release_reentry_terminal_and_f140_claim
```

Raw result: `1 passed in 639.18s (0:10:39)`. This was one pytest process plus at most four supervisor/child Python processes (maximum five total), with an isolated scheduler root under `/tmp/pytest-of-xzf/pytest-277/test_real_pending_graph_releas0/v15-supervisor-scheduler`. It used fake fixture telemetry and an empty sentinel dataset; it did not use CUDA execution or train.

The worker 1 peer had a real claim for run `317538ba0e1c6a0bb62a`, full canonical SHA `317538ba0e1c6a0bb62a8db994d7842b72f1531b9dc172634b108bf6d2b7fe1c`, nonce `abe84fa6ae1647feaf80f1ab1407e176`, child PID/start ticks `2121581/37872487`, and supervisor PID/start ticks `2121052/37866694`. The saved clone evidence records simultaneous held canonical, worker, and GPU flocks and matching owner identities. Heartbeat/progress grew from 1 to 2,806 rows while worker 0 deferred. The authenticated deferred result has boundary `SERIALIZED_PRECLAIM_MEMORY_CHECK`, return code 75, `automatic_respawn=false`, `peer_stop=false`, and `hold_unchanged=true`. The resource receipt reports no canonical claim, attempt, or checkpoint, and the phase log has no `training_before` phase. The test passed its no-training, identity, no-respawn, peer-liveness, and progress assertions.

Evidence retained at `/tmp/pytest-of-xzf/pytest-277/test_real_pending_graph_releas0/`:

- `v15-live-peer-continuity-evidence/owner_and_locks.json` SHA-256 `93be81f7661cb6e853be2f464075e92442c0dd983ccc408533133680a59f8254`.
- `v15-live-peer-continuity-evidence/v15-live-peer-heartbeats.jsonl` SHA-256 `22d192d655c149098c29651f555fe7ae160c75343b87ea2c3527227509f9a029` (2,806 rows).
- `v15-live-peer-continuity-evidence/trajectory_worker1_1_abe84fa6ae1647feaf80f1ab1407e176.phases.jsonl` SHA-256 `eea03cbcb0990d38acc8183f76cef5c5e1e9a7f21f873d3c65f36a1714cd8e92`.
- `v15-live-peer-continuity-evidence/audited_worker_1.jsonl` SHA-256 `3a289e78515f1fe1862115604243f1b41bfd2da909d3d099f08e728beb68ba4d`.
- `v15-live-peer-continuity-evidence/v15-live-peer-ready.json` SHA-256 `45bf3a8dd04c98b6e2bdee711f8a50cb7c708e41fd32d1668da3f3aff49cdae1`.
- `v15-boundary-evidence-SERIALIZED_PRECLAIM_MEMORY_CHECK/trajectory_worker0_1_62b0abd4645c4283b836fc99ae593e21.receipt.json` SHA-256 `836b2ee203fec0ee6c4b0f4346bf22e14a7ac5235699d31bc21f97411debf052`.

The full isolated peer clone is removed by the harness after the run; the evidence directory above is the retained subset. No production lock was intentionally acquired by this test. Clone lock and log COW checks precede dispatch. The current production execution-state lock is empty (SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`), inode `54657656`; no pytest/F0 supervisor/trajectory child remained after completion.

## Verification and historical results

All Python commands in this revision used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`; Python tests were run sequentially. The current B1–B5 targeted group and its corrected single-test rerun are recorded above. The new peer-continuity E2E is also a current-revision test. Earlier, before the latest peer-continuity extension and NO-GO fixture assertion edit, the v2 implementation report recorded sequential regression results of `84 passed, 1 warning in 833.22s`, `40 passed in 8.02s`, and `44 passed in 94.91s`; these are historical v2 results, not a full regression rerun against the exact test hash in this report. The standalone no-peer two-boundary E2E in that earlier report was `1 passed in 590.28s`. It remains historical evidence only.

No test in this report analyzes scientific results. No training function was invoked. The current test source hash is bound below; the earlier test-suite counts are not represented as verification of changes made after those runs.

## Production protection and B6 disclosure

Read-only checks after the new peer E2E and corrected single-test rerun found the production hold still `active=true`; no pytest, F0 supervisor, trajectory child, or worker process remained. The following before/after values are compared to the previously recorded current snapshot and independently checked again after these clone-only tests:

| Protected production artifact | Before/current baseline SHA-256 | After SHA-256 |
|---|---|---|
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` | same |
| Canonical v1.4 execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` | same |
| v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` | same |
| v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` | same |
| v1.4 migration ledger | `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30` | same |
| Frozen search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` | same |
| Search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` | same |
| Immutable remediation no-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` | same |
| Production `audited_worker_0.jsonl` | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` | same |
| Production `execution_state.lock` | empty file SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | same |

The baseline worker-log/lock hashes are post-incident observations, not recovered pre-incident identities. B6 remains **OPEN/FAIL**: nine earlier test `CHILD_STARTED` rows remain appended at production worker-log lines 130–138; the pre-edit full queue/status/run/checkpoint inventory and the original runner diff were never retained. Those preimages cannot be inferred or restored. The v2 report's last full current inventory was 288 queue entries, 140 F140 identities, 51 H51 identities, 155 status paths, 115 canonical-lock paths, 109 existing queue-key run roots, and 535 checkpoint files; its inventory digests are historical observations and were not recomputed for this report. The clone-only tests in this revision did not write production queue/status/run/checkpoint files, but this statement does not repair the missing historical baseline or close B6. The separate incident review also records prior production flock access through a hardlinked clone; its status is not changed by current COW.

No production migration, release, hold change, gate update, queue/status/run/checkpoint write, production worker launch, or training occurred. The new release remains inactive/unpublished. Preserve the active hold, prior review FAIL, all incident rows, and B6 OPEN/FAIL pending independent review and a separately authorized disposition. This report does not advance the work beyond IMPLEMENT.

## Current implementation and test hashes

| File | SHA-256 |
|---|---|
| `experiments/stage_F0_successor_v15.py` | `963be09e1793c6663195ed10c40a7e37ef7e981e3c9d96fbc5d6a65b04bd26a9` |
| `experiments/stage_F0_worker_control.py` | `69e332244fbac4fb7d957cc6a5b398733ee88e679015ec42263b656e88961717` |
| `experiments/stage_F0_queue.py` | `19a684f7d81aedaa449b4af84ee51eae1f2603607404d3526c9f67f56ba5b21e` |
| `experiments/stage_F0_run.py` | `4c94a52d54212bd872efa5499b1a7d55eacaecb4e5d2957c88f4d95ca2837377` |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/stage_F0_migration.py` | `9e4c903277505a0e513f3e5f673ce3e263371722bf807a4a0cd7c4b2e4d103a0` |
| `experiments/stage_F0_outputs/runs/run_search_claim_worker.py` | `e44114d1d9f6a12635e4c0d0d42ed55d32b272bf775c6e42d966af839c40ac90` |
| `experiments/stage_F0_outputs/runs/run_search_trajectory_child.py` | `0808eee556cd73f056caaea719ad6d61d5c735dce43b1fc12933f82c060e1a6e` |
| `experiments/stage_F0_outputs/runs/run_search_worker.sh` | `ea43ee4d78059bd7baf191602b6643364eb510025e50bdf657405479b8568dd4` |
| `experiments/stage_F0_outputs/runs/run_f0_python_task.sh` | `d8a317ee314c4eae0b2b10d5b8e2e9cfce345ae5164b6104a0452c02dfe34e17` |
| `experiments/stage_F0_outputs/preflight/run_first_search_epoch.py` | `a914327267cb650f8011e56f0164268fcee131aa1c361b5a145e2c53431d63e3` |
| `experiments/tests/test_stage_F0_successor_v15.py` | `54b5b3e5575e8eb8adc87aa93b9f452b2e4f2ecdd3a69adc9953307577b5c97b` |
| `experiments/tests/test_stage_F0_worker_control.py` | `2cee640070dffe9deb209d46939ad1db7e3035dc7f9f5e17b3cc86b0c9505e50` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |
| `experiments/tests/test_stage_F0_launcher.py` | `b2761475f4b6c98476964cab8fea6918a188c2bd255234a7cd2a8e86cf54a824` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `079cbae1940a1c37671e323f85fec3c25babc5b80ccabcb470fb1a2e97c3b5e6` |
| `experiments/tests/test_stage_F0_migration.py` | `4aadbc98ee01680e228a05d1c550c4b28c3f4c5748ce717fcdddd342114149d4` |
| `experiments/tests/test_stage_F0_queue.py` | `c297862e1e8a35d07668b675b8eaf9ddde1baecc0b0b8a1b46088d292e288b36` |
| `experiments/tests/conftest.py` | `bc1ab7562c07a4f68711db525138deb313fa9e6da93caca6d7cd616a86631bb7` |

This report is the only new report file for this revision. No scientific conclusion is drawn.
