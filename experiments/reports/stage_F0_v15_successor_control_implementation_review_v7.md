# REVIEW REPORT: F0 v1.5 implementation v7 — qualification precheck

1. **Verdict: PASS, scoped to `QUALIFICATION_PRECHECK_ONLY`.**

Independent reviewer `/root/f0_v15_implementation_review` reviewed approved DESIGN v2 SHA `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`, implementation v7 SHA `5f2365fb730e0b2cb591e58eace9cbff5ba6c365056a3dd538c4c97412946331`, exact test SHA `10c7e1e673de0b8e3b36526701bc1a03bb301fcb6e44ee14bf4b9d37333c2350`, and the v5/v6 marker findings. **`qualified_RUN_precheck=true`: the reviewed code may proceed to a separate bounded, isolated, non-training qualification RUN.** This is not a completed qualification verdict, staged-gate GO, scientific baseline attestation or production release review.

**All full qualification criteria remain unclosed. B6 remains OPEN/FAIL. Production/scientific RUN and principal release remain NO-GO, and active production hold must remain.** This code-precheck PASS does not override those gates.

2. **Requirement-by-requirement verification**

| Contract | Actual implementation/check | Result |
|---|---|---|
| V5 missing later markers | real parent fixture invokes helper for all five records before consuming later resume/child marker | Corrected |
| V6 partial publication | helper polls read/JSON/object/finite numeric timestamp, treating empty/truncated/read/encoding errors as pending | Corrected; actual v6 scheduling counterexample now succeeds |
| Ordered rendezvous | snapshot → peer_wait → peer_claim → supervisor_resume → child_defer strictly increases | Independent positive test and invalid-order probe pass |
| Early producer exit/partial marker | helper records complete/invalid/missing states, parse errors, returncode and exited stdout/stderr before raising | Independent early-exit test passes |
| Bounded timeout | same diagnostic path records alive state, elapsed/deadline and missing/invalid records without blocking for live pipes | Independent timeout probe passes |
| Actual Path.write_text test | coder fixture pauses after real open returns; verifies size-zero visibility; helper retries until writer completes | Independent copied-source test passes |
| Parent coordination outside canonical/state lock | supervisor wrapper waits after real snapshot capture and before its return/child spawn; no child runner-main wait reinstated | Inspected; compile/order test passes |
| Process reporting | peak focused count correctly stated as three for pytest/supervisor/child, within five | Corrected; reviewer checks themselves use at most two |
| C1/C2 runtime joins/observation constraints | runtime/source unchanged at `0b8020e5…427c2`; prior corrected schema and ledger predicates retained | No new runtime algorithm or source drift observed |
| DESIGN §§8/10 final qualification | actual both-boundary changed-peer supervisor E2E, controlled long reentry, required fault/restart/later-hold and current regressions not run here | Pending separate RUN; not implied by PASS |
| B6 preservation | incident disposition and missing historical baseline remain OPEN/FAIL | Production gate still blocked |

No model, objective, mathematical formula, tensor/mask/routing/weighting or gradient path changes are introduced by this test-only checkpoint. Missing original runner bytes/full pre-edit inventory cannot be reconstructed or scientifically attested by this code precheck.

3. **Bugs/deviations found**

No unresolved blocking implementation issue was found for the **bounded isolated qualification precheck scope**. The concrete v5 absent-file and v6 existing-but-incomplete JSON schedules are corrected. Invalid/read/decode cases now enter the bounded poll and unified failure path instead of raising unarchived parse errors. The runtime and scientific source hashes remain those previously reviewed.

B6 OPEN/FAIL and remaining DESIGN §§8/10 acceptance evidence are explicitly retained as separate gates, not relabeled as code bugs merely because the mandated qualification RUN has not yet occurred. A qualification failure must return observed raw evidence to implementation/review; this PASS does not predict its result or authorize retries of scientific canonical keys.

4. **Corrections made**

None to repository implementation/tests/production. Reviewer made an ordinary copy clone, unique absolute basetemp and external archive for bounded CPU checks and adds this review/evidence only. No long real E2E, model training, production migration, gate/hold mutation or release was run.

5. **Independent checks and remaining risks**

Clone root `/tmp/f0-v15-v7-review-20261014-a/clone`; basetemp `/tmp/f0-v15-v7-review-20261014-a/basetemp`; external raw archive `/tmp/f0-v15-v7-review-20261014-a/archive`. Mandatory project task wrapper and project Python were used.

Copied-source focused command selected:

- `test_parent_peer_rendezvous_waits_for_both_later_markers_and_orders_them`
- `test_parent_peer_rendezvous_archives_early_supervisor_exit`
- `test_parent_peer_rendezvous_retries_marker_opened_before_write`
- `test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral`

Raw independent result: **4 passed in 0.75s**. Output retained as `archive/independent-rendezvous.stdout.txt`. These run actual helper positive/partial/early-exit and wrapper compiler/order cases; they are not real two-worker qualification E2E.

Independent `boundary_probes.py`, raw stdout and JSON in that archive, exit 0, confirmed:

- live-producer missing-marker timeout writes `RENDEZVOUS_TIMEOUT` with failure archive/alive state;
- complete but out-of-order records write `RENDEZVOUS_MARKER_ORDER_INVALID` and failure archive;
- the **same actual PathBase.write_text open-before-write scheduling point used in v6** produces an existing size-zero fifth marker. V7 waits about 0.103 seconds for delayed producer release and then succeeds; producer exits 0, valid completed marker retained, no unexpected diagnostic or JSON exception. The runtime/helper was not monkeypatched for this replay.

Coder final raw `8 passed in 0.75s` and preceding `8 passed in 0.83s` are independently hashed against the three-entry manifest. Final stdout SHA `1558b4d8f56605cb75539366aacdb2b0fb3a49bc2fc2d93cef48055fc5408786`; manifest SHA `5e4a64af5a3d7b497f9cf29c05ec49184440be2a02433f00270634e333be3999`. No self-entry/circular hash is used. Manifest binds those two stdout files and retained v6 manifest. This verifies the reported focused outcomes, not unexecuted full acceptance cases.

Required next qualification evidence remains actual both-boundary changed-member peer authentication/continuity and supervisor orderly deferral/no-respawn, controlled >60-second exact-PREPARED reentry with immutable audit/identity/COMMITTED preservation, specified fault/fsync/rename/later-hold opportunities and current relevant regressions. Use clone-only state and fake/sentinel non-training paths, archive raw evidence outside pytest cleanup trees, preserve failure outputs, and observe the global five-Python-process cap. Do not access real research/test data or production mutation APIs during that qualification.

Current frozen gate/hold/queue/remediation and both production audit hashes are unchanged, F140 has no materialized status/lock/run paths, hold remains active and successor production ledgers absent. B6 source-preimage/evidence deficiencies remain unclosed. No efficacy/AUC/loss/domain conclusion follows this review.

6. **Whether experiments are safe to run**

**Separate isolated non-training qualification RUN: code precheck PASS. Production/scientific RUN and principal release: NO-GO.** B6 remains OPEN/FAIL and production hold active. Full qualification, incident disposition and independent staged graph review must be handled before any production admission.
