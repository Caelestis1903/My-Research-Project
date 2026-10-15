# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v3

## 1. Verdict

**FAIL. qualification_RUN_precheck=False.** Previous terminal-oracle and interrupted-PREPARED nonce defects are corrected statically. Recovery coverage remains blocked by three MAJOR boundary-classification defects in the new injection implementation. No matrix was executed, so this report gives static counterexamples and predicted failure modes, not observed pytest failures. Return findings to coder and review v4 separately.

No isolated Q3 qualification RUN, production change, claim/release, staged graph GO, scientific RUN, training or GPU action is authorized. B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Exact v3 identity | Test SHA 163aa615984eb729d7428d53aa88b6973c5327252acec7b615a294abeea796ae; implementation report SHA c695913193e9a435a38ba5579753c6dc146fba89c3d2d9195bb33131d3fa124d. Ordinary frozen copies are archived externally; mutable successors excluded. |
| Only harness changes/baselines | PASS in approved baseline03 exp/m3oe .py/.sh inventory: only test file changes. Runtime remains 0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2. No mathematical, tensor, routing, weighting, gradient or scientific algorithm changes. |
| Previous R1 terminal classification | CLOSED STATIC: :2586–2590 requires exact v1.5 schema, COMMITTED state, true launch flag and current ledger SHA; :2608–2609 asserts expected publication-boundary terminality. Old v1.4 source flag alone no longer selects final archive/admission success. |
| Previous R2 concrete coverage | NOT CLOSED: AST counts 42 unique PREPARED-or-later specs (30 forward+12 recovery), plus eight preimage cases. Twelve recovery entries now name intended existing archive/fence paths, but actual predicates are incorrect (three new findings below). Three forward dirs also regress. |
| Existing object/path callsites | Intended archive seed hooks leave release/pending/final archives visible; mutable seeds leave canonical pending/pointer/released hold before its next event. Retry file classifier recognizes pending canonical, pointer and hold. Yet six recovery file points cannot match, and three archive parent points fault during seed. Three mutable parent retry cases appear statically reachable; no execution PASS is claimed. |
| Previous R3 immediate foreign nonce | CLOSED STATIC: prepared_ledger label now attempts valid nonce-derived alternate identity before original resume (:2188–2221), snapshots eight graph/log paths and query count. It runs after test-only later-hold injection/restoration but before original recovery; durable PREPARED remains unchanged. Pre-PREPARED archive cases correctly have no recorded nonce to reject. |
| Visible-state/admission oracles | Ledger exact prefix, PREPARED/source/audit bytes and preimage digests retained; expected canonical phase, pointer and hold projections extended to recovery entries. Incomplete gate rejects; exact terminal graph may validate. Oracles cannot execute for the misclassified fault cases. |
| Nonce/later-hold/recovery | Original matrix and shared new case tail preserve foreign nonce rejection, later active hold, original-identity recovery, exactly one ordered event per release phase, unchanged immutable PREPARED/audit/source and terminal canonical/archive equality. These are authored assertions, not run evidence. |
| Rename survival branches | Existing final rename-visible COMMITTED_IDEMPOTENT branch and simulated lost-rename pending restore remain unchanged. Four real post-rename/pre-parent hooks are distinct from fsync injection. No actual reboot/power-loss result is inferred. |
| Terminal all bytes/no telemetry | Six graph artifacts plus release/observation ledger bytes and query counter are compared; terminal later hold rejects without graph/log/query mutation, restored exact released clone projection ACK stays byte exact. Unexecuted. |
| Isolation/COW/process | Ledger COW/detached worker logs/locks and per-case/base finally cleanup unchanged. Reviewer used sequential project Python static audits with external scheduler, no reviewed-code imports or tests. Future RUN must use an independent ordinary-copy clone and unique basetemp. |
| Coder checks | Coder reports compile PASS and 2/62 collect-only, 60 deselected, 0.13s. No bound raw collection output/manifest independently verified here. Matrices explicitly NOT EXECUTED. Reviewer AST parse PASS independently. |

### Boundary inventory

The 30 forward fsync specs remain: 6 immutable archive create fsyncs; 10 post-PREPARED ledger fsyncs; 4 observation fsyncs; 8 atomic temp/rename-parent fsyncs; 2 terminal gate-fence fsyncs. Eight preimage specs add 6 preimage archive fsyncs and 2 PREPARED append fsyncs. Twelve new retry specs intend six existing immutable archive fsyncs (runtime :2398/:2400 × three objects) and six mutable-path fence fsyncs (runtime :704/:707 at callers :2423/:2437/:2460).

Actual static defects affect three forward archive :dir cases, all six recovery :file cases, and three recovery archive :dir seed cases. These are twelve distinct authored cases; the remaining three mutable recovery :dir cases are not declared failing without execution evidence. All exact specs, duplicate keys, computed effective values and comparison outcomes are in the evidence JSON.

## 3. Bugs/deviations found

### Q3-V3-R1 — MAJOR: duplicate directory keys overwrite forward mappings

At :2502–2519, release_archive, pending_archive and final_archive each occur twice. Python dictionary construction retains the later value: recovery:release_archive:dir, recovery:pending_archive:dir, recovery:final_archive:dir. Thus expected_dir_boundary can never equal the requested corresponding forward name release_archive:dir, pending_archive:dir or final_archive:dir at :2532–2535. The intended fsync exception is not raised; pytest.raises would fail if the rest of the valid graph completes normally.

This regresses the forward parent fsync coverage required by DESIGN §10:162. Required correction: maintain mode-aware, unique mappings for both forward and existing-object boundaries; verify no duplicate keys and each intended forward/retry name resolves separately.

### Q3-V3-R2 — MAJOR: six recovery file names cannot match

At :2497 file_boundary is file_key + ":file". At :2498 it compares directly with active_boundary. Every requested recovery file string is recovery:<object>:file, while classifiers return only <object>. Even after :2548–2549 sets active_boundary to the requested recovery name, equality is false for release_archive, pending_archive, pending_gate, release_pointer, hold and final_archive. The six intended retry file fsync faults are unreachable.

Required correction: compare one normalized boundary identity with explicit forward/recovery mode and exact callsite/path; retain separate forward predicates. A match assertion after a call does not compensate for an impossible selector.

### Q3-V3-R3 — MAJOR: archive directory retry fault fires during seed creation

At :2538–2547 the selected fsync hook is installed before the named seed commit. Before state receives active_boundary, :2491 defaults to boundary, which is already recovery:<archive>:dir. With the effective dictionary from R1, initial durable_bytes_create parent fsync matches the retry label for release/pending/final archive. It raises Q3 injected after parent fsync while creating the artifact, before the expected named RELEASE_ARTIFACT_DURABLE/PENDING_GATE_RENAME/FINAL_GATE_ARCHIVE seed hook.

The enclosing seed pytest.raises expects "fault injection at", so that premature exception is incompatible. More importantly, it never exercises the existing-artifact branch; a matching filename and fsync alone is not a retry-callsite proof. This remains defective even if merely widening the accepted exception message.

Required correction: disable selected injection during seed, enable it explicitly only after the expected named seed hook, distinguish create/existing mode and record/assert seed completion plus second-call fault order and identity. This is a harness correction, not a research redesign.

## 4. Corrections made

None to reviewed implementation/test/report, production or previous attempts. Created external ordinary frozen copies, AST/static expression audit, source diff and this new report/evidence. No runtime/harness module was imported or executed; no pytest, long matrix, worker, release transaction, torch/model/sample/checkpoint/GPU work occurred.

## 5. Remaining risks

The static audit directly reads literal dictionary/spec values and evaluates ordinary string equality; it does not simulate syscall durability or claim execution coverage. It proves selector contradictions for exact v3 source while preserving the distinction from observed test output. Coder basic checks cannot catch these runtime paths. Corrected successor still needs independent code review and then bounded isolated qualification RUN with raw evidence, exact source hashes and protection/slot checks. Source-only hash comparison cannot reconstruct B6 pre-edit inventory or prove all transient filesystem/lock activity absent.

## 6. Whether experiments are safe to run

**NO-GO for the isolated Q3 matrix at this checkpoint.** Correct all three selectors/seed issues and review the successor first. Even a later implementation PASS permits only a separately scoped isolated qualification RUN, not production release or scientific work. B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain.

### Evidence bindings

- `/tmp/f0-v15-Q3-review-v3-20261014-a/test_stage_F0_successor_v15.py` — SHA `163aa615984eb729d7428d53aa88b6973c5327252acec7b615a294abeea796ae`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v3.md` — SHA `c695913193e9a435a38ba5579753c6dc146fba89c3d2d9195bb33131d3fa124d`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/audit.py` — SHA `1a465a138fdbf205a4bbef50568b1c1c2b3a29f4e1f1b83642c91affa3670514`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/audit.stdout.txt` — SHA `e61592f496a7a07f15d4cfda7dc113c2107458da345b8de359c91689c250b41d`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/static_audit.json` — SHA `f55127361344d049d620e59a7a70b279155505b2ea196a3467bcf11b957dbbab`.
- `/tmp/f0-v15-Q3-review-v3-20261014-a/source-vs-v2.diff` — SHA `fe297ef06c096c5fadd4561a1523dfd1cdab1d29026fc405be7e11f440ba7071`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v3_evidence.json` — SHA `89d4b60dee152cbb1588d6211c60e7a961bc5c633d43e6f4928b2db61c8cf77b`.
