# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v4

## 1. Verdict

**FAIL. qualification_RUN_precheck=False.** All three v3 selector/seed defects are corrected statically, but a new MAJOR FD-path assertion defect blocks four temporary-file-fsync cases. The fault correctly occurs before rename; the test incorrectly demands its temporary file path equal the destination. No matrix was run: this is a static path/control-flow counterexample, not observed pytest output.

Do not start the isolated Q3 qualification matrix at this checkpoint. Return the specific assertion finding to coder and review v5 separately. B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. No production/scientific action, staged graph GO or release is permitted.

## 2. Requirement-by-requirement verification

| Requirement | Independent result |
|---|---|
| Frozen v4 source/report | Test SHA a78b67b3d6d0e0eef74ce6c0f7bfdd4a6b7708ed4c813fff9754b57d3d274019; coder report SHA 631dce94630afc8f71dd1c326bbefaffd1af3d901c4eb7941eef8fd9622ff774. Ordinary copied snapshots exclude mutable successors. |
| Test-only/baseline scope | Approved baseline03 exp/m3oe .py/.sh comparison identifies only the harness changed. Runtime SHA remains 0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2. No model/algorithm/formula/shape/mask/routing/weight/gradient changes. |
| V3 R1 collision-free directory selection | CLOSED STATIC: 21 rows use (class,phase,boundary), 21 distinct (class,phase) pairs and 21 unique IDs. Three forward archive dirs coexist with their recovery dirs; no dictionary overwrite. |
| V3 R2 recovery-file normalization | CLOSED STATIC: phase=recovery builds recovery:<class>:file, exactly matching all six declared recovery file IDs. Forward IDs retain <class>:file. |
| V3 R3 seed/active order | CLOSED STATIC: seed phase has no directory rows; seed file IDs lack recovery prefix and cannot match requested recovery IDs. All 12 selectors disabled during seed. Exact named seed exception/state and empty matches asserted, then phase changes to recovery, state resets and selected hook runs. |
| Boundary/cardinality/callsite mapping | Independent AST counts 42 unique forward/recovery specs (30+12), eight unique preimage specs and 21 directory rows. Original normal/create/append/replacement/final-fence and new existing-archive/mutable-fence callsites retained. Each injected case asserts one match, class, phase and FD kind. Correct regular path verification FAILS for four temp classes (R1 below). |
| Preimage descriptors | Eight-case loop records expected file-fsync then optional control-dir-fsync sequence; exact archive/ledger path and expected FD kind asserted, distinguishing prepared_ledger from content-addressed preimages. |
| Pre/post visible phase, nonce, later hold | Prior exact ledger prefix, PREPARED/source/audit/preimage bytes, source/pending/final projection, pointer/hold state, admission rejection/valid terminal, foreign-nonce and later-hold preservation assertions remain. No claim that these now pass execution; four cases halt earlier at path assertion. |
| PREPARED immediate identity rejection | Both PREPARED append file/parent cases still reject a valid nonce-derived foreign transaction with eight graph/log paths and query count preserved before original resume. Pre-PREPARED archive points correctly have no durable nonce. |
| Same identity/append-once/immutable graph | Shared case tail still requires six release events in order once, exact PREPARED line/source/prelaunch unchanged and canonical final archive equality. Eight preimage cases retain single PREPARED/COMMITTED and exact captured PREPARED where present. |
| Terminal/rename semantics | Four-part v1.5+COMMITTED+true launch flag+current ledger SHA oracle plus expected boundary phase retained; old v1.4 authorized gate remains nonterminal. Existing visible final-rename and simulated lost-rename branches retained. File fsync before rename is distinct from rename-before-parent and parent-fsynced outcomes. |
| Terminal no telemetry/log byte mutation | Six graph artifacts plus both ledgers compared; terminal later-hold rejects and restored exact released clone projection ACK issues no query. Assertions remain unexecuted. |
| COW/cleanup/authorization | Detached ledgers/logs/locks, atomic mutable replacements and finally cleanup unchanged. Reviewer only creates new review files/external snapshots; no production locks/claims/gates/holds/results modified. Any future RUN remains independently built ordinary-copy clone, unique basetemp, external scheduler and <=5 task Python processes. |
| Report/testing evidence | Coder reports compile PASS and 2/62 collected, 60 deselected, 0.12s. No bound raw collection output/manifest independently verified here. Independent AST parse/static audit succeeds; no matrix/pytest/harness executed. |

## 3. Bugs/deviations found

### Q3-V4-R1 — MAJOR: temporary descriptor equated with destination

At test :2637–2640 expected_fd_path maps pending_temp→canonical, pointer_temp→release_pointer, hold_temp→hold and final_gate_temp→canonical. At :2643–2644 every regular descriptor requires Path(match["fd_path"]) == expected_fd_path.

Runtime _atomic_replace_bytes :626 creates tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex), opens that file and calls os.fsync(fd) at :633. The hook calls real_fsync and raises immediately afterward; os.replace(tmp,path) at :636 is never reached in these file-fsync cases. /proc/self/fd therefore records the temporary path; runtime finally closes its FD while leaving the temporary file visible. It cannot equal the destination, for any valid UUID suffix.

Affected four specs:

- pending_temp:file: descriptor is execution_isolation_gate.json.tmp.<uuid>; expected is canonical gate.
- pointer_temp:file: descriptor is global_hold_release.json.tmp.<uuid>; expected is current release pointer.
- hold_temp:file: descriptor is global_hold.json.tmp.<uuid>; expected is current hold.
- final_gate_temp:file: descriptor is execution_isolation_gate.json.tmp.<uuid>; expected is canonical pending gate.

Thus all four have a statically inevitable assertion failure before visible-state/restart oracles, despite correct injection. DESIGN §10:162 requires distinguishing the actual file-fsync point from rename/publish semantics; the destination is still the prior source/pending/hold projection.

Required correction: for *_temp regular FDs verify same destination directory, precise temporary basename prefix/suffix and ordinary-file identity without alias, expected payload and unchanged destination phase. Preserve exact equality for actual canonical/archive/ledger/fence FDs and exact parent for directory points. Do not weaken or remove FD/path/callsite proof, and do not treat a temporary inode as the destination.

## 4. Corrections made

None to implementation, test, coder report, prior artifacts or production. Created ordinary frozen copies, source diff, independent AST/static-expression audit and new report/evidence only. The static audit reconstructs literal row/prefix/path comparisons without importing or running reviewed functions. No pytest, long matrix, worker, release/claim, model/checkpoint/data or GPU operation occurred.

## 5. Remaining risks

Unique rows and normalized identifiers show static selection consistency, not observed exactly-once injection. Each test records one match then raises, but execution evidence is still required after correction. The temp-path counterexamples are mathematical Path inequality under the actual runtime construction/order, not actual syscall probes. Coder compile/collection evidence is report-only and cannot validate these branches. Approved source inventory comparison does not prove universal transient filesystem/lock preservation or reconstruct B6 historical preimages.

## 6. Whether experiments are safe to run

**NO-GO for isolated Q3 qualification RUN at v4.** Correct the MAJOR assertion, bind and independently review the successor. A future implementation PASS allows only a separately bounded isolated Q3 qualification RUN; production release, staged graph GO and scientific RUN require separate gates. B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain.

### Evidence bindings

- `/tmp/f0-v15-Q3-review-v4-20261014-a/test_stage_F0_successor_v15.py` — SHA `a78b67b3d6d0e0eef74ce6c0f7bfdd4a6b7708ed4c813fff9754b57d3d274019`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v4.md` — SHA `631dce94630afc8f71dd1c326bbefaffd1af3d901c4eb7941eef8fd9622ff774`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/audit.py` — SHA `79446ec0d628770a905e0ed5e0e2b8d8b87a2db9b479a477e7fc56795d2f2cc2`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/audit.stdout.txt` — SHA `91cd3aa98a9cf303aad278e3bc7a60bcc4f7edfe5d6dd64096be26655662b4ef`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/static_audit.json` — SHA `eeccc52a2f94960827e21808b5787f95425a61f13e706b5a745d786d6ed6666b`.
- `/tmp/f0-v15-Q3-review-v4-20261014-a/source-vs-v3.diff` — SHA `53ed345caec14f4d93b211b6a5705ef7008ad6654906b28c013f3d49650abfe0`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v4_evidence.json` — SHA `446254c617037d9913b688f6b3a1ef63765fbf9f0251716157a8309c84451150`.
