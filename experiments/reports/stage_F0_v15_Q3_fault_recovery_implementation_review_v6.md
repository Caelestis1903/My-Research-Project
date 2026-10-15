# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v6

## 1. Verdict

**PASS — implementation readiness for a separate bounded isolated Q3 qualification RUN only. qualification_RUN_precheck=True.** All prior implementation blockers are addressed statically; no new blocking defect was found in the frozen v6 harness. The reviewer executed no test/matrix, so Q3 execution is not accepted and Q3 remains NOT CLOSED.

B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO remain. This PASS does not permit production release/claims, staged graph GO, scientific RUN, training or broader experiment execution.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Frozen source/report/design | Test SHA 80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9; coder report SHA 1667b761c43d68f8d33452b6557409a2a8caf15735a701ffb335f84e82f44723. Runtime SHA 0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2 unchanged. DESIGN v2 SHA 8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316. Ordinary frozen copies exclude successor edits. |
| Truly lazy payload selection (v5 blocker) | CLOSED STATIC: :2675–2684 uses if/elif by expected_file_key. Pending/pointer/hold branches contain no final_archive call; only guarded final_gate_temp branch reads it at :2682. Unknown class raises AssertionError. AST branch trace confirms absence of eager dictionary/lazy-default trap. |
| Temp FD versus destination (v4 blocker) | All four regular *_temp cases retain exact destination parent, filename+.tmp.+32 lowercase hex suffix, recorded regular FD, temp bytes captured after real fsync and before injected exception, actual temp readback and captured/current old destination bytes. Rename at runtime :636 has not happened for file fsync :633. Non-temp regular FDs still equal actual destination; dir FDs equal actual parent. |
| Temp payload/phase | Pending temp exactly pending gate; pointer temp exactly release payload; hold temp SHA exactly PREPARED released_hold_sha256 with active:false; final temp exactly immutable final archive plus COMMITTED state. Old destination is source canonical gate/pointer/hold for first three, pending gate for final. Hold byte self-comparison is supplemented by exact PREPARED digest, so not a tautological payload oracle. |
| Phase-aware/unique 42+8 mapping (v1–v3 blockers) | AST independently verifies 42 unique specs =30 forward+12 recovery; 21 distinct directory (class,phase) rows with unique IDs. Six recovery regular IDs normalize to recovery:<class>:file; all 12 selectors suppress during explicit seed phase. Eight preimage specs are byte/AST-equivalent in behavior to reviewed v4; descriptors require exact file then optional parent sequence. |
| Exact seed-before-recovery callsite | Each retry case uses the specific named seed hook, checks exact exception and partial source object, requires no injected match, then enables recovery phase and resets match state. Existing release/pending/final archives map to runtime :2398/:2400; expected pending canonical/pointer/released hold map to :704/:707 via :2423/:2437/:2460. Selected hook calls actual fsync then records class/phase/kind/path and raises. Exactly-one recorded match assertions authored; actual match counts await RUN. |
| Ledger/observation/rename boundaries | Three preimage archive file/parent pairs + PREPARED append pair; three immutable create pairs; five later ledger append pairs; two observation-stage pairs; four atomic temp/rename-parent pairs; final gate fence pair; six existing-object retry pairs. Existing named matrix covers four selected actual post-rename/pre-parent interruptions and named publication/event points. Each class points at distinct state/callsite, not merely a renamed aggregate hook. |
| Exact interrupted state/admission | Ledger visible prefix, raw PREPARED/source/prelaunch, preimage digests, canonical source/pending/final bytes, hold/pointer/release projections and relevant archive presence checks retained. Incomplete graphs reject admission; only exact expected v1.5 final publication is terminal. |
| Foreign identity/later incident hold | Both PREPARED append interruptions reject valid alternate nonce before original resume, preserving eight paths and query count. Every PREPARED-or-later shared case checks alternate identity and later active hold; original clone hold only restored for same-identity recovery. Terminal later hold rejects and stays exact. Pre-PREPARED archives have no durable nonce yet and correctly use source/hold preservation oracles. |
| Same identity/append-once/immutable graph | Ordered six release events once, exact original PREPARED line/source/prelaunch, preimage digests and final canonical/archive equality retained. Eight preimage cases require one PREPARED and COMMITTED and exact captured PREPARED where present. |
| Terminal B1/no telemetry | Terminal requires v1.5 schema, COMMITTED state, true launch flag and exact current ledger SHA; actual classification must match expected boundary. Old authorized v1.4 source gate remains nonterminal. Six graph files plus release/observation ledgers remain exact during ACK; query counter unchanged. Visible final rename and simulated lost-rename branches retained. |
| Source/scientific/COW preservation | Approved baseline03 exp/m3oe .py/.sh comparison finds only harness changed; nine protected current hashes match accepted baseline exactly. Runtime/model/scientific behavior unchanged; no math/tensor/mask/routing/weight/gradient modification. Ledger/log/lock COW and finally cleanup retained. Source/hash checks are bounded, not universal transient-side-effect proof. |
| Actual verification limits | Reviewer AST/static audit succeeds in project environment, external scheduler, one Python at a time; no reviewed runtime/harness imported or executed. Coder reports compile PASS and 2/62 collected, 60 deselected, 0.14s; no bound raw collection output independently verified. Compilation/collection does not establish matrix PASS. |

## 3. Bugs/deviations found

No remaining CRITICAL, MAJOR or MINOR implementation blocker identified within this exact static Q3 scope. V1 missing syscall/state oracles, v2 old-authorized gate/recovery/mid-PREPARED identity gaps, v3 duplicate maps/recovery prefix/seed-trigger errors, v4 temporary-path equality and v5 future-artifact eager read are all addressed statically. No claim is made that all tests actually pass or every qualification criterion is executed.

The only change from frozen v5 is class-specific payload branching. Its final archive read is dominated by expected_file_key == final_gate_temp. Other dictionaries in temp metadata construct byte projections or Path objects without reading future artifacts. Every early selected temp branch uses only its current immutable input/captured bytes, while later final branch reads the already-created final archive. Existing selector/state/terminal code and the preimage function remain unchanged.

## 4. Corrections made

No reviewed implementation/test/report, prior evidence, protected or production changes. Created new ordinary frozen copies, actual diff, independent AST/static/source/protected-hash audit and this report/evidence. No pytest, matrix, worker, claim/release transaction, model/checkpoint deserialization, sample access or GPU operation occurred.

## 5. Remaining risks

All 50 fsync specifications and retained named fault/restart checks remain unexecuted at this checkpoint. Static matching, branch dominance and assertions establish implementation readiness, not durability or recovery evidence. Runtime timing, filesystem behavior, historical validator dependencies, slot behavior and actual match counts can still fail during qualification. Preserve raw failures rather than infer success from collection. Read-only frozen checkpoint/source hashes used by existing validators remain distinct from model loading; they cannot reconstruct B6 pre-edit evidence or justify scientific work.

## 6. Whether experiments are safe to run

**A separately orchestrated bounded isolated qualification RUN is permitted for these exact focused nodes only:**

- test_each_release_fault_boundary_uses_independent_prepared_clone
- test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity
- test_each_release_syscall_boundary_has_independent_restart_oracle

Use a fresh independently verified ordinary-copy clone of the bound source graph, unique basetemp, external evidence archive/scheduler and required project Python wrapper/slot accounting. Keep task Python concurrency <=5 and concurrent roots independent. Use synthetic target telemetry and administrative clone graphs only; no production files/locks/claims/release, training, model/checkpoint deserialization/forward, samples or GPU math. Necessary existing historical raw-byte SHA validation is bounded read-only and does not expand source/data scope.

Record exact command/exit/stdout/stderr, source and protected before/after SHAs, process/slot evidence, every boundary descriptor/phase/match count/visible state/recovery/terminal result, and archive manifest. Preserve failures. Independent execution review must accept raw evidence before Q3 closure. This PASS does not itself start RUN or grant staged graph GO/production/scientific authority. B6 OPEN/FAIL, forward NOT_GRANTED and production NO_GO remain.

### Evidence bindings

- `/tmp/f0-v15-Q3-review-v6-20261014-a/test_stage_F0_successor_v15.py` — SHA `80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v6.md` — SHA `1667b761c43d68f8d33452b6557409a2a8caf15735a701ffb335f84e82f44723`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/audit.py` — SHA `d458b42e1a036bea3d7fecf0e4bca672861d7af234c389d82e8d4e92decc8ece`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/audit.stdout.txt` — SHA `1c51d9115a39d0715fcaea3b91a50298d3ee45a438fda4e6d6fdeb62414853db`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/static_audit.json` — SHA `61757da102bf5f3949a8ca0381839ec90935bfa67a2376530dcc17242d784b39`.
- `/tmp/f0-v15-Q3-review-v6-20261014-a/source-vs-v5.diff` — SHA `7feaa48bc04a995d54b9824d88342cad994ebed4bed7f2483a6848867322421d`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v6_evidence.json` — SHA `bc863621a092418ec0884deff52059b68cfede3212a1eccaf6083f0dce5b1322`.
