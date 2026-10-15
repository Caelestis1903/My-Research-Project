# REVIEW REPORT — v1.5 Q3 fault-recovery implementation v1

## 1. Verdict

**FAIL. qualification_RUN_precheck=False; Q3 remains NOT CLOSED.** Two MAJOR coverage defects prevent treating this harness as implementing DESIGN v2 §10's complete fault/recovery criteria. This is a static implementation verdict, not a failed matrix result; the reviewer executed no matrix, pytest, training or production action. Return these findings to coder; review the successor separately.

B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. No release, staged graph GO, scientific RUN or production recovery is permitted.

## 2. Requirement-by-requirement verification

| Requirement | Verification |
|---|---|
| Only test harness source changes | PASS in approved baseline03 source scope: independently compared all archived exp/m3oe .py/.sh identities; only test_stage_F0_successor_v15.py differs. Frozen diff is captured externally; this repository has no Git metadata, so comparison uses accepted attempt03 source preimage f619d97b… rather than git diff. Runtime remains 0b8020e5…; B6 tool remains f76433a6…; no mathematical/tensor/mask/gradient/scientific behavior change. |
| Actual file-vs-directory fsync | PARTIAL: new function at :1991–2125 loops gate/hold/release_pointer/prepared_ledger × file_fsync/parent_directory_fsync. /proc/self/fd path plus fstat distinguish regular FD from control-dir FD. real_fsync executes first, then raises; thus this is failure after successful file fsync or after successful parent fsync, not a simulation of failed syscall. R1 identifies omitted boundaries. |
| Preimage/PREPARED visible state | PARTIAL: :2089–2108 confirms active hold, exact selected source/archive bytes and absent PREPARED ledger for archive interruptions; PREPARED interruptions require one complete PREPARED row. Resume checks same transaction ID and one COMMITTED. It lacks full graph/prepared-raw and all required negative oracles; R2. |
| Different nonce | PASS for original 14-hook matrix: :1829–1858 derives uuid5 from a valid new nonce, updates pending/release IDs and pending hash, reaches transaction-ledger identity rejection rather than malformed nonce validation, compares eight graph/ledger paths. Not implemented in new eight syscall cases; R2. |
| Later hold preservation | PASS at original 14 recovery points (:1888–1913), with hold/ledger/gate byte preservation; added terminal later hold :1943–1966 compares six artifacts and both ledgers. New eight syscall recovery points omit later hold; R2. |
| Final rename visible/nonvisible | Existing visible branch at :1706–1725 sees exact final bytes and COMMITTED_IDEMPOTENT after rename-before-dir-fsync; independent matrix :1915–1928 restores only clone pending bytes to simulate lost rename and recovers. These demonstrate selected restart outcomes; no actual crash/reboot durability is claimed. |
| Terminal exact bytes/no telemetry/append | Added :1936–1980 snapshots canonical gate, hold, immutable release, pending archive, release pointer, final archive, release ledger and observation ledger. Same-identity ACK returns COMMITTED_IDEMPOTENT, leaves bytes exact and query count unchanged. Runtime :2218–2245 returns before query/observation/append for exact terminal graph. Written assertions are meaningful but unexecuted. |
| COW and production preservation | New writes target base/case roots; worker logs/locks detach; original matrix explicitly detaches release/observation ledgers before append. New base has no ledger before per-case creation. Runtime mutable gates/holds/pointers use atomic replace; new artifacts O_EXCL. No source/production edits by reviewer. These roots must remain inside a separately built isolated clone when RUN; directly executing in the real workspace is not authorized. |
| Coder evidence | Report honestly states matrices NOT EXECUTED, only syntax and collection. AST parse independently succeeds. Coder's 2-tests-collected/py_compile assertions are report-only here: no bound raw output/manifest was independently verified. Collection cannot validate recovery assertions. |

## 3. Bugs/deviations found

### Q3-V1-R1 — MAJOR: most syscall boundaries still aggregated

DESIGN v2 §10:162 requires fault injection after every file rename/fsync and ledger append boundary. Attempt4 Q3 explicitly rejected named hooks as sufficient (§Q3:30). Test :1785–1792 retains fourteen hooks; :2030–2031 adds only three preimage helpers plus PREPARED append, with two fsync categories.

Runtime _append_jsonl_durable :559–575 fsyncs the file and then parent; durable_bytes_create :604–621 does likewise; _atomic_replace_bytes :624–643 fsyncs the temporary file, renames it, then fsyncs parent. The release artifact, pending archive and final archive durability calls at :2401–2504 still have no separate file/parent interruption. Canonical pending gate, pointer, hold and final gate replacements lack their temporary-file-fsync interruption. PREPARED is the only release-ledger event whose actual file/parent fsyncs are independently faulted; later events remain aggregated, including COMMITTED. Administrative observation appends likewise have no individual syscall case in this addition.

Correction required: cover each remaining actual boundary with identifiable FD/path/phase and exact state/recovery oracles. Keep rename surviving/not-surviving branches. No runtime or scientific redesign is requested.

### Q3-V1-R2 — MAJOR: new syscall cases lack required recovery-negative/state assertions

At test :2079–2117 each new syscall case goes from interruption assertions directly to same-identity commit. None injects a later incident hold or attempts a correctly nonce-derived alternate transaction there, although DESIGN §10:162 requires later hold at every recovery/acknowledgement opportunity. None calls the admission validator to prove the incomplete graph rejects launch.

The PREPARED branch captures parsed event/ID (:2093–2098), then after recovery checks row0 event/ID and COMMITTED count (:2114–2117). It does not save/compare the exact PREPARED first-row bytes or count PREPARED once after recovery; changed extra PREPARED fields would not be caught by these explicit assertions. For preimage interruptions only the selected input source is compared, so mutations of other graph artifacts at that point are not directly detected.

Correction required: snapshot full relevant preimage/partial graph and exact PREPARED bytes; test alternate identity and later hold without graph mutation; assert incomplete admission rejects; restore only clone test state and recover the same identity with exact PREPARED bytes and one PREPARED/COMMITTED; finish with terminal byte/no-query checks. An equivalent phase in another named-hook case does not supply the explicitly required per-interruption oracle.

## 4. Corrections made

None to implementation, tests, reports under review or production. Copied v1 source/report into a new external review directory before coder successor edits, parsed AST, captured actual diff and wrote this new review/evidence only. No code imported/executed from the reviewed harness.

## 5. Remaining risks

All recovery assertions are currently unexecuted. Static inspection establishes the selected injections and missing oracles, not successful runtime recovery. A separately reviewed successor still needs isolated qualification RUN with unique basetemp, protected identities and raw outputs. Coder report's basic checks have no independently bound raw log here. The source-only preservation comparison excludes filesystem metadata, non-source mutable production artifacts and transient locks; no universal no-side-effect proof is inferred.

The unchanged runtime implements terminal durability fencing and exact graph validation; this report does not recertify every earlier runtime invariant or close complete DESIGN §§8/10. Scientific formulas, shapes, routing, weighting and gradients are unchanged and not applicable to this harness increment.

## 6. Whether experiments are safe to run

**NO-GO for the proposed complete Q3 qualification matrix until MAJOR findings are corrected and successor reviewed.** No production action, release or scientific RUN is approved. Historical B6 OPEN/FAIL remains even after a future Q3 pass. A later implementation PASS would permit only its separately scoped isolated qualification RUN, with actual acceptance still dependent on raw execution evidence.

### Bound evidence

- `/tmp/f0-v15-Q3-review-v1-20261014-a/test_stage_F0_successor_v15.py` — SHA `95625db76bdf69ad3b351f100fd952447a6ed2834444e6de9b4499aa62a126f2`.
- `/tmp/f0-v15-Q3-review-v1-20261014-a/stage_F0_successor_v15.py` — SHA `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- `/tmp/f0-v15-Q3-review-v1-20261014-a/stage_F0_v15_Q3_fault_recovery_implementation_v1.md` — SHA `09648dd345c1febfa9e2a1d0ac79f315b7f21e68e6dc8ef14c9523a26a5331a9`.
- `/tmp/f0-v15-Q3-review-v1-20261014-a/source.diff` — SHA `6a5342e6882f906fb3358d0a401e3585f0fb3b9adc28a3649582775282ae463b`.
- `/tmp/f0-v15-Q3-review-v1-20261014-a/static_audit.json` — SHA `80182b17c8ed8faed13bc5f5abff8de4e1e4c7d91aa833ae15df7b4449d3577a`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v1_evidence.json` — SHA `b1504ef5a8ef8dc9794f925542a4849ddca86269390bdf2b3d3628f7ee933b8f`.
