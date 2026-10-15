# REVIEW REPORT — Q3 four frozen incident clone-input supplements

## 1. Scoped verdict

**PASS for the exact input-copy scope only.** A fresh ordinary-copy isolated clone may include the four listed incident files, in addition to the previously reviewed remediation candidate. These are frozen dependencies already required by the unchanged v6 harness/runtime. This does not authorize copying anything further or starting a test/retry/RUN. No files were copied by this review.

B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. The v6 implementation PASS still permits only its separate bounded qualification workflow; this scope report does not issue the next RUN instruction or waive clone preflight.

## 2. Requirement-by-requirement verification

Independent source reads used O_RDONLY|O_NOFOLLOW after lstat validation. Each source and every ancestor was ordinary/no symlink; nlink=1. Source lstat before, opened fstat, fstat after and lstat after matched device/inode/type/size/mtime/ctime. Hashes exactly equal runtime's frozen values. All four are absent from attempt01 clone; no missing file was invented or overwritten.

| Exact filename under experiments/stage_F0_outputs/runs/worker_logs/ | Bytes | Frozen/current SHA-256 |
|---|---:|---|
| `worker_1_failure_1791870436_8fc4a33fc7574a368312b4965e24a283.json` | 1582 | `1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4` |
| `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.stderr.log` | 921 | `2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a` |
| `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.phases.jsonl` | 1096 | `49ddf8f36f62da00fb3f19d1105ae5f390070c393b0744581e757142393e7cba` |
| `trajectory_worker1_1_8fc4a33fc7574a368312b4965e24a283.stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The empty stdout is an existing stable ordinary file with size0 and its real FD read hashes to the empty digest; this is not an absence assertion. Preserve raw bytes exactly; do not parse/rewrite, clean logs, synthesize an empty placeholder or create the missing receipt. These files preserve the known no-claim bootstrap incident; they do not repair B6 historical gaps.

The two DESIGN/review origins were independently verified in the existing attempt01 clone `/tmp/f0-v15-Q3-run-20261014-a/clone`: exactly one manifest row each with reviewed_binding:true, correct hash and current actual ordinary single-link clone bytes. Current reviewed source bytes also match, and source/clone device+inode differ. Builder `build_ordinary_clone_v2.py:24–25,99–112` supplies them through CURRENT_BINDINGS with source=REPO/relative; their provenance is separately reviewed live input, not B6 raw.

| Existing clone relative path | Bound and actual SHA-256 |
|---|---|
| `experiments/reports/stage_F0_v15_successor_control_design_v2.md` | `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316` |
| `experiments/reports/stage_F0_v15_successor_control_design_review_v2.md` | `9dd314ddd4b1ae7622436e00195888ab682089e65bf68649816afaeaeef392eb` |

Original clone_manifest SHA `b3b91752996a4cbb023c51ca53933304aa7b3471976000902e2fe37f58e984b4`; builder SHA `c0121c83d8a0a855901b6bb5a64af8dfc0c050d4d924e2b10fc2eb83ea1ca12d`. Current stable observations validate existing identities; they do not retrospectively manufacture missing before-copy provenance.

## 3. Manifest schema/provenance and limits

Use a fresh `F0-Q3-CLONE-INPUT-PROVENANCE-v1` manifest (template and required fields in evidence JSON): bind attempt ID/time window, baseline03, preserved attempt01 failure/input manifests, v6 test/runtime/coder/review, prior candidate-scope review and this new scope review/evidence.

For exactly five live supplements (candidate+these four), label `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` and record source path/role/expected and observed hashes, before/open-FD/after stats, no-symlink ancestry, observation/copy UTC, destination unique ordinary-file stat/inode/readback hash and source-after-copy hash. Destination must have nlink1 and an inode different from source/archives, with no hardlink/symlink/reflink shortcut. Publish only within a fresh isolated clone; source remains read-only. Do not attribute these payloads to baseline03 raw or alter that baseline, prior attempts, reports, pointers or production lock state.

For the two design documents, preserve `EXISTING_REVIEWED_INPUT` origin references to exact source and unique manifest rows, builder/hash and current revalidation. Their existing clone hashes are verified; no permission to change their content is needed or granted.

Keep origin/copy history distinct from the final clone inventory: every final relative destination must have exactly one unambiguous final SHA/origin row. Reviewed overlays must not leave conflicting old/current SHA rows presented as a single final inventory. Complete all nineteen frozen-artifact path/type/hash checks plus reviewed source identities and receipt/F140 absence checks before any test. No test, telemetry/model initialization, claim/release or source bypass belongs to this input setup review.

## 4. Corrections/execution

None to source, protected production, clone, attempts, implementation or prior evidence. Reviewer wrote only independent external audit outputs and this new report/evidence, using sequential project Python with external process slots. No clone/input copy, pytest, worker, training, checkpoint/model/data loading, GPU action or RUN occurred.

## 5. Remaining risks

New inputs are honest current post-incident snapshots matching already frozen SHA values. They are not baseline03 archived originals and cannot recover missing pre-edit inventory, runner preimage/diff or historic shared-lock observations. Existing docs' source route is supported by manifest plus builder and current hashes; original before-copy descriptor snapshots were not recorded, so do not claim stronger historical provenance.

This is a bounded six-path origin audit (four source supplements plus two docs/source-copy pairs), not acceptance of the failed attempt01 matrix or of a corrected builder. A new builder and new clone must still pass their own identity/alias/full-dependency manifest preflight; preserve setup failures and their outputs. This report does not excuse any package/environment/process-slot or broader raw-evidence requirement.

## 6. Recommendation

Include exactly these four inputs alongside the previously scoped candidate in the fresh ordinary-copy plan, with the stated schema/provenance bindings. Complete and review clone preflight before parent separately instructs the runner. **RUN_authorized=false; retry_authorized=false.** B6 OPEN/FAIL, NOT_GRANTED and production NO_GO remain.

### Evidence bindings

- `/tmp/f0-v15-Q3-incident-input-scope-20261014-a/audit.py` — SHA `648def5a2d620c63120d343420ccf04186575e9acdbd11535dd4b8b8d66807cb`.
- `/tmp/f0-v15-Q3-incident-input-scope-20261014-a/audit.stdout.txt` — SHA `670c0f352df06f332c69f0038452fe287fbd4d6a75c4b91368e871be90f339fd`.
- `/tmp/f0-v15-Q3-incident-input-scope-20261014-a/audit.stderr.txt` — SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `/tmp/f0-v15-Q3-incident-input-scope-20261014-a/scope_audit.json` — SHA `dc6e21d5d427ca1f535b01b30809d248d4f2fab7c7f3dbbdce94eb00ef9fb0a8`.
- Evidence JSON `experiments/reports/stage_F0_v15_Q3_attempt01_incident_input_scope_review_v1_evidence.json` — SHA `ae65192d7f13b567d4e366873a6d55d9bf1702f5154cd924d3fadb7a716a300a`.
