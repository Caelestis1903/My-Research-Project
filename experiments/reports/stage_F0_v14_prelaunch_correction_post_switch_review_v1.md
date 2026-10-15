# REVIEW REPORT — v1.4 actual canonical prelaunch switch

Review date: 2026-10-13. Independent reviewer. Scope: separate read-only post-switch verification required by the accepted correction specification and artifact review.

## 1. Verdict

**GO** for the completed administrative canonical switch and subsequent preparation of the complete v1.4 pending gate. Implementation-review equivalent: **PASS**. No blocking deviation was found. This report grants no release or training authorization; the hold is still active and the production gate is unchanged.

The specification is `experiments/reports/stage_F0_v14_prelaunch_correction_design_v1.md`, SHA256 `e46285ce811d0a4aef439e921609293b9457e25173f064e473b0bb4dbfaa6113`. Exact prepared-artifact approval is `experiments/reports/stage_F0_v14_prelaunch_correction_artifact_review_v1.md`, SHA256 `6219b826c2c71cfb463807268360057610ac6f79aebe22d16111cc20e2d5fff5`. That review's sole MINOR attestation typo was already resolved by a separate preserved v2 attestation before the switch.

## 2. Requirement-by-requirement verification

### Actual ordinary canonical file and retained original

All paths in this table are relative to `experiments/stage_F0_outputs/runs/`.

| Artifact | Actual SHA256 | Device / inode | Mode / bytes / links |
| --- | --- | --- | --- |
| `execution_control/prelaunch_audit_v14_fresh_search.json` | `8945203e534e033de8ab3bf5eea60bbc367e110cef24809710f5615c46f4da9e` | `2049 / 54658485` | `0444 / 61321 / 1` |
| `execution_control/prelaunch_audit_v14_fresh_search_corrected_v2.json` | `8945203e534e033de8ab3bf5eea60bbc367e110cef24809710f5615c46f4da9e` | `2049 / 54658482` | `0444 / 61321 / 1` |
| `execution_control/prelaunch_audit_v14_fresh_search_original_12c723a4.json` | `12c723a4fff4115e68bf03c4283adcca6b225cf739fdf1e5dc0846215ac59c39` | `2049 / 54658476` | `0444 / 52937 / 1` |
| `execution_control/prelaunch_audit_v14_fresh_search_switch_v1.json` | `12d9296e1ba7c45ca40b553d2ddda9a113787d1b0a9b756c2c0b269580238c78` | `2049 / 54658486` | `0444 / 8842 / 1` |

The reviewer independently checked `lstat`, raw bytes and SHA256. Every table entry is an ordinary regular file, not a symlink. Actual canonical bytes equal the approved corrected v2 bytes exactly. The old canonical inode remains reachable through the original archive with its original bytes/SHA; its link count fell from 2 to 1 because the original canonical directory entry was replaced. The canonical inode changed to a distinct new inode as required for ordinary-file replacement. The temporary switch path is absent.

The correction manifest is unchanged at SHA256 `24cbe29da3b12e09805922223509e5f5b58dc7ccc79224d219649552f912fd0d`. The switch record binds that exact manifest, the exact artifact-review path/SHA and exact original/corrected SHA values. The canonical's required positive integer `disk_free_bytes=291559395328` remains equal to its same-preparation nested `disk.free_bytes`.

### Durable switch event and implementation evidence

The actual switch event schema is `F0-PRELAUNCH-CORRECTION-SWITCH-v1`, ID `f0-v14-prelaunch-correction-v2-switch-20261013`, decision `SWITCH`, with `switch_performed=true` and `release_or_launch_authorized=false`. Before identity `2049/54658476`, after identity `2049/54658485`, and retained archive identity `2049/54658476` agree with this and the preceding independent review. Before/after byte hashes are the exact approved hashes above.

The event records successful locked graph/resource/absence checks, exclusive creation, temporary-file fsync, directory fsync before replacement, `os.replace(temp, canonical)`, directory fsync after replacement and archive preservation. The retained actual source `/tmp/correction_switch.py`, independently inspected SHA256 `36b4eeec57240c29f7e52e51e31241d886306b71b1b2568413e069581a987828`, implements these steps under `execution_state.lock`. It writes/flushed/fsyncs temporary file contents, sets mode `0444`, fsyncs the directory, immediately rechecks the original/archive/corrected/manifest/active-hold identities and release absence, atomically replaces the directory entry and fsyncs the directory. It runs both unchanged validator checks, exclusively writes/flushed/fsyncs the immutable event and fsyncs its directory before releasing the execution-state lock. It never writes the gate, hold, migration ledger, statuses or training assets.

Historical fsync execution is supported by the retained source and completed durable event, not a syscall trace. The current filesystem state independently matches the recorded outcomes. The code fsyncs temporary file contents before its final `fchmod`; this review verifies the current final mode rather than asserting separately traced durability of that permission change. Content/entry durability steps required for the approved switch are present.

The record's observation stamp is `2026-10-13T05:17:26.081262Z`; GPU query timestamp is Unix `1791868645.3632305`. Its `switch_at_utc=2026-10-13T05:17:42.082111Z` is assigned after the post-switch validator calls, so it records completion/finalization of the switch procedure rather than the exact `os.replace` syscall instant. This timing interpretation is verified directly in the source.

### Actual unchanged runtime acceptance and rejection

The reviewer ran the real `validate_current_gate(..., require_release=False)` twice against the actual canonical path and an in-memory complete v1.4 pending gate constructed from the immutable migration gate. No file-read/hash substitution was used and no production gate was written. The candidate explicitly includes `postmigration_review_verdict: GO`, exact structured review path/SHA, current committed active-hold/migration bindings and forbidden-scope false fields.

1. Original-SHA candidate rejects with `v1.4 prelaunch audit is missing or changed`.
2. Corrected-SHA candidate passes the entire unchanged validator at `2026-10-13T05:20:15.679139+00:00`, returning gate, snapshot, hold, historical H49, prelaunch and scientific search graph.

That full pass independently rechecks archived source snapshot, frozen C97/incident inventories, H49 historical protection, exact H51/F140 identities, scientific gate/control amendment, control inventory, migration ledger, status transitions, complete prelaunch conditions and structured postmigration GO review. This is the actual post-switch verification required by the earlier artifact review; it supersedes its prospective single-path read simulation for this boundary.

### Hold, migration, source and production gate remain unchanged

- Active hold is true, SHA256 `29597e3ca48a14137c1214874c3281913436a8683174c4d776504c23a663fb86`, with exact 51-member H51 preservation.
- Migration ledger SHA256 remains `b91d912c3d7f9be9b896e19c6fa7744e92872c2ea870decedef00a5ac1087e30`.
- Source snapshot SHA256 remains `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533`.
- Immutable migration pending gate SHA256 remains `14df05478e1f75bc2f2377fe5305b776e0fb659b33453c4b9d247fb652757a12`.
- Production gate SHA256 remains `13e5e1dd352faaa46d297aa6651ec7acf6df80c32f8636360a5cfa7a30dfdac7`.
- `global_hold_release_v14.json` and `postabnormal_v14_release.jsonl` are absent; there is no v1.4 release PREPARED/COMMITTED transaction, and the hold contains no `V14_PRINCIPAL_FRESH_ONLY_RELEASE` event.

The reviewer rechecked canonical/archive/hold/migration/production-gate hashes and release absence after both actual runtime checks. No algorithm, research formula, tensor/mask/routing/weighting semantics, gradient path, baseline or hypothesis changed in the inspected switch source.

### Independently refreshed resources and queue state

At approximately `2026-10-13T05:19:59.499536+00:00`, independent `nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits` returned:

| GPU | UUID | Used / total MiB | Used fraction |
| --- | --- | --- | --- |
| 0 | `GPU-4ac4b341-3045-1093-fec3-492516863e5a` | `4222 / 24576` | `0.17179361979166666` |
| 1 | `GPU-2a59fb49-6bef-5320-74ab-ffb24d056908` | `15 / 24576` | `0.0006103515625` |

Both target UUIDs match the approved audit, and memory fractions are strictly below 0.90. GPU compute utilization was not used as an eligibility gate.

The actual fixed queue still has 288 entries: 97 `VALIDATION_COMPLETE`, 51 `RECOVERY_BLOCKED`, 140 implicit untouched `UNCLAIMED`, zero `CLAIMED` and zero `RUNNING`. All F140 canonical locks remain absent by the full source-snapshot/runtime checks. The reviewer independently opened existing lock files read-only and probed nonblocking exclusive locks: all 114 existing paths from the 293-path required production inventory were available, all 179 absent paths remained absent, and all five existing Python scheduler slots were available. No missing lock or slot was created.

Independent `/proc` enumeration found the reviewer project Python process and the unrelated Qwen uvicorn service, with no F0 worker/trajectory-child Python process. Current output filesystem `shutil.disk_usage` returned total `983344672768`, used `641764356096`, free `291553619968` bytes; free space is positive.

## 3. Bugs/deviations found

No CRITICAL, MAJOR or MINOR post-switch correctness defect was found. The timestamp and historical syscall-evidence limitations above are explicit bounds on what this read-only review can attest, not additional execution permission. The earlier attestation typo is already corrected and was not reintroduced.

## 4. Corrections made

None. The reviewer did not modify any production file, source, lock contents, gate, hold, ledger, status, attempt or training asset. Only this distinct review report was added.

## 5. Remaining risks and boundary

The immutable audit still intentionally carries its preparation timestamp `2026-10-13T05:08:14.724918Z`. The switch event and this reviewer report provide later observations; they do not rewrite or mislabel the original audit measurement. The unchanged validator verifies audit bytes/fields and does not enforce a wall-clock maximum age. GPU/process/claim/slot/lock/disk observations describe their collection times and must be refreshed before a separate principal release and before workers claim keys. This report does not promise that resources remain unchanged afterward.

Create the full new pending gate with corrected canonical SHA, `postmigration_review_verdict: GO`, structured review path/SHA and exact correction manifest/artifact-review/post-switch-review bindings. Preserve the old production gate bytes and immutable pending gates. Validate that actual gate before the separate principal-release transaction. The correction manifest/review bindings are additional review provenance; their existence does not bypass the runtime release graph.

## 6. Whether experiments are safe to run

**Not yet authorized by this review.** The actual administrative switch is **GO**, both real validator outcomes are verified, and pending-gate preparation may proceed. Training remains behind the active hold until the complete new pending gate validates, separate principal release succeeds and fresh memory/process/lock checks satisfy the approved scope. H51 retry/recovery remains forbidden.
