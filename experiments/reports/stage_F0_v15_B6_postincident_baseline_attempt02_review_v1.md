# REVIEW REPORT — B6 post-incident baseline attempt02

**Verdict: FAIL. Baseline acceptance: NOT_ACCEPTED.** The archived observations that exist are internally consistent, but the submitted scope omits required source dependencies and a referenced raw reproduction script. This attempt cannot be marked `ACCEPTED_BOUNDED`. The prior B6 tool-v3 implementation PASS is unchanged; this is a separate review of the actual collection. B6 historical disposition remains `OPEN/FAIL`, forward disposition `NOT_GRANTED`, execution `NO_GO`. No staged graph GO, release, claim or scientific RUN is authorized.

## 1. Scope and bindings

Independently reviewed existing artifacts only; no recollection, E2E, training, protected write or repair was performed. A single project-Python audit process streamed existing archived bytes and inspected stored/current protected identities. The raw audit is preserved externally, with all checked payload records. No audit process remains running.

- Baseline manifest: `research_F/F0/evidence/B6/f0-v15-postincident-20261014-02/manifest.json`; SHA `570c22e80520aff3fa602005e1ef7128c09372f6097071bb365b15654a5f74a4`.
- External artifact manifest: `/tmp/f0-b6-postincident-20261014-02-run/final_artifact_manifest.sha256`; SHA `988d32d55f9993c5cad887077e5881df840deb8c8895538075a224a4417e1ba0`.
- Approved policy DESIGN v1 SHA `4084a9ec3054398e6b78ff2a1b1f8bcf55d1556b40fdff0a4c12eeae2941ff5a`; design-review SHA `5d37146a57737f510469f77283255ab260e6e50adefba193bf404cece01544d6`.
- Review evidence: `experiments/reports/stage_F0_v15_B6_postincident_baseline_attempt02_review_v1_evidence.json`; SHA `a32960b01f877759958aa1418b779a84b9d41e83fc569942d2feb7882a9a3749`.

- Independent raw evidence: `/tmp/f0-B6-baseline02-review-20261014-a/audit.py`; SHA `cfc7925c7cbf565a7e9de4f75cf090081be02185bd5ea74a7c744f4b89a94bfc`.
- Independent raw evidence: `/tmp/f0-B6-baseline02-review-20261014-a/audit.stdout.txt`; SHA `53385400a46196f64f8d718d74e99441e3eb7d82a4469812eff1180bc1bd941b`.
- Independent raw evidence: `/tmp/f0-B6-baseline02-review-20261014-a/audit.json`; SHA `311bdad18ebf6705d84b8e62251f0543372c6ac329e70db69488aee69d6e3948`.

## 2. Requirement-by-requirement verification

| Requirement | Result and evidence |
|---|---|
| Exact scope and nonoverlap (§5.3) | **FAIL completeness**: eleven known runtime dependencies omitted. Scope roots exp/runs are nested, but selected regular payloads do not alias; root nesting alone is not a duplicate source claim. |
| Deterministic double inventories (§5.3) | PASS for declared scope: 2462 rows in each; sorted unique paths; both digest bindings recomputed; path/kind/bytes/key-stat projection equal. |
| Presence/absence and F140 identity (§5.3) | PASS: actual frozen snapshot fresh_ids match all 140 member identities; exactly 560 four-role paths; all absent in both inventories and independent current lexists check. Broken symlinks would not count as absent. |
| Archive bytes, readback and no alias (§5.3) | PASS for all existing payloads: 3530 hashes, 15942500076 bytes, zero unverified, zero digest/kind/nlink/inode issues. Regular and symlink-evidence copies are ordinary single-link files; archived inode differs from recorded source. |
| External manifest integrity (§5.4) | PASS: all 16 listed artifacts independently hash-match. This does not cure omitted scope. |
| Durable retained raw evidence (§5.4) | **FAIL one referenced script**: all 217 attempt4 manifest hashes have archived bytes; seven of eight tool-review nested-manifest references have archived bytes, one does not. |
| Incident and synthetic log honesty (§5.4) | PASS within recorded scope: raw worker0 log preserves nine CHILD_STARTED lines 130–138; synthetic peer-log freshening is explicitly disclosed as SYNTHETIC_FRESH_PEER_LOG. Attempt4 evidence is retained, not elevated into production continuity proof. |
| Quiescence and freshness (§5.3) | BOUNDED: four snapshots were taken 12.791–56.280 ms before collector start. They are point observations; rearchiving them in the second read does not establish continuous or second-live-window surveillance. |
| Protected preservation and no production lock/claim (§6/§10) | PASS bounded evidence: nine before/after listed hashes equal; empty diff; all 19 additional v7 protected current checks match. Collector/driver static control flow contains no production claim/release/flock or model work. Hash equality does not prove absence of all transient external lock activity. |
| Tool/policy identity and process cap (§5.3/§9) | Verified bound identities; raw command uses project environment and external scheduler. Quiescence observes one collector project Python; reviewer audit peak one. No full-window process monitor was supplied, so transient global counts are not certified. |
| Status never auto-authorizes (§5/§7) | PASS: baseline remains COLLECTED_UNREVIEWED, prospective RECORDED_UNREVIEWED, forward NOT_GRANTED, execution NO_GO; historical proof fields false. Review does not alter these artifacts. |

## 3. Bugs/deviations found

### BASE02-R1 — MAJOR: required scientific/runtime source omitted

DESIGN §5.3, line 54 explicitly includes current related scientific/runtime/control source and its inventory. The preflight driver `collect.py:114–141` selects experiment Python files, skips `history_exp` at line 117, and only adds a limited frozen control list. The exclusion at line 170 treats the entire history tree as outside scope. No scope root includes the sibling m3oe runtime dependency tree. A ban on data samples does not exclude required model source. The following exact dependencies from the v7 runtime inventory have no regular source-path inventory/archive entry:

| Runtime inventory key | Current SHA-256 recorded by independent audit |
|---|---|
| `../m3oe/MDMTRec/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `../m3oe/MDMTRec/basic/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `../m3oe/MDMTRec/basic/activation.py` | `152c5909e966055f83b7ec49de7634c853fbae8b029fbd37759371c14c921046` |
| `../m3oe/MDMTRec/basic/features.py` | `116025a45517021bfe86447b8229e59cf04feb665549544ce30173d04421a2b0` |
| `../m3oe/MDMTRec/basic/initializers.py` | `a9a6a2923eebeca7f541ffaddbccf2671a42e7f9b6a1ec8d38ba2872c53f6196` |
| `../m3oe/MDMTRec/basic/layers.py` | `e3eb8db91ff0aaf79c8e389107df5f64841992fab02a18b29c37bb446b22763a` |
| `../m3oe/MDMTRec/models/multi_domain/__init__.py` | `e11c58c743dd3125f0a85eeb89c0266706b4716b10700eb99ed1f18d19cb23e8` |
| `../m3oe/MDMTRec/models/multi_domain/model.py` | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |
| `../m3oe/MDMTRec/utils/__init__.py` | `13c62ae9d561769c4597c2f279bf03ab25363ad64ada1e77af58c4f7979cc7a3` |
| `../m3oe/MDMTRec/utils/data.py` | `71d08bd9487afc702473d8813449d21ffd85c223ed05457791486b6df708501a` |
| `history_exp/first/p7_models.py` | `7b727c4c933e93860eb9ce1dd57d58f4c40adbf3975d2d7ca6dd76afb8b6e839` |

The evidence JSON preserves absolute source paths and both reviewed/current hashes. Matching bytes incidentally appearing under another file, such as empty __init__ files, do not bind the omitted path identity. This is a collection-plan completeness defect, not evidence that existing copies were corrupted.

### BASE02-R2 — MAJOR: retained manifest references unarchived raw script

DESIGN §5.4, line 66 requires complete ordinary-copy retention and explicit failure on missing raw evidence. `collect.py:18–23,147–149` selects the tool-review `archive` directory, while that archive manifest also references a script outside it. The stored `review:REVIEW_RAW_MANIFEST.json` references:

- Path: `/tmp/f0-B6-policy-v3-review-20261014-a/probe.py`.
- SHA-256: `5b265e759aa35aa2328a0b6b0a45ffc3877d697fe975b9325699ed6da87d9a5d`.

No regular payload in either baseline inventory contains that digest. Its current /tmp availability, if any, is not durable retention in this baseline. Seven other nested references and all 217 attempt4 references have archived bytes. Fixing the missing reference and source scope requires a separately authorized successor collection/evidence revision; attempt02 must remain unchanged.

## 4. Corrections made

None to implementation, collection manifest, production files or previous reports. Only this new review report, its evidence JSON, and independent external audit outputs were created. Neither blocking omission was silently repaired.

## 5. Remaining risks and evidence limits

Collection window: `2026-10-13T21:10:19.423981Z` through `2026-10-13T21:18:25.406132Z` (approximately 486 seconds). Two equal inventories show stable observed projections over this bounded collection, not historical preservation or continuous process/lock surveillance. `/proc/locks` is a point snapshot; an observed external scheduler slot holder is not proof of a production deadlock. Static producer control flow and protected equality must not be presented as the established cause or absence of unrelated transient lock events.

Current protected-byte equality covers the listed nine plus nineteen paths. It cannot reconstruct the missing incident pre-edit hash, complete pre-incident inventory, or original scientific runner preimage. The preserved raw log confirms the incident rather than curing it. Synthetic fixture peer logs support only their recorded cases. No AUC/loss/model conclusion is drawn.

## 6. Whether experiments are safe to run

**NO-GO for release, staged graph GO and scientific RUN.** The actual baseline acceptance is FAIL until both MAJOR completeness issues are corrected and a new bounded artifact set is independently reviewed. No policy/tool PASS or raw hash success closes B6. Historical B6 remains OPEN/FAIL permanently disclosed; a future accepted baseline would mean only ACCEPTED_BOUNDED for its exact scope, with forward disposition separately reviewed. This report does not authorize recollection or any production action.
