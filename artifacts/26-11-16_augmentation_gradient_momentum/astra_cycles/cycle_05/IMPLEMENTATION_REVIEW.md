# REVIEW REPORT

## 1. Verdict: FAIL

One **MAJOR** blocking issue, C05-R1. Experiments are **not safe to proceed to engineering** with this source. Return to coder before preparation, source freeze or owner startup.

Reviewer: **Confucius / 01a1e9eb-54e3-7211-8fe1-2bb686e0183e**, gpt-6.1-sol/high, fork_context=false. Parent: 01a1dfb8-605b-7430-8462-5fd3da0b048f. Distinct from Lorentz and both coders Laplace/Lovelace. No descendants; permanent return after handoff.

Actual candidate source: `3f32ce34d76d2ab79a3b8fcb447722862310f7cecb6d037b326626b03e9d7e74`, **55 files**, exactly immutable C04 source48 plus seven new files. All55 paths/hashes and installed Adam identity are recorded in IMPLEMENTATION_REVIEW.json. Source48 receipt hash `10f3000ae43e2a394b71cdb2febc291888f06bf216dd99c05ddd7b842186fec3` and source identity `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f` match. Complete specification byte hash `a1f61ae81535bb33873da72185d9400187c5b988e81c0d3a5019c5505e821af9`; design7 `9a7e826794f4cdc3fcc7c437e1462149a1dc9372b5d5e3a38981f69fb3af299c`; research6 `bfd6230c9cae8971a3a8d793027fc54ba6209d193429340107bf64bb22dc4caf` match.

## 2. Requirement-by-requirement verification

- **§1-4 — PASS_STATIC**: Research question, sole intervention and held-constant matrix. U only; ML/KR seeds2022..2026,50 epochs,40960 FP32; exactly30 C/W/S controls; no efficacy stopping or extra arm.

- **§5 — PASS**: U mathematics, support, reductions and null semantics. core.py:59-84 uses frozen C/H/W/S with diagnose=False and d-major/t-minor (w*a)*cell.mean; exact shared W/U cell masses; frozen m order; scheduled detached statistics; mixed-label lambda0 H=null/not_computed_delegate.

- **§5.1 — PASS**: Source CLEAN delegates, Adam/EMA, shapes, gradients and gates. core.py:269-303 delegates whole nonordinary call; ordinary one forward/backward/source Adam/EMA. Source [F,B] to [B,2], detached gate, None-versus-zero behavior and private counters verified in actual pinned functions and hash-bound core fixtures.

- **§5.2 — PASS**: Common-state diagnostics and scientific neutrality. C/S/U/W; exactly3 extra sweeps and4 sequential real source-Adam clones; no clone forward; exact actual U virtual post hashes/update zero; live/prestate/RNG checks; final core fixtures27 calls/14points/56virtual steps reused.

- **§6-6.2 — PASS**: Immutable source/design closure and separate native identities. Actual55 hash equals runtime final report; exact immutable source48 union seven new files; both coder JSON/MD reports read; designer/coders/reviewer distinct; external Adam pinned.

- **§7.1 — PASS_STATIC**: Test-opaque cached projection and numeric reuse. Explicit train/validation/features attribute projection; no Test dereference; manifest control pointers1530; C03 initialization/order binding; source control selection preserved. No preparation or data unpickling performed in review.

- **§7.2 — PASS_STATIC**: Physical/logical/staging roots and RAM-only state. Canonical R5/N5/S5 writers, containment/symlink checks, explicit mapping; Tensor/array/bytes rejected from JSON; snapshot/EMA/Adam/RNG/loader/BytesIO remain RAM.

- **§7.3-7.4 — PASS_STATIC**: Launch gates, owner/ACK/recovery and selection. Real generic-server methods/responses inspected; byte-bound native reviews consumed; epoch transactions preserve ACK chronology/lost-reply reconciliation; strict independent online/EMA AUC selection with unscaled same-state LL; same-owner recovery and no formal release.

- **§7.5 — PASS_STATIC_WITH_REUSED_CPU_EVIDENCE**: Named telemetry, physical-fit admission, terminal facts and witnesses. Named NVMLv2 ABI/three fresh whole attempts; .005s raw logs; current85/85/90 plus measured physical100 fit per card; PID/job/attempt/RUNNING/FULL pending receipt; terminal core before independent cleanup; append-only immutable raw/prefix/envelope refs. Existing bound runtime fixtures cover injected failures and boundaries.

- **§7.6 — PASS_STATIC_DOWNSTREAM_EVIDENCE_REQUIRED**: Four engineering cases and consumed resource/parent release gates. Eight-call plan/case, source lambda0 fork, immutable prefix, restored suffix, exact diagnostic comparisons and full validation; resource/native/owner/review/witness/ACK/release fields consumed. Actual four GPU cases remain RUN-stage work.

- **§7.7 — FAIL**: Complete numeric archive, mapping/readback/capacity and publication readiness. C05-R1: required pinned design and actual engineering hash observations rejected before archive creation. Explicit mapping, capacity/readback/parts code otherwise inspected.

- **§8 — PASS_REUSED_FINITE_EVIDENCE_WITH_INTEGRATION_GAP**: Minimal CPU fixtures. Reused hash-bound core and runtime passed results without broad rerun. Reviewer added read-only archive-input reproductions because coder archive fixture did not cover actual required document schemas.

- **§9 — PASS_STATIC**: Useful/physical budget, raw normalized exports and paired arithmetic. 10 runs/500 ACK/17250 live steps,40/120/160 diagnostics. Normalized point/parameter joins;400/2400 endpoints/cells,4000/24000 curves,500phase/103500weights;17x2x5x34x5=28900 paired and5780 summaries; ddof1/t4 and no epoch favorable direction.

- **§10-11 — BLOCKED_BY_C05-R1**: Stage acceptance and termination. No prepare/owner/engineering until reviewed source has no blockers. Parent retains analysis/publication and C05 delivery then USER_STOPPED; no C06/Test path authorized.

Both AGENTS files, W/start.md decoded GB18030, W/buchong.md, current runbook, full accepted specification, all seven new files, both coder reports in MD/JSON, necessary pinned numerical/storage APIs and installed Adam identity were inspected. Both workspace stop controls are ACTIVE/manual_stop_requested=false/termination_reason=null, stop_after_current_cycle=5/cycle06_authorized=false; all four sentinels absent.

## 3. Bugs/deviations found

**C05-R1 — MAJOR / blocking: the numeric archive guard rejects mandatory provenance.**

`tables.py:363,378` rejects any dictionary with a key in `{selected_models,model,ema,optimizer,rng,loader}`, regardless of the value. `checked_member` applies it to JSON/JSONL at lines392–394. `archive_members` requires both the raw engineering logs at line413 and all seven design artifacts at line424.

Two concrete required producers conflict with that guard:

- Accepted `research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/EXPERIMENT_MANIFEST.json:166` contains `"model": "source M3oE fixed1"`. Calling the real `checked_member` on this unchanged, byte-bound design input raises `TechnicalPause: Opaque training state field in public record`.
- `runner.py:439–443` returns model/EMA/Adam/RNG/loader **hash strings** under those names. Lines756–764 normalize activity and write that observation in every engineering call log. The real `runner.observation` on a CPU fixture produces five 64-character strings; passing its production-shaped normalized observation to `numeric_document` raises the same error. No forward/backward/Adam/training call was used in this reproduction.

Consequently pack cannot produce the required final archive even if all training completes. Fixing source only after its run freeze would change the reviewed55 identity. Coder’s archive fixture covers a generic `{count,test_accessed}` record and rejected actual-state examples, so its PASS does not cover these required production inputs.

**Minimal correction:** distinguish allowed hash/text provenance from actual state payloads in the archive validator. Preserve credential/tensor/array/bytes/opaque-state rejection. Accept the pinned manifest and production engineering hash observations; retain both in the archive and leave immutable design unchanged. Add one narrow CPU fixture for these two accepted inputs plus existing actual-state/credential rejection. Return corrected source/report identities through fresh independent review before preparation.

## 4. Corrections made

None. Review scope prohibits editing source/specification/coder inputs. Authored only IMPLEMENTATION_REVIEW.md/.json and IMPLEMENTATION_REVIEW_CHECKS.json. C05-R1 remains open; no PASS WITH FIXES claim.

## 5. Remaining risks

Actual four-case GPU engineering, eight ACK/restore proofs, independent resource signoff and parent release remain downstream gates. Numerical core and runtime finite results were reused after exact source/report hash binding; no broad duplicate suite was run. The reviewer’s added evidence is limited to RAM syntax compilation, identity/stop reads and the concrete read-only archive-input reproductions.

## 6. Whether experiments are safe to run

**No — experiments_safe_to_proceed_to_engineering=false.** Real `protocol.check_review(IMPLEMENTATION_REVIEW.json,'code',source_hash)` rejects this FAIL with `TechnicalPause: Review has blockers`, so the receipt cannot release preparation. Return C05-R1 to coder; a fresh distinct reviewer must verify the corrected candidate. This reviewer returns permanently.

Skipped data unpickling/preparation, source freeze, owner IPC/server startup, GPU queries/training and Test; zero reviewer training operations. The remaining concrete blocker is required archive completion.
