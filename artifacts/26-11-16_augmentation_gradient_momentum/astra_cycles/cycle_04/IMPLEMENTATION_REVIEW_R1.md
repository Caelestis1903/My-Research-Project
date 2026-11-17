# REVIEW REPORT — Cycle04 CODE REVIEW_R1

**1. Verdict: PASS.** The three original MAJOR runtime blockers and the MINOR cost omission are corrected in the actual R1 implementation. Focused independent CPU checks pass. This clears parent preparation, the new foreground owner and the four registered real engineering cases. **Formal training still requires the separate actual engineering/resource signoff and parent acceptance/release of the four temporary keys.**

Reviewer: `01a1e6e3-bfaa-73a2-a96e-a16123dd9999`; original parent: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. All contributing coders excluded from this reviewer's identity: `01a1e54b-0376-79c0-b1a7-20a6893345c7`, `01a1e54b-03a0-7613-a537-dba4cc0f0e80`, `01a1e6cd-54c0-7a11-80ca-fb7194be1f15`. This is a fresh independent review; no descendants or detached sessions.

Source48: `00ce7925c3589deb239f57e8374673e5afe8e3fe155e159dd34c140ae08cd553`. Initial FAIL source48: `51e1e9e151858ffcc22fa10fb1c34a4f4921e8873436c70991d85c261f63e6cb`.
Seven-design: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`. Six-research: `a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a`.

**2. Requirement-by-requirement verification**

Read the complete459-line A4 design, DESIGN_ACCEPTANCE, ASTRA_HANDOFF, both combined coder reports, original core/runtime reports and CPU receipts, initial FAIL review, repair report/CPU receipt and PARENT_SOURCE_REVIEW_CANDIDATE_R1. Read all seven actual new E files. Traced their imports and consumed frozen model/loss/Adam/EMA/evaluator/RNG/copy/owner APIs; unchanged import-closure/core proof is reused only after exact hash comparison.

| SPEC | Requirement | Verification |
|---|---|---|
| 1–3, 8, 11 | Research intervention and causal scope | **PASS_STATIC** — Only constant W/S new arms; exact numeric C/N reuse; no new ranking update, normalization, arm, schedule, architecture or efficacy decision. Fifteen registered comparisons and primary selected W/EMA minus C/EMA remain fixed. |
| 4 | Frozen protocol and baselines | **PASS_STATIC** — Natural data, fixed1 M3oE, detached gate, source Adam with coupled unscaled WD and exact source EMA; five seeds2022–2026,50 epochs,batch40960,FP32. All44 unchanged source files match initial candidate; external Adam unchanged. |
| 5.1–5.2 | Loss and coefficient formulas | **PASS_STATIC_AND_REUSED_EXACT_EVIDENCE** — Inspected direct ordered .75C+.25sum(w*pi^2*class_BCE), S=mC, m=1-.25sum(w*2pi0pi1). Actual mass remains m in [.875,1]. No rank gradient in live W/S. Source N used only for diagnostic reference/checks. |
| 5.2–5.3 | Shapes, masks, null statistics and row identities | **PASS_STATIC_AND_REUSED_EXACT_EVIDENCE** — [F,B] inputs/[B,2] labels, six cells, absent versus empty support, detached coefficients, W versus S task allocation, structural null reasons and exact full/tail source IDs preserved. |
| 5.3 | Contexts, counters and private steps | **PASS_STATIC_AND_CPU** — Formal and engineering clocks are separate. Saved counts exactly optimizer_updates/ema_updates; operations and every parameter's None/tensor-zero/private-step status checked separately. |
| 5.4 | Gradient paths and real-Adam diagnostics | **PASS_STATIC_AND_REUSED_EXACT_EVIDENCE** — Three extra autograd sweeps and four sequential disposable source Adam clones from actual prestate; None preserved; live arm virtual/live hashes exact; full-coordinate FP64 vector norms/cosines and zero_norm nulls. |
| 5.4–5.5 | Source delegates and exact observational neutrality | **PASS_STATIC_AND_REUSED_EXACT_EVIDENCE** — lambda0/all-single use actual source_clean_step. Scoped forward hook removes itself before deepcopy, clone optimizers have no live hook, finally cleanup retained. Exact replay/state/RNG/loader contracts remain distinct from registered FP32/FP64 algebra tolerances. |
| 6.1–6.3 | Write roots, import closure and immutable provenance | **PASS_INDEPENDENT_HASH_CHECK** — Read all seven package files and inspected frozen imports/consumed APIs. Compiled all six new Python files in memory. Independently reproduced source48 and seven/six design digests; only protocol/runner/tables/README changed;9 parent-bound report receipts verified. |
| 7.1 | Test-opaque cache loader | **PASS_STATIC_AND_CPU** — Only train/validation arrays, IDs and feature schema are projected. No source_cache, Test traversal or reconstruction. Opaque fixture passed; real cache/raw verification deferred to preparation. |
| 7.1 | Historical numeric controls and selection | **PASS_STATIC_AND_CPU** — Actual CLEAN and NQR result/E1 bytes parsed through exact index; missing CLEAN RNG/count/continuation fields stay unavailable. Preparation verifies1020 files; W/S must match source initialization and500 order identities without C3 updated-state prefix equality. |
| 7.2, 7.7 | Frozen jobs and scientific budgets | **PASS_STATIC_AND_CPU** — 20 new jobs,20 reused controls,1000 epochs,34500 live forwards/backwards/Adam/EMA;80 diagnostic points/240 extra sweeps/320 virtual steps. Engineering stays scientific_epochs0 and never complete50. |
| 7.3 | Snapshot isolation, restore and independent strict selection | **PASS_STATIC_AND_CPU** — Immutable CPU copies, isolated populated Adam restore, RNG restored last, loader continuation; scientific identity excludes physical timing/diagnostic work while full continuation binds actual payload. Earliest strict online/EMA maximum retains ties and associated LL/hash. |
| 7.4 | Frozen owner API and chronology | **PASS_STATIC_AND_CPU** — Compared actual serve/register/recover/commit/get/complete API and two-key counts. ACK/status/get/full hash/journal/numeric acceptance consume actual fields; private guardian/local owner material excluded from public receipts; canonical N and protected old-owner exclusion retained. |
| 7.4, 7.6 | Uncommitted replay and lost ACK | **PASS_R1_FIXED** — Attempt-scoped transactions, canonical transaction-bound ACK and exact journal reconciliation verified through real entrypoints using in-memory owner/files. |
| 7.5–7.6 | Engineering recovery before signoff | **PASS_R1_FIXED** — Fixed key and scoped fresh review; prefix restart and precommit-proof-backed ACK2 finalization executed on CPU with real core/Adam/evaluator. Actual12-call recovery and zero-call finalization preserve physical costs and proof hashes. |
| 7.5–7.6 | Full-process profiling and admission | **PASS_R1_FIXED_STATIC_AND_CPU** — Main-thread context/allocator ordering corrected; staged telemetry/envelope fixture passed. Current85/85/host90, usable100 fit, first fitting KR-first queue, independent cards, pending first-FULL ACK/resample and no fixed cap/forecast remain. |
| 7.5, 10.2 | Four real cases and independent resource signoff | **PENDING_RUN_GATE** — Code fixes the four assignments and full/tail lazy/populated diagnostic/replay plan. No actual case/profile/owner evidence exists yet; CPU telemetry does not satisfy this gate. |
| 7.6 | Supervision, stop, recovery and Test seal | **PASS_STATIC_AND_CONTROL_READ** — Both workspaces ACTIVE/manual_stop=false, root/artifact sentinels absent. Retained mechanical subprocess handles; no AI/driver lifecycle. USER_STOPPED blocks admission/recovery; partial epochs uncommitted. No Test-opening route introduced. |
| 7.7 | Formal selected-RAM retention and parent acceptance | **PASS_STATIC_PENDING_RUN** — All20 selected online/EMA pairs retained;1000 ACKs and actual191017920 tensor bytes checked at handoff. Parent direct get/hash/validation acceptance required before closing formal runner. Only accepted temporary engineering keys can be released. |
| 9.1–9.3 | Numeric extraction, seed statistics and physical costs | **PASS_STATIC_AND_CPU** — ACK-bound scientific epochs; source six-cell/pooled metric names; selected/E1/E10/E30/E50 and50 curves; actual tail row weights; scheduled count denominators; n5 paired sample SD/registered interval; negative LL favorable. R4 diagnostic/restore/serialization records independently exercised. |
| 10.1 | Finite implementation evidence | **PASS_FOCUSED_CPU** — Fresh execution protocol6/runner10/tables5 groups PASS under no-write/no-IPC/process audit. Exact core54+10/112-clone/four-M3oE evidence reused by code/report hash; broad core/old QR suites not rerun. |
| 10.2–10.3 | Stage gates and research authority | **PASS_CODE_ONLY** — Fresh code reviewer excludes all3 contributing coders. Parent may proceed to preparation/new foreground owner/four real cases. Formal RUN still requires separate fresh resource signoff and parent-accepted four-key release. Analysis/publication/Astra and uncapped outer loop remain parent responsibilities. |

The production objective is directly `W=.75C+.25sum_c(w_c sum_a pi_ca² class_BCE_ca)`; `S=mC`, `m=1-.25sum_c(w_c2pi_c0pi_c1)`. Coefficients/counts are detached, actual BCE mass is m, and source coupled decay is unscaled. None/tensor-zero/private-step distinctions, ordinary detached gates, scoped source identity hooks, sequential real Adam clones, exact actual/virtual replay and source EMA are preserved. No research intervention or baseline was changed by R1.

**3. Bugs/deviations found**

No new implementation issue or remaining blocker was identified. Original issue classifications are retained below with their resolutions.

**C04-R1 — MAJOR — RESOLVED.** `runner.py:218,241`, `protocol.py:546`: transactions now live at `transactions/<key>/attemptN/epoch_XX.json`. The canonical ACK binds the successful physical transaction file/hash. Actual journal attempt/path selects reconciliation; full continuation is still checked against the exact returned payload. Both formal and engineering failed-before-commit -> reviewed recovery -> next-attempt replay fixtures passed. Old failed transaction bytes remain, canonical epoch extraction selects the successful attempt, and already-committed/local-ACK-lost reconciliation adds no commit. A changed full payload is rejected.

**C04-R2 — MAJOR — RESOLVED.** `runner.py:409,446,481,752,765,816`, `protocol.py:496,508`: `smoke --recover --case-id --recovery-review` binds one existing fixed engineering key, source/design/pause/owner and last ACK before signoff exists. A retained prefix restores its saved permutation/model/EMA/populated Adam/loader/RNG and performs genuine remaining source/lambda0 and neutral suffix proofs. Suffix ACK2 can finalize with no training only after its precommit hashed eight-call proof verifies diagnostics, fork/replay observations, selection/full validation and prefix ACK. Changed proof/source/design/pause/owner is rejected. Four keys, scientific_epochs0, no complete50 and accepted-release gates remain.

Fresh actual-entrypoint CPU outcomes:

| Failure fixture | Returned physical calls | Diagnostics | Extra sweeps | Virtual Adam | Recovery added calls | Restore records | Retained profiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| After call6 with prefix ACK1 retained | 12 | 3 | 9 | 12 | 6 | 8 | 2 |
| After suffix ACK2 before case receipt | 8 | 2 | 6 | 8 | 0 | 4 | 1 |

Each verified plan stays8/2/6/8; actual work is retained separately. The resource consumer verifies real physical call logs and summed case costs rather than rewriting replayed work to8. Original profiles are retained with hashes and combined measured maxima.

**C04-R3 — MAJOR — RESOLVED.** `runner.py:68,75,87,155,506,662`: monitoring begins before context; the main worker initializes CUDA and enables allocator statistics before model/EMA/Adam allocations. The sampler does not initialize CUDA. The actual sampler fixture measures own900/reserved500/non-torch400/envelope900; EMA own1100/reserved700 keeps non-torch400. The former startup double count is removed. Current85 memory/current85 power/host90, usable100 physical fit, pending full-allocation ACK/resample, first-fitting queue and per-card independence remain unchanged. These are CPU telemetry fixtures; real four-case CUDA profiles remain required.

**C04-R4 — MINOR — RESOLVED.** `runner.py:163,508,668`, `tables.py:225,239,317`: restore elapsed time uses CUDA completion barriers with RNG restored last and is logged with actual attempt/role before subsequent failure. Diagnostic_seconds is extracted from returned physical batches, including failed/replayed work. Every retained physical transaction contributes its measured serialization cost. Incomplete/unreturned work uses null; durations are not inferred by subtracting unrelated totals. Logical epoch/transfer summaries and overlapping physical components remain explicitly separate.

**4. Corrections made**

No source corrections were made by this reviewer. The fresh repair coder's corrections above were inspected and verified. Only these two review files are reviewer writes.

All commands used W and `PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`. All six Python files compiled from source text in memory. Independently recomputed source48, seven-design and six-research hashes; compared the actual candidate against the parent manifest and verified nine report receipts. Exactly44 source files and installed Adam are unchanged; the only changed closure members are protocol/runner/tables/README.

Executed the focused `protocol.finite_checks()`, `runner.finite_checks()`, `tables.finite_checks()`: **6/10/5 groups PASS**, including actual commit/recover/epoch consumers, real-core CPU engineering recovery, cost extraction and staged telemetry. Complete command/results/input hashes are in the companion JSON. The harness audited against disk writes, real IPC and subprocesses. Its first attempt stopped before runtime fixtures because Adam's lazy installed PyTorch import tried to create a temporary distributed template. The final harness generated that normal unused module in RAM, restored stdlib IO hooks, and passed with the audit enabled; numerical helpers/Adam were unchanged.

Reused exact core/checks evidence:54 core calls +10 direct source calls,112 cloned source Adam steps and4 actual-M3oE CPU comparisons, with code hashes `790ae97d01a3ab3d39e42712580e333aa017b96d5869f36aed682181b03de683` / `7bbf10c3e4249d9a3af7e1ad80b7c93267a66f99ddc52e3050c709ba99605ae1`. The broad core suite, old QR/smoke suites and historical source-tree audit were not repeated.

**5. Remaining risks**

N is absent. Preparation must verify real raw/cache train-validation bytes and all1020 historical numeric-control files. Actual new-owner IPC, four CUDA cases, full-process/context/diagnostic residency, compatible hardware envelopes and fresh resource approval remain RUN-stage gates. CPU telemetry does not supply that evidence.

Formal execution must verify source initialization/order identities,1000 whole-epoch ACKs, registered logical/physical costs and all20 retained selected pairs. Parent direct get/hash/chosen-validation acceptance is required before closing formal RUN. Both workspaces were ACTIVE/manual-stop=false with no root/artifact stop sentinels at the review read. Test remains sealed.

**6. Whether experiments are safe to run**

**Yes for parent preparation/new foreground owner/the four registered real engineering cases after their existing preparation and stop checks. Formal training: not yet authorized.** First obtain actual four-case engineering/resource PASS from a separate fresh reviewer, then parent-accepted release of exactly those four temporary keys with actual owner release flags, and retain all measured admission gates. This code review does not replace any of those gates.

No GPU query, training launch, real owner/old-owner IPC, real cache/Test access, tensor file, source/design/context/rootstate/Git edit or child dispatch occurred. This reviewer returns control permanently to the original parent; do not resume or reuse this context.

