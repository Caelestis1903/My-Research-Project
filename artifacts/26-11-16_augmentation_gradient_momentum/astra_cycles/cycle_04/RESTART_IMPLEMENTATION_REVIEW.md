# REVIEW REPORT

1. **Verdict: PASS.** Zero CRITICAL, MAJOR or MINOR implementation findings; zero blockers. No source corrections.

Reviewer: `01a1e753-1214-7ec0-8183-5dc262103865`, distinct from all four contributing coders listed in the JSON. Read-only restart implementation review, under parent `01a1dfb8-605b-7430-8462-5fd3da0b048f`.

Actual source48: `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`.
Unchanged scientific design: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`.

2. **Requirement-by-requirement verification**

| Requirement | Verification |
| --- | --- |
| Restart §7.1; §10: Historical source preservation | PASS — 48 snapshot bytes match the old source receipt and RAM-loss mapping; old receipt/hash recipe verified. Exactly the authorized three live files differ; 45 match. Review performed no attempt1 writes. |
| Restart §7.2: Canonical attempt2 and fresh jobs | PASS — Only ATTEMPT plus two canonical-write diagnostic strings changed in protocol.py. Existing source_paths/digest/config/job/prepare recipes remain unchanged. Independently generated 20 unique keys disjoint from all 24 historical formal/engineering keys; formal scientific job fields match exactly except source_hash/key. |
| Restart §5; §7.3: Pending launch generation | PASS — Before Popen, captures prior owner attempts+1. Immediately after launch, records key/PID/expected_attempt and active exact job. Release requires live matching active child, exact job, PID, attempt, RUNNING and allocated is True. Fresh absent keys expect registration1, not experiment attempt2. |
| Restart §5; §7.3: Registration and actual allocation ACK | PASS — Frozen serve register resets allocated=False and increments attempts; worker registers with os.getpid(). Actual first train_step, lazy Adam/diagnostics, and profile measurements precede allocation_ack. ACK now records actual worker_pid; no server protocol change. |
| Restart §5; §9: Release, telemetry, audit and exits | PASS — Qualified release samples current NVML and records expected/observed PID/attempt, source/design and device before dropping pending. Admission samples again before next launch. Exit cleanup matches both key/PID; nonzero exit sets PAUSE without ACK-release evidence. |
| Restart §5; §7.3: Admission semantics | PASS — Unchanged admission uses per-card current memory<85%, power<85%, host RAM<90%, pending exclusion, and current used+remaining measured full-process residency<=CUDA-usable100%. First-fitting queue, dataset/arm/hardware envelope, resampling and independent device loop retained. No cap/projected85/predicted power/both-card requirement added. |
| Restart §4–5; original §5.1–5.2: Mathematics and weighting | PASS — Byte-identical core inspected: direct ordered W=(1-lambda)C+lambda A, A=sum(w*pi²*class_BCE), S=mC, m=1-sum(w*lambda*2*pi0*pi1). Detached support weights; actual applied mass m, no renormalization or live ranking gradient. |
| Restart §4–5; original §5.2–5.3: Shapes, support, masks and routing | PASS — Byte-identical validation for source [F,B], FP32 [B,2], six domain/task cells, full40960/tails21927/40052, missing/absent-class nulls and source identity delegates. Matched-mass S and heterogeneous W task/cell semantics preserved. |
| Restart §4–5; original §5.4–5.5: Gradient paths and observational diagnostics | PASS — Byte-identical train_step inspected: only selected W/S loss backward updates live parameters; N only diagnostic reference. Three extra gradients, four sequential isolated actual source Adam clones, None versus zero gradients, exact virtual/live checks and one source EMA remain intact. Reuse exact unchanged core/checks evidence by verified hashes. |
| Restart §4; §7.4–5: Freeze, controls and reviewer separation | PASS — Actual source48 and external installed Adam verified. Seven-file design, six-file research bundle and control index retain exact hashes; index enumerates20 results+1000 epochs. Four genuine coder report identities preserved; check_review unchanged and this reviewer differs from every coder. |
| Restart §7.6: Fresh foreground owner boundary | PASS — README specifies parent prepare then direct foreground serve(attempt2), private bindings and new source/design/server identity; no detached helper invocation was added. Frozen owner/IPC/snapshot machinery unchanged. Actual new-owner availability/binding remains the parent's subsequent gate. |
| Restart §7.7–8: Fresh engineering and resource gates | PASS — CASES and source-bound engineering validation unchanged: four prescribed GPU/dataset/arm cases, separate zero-scientific-epoch clocks, exact replay/neutrality/counter checks and measured profiles. Formal run still requires distinct signoff, parent acceptance and actual release flags. No GPU evidence was represented as already obtained. |
| Restart §7.9; §4: Full formal matrix and baselines | PASS — Unchanged 2 datasets ×2 arms ×5 seeds2022–2026 ×50 epochs,1000 useful ACK epochs/34500 Adam+EMA updates. Fresh root/jobs start without inherited ACK credit. Source initialization/order, deep restore/RNG-last, selection, commit/recovery/export and existing numeric-only CLEAN/NQR controls remain unchanged. |
| Restart finite CPU acceptance: Targeted verification | PASS — Reviewed coder's exact diffs and bounded6/11/5-group outputs. Independently compiled edited Python in memory, verified identities, and executed unmodified production run/admission/measured_envelope/_supervisor_exit AST bodies with mocked external boundaries: all eight cases plus eight admission assertions passed. No broad core or bounded-suite rerun. |
| Restart §6; §9; parent handoff: Reporting and termination scope | PASS — README migrates current commands to attempt2 and restart review, preserves attempt1 costs as history with zero useful-budget credit, and requires full current report/review/context/publication then USER_STOPPED; no C05/Astra. Parent report tools are outside coder scope and were not modified by this review. |

3. **Bugs/deviations found**

None in the candidate implementation. The historical MAJOR stale-generation release bug is corrected: an old allocated=true cannot satisfy the launched PID, exact registration attempt, exact job, RUNNING and live-child conditions. No unrequested algorithm, weighting, resource threshold or baseline change was found.

4. **Corrections made**

None; source edits were prohibited. The only reviewer fixture correction was to filter the24 historical job files by purpose=formal before asserting20 formal jobs; the other four files are engineering jobs. The corrected identity check passed. No repository defect was inferred from that harness error.

Independent validation used the explicit project Python with PYTHONDONTWRITEBYTECODE=1 and stdlib-only stdin scripts. It compiled the two edited Python files in memory, verified all48 source paths/digests and preserved snapshot bytes, unchanged45 files, external Adam, accepted design/addendum hashes, contributor/core evidence hashes, unchanged control index and20 fresh actual-recipe jobs. No bytecode/temp file was generated.

A single deterministic in-memory transcript executed the unmodified production run(), admission(), measured_envelope() and _supervisor_exit() AST bodies with mocked storage, telemetry, processes and file boundaries. All eight required cases passed: absent record; old PID+attempt allocated=true; correct PID/wrong attempt; wrong PID/correct attempt; matching generation allocated=false; qualified release once with fresh resampling before next same-card launch; independent GPU1 admission while GPU0 pending; nonzero child exit9 causing PAUSE without allocation release. Recovery registration6 and fresh registrations1 were distinguished from experiment attempt2. Eight strict admission/physical-fit assertions passed.

The coder's source-bound bounded protocol/runner/tables results (6/11/5 groups) were inspected and reused. Unchanged core/checks numerical evidence was reused by independently verified hashes; math, shapes, support, selected-loss gradients and source Adam/EMA paths were inspected directly. No broad suite or duplicate core run was performed.

5. **Remaining risks**

Current GPU availability, foreground-owner state and actual hardware engineering are subsequent gates; this review did not measure them. Parent must prepare the new root, verify a genuinely fresh foreground owner and actual runner metadata, complete all four fresh-source engineering cases, obtain a distinct resource review, and perform actual acceptance/four-key release before formal training. Existing prepare still verifies raw/train/validation provenance and all1020 numeric control files.

Attempt1's lost model/Adam/RNG states remain unavailable. Its275 ACK epochs/7733 useful updates are historical only and cannot resume or count toward attempt2. The review establishes preserved source/index/job identities, not present historical tensor availability.

6. **Whether experiments are safe to run**

**Code gate passed.** Proceed to parent preparation and the required supervised engineering stage after owner/metadata checks. **Formal RUN is not yet released**; it remains conditional on fresh actual engineering, resource signoff and parent release. Complete only Cycle04, then report/review/context/publication and USER_STOPPED; no C05/Astra.

Skipped GPU queries/training, owner IPC, dataset-cache/Test access, broad tests, Git and child agents. Only these two new review reports were written; old reviews and attempt1 were left untouched.

