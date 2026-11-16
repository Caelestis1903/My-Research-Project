# Astra self-critique and completeness audit
This is design self-critique, **not** the distinct implementation review required before RUN.

## Chosen experiment and strongest objections
The selected A matrix changes the odd/even structure of NDP-A's objective, while B and C remain distinct unselected alternatives. A's strongest objection is that deleting the first-order effect may simply return nearly to CLEAN. The predeclared CLEAN comparison prevents calling such recovery a useful augmentation result. The isotropic control asks whether the remaining axis has value; it does not match curvature or Adam-update magnitude.

Second, old ANTI2 already weakens an argument based only on reduced estimator variance. AXIS-IID2 is required: identical conditional expected objective, same forward count, independent signs. If AXIS-EVEN and IID2 are indistinguishable, the experiment supplies no endpoint evidence for antithetic pairing even if their local estimator variances differ. No local variance diagnostic will be promoted to a research success criterion.

Third, the task direction becomes largely label-sign independent after symmetrization. Claims about retaining supervised ascent direction would misdescribe the method. Its surviving information is task-logit geometry; targets remain in BCE. This is why a task-axis claim must clear ISOTROPIC-EVEN.

Fourth, the exact identity at a common model does not isolate causal contributions after 50 epochs of nonlinear feedback. A versus NDP-A is a controlled intervention on the objective, not an additive attribution of final AUC. Two additional forward graphs cost real compute. Report worker-seconds and efficiency; do not call equal Adam counts compute equality versus historical NDP-A.

Fifth, the probe reattaches gate inputs whereas the training graph preserves the original detached gates. The inherited probe is an information source, not the exact training-gradient Hessian. The Taylor interpretation is conditional and local. No architecture or detach behavior is “fixed” during implementation.

Sixth, .03 and .25 are inherited experimental constants, not newly optimized settings. Failure of this matrix does not falsify every possible even-view method; it does falsify or weaken its registered fixed-budget predictions. Do not respond with hidden alpha/lambda/gamma search or an extra confirmation seed.

## Evidence audit
- STRONG engineering evidence: complete TV-GRM trajectories and all gate/support/cap/SHUFFLE checks passed; historical exact controls were preserved.
- MEDIUM research evidence: no tested TV-GRM candidate yielded a selected-EMA positive mean on both datasets; KR recovery from NDP-A is consistent and CI-positive.
- WEAK mechanism evidence: residual history, directional benefit beyond SHUFFLE, general domain-size causality, and any present efficacy claim for the new candidates.
- Contradiction retained: ML .20 is +.000153992 AUC (3/5, CI crosses zero) and improves LL in all five seeds. Therefore the all-gamma/all-dataset-nonpositive gate is FALSE.
- Size explanation fails a simple pattern check: KR's two smaller domains move oppositely; ML's smallest and largest can improve together. Populations are descriptive. The baseline AUC/LL table supplies performance context without treating size as difficulty.
- Macro/traffic, AUC/LL, online/EMA and E50/selected disagreements are separate outcomes; none substitutes for the primary selected fixed-EMA Macro AUC.
- Existing CLEAN fixed EMA is an internal source control, not a newly declared published-model baseline.

## Runtime storage finding and bounded resolution
The existing TV guardian's live server implements register/commit/complete/get/release. It has no new-root argument: commit appends to its existing epoch_commit_ledger.jsonl, and complete writes a result under its existing runs/<newkey>. The source also has an ensure_guardian launcher; that launcher and the old scheduler/driver must **never** be called.

A narrow client can use the existing live storage service with distinct cycle-qualified job keys. Before any such writes, the runner records the live journal's byte length and SHA-256 and the existing key set in the new cycle artifacts. All old records and their journal prefix must remain byte-identical; published source, frozen raw archive, old tables and reports remain unchanged. New cycle epoch files/results are canonical in the new cycle directory, with absolute epoch paths sent to the shared guardian; its automatic new-key result is only a storage receipt mirror. Append-only live storage growth is explicitly distinguished from the archived historical evidence identity. This requires no replacement guardian or background AI.

At completion the guardian's existing complete operation retains only selected online/EMA models. Parent marks discarded cycle keys and releases their RAM after the complete evidence handoff; release only those keys, never old retained states. Keep at most one explicitly retained incumbent five-seed set plus the current cycle after each cycle's evidence decision, unless the existing eligible-final-verification gate requires more. Any pre-existing protected states remain protected. Resource admission includes all retained states and transient snapshot/IPC copies. Pending tasks wait if RAM headroom cannot admit them; there is no process-count cap.

Hard execution blocker if the guardian is absent, incompatible, cannot distinguish new keys, or cannot preserve the old prefix/records. PAUSE; do not restart or replace it. This child did not connect to or alter guardian state, so liveness and capacity remain runner preflight gates.

## Completeness status and blockers
The algorithm equations, branch order, RNG, source controls, parameter activity, precision, seed matrix, selection, statistics and next-stage responsibilities are specified in CODEX_EXECUTION_PLAN.md. There is no unresolved numerical hyperparameter or additional experiment hidden behind this handoff.

Implementation and its source hash do not yet exist; independent reviewer approval, actual identity checks, live guardian admission and GPU smoke are pending. They are mandatory RUN gates, not claims already satisfied by this research document. A missing exact-control file, changed preprocessing/model/objective/optimizer identity or failed equality check blocks reuse and the frozen 30-run matrix. Parent must issue a revised design for source-matched controls, then fresh implementation/review; no automatic budget expansion.

Precise DT-PGNA/TGA historical equations were not located in the provided local research-plan/report material. That blocks a comprehensive historical novelty claim. It does not change the supplied equations; if those missing records duplicate A, parent must reject redundancy before implementing. No scientific efficacy claim is ready for publication.

## Failure and stopping rules
Bad loss/gradient, identity mismatch, resource violation, incomplete raw records, missing RAM state or failed correctness check -> technical PAUSE, preserve recoverable RAM, then repair through fresh coder/reviewer where source changes.
AUC NO-GO, no continuation signal, contradictory endpoints, or Astra STOP -> write honest evidence/context and request fresh research from the original parent, with Test sealed. These are candidate outcomes, not normal termination of the research line.
Explicit user stop -> safely finish the current atomic epoch, validation and acknowledged RAM commit when technically possible; stop dispatch and summarize. Only manual stop plus final freeze and the existing eligible five-seed selected-RAM verification gate can authorize Test. This cycle contains no Test implementation or automatic Test action.
Outer maxima remain null; the finite 30-trajectory experiment matrix is a scientific design, not a whole-line cap.

## Review instructions
A distinct fresh reviewer must inspect these six files, actual implemented source and coder report, including transitive frozen dependencies and the guardian client call graph. Reviewer must distinguish exact identities from numerical tolerances, verify no Test path/driver/guardian startup call, and verify that changes do not reach historical source or root state. PASS or PASS WITH FIXES is sufficient only with zero remaining blocking issues. FAIL returns to a fresh coder; every new review uses a different fresh child. This self-critique cannot satisfy that gate.

