# Independent review — four-seed recovery and data memory mappings

Verdict: **PASS**. Blocking issues: none. This is implementation review and prospective execution authorization, not a research result or permission to analyze fewer than four training seeds.

The reviewer independently read start.md/AGENTS.md, DESIGN_FOUR_SEED_RECOVERY_ADDENDUM.md, both CODER_IMPLEMENTATION_REPORT_RECOVERY_DATA.md and CODER_IMPLEMENTATION_REPORT_RECOVERY_CAMPAIGN.md, the actual eleven changed/new implementation/configuration files, relevant unchanged export/delivery paths, and the previous authorization and reviews. Coder and reviewer are different agents. No formal training or performance analysis was performed by this reviewer.

## Four-seed evidence and stage flow

The effective configuration fixes development seeds5201–5204 and retains independent locked seeds6201–6204. Dataset, native, pilot, base, groups, demand, formation, reader, controls and pinned-source configuration objects exactly equal the original09 configuration. Practical mean LL/AUC/cell thresholds also exactly match. Only the required seed-direction condition now applies to every development seed. The original fetched plan file remains unchanged.

The decision code blocks configurations with fewer than four distinct development seeds and incomplete seed/method inventories. Cross-seed functional intersections use all four seeds. KuaiRand K1_WIDE and SOURCE_INIT_RANDOM are mandatory evidence-completion conditions unioned with any newly triggered S3 subset. S3 blocks when any required control is missing for any development seed. Completed in-memory formations/readers/diagnostics are reused; the new completion receipts identify their actual statistical seed. Final freeze recomputes stage decisions and checks actual prefix/method receipts across four seeds, plus the hash-bound existing SCALE receipts. Empty inventory is BLOCKED_INCOMPLETE, not fabricated PASS. Exposed-role metrics remain behind the global freeze.

## Memory and state boundary

RowDataStore accepts explicit NumPy arrays only, rejects object dtype, writes content-addressed ordinary NPY data in bounded chunks, verifies dtype/shape/content hashes, and returns read-only mappings. Relevant call sites explicitly cover native H/Z, pilot hidden activations, base OOF/refit logits, final and single-function prediction dictionaries, retained feature-bank values, formed function outputs, and three diagnostic request arrays. Existing dictionaries are updated in place so prediction closures continue reading the same dictionary objects. Fixed-content banks keep their prior release behavior.

The ledger creates writable bool mappings only on a new method/role/task/purpose key; membership, duplicate-row counting, record order and exact unique-row summaries remain unchanged. Page-reclamation calls visit known data mappings only. No model, optimizer, RNG snapshot, module, formation training_state or recursive object graph is written to a cache. Native and pilot update equations, initialization streams, row plans and selection logic remain unchanged by inspection. Actual old-seed native-selected/native-epoch24/pilot hashes and selected epochs must match during replay; the new overall source/configuration identity is explicitly different.

GPU admission still requires each card's memory and power below85%. Host admission reserves8GiB, reserves pending prefix working sets and uses measured peak RSS with15% headroom after the first observed prefix per dataset. The first prefix peak and eight-context steady state are runtime evidence, not assumed measurements. An insufficient idle-host budget raises an explicit error; it does not silently reduce batch, epochs or data. The registered fallback remains available if actual memory proves insufficient.

Cleanup enumerates only the current v2 prefix row-data files named by their manifests. It requires accepted final exports and worker termination first, verifies resolved ownership, and leaves final paired/candidate/function exports and old v1 evidence outside the deletion set. No training-state file format or new swap file is introduced.

## Independent checks actually executed

- Actual MovieLens16384-row feature arrays, a noncontiguous view and FP64 predictions: exact values, dtype, shape/order and both original/chunked hashes before and after mapping and reuse.
- Prediction dictionary/object identity preserved; small live model state and Torch RNG unchanged.
- Original bool-mask ledger versus new writable mappings: overlapping and duplicate rows, different methods/tasks, multiple passes and an empty access gave identical records and summaries.
- Actual native features for4098 MovieLens rows spanning three domains: source PilotAccess cached raw hidden activations and FP64-statistics/FP32-return standardization matched direct frozen forwards bit-for-bit; no fitting occurred.
- Independent synthetic gate checks: fourth-seed opposite direction rejected; three-of-four and old two-seed configuration blocked; old two-seed freeze rejected; mandatory KuaiRand control union and a missing fourth-seed control blocked as specified.
- Actual empty new campaign inventory blocked analysis while both existing SCALE datasets had four hash-matching completion receipts. Synthetic host-memory values independently checked the8GiB reserve and measured-peak update.

Receipts: RECOVERY_DATA_INDEPENDENT_CHECKS.json and RECOVERY_GATE_INDEPENDENT_CHECKS.json. These are implementation equivalence checks, not four-seed performance evidence. All actual training completion, replay hash comparisons, full-role raw-X tolerance≤1e-6, fused deployment timing, mechanism exports, final four-seed acceptance and export acceptance remain required RUN-stage obligations. Do not infer them from this PASS.

The issued authorization binds the exact current Python/source-manifest file map and effective configuration hash. Any later source/configuration change requires appropriate independent review before affected execution; this review does not authorize arbitrary runtime injection, a partial two-seed conclusion, or an untriggered experiment.
