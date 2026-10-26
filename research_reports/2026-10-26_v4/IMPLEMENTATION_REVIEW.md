**Latest scope:** development native/pilot, P1 four-OOF, and conditionally triggered internal P2/P3 are admitted after fixes. P4, exposed scoring and final cleanup require later scope review. Historical admission records follow.

**Latest scope:** native/pilot prefix and P1 eta/four-OOF execution are admitted after fixes; P2/P3/P4 and exposed scoring remain unadmitted. The original prefix-only record below retains its historical scope.

# REVIEW REPORT — DFD-v4.0

## 1. Verdict and exact admission scope

**PASS WITH FIXES — native/pilot prefix only. All blocking findings in this admitted scope are corrected.**

The reviewer independently read DESIGN_SPECIFICATION, locked files04/05, AGENT.md, the current prefix/library coder reports and actual source. Admission is limited to the four development native24→selected-cache→pilot12 prefixes for ML/KR seeds5101/5102, including the retained required native epoch1 and pilot pass1 measurement boundaries. It does **not** admit P1 eta/OOF execution, P2/P3, P4 adaptations, or exposed benchmark scoring. Those require their own source-matched review.

Admitted prefix source SHA256: `1273530d12170a63fdda6c68f1b037e5aa67ccf1487a58cf5934518a38b17850`. Exact dependency files/hashes are in REVIEWER_PREFIX_ACCEPTANCE.json. Later demand/CLI additions cannot change the reviewed native/pilot code silently.

## 2. Requirement-by-requirement verification

| Requirement | Independent inspection / evidence | Result |
|---|---|---|
| Actual natural data and fit-only encoding | RoleBundle independently rebuilt every role and checked raw SHA/domains/labels, all saved x/y bytes, fit-only vocabulary/scaler, atomic disjoint/exhaustive rows and exact natural fit counts. Both full verifications completed. | PASS |
| Raw task/user identities | ML rating>3/rating>4 and raw user_id/movie_id retained. KR is_click/long_view taken from raw columns; no raw user/item identity is invented. | PASS with declared KR support limitation |
| Canonical fixed native | Seven vendor dependencies match exact c5618e28 remote Git blobs. Actual original model is initialized first, learned-weight state is removed and copied into fixed. No learned fusion/AutoML optimizer is present. | PASS |
| Native packet and gradient graph | Independent actual-schema CPU checks: sigmoid parity ML max5.96e-8, KR0; shared[24,4,64], h[24,256] exactly equals four expert-wise affine-free LN concatenation. Hooked real backward confirms gate inputs require_grad=False. | PASS |
| Native training target/batch/optimizer | Production now uses a full physical40960 batch, short last retained, sum BCE/(2*actual logicalB), exactly one coupled Adam(.001,.9,.999,1e-8,wd1e-5) update. FP32/noAMP/noTF32. Main-plan RNG namespace20 and canonical-init10 are isolated. | PASS |
| Actual full-batch numerical path | Independent GPU checks on the actual first planned40960 rows of both schemas: production physical path vs direct full reference loss error0 and maximum gradient error0. Peak allocated about0.975GB ML/1.080GB KR during this bounded check. These are check-specific observations, not training throughput or final peak claims. | PASS |
| Native checkpoint selection | Actual code runs24 epochs, eligible12–24 validation macroLL minimum, strict comparison keeps earliest ties, epoch24 retained temporarily, selected model permanently frozen before cache. Evaluation uses uniform epsilon1e-7 LL and raw-probability AUC. | PASS |
| Cache identity and permissions | h/z are FP32 global-row memmaps with no labels, actual shared output, selected-native/extractor/encoder/input identity. Resumed pilot→cache and selected-native→cache dependencies are explicitly validated. Exposed labels are not used by this prefix. | PASS |
| Six pilot functions and ownership | Six independent256→32→1 heads, hiddenXavier/output0,49542params; namespace30 initialization/31 plan,12passes, natural loss/(2B), h/z detached. Only actual owner heads enter the training graph. | PASS |
| None-gradient Adam semantics | On real frozen ML schema features, source0 pilot update leaves all source1/2 grads None and those parameters absent from Adam state, including after resumed next step. | PASS |
| Complete continuation | Actual serialized model/optimizer/gradients/modes/requires_grad/RNG payload was restored; its next pilot Adam step is bitwise equal to uninterrupted execution. Identity is validated before restoration. Temporary test state was removed. Prefix cursor/history/full selected state are present in actual production payloads. | PASS |
| Logging, finite matrix and cost boundaries | Explicit source/data/config/plan identity, native/validation/pilot leaves and label-access journals; deterministic full matrix remains fixed. Retained first required component measurement is not an extra quality trial. Complete-stage budget admission remains orchestrator-owned after measurements. | PASS within admitted scope |
| User storage override | Actual paths write only this artifact_root/_temporary; no new /home_new write. Rolling complete state is temporary for continuation/selection/dependencies, then models and large caches must be removed after final predictions. Final absence of model re-inference is a user-requested retention limit. | PASS, final cleanup obligation remains |
| Strong controls / later stages | P1 four OOF strategies and P2/P3/P4 training are outside this admission. No historical baseline or previous result is overwritten or substituted. | NOT YET ADMITTED |

Full data receipts are in reviewer_p0/{movielens,kuairand}_DATA_ROLE_MANIFEST.json; native graph receipt in REVIEWER_NATIVE_CHECK.json; numerical/state/source receipt in REVIEWER_PREFIX_ACCEPTANCE.json. Historical source identity and pinned source provenance have separate factual receipts.

## 3. Bugs/deviations found

| Severity | Finding | Resolution before any research trajectory |
|---|---|---|
| CRITICAL | Generic prefix payload used explicit dependencies={} plus **other while pilot passed dependencies, causing a duplicate-keyword TypeError at first pilot checkpoint. | Coder constructs the base then updates fields; actual serialized pilot payload was independently exercised. |
| MAJOR | Prefix source identity initially included incomplete eta/CLI files; adding later modules would invalidate a correctly resumable native state. | Actual dependency scope is now native/pilot-only and source-matched independently. |
| MAJOR | load_prefix did not explicitly validate its saved pilot/cache and selected-native/cache dependency hashes. | Checks added and independently inspected. |
| MAJOR, unadmitted optional optimization | ML physical4096 accumulation failed the fixed atol1e-6/rtol1e-4 gradient check in19 elements across3 tensors; maximum difference4.02e-6. KR passed, max5.07e-7. Loss differences were3.73e-8/7.45e-9; full denominators were correct. | No tolerance was relaxed. The optional optimization is not admitted; both datasets/pilots now use the original full40960 physical path, which fits GPU0 and passed independent direct-reference checks. Original failure is preserved in REVIEWER_MICROBATCH_DIAGNOSTIC.json. |

Separate early eta/OOF findings were sent to its coder before execution (duplicate simplex alpha key, diagnostic clipping used in DIRECT training, incomplete head/source identity and missing train-fold constant error diagnostic). These do not change the above prefix verdict and are not claimed resolved/admitted by this report yet.

## 4. Corrections made

Implementation corrections were made by the assigned coder; the reviewer did not redesign the experiment. The reviewer independently verified the corrected actual source and numerical paths. Full physical batching is already the original logical-batch training definition; using it changes no loss, sample rows, learning rate, Adam update count, seed or selection rule.

The user's later storage instruction supersedes permanent checkpoint retention. Actual scoped paths and coder report now record this override. No old /home_new folder is modified by this review.

## 5. Remaining risks and limits

No blocker remains in the admitted native/pilot prefix. P1 eta/OOF and later conditionally triggered stages still need independent review. Final cleanup and result/prediction/source identity receipts must precede deletion of temporary complete state; after user-requested deletion a final predictor cannot be re-run from retained weights. KR original-user bootstrap/support cannot be claimed without a trustworthy row mapping. These are explicit scope limits, not efficacy conclusions.

## 6. Safe to run

**Yes: the exact source-matched native/pilot prefix is safe to run.** The runner may execute the required retained native epoch1 boundary, complete measured budget/plan admission, then continue the same four prescribed development prefixes. It must not run eta/OOF, formation/readout, formal seeds or exposed scoring under this scoped approval. No research-quality result has been analyzed by this reviewer.


## 7. P1 eta / four OOF policies — additional independent admission

**Verdict: PASS WITH FIXES. No blocking issue remains in this P1 scope.** This extends the initial prefix-only approval through P1 OOF scoring and the registered dataset-level decision. It does not admit formation, formal seeds, or exposed evaluation. Exact current hashes are in REVIEW_P1_APPROVAL.json.

| Requirement | Actual inspection and independent evidence | Result |
|---|---|---|
| Fold isolation | Per-domain model-seed-independent PCG folds; both tasks share row fold. Eta/LOGIT/DIRECT train each heldout complement only; CONST fits that same train fold. Actual train/heldout row hashes and trained head/optimizer hashes are retained. | PASS |
| Inputs/normalizer | FULL261=[h256,z0,z1,f0q,f1q,f2q]; LOGIT keeps corresponding final5 coordinates. Normalizers use target fit frozen features only, std floor.001 and clip±10. No heldout label enters normalization. | PASS |
| Initialization/optimization | Hidden Xavier/bias0, output weight0; probability bias uses train-fold positive rate. DIRECT prior[.9,.05,.05]. Fixed20passes, batch512, Adam.001/wd0, separated kind/fold streams; no heldout selection. | PASS |
| Demand/action | FP64 diagnostic logits clip1e-7; exact analytic BCE mu with eta, no y argument. Hard argmax OFF/source0/source1 with exact tie OFF/source-ID. No m/tau in P1 actions. | PASS |
| Strong DIRECT | Separate stable true half-mixture logits, no diagnostic clip in training; actual convex-mixture BCE. Independent extreme-logit check preserves±1000 own logits and remains finite. | PASS |
| Constant control | Prescribed three FP64 SLSQP starts/bounds/equality/analytic gradient/maxiter/ftol, component protection1e-15, explicit feasibility/FWgap≤1e-7. Heldout does not select coefficients. | PASS |
| Continuation and frozen prefix | Independent64-real-row semantic checks for eta, LOGIT and DIRECT: interrupted pass1→pass2 equals uninterrupted2pass bitwise for parameters and Adam; frozen native/pilot unchanged. Temporary test models removed. These are semantic checks, not research trajectories. | PASS |
| Numeric/axes | Analytic BCE identity max error1.776e-15; shared row folds and all six actual train/heldout partitions checked disjoint. | PASS |
| Actual cost path | One additional64-row actual head pass with production CostLedger created exactly one cost leaf and one matching access record; no duplicate keyword, no model left. | PASS |
| Outputs | Original rows/fold/candidates/eta/mu/actions/four p/loss, train-fold constant-d error, fixed OOF quartiles and actual trained-state identities are exported. Source/cache/pilot/data identity checked on reuse. | PASS |
| Recovery/decision | Partial OOF export is rebuilt from completed head states. decide_p1 validates completion sidecar, NPZ SHA and stored/current checkpoint/source identities before the exact both-seed gain/cell/LOGIT conditions on P_form OOF. Missing/invalid output is INCOMPLETE. | PASS |
| Later stages | Full-pool refit/m/tau/source formation execute only after P1 passing and separate conditional-stage admission. | NOT YET ADMITTED |

P1 corrections, all before research eta/OOF execution:

| Severity | Finding | Correction independently checked |
|---|---|---|
| CRITICAL | CONST receipt duplicated alpha as explicit and unpacked keyword. | Base record then update; no duplicate. |
| CRITICAL | Actual CostLedger call duplicated its component argument with RNG component metadata; ledger=None smoke had not exercised this. | Field renamed rng_component; actual ledger path independently ran successfully. |
| MAJOR | DIRECT trained on diagnosis-clipped logits. | Stable un-clipped log-probability/log-complement half-mixture logits, diagnostic clip retained only for d/mu. |
| MAJOR | Head/OOF identities lacked source/pilot-feature or actual trained-head identity. | Immutable source/cache/pilot-feature/train-row/normalizer plus actual head/Adam hashes recorded and validated. |
| MAJOR | Interrupted export could block same-state recovery, and leftover metrics could appear complete. | Completion/NPZ/checkpoint/source consistency enforced; only export is rebuilt, finished fold fitting is not repeated. |
| MINOR, required mechanism output | Missing train-fold constant-d comparator and four per-row losses. | Exported without altering fitting/actions. |

Independent receipt: REVIEWER_P1_ACCEPTANCE.json, including the original numerical-check source and the final log-field-only revision checked on its actual ledger path. Numerical math/initialization/update code did not change in that last revision. No extra research seed, quality run, hypothesis, threshold or ablation was introduced. No P1 research result was analyzed by this reviewer.


## 8. Conditional internal P2/P3 — additional independent admission

**Verdict: PASS WITH FIXES. All blocking findings in this scope are corrected before any conditional research trajectory.** This admits the locked internal matrix only after its dataset-level predecessor gate passes; it does not turn an untriggered stage into a required run. Exact source/files are in REVIEW_CONDITIONAL_APPROVAL.json. Current pipeline SHA256 is `7fe9347ba347e8ed6f6bc78199adb68a62b636117998cd57c3c82bce8b3e8514`.

| Requirement | Actual implementation independently verified | Result |
|---|---|---|
| Full-pool demand/weights | Fresh eta refit20 only after P1 passing; observed full-pool m, equal-target population SD around pooled weighted mean, tau floor.001. Actual-source h/z/domain retained for target extrapolation; fixed w=.1+1.8softmax, per-row/task sum2. No eta/source/native target gradient chain. | PASS |
| Formation and paired controls | Twelve independent source/task/intended-target256→32→1 functions copied from own pilot with paired output-weight-only perturbations/fresh Adam. Twelve passes, actual owner labels, denominator4B, no owner-sum substitute. Full physical40960, short last kept, one coupled Adam. | PASS |
| Permissions/state | Frozen h/z/pilot/eta; active source losses only. Independent initialized-absent-source test preserves parameter/Adam m/v/step and None grads. Actual complete-state next-step formation resume has parameter/optimizer error0. | PASS |
| GEOM/SHUFFLE | Prescribed fit-only K2/subset/seed/n_init/temperature; fixed whole two-coordinate vector shuffle within source/task/y/confidence decile. Completed weights restored directly, partial GEOM restores frozen centers/tau without re-fitting. Independent exact joint-vector preservation passed. | PASS |
| LOCAL_CAP/POOL | LOCAL_CAP24 independent local heads, per-function4B coefficient, no foreign labels. POOL new two-task pooled pilot12, then all12slots allfit12 with12B denominator; original own pilot remains component0. Different parameter/support costs remain explicit. | PASS |
| Ordinary readers | Local plus all4foreign functions,263 input; target fit frozen normalizers. Every soft arm P_form20 then P_read10 with same Adam state. Only full six-head pass0–10 groups/OFF considered; strict macroLL comparison keeps OFF/earliest ties. No per-cell stitching. | PASS |
| Direct controls | H_DIRECT full258 and actual target labels; ETA_DIRECT deep-copies refit eta parameters AND Adam, freezes original, continues P_read10. Their own complete six-head candidates pass0–10, no inappropriate OFF rescue. | PASS |
| Strong constant | Same UNIFORM bank, unique P_form∪P_read labels, exact prescribed FP64 simplex solver and FWgap. One complete constant six-head group vs exact OFF on P_cal. | PASS |
| Budget bridge | U_BUDGET loads the exact UNIFORM prefit20 full identity and complete phase/pass21/head0/step0 boundary, continues to60 before P_read10. It never branches after P_read. | PASS |
| Whole predictor and endpoints | Raw-X capsule contains all actual native/pilot/bank/readout/normalizer state. Final combined hash now explicitly includes native/pilot/bank/readout/exact-policy identities, including actual bank on OFF; same identity is in raw predictions/COMPLETE/capsule. Fixed read-pass10 endpoint has separate complete identity. Per-run reconstruction≤1e-6 remains a runtime completion condition. | PASS |
| Decision gates | All nine P2 references retained; both-seed positive LL gains/mean≥.0005, average AUC protection, maximum over cell mean-seed LL harm≤.003, and specified nondegeneracy. P3 compares both U_BUDGET and POOL identically. No in-stage scientific early exit. | PASS |
| Bounded diagnostics | Two source deletions, four UNIFORM function replacements and two intent swaps at each fixed model's original alpha. No rerouting/retraining. OOF cuts/confidence/support groups fixed in advance. These are descriptive APIs, not proof of complete-training necessity. | PASS |
| Retained measurements | New formation1/reader1 pauses occur only after saving complete pass state and next cursor. Resume continues the same trajectory; stop flags do not change method identity, optimizer or loss. | PASS |
| Precision/storage/cost | FP32 parameter updates, stable FP64 mixture risk, explicit nonoverlapping leaves/access records; current/selected/prefit states temporarily under this campaign only. Final deletion/exposed freeze is outside this admission. | PASS within scope |

Independent minimal conditional numerical execution passed: formation loss error0, optional small-tensor split-gradient error1.86e-8 (production uses full40960), absent initialized Adam unchanged, frozen inputs/pilot no gradients, complete next-step resume error0, correct five-candidate ordering/exact OFF, FP64 mixture error6.66e-16, K3/K5 simplex FWgaps8.31e-9/0, exact stratified joint vectors, OFF whole-policy tie. Semantic test temporary model files were confined to this campaign and removed. See REVIEWER_CONDITIONAL_ACCEPTANCE.json. The final added component-identity and complete-pass pause fields were statically inspected; no extra research or repeated numerical trial was needed.

| Severity | Finding | Correction verified |
|---|---|---|
| MAJOR | Resume recomputed GEOM KMeans/responsibilities instead of reusing the frozen choice. | Restore matching saved weights, or exact frozen centers/tau if weight export was incomplete; no new clustering. |
| MAJOR | CONST/U_BUDGET borrowed UNIFORM state without full source/data/prefix/method/stage/config/plan validation. | Full dependency identity required; U_BUDGET also validates the exact prefit20 cursor. |
| MAJOR | Selected predictor hash omitted explicit actual formed-bank state. | Combined native/pilot/bank/readout identity now retained in all selected outputs and endpoint identity separately. |
| MINOR, declared execution choice | Small-network physical4096 optimization was unnecessary beside authoritative full40960 native path. | Parent authorized full40960 physical formation/pooled passes for all paired arms, unchanged logical target/Adam count; report updated. |

**Safe to run:** yes, via configure() plus the source-matched resume path, only for dataset stages whose exact predecessor gate passed and whose complete-package resource budget is admitted. ML P2 may start independently of KR's pending P1. All triggered outputs must complete; a strong early opponent is not an exit condition. P4 external adaptations, exposed scoring/final method freeze and final cleanup are not approved by this section.
