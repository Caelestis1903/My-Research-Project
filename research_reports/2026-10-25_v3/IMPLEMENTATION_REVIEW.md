# Independent implementation review — MDMT-NEXT-v3.0

**Verdict: PASS WITH FIXES. All blocking findings listed below were corrected before P1 admission.**

Reviewer independently read the approved DESIGN SPECIFICATION, the three coder reports and actual implementation. This is a code/semantic admission review, not an analysis of research effectiveness. Reviewed aggregate source SHA256: `3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2` (71 new/imported source files). Machine receipt: `artifacts/mdmt_next_v3/v3_0/IMPLEMENTATION_REVIEW.json`.

Admission covers bounded P0 diagnostics, 52 core logical 24-epoch trajectories and their eight physical warmups, four audited historical aux resumes, and six audited historical external-method resumes to epoch24. Conditional P2 implementations and fresh-seed 50-epoch external/formal execution require review when their fixed triggers activate; this verdict does not claim those currently dormant implementations exist. Complete-package budgets must be frozen from retained real component measurements before bulk execution, as the CLI enforces.

## Requirement-by-requirement verification

| Requirement | Actual implementation and verification | Result |
|---|---|---|
| Real natural data and historical identities | Data reuses audited prepared arrays/fit-only encoding; raw/array/role verification remains enabled. P_form/P_read independently regenerated and compared to exact old aliases. No label-conditioned sampling or proportion changes. | PASS |
| Native prediction graph | Actual pre-sigmoid logits are returned by the mechanically mirrored adapter, `risk=False`; historical gate detach remains. Both real schemas independently checked against actual historical native forward. | PASS |
| Native ownership | Live module groups include embedding, STAR shared pieces, shared and task experts; complete unique parameter coverage checked. Private gradient remains natural. | PASS |
| Logical training and Adam | Logical batch40960, short final batch retained, FP32/no AMP/TF32, coupled Adam defaults fixed. LPS microbatch accumulation has one optimizer step; scalar denominators use the entire logical source/task set. | PASS |
| C8 and STATIC | Actual p-based coefficients, quarter-min mass transfer, legal/dedup masks and BASE fallback retained; ML action5/KR BASE frozen. Shared and private gradients enter one real Adam update. | PASS |
| RHD history timing | Twelve warmup coordinates4.5–10.0; FREEZE uses those twelve only. CUM/EMA forecast before reveal, then restore/observe before the next main block. Missing action entries stay missing; EMA last32/half-life8. | PASS |
| PB initialization/candidates | One full-C8 event at10.0, then STATIC/history-best/oldest remaining, at most three. H1 and holding-block estimands remain distinct. First main block is not repeated. | PASS |
| PB acceptance permissions | P_update chooses the candidate. Only STATIC and that candidate are scored on P_cal. Two tasks are averaged within row before finite-pool paired SE; acceptance is strict improvement>SE. Unselected cal candidates are not searched. | PASS |
| Exact branch transactions | Model, buffers, modes, true Adam, gradients, RNG and registered controller state restored; journals remain outside rollback. H>1 repeats actual forward/backward with changed states. Independent full40960/H2 semantic checks passed. | PASS |
| LPS capacity/features | Expert-wise LN→h256; detached bank inputs; row-normalized fixed Gaussian projection→a32; u34 uses detached native logits, no geometry. Three×four rank4 blocks and shared-within-source readers have specified sizes. Full owner sum equals rank16 forward and gradients. | PASS |
| LPS direct fit permissions | Real owner route applies detached w exactly once. Native/reader gradients match unweighted values; foreign bank has no direct target gradient. Inactive bank/reader operations retain None semantics. Allocator/read heads are outside main optimizer. | PASS |
| Formation estimator | Fit512/source and P_form512/target, natural source mass, current real bank moments/step, coupled functional Adam, create_graph=True. Only allocator commits; virtual bank/moments are discarded. Own z0 stays real; foreign virtual graph remains differentiable. | PASS |
| Formation outer target | .75 deployed and .25 coverage BCE computed separately, then added. Uniform reference shares support/state; φ gradients, routing support, displacement and no-submit identity recorded. | PASS |
| Six core LPS arms | Independent all-OFF L0, source-coarse L1, uniform-block L2, scalar-meta L3, block-meta L4 and convex constant-read L5 preserved. Read-family warmup forks before arm-specific refresh; L0 cannot fork it. | PASS |
| Read refresh and strong controls | Independent six-head Adam/shuffles; first10/later2 passes; complete old policy+optimizer and full-pass candidates; one whole-policy P_cal choice. L5 FP64 box-constrained convex solver, old/two-start whole-policy candidates. | PASS |
| L0 local predictors | MONO/FEATURE fit full P_label, no foreign output/key, independent validation AUC selection and full selected state. Feature fitting, repeated label use and complete-pipeline inference cost recorded. | PASS |
| Coverage and stage ordering | One seeded whole logical batch per consecutive ten batches from epoch6, carried across epochs; other batches OFF before11/deployed afterward. Formation/read refresh precedes even-epoch validation; odd epochs also validate. | PASS |
| Local reading attribution | Selected/terminal deletion and label-free gate groups retained. L4 now includes paid same-bank convex-constant versus conditional reading, without bank/native/reader mutation. | PASS |
| Selection/evaluation | Eligible≥12, RHD validation LL/LPS validation AUC, strict comparisons preserve earliest ties; common24 retained. Main and local predictors use their own true deployment predictions. Q_explore remains development; exposed outer scoring requires explicit final freeze. | PASS |
| Historical resumes/baselines | Original aux/native/discrete algorithms and STEM/AutoIFS/STGU imports preserved. Ten real saved states independently restored with equal model/optimizer/gradients/modes/flags/RNG. AutoIFS nested optimizer/workflow and STGU state equal. Four aux actor numerical fields equal. | PASS |
| Epoch10 aux continuation | True ML K4 aux0 resume skips exactly the already committed begin-at11 refresh; later refreshes remain original. Old selected aliases and validation prefix preserved. | PASS |
| Source/review/run gates | Exact config membership, source-matched review admission, dependencies, explicit resume and complete-package budget guard inspected. Source/controller identity added to RHD true resume; no unreviewed method is automatically enqueued. | PASS |
| P0 scope | Strict-prefix four H1 traces, separately named two H5 snapshots, honest missing tree candidates, fixed old K1/K4 source-response forwards and matched original calibration pools. No invented raw observations or new full trajectories. | PASS |

## Independent semantic execution

Executed under `CUDA_VISIBLE_DEVICES=0` using the verified Python3.13/torch2.9 environment, in isolated temporary directories. GPU1 was not initialized. These examples are excluded from research-quality trajectories.

- Both actual native schemas: probability maximum absolute difference **0**; expert output `[96,4,64]`, embedding widths113/544. Actual full40960 batch private-gradient maximum error **3.7253e-9**. Initialized None versus zero Adam semantics, H2 current-state recomputation, main-plan immutability and complete rollback digests passed.
- Rank16 capacity: FP64 forward error **1.3878e-17**, gradient error **1.6653e-16**. Explicit one-w bank reference, forward/native/reader differences **0**; foreign direct bank gradients None and inactive Adam states absent.
- Full-source/task scalar normalization error **0**; intentionally normalizing separate chunks differs by **.012939**, showing the test detects the prohibited shortcut. Ten whole-batch coverage decades each contain one selected batch.
- Initialized coupled functional Adam versus real cloned Adam: maximum error **0**, including nonzero decay and None skipping.
- FP64 hypergradient: analytic **9.8345122763e-7**; central-difference relative errors **1.31e-7–8.84e-7** across the frozen step sweep. Production small real-batch/full512-support-and-query meta path produced a finite nonzero allocator gradient; real model and main optimizer were unchanged.
- Ten actual historical last checkpoints: model, true outer optimizer, gradients, requires_grad, modes and RNG numerical identities match after restoration; actual AutoIFS nested workflow and STGU state match. Four aux actor module/optimizer state_dicts and all numerical fields match. Raw object-pickle digest differences for nested nn.Module/optimizer instances were not misclassified as numerical drift.

## Bugs/deviations and corrections

| Severity | Finding | Correction/status |
|---|---|---|
| CRITICAL | Production meta support contained z0, while from_features supplied z0 again through explicit kwargs; first meta step would raise TypeError. | Owner changed to copy/update packet merging. Independent production meta path passed afterward. |
| CRITICAL | Combined semantic entry imported nonexistent lps_semantics instead of semantics_lps. | Correct import inspected; the two actual acceptance modules independently executed successfully. |
| CRITICAL | Initial generic cost/status dict constructors could duplicate additive/checkpoint kwargs used by real event/final status paths. | Owner changed to construct then update rows; affected calls and current code inspected. |
| MAJOR | RHD true resume could accept another controller/source under the current method name. | Source/data/dataset/seed/kind validation added before restore; allowed warmup/PB_HISTORY parents explicit. |
| MAJOR | Closed-loop L4/L5 alone omitted the requested same-bank constant/conditional attribution. | Paid fixed-state convex-constant comparison added at L4 selected/common endpoint; real model identity checked. |
| MAJOR | L0 inference receipt initially timed only OWN, omitting MONO/FEATURE overhead. | Complete native+owner+local-head timing and parameter counts added for each predictor, including warmup work. |
| MAJOR, historical transport | Old aux epoch10 resume would repeat an already completed read-policy refresh. | Isolated bridge skips that single duplicate refresh and records it; original artifacts untouched. |

No research result required rerunning: these corrections preceded P1 execution. Reviewer requested fixes from owners and did not redesign algorithms or change hypotheses/constants.

## Remaining boundaries and safety to run

**Current admitted experiments are safe to run within the recorded scope. No blocking issue remains.** Retained component measurements must continue to their prescribed endpoints; a timing fragment is not a quality-based early exit. The reviewed CLI requires a concrete frozen component/count budget before bulk execution.

This verdict does not establish algorithm efficacy, external-paper score reproduction, or new unseen-data confirmation. Old tree candidate/mid-refresh evidence is genuinely missing; P0 reports that limit. Concurrent process wall totals are not precise GPU hours. Conditional P2/formal code, exact package budgets, scientific decisions and final report aggregation remain explicit subsequent work.
