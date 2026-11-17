# REVIEW REPORT

1. **Verdict: FAIL.** Scientific numeric evidence: **PASS**. One open **MAJOR** publication/retention issue; no CRITICAL or MINOR findings. No input corrections made.

Reviewer: Darwin, native reviewer_id `01a1e959-925f-7742-913b-f34c65ffea24`. Original parent: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. This fresh review is distinct from all four coders and preceding reviewers. Root contract remains **CHANGE, hash=allow**.

**Blocking finding B1:** the bound Cycle04 RAM owner PID2476891/session67406 is absent on the current host `ubun`. Repeated procfs checks and the current xzf process inventory show no Cycle04 owner; protected TV owner PID1025015 remains visible. Yet [REPORT.md](/home/xzf/Code/phase0-3_tvg_rm_20261116/research_reports/completion_weight_attribution_20261117/REPORT.md:160), [DATA_AVAILABILITY.md](/home/xzf/Code/phase0-3_tvg_rm_20261116/research_reports/completion_weight_attribution_20261117/DATA_AVAILABILITY.md:5) and EXP_AI_CONTEXT.md:15 state that the selected states remain in that owner. Last timed observation: 2026-11-17T12:02:24.146007+00:00. The JSON records the exact input hashes and observation.

The parent’s direct acceptance at 2026-11-17T11:21:33.883200+00:00 remains valid historical evidence: 40 selected models, 20 pairs, 191017920 bytes, 1000 ACKs and 34500 useful updates. This review independently verified its numeric bindings and inspected its direct get/hash implementation. That receipt does not establish present availability. B1 does **not** invalidate the completed numerical results or change MIXED.

2. **Requirement-by-requirement verification**

| ID | Requirement | Finding |
|---|---|---|
| R01 | Governing policy, stop controls and identity | PASS: Both workspaces ACTIVE, manual_stop_requested=false, no sentinel; current scope is finish C04 then exactly one full C05, no C06. Darwin is distinct from all four named coders and prior reviewers. Root CHANGE/hash=allow preserved. |
| R02 | Frozen source, design and runtime | PASS: All 48 installed source files match source_receipt; canonical bundle d1930f26... verified. Seven accepted design files and six-file research bundle match acceptance. Installed torch/optim/adam.py matches b099132d... . |
| R03 | Specification and algorithm scope | PASS: Inspected direct W reduction, S=mC, support/mask coefficients, actual scalar backward, source Adam and one live EMA. Reused unchanged source-bound code/engineering proofs after byte identity. No new algorithm, arm or baseline change. |
| R04 | Mathematics, shapes, masks, weights, gradients | PASS: [F,B]/[B,2]/[B] routes and probability BCE remain as designed; counts are host constants, W and S remain differentiable. All 207000 coefficient rows satisfy mass/support identities; matched W/S m equals exactly at each seed/epoch/batch. R has no live update path. |
| R05 | Baseline, natural-data, seed and selection bindings | PASS: Exact 20 reused controls / 1020 file hashes and archive copies verified. All 50 order hashes and initial hashes agree per dataset/seed. Independent strict AUC selection, same-state loss and chosen hash verified for both estimators. Manifest/data/config bindings preserved; no cache/Test unpickling. |
| R06 | Full useful budget and acknowledged records | PASS: 20 new x50 epochs, 1000 formal ACK/epoch/transaction/batch links, 34500 updates and all final operation counters independently reconciled; old275 epochs contribute zero. |
| R07 | Raw endpoints and curves | PASS: All400 ENDPOINTS and4000 CURVES records matched actual raw result/epoch metrics, epoch/order/model hashes and selected states. Numerical appendix endpoint lines match. |
| R08 | All comparisons and seed statistics | PASS: 15 families x2 datasets x5 endpoints; 25500 parent paired rows and5100 summaries over34 metrics including epoch, plus24750/4950 frozen export rows. Independent centered sample SD/(n-1), t4 intervals, sign/favorable counts and2720 endpoint means match. |
| R09 | Primary decision and eligibility | PASS: W EMA-C EMA is MIXED: ML mean negative, KR positive, both intervals cross0, both miss+.003; KRd0t0 mean below-.001. Separate loss benefit retained; no champion or Test release. |
| R10 | Conditional mechanisms and contradictions | PASS: Verified task1 W-S LL gains, online/EMA distinction, ML N-W E1/E50 reversal, KRd2 rank gain/d0 LL harm/d1 LL gain and macro/traffic/pooled contradictions. No mediation percentages, class-only/balancing, Adam-LR equivalence, universal CL/momentum or SOTA claim. |
| R11 | Diagnostics, annotations and null semantics | PASS: 93600 raw diagnostic rows at80 committed points,17888 mean rows and4160 algebraic/Boolean annotations checked. Actual-versus-virtual hashes match; values are grouped within seed before between-seed summaries; null reasons retained.48000 statistic rows and19200 means verified. |
| R12 | Physical execution and costs | PASS: Independent gzip reading of1009 physical batch files confirms34640 returned calls,140 outside ACK files,81/243/324 physical diagnoses/sweeps/virtual steps versus80/240/320 useful.30 processes=20COMPLETE/8PAUSED/2FAILED_OBSERVED.28 cost means checked; overlapping times not added. |
| R13 | Recovery provenance and historical corrections | PASS: Fresh Dirac recovery02 PASS supersedes historical hash-denied partial preparation; source bindings match.111 recovery ACK workflows sum61.909074446884915s; original wrong-zero raw report hash preserved and parent correction exact. Retired40 processes/275 ACKs/7733 updates and10523.255015449016s independently recounted separately. |
| R14 | Raw archive coverage, integrity and exclusions | PASS: 89 parts and concatenated archive SHA verified; all6611 member hashes/sizes checked against stream and current originals;4243 new+1020 controls+1348 retired including48 frozen source files. Exact eligible tree coverage, gzip CRC, safe regular paths, excluded tensor suffixes/private owner files and known owner credential values checked. |
| R15 | Scientific figures and maintained contexts | PASS_WITH_AVAILABILITY_EXCEPTION: Read FIGURES.py and visually inspected all three actual PNGs against verified data. All seven current context updates preserve scoped evidence strengths and contradictions;16 ledger metric rows match. Current RAM assertions are covered by B1. |
| R16 | Historical direct selected-RAM acceptance | PASS: PARENT_ACCEPT.py uses actual get/state_digest and validates selected epoch/validation/ACKs. Its recorded40-model/20-pair/191017920B acceptance, result hashes, handoff hash and all numeric rows match. |
| R17 | Current selected-state retention and truthful availability | FAIL: The bound ownerPID2476891 is absent from current host ubun procfs on repeated reads; ps -u xzf shows no C04 owner, while protected TV owner1025015 remains visible. Current report/data-availability/context present-tense retention assertions are false as observed. No40-model pull or owner IPC performed. |
| R18 | Publication status and continuation | BLOCKED_BY_B1: Main publication remains parent's next step, not a completed action. Numeric evidence is valid for the scoped conclusions. Current package needs truthful RAM-loss/disposition reporting before final release; no additional user permission, new research or automatic rerun is requested. |

Frozen scientific source48: `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`.

Seven-file design: `2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7`.

Six-file research bundle: `a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a`.

The actual source48, seven accepted design files, external installed Adam and the distinct restart/code, engineering and recovery02 PASS receipts all match their recorded bytes. Unchanged source-bound mathematical, gradient, replay and engineering proofs were reused after those identities were established. I inspected the actual completion loss and live/diagnostic update paths; no broad tests or training reruns were performed.

W is constructed directly as `.75*C+.25*sum(w*pi²*class_mean_BCE)`; S is the same batch’s `m*C`, with `m=1-.25*sum(w*2*pi0*pi1)`. There is no division by m. Only the actual W/S scalar backward reaches live Adam, followed by one live EMA update. Counts are host constants; model probabilities and source parameters remain differentiable. Ordinary input/target/domain shapes, masks, absent-class nulls and None-versus-tensor-zero optimizer semantics remain source-bound. Counterfactual C/S/W/N clones use the actual source Adam without applying N’s gradient to the live model.

All207000 coefficient rows satisfy the recorded support/mass identities. W/S m is exactly equal at every corresponding batch; five-seed mean m is ML0.8947690674512807 and KR0.8994699862685044. This is combined class/cell/task allocation, not an isolated class-weight experiment.

**Independent arithmetic and source records**

I read all20 exact reused control results and1000 control epoch files, verified their hashes and archive copies, then followed all1000 new ACKs to the exact epoch, transaction and compressed batch identities. ACK event chronology and actual server journal linkage match. Initial hashes and all50 epoch order hashes match the corresponding natural-source controls. Each estimator independently selects the earliest strict AUC maximum; its LL, cell metrics, epoch and state hash stay bound to that same state.

All400 raw endpoints and4000 curve records match actual result/epoch values and hashes. Parent tables contain25500 paired-seed rows,5100 summaries and2720 endpoint mean rows. I separately verified the frozen exporter’s24750 paired rows/4950 summaries. Coverage is15 contrast families, two datasets, five endpoints and both online/fixed EMA routes; the parent adds epoch as the34th metric. Each paired group has exactly seeds2022–2026.

Independent arithmetic used `math.fsum` and an explicit centered squared-deviation sum divided by4, with `mean ± 2.7764451051977987*SD/sqrt(5)`. Means, sample SD, intervals, positive/negative/zero and favorable/unfavorable counts match. All numerical-appendix endpoint/hash lines match their sources. Figure code was read and all three actual PNGs were visually inspected.

| Selected W EMA − CLEAN EMA | Mean AUC | Sample SD | Descriptive95% interval | Positive seeds | Mean LL | LL-lowering seeds |
|---|---:|---:|---|---:|---:|---:|
| MovieLens | −0.00015462144763278562 | 0.00045484050930574133 | [−0.0007193805329623899, +0.0004101376376968186] | 2/5 | −0.0003053164734484248 | 3/5 |
| KuaiRand | +0.0025550857333183517 | 0.004445800598895028 | [−0.002965104813522348, +0.008075276280159051] | 3/5 | −0.003942334325717012 | 5/5 |

Both primary AUC intervals cross0 and both means fail+.003. ML’s minimum cell mean−0.00034086049986612463 passes−.001; KRd0t0−0.0031413726095260277 fails. The category **MIXED**, failed qualification, no champion and Test=false are correct. KR Macro LL’s interval [−0.005845996309511461,−0.0020386723419225632] is negative; this is retained separately from AUC qualification.

**Scientific claim review**

- W−S task1 LL decreases for all five seeds in both datasets: ML−0.002988932380749809, KR−0.00512458085444335, both intervals negative. MEDIUM conditional evidence for heterogeneous allocation is appropriate.
- ML N−W online selected AUC is−0.000412488607235173, five negative seeds, negative interval. Matched selected EMA AUC is−0.0001888954520666175, two positive seeds and an interval crossing0. EMA selected LL increases for all five seeds. The report preserves the estimator distinction.
- ML N−W EMA AUC reverses from E1+0.01927757693704717, five positive seeds and positive interval, to E50−0.001496855328235025, five negative seeds and negative interval; E50 LL increases. The report correctly treats this as conditional stage evidence and leaves ranking necessity unresolved.
- KR W−C d2 AUC+0.009201239807310201 is five-seed favorable; d0 LL+0.002005434900329056 is five-seed adverse; largestd1 LL−0.006935433446059669 is five-seed favorable. These contradict a uniform small-domain-gain/large-domain-cost story. Natural priors, populations, domain identity and baseline performance remain competing explanations; size-as-main-cause is WEAK.
- KR macro AUC+0.0025550857333183517 coexists with sample-weighted cell AUC−0.00016677874149220173 and pooled-task mean AUC+0.0000044399773567516124, the latter two only1/5 positive. Pooled LL improves for5/5. Aggregations are correctly distinguished; pooled and traffic-weighted LL are not counted as independent repetitions.
- Local diagnostics are scoped to the current host state. No mediation percentage, class-only or balancing identification, Adam learning-rate equivalence, universal curriculum/momentum efficacy, inevitable difficult-domain M3oE failure or SOTA claim was found. Reused C/N and the identical C3/C2 prefix are not counted as new independent replications.
- All seven maintained current context updates preserve these scientific conclusions and the historical contradictions. Their hashes are in the JSON. Present-tense RAM statements are the exception in B1.

**Diagnostics, baselines and costs**

Verified93600 raw diagnostic rows from80 committed points,17888 numerical/null mean rows,4160 formula/Boolean annotations,48000 statistics rows and19200 statistic means. Within-seed aggregation, explicit unavailable values, actual/virtual hashes and algebraic deltaEMA annotations match. Twelve baseline cell and six baseline domain records match exact CLEAN files and natural train/validation count/label receipts; no cache was opened.

Independent gzip reading confirms1009 physical batch files,34640 returned calls and140 outside current ACK-linked files,1408883900 physically returned rows,81 diagnoses,243 extra sweeps and324 virtual steps. Useful totals are34500 calls,1403149500 rows,80/240/320 diagnostics/sweeps/virtual steps. Final useful operations also confirm1000 epochs,2806299000 task entries and no extra probe/view forward or algorithm RNG draw.

Thirty formal processes normalize to20COMPLETE,8PAUSED and2FAILED_OBSERVED. Known worker duration is60790.55736127368s with two missing terminal durations; supervisor process-wall is68156.81714725494s. Returned-call duration61546.47154270578s and controller segments14179.961083152913s have different, overlapping coverage. They cannot be added. All formal worker PIDs and recorded controller PID are absent; storage-owner absence is separately material.

The parent’s correction of111 recovery02 ACK workflows to61.909074446884915s matches the exact COSTS.csv and original report hashes. The wrong-zero historical runner report remains archived unchanged. Twenty-eight cost-mean rows, raw inputs and available stdout fields match.

Retired attempt1 independently has40 formal processes,275 ACK epochs/7733 updates,24PAUSED/15FAILED/1FAILED_OBSERVED, known worker duration10523.255015449016s and one missing terminal. It contributes zero current completion credit. Historical hash-denied preparation remains historical; genuine distinct Dirac recovery02 PASS supplies the resolved identity gate.

**Archive verification**

Streamed the existing archive without copying it to home. All89 parts00..88 match their individual sizes/hashes; their combined hash and the unsplit staging archive match:

`5841209454b679425a25e910042e90056553a03521507c30b9287e69b406d5e2`.

Compressed size740197954B;6611 regular members;4077601426 uncompressed bytes. Independently hashed every member and every original path:4243 current numeric members,1020 exact control files and1348 retired numeric/source members, including all48 preserved retired source files. Eligible current/retired trees have exact member coverage, not a sampled subset. Required result/epoch/ACK/transaction/batch/tables/source-receipt/control/recovery/parent-acceptance inputs are included. The outer gzip checksum/trailer also passes.

Members use safe relative regular-file paths; guardian.json, private STORAGE_OWNER.json and tensor/checkpoint/pickle/array suffixes are absent. Scanning found no known actual current/retired owner authkey value, recognizable GitHub credential or private-key material. Current source48 is separately hash-bound for publication; the archived source snapshot is the retired source, correctly identified as such.

Selected input hashes (all57 report/context/key input hashes and all source/design identities are retained in JSON):

| Input | SHA256 |
|---|---|
| REPORT.md | `47e2c53e0960e77d4ec5d6e1690ccb685c450d05a5113d88793307d5eae1db87` |
| DECISION.json | `b99023c50c9cb583567fabc3624109c074072a1b5c64c1270c8a6e686afd49de` |
| ANALYSIS.py | `52e89119bdfd36034d3814505244f747e771dcf94f3277810689b2e149d8cebe` |
| TABLE_RECEIPT.json | `05e3044eec6753f471ebf8f71c7c308dea9ae0a73316c5cb56f1ea5d9085c4a5` |
| PARENT_STORAGE_ACCEPTANCE.json | `c68d93815c0e8c9f60e88898ecf73ac0124243dfeb3b25ccd061b208a5f7d0e7` |
| RAW_RUNS_MANIFEST.json | `a44b00cab058f877888898a4a33c05dd2e191779c6673b82828db015d590dce1` |

3. **Bugs/deviations found**

**B1 — MAJOR, OPEN:** current selected-model availability conflicts with the publication inputs. The bound owner is absent, so the40 accepted selected models cannot currently be accessed through that owner. This affects retention, reproducibility and any action requiring those live states. It does not change historical validation metrics, their registered comparison or full-budget completion.

Parent must record the actual storage-loss/disposition evidence and correct present-tense availability in the report, data-availability page and affected current context, or establish a genuinely surviving provenance-bound state. Preserve original frozen N, its archive and the historical acceptance. Do not turn the stale receipt into current liveness, fabricate tensors, open Test or discard numerical results. This finding requests no training or replacement guardian.

4. **Corrections made**

None. Only FINAL_EVIDENCE_REVIEW.md/json are written. The historical cost-field correction was already made by the parent and independently verified here. No source/report/context or frozen-N input was edited.

5. **Remaining risks**

B1 remains the sole release blocker. Its exit time and cause are unknown. Scientific limits remain five paired seeds, shared/adapted validation populations, correlated comparisons, one model and two natural datasets; size/class/task/optimizer/EMA paths are not independently isolated. Two failed-worker durations and unreturned work remain unknown, and the NVML failure’s concrete API remains unidentified. These limits are represented in the report and do not require new experiments for this review.

6. **Whether experiments are safe to run / publication readiness**

**Current release: FAIL; publication_ready=false; experiments_safe_to_continue=false pending B1 disposition.** The verified numeric evidence supports the stated scoped scientific conclusions and can be preserved and reused as numeric controls. The present package must not be released as currently retaining40 live selected models.

Actual main publication remains the original parent’s next step; this review performed no publication action. The existing user authorization is finish C04, then exactly one full C05, then stop without C06. No additional permission is requested. Test remains **false/sealed**.

Skipped GPU queries/training, model repulls/owner IPC, cache unpickling, Test, broad tests and execution of mutable report generators. Current RAM state is the concrete unresolved limitation. This fresh reviewer returns permanently after these two reports; no descendants, independent AI session or context reuse.
