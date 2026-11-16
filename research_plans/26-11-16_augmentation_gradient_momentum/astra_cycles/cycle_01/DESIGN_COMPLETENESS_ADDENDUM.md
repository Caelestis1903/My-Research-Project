# DESIGN SPECIFICATION — cycle_01 completeness addendum

Date: 2026-11-16. Designer verdict: **PASS, implementable design; parent hard-completeness gate remains required.** No implementation or execution approval is implied. This addendum resolves only the missing DT-PGNA/TGA equivalence audit and historical-storage contamination. It supersedes the unresolved-history statements and shared-guardian write proposal in the six original Astra files; their candidate comparison, literature, scientific equations, historical identities and evidence remain preserved. No new literature search was performed.

## 1. Research question

Does deleting the odd-in-perturbation component of full-support NDP-A improve selected fixed-EMA validation Macro AUC? Do task-axis geometry and exact sign pairing contribute beyond norm-matched isotropic smoothing and independent signs? Before implementing, is this intervention mathematically different from the stopped DT-PGNA/TGA pilot, and can its raw evidence be stored without changing historical inputs?

## 2. Hypothesis being tested

Keep the registered hypotheses separate: AXIS-EVEN improves over CLEAN; improves over one-sided NDP-A; improves over ISOTROPIC-EVEN if task-axis geometry matters; improves over AXIS-IID2 if exact pairing benefits endpoints. None is presumed true. Equal expected objectives of AXIS-EVEN and AXIS-IID2 do not imply an endpoint advantage. Historical non-equivalence is a source/math finding, not efficacy or publication novelty.

## 3. Variables changed

Scientific factors remain exactly the three selected arms: AXIS-EVEN, AXIS-IID2, ISOTROPIC-EVEN. Relative to NDP-A, A replaces its one-sided view objective with the even pair. The two controls change sign sampling or axis distribution as already specified. This addendum changes only storage ownership/root and design provenance; it adds no scientific factor, run, hyperparameter or baseline.

## 4. Variables held constant

Two datasets; original rows/splits/labels/routing/encoders/scalers; fixed1 M3oE architecture and detached training gates; frozen full-derivative probe; alpha=.03, mixture lambda=.25; natural row-task support and mass; FP32, batch40960 including tail; seeds2022–2026; 50 epochs; same initialization and private loader; one coupled Adam and fixed EMA per batch; selection, metrics and statistics in CODEX_EXECUTION_PLAN. Exactly 30 new trajectories, 1500 epochs, 51750 Adam and 51750 EMA updates. Twenty source-indexed CLEAN/NDP-A trajectories are reused only after their existing identity gate. B/C remain unselected. Original data and Test remain sealed/unchanged under the existing policy; this design performs no Test access.

## 5. Mathematical/algorithmic definition and exact historical audit

### 5.1 Verified identities

Historical root H=`/home/xzf/Code/phase0-3-20261115`. Current root W=`/home/xzf/Code/phase0-3_tvg_rm_20261116`. Paths below are relative to those roots. SHA-256 values are hashes of file bytes, not Git object IDs.

Pilot `attempt1/run_manifest.json` records implementation commit `bf36e454adc2d0c569218a7993834f6d56cfc97f` and aggregate source SHA-256 `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`. Its complete 59-file receipt was checked: every local file hash matches, and canonical hashing of its file list reproduces the aggregate. Both pilot source files below also match the bytes at that Git commit. The audit-base commit `79fbd441c6c87cff30838c65bf97ad6ec7f45abb` is provenance, not the trained implementation identity.

| Root / path | SHA-256 |
|---|---|
| H `experiments/heterogeneity_aware_20261115/core.py` | `ec3fcfafc0e3487439b9552204fb14f1514eb18a42a13323648f00796836819d` |
| H `experiments/heterogeneity_aware_20261115/protocol.py` | `a72a5eb9345cd9b0889cbe99541a41173e110ec892cc207274eec98757394f5a` |
| H `research_reports/2026-11-15_heterogeneity_aware/DESIGN_SPECIFICATION.md` | `4f0f15e33141cd347d5a0e6f7dcd6aa8c80b70f842a5b93a6cefc546f31f0cd2` |
| H `research_reports/2026-11-15_heterogeneity_aware/REPORT.md` | `6cf1f9183f2586eef5a97dff7244a37a2a4fb5cbf32140f3d9154043826a6fc0` |
| H `artifacts/heterogeneity_aware_20261115/attempt1/source_receipt.json` | `f3acf374fee6beec22b735af5b2d1ce7fc31964eb2e9afec542533c8fe1dea2b` |
| H `artifacts/heterogeneity_aware_20261115/attempt1/run_manifest.json` | `63fbaa6731aabcc18a187110effbb887787bacd50933bcfae40f30f9033cbeb8` |
| W `experiments/auxiliary_objective_attribution_20261115/core.py` | `0999b534aba4f632f9fbd35a3313d5eeb4fd04405d323f881d2e39a7f4a2bfe3` |
| W `experiments/auxiliary_objective_attribution_20261115/protocol.py` | `2d8cea2d9319e437b6fa48fb99d6d2debdf8b972f1b9d60f1feef897e1151ebe` |
| W `experiments/auxiliary_objective_attribution_20261115/stage2/core.py` | `fe662da1b8c00229a6a234bb7e00d659fab1d278470092531cdeefa6bb2f3d58` |
| W `experiments/auxiliary_objective_attribution_20261115/stage2/protocol.py` | `66ea0d1e6461711c88e2a68ca2ca32ed02e713967f06d680c96f5540816fe731` |

The auxiliary workspace's Stage1 and Stage2 run manifests record Git commit `e5738e52294f064c23a9b953fe74abb84de713df`, respectively source aggregates `a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0` and `831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`. Current auxiliary core and Stage2 core bytes also match published current commit `c4dc36bd38bd4f7a2a3319377a8289ba8185532d`. No new training-source hash exists yet; coder must receipt all transitive imports, including RAM helpers.

### 5.2 Actual pilot math, not names

Let C=mean BCE over all 2B natural row-task entries. Pilot REPORT.md:3 maps A=DT-PGNA, B=TGA-Pair, C=TGA-Example. In pilot core.py:45–63, for each domain/task, k=min(128,n_positive,n_negative). Pair sampling selects k positives and k negatives without replacement within each class and interleaves (+,-); example sampling instead selects 2k distinct natural rows of that domain for that target task. Thus example sampling is still budgeted by both-class availability. A row can appear for both target tasks. These are selected target entries, not two opposite views of the same entry.

With target logit Jacobian g and other-task Jacobian h, DT-PGNA uses q=g-1[||h||²>1e-20]*(h·g/||h||²)*h, with scalar reductions in float64; TGA-Pair uses q=g. The initial positive-row displacement is -alpha*r*q/(||q||+1e-12), negative-row displacement is +alpha*r*q/(||q||+1e-12), where r=RMS(e). For each pair, m=z_pos-z_neg and dm=g_pos·delta_pos-g_neg·delta_neg. Reject if either member's direction norm<=1e-10 or r<=1e-12, or |m|<=1e-7 or |dm|<=1e-12; otherwise multiply both displacements by min(1,|m|/(|dm|+1e-12)). DT-PGNA additionally measures other-task logit drift, halves both radii at most twice, then drops pairs still exceeding 1e-3. TGA-Pair does not do this. See core.py:66–141. Conditional B_MATCH retains the projected route's acceptance/radii but replaces its orientation with unprojected g; it is not an even pair and is not a new arm here.

TGA-Example instead uses b=(sigmoid(z_t)-y_t)*g, delta=alpha*r*b/(||b||+1e-12), dropping near-zero b/r; no projection or pair-margin cap (core.py:100–117). All views detach; the ordinary embedding lookup stays trainable. The full gate-input derivative is probe-only, just as in the current source.

For accepted target-entry multiset S with M>0, actual pilot core.py:170–192 trains

`L_pilot = C + .25 * [2/M * sum_(i,t in S) BCE(p_t(e_i+delta_it),y_it)].`

Total coefficient mass is 1.5, with selected-support auxiliary mass .5, rather than a natural mass-one mixture. No accepted view gives C alone. Pilot `protocol.py:43–55` fixes this objective and ordinary coupled Adam. Pilot RANDOM/GAUSSIAN first use the projected pair route's accepted support and final radii, then replace each accepted displacement by one normalized coordinate-Rademacher/Gaussian direction; if no pair survives they use the example-sampling fallback and alpha*r radii (core.py:142–158). These are neither same-input antithetic pairs nor current full-support norm matching.

Current Stage1 auxiliary core.py:74–132,175–200 preserves B/C support, displacement and extra-target-BCE math; Z-PAIR/Z-EXAMPLE change only accepted deltas to zero, preserving support and coefficient mass. They do not implement even smoothing. Stage2 `make_views` and `losses` (core.py:23–58) repair support/mass: every row contributes each task, invalid directions become zero rather than dropped, and `L_N=.75*C+.25*Lplus`. There is no projection/cap/backtracking there. This exact source supplies the new deltas and the historical NDP-A comparator.

### 5.3 Selected math and non-equivalence finding

For the unchanged detached NDP-A delta on every row-task, define Lplus and Lminus as natural 1/(2B) target BCE at +delta and -delta. Then

`E=(Lplus+Lminus)/2-C; O=(Lplus-Lminus)/2;`

`L_A=.75*C+.125*Lplus+.125*Lminus=C+.25*E;`

`L_N=C+.25*(E+O); L_A-L_N=-.25*O.`

This is an exact identity at a common state with the deltas held fixed. It also describes training-gradient differences through the ordinary detached-gate graph; it is not differentiation through the probe. One-sided BCE generally has O!=0: for a locally affine logit z(e)=w·e, small nonzero delta gives O=(sigmoid(z)-y)*(w·delta)+O(||delta||³). The finite identity is exact; the expansion is only explanatory. Hence neither the one-sided TGA-Example/NDP-A nor paired-different-example TGA-Pair implements this objective. DT-PGNA also changes orientation/acceptance/radius; removing projection alone cannot recover A. The pilot's selected-support mass1.5 cannot in general be transformed into A's full-support mass1 by an optimizer learning-rate relabeling, especially with Adam epsilon and coupled weight decay. No optimizer novelty is claimed: all use the same Adam/EMA family; the changed data gradient feeds its ordinary moments.

In A every natural entry has clean weight .75/(2B) and two replica weights .125/(2B) each, retaining zero directions. All-zero deltas delegate to the exact clean source branch. AXIS-IID2 uses two independent per-row/task/replica Rademacher signs; its conditional expected loss equals L_A. ISOTROPIC-EVEN uses the actual row-task ||delta|| radius with independent Gaussian-normalized axes and the already specified zero-norm fallback. Both have the same four target-task view forwards as A. Pilot RNG is keyed numpy SHA256(dataset,seed,epoch,global_step,stream), sampling selected entries/noise; new RNG is the already registered private torch SHA256 axis_even_v1 stream keyed by dataset/seed/epoch/batch/task, with no loader/global consumption. Preserve its exact key/draw rules; do not replace it with pilot RNG.

**Finding: materially different from the audited pilot and current one-sided/zero-delta objectives.** Clearance is local non-equivalence only. Sign pairing, smoothing, and the Jacobian axis are not individually new. The original Gaussian ANTI2 distinction and its negative endpoint evidence remain as documented by Astra. Do not claim globally novel/unpublished math, revive the stopped pilot, or use its results as matched new-arm endpoints. If additional exact historical source is shown to implement the same full-support even objective and controls under this protocol, reject redundancy before implementation; do not alter A's formula/budget to evade that finding.

## 6. Exact affected modules/files

This design task owns only this new file, minimal relevant sections of CODEX_EXECUTION_PLAN.md, and EXPERIMENT_MANIFEST.json in this directory. The other four Astra Markdown files remain original evidence. The parent must hand off all seven files with this addendum's two explicit overrides.

Future implementation remains the already suggested `experiments/task_axis_even_views_20261116/` core/protocol/client runner and required entry files. Reuse W's frozen `task_view_gradient_residual_momentum_20261116/runner.py::serve`, `conditional_information_momentum_20261102_v3/guardian.py::request`, and its transitive BytesIO IPC functions. No historical source, tests, driver, results, tables, history or state files are changed here. New numerical outputs belong only to `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/` (N below).

## 7. Implementation requirements — smallest supported storage isolation

### 7.1 Why a new root is required

The frozen TV storage server is W `experiments/task_view_gradient_residual_momentum_20261116/runner.py::serve(attempt)`, lines62–142, SHA-256 `5ddcb68c206c6d78ad6c3cecf59441081aa9a9ddde3233ade4af55d158b32cfa`, matching published c4dc36bd. It appends commit records at `attempt/epoch_commit_ledger.jsonl` (102–109) and writes completion JSON at `attempt/runs/<key>/result.json` (110–116). No per-request output-root override exists. Old `protocol.py::results` (157–164; SHA-256 `c97acf7879b57c3e9ee49ef84633b74366a934e9e7c9787ed08cf7a2f19b2f64`) globs all old-root result files then adds 20 reused controls; `tables.py::export` (10–15; `d45c89dd79620d2cb2e16412107d9b383fa69f00cfe48dd4fc742f829fc13715`) requires exactly70. The old root currently has50 local results. A new-key mirror there makes71 immediately, even if all old bytes/prefixes survive. Prefix protection does not solve collection contamination.

### 7.2 Reuse serve with a different attempt, not a new storage implementation

During reviewed RUN only, create a **new cycle-root mechanical RAM owner** by calling the unmodified TV `serve(N)` directly in a persistent foreground tool process, using the project Python. Example invocation specification (not executed by this designer):

`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -u -c 'from experiments.task_view_gradient_residual_momentum_20261116.runner import serve; serve("/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1")'`

The original parent owns that current-conversation tool session and PID; the active native runner supervises its status and all mechanical workers. Use no shell `&`, detached process/session, launcher, daemon, AI/session invocation or orchestration framework. This is additional mechanical storage for N, not replacement/repair/restart of the protected old guardian. The explicit current user instruction authorizes this new isolated owner despite older generic no-new-guardian wording. Never call either `ensure_guardian`, old main/prepare/schedule/worker/retain/recover entry points, or an old driver action. Importing the frozen runner solely for `serve` is the narrow exception to the plan's generic entry-module prohibition; reviewer checks its transitive import graph, not a rewritten server. Its module `main` is guarded by `if __name__ == '__main__'`.

Root N must resolve to its own directory, outside every old attempt directory; no alias/symlink to an old attempt and no copied old guardian.json. Refuse to start a second owner over an existing owner receipt. Attach to the recorded live N owner after a handoff; missing/incompatible/dead owner means PAUSE, never an automatic restart over lost RAM. New `serve` generates its own socket/authentication, writes only N/guardian.json (0600), and starts with empty records. Keep credentials there, not in public manifests/handoff prose.

Use unchanged `guardian.py::request(N,command,...)` (SHA-256 `b819b0a482a382e91a792a27009c4e28084318be03398c7eb49f7e5cfe70f284`), which reads N/guardian.json. Every new register/allocated/commit/complete/paused/failed/get/recover/release request targets N; **no mutating request targets the old guardian**. The TV serve API, not the older generic guardian's serve, is required: TV supplies selected-state compaction, paused/recover and no retry-count cap. Old generic serve has a different retry contract and does not do the required full compaction.

The only lazy driver-module reach from TV serve is its recover branch's read-only stop predicate: TV protocol.stop_requested imports autonomous.stop_requested, which only reads both controls. Permit that audited predicate, not driver main/update_state/start/run/recover. `autonomous.py` SHA-256 is `19bbb5e075a3928997ba6118da9e420f27609554615556bc25c97a2cc0f819fc`; no top-level process launch was found. The new runner also checks both controls before each dispatch/launch/recovery. No old-root PAUSE/recovery/state write is authorized.

### 7.3 Transactions, recovery and retention across fresh agents

New job keys retain the registered full config/source/cycle identity, with the **seven-file** bundle hash. Worker snapshots keep current model/EMA/optimizer, loader/global RNG, epoch/global_step, counts, selected_models, selected_epochs, selected_validation, best scores, initial hashes, endpoint summaries and physical costs. No residual state. Release local IPC snapshot references after ACK. `send_memory`/`receive_memory` use torch serialization only through `io.BytesIO` and socket bytes, never a pathname; exact implementation is vendored `vendor_ga31/protocol.py:94–99`, SHA-256 `f71e0944cc08975597b2074ae4567c1fe817be830b6fdc67d8714f4662ad0c0f`.

Write new numerical batch/epoch files under N; send absolute N epoch_file paths in commits. Commit increments the owner's epoch by exactly1 and replaces its one continuation snapshot; wait for RAM_COMMITTED before advancing. At50, existing complete retains precisely selected_models/selected_epochs/selected_validation, drops continuation model/moments/RNG, and writes the sole canonical N/runs/<key>/result.json. Result payload contains numbers/strings/hashes, no model tensors. Thus journal, metrics and results are all cycle-local, with no mirror anywhere under the old root. The new exporter reads exactly30 N results plus the immutable20-control index; never call the old70-row exporter on N or modify it to accept additional keys.

For a technical pause the worker records paused/failed without discarding the last acknowledged state. A fresh runner reattaches via N/guardian.json, verifies source/design/job/epoch identity, and registers PAUSED jobs for continuation. Dead RUNNING/FAILED jobs use the existing serve recover command only after **all** recorded workers are dead and both stop controls permit recovery (serve lines85–98); never send global recover while any worker is live. Recovery writes only N/recovery_ledger.jsonl and may remove only N/PAUSE_TRAINING. Recovering epoch50 before complete reuses the full committed snapshot to finalize without retraining; already COMPLETE jobs cannot register again. Lost RAM is an explicit technical PAUSE/new reviewed attempt decision by parent, not a disk restore or silent rerun.

At runner completion, hand off a cycle-only numeric ownership receipt containing owner root/PID/socket identity (no auth secret), parent's persistent tool-session handle, key/status/epoch/counts, selected hashes, retained_bytes and source/design hashes. Parent explicitly accepts storage supervision **before** closing the runner permanently. The same mechanical process remains in the original parent's owned tool session through ANALYZE and fresh Astra/coder/reviewer/runner lifecycles; fresh agents get files/endpoint identity, not inherited context or a resumed AI. No storage-owner agent is created. If the tool lifecycle cannot preserve that owned process across runner closure, PAUSE before closure/admission and report the exact limitation; no detached fallback is allowed.

Retain at most one continuation per unfinished trajectory and two selected states per completed trajectory until the evidence decision. Parent then releases explicitly discarded **new-root** keys, retains only designated incumbent/eligible sets under the existing final-freeze policy plus the current cycle, and carries forward their owner receipts. Do not accumulate discarded tensors across unlimited cycles. Protected old states/owner remain untouched and counted in host RAM. The existing shutdown command is allowed only for a new owner with zero retained states and zero live workers; never shut down an owner holding an incumbent or any old owner. Host admission includes old/new retained tensors, pending allocations, workers and transient CPU/BytesIO copies; GPU and power gates remain unchanged.

## 8. Ablations required

Unchanged: AXIS-IID2 and ISOTROPIC-EVEN, plus the exact historical CLEAN and NDP-A contrasts. No projected/pair/TGA rerun, ANTI2 rerun, extra control, extra seed, extra epoch, HPO or B/C implementation. Existing direct correctness and reviewed full-batch smoke gates remain. Verify storage routing statically and during those existing gates; do not add a research matrix to test IPC.

## 9. Expected logging requirements

Keep the execution plan's batch/epoch/support/RNG/Adam/EMA/loss/AUC/LL/cell/task/domain/cost records unchanged. Add only cycle-local storage receipts: owner/source identity, distinct resolved root/socket/PID, ACK epochs, retained bytes/key inventory, and explicit parent handoff/release decisions. Before and after smoke/formal writes, record in N the old journal's **whole** byte length/hash and sorted old runs/result path+hash inventory; require exact equality, not unchanged prefix. The old control index stays byte-identical, so50 local+20 reused remain the same70 inputs. A read-only old-owner status inventory may be compared if available; never register a probe/new key there. Do not publish socket credentials or serialize tensor states to logs/JSON. Full30 coverage is required before the parent's scientific decision; designer has produced no results.

## 10. Acceptance criteria for reviewer

1. Parent's hard-completeness gate consumes all seven files. Historical mapping, receipt/commit hashes, support, caps/projection, RNG and loss-mass differences above must be traceable to actual files; no global novelty or efficacy inference is accepted.
2. Actual implementation matches the unchanged A/control equations and exact30-run/51750-update contract; source CLEAN/NDP-A/null identities, parameter activity, Adam/EMA and all existing correctness gates still pass. Reviewer remains distinct from coder; this design verdict cannot substitute for implementation review.
3. Source call graph shows only frozen serve at N and request(N,...), BytesIO-only tensor IPC, no old mutating request/output path, ensure_guardian, autonomous action or detached/AI launcher. New numeric export is30+20; old export inputs remain50+20 byte-for-byte. Frozen helpers themselves are unmodified.
4. Current-conversation parent owns the mechanical session across native runner close/handoff, with liveness/source/state-hash checks before reattachment. Complete compacts selected states; release/shutdown respect retained states; technical recovery does not lose or duplicate epoch updates. Live supervision/resource readiness and actual smoke remain RUN gates, not designer claims.
5. Bundle identity uses exactly seven cycle-relative files: the original six plus DESIGN_COMPLETENESS_ADDENDUM.md. Sort `[relative_path, SHA256(file_bytes)]` pairs lexicographically by path, UTF-8 encode their JSON array with `separators=(",",":")` and no appended newline, and SHA-256 that payload. Store the value in a separate run/review receipt, never inside any of the seven files. Source/job/review receipts bind this value. The parent finalizes it after all design edits.
6. No MODEL FILES, tensor/optimizer/checkpoint/pickle spill, training/Test execution, history mutation, driver action, commit or push by this designer. Both stop-control observations were ACTIVE/manual_stop_requested=false/test_accessed=false with absent sentinels; every future dispatch must recheck.

## 11. Conditions that would falsify the hypothesis

Keep all registered directional falsifiers. Failure to improve CLEAN rejects the primary endpoint prediction at these fixed settings. Recovery over NDP-A without beating CLEAN supports damage removal only. Failure to beat NDP-A weakens/rejects the registered odd-component endpoint account; failure to beat isotropic gives no demonstrated task-axis value; comparable AXIS-EVEN/IID2 endpoints give no demonstrated pairing benefit. A CI crossing0 is unresolved rather than proof of equality; LL/pooled/online/late-epoch gains cannot rescue failed selected-EMA Macro AUC. Analyze every domain/task, population, baseline and stage; do not equate small domains with difficult ones. Parent labels evidence STRONG/MEDIUM/WEAK and preserves contradictions.

Independently, exact historical equivalence would falsify the local non-redundancy finding and block implementation. Inability to isolate all new writes or preserve supervised RAM across fresh-agent handoffs is an engineering FAIL/PAUSE, not evidence for or against efficacy. No scientific formula/budget expansion is authorized as a workaround.
