# CODER IMPLEMENTATION REPORT — execution and accounting

Implemented the approved design in `experiments/ema_next_20261111/{__init__,protocol,decision,state,scheduler,run,report}.py` and `tests/test_execution.py`. Scientific operators, trajectory/evaluation and science tests are owned by the separate science coder. Existing model, data, loss, optimizer and historical result files were not edited.

The execution package resolves the pinned plan, records its one C risk-tie correction without changing raw planning files, verifies source/data identities, prepares source panels, and expands the 740-epoch mandatory core. Its finite controller handles parent B/C collection, all MATCH shorts, conditional RAW/full A/AB, conditional KD controls, one inherited Gaussian family, global freeze, Test, tables and RAM release. Decisions require complete five-seed records and use the prescribed cell guards and numerical gates. The no-feasible-KD branch records the best unprotected result and does not amplify it. C freezes one shrinkage with its matched PRED/PARAM control. Full MATCH/RAW ties prefer RAW.

RAM state uses an independent guardian with incremental immutable packets, descriptor references, old/new epoch commits, worker ownership, bounded replay and no model files. Parents retain E20/E35 anchors; losing MATCH states are released before RAW; E35 shorts retain only scalar outputs; final compaction removes unused prefixes and teacher initialization references after consumers finish. Whole-block forecasts enumerate best packets, prefixes, anchors, sparse history, live/export/worker copies, prior Clean/Gaussian endpoints and conditional phase bounds. Runtime admission checks each GPU independently, host RAM, the 4 GiB guardian predicate and the 7 GiB physical tensor bound. Teacher probabilities are cached in RAM once per teacher/seed.

The CLI supports audit, prepare, smoke, execute/resume, status, analyze, freeze, evaluate, report and release. Execute requires an independent hash-bound review receipt and both hash-bound GPU smoke receipts. Smoke performs source equivalence, a full real epoch and full registered readout sweep, A and KD steps, and a worker exit/recovery roundtrip. It does not select scientific candidates.

Final export includes all committed curves, seed/cell endpoints, paired original values and deltas with t4 intervals, training/Test costs, all actual decisions and untriggered branches, HPO selection, source provenance and artifact hashes. Gaussian-parent Clean students have their own condition and explicit Gaussian-parent O/F comparison. Formal comparisons cover each completed condition against all three published models. AB has separate independently selected and E50 same-time Validation interaction tables. The generated report contains numerical results for the orchestrator's scientific interpretation.

## Checks

Command:

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/ema_next_20261111/tests
```

Result: **16 passed in 11.87 seconds**. One inherited PyTorch TF32 API deprecation warning; source TF32 settings remain unchanged.

Coverage includes finite matrix/history counts, five-seed completeness and equality boundaries, strict checkpoint versus HPO tie behavior, immutable storage accounting and prefix release, failed commit preservation, independent GPU admission, and the ten science checks listed in `CODER_SCIENCE_REPORT.md`. Package syntax checks also passed.

No scientific training or research-result analysis was performed by this coder. GPU smoke and the prescribed training queue remain runner work after independent PASS. No unrelated regression suite, dependency change or additional experiment was introduced.

## Runtime revision: CUDA smoke initialization

The first runner smoke stopped before model execution because a fresh process called `torch.cuda.reset_peak_memory_stats('cuda:0')` before CUDA device initialization. Added `torch.cuda.set_device(device)` immediately before the reset. A fresh-process focused check confirmed device 0 selection and a zero-allocation peak-stat reset. This revision changes smoke initialization only; no training or scientific record existed at failure.

## Scheduler-only revision: pending CUDA reservations

The first five ML workers started on GPU 0 before their delayed CUDA allocations were visible. Device use reached 21,941 / 24,576 MiB. Each worker subsequently reported 1,346 MiB process GPU use and 934.894 MiB peak tensor allocation. Healthy workers and scientific records were preserved.

Added the external adapter `experiments/ema_next_admission_20261111.py`; the scientific package and its frozen hash `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6` remain unchanged. It discovers exact-attempt workers by PID/start/device/job before guardian registration, rejects duplicate pending jobs, and reserves each worker's unallocated GPU remainder. A missing per-PID reading reserves the full predicted amount. Measured process and nonallocated-memory high-water values persist across controller recovery. Host and tensor forecasts include discovered pending processes. The adapter requires its own hash-bound independent PASS and records old/new controller identity.

Focused checks: `python -m pytest -q experiments/test_ema_next_admission_20261111.py` — **2 passed in 6.05 seconds**. Live `--inspect` loaded no models and launched no processes; it computed ML/KR candidate reservations of 1,479,829,504 / 1,668,911,616 bytes from current measured overhead. Controller-only replacement remains runner work after review.

## Scheduler-only revision 2: parent cohort tensor reservation

The original runtime admission started with the complete five-seed forecast and then added every active worker's closure again. For KR this rejected a second parent even before completed KR states accumulated. The external adapter now reserves the actual parent cohort. The full-five-seed prepare check, scientific package, active workers and guardian remain unchanged. Nonparent jobs, mixed cohorts and registered workers absent from process discovery retain the original admission path.

All formula terms below are bytes. `P` and `S` are the proposed dataset's total and shared parameter counts; `FP32P=4P`. The immutable closure `C=4*(sum(best)+sum(prefix)+sum(anchors))` comes directly from that dataset's prepared layout: independently selected readouts, prefix winners, and E20/E35 anchors including sparse F history. `L=4*layout.active` covers theta, optimizer m/v, F and active history. `M=1 MiB+200*runtime_buffer_bytes` is the original metadata allowance. The incoming old/new overlap is `I=L+4*(12P+3S)+M`. `B=status.payload_bytes` is the whole actual guardian payload, including previous datasets, completed seeds and current active states; nothing is subtracted. `n` counts every exact-attempt discovered pending/active parent process plus the proposed parent.

The guardian predicate is `G=B+n*(C+2L+M)+I`. The physical forecast is `G+n*(C+L+5*FP32P+M)+I`: each worker retains its immutable closure and live state, the original five-vector scratch allowance, and metadata. The final extra `I` covers the worker/export copy alongside guardian-deserialized incoming tensors during IPC. The existing 4 GiB / 7 GiB caps still apply. Original current-payload safety floors are retained by passing this cohort peak through the original admission function with zero additional active-count increment; the cohort has already counted those processes. GPU process high-water reservations, missing-reading full reservation, duplicate-job rejection, host checks and power limits remain active.

With the prepared conservative prior-dataset endpoint bound `B=0.563649088 GiB`, KR has `C=0.595167309`, `L=0.090203471`, and `I=0.170740344 GiB`. Four parents yield `G=3.840592690 GiB`, physical `6.876993701 GiB`; five yield `4.617143504 GiB` / `8.369959682 GiB` and are rejected. Runtime uses actual `B`, so retained state accumulation can reduce concurrency. Finishing workers release worker/live copies before additional parents are reconsidered.

Focused checks: **4 passed in 2.93 seconds** in `experiments/test_ema_next_admission_20261111.py`, covering pending GPU reservations, duplicate pending jobs, four-parent admission/five-parent rejection, retained-payload accumulation, original prepare immutability and nonparent fallback. Read-only live inspection observed controller PID 1830466 (start 19455337), four A_MATCH workers and guardian payload 668,556,688 bytes. No process was launched or stopped by these checks.

Adapter SHA-256: `81a9ab3b90c5c875e2631288bb4eb71fb7335f44a1c86921d9835f0bc55f5daa`. Scientific SHA-256 remains `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`. A reviewed controller replacement must supply `--previous-controller 1830466`; the adapter records the new controller identity and archives the preceding revision receipt before writing the new one. Independent adapter review and runner-controlled replacement remain required; no scientific test or training was rerun by the coder.

The exact previous adapter is preserved as `scheduler_revisions/admission_07a2348e.py`, verified against SHA-256 `07a2348ecf269c8a6843351a2721fd9e6e6788a4636b5cc11e7aef1ffbcb2de4`. `scheduler_revisions/07a2348e_to_81a9ab3b.diff` records the complete transition to this revision.

## Scheduler-only revision 3: frozen-decision JSON reentry

Controller PID 1859758 stopped during dataset reentry at `gate_write(PARENT_BC)`. Read-only recomputation from the committed parent curves showed Python equality false but `saved == json.loads(json.dumps(recomputed))` exactly true, including all metric/config/selection values. Fresh `points` used integer seed keys; JSON reload used string seed keys. The existing ML PARENT_BC file SHA-256 was `f982a211b8af213c454c5ee8f4d4a966046cbcbc07b5ecb7a98dbabdd9b6fd15`.

The external adapter now supplies the shared runtime `scheduler.gate_write`: it compares JSON-normalized values, returns without rewriting an existing equal decision, and raises the original error for any actual changed value. New decisions still use the original JSON writer. This fixes reentry without changing scientific computations, selections, frozen decision files, workers or guardian.

Focused check: `python -m pytest -q experiments/test_ema_next_admission_20261111.py -k frozen_gate_reentry` — **1 passed, 4 deselected in 5.18 seconds**. The check covers int-seed JSON roundtrip, identical bytes/mtime on repeated equal entry, and rejection of a 0.00000001 metric change while preserving the frozen file. No GPU/Test or research execution was performed by the coder.

Adapter SHA-256: `3b2e089e89356fabc0012fb81a13f36f16c81df12e732aa7bf22a83735ce501e`; scientific SHA-256 remains `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`. The exact prior source is `scheduler_revisions/admission_81a9ab3b.py`, verified against its full preceding hash; `scheduler_revisions/81a9ab3b_to_3b2e089e.diff` preserves the complete revision. Independent review and runner-controlled resume follow this change.

## Scheduler-only revision 4: count active parent closure once

At KR E37/E38/E36, the previous formula included the three active parent closures in actual guardian payload and reserved their complete future closures again. It estimated a fourth parent at 4.641049 / 7.677450 GiB, despite available GPU capacity. Read-only inspection identified the first KR admission before any KR worker: job `P.C.kuairand.2022`, time `1794386059.6129935`, frozen preceding ML payload **288,103,704 bytes**. Its canonical record SHA-256 is `9730465a2dd7871437d09a8aab7615f3f4ce38531a2dfdf66c4f81a67f3a5b3d`; ML DATASET_COMPLETE SHA-256 is `b6a4ae9c8648adfde4091aa84d05cb7f5131a237d00f92c6ca9a79d205ff6af2`.

The small `scheduler_revisions/KR_PARENT_BASELINE.json` receipt binds these records, their time/job/payload, this attempt's absolute path, scientific hash and guardian PID/start/boot identity. Its SHA-256 is `5c2a5639e5f55ccac8417f1ac7ae81ac1a4a5ecab1f2cefca8b7deeefbd8cf35`. Before use, the adapter verifies those bindings, the original admission's empty worker list/count, the completed preceding dataset, and that the only current KR rows/processes are initial Clean parents. A retained unfinished row absent from discovery and distinct from the proposed job prevents baseline use. Other stages or mismatching evidence keep the previous formula.

For this verified phase, `B=288103704`, `d` is the number of completed KR parents without a discovered process, and `n` counts every discovered pending/active process plus the proposed parent. A completed row whose process has not exited is covered by the full active allowance instead of counted twice. The bound is `G=B+d*(C+M)+n*(C+2L+M)+I`; physical remains `G+n*(C+L+5*FP32P+M)+I`. Terms retain the revision-2 object mapping. Each completed parent keeps its whole immutable closure allowance, while active parents receive the entire future closure/live allowance. Original current-payload floors, 4/7 GiB caps, pending GPU reservations, duplicate detection, host/power checks and nonparent handling remain active.

| Completed d | Cohort n | Guardian GiB | Physical GiB | Tensor admission |
|---:|---:|---:|---:|---|
| 0 | 4 | 3.545261085 | 6.581662096 | Allow |
| 0 | 5 | 4.321811900 | 8.074628077 | Reject |
| 3 | 2 | 3.780591071 | 5.384161748 | Allow fifth seed |
| 4 | 1 | 3.600184128 | 4.487339638 | Allow last seed |

Focused adapter checks: **7 passed in 5.06 seconds**. New checks cover the four cases above through the original admission caps, pending cohort counts, retained completed closures, receipt/data hash mismatch, unknown unfinished process, and nonapplicable-stage fallback. Existing pending-GPU and frozen-decision checks also passed. A read-only live call verified the receipt and produced d=0/n=4, actual payload 1,522,809,672 bytes, G=3,806,695,104 / physical=7,067,005,864 bytes. No process was started or stopped.

Adapter SHA-256 is `60e76abc3231c9846d3c97ac2d8d894a98f1df8b527f5c5eb1c398fae3bf6e84`; scientific SHA-256 remains `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`. Exact prior source `scheduler_revisions/admission_3b2e089e.py` and `scheduler_revisions/3b2e089e_to_60e76abc.diff` preserve the transition. Independent review and runner-controlled controller replacement remain outstanding at coder handoff.

## Scheduler-only revision 5: bounded Clean A cohort

The KR A_MATCH stage admitted one worker and rejected a second because the original admission applied the entire five-parent peak to every family, then added active-worker allowances. The actual A state graph has only O/F plus at most one frozen B, no new anchors or C policies, and one history window. Its FP32 bound is selected O/F/B including B's parent F at most 5P, inherited prefix at most 4P, live theta/m/v/F 4P, and sparse history at most hP. With h<=6, the total is at most 19P and is covered by the existing `E=20*FP32P+32 MiB` allowance. E20→26 completion retains this state for possible continuation; E35→41 completion immediately drops it. Continuation stays under the same bound and E50 completion retains selected readouts only.

For Clean A_MATCH/A_RAW only, the adapter now removes the unrelated five-parent floor and keeps the original current-payload branch: `G=B+(2n+1)E`, physical `B+(4n+1)E+nD`. `B` is the full actual guardian payload without subtraction; `n` includes every discovered active/pending process plus the proposed job; `D` remains the prepared entire parent closure plus live/history bound, covering full parent retrieval during fork initialization. The remaining worker/export/IPC allowances and 4/7 GiB caps are preserved. Completed E20 states accumulate in B, while E35 release reduces B naturally.

The branch requires every proposed/discovered job to be same-dataset Clean A_MATCH/A_RAW, with ordinary 20→26 or 35→41 ranges, or the existing 26→50 continuation. It rejects custom readout menus, nonregistered/non-B_SPLIT frozen B, changed P/steps/rho, history bounds above six, mixed families and missing discovered registered workers. The pinned trajectory implementation supplies the O/F menu and removes anchors during the fork. All nonapplicable stages, including KD and Gaussian, retain their previous admission behavior.

At inspected B=1,367,218,872 bytes, KR E=162,694,752 and D=735,911,272 bytes: four A workers forecast 2.637013458 / 6.590666942 GiB; five forecast physical 7.882122904 GiB and are rejected. A later read-only live check with B=1,425,345,176 and one existing worker computed the second-worker forecast G=2,238,818,936 / physical=4,361,420,488 bytes.

Focused adapter tests: **9 passed in 3.71 seconds**. New checks cover four-worker admission/fifth-worker rejection with unregistered pending processes, exact inclusion of retained E20/released E35 payload, allowed continuation and mixed/nonapplicable fallback. The earlier GPU, parent-baseline and frozen-decision checks remain passing. No scientific execution or process launch/termination was performed by this coder.

Adapter SHA-256: `1755e97d6bc087d555f48145cb3bc3c5a0c70b60afa5bfb263b678ae82efc346`; scientific SHA-256 remains `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`. Exact prior source `scheduler_revisions/admission_60e76abc.py` and `scheduler_revisions/60e76abc_to_1755e97d.diff` preserve the change. Independent review and runner-controlled controller replacement follow coder handoff.
