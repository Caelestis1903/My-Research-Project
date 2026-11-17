# Cycle04 execution history

## Operational restart

The first attempt lost its foreground RAM owners after a session interruption. Its 275 committed epochs and 7,733 useful updates remain historical numeric evidence and cost; they receive no completion credit in attempt2. Training state was never recovered from disk. The stale allocation flag in that attempt's recovery was also found to release pending startup protection before the new worker registered. A new DESIGN, implementation, distinct code review, four real GPU engineering cases and distinct resource review approved the PID/registration-attempt repair before attempt2 started. Model, objective, optimizer, data and scientific budget were unchanged.

The retired attempt's physical work is recorded separately in `RETIRED_ATTEMPT_COSTS.json` and `RETIRED_PHYSICAL_ATTEMPTS.csv`. It had forty formal worker processes: twenty-four PAUSED, fifteen FAILED and one failure observed only by its supervisor. Known returned worker durations total 10,523.255015449 seconds; one failed terminal duration is unknown. Supervisor process-wall durations overlap worker durations and must not be added to them.

## Attempt2 first formal segment

The initial formal controller stopped at 771/1,000 formal committed epochs and 29,021/34,500 useful updates. Thirteen trajectories were complete; five retained partial states and two unstarted jobs remained. A KuaiRand COMPLETION-BCE-025 seed2026 worker failed with `NVML measurement failed code=3`. Its last accepted epoch was40. The failed worker's cleanup re-raised the same sampler error before its terminal profile/cost receipt was written. The supervisor observed 6,295.269635677338 process-wall seconds; this is not an exact returned training-call duration.

The parent directly accepted all eighteen registered RAM states and their last ACK/transaction bindings. A fresh independent reviewer approved the unchanged-source recovery. A fresh native Luna/high runner then resumed the five partial trajectories and started the two remaining jobs. Completed trajectories were preserved. The runner and reviewer contexts were distinct and were closed permanently after their handoffs.

## Attempt2 recovery01 technical pause

The recovery controller ran for 1,418.0120804079343 elapsed seconds and returned PAUSED. Fifteen trajectories were complete at the end, with 889/1,000 formal committed epochs and 32,251/34,500 useful updates. Engineering commits are excluded from these totals. Seven physical workers had started and exited: six exit0 and one exit1. The sum of observed process-wall durations was 6,973.316387176514 seconds; these processes overlapped. Known returned calls total2,876 across the segment, with one failed worker's partial count unknown.

MovieLens COMPLETION-BCE-025 seed2026, PID2619341, failed in resource-profile marking. Both the sampler check and cleanup reported `NVML measurement failed code=3`. Its supervisor observed 689.6625649929047 process-wall seconds; the worker did not return a complete call count. Four other active trajectories paused safely. All seven workers and the controller exited; the original attempt2 RAM owner PID2476891 remained alive. The fresh raw runner report is `attempt2/FORMAL_RUNNER_REPORT_RECOVERY_01.md` with its JSON counterpart. No efficacy analysis, final handoff or full-table export was performed by that runner.

Subsequent direct reading of this failed worker's compressed batch logs found400 valid returned batch records: twenty at each of E1–E20, and an empty readable E21 log. Thus the missing terminal count does not erase those400 observed calls. Combined with2,876 returned calls in the other six terminal records, at least3,276 returned calls are evidenced for recovery01, compared with3,230 new useful committed updates. The difference of46 includes replay or work outside the newly committed frontier; any call that failed before returning a numeric record remains unknown. Process-wall and training-call durations remain separate.

The five unfinished last-commit points are KR MASS-SCALE-025 seed2025 E49; ML COMPLETION-BCE-025 and MASS-SCALE-025 seed2025 E26 each; ML COMPLETION-BCE-025 seed2026 E20; and ML MASS-SCALE-025 seed2026 E18. Thus111 formal epochs and2,249 useful updates remain. The corresponding ACK files exist. A new direct RAM/ACK acceptance and distinct recovery review are required before the next runner starts.

Two fresh telemetry observations on each card succeeded for handle, memory, power, enforced power limit and process-list queries; their raw return codes are in `attempt2/NVML_RECHECK_02.json`. NVIDIA defines code3 as NOT_SUPPORTED. The historical failure did not record which query returned it, so those successful rechecks do not locate its cause. [NVIDIA NVML error definitions](https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceEnumvs.html).

## Current continuation boundary

The user has authorized completion of Cycle04 and exactly one additional full Cycle05, followed by termination. Cycle06 is not authorized. Test remains sealed. During recovery preparation, Stop That Shit Guard blocked identity/publication hashing with HASH_NOT_AUTHORIZED under ARMED/change, hash=deny. The partial independent review therefore certified only static/numeric fields and supplied no RUN release. The user subsequently explicitly enabled hash=allow.

After that permission change the original parent directly verified all20 live states: fifteen compact selected pairs and five exact whole-epoch continuations. It also verified all889 ACK/epoch/transaction/batch identities and frozen source/design/engineering bindings, with20 retained states totaling266,400,952 tensor bytes. `attempt2/PARENT_FORMAL_PAUSE_ACCEPTANCE_02.json` records the acceptance at2026-11-17T10:49:21.795210+00:00; `attempt2/NVML_RECHECK_03.json` records four fresh successful telemetry samples. A fresh independent recovery reviewer receives those actual results before any replacement runner starts. The old runner and partial-review contexts remain permanently closed.

## Completed recovery02 and final RAM acceptance

Fresh independent Dirac01a1e920-d526-7303-91e0-dadf1ecef937 verified the current source/design,889 committed bindings and full20-key frontier, then returned PASS with no blockers in RECOVERY_REVIEW_02. Its context was closed permanently. The parent separately verified the fresh native DEFAULT runner Hegel01a1e92b-2bba-7b82-8451-832dbb39770a as gpt-6-luna/high under the original conversation before giving GO.

ControllerPID2762603/toolsession6629 resumed five retained trajectories and completed the remaining111epochs/2249 returned calls. It reported FORMAL_COMPLETE, exit0, foreground segment1073.0433918869821 seconds, with no new failure or unknown partial call. The parent observed five actual CUDA workers during startup (GPU0=2/GPU1=3); all completed and exited. This is supervised foreground mechanical work, not a detached AI session.

RUNNER_HANDOFF and all18 CSV exports plus full resource audit completed. The parent directly accepted40 selected model hashes/epochs/validation,1000 ACKs/34500 useful updates and20 compact selected pairs totaling191017920 tensor bytes. Raw RECOVERY_02 reports distinguish the two older NVML failures; after reporting, Hegel was permanently closed and N was frozen for numeric archiving.

Parent batch-ledger counting found34640 returned calls,140 outside the current committed batch files, and81 physical diagnostic points versus80 useful points; physical extra sweeps243/virtual steps324 versus240/320 useful. Both missing worker terminal durations remain unknown. The recovery02 raw report's snapshot-transfer seconds0 was a field-reading error:111 ACK commit/status/get workflow records sum61.909074446884915 seconds. COST_ACCOUNTING_CORRECTION.json binds the exact original report and COSTS.csv; historical archive bytes are preserved.

The completed archive contains6611 members (4243 current numeric,1020 exact controls,1348 retired numeric/source),4077601426 uncompressed bytes,740197954 compressed bytes in89 parts. Full archive SHA256 is5841209454b679425a25e910042e90056553a03521507c30b9287e69b406d5e2. No tensor or private storage credential file is archived. Parent analysis and figures are complete; the registered efficacy category isMIXED. Report/context evidence review and actual main publication precede the one additional authorized Cycle05.

## Post-acceptance selected-model availability loss

Darwin's final evidence review passed numerical, statistical, scientific and full archive checks but raised MAJOR B1: owner2476891 was absent at2026-11-17T12:02:24.146007+00:00, contradicting present-tense retention statements. Parent procfs checks confirmed absence and toolsession67406 returned exit143, consistent with the conventional128+SIGTERM status; the sender, exact exit time and trigger were not established. TV storage process1025015 remained visible. The earlier40-model acceptance at11:21:33.883200 remains historical evidence, not a current availability claim.

SELECTED_RAM_AVAILABILITY_UPDATE.json records that the selected models are now unavailable and Test eligibility isfalse. Completed metrics,1000epoch identities, frozen source/controls/archive andMIXED are unchanged. Reports and all seven maintained contexts disclose the changed availability; the frozen N and archive remain intact. No model was fabricated, reconstructed from disk, automatically retrained or assigned a replacement owner. A scoped fresh evidence R1 review checks this disposition before publication. The next independently initialized Cycle05 remains authorized after C04 publication; it cannot consume the lost models as a warm start.
