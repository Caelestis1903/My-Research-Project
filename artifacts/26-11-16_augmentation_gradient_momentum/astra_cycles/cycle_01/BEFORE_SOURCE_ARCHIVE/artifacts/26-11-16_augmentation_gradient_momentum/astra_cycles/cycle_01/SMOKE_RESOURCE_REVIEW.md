# REVIEW REPORT — actual smoke resource admission

1. **Verdict: FAIL — one MAJOR blocker; FORMAL30 must remain unstarted.** This supplements the accepted R1 static/CPU review; it does not invalidate scientific math or the recorded correctness identities.

   Accepted/current source: `2ee2bdddb67861e65449cb3577c1d835f13395000a90b4ac101508c855f7b1a1` (all79 source-file bytes and canonical aggregate independently verified).
   Actual `attempt1/SMOKE_REPORT.json` SHA-256: `4d50a42e5a56e0c2df2392b18ae40cec8768714f4315b63c6148efe7113521f1`.

2. **Requirement-by-requirement verification**

| Requirement | Finding |
|---|---|
| Full40960 FP32 / source identities | Six recorded steps each have one Adam/EMA update. Full/tail signature equality independently checked for null vs CLEAN and delegated one-sided vs NDP-A. Tails ML21927/KR40052. Reusable correctness evidence. |
| Historical isolation | `old_before == old_after` exactly within the actual smoke receipt; no new inventory dump or historical write. |
| Independent memory AND power<85% | `choose_device` implements the strict conjunction correctly. Positive memory reservations remain; no memory blocker found in this narrow review. Power-reservation semantics fail below. |
| Projected process-power reservation | **FAIL / RESOURCE-1 MAJOR.** A zero change in short-window board telemetry cannot be transferred as a zero incremental budget to another loaded card. |
| Actual monitoring | Smoke's sampled maxima pass the actual85% check. Schedule resamples board load and can block later admissions, but cannot thereby validate the already admitted worker's zero projected budget. Only memory reservations are updated from reported worker peaks. |
| Owner / source / Test | Receipt binds original foreground session59365/PID1570727 to source2ee2; no owner interaction. Formal dispatch/launch/progress/commit files absent. Both stop controls ACTIVE/manual_stop=false/Test=false; sentinels absent. Test stays closed. |

3. **Bug/deviation: RESOURCE-1 / MAJOR**

| Dataset | Arm order AXIS-EVEN / AXIS-IID2 / ISOTROPIC-EVEN: pre→peak W | Dataset reservation |
|---|---|---:|
| MovieLens | 29.78→125.21 /157.24→195.61 /195.61→195.61 |95.43 W|
| KuaiRand |162.15→162.15 /206.31→206.31 /205.36→205.36 |0 W|

`runner.py::PowerSamples` (498–520) samples board draw. `smoke` starts this window after model/input allocation (540–550), computes `max(0, maximum-first_sample)` (559–560), and takes the dataset maximum (582–584). The subtraction is arithmetically correct; it does not identify transferable process power. The evidence cannot distinguish sensor lag, previous-arm activity or other load as causes of the high first sample.

`schedule` (330–332,394–396) feeds this0 W unchanged into `choose_device` (149–160). With sufficient memory, supplied GPU0 draw257–283 W projects257–283 W, below `.85*350=297.5 W`; thus a new KR worker is admitted without budgeting its load. A CPU-only fixture executed the actual AST-extracted function: at283/350 W,0 W admits;95.43/206.31 W reject. Memory in that fixture was synthetic and admissible; this is **not** a claim that a real GPU0 launch or power excursion occurred. Later actual sampling cannot repair the initial projection.

4. **Corrections made: none. Minimal correction for coder/reviewer**

Change only the reservation interpretation in `runner.py::schedule`: use the maximum measured board peak per dataset as a conservative admission budget, at least the old increment: ML195.61 W/KR206.31 W. Label this a conservative observed board-power budget, not an isolated process increment. Keep original telemetry untouched. The existing strict power/memory predicate, pending handling and actual resampling stay in force. No new sampler, watt floor, algorithm, precision/batch change or smoke rerun is needed for that bounded correction.

For KR this projects463.31–489.31 W on the supplied loaded GPU0, excluding it, and219.31–226.31 W on supplied idle GPU1; live memory/host/power still control admission. The same correction also avoids treating ML's95.43 W as a calibrated per-arm bound.

**The arithmetic edit alone is insufficient:** runner source is hashed. `protocol.py::validate_attempt` (218–239) requires the immutable manifest/smoke source to equal current source; `check_review` requires a fresh exact-source review. Preserve all original receipts, including the smoke bytes above. Append an explicit reviewed superseding run/source/admission receipt binding original evidence, revised source, fresh PASS review, unchanged scientific dependencies/design and the SAME owner root/endpoint/PID/session. Use its effective formal manifest consistently in `validate_attempt`, `validate_continuation` (259–263), and `runner.py::require_storage` (91–99); retain original owner bootstrap identity and expose the revised verified source in handoff. Formal jobs must record the revised actual source. Do not broadly relax hash checks, rewrite receipts or replace/restart owner1570727/session59365. Frozen serve itself remains unchanged. These necessary provenance touches are confined to runner.py/protocol.py plus appended review/run receipts.

**Existing identities can be reused** after a fresh narrow independent review explicitly binds their old smoke/source hashes to unchanged numerical/worker paths under the revised admission code. Existing validators do not currently allow such reuse across aggregates. Reuse retains original results and does not assert that the original0 W was valid or grant formal execution approval.

5. **Remaining risks**

No observed actual85% breach is established. Existing short-window peaks are measured budgets, not guarantees of every future peak; actual monitoring remains required by the approved plan. This review does not clear future live admission. Scientific formulas, FP32 and full batch are unaffected by this finding; prior R1 acceptance remains source-specific.

6. **Experiments safe to run: NO, pending the bounded fix and fresh review.** Return to the original parent; keep formal30 unstarted and the existing owner alive. `test_accessed:false`.

Only these two requested artifacts were written. No GPU query/run, smoke rerun, source/design/shared-state change, storage IPC, agent/session dispatch, commit or publish occurred. No broad regression or fresh storage-liveness check; parent permanently closes this reviewer after handoff.
