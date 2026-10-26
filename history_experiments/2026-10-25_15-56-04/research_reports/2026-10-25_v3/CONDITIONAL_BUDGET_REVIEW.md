# Independent review — conditional cost proposal helper

**Verdict: PASS WITH FIXES. Blocking fixes are corrected. P2 budget proposal is safe for orchestrator adoption.**

This is static source review plus inspection of the corrected cost-only proposal metadata. No new quality records or arrays were inspected, no GPU command was run, and no experimental source was modified.

## Requirement-by-requirement verification

| Requirement | Verified implementation | Result |
|---|---|---|
| Inputs and proposal safety | Reads cost journals, fixed role counts, config/status/manifests/locks and process/scheduler journals. No epoch-quality, prediction or outcome input path. Writes a new proposal only, refuses existing paths and BUDGET_FREEZE.json. | PASS |
| P2 exact queue | Requires exactly CAL/OWNER/GEOM × two datasets plus KR R7, seed3101. READY requires every retained trajectory reach epoch11 and all required rates/residual evidence exist. | PASS |
| Physical counts | R7 suffix27 events, <=81 H-block branches /1620 KR shadow updates; reused first initialization/main block excluded from new physical work. CAL/OWNER each72 formation steps and24 complete six-head read passes, GEOM zero meta plus one geometry fit and24 passes. Cache, validation, save and measured startup/restore costs included. | PASS |
| Remaining work | Subtracts actually observed normalized units, retains conservative unmatched ceilings; COMPLETE endpoint trajectories owe no fictitious extra shortlist/cal work. Independently scheduled formal exposed evaluation remains separately reserved. | PASS |
| Rates | Uses actual per-unit maxima with n, origins and explicit transfer labels, no invented P90. Missing required rates produce unresolved/null reservation. GEOM/projection work within main/cache/read/eval timers is retained. | PASS |
| Shared and independent deployment | Shared warmup and R7 firstblock paid once in research physical work. Standalone estimates separately add necessary warm/native prefixes and R7 initialization. Exclude research final scoring/deletion/inference and terminal simple feedback. Estimates explicitly exclude unmetered process residual and inseparable work remains retained. | PASS |
| Future formal counts | Fixed KR20 methods plus4 fresh warmups from immutable BEST_SIMPLE lock. Warm H1 feedback96 steps; PB independent245 branches; HB4900 shadow updates; HISTORY physical237 branches plus shared8 initialization. FreshSTGU50 uses explicit3×capacity rate transfer, not a fabricated observation. Future retained revision requires a fragment per kind and takes applicable maxima. | PASS, proposal generation only |
| Process wall | Exclusive leaves only; inclusive parents excluded. Matches canonical process START/EXIT, subtracts leaf seconds, includes measured scheduler import gap or explicitly estimated additional residual allowance. Reports unresolved intervals. | PASS AFTER FIX |
| Margin and boundary | 1.35 × remaining component plus process reservation, plus separate reporting allowance. Serial workload/process seconds are not GPU-hours, energy, statistical bounds or training deadlines. | PASS |

## Bugs and corrections

**MAJOR, corrected by helper owner:** conditional CLI EXIT lacks pid. Original overhead() required exact (run_id,pid), omitting every P2 residual. Current code pairs such an EXIT only to the unique open START for that run, pops the matched entry, records pairing provenance and supports formal journals. The corrected proposal contains7 missing-pid pairings, zero open starts and zero unmatched exits. The older proposal remains archived rather than overwritten.

**MINOR conservative accounting issue, corrected:** legal shortlists or selected calibration work may be below theoretical ceilings. COMPLETE training no longer appears to owe additional fictitious probe work; independently scheduled formal exposed evaluations remain separate.

## Remaining risks and release scope

Observed maxima with1.35 margin are forecasts, not proven runtime upper bounds; contention, solver/allocator/data/IO changes can exceed them. Unmeasured deletion/inference terms are declared scaled allowances. Standalone component/prefix estimates are partial and cannot support precise minimal-deployment claims. The32-step retrospective replay and old-STGU LL scoring need their own approved budget if not already separately covered.

This receipt releases only adoption of the corrected P2 proposal for the seven already reviewed runs. It does not freeze methods, alter source approval, authorize formal training, or impose a timeout. Formal proposal generation is mechanically safe, but formal execution still requires the parent's method/confirmation lock, complete positive component budget and existing source-scoped approval.

## Source and corrected proposal identities

- Helper SHA256: `9f76c0ee90d10ea7ef510a485863f0cc2119b20a50298d766c5d984b18372317`
- Proposal: `BUDGET_PROPOSAL_P2_RESIDUAL_CORRECTED.json`
- Proposal SHA256: `1f45a1f08af0905e1b142529d0b05b8b0c40650742e00b0f8197d82e51eff3d9`
- Composite experiment source: `31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276`
- Corrected remaining reservation: `2884.7477526271578` seconds; cost-only forecast, not timeout.
