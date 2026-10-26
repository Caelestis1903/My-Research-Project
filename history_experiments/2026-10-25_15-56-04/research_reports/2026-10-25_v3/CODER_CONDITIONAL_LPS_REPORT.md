# Conditional LPS implementation report

Implemented the approved `CONDITIONAL_DESIGN_ADDENDUM.md` without changing the frozen 71-file base (`3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2`). This coder has not launched a formal/P2 trajectory or analyzed research outcomes. Independent conditional review and complete-package cost admission remain required.

## Owned files

- `experiments/mdmt_next_v3_conditional_20261025/lps_extensions.py`: SHA256 `2aa1126eb9608b4979ecc95a6c5a5679e38ae2b7ccc88c9ef330f0b3a6a35e85`.
- `experiments/mdmt_next_v3_conditional_20261025/semantics_lps_extensions.py`: SHA256 `ba2c4d3679f6cf396de5bc0813104961272103439c5f8b8e124c0423113c13c7`.

Entry: `ExtensionLpsTrainer(cfg,campaign,out=None,device='cuda:0')`, inherited fixed-epoch `train(resume=None,until=None)`, explicit `predict_logits(x)`, and overridden deployment/evaluation contexts. Semantic entry: `validate_semantics(campaign,device)` in `semantics_lps_extensions.py`.

## Exact changed mechanisms

**CAL_PROB_CLIP:** six zero-final-layer `34→19 ReLU→1` probability residual heads, P_read un-clipped BCE, full six-head/optimizer candidates including old state, P_cal scoring by actual projected deployment BCE. FP64 active-set coefficients solve the unique minimum-L2 box-constrained read with the frozen equality tolerance. Every coefficient context explicitly receives current u, real owner z0, eight real residuals and domain. Main fit detaches alpha; exactly-unused foreign readers are returned to grad=None before Adam. Meta support/query freeze alpha from their current real packets before any virtual bank replacement. Both separate outer backgrounds and L4-only phi updates remain unchanged.

Complete-packet overrides cover main, meta support/query, P_cal policy risk, read refresh statistics, evaluation, deletion and inference. There is no hidden global feature state or overloaded two-argument CAL gate. Context journals retain current-real packet hashes, raw correction/projected delta/interval summaries, coverage/saturation, coefficient usage, and FP64/FP32 arithmetic error. Foreign deletion reports both actual reprojected CAL deployment and the frozen-original-read response, so a read-dependent policy is not accidentally evaluated as a constant gate. Native/owner deletion semantics remain fixed-state risk responses.

**OWNER_META:** only the outer query changes. Each target uses its own virtual bank with detached current native logits, and its real-current foreign residual background. The .75 deployed/.25 .5 risks are computed separately. The same dynamic outer function is used for the uniform reference. Source fit supports, masses, bank-only functional Adam, read heads and schedules remain the base L4 definitions. Query gradients never submit real bank/reader/native/moment updates.

**GEOM:** one-time fit-only current-a subset, at most20000/source; deterministic PCG subset and specified KMeans++ seeds/5 starts/300 iterations/tolerance. Temperature uses the complete rho entropy including its .025 floor, median-positive-distance bracket and 60 log bisections, with explicit unreachable-endpoint handling. Centers/tau and exact fit IDs are saved in full allocator state. Current a drives later geometric weights, same weights for both tasks and the same owner backward route. GEOM has no learned allocator optimizer, zero meta steps, zero P_form accesses and no uniform virtual reference; read refresh stays 10+7×2 passes.

## Provenance and lifecycle

Construction transparently uses the base L4 object constructor, then installs the actual conditional method/config and its real policy/allocation before any step or checkpoint. The initialization receipt states this adapter boundary; every serialized config/checkpoint retains its actual conditional method and composite source.

`source.verify_checkpoint` checks the unchanged composite on resumes, and explicit certified base source/checkpoint SHA/full-state/data/seed/boundary on warmup forks. Only the exact shared epoch10 pre-refresh state is imported; the parent hash is never relabelled. Prefix validation provenance is retained. Conditional save adds a persistent post-initial-refresh epoch10 alias, preventing epoch11 from overwriting the only probability/geometry initialization snapshot. True resume restores full heads/normalizers/main/meta optimizers/geometry and avoids repeated initialization.

Changed startup, fork/restore, geometry fit, projection within main/meta/read/eval, repeated cached label use, complete inference/projection and metadata I/O are billed. `predict_logits` includes actual FP64 projection but excludes optional provenance hashing/journal I/O; inference metadata is separately paid and described. All eight foreign blocks are still computed; no sparse-speed claim is implied.

## Focused acceptance run

Syntax compilation passed. Compact semantic examples ran with `CUDA_VISIBLE_DEVICES=0`, including actual verified shared-state forks and single changed-path meta/main examples in isolated temporary directories, excluded from P2 quality tables. Receipt: `artifacts/mdmt_next_v3/v3_0/conditional_lps_semantics.json`.

| Check | Actual acceptance observation |
|---|---|
| Signed/zero/boundary/inside/outside projection | equality error0; all coefficients in[0,1] |
| Independent SciPy constrained minimum-L2 fixture | maximum coefficient difference4.16e-17; free KKT ratio spread0 |
| Clipped selection vs raw probability risk | controlled fixture risks differ, confirming actual deployment is scored |
| CAL production meta path | phi hypergradient norm6.51e-6; real model/main Adam unchanged; phi updated |
| CAL current-real coefficient context | virtual content changes outer risk while saved alpha stays identical |
| CAL real fit permissions | no probability-head or foreign-source gradient; native/owner paths active; all-zero alpha gives foreign-reader grad=None |
| CAL complete candidate and resume lifecycle | 11 complete initial candidates; optimizer restored; full conditional resume equal; post-refresh epoch10 retained |
| OWNER changed path | phi norm2.31e-4; isolated target gradient reaches virtual owner only, not virtual foreign; real foreign background matters |
| GEOM | deterministic fit/tau; weight sum error4.77e-7, values within[.1,3.7]; changed current a changes weights |
| GEOM frozen resume and later refresh | geometry identical after restore; no meta optimizer/steps/P_form; two read passes use only P_read/P_cal |

Two initial checks exposed fixture issues rather than algorithm defects: the inherited base cross-target outer helper requires all three target packets, and a uniform shift of normalized a leaves all center distances offset equally. The CAL fixture now provides all three queries; the GEOM fixture changes two coordinates with opposite signs. No design rule, tolerance or research configuration was changed to address them.

No unresolved conditional LPS blocker is known to this coder. Final composite source/receipt must be frozen after the other conditional worker and reviewer finish. The above are semantic execution checks, not evidence of algorithm effectiveness.
