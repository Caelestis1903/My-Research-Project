# DESIGN ADDENDUM — FP64 SVD execution placement

Approved DESIGN change before scientific dispatch, 2026-11-02. Applies to DESIGN_SPECIFICATION sections7/12 and `views_ema_field.null_basis`.

**Observed reason.** Parent reports all12 initial Stage0 checks PASS, no S1 trajectories dispatched, and KR/F1 full steps18.019/14.491s versus other KR methods around.5s. Its CUDA FP64 batched SVD on approximately[1843,2,33] is the concrete suspected bottleneck.

**Approved change.** Coder may make one real-size CPU/CUDA null-basis comparison, then move only the already-normalized FP64 [n,2,F] SVD to CPU if the measured CPU path including transfers materially improves runtime. Use the same `torch.linalg.svd(...,full_matrices=False)`. Return retained FP64 basis (and singular values if needed) to the original device; normalized J and all subsequent projection/scaling remain on their current devices. Preserve singular cutoff `s>1e-6*smax`, absolute zero1e-12, rank masks, fallback conditions, cap, dtype of final delta and every RNG key/draw. Do not substitute a different projector formula, truncated algorithm or approximate precision.

Freeze one placement for all F1/FR/F_PR and both datasets; use CPU normally for already-CPU inputs. No per-row, per-epoch, per-seed or performance-result-dependent backend choice. If the measured bottleneck does not improve materially, retain the existing implementation and record that measurement.

**Minimum acceptance.**

- Compare both backends on the same real-size Jacobian and fixed vectors held in RAM. Compare ranks and projected vectors, not singular-vector signs. Rank masks must agree. FP64 projected vectors must satisfy allclose(atol=5e-10,rtol=1e-8), with normalized-J residual<=1e-9 times max(1,vector norm). Reuse existing rank0/1/2 and degenerate checks.
- At an identical actual model/EMA/RNG state, final FP32 deltas must satisfy allclose(atol=2e-7,rtol=1e-5), with unchanged fallback decisions and counters. Keep this paired check bounded to the same microcheck; no research trajectory or seed is added.
- Run the existing focused F semantic check. Under the changed source, repeat real two-full-batch-step F1 Stage0 for ML and KR, including existing diagnostics, memory and physical costs. No test evaluation or tuning. Reuse unaffected Stage0 observations.
- Independent reviewer inspects the small implementation diff, coder report and numerical checks. Scientific dispatch waits for PASS or resolved blocking fixes.

**Records and costs.** Preserve old source/Stage0/forecast records, append the optimization receipt and new source hash, and charge the microcheck and repeated Stage0 work. Before first scientific dispatch, derive the affected job estimates from the newly measured implementation and freeze the resulting dispatch reservation; retain both forecast versions. Do not enlarge a frozen reservation to absorb failures. Existing1.20 physical-cost cap, retry policy, queue,50 epochs and all scientific gates remain unchanged.

Designer performed source inspection only; coder/runner produce the numerical evidence.

