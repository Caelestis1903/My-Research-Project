# Cost measurement scope

Training receipts report wall time after dataset/model initialization, including periodic diagnostics/checkpoint I/O; jobs are co-scheduled. Queue events additionally bind process start/end. Torch peak_reserved_mib excludes driver/context memory; nvidia-smi admission uses actual device memory.

Cost profiles measure supported dense forward/backward FLOPs and model-only CPU timings on a small stratified train-meta batch while other CPU/GPU jobs are active. They are not isolated hardware latency benchmarks. Unsupported embedding/normalization/optimizer operations are explicitly excluded, so profiled FLOPs are lower-bound coverage. Scaling to total examples is a profile-based estimate, not a claim every training operation was counted. Original's auxiliary architecture passes must be included separately when estimating cost.

Frozen test readout separately records full physical-batch forward times with co-scheduled GPUs. None of these measurements, by itself, establishes production latency or hardware speedup. Nominal serialized, requires-grad trainable and functionally used capacity are distinct; explicitly disabled modules can remain serialized. Earlier CPU profile JSONs without a measurement_context field have this same shared-host measurement scope.

An external user process main_capsid.py was observed onGPU0 from approximately14:10 UTC during the final training segment/reruns. It was not controlled or interrupted. Together with intentional co-scheduling and diagnostics, this is another reason to treat wall times as recorded process durations rather than isolated algorithm speed or physical GPU-hour estimates.

## Complete training accounting

The856 successful full-budget receipts sum to88.4116 process-wall hours,47,386,839,000 primary row exposures and44,729,656 original-protocol auxiliary row exposures. The declared small-batch supported-dense profile extrapolation sums to149.927 PFLOPs, excluding optimizer/embedding/normalization, diagnostics/evaluation and unsupported operations. This is an estimate, not measured full-run hardware FLOPs. Archived partial attempts are excluded from these successful-run totals.

Raw formal queue events span2026-10-19T05:14:59.346936+00:00 to 2026-10-19T14:40:23.259968+00:00 (9.4233 hours of orchestration wall time). This interval across two shared3090 devices is not an isolated physicalGPU-hour estimate. Some device time belongs to external work, CPU/data/checkpoint work and co-scheduling.

CPU measurements demonstrate why latency rankings would be unsafe: identical-size KR fixed/original/original_sample graphs have observed768-row median forwards around.327/.024/.322 seconds under different shared-host load. Treat them as raw measurements, not a claimed original-model speedup. Structural parameter counts and supported-dense operator counts are reported separately in offline_readout/model_cost_profiles.csv and full per-run exposure estimates in development_readout/evidence/costs_with_profile_estimates.csv.
