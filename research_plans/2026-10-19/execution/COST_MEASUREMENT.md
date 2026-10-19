# Cost measurement scope

Training receipts report wall time after dataset/model initialization, including periodic diagnostics/checkpoint I/O; jobs are co-scheduled. Queue events additionally bind process start/end. Torch peak_reserved_mib excludes driver/context memory; nvidia-smi admission uses actual device memory.

Cost profiles measure supported dense forward/backward FLOPs and model-only CPU timings on a small stratified train-meta batch while other CPU/GPU jobs are active. They are not isolated hardware latency benchmarks. Unsupported embedding/normalization/optimizer operations are explicitly excluded, so profiled FLOPs are lower-bound coverage. Scaling to total examples is a profile-based estimate, not a claim every training operation was counted. Original's auxiliary architecture passes must be included separately when estimating cost.

Frozen test readout separately records full physical-batch forward times with co-scheduled GPUs. None of these measurements, by itself, establishes production latency or hardware speedup. Nominal serialized, requires-grad trainable and functionally used capacity are distinct; explicitly disabled modules can remain serialized. Earlier CPU profile JSONs without a measurement_context field have this same shared-host measurement scope.
