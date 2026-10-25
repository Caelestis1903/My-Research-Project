# Independent review — final common inference benchmark

**Verdict: PASS. No blocking cost or forward-identity issue found.** Static review only; no GPU execution or quality-result inspection occurred during this review. The helper remains outside the frozen experiment package and does not modify its source.

Reviewed file: `artifacts/mdmt_next_v3/v3_0/analysis_scripts/benchmark_inference.py`. SHA256: `f50a14226efbd2d0f679c2c5bcaa7d17af257171682057444797dcf2368cf57b`.

| Requirement | Actual implementation | Result |
|---|---|---|
| Final-only admission | Requires all52 core and10 aux/external manifest entries COMPLETE at their declared endpoints, explicit FINAL_METHOD_FREEZE run membership, current independently reviewed experiment source, and matching optional inference source identity. | PASS |
| Fixed input and hardware | Same first4096 sorted fit row IDs per dataset, encoded feature values, FP32/no AMP/TF32, physicalGPU0. Input/row hashes and domain counts retained. No labels enter benchmark forward or checkpoint selection. | PASS |
| RHD | Direct restored native model eval forward; training controller and probe work omitted appropriately. | PASS |
| LPS | Restored current feature/projection, full owner sum and actual continuous/coarse/constant read policy match LpsTrainer.evaluate. Allocator and coverage are excluded from deployment. | PASS |
| L0 local predictors | Each MONO/FEATURE pipeline restores its own primary selected full checkpoint and applies its own normalized local correction to actual own z0. | PASS |
| External adapters | STEM and STGU use their restored deployed model directly. AutoIFS requires reuse ticket and preserves frozen selector network and selector input embedding in the model. Workflow/rewind are discarded. | PASS |
| Optional old aux | Mechanical original Trainer.choose path retains real CPU selector/normalization and transfer overhead; no proxy read head replaces it. | PASS |
| State/memory separation | Training optimizer/moments, saved gradients, formation allocator, inactive local heads, controllers and workflow rewind references are released before GC/empty_cache and baseline/peak measurements. Required deployed modules survive. | PASS |
| Parameter inventory | Unique stored-module parameter and buffer counts; explicit distinction between requires-grad-at-restore and inference-resident totals. No claim all resident weights are active on every row. | PASS |
| Latency/cost | Five warmups and30 synchronized no-grad eval calls; P50/P95 linear quantiles, throughput and all measured warmup/timed seconds retained. Initialization/hash/JSON cost excluded from forward latency but included in benchmark-section cost. | PASS |
| Memory | Baseline allocated/reserved bytes measured after cleanup; peak totals and baseline-subtracted increments reported separately. Footprint definition includes model/input/temporary state and disclaims theoretical minimum deployment memory. | PASS |
| Isolation | GPU0 compute PIDs checked before/after restoration and every five repetitions. Other jobs cause a retained stop; none are terminated. GPU mask set before importing torch. | PASS |
| Failure reporting | Missing/unsupported selected pipelines retain explicit reasons; hardware/runtime interruption stops with retained records. No timing-based model reselection occurs. | PASS |

The default36 measurements cover seed3101 core pipelines on both data, their four additional L0 local predictors, and six complete external adapters. Optional old aux adds four measurements. They characterize this fixed input batch/hardware setting, not all domain mixtures or seeds.

One **MINOR wording issue** was sent to the owner: the existing identity loader hashes and memory-maps the label artifact while constructing a trainer, so literal statements that no label array is read/loaded are too broad. Label values are not materialized for the benchmark forward, and no quality scores or labels select a model. This has no path to changed forward latency, predictions, or the stated memory scope because identity restoration is outside timed inference. The owner may clarify that text; a changed helper hash must be recorded before execution.

Safe to run after final method decisions/freezing and GPU0 becomes idle. This review does not admit new research algorithms, change the frozen source, or establish any quality/efficiency superiority. Parent retains scientific interpretation and final scheduling.
