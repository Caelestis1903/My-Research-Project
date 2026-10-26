# Delivery and cost audit — read-only findings

Scope: inspect the currently reviewed, already resident execution path; no code edits, service commands, training, timing experiment, or process restart performed in this audit. The four completed S0 contexts must remain in their existing workers. Current S1 execution and original training identities are retained.

## Actual measurement coverage

| Component | Actual field | Scope and limitation |
|---|---|---|
| Native | `epoch_metrics.jsonl[*].seconds` | Each epoch includes training and validation; not isolated GPU kernel time |
| Source/two local pilots | `training.jsonl[*].seconds` | Per pass, recorded individually |
| Shared prefix | `PREFIX_COMPLETE.json.elapsed_seconds`, `peak_cuda_bytes` | Actual process wall total and CUDA allocator peak; includes native, terminal/selected feature export, source/local pilots; does not include raw-data audit/model construction before timer |
| B OOF/refit | `bundle.diagnostics.base_seconds` | Actual combined fold/refit plus all-role B prediction time; currently RAM only, not included in final export |
| Three direct controls | `bundle.diagnostics.direct_seconds` | Actual combined time for all three families; currently RAM only; per-family wall times were not measured and cannot be reconstructed from step counts |
| Fixed-content readers | `PredictionBundle.diagnostics.elapsed_seconds` | Includes fitting lambda candidates and all-role selected predictions; does not isolate raw feature preparation |
| Probability gate | same field | Includes its fitting/selection and role predictions |
| Formation | terminal function `metadata.elapsed_seconds`, `parameter_count`, query/aux visits, effective steps | Actual per-function cumulative wall; sum terminal entries once, never all cumulative pass values. K1 reused functions specify zero additional training cost |
| RRR | each returned start bank identity | Actual per-start duration; keep both starts if triggered, including the nonselected start |
| Correctness callback | `_raw_x_validate` | Numerical agreement only. Its independent H/Z reconstruction cache is reused, so this is not deployment latency |

No code in the reviewed path times a standalone deployed full-system forward call, creates a per-request active-parameter table, or emits a complete shared-vs-standalone cost accounting table. Existing raw timings must be exported before contexts are released. Missing historical per-direct-family wall measurements must be labeled unmeasured, not prorated by steps or invented. Method compute cannot simply be summed over all shared prefixes as though every deployed method trains every control.

## Minimum safe technical strategy requiring parent approval

Add one explicitly versioned **post-training delivery supplement**, independently reviewed before S2. Do not edit native/base/reader/formation algorithms or restart the resident server. Because `mechanisms` has not yet been imported in workers, its first import can install a narrowly named delivery callback into the existing `campaign._summary` global. That callback receives the complete existing `(bundle,predictions,formations,diagnostics)` objects, delegates the original summary unchanged, and only exports/times frozen computation. It does not inspect caller frames, capture or pickle training state, modify weights/optimizers/RNG/selection, or create new training trajectories. The installation itself is explicit and logged.

The supplement identity must state original resident core source/config hashes and loaded module identities separately from the supplement file hashes and installation event. A refreshed stage authorization covers the new report-only bytes. The final report must not claim that the old resident campaign code was hot-reloaded or that prefixes were trained under the collector's later hash. A hook that changes the already-running S1 training is unnecessary.

At the first post-S2 summary for each prefix, retain the complete numerical cost fields above and count actual RAM modules/coefficient metadata. Immediately before memory release, the same callback/export collector can benchmark the frozen final selected models after all registered branches exist. A one-time per-prefix marker avoids duplicate timing and identifies any later triggered methods separately.

Minimum benchmark: raw `Q_explore` input rows in fixed natural order, a fixed documented batch (e.g.4096), no labels, a small fixed warmup/repeat count, CUDA synchronization around wall timing, measured output parity with cached selected predictions≤1e-6. Include native→B→content→selected policy and device transfers. Use the same device/batch/dtype/backend for every compared method. Preserve exact OFF and group cuts. Record batch/rows/domain composition, device/model/source identity, median/mean/min/repeated timings, and maximum output discrepancy. This is an execution measurement, not an additional research arm or model-selection signal.

FULL-source and formed readers should both use exact coefficient fusion where available: retain each necessary hidden transformation and fold its fixed reader/normalization/output coefficients into the final affine map. Validate the fused path numerically before timing. Count total trained method parameters separately from stored fused deployment parameters and per-target/group active parameters; zero selected content/OFF removes the corresponding inference branch. Report source/local pilot training costs even when their output heads are not deployed. Direct, FULL_SOURCE_ALL, GROUP_CAL/B_REFIT, RCF, POOLED_TARGET, LOCAL_ACTIVE are the minimum explanatory comparison set; include additional actually triggered methods without new training.

## Existing field completeness

Formation terminal records already preserve projection `c`, pre/post S/R/K/T, actual steps/label visits, initialization and final function hashes. Per-arm actual request/anchor exports, sourceφ, single-function predictions, group/deletion/replacement profiles, source support, and OOF/refit drift are present in the reviewed contract. Do not reopen historical training to improve report cosmetics. Only the missing cost/parameter/inference accounting and propagation of the already-measured RAM timing fields require this delivery supplement.

No permission is requested from the user: this audit proposes a concrete internal implementation route for parent approval and required independent review. Actual deployment timings remain **NOT_MEASURED** until the approved supplement executes.
