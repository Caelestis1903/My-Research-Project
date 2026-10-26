# CODER IMPLEMENTATION REPORT — Delivery measurement preservation

Implements `DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM.md`. No formal training, research-effect analysis, actual timing, resident RPC, restart, export, bootstrap, or model mutation was performed by this coder. Current evaluate was left running.

## Changes and preserved scope

Only delivery.collect_delivery changes among existing delivery definitions. FrozenDeployment, benchmark_predictor, its3warmups+10timings/natural4096 raw-X rows/synchronization/transfer scope/1e-6 parity, all cost formulas and parameter definitions retain identical ASTs. Existing reader numerical definitions also retain identical ASTs; its final hardcoded delivery-only hook is added. New helper is delivery_observation_preservation.py. Pre-change delivery/readers bytes and SHA are in evidence/pre_delivery_preservation.

The collector creates identity-bound immutable per-method JSON observations under delivery/measurement_records, outside run-index recursion, before updating the canonical summary. Content-addressed record names, exclusive hard-link publication and byte-for-byte existing-content verification prevent overwrites. Canonical JSON uses a complete temporary pure-data file, fsync and atomic replace; the training-cost CSV is replaced atomically from unchanged real costs.

Identity includes config and exact prefix identity(native/pilot/base/data/source), dataset/seed, frozen prediction identity/selection, formation identity where applicable, all4096 actual row IDs/hash, feature schema, the unchanged recipe, device/Python/Torch/CUDA/thread environment and the compiler/benchmark AST hash. MEASURED entries must retain ten positive finite actual samples and both fused-reference/retained-p errors within1e-6. Stored deployed parameter recipe must also match the current unchanged compiler.

Existing verified MEASURED records are reused with their exact10timings and source path/SHA. False calls never benchmark and never replace valid MEASURED entries with NOT_MEASURED. True calls skip valid entries, benchmark only permitted missing methods and preserve existing nonrequired failures. RANDOM_FEATURE parity failure reason/error remains an observed NOT_MEASURED result; its compiler is not changed or repeatedly retried. Required coverage and all-method coverage are distinct: required_measurement_status may be COMPLETE while global status is PARTIAL_OBSERVATIONS. current_call_benchmark_requested describes the current request separately from persistent measurement sources.

Legacy True receipts/snapshots are not relabeled as new measurements. Snapshot bytes/SHA, exact prefix/config identity, frozen TRAINING_COMPLETE prediction identity/selection/prefix, original immutability assertion, raw-row/order/schema/environment/precision/recipe/parity and actual10samples are checked. Formed methods additionally require an actual formation.json identity match. Where original source mtime exists, it cannot predate the method-completion receipt. Original measurement source metadata is retained; missing original timestamps/formation fields are stated, with explicit crosscheck paths/SHA rather than guessed values. Imported records record the import time separately. Missing or mismatched evidence becomes a missing measurement, not fabricated timing. Allowed snapshot paths stay inside the owned attempts root. Successfully imported records are reused idempotently.

## Existing normal RPC entrypoint

The reader tail reloads only the fixed own delivery-preservation helper and calls run_on_reader_reload. Parent/client/other processes are NO_OP. Only exact campaign._worker code in a normal reload-readers frame, typed PrefixBundle plus predictions/formations/diagnostics/config dicts, frozen=True, exact eight registered worker PID/PPID2088990, fixed config/root and original FROZEN SHA may proceed. Definitions alone do not time anything: a dedicated independent PASS authorization with operation fill_missing_required is mandatory.

The token binds this design and exact new delivery/readers/helper source SHA, all original frozen service/worker/config identities, a per-worker whitelist that is a subset of the fixed nine REQUIRED methods and verified snapshot paths/SHA. The same token can be repeated without retiming valid entries. Full frozen fingerprint before/after uses the existing extended raw observer proof, covering model/terminal/optimizer/mode/gradient/RNG/normalizer/reader/p/selection identities. The helper invokes only collect_delivery with bounded allowed methods; it never calls training, export, bootstrap, decide, freeze or finalize. It writes per-worker preservation receipts with reused/new/missing actions and real version-record references.

Existing raw-observer authorization also binds the readers SHA and must be refreshed by the independent reviewer for this new tail. Its reconstruction functions and guard source are unchanged. No FROZEN/config/training identity is rewritten. Concrete token fields and current SHA are in DELIVERY_PRESERVATION_INSTALL_REQUIREMENTS.json; that file is not an authorization and this coder wrote no PASS token.

## Verification actually executed

`verify_delivery_observation_preservation.py` ran entirely with small mock observations and synthetic timing values in a distinct verification artifact root. No real benchmarks or training occurred.

- False→False never called benchmark or fabricated measurement.
- True MEASURED→False retained all ten samples and exact measurement source references.
- Repeated authorized fill did not remeasure valid entries or change immutable record bytes/mtime.
- One missing authorized REQUIRED method was called exactly once; other missing methods stayed explicit.
- RANDOM_FEATURE's mocked original parity failure and max error remained NOT_MEASURED; nine-method coverage did not claim all methods MEASURED.
- Snapshot SHA, prefix/config, policy, row order, environment and formed identity mismatches were rejected; matching actual mock completion/formation receipts allowed legacy reuse.
- Immutable overwrite was refused; no temporary pure-data files remained after atomic writes.
- Wrong PID/parent/config/frozen SHA/source/verdict/whitelist were refused by the same pure validators used in the typed hook; ordinary client installation was NO_OP.
- AST comparison confirms the only original delivery function changed is collect_delivery; compiler, benchmark, training costs and all original reader numerical definitions are unchanged. The three Python files passed py_compile.

Evidence is DELIVERY_PRESERVATION_BASIC_CHECKS.json. Core real timing remains pending independent PASS, natural completion of the current evaluate, and one authorized normal reload-readers. The intended real fill is missing REQUIRED methods only; surviving True snapshots are reused, and lost MovieLens5201 timing cannot be reconstructed from absent data. Real72-required-measurement acceptance and unchanged frozen-state receipts remain runner-stage work.

A final narrow data-integrity check verifies immutable record content hashes against their content-addressed filenames before reuse. The valid existing mock record was accepted and an identical body under a mismatching name was rejected. Core timing/storage mock tests were not unnecessarily repeated.

## Current SHA256

- `experiments/mdmt_rcf_20261026/delivery.py`: `94a44fde6540b7513da9e98bf3e53f97ff4aca64d64bd436ed47e50090f7dc3c`
- `experiments/mdmt_rcf_20261026/delivery_observation_preservation.py`: `d46324c30f3eac44936bb8f91e9e51b69291bfd33afd51dbccad383fd1213366`
- `experiments/mdmt_rcf_20261026/readers.py`: `8004af8e37635de8b2fc24b9bbcaed523826c7594b5d76577dec5e0d266c938d`
