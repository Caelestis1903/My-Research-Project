# CODER IMPLEMENTATION REPORT — Recovery A: row-data memory storage

This implements the data-storage portion of `DESIGN_FOUR_SEED_RECOVERY_ADDENDUM.md`. No research performance analysis or formal training was performed by the coder.

Changed files: `memory_data.py`, `protocol.py`, `prefix.py`, `baselines.py`, `readers.py`, `formation.py`, and `mechanisms.py` under `experiments/mdmt_rcf_20261026/`. Campaign/configuration/gates are owned by Coder B and reported separately.

`memory_data.py` has explicit NumPy row-data entrypoints only. Arrays at least 256 KiB are persisted as ordinary NPY files under the owning prefix's `row_data_cache`; writes and compatible dtype/shape/data hashes use blocks of at most approximately 8 MiB (a single large row is the minimum block). Completed mappings are read-only. The hash has exactly the original array_hash dtype/shape/C-order byte definition. Content identity in filenames prevents a changed-data overwrite. The small arrays remain ordinary arrays. No module/object-graph/tensor/pickle/state serialization occurs.

Native H/Z are mapped immediately per role before pilot training. Pilot raw hidden activations are mapped at cache creation; standardization, numerical dtypes, statistics, and forward chunking remain unchanged. B OOF/refit logits, direct/probability-reader predictions, all linear-reader predictions including single-function diagnostics, retained feature banks, formed function outputs/features, and the three diagnostic request arrays are explicitly mapped at their data boundaries. Prediction and FeatureBank dictionaries are updated in place, preserving closure references. Fixed-content feature banks keep their existing post-selection release behavior.

LabelLedger allocates a writable file-backed full-length bool access mask only on a new key. Original row membership checks, duplicate-row counting semantics, record ordering, row hashes, passes, and summary calculation are unchanged. This also removes the old unconditional large np.zeros evaluated inside every setdefault call. The ledger masks are access data, not model state.

`release_data_pages` flushes and requests MADV_DONTNEED only for known data mappings, then collects freed temporaries. It never visits or releases live model tensors, Adam, RNG snapshots, or formation training_state. The cache receipt lists immutable row-data mappings and mutable label-mask paths for owned cleanup after export acceptance. Scientific state remains in RAM. Startup/concurrency and final cleanup are handled by Coder B.

## Basic verification actually executed

- py_compile passed for all seven changed Python files.
- Actual MovieLens prepared fit rows (16,384) were mapped with float32, noncontiguous view, and float64 prediction data. Values, dtype, shape/order, original array_hash and chunked hash matched exactly.
- PredictionBundle object and its original probability dictionary identities survived mapping; contents remained exact.
- Original boolean-mask ledger rules were replayed against the new writable mmap across overlapping rows, duplicate rows, separate methods/tasks, multi-pass and empty accesses. Every record and summary matched exactly.
- A real-row-derived H matrix and frozen untrained PilotBank tested cached raw hidden activations. The mapped float32 output exactly matched the direct PyTorch forward and reused the same cache object. No pilot training occurred.
- Mapping did not change a live small model or Torch RNG. No model files were written.

Evidence is `DATA_MAPPING_BASIC_CHECKS.json`; this is implementation verification, not a multi-seed performance experiment. Prefix selected/epoch24/pilot science hashes must be checked against old v1 when the runner performs the authorized same-seed replay. RAM admission and first actual worker peak measurement remain runner-stage obligations. Independent reviewer verdict is not asserted by this coder.

## File SHA256

- `experiments/mdmt_rcf_20261026/memory_data.py`: `e8ade42f83900257a0813f950f57d2b23cd5e59c4bb92c0d552b47877503f639`
- `experiments/mdmt_rcf_20261026/protocol.py`: `31973483853ed8c79e1cd729d73525f5e3c548e2e48e69cf40958a382a9ee061`
- `experiments/mdmt_rcf_20261026/prefix.py`: `787d353b887afcfea5a102aa8cc15787f66c16a95a0bc80b72b487421115fd1c`
- `experiments/mdmt_rcf_20261026/baselines.py`: `24a12bfb2033b2fa3e9dde26b0fe0ed4a61ce7dd19ac59c286401dcf83911760`
- `experiments/mdmt_rcf_20261026/readers.py`: `1a9f578f0dc7bb7732b4c61a13320955f71c4340e6a4232c96f5dfd9a12d6493`
- `experiments/mdmt_rcf_20261026/formation.py`: `0b241ffb871bd73d2073db2dfa830c55f2b6ceb210988dc4eb51f2c6ad6a9906`
- `experiments/mdmt_rcf_20261026/mechanisms.py`: `17d6071b67a4dbf43cc86ad0953a21bdb4ea88a6d29876cb25da9ff1549efb88`
