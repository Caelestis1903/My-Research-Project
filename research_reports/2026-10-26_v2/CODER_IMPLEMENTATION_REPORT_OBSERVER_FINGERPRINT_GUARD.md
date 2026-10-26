# CODER IMPLEMENTATION REPORT — Stable observer code-identity guard

The first authorized normal reload-readers attempt failed before either callable assignment at the old marshal(code) identity guard. Its failure receipt `failures/movielens_5201_1793025888.json` remains unchanged. No model/state/policy/RNG/p or FROZEN change is authorized by this correction. This coder has not contacted the resident service.

## Narrow correction

The guard now requires fingerprint_scheme CODE_SEMANTIC_V1. It hashes canonical JSON of code name/qualname, argument counts, nlocals, stacksize, flags, actual co_code bytes, exception table, ordered names/varnames/freevars/cellvars and recursively typed constants. Nested code uses the same definition; tuple and frozenset preserve type, frozenset members sort by canonical representation; float/complex preserve exact IEEE bits, and bool/int/string/bytes/None/Ellipsis have separate tags. Filename, first line and line table are metadata, excluded from semantic identity. Unsupported constant types reject rather than falling back. It never accepts an unknown function or bypasses a mismatch.

Old code identities are still derived strictly from the preserved original baseline source with original source SHA24a12bfb2033b2fa3e9dde26b0fe0ed4a61ce7dd19ac59c286401dcf83911760, compiled without executing function bodies. New function identities remain bound to the exact independently reviewed observer file SHA. Existing source/PID/PPID/typed worker/op/frozen/config/PASS guards remain mandatory. The replacement allowlist is still exactly baselines.raw_features and baselines._raw_source. No canonical reconstruction mathematics or training/readout definitions changed.

Any old/new semantic mismatch writes a small REJECTED_BEFORE_ASSIGNMENT receipt under observer_raw_reconstruction/guard_rejections, including actual and expected semantic/marshal hashes and actual module/name/qualname/co_filename/firstlineno/flags. No model state is recorded, and the exception remains fatal to installation. Marshal fingerprints are diagnostic only.

The first failed attempt had already imported the old observer module into the worker's sys.modules. The reader's fixed final hook now refreshes exactly that own observer module via importlib.reload when loaded, then calls the same typed installer; otherwise it imports the same hardcoded module. It never reloads baselines or training modules. This is required to make the new reviewed guard execute on the next normal RPC rather than leaving the failed marshal guard cached.

## Actual minimum verification

`verify_observer_semantic_fingerprint.py` tested each of the two original archived code objects through direct compile, code and full-module marshal roundtrips, types.FunctionType, timestamp-pyc representation, and changed location/line-table metadata. Every representation produced the same semantic fingerprint. Mutated opcode, constant or global name produced a different fingerprint in every case. Typed scalar and float sign bits remain distinct; frozenset ordering is stable. No archived function body was executed, no fitting occurred, and no resident RPC was used.

The baseline AST audit still finds only the two authorized observer delegate definitions changed; all other original baseline definitions and every original numerical reader definition are identical to the pre-observer snapshots. The changed modules pass py_compile. Core canonical reconstruction GPU fixtures are not rerun: this correction only changes code identification and the fixed loader. The real full-row 1e-6 after-case remains pending normal installation and evaluate, not claimed repaired.

Evidence: `OBSERVER_SEMANTIC_FINGERPRINT_BASIC_CHECKS.json`. The independent reviewer must sign a new exact token with fingerprint_scheme CODE_SEMANTIC_V1, source_sha256, old/new_callable_fingerprints from that evidence, and optional old/new_marshal_fingerprints for diagnostics. All original exact service/worker/config/FROZEN identity fields remain unchanged. No PASS authorization is written by this coder.

## Current source SHA256

- `experiments/mdmt_rcf_20261026/raw_reconstruction_observer.py`: `945136f72d577ee22aa4e98a3446f2c930f1534e605c7fd62f07bc7f01e989c3`
- `experiments/mdmt_rcf_20261026/baselines.py`: `a7207fa765644279bc563ea0926a57814e61a310a8f2d2870eb39e354f4ac0e4`
- `experiments/mdmt_rcf_20261026/readers.py`: `7a77f17572d0c40177cbb82a79600a4250f1f8fb92e137994c2847586d6bd109`

## Semantic callable fingerprints

```json
{
  "old": {
    "_raw_source": "a0f3943be30d80ebb2a8b134cf53f123004a56337d5f246e134dc1ea624b5840",
    "raw_features": "adbdb96aafbd9b7cb7247330c10650fbd9d6dfc8ca5867e1537e882f1944c7a6"
  },
  "new": {
    "raw_features": "820902496e97efcc62e9df70547f2c5f814a030703e9874ac9740d99d194f27d",
    "_raw_source": "2712a2a2ef9b67052ad8c26ea8f44a7fdf63f1abc433e53f9ebc48c4981fe3a9"
  }
}
```
