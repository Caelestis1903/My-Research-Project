# CODER IMPLEMENTATION REPORT — Frozen raw-X reconstruction observer

Implements `DESIGN_RAW_RECONSTRUCTION_OBSERVER_ADDENDUM.md`. This is an observer repair; no formal training, parameter selection, benchmark exposure decision, or scientific effect analysis was performed by this coder. The resident service was not contacted.

## Actual failure and bounded change

The preserved first normal evaluate failed MovieLens seed5201 FULL_SOURCE_ALL at maximum p error 2.1976759427277948e-6 against the unchanged 1e-6 acceptance rule. Native raw-X H/Z errors were 0. Existing source bytes were preserved at `evidence/pre_raw_reconstruction_observer/`. The defect candidate is target masking before downstream forward, which differs from the full-role content creation graph.

`raw_reconstruction_observer.py` reconstructs source/local/initial activations on each frozen head's existing device, full role with 40960 chunks, then applies the original FP64 normalizers / float32 cast before target masking. Initial normalizers come from independent raw-X fit H and the initial CPU head; they never read PilotAccess hidden caches. The separate small observer-statistics cache binds recipe, native hash, fit row hash and initial-head state hash. It does not change original normalizers.

Pilot residuals use the canonical one-dimensional hidden @ flattened output weight + float bias. Formed content uses only a RAM deepcopy of the frozen module on bundle.device, full-role chunks of 16384 and original CPU NumPy scale placement. Original modules, modes, grads, requires_grad and devices are untouched. Projections preserve their existing matrix expression after canonical source features. The delivery observation role remains the full supplied observation packet, with independent full-fit initial statistics.

`baselines.py` changes exactly raw_features and _raw_source into same-signature delegates for future fresh imports. Existing resident workers are updated through the fixed normal reload-readers import hook. The installer does not reload baselines or change training function objects. It replaces exactly baselines.raw_features and baselines._raw_source. All original numerical reader definitions and all other original baseline definitions retain identical ASTs.

## Typed normal installation contract

Fresh clients, parent and unregistered processes are NO_OP. Only the exact existing campaign._worker code frame, current op reload-readers, frozen=True, registered dataset/seed/PID, PPID2088990, unchanged config hash and owned artifact root may install. The frozen manifest SHA is fixed at 09def5e2f43786f9c2875e5ae466845a10ec7db47a3c6edd28379827b72e927a. No new RPC, eval/exec, object scan, trained fixture dump or arbitrary function target exists.

The independent reviewer must create OBSERVER_RAW_RECONSTRUCTION_AUTHORIZATION.json with verdict PASS or PASS WITH FIXES and no blocking_issues, exact service_pid/config_hash/frozen_manifest_sha256/worker_pids, source_sha256 for the three files, old_callable_fingerprints and new_callable_fingerprints. Expected values are in RAW_RECONSTRUCTION_OBSERVER_BASIC_CHECKS.json and module constants. The old callable code fingerprints were derived without execution by compiling the archived original text with the original absolute baselines.py co_filename and extracting its two code objects. New fingerprints are the actual observer callables. Source SHA authorization is checked before any assignment; only the old exact pair or already installed new exact pair is accepted.

Before/after fingerprints cover native/source/initial/local/base/direct/gate/formed and retained terminal model values, modes, grad requirements, existing gradients, devices, reader coefficients, normalizers, RNG, selected and terminal predictions, selection/identity and retained RAM optimizer/training states. Only hash results are written, never training state. Five baseline training function object identities must also remain identical. Per-worker installation receipts record the full before/after proof, source identities and pending actual parity. FROZEN/config/retained p are not rewritten.

## Basic verification actually executed

`verify_raw_reconstruction_observer.py` ran with project Python and repository PYTHONPATH, using 41063 rows to cross the 40960 boundary. Source/local frozen heads were CUDA, initial heads CPU, and the formed reference used the actual canonical GPU16384 path. These are numerical fixtures without fitting.

- FULL_SOURCE, FULL_SOURCE_ALL, FULL_TARGET, RANDOM_FEATURE and PILOT_LOGIT reconstructed content exactly matched the canonical full-role oracles (max error 0, float32).
- Pilot residuals exactly matched canonical PilotAccess residuals.
- New formed full-role GPU output matched exactly; the old CPU/target-first alternative differed by 1.7881393432617188e-7 in this fixture.
- Old target-first source content differed by approximately 5.72e-6 to 8.58e-6, while the new full-role path was exact.
- The reconstruction ran after the training hidden/output/FeatureBank caches were replaced with access-forbidden dictionaries. Those caches were only used earlier to construct fixture comparison oracles.
- Model values, modes, requires_grad, grads, devices and RNG were unchanged; a non-worker install was NO_OP.
- Baseline AST comparison found only the two authorized observer definitions changed; every original reader definition remained identical. The three files passed py_compile.

These checks do not claim the actual trained case is repaired. The real before is preserved; after independent PASS and normal installation, the same complete normal evaluate must establish actual ML5201 parity and every actual method/context must pass the existing full-row 1e-6 gate. No tolerance is relaxed, no frozen coefficient/p is changed, no retraining is authorized. Actual post-install raw/formed parity and all eight export acceptances remain pending runner work.

## Final approved-scope measurement extension

After all independently reconstructed output arrays are complete, a separate terminal observer helper may compare cached values only as an oracle. It requires the already-authorized installed frozen worker, actual fit role, exact registered PID/PPID and unchanged FROZEN SHA. FULL_SOURCE_ALL records one source/task raw hidden and standardized float32 maximum error; RCF records one actual function full-role GPU16384 output maximum error and one old CPU/target-first comparison. It writes small JSON under observer_raw_reconstruction/comparisons outside run-index recursion. No compared cache enters output construction or changes p, policy, scale, normalizers or training objects. True logit error is explicitly NOT_MEASURED; the original separate full-row probability gate remains mandatory at 1e-6. Existing matching receipts can be reused.

The installer now explicitly checks PrefixBundle and prediction/formation/config dict types before fingerprints or assignment. The final extension changes only this terminal guarded comparison and type checking. The already successful core GPU fixture was not repeated; a fresh-process call verified terminal comparison and installation are NO_OP outside authorized runtime. RAW_RECONSTRUCTION_OBSERVER_BASIC_CHECKS.json preserves the core fixture's original source SHAs separately from final source SHAs and updated callable fingerprints. Independent reviewer must audit the final delta before any service contact.

## Current source SHA256

- `experiments/mdmt_rcf_20261026/raw_reconstruction_observer.py`: `2925fa4f40adc63b64f69091bc029c839dbd2c40c6bb63bb7c22002210b2a4fc`
- `experiments/mdmt_rcf_20261026/baselines.py`: `a7207fa765644279bc563ea0926a57814e61a310a8f2d2870eb39e354f4ac0e4`
- `experiments/mdmt_rcf_20261026/readers.py`: `8f8d0ee71ea39e89a6e73fa8b8e02918e93622f0501f1c0b113fdf37cc7fae8d`
