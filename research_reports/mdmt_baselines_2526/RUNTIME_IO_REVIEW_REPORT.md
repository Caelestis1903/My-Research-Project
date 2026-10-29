# Runtime I/O R2 independent review

Date: 2026-10-29. Amendment: `MDMT-SOTA-2526-RUNTIME-IO-R2`.

**Latest verdict: PASS — IOFIX1, candidate02 (2026-10-29).** IO1/IO2 below are now closed. This permits only the specified full-budget HyperGate/KuaiRand/seed2022 R2 replay from initialization. It does **not** release the remaining nine formal trajectories or E immediately. The real replay is still pending; first-five scientific equality and successful normal completion are required before further resumption. The original candidate01 FAIL text below is retained as history, and its exact pair/receipt/report are archived under `artifacts/mdmt_baselines_2526/runtime_io_review_history/candidate01/`.

## 1. Verdict

**FAIL. Do not start the bounded real retry yet.** The resident-X implementation itself is correct and narrowly scoped. Two source/evidence authorization gaps in the new mixed-revision validator must be corrected before review can permit the retry. No real replay or E evaluation was performed by this reviewer.

Machine-readable receipt: [RUNTIME_IO_REVIEW_RECEIPT.json](RUNTIME_IO_REVIEW_RECEIPT.json). It preserves the exact candidate R1/R2 bytes and metadata reviewed here. This is a failed review receipt, not an authorization. The current project amendment remains `REVIEW_REPLAY_PENDING`.

## 2. Requirement verification

| Requirement | Independent result |
|---|---|
| Change only feature storage, not model mathematics | Correct: the complete data.py diff replaces only `x=np.load(...,mmap_mode="r")` with ordinary load plus `x.setflags(write=False)`. Y load and role authorization are unchanged. |
| Fixed exact R1/R2 pair | Candidate inventories each contain 18 identical paths, with 15 unchanged byte maps and exactly 3 changed files: data.py, evaluation.py, report.py. R1 equals the actual PRE_FORMAL_FREEZE snapshot; current package equals the candidate R2. No model, initialization, loss, optimizer, training, sampler, config or reference bytes changed. **The validator does not yet bind that pair to the independent review receipt: IO1.** |
| Preserve successful R1 trajectories | Independently read 20 completed formal run receipts and their actual source snapshots. All 20 equal R1 and their configuration/source/prepared-manifest identities equal the amendment's declared originals. No receipt/checkpoint/source file was edited by the reviewer. |
| X values, permissions and RNG | Root's actual X-only feature receipt reports full float32 equality at MovieLens `[1000209,8]` and KuaiRand `[2507666,34]`, ordered/reversed fit/F/V batch equality, read-only ordinary owned X, unchanged Python/NumPy/Torch CPU RNG and no Y loading. Inspected the receipt and affected code; did not repeat the already completed old-mapping read. Independent synthetic tests verify ordinary X, mapped Y, no constructor label indexing, E/S/legacy denial and exact supported-model logits/loss/gradients/Adam/state/RNG equality. |
| Default source freeze remains strict | Correct: without the explicit amendment, the original full-source byte equality check remains active and mixed sources fail. |
| Exact run/checkpoint/config/data guards retained | Existing 30-run, formal phase, path, effective method configuration, source, dataset and selected/fixed-final checkpoint checks remain. **The amendment's own original configuration/source/prepared-manifest declarations are not cross-checked against the runs: IO1.** |
| Per-run R1/R2 association and active evaluator | Implemented: mixed freeze classifies exact full source maps as R1/R2; amended E dispatch checks that run's map and current installed R2 package before existing authorization/label access. Missing/unknown revisions and current-package drift are rejected in synthetic tests. |
| Completed review/test/replay evidence | **FAIL:** review is a text substring check and replay uses status/Boolean fields without checking the referenced actual run/logs. Both were independently reproduced: IO1/IO2. |
| Bounded replay protocol | Design correctly limits the first retry to full-budget HyperGate/KuaiRand/2022, with first-five scientific fields exactly equal before the other nine entries proceed. Actual replay is pending, so final freeze and E remain forbidden. |

## 3. Blocking findings

### IO1 — MAJOR: the approval/evidence is not bound to the exact reviewed byte pair and original identities

`report.py:_validate_runtime_io_amendment` checks a review receipt by searching its text for the amendment ID and substring `PASS`. It never compares the source pair in the amendment with the pair actually reviewed. Its `allowed_changed_paths` check therefore permits a newly regenerated data/report/evaluation pair to reuse the old review text. Requiring the installed package to equal that newly regenerated R2 does not close the gap: both candidate and current source can change together without a new independent review.

Independent controlled proof used a temporary copy of the candidate, without editing or executing altered repository code. The probe changed candidate R2 data.py to eagerly load Y, updated its full source maps and unified diff, and simulated matching installed bytes by overriding only the source-inventory query in the probe process. The unchanged review text and unchanged old CPU evidence were still accepted: `ACCEPTED_DIFFERENT_PAIR_WITH_FORBIDDEN_EAGER_Y_USING_OLD_REVIEW_TEXT`. This violates the explicit finite-pair/no-eager-Y contract.

The amendment also records `configuration`, `source_commits` and `prepared_data_manifests`, but these declarations are not compared with each run in the amended freeze. The existing all-run common-identity checks must remain, but they do not replace binding the runs to this amendment's approved originals.

**Minimum correction:** use a structured independent review receipt containing the exact reviewed R1/R2 maps (or an immutable full-content record), amendment ID, verdict, original config/source/data identities, and actual CPU/feature evidence association. Validate full byte/metadata equality against that receipt, not a verdict substring. Cross-check each actual run's config/source/data against the exact amendment originals. Do not add hashes, signatures, tokens or generic version rules. A changed candidate may be archived/resealed before a new review; an old approval must not authorize new bytes. Add a focused regression that changes an allowed source file and candidate/current bytes together while leaving the review receipt unchanged, and one original-identity mismatch rejection.

### IO2 — MAJOR: replay completion is accepted without the actual replay run or trajectory evidence

The validator accepts a replay JSON with status `PASS`, method/dataset/seed, `completed_epochs>=5`, `first_five_trajectory_exact=true` and a success-status string. It does not open the actual retry run receipt or old/new epoch logs and does not recompute the first-five scientific comparison. The independent probe supplied nonexistent run and log paths with those fields; `_validate_runtime_io_amendment` accepted it: `ACCEPTED_PASS_TEXT_AND_REPLAY_NONEXISTENT_RUN_LOGS`.

**Minimum correction:** require actual paths to the target completed formal retry run, its exact R2 implementation snapshot and both the preserved failed-attempt and retry epoch logs. Validate the real full-run status/identity/config/data/source and recompute the specified scientific first-five comparison from those records, including actual present per-cell V and train loss/update/exposure fields. Reject missing logs/run, fewer than five completed epochs, any scientific difference, or a non-R2 retry even when summary flags claim success. Reuse the existing field structures; do not build a general replay framework. Keep final freeze blocked before successful real replay. Synthetic fixtures may provide small real fixture JSON/log files to test this without starting training or reading real E.

## 4. Independent evidence

Command:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_sota_baselines_2526/tests/test_pipeline.py experiments/mdmt_sota_baselines_2526/tests/test_runtime_io.py -q
```

Result: **16 passed in 1.74s**. This passing test result is retained alongside the uncovered evidence-binding gaps. Probe outputs and paths are recorded in the JSON receipt; all probe amendments/review/replay artifacts were temporary synthetic records, never the project's actual pending authorization.

The reviewer inspected the complete before/after data/evaluation diff and new report validator, source inventories, design, implementation and CPU/feature records. Direct source checks established R1 equals PRE_FORMAL_FREEZE and all 20 current successful formal runs, current package equals candidate R2, and all successful run config/source/data fields equal the declared originals. No checksum was computed.

## 5. Corrections made and remaining limits

No implementation correction was made; ownership remains with the pipeline coder. Only this report and its exact-candidate JSON receipt were created. Return IO1/IO2 for minimal correction, then request a focused follow-up review. Previously verified X mathematics, source-scope and CPU equivalence evidence can be reused while unchanged.

Even a later PASS permits only the bounded first retry, not immediate E release. The NumPy SIGSEGV root cause remains unresolved. The full-budget retry must finish and its first five scientific epochs must compare exactly before the remaining nine formal runs proceed. Another native crash or a scientific mismatch triggers the stated stop rule. Successful R1 trajectories and failed attempts must remain unchanged. The final amended freeze requires all 30 valid trajectories and actual completed replay evidence.

## 6. IOFIX1 candidate02 follow-up review

**Verdict: PASS for the bounded replay only.** The reviewer inspected the revised validator and exact resealed candidate02 source pair, the implementation report's IOFIX1 appendix, the current CPU evidence and affected regression cases. No implementation file was changed by the reviewer.

Candidate02 retains the same 18 source paths and 15 unchanged files; only data.py, evaluation.py and report.py differ from R1. Current installed bytes equal the candidate02 R2 map. Previously established R1/PRE_FORMAL identity, unchanged 20 successful R1 run identities, feature-value/ordering/RNG evidence and the minimal resident-X mathematics remain applicable; they were not repeated or expanded into unrelated tests.

### IO1 closure

The validator now parses an actual structured independent-review JSON. Its amendment ID, status and verdict must match, and its `reviewed_pair` must equal all six exact candidate objects: full R1 snapshot, full R2 snapshot, changed paths, original configuration, source commits and prepared-data manifests. The full current CPU and feature evidence dictionaries must also equal those covered by that review. The original text-substring authorization is removed.

The affected test changes candidate R2 data.py and simulated current bytes together to the forbidden eager-Y variant while retaining the old reviewed pair. The validator rejects it. Each actual formal run is now checked against the amendment's original configuration/source/data through `_validate_run_originals`; a changed source declaration is rejected before the matrix can freeze. Exact byte-pair/current-source checks and all original matrix/checkpoint guards remain active. **IO1 CLOSED.**

### IO2 closure

Replay validation now opens the fixed archived HyperGate/KuaiRand/2022 failed run and crash receipt, derives the archived R1 snapshot location without rewriting the old run metadata, and verifies exact R1 bytes and original run identities. It opens the referenced retry run and JSONL file, requires the correct completed formal method/dataset/seed, actual R2 snapshot and actual selected/fixed-final paths, then parses the first five complete ordered train records from both logs.

Those records must include train loss, update/exposure counts, components, temperature and validation covering all six cells with macro metrics. The comparison excludes only `seconds`; every other present scientific/support field must be identical. The tests accept the complete mixed R1/R2 synthetic matrix only with this actual-record fixture and reject nonexistent run/log paths despite success flags, fewer than five epochs, a changed train-loss value and a retry containing R1 source. The normal global freezer still independently opens and checks every actual matrix checkpoint. **IO2 CLOSED.**

### Independent command and result

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_sota_baselines_2526/tests/test_pipeline.py experiments/mdmt_sota_baselines_2526/tests/test_runtime_io.py -q
```

**16 passed in 1.77s.** The validator test reads existing failed-attempt metadata and logs, but replacement run/log/checkpoint records are temporary synthetic fixtures. This is not the real R2 replay, and no real training, retry, E-label access, dependency installation or checksum computation occurred.

The new `RUNTIME_IO_REVIEW_RECEIPT.json` binds this exact candidate02 pair and the unchanged actual-feature/current CPU evidence. It records candidate01's failed-review archive and the bounded release scope. The actual amendment still has pending replay and cannot pass final-freeze validation merely because this independent review now passes.

### Resumption conditions and limits

The orchestrator may now populate the review receipt reference and start **only HYPERGATE/kuairand/2022**, using the original full training budget/configuration/device/RNG setup plus the reviewed resident-X loader. Preserve both failed attempts and the 20 successful R1 trajectories exactly. Compare the actual retry's first five scientific epochs and require successful normal completion. Any difference or another native crash stops further launches under the existing design. Only then may the remaining nine formal runs resume, and only the complete 30-run amended freeze with actual replay evidence may precede E.

This PASS establishes the bounded implementation/evidence contract; it does not locate or prove resolution of the native NumPy fault. Actual R2 replay, GPU stability and final empirical results remain unmeasured by this review.
