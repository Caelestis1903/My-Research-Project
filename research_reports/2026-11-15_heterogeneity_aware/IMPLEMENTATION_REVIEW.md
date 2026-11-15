# Independent implementation review

Final verdict: **PASS WITH FIXES — R1 corrected; no blocking issues remain.**

Reviewed final source SHA256: `a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491`.

The review independently compared DESIGN_SPECIFICATION.md, SOURCE_MAP.md, CODER_IMPLEMENTATION_REPORT.md, all seven new Python files and the reused fixed1 model, evaluation and RAM guardian paths. No research training or Test prediction was run.

## R1 correction and final review

The orchestrator approved DESIGN_SPECIFICATION section9 before this re-review. The correction is confined to `run.py::check_sensitivity` and its smoke call. The helper copies the original FP32-initialized coefficients into an FP64 numerical probe, uses fixed step1e-5 at the already declared strict double-precision tolerances, and separately checks the actual FP32 sensitivity against that same-coefficient probe at the unchanged FP32 tolerances. Ordinary FP32/probe logit equality is still exact. All full-batch training, augmentation and evaluation paths retain their original FP32 configuration.

The corrected helper was verified on the same exact four actual-source rows in each dataset. MovieLens/KuaiRand maximum finite-difference absolute discrepancies were `3.50265e-11` / `2.95627e-11`; maximum FP32-versus-FP64 sensitivity discrepancies were `2.60029e-7` / `5.18891e-7`. Both checks passed without changing comparison tolerances or training hyperparameters. The reviewer inspected the helper, approved design amendment, corrected coder report and stable final source identity, and reused this actual-data verification evidence. R1 is resolved.

The final review permits the planned two-dataset GPU smoke, followed by research runs only when that smoke passes. GPU full-batch feasibility and diagnostics remain the runner's next required gate.

## Prior review history — preserved FAIL evidence

Initial verdict: **FAIL** on source `fa4db1090ec06d35dc205f412afa6b3e58880b629fb71c6f3be199fccdbff50e`.

**R1 — Relevant / previously blocking, now resolved: the initial KuaiRand smoke finite-difference probe failed at its fixed step.**

`experiments/heterogeneity_aware_20261115/run.py:212–215` uses an absolute embedding displacement of `1e-3` and checks the central difference against the local derivative with `rtol=1e-2, atol=1e-4`. The source sparse embeddings initialize with standard deviation `1e-4`. On the exact real four rows selected by the smoke seed and loader, the KuaiRand initial model does not satisfy this finite-step local approximation. The same CPU reproduction passes for MovieLens.

Actual KuaiRand row2/task0: finite difference `-0.3065168857574463`, full-function autograd derivative `-0.29870766401290894`; absolute discrepancy `0.0078092217445373535`, relative discrepancy `0.026143359020352364`. One of eight entries fails. Thus the specified two-dataset smoke cannot currently complete under the implemented numerical check.

Fix only the numerical probe: use a smaller fixed step that passes at an actual smooth source-model point, or select a smooth point as the design requires. Keep the declared comparison tolerances, complete function derivative, model, training objective, perturbation radii and training budget. Verify the corrected rule on the actual KuaiRand source model/rows and refresh the implementation report/source hash. This finding does not establish an error in the augmentation Jacobian.

## Verified implementation contracts

- The probe reconnects `star_mlp + skip_conn` to gate pre-hooks only during sensitivity extraction; ordinary training retains the source gate detach. Tower lookup uses the source `task*3+domain` ordering. Hooks are removed in `finally`.
- Sampling, exact rank-one projection, zero-gradient skips, pair cap, two halvings, target-only auxiliary BCE and detached perturbations follow the specification. RANDOM uses Rademacher coordinates; GAUSSIAN uses normalized Gaussian coordinates. `B_MATCH` uses A's accepted norms/support and B's unprojected direction.
- Clean loss is source sample-mean probability BCE. Each training batch contains one backward, Adam step and after-Adam EMA update. Source optimizer settings, natural shuffle, batch40960 and 50epochs are preserved.
- Five allowed seeds and full50 checks precede validation-based selection. HPO is bounded. Test evaluation loads only states listed in a frozen identity record; development code does not predict on Test.
- The reused guardian owns selected/continuation states in RAM through BytesIO IPC. Only data arrays/metadata use pickle disk caching. Cache creation checks source-derived identities; worker records retain model/config/row-order hashes without writing weights to disk.
- Per-GPU admission checks memory and power independently at85%, checks host RAM at90%, and uses the measured full-batch peak. The scheduler does not lower batch size on OOM.
- Flat exports preserve all six cells, Macro/Traffic/Pooled metrics, independent online/EMA selection and seed-paired statistics.

## Validation evidence

Coder's five compact CPU checks, syntax check and CLI-help evidence were inspected and reused. The reviewer additionally ran the exact implemented finite-difference rule, on CPU, with audited real dataset preprocessing, source model factory, seed2022 and the first four rows of the source loader order. MovieLens passed; KuaiRand failed as recorded above. This command built no disk cache and accessed no Test prediction/metric.

R1's correction and passing final review are recorded above. No unrelated validation matrix was added.
