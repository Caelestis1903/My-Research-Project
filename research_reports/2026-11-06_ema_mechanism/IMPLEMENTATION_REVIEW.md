# Independent implementation review

Verdict: **PASS WITH FIXES**. R1 and R2 below are corrected and checked. There are no remaining blocking findings for the frozen experiment.

Reviewer: independent reviewer agent, separate from all three implementation agents. Review date: 2026-11-06. Reviewed the approved `DESIGN_SPECIFICATION.md`, all three `CODER_IMPLEMENTATION_REPORT_*` files, the actual new package and tests, and the frozen plan's mathematical, implementation and evaluation contracts.

## Reviewed identity

| Item | Identity |
|---|---|
| Source commit | `40705dfe0bb4fe614bdca835c0bae567348c87d0` |
| Plan commit | `9fb59d5a53bf0efa555caeaca03e1ffa3812b686` |
| New implementation SHA256 | `31850f6a4f137b6f37f9dfefaf7c6b8de8d41053f5a132a77bdc50d36ed8b11b` |
| Audited source-files SHA256 | `49c109a753bdd07ad5acf7c2e919830d7611296eb839fba1b73efc917a2c26c0` |
| Manifest SHA256 | `b65ae44febf828cb0cf844bf8d0226567921d6e675396a8b6beed7e50c3640dd` |
| Plan-files SHA256 | `d9cecac2ab0fc87f53c425b355192ee6198354ddfe8cf22a7dc65595129534da` |

The implementation digest is produced by `protocol.implementation_identity()` over all Python files in the new package, including its tests. Existing tracked source files have no working-tree changes. The sixteen frozen source-file SHA256 and Git-blob checks pass through the actual resolver.

## Findings and corrections

| Finding | Effect on this experiment | Correction and check |
|---|---|---|
| R1 — FP32 CUDA scatter accumulation in observation reductions | M2 direction scores lacked the specified ordered, sufficiently precise accumulation; atomic variation could also conflict with exact epoch replay identity. | Fixed twelve-group order and coordinate-preserving float64 sums replace scatter atomics. Innovation/geometry differences and products, M2 displacement differences, gradient products and energies are formed in float64 before reduction. Source training and EMA arithmetic are unchanged. Cancellation/repeat, finite-difference, source-equality, RAM-continuation and observer-schema checks pass. |
| R2 — guardian tensor payload was counted but actual guardian RSS was absent | The required resource report would omit guardian process memory and conflate the model-payload cap with process memory. | Guardian status records actual `VmRSS` and `ru_maxrss`; committed epoch costs and RAM summaries include `peak_guardian_ram_bytes`. The existing COSTS maximum aggregation retains that field. Smoke receipts include worker RSS. The commit/RSS check passes. |

## Specification checks

- Source clean/G-D0 loss is called directly, retaining its clean reduction order, replacement semantics, actual-tail denominator and original Gaussian noise identity. Adam settings, initialization, loader order, source gate detach and after-Adam F update match the frozen protocol.
- Observer functions operate through scratch/functional parameters. Persistent physical EMA aliases, pulse start/end clocks, group-coordinate copies, E20/E40 retention, strict complete-Validation selection and selected/E10/E50 RAM snapshots correspond to the specification. The schema retains 72 cell/group rows and marks the frozen 24 supported relationships.
- Fixed training panels use disjoint hashed row pools and independent generators. V membership comes from Validation; its row side is seed-independent. Rank pairs, panel hashes and identities are persistent. Original FP32 probabilities feed AUC; the original float64 log-floor convention feeds LogLoss.
- Both full-functional and source-detach M2 are computed on P_A only. M0/M1/M3 and P_B replications have the required direction and population definitions. Full functional derivative and source-forward equality checks pass.
- Future targets come from the new pulse at origin+5. The implementation uses other four seeds' early origins and one V row side for fitting, the held-out seed's late origins and opposite side for scoring, then exchanges sides. Fold counts are 576/240. Continuous scaling fits the training fold only; ridge uses summed squared error plus alpha=1 with an unpenalized intercept.
- All eight fixed ridge models, all five simple references, four raw diagnostics and a fit-fold uniform action are retained. F is an allowed action with the specified tie order. Summaries aggregate correlated observations within seed and retain every registered condition/cell/group/cell-group scope.
- C_CAL and C_BG use their registered five-seed triggers. C_BG freezes at most two groups and reads retained RAM vectors; temperature changes only the LL diagnostic. Neither control starts training.
- Source/plan/implementation/data/panel identities, RAM-first epoch transactions, rollback, replay comparisons and finite restart limits are present. Prediction NPZ fields, dtype/shape and byte caps are checked; parameters and optimizer state use RAM IPC. The scheduler admits at most two workers with the per-device memory/power and host-RAM limits.
- Freeze checks all twenty jobs, all fifty epochs, five seeds per condition, exact selected epochs and RAM endpoint hashes. The independent Test subprocess verifies the binding before opening Test arrays, reuses pinned same-state archived endpoints and deduplicates physical aliases/states.
- S0 actually extracts the available A/B/C complete Validation curves and HPO tables into `ARCHIVED_VALIDATION_CURVES.csv` and `ARCHIVED_HPO.csv`, with source path/commit/hash. It also reconstructs the archived factorial and seed-paired interactions. The implementation therefore covers the required zero-training curve extraction.
- Report generation consumes the real endpoint, pulse, prediction and control tables, generates the required PNG/PDF figures and checks required artifacts and the observation disk cap. The orchestrator supplies the final scientific interpretation from the completed results.

## RAM payload check

The reviewer exercised the actual `export_session` structure on both source models, filling every physical readout's selected/E10/E50 entries and both retained F/fast/slow anchors. `cpu_copy` performed its actual recursive copies; an upper bound for all Adam first/second moments and scalar steps was added. The worker stores scalar `best_scores`, with no duplicate selection-state container.

| Dataset | Complete-job payload upper bound |
|---|---:|
| MovieLens | 74,266,156 bytes |
| KuaiRand | 180,809,592 bytes |
| Ten jobs per dataset combined | **2,550,757,480 bytes** |
| Guardian model-state cap | **4,294,967,296 bytes** |

Active pulse windows occur before E50 and have fewer fixed endpoint snapshots, so the complete-job bound also covers the retained model payload of active jobs. Process RSS, IPC copies and read-only data mappings are accounted separately by the runtime measurements.

## Verification and runtime handoff

Independent project-environment check: `python -m pytest experiments/ema_mechanism_20261106/tests -q` — **24 passed**. After the RSS-only revision, the affected commit/rollback/RSS test was independently rerun — **1 passed**; the coder's complete core suite also reports **9 passed**. The unchanged source setup emits its PyTorch TF32 API deprecation warning.

The reviewed implementation may proceed through `audit`, `preflight`, `validate`, both original-batch dataset smokes and the fixed twenty-job collection, followed by S1–S4, endpoint freeze, independent Test evaluation, report and release. Runtime validation and both smoke receipts must bind the implementation hash above. Copy this review to the run output before `train`.

GPU smoke and scientific trajectories were not run during review. Their measured resource and full-budget results are produced by the prescribed RUN stage.
