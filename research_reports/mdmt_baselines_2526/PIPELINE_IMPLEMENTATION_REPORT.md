# Pipeline implementation report

Status: **IMPLEMENTATION_COMPLETE; independent review pending.** Real-data training remains `PENDING_APPROVAL`; no train or E-evaluation command was run.

## Delivered

- Strict protocol configuration and executable `audit`, `prepare-data`, `smoke`, `train`, `evaluate`, `freeze`, and `report` commands in `experiments/mdmt_sota_baselines_2526/`.
- Hash-free T0 preprocessing and direct-value row/label/role checks. The four historical preprocessor/manifest JSONs are versioned in `experiments/mdmt_sota_baselines_2526/reference_data/` because the remote reference checkout does not track them. Their source paths are documented beside the snapshot. Recorded historical checksum strings remain provenance only; no digest was run.
- Training over T0∪F, V-only checkpoint selection, Q development evaluation, test-locked checkpoints, and the global 30-run freeze gate before E. Checkpoints retain model, optimizer, stage, RNG, next epoch/sampler position, source/config/data identity, and AutoIFS rewind/selector/ticket/temperature state.
- AutoIFS stage logs record optimizer ownership, rewind epoch, fresh reuse model/optimizer, direct selector transplant equality, gate freeze, and per-domain/task pruning fractions with denominators. Temperature follows the pinned trainer's rewind-before-final-temperature-increase order.
- Metrics/report output includes macro AUC/LogLoss, count-weighted natural AUC/LogLoss, pooled AUC by task, defined support counts, paired same-seed cell deltas, seed-level sample SD, runtime, and explicit `NOT_RUN` entries.
- `audit` reads the checked-in source manifest and static audit, probes actual pinned Git commits and component/license paths, and reports the current data/source inventory. It does not compute hashes.

## Verification

All commands used `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`.

- `python -m pytest experiments/mdmt_sota_baselines_2526/tests -q`: **10 passed**. Coverage includes synthetic prepare/role guards, metric support and weighted natural AUC, paired/report fixtures, checkpoint optimizer/RNG next-step replay, and model adapter checks.
- `python -m experiments.mdmt_sota_baselines_2526 smoke`: **CPU_SYNTHETIC_SMOKE_PASS** for M3OE_FIXED1, HYPERGATE, and AUTOIFS; each passed finite loss, backward, deterministic inference, and checkpoint parity. AutoIFS also passed synthetic search→rewind→prune→reuse with four optimizer updates.
- `smoke --method AUTOIFS --dataset kuairand --seed 2025 --output /tmp/mdmt_sota_smoke_options_20261028_a`: **PASS**, confirming the selected model/dataset/seed/output options take effect.
- `audit --output /tmp/mdmt_sota_baselines_2526_audit.json`: **PASS**. HyperGate, RecStudio, and AutoIFS commits matched the manifest; RecStudio's MIT license was present, AutoIFS's license was absent and is disclosed. The copied reference JSONs and both prepared-array sets were present; `checksum_revalidated=false`.
- Orchestrator-run CPU `prepare-data` receipts report MovieLens 1,000,209 rows and KuaiRand 2,507,666 rows with `READY_DIRECT_VALUE_CHECKS`; row roles and all 8 role × 6 task/domain support counts matched retained manifests by direct values.

These checks are integration evidence only. Smoke used synthetic rows and its scores are not research results. No benchmark training or test evaluation has run.

## Remaining review limits

HyperGate's complete paper configuration was unavailable; its registered values remain labeled `UNVERIFIED_PROPOSAL`. The result can claim a unified official-core adaptation only after review and user approval, not an as-published numerical reproduction. AutoIFS source remains external because its pinned repository has no license file. Independent reviewer assessment and explicit user RUN approval are still required before any real-data training.

## Initial independent review remediation — F3–F7

The initial **FAIL** remains recorded in `REVIEW_REPORT.md`; this section records implementation evidence only and does not claim re-review or PASS.

- **F3:** `build_freeze` now opens all 60 formal selected/fixed-final checkpoint files, verifies their exact run directory and checkpoint paths, embedded protocol/phase/method/dataset/seed/config/source/data/environment/package-snapshot identity, checkpoint stage/epoch/state schema, and run-recorded metadata before registering an entry. E evaluation requires phase `formal`, exact absolute run directory/manifest, exact selected or fixed-final path and direct checkpoint metadata equality with that global entry. `Data` construction no longer scans label values; E labels are indexed only after authorization. Synthetic tests accept an exact formal entry and reject development phase, changed path/config/source, missing checkpoint and altered embedded identity.
- **F4:** pruning now counts strict `gate_logit > 0` as removed, and `<= 0` as retained, with explicit denominator. The fixed positive/negative/zero selector fixture checks 100%, 0%, and strict-zero behavior.
- **F5:** Markdown now renders actual formal selected-E values and seed means/sample SD, separately renders formal fixed-final E and development Q, leaves missing values blank, and reserves an empty custom-method row. The experiment manifest contains run/config/source/data/environment/checkpoint/evaluation references. Pairing checks common protocol, phase, split/data, roles, training and metric identities while allowing model-specific configurations to differ; incompatible split fixtures are rejected.
- **F6:** each valid run output is first written as `STARTED`; caught setup/training/save/evaluation exceptions write `FAILED`, stage/type/message/elapsed/partial-file fields and are re-raised. Each run stores a direct copy of the executed package source and runtime environment without hashes. Selected-checkpoint timing uses a bounded 256-row V batch, five warmups, 25 measured repetitions, CPU/GPU synchronization, and exports P50/P95 plus metadata to `runtime_cost.csv`.
- **F7:** audit/report accept effective method/dataset/seed filters. Evaluation supports custom output directories and optional run identity assertions; external metric paths are written to a run-local evaluation index that report reads.
- **F8 reproduction guide:** the source checkout, development-matrix and fresh-environment edits were delegated to the model coder; their delivery remains pending this pipeline report update.

Targeted evidence on 2026-10-28, using the configured project Python:

- `python -m pytest experiments/mdmt_sota_baselines_2526/tests -q`: **19 passed**. This includes exact formal freeze authorization and failures, report values/provenance/pair checks, failure receipts, source snapshot, CPU timing, CLI filters, and custom-evaluation discoverability.
- `python -m experiments.mdmt_sota_baselines_2526 smoke`: **CPU_SYNTHETIC_SMOKE_PASS** for all three methods, including AutoIFS search/rewind/prune/reuse. No real rows were used.
- `python -m experiments.mdmt_sota_baselines_2526 audit --method AUTOIFS --dataset kuairand --seed 2025 --output /tmp/mdmt_audit_filter_20261028.json`: selected one-trajectory inventory and verified the pinned AutoIFS commit.

No real training, E evaluation, dependency installation, or checksum calculation was performed in this remediation pass.

## Follow-up review remediation — FIX2

The initial **FAIL** and the later reviewer-reproduced FIX1 gaps remain part of the review history. This section records the FIX2 implementation and synthetic verification only; it does not claim reviewer PASS.

- Formal freeze now locks each method's complete `config.models[method]` identically across its datasets and seeds, while allowing different methods to retain their own model configurations. Report aggregation rejects within-method configuration drift across the selected phase and seeds. Paired baseline identity now also includes pinned source commits.
- Freeze reads each run's declared implementation snapshot as direct file bytes, checks its inventory and recorded sizes, and compares every formal run's source bytes. It retains the canonical bytes in the freeze record as base64 text for later exact comparison. E evaluation checks the current run snapshot against these frozen bytes before reading labels; a failed recheck clears any prior E authorization.
- Implementation snapshots recursively include the four nested preprocessor/data-manifest JSON files under `reference_data/{movielens,kuairand}`.
- Runtime environment capture follows `seed_all()`. The failure-receipt fixture starts with both TF32 flags set to `tf32` and verifies the receipt records the actual post-seeding `ieee` values.

FIX2 verification on 2026-10-28, using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m pytest experiments/mdmt_sota_baselines_2526/tests -q`: **21 passed**. Synthetic regressions cover synchronized single-run HyperGate `lambda_domain` drift in both checkpoints, same-size source-byte mutation, E-time source recheck and label lock, method configuration drift across formal seeds, paired source-commit mismatch, all four nested JSON snapshot files, and post-seeding TF32 environment capture.

No real-data training, E evaluation, package installation, or checksum calculation was performed. The review report history remains unchanged and awaits independent FIX2 follow-up review.
