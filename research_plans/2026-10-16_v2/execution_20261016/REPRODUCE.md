# Reproduction and artifact scope

The execution reports and machine-readable ledger determine completion. A running process, an interim analysis receipt, and a complete producer manifest are different states. Do not launch duplicate workers while either queue remains active.

Working tree: `/home/xzf/Code/phase0-3-20261016-git`, branch `codex/execute-2026-10-16-v2`. Python: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`. Run commands from this working tree with `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=2`.

## Frozen training

`wave1.json` is the exact training manifest. It records every argument and unique attempt directory. `wave1_status.json` records the resolved command, PID, device, timestamps, and exit code. Every successful run contains its config, source hashes, artifact hashes, complete epoch metrics, batch row hashes, exposure, selection freeze, optimizer observations, and predictions. Calibration is validation-only; its frozen decision is `sgd_calibration_decision.json`.

The base training worker and GPU scheduler are pinned to `978592f1f089285888aa1da97be3972a6c82a18f`. AdaTask's additional worker is pinned to `13a469b2fc0e70881a61e48fe2706ae574c58144`. Use the manifest's per-task pin, not the final report commit, to reconstruct the actual training source. Training refuses source drift. Do not edit producer code while runs are active.

The persistent scheduler entry point is `experiments/v2_queue_20261016.py`; inspect its CLI before a recovery launch. Successful attempt directories are immutable and skipped. An incomplete attempt must be audited and resumed with a verified recovery mechanism or assigned a new attempt directory; it must never be relabeled successful. No seamless mid-epoch resume is claimed.

## Readout order

1. Run `experiments/v2_analyze_20261016.py --phase FINAL` only after training finishes; it recomputes selected prediction metrics, binds rows and labels to the raw files, checks hashes and pairing, and generates aggregate/per-cell/paired tables.
2. Run `experiments/v2_p1_decision_20261016.py` after all 16 P1 Sample runs and the P1 numerical analysis. Its output requires primary scientific review and does not launch a branch.
3. `experiments/v2_readout_queue_20261016.py` mechanically runs the preregistered 16 Adam validation-grid selection bootstraps, eight natural-arm gradient probes, and four weight-equivalence audits. Successful receipt directories must not be overwritten. These analyses use immutable states and do not update the model.
4. Run `experiments/v2_probe_20261016.py --differences`, then `experiments/v2_domain_analysis_20261016.py`, `experiments/v2_geometry_analysis_20261016.py`, and `experiments/v2_exposure_analysis_20261016.py` after all applicable producer and probe receipts exist.
5. Shadow-state counterfactuals (`experiments/v2_shadow_20261016.py`) require P1-A; their existence in the repository does not mean that their conditional branch was triggered or executed.

The analysis review receipt contains the exact analysis source hashes. Reports distinguish primary self-review from independent review, consistent with the invoked `start.md` delegation restriction.

## Data and model states

Raw KuaiRand: `/home/xzf/data/kuairand/processed/kuai_rand.csv`; raw MovieLens: `/home/xzf/data/ml-1m/ml-1m.csv`. Dataset hashes and preprocessing provenance are in `dataset_provenance.json` and each run's config. Seeds: 2022, 2023, 2024, 2025. KuaiRand has 49 updates per epoch; the MovieLens `d2_large` training configuration has five. Cross-environment training budgets are therefore not equivalent.

Raw datasets, full `.pt` states, required-epoch prediction arrays, engineering profiles' temporary tensors, and complete stdout logs stay on the execution host. The repository retains source/config/provenance, metrics, selected validation/test predictions, selected-state hashes, probe pools and projections, diagnostics, and reports. Producer manifests intentionally include hashes of both tracked evidence and local-only files. They are not claims that every hashed artifact was uploaded.

Tests are forensic readouts because the project has previously used them. No checkpoint, hyperparameter, conditional branch, or rerun is selected using test performance. The 200 validation-row bootstraps quantify conditional uncertainty on the seven saved selectable epochs, not all 50 epochs or independent retraining uncertainty.

## Scientific limits

Cyclic controls preserve the exact realized domain-count sequence within each q, and change within-domain repetition, coverage, and temporal dependence jointly. Coverage-threshold comparisons use unequal budgets and coarse epoch resolution. Neither is an exact exposure-only causal intervention. Gradient covariance is measured on eight disjoint 32-row microbatches per domain in 256 selected coordinates per block. Spectra and synthetic fixed-theta autocorrelation do not establish full-coordinate production-gradient covariance or performance mediation.

The historical Stage2D learned-fusion/two-optimizer protocol is kept separate from the explicitly required v2 fixed-fusion/single-optimizer protocol. A weaker new-protocol replication cannot retroactively invalidate a hash-verified historical result.
