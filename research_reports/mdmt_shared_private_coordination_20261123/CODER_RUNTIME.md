# Coder implementation report for S0 and S1 runtime

## Changes

- `cli.py` connects the bounded Train/Validation audit, frozen panels, source and review receipts, resource audit, real first/tail CUDA fidelity, and RAM recovery smoke to the staged commands.
- `training.py` executes the canonical J_ORIG trajectory with native Adam and EMA. It records actual batch visitation against the canonical permutation, evaluates both readouts after every epoch, and selects their earliest strict Validation Macro AUC maxima independently.
- Each epoch transaction binds online, EMA, native Adam, module modes, Python/NumPy/CPU/CUDA RNG, generator, permutation, cursor, scalar ledger, selected state and run identity. The scalar intent binds the expected packet hash before IPC. A fresh guardian get and hash confirms the commit or recovers an interrupted publication. End-E10/E25/E40 anchors use separate keys. A disposable fork key is read back into an independent clone and released after diagnostics. The supervisor verifies the stored anchor, panel identity and the first two batch hashes against the next actual baseline epoch.
- `scheduler.py` registers ten deterministic logical IDs, one J_ORIG per dataset and seed. Separate attempt IDs identify process costs and retries. Admission uses current per-card memory and power below 85% and current host RAM below 90%. It waits for a fresh sample after startup and uses measured full-process residency and CUDA-usable capacity for physical fit after an actual S1 CUDA OOM. A resource pause preserves the verified epoch and lets the foreground runner resume the same logical run.
- The unchanged v3 `guardian.serve` owns tensor state inside the D2 artifact directory. Its transport closure command is named `failed` and clears a storage slot's worker PID. D2's scientific completion and pause states come from the D2 receipts. The implementation map is [IMPLEMENTATION_MAP.md](../../artifacts/mdmt_shared_private_coordination_20261123/S0/IMPLEMENTATION_MAP.md).

## Checks completed

- The project Python compiled the package and imported the CLI. The runtime unit suite passed all five tests. It covered the registered matrix, admission thresholds and physical fit, 50-epoch permutation and natural tail coverage, strict independent selection, mocked guardian intent recovery and replay of the next native Adam/EMA update, stop controls and the Test seal. PyTorch warned that the frozen TF32 setter API is deprecated.
- `cli prepare` passed on canonical Train/Validation data. [SOURCE_MANIFEST.json](../../artifacts/mdmt_shared_private_coordination_20261123/S0/SOURCE_MANIFEST.json), [DATA_MANIFEST.json](../../artifacts/mdmt_shared_private_coordination_20261123/S0/DATA_MANIFEST.json), and the frozen [MovieLens](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_movielens.json) and [KuaiRand](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_kuairand.json) panel receipts are present. The data manifest reports `test_status=SEALED`.
- `cli review-input` verified the frozen source bytes and passing S0 scalar records and wrote [REVIEW_INPUT.json](../../artifacts/mdmt_shared_private_coordination_20261123/S0/REVIEW_INPUT.json).

## Foreground runner commands

Run from the `phase0-3_mdmt_coordination_20261123` worktree with the current stop controls clear.

```bash
D2_PYTHON=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.cli review-input
```

The independent reviewer should write `research_reports/mdmt_shared_private_coordination_20261123/IMPLEMENTATION_REVIEW.json` with the `design_sha256`, `implementation_hash`, `files` and `evidence` fields from the finalized review input plus their `verdict` and `blocking_issues`. A passing verdict is `PASS` or `PASS WITH FIXES` with an empty blocking list.

```bash
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.cli smoke --gpu 0
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.cli run
```

The smoke's `--gpu` is a physical NVIDIA index. Verify `S0/GPU_SMOKE.json` and `S0/RESOURCE_AUDIT.json` after smoke. S1 writes `S1_RUN_MANIFEST.json`, `S1_RUN.json`, `S1_ADMISSIONS.jsonl` and `S1_PHYSICAL_CAPACITY.json` at the artifact root. Inspect `runs/<logical_id>/attempts/<attempt_id>/` for each worker log, allocation, resume and epoch RAM receipt. The accepted per-run result, epoch metrics, anchors and diagnostic receipts are under `runs/<logical_id>/`. The completed stage writes `ANCHOR_INDEX.json`, `BASELINE_ENDPOINTS.csv` and `COSTS.csv`, then records `COMPLETE`. The corrected matrix contains 30 anchors and 570 named forks.

On a resource pause, keep the D2 guardian and rerun the same foreground command when current admission conditions permit. A failed or exited RAM owner is a technical recovery case. A new D2 owner requires repeating the reviewed smoke with a new `--guardian` directory inside the D2 artifact root and passing that same directory to `run`. Jobs without verifiable RAM replay the frozen logical identity, with past physical attempts retained in the ledger.

## Pending independent checks

The reviewer must assess the final source, the CUDA preflight and live guardian behavior. The coder checks used a synthetic CPU resume fixture and a mocked transport. CUDA smoke, the ten scientific trajectories, Test access and scientific analysis are outside these coder checks. CUDA intervals are timed on the stream around work phases and can include idle gaps during synchronization.
