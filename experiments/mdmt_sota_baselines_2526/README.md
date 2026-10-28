# MDMT SOTA baselines

Package: `MDMT-SOTA-2526-v1`. It adapts pinned HyperGate and AutoIFS source to the repository's MovieLens/KuaiRand row contract and compares them with `M3OE_FIXED1`. HyperGate's three gates remain enabled. AutoIFS retains search, rewind, hard pruning and reuse. The four upstream AutoIFS affine bias dimensions receive the documented output-shape correction.

The default HyperGate configuration contains explicit values marked `UNVERIFIED_PROPOSAL`; its official paper configuration was not available during source audit. The default therefore supports an official-core unified adaptation run, not a claim of exact as-published numerical reproduction. AutoIFS's original KuaiRand schema/split also differs from this repository's input contract.

The two historical `PREPROCESSOR.json` and `DATA_MANIFEST.json` pairs are stored under `reference_data/`. They were copied from the local historical paths recorded in `reference_data/README.md` because the independent repository snapshot does not contain those JSON files. They are read-only reference inputs; prepared arrays are rebuilt separately under `artifacts/mdmt_baselines_2526/data/`.

## Commands

Use the repository Conda environment:

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_sota_baselines_2526 audit
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_sota_baselines_2526 smoke
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_sota_baselines_2526 prepare-data --dataset movielens
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_sota_baselines_2526 prepare-data --dataset kuairand
```

`smoke` uses generated CPU rows and does not open either CSV. `prepare-data` is CPU data preparation; it rebuilds only below `artifacts/mdmt_baselines_2526/data/`, compares T0-derived preprocessing and encoded values directly, and records that the existing raw checksum was not recomputed.

Smoke accepts `--method`, `--dataset`, `--seed`, and `--output` (an empty or new directory). With no method it checks all three; its report identifies the synthetic dataset/seed and output directory. `audit --output PATH` writes the source and data inventory JSON as well as printing it.

Real training commands are intentionally left to the orchestrator after the user's RUN approval. A single trajectory is run as:

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_sota_baselines_2526 train --method HYPERGATE --dataset movielens --seed 2022 --phase development
```

The training command produces a selected and fixed-final checkpoint, V metrics, stage logs and a `TRAIN_COMPLETE_NOT_EVALUATED` run receipt. It never reads test labels. See `research_reports/mdmt_baselines_2526/reproduction_guide.md` for fixed-source checkout and the complete post-approval sequence.
