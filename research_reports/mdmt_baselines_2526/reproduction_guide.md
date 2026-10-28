# Reproduction guide

This guide records environment setup, pinned-source checkouts, data preparation and the approved run matrix for `MDMT-SOTA-2526-v1`. **Every real-data train command below requires the user's explicit RUN approval.** Review PASS and data readiness alone do not authorize training. Do not infer experimental results from CPU smoke output.

## Python environment

For a new environment at the repository-approved prefix, run these commands once. The lock file already sets the PyTorch 12.6 wheel index; do not edit it.

```sh
source /home_new/xzf/envs/miniconda3/etc/profile.d/conda.sh
ENV_PREFIX=/home_new/xzf/envs/miniconda3/envs/phase0-3-env
conda create --prefix "$ENV_PREFIX" python=3.13.5 pip
conda run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install -r requirements-phase0-3.lock
conda run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install PyYAML==6.0.2
PY="$ENV_PREFIX/bin/python"
```

`PyYAML==6.0.2` is not listed in the existing lock, but the reused `mdmt_basp_v1` initializer imports the repository config through PyYAML. It is already present in the inspected project environment; the extra command is only for a genuinely empty environment and does not change the lock. Do not install AutoIFS's historical `requirments.txt` into this environment.

The checks for this deliverable used the existing `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` environment recorded in `ENVIRONMENT.json`. A fresh environment was not created or installed during this task.

When using that existing environment, set `PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` and skip the create/install commands. The recorded GPU snapshot shows both devices occupied; use `audit` to recheck resources after approval, before scheduling any training.

## Pinned external source checkouts

Use these paths because they exactly match `default_config.json`. The parent HyperGate commit retains its RecStudio gitlink; a separate RecStudio checkout at that same commit is placed at the configured runtime path, so no nested path override is needed.

```sh
SOURCE_ROOT=/tmp/mdmt_sota_sources_2526
mkdir -p "$SOURCE_ROOT"

git clone --filter=blob:none --no-checkout --branch main \
  https://github.com/XuHwang/HyperGate.git "$SOURCE_ROOT/HyperGate"
git -C "$SOURCE_ROOT/HyperGate" checkout fdc82e3f3feee23694a8123f91092615f977900e
git -C "$SOURCE_ROOT/HyperGate" ls-tree HEAD RecStudio

git clone --filter=blob:none --no-checkout --branch multi-domain-multi-task \
  https://github.com/XuHwang/RecStudio.git "$SOURCE_ROOT/RecStudio"
git -C "$SOURCE_ROOT/RecStudio" checkout f08cca607e998e8be4c5e915a393346baffa71f6

git clone --filter=blob:none --sparse --branch main \
  https://github.com/ChaohuaYang/AutoIFS.git "$SOURCE_ROOT/AutoIFS-pinned"
git -C "$SOURCE_ROOT/AutoIFS-pinned" sparse-checkout set --no-cone \
  '/README.md' '/requirments.txt' '/autoifs_trainer.py' \
  '/modules/autoifs.py' '/modules/layers.py' '/utils/**' '/dataloader/**'
git -C "$SOURCE_ROOT/AutoIFS-pinned" checkout 65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e
```

The HyperGate `ls-tree` output should identify the unchanged RecStudio gitlink at `f08cca607e998e8be4c5e915a393346baffa71f6`. The separate checkout above places that pinned source at `/tmp/mdmt_sota_sources_2526/RecStudio`, matching the default config.

The AutoIFS sparse checkout includes the loaded core/layers, its trainer and import support, README and historical requirements inventory. It omits the large `data/ml-1m.zip` archive and the original data-conversion scripts. The new adapter reads this checkout's `modules/autoifs.py` and `modules/layers.py`; it does not use the original TFRecord trainer. AutoIFS has no LICENSE file in the pinned checkout; its source remains external and is not copied into this package.

If any target directory already exists, inspect its Git HEAD and sparse paths rather than cloning over it. The package `audit` command checks the configured roots and pinned commits.

## Audit, synthetic checks and data preparation

These checks do not train on real rows:

```sh
"$PY" -m experiments.mdmt_sota_baselines_2526 audit
"$PY" -m experiments.mdmt_sota_baselines_2526 smoke
"$PY" -m pytest experiments/mdmt_sota_baselines_2526/tests -q
```

`audit` is read-only and reports configured paths, pinned commits, source components, license presence, local data and GPU inventory. `smoke` uses generated CPU rows, not either CSV. It accepts optional `--method`, `--dataset`, `--seed` and `--output`; without a method it checks all three adapters, including the synthetic AutoIFS search→rewind→ticket→reuse transition. Smoke scores are not experiment results.

`DATA_READINESS.json` records the currently restored MovieLens and KuaiRand arrays and direct-value checks. On a clean workspace where the owned arrays are absent, rebuild them only at absent output paths:

```sh
"$PY" -m experiments.mdmt_sota_baselines_2526 prepare-data --dataset movielens
"$PY" -m experiments.mdmt_sota_baselines_2526 prepare-data --dataset kuairand
```

The raw CSVs must exist at the paths in `default_config.json`. Preparation refuses to overwrite existing destinations, refits preprocessing on T0, compares the encoded values and role rows directly, and records `checksum_revalidated=false`. The versioned files under `reference_data/` are historical preprocessing/manifest JSONs, not row arrays. No checksum is recomputed.

## Development matrix after approval

The development budget is **8 SOTA trajectories**: `HYPERGATE` and `AUTOIFS` × MovieLens and KuaiRand × seeds 2022 and 2023. `M3OE_FIXED1` is formal-only.

First inspect the four canonical seed-2022 trial paths under:

```text
artifacts/mdmt_baselines_2526/runs/development/{movielens,kuairand}/{HYPERGATE,AUTOIFS}/seed2022
```

If a trial has a complete run receipt and exactly matches the approved development budget, config, source, data and metric identity, reuse it as that development trajectory. Do not issue `train` again into its nonempty output directory. If a matching seed-2022 trajectory is absent, run that method/dataset once using the command below; if an existing path is partial or has a different identity, stop and resolve it rather than overwrite it.

```sh
# Execute only the lines whose canonical seed-2022 path is absent.
# Reuse a completed matching trial by leaving its line unexecuted.
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method HYPERGATE --dataset movielens --seed 2022 --phase development
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method AUTOIFS --dataset movielens --seed 2022 --phase development
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method HYPERGATE --dataset kuairand --seed 2022 --phase development
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method AUTOIFS --dataset kuairand --seed 2022 --phase development
```

Once matching seed-2022 trials are reused, add only the four seed-2023 SOTA trajectories. Skip any already-complete trajectory only after confirming its full identity; `train` refuses nonempty output directories.

```sh
# Assumes all four canonical seed-2023 paths are absent. If one exists,
# verify its identity and omit only that dataset/method iteration.
for dataset in movielens kuairand; do
  for method in HYPERGATE AUTOIFS; do
    "$PY" -m experiments.mdmt_sota_baselines_2526 train \
      --method "$method" --dataset "$dataset" --seed 2023 --phase development
  done
done
```

Do not run M3OE_FIXED1 in development or repeat seed-2022 trajectories that were already completed under the same full development identity. The four seed-2022 and four seed-2023 SOTA trajectories make the eight-trajectory budget.

## Formal matrix, freeze and E evaluation after approval

After development, freeze the model, source, data, config and selection rules. The formal comparison is **30 fresh trajectories**: all three methods × both datasets × seeds 2022–2026. Formal seed-2022/2023 runs are new trajectories in the `formal/` output tree; do not reuse development weights. Seeds 2024–2026 are unseen during development. Run this loop once against empty formal output paths; the trainer refuses to overwrite an existing run directory.

```sh
RUN_ROOT=artifacts/mdmt_baselines_2526/runs
for dataset in movielens kuairand; do
  for method in M3OE_FIXED1 HYPERGATE AUTOIFS; do
    for seed in 2022 2023 2024 2025 2026; do
      "$PY" -m experiments.mdmt_sota_baselines_2526 train \
        --method "$method" --dataset "$dataset" --seed "$seed" --phase formal
    done
  done
done
```

Only after all 30 formal run receipts complete, create the global freeze:

```sh
FREEZE=artifacts/mdmt_baselines_2526/final_freeze.json
"$PY" -m experiments.mdmt_sota_baselines_2526 freeze \
  --run-root "$RUN_ROOT" --output "$FREEZE"
```

E is read only after that freeze accepts the complete formal matrix. The following loop evaluates each formal checkpoint view against E and records the formal identity. It is included as a procedure, not authorization to read E before the run gate is approved.

```sh
for dataset in movielens kuairand; do
  for method in M3OE_FIXED1 HYPERGATE AUTOIFS; do
    for seed in 2022 2023 2024 2025 2026; do
      run_dir="$RUN_ROOT/formal/$dataset/$method/seed$seed"
      for view in selected fixed_final; do
        "$PY" -m experiments.mdmt_sota_baselines_2526 evaluate \
          --run-dir "$run_dir" --role E --view "$view" --freeze-manifest "$FREEZE" \
          --method "$method" --dataset "$dataset" --seed "$seed"
      done
    done
  done
done
```

Create the comparison only after the required evaluations are present:

```sh
"$PY" -m experiments.mdmt_sota_baselines_2526 report --run-root "$RUN_ROOT" \
  --output research_reports/mdmt_baselines_2526 \
  --method M3OE_FIXED1 HYPERGATE AUTOIFS --dataset movielens kuairand \
  --seed 2022 2023 2024 2025 2026
```

## Interpretation and provenance limits

HyperGate uses the registered `UNVERIFIED_PROPOSAL` values because a complete paper configuration was unavailable. Results may be described as an official-core unified adaptation, not as exact as-published numerical reproduction. AutoIFS uses the documented row-schema/selection adaptation and its external pinned source; its original TFRecord/KuaiRand schema is not claimed identical to this protocol. The historical E partition is already exposed and must be labeled as reassessment.

The environment, pinned-source identities, data receipts, implementation reports and run identities are part of the result record. Commands in the post-approval sections are documentation only; no real training, E evaluation, dependency installation or fresh environment creation was performed while preparing this guide.
