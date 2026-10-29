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

When using that existing environment, set `PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` and skip the create/install commands. The initial GPU snapshot preceded execution. Recheck resources with `audit`; this campaign used only `cuda:1`, as recorded in `RUN_AUTHORIZATION.json`. Set `DEVICE=cuda:1` for the train/evaluate commands below after confirming that device is available.

## Restore metadata in a published checkout

The connector could not upload the large complete JSON records. They are published as exact byte parts, listed in [PUBLICATION_PARTS.json](PUBLICATION_PARTS.json); the working workspace retains the complete originals. In a fresh GitHub checkout, run this once from the repository root after setting `PY` above. The source package and result CSVs are ordinary files.

```sh
"$PY" - <<'PY'
import json
from pathlib import Path
index = json.loads(Path('research_reports/mdmt_baselines_2526/PUBLICATION_PARTS.json').read_text())
for entry in index['files']:
    data = b''.join(Path(part).read_bytes() for part in entry['parts'])
    assert len(data) == entry['bytes'], entry['path']
    target = Path(entry['path'])
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
print('Complete metadata records restored without hashing or dependency changes')
PY
```

Before publication, every part set was concatenated and compared directly with its original bytes. No JSON values, source snapshots or historical receipts were rewritten for this transport workaround. See [PUBLICATION_ARTIFACTS.md](PUBLICATION_ARTIFACTS.md) for the file list.

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
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method HYPERGATE --dataset movielens --seed 2022 --phase development --device "$DEVICE"
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method AUTOIFS --dataset movielens --seed 2022 --phase development --device "$DEVICE"
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method HYPERGATE --dataset kuairand --seed 2022 --phase development --device "$DEVICE"
"$PY" -m experiments.mdmt_sota_baselines_2526 train --method AUTOIFS --dataset kuairand --seed 2022 --phase development --device "$DEVICE"
```

Once matching seed-2022 trials are reused, add only the four seed-2023 SOTA trajectories. Skip any already-complete trajectory only after confirming its full identity; `train` refuses nonempty output directories.

```sh
# Assumes all four canonical seed-2023 paths are absent. If one exists,
# verify its identity and omit only that dataset/method iteration.
for dataset in movielens kuairand; do
  for method in HYPERGATE AUTOIFS; do
    "$PY" -m experiments.mdmt_sota_baselines_2526 train \
      --method "$method" --dataset "$dataset" --seed 2023 --phase development --device "$DEVICE"
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
        --method "$method" --dataset "$dataset" --seed "$seed" --phase formal --device "$DEVICE"
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

For the approved runtime-I/O repair, retained successful runs may carry exact source revision R1 while the bounded HyperGate/KuaiRand/2022 replay and subsequent runs carry R2. The ordinary `freeze` command above keeps strict single-source behavior and rejects a mixed matrix. Use the explicit amendment API only after the source-equivalence record contains independent review PASS, both-dataset x-only CPU equivalence, and the exact full-run replay receipt; while that record is pending, this call must reject:

```sh
"$PY" -c 'from experiments.mdmt_sota_baselines_2526.report import build_freeze; build_freeze("artifacts/mdmt_baselines_2526/runs", "artifacts/mdmt_baselines_2526/final_freeze.json", runtime_io_amendment="research_reports/mdmt_baselines_2526/RUNTIME_IO_SOURCE_EQUIVALENCE.json")'
```

The amended freeze binds every run to its direct-byte R1 or R2 snapshot, preserves both checkpoint identities, and binds E evaluation to the current approved R2 package. The bounded replay is only the original HyperGate/KuaiRand/seed-2022 trajectory from initialization; compare the first five complete epochs exactly and finish its unchanged full training protocol before launching the remaining nine formal entries. If the first-five comparison differs or a native crash recurs, stop and retain the attempt evidence.

E is read only after that freeze accepts the complete formal matrix. The following loop evaluates each formal checkpoint view against E and records the formal identity. It is included as a procedure, not authorization to read E before the run gate is approved.

```sh
for dataset in movielens kuairand; do
  for method in M3OE_FIXED1 HYPERGATE AUTOIFS; do
    for seed in 2022 2023 2024 2025 2026; do
      run_dir="$RUN_ROOT/formal/$dataset/$method/seed$seed"
      for view in selected fixed_final; do
        "$PY" -m experiments.mdmt_sota_baselines_2526 evaluate \
          --run-dir "$run_dir" --role E --view "$view" --freeze-manifest "$FREEZE" \
          --method "$method" --dataset "$dataset" --seed "$seed" --device "$DEVICE"
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

The environment, pinned-source identities, data receipts, implementation reports and run identities are part of the result record. The initial guide was prepared before execution. RUN was approved on 2026-10-29; actual completion is recorded in the execution reports, rather than inferred from the commands here. No dependencies or global shell defaults were changed.

## Runtime incidents and process-local thread profile

Five native SIGSEGV process incidents are preserved in `RUN_INCIDENTS.json`. Resident-X R2 did not eliminate the failure: the third occurred during HyperGate/KuaiRand/2025 validation. The finite D4/D1 diagnostics both passed without reproducing it. A fourth incident occurred under D1 after AutoIFS/KuaiRand/2022 had saved its selected/final checkpoints, during posttraining fixed-final V validation. Single-thread execution is therefore insufficient to eliminate the failure; the exact native root cause remains unresolved.

`DESIGN_RUNTIME_CRASH3_ADDENDUM.md`, `CODER_RUNTIME_CRASH3_REPORT.md` and `RUNTIME_CRASH3_REVIEW_REPORT.md` describe the reviewed external launcher. It keeps the R2 package unchanged, sets OMP/MKL/OpenBLAS/NumExpr limits to1 before library imports and Torch intra/inter-op counts to1, and records each process in a separate sidecar. Example invocation, subject to the recorded replay and full-matrix freeze gates:

```sh
"$PY" research_reports/mdmt_baselines_2526/runtime_crash3_tool.py launch \
  --sidecar "$RUN_ROOT/runtime_profiles/unique_command.json" -- \
  --config experiments/mdmt_sota_baselines_2526/default_config.json \
  train --method HYPERGATE --dataset kuairand --seed 2025 --phase formal \
  --device cuda:1 --output "$RUN_ROOT/formal/kuairand/HYPERGATE/seed2025"
```

Each invocation needs a new sidecar path. The launcher also accepts the existing evaluate CLI argv unchanged; it does not bypass E authorization. The historical campaign contains multiple actual CPU thread profiles, so its wall-time costs are not a controlled thread-performance comparison. Original successful run records are retained without retroactive environment edits.

The fourth incident was diagnosed once under gdb using the existing checkpoints. Both complete V views matched the logged selected/final reuse epochs exactly and the inferior exited normally: `NOT_REPRODUCED`, not a repaired root cause. A separate independently reviewed, one-use `runtime_posttrain_recovery.py` then completed this existing trajectory's bookkeeping and selected inference timing with zero optimizer/backward steps. Its original `TRAINING` run.json bytes, failed-process receipt and checkpoint identities remain preserved under `runtime_posttrain_recovery/AUTOIFS_kuairand_seed2022/`.

For that one trajectory, exact original `training_seconds`, `total_elapsed_seconds` and `peak_cuda_bytes` are null/NA. The actual logged epoch-duration lower bound, original process wall time and new recovery/inference measurements are separately labelled. Do not replace the missing costs with zero or count this existing training both as a recovered seed and an abandoned independent attempt. The recovery tool is restricted to these historical artifacts and is not part of a fresh campaign's normal training procedure.

The fifth incident occurred during AutoIFS/KuaiRand/2023 pruning statistics before selected/final checkpoints were saved; the preceding recovery could not apply. One separately reviewed original-seed, original-budget full training process under gdb completed normally, including pruning and both checkpoints. Its first14 scientific epoch records exactly matched the archived failed attempt except elapsed seconds (`CRASH5_FULL_PROCESS_RAW_RESULTS.json`). Debugger success is not evidence that the native cause was repaired. The remaining original 2024–2026 runs use the same debugger instrumentation for fault visibility, with distinct actual timing provenance and no added seeds or optimizer budget.

For a fresh campaign entirely on the delivered R2 source, the ordinary strict single-source freeze is appropriate. The explicit R1/R2 amendment API above is for this historical mixed-source campaign and requires its actual preserved evidence; a fresh run does not need to recreate failed attempts.

For this recorded historical campaign, after its30 formal entries and60 E views are complete, regenerate the descriptive analysis and historical execution status with the command below. This analysis helper validates the recorded recovery and seven D1 training profiles; for a fresh all-R2 campaign, use the generic report CLI above and retain that campaign's own runtime provenance rather than reusing historical recovery receipts.

```sh
"$PY" research_reports/mdmt_baselines_2526/analyze_results.py
```
