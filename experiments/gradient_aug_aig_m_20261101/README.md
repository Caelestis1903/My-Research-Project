# AIGM01 finite experiment

Frozen specification: [DESIGN_SPECIFICATION](../../research_reports/2026-11-01_aig_m/DESIGN_SPECIFICATION.md).
The package reuses the unchanged GAM01/GA31-R3 data, model, objective, Gaussian view, metrics and memory transport.

From the repository root, after independent review and an immutable code commit:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_aig_m_20261101/tests/test_semantics.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_aig_m_20261101.run --mode check --device cuda:1 --code-commit CODE_SHA --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_NEW_check
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_aig_m_20261101.run --mode all --devices 0,1 --code-commit CODE_SHA --stage0-receipt artifacts/gradient_aug_aig_m_20261101/attempt_NEW_check/stage0_checks.json --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_NEW
```

S1/S2/S3/S4 run 16/12/8/8 full development trajectories. S5 reuses strictly identical development RAM trajectories and trains the remaining seeds. Each dataset independently admits or rejects AIG; the nominal budget is 64/69/74 and the actual unique trajectory count subtracts explicitly logged RAM cache reuses. Four shadows share every online trajectory. Five-seed confirmation includes development seeds 2022/2024.

`S*_decision.json`, `progress.json`, and each worker's `stdout.log` expose stage progress. Thresholds are computed only from S95 E1 training statistics. `formal_configuration_frozen.json` fixes methods and shadow rates before S5; `selection_frozen.json` fixes model/epoch identities before unified test. No weights or checkpoint files are written. A process exit releases RAM states; restart requires fresh complete trajectories.

Offline numerical tables, after the complete run:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_aig_m_20261101.analyze --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_NEW
```

Training CUDA-event intervals run from each batch's input transfer through online/shadow updates, including host scheduling gaps; validation/test event sums cover model forwards and exclude CPU metric calculation. All costs remain separate from wall time and are affected by concurrent GPU work. The AIG logged scalar `reference_fixed_mixed_loss` is the fixed-maximum-mixture reference; its parameter-group-dependent update has no single matching scalar objective.
