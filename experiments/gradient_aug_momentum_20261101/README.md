# GAM01

Frozen design: `research_reports/2026-11-01_momentum/DESIGN_SPECIFICATION.md`.
The package imports the unmodified `experiments.gradient_aug_20261031_v2` namespace.

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_momentum_20261101/tests/test_semantics.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.run --mode check --run-dir artifacts/gradient_aug_momentum_20261101/attempt_<timestamp>_check --device cuda:1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.run --mode all --run-dir artifacts/gradient_aug_momentum_20261101/attempt_<timestamp> --devices 0,1 --code-commit <reviewed-implementation-commit>
```

Run `check` after independent implementation review, then `all` after Stage0 PASS.
No model or optimizer state is written to disk. All selected/E10/E50 states and recovery
checks use RAM. An interrupted coordinator requires rerunning the affected matrix.

`all` executes the forty paired trajectories, freezes the validation/D1 P2 gate,
then freezes every checkpoint before unified testing. If P2 triggers, the coordinator
retains RAM and waits for `phase2_ready.json` in that attempt. The separately reviewed
conditional implementation supplies `phase2.execute(bundles,p,results,out,devices,ready)`.
The ready JSON must contain `review_verdict`, `blocking_fixes_complete`,
`implementation_commit`, and `budget_frozen`. No P2 algorithm is implemented here.

Numerical tables can be regenerated after training with:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.analyze --run-dir artifacts/gradient_aug_momentum_20261101/attempt_<timestamp>
```
