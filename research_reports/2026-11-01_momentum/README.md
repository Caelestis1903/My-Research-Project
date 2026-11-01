# GAM01 完整实验交付

[完整分析报告](REPORT.md) · [逐seed结果与checkpoint身份](SELECTED_SEEDS.md) · [执行事实](RUNNER_REPORT.md) · [原始工件](RAW_ARTIFACTS.md)

完成两数据集×四臂×五seed，共40条50epoch轨迹、2000epoch、69000次参数更新，正式失败0。全部选择冻结后统一测试，240条online/shadow selected/E10/E50记录完整。P2两数据均未触发，本轮停止。

- **MovieLens P0：TRADEOFF**。selected AUC +.004399（5/5），LogLoss +.007800（4/5恶化）。
- **KuaiRand P0：GO**。selected AUC +.004616（4/5），validation/test预设保护未触发。
- **A3差分长记忆：未建立独有增量**。相对普通增强Adam，两数据test AUC均差为负。

## 代码和证据

新代码在 [experiments/gradient_aug_momentum_20261101](../../experiments/gradient_aug_momentum_20261101/README.md)，复用未改动的GA31-R3命名空间。执行代码commit为 `89edb99a389c56fde86460af677a48e14fbcf116`；发布分支为 [26-11-01_v2](https://github.com/Caelestis1903/My-Research-Project/tree/26-11-01_v2)。

| 内容 | 文件 |
|---|---|
| 设计→实现→独立审查 | [DESIGN_SPECIFICATION](DESIGN_SPECIFICATION.md)、[IMPLEMENTATION_REPORT](IMPLEMENTATION_REPORT.md)、[REVIEW_REPORT](REVIEW_REPORT.md) |
| 四项CPU检查、真实CUDA与RAM恢复 | [formula_checks](formula_checks.json)、[semantic_checks](semantic_checks.json)、[stage0_checks](stage0_checks.json)、[checkpoint_restore_checks](checkpoint_restore_checks.json) |
| 配置、源码、环境、数据身份 | [resolved_config](resolved_config.json)、[source_manifest](source_manifest.json)、[environment](environment.json)、[data_manifest](data_manifest.json) |
| 原值、完整轨迹、所有域任务 | [selected_metrics](selected_metrics.csv)、[test_metrics](test_metrics.csv)、[validation_metrics](validation_metrics.csv)、[epoch_metrics](epoch_metrics.csv)、[per_cell_metrics](per_cell_metrics.csv) |
| 五seed配对、SD和区间 | [paired_effects](paired_effects.csv)、[effect_summary](effect_summary.csv)、[arm_summary](arm_summary.csv) |
| 差分状态及shadow | [optimizer_state_diagnostics](optimizer_state_diagnostics.csv)、[diagnostics_summary](diagnostics_summary.csv)、[shadow_metrics](shadow_metrics.csv) |
| 门与选点冻结 | [P2 validation门](phase1_validation_decision.json)、[selection_frozen](selection_frozen.json)、[decision](decision.json) |
| 运行、失败和成本 | [run_ledger](run_ledger.csv)、[launch_ledger](launch_ledger.csv)、[failure_ledger](failure_ledger.csv)、[costs](costs.csv)、[completion](completion.json)、[state_release](state_release.json) |

## 复算与完整重训

数表和图可离线复算。先按[原始工件索引](RAW_ARTIFACTS.md)解包恢复正式attempt，再在仓库根目录运行：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.analyze --run-dir artifacts/gradient_aug_momentum_20261101/attempt_20261101_1340
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_momentum/make_figures.py --run-dir artifacts/gradient_aug_momentum_20261101/attempt_20261101_1340
```

完整重训使用新attempt目录：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_momentum_20261101/tests/test_semantics.py
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.run --mode check --device cuda:1 --run-dir artifacts/gradient_aug_momentum_20261101/attempt_NEW_check
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_momentum_20261101.run --mode all --devices 0,1 --code-commit 89edb99a389c56fde86460af677a48e14fbcf116 --run-dir artifacts/gradient_aug_momentum_20261101/attempt_NEW
```

数据路径和身份沿AGENTS.md及本轮配置；CSV不随交付上传。模型与优化器状态只存RAM，本次已释放；模型复评需要重训。无额外依赖、合同外算法或超参搜索。
