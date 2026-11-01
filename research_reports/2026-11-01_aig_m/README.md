# AIGM01 完整实验交付

[完整科学报告](REPORT.md) · [逐阶段搜索报告](SEARCH_REPORT.md) · [原始执行事实](RUNNER_REPORT.md) · [代码入口](../../experiments/gradient_aug_aig_m_20261101/README.md)

完成56条唯一50epoch训练、2800epoch、96600次参数更新；名义64条中的8条严格RAM缓存已单列。90条正式测试端点齐全，训练失败/OOM0，模型状态已按要求从RAM释放。

- EMA：ML selected test ΔAUC +.003849（5/5正），KR +.004616（4/5正），均通过预设门。ML LL仍平均+.001249、4/5恶化。
- 普通增强：ML NO_PROMOTION；KR TRADEOFF，macro/LL改善伴随最大域两任务的AUC下降。
- 固定SAD和AIG-M：开发增量门未通过，停止晋级；没有追加搜索或五seed候选测试。

## 入口和数值

| 内容 | 文件 |
|---|---|
| 设计→实现→独立审查 | [DESIGN](DESIGN_SPECIFICATION.md)、[IMPLEMENTATION](IMPLEMENTATION_REPORT.md)、[REVIEW PASS](REVIEW_REPORT.md) |
| Stage0、运行与成本 | [Stage0](STAGE0_REPORT.md)、[runner](RUNNER_REPORT.md)、[completion](raw_attempt_tables/attempt_20261101_1720/completion.json)、[costs](raw_attempt_tables/attempt_20261101_1720/costs.csv) |
| 来源和历史复算 | [evidence](EVIDENCE_REVIEW.md)、[source receipt](SOURCE_FETCH_RECEIPT.json)、[execution identity](EXECUTION_IDENTITY.json)、[historical pairs](historical_paired_effects.csv) |
| 全部开发checkpoint与门控 | [development checkpoints](development_checkpoints.csv)、[development summary](development_summary.csv)、[mechanism](aig_mechanism_summary.csv)、[thresholds](raw_attempt_tables/attempt_20261101_1720/thresholds_frozen.json) |
| 五seed正式原值与配对 | [selected](raw_attempt_tables/attempt_20261101_1720/selected_metrics.csv)、[test](raw_attempt_tables/attempt_20261101_1720/test_metrics.csv)、[validation](raw_attempt_tables/attempt_20261101_1720/validation_metrics.csv)、[paired effects](raw_attempt_tables/attempt_20261101_1720/paired_effects.csv) |
| 冻结与去留 | [formal configs](raw_attempt_tables/attempt_20261101_1720/formal_configuration_frozen.json)、[selected identities](raw_attempt_tables/attempt_20261101_1720/selection_frozen.json)、[decision](raw_attempt_tables/attempt_20261101_1720/decision.json) |
| 运行、失败和释放 | [trial ledger](raw_attempt_tables/attempt_20261101_1720/trial_ledger.csv)、[launch ledger](raw_attempt_tables/attempt_20261101_1720/launch_ledger.csv)、[empty failure ledger](raw_attempt_tables/attempt_20261101_1720/failure_ledger.csv)、[RAM release](raw_attempt_tables/attempt_20261101_1720/state_release.json) |
| Stage0原始收据 | [check attempt](raw_attempt_tables/attempt_20261101_1720_check/)、[stage0 checks](raw_attempt_tables/attempt_20261101_1720_check/stage0_checks.json) |

训练执行commit：`2cec25ab5b32411855a13ab3143678a5e980ea3f`。正式五seed包含开发2022/2024，历史test也用于过往研究判断；当前区间描述固定自然划分的训练随机性。源码和环境版本、数据身份、选择规则详见REPORT与[依赖锁](../../requirements-phase0-3.lock)。

## 原始归档和复算

[完整check+all归档](raw_attempts_2026-11-01_aig_m.tar.gz)含687个文件、56份stdout和56份stderr，以及逐run配置、epoch、batch、view、split行号和各类数值；不含权重/checkpoint二进制。压缩包SHA256：

`23dc300c1fdafc9d477e1c975759f4d1b3b3af45b33224fb03e5df52bbede09a`

[归档校验文件](raw_attempts_2026-11-01_aig_m.tar.gz.sha256) · [逐文件SHA256](raw_attempts_2026-11-01_aig_m_SHA256SUMS.txt)。归档内路径从两个attempt目录开始，在仓库根目录运行：

```bash
mkdir -p artifacts/gradient_aug_aig_m_20261101
tar -xzf research_reports/2026-11-01_aig_m/raw_attempts_2026-11-01_aig_m.tar.gz -C artifacts/gradient_aug_aig_m_20261101
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_aig_m_20261101.analyze --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/development_tables.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/make_figures.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/build_report.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
```

所有复算只读数值文件。模型复评需用代码入口重训，使用新的空attempt目录；Stage0 PASS receipt、代码commit和源码manifest必须一致。
