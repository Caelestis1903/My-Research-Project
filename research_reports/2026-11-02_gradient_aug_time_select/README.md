# GATS02 研究交付入口

研究结论见 [REPORT.md](REPORT.md)，完整合同见 [设计规范](DESIGN_SPECIFICATION.md)，执行身份与方法见 [METHODS_AND_REPRODUCIBILITY.md](METHODS_AND_REPRODUCIBILITY.md)。本轮研究代码位于 [新实验包](../../experiments/gradient_aug_time_select_20261102/README.md)，历史依赖包保持原始tree身份。

实际恢复执行commit为 `3a164c6d6e3847883d1f2ae23c7d8e76ee47b6db`，源码manifest为 `95c5d28204849e2e815ff33753339a1ef115fdbb2401886923435b6e9511299d`；55个实际执行源码blob均已核对。首次执行 `ac2ec5cc70be7dcb880a7dde1015e4634726fd96` 的失败、开发数值和额外成本单独保留。

## 运行与复现

从仓库根目录使用项目环境，不使用Conda base。真实CSV路径、SHA与预处理身份见 [DATA_VERIFICATION.json](DATA_VERIFICATION.json)，环境锁为根目录 `requirements-phase0-3.lock`。必要CPU回归读取真实旧端点，fresh clone须先按归档说明恢复failed_0005数值日志。模型与所有训练状态只在RAM；需保持协调器存活至统一测试及释放结束，没有可下载的权重checkpoint。

本次恢复的实际命令结构如下；stdout/stderr必须在run-dir外另设文件，run-dir初始为空：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_time_select_20261102/tests
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode check --device cuda:1 --code-commit 3a164c6d6e3847883d1f2ae23c7d8e76ee47b6db --recovery-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005 --task-start-unix 1793549242.1426222 --run-dir artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315_check
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_time_select_20261102.run --mode all --devices 0,1 --code-commit 3a164c6d6e3847883d1f2ae23c7d8e76ee47b6db --recovery-attempt artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005 --task-start-unix 1793549242.1426222 --stage0-receipt artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315_check/stage0_checks.json --run-dir artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315
```

以上是**历史执行命令**，原始12小时deadline不会因重新运行延长。未来从头复现实验需使用新的空目录，在check/all命令中省略 `--recovery-attempt` 与 `--task-start-unix`，以该次新任务的实际起点计时；不要修改旧attempt或把过期起点换写成旧任务继续运行。新GPU实验须先检查可用资源并取得匹配源码的新Stage0 PASS。

仅重现最终数值表与图，不需要GPU或模型权重：先按 [归档说明](raw_archive/README.md) 恢复数值日志，再从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-02_gradient_aug_time_select/finalize_analysis.py --run-dir artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0315
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-02_gradient_aug_time_select/make_final_figures.py
```

阶段A0/A1与机制诊断的重算入口为 `analyze_a_group.py`、`analyze_mechanisms.py`、`make_time_figures.py`，均接受 `--run-dir`；最初阶段表来自失败attempt的完整开发结果，跨attempt身份相同的证据另存。最终表以完成恢复队列为准。

## 证据导航

- 合同与审查：[DESIGN](DESIGN_SPECIFICATION.md)、[IMPLEMENTATION](IMPLEMENTATION_REPORT.md)、[初次REVIEW](REVIEW_REPORT.md)、[恢复DESIGN](RECOVERY_DESIGN_ADDENDUM.md)、[恢复IMPLEMENTATION](RECOVERY_IMPLEMENTATION_REPORT.md)、[恢复REVIEW](RECOVERY_REVIEW_REPORT.md)。
- 运行事实：[RUNNER_REPORT](RUNNER_REPORT.md)、[STAGE0_REPORT](STAGE0_REPORT.md)、[失败attempt](FAILED_ATTEMPT_RUNNER_REPORT.md)、[首次Stage0](INITIAL_STAGE0_REPORT.md)。
- 阶段分析：[A组](A_GROUP_VALIDATION_ANALYSIS.md)、[A1/C2](A1_C2_VALIDATION_ANALYSIS.md)、[B1](B1_VALIDATION_ANALYSIS.md)、[B2](B2_VALIDATION_ANALYSIS.md)、[B3](B3_VALIDATION_ANALYSIS.md)。这些阶段报告保留当时等待控制的上下文，最终裁决见REPORT及decisions目录。
- 原始与派生数值：`tables/`包括全端点/逐seed配对/样本SD与t95%区间/域任务人口与基线/成本；`decisions/`包括控制、候选、冻结、完成和RAM释放收据；完整原始日志见 `raw_archive/`。
- 已知测量范围：[DIAGNOSTIC_ACCOUNTING_REVIEW](DIAGNOSTIC_ACCOUNTING_REVIEW.md)。未使用mask的增强臂actual perturbation energy为N/A，原始零不表示没有Gaussian扰动；物理成本与科学样本数分开。
