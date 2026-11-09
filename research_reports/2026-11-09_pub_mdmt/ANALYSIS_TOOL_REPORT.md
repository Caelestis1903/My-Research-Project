# 诊断数值汇总工具交接

状态：IMPLEMENT COMPLETE。新增 `analyze_diagnostics.py`；实验包和运行产物保持当前内容。脚本只读取训练侧诊断与短步BCE记录，科学解释由orchestrator完成。

执行：

```sh
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
$PY research_reports/2026-11-09_pub_mdmt/analyze_diagnostics.py --self-check
# 全部训练完成、EXPERIMENT_COMPLETE.json为COMPLETE后：
$PY research_reports/2026-11-09_pub_mdmt/analyze_diagnostics.py \
  --out artifacts/pub_mdmt_20261109/attempt1
```

本次仅执行内置fixture，结果 **PASS**。覆盖精确(epoch,step)连接，附近step排除，六格Spearman排序，ties/zero/missing，C0两任务向量保留，实际solver残差字段与refresh计数，非对角cosine及三个阶段边界。真实训练日志的汇总由全训练结束后的主agent执行。

输出到artifact根下 `tables/`：

- `DIAGNOSTIC_SEEDS.csv`：dataset/arm/seed/group/阶段1–10、11–30、31–50。包含实际diag条数、非空solve记录数、solver失败与fallback条数；raw residual最大值只取实际solve字段。alpha sum为每条diag的sum再取均值，alpha min/max为该seed/group/阶段内分量极值，alpha mean为已记录分量均值。候选norm均值、唯一非对角候选cosine均值、actual/data utility正负零比例，以及应用/自然Adam更新norm和ratio均值分别保留。
- `MECHANISM_PAIRS.csv`：S组diag与SHORT_STEP combined按(epoch,step)精确内连接，保留原始utility向量和六格BCE变化。C1/C2通过SHORT_STEP的候选身份确认六个真实cell，再对utility和负BCE变化计算Spearman；ties用平均rank，常量向量的rank相关为空。sign agreement只在两侧均非零的cell计算，零值及比较分母单独记录。缺失cell或utility保留原值并记录对应状态。C0保留两任务向量及覆盖，不做六格rank。
- `PAIR_COVERAGE.csv`：每dataset/arm/seed的首batch观察数、S diag数、实际可连接窗数、缺配窗数、六成员完整loss窗数、有效rank窗数，以及首batch/配对epoch和精确epoch-step清单。配对数量由实际日志决定，包含实际额外refresh，未使用附近step补齐。

`refresh_solve_count`表示该组实际调用solver的记录数。P1 D/T继承S系数，其本地solve计数为0。所有汇总行保留seed身份；脚本不把window视为独立seed，也不进行跨window显著性检验。Test指标不参与该工具。
