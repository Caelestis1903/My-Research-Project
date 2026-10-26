# 复算、来源和后续入口

从项目根目录执行，依赖既有phase0-3-env和/home/xzf/data。源规划基底b17cb2707fc2448f09f21fd12279b2b3856c7756；新代码确切hash见各receipt，Git归档见publication_status.json。历史baseline/code/data identity与本轮发布身份分别记录，不把后来的归档commit当启动commit。

## 只复算已保存数值（无需dataset或GPU）

```bash
OPENBLAS_NUM_THREADS=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-16/execution_20261016/analyze_numeric.py
python3 research_plans/2026-10-16/execution_20261016/build_novelty_matrix.py
```

analyze_numeric验证8case、56产物hash、32个旧/新covariance后重建汇总图表。需要旧C2_numeric_oracle_20261016目录（已在规划基底仓库）。它不调用模型/训练。

## 完整数值重新执行

state_oracle.py默认拒绝覆盖已有case。确需独立复现时，在research_plans/2026-10-16下使用新的**同层级执行目录**复制state_oracle.py、analyze_numeric.py；不要复制旧numeric_oracle输出。先运行`--check`，再按两个dataset、2022–2025逐项运行：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-16/NEW_EXECUTION_DIR/state_oracle.py --check
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/xzf/miniconda3/envs/phase0-3-env/bin/python research_plans/2026-10-16/NEW_EXECUTION_DIR/state_oracle.py --dataset kuairand --seed 2022
```

实际本次八条命令、PID、退出码、起止时间在numeric_queue_status.json。run_numeric_queue.py是本次持久化机械CPU队列；已全部退出0。它不是GPU训练launcher。

## 历史取证复算

audit_and_selection.py依赖本地历史工件与已有B1分析表；它不重新读400份大state，不代替原审计，只复核receipt/关键静态产物。原始权重/数据/完整预测未随新提交上传；其本地路径在artifact_inventory.csv。发布clone没有本地原始权重时，缺失必须保持INCOMPLETE。

recover_ao04.py已完成attempt3。若新独立运行，显式指定未存在的`--output-name`；勿覆盖旧目录。`--reuse-matches`只接受记录过的完整预测匹配，且重新核对state与NPZ哈希。attempt2的3项全预测绑定在Git保留其inventory与日志；attempt3按同hash复用。B1/2023没有原config时只做结构索引，不构造伪原配置。

bootstrap_validation.py要求attempt3 row绑定和原B0/B1预测；输出六cell AUC/LL条件区间，不能合成macro CI或称独立holdout。

## 推进正式研究所需输入

1. 原S5/S6/S6-R2：run config、source/version、split/init/batch identity、selected/common/final完整model/optimizer/RNG和逐行预测。当前372个显式历史路径仍缺；检查替代内嵌state后仍缺的范围见gate_decisions.md。
2. 可满足原规划的审查流程；本轮start.md禁止非机械子代理，不能自填独立PASS。
3. 若上述条件解决，按原2D冻结四臂/四seed/计算预算/selection/guards并执行；不由本轮数值结果自动触发。Stage4/4B/6/E1依各自前提推进。
4. 任何方法确认使用未开发的新temporal/external holdout；旧test继续只能法证。
