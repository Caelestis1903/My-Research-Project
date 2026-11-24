# 00 — Codex 执行手册

## 1. 任务与执行权限

执行目标是把历史信息机制变成可比较的完整训练方法：在固定自然 MDMT 协议下实现 C1/C2，完成相关诊断、必要匹配控制、五 seed 评估与报告。当前提交交付规划，不运行模型。后续用户指令启动本手册时，先记录该任务允许的训练与 Test 范围。

本手册沿用根 `AGENTS.md`、`start.md` 中对自然数据、种子、RAM 状态与资源调度的要求。历史 Cycle04/Cycle05 已停止；本研究使用新 study ID `history_information_20261124`，不恢复旧自循环。当前用户明确指定本方案提交 main；未来实验代码、日志、报告按其执行任务指定分支交付。

## 2. 先读取什么

1. 读取本目录 README、01、03、05、06、08 和机器 manifest。
2. 从固定提交复核 `mdmt_research_context/01_AI_EXPERIMENT_INDEX.md`；按01的相关实验编号核对02、03。AI 不获取 `04_HUMAN_RESEARCH_REVIEW.md`。
3. 按11的固定路径取得源实现与数字；按07的源码闭包恢复清单核对依赖、模型、CSV和预处理。
4. 本轮公式以05为准，矩阵以06/manifest为准，阶段门以08为准。如文件对同一科学定义存在冲突，先生成一项具体修订并完成独立审查，再创建训练队列。

## 3. 四阶段工作流

| 阶段 | 必做工作 | 完成产物 | 进入后续的依据 |
| --- | --- | --- | --- |
| S0 / 第一阶段 | 来源身份、历史端点复用、理论和假设冻结、设计交接 | `SOURCE_MANIFEST.json`、`REUSE_REGISTER.csv`、`DESIGN_SPECIFICATION.md`、`TEST_POLICY.json` | 自然协议、配对支持、算法零点和比较集合明确 |
| S1 / 第二阶段 | 必要父轨迹重建、D1–D4诊断；同轨迹 online/EMA与shadow采集 | 诊断表、父状态收据、`MECHANISM_REPORT.md` | 完整模型与原生更新通过关键验收；科学信号正负均进入S2 |
| S2 / 第三阶段 | C1/C2四个非零配置的40条full50；双读出与完整搜索 | 完整曲线、五seed端点、配置选择表 | 按08选出每个家族候选或关闭该家族 |
| S3 / 第四阶段 | 有收益候选的M/S/P控制；有条件组合、范围、骨干、Gaussian；全局冻结和允许的Test评价 | `FINAL_FREEZE.json`、完整对照、`FINAL_REPORT.md`、来源与成本、研究决策 | 所有触发工作完成，结果与未触发原因登记并交付 |

S1不能用单步余弦、局部AUC、同状态面板不显著或短期负收益取消S2。若S2两个家族的完整训练均无继续价值，关闭扩展模块，仍交付完整负结果、已排除解释与下一步研究决策。

## 4. 实现与独立审查

主代理负责科学问题、资源与阶段调度。先由设计代理产出规范；编码代理实现；另一个独立审查代理核对公式、状态与对照。编码与审查由不同原生子代理承担，使用显式文件交接；新子任务采用独立上下文。研究/编码/审查子代理沿用主代理模型，机械runner按当前环境可用能力分配。不存在指定型号时记录实际模型，保持角色独立。

审查通过且blocking问题修复后，启动当前会话监督的训练进程。runner写原始数据，主代理负责科学解释。不得通过独立后台AI或旧 autonomous driver 扩展研究。

## 5. 入口命令合约

以下 `experiments.history_information_20261124` 是**编码阶段需创建的新模块接口**。当前仓库已有可复用接口见11；这些新命令在实现审查通过后使用。

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli audit --plan research_plans/26-11-24_history_information/experiment_manifest.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli check --scope scientific_contracts
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S2
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S2
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S2
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S3a --decisions artifacts/history_information_20261124/decisions.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S3a
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S3a
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S3b --decisions artifacts/history_information_20261124/decisions.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S3b
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S3b
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S3c --decisions artifacts/history_information_20261124/decisions.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S3c
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S3c
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli materialize --stage S3d --decisions artifacts/history_information_20261124/decisions.json
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli run --stage S3d
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli analyze --stage S3d
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli freeze
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli evaluate --split test
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.history_information_20261124.cli report
```

S3a只运行S并冻结C/S代表；S3b运行相应M/P；S3c按规则运行或跳过C12后冻结Cstar；S3d运行已触发N/HyperGate/Gaussian。空阶段写`NOT_TRIGGERED`并向下一阶段透传决策，不制造空训练。S0若发现发表基线身份需修复，另以同样`materialize/run/analyze --stage BASELINE_REPAIR`执行，仅在需要时实例化，freeze前验收。

每个命令独立调用。`materialize`只生成通过对应规则的任务；`run`必须检查当前停止控制、审查收据、现行资源和状态可用性。`evaluate`检查冻结哈希、Test权限及selected RAM实际读回。未获本轮Test权限时，保留冻结模型和完整Validation报告，并登记`TEST_PENDING_AUTHORIZATION`；不以旧实验的许可代替本轮许可。

## 6. 复用与恢复

- 先读取已提供的历史重分析文件。相同轨迹/seed的数字复用不重复训练、不增加n。
- 需要同状态干预而原RAM已消失时，只恢复B0/B1/H0各一次；同一恢复同时产生后续所需shadow、双读出和selected状态。父轨迹重建不选择表现最好的attempt。
- HPO候选50epoch完整完成，全部配置都有两个selected读出；禁止以某配置E10较差替换正式预算。
- 恢复状态含模型/buffers、optimizer、两个时钟、shadow、EMA、RNG、loader游标、最佳读出和数据/源码身份。只保存日志、数值、代码与报告到磁盘，训练状态保存在RAM并读回核验。
- 现有CSV可通过冻结loader重建预处理；不把旧prepared目录缺失当成CSV不可用。

## 7. 停止与结案

`TECHNICAL_PAUSED`对应实际身份错误、非有限更新、状态丢失或资源不足；保留失败证据并修复。`SCIENTIFIC_CLOSED`对应完整训练和匹配控制后的无独立增量；完成报告后关闭分支。用户停止时安全完成或checkpoint当前原子步骤，并报告已完成、未完成和状态位置。

本研究采用有限矩阵及预定条件模块，不启动新一轮开放式机制搜索。所有已触发工作均完成、报告复核、上下文更新和远端交付后，记录`COMPLETED`。新方法若被判为辅助组件，依12给出具体下一研究问题，等待下一任务定义。
