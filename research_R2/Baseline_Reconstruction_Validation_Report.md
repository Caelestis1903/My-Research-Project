# B0 重建与验证记录

2026-10-12，状态 **COMPLETE / G1 PASS**。两数据集分别按已锁定当前runtime重建B0 seed2022/2023；四run均完整50epoch且非smoke，退出码0。旧编排stub保留，不作为baseline。

原始selected epoch和全精度validation macro AUC、run manifest SHA256见 `preflight/B0_completion_raw_20261012.json`。两数据集四份G1均PASS，详细预测/指标重放误差、完整性、初始化与曝光、选模复算和test封存原始证据见 `preflight/G1_all_pass_20261012.json` 及各run的 `g1_replay.json`。未评价test。

正式累计4 attempt/200 run-epochs/0 resume；新根 `formal_runs_20261012/`，四份初始化三件套位于 `formal_init_20261012/`。原有代码hash对账无变化。G0真实数据路径适配见 `preflight/runtime_path_adaptation_20261012.json`。下一固定波次为B1两数据集×两seed四进程，再为B2四进程；本报告仅记录执行验收，不分析科学结论。
