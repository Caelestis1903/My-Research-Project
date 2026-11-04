# 旧A三模型30条轨迹重选审计

逐条只读核对原formal run.json、完整最终阶段epoch日志、final_freeze身份和六格E端点。未运行训练或前向，未加载模型权重。

30条均有覆盖其历史实际最终阶段的完整V曲线。按该曲线严格最大AUC、并列最早重选：13条可直接接入同epoch完整历史E端点；17条缺对应选中状态/端点；不完整曲线0条。

30条主表均INCOMPATIBLE：原train为T0∪F、预处理拟合T0；域人口与本轮完整自然train不同。可重选项只进入OBSERVED_LEGACY_TRAJECTORY历史附录，保留旧LL早停及原raw-logit AUC/softplus LogLoss语义。

旧标量收据没有tensor model hash；CSV的model_hash留空，checkpoint_file_sha256标识实际已存在的checkpoint文件，checkpoint_identity_sha256绑定原冻结的阶段/epoch/模型meta和schema。两者分别命名，文件摘要不替代tensor摘要。

| 数据 | 方法 | 可重选seed | 缺状态seed |
|---|---|---|---|
| movielens | M3OE_FIXED1 | — | 2022,2023,2024,2025,2026 |
| movielens | HYPERGATE | 2023,2024 | 2022,2025,2026 |
| movielens | AUTOIFS | 2022,2023,2024,2025,2026 | — |
| kuairand | M3OE_FIXED1 | — | 2022,2023,2024,2025,2026 |
| kuairand | HYPERGATE | 2022,2023 | 2024,2025,2026 |
| kuairand | AUTOIFS | 2022,2024,2025,2026 | 2023 |

逐条路径、SHA256、完整候选窗口、选中epoch、保存视图及E端点见LEGACY_A_RESELECTION_AUDIT.csv；对应V曲线、原冻结checkpoint身份和完整E指标见LEGACY_A_RESELECTION_EVIDENCE.json。重新运行本目录audit_legacy_a.py可从只读历史产物复建此审计。
