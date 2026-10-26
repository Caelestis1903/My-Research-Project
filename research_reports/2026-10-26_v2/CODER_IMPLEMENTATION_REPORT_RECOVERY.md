# CODER IMPLEMENTATION REPORT — 四种子恢复统一入口

状态：两部分实施完成，等待独立 reviewer 验收；未启动正式训练或分析真实效果。

依据：`DESIGN_FOUR_SEED_RECOVERY_ADDENDUM.md`。

- Coder A 数据内存部分：`CODER_IMPLEMENTATION_REPORT_RECOVERY_DATA.md`。对应 `DATA_MAPPING_BASIC_CHECKS.json`，逐行数据 mmap / ledger 等义外移；不保存模型或优化器。
- Coder B 编排部分：`CODER_IMPLEMENTATION_REPORT_RECOVERY_CAMPAIGN.md`。对应 `RECOVERY_CAMPAIGN_BASIC_CHECKS.json`，四种子门槛、已开始控制证据补全、幂等、重放核 hash、主机内存准入、逐方法训练 receipt 与清理白名单。

有效配置：`experiments/mdmt_rcf_20261026/config_four_seed.json`，由原计划派生，原 09 文件保持原字节；hash `cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6`。

新运行目录：`artifacts/mdmt_rcf_20261026/v2_four_seed`。旧 v1 完整保留；SCALE 已有 8 条四种子完成证据只核身份复用，不重训。

主实验当前无新训练，实际 `FOUR_SEED_ACCEPTANCE.json` 是 `BLOCKED_INCOMPLETE`，不可据此写正式研究结论。reviewer PASS 后由 runner 执行真实首 prefix 峰值 / 同 seed 三模型 hash / 所有方法四种子完成 / 外层冻结 / 全导出验收。
