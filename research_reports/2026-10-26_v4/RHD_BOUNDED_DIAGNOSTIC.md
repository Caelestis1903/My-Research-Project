# RHD 有界诊断与历史动作复核

本轮不启动第二条完整研究线。2026-10-26 04:33 UTC 实查时，旧 checkpoint 根 `/home_new/xzf_mdmt_next_v3_2026-10-25_15-56-04` 已不可访问，旧 artifact 的 checkpoints 链接不能恢复状态。因此 KR seed3101/3102 的完成 epoch12/20 四个规定锚均为 **MISSING_ARTIFACT**；不替换 epoch、seed 或方法，不重训旧 baseline，新增分叉更新为0。启动初期只曾确认目录列表可见，未把当时的文件列表当成本轮完整状态验收。日志中的历史 checkpoint hash 不能替代现存模型/Adam/RNG/row-plan 字节。

本地历史日志仍可读。对旧四正式 seeds 的 `policy_decisions.jsonl` 直接汇总完成 epoch11–50 的动作占用率如下。KR 的 STATIC 就是 BASE；统计单位为既有训练 seed。这里没有训练新模型或重新选择方法。

| seed | PB_H1 非BASE | PB_HB 非BASE | PB_HISTORY 非BASE |
| --- | ---: | ---: | ---: |
| 4101 | 900/1600 (56.25%) | 680/1600 (42.50%) | 1580/1580 (100%) |
| 4102 | 780/1600 (48.75%) | 700/1600 (43.75%) | 1540/1580 (97.47%) |
| 4103 | 680/1600 (42.50%) | 500/1600 (31.25%) | 1580/1580 (100%) |
| 4104 | 740/1600 (46.25%) | 560/1600 (35.00%) | 1380/1580 (87.34%) |

HISTORY 自有日志不含共享首半个 epoch，因此其分母为1580，未补造缺失块。完整按11–20/21–30/31–40/41–50划分的八动作计数、日志路径和 SHA256 存于 `artifacts/mdmt_dfd_v4/v4_0/rhd_historical_audit/{old_action_occupancy.csv,sources.json}`。HB 晚期41–50非BASE占比依次为30%、50%、35%、35%；相应 HISTORY 为100%。这与更保守地保持 BASE 的解释相容，但信息来源、动作身份和接受频率同时变化，观察性统计不能识别频率的独立原因。

历史固定提交的正式报告同时显示：HB 相对 BASE 的 selected test 平均 ΔLL 为 +0.00074075，而共同 epoch50 为 −0.00875428；所有20条轨迹的 epoch50 LL 都差于各自合法 selected。该事实属于旧版本，不是 DFD-v4 结果。它支持把晚期少恶化与最佳可用 checkpoint 质量分开，不能据占用率重述主指标成功。[固定历史复评](https://github.com/Caelestis1903/My-Research-Project/blob/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0/research_reports/2026-10-25_v3/FORMAL_LOCKED_REEVALUATION.md)

证据强度：已存动作日志占用差异为 **STRONG** 的记录性证据；“少行动解释晚期减损”是 **WEAK** 的因果解释。共享/私有四叉的风险响应、AUC与六格变化本轮均 **未测**，不能判定 RESEARCH_REOPEN_ELIGIBLE。若未来恢复四个原定完整锚，才可依原方案执行固定分叉；本轮不自行打开替代实验。
