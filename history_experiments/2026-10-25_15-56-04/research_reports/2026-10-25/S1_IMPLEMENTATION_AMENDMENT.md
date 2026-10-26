> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`467fa43c7a0e93e8f177ed7f78be6be4d2f5657212af206a92a05dbe05bf13e6`。

# S1 implementation amendment

Critical bridge-only correction discovered by inspecting the old bank: the historical mean64 input is raw expert mean, whereas the first bridge implementation averaged the individually normalized experts. The concat256 K1/K4 arms correctly use expert-wise LayerNorm and are unaffected. Preserve both original mean attempts; rerun only mean-K1 on both datasets under explicit corrected IDs before S1 carrier analysis. No outer evaluation occurred.

Other changes are lifecycle-only or new stage code: static calibration no longer spends time replaying unused shadow controllers; cache time is separated from actor fitting time to avoid duplicate accounting. Native and concat S1 predictor trajectories are unchanged. Source receipts per stage retain both versions. Review remains primary-agent, not independent.

Checkpoint storage is at /tmp/mdmt_next_v2_20261025_checkpoints, explicitly linked from the campaign root, because /home has 24GB free and the system volume has 179GB. This is a local temporary filesystem path; preserve it with the campaign when archiving.

The sealed warmup feature packets originally recorded the previous completed epoch at both half and end anchors. End anchors should count the epoch whose final fit batch has completed. This affects only the state-feature stage coordinate, not native training, actions, utility labels, risk rows or support. The controlled bootstrap now mechanically derives coordinate9 from the known anchor index, log1p(floor(4.5+.5*j))/log51; original files remain unchanged. Production end anchors commit completed epoch before measuring. Controllers are rebuilt by strict temporal replay, not by fitting all labels at once. FREEZE_WARMUP has no reported prequential scores on its first12 selection anchors.
