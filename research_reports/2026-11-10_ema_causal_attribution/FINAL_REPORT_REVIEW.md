# 最终科学报告有限独立证据核对

日期：2026-11-11。Verdict: **PASS WITH FIXES**。两处身份/措辞精确化已完成，无剩余阻塞项。

本次核对版本 `FINAL_REPORT.md` SHA-256：`3e56ec0c798e03d5877840bf558a95159d393fa5b7a3466ac0a71b70cd0d01c8`。依据为完整1604条 `EVIDENCE_SUMMARY.json`、`REPORT_ADDITIONAL_EVIDENCE.json`、相关原始标量表和 producer 定义。本审阅只核查关键数值、比较身份和证据适用范围；主科学裁定由根代理负责。

## 已闭合的两处修正

1. 首页 Q4 原称“rho预测器全部退化为统一Fast”，范围过宽。原动作频率显示 carry/direct 的 all/fast 次数为 ML-C 19/25、KR-C 22/25、ML-G 19/25、KR-G 24/25。报告现明确限定 C_design/C_simple；新增 `FINDINGS_CLARIFICATIONS.md` 说明冻结文本中的名称范围，保留原 findings/freeze/Test 哈希。未改变数值或主方向。
2. 训练来源表原未直接标出混合人口及阶段。现表头明确 late E35/E50、h=ML40/KR138、recent-only ΔAUC 来自全 V、净恢复来自 V_mech 固定 pair、目标变化来自 P_B 自然面板，避免将三列误读为同一全 V 统计。

## 有限核对结果

- 原 ENDPOINTS 的四条件独立 selected F−O Macro AUC 分别为 `0.003848844882298519`、`0.004616399443180878`、`0.003856801819326583`、`0.007766927498394205`，与报告一致。对 AUC/LogLoss 逐 seed 复核独立 F−O 与同刻 F−OatF；后者 epochs 匹配。两种比较未混用。
- MECHANISM_FINDINGS 的主方向为 C；findings 哈希匹配 freeze，Test 收据绑定 freeze 哈希，freeze 时间早于 Test 完成时间，逻辑读出数为100。报告将 C 写为下一轮短程反馈判别，而非已经证实长期训练收益。
- 从两 Clean 原始 SOURCE_ASSAY 与相同 seed/anchor/population 的 O 指标，独立复算16条 AUC/LL 汇总及五 seed 数值。全部匹配附加证据；源记录 scope 仅 EMBED、anchors 仅 E20/E35。current_preconditioned 保留 vhat、使用当前 gradient+coupled WD、匹配真实组步幅；momentum_isotropic 保留一阶方向而去掉坐标预调节。报告没有把该局部虚拟一步写成已反馈训练实验。
- C_PRIMARY/recent、old 均按删除态解释；四条件实际选定 X 均为 C_PRIMARY/recent，故 X_F 的 00=F−recent、11=F 身份正确。
- EMBED 的 O/F 两背景、recent/old 两独立项、交互与联合项与附录一致。逐 seed 检查加法闭合，以及四条件两个背景的交互均为正。报告明确 F 背景是其余坐标的背景，没有把 old 单独负效应直接等同于删除 old 有益。
- 准确 pair corrected/broken 使用已归一化 AUC 贡献；报告描述包含平局半贡献，并区分全 V 准确账本与固定抽样重叠/示例。q−F 明确限定同 anchor 的 V_mech，未冒称独立 selected Test 性能。前序附录审阅的归一化与配对检查继续适用。
- 融合删除明确为 E35+tF，未标成 E35/E50 的 late。细融合留出不稳定和完整切口证据分别陈述；未把 task 输出项效果当成 task-private 参数 EMA 必然有效。
- KR mechanism7 的 V_mech oracle 均值约 `0.0011583` / `0.0012484` 与保存汇总相符。报告明确 KR 为 ROOM_WITHOUT_ACTIONABLE_SIGNAL；未把训练可见规则无增量写成“没有空间”。Oracle、静态、拟合规则、carry/direct 的身份分开。
- 报告给出具体局限的实际影响：q 缺完整独立 Test、Adam来源仅局部虚拟步、path 未闭合不能作积分份额解释、KR细融合/窗口来源更弱、有限早期LOSO训练单元不代表全部信号不可用。未以笼统免责声明替代这些边界。

## 范围与交付

未发现需改变主要数值、重算科学结果或推翻所述证据身份的错误。本次没有重复计算全部大表、拟合、运行模型或实验；只新增本审阅文件，未修改报告、代码或原表。四条件图表导出与 release 收据属于正在完成的交付步骤，不作为此次证据核对的阻塞项。
